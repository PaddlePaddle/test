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
    return [85][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_414_0_0(self, parameter_6, parameter_4, parameter_2, parameter_3, parameter_5, parameter_0, parameter_1, parameter_7, parameter_8, data_0, data_3, data_6, data_1, data_4, data_7, data_2, data_5, data_8):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('-1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_1 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_2 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_3 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_4 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_5 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_6 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_7 = full_0

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(parameter_0, full_0, float('0'), True)

        # pd_op.exp: (1xf32) <- (1xf32)
        exp_0 = paddle._C_ops.exp(scale_0)

        # pd_op.multiply: (-1xf32) <- (1xf32, -1xf32)
        multiply_0 = exp_0 * data_0

        # pd_op.add: (-1xf32) <- (-1xf32, 1xf32)
        add_0 = multiply_0 + parameter_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], float('0.5'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_8 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_9 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_10 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_11 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_12 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_13 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_14 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_15 = full_1

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(add_0, full_1, float('0'), True)

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(parameter_1, assign_7, float('0'), True)

        # pd_op.exp: (1xf32) <- (1xf32)
        exp_1 = paddle._C_ops.exp(scale_2)

        # pd_op.multiply: (-1xf32) <- (1xf32, -1xf32)
        multiply_1 = exp_1 * data_1

        # pd_op.add: (-1xf32) <- (-1xf32, 1xf32)
        add_1 = multiply_1 + parameter_1

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(add_1, assign_15, float('0'), True)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_2 = scale_1 + scale_3

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(parameter_2, assign_6, float('0'), True)

        # pd_op.exp: (1xf32) <- (1xf32)
        exp_2 = paddle._C_ops.exp(scale_4)

        # pd_op.multiply: (-1xf32) <- (1xf32, -1xf32)
        multiply_2 = exp_2 * data_2

        # pd_op.add: (-1xf32) <- (-1xf32, 1xf32)
        add_3 = multiply_2 + parameter_2

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(add_3, assign_14, float('0'), True)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_4 = add_2 + scale_5

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(parameter_3, assign_5, float('0'), True)

        # pd_op.exp: (1xf32) <- (1xf32)
        exp_3 = paddle._C_ops.exp(scale_6)

        # pd_op.multiply: (-1xf32) <- (1xf32, -1xf32)
        multiply_3 = exp_3 * data_3

        # pd_op.add: (-1xf32) <- (-1xf32, 1xf32)
        add_5 = multiply_3 + parameter_3

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(add_5, assign_13, float('0'), True)

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(parameter_4, assign_4, float('0'), True)

        # pd_op.exp: (1xf32) <- (1xf32)
        exp_4 = paddle._C_ops.exp(scale_8)

        # pd_op.multiply: (-1xf32) <- (1xf32, -1xf32)
        multiply_4 = exp_4 * data_4

        # pd_op.add: (-1xf32) <- (-1xf32, 1xf32)
        add_6 = multiply_4 + parameter_4

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(add_6, assign_12, float('0'), True)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_7 = scale_7 + scale_9

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(parameter_5, assign_3, float('0'), True)

        # pd_op.exp: (1xf32) <- (1xf32)
        exp_5 = paddle._C_ops.exp(scale_10)

        # pd_op.multiply: (-1xf32) <- (1xf32, -1xf32)
        multiply_5 = exp_5 * data_5

        # pd_op.add: (-1xf32) <- (-1xf32, 1xf32)
        add_8 = multiply_5 + parameter_5

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(add_8, assign_11, float('0'), True)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_9 = add_7 + scale_11

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(parameter_6, assign_2, float('0'), True)

        # pd_op.exp: (1xf32) <- (1xf32)
        exp_6 = paddle._C_ops.exp(scale_12)

        # pd_op.multiply: (-1xf32) <- (1xf32, -1xf32)
        multiply_6 = exp_6 * data_6

        # pd_op.add: (-1xf32) <- (-1xf32, 1xf32)
        add_10 = multiply_6 + parameter_6

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(add_10, assign_10, float('0'), True)

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(parameter_7, assign_1, float('0'), True)

        # pd_op.exp: (1xf32) <- (1xf32)
        exp_7 = paddle._C_ops.exp(scale_14)

        # pd_op.multiply: (-1xf32) <- (1xf32, -1xf32)
        multiply_7 = exp_7 * data_7

        # pd_op.add: (-1xf32) <- (-1xf32, 1xf32)
        add_11 = multiply_7 + parameter_7

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(add_11, assign_9, float('0'), True)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_12 = scale_13 + scale_15

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(parameter_8, assign_0, float('0'), True)

        # pd_op.exp: (1xf32) <- (1xf32)
        exp_8 = paddle._C_ops.exp(scale_16)

        # pd_op.multiply: (-1xf32) <- (1xf32, -1xf32)
        multiply_8 = exp_8 * data_8

        # pd_op.add: (-1xf32) <- (-1xf32, 1xf32)
        add_13 = multiply_8 + parameter_8

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(add_13, assign_8, float('0'), True)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_14 = add_12 + scale_17

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_16 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_17 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_18 = full_2

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(data_0, full_2, float('0'), True)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_15 = scale_18 + data_3

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_16 = add_15 + data_6

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(data_1, assign_18, float('0'), True)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_17 = scale_19 + data_4

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_18 = add_17 + data_7

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(data_2, assign_17, float('0'), True)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_19 = scale_20 + data_5

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_20 = add_19 + data_8

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(add_4, assign_16, float('0'), True)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_21 = scale_21 + add_9

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_22 = add_21 + add_14
        return full_0, exp_0, multiply_0, full_1, scale_1, assign_7, exp_1, multiply_1, assign_15, scale_3, add_2, assign_6, exp_2, multiply_2, assign_14, scale_5, assign_5, exp_3, multiply_3, assign_13, scale_7, assign_4, exp_4, multiply_4, assign_12, scale_9, add_7, assign_3, exp_5, multiply_5, assign_11, scale_11, add_9, assign_2, exp_6, multiply_6, assign_10, scale_13, assign_1, exp_7, multiply_7, assign_9, scale_15, add_12, assign_0, exp_8, multiply_8, assign_8, scale_17, add_14, full_2, scale_18, add_15, assign_18, scale_19, add_17, assign_17, scale_20, add_19, assign_16, scale_21, add_21, add_16, add_18, add_20, add_22



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

    def forward(self, parameter_6, parameter_4, parameter_2, parameter_3, parameter_5, parameter_0, parameter_1, parameter_7, parameter_8, data_0, data_3, data_6, data_1, data_4, data_7, data_2, data_5, data_8):
        return self.builtin_module_414_0_0(parameter_6, parameter_4, parameter_2, parameter_3, parameter_5, parameter_0, parameter_1, parameter_7, parameter_8, data_0, data_3, data_6, data_1, data_4, data_7, data_2, data_5, data_8)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_414_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_6
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_0
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # data_0
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # data_3
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # data_6
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # data_1
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # data_4
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # data_7
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # data_2
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # data_5
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # data_8
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_6
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_0
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # data_0
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            # data_3
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            # data_6
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            # data_1
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            # data_4
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            # data_7
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            # data_2
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            # data_5
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            # data_8
            paddle.static.InputSpec(shape=[None], dtype='float32'),
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