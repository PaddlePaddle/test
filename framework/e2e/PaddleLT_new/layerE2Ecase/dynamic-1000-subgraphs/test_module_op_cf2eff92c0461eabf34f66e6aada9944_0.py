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
    return [95][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_336_0_0(self, data_0, data_1):

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # pd_op.split_with_num: ([-1xf32, -1xf32, -1xf32, -1xf32]) <- (-1xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(data_0, 4, full_0)

        # builtin.split: (-1xf32, -1xf32, -1xf32, -1xf32) <- ([-1xf32, -1xf32, -1xf32, -1xf32])
        split_0, split_1, split_2, split_3, = split_with_num_0

        # pd_op.split_with_num: ([-1xf32, -1xf32, -1xf32, -1xf32]) <- (-1xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(data_1, 4, assign_0)

        # builtin.split: (-1xf32, -1xf32, -1xf32, -1xf32) <- ([-1xf32, -1xf32, -1xf32, -1xf32])
        split_4, split_5, split_6, split_7, = split_with_num_1

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_0 = split_0 + split_2

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], float('0.5'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_1 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_2 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_3 = full_1

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(add_0, full_1, float('0'), True)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_1 = split_1 + split_3

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(add_1, assign_3, float('0'), True)

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_0 = split_2 - split_0

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_1 = split_3 - split_1

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_2 = split_4 + split_6

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(add_2, assign_2, float('0'), True)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_3 = split_5 + split_7

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(add_3, assign_1, float('0'), True)

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_2 = split_6 - split_4

        # pd_op.assign: (-1xf32) <- (-1xf32)
        assign_4 = subtract_2

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_3 = split_7 - split_5

        # pd_op.assign: (-1xf32) <- (-1xf32)
        assign_5 = subtract_3

        # pd_op.maximum: (-1xf32) <- (-1xf32, -1xf32)
        maximum_0 = paddle.maximum(split_0, split_2)

        # pd_op.maximum: (-1xf32) <- (-1xf32, -1xf32)
        maximum_1 = paddle.maximum(split_1, split_3)

        # pd_op.maximum: (-1xf32) <- (-1xf32, -1xf32)
        maximum_2 = paddle.maximum(split_0, split_4)

        # pd_op.maximum: (-1xf32) <- (-1xf32, -1xf32)
        maximum_3 = paddle.maximum(split_1, split_5)

        # pd_op.minimum: (-1xf32) <- (-1xf32, -1xf32)
        minimum_0 = paddle._C_ops.minimum(maximum_0, split_6)

        # pd_op.minimum: (-1xf32) <- (-1xf32, -1xf32)
        minimum_1 = paddle._C_ops.minimum(maximum_1, split_7)

        # pd_op.minimum: (-1xf32) <- (-1xf32, -1xf32)
        minimum_2 = paddle._C_ops.minimum(split_0, split_4)

        # pd_op.minimum: (-1xf32) <- (-1xf32, -1xf32)
        minimum_3 = paddle._C_ops.minimum(split_1, split_5)

        # pd_op.maximum: (-1xf32) <- (-1xf32, -1xf32)
        maximum_4 = paddle.maximum(maximum_0, split_6)

        # pd_op.maximum: (-1xf32) <- (-1xf32, -1xf32)
        maximum_5 = paddle.maximum(maximum_1, split_7)

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_4 = minimum_0 - maximum_2

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_5 = minimum_1 - maximum_3

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_0 = subtract_4 * subtract_5

        # pd_op.greater_than: (-1xb) <- (-1xf32, -1xf32)
        greater_than_0 = minimum_0 > maximum_2

        # pd_op.cast: (-1xf32) <- (-1xb)
        cast_0 = paddle._C_ops.cast(greater_than_0, paddle.float32)

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_1 = multiply_0 * cast_0

        # pd_op.greater_than: (-1xb) <- (-1xf32, -1xf32)
        greater_than_1 = minimum_1 > maximum_3

        # pd_op.cast: (-1xf32) <- (-1xb)
        cast_1 = paddle._C_ops.cast(greater_than_1, paddle.float32)

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_2 = multiply_1 * cast_1

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_6 = maximum_0 - split_0

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_7 = maximum_1 - split_1

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_3 = subtract_6 * subtract_7

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_4 = assign_4 * assign_5

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_4 = multiply_3 + multiply_4

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_8 = add_4 - multiply_2

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_6 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_7 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_8 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_9 = full_2

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(subtract_8, full_2, float('1e-10'), True)

        # pd_op.divide: (-1xf32) <- (-1xf32, -1xf32)
        divide_0 = multiply_2 / scale_4

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_9 = scale_0 - scale_2

        # pd_op.assign: (-1xf32) <- (-1xf32)
        assign_10 = subtract_9

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_5 = subtract_9 * assign_10

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_10 = scale_1 - scale_3

        # pd_op.assign: (-1xf32) <- (-1xf32)
        assign_11 = subtract_10

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_6 = subtract_10 * assign_11

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_5 = multiply_5 + multiply_6

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_11 = maximum_4 - minimum_2

        # pd_op.assign: (-1xf32) <- (-1xf32)
        assign_12 = subtract_11

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_7 = subtract_11 * assign_12

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_12 = maximum_5 - minimum_3

        # pd_op.assign: (-1xf32) <- (-1xf32)
        assign_13 = subtract_12

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_8 = subtract_12 * assign_13

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_6 = multiply_7 + multiply_8

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(add_5, assign_9, float('1e-10'), True)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(add_6, assign_8, float('1e-10'), True)

        # pd_op.divide: (-1xf32) <- (-1xf32, -1xf32)
        divide_1 = scale_5 / scale_6

        # pd_op.divide: (-1xf32) <- (-1xf32, -1xf32)
        divide_2 = subtract_2 / subtract_3

        # pd_op.divide: (-1xf32) <- (-1xf32, -1xf32)
        divide_3 = subtract_0 / subtract_1

        # pd_op.atan: (-1xf32) <- (-1xf32)
        atan_0 = paddle._C_ops.atan(divide_2)

        # pd_op.atan: (-1xf32) <- (-1xf32)
        atan_1 = paddle._C_ops.atan(divide_3)

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_13 = atan_0 - atan_1

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('0.405285'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(subtract_13, full_3, float('0'), True)

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_9 = scale_7 * subtract_13

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('-1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_14 = full_4

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(divide_0, full_4, float('1'), True)

        # pd_op.assign: (-1xf32) <- (-1xf32)
        assign_15 = scale_8

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_7 = scale_8 + multiply_9

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(add_7, assign_7, float('1e-10'), True)

        # pd_op.divide: (-1xf32) <- (-1xf32, -1xf32)
        divide_4 = multiply_9 / scale_9

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_10 = divide_4 * multiply_9

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_8 = assign_15 + multiply_10

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_9 = add_8 + divide_1

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(add_9, assign_6, float('0'), True)

        # pd_op.mean: (xf32) <- (-1xf32)
        mean_0 = paddle._C_ops.mean(scale_10, [], False)

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full([1], float('10'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(mean_0, full_5, float('0'), True)
        return full_0, split_0, split_1, split_2, split_3, assign_0, split_4, split_5, split_6, split_7, full_1, scale_0, assign_3, scale_1, subtract_0, subtract_1, assign_2, scale_2, assign_1, scale_3, subtract_2, subtract_3, maximum_0, maximum_1, maximum_2, maximum_3, minimum_0, minimum_1, minimum_2, minimum_3, maximum_4, maximum_5, subtract_4, subtract_5, multiply_0, cast_0, multiply_1, cast_1, multiply_2, subtract_6, subtract_7, multiply_3, assign_4, assign_5, multiply_4, add_4, full_2, scale_4, divide_0, subtract_9, assign_10, multiply_5, subtract_10, assign_11, multiply_6, subtract_11, assign_12, multiply_7, subtract_12, assign_13, multiply_8, assign_9, scale_5, assign_8, scale_6, divide_1, divide_2, divide_3, atan_0, atan_1, subtract_13, full_3, scale_7, multiply_9, full_4, scale_8, assign_7, scale_9, divide_4, multiply_10, assign_14, assign_15, add_8, assign_6, scale_10, full_5, scale_11



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

    def forward(self, data_0, data_1):
        return self.builtin_module_336_0_0(data_0, data_1)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_336_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # data_0
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # data_1
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # data_0
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            # data_1
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