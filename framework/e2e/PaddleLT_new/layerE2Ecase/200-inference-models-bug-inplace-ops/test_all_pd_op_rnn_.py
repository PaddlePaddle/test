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
import itertools

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
        return True, "last stage failed."
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
    ATHENA_ENABLE_TRY_RUN=False,
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
    if enable_cinn is None:
        return True
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

def ApplyToStatic(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        input_spec=net.get_input_spec(),
        build_strategy=build_strategy,
        full_graph=True,
    )

class InstanceTrait:

    @classmethod
    def instance(cls):
        if cls.instance_ is None:
            cls.instance_ = cls()
        return cls.instance_

    @classmethod
    def static_instance_with_cinn(cls):
        if cls.static_instance_with_cinn_ is None:
            cls.static_instance_with_cinn_ = ApplyToStatic(
                cls.instance(),
                use_cinn=True
            )
        return cls.static_instance_with_cinn_

    @classmethod
    def static_instance_without_cinn(cls):
        if cls.static_instance_without_cinn_ is None:
            cls.static_instance_without_cinn_ = ApplyToStatic(
                cls.instance(),
                use_cinn=False
            )
        return cls.static_instance_without_cinn_


class CinnTestBase:

    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def _test_entry(self):
        dy_outs = self.train(use_cinn=False)
        cinn_outs = self.train(use_cinn=GetEnvVarEnableCinn())

        for cinn_out, dy_out in zip(cinn_outs, dy_outs):
          if type(cinn_out) is list and type(dy_out) is list:
            for x, y in zip(cinn_out, dy_out):
              self.assert_all_close(x, y)
          else:
            self.assert_all_close(cinn_out, dy_out)

    def train(self, use_cinn):
        if GetEnvVarEnableJit():
            net = self.prepare_static_net(use_cinn)
        else:
            net = self.prepare_net()
        paddle.seed(2024)
        out = net(*self.inputs)
        return out
    
    def prepare_data(self):
        self.inputs = self.get_inputs()
        for input in self.inputs:
            input.stop_gradient = True

    def prepare_net(self):
        return self.get_test_class().instance()

    def prepare_static_net(self, use_cinn):
        if use_cinn:
            return self.get_test_class().static_instance_with_cinn()
        else:
            return self.get_test_class().static_instance_without_cinn()

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
class PrimitiveOp_7fcf25fa56bbdc608ffb9f12edc85280(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_2_8, arg_2_9, arg_2_10, arg_2_11, arg_2_12, arg_2_13, arg_2_14, arg_2_15, arg_4):
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        input_2 = [arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_2_8, arg_2_9, arg_2_10, arg_2_11, arg_2_12, arg_2_13, arg_2_14, arg_2_15]
        input_3 = None
        input_4 = arg_4
        return (lambda x, f: f(x))(paddle._C_ops.rnn_(input_0, input_1, input_2, None, input_4, float('0'), True, 512, 256, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='uint8'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2a4942c47cac19508b36b9a0305bcb15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fcf25fa56bbdc608ffb9f12edc85280
    def get_inputs(self):
        return [
            paddle.uniform([25, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(2, dtype='uint8').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6c51a1b747f43917a49dd883db19e633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fcf25fa56bbdc608ffb9f12edc85280
    def get_inputs(self):
        return [
            paddle.uniform([25, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0, dtype='uint8').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0feca09cf4089703e200a0df153fb5fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_2_8, arg_2_9, arg_2_10, arg_2_11, arg_2_12, arg_2_13, arg_2_14, arg_2_15, arg_4):
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        input_2 = [arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_2_8, arg_2_9, arg_2_10, arg_2_11, arg_2_12, arg_2_13, arg_2_14, arg_2_15]
        input_3 = None
        input_4 = arg_4
        return (lambda x, f: f(x))(paddle._C_ops.rnn_(input_0, input_1, input_2, None, input_4, float('0'), True, 480, 96, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='uint8'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c14f4d22dd0bff7d03495540c32f5270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0feca09cf4089703e200a0df153fb5fc
    def get_inputs(self):
        return [
            paddle.uniform([25, 1, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 1, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 1, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0, dtype='uint8').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_765096dee906963bee64d24ef8bcfe33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_4):
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        input_2 = [arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7]
        input_3 = None
        input_4 = arg_4
        return (lambda x, f: f(x))(paddle._C_ops.rnn_(input_0, input_1, input_2, None, input_4, float('0'), True, 512, 256, 1, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='uint8'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f9f492341eb75da7535d53a0416e5568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_765096dee906963bee64d24ef8bcfe33
    def get_inputs(self):
        return [
            paddle.uniform([26, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(2, dtype='uint8').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_03ee612af40fd5304ec9df281704d294(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_4):
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        input_2 = [arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7]
        input_3 = None
        input_4 = arg_4
        return (lambda x, f: f(x))(paddle._C_ops.rnn_(input_0, input_1, input_2, None, input_4, float('0'), True, 256, 256, 1, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='uint8'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e7090218b7fd02b20b6486eb064a47fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03ee612af40fd5304ec9df281704d294
    def get_inputs(self):
        return [
            paddle.uniform([26, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(2, dtype='uint8').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2ce3e13aed01e93a6ef488b2f6d75211(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_765096dee906963bee64d24ef8bcfe33
    def get_inputs(self):
        return [
            paddle.uniform([26, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0, dtype='uint8').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_16dbbd3d78d91790cc9044aa650c794c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03ee612af40fd5304ec9df281704d294
    def get_inputs(self):
        return [
            paddle.uniform([26, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0, dtype='uint8').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f835adcb02f1055f74c37b68d074d6e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_2_8, arg_2_9, arg_2_10, arg_2_11, arg_2_12, arg_2_13, arg_2_14, arg_2_15, arg_4):
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        input_2 = [arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_2_8, arg_2_9, arg_2_10, arg_2_11, arg_2_12, arg_2_13, arg_2_14, arg_2_15]
        input_3 = None
        input_4 = arg_4
        return (lambda x, f: f(x))(paddle._C_ops.rnn_(input_0, input_1, input_2, None, input_4, float('0'), True, 512, 256, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[4, None, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[4, None, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='uint8'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5cd362c77b71a211039d4a4fceb19013(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f835adcb02f1055f74c37b68d074d6e9
    def get_inputs(self):
        return [
            paddle.uniform([25, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(2, dtype='uint8').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_18a4a522190c5ef5cc1e8897cf9f3c9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f835adcb02f1055f74c37b68d074d6e9
    def get_inputs(self):
        return [
            paddle.uniform([25, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0, dtype='uint8').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b65592e05f3afe12253ce90f13e0486b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_2_8, arg_2_9, arg_2_10, arg_2_11, arg_2_12, arg_2_13, arg_2_14, arg_2_15, arg_4):
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        input_2 = [arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_2_8, arg_2_9, arg_2_10, arg_2_11, arg_2_12, arg_2_13, arg_2_14, arg_2_15]
        input_3 = None
        input_4 = arg_4
        return (lambda x, f: f(x))(paddle._C_ops.rnn_(input_0, input_1, input_2, None, input_4, float('0'), True, 480, 96, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25, None, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[4, None, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[4, None, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='uint8'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f791d8209e4be539faa2fc57e5db1cc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b65592e05f3afe12253ce90f13e0486b
    def get_inputs(self):
        return [
            paddle.uniform([25, 1, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 1, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 1, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0, dtype='uint8').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0aec515d8deb281984eb78b77a2bcec3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_4):
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        input_2 = [arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7]
        input_3 = None
        input_4 = arg_4
        return (lambda x, f: f(x))(paddle._C_ops.rnn_(input_0, input_1, input_2, None, input_4, float('0'), True, 512, 256, 1, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[26, None, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[2, None, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[2, None, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='uint8'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_33a73633c74621f0fce2439af4c49241(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aec515d8deb281984eb78b77a2bcec3
    def get_inputs(self):
        return [
            paddle.uniform([26, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(2, dtype='uint8').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8422642519497d4f3aa4620c391cf1ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_4):
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        input_2 = [arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7]
        input_3 = None
        input_4 = arg_4
        return (lambda x, f: f(x))(paddle._C_ops.rnn_(input_0, input_1, input_2, None, input_4, float('0'), True, 256, 256, 1, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[26, None, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[2, None, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[2, None, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='uint8'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a2074ef8e8165d7685e4ff01a0202edf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8422642519497d4f3aa4620c391cf1ca
    def get_inputs(self):
        return [
            paddle.uniform([26, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(2, dtype='uint8').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1ed7fe9b24e661a24f1267a231cb123(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aec515d8deb281984eb78b77a2bcec3
    def get_inputs(self):
        return [
            paddle.uniform([26, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0, dtype='uint8').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f292a3ec73230ca3265e1f5bd99f137e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8422642519497d4f3aa4620c391cf1ca
    def get_inputs(self):
        return [
            paddle.uniform([26, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0, dtype='uint8').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()


if __name__ == '__main__':
    unittest.main()