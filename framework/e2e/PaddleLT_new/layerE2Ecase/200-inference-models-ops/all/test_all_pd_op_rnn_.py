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
class PrimitiveOp_6b8198b29ecade32686844eda124cb46(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_4):
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        input_2 = [arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7]
        input_3 = None
        input_4 = arg_4
        return (lambda x, f: f(x))(paddle._C_ops.rnn(input_0, input_1, input_2, None, input_4, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

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
class TestPrimitiveOp_dee42a58fcaa0df2d022bc35f3a1884a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b8198b29ecade32686844eda124cb46
    def get_inputs(self):
        return [
            paddle.uniform([40, 6, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 6, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 6, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_13f626a3c81ef76706086f0b37a9124d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b8198b29ecade32686844eda124cb46
    def get_inputs(self):
        return [
            paddle.uniform([40, 1, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_b950ec76ffcfa5b6d0104ac9c1c1bad1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_4):
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        input_2 = [arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7]
        input_3 = None
        input_4 = arg_4
        return (lambda x, f: f(x))(paddle._C_ops.rnn(input_0, input_1, input_2, None, input_4, float('0'), True, 512, 256, 1, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

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
class TestPrimitiveOp_f4be1d8a425e0c661a6b959e21df6365(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b950ec76ffcfa5b6d0104ac9c1c1bad1
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

class PrimitiveOp_a93ec554fe6577e67536c1712f927c33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_4):
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        input_2 = [arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7]
        input_3 = None
        input_4 = arg_4
        return (lambda x, f: f(x))(paddle._C_ops.rnn(input_0, input_1, input_2, None, input_4, float('0'), True, 256, 256, 1, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

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
class TestPrimitiveOp_e0f9b252d68e56bb37ef448caa1f9e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a93ec554fe6577e67536c1712f927c33
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

class PrimitiveOp_f0ba6092c9be8d8d2ea398e395f85e6c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_2_8, arg_2_9, arg_2_10, arg_2_11, arg_2_12, arg_2_13, arg_2_14, arg_2_15, arg_4):
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        input_2 = [arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_2_8, arg_2_9, arg_2_10, arg_2_11, arg_2_12, arg_2_13, arg_2_14, arg_2_15]
        input_3 = None
        input_4 = arg_4
        return (lambda x, f: f(x))(paddle._C_ops.rnn(input_0, input_1, input_2, None, input_4, float('0'), True, 512, 256, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

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
class TestPrimitiveOp_734c908e92a88b19ed46757c12194373(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0ba6092c9be8d8d2ea398e395f85e6c
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

class PrimitiveOp_103c411e1e1114dd3d547667b5c2fc32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_2_8, arg_2_9, arg_2_10, arg_2_11, arg_2_12, arg_2_13, arg_2_14, arg_2_15, arg_4):
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        input_2 = [arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_2_8, arg_2_9, arg_2_10, arg_2_11, arg_2_12, arg_2_13, arg_2_14, arg_2_15]
        input_3 = None
        input_4 = arg_4
        return (lambda x, f: f(x))(paddle._C_ops.rnn(input_0, input_1, input_2, None, input_4, float('0'), True, 480, 96, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

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
class TestPrimitiveOp_cbfac81c229750be245e6a9d1941c843(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_103c411e1e1114dd3d547667b5c2fc32
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

class PrimitiveOp_1e94daeae8d3ff039e84e3a07663a507(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_4):
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        input_2 = [arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7]
        input_3 = None
        input_4 = arg_4
        return (lambda x, f: f(x))(paddle._C_ops.rnn(input_0, input_1, input_2, None, input_4, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40, None, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[2, None, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[2, None, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='uint8'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b74bd5396d991916225ce8a46e0b5723(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e94daeae8d3ff039e84e3a07663a507
    def get_inputs(self):
        return [
            paddle.uniform([40, 6, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 6, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 6, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a29cefe1c66c1eb8786631dfad88eb9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e94daeae8d3ff039e84e3a07663a507
    def get_inputs(self):
        return [
            paddle.uniform([40, 1, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_2feb485bfd9f9c07cd8b0f30df76653e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_4):
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        input_2 = [arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7]
        input_3 = None
        input_4 = arg_4
        return (lambda x, f: f(x))(paddle._C_ops.rnn(input_0, input_1, input_2, None, input_4, float('0'), True, 512, 256, 1, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

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
class TestPrimitiveOp_4fe5d36b02e74938695d2ef49052cfa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2feb485bfd9f9c07cd8b0f30df76653e
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

class PrimitiveOp_e36353f78fb272c5de1710ce7ef1b087(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_4):
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        input_2 = [arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7]
        input_3 = None
        input_4 = arg_4
        return (lambda x, f: f(x))(paddle._C_ops.rnn(input_0, input_1, input_2, None, input_4, float('0'), True, 256, 256, 1, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

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
class TestPrimitiveOp_d5a160619488d2590e270b42186c6f51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e36353f78fb272c5de1710ce7ef1b087
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

class PrimitiveOp_f17dfad96e52530a12837eaca3a7f2d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_2_8, arg_2_9, arg_2_10, arg_2_11, arg_2_12, arg_2_13, arg_2_14, arg_2_15, arg_4):
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        input_2 = [arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_2_8, arg_2_9, arg_2_10, arg_2_11, arg_2_12, arg_2_13, arg_2_14, arg_2_15]
        input_3 = None
        input_4 = arg_4
        return (lambda x, f: f(x))(paddle._C_ops.rnn(input_0, input_1, input_2, None, input_4, float('0'), True, 512, 256, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

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
class TestPrimitiveOp_0d30c4af967618d39a3d364c25f39818(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f17dfad96e52530a12837eaca3a7f2d4
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

class PrimitiveOp_0be9406f923967cc3f8ba5a42ed5dabb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_2_8, arg_2_9, arg_2_10, arg_2_11, arg_2_12, arg_2_13, arg_2_14, arg_2_15, arg_4):
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        input_2 = [arg_2_0, arg_2_1, arg_2_2, arg_2_3, arg_2_4, arg_2_5, arg_2_6, arg_2_7, arg_2_8, arg_2_9, arg_2_10, arg_2_11, arg_2_12, arg_2_13, arg_2_14, arg_2_15]
        input_3 = None
        input_4 = arg_4
        return (lambda x, f: f(x))(paddle._C_ops.rnn(input_0, input_1, input_2, None, input_4, float('0'), True, 480, 96, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

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
class TestPrimitiveOp_8147f6bec14d89a47661956a47046a78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0be9406f923967cc3f8ba5a42ed5dabb
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


if __name__ == '__main__':
    unittest.main()