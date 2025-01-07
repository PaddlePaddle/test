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
class PrimitiveOp_3c279fcf624358af75f98c991d4a536a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return paddle._C_ops.stack(input_0, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_457fa1bf07d2bb5a7e35560806064057(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c279fcf624358af75f98c991d4a536a
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(26, dtype='int32').reshape([]),
            paddle.to_tensor(512, dtype='int32').reshape([]),
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

class PrimitiveOp_fe14253db62883325e72e0008e5ed016(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e9c547a1d9bf1f4f39702acca9401873(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe14253db62883325e72e0008e5ed016
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(26, dtype='int32').reshape([]),
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
class TestPrimitiveOp_aa29759bd218816429c7f9bccb942a0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe14253db62883325e72e0008e5ed016
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(256, dtype='int32').reshape([]),
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
class TestPrimitiveOp_d0407ece62d9d84d06de8a1108fbeb2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c279fcf624358af75f98c991d4a536a
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(501, dtype='int32').reshape([]),
            paddle.to_tensor(30, dtype='int32').reshape([]),
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
class TestPrimitiveOp_83761faddbe94c0308aef07cbbec47bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c279fcf624358af75f98c991d4a536a
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(501, dtype='int32').reshape([]),
            paddle.to_tensor(4, dtype='int32').reshape([]),
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

class PrimitiveOp_75859968a5b344dd7084fc4f97f2403e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0):
        input_0 = [arg_0_0]
        return paddle._C_ops.stack(input_0, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0a722fc66ea96cba127f30f4c651d42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75859968a5b344dd7084fc4f97f2403e
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
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
class TestPrimitiveOp_b8dfd80de89084a85ed3b1b92e398d88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe14253db62883325e72e0008e5ed016
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(40, dtype='int32').reshape([]),
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
class TestPrimitiveOp_c85e283af7c5150d48b85a03711726a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c279fcf624358af75f98c991d4a536a
    def get_inputs(self):
        return [
            paddle.to_tensor(2, dtype='int32').reshape([]),
            paddle.to_tensor(6, dtype='int32').reshape([]),
            paddle.to_tensor(128, dtype='int32').reshape([]),
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
class TestPrimitiveOp_427c2836ccc7d2ff48f727f1bd0782aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c279fcf624358af75f98c991d4a536a
    def get_inputs(self):
        return [
            paddle.to_tensor(2, dtype='int32').reshape([]),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(128, dtype='int32').reshape([]),
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

class PrimitiveOp_6769b7dc79ec34e4f7b3a06512394444(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_da96f0ab0cd3c5fc4f11077ea4ada5ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6769b7dc79ec34e4f7b3a06512394444
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_87879e80cca92dcb8a1263c64b93b088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe14253db62883325e72e0008e5ed016
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(1, dtype='int32').reshape([]),
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

class PrimitiveOp_650dedbe87862afc2fcfc017666cf1ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float64'),
            paddle.static.InputSpec(shape=[None], dtype='float64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d9244bda9a9db80dcb34001988ac8e39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_650dedbe87862afc2fcfc017666cf1ba
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype='float64').reshape([10]),
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
class TestPrimitiveOp_1970776f8140ea1c81c2a8ae530cb2af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_650dedbe87862afc2fcfc017666cf1ba
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float64').reshape([10]),
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

class PrimitiveOp_c5a9cd481ddfb12d017f1691531161d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float64'),
            paddle.static.InputSpec(shape=[None, None], dtype='float64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_83ba48cc06c504b721d882f736e25245(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5a9cd481ddfb12d017f1691531161d9
    def get_inputs(self):
        return [
            paddle.uniform([100, 32], dtype='float64', min=0, max=0.5),
            paddle.uniform([100, 32], dtype='float64', min=0, max=0.5),
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

class PrimitiveOp_ab157d187a43382ed9becd90d46416bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e43a1f9db86969169373660e3bdde5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab157d187a43382ed9becd90d46416bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_5e904c94e2fa425ee397850af9483725(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c279fcf624358af75f98c991d4a536a
    def get_inputs(self):
        return [
            paddle.to_tensor(2, dtype='int32').reshape([]),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(256, dtype='int32').reshape([]),
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
class TestPrimitiveOp_5cb480d8a924b93c39f0ff25f1ea6612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c279fcf624358af75f98c991d4a536a
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int32').reshape([]),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(96, dtype='int32').reshape([]),
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
class TestPrimitiveOp_c82e3a2d3553bfd7ad0ee9b5adb7640f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe14253db62883325e72e0008e5ed016
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(96, dtype='int32').reshape([]),
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

class PrimitiveOp_9629cc98e1893616e9e54ec96c0f1ed7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8a8cb107b8ef131557229ad4cb4fc08c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9629cc98e1893616e9e54ec96c0f1ed7
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[48, 80], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[48, 80], dtype='int64'), 'int64'),
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

class PrimitiveOp_7ff54aeb46e416d7a815006a705ca84a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fa88a72684524776e9c5fdc7dec5b315(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ff54aeb46e416d7a815006a705ca84a
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 48, 80], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_ca2a7339fdfea6cfacd9ca489a04fa0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9629cc98e1893616e9e54ec96c0f1ed7
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[96, 160], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[96, 160], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_663f5b0c5c936b8b341bd6e2ee43cf10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ff54aeb46e416d7a815006a705ca84a
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 96, 160], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_826d8c46ac0b5d55189f68f45eae0989(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9629cc98e1893616e9e54ec96c0f1ed7
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[192, 320], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[192, 320], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_f837dfae74092b61ebf9c740c09113b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ff54aeb46e416d7a815006a705ca84a
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 192, 320], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_0db994d38a08ad1db4419e5f9c42af90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9629cc98e1893616e9e54ec96c0f1ed7
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[180, 320], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[180, 320], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_1b4052b247b9657499e255a90d7ad3fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ff54aeb46e416d7a815006a705ca84a
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_309d6894ea5bfd1f15619c57aeda98ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d57376d4cbb1b5a4d40adbae02bcbda9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_309d6894ea5bfd1f15619c57aeda98ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 720, 1280], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 3, 720, 1280], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_0f7ca421af1b20adcce4483d049831a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_efb08ada5f26eaa3e12b16a3fffa61a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f7ca421af1b20adcce4483d049831a1
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[16800], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[16800], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_2cade5a402e67b5da0f90fa810ef453f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f7ca421af1b20adcce4483d049831a1
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[4200], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[4200], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_654c436ece0de80c9f60cbff6f1f84fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f7ca421af1b20adcce4483d049831a1
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1050], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[1050], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_caa7ed155b1d19b30a2bbe3110b2cfd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f7ca421af1b20adcce4483d049831a1
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[273], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[273], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_e90137ab9a04a898275de02c2a7851a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f7ca421af1b20adcce4483d049831a1
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[77], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[77], dtype='int64'), 'int64'),
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

class PrimitiveOp_80b1fc4c999a818f449d8583d8383364(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_25f9b03dc5c2df2598b37553f206f507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80b1fc4c999a818f449d8583d8383364
    def get_inputs(self):
        return [
            paddle.uniform([1, 16800], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 16800], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 16800], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 16800], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_dc403b110350bb55b926e095f5294f88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80b1fc4c999a818f449d8583d8383364
    def get_inputs(self):
        return [
            paddle.uniform([1, 4200], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 4200], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 4200], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 4200], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_aebfdff694d4b9641294e7109ae7d761(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80b1fc4c999a818f449d8583d8383364
    def get_inputs(self):
        return [
            paddle.uniform([1, 1050], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1050], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1050], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1050], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_2893b9cb8640ff2b2078628f21a5b6ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80b1fc4c999a818f449d8583d8383364
    def get_inputs(self):
        return [
            paddle.uniform([1, 273], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 273], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 273], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 273], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_d72d72af6b8b15af9a53e944cfb2af3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80b1fc4c999a818f449d8583d8383364
    def get_inputs(self):
        return [
            paddle.uniform([1, 77], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 77], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 77], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 77], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_71cc0f07db57866d77e7b57c3e2babf9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_20323862a65777b9d20fe541bf7a059c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71cc0f07db57866d77e7b57c3e2babf9
    def get_inputs(self):
        return [
            paddle.uniform([80, 80], dtype='float16', min=0, max=0.5),
            paddle.uniform([80, 80], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_1fc004a8ba7dad2a9493d09a8c64d11f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71cc0f07db57866d77e7b57c3e2babf9
    def get_inputs(self):
        return [
            paddle.uniform([40, 40], dtype='float16', min=0, max=0.5),
            paddle.uniform([40, 40], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_7633545e5e89551273d16ae32ffd6d6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71cc0f07db57866d77e7b57c3e2babf9
    def get_inputs(self):
        return [
            paddle.uniform([20, 20], dtype='float16', min=0, max=0.5),
            paddle.uniform([20, 20], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_f4562a3fdc807d6e1bd4d5065f470726(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ad30119ebf8da2dd9ae504a140b89f3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4562a3fdc807d6e1bd4d5065f470726
    def get_inputs(self):
        return [
            paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9391f12ab2406f96940a794c47af42ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4562a3fdc807d6e1bd4d5065f470726
    def get_inputs(self):
        return [
            paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a4dfed59c9c6e564bc00340c4d0357bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4562a3fdc807d6e1bd4d5065f470726
    def get_inputs(self):
        return [
            paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5cf019ef85c580eb54eb9e1b85d5ec4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c279fcf624358af75f98c991d4a536a
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(501, dtype='int32').reshape([]),
            paddle.to_tensor(256, dtype='int32').reshape([]),
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

class PrimitiveOp_22261e13e6b63718b7db952862a1081a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bdf742e662d1bf02f18b943d50df3fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22261e13e6b63718b7db952862a1081a
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_376268c382685a9a3361cfe53ab481fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_343f475624082a50ee5322129a1e03bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_376268c382685a9a3361cfe53ab481fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 16800], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 16800], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 16800], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 16800], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_64960acc0ae1533d1c00640bd9cbe8af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_376268c382685a9a3361cfe53ab481fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4200], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8f5dff39dd866068742e4956828567e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_376268c382685a9a3361cfe53ab481fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1050], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1050], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1050], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1050], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8a82b3bd0258d4a17766e94232fbe649(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_376268c382685a9a3361cfe53ab481fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 273], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 273], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 273], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 273], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d0f7ad022a4b5d6013b488a5805a405a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_376268c382685a9a3361cfe53ab481fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 77], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 77], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 77], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 77], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_b82a934ec9f0a629bd64336b032cefef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_555fa9e40c57688b5d69a8dc10d30334(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b82a934ec9f0a629bd64336b032cefef
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7bc4f8e69482d7917295aff609444258(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b82a934ec9f0a629bd64336b032cefef
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6fb9f9ffa50529db22615cf223e72100(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b82a934ec9f0a629bd64336b032cefef
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5592740ace4c7134d789dc9b50f242e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b82a934ec9f0a629bd64336b032cefef
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_3ca1131b5cd2d430cd169e5e4f5f5461(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4e39cdba247c98bd467105e7a37791de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ca1131b5cd2d430cd169e5e4f5f5461
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 720, 1280], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 720, 1280], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_450e50b5b2c41128ddaafc1388186e9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e371f8a991f8dff12d50f5965d7bc82f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_450e50b5b2c41128ddaafc1388186e9b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_d1921b6594a40db38b8432aa2a508798(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0c8cf1b87b9d6e5b42fe0439a30918c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1921b6594a40db38b8432aa2a508798
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_a9f0881d443ed9edbdd48a49615a839a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10], dtype='float64'),
            paddle.static.InputSpec(shape=[10], dtype='float64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8a29a0185224c73ee5c2351208b82dc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9f0881d443ed9edbdd48a49615a839a
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype='float64').reshape([10]),
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
class TestPrimitiveOp_122747f9fdc0ddfda092f12be88eafb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9f0881d443ed9edbdd48a49615a839a
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float64').reshape([10]),
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

class PrimitiveOp_1fd84281c9c11cc6c918c79c3f7071a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 32], dtype='float64'),
            paddle.static.InputSpec(shape=[100, 32], dtype='float64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_95797d421e5de1bf11a050038ce38592(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fd84281c9c11cc6c918c79c3f7071a3
    def get_inputs(self):
        return [
            paddle.uniform([100, 32], dtype='float64', min=0, max=0.5),
            paddle.uniform([100, 32], dtype='float64', min=0, max=0.5),
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

class PrimitiveOp_523c0ce1a82b41a8460fcf2edfe67bf7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b0d3526af59ce26a825dbfb04e81bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_523c0ce1a82b41a8460fcf2edfe67bf7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_418bebd5ed0d22d6398928dc48fa3c24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[48, 80], dtype='int64'),
            paddle.static.InputSpec(shape=[48, 80], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9a9a137b656da29425b64ea1f053f604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_418bebd5ed0d22d6398928dc48fa3c24
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[48, 80], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[48, 80], dtype='int64'), 'int64'),
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

class PrimitiveOp_57658105089dfcedff4bca6b1ebd7add(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 80], dtype='float16'),
            paddle.static.InputSpec(shape=[1, 48, 80], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b6cd05a09a498f7129a181ea28fb3fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57658105089dfcedff4bca6b1ebd7add
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 48, 80], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_4392290d59c69f23ebb30b027a3d0c52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[96, 160], dtype='int64'),
            paddle.static.InputSpec(shape=[96, 160], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7dd8c381b40caa43e5cf5159ae09a38e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4392290d59c69f23ebb30b027a3d0c52
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[96, 160], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[96, 160], dtype='int64'), 'int64'),
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

class PrimitiveOp_054414d8920c6ae89837152ddcd3c060(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 160], dtype='float16'),
            paddle.static.InputSpec(shape=[1, 96, 160], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fa3388218816b8bdd8d63c228860fa8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_054414d8920c6ae89837152ddcd3c060
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 96, 160], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_7f7f836c434bc20a1c5e47b516b0db51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[192, 320], dtype='int64'),
            paddle.static.InputSpec(shape=[192, 320], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf82a145cb65001e5146503b921b8a45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f7f836c434bc20a1c5e47b516b0db51
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[192, 320], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[192, 320], dtype='int64'), 'int64'),
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

class PrimitiveOp_08d1fc531e9cffbde81f0f4b433cb9c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[1, 192, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0cd9d17d8c0860f571a917c38828c854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08d1fc531e9cffbde81f0f4b433cb9c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 192, 320], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_506dd900d05f31945c1aeedf8e8a43e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[180, 320], dtype='int64'),
            paddle.static.InputSpec(shape=[180, 320], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2f747ba44f41222f85793b66cd61489f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_506dd900d05f31945c1aeedf8e8a43e1
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[180, 320], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[180, 320], dtype='int64'), 'int64'),
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

class PrimitiveOp_72459f194bd6b9967d927180375ec5f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 180, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[1, 180, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7169df9e885d852333a98a764c4c95c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72459f194bd6b9967d927180375ec5f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_0669c93d4506a4a658a3406e8cd92d03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 720, 1280], dtype='float16'),
            paddle.static.InputSpec(shape=[1, 3, 720, 1280], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5a006f56139e1a2fee6109cae4fe61f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0669c93d4506a4a658a3406e8cd92d03
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 720, 1280], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 3, 720, 1280], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_2af03150064b421b2d71c9274c1334f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80, 80], dtype='float16'),
            paddle.static.InputSpec(shape=[80, 80], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9f94e6565a3fa5d7dcb1b81fc8380b8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af03150064b421b2d71c9274c1334f9
    def get_inputs(self):
        return [
            paddle.uniform([80, 80], dtype='float16', min=0, max=0.5),
            paddle.uniform([80, 80], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_14b9d8333d0e3c3bf2246fc9026a31dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40, 40], dtype='float16'),
            paddle.static.InputSpec(shape=[40, 40], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_601e52982ede487ee13f87a619863f40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14b9d8333d0e3c3bf2246fc9026a31dd
    def get_inputs(self):
        return [
            paddle.uniform([40, 40], dtype='float16', min=0, max=0.5),
            paddle.uniform([40, 40], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_a53aea7634d7c02d587aef449b840654(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 20], dtype='float16'),
            paddle.static.InputSpec(shape=[20, 20], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d94e496080dc3e57a5aeffae8a4f594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a53aea7634d7c02d587aef449b840654
    def get_inputs(self):
        return [
            paddle.uniform([20, 20], dtype='float16', min=0, max=0.5),
            paddle.uniform([20, 20], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_2cea06b6f5afe7b89f6b47f16ac8a303(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[80, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_acdf880c19faeb73e0cc61dd773ba44c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cea06b6f5afe7b89f6b47f16ac8a303
    def get_inputs(self):
        return [
            paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_b1aa40ecca42422ba28ea17a7353ecef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7bdee52edd6927b2bfde6a6e968f7f3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1aa40ecca42422ba28ea17a7353ecef
    def get_inputs(self):
        return [
            paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_fc9e66840acad803e10bdb2c9f2bb760(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[20, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d721ad6caac645d78b32f1bda4922f39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc9e66840acad803e10bdb2c9f2bb760
    def get_inputs(self):
        return [
            paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_3c0f674954540826db14a99cdd0f9f74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e13a54994f2c72e97e552b210c6fcc26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c0f674954540826db14a99cdd0f9f74
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_3b6a7b9b261e5b541a8b29ede926062f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 48, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14800072a33097727740fc71e566e80e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6a7b9b261e5b541a8b29ede926062f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 80], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_f32ae5045a41c4238d56c54574cd6f0c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c1cc082b081a963eba0b7914c4139b3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f32ae5045a41c4238d56c54574cd6f0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_f9fe0e1ceaef41d2fcf42c6326d90d05(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 192, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f248088f958bafb1ba869d05679d0d60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9fe0e1ceaef41d2fcf42c6326d90d05
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_e7af695db20fb55c8e46ddc1aa8f29dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 180, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 180, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_72d594b6bdee2618a140e48ad3e054f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7af695db20fb55c8e46ddc1aa8f29dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_8d3d5d62ad425335232bd62a46c47c4f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 720, 1280], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 720, 1280], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fdef9ffc87f50c90add3e04d3a9c559d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d3d5d62ad425335232bd62a46c47c4f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 720, 1280], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 720, 1280], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_147c479eab55ebcfbf3a3e4de5cfc8b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8356a4f6e2979aaf02432724d3596eab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_147c479eab55ebcfbf3a3e4de5cfc8b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
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