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
class PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e6c832ebb948c6884bca1d02ede63ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 64, 64], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_45c355eddc3fcb65de1044f0d229108b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = -1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_893bcc96db059412502e5787bc4fdfe9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45c355eddc3fcb65de1044f0d229108b
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21504, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8ce6bbfa024b12e6d652f009de88b941(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 24, 36], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_00082eb7ec4367861c66cd6d56507571(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 24, 36], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_d5ec4efca318aba690efc1d510e89395(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
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
class TestPrimitiveOp_a55ed8b514057852008e25b47cbac0d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5ec4efca318aba690efc1d510e89395
    def get_inputs(self):
        return [
            paddle.uniform([43, 448, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_cda29f2042bf32fa9f8c2460d6ce16ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
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
class TestPrimitiveOp_57d74b363a0f097924cda6dac3e0a801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cda29f2042bf32fa9f8c2460d6ce16ab
    def get_inputs(self):
        return [
            paddle.uniform([129024, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([32256, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([8064, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2016, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([528, 4], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_50cfdaea76fd09bb791d2cad11dcaaef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 129024, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32256, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8064, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2016, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 528, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_897e0c8d60b7e6f71a523f59a7ee0e76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 129024, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32256, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8064, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2016, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 528, 4], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_f6593d6be4f408d652085a36042a29c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7e5c995700c37cc8dee95f30e96ba2c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6593d6be4f408d652085a36042a29c9
    def get_inputs(self):
        return [
            paddle.uniform([300, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
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
class TestPrimitiveOp_5a66a4d2d236b9b2a81ee07b0d132ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 152, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 152, 152], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2ca207d954c5b2fd2ef2f20e349e0140(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_eea49c69ade9e2c79fb2e79ab2ab5dd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_8b5119acf06cbcee968b91525f140d0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8eb490ffa21b5b19ed754fbb05b502fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b5119acf06cbcee968b91525f140d0b
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.09477460384368896, -0.4902087152004242, -0.4203163981437683, -0.30769187211990356, -0.4971514344215393, -0.36976104974746704], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.5615507960319519, -0.18464675545692444, -0.27692711353302, -0.41350317001342773, -0.2525683343410492, -0.38656434416770935], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0754191875457764, 0.5247361660003662, 0.7928956151008606, 1.3002996444702148, 0.6827868819236755, 0.9622427821159363], dtype='float32').reshape([6]),
            paddle.to_tensor([0.9421992897987366, 1.1555519104003906, 0.8007746934890747, 1.095479965209961, 0.7881323099136353, 1.0217437744140625], dtype='float32').reshape([6]),
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
class TestPrimitiveOp_ce8faeece98edd999267dbc14875ad78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6593d6be4f408d652085a36042a29c9
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
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
class TestPrimitiveOp_63632d759fd7eae9844ddaaa7ecd6d87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6593d6be4f408d652085a36042a29c9
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
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

class PrimitiveOp_25b669a33790e1cb93baf70e0dad6601(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = -1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4ad4df98d7f8525fb5ac98b6e98165ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25b669a33790e1cb93baf70e0dad6601
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9e31fdc5647b7fcc704447bb6296130b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cda29f2042bf32fa9f8c2460d6ce16ab
    def get_inputs(self):
        return [
            paddle.uniform([115200, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([28800, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([7200, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1800, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([450, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fd77dca00b0623daab7bf61e7bcf507e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 115200, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 28800, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 7200, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1800, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 450, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_de88e18cd80145fe3764a89ca7907dab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 115200, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 28800, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 7200, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1800, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 450, 4], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_994ddb9387838e8b249c614656d2d184(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_76024ca23ce62d5d417ebd7dc7a16284(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_994ddb9387838e8b249c614656d2d184
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4096, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 16384, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5c36307159700486aa846430d36e73c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_994ddb9387838e8b249c614656d2d184
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4096, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 16384, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_eb5b547993fee6e09572f7683eaf029b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_da4c4176715139b57b811b70637bc0c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_96a63b6b5fabadaf12d7078f97362eb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6593d6be4f408d652085a36042a29c9
    def get_inputs(self):
        return [
            paddle.uniform([100, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
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
class TestPrimitiveOp_b7d2891a187181eb8478b50b02a801b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 104, 104], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4558b4cbefde7d6410cc0fca907e7b5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 60, 24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2ffaa78d84e88062ceffe748ab249818(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9c3cf8d6db022e74baff202344f7af59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2f4a5478c161b7410b8ae87b2e0688ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_7f2a4420777aa70a0a6a928c523991ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b2953c3120e2bcfe594d443aaba3e0cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f2a4420777aa70a0a6a928c523991ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 150, 384], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_5b11814883badfe7c4e659ead5e6020c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0):
        input_0 = [arg_0_0]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d7ea35e4d80d34616ccb7b93c5f43af1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b11814883badfe7c4e659ead5e6020c
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'), 'int64'),
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

class PrimitiveOp_4ac2fc2fbbd23b56ccb218de37a7f0c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 2
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c080cf903e3d31c11f56ca8c2fcf9a43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ac2fc2fbbd23b56ccb218de37a7f0c0
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_7ddaefe74e94a71b89f55c641b2f02dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_994ddb9387838e8b249c614656d2d184
    def get_inputs(self):
        return [
            paddle.uniform([1, 9216, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2304, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_79d3de1d92c880ffbd6674416a35270e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_994ddb9387838e8b249c614656d2d184
    def get_inputs(self):
        return [
            paddle.uniform([1, 9216, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2304, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_615fe80b371124bdf69af85ebaa9ac41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_994ddb9387838e8b249c614656d2d184
    def get_inputs(self):
        return [
            paddle.uniform([1, 9216, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2304, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_b4c82d897a5f6ce2fa44b242736a2356(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b858514a8ebb0a5822d4098f4b1038ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4c82d897a5f6ce2fa44b242736a2356
    def get_inputs(self):
        return [
            paddle.uniform([9216, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([2304, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([576, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5982f40baee0f11e7c5b00d45facb975(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4c82d897a5f6ce2fa44b242736a2356
    def get_inputs(self):
        return [
            paddle.uniform([9216, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2304, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([576, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7791b1dca24050cd0ad680df63e68266(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25b669a33790e1cb93baf70e0dad6601
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9b4338bbb5e823961144f4b535581173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7c64dab4104a47b063be65833b440a9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 48, 48], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2309fb34a48b2c90d3b1ced53ef0b722(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_44e1b6c2aecd2c41a7d9cfd6808f8bc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_22d1f92ef4d41eb6531a218349aaac3b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0):
        input_0 = [arg_0_0]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f64a3979fbde7168ef13c5654530c8a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22d1f92ef4d41eb6531a218349aaac3b
    def get_inputs(self):
        return [
            paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d4fcd8057f9227c1e2d047071f5d4408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 112, 112], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_f61c5af0b1c3ec118ef972a224474bd0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5ce0406078485cf2a2085d5cf7565604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f61c5af0b1c3ec118ef972a224474bd0
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 48, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 48, 28, 28], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_506063ee61f999cca591ba0fe8363fc8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0):
        input_0 = [arg_0_0]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1fb328d9f9576dba818fe2e84406d2c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_506063ee61f999cca591ba0fe8363fc8
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[86970], dtype='int64'), 'int32'),
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
class TestPrimitiveOp_42d542af1deea7523621814cf8556cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f1415bbdb9bc34ac45855415c69821f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e8ef99dd6fcc1c0ee662008bd9ecc1c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 34, 34], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a2fd52c249585b602fefdaf42bff196f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 64, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0e3ad9184b4472ef627ad220ef542f7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_506063ee61f999cca591ba0fe8363fc8
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[242991], dtype='int64'), 'int32'),
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
class TestPrimitiveOp_5e2ceb8881f703f35173c30ce25833ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5ace858c2c1b649a728ac10f2c750a4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_96a886ca27f7569194e66408def5d8ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 13, 13], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4e1e5fae391d887eac734a0a9a0a066b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 34, 128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4f7681644eb7021db992f22f4e05a6f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 160, 240], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_bfeb32be76e9d30a8227bf0960bfa3e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2d24ac9ee929c6336de2c649ffb563c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_506063ee61f999cca591ba0fe8363fc8
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[220968], dtype='int64'), 'int32'),
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
class TestPrimitiveOp_03102ff51f83d470d878445ae96fd6b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 60, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_66a1d65aca051e16e6d86dfd9586ee96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 8, 8], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5f34532a0a9e1a956f5be4e15ef113dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25b669a33790e1cb93baf70e0dad6601
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6477fae047202e1ab97d1807e55ecde4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4629551be84bd7497d70085026bd1206(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ea433430b93b3aba5d22bd7ca12ef45a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8c33e594d907fc2c4ee408f193f3d70a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 25, 38], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0fe372a5825c499c60551ddb7e84905f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25b669a33790e1cb93baf70e0dad6601
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4424b3e76e388cfaac2a5ba10cf9d1da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_506063ee61f999cca591ba0fe8363fc8
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[153450], dtype='int64'), 'int32'),
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
class TestPrimitiveOp_6aa0a30e33e3f5ce2317d0203c4e1529(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_2544148fd649c6f84641ce0a1fcd5e7f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_20f4eef8232c7ee47a4d6561b6266701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2544148fd649c6f84641ce0a1fcd5e7f
    def get_inputs(self):
        return [
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_97558b7e42d848210393ce2dcaa28a32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 30, 30], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_64295394cdbe267e94e6e8d4bb826abc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0):
        input_0 = [arg_0_0]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe46dbcc6900a80f4bb87210b3eccb70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64295394cdbe267e94e6e8d4bb826abc
    def get_inputs(self):
        return [
            paddle.to_tensor([[4], [7], [8], [2]], dtype='int64').reshape([4, 1]),
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
class TestPrimitiveOp_eba99d511a69a5d968f5f815dcbbf32e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_506063ee61f999cca591ba0fe8363fc8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_172d33c26ad57defc9895780336dfaf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_506063ee61f999cca591ba0fe8363fc8
    def get_inputs(self):
        return [
            paddle.to_tensor([9, 9, 3, 6], dtype='int32').reshape([4]),
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

class PrimitiveOp_cfbbd3082f240e5fae6faf8c4a7114f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0):
        input_0 = [arg_0_0]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c952e9533f887c1708ed24e18a453645(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfbbd3082f240e5fae6faf8c4a7114f2
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[4, 28, 28], dtype='int64'), 'int32'),
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

class PrimitiveOp_04275cd298b5ef067e74f4e89fd3068a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0):
        input_0 = [arg_0_0]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1de5702cdcb09717b615f614dbd5eb5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04275cd298b5ef067e74f4e89fd3068a
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03318215161561966, 0.3026772439479828, 0.3224482238292694, 0.37125200033187866], dtype='float32').reshape([4]),
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
class TestPrimitiveOp_187e7d3154fb93c73f0552c626b2e7f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 22, 22], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_95ea97c25a285c20dcc2d8a772dbdd78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 60, 16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f947dfe621e9ac2ba267162e1e0a1eb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64295394cdbe267e94e6e8d4bb826abc
    def get_inputs(self):
        return [
            paddle.to_tensor([[4], [7], [8], [2], [2], [9]], dtype='int64').reshape([6, 1]),
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
class TestPrimitiveOp_b49459a449aa802dfd105066e61e081e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_506063ee61f999cca591ba0fe8363fc8
    def get_inputs(self):
        return [
            paddle.to_tensor([9], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_23607ee48ad6a9cb48dd07ee2858bcbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_506063ee61f999cca591ba0fe8363fc8
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 6, 3, 3, 1, 7], dtype='int32').reshape([6]),
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
class TestPrimitiveOp_b0b44ca400465d2f22207257860e4632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfbbd3082f240e5fae6faf8c4a7114f2
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[6, 28, 28], dtype='int64'), 'int32'),
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
class TestPrimitiveOp_cfbcb312486b80054c4db8b1106e6a5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04275cd298b5ef067e74f4e89fd3068a
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08213967829942703, 0.4675028324127197, 0.4983418881893158, 0.24993745982646942, 0.3551784157752991, 0.27458786964416504], dtype='float32').reshape([6]),
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
class TestPrimitiveOp_677fdba03b0c4b419a8aeebd0c0f5c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 19, 256, 512], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_aa05f9125e2e5470c672297d4a8268a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 64, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 64, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 64, 240, 240], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_552be08ae67acd5e36eb035076df3852(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_506063ee61f999cca591ba0fe8363fc8
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[185691], dtype='int64'), 'int32'),
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
class TestPrimitiveOp_5186fe327213470f10aa663068d5ac47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25b669a33790e1cb93baf70e0dad6601
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5094551cc22ca782168772384d3e217c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 20, 30], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_933b29492cd23d89cebc9d80666445a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 20, 30], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cdda87d65855f78ab58e39f5d6bc3c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_849dc6b4aab87a4bee54677ef38f9a86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22d1f92ef4d41eb6531a218349aaac3b
    def get_inputs(self):
        return [
            paddle.uniform([242991, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0ec8344daa8ef92d755f1e0cab1d1bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_057dbb88ca355eb6445df49f150674e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_994ddb9387838e8b249c614656d2d184
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1024, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7a58c387a7109a697e4cbdbe2c37d85f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_994ddb9387838e8b249c614656d2d184
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1024, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fb25433e917ededb33dacde172a67b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_994ddb9387838e8b249c614656d2d184
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1024, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b08a2eaf28964cd21dd2e5b3507cb178(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4c82d897a5f6ce2fa44b242736a2356
    def get_inputs(self):
        return [
            paddle.uniform([4096, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6de97417fac6fa0aa4f42846f177bfba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4c82d897a5f6ce2fa44b242736a2356
    def get_inputs(self):
        return [
            paddle.uniform([4096, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a2a3ec8995cdb74e38506f38cf7bc0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25b669a33790e1cb93baf70e0dad6601
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cd83874a2d09269ce44978697294cf62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6593d6be4f408d652085a36042a29c9
    def get_inputs(self):
        return [
            paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
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
class TestPrimitiveOp_1f65ee76d6aef6ad294285425d9ee2ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 44, 44], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a387a747606bf37dea0b636fe2410977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5ec4efca318aba690efc1d510e89395
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fb6452018f1f287d9a1087e316522a3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22d1f92ef4d41eb6531a218349aaac3b
    def get_inputs(self):
        return [
            paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0eb3737c12fa8eca21b1820fbdd1bda3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4e99a2ec631f66d6da63ba44032a2530(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 128, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c8e0c2336b1f7a77b884dba60169de50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 96, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 96, 96], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b44f9213c240b94b1cb5f3e8079213c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f2a4420777aa70a0a6a928c523991ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5e38efd32e78a897749e4e85a5f23d3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5ec4efca318aba690efc1d510e89395
    def get_inputs(self):
        return [
            paddle.uniform([43, 224, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 128, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 128, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 128, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 128, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 128, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_eba972c0eaaed8ed33337905c3582cda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_506063ee61f999cca591ba0fe8363fc8
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[113061], dtype='int64'), 'int32'),
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
class TestPrimitiveOp_ed0736566f45bbf7c80c59aa2e917218(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b5119acf06cbcee968b91525f140d0b
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.31164759397506714, -0.29729509353637695, -0.6793718934059143, -0.2461491823196411, -0.4240976870059967, -0.5619874000549316], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.380005419254303, -0.4661708474159241, -0.5894411206245422, -0.7768030166625977, -0.7117545008659363, -0.5895625948905945], dtype='float32').reshape([6]),
            paddle.to_tensor([0.8328929543495178, 0.8114372491836548, 0.7676553130149841, 1.0613429546356201, 0.7502427101135254, 0.7692978382110596], dtype='float32').reshape([6]),
            paddle.to_tensor([0.8962369561195374, 1.0939757823944092, 0.5926178097724915, 0.8562833070755005, 0.7293772101402283, 0.6426240801811218], dtype='float32').reshape([6]),
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
class TestPrimitiveOp_ed697eb290fd12e67a16f6f8f2e721a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8b1bd4bc5b0cf5ed95765eec529e2a19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f61c5af0b1c3ec118ef972a224474bd0
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4f93a36130b2fa61ccca08bed42fda6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2d53866eac6a0f1b014dd0d867f70088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2b976e199dfd592f40a39d1912e01a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25b669a33790e1cb93baf70e0dad6601
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b7ad6957158d776e5036e4da6667276e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_81db455c359b1ec76b660e8ebd15e7da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6593d6be4f408d652085a36042a29c9
    def get_inputs(self):
        return [
            paddle.uniform([6, 256, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
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
class TestPrimitiveOp_e4077be39f80c6049f35f3f6785a22a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7b65948c694788f97e492a70c132ece5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_607fda637fe24340b23b2814e1c1c96b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5cbfd507d063c924fe789d241868c9c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 48, 112, 112], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5d6bd81a624c9101977bda8bcbf86e3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e377873fbff61da1f0bf1f1e28f30c89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a631f0494b09202476b82a636a808c10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_85f296abf34a8bbbf8e337ff5991a1a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_19bb63819e2459e8268ddf65ed546ea6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6ab5412180b5f93037f03f3d664856ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b4af008fc41d8c96cfb145dc66732ba3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9dbd80abe061e89ba6e9c7f06ca0aceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6593d6be4f408d652085a36042a29c9
    def get_inputs(self):
        return [
            paddle.uniform([3, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
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
class TestPrimitiveOp_51cfbed3d983d31c452654214124a3d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_57a4e8768c68d3781031ee31f78cc327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 160, 160], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9820fc1eaf923462c3c7cfa7b5a6de5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_46d349535cfcd7cb4f31c20aea73fc85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cda29f2042bf32fa9f8c2460d6ce16ab
    def get_inputs(self):
        return [
            paddle.uniform([139392, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([34848, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([8712, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2178, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([528, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5f059cc1b782fea79763e5f28bd7af14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 139392, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 34848, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8712, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2178, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 528, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a229de0b97c9b12e71feb093848aac85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 139392, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 34848, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8712, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2178, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 528, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fbc14792d3e4c8ff49432b0c09c9d60a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b4d2a765c21f63abe6cd5d30201bdcc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 30, 30], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5daed8498b4b727fbdc509c5d7fb38de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d75f6c9c3742452616cfcec4ee7cdb57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5ec4efca318aba690efc1d510e89395
    def get_inputs(self):
        return [
            paddle.uniform([43, 512, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c572fdec18f4eb36560fd9b440c32efe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 168, 168], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 60, 168, 168], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f73364d2ca5fcd0617a6bfe5e9c3e03b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22d1f92ef4d41eb6531a218349aaac3b
    def get_inputs(self):
        return [
            paddle.uniform([217413, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a35c8b9614e5ea57017cfe5de3963539(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_506063ee61f999cca591ba0fe8363fc8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 6], dtype='int32').reshape([2]),
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
class TestPrimitiveOp_6309e42b2ac69e139c34713874f2c8e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_da6b6c060ee3f37d89472a2a86888363(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4e40e9cd48f96da913c42ccb9fae1ae7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_97ec8e4a88bfaea6e89d18edcb76c05f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3ebb69326bcfe47215783086f44432d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_63b55dfb094b68a790fe836d00411b3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6545976066f480dcbbeb11467fd4ba45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8847c3243e8d9403d676f114ba972a41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_99e2cb0215d994e7ed13b60caddc0174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_86562cffaa667ff9e5cdedb9b8952931(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_96a7cdd84bbd9f6a0e063d6c3c38ed96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 34, 160, 160], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b2bd7c5c15cca49e075552c5e795830d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 60, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4fd246f6e2edb552bccb75e150ca9435(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 52, 52], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a8bd9984ce60df131413b796a5dbcbc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5ec4efca318aba690efc1d510e89395
    def get_inputs(self):
        return [
            paddle.uniform([11, 448, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4a67b96c1ecbe85471e5f42e3a092122(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 28, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 28, 50], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_177e28f62559716ab3108a6abc3954a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 76, 76], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_bd0d8877c76b5c94043677733d2af53f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25b669a33790e1cb93baf70e0dad6601
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c5708695f0a4e880f511c7171b7999fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6593d6be4f408d652085a36042a29c9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
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
class TestPrimitiveOp_6bd04a32e779c4064fcf3f517217db43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2b014216d08864d85f6b9ccb1933c83f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_506063ee61f999cca591ba0fe8363fc8
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 9], dtype='int32').reshape([2]),
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
class TestPrimitiveOp_3cc3a177b78cbb4c5e5d8ab0b61682f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3b1a8334621509b108ea4f8975a8b8c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_994ddb9387838e8b249c614656d2d184
    def get_inputs(self):
        return [
            paddle.uniform([1, 6400, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1600, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 400, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_aae37c7f5a9b83bd22ef598d4b0afb51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_994ddb9387838e8b249c614656d2d184
    def get_inputs(self):
        return [
            paddle.uniform([1, 6400, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1600, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 400, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8da7554db56980690c50586e7305fc8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_994ddb9387838e8b249c614656d2d184
    def get_inputs(self):
        return [
            paddle.uniform([1, 6400, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1600, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 400, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_067067ef991c6353c62674599285c57b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4c82d897a5f6ce2fa44b242736a2356
    def get_inputs(self):
        return [
            paddle.uniform([6400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1600, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([400, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d80dd25d5d839c9eb225967b7cb3f7dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4c82d897a5f6ce2fa44b242736a2356
    def get_inputs(self):
        return [
            paddle.uniform([6400, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1600, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([400, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_86df21567526a882232bff0fdead57f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39, arg_0_40, arg_0_41, arg_0_42, arg_0_43, arg_0_44, arg_0_45, arg_0_46, arg_0_47, arg_0_48):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39, arg_0_40, arg_0_41, arg_0_42, arg_0_43, arg_0_44, arg_0_45, arg_0_46, arg_0_47, arg_0_48]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

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
class TestPrimitiveOp_a7487248bf5d0f726e8b8453de99bceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86df21567526a882232bff0fdead57f9
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b4896a7e0100d30e05149ce08590c86e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f61c5af0b1c3ec118ef972a224474bd0
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fec1558b9663f3dc992f4753015ee316(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2ea47dfa0f408ee2c9333a2d8af87e0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_506063ee61f999cca591ba0fe8363fc8
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[205923], dtype='int64'), 'int32'),
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
class TestPrimitiveOp_0fe9135646b8cc873d334ea3fa1a9648(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 176, 264], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d0023b61a6ca3d09773fbb695a696709(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25b669a33790e1cb93baf70e0dad6601
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_70fbf6f2b00ebee8b4185d78e854b4b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25b669a33790e1cb93baf70e0dad6601
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8480bd8c3ba940768e363ec90ea3e014(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 184, 328], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5ed44757b4d2a4e84019296db57d245a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5f5eea6e1d24e68e0895359358e4b685(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6593d6be4f408d652085a36042a29c9
    def get_inputs(self):
        return [
            paddle.uniform([7, 64, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 64, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 64, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 64, 7, 7]),
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
class TestPrimitiveOp_b8ae946a046e7a493329814df9e924fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 136, 136], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 136, 136], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1a851adb9cd07e5fade66d89c3f2b706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_c76e1849d0da6371394c425aa0f4ae37(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 2
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8a3a66acc250c256771ec8b7fa7858a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c76e1849d0da6371394c425aa0f4ae37
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4, 1, 13, 19], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_2e3b0a26efe3eb3bbbb94e79340b4b99(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9a661d53b1cb8efc9c6ff8e3dfcf8ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e3b0a26efe3eb3bbbb94e79340b4b99
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int32').reshape([1]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
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

class PrimitiveOp_0d6f04ac2bc29ae4b9733a9b1bb96a38(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a3a9c65361bb4e5507c86318c7b80935(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d6f04ac2bc29ae4b9733a9b1bb96a38
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 2, 3, 4], dtype='int32').reshape([4]),
            paddle.to_tensor([0, 0], dtype='int32').reshape([2]),
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
class TestPrimitiveOp_edb3117604e584cfe26670ce3ac333c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d6223945108414c71977c30cc4bd59d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f61c5af0b1c3ec118ef972a224474bd0
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 80, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fb09264f23f8fc2d08ed90fc659392ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 44, 44], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4a2ce7cab0af74aa5733eaa3deaa4f6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cda29f2042bf32fa9f8c2460d6ce16ab
    def get_inputs(self):
        return [
            paddle.uniform([182400, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([45600, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([11400, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2850, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([741, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b46f27c4571c546fb151b94e6c29f1ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 182400, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 45600, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 11400, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2850, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 741, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3da2104cd8095c3873e09832d70a0bf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 182400, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 45600, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 11400, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2850, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 741, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_df742fd474ba553bff41fc9eeea442b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22d1f92ef4d41eb6531a218349aaac3b
    def get_inputs(self):
        return [
            paddle.uniform([86970, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_373225b0136119f77a9bbbd11e34e460(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22d1f92ef4d41eb6531a218349aaac3b
    def get_inputs(self):
        return [
            paddle.uniform([205923, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_392e4414f7cbea2031fe1162c4a75354(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e1c5fe2dac3b626f6b639e40dc3c5fd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_96effc657b5baec65b43fd479fe4ffe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_506063ee61f999cca591ba0fe8363fc8
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[123783], dtype='int64'), 'int32'),
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
class TestPrimitiveOp_9bf5723424849f0b2363ae7660f84091(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22d1f92ef4d41eb6531a218349aaac3b
    def get_inputs(self):
        return [
            paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c9434fb59bd9737c58bc4ac8d0f28539(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_84fffeb0eb53456c23dd636a99a81109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 104, 104], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2bb29d09e89c75d7cd916c701d5b6bfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 184, 184], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 184, 184], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f2e91fc4882eac506b68f54d439b62f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_506063ee61f999cca591ba0fe8363fc8
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[171888], dtype='int64'), 'int32'),
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
class TestPrimitiveOp_703e4939afa2d368df0ecbda16fc1f00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f438aed2373b8b582926eb6be6517c14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4c82d897a5f6ce2fa44b242736a2356
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([784, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3136, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_76f83bb627bf6befa9de239a9329d80b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4c82d897a5f6ce2fa44b242736a2356
    def get_inputs(self):
        return [
            paddle.uniform([196, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([784, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3136, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_bbc31d858bb54870e94cd19a1c61c05a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4c82d897a5f6ce2fa44b242736a2356
    def get_inputs(self):
        return [
            paddle.uniform([196, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([784, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3136, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9ae817eddf674aaa236ad641f6a3dad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a7c6242ddd624d4ff2b682e3a332010d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4b514c119248faf079121294c1574c9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_980593cf33aa08377d499349ea8af138(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_663e696f36c22b6ff0caf4148f140bd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e231b8cb97ffd65b37180b1386b0d653(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_230f94577e86e481f475d27f4a233007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cc8b93f54ec0488c8e976b815f8bcadf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cda29f2042bf32fa9f8c2460d6ce16ab
    def get_inputs(self):
        return [
            paddle.uniform([65280, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([16320, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4080, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1020, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([270, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_42771142d65185251b60e925bf6faba4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 65280, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 16320, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4080, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1020, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 270, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cc37f7f19687a86024cb5dfd97a7cb00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 65280, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 16320, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4080, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1020, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 270, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_efb771570908d4d8b269bdec0af7c283(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ac997fc0046bba7d6e17ec2e0a925ddf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6593d6be4f408d652085a36042a29c9
    def get_inputs(self):
        return [
            paddle.uniform([5, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
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
class TestPrimitiveOp_011718cac472710aaf51dbd4b1f9c73d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 22, 22], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8d740e90aa4e210195479970f9f9356c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_506063ee61f999cca591ba0fe8363fc8
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[217413], dtype='int64'), 'int32'),
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
class TestPrimitiveOp_3e152d77fe86c9a4143ec80cfdf607ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22d1f92ef4d41eb6531a218349aaac3b
    def get_inputs(self):
        return [
            paddle.uniform([113061, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_744d1a2f78110fdf31d80a75a79c03a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1f006d4c6cdfe44384c039d496b43d7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 18, 18], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_93e44de0ba5d0c5107f2a414f559038e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f61c5af0b1c3ec118ef972a224474bd0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7cbc64d9d5db1f0ed5ee6b504ae284b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5ec4efca318aba690efc1d510e89395
    def get_inputs(self):
        return [
            paddle.uniform([11, 512, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_25bb7a6a1093fd9aae2fb532ddc04536(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5ec4efca318aba690efc1d510e89395
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_598b97e1a629756acfbdcf73334dcbf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a53f014b2de7292d1318d4ca083816ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 9, 9], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5b327c1b3970a6d315c3f683b8ea1429(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_da83f33be89451d6761b40510d4347d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 88, 88], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b18635f2a2cac9a95956069834c85a61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b1c17523f4973a1477cc9476addaafa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 232, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 232, 16, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4173a36ecd034ca971a2ca671e9d0e20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 16, 128, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_33b640b20257b375b45592a8972cb089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 52, 52], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7d09f88b36e74eb9a1b1a602f5b6a3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22d1f92ef4d41eb6531a218349aaac3b
    def get_inputs(self):
        return [
            paddle.uniform([123783, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d61b6ee279278f23fcff4b674a1aaf83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 13, 13], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_44ceaa9053757aab2615e0a26a40e6d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f61c5af0b1c3ec118ef972a224474bd0
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_32da615ad800bf70b206534bead2e80b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_325dc4419540654d93092a2488d47f62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 15, 15], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9dd7c007934d66258eca217f15d0ad6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 14, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 14, 25], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a8fa96a7aacdb89f37322e1c635f77fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 42, 42], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 42, 42], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d2649b7b2bd6b98f0c99bf17ac29e7e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 60, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9da284bc4b145580d93ba37300a3e078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f61c5af0b1c3ec118ef972a224474bd0
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_1fefbfa839e30c4185c1743a3a86718c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
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
class TestPrimitiveOp_5ebbe7c3441b9c857cb4c99c75062222(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fefbfa839e30c4185c1743a3a86718c
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 7, 7], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3da932d966104c53fc76050717e96db2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_39d4f5b3c0e7d9ce3dc170218e705224(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25b669a33790e1cb93baf70e0dad6601
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cf19b40a9aa47375a88053b9f81675e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([22, 100, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 100, 7, 7], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_30ef170cfb8ffdc2633b0fa5b0cbe470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c8dfe551ff5e1107dc7245c23e49709a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25b669a33790e1cb93baf70e0dad6601
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f072e9da8604398f512560405316f6f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_bab4e0da7304d988c9fc286cafea060b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64295394cdbe267e94e6e8d4bb826abc
    def get_inputs(self):
        return [
            paddle.to_tensor([[4], [7], [8]], dtype='int64').reshape([3, 1]),
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
class TestPrimitiveOp_65727911b0467c80d188ba4577e607e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_506063ee61f999cca591ba0fe8363fc8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 9, 9], dtype='int32').reshape([3]),
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
class TestPrimitiveOp_daee2e0e59655c7ebb1628eec5a42e1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfbbd3082f240e5fae6faf8c4a7114f2
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[3, 28, 28], dtype='int64'), 'int32'),
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
class TestPrimitiveOp_835f46af0140d1cb1e0483f1f3416262(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04275cd298b5ef067e74f4e89fd3068a
    def get_inputs(self):
        return [
            paddle.to_tensor([0.240517258644104, 0.41334110498428345, 0.4914228022098541], dtype='float32').reshape([3]),
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
class TestPrimitiveOp_cf94ffdd27ed5fbb210d17035c1fd618(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5ec4efca318aba690efc1d510e89395
    def get_inputs(self):
        return [
            paddle.uniform([43, 512, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0610e74631b212579d9e4b5f41baf392(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f61c5af0b1c3ec118ef972a224474bd0
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_6a7717ea8be4c5963bbf56d4ac99ea01(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

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
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_24c2edcfd2df95a02a230d489f178cdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a7717ea8be4c5963bbf56d4ac99ea01
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_eba80a85ad6bb7a4b6518b8a9bda045f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6ae4bb94264d2fae9fc0857f6fed6f7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22d1f92ef4d41eb6531a218349aaac3b
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3930245339870453, 0.30191943049430847, 0.31097444891929626, 0.011108126491308212], [0.1884414106607437, 0.31346622109413147, 0.012437473982572556, 0.2569693624973297]], dtype='float32').reshape([2, 4]),
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
class TestPrimitiveOp_65b7049832b01ea6415eea3a128262d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22d1f92ef4d41eb6531a218349aaac3b
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07804898172616959, 0.3275541067123413, 0.11523129791021347, 0.2340201884508133]], dtype='float32').reshape([1, 4]),
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
class TestPrimitiveOp_fe3cd5fe048ed5cc982f8197b26c29e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22d1f92ef4d41eb6531a218349aaac3b
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.25656959414482117, 0.06855367124080658, 0.07685001939535141, 0.10500314831733704], [0.04176953807473183, 0.3559780716896057, 0.32213693857192993, 0.052480366080999374]], dtype='float32').reshape([2, 4]),
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
class TestPrimitiveOp_125c3e88ba7a11044cc891f610aa5232(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_91fd12148b3b2a34497c5eb1dfa596dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_994ddb9387838e8b249c614656d2d184
    def get_inputs(self):
        return [
            paddle.uniform([1, 4624, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1156, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 289, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b93966b4b023cd0f935098b0f7790ce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_994ddb9387838e8b249c614656d2d184
    def get_inputs(self):
        return [
            paddle.uniform([1, 4624, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1156, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 289, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a35ab7a81edb76de5703e887ad3b8bec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_994ddb9387838e8b249c614656d2d184
    def get_inputs(self):
        return [
            paddle.uniform([1, 4624, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 289, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8f2901a138942193b2e01079cda819bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4c82d897a5f6ce2fa44b242736a2356
    def get_inputs(self):
        return [
            paddle.uniform([4624, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1156, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([289, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a4aaafaa3ab6d5f35eb49f7d3f4de841(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4c82d897a5f6ce2fa44b242736a2356
    def get_inputs(self):
        return [
            paddle.uniform([4624, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([289, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_718f27ff7a9095cee15f07ecff10562f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9bd9663e3f4c3783c9b4eafd3162a8af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22d1f92ef4d41eb6531a218349aaac3b
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.38766393065452576, 0.03835798427462578, 0.005889374762773514, 0.208037331700325], [0.19441747665405273, 0.05325768142938614, 0.1256839781999588, 0.34065794944763184]], dtype='float32').reshape([2, 4]),
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
class TestPrimitiveOp_9269f8f6a5b25e0ca262a6b864ebca51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22d1f92ef4d41eb6531a218349aaac3b
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.24676227569580078, 0.09557920694351196, 0.211879700422287, 0.015435690991580486]], dtype='float32').reshape([1, 4]),
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
class TestPrimitiveOp_60be2ae6849c48df8a4409ba4455e065(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22d1f92ef4d41eb6531a218349aaac3b
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.23169058561325073, 0.4083530008792877, 0.01863524690270424, 0.27500462532043457], [0.030609913170337677, 0.457322359085083, 0.3447035253047943, 0.37497127056121826]], dtype='float32').reshape([2, 4]),
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
class TestPrimitiveOp_5824c9034726bb62865202f6ca4b11bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cda29f2042bf32fa9f8c2460d6ce16ab
    def get_inputs(self):
        return [
            paddle.uniform([84864, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([21216, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5304, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1326, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([351, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1bf3bfdddaa82bf9de422defcb78c6b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 84864, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21216, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5304, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1326, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 351, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4b060ba6c60ea31dd2d586859070e315(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 84864, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21216, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5304, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1326, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 351, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_80ba77fffb801a55b78cf292e9f63dfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6593d6be4f408d652085a36042a29c9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
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
class TestPrimitiveOp_40d7141b9a2200f4844de7bbac2f8287(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0ba36dc74658b95f6606aad02535eebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4096, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_14f6b2111b6de9ceb017f2f28b12ec7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4096, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1024, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4429f55dd40ca05c39edc536c99d8976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0328fbf02a14ee3c293c033353761d91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_eacdf139902f6e34dd688e79abece69d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_37cc45b594d0d3a9eb484bb01bc62a8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 30, 30], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_bf9f0dd59f23875a74dce90ad269278e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f61c5af0b1c3ec118ef972a224474bd0
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ebcdafa450c90dbb10533b6b130526a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 15, 25], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_936eb130b6a258013e1fc6fa6e39ac43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 15, 25], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_426b2411c99d55b85738ab29c0c8486d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 256, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1979e9a227fde549412106174820b261(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 128, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6b2d63702da30a1a33aef676f7034c8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f18b59d3345d635da3ff8c992d14b285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6bf7e4daf32e808cb58c35efabb8d53b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64295394cdbe267e94e6e8d4bb826abc
    def get_inputs(self):
        return [
            paddle.to_tensor([[4], [7]], dtype='int64').reshape([2, 1]),
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
class TestPrimitiveOp_a2f6220a8c02261f497c1a2744599486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_506063ee61f999cca591ba0fe8363fc8
    def get_inputs(self):
        return [
            paddle.to_tensor([8], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_4cec4bba30dab15cf9b7bf35d890de4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_506063ee61f999cca591ba0fe8363fc8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 2], dtype='int32').reshape([2]),
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
class TestPrimitiveOp_6c3e6d93e6ad5ec1b0b66404f2dcdf5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfbbd3082f240e5fae6faf8c4a7114f2
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[2, 28, 28], dtype='int64'), 'int32'),
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
class TestPrimitiveOp_3aaa0c143261f27b87aa63b717f806e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04275cd298b5ef067e74f4e89fd3068a
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2746356129646301, 0.23825183510780334], dtype='float32').reshape([2]),
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
class TestPrimitiveOp_9f5195a7b33e053752187b11dd72c026(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4970bbf68ee49383d58be231d7b01c92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6593d6be4f408d652085a36042a29c9
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
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
class TestPrimitiveOp_6e0be06527a453f0328c1327524590c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_dc76586a3ff904b4451a3b8d852a7714(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_59d2a56350f02f62a7b1a511fc881189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c99e028e7d0181e72ad1d6ce60002a75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f61c5af0b1c3ec118ef972a224474bd0
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0b485842f097d3de9ef3dff40307bbb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 8, 8], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_afa6540cb089a89d0e0a7d0ebbc40efa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f31246f62fb2d5e19d7dc61da64ba642(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6f754818c50f07835c0262161f49ee70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 92, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 92, 92], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1a13a4e6f664abd638c84ca99c0c8abe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_77d8c8d470763557d6641ec1942db7a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 44, 44], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_408decb4aac7cee1cff7c907674e013b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cda29f2042bf32fa9f8c2460d6ce16ab
    def get_inputs(self):
        return [
            paddle.uniform([139392, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([34848, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([8712, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2178, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([561, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9617d83f88f399777cc88e4430b7fab0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 139392, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 34848, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8712, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2178, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 561, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ee71f250b4725c41711ea2f73f53c047(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 139392, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 34848, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8712, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2178, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 561, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fdf286aee18dbf11579a672407a0518a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8ad71b5c2993dc8f8e6066705edae464(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 52, 52], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_438256d520039729db8584fe689be84d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 40, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6141331022df4bbaff6e9745fe47d3fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_de47c33951f41e927c554b568b37ebaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5aa99dbe2865365f1e22fc32e0ee1bb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_20a75995b40df328225035e3e322effc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c76e1849d0da6371394c425aa0f4ae37
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4, 1, 50, 76], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_81bb08deaef1510428e18ac74178841e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 46, 46], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 46, 46], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fbfc1294556b6eefffb5d17237d2b3af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cda29f2042bf32fa9f8c2460d6ce16ab
    def get_inputs(self):
        return [
            paddle.uniform([15200, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([950, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([247, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([70, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_71d308473e3f753f4f7d11cd8d5dc775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cda29f2042bf32fa9f8c2460d6ce16ab
    def get_inputs(self):
        return [
            paddle.uniform([15200, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([950, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([247, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([70, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b9217782902ddd67b0cc5532e7ceb965(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cda29f2042bf32fa9f8c2460d6ce16ab
    def get_inputs(self):
        return [
            paddle.uniform([15200, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([950, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([247, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([70, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cdbcd59507e5f1b428bac5838023d444(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a8fcbe6e6a398762c2638868963efcf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 64, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e9f4c7c5cac2c40f1c333b30148303b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cda29f2042bf32fa9f8c2460d6ce16ab
    def get_inputs(self):
        return [
            paddle.uniform([163200, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([40800, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([10200, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2550, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([663, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_479e1dc22f46b6ab22660701a789fffa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 163200, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40800, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 10200, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2550, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 663, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_31794f73ca26f49ba0d718f09276a9e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 163200, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40800, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 10200, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2550, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 663, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6e9b948b32b6ef993708a80192248730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cf42112ad6b306e1e53ed65350e642bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 84, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 84, 84], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c0fd7663616cae5b864920ac96125a06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6766db5f20662a716fbf702d0c62a143(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 80, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ad92b53a776f32f99d210f924701652e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 48, 48], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e21cec8415282dc87e290bfb57883769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_704141b6e6d90dd2037128cd67582782(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2ed1968e506492450bb9950e68ba5023(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cda29f2042bf32fa9f8c2460d6ce16ab
    def get_inputs(self):
        return [
            paddle.uniform([92928, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([23232, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5808, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1452, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([363, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3d3b4dd41aecfc57a23473a369653277(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 92928, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 23232, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5808, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1452, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 363, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b0a199fb7da4e14e374ece1dbe5b5783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 92928, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 23232, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5808, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1452, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 363, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b219686c7088bfa8e3126889d17ee837(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2c1f9119f827a74d01cd5d2fa7911ff9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b49d54f039e3fc4abbf523986aac5031(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f5519ec8bbe32224b02bfd92140ae841(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d6fae87ba82c51a05c1f7d1b2f8d6a5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_960d9f9bf76a04bd56afceceebe388f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f2a4420777aa70a0a6a928c523991ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 150, 768], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_783bc417afdbf272f828e9c3ce1441bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 36, 36], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_60bd774f8280bffbc9366419ae157f9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 116, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 116, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_52f7ce0f576151921bd67c5d5a0aedc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f2bcd99ba53b84c7779fa14d669f54aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_225bf43a9e797bc115600d9068abf664(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 15, 15], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_161198f1ab815e5ecabd73075a8c021c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_897b877b14c2338bc411186b051a8a50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ad5187ddeba4950461c6dbb5bddb9abe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 120, 120], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b028dd8ce771139a79dbfe60d9457b07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fa3b230c8c67cf401151f35ebc804b52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cda29f2042bf32fa9f8c2460d6ce16ab
    def get_inputs(self):
        return [
            paddle.uniform([154560, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([38640, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([9660, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2415, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([648, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7bab3c49e4be2fcc6a2ab80684ad619b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 154560, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 38640, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9660, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2415, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 648, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_aa38bf1cc7cb00787931d38a861e3647(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 154560, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 38640, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9660, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2415, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 648, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d6d65359bd17ac3676f0266187364d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9be12347113f2f6cef15732776710dea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2b0b78dca0257adb38d1bf2d4dedaf17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_75bfdace60898c9462d748f7a57f001a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9cfbcfa368292c187cc57d16f042b584(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5dc64820feb5f125fdf6945ae7a47913(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d8dff7e8eac7e0d4391bc25fc57eaf95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 38, 38], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_92eb00821c1625ff2b4c434a4f2328c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a84cf9afb534db938fe7228b14642174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5ec4efca318aba690efc1d510e89395
    def get_inputs(self):
        return [
            paddle.uniform([11, 224, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 128, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 128, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 128, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 128, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 128, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c6e40c75dd588be8c9c3c46c5880f3cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 60, 256, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e10ada60a63bee4deec59f650655c9f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f61c5af0b1c3ec118ef972a224474bd0
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d1fe52cc5af2d875eb62073a81cb0b76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 18, 18], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_b1b4603f75901cd6c46094dc422f0b3a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39, arg_0_40, arg_0_41, arg_0_42, arg_0_43, arg_0_44, arg_0_45, arg_0_46, arg_0_47, arg_0_48, arg_0_49, arg_0_50, arg_0_51, arg_0_52, arg_0_53, arg_0_54, arg_0_55, arg_0_56, arg_0_57, arg_0_58, arg_0_59, arg_0_60, arg_0_61, arg_0_62, arg_0_63, arg_0_64, arg_0_65, arg_0_66, arg_0_67, arg_0_68, arg_0_69, arg_0_70, arg_0_71, arg_0_72, arg_0_73, arg_0_74, arg_0_75, arg_0_76, arg_0_77, arg_0_78, arg_0_79, arg_0_80, arg_0_81, arg_0_82, arg_0_83, arg_0_84, arg_0_85, arg_0_86, arg_0_87, arg_0_88, arg_0_89, arg_0_90, arg_0_91, arg_0_92, arg_0_93, arg_0_94, arg_0_95, arg_0_96, arg_0_97, arg_0_98, arg_0_99, arg_0_100, arg_0_101, arg_0_102, arg_0_103, arg_0_104, arg_0_105, arg_0_106, arg_0_107, arg_0_108, arg_0_109, arg_0_110, arg_0_111, arg_0_112, arg_0_113, arg_0_114, arg_0_115, arg_0_116, arg_0_117, arg_0_118, arg_0_119, arg_0_120, arg_0_121, arg_0_122, arg_0_123, arg_0_124, arg_0_125, arg_0_126, arg_0_127, arg_0_128, arg_0_129, arg_0_130, arg_0_131, arg_0_132, arg_0_133, arg_0_134, arg_0_135, arg_0_136, arg_0_137, arg_0_138, arg_0_139, arg_0_140, arg_0_141, arg_0_142, arg_0_143, arg_0_144, arg_0_145, arg_0_146, arg_0_147, arg_0_148, arg_0_149, arg_0_150, arg_0_151, arg_0_152, arg_0_153, arg_0_154, arg_0_155, arg_0_156, arg_0_157, arg_0_158, arg_0_159, arg_0_160, arg_0_161, arg_0_162, arg_0_163, arg_0_164, arg_0_165, arg_0_166, arg_0_167, arg_0_168, arg_0_169, arg_0_170, arg_0_171, arg_0_172, arg_0_173, arg_0_174, arg_0_175, arg_0_176, arg_0_177, arg_0_178, arg_0_179, arg_0_180, arg_0_181, arg_0_182, arg_0_183, arg_0_184, arg_0_185, arg_0_186, arg_0_187, arg_0_188, arg_0_189, arg_0_190, arg_0_191, arg_0_192, arg_0_193, arg_0_194, arg_0_195):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39, arg_0_40, arg_0_41, arg_0_42, arg_0_43, arg_0_44, arg_0_45, arg_0_46, arg_0_47, arg_0_48, arg_0_49, arg_0_50, arg_0_51, arg_0_52, arg_0_53, arg_0_54, arg_0_55, arg_0_56, arg_0_57, arg_0_58, arg_0_59, arg_0_60, arg_0_61, arg_0_62, arg_0_63, arg_0_64, arg_0_65, arg_0_66, arg_0_67, arg_0_68, arg_0_69, arg_0_70, arg_0_71, arg_0_72, arg_0_73, arg_0_74, arg_0_75, arg_0_76, arg_0_77, arg_0_78, arg_0_79, arg_0_80, arg_0_81, arg_0_82, arg_0_83, arg_0_84, arg_0_85, arg_0_86, arg_0_87, arg_0_88, arg_0_89, arg_0_90, arg_0_91, arg_0_92, arg_0_93, arg_0_94, arg_0_95, arg_0_96, arg_0_97, arg_0_98, arg_0_99, arg_0_100, arg_0_101, arg_0_102, arg_0_103, arg_0_104, arg_0_105, arg_0_106, arg_0_107, arg_0_108, arg_0_109, arg_0_110, arg_0_111, arg_0_112, arg_0_113, arg_0_114, arg_0_115, arg_0_116, arg_0_117, arg_0_118, arg_0_119, arg_0_120, arg_0_121, arg_0_122, arg_0_123, arg_0_124, arg_0_125, arg_0_126, arg_0_127, arg_0_128, arg_0_129, arg_0_130, arg_0_131, arg_0_132, arg_0_133, arg_0_134, arg_0_135, arg_0_136, arg_0_137, arg_0_138, arg_0_139, arg_0_140, arg_0_141, arg_0_142, arg_0_143, arg_0_144, arg_0_145, arg_0_146, arg_0_147, arg_0_148, arg_0_149, arg_0_150, arg_0_151, arg_0_152, arg_0_153, arg_0_154, arg_0_155, arg_0_156, arg_0_157, arg_0_158, arg_0_159, arg_0_160, arg_0_161, arg_0_162, arg_0_163, arg_0_164, arg_0_165, arg_0_166, arg_0_167, arg_0_168, arg_0_169, arg_0_170, arg_0_171, arg_0_172, arg_0_173, arg_0_174, arg_0_175, arg_0_176, arg_0_177, arg_0_178, arg_0_179, arg_0_180, arg_0_181, arg_0_182, arg_0_183, arg_0_184, arg_0_185, arg_0_186, arg_0_187, arg_0_188, arg_0_189, arg_0_190, arg_0_191, arg_0_192, arg_0_193, arg_0_194, arg_0_195]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

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
class TestPrimitiveOp_32cab10c8e50f2f931fb6d322d065030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1b4603f75901cd6c46094dc422f0b3a
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9d3a258501ae313e7210c17218f19aaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 9, 9], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_130e8750601a4f0f625906b474ea09ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 68, 68], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c9e6207fa0418118dc2a59361f0509bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cda29f2042bf32fa9f8c2460d6ce16ab
    def get_inputs(self):
        return [
            paddle.uniform([165888, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([41472, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([10368, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2592, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([648, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5f5d0b6974fa9805877e98e88ed87ca6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 165888, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 41472, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 10368, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2592, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 648, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_860654f2bed89bb24bbc870e92830a95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eef316d18b5fb11186e8298f0d1ffd0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 165888, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 41472, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 10368, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2592, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 648, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_aa0ed63fa18c92d0b252ecf436f49151(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_994ddb9387838e8b249c614656d2d184
    def get_inputs(self):
        return [
            paddle.uniform([1, 5184, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1296, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 324, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_abbcbfe2f491fc69098e19932b81c15b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_994ddb9387838e8b249c614656d2d184
    def get_inputs(self):
        return [
            paddle.uniform([1, 5184, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1296, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 324, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_93f1c7ae9a8cff55e4c5d06792c59950(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_994ddb9387838e8b249c614656d2d184
    def get_inputs(self):
        return [
            paddle.uniform([1, 5184, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1296, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 324, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e4524d9e330785af3eca8a5f3f93d335(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4c82d897a5f6ce2fa44b242736a2356
    def get_inputs(self):
        return [
            paddle.uniform([5184, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1296, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([324, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3b97c831b4b5bc78ab0b40458ed7f759(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4c82d897a5f6ce2fa44b242736a2356
    def get_inputs(self):
        return [
            paddle.uniform([5184, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1296, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([324, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9144ba3bb6760400f503f022fa6bf4de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25b669a33790e1cb93baf70e0dad6601
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b4016af76931412297fb95dc8e2703f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_dc14b264dedb879c56c44323d7f3cf03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_506063ee61f999cca591ba0fe8363fc8
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[185658], dtype='int64'), 'int32'),
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
class TestPrimitiveOp_273f16646a252298b0e362818fb637a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22d1f92ef4d41eb6531a218349aaac3b
    def get_inputs(self):
        return [
            paddle.uniform([220968, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e0eb4896fdf4b27a46ea6a3d60f584d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86df21567526a882232bff0fdead57f9
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4c32c25f30bc2f9639408751b6570c56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a7717ea8be4c5963bbf56d4ac99ea01
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_aaf553944ac6d15760f22eef206dd440(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 48, 48], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9089f8027dfceb96f4e59d9a3591c9db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_10276f66407483172edd90e09707293d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c76e1849d0da6371394c425aa0f4ae37
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4, 1, 25, 38], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9c88d9e18d7e6ef816ba1a3364507aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 48, 48], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f826e5047fedbdac52df421a234aa47d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9b02dd8ba6d2b5e4d09c08bb28d31025(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cdc49fe5a86167cdad2f85bc9794b61f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c76e1849d0da6371394c425aa0f4ae37
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4, 1, 7, 10], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4b1789a6099a51e0e0b04a04ea45343e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 58, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 58, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_460716ad6e63163a4b32ce8cee5d765d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e842a9119a9dae0bf38ba8469b31ac6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8a4c853c27d1125a34da22396c18ad0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22d1f92ef4d41eb6531a218349aaac3b
    def get_inputs(self):
        return [
            paddle.uniform([185658, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_299dfc5e28e0baa4d020ea013db610b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5b547993fee6e09572f7683eaf029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6c367a316bf4341db60d8c30e0243f6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c76e1849d0da6371394c425aa0f4ae37
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4, 1, 100, 152], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_39cf29139d6456a72215f43ae233535d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5ec4efca318aba690efc1d510e89395
    def get_inputs(self):
        return [
            paddle.uniform([11, 512, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1fb7b765aff5785130387da30b914291(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_197fab3f1269fd6ba5830e1f082384d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 256, 256], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_3ef70478fc712820490bd8bb9d39d979(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a3a2c631a08e9ff6737629170d8911a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ef70478fc712820490bd8bb9d39d979
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 64, 64], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_1f20c22d7c3063914256129accdfdcac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = -1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 21504, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dc070b47a8e63d7a521dcfb14facade4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f20c22d7c3063914256129accdfdcac
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21504, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_9c7001a5882bd456f3334bad84eb8c5a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_97782d2d2629d8db5968e692cf0c8585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c7001a5882bd456f3334bad84eb8c5a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 24, 36], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_799d9ffc24efea3ff249b73a82fa3636(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 24, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d462f1e290b906659047fa4f67f6585d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_799d9ffc24efea3ff249b73a82fa3636
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 24, 36], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_1e36cc2ee7a9d2a939cb3479f45d3d7a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_568f877109f370e493bfc5be0a6d105b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e36cc2ee7a9d2a939cb3479f45d3d7a
    def get_inputs(self):
        return [
            paddle.uniform([129024, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([32256, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([8064, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2016, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([528, 4], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_c2880837030fd9e89aaee8956486156b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ad884ffebd0d9c96038fbb837aded1cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2880837030fd9e89aaee8956486156b
    def get_inputs(self):
        return [
            paddle.uniform([1, 129024, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32256, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8064, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2016, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 528, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_b43068f6ce3c4b1ede264886e9f148b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_28952f729381c0bf20dd68bc4635722f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b43068f6ce3c4b1ede264886e9f148b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 129024, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32256, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8064, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2016, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 528, 4], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_2473ed47afd0ea22b4a9727c1ace3e33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_017dc236e70a682b0a27233fe8e571e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2473ed47afd0ea22b4a9727c1ace3e33
    def get_inputs(self):
        return [
            paddle.uniform([300, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
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

class PrimitiveOp_31a8a5dba3bdb94943a131627fcfb21b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d66f08ff2f16a7a0aa9dea0339b3a2f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31a8a5dba3bdb94943a131627fcfb21b
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 152, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 152, 152], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_433205b39c217d20dd7611bfb9b99056(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5dedb7a83f328e2ccfb92bbf9e313078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_433205b39c217d20dd7611bfb9b99056
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_80367a8b42b048ced4695c0aab6596b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4958af4f9827ad8f7f4f5f8aa3aa644f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80367a8b42b048ced4695c0aab6596b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4e2f3c4cad9e2716278f430f850f728e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2473ed47afd0ea22b4a9727c1ace3e33
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
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

class PrimitiveOp_b7af34eb9ac2afbc36c2831855253b28(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2a9a3d97a7db79f82a4323a9c1576209(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7af34eb9ac2afbc36c2831855253b28
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
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

class PrimitiveOp_73562921390a687324f2ee21fcd3144a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = -1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cd0b6572416c1aceb3e53f1488ef9f75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73562921390a687324f2ee21fcd3144a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_45d036f411a5d30f66685966d6d199e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e36cc2ee7a9d2a939cb3479f45d3d7a
    def get_inputs(self):
        return [
            paddle.uniform([115200, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([28800, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([7200, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1800, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([450, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_60a8998c22b4a4b9986f481e79e22ad7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2880837030fd9e89aaee8956486156b
    def get_inputs(self):
        return [
            paddle.uniform([1, 115200, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 28800, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 7200, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1800, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 450, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5f6aeb469b786afaff212515c92255c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b43068f6ce3c4b1ede264886e9f148b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 115200, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 28800, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 7200, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1800, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 450, 4], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_98281a708fe0868e28534120d356331f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, None, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, None, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b2b3dcc5d0f3b00b6f634324f852f9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98281a708fe0868e28534120d356331f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4096, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 16384, 2], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_47c4442d9152104f49dca64a513f7e2e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_da068ec8a2b875f7a7147aec2ece11e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47c4442d9152104f49dca64a513f7e2e
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4096, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 16384, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b48b37aa737086655af4eb591aef0a04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2473ed47afd0ea22b4a9727c1ace3e33
    def get_inputs(self):
        return [
            paddle.uniform([100, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
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

class PrimitiveOp_a2dd67f5f09297d85ceb86433100e780(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 36, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 36, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f4cdb04cc884c6b1aef3184ed3a9f54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2dd67f5f09297d85ceb86433100e780
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 104, 104], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_13f45e83991fd1b23933f593973e75ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 60, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2b8f71505ac3a2a39b7f096099c30a9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13f45e83991fd1b23933f593973e75ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 60, 24, 24], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_c26f907bce7efcedb842206223512c93(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 20, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bc6819f24b1c6e605e62e2a398128da8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c26f907bce7efcedb842206223512c93
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_58ba41064749534a0212c914bca3c9cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_559d83bc85a9b08f2ef32a73dcf562d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58ba41064749534a0212c914bca3c9cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_35094ce73bfe260ff4542a7e890b9c0c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cd820435a005aadc03e2208f3eaad4e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35094ce73bfe260ff4542a7e890b9c0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_997347c0e9cbb0b30a1bd984317e156b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 150, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b560d363c5f7e22420efa4ef945df2f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_997347c0e9cbb0b30a1bd984317e156b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 150, 384], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_c3f91e27db77f9cacfdc08ad834b9ab9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0):
        input_0 = [arg_0_0]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 500, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_536d9261fade5955b1a53a79b9c8b891(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3f91e27db77f9cacfdc08ad834b9ab9
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'), 'int64'),
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

class PrimitiveOp_09a1672047d76d6f986fd618c8b8c535(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 2
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 500, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[None, None, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b729778be5622c5b75a281a2a02094ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09a1672047d76d6f986fd618c8b8c535
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'), 'int64'),
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

class PrimitiveOp_9348f06e5615d4c76b7afb4283402a2e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[9216, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[2304, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[576, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_30b44dc0b881ac24a4ffa74ca121a998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9348f06e5615d4c76b7afb4283402a2e
    def get_inputs(self):
        return [
            paddle.uniform([9216, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([2304, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([576, 2], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_784d2d3c35923690784b8ef5e87ebc6e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[9216, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2304, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[576, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7b54de89f471c9e2975a232766978e60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_784d2d3c35923690784b8ef5e87ebc6e
    def get_inputs(self):
        return [
            paddle.uniform([9216, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2304, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([576, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_fd897bc05ad0fb873a430e6552494325(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = -1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12096, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 12096, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f820f91142474712d70ce793ce1bad40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd897bc05ad0fb873a430e6552494325
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_ea8cdc047f5192688d0482793e9ce0e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_89e2fa581d11d9e547a10783341b1154(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea8cdc047f5192688d0482793e9ce0e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 48, 48], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f65bf268f0e7799eebc0699dd16d8922(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35094ce73bfe260ff4542a7e890b9c0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_62636c82df265cf47be465804593c945(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2473ed47afd0ea22b4a9727c1ace3e33
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
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

class PrimitiveOp_ee005a50700d095e972b44ded04f19e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d3d14957004ff54b046a009282646227(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee005a50700d095e972b44ded04f19e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 112, 112], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_0f22b0198b08fd1badb247475a1e621a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_680ce8db931d1b50a789e1950fa6796b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f22b0198b08fd1badb247475a1e621a
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 48, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 48, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d0c479bf15a598b00d5b96fb4391551e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee005a50700d095e972b44ded04f19e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9dfa4a2822ef16c44571888e775e7d54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea8cdc047f5192688d0482793e9ce0e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f57fcd51423c20b9fef43efe8c485b96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea8cdc047f5192688d0482793e9ce0e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 34, 34], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1af046f281fd1fc04f04f4425cdea520(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea8cdc047f5192688d0482793e9ce0e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 64, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_aa5b284bc3f296912f8983250ba5ec41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_433205b39c217d20dd7611bfb9b99056
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_df2fa1672db466476fad4c930dcb9cca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ef70478fc712820490bd8bb9d39d979
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 13, 13], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_59ea20785fe331592f168fb5a36d4602(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 34, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a4edf0cbf1c8e7c4393b2372406e3cc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59ea20785fe331592f168fb5a36d4602
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 34, 128, 128], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_4252d85ff71455285e56365d7ad139aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 36, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c4c7d5f7386fae08f7ff03a7d59cc088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4252d85ff71455285e56365d7ad139aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 160, 240], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_17a718f4eb174b49b690bc34bce5c76a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f9592204ba520ef8d552d284991ff684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17a718f4eb174b49b690bc34bce5c76a
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 60, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_68638f35f6c7420c5173ae814c6d9ba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73562921390a687324f2ee21fcd3144a
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_dd49adb68a7bbe1bce9e5b8b0d287845(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ef70478fc712820490bd8bb9d39d979
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a32bddc7bc20ccc0e855acfc33e786ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c7001a5882bd456f3334bad84eb8c5a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_5e958785ac0c63d45d513c6d4a9fa256(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47a47617e34fd7e8cf0208eca6c48c8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e958785ac0c63d45d513c6d4a9fa256
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 25, 38], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4906bfe41dbdadf3eee91898dbce14aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73562921390a687324f2ee21fcd3144a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_a087c0ed7f4aaa3471b1cfc37d423f6d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c4a9abc8392e5e0321fbc39adf67bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a087c0ed7f4aaa3471b1cfc37d423f6d
    def get_inputs(self):
        return [
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_acdbd82ef2c0cf905efdaef83eaa2874(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_969de9cfc415698cb0b7f35e92581bc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acdbd82ef2c0cf905efdaef83eaa2874
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 30, 30], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_db673b643e4c55dfaf0fbee076dfe6c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ef70478fc712820490bd8bb9d39d979
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 22, 22], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_67b3c62fed5aab8bd14155a9c4434107(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13f45e83991fd1b23933f593973e75ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 60, 16, 16], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_942bd872d64afcbcb05e92db13cb4665(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 19, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b7159b496d6484e443a2168a0fee6d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_942bd872d64afcbcb05e92db13cb4665
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 19, 256, 512], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_22818e8e2bfa8be25c3d27ec76799d7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ed91feed0b31e247af3414002751b8a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22818e8e2bfa8be25c3d27ec76799d7c
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 64, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 64, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 64, 240, 240], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0058a7ced8b7cb0607adcbba77701079(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73562921390a687324f2ee21fcd3144a
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_80a51fcb51a6db87c1dc026c91c880dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c7001a5882bd456f3334bad84eb8c5a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 20, 30], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_6dcaa1ce61fa95268ed345d2825b34e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 20, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f6cb9c4e4811026c179af3bede4d12f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dcaa1ce61fa95268ed345d2825b34e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 20, 30], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ee801015452693b8c89255e1e56aab52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35094ce73bfe260ff4542a7e890b9c0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_eaa306f0a7ddaabb0f22240afc6e93c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4096, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ffd6cf39f274ab29ae48c8521f1d717b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eaa306f0a7ddaabb0f22240afc6e93c1
    def get_inputs(self):
        return [
            paddle.uniform([4096, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_6a5dd1f63dc051ec95eec065fab54d1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4096, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e08483096722d7cb393aee51db6fcce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a5dd1f63dc051ec95eec065fab54d1c
    def get_inputs(self):
        return [
            paddle.uniform([4096, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_fb24588033873f0f70bbd871fd7d9ed2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = -1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 5376, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 5376, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ecf8efb79e1bd95879d3f874b2ba35f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb24588033873f0f70bbd871fd7d9ed2
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c7ed931ece992e306ec237ad2432cffc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2473ed47afd0ea22b4a9727c1ace3e33
    def get_inputs(self):
        return [
            paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
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

class PrimitiveOp_4be3a8231da702deb8fef07e9508237e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_24fb430ff16590d0ef7acd6be6ccc458(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4be3a8231da702deb8fef07e9508237e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 44, 44], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_de589d50f235561d57a3cc192c3641f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e4b9226574784917e3115f8de4dfcb9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de589d50f235561d57a3cc192c3641f1
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_59bcbb2c01296c9247b85ef328d27e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee005a50700d095e972b44ded04f19e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 96, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 96, 96], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_7860456e5b218ff3d426a422de58ab74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_16be45e8b0a91006833d32144090c317(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7860456e5b218ff3d426a422de58ab74
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_1263204eb69cc4a39289f84e2bb3d695(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 288, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 288, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_273cb936b47dfc9c46b99efb76d2df7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1263204eb69cc4a39289f84e2bb3d695
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_e4b495b679f7bed8eb4fc8908ab8a566(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_675e0929cf761644f4174e7672923287(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4b495b679f7bed8eb4fc8908ab8a566
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_c1045e4f9b39505f73b96d9d5348b612(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4ce01bd33b1b0a669944637f05c82a7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1045e4f9b39505f73b96d9d5348b612
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b235bfcfebfc2b6cc16d78c914d923cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80367a8b42b048ced4695c0aab6596b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_2faba8c3833f218e16392764900cbf37(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eeda9ffbe68f0e20bf5d38a26767d9ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2faba8c3833f218e16392764900cbf37
    def get_inputs(self):
        return [
            paddle.uniform([6, 256, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
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
class TestPrimitiveOp_883a238ab4f1eec1d7c6e686570acffb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1263204eb69cc4a39289f84e2bb3d695
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_25d9a75ddb3c07aba46610fbf9c972e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 97, 97], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d155940198f6c4620f3aa951cd9970a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25d9a75ddb3c07aba46610fbf9c972e8
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_42d6a0f9a890417ebf957f451d15f96d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35094ce73bfe260ff4542a7e890b9c0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_65421f764ad74d8baa372b24f3a7198b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee005a50700d095e972b44ded04f19e7
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 48, 112, 112], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_1968fb9580f0fe4daed07f40792aff24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_57ff6e92410f3b963541c37d43f17b30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1968fb9580f0fe4daed07f40792aff24
    def get_inputs(self):
        return [
            paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_266cf604a3222aa618900f71757e04f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acdbd82ef2c0cf905efdaef83eaa2874
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_0870e77c53e7835f2d5a4b4e5a6f6806(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3b718551a2196082ea0073cad1e649e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0870e77c53e7835f2d5a4b4e5a6f6806
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ba9333aaf6484aaa8e50550d208f85cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0870e77c53e7835f2d5a4b4e5a6f6806
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c86d9a37bc2d3ad6887a29553a555628(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35094ce73bfe260ff4542a7e890b9c0c
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_834cb9311a1b6cce4d8d0ea085e09d2a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c6ebd01e61eef3efe5f4e4cb6a3e05c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_834cb9311a1b6cce4d8d0ea085e09d2a
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8d3aebe362fbe4ef631233397d546f09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_834cb9311a1b6cce4d8d0ea085e09d2a
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e527d9a957790b40de8d9879aafc6784(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2473ed47afd0ea22b4a9727c1ace3e33
    def get_inputs(self):
        return [
            paddle.uniform([3, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
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
class TestPrimitiveOp_45c8153d90221c1edbf19d62a432c173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4be3a8231da702deb8fef07e9508237e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c62746697649cd4098e469c2cd9969ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31a8a5dba3bdb94943a131627fcfb21b
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 160, 160], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_628c0491bf32a7c9b9d8ee2e7cb89cab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 20, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 20, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_95d095914b49da932cd52d0737e36724(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_628c0491bf32a7c9b9d8ee2e7cb89cab
    def get_inputs(self):
        return [
            paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_43147cb1457bf6f332746b83596f50c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e36cc2ee7a9d2a939cb3479f45d3d7a
    def get_inputs(self):
        return [
            paddle.uniform([139392, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([34848, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([8712, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2178, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([528, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_859ec154058fb413db0368585dc59068(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2880837030fd9e89aaee8956486156b
    def get_inputs(self):
        return [
            paddle.uniform([1, 139392, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 34848, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8712, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2178, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 528, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1739073dd591395d092696790cc55861(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b43068f6ce3c4b1ede264886e9f148b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 139392, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 34848, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8712, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2178, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 528, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3a72db905628d4790b0c3b3856ed6ac9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee005a50700d095e972b44ded04f19e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_04cf0d5cdfcbd82d8da8e87e386c2b95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea8cdc047f5192688d0482793e9ce0e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 30, 30], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_2e7dfd16fcb884a13d005fc87aeec173(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_af1c963907703103b81adf73de714bcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e7dfd16fcb884a13d005fc87aeec173
    def get_inputs(self):
        return [
            paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2ee444c813b159856c17705105100e59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13f45e83991fd1b23933f593973e75ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 168, 168], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 60, 168, 168], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a99c3c0ea1c6a0618b545851e70cb3fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea8cdc047f5192688d0482793e9ce0e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8af0a3b20840531296eeac9fe6f9a8ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea8cdc047f5192688d0482793e9ce0e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cb295e2363c6356102345dc782132bc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acdbd82ef2c0cf905efdaef83eaa2874
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c788f84240c1308d15ae83de7a009d3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0870e77c53e7835f2d5a4b4e5a6f6806
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_126600faf5437b9e19ef73afeb963d6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0870e77c53e7835f2d5a4b4e5a6f6806
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_493574f6e82becfd1d80d9809d992b4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35094ce73bfe260ff4542a7e890b9c0c
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6d649bc33ea955c992c64de13210d855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_834cb9311a1b6cce4d8d0ea085e09d2a
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9b076d6e031168d1eb1f1bdb9af88e27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_834cb9311a1b6cce4d8d0ea085e09d2a
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8ff594afce2277f1ced1098c096f1299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22818e8e2bfa8be25c3d27ec76799d7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3856f3d7b42a3613b4fda16d6e08d424(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59ea20785fe331592f168fb5a36d4602
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 34, 160, 160], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_595132b9aac53f462d70291c3abeef01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acdbd82ef2c0cf905efdaef83eaa2874
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 60, 60], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_bcd52a0213178c2480a8285fef8f8ab2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4bb4e6081ed61de7b5ce70b4ef375281(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcd52a0213178c2480a8285fef8f8ab2
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 52, 52], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5afcc643ee73a5d419b6fe1b248f013f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee005a50700d095e972b44ded04f19e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 76, 76], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_078cd73e28e15c599491af1e333f778b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73562921390a687324f2ee21fcd3144a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_737adf6c377a8d81dfaf2b7d5e4f263b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2faba8c3833f218e16392764900cbf37
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
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
class TestPrimitiveOp_60a91e5f26599efdfb7c47e17238644f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35094ce73bfe260ff4542a7e890b9c0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5e184783b84ee440a02c0b8d673f2cf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80367a8b42b048ced4695c0aab6596b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_ed4fcc16a3cd13684fe012d39dc0a27f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6400, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1600, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[400, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2b8ebbdb3882bcdaf39d4518ca4e5b8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed4fcc16a3cd13684fe012d39dc0a27f
    def get_inputs(self):
        return [
            paddle.uniform([6400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1600, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([400, 2], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_0dd12bf482b30bb58df1b32b3a62a0f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6400, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1600, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[400, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8792206db1a8ef2f39570cba3a20073e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0dd12bf482b30bb58df1b32b3a62a0f1
    def get_inputs(self):
        return [
            paddle.uniform([6400, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1600, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([400, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_003be30fb7679c8475e24644c7e8f579(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = -1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8400, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 8400, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_85ddda60829640154b47750e7cd5bab8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_003be30fb7679c8475e24644c7e8f579
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_e69dbf8c33829a79a8908ef294a961fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39, arg_0_40, arg_0_41, arg_0_42, arg_0_43, arg_0_44, arg_0_45, arg_0_46, arg_0_47, arg_0_48):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39, arg_0_40, arg_0_41, arg_0_42, arg_0_43, arg_0_44, arg_0_45, arg_0_46, arg_0_47, arg_0_48]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d95316f05b7de73be0ccf400101e6ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e69dbf8c33829a79a8908ef294a961fa
    def get_inputs(self):
        return [
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_2a568e9bea98897d3e347931bd1a4e1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ff1c893a26122431a14427156d2cbbe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a568e9bea98897d3e347931bd1a4e1b
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b72ca44868236ab85b5563e2739b42c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4252d85ff71455285e56365d7ad139aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 176, 264], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_badca310622ad9c9deda73a6cd34b284(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73562921390a687324f2ee21fcd3144a
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c4f2e1353ce30ce5e9f17a421229f992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22818e8e2bfa8be25c3d27ec76799d7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 184, 328], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_7e5d5492c9b32674d0b705b12e1c1f4d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_734d3c0d52aaa388ed32fe2d5f0c1599(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e5d5492c9b32674d0b705b12e1c1f4d
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_21d60c56c258fa3b5abab0b4e7f94226(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2473ed47afd0ea22b4a9727c1ace3e33
    def get_inputs(self):
        return [
            paddle.uniform([7, 64, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 64, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 64, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 64, 7, 7]),
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
class TestPrimitiveOp_c6440a0d95c1372db6df932a5264bbe1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31a8a5dba3bdb94943a131627fcfb21b
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 136, 136], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 136, 136], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2ab7b6b4c9aaf0cc312b7cf7bf9727fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35094ce73bfe260ff4542a7e890b9c0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_35cc2be78a5a2b9d8305bb0c6864a26b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 2
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 4, 13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4, 1, 13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d78dba6c47344273d87399c3e19896b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35cc2be78a5a2b9d8305bb0c6864a26b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4, 1, 13, 19], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_c850c6281e2c27bd401456c79b2cf6a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_379048018d356eea2e8a1ca10d075851(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c850c6281e2c27bd401456c79b2cf6a8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int32').reshape([1]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
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

class PrimitiveOp_9ab3a71d690711355761a3cb0d9bc16e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_deb8a2e5ba5351a85a42cf9bbba7ad2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ab3a71d690711355761a3cb0d9bc16e
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 2, 3, 4], dtype='int32').reshape([4]),
            paddle.to_tensor([0, 0], dtype='int32').reshape([2]),
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

class PrimitiveOp_bc6a404ee87fd81db23031ff132a25da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_095bb805361f7cddf005445aaa594761(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc6a404ee87fd81db23031ff132a25da
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 80, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8c27fc94dc3b6fcb31759eb576959ce3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_433205b39c217d20dd7611bfb9b99056
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 44, 44], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_dcbdccf129feb124484aa6ccd296419d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e36cc2ee7a9d2a939cb3479f45d3d7a
    def get_inputs(self):
        return [
            paddle.uniform([182400, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([45600, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([11400, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2850, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([741, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_93bec5967ac8dde2bb2239b16f91a239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2880837030fd9e89aaee8956486156b
    def get_inputs(self):
        return [
            paddle.uniform([1, 182400, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 45600, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 11400, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2850, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 741, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6c7b952d7ada35a9c10a6ed55c9fb6de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b43068f6ce3c4b1ede264886e9f148b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 182400, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 45600, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 11400, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2850, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 741, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6464c36fd37c49e44786b7b126f936e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee005a50700d095e972b44ded04f19e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_163b59d340b510113d7f0315b2c07261(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1525ffdf01c3a37a1a0706d0c725b820(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_163b59d340b510113d7f0315b2c07261
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b856465e245c4b8dd7b1cde368e98d2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31a8a5dba3bdb94943a131627fcfb21b
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 104, 104], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a0ce61346d824d2506ce36236ba369b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee005a50700d095e972b44ded04f19e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 184, 184], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 184, 184], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cff70f840d446ec21582443fad33ce93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea8cdc047f5192688d0482793e9ce0e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_45a8ff39567bf44c93f5e76a64945d7e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[784, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[3136, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_62ae58d743bf7560619d210b04f501c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45a8ff39567bf44c93f5e76a64945d7e
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([784, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3136, 4], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_220fbf25f413f72a49cb69231bfd983b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[784, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[3136, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_80abe811823414770c3ce466e2ca20dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_220fbf25f413f72a49cb69231bfd983b
    def get_inputs(self):
        return [
            paddle.uniform([196, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([784, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3136, 2], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_21a8447fe63026eb4bbb8cf2da97df5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[784, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3136, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7da46c89116d7a6adde5569f54d8f6e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21a8447fe63026eb4bbb8cf2da97df5d
    def get_inputs(self):
        return [
            paddle.uniform([196, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([784, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3136, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d294c19d0003dec0e9ee3a25fef96a6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acdbd82ef2c0cf905efdaef83eaa2874
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_694c21c347bcf52564db466e0a794f84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0870e77c53e7835f2d5a4b4e5a6f6806
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7a5f95e47d0b852d040f024478e3b6d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0870e77c53e7835f2d5a4b4e5a6f6806
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1f45fb93d486b391801627749c96607e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35094ce73bfe260ff4542a7e890b9c0c
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8a071a9e373118220ee878ae9a6d643d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_834cb9311a1b6cce4d8d0ea085e09d2a
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d5a6d26d1c407f8a60faabd21396853b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_834cb9311a1b6cce4d8d0ea085e09d2a
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a02788bf16707b1516d483b62a72f468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e36cc2ee7a9d2a939cb3479f45d3d7a
    def get_inputs(self):
        return [
            paddle.uniform([65280, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([16320, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4080, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1020, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([270, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_41fd59a264d7102df456ed8ce941d93c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2880837030fd9e89aaee8956486156b
    def get_inputs(self):
        return [
            paddle.uniform([1, 65280, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 16320, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4080, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1020, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 270, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_685591e04b1852e6e3f7f67c5d1ecff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b43068f6ce3c4b1ede264886e9f148b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 65280, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 16320, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4080, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1020, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 270, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cb57ec731d5d99e636b0242c52f33a43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2473ed47afd0ea22b4a9727c1ace3e33
    def get_inputs(self):
        return [
            paddle.uniform([5, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
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

class PrimitiveOp_55f060b21c167e9669f2c30d2abaf2b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 200, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 200, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5efce9618e23191047af138b588d201e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55f060b21c167e9669f2c30d2abaf2b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 22, 22], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d72fe73c861bc53530753fa120fcf56b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4be3a8231da702deb8fef07e9508237e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 18, 18], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_8bb7aa41ef536444effdda33871581f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f7a6ebac13c95323c24a3eb483185b0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8bb7aa41ef536444effdda33871581f2
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b73a2b40d035c5fcbf76822af82419a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ef70478fc712820490bd8bb9d39d979
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 9, 9], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_219efd117b813fa533910f56d0fb7124(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea8cdc047f5192688d0482793e9ce0e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9e1640e60638ad0ab30be13f1b6c068e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e7dfd16fcb884a13d005fc87aeec173
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 88, 88], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7e404f619ee4bfbf6713086c209ed826(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee005a50700d095e972b44ded04f19e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0b7f926c66df9e1814aec6d3e69abaaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e7dfd16fcb884a13d005fc87aeec173
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 52, 52], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_940506351a08bd6be9f14c3fa3744caf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55f060b21c167e9669f2c30d2abaf2b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 13, 13], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_4c5c9dea225188d5c02493008b2f87b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2219a469bb157a269e42a8430849d563(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c5c9dea225188d5c02493008b2f87b4
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_adaeb5f0cfc7aa387e49480b77625426(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35094ce73bfe260ff4542a7e890b9c0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 15, 15], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_01e2b0de16bcb2649d00b3d8ddfc936a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ef70478fc712820490bd8bb9d39d979
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 42, 42], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 42, 42], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_87a9a5505efdf4ab96e491291668be5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcd52a0213178c2480a8285fef8f8ab2
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 60, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b918710ee6f7673e2a3c087b57dd61fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4b495b679f7bed8eb4fc8908ab8a566
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_28df817ca7946ef50c38e50848170d48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ecf8eb61f5806e700503d0cd6f6d60e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28df817ca7946ef50c38e50848170d48
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 7, 7], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_69051f2f6bbb0cd4fe2dd2f1302278ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_433205b39c217d20dd7611bfb9b99056
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_56bc072fe13aa86a41c5e831696011ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 100, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9850893c956c1bebf30878410621d0e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56bc072fe13aa86a41c5e831696011ea
    def get_inputs(self):
        return [
            paddle.uniform([22, 100, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 100, 7, 7], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f8ec655881e16c7b6593a352b231f9bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35094ce73bfe260ff4542a7e890b9c0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_fe7a892dc61053d0b98518a58ccb1b47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ff8f167270f8543707b9495858129fa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe7a892dc61053d0b98518a58ccb1b47
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3129ccfe0afc3b46bf130590e7234519(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73562921390a687324f2ee21fcd3144a
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_15fa5a41f857c2a3eb0c26c9727b05ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bdd7a556e43d26ba279ef134f54d0122(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15fa5a41f857c2a3eb0c26c9727b05ad
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_53d9df58267c3f9b4e291bfd6198b3b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea8cdc047f5192688d0482793e9ce0e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_56b393be6cfa3c3ec8f2a0a6fffa471f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0):
        input_0 = [arg_0_0]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e5d14a18819acc4b7fb2d14c3ee4b7b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56b393be6cfa3c3ec8f2a0a6fffa471f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07804898172616959, 0.3275541067123413, 0.11523129791021347, 0.2340201884508133]], dtype='float32').reshape([1, 4]),
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
class TestPrimitiveOp_7bf072fe8ae25fbbc20df05f10081ff1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56bc072fe13aa86a41c5e831696011ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 26, 26], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_0013f65349bb7ff6da9861b50fff7746(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4624, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1156, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[289, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a000e1014f1b2cf8217a6c3c24b29d1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0013f65349bb7ff6da9861b50fff7746
    def get_inputs(self):
        return [
            paddle.uniform([4624, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1156, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([289, 2], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_be5758d4fc777ab887be59d0abb0456a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4624, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1156, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[289, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47c45654b43916fa4b92aa63b235ff83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be5758d4fc777ab887be59d0abb0456a
    def get_inputs(self):
        return [
            paddle.uniform([4624, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([289, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_6b5aa7f6d892b7c0f01c4739f2b2a108(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = -1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6069, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 6069, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_91488d381f091b8d2fdda589e5ad6d37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b5aa7f6d892b7c0f01c4739f2b2a108
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_21af8e1e1293bbb3c9dc0dee3d56d579(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35094ce73bfe260ff4542a7e890b9c0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_52c51e79b3123ad68fb1fe473596c790(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56b393be6cfa3c3ec8f2a0a6fffa471f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.24676227569580078, 0.09557920694351196, 0.211879700422287, 0.015435690991580486]], dtype='float32').reshape([1, 4]),
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
class TestPrimitiveOp_2c2bcb923c5f5c67697dea96ad294b71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e36cc2ee7a9d2a939cb3479f45d3d7a
    def get_inputs(self):
        return [
            paddle.uniform([84864, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([21216, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5304, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1326, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([351, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_264bc84703a75cebdf662068763ce9bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2880837030fd9e89aaee8956486156b
    def get_inputs(self):
        return [
            paddle.uniform([1, 84864, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21216, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5304, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1326, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 351, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2a8d49ab635a1fdcd73af69e71c4b8d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b43068f6ce3c4b1ede264886e9f148b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 84864, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21216, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5304, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1326, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 351, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3f0ebdfa7c65e96061281bceb9b3f323(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2473ed47afd0ea22b4a9727c1ace3e33
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
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
class TestPrimitiveOp_65d4416c5fdd97b3d3aed4445796e6d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31a8a5dba3bdb94943a131627fcfb21b
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 32, 32], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_b0e6e811c435d5d6c91668b303021540(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, None, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, None, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, None, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, None, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dba223453dc918ca293bd327bf14cf4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0e6e811c435d5d6c91668b303021540
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4096, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_51fba8c35a14a034259a7d755a384e6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2880837030fd9e89aaee8956486156b
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4096, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1024, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a5f70fdfd3c9ce8822ff2edd6ae8f5e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4be3a8231da702deb8fef07e9508237e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a8d5df3082eeea6946e4cdb63b3916fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35094ce73bfe260ff4542a7e890b9c0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_733a647afe94d03bf10a2d6cf5e8c686(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80367a8b42b048ced4695c0aab6596b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_41b7e6b497ddcd7284ce5b253f1f95c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73562921390a687324f2ee21fcd3144a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fbcec4a7b6923e5f3e8cfcea4d24f519(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e5d5492c9b32674d0b705b12e1c1f4d
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 30, 30], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6d4c324f69c45e11b8e7e7c5f9fd8941(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4b495b679f7bed8eb4fc8908ab8a566
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0a2f70228a14ee3d9307694c838db73d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c7001a5882bd456f3334bad84eb8c5a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 15, 25], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_141a0120a50699705a20d905073c94fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 15, 25], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b9a3706b0de964d02f5687bbea3312a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_141a0120a50699705a20d905073c94fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 15, 25], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ce73ae6d6ad25925d17afb64f866424d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31a8a5dba3bdb94943a131627fcfb21b
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 256, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fdcd6e2e8740272d63b43e001d8e92e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80367a8b42b048ced4695c0aab6596b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_13b58353e22186692abb7a64a7e954f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea8cdc047f5192688d0482793e9ce0e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5ec32572fa178046563d58c27479cf5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2faba8c3833f218e16392764900cbf37
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
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
class TestPrimitiveOp_b2b49489b651b69c1be80e8c3cd5cbd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4b495b679f7bed8eb4fc8908ab8a566
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_40e90f86f9d526f6823b36003661876c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 20, 128, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b5677a8e0a8c31efebe8f450dce9c703(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40e90f86f9d526f6823b36003661876c
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_27441bf90e3de7d07f1932b5046c5602(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 40, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 40, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb1e2ae73211edf747325afd0e1d3e1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27441bf90e3de7d07f1932b5046c5602
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_440714f3ea35e8cfea2b16ef7e937da9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 32, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 80, 32, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_db130a9acc660218715d808c9055abc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_440714f3ea35e8cfea2b16ef7e937da9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_f14a46b7ed25163da0c7d615cbe7c8ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 160, 16, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 160, 16, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_203ab9de46a76657850b90cd1d362d58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f14a46b7ed25163da0c7d615cbe7c8ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_57812d93c9b1922794bc140b1c510c39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea8cdc047f5192688d0482793e9ce0e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 92, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 92, 92], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9ffc30a947a491ed3ddbc36cd47005ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea8cdc047f5192688d0482793e9ce0e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_99c6584f735ec00619221754d8d083fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56bc072fe13aa86a41c5e831696011ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 44, 44], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_89c74effe3b3261d67f9141697d68c3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e36cc2ee7a9d2a939cb3479f45d3d7a
    def get_inputs(self):
        return [
            paddle.uniform([139392, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([34848, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([8712, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2178, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([561, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_034a0f00889d70e084a4d62097bffd04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2880837030fd9e89aaee8956486156b
    def get_inputs(self):
        return [
            paddle.uniform([1, 139392, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 34848, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8712, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2178, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 561, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fa14e048016ee81a57161eda9fc3b294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b43068f6ce3c4b1ede264886e9f148b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 139392, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 34848, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8712, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2178, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 561, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ddd447a1e49437621fec44f9041def30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee005a50700d095e972b44ded04f19e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 52, 52], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6aa43c53793c325c959da22c8114fb46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee005a50700d095e972b44ded04f19e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 40, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5d4b04c9948ef0dde72a080d8fab2a95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73562921390a687324f2ee21fcd3144a
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7277daae7d253c5b8c4ada1ca916da08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35094ce73bfe260ff4542a7e890b9c0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_d506faf8d8511adb893f05643f893b9a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 180, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 180, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_99c82c00240b871a21f53db89eeaa896(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d506faf8d8511adb893f05643f893b9a
    def get_inputs(self):
        return [
            paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_776d46796d67233fd4f70cb4bb2fc4d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe7a892dc61053d0b98518a58ccb1b47
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_a36a8a99f1853030b799bb90cf862fb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 2
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 4, 50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4, 1, 50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_06c4e47e00ace6f1ed2b3850001ebc11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a36a8a99f1853030b799bb90cf862fb7
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4, 1, 50, 76], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e16e461630291636badde741b41f43fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35094ce73bfe260ff4542a7e890b9c0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 46, 46], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 46, 46], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_7cb56efd6915efc62d5c19e0f6666f2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15200, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[3800, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[950, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[247, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[70, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2deef7e6cb7b4a7e529c4a28ed8efb30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cb56efd6915efc62d5c19e0f6666f2c
    def get_inputs(self):
        return [
            paddle.uniform([15200, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([950, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([247, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([70, 4], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_212d0b56ebc188eb238a83fe54080122(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15200, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[3800, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[950, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[247, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[70, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f361b8665e4ec6be7bc8623c08e0a4b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_212d0b56ebc188eb238a83fe54080122
    def get_inputs(self):
        return [
            paddle.uniform([15200, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([950, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([247, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([70, 2], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_c3eb6da7b44ddb5a5e08f70d5a414e8c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15200, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3800, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[950, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[247, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[70, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_824f0c5a3b7512c127459bdf53df860f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3eb6da7b44ddb5a5e08f70d5a414e8c
    def get_inputs(self):
        return [
            paddle.uniform([15200, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([950, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([247, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([70, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_515cb01a0690293aea36bba5c28489ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e5d5492c9b32674d0b705b12e1c1f4d
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 64, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a0ba5b5f644b1889f09882623f608156(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e36cc2ee7a9d2a939cb3479f45d3d7a
    def get_inputs(self):
        return [
            paddle.uniform([163200, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([40800, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([10200, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2550, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([663, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f1f19ae99fb427136d6e88c11a988069(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2880837030fd9e89aaee8956486156b
    def get_inputs(self):
        return [
            paddle.uniform([1, 163200, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40800, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 10200, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2550, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 663, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_29443e8199ceb51e88604a57bf70ae0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b43068f6ce3c4b1ede264886e9f148b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 163200, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40800, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 10200, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2550, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 663, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4dd316a1ebffd802c5af6d0c4dff3d13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4be3a8231da702deb8fef07e9508237e
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_25f43ccbbdef1478d2be40cf69631b22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4be3a8231da702deb8fef07e9508237e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 84, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 84, 84], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_08f44d1bfec9862a4848b38f1d60bc25(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_430bc09523f02733db3dede30476bf76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08f44d1bfec9862a4848b38f1d60bc25
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fd51e178700cd914608cc1fd78a58ed8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_433205b39c217d20dd7611bfb9b99056
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_da48a99991fbb98d6bbaf6a0da75141e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31a8a5dba3bdb94943a131627fcfb21b
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 80, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4e70cc65b2c61d6efec698f3dc71e146(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17a718f4eb174b49b690bc34bce5c76a
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 48, 48], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9ed15b2f4d21a3e97bcd9cc9573eb8b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acdbd82ef2c0cf905efdaef83eaa2874
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 24, 24], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_f050d0b25d07ca0d4e1f1bede5c88a23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c9403e9ebda99b8a4c40e6ee62036afb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f050d0b25d07ca0d4e1f1bede5c88a23
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_85ef37be271088ea6055049f9f7b60ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e36cc2ee7a9d2a939cb3479f45d3d7a
    def get_inputs(self):
        return [
            paddle.uniform([92928, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([23232, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5808, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1452, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([363, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7047af1512b4c2f945e715d60e0f5362(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2880837030fd9e89aaee8956486156b
    def get_inputs(self):
        return [
            paddle.uniform([1, 92928, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 23232, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5808, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1452, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 363, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8dd35adb2d338d04230ccd221121ea3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b43068f6ce3c4b1ede264886e9f148b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 92928, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 23232, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5808, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1452, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 363, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4b8fd9585e1374dfa24ad5fc5e415dce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2dd67f5f09297d85ceb86433100e780
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d7160fc882b36a2029c1cd29f033efa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcd52a0213178c2480a8285fef8f8ab2
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f0e4786dd1938f623a6dac14e8e5ca03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35094ce73bfe260ff4542a7e890b9c0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_01b4d83f12dedf1f533af47e668e8992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13f45e83991fd1b23933f593973e75ef
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_38a9ec47e99de0ca167215f0a2997bc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1263204eb69cc4a39289f84e2bb3d695
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_be61e38284a41d8ab66f68e234029242(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 150, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dcb118a7634d429cc3e28e82acff0bbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be61e38284a41d8ab66f68e234029242
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 150, 768], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_74c904bc1392991f9b21000111e074ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e7dfd16fcb884a13d005fc87aeec173
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 36, 36], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_724d6d5e907699ab6ec09fe6bd9051a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de589d50f235561d57a3cc192c3641f1
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 15, 15], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c8c36acadf5d6fbedc0678a337fba63b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea8cdc047f5192688d0482793e9ce0e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_42a963b34344a643967cf0ad565b7332(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_433205b39c217d20dd7611bfb9b99056
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_78217e3fe5494aa5b770dd83b7c4f283(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2dd67f5f09297d85ceb86433100e780
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 120, 120], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_712e9d34169bf3e600ccf0c50c45d23f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8bb7aa41ef536444effdda33871581f2
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9d93281595fb6950c3a3174012dac230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e36cc2ee7a9d2a939cb3479f45d3d7a
    def get_inputs(self):
        return [
            paddle.uniform([154560, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([38640, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([9660, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2415, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([648, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0f41700fcb082a34aae697db74fa576d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2880837030fd9e89aaee8956486156b
    def get_inputs(self):
        return [
            paddle.uniform([1, 154560, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 38640, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9660, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2415, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 648, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_08b3b96ac1cff10fc316d9cb320860fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b43068f6ce3c4b1ede264886e9f148b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 154560, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 38640, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9660, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2415, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 648, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_44628fd9a4a880360c303ce73202fd81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acdbd82ef2c0cf905efdaef83eaa2874
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_11c07a4237c0da75ab587d049fbf3ea4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0870e77c53e7835f2d5a4b4e5a6f6806
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cea0c271e7e018dabd2f8a608a55b148(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0870e77c53e7835f2d5a4b4e5a6f6806
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f8f9d450573ba11881e1d556ff955298(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35094ce73bfe260ff4542a7e890b9c0c
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2063adc7178d3d481145a4005543af21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_834cb9311a1b6cce4d8d0ea085e09d2a
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8e333b8d12ae859597b2f6f550ea7fc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_834cb9311a1b6cce4d8d0ea085e09d2a
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_213d83ae274763dade838bc9fc911f16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea8cdc047f5192688d0482793e9ce0e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 38, 38], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_588b5da3a4e67022f74c6f9143105737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13f45e83991fd1b23933f593973e75ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 60, 256, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8fd929ea0c820f2655e2e6921433819c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4b495b679f7bed8eb4fc8908ab8a566
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_52e7559c6ca88f4e39ea444caa0b510d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56bc072fe13aa86a41c5e831696011ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 18, 18], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_16fe7b83a58c274ef1573801d7f06061(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73562921390a687324f2ee21fcd3144a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_201d6e620dc8e5feb728d0c6a2b005c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39, arg_0_40, arg_0_41, arg_0_42, arg_0_43, arg_0_44, arg_0_45, arg_0_46, arg_0_47, arg_0_48, arg_0_49, arg_0_50, arg_0_51, arg_0_52, arg_0_53, arg_0_54, arg_0_55, arg_0_56, arg_0_57, arg_0_58, arg_0_59, arg_0_60, arg_0_61, arg_0_62, arg_0_63, arg_0_64, arg_0_65, arg_0_66, arg_0_67, arg_0_68, arg_0_69, arg_0_70, arg_0_71, arg_0_72, arg_0_73, arg_0_74, arg_0_75, arg_0_76, arg_0_77, arg_0_78, arg_0_79, arg_0_80, arg_0_81, arg_0_82, arg_0_83, arg_0_84, arg_0_85, arg_0_86, arg_0_87, arg_0_88, arg_0_89, arg_0_90, arg_0_91, arg_0_92, arg_0_93, arg_0_94, arg_0_95, arg_0_96, arg_0_97, arg_0_98, arg_0_99, arg_0_100, arg_0_101, arg_0_102, arg_0_103, arg_0_104, arg_0_105, arg_0_106, arg_0_107, arg_0_108, arg_0_109, arg_0_110, arg_0_111, arg_0_112, arg_0_113, arg_0_114, arg_0_115, arg_0_116, arg_0_117, arg_0_118, arg_0_119, arg_0_120, arg_0_121, arg_0_122, arg_0_123, arg_0_124, arg_0_125, arg_0_126, arg_0_127, arg_0_128, arg_0_129, arg_0_130, arg_0_131, arg_0_132, arg_0_133, arg_0_134, arg_0_135, arg_0_136, arg_0_137, arg_0_138, arg_0_139, arg_0_140, arg_0_141, arg_0_142, arg_0_143, arg_0_144, arg_0_145, arg_0_146, arg_0_147, arg_0_148, arg_0_149, arg_0_150, arg_0_151, arg_0_152, arg_0_153, arg_0_154, arg_0_155, arg_0_156, arg_0_157, arg_0_158, arg_0_159, arg_0_160, arg_0_161, arg_0_162, arg_0_163, arg_0_164, arg_0_165, arg_0_166, arg_0_167, arg_0_168, arg_0_169, arg_0_170, arg_0_171, arg_0_172, arg_0_173, arg_0_174, arg_0_175, arg_0_176, arg_0_177, arg_0_178, arg_0_179, arg_0_180, arg_0_181, arg_0_182, arg_0_183, arg_0_184, arg_0_185, arg_0_186, arg_0_187, arg_0_188, arg_0_189, arg_0_190, arg_0_191, arg_0_192, arg_0_193, arg_0_194, arg_0_195):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39, arg_0_40, arg_0_41, arg_0_42, arg_0_43, arg_0_44, arg_0_45, arg_0_46, arg_0_47, arg_0_48, arg_0_49, arg_0_50, arg_0_51, arg_0_52, arg_0_53, arg_0_54, arg_0_55, arg_0_56, arg_0_57, arg_0_58, arg_0_59, arg_0_60, arg_0_61, arg_0_62, arg_0_63, arg_0_64, arg_0_65, arg_0_66, arg_0_67, arg_0_68, arg_0_69, arg_0_70, arg_0_71, arg_0_72, arg_0_73, arg_0_74, arg_0_75, arg_0_76, arg_0_77, arg_0_78, arg_0_79, arg_0_80, arg_0_81, arg_0_82, arg_0_83, arg_0_84, arg_0_85, arg_0_86, arg_0_87, arg_0_88, arg_0_89, arg_0_90, arg_0_91, arg_0_92, arg_0_93, arg_0_94, arg_0_95, arg_0_96, arg_0_97, arg_0_98, arg_0_99, arg_0_100, arg_0_101, arg_0_102, arg_0_103, arg_0_104, arg_0_105, arg_0_106, arg_0_107, arg_0_108, arg_0_109, arg_0_110, arg_0_111, arg_0_112, arg_0_113, arg_0_114, arg_0_115, arg_0_116, arg_0_117, arg_0_118, arg_0_119, arg_0_120, arg_0_121, arg_0_122, arg_0_123, arg_0_124, arg_0_125, arg_0_126, arg_0_127, arg_0_128, arg_0_129, arg_0_130, arg_0_131, arg_0_132, arg_0_133, arg_0_134, arg_0_135, arg_0_136, arg_0_137, arg_0_138, arg_0_139, arg_0_140, arg_0_141, arg_0_142, arg_0_143, arg_0_144, arg_0_145, arg_0_146, arg_0_147, arg_0_148, arg_0_149, arg_0_150, arg_0_151, arg_0_152, arg_0_153, arg_0_154, arg_0_155, arg_0_156, arg_0_157, arg_0_158, arg_0_159, arg_0_160, arg_0_161, arg_0_162, arg_0_163, arg_0_164, arg_0_165, arg_0_166, arg_0_167, arg_0_168, arg_0_169, arg_0_170, arg_0_171, arg_0_172, arg_0_173, arg_0_174, arg_0_175, arg_0_176, arg_0_177, arg_0_178, arg_0_179, arg_0_180, arg_0_181, arg_0_182, arg_0_183, arg_0_184, arg_0_185, arg_0_186, arg_0_187, arg_0_188, arg_0_189, arg_0_190, arg_0_191, arg_0_192, arg_0_193, arg_0_194, arg_0_195]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f13808272353acc79e98c3356a7685cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_201d6e620dc8e5feb728d0c6a2b005c8
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a4ca238d83adcd483ef8632011e1d3ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55f060b21c167e9669f2c30d2abaf2b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 9, 9], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5d86eb90d1a45deb5f87bd65be488d75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee005a50700d095e972b44ded04f19e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 68, 68], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5b74c1784b7b629c10b90949f925adcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e36cc2ee7a9d2a939cb3479f45d3d7a
    def get_inputs(self):
        return [
            paddle.uniform([165888, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([41472, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([10368, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2592, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([648, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_acaee8141221e6840c86f5b10440e76f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2880837030fd9e89aaee8956486156b
    def get_inputs(self):
        return [
            paddle.uniform([1, 165888, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 41472, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 10368, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2592, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 648, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d8404abf0d08d6541eae2e777a7be208(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b43068f6ce3c4b1ede264886e9f148b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 165888, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 41472, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 10368, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2592, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 648, 4], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_fea50b81eadfb6e65f3b9105739c8103(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5184, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1296, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[324, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f049bce3d7e2b4efb7906da70df6884c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fea50b81eadfb6e65f3b9105739c8103
    def get_inputs(self):
        return [
            paddle.uniform([5184, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1296, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([324, 2], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_04b05c3b0b860bd806bd03d0d3bc2b1a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5184, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1296, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[324, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b38b4b4a1442a97f0fd9080890e2a63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04b05c3b0b860bd806bd03d0d3bc2b1a
    def get_inputs(self):
        return [
            paddle.uniform([5184, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1296, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([324, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_ec69a66b5ca92d95b12b9cb32924e6eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = -1
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6804, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 6804, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e2c5b1f576c46252b6ad32739dad98c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec69a66b5ca92d95b12b9cb32924e6eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f7137ce5660714d1b9f9db4121734bca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee005a50700d095e972b44ded04f19e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 24, 24], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_625541baa97e8375efca3c0211ae9bdc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39, arg_0_40, arg_0_41, arg_0_42, arg_0_43, arg_0_44, arg_0_45, arg_0_46, arg_0_47, arg_0_48):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39, arg_0_40, arg_0_41, arg_0_42, arg_0_43, arg_0_44, arg_0_45, arg_0_46, arg_0_47, arg_0_48]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8787c08f4079aa99ccae30177bd65d33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_625541baa97e8375efca3c0211ae9bdc
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_75befd2da3390e711a076ff2c533fea8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15]
        input_1 = 0
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c919edc664ea8563b3b85c4d6809b56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75befd2da3390e711a076ff2c533fea8
    def get_inputs(self):
        return [
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e85a96a45dc452d2036afa112293ae57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31a8a5dba3bdb94943a131627fcfb21b
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 48, 48], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b719ca7d4ab4c1319d49bf63623989c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_433205b39c217d20dd7611bfb9b99056
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_fdb430ce91a428af8af82cf99309405f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 2
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 4, 25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4, 1, 25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f42c99a7ff3669781f08ef2b8d938919(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fdb430ce91a428af8af82cf99309405f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4, 1, 25, 38], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9bcf0d6d6da5da7d6401bd8900fe5b41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acdbd82ef2c0cf905efdaef83eaa2874
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 48, 48], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ccde8bdd77fd9b234e021ac33ee3f05b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_433205b39c217d20dd7611bfb9b99056
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b16318ea5eadf6ff268327356a001ba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee005a50700d095e972b44ded04f19e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_416e331e339d6f85b4c06f8dc879a207(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 2
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 4, 7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4, 1, 7, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3df121c639cf7886becd9dde76b77b36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_416e331e339d6f85b4c06f8dc879a207
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4, 1, 7, 10], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fcd58849a9ba714250b744be10e6975d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe7a892dc61053d0b98518a58ccb1b47
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_2322a5bee5442e2b25895f1647dda00a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        input_1 = 2
        return paddle._C_ops.concat(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 4, 100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4, 1, 100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d893f9eda4b3e4d3b011e5210bba06f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2322a5bee5442e2b25895f1647dda00a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4, 1, 100, 152], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fae6ce4a21fa9ba0350c4a879f016b47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2dd67f5f09297d85ceb86433100e780
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 36, 256, 256], dtype='float32', min=0, max=0.5),
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