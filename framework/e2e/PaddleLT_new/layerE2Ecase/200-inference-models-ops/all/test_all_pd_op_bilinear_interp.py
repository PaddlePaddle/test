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
class PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_60691597a43d67d5e147ad0832b02965(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([64, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1bc4c189e75d584ffa982380714077ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_db2f8542ccc41c1b9233f16712ac88ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([512, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_316665c7facf39dc90da0e23698c3734(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d9ff1460edf827e76b3e675cfda154a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1024, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b5edae41fa5789890046f31170e3a72a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 64, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_49839d4eff4229867c645bd446916b01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5edae41fa5789890046f31170e3a72a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 64, 256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f5789d977234efe7290abce8f3f89e00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 8, 6, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4e02491c8f4b28747a359454148c460c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5789d977234efe7290abce8f3f89e00
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 4, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d73144cfa0de36853d76314cabc3e262(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 16, 12, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_20eda6e410493335c24685f92c1c195f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d73144cfa0de36853d76314cabc3e262
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 8, 6], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e1cc0356eef3ab8662516fb176c0acf6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 24, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9b97433ac7b65faf7bc38efc6e480482(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1cc0356eef3ab8662516fb176c0acf6
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 16, 12], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a604317e9b10d6a13800bffd5c729d0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6df726b43df67c37681c8b9227ee9e8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1024, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6d172ade076f88096d751a88f195186e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b34c344589f7d07507b2b342aee5338(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d172ade076f88096d751a88f195186e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9a741a7573ccc152e7ac388033bff235(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d172ade076f88096d751a88f195186e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 2, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4fc2acd4b231bc0b3fb25d495093e0be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d172ade076f88096d751a88f195186e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 3, 3], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d9e460c6f02769eb5ba5e7beb5641dd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d172ade076f88096d751a88f195186e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 6, 6], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7dad7ccb4d9b23b90085233c0c593062(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 100, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2dd25015eeda298892d309ba5404bd1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7dad7ccb4d9b23b90085233c0c593062
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]], dtype='float16').reshape([1, 1, 2, 6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec8f4b9406f01cdf11f4a95eddf5e8f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 64, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0f468842549aaa0fb65c51488b772ac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 32, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea4a844b2f7a74bf2b00668b89cff724(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 32, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([64, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b42cc41c0262cb5955f3889bb2f756b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 16, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_42edccb841053e771f257fb17f49c408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 16, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([64, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_17f4354d6d818904b363327df3122eb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 16, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4ec981f71d04562ebfd28252aa568c10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 64, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8ed5908d0e597f7fff2bfdfd32fdd0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 32, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b1cb9bde8645c691e83d8b71e24b7f87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 16, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2748d80a2272929ab73b82c6d8cd7bbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([512, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c4f7c12740981fc44823fac8afc533b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8c94b84542b404a22cd746d10c07c851(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_760219b6c2c8945504cc33837798066d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 8, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c828ba152b0a3c5a6b2161def7ee19d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 4], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b206eeb107a3e5513a6087135d368e90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_410a156a98f25a31ff13d4f3db82cb8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 14, 25], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([112, 199], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c37c12e6c316821ac3299c964ef09e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 112, 199], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([224, 398], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0235946f5965b4a1224600de8cc26073(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_10633f2b10a28add3fad3360a7510967(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0235946f5965b4a1224600de8cc26073
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_84fc328790c3774bf7c0ffe8ee78aed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0235946f5965b4a1224600de8cc26073
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_24add067ed45803e74a093910d448416(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0235946f5965b4a1224600de8cc26073
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_61a7580cfb364e1c318c14c62cf7def6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0235946f5965b4a1224600de8cc26073
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 6, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e6dafadb05064ece1244b6610f503a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0235946f5965b4a1224600de8cc26073
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ba8d9e2b0841c03a5dd3be4f5ee260a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0235946f5965b4a1224600de8cc26073
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([256, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0ab16435765dc348673ba1c97bd063cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0235946f5965b4a1224600de8cc26073
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([256, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13dfcbd62efd54be8245fefbd1c77975(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1024, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_880420458e8f0925ee799b9d176629c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 192, 320, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1454067d86023a493801d2d44cf52deb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_880420458e8f0925ee799b9d176629c5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_dce3a74c154ec29d22075ed89a7151f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c84186d18b97086352d6374bca7a7196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dce3a74c154ec29d22075ed89a7151f8
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 48, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46eeb60c7157a6578a085fdde56e39d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dce3a74c154ec29d22075ed89a7151f8
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 96, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2ca05cd521f1747e5679a5dffeb08c8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 180, 320, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1660bf94874f19e5faf782e0a2027a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ca05cd521f1747e5679a5dffeb08c8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 192, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_56a696964211f1dbbae976065755d726(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e8d9770ebb82a7ba27bd86eb0b58fb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56a696964211f1dbbae976065755d726
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2211bbb82ea56bb3f1354e5e7f21451a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_62b884c8a406c2c9b9c5acd3b7b899f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 2, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6c961d56349241af920f2820ff4b8dbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 4], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2a7c2adc038ba94f23af69f6dad69730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_74711c0c553ba6d4f239d7eb6f8c1b0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([64, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_079e5e9aa8899c95e966bb216fc09efa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1534a33689c552e26dc757a5a5e9ff2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1024, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f2ca98214e6f9b078e734963390e1cc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d172ade076f88096d751a88f195186e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1d247f3334c9896f630fc63eb2feb16b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d172ade076f88096d751a88f195186e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 2, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_98dee9a0aad3a16128120b7593ce3c62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d172ade076f88096d751a88f195186e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f0ed3d827ab05670a9146cb045251fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d172ade076f88096d751a88f195186e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 6, 6], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8c628ed9edd9a377f25252ee8cb8b102(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d172ade076f88096d751a88f195186e
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_71cf9a2ab789b11bc90fc10a169c2685(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d172ade076f88096d751a88f195186e
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([256, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d9d6fdf4c408821e898036ce910ca132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d172ade076f88096d751a88f195186e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([256, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_293263eafc982827bde19b407d4bf23f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 256, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1024, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f721e9db2ccb4790abbf09f843ec2a28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c3e3b737345d4c0008147463a3e745c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d575b65871139e3926d6c6cf6ba22e35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3ad4ea411ccfb2e980db3a8c937bc7e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1decefe03104d1f31aeca441c515a811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_738fc21d87e78c2578705378847af8bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_44383b7eb600d5886268facf2b063162(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_24244376ee1a5297f9429c1fbc089609(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ede605f0f6fa95bc54c8f440264b3b6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e18c7e0db624832c4f3d704b5a8910f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([512, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_75ee6fc0ac8f54d30e87af28b75098c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f1db4f3a2610a3325ae6f61889426c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec5d7ecf3941d6e5df681d174a09a137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3a5b6fc8f76b059f6bb5e1b702dc6118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c96ce5f9ae1142b411f3d40d8954ce36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f7d5fe75ee572739c2ad8baa95c18aa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea917296cd118db4087e1798914376a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1024, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aa65845b3d60de76969811429d14a711(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b07a296d7fc3a01260656eeb905078d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ade55ad62045f3b97792ae14fafbcaaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([512, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c8cc6c9b93d3b0b5d08a31956619f441(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d172ade076f88096d751a88f195186e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_920c48564231665ba0fe2a57405d6b61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d172ade076f88096d751a88f195186e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_71666d6dd34cf0c9a3268e1c2748e865(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_19e79e96734c83568a98bd6c3f28f44d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71666d6dd34cf0c9a3268e1c2748e865
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2a9617909d3c5fcb859e577a9767c194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71666d6dd34cf0c9a3268e1c2748e865
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 512], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e0b53ebe25e58c7614d9b39444371dcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512, 1024], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0d1053409e6c755a91c5736a801aea9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_da6d8b909bae2b075b2b451a54bb3f20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1c38fb8361c25ef886f76cf89121c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1024, 2048], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_157ea28316a03a7de9e0385350fc9b00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 192, 320, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dfa09ac2a2eec1782effbc08284fb111(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_157ea28316a03a7de9e0385350fc9b00
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 180, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d71791b8574b3ece096725031ee416d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8cc4b0e206de335a2e6e0bfbf3f5a683(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d71791b8574b3ece096725031ee416d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 48, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b62a5d21e9c4d444cdae54558cfce87f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d71791b8574b3ece096725031ee416d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 96, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fc67b0940969029d5b1b939acf35c52e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 180, 320, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_705f6694b8202c0728b9afaead71f572(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc67b0940969029d5b1b939acf35c52e
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 192, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bbf1cde46fe1b3e37979c812068af0df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ff003d32314851e01b71356309cc7e06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbf1cde46fe1b3e37979c812068af0df
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 180, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dee87ace87f69ab2c1ad1f84a564dc32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0235946f5965b4a1224600de8cc26073
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f23447958091fd8a012c666bae386abf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0235946f5965b4a1224600de8cc26073
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e4b5a9f218ae362f17b5b551003e506(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0235946f5965b4a1224600de8cc26073
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_feaaac6cf1524d249bc60583dc1375c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0235946f5965b4a1224600de8cc26073
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 6, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c33965c8b26499b68d638b7c8f492bfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eff7d560783db8c86408d4d17ecaa33e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_05e0894d76f303aa278547b35cd356f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5585d2176c1bca634cb93b1c4abc582d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f2b44a89aa46c9858c3fe7536c8860d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f0e12afb3095bff7420400964accf472(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 8, 6, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8e17226663fb734f7860849644b2dd32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0e12afb3095bff7420400964accf472
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 4, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_deea7be378e111959ff5115aba0ceb43(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 16, 12, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47819d739ca25a9b91516b5c77dc0c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_deea7be378e111959ff5115aba0ceb43
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 8, 6], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_377a7709b69021f3fa849a54c29eca14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 24, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4fa8d49cad4a9e6d3e44377b8eb780ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_377a7709b69021f3fa849a54c29eca14
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 16, 12], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5b64cb65fd5a45b210ac36f18e3953fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 100, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c6e2bb93a444c153c418cbcfc602c83a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b64cb65fd5a45b210ac36f18e3953fb
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]], dtype='float32').reshape([1, 1, 2, 6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_359fe2d9fc6d61698c791113442cb3ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 64, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bb358c3830528c1e632c57ffd328d377(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_359fe2d9fc6d61698c791113442cb3ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 64, 256], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc28c7a609be32d7de86e540a986aa77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0235946f5965b4a1224600de8cc26073
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_76bd547c78caf8def1f0a2fcaabd530a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0235946f5965b4a1224600de8cc26073
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8fffb8d535e6a7b995a99318dff158e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5ccd5a0a9693c28602003bee1d2b87b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fffb8d535e6a7b995a99318dff158e6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3496dc5b8cee8dc723ff9c8a853b6418(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fffb8d535e6a7b995a99318dff158e6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f60a6221eaba38103919f7b6091c42a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_568663e9ec8cc3a659f64370b2ed36be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f28ec3644a19ce3e812cf90b5d28760b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1024, 2048], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b0389c601d00670ec9997501f1dd4fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 14, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([112, 199], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d841153d64c748f7d2eaa00fe1ab2386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcb80c1466e8d4dedfc987dda5d54ee5
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 112, 199], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([224, 398], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f3bcea0dd8d787d75a20a0eab81400ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77333c59e97caf1f1b28c86f8d7f19d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bcea0dd8d787d75a20a0eab81400ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([64, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_72a241f5d850c5c57e7b53f17707f381(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a88f888637e2c8e0f9cefe4b69ee4b9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72a241f5d850c5c57e7b53f17707f381
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7f577e766aa22a3928af1279f8d258df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d314f9c942c3fa7e0b8556d800ed7df6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f577e766aa22a3928af1279f8d258df
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([512, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e50b6a2ffebb57957d6aa1f2c0d23355(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_64b16134d32c0db2e9a7ba9950a6b45a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e50b6a2ffebb57957d6aa1f2c0d23355
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fd2be78cc5cf79f7cb811e8928a590b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 19, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e71b7053b7211b17a37e3e411ca3c2d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2be78cc5cf79f7cb811e8928a590b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1024, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c4d6b5feb3b141df5ecac2262f55007c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 64, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 64, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7b20d2b21e0f2145fd43e841aba097a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4d6b5feb3b141df5ecac2262f55007c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 64, 256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_346dc76e7022c40c5a7f5338c4e89f6e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 8, 6, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 160, 4, 3], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b31be8aa0d84c9db20cbd1018ecbab04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_346dc76e7022c40c5a7f5338c4e89f6e
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 4, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bd9d3244333de7ef0e3920653ed6007b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 16, 12, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, 8, 6], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_57052eacd590b8725b365ea02afaec8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd9d3244333de7ef0e3920653ed6007b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 8, 6], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_48abf86608ab8fd99acb8bddb434c191(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 24, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, 16, 12], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_daceab60b5da579f339e8162c88fc098(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48abf86608ab8fd99acb8bddb434c191
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 16, 12], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_10b943cec2765dcc68ed16ed575937c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_be8cac51c2eb461e58fd8334c4409192(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10b943cec2765dcc68ed16ed575937c7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9972508d73c12cc18f15767d2cfeea7f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 19, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_adfb2fca42cd0b22057c5931c0a6431b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9972508d73c12cc18f15767d2cfeea7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1024, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_18c16f761431c93c91d21a14b1b1f995(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_64880573da54f993d0c0f2209d93b1d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18c16f761431c93c91d21a14b1b1f995
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6fecb8e421605d236ea7eac75b642a8a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 2, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d67607b79b9afb252dae20b5fbc58278(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fecb8e421605d236ea7eac75b642a8a
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 2, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a7d70f13b53b4a43912668875341fb8c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 3, 3], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f089422d5e7d48c6c3e36163b6ab52ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d70f13b53b4a43912668875341fb8c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 3, 3], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f15604e8a9bc318c6f2ed63a75125cae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 6, 6], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bcc535c939a0fd3c0d915cc8e7b26735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f15604e8a9bc318c6f2ed63a75125cae
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 6, 6], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_abaa0b20cb269aaeda7e155bfd3a177f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 100, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 2, 6], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5867163b8e49e61b9eb9f02a1e986fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abaa0b20cb269aaeda7e155bfd3a177f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]], dtype='float16').reshape([1, 1, 2, 6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a8b881c6d1cb701f9f77804cfb528525(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_309f239883e57f387580388cbbe713d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8b881c6d1cb701f9f77804cfb528525
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 64, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_be4f614c143a12c3eb010ba0a9fe67e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8b881c6d1cb701f9f77804cfb528525
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 32, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1f7098cc874504fc6e1bea14f4466095(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 36, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b155d06127b8ff20eebea90ef6542ef2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f7098cc874504fc6e1bea14f4466095
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 32, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([64, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aa3ac0ee3a09a48a6688c2bbbdbc2f6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8b881c6d1cb701f9f77804cfb528525
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 16, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11b3b695447f009695b89580558d9a70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f7098cc874504fc6e1bea14f4466095
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 16, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([64, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cc30f1da0972bf3d02368364b6808263(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5be574c8cba6f13f28318cb168064f55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc30f1da0972bf3d02368364b6808263
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 16, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cad069444683a9fae5d4f3aa41b5c399(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f7098cc874504fc6e1bea14f4466095
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 64, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9aeac27702bbf7f002699e2efc8000f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc30f1da0972bf3d02368364b6808263
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 32, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1681b1672b37c3fcb9a0c475acc17cae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_48feadc2ff315484ef41f2f151f5427e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1681b1672b37c3fcb9a0c475acc17cae
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 16, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_690b3504d3a5764d54358d1ea410c6be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2be78cc5cf79f7cb811e8928a590b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([512, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_690bd7afde7786752f4e5514bbbae30b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 32, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_94183d15fa7cc75cce6e41593ebf2ab0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_690bd7afde7786752f4e5514bbbae30b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_85729e402485b797d4c9cf1d885fec55(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 16, 16], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_470eb36833ba26045caf8efe7c21fd78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85729e402485b797d4c9cf1d885fec55
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6891b1ef3bb1b550d46f1c1026cb1f7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 8, 8], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe40cc3c265ffa3549446348f0a6c420(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6891b1ef3bb1b550d46f1c1026cb1f7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d4637b38b2a54af4f407dc9db126e540(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 4, 4], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f39289b0e042a9d70a42647626e19a8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4637b38b2a54af4f407dc9db126e540
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 8, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6b2615acb896d08afa31c7f2ee9b1246(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 2, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f3b7f612ca916a4f78b9d4c896ea03c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b2615acb896d08afa31c7f2ee9b1246
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 4], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_adf85351ce3f2c1a27306c095ed91f89(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_95bb31548043290b67af3be1a7542937(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adf85351ce3f2c1a27306c095ed91f89
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e2724d6b7c23c06206bfaad8778f3735(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 19, 64, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_17f34ee31e4f23d221af688066f85861(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2724d6b7c23c06206bfaad8778f3735
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1024, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8b9f7ae840560ad6d7eb7dfb6ef5b2cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5231e597c77eaecdba6e6fab4740913c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b9f7ae840560ad6d7eb7dfb6ef5b2cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 14, 25], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([112, 199], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47123e9cc17a7a58e07fa700ab407cb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f577e766aa22a3928af1279f8d258df
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 112, 199], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([224, 398], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b4ec18a30d29de93fc5b2f0ac7e5f2cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8560f061becc00bc1d7247c5b08c303a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ec18a30d29de93fc5b2f0ac7e5f2cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e5d991659672f97c336d3bc8f0c28c24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 2, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2012e56b8a0c0741437ee0ba9c5ce04a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d991659672f97c336d3bc8f0c28c24
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c827a63aa32aac210361c643d6ece8c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_477641788c07b231f73f95e3773a9f92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c827a63aa32aac210361c643d6ece8c6
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_dc3fd2d73cc46b53866df6594cf9d938(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 6, 6], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6888aa9a517938359d86f4b97c6c1808(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc3fd2d73cc46b53866df6594cf9d938
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 6, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_103aab0d7f8fa2b06c582a262e4ec78a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a4714d4f5e938f81de1f65708ebea3b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_103aab0d7f8fa2b06c582a262e4ec78a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c83a5da5516bc04f92c4084739ebd7bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_103aab0d7f8fa2b06c582a262e4ec78a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([256, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_04532b358f4e11908b06cf6d0226ad5a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b6fffec27fe319d7b1cabf82da27c45b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04532b358f4e11908b06cf6d0226ad5a
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([256, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c36199e395c2b652e4ac526763875c18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9972508d73c12cc18f15767d2cfeea7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1024, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a44f7cb9a01555e745286543a7a785bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 192, 320, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 180, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5f569539692efe23203882ac5132ef82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a44f7cb9a01555e745286543a7a785bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_38e2fcafc5bdd980968d0f6ffaaed50c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 48, 80], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f15cc669da85cfac1e7c62318ab4ea5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38e2fcafc5bdd980968d0f6ffaaed50c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 48, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_faf7f3f92b62101b88f06763fb5b0fa1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 96, 160], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_54c1b283949c99e0af6dad39c2da72c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_faf7f3f92b62101b88f06763fb5b0fa1
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 96, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6ff1d290e44a5e5653eb9f7a3e0a9908(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 180, 320, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 192, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8fa449684eaabdf52cf54e97303872da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ff1d290e44a5e5653eb9f7a3e0a9908
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 192, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a24272f1a6b069055e624c83f3b4370f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 180, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_420f64c523d1130b2484a06eabe9b104(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a24272f1a6b069055e624c83f3b4370f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f2a960f447f65667dbffe14036735a1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f229aca1ee0f8dcaa46d92bde04c14da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f2a960f447f65667dbffe14036735a1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_24d950fd5fa3ff005ddf100429c869e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 2, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_035d917c0841740a943b9310a3325328(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24d950fd5fa3ff005ddf100429c869e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 2, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f78623f183772e03a283b0d1ad8be53d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 4, 4], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d65505adb69151f1d87df5f62da9d29e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f78623f183772e03a283b0d1ad8be53d
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 4], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_19685d98167272d5296c7bd0ea669066(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e50b6a2ffebb57957d6aa1f2c0d23355
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_caa481b1a208bbafe813f06c13a00dd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e50b6a2ffebb57957d6aa1f2c0d23355
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([64, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2637658b1fb793988dfe00f653b85448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b9f7ae840560ad6d7eb7dfb6ef5b2cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9a674aa1eede8ec4e1838495b71f4011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2be78cc5cf79f7cb811e8928a590b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1024, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7e2b8252af80e5bcbd6488a23a338465(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1869d870e912548ff6c7a096916a211(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e2b8252af80e5bcbd6488a23a338465
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_44a9675cbed705257864b8ee7196f396(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 2, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14d4ad411a5ddf4776f8c8ab7d396504(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44a9675cbed705257864b8ee7196f396
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 2, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_12e8b9744515b360e3d729f7f2c0cf37(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 3, 3], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e54aebc3c405d11e11371eacb2571a77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12e8b9744515b360e3d729f7f2c0cf37
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e279c5a2b22192c8da7620ceddf6fb06(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 6, 6], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_452f14f3333ad93abb00c71d438b3acc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e279c5a2b22192c8da7620ceddf6fb06
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 6, 6], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ed791e5c41d62aee7c5ea89d7143adfc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_85867eb4a0ef58cf3b0b6e98ba97e39a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed791e5c41d62aee7c5ea89d7143adfc
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c27473432c9cc480e5982f7fd4ffccba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed791e5c41d62aee7c5ea89d7143adfc
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([256, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a07c6ae12f8c0f1485902f8d5351d0c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0ba657760448173a715d27e7d2846714(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a07c6ae12f8c0f1485902f8d5351d0c9
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([256, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a021821fe075ec1a765b8f76741009c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2be78cc5cf79f7cb811e8928a590b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 256, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1024, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ab7b52f3874b5e9f057e6e7975eaab80(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1457d5a2f0a826d869b71ea5b5286fc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab7b52f3874b5e9f057e6e7975eaab80
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a105ac49c4f1886ced53036ac012f474(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab7b52f3874b5e9f057e6e7975eaab80
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_aec10b506dddb652b448d1b7aaef4461(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 36, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dcef461895b1ee856e77396b1bb05b95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec10b506dddb652b448d1b7aaef4461
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_03f10bd99d316d37575d1c7d9d891378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab7b52f3874b5e9f057e6e7975eaab80
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_abf7afffeb1ce758276d294f09912654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec10b506dddb652b448d1b7aaef4461
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e557e1da37fa1a5b4df5571957875c1e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_729ed8188eaf78c1c916328e4e5080fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e557e1da37fa1a5b4df5571957875c1e
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c33e7b41a2a8a8984ffac9edab2be58b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec10b506dddb652b448d1b7aaef4461
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_27da96f84aebe373d42edd1c04c02c30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e557e1da37fa1a5b4df5571957875c1e
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2993c9350abb0f78d7afb9e1df4fc9b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8330b7a9a251fbb47938147742faa1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2993c9350abb0f78d7afb9e1df4fc9b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_24753eb4d13b67e08bf355a498b65d8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9972508d73c12cc18f15767d2cfeea7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([512, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_46d5003cfbf99b985f0fc28c46640562(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1cf62a7bc8b2ad29a76b781184c11a1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46d5003cfbf99b985f0fc28c46640562
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_22d315c1bea653104939c5391c43d5f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 2, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c10447a4d97b448415f722f2bf37abc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22d315c1bea653104939c5391c43d5f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_77965f7a60cd62c1d78aeaae0978367f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 4, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f59b5751fc3d01ca390500bcd38246c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77965f7a60cd62c1d78aeaae0978367f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_016f3cd6eedac4e0a0f2f3cfa587a104(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10b943cec2765dcc68ed16ed575937c7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8109565d731595eb51f986f53375a246(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10b943cec2765dcc68ed16ed575937c7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b16d34707eadd53e800e062e4633b8a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_61ad3a670d4cff840fed2a379ab410e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b16d34707eadd53e800e062e4633b8a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7af4526f1753461e7ddf121f93dc863f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9972508d73c12cc18f15767d2cfeea7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1024, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8ca7db181501ef8d115c8a7e39d98fb9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_62c2bb5e06bbb5af6836f1590d3dc72e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ca7db181501ef8d115c8a7e39d98fb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a7ea03c58793727267760e81e16f2d39(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ad1d3b20883cf25221ce7f75034311a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7ea03c58793727267760e81e16f2d39
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ea7ce21b320a835f3dab4646922a8c40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b0029e6e966183d83046172e33243d28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea7ce21b320a835f3dab4646922a8c40
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([512, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_71500aa7c41fea3c5a8ff1f4ae0f7206(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_61ab8bd62003ab9638d79016a6bdf413(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71500aa7c41fea3c5a8ff1f4ae0f7206
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b516697e96e7872f2573a944e6d0ddb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71500aa7c41fea3c5a8ff1f4ae0f7206
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a63e9735040763b14b9d27ada4d392ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8475248081b43ac8533f1b2a7ae25fca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a63e9735040763b14b9d27ada4d392ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c03a12a0674e3c88c4dc8261c75c795(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a63e9735040763b14b9d27ada4d392ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 512], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d3f93b90dfb838a81c42c8f314e51fc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72a241f5d850c5c57e7b53f17707f381
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512, 1024], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4e931077a38af02cc9b7e25b362c61a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72a241f5d850c5c57e7b53f17707f381
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4672bb0adc148cd98a744d8d9116759a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2be78cc5cf79f7cb811e8928a590b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1024, 2048], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7445f5c34d673cbc8710e6f763da1b41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 192, 320, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 180, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3b896b0ec63df107c31d3f953993ebf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7445f5c34d673cbc8710e6f763da1b41
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 180, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_032b18a2febf9b76a94d03ebed594a2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 48, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a4b55bafcfd9f1aca0756d1ecc3ad86f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_032b18a2febf9b76a94d03ebed594a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 48, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7bc9145b7dcec06556f4089112db9d96(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 96, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_99686597650ce4a7472d9e2f8f9df4bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bc9145b7dcec06556f4089112db9d96
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 96, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d73acdaf2ea9a62b3f6ebeca7b07c36e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 180, 320, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 192, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dbd47e6477fe6c90ab74ad8a6194b4e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d73acdaf2ea9a62b3f6ebeca7b07c36e
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 192, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c194639392c95be3172d2321f7b5187e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 180, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_61853436382819bf2ef387752ec627d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c194639392c95be3172d2321f7b5187e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 180, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_aa08ad6fbab4b46341e734134ff9e9ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4a1915db416f46aa5dab662b1d3b4a5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa08ad6fbab4b46341e734134ff9e9ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f83e980ff18741e7364737e92160e7ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 2, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90131b370891049fdb96dee38ae7c1f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f83e980ff18741e7364737e92160e7ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ad6bfecb080f7b7adca8bb43d9273a78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0c5524c426ff8014de3c40ec74f76bb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad6bfecb080f7b7adca8bb43d9273a78
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_46620dbdfd3d671ce4c39129b388a943(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 6, 6], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_04cd2d0c53b35bfb6627d85b0bedfbc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46620dbdfd3d671ce4c39129b388a943
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 6, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bbc3c83fb19745b1787bbe673d40867a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8c8ff767ca7cfe42993b07c945783648(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbc3c83fb19745b1787bbe673d40867a
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_83bcd87dfdc798d9cfdf7a4a4b0c71dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a77f6950f9653614f716ec411f20eed3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83bcd87dfdc798d9cfdf7a4a4b0c71dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e1aab62c0cb80293e82b6ec0189cc51f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 8, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fa31f5e069fec09d0ed23ac7261cf846(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1aab62c0cb80293e82b6ec0189cc51f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f9b59f99229b7fbb61a251358c6861f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 4, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_27dabe9c4b5ce8bc072ee851fb4358ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9b59f99229b7fbb61a251358c6861f2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0f99405b1808c2117733220c15266cef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 2, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a7c6e026104c5fbc958c83229122b44d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f99405b1808c2117733220c15266cef
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4fc12d02f8f1cbc55ab21848bcc24b11(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a6dadbc3591264e70b947d7cd380e106(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fc12d02f8f1cbc55ab21848bcc24b11
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5d232512d33e728049b334da1a2180fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 19, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_edefbb66a808a713cb68e02092fce512(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d232512d33e728049b334da1a2180fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1024, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a1bb88e839fb4930a7a583e87e5c2a6c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 8, 6, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 160, 4, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_266d41b882a408893a40e631dd980ff1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1bb88e839fb4930a7a583e87e5c2a6c
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 4, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f535319f8482cae4171bd38d0011be86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 16, 12, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, 8, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_615b988cb61710adf527769b0502a284(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f535319f8482cae4171bd38d0011be86
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 8, 6], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a4b39f595dadab95bb4692aabfe6edab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 24, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, 16, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6271e21ae827b2ec7fdf7f546b616685(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4b39f595dadab95bb4692aabfe6edab
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 16, 12], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e9d92523ba055a25067d83f3dc47ad20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 100, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 2, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_84bdc2b60fe822c8a6db7c2abe25770c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9d92523ba055a25067d83f3dc47ad20
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]], dtype='float32').reshape([1, 1, 2, 6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5801d24d88bca06d773c68c56d9b4f6b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 64, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 64, 256], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_20cd196fdcf766c83e1bd7745f6147cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5801d24d88bca06d773c68c56d9b4f6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 64, 256], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_309e51e563d9355212680ce39efa096b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_946adf7a3dcd560160503307e09fad2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_309e51e563d9355212680ce39efa096b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3e82c726ae90e1d1704702ff0651e29b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_309e51e563d9355212680ce39efa096b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_940eaf298719afab2c5bda303f7f220b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13c99d3ebc90c7cdbceb0a695ecc0dd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_940eaf298719afab2c5bda303f7f220b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a2194e69f11cdb63ee35c40b41fab126(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_940eaf298719afab2c5bda303f7f220b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c2c5ad1e004435b5b3c2721a8011232c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7ea03c58793727267760e81e16f2d39
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c83d746b97d0db60cfc207d07ad79d25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7ea03c58793727267760e81e16f2d39
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4894e1f9c1169fc472043afb0c58351b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9972508d73c12cc18f15767d2cfeea7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1024, 2048], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b2a4e608ee572ddbffae65162413bbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b16d34707eadd53e800e062e4633b8a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 14, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([112, 199], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b1cbb4fa1a72b62425471af930ba23d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea7ce21b320a835f3dab4646922a8c40
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 112, 199], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([224, 398], dtype='int32').reshape([2]),
        ]


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