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
class PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return paddle._C_ops.stack(input_0, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_02d83dae1e917926e55e00a5fd6566d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(24, dtype='int64').reshape([]),
            paddle.to_tensor(36, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c5a0036e707a47e1660af4321f83d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(11, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e81043c99e61d070271f67db6c343000(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(43, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_639da68aedf63f6f9f029baf9d8aee48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_862164ee7d140b1821f557d96ea787f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(3549, dtype='int64').reshape([]),
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(19, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_eacf2a314f8698a8181739e27e670aff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2fb9752afe97b536d37749a92bb4aa5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
    def get_inputs(self):
        return [
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_984dbe1a4816f690fb38e43fe53d9901(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return paddle._C_ops.stack(input_0, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7401fc2c29433fa0d70d61579efd3c19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_984dbe1a4816f690fb38e43fe53d9901
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1024, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_752418967961f2b95dcb50a5a2613292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
    def get_inputs(self):
        return [
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6ff6c36cdc41273fe96d2dc03ce654f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_984dbe1a4816f690fb38e43fe53d9901
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(4096, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7364274f27152ff34074317b2462e726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
    def get_inputs(self):
        return [
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c0d521dbbd12677824e5aebb3d19eb0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_984dbe1a4816f690fb38e43fe53d9901
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(16384, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_162f5cd7d1ac6344ecff665b334ba41b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
        return paddle._C_ops.stack(input_0, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8bfce412cd80d6209973c1eb2d4ad43d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_162f5cd7d1ac6344ecff665b334ba41b
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(2, dtype='int64').reshape([]),
            paddle.to_tensor(20, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor(256, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e3c962bdfa7219d812e8e3506edf9088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(40, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor(256, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32b1b445649738a998a9a3d1a7065d4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_162f5cd7d1ac6344ecff665b334ba41b
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(2, dtype='int64').reshape([]),
            paddle.to_tensor(40, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e2d32b8f142ff2d3f493fdc090bafa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(80, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_702ceb294347ec06392cde7fee656040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_984dbe1a4816f690fb38e43fe53d9901
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(-1, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b6d07cfcf37e6e3e9168ce485559521c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor(16, dtype='int64').reshape([]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f16e92dd15f5961348710a4f7d058ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
    def get_inputs(self):
        return [
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f49a33cdc4971b9bf54b1d4b8bc93e08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
    def get_inputs(self):
        return [
            paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0522874b13a5325c9dad4751d571b1d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
    def get_inputs(self):
        return [
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4d9d1fc9430ce0f4f90f2d727f4a8028(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(43, dtype='int64').reshape([]),
            paddle.to_tensor(7, dtype='int64').reshape([]),
            paddle.to_tensor(7, dtype='int64').reshape([]),
            paddle.to_tensor(768, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7c53d63065f058bb2922e2b7ad86bb5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(3, dtype='int64').reshape([]),
            paddle.to_tensor(512, dtype='int64').reshape([]),
            paddle.to_tensor(1024, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_67d72cdcf2abe3bc347c80bb1c523fc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_984dbe1a4816f690fb38e43fe53d9901
    def get_inputs(self):
        return [
            paddle.to_tensor(2, dtype='int64').reshape([]),
            paddle.to_tensor(256, dtype='int64').reshape([]),
            paddle.to_tensor(21, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_428275c1d6a992946c7610ec57f7170e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5]
        return paddle._C_ops.stack(input_0, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ce60eae93872ee22f9a0efc7a05a96d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428275c1d6a992946c7610ec57f7170e
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor(13, dtype='int64').reshape([]),
            paddle.to_tensor(13, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6bce68ab7110e3c0ff63909adb97dc27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(7581, dtype='int64').reshape([]),
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(17, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_332fba60d16119056bad139380f99385(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428275c1d6a992946c7610ec57f7170e
    def get_inputs(self):
        return [
            paddle.to_tensor(22, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(24, dtype='int64').reshape([]),
            paddle.to_tensor(48, dtype='int64').reshape([]),
            paddle.to_tensor(2, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c8ed67ed53882fb43d3613406697ba71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(22, dtype='int64').reshape([]),
            paddle.to_tensor(48, dtype='int64').reshape([]),
            paddle.to_tensor(48, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ba0c9d2524eeed5c6546b4f6776e0eb2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return paddle._C_ops.stack(input_0, -1)

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
class TestPrimitiveOp_b28d154968ace8a12f067d877296e13f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba0c9d2524eeed5c6546b4f6776e0eb2
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e695958cf248446c52eaaddcd0e1fed6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(25, dtype='int64').reshape([]),
            paddle.to_tensor(38, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9ec86e93d3546559c33f5ab6d9e5a83a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428275c1d6a992946c7610ec57f7170e
    def get_inputs(self):
        return [
            paddle.to_tensor(6, dtype='int64').reshape([]),
            paddle.to_tensor(2, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(12, dtype='int64').reshape([]),
            paddle.to_tensor(24, dtype='int64').reshape([]),
            paddle.to_tensor(192, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f39bc2c06539729e99a22a58cf81b2cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(6, dtype='int64').reshape([]),
            paddle.to_tensor(24, dtype='int64').reshape([]),
            paddle.to_tensor(24, dtype='int64').reshape([]),
            paddle.to_tensor(192, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dca92d663bdb3fda47dc37d1514be04b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor(16, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3dc15dd9c48480d82ff0bc58ad93196f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(4725, dtype='int64').reshape([]),
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(17, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e2250861d076c69d1294ef7280e490f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor(16, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_09e4846cd1e29e7c1bbba5e5d4e9fc06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_162f5cd7d1ac6344ecff665b334ba41b
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(577, dtype='int64').reshape([]),
            paddle.to_tensor(3, dtype='int64').reshape([]),
            paddle.to_tensor(12, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7b96360ebeeb9665e845a0faaee59161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_984dbe1a4816f690fb38e43fe53d9901
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(577, dtype='int64').reshape([]),
            paddle.to_tensor(768, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_67726fffc685413d66b527f4eb3b5d4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(43, dtype='int64').reshape([]),
            paddle.to_tensor(14, dtype='int64').reshape([]),
            paddle.to_tensor(14, dtype='int64').reshape([]),
            paddle.to_tensor(384, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f8bcc3f7a754c1283edbf54176cff48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(16, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_57e9d64e0a4bd5e17ab5f2d718ccbae4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428275c1d6a992946c7610ec57f7170e
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(48, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd6be47e0d54092f5d5ab7ff3d66e97c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
            paddle.to_tensor(48, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5963f3b85f2c3ab84ea0fb2815ac11e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(43, dtype='int64').reshape([]),
            paddle.to_tensor(28, dtype='int64').reshape([]),
            paddle.to_tensor(28, dtype='int64').reshape([]),
            paddle.to_tensor(192, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_60e2e7d8d764feb629d1c7c71a741875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(256, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ca59e3cae5b42185a3674366397ccbd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8efe566b125d2d7034fbe76b3b2b71df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(8400, dtype='int64').reshape([]),
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(17, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_594e3b0595d43ec3646060c107d955cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(20, dtype='int64').reshape([]),
            paddle.to_tensor(30, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a92fb808c75a301c120b9060dc48887f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(43, dtype='int64').reshape([]),
            paddle.to_tensor(56, dtype='int64').reshape([]),
            paddle.to_tensor(56, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_569849562a539f552929a758a89f8c0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
    def get_inputs(self):
        return [
            paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ffca199f04361becdb305269f1fdbbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7b214274d8db1e9d3c13df256076d942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(3549, dtype='int64').reshape([]),
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(17, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a390af980acd11ba4c580cdbaf86526d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_43309eb98c070c3b8757784d8d9ee74a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(512, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a30e543238e5ab628166697233d08e12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428275c1d6a992946c7610ec57f7170e
    def get_inputs(self):
        return [
            paddle.to_tensor(10, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(2, dtype='int64').reshape([]),
            paddle.to_tensor(24, dtype='int64').reshape([]),
            paddle.to_tensor(12, dtype='int64').reshape([]),
            paddle.to_tensor(192, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f12f9a7342dba3031232bdf727d9ec9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(10, dtype='int64').reshape([]),
            paddle.to_tensor(24, dtype='int64').reshape([]),
            paddle.to_tensor(24, dtype='int64').reshape([]),
            paddle.to_tensor(192, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3096c074b763153d7cf078196017f8c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0a2f9e37df3c5c3c5392c522a87dcdd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3096c074b763153d7cf078196017f8c2
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(2, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_19658f6df841a4272dfa975d6e1e3129(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3096c074b763153d7cf078196017f8c2
    def get_inputs(self):
        return [
            paddle.to_tensor(98, dtype='int64').reshape([]),
            paddle.to_tensor(99, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_27a3f2877458b6d1bc1f65881c4ecc98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(192, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f3f6718c102448c10709d85cd6c2f115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(11, dtype='int64').reshape([]),
            paddle.to_tensor(7, dtype='int64').reshape([]),
            paddle.to_tensor(7, dtype='int64').reshape([]),
            paddle.to_tensor(768, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_af55f5f5623fa793ebd965fcb72c6355(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1280, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f507d8887a88f78b4ac2639c6169e58d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428275c1d6a992946c7610ec57f7170e
    def get_inputs(self):
        return [
            paddle.to_tensor(6, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
            paddle.to_tensor(48, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_da2a69f7914bcafacda9ed5299628e99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(6, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
            paddle.to_tensor(48, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_581f196adad6b125da93ad297c8ed421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_984dbe1a4816f690fb38e43fe53d9901
    def get_inputs(self):
        return [
            paddle.to_tensor(7, dtype='int64').reshape([]),
            paddle.to_tensor(256, dtype='int64').reshape([]),
            paddle.to_tensor(19, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b25dfe7dfd76bd217ffa1175dcbc0e65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428275c1d6a992946c7610ec57f7170e
    def get_inputs(self):
        return [
            paddle.to_tensor(22, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(12, dtype='int64').reshape([]),
            paddle.to_tensor(12, dtype='int64').reshape([]),
            paddle.to_tensor(768, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d5fa25e964fb6dacccd0275e91b21a26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(22, dtype='int64').reshape([]),
            paddle.to_tensor(12, dtype='int64').reshape([]),
            paddle.to_tensor(12, dtype='int64').reshape([]),
            paddle.to_tensor(768, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7573320e1578ee160b3aa6424e8a8ab0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428275c1d6a992946c7610ec57f7170e
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(24, dtype='int64').reshape([]),
            paddle.to_tensor(48, dtype='int64').reshape([]),
            paddle.to_tensor(2, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_81b05196e2dfeb6b306bbdfaee05f853(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(48, dtype='int64').reshape([]),
            paddle.to_tensor(48, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5cb9593eaff70dc135e7c1a8f8e39bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(512, dtype='int64').reshape([]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_537ec57b9097b35d7c1928a9375332d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_162f5cd7d1ac6344ecff665b334ba41b
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(2, dtype='int64').reshape([]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8091fe99b042906138e2bfb1b3ee026c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_984dbe1a4816f690fb38e43fe53d9901
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(512, dtype='int64').reshape([]),
            paddle.to_tensor(256, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c77c37b258d316674332f551c0604c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428275c1d6a992946c7610ec57f7170e
    def get_inputs(self):
        return [
            paddle.to_tensor(6, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(12, dtype='int64').reshape([]),
            paddle.to_tensor(12, dtype='int64').reshape([]),
            paddle.to_tensor(768, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_17a1bee9c335494e5020bba68a55185a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(6, dtype='int64').reshape([]),
            paddle.to_tensor(12, dtype='int64').reshape([]),
            paddle.to_tensor(12, dtype='int64').reshape([]),
            paddle.to_tensor(768, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4ea40309794e5eeffe6d1401501c0963(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428275c1d6a992946c7610ec57f7170e
    def get_inputs(self):
        return [
            paddle.to_tensor(10, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
            paddle.to_tensor(48, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6d014c3cb4fea51df3cea2716c4c5582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(10, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
            paddle.to_tensor(48, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_72342f508736d27acc789df0015b5598(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7]
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
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8464b357471eee6d7787d624fae7074a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72342f508736d27acc789df0015b5598
    def get_inputs(self):
        return [
            paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f0489c560b79aadd7a410490ceabc46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72342f508736d27acc789df0015b5598
    def get_inputs(self):
        return [
            paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4577198ecbe4d4afa79707cf39f5bc21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(256, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9b0006a647671256dd5acb399f16b82c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428275c1d6a992946c7610ec57f7170e
    def get_inputs(self):
        return [
            paddle.to_tensor(22, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
            paddle.to_tensor(48, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b8037637932f5c31e2936cf70a978a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(22, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
            paddle.to_tensor(48, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6121a0e4f9d5a428f3ea0ff772b29d74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_162f5cd7d1ac6344ecff665b334ba41b
    def get_inputs(self):
        return [
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(2, dtype='int64').reshape([]),
            paddle.to_tensor(36, dtype='int64').reshape([]),
            paddle.to_tensor(28, dtype='int64').reshape([]),
            paddle.to_tensor(50, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a2b0f8a60125987b6fd91175ea499995(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(72, dtype='int64').reshape([]),
            paddle.to_tensor(28, dtype='int64').reshape([]),
            paddle.to_tensor(50, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a3756011246c4171d7ebdff1837fcad3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(4116, dtype='int64').reshape([]),
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(17, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5ccb059ab21bb170bc1e88804f3cf3e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac62d34b790a6a3e3243514d09f7fc38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
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
class TestPrimitiveOp_aabb38f9ce2d33ae4a6a0331380c7e14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
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
class TestPrimitiveOp_5dc53e19bd1eca425deb09a4a5f8d283(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
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
class TestPrimitiveOp_6134b0b508ec1d0143ca3de0c29f7d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72342f508736d27acc789df0015b5598
    def get_inputs(self):
        return [
            paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6aa70b81235bf5fa83c4ae0d73f7a1e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(6069, dtype='int64').reshape([]),
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(17, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_19e2e3b028768b3f69e975a4471852f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(11, dtype='int64').reshape([]),
            paddle.to_tensor(28, dtype='int64').reshape([]),
            paddle.to_tensor(28, dtype='int64').reshape([]),
            paddle.to_tensor(192, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b27a8c2fee318eb551711c393b802fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428275c1d6a992946c7610ec57f7170e
    def get_inputs(self):
        return [
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(512, dtype='int64').reshape([]),
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor(16, dtype='int64').reshape([]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_24f907396849276de8efe9c1f5c71572(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(512, dtype='int64').reshape([]),
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(16, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f9d24c2e068663377c787a9b5217077c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428275c1d6a992946c7610ec57f7170e
    def get_inputs(self):
        return [
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor(52, dtype='int64').reshape([]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor(202, dtype='int64').reshape([]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8028ee29f65240af4c8cdafd03568312(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(512, dtype='int64').reshape([]),
            paddle.to_tensor(52, dtype='int64').reshape([]),
            paddle.to_tensor(202, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eb7f04141f42009f13d70cec01708447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_162f5cd7d1ac6344ecff665b334ba41b
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(1025, dtype='int64').reshape([]),
            paddle.to_tensor(3, dtype='int64').reshape([]),
            paddle.to_tensor(6, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_18520e738ab9ef9b058bc2141e45b497(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_984dbe1a4816f690fb38e43fe53d9901
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(1025, dtype='int64').reshape([]),
            paddle.to_tensor(384, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_91a44c46039426391aafbeda23ee42fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(512, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc4724f2cfb2bfb82acd136e47245c33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(16, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(150, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4518be139fad8135f5de7a3cfc553457(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return paddle._C_ops.stack(input_0, -1)

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
class TestPrimitiveOp_bfc9f947fa13ddbc31ca3edf2c531047(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4518be139fad8135f5de7a3cfc553457
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34e9342f7a3d8720587e90026eaf7909(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_576dd2f5d2e997d7909885f3a17cc48a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4518be139fad8135f5de7a3cfc553457
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90cf2f8b6576a11e0f84161c15d926c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47a0e71e73bc2e1ad3f9581000855166(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4518be139fad8135f5de7a3cfc553457
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cba90d97395c505df285d29f79d295f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5ddc7a230074524aa11680f03577bb64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428275c1d6a992946c7610ec57f7170e
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(16, dtype='int64').reshape([]),
            paddle.to_tensor(512, dtype='int64').reshape([]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b05e264e8e6c20a6f7c5e1d7ead58b9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(512, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34bf75c51c9cc8c3af3349d715c4e25d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428275c1d6a992946c7610ec57f7170e
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(13, dtype='int64').reshape([]),
            paddle.to_tensor(13, dtype='int64').reshape([]),
            paddle.to_tensor(512, dtype='int64').reshape([]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4e3d3bedbdfbfe0ffbc88fc4a69dc4fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(512, dtype='int64').reshape([]),
            paddle.to_tensor(104, dtype='int64').reshape([]),
            paddle.to_tensor(104, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_31fdeb7aeea9079bb99f014a4ca36307(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(160, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_80c9ecfab86fbf74b0036a825dd9e1c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1024, dtype='int64').reshape([]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_44989b84e89537c9ecd64149d03f332d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_984dbe1a4816f690fb38e43fe53d9901
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1024, dtype='int64').reshape([]),
            paddle.to_tensor(256, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_53d777e8d7394477b828f0f8ec509e1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5124c69283a997149c695ef83a03d516(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_162f5cd7d1ac6344ecff665b334ba41b
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(2, dtype='int64').reshape([]),
            paddle.to_tensor(232, dtype='int64').reshape([]),
            paddle.to_tensor(16, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2605dcf46eae6efa2c8144337f307a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(464, dtype='int64').reshape([]),
            paddle.to_tensor(16, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_010124046f38972d8b0459fa456e4257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_162f5cd7d1ac6344ecff665b334ba41b
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(2, dtype='int64').reshape([]),
            paddle.to_tensor(16, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor(256, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_efea401c83af744262858d6bf2dd9dd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor(256, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1277775feff48ad6f70be5051717c302(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor(512, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4e30275be0176e8b6e5b875718b86148(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor(16, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor(160, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_37f5acf7cde517e1a650c0504b4da50d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(11, dtype='int64').reshape([]),
            paddle.to_tensor(14, dtype='int64').reshape([]),
            paddle.to_tensor(14, dtype='int64').reshape([]),
            paddle.to_tensor(384, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a5fc8478c11bc4d15e5e254f83278464(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428275c1d6a992946c7610ec57f7170e
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(2, dtype='int64').reshape([]),
            paddle.to_tensor(24, dtype='int64').reshape([]),
            paddle.to_tensor(12, dtype='int64').reshape([]),
            paddle.to_tensor(192, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2cbd67a32f0e97f5b2ba174dcb8bb940(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(24, dtype='int64').reshape([]),
            paddle.to_tensor(24, dtype='int64').reshape([]),
            paddle.to_tensor(192, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2b2f8be06ed089b6bc962e8e1bae556b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_162f5cd7d1ac6344ecff665b334ba41b
    def get_inputs(self):
        return [
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(2, dtype='int64').reshape([]),
            paddle.to_tensor(72, dtype='int64').reshape([]),
            paddle.to_tensor(14, dtype='int64').reshape([]),
            paddle.to_tensor(25, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_87295e7fab4e1ac03806057af613fa16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(144, dtype='int64').reshape([]),
            paddle.to_tensor(14, dtype='int64').reshape([]),
            paddle.to_tensor(25, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4527907883baa2bf4fb7eec7d54189f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428275c1d6a992946c7610ec57f7170e
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(16, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_83148b7b0153fae6422e0be4615a2e5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(11, dtype='int64').reshape([]),
            paddle.to_tensor(56, dtype='int64').reshape([]),
            paddle.to_tensor(56, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a57b68b411a1b895b288056f4442bb22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor(256, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_06defccc2b109bb5df1e8f6e054b09b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(9261, dtype='int64').reshape([]),
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(17, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_56ccbb2049af834242cdd114853a79d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428275c1d6a992946c7610ec57f7170e
    def get_inputs(self):
        return [
            paddle.to_tensor(10, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(24, dtype='int64').reshape([]),
            paddle.to_tensor(48, dtype='int64').reshape([]),
            paddle.to_tensor(2, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_730339bf46d28f0045a986a53a3641ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(10, dtype='int64').reshape([]),
            paddle.to_tensor(48, dtype='int64').reshape([]),
            paddle.to_tensor(48, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_98ec5fb022671d618fc0bf24e9e16ffc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428275c1d6a992946c7610ec57f7170e
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(12, dtype='int64').reshape([]),
            paddle.to_tensor(12, dtype='int64').reshape([]),
            paddle.to_tensor(768, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1e4007864d7638f4b20ee8699188386c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(12, dtype='int64').reshape([]),
            paddle.to_tensor(12, dtype='int64').reshape([]),
            paddle.to_tensor(768, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f2217591ad10ace98fe047a4bff7ca87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
    def get_inputs(self):
        return [
            paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ab9bf43fde4707ca5885b9a6e9c9e291(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
    def get_inputs(self):
        return [
            paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fbd2f6fd47071e0084ab812e087d47c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
    def get_inputs(self):
        return [
            paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ffbb68ca5a2ae895bad835055965e82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(2048, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_39857b298a5b2b62d312dab52daef25d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4322fcc7ae433f4e659ef8344ec4e03d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_984dbe1a4816f690fb38e43fe53d9901
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(256, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_072e8ee44ebb666754e58f6380b54c8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
    def get_inputs(self):
        return [
            paddle.uniform([8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 8], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b084e60a5fabad852133206744d118a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_984dbe1a4816f690fb38e43fe53d9901
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4d2d036b7e4b3121e4ea5756b88d7383(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(320, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_277831d8f2813096aaad390e8c417ead(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(2100, dtype='int64').reshape([]),
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(17, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_753af175a5c02d527ce98ce5bb36db8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(512, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fdcc1409dc6082b5aea7f2a5578302d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(15, dtype='int64').reshape([]),
            paddle.to_tensor(25, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8871eb439e65f4e0f578741e7ce7aed2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72342f508736d27acc789df0015b5598
    def get_inputs(self):
        return [
            paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5533105375541e882b07f5bc07d41eb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_162f5cd7d1ac6344ecff665b334ba41b
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(1025, dtype='int64').reshape([]),
            paddle.to_tensor(3, dtype='int64').reshape([]),
            paddle.to_tensor(12, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ae9828f4fe6eb349089fc46840eca333(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_984dbe1a4816f690fb38e43fe53d9901
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(1025, dtype='int64').reshape([]),
            paddle.to_tensor(768, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d21ec0ad3ed50b60cac3378c3999198d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428275c1d6a992946c7610ec57f7170e
    def get_inputs(self):
        return [
            paddle.to_tensor(22, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(2, dtype='int64').reshape([]),
            paddle.to_tensor(24, dtype='int64').reshape([]),
            paddle.to_tensor(12, dtype='int64').reshape([]),
            paddle.to_tensor(192, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b8932728c90686669146c13b3fd9936b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(22, dtype='int64').reshape([]),
            paddle.to_tensor(24, dtype='int64').reshape([]),
            paddle.to_tensor(24, dtype='int64').reshape([]),
            paddle.to_tensor(192, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c148efb69d428316bbeb6bdebabd428(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1024, dtype='int64').reshape([]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e8a4a26ebb0e22b6ece26f80ee7e2e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_162f5cd7d1ac6344ecff665b334ba41b
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(2, dtype='int64').reshape([]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2c8ef41c9e4ebe8ab6eb9a0e8b186beb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_984dbe1a4816f690fb38e43fe53d9901
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1024, dtype='int64').reshape([]),
            paddle.to_tensor(512, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b3ed021237c71e8cc96566e4b0c8797a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(11109, dtype='int64').reshape([]),
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(17, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_442271e4039bfb5c8d1acce9cdb1e809(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(1280, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_274c204270bc58757c3f6f7bffa36d75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428275c1d6a992946c7610ec57f7170e
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(11, dtype='int64').reshape([]),
            paddle.to_tensor(7, dtype='int64').reshape([]),
            paddle.to_tensor(7, dtype='int64').reshape([]),
            paddle.to_tensor(384, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_031248751071ba74828d3a29c709d8fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(28, dtype='int64').reshape([]),
            paddle.to_tensor(77, dtype='int64').reshape([]),
            paddle.to_tensor(384, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2856da90a76f16fe914b2605fc88ce36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4518be139fad8135f5de7a3cfc553457
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_913abfbd83913eac6bc48f21271ae757(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_23604dbc3be83177be072fa27d4d2ff7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4518be139fad8135f5de7a3cfc553457
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f08d44d6655904cc4eec3e569c59bb27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c4d9b8e3761bf095c841e32a47a762e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4518be139fad8135f5de7a3cfc553457
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_db71363f2f9ca35dd28b9736c9e2799f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a2d1af89bccf1406e951cc00e26f0d6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4518be139fad8135f5de7a3cfc553457
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5cd59efbd4d9e82a9ccb263aa1808e8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d75339a9dc5f55e991d19a168d73330a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4518be139fad8135f5de7a3cfc553457
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dc90be5c352a78f54290d9361babdf57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2c6430b6bc3d44f2cc63cd9370e72a4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428275c1d6a992946c7610ec57f7170e
    def get_inputs(self):
        return [
            paddle.to_tensor(6, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(24, dtype='int64').reshape([]),
            paddle.to_tensor(48, dtype='int64').reshape([]),
            paddle.to_tensor(2, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_83646188593410ef75041fc61fef0f2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(6, dtype='int64').reshape([]),
            paddle.to_tensor(48, dtype='int64').reshape([]),
            paddle.to_tensor(48, dtype='int64').reshape([]),
            paddle.to_tensor(96, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb72a9904dd2942903d3d6c46b2c2eae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(320, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c44333748f0614d49ad8092f158adf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(160, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e4b0c5a57ec3efd71e6890411d82d2bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_162f5cd7d1ac6344ecff665b334ba41b
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(2, dtype='int64').reshape([]),
            paddle.to_tensor(80, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4de5258e0f89c48dd744361a886c3232(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(512, dtype='int64').reshape([]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e463b94e3bfe1b0d2f682da4a8a68176(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_984dbe1a4816f690fb38e43fe53d9901
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(512, dtype='int64').reshape([]),
            paddle.to_tensor(512, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1613cbd37412fe24fe707f910ea6b339(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_162f5cd7d1ac6344ecff665b334ba41b
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(2, dtype='int64').reshape([]),
            paddle.to_tensor(116, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_923f08c49950e517b23cd76a901679eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(232, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_78a41e51b963fcb49c1216958c6f01b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(256, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_663cfe613281ee08ce878c0a0b0f8ef5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor(256, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f366e76704d11bdef2ebbe7b92cf1b1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(16, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(320, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_93f3e6f8b1a44e21abb61b96432bc88c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(320, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_889b94fbb5c3e8b465b6c772a215f87d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(3024, dtype='int64').reshape([]),
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(17, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2c1eb190a81e5f582f2f1136be6f62e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
    def get_inputs(self):
        return [
            paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7cbd90221363656e2bbd98785d5ca5b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
    def get_inputs(self):
        return [
            paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ace5a7c9ae3549b9bb9a280816c4071(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eacf2a314f8698a8181739e27e670aff
    def get_inputs(self):
        return [
            paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_417b672d3f74ba88af002d2f99e5fda8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_162f5cd7d1ac6344ecff665b334ba41b
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(1174, dtype='int64').reshape([]),
            paddle.to_tensor(3, dtype='int64').reshape([]),
            paddle.to_tensor(6, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_739a7bd62a3b06f06e923c817b628603(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_984dbe1a4816f690fb38e43fe53d9901
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(1174, dtype='int64').reshape([]),
            paddle.to_tensor(384, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb957aa37c864e6d88704cb956f47059(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_984dbe1a4816f690fb38e43fe53d9901
    def get_inputs(self):
        return [
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor(256, dtype='int64').reshape([]),
            paddle.to_tensor(150, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d7da39f80a51243a5c708524427049c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int64').reshape([]),
            paddle.to_tensor(16, dtype='int64').reshape([]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor(160, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_94c308eb7d59ab7d6f94c4f6d53a3906(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_162f5cd7d1ac6344ecff665b334ba41b
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(2, dtype='int64').reshape([]),
            paddle.to_tensor(58, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2693f50a23ac199f114043cba02cac66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(116, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c7bf5f5c2f0aa60babee97f6cb83367(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_162f5cd7d1ac6344ecff665b334ba41b
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(1174, dtype='int64').reshape([]),
            paddle.to_tensor(3, dtype='int64').reshape([]),
            paddle.to_tensor(12, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c14f2a3a84d9a9ebbcf21fd1a823d314(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_984dbe1a4816f690fb38e43fe53d9901
    def get_inputs(self):
        return [
            paddle.to_tensor(-1, dtype='int64').reshape([]),
            paddle.to_tensor(1174, dtype='int64').reshape([]),
            paddle.to_tensor(768, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_431743ad31de6764fb6f36caf6860b54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd3372f1645cd10abac1dfa8b467488c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor(512, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b02c23dad2fa990b1fc56d9703d3b45a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b54416ccb4ef99b8e4c6b7ab6c6669e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b02c23dad2fa990b1fc56d9703d3b45a
    def get_inputs(self):
        return [
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1816137c4cdae44ca6a853479163bcf0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[48, 48], dtype='float32'),
            paddle.static.InputSpec(shape=[48, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_148c971393397dab5d1b8295104687ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1816137c4cdae44ca6a853479163bcf0
    def get_inputs(self):
        return [
            paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f4fc6025cf739a9b42bfdcae340cdea1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b98d9c3061448697ebcdd083faf4fb90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4fc6025cf739a9b42bfdcae340cdea1
    def get_inputs(self):
        return [
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7525e65fbdb3bfdd6e3bc64db43a7840(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            paddle.static.InputSpec(shape=[300], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e5d8e8d42e3a507bea50d9c861fd2ea2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7525e65fbdb3bfdd6e3bc64db43a7840
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ca73333d95011cff54b2b1614b0a63fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_94896bbd4ac7012d9add668de0a468c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca73333d95011cff54b2b1614b0a63fd
    def get_inputs(self):
        return [
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_04c70895134e3deff0ee2f7ee554a62f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8e1ca8d286f8dd0eb7cb2e453076d9f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c70895134e3deff0ee2f7ee554a62f
    def get_inputs(self):
        return [
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4daf993b05c99e5666fdfb18d4f7a6ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc25a18964eb2f496bdf3e4d5f32fc74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4daf993b05c99e5666fdfb18d4f7a6ff
    def get_inputs(self):
        return [
            paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_06e4151fef49aa6584f2eda04e34d24e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 49], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_24ed90a5dcf01a6ef8fa5747686dcf4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06e4151fef49aa6584f2eda04e34d24e
    def get_inputs(self):
        return [
            paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f3b1b5af5ef5e7a9030f2cb589f12ba4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 784], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 784], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 784], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 784], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 784], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 784], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 784], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 784], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b956fe73b086df6f454bf1b5fc3f7baa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3b1b5af5ef5e7a9030f2cb589f12ba4
    def get_inputs(self):
        return [
            paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_57f5147fe84e62b63f73207becf07593(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[80, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_713f09997c48f2fc8e1e0953aeadb333(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57f5147fe84e62b63f73207becf07593
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

class PrimitiveOp_3b4f23f4994f38524088658aae31f574(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9a2068628c6e13589d8dc3aa08932642(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b4f23f4994f38524088658aae31f574
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

class PrimitiveOp_8ec9b06ee6f9dc47f23e7e10ee9b139d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[20, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c61fb0f9c8edd41967dd73428ad29b0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ec9b06ee6f9dc47f23e7e10ee9b139d
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

class PrimitiveOp_ab299f9d675c8b37232dc3ee4b5a892e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 196], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 196], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 196], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 196], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 196], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 196], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 196], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 196], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cbbd2e9e5c71eedaea2452ac1ea0c8a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab299f9d675c8b37232dc3ee4b5a892e
    def get_inputs(self):
        return [
            paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_56c0fe92bbd1185bbc8b13ced27754ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2236b9f8dc9f67beb94c49b1039e1b85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56c0fe92bbd1185bbc8b13ced27754ba
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_dddf0bfbf0835c3eb697fa98910aeb94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f885a8363327908396ce947f11b38aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dddf0bfbf0835c3eb697fa98910aeb94
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_99b175a5dd2ea9af47b4d4a1b9621d8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a981fcfbe6727142c1425c987499ff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99b175a5dd2ea9af47b4d4a1b9621d8e
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e695f0db3bc8eacf0bb01bd7f8342af1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4126073449b1876d71316141978a528(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e695f0db3bc8eacf0bb01bd7f8342af1
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cfc245b560245520aba3659bbb0338f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c02e488fe32c58fbc1d6fdbbed2ae5bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfc245b560245520aba3659bbb0338f2
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8dcd02f336074d05fa74af228ba9bbc7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_afc4206926eb11879d3d4868f4d8bfc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dcd02f336074d05fa74af228ba9bbc7
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_edd854536d41c76496819025abccba4e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[68, 68], dtype='float32'),
            paddle.static.InputSpec(shape=[68, 68], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b808abdc0f515e28e9f186edcc1eedd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edd854536d41c76496819025abccba4e
    def get_inputs(self):
        return [
            paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_43e8c5b63449d90ecab249b2e271cf9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[34, 34], dtype='float32'),
            paddle.static.InputSpec(shape=[34, 34], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a636c7c8a523c0628e97978171870cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43e8c5b63449d90ecab249b2e271cf9b
    def get_inputs(self):
        return [
            paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b19974656f83d05f6043cda52ceda9d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[17, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d75fc92890e30716faef67723bd4296(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b19974656f83d05f6043cda52ceda9d0
    def get_inputs(self):
        return [
            paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9ad10784831e5672503e0abb72891d98(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 784], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 784], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 784], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 784], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 784], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 784], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 784], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 784], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cdaffa3d3b29953b595ccc53937718f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ad10784831e5672503e0abb72891d98
    def get_inputs(self):
        return [
            paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_63f41f38f880571996e8b99b49d4c66d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ed7673b78f777bf545e276be483ffea1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63f41f38f880571996e8b99b49d4c66d
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5b3734114b87a88f80964fdc136d4675(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38180c6ff2bf1ba13cde96d0ccc82390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b3734114b87a88f80964fdc136d4675
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c966adf65a6b31e06e9f54325986c96b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d44c4a597be267d5049a9bdd26985921(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c966adf65a6b31e06e9f54325986c96b
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_742b32c21aaa3b4ff55cc675508c309a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32f6695ce5a2c9861a3750420a525c40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_742b32c21aaa3b4ff55cc675508c309a
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0d088f5897e0fe95f6ff0af133f2116c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ed6012d1f60779d5a862187849756d5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d088f5897e0fe95f6ff0af133f2116c
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a22ecc6cd8fa477e5433e08565a41352(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e07b52c07b3888dcef396eb014acb0fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a22ecc6cd8fa477e5433e08565a41352
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_148a8ca3b47e34d9d2a4758e66a30078(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_de698f36644e20dac298f6514bafeb3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_148a8ca3b47e34d9d2a4758e66a30078
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b8a56e823b92349d716b9ea865fd0551(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_df8aa977f38f8524bde61ecaa88247f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8a56e823b92349d716b9ea865fd0551
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2fbfe8d7950f65a4f7f0c57fa50f8f42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a4f583c709c7a63f537270ceaec4d70a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fbfe8d7950f65a4f7f0c57fa50f8f42
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b43f7629bffe88ec3392d1d75ae5b008(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_74255c634de7fc9be06f731339285d91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b43f7629bffe88ec3392d1d75ae5b008
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_395ad68914eaba05f5ce2885ebbefa29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[72, 72], dtype='float32'),
            paddle.static.InputSpec(shape=[72, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_55fd68b0762494682d7421995567c4a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_395ad68914eaba05f5ce2885ebbefa29
    def get_inputs(self):
        return [
            paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6cda4cd7dbfc76d9e8313a7cc422cda7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[36, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a3b0dd1d781299dda5c1db08495f98f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cda4cd7dbfc76d9e8313a7cc422cda7
    def get_inputs(self):
        return [
            paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bf7147959fb0dca717221a6c35fdb8bd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[18, 18], dtype='float32'),
            paddle.static.InputSpec(shape=[18, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_00c374500d1890265caf90547df0455d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf7147959fb0dca717221a6c35fdb8bd
    def get_inputs(self):
        return [
            paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
        ]


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