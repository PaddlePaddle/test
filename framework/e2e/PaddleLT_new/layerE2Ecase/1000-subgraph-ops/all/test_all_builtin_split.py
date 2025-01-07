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
class PrimitiveOp_1848e60d49960e74b723e5f73421c99d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bb4ed6179b2a01843f7b0404f8c4d82d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_88bc0c57e668f9e2df7223998fb5ca57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e734ca0b10721d7d688a261329cb981a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return input_0

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
class TestPrimitiveOp_1dc29ff3e00161fc6a324b0d49f2136b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return input_0

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
class TestPrimitiveOp_cd8413f40117fe087071a90c85e9eb53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([300], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_092a8d62914230a463eeb9d7215936c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c775f226a8b2b694c6dba8eb98444794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a81a43d3a597908b508deaae85c43d9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c0d35f3b5a0f30e8f44d6221d391b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([8], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_79f65a44c58b27995e90007b4a0ebd5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33935546875, 0.16132095456123352, 0.045737527310848236, 0.3098590075969696], [0.3117451071739197, 0.10883624106645584, 0.24767731130123138, 0.24308723211288452]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_97e60f0d8187517dbccc2e56bc869089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_05b8bde67313b7d9c11fce33ea18321d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
class TestPrimitiveOp_edba7fda5fe10881888a3f3b2ba969ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_144ad015232a4ec9babf8748384970a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_959ccfa6b5df262a29ac321871188c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_6e49408cda69bce22d03907e6a4ad98e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_cca586620806f9812960aa2a71b76721(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_690b381e3831f63d2c4d59a651a79570(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([100], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3bab8179aa3694797c877d3b359028b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e659c0e79600def1c3a8d34b28716f9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 6, 6], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 6, 6], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_61f43a31ca5da36e676a6ff28ff71b21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 52, 52], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f711307187e6a459aa52e55d4d96254e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f2eafe799c9e741dde083e76a9c66de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_240cd57f8d226b2bdfadfe5f4ae41924(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_56d69b5b0bcbfad9b50f41c85652f5e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_5c828b0cc0c36d336ec79f8430d27e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
class TestPrimitiveOp_19d110fc29f042d2c5506bd7fa43bfa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 60, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4386e65e3847411e1b8a1382e02f3ddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_082d50bed6d2d2deb2b22cb0f7e145ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 11, 11], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9f636dacfe8f382924144ac1ce6a80f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3767479658126831, 0.09100621193647385, 0.187086820602417, 0.25773152709007263], [0.2633567452430725, 0.00639107171446085, 0.47647345066070557, 0.33961549401283264]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_41364762b0d90a75a6eda5186e6f0202(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

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
class TestPrimitiveOp_2597f29bf1b5e3789e6c900f08d6c468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41364762b0d90a75a6eda5186e6f0202
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf61bf4af558c4a79a30a62f8b82e16b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_961792c66c633a0620e72d153f6e63fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 9, 9], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b2129367263bf0a2cc9c6b900b6f92dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return input_0

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
class TestPrimitiveOp_0b87b0efd598f8efd228d1d81cc38c5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2129367263bf0a2cc9c6b900b6f92dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.028633873909711838, 0.306168794631958]], [[0.3193291425704956, 0.09081225097179413]], [[0.08506157994270325, 0.29002857208251953]], [[0.04719119146466255, 0.3807370364665985]], [[0.11347969621419907, 0.4741647243499756]], [[0.32951560616493225, 0.25431615114212036]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.10314705222845078, 0.4166870415210724]], [[0.39106929302215576, 0.08782759308815002]], [[0.12779450416564941, 0.1573505848646164]], [[0.1394997388124466, 0.21213169395923615]], [[0.18512240052223206, 0.08317950367927551]], [[0.3243594765663147, 0.45078641176223755]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.322207510471344, 0.2033071368932724]], [[0.09215298295021057, 0.3187134861946106]], [[0.12491016089916229, 0.2946590185165405]], [[0.20400087535381317, 0.21122798323631287]], [[0.11348603665828705, 0.3973521292209625]], [[0.4078381359577179, 0.4132659137248993]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.19120168685913086, 0.44401031732559204]], [[0.08718405663967133, 0.4174824655056]], [[0.020174739882349968, 0.04101495072245598]], [[0.09167716652154922, 0.4122407138347626]], [[0.3388717770576477, 0.13130366802215576]], [[0.19049964845180511, 0.4378237724304199]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6a175d0c0530a3c5162117bbcd513875(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

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
class TestPrimitiveOp_650cd037fe99bd55c8d5958fd68a9ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a175d0c0530a3c5162117bbcd513875
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.018874194473028183, 0.3943251371383667], [0.489003986120224, 0.09215033799409866], [0.4628400206565857, 0.35706427693367004], [0.056388091295957565, 0.28063154220581055], [0.12517590820789337, 0.21209801733493805], [0.03224913030862808, 0.1864347755908966]]], dtype='float32').reshape([1, 6, 2]),
            paddle.to_tensor([[[0.23993122577667236, 0.3653700053691864], [0.24037961661815643, 0.21902430057525635], [0.11654946208000183, 0.05408056080341339], [0.08582931011915207, 0.20485243201255798], [0.343675434589386, 0.20209373533725739], [0.29344651103019714, 0.33133837580680847]]], dtype='float32').reshape([1, 6, 2]),
            paddle.to_tensor([[[0.2596043646335602], [0.3882347643375397], [0.07454115897417068], [0.45202702283859253], [0.1090674102306366], [0.15080255270004272]]], dtype='float32').reshape([1, 6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_74c6c37652caafcb9ffba3c1e0587aa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a32f6f08b9ebf9769240cea6e858a64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 24, 24], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ab6171e1d9ea31c6c02f10ec574dd0dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 12, 12], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_95fb916c9c329a94f991bb7f30bb99ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14884699881076813, 0.47242850065231323, 0.3183194398880005, 0.24894773960113525], [0.47321370244026184, 0.18095393478870392, 0.40104353427886963, 0.2941465675830841]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_43083f02af5d39f7b67d7ee0c0c0260c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_72aa8d8d05456e8611725c8efe07ea3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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

class PrimitiveOp_d5dc53de0d2688a5b4748447c9855df4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return input_0

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
class TestPrimitiveOp_89161b457bd2a2bc6708b5c78c1eaa8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5dc53de0d2688a5b4748447c9855df4
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
class TestPrimitiveOp_ea47b32c79f402891afd52bdc17c1919(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
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
class TestPrimitiveOp_979426c7c8aa6341f1f78e034608ae3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_12bf2ac8c534ef4d25941c2dc1369827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 24, 24], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b608b7e54c3e9c0db3bd2ae1327a0621(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
class TestPrimitiveOp_d72736ee64e7806324ddc10902ba3424(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 15, 15], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_58ff7aa29e7c894b62adecece32e7017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_45753d63be77e5a306508a8b2d5e948f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 36, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f539f7f0f20681833c8b0ce92cd01122(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([1843, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1843, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1843, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1843, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b4ab8938db859088e381c651a072219d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 8, 8], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_70b9d3bbbd752904d5da78c302fc4f5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
class TestPrimitiveOp_632825a5440eead2cdfc0d10e827dc22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f60c81c970ad1c48f24c5a7d5db3dd2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_068144b324fbc9d974dcb67738c5cad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
class TestPrimitiveOp_84141244e7e3593c5a62475f3560de0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14373856782913208, 0.42031458020210266, 0.17581358551979065, 0.421566367149353], [0.1028633564710617, 0.3344844579696655, 0.20517560839653015, 0.34165823459625244], [0.1792476326227188, 0.4266873598098755, 0.23294998705387115, 0.011388166807591915], [0.33024778962135315, 0.1721087098121643, 0.1939237117767334, 0.2781818211078644], [0.22236931324005127, 0.27251046895980835, 0.35136815905570984, 0.011399761773645878], [0.3315168023109436, 0.37677645683288574, 0.11405857652425766, 0.16159768402576447], [0.40157678723335266, 0.17235715687274933, 0.19853566586971283, 0.16763707995414734]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5f2a9a34d9ce095a6d373a978df49d93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11c9f70cbe998840308baa5ba8c1dbd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b38f7c0d5748f90ac42d1cfceb596f6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7]
        return input_0

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
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dda2a47081d8eb070b1881dc1fd6993a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4b561d65e8cf1cccbeb94f9358c9c2bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0376352034509182], [0.04618304222822189], [0.19655629992485046], [0.2655533254146576], [0.07565709203481674], [0.4892984926700592], [0.1961444914340973], [0.37420839071273804], [0.28486770391464233]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3668404221534729], [0.25006723403930664], [0.4367183446884155], [0.3702875077724457], [0.01806609332561493], [0.3522672653198242], [0.4979040324687958], [0.4118877649307251], [0.1997719407081604]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3040970265865326], [0.14878807961940765], [0.49822190403938293], [0.4209156930446625], [0.25913557410240173], [0.3326629102230072], [0.04537851735949516], [0.4687543213367462], [0.3520947992801666]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.09477832168340683], [0.034025467932224274], [0.19512628018856049], [0.2819446623325348], [0.021398765966296196], [0.4702199399471283], [0.48925352096557617], [0.009847324341535568], [0.10192611813545227]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c5048f1c0b7baedfc9a816e20c48b694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.059051621705293655], [0.3663615584373474], [0.11374413222074509], [0.05777735263109207], [0.11371004581451416], [0.33100050687789917], [0.13957619667053223], [0.23278938233852386], [0.17369438707828522]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.03673207759857178], [0.49808135628700256], [0.20493631064891815], [0.4672515094280243], [0.2122543603181839], [0.12291333824396133], [0.29448965191841125], [0.30679893493652344], [0.03264906257390976]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.4014131426811218], [0.22237733006477356], [0.10303381830453873], [0.2967050075531006], [0.4305284917354584], [0.34510764479637146], [0.00745149701833725], [0.07239554077386856], [0.18927383422851562]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3714483380317688], [0.40653151273727417], [0.10373926907777786], [0.18074949085712433], [0.06578893959522247], [0.28882426023483276], [0.0583321675658226], [0.4913760721683502], [0.3878237009048462]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cbc4d7cd667c46d47f21de001c4e895c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6933fc52eaa223c14c2741bab6c14bb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
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
class TestPrimitiveOp_a584fcd652d89ea075bfce9f2062e1b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
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
class TestPrimitiveOp_f395f03fafee203c555a94643e375026(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
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
class TestPrimitiveOp_f6d6a9ddb8f95fbf71f1c24be8d10219(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
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
class TestPrimitiveOp_7778eed994216c0e5523a822369834e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 4, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e9ea3a6b871abe3b5aaf3a7a606de0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([5583, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5583, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5583, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5583, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b5f7f362f5ad5b569ecb7d82e6494e97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
class TestPrimitiveOp_ae021050230c6fdaa1d738576612010d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f44f25951247cc47c96f06e0b0124ed1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14185766875743866, 0.2782001793384552, 0.4330953061580658, 0.4712361693382263], [0.16310042142868042, 0.08349864184856415, 0.08359965682029724, 0.249087393283844], [0.3515278100967407, 0.23490144312381744, 0.1178266704082489, 0.30084607005119324], [0.21413810551166534, 0.4333586096763611, 0.09454689174890518, 0.47111082077026367], [0.003853335976600647, 0.4897143244743347, 0.481177419424057, 0.04118751734495163], [0.12994852662086487, 0.14688752591609955, 0.3679650127887726, 0.046101413667201996]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3af90ab907c9af06442c56824c7738ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([6], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0257890e925a139b16372a5d1da55ec8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6dec380f21e65c9fcb0a4d619a1cc373(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90c72ac2fc530513aef7cd3da0785a30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46da71705d543aa98a7bd98ab7a47bf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.058449819684028625, 0.09226198494434357, 0.038015492260456085, 0.4389108717441559], [0.1974448561668396, 0.45112109184265137, 0.4329894781112671, 0.4065325856208801], [0.14642252027988434, 0.20309582352638245, 0.025423023849725723, 0.03931766375899315]], dtype='float32').reshape([3, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ddc5f29dbd04067f08de4222dfec09e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4a1124fc591ac9867819bf737c738110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 34, 34], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_17043484afab04c36688a2001e6f7b9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7ed4049586f80e8798ec3607daf8ed92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
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
class TestPrimitiveOp_dea896ef235528715b6a2535f02076e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 44, 44], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f0740cf76dfcc5c2cbf7abd4fdc48a1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d4bb432e730ad8ef66231d92c5203859(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 18, 18], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_499fbde81d3808b6ea0a46a48f40c6f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 17, 17], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f607bff1041daf3617005641d1e80184(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e1538baf0702a5a965023a2fbe99923(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_21353360348922b803268b71dc2366a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 22, 22], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3905781722be7ea62882ef5b2d11bbd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5dc53de0d2688a5b4748447c9855df4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20842023193836212, 0.2758379578590393, 0.27349337935447693, 0.37534666061401367, 0.46773508191108704, 0.21818071603775024], dtype='float32').reshape([6]),
            paddle.to_tensor([0.42704054713249207, 0.06060203164815903, 0.37507253885269165, 0.2784046232700348, 0.22594958543777466, 0.4257948100566864], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3536610007286072, 0.4108772277832031, 0.4120214581489563, 0.15821325778961182, 0.2899235188961029, 0.08802277594804764], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0032873579766601324, 0.4201934039592743, 0.23457805812358856, 0.10703792423009872, 0.2224547415971756, 0.4281674027442932], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_30492e6d0f730f43fd94a79025bdd375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5dc53de0d2688a5b4748447c9855df4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11142879724502563, 0.011232888326048851, 0.26740866899490356, 0.2513158917427063, 0.18938873708248138, 0.4642762541770935], dtype='float32').reshape([6]),
            paddle.to_tensor([0.48052719235420227, 0.1542738527059555, 0.4869168698787689, 0.44539231061935425, 0.05634588375687599, 0.15570500493049622], dtype='float32').reshape([6]),
            paddle.to_tensor([0.28813254833221436, 0.21503736078739166, 0.4214804172515869, 0.27635470032691956, 0.17865359783172607, 0.20482513308525085], dtype='float32').reshape([6]),
            paddle.to_tensor([0.011191552504897118, 0.10982688516378403, 0.01682184264063835, 0.1927325576543808, 0.01876295544207096, 0.3875589072704315], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a82ccb1b3df2137daaa031b91ca52aab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([1724, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1724, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1724, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1724, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fa0722a8bef29126a25f5a0594a17d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 5, 5], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aae150eb0fc942f51c08117b5ecb1d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a175d0c0530a3c5162117bbcd513875
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_80c055ec17938795cd0f94d79f2f403d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12616214156150818, 0.0712435245513916, 0.326055109500885, 0.3594553768634796], [0.03403058275580406, 0.0038598976098001003, 0.3403976857662201, 0.3463982343673706]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_31f3771d77b3eb287bde90ee8b122515(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
class TestPrimitiveOp_9e5d3216a71c06e484b9f96b317c1c3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.42137449979782104, 0.1966567188501358, 0.4952227473258972, 0.05661465600132942]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c451a4630aa40476f8c12a2dd8b667f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a00e76e3ff79a2bd5e025d26b6a16aa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 17, 128, 128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_30cc89538f842dbc1ec95e3f6bef6063(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b9dea30d0cbeb10588a5bd833f33b1e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_373feead1a9fbbed37952fe0804f5e40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_bcbe28e378696ed8518cfa7a8a7b580f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_47ff3884c46ce473bca1559ef26f47ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41364762b0d90a75a6eda5186e6f0202
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 49, 8, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d6ceac466e65ed955a6a19d648edaff9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 13, 13], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ee48c8e4f066a56ab10cb0112564321f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41364762b0d90a75a6eda5186e6f0202
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
class TestPrimitiveOp_4c7e8a203eec421bfdb121d3019855c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 36, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a51d1c57ef055c27da9ed5f2f70ba499(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a88937a4ba4b9baec3388c0967dd94bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e547a28ba2aa7721d42891e822f7d708(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d5c47e276a7632be0d355052650efdf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
class TestPrimitiveOp_5a231c7d07f8191820a7acb723c01c80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([1492, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1492, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1492, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1492, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4233c14a8059d68bd80d489a3d3ae3a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
class TestPrimitiveOp_8c5f3c7843830a001e0e98f85b5f618f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77bc556f36c2162ce7909f25b18d7ccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_feb7dad83e09ea39cada7d0775a0210c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3760044276714325, 0.12422966212034225, 0.04910736158490181, 0.13082978129386902], [0.1478123515844345, 0.18232350051403046, 0.4741666913032532, 0.3131527900695801], [0.4784177541732788, 0.1411914825439453, 0.4344243109226227, 0.31855785846710205], [0.14063599705696106, 0.0371808297932148, 0.11269502341747284, 0.12853483855724335], [0.345084547996521, 0.22814075648784637, 0.3940271735191345, 0.28395769000053406], [0.3161979615688324, 0.4849144220352173, 0.43376392126083374, 0.27867332100868225], [0.0065774694085121155, 0.16474632918834686, 0.48897072672843933, 0.450259804725647]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d316ad8b4d59db3809f6533ddbe012c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2633066773414612, 0.17589786648750305, 0.1740284264087677, 0.023862803354859352]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5047874ce14665dd3653812dce54ca6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7da23b76417507432cf3a3bcb410c56b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41364762b0d90a75a6eda5186e6f0202
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8cd10dc14dd590d8a541c15e6280027a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.46800416707992554]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.48410019278526306]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.16707240045070648]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.03597502410411835]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dee6ab0ccd0426cf8f108a69fb28088c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22265328466892242]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.38656848669052124]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.24377180635929108]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3414774537086487]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aa24ae4a2f5342072d0366048128381e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 8, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196, 8, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8962e7ce7731259d207dd9382eb2926b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.007817946374416351], [0.1666908711194992], [0.19357992708683014], [0.4992569386959076], [0.4388372004032135], [0.25200337171554565]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.030296431854367256], [0.23925189673900604], [0.27806979417800903], [0.09709051251411438], [0.4757535755634308], [0.4532786011695862]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.30872491002082825], [0.32114145159721375], [0.030900035053491592], [0.3291957676410675], [0.13353675603866577], [0.17648190259933472]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.06679902225732803], [0.2054377794265747], [0.36668092012405396], [0.306477427482605], [0.43354904651641846], [0.24444907903671265]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0611823f86c8511b7529d5db046f712b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.37052908539772034], [0.13874346017837524], [0.10015410929918289], [0.2557911276817322], [0.1847122311592102], [0.24399599432945251]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.14124219119548798], [0.04379585385322571], [0.43415820598602295], [0.15889577567577362], [0.3719024062156677], [0.0006744983256794512]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3794514834880829], [0.10002264380455017], [0.24930144846439362], [0.05756419524550438], [0.35914668440818787], [0.47518736124038696]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.2596014738082886], [0.2903253138065338], [0.1237405389547348], [0.06167442724108696], [0.4700954556465149], [0.43061691522598267]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_56eec34e495f7f3c2c307499df1c6cb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2129367263bf0a2cc9c6b900b6f92dc
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
class TestPrimitiveOp_d28d1ebe3696cca2ecbb0aa214b7cd23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_da2bac44525413c24a709f2eee72de2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_77b5f380ded411bbb99efe60e054cb8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_014fe8222e4aca10d895aef341bbbb40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a175d0c0530a3c5162117bbcd513875
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fd838c0c16315be90fb5ddba34cc6997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 32, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c379250b4be726c275b44e808e61819(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3426182270050049, 0.371687114238739, 0.17967729270458221, 0.05511809512972832], [0.028130333870649338, 0.08196408301591873, 0.17705516517162323, 0.4715835154056549], [0.27289482951164246, 0.43746933341026306, 0.3031827509403229, 0.16167962551116943], [0.20820161700248718, 0.023066138848662376, 0.3269689381122589, 0.34487488865852356], [0.254242867231369, 0.17583134770393372, 0.4541226923465729, 0.3056456446647644]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d33ee0f7804e19ae5fa954d05146406(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([5], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f4ebbbaec636b25a6951b47c893cc2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06615816801786423, 0.17510589957237244, 0.12482108920812607, 0.3349984884262085], [0.09679903835058212, 0.35872501134872437, 0.2356889545917511, 0.3929291069507599], [0.4208967387676239, 0.013075667433440685, 0.12194177508354187, 0.09354513138532639], [0.16531191766262054, 0.45159438252449036, 0.43057894706726074, 0.33081603050231934], [0.29036158323287964, 0.4901483356952667, 0.10050932317972183, 0.4676995575428009], [0.3282496929168701, 0.01271427795290947, 0.07691396772861481, 0.007982900366187096], [0.3301147222518921, 0.16804207861423492, 0.19386331737041473, 0.050972338765859604]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f47b065ac3d04d3c083c91d06ea73cd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4841764569282532, 0.4656074643135071, 0.32657986879348755, 0.29411062598228455], [0.1873297095298767, 0.25933483242988586, 0.2513105869293213, 0.16280633211135864], [0.17818327248096466, 0.274533212184906, 0.3206072151660919, 0.4815177321434021], [0.3417300879955292, 0.23737823963165283, 0.24936912953853607, 0.44680628180503845], [0.2608222961425781, 0.1117267906665802, 0.1882377415895462, 0.0022811219096183777], [0.11898007988929749, 0.39454466104507446, 0.33293068408966064, 0.1850283145904541], [0.3458210229873657, 0.4326430857181549, 0.10277026146650314, 0.4839790463447571]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e04b5c38fcf5d2371528b137bc751e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 96, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 96, 96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1982626b5b29898e572e3cb7cce23925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 88, 88], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_659c84a3bc8edaf2b8a2e0d5527f1851(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07233921438455582, 0.2189362794160843, 0.08887238800525665, 0.32634830474853516]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e36ed182d8f298a6b0709ece386d4533(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a175d0c0530a3c5162117bbcd513875
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f3ca06e9d4e684c2f9d147a77d4cb5c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d1fee40e0a8a33a93b487b66bdf3ffe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([2079, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2079, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2079, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2079, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4db40d33bb75ea13a6fcf54ed493f21f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2129367263bf0a2cc9c6b900b6f92dc
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
class TestPrimitiveOp_bea112607e2311e80da5e571a9c86161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_17d05eb6382d2ea58862dbdb8e9865f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_dadacece982240c8744107d28c88cd17(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
        return input_0

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
class TestPrimitiveOp_77a70d4deb03de21ad869269414b83cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dadacece982240c8744107d28c88cd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_311ef798a7fd8ccd1a77bae182d94cc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([4526, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4526, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4526, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4526, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_78bad22fe49e21cec6a67e7126a6ddd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
class TestPrimitiveOp_9694e6ff00cb6491a017396a14f18581(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 17, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6311ec15e87a13a412f88d561891c7c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([22, 600, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 600, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c2aa2ae6cd5171f3927a8021bd74d5a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12845773994922638, 0.16413450241088867, 0.3531971573829651, 0.39715954661369324], [0.3552411198616028, 0.1575901061296463, 0.3620745837688446, 0.0040739066898822784], [0.3909343183040619, 0.4800011217594147, 0.08939668536186218, 0.31853631138801575], [0.371174693107605, 0.048930808901786804, 0.41185128688812256, 0.37662333250045776], [0.45312097668647766, 0.09228237718343735, 0.039329834282398224, 0.2555757462978363]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c08694a3343c8e3c36b181d0b1fefe4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([1046, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1046, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1046, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1046, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c63d5f01dedb07a889a46f7cc76af9ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
class TestPrimitiveOp_f4e6ed5bdb7d782bc788cf9d057035bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_871a86c50ec55a304be42675abf7d075(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_abe25245c08fed1e8471d926e0887bb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c61df594738380170c139198793e8c14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 192, 288], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ca132ebf48f034a1a571349add59d0e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_419ba98224dc93f12ae7fb3713e0baa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a175d0c0530a3c5162117bbcd513875
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3e5058132190a639d5d26cd388a876ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_16d86ea15b393d4a480f7fa28100d85d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_1f945e2bca3413828b3e004f7913a906(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_4ea37fd3f4008fe17ad4b9936174e214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08773402124643326], [0.46262967586517334], [0.004426985047757626], [0.064740389585495], [0.2718239426612854]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.11126096546649933], [0.2779003977775574], [0.12715952098369598], [0.03098335675895214], [0.22021512687206268]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.24352209270000458], [0.11840708553791046], [0.20973068475723267], [0.31463444232940674], [0.25592753291130066]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.4568329155445099], [0.2669721841812134], [0.11313679069280624], [0.41974976658821106], [0.3077079951763153]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_45e754ff9456f6c227ecd7407a8be873(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2749200761318207], [0.015816230326890945], [0.03016461431980133], [0.20059402287006378], [0.0772194042801857]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0596802644431591], [0.2538567781448364], [0.03441440314054489], [0.47248581051826477], [0.3944278955459595]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.057828549295663834], [0.022255240008234978], [0.08091912418603897], [0.022901371121406555], [0.4846023917198181]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.1068728119134903], [0.473810613155365], [0.3731231987476349], [0.48226043581962585], [0.17963165044784546]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_41a1ae48724f72263a42b90132b61ff2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12460564821958542, 0.22093810141086578, 0.22483277320861816, 0.3253820240497589]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8ee462787e16133403b927ba886fa509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 68, 68], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_379b64667ba43ad8a6303d8f433aa279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_63cb58fa23a3f6d184ec8519e1128f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 72, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 72, 72], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3a46aebaabcd91c85fde2923b84824f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
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
class TestPrimitiveOp_856ee7559a12dee65d21e7b25fc57236(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a175d0c0530a3c5162117bbcd513875
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_715d31ab33200caceb33d78602eb679e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 12, 12], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_70876e4edca7828244d87fbd713011e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22656188905239105, 0.04892861843109131, 0.18918175995349884, 0.1263914257287979]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4870f1f9e7f83a33bff195cb6b8129b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14fc5f240e2cb1646d1c5230c304e9b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_66b5c4abbb1f09af8b86f8b8964fb96c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([2335, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2335, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2335, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2335, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8cfda85918c32e88812353b636f4108e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([2986, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2986, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2986, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2986, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a44b0d22cede5d69f07ba032ef36985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([3783, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3783, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3783, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3783, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1e3f1ed672a768d70e7236e2eba48e55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4ffee4989e420692e825f235cfe14206(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 16, 16], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6043f2ed5775e2fe5e86d512dd5ec4cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_56e1566c86f681cde1f2f08c29e2357f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
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
class TestPrimitiveOp_fa60117ce2008ebdaf0ff637a39e04aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a175d0c0530a3c5162117bbcd513875
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_55a07bccdf4f629b02a8223851564ca8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 8, 8], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1d28e17ba3d110c98ad3d3fcbc282d5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 120, 200], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ee1367c7bf949c3dc1f32605fa8f568c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_4d24e06693772dd9aa359672efd4e391(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_e4a9bfed4497c1a883828138252bdc96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_d7fb66e666b52b25a9cf3332d74459d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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

class PrimitiveOp_1d4039695f3562fdf95293b9ad4f83a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
        return input_0

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
class TestPrimitiveOp_aeebba240e2521aa78ec7a295229e9e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d4039695f3562fdf95293b9ad4f83a7
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
class TestPrimitiveOp_cadbf52be64784c09df1a2d074e92f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4a3d21c9c23d377f912866c3f5c0347a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_129481c0f4a5ab099c4bd60676ef6c7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
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
class TestPrimitiveOp_b47fc31c8da2676010f14764861aa462(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7afb92e405eb5eb06ec82660d818de6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
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
class TestPrimitiveOp_5ef31bd6ee7d4c1ba6cabdcde218f435(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a175d0c0530a3c5162117bbcd513875
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9028ffbe315a6dbd44c2306d17844906(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
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
class TestPrimitiveOp_a5287e1a5bf93c32e72d7741f4cbfeca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.353889137506485, 0.04948345944285393, 0.21553945541381836, 0.37403661012649536]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ddd6b809578785fb1798c9e317e00d0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15059156715869904], [0.024037236347794533], [0.4898376762866974], [0.2543194890022278]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.43786975741386414], [0.2647301256656647], [0.011273171752691269], [0.1107817143201828]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.23596063256263733], [0.18666329979896545], [0.06731393188238144], [0.3615970313549042]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.07578111439943314], [0.05176808685064316], [0.18972070515155792], [0.19638481736183167]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_98c5b099da0c4fdbf0b22b9547f3fa65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3180879056453705], [0.30447274446487427], [0.15876974165439606], [0.10897774249315262]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.05201704055070877], [0.07029230147600174], [0.4768783748149872], [0.3339034616947174]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.004064901731908321], [0.173634335398674], [0.1498268097639084], [0.34006810188293457]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.16424576938152313], [0.19745327532291412], [0.15565119683742523], [0.40988653898239136]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_514e19c9737cb70487638acb2aece79b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f3ac1a218a0edb0ea3abdb0a18ad5c3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a175d0c0530a3c5162117bbcd513875
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d30cceabf35beb58a3480764c1ac194f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([2041, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2041, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2041, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2041, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b94255054c2a003ede54f61a4dce5267(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41364762b0d90a75a6eda5186e6f0202
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 4, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196, 4, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196, 4, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4d7fcc96d9d3a873694636b98456990(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_caaecd4c0edc0e3690ae17e823aa4c76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_de783c1c948fa237947766dc25da6dce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
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
class TestPrimitiveOp_805263bdb92fe1c58cb0d47d23a100c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
class TestPrimitiveOp_0e1194733ca52015955f5b7f5a03be9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
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
class TestPrimitiveOp_96ca494ba9ac9edbada9de7554cd80d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41364762b0d90a75a6eda5186e6f0202
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 12, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 16, 12, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 16, 12, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3fcd0b0eaa8cefc8603c73bd42fe3dfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 18, 18], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4b98b19fca2e17973d4f230a12ba61dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 14, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b09719fba1c8851eb89c963f8670262b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 48, 48], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5685b80eec09d7cff5bb5279fb3538d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([4210, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4210, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4210, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4210, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3ea579974162fa95c89dbf0f9e0a1d99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12450958043336868, 0.052685387432575226, 0.3968353867530823, 0.15485253930091858], [0.34951451420783997, 0.32297182083129883, 0.38014039397239685, 0.47485774755477905], [0.2576924264431, 0.4425520598888397, 0.14566507935523987, 0.4091264307498932], [0.19692909717559814, 0.3446423411369324, 0.3547402620315552, 0.12531565129756927], [0.4130346477031708, 0.48923182487487793, 0.30839577317237854, 0.05351301282644272], [0.2585386633872986, 0.3128417432308197, 0.34252235293388367, 0.39456647634506226], [0.3814266622066498, 0.10656841099262238, 0.28748100996017456, 0.2259080708026886]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_18e9ec9ebba619e84ffabe7e135e1eda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.13486644625663757, 0.05036383494734764, 0.17426297068595886, 0.035235773772001266], [0.45595043897628784, 0.45600008964538574, 0.41477057337760925, 0.39209648966789246], [0.254535436630249, 0.358079731464386, 0.4156394898891449, 0.2574434280395508], [0.13503143191337585, 0.07170978933572769, 0.31542617082595825, 0.4614073634147644], [0.17103426158428192, 0.2569636106491089, 0.24876902997493744, 0.303025484085083], [0.4615570306777954, 0.3139835000038147, 0.48396337032318115, 0.15067026019096375]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46bcbd4c4302e9485e730ae9d92c692c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_529990e39fbc574e0b8f6e8e56d16b87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 49, 16, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_72c239abbbaca8cf2c19e6e35ade7786(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f106e9dda5232c93243dff7fc2676c69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_44124b1631ce8b732a6280243ef6e81e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7d8e9bcd2df434a2ee27726613b97239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9fca06b6355a006742622c9b2ac24067(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33935546875, 0.16132095456123352, 0.045737527310848236, 0.3098590075969696], [0.3117451071739197, 0.10883624106645584, 0.24767731130123138, 0.24308723211288452]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1457086f8118f6ae44c94f8b23f05743(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c52b5fc607993875a61bf26722edbc5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1457086f8118f6ae44c94f8b23f05743
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
class TestPrimitiveOp_4d39cebbb9ce8b420f2268eb38ce6406(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_70bb88f93de413821c8609aac9786df4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9392b30c0e30f025725ed475fb5621b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dc551b7c1c58b70a0dd7762fcf26dba3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 6, 6], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 6, 6], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7c31944942426a78ec58610628e274a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 52, 52], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8d508cf2cbc394b8467027eb29c340e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0510205a9c7efdd4d2de55a77f5418b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d508cf2cbc394b8467027eb29c340e5
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

class PrimitiveOp_58d1fc007f768492e002012cdf5c79e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[48, 48], dtype='float32'),
            paddle.static.InputSpec(shape=[48, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8ccc9c595004109166c3fae8d05cdc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58d1fc007f768492e002012cdf5c79e9
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

class PrimitiveOp_0a4b9c90c22e6ec6569fdb099c53a9f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1701b798f38005fd8e96c92e7e1e32af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a4b9c90c22e6ec6569fdb099c53a9f1
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
class TestPrimitiveOp_353cef533498117db7a6e1f950fb36af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 60, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8c6f2b983f9d859f3383c79a15bbeee4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7b754fd89cd1714facabcab6a67ce3f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 11, 11], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a9bec26e4e224094b9925cc10be22a18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3767479658126831, 0.09100621193647385, 0.187086820602417, 0.25773152709007263], [0.2633567452430725, 0.00639107171446085, 0.47647345066070557, 0.33961549401283264]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_536554d4068550f339721ff7091de2ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

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
class TestPrimitiveOp_738ad3ab4bec68e89c6e3dbf4fc36676(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_536554d4068550f339721ff7091de2ab
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c1782ba106b739e0998af469f53e9819(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_747308f836dc0eb5fd5914713203fec5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 9, 9], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07971b1ceb2a33e1f71089fbe82aab1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d1b2370ecbb4efe49164afd11ec0b4bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 24, 24], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7d4b4b7a886369b5abf5c9b7038c96f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 12, 12], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_40cc2ba11734fe4e9bd17ce65f8bae96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14884699881076813, 0.47242850065231323, 0.3183194398880005, 0.24894773960113525], [0.47321370244026184, 0.18095393478870392, 0.40104353427886963, 0.2941465675830841]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c6f323ffac33f8de847925ea46a7253e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dc93c5815c78619d759c2645eef014e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1457086f8118f6ae44c94f8b23f05743
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

class PrimitiveOp_5c3ae42c817ec934892ae13bad46f641(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return input_0

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
class TestPrimitiveOp_adeaa095b592912bbd3a3f841506a5a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c3ae42c817ec934892ae13bad46f641
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

class PrimitiveOp_94acec3b8c48b789729b93f51dc717e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e5982e2aec6679593f25fdd4d8b48319(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94acec3b8c48b789729b93f51dc717e9
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
class TestPrimitiveOp_cc65f589714792b7d937f9e9ec521abe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1457086f8118f6ae44c94f8b23f05743
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
class TestPrimitiveOp_c7d6a2ce7bf0245f9bed82faedd832d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 15, 15], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_69a3489015c06588a98698e2063d474b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5415611ea51c16a8b469e5cb3650d97b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([1843, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1843, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1843, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1843, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_103c1f0030162dfbc30c343714668a20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1457086f8118f6ae44c94f8b23f05743
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

class PrimitiveOp_6a94b94f403500d4ee1dde12643c597d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d20f1964c4888a17821a1a626aea80e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a94b94f403500d4ee1dde12643c597d
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

class PrimitiveOp_093122e485d209c87b4a878cf06c16f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_88631d0e73f108117223f60cdea95a67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_093122e485d209c87b4a878cf06c16f4
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

class PrimitiveOp_b63825d1a4ae907c88ba57f8f5b33c40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_25051e3f9454a43e201351daa822d29c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b63825d1a4ae907c88ba57f8f5b33c40
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
class TestPrimitiveOp_f6b586df17f38fc09fb94a7a25c596a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14373856782913208, 0.42031458020210266, 0.17581358551979065, 0.421566367149353], [0.1028633564710617, 0.3344844579696655, 0.20517560839653015, 0.34165823459625244], [0.1792476326227188, 0.4266873598098755, 0.23294998705387115, 0.011388166807591915], [0.33024778962135315, 0.1721087098121643, 0.1939237117767334, 0.2781818211078644], [0.22236931324005127, 0.27251046895980835, 0.35136815905570984, 0.011399761773645878], [0.3315168023109436, 0.37677645683288574, 0.11405857652425766, 0.16159768402576447], [0.40157678723335266, 0.17235715687274933, 0.19853566586971283, 0.16763707995414734]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f0f87da5d1e5a66cd8e407e42a08944(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0bf89928138f5b24110587c67e8fa6e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f92a555e56ec4598276ce514961795b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9702dd4de525a1a0e32240c2d0e4d5a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 4, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_475eb47fb8dcccbbc31abab26ec8d137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([5583, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5583, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5583, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5583, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fd63e9365b1931d59162ae47dc77e0b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ac133a3eda5ef2ce2a10652b267b9c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14185766875743866, 0.2782001793384552, 0.4330953061580658, 0.4712361693382263], [0.16310042142868042, 0.08349864184856415, 0.08359965682029724, 0.249087393283844], [0.3515278100967407, 0.23490144312381744, 0.1178266704082489, 0.30084607005119324], [0.21413810551166534, 0.4333586096763611, 0.09454689174890518, 0.47111082077026367], [0.003853335976600647, 0.4897143244743347, 0.481177419424057, 0.04118751734495163], [0.12994852662086487, 0.14688752591609955, 0.3679650127887726, 0.046101413667201996]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f6b8896b417264f9f0afe8e219cefe9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_eeb0b9bab044b09395736d2a9f87a00e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 8, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f124bab359a957b381a6b58944d87fde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeb0b9bab044b09395736d2a9f87a00e
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_521b3532a5cf474d80a43a789797cc1d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c160a65581f855c57038ac0613bde1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_521b3532a5cf474d80a43a789797cc1d
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e6c54ca04a6870dc60b0c9268037d803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.058449819684028625, 0.09226198494434357, 0.038015492260456085, 0.4389108717441559], [0.1974448561668396, 0.45112109184265137, 0.4329894781112671, 0.4065325856208801], [0.14642252027988434, 0.20309582352638245, 0.025423023849725723, 0.03931766375899315]], dtype='float32').reshape([3, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_119b12e7f83ba8739a2ea9c81b0d1603(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_24d21c474b6d10b419d2aa8de3ff16f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7fcaf03c4a02d1b0253f5797aedc251f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24d21c474b6d10b419d2aa8de3ff16f1
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
class TestPrimitiveOp_372b53cca9711d495286672dc7f1828b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 44, 44], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_37cf37afe65d5db7a40d6a8556820c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24d21c474b6d10b419d2aa8de3ff16f1
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cac049235f5ed539d2fc14f4aa46d3e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 22, 22], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13d6b5967f8ddd596648a8e80968260c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([1724, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1724, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1724, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1724, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a5853ed03cf16091682292f1bb647ce3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 5, 5], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_de007bd9a6ceecee7266f8c81ecb4d7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e933cadd3d66798db273d8805dd117f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de007bd9a6ceecee7266f8c81ecb4d7c
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_933cf9cd2ca7320aaf16b2861cfbc744(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12616214156150818, 0.0712435245513916, 0.326055109500885, 0.3594553768634796], [0.03403058275580406, 0.0038598976098001003, 0.3403976857662201, 0.3463982343673706]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b2f66766bb20ccfa8fb08852128e9249(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1457086f8118f6ae44c94f8b23f05743
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
class TestPrimitiveOp_21aece329c6b03ac4401f324b7e2a1fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.42137449979782104, 0.1966567188501358, 0.4952227473258972, 0.05661465600132942]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_64a159158ae7661df5ba7e3ad51dc217(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_67bd6782fa2a543cd39bf781531d17db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[80, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fae74374c8409a262a23a3b7cc999dd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67bd6782fa2a543cd39bf781531d17db
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

class PrimitiveOp_15ec87db4e5f15620d1e1a3ba7cb20cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5a1259edcf78f64e65cb4acab5d68ec5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15ec87db4e5f15620d1e1a3ba7cb20cb
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

class PrimitiveOp_0a80ba214ef30b26b24f63ad9db0f30c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[20, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68971d2d93a35cc4a967696a965e7b3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a80ba214ef30b26b24f63ad9db0f30c
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

class PrimitiveOp_475f1705db145005d19fd6cad5a379b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 49, 8, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 49, 8, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 49, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e14024133ec17691379f3028a0523d7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_475f1705db145005d19fd6cad5a379b3
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 49, 8, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f0363af6fdb977e9199522a6ccfaa97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 13, 13], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b2a7fb5cf0c88681876d3b0e6adc45f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

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
class TestPrimitiveOp_c305886d3a6f8da17ef9467f6ff7c6fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2a7fb5cf0c88681876d3b0e6adc45f4
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
class TestPrimitiveOp_6a1b3a1c6e2bc037d94413a40c69bf45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 36, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1dcedba4fb0f6e11120f19dc6dca36ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cc54c9821270174de1a21835190105e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_69ef002098dbedb69e95fad437de154f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1457086f8118f6ae44c94f8b23f05743
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
class TestPrimitiveOp_04c1fd2650dd844f4f17cd0b401bc1da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([1492, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1492, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1492, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1492, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6c0cc4c22ad720106d9d330841ac38a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_320488096ec081cbae80959ae476b2c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3760044276714325, 0.12422966212034225, 0.04910736158490181, 0.13082978129386902], [0.1478123515844345, 0.18232350051403046, 0.4741666913032532, 0.3131527900695801], [0.4784177541732788, 0.1411914825439453, 0.4344243109226227, 0.31855785846710205], [0.14063599705696106, 0.0371808297932148, 0.11269502341747284, 0.12853483855724335], [0.345084547996521, 0.22814075648784637, 0.3940271735191345, 0.28395769000053406], [0.3161979615688324, 0.4849144220352173, 0.43376392126083374, 0.27867332100868225], [0.0065774694085121155, 0.16474632918834686, 0.48897072672843933, 0.450259804725647]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0ecac9b799bbc21367eadff410a126ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2633066773414612, 0.17589786648750305, 0.1740284264087677, 0.023862803354859352]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a8c97900aa90cbd9ed273fb452623471(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

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
class TestPrimitiveOp_f51834900e58625088829918fdebb71c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8c97900aa90cbd9ed273fb452623471
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_71067eebaf355996c10a9d4e28f761ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 196, 8, None], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 196, 8, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36c6ebb7c0a3c180ff8d91f173024381(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71067eebaf355996c10a9d4e28f761ca
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 8, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196, 8, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8757ff1dd8a466f4187c9c52f922399a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return input_0

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
class TestPrimitiveOp_136526e4ae24988a63676be828cf4c52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8757ff1dd8a466f4187c9c52f922399a
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

class PrimitiveOp_8b04bb0e505aa517ac1b97d645951e0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b64fa05dc9ef1967fa01e4f1472f7157(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b04bb0e505aa517ac1b97d645951e0b
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

class PrimitiveOp_514e2b7832361ee7ba5c03c69f8f57fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b5fa6b219c737bfe92a8959930f84ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_514e2b7832361ee7ba5c03c69f8f57fa
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

class PrimitiveOp_21554e657e43f93a58b530ef9ae88388(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3a34f249e62c4bb3fadf15c55c13b01e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21554e657e43f93a58b530ef9ae88388
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

class PrimitiveOp_e6b1002cf5bd85949783ef2c25a960e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7541237d00e1f42ae5579df35ad46461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b1002cf5bd85949783ef2c25a960e4
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_16f467c540e90b8d1b2b841df1e46fde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 32, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_39ee043e484a047c14a1fcfdfdda05a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3426182270050049, 0.371687114238739, 0.17967729270458221, 0.05511809512972832], [0.028130333870649338, 0.08196408301591873, 0.17705516517162323, 0.4715835154056549], [0.27289482951164246, 0.43746933341026306, 0.3031827509403229, 0.16167962551116943], [0.20820161700248718, 0.023066138848662376, 0.3269689381122589, 0.34487488865852356], [0.254242867231369, 0.17583134770393372, 0.4541226923465729, 0.3056456446647644]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14f220a1a5b248767cb997c1039ebc74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06615816801786423, 0.17510589957237244, 0.12482108920812607, 0.3349984884262085], [0.09679903835058212, 0.35872501134872437, 0.2356889545917511, 0.3929291069507599], [0.4208967387676239, 0.013075667433440685, 0.12194177508354187, 0.09354513138532639], [0.16531191766262054, 0.45159438252449036, 0.43057894706726074, 0.33081603050231934], [0.29036158323287964, 0.4901483356952667, 0.10050932317972183, 0.4676995575428009], [0.3282496929168701, 0.01271427795290947, 0.07691396772861481, 0.007982900366187096], [0.3301147222518921, 0.16804207861423492, 0.19386331737041473, 0.050972338765859604]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0254ce6c498d67bf3e5453c2c9f9d474(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4841764569282532, 0.4656074643135071, 0.32657986879348755, 0.29411062598228455], [0.1873297095298767, 0.25933483242988586, 0.2513105869293213, 0.16280633211135864], [0.17818327248096466, 0.274533212184906, 0.3206072151660919, 0.4815177321434021], [0.3417300879955292, 0.23737823963165283, 0.24936912953853607, 0.44680628180503845], [0.2608222961425781, 0.1117267906665802, 0.1882377415895462, 0.0022811219096183777], [0.11898007988929749, 0.39454466104507446, 0.33293068408966064, 0.1850283145904541], [0.3458210229873657, 0.4326430857181549, 0.10277026146650314, 0.4839790463447571]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2f0cb283597fffeb5d2ffb37286cedd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 88, 88], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_79a210a1d5aec6ef7c14d23b9177bf78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07233921438455582, 0.2189362794160843, 0.08887238800525665, 0.32634830474853516]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8681c1a1016d2414d6f73dfd91b7dc98(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7877e551beef61234d148b30b57aefa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8681c1a1016d2414d6f73dfd91b7dc98
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f6c02d115d998e694a638a42fc1aaba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([2079, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2079, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2079, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2079, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5b94c1e4afa16f5d606b25312e3a9502(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return input_0

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
class TestPrimitiveOp_ff089a0cf0b96f549c98387124297839(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b94c1e4afa16f5d606b25312e3a9502
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

class PrimitiveOp_84833a8c576c88ad654afd914b4c47a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
        return input_0

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
class TestPrimitiveOp_2edb5833cb02b40945ae785e965f66e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84833a8c576c88ad654afd914b4c47a9
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38e72f38a18c40ba89b873f15f16ee54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([4526, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4526, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4526, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4526, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bf825eb8319922fafc68039371329d21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 600, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 600, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_74bf508f92c72f41168fb4bfb8cc4118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf825eb8319922fafc68039371329d21
    def get_inputs(self):
        return [
            paddle.uniform([22, 600, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 600, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_993ea0795b4701d217cf3aa11e86f97d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12845773994922638, 0.16413450241088867, 0.3531971573829651, 0.39715954661369324], [0.3552411198616028, 0.1575901061296463, 0.3620745837688446, 0.0040739066898822784], [0.3909343183040619, 0.4800011217594147, 0.08939668536186218, 0.31853631138801575], [0.371174693107605, 0.048930808901786804, 0.41185128688812256, 0.37662333250045776], [0.45312097668647766, 0.09228237718343735, 0.039329834282398224, 0.2555757462978363]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd4a4abf764e5ebc7e5bcfa71bbf35c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([1046, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1046, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1046, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1046, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b693a8b44c9655702db6377742531e94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f9a9c6107356ebb24be6f20dab5ae6cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1457086f8118f6ae44c94f8b23f05743
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
class TestPrimitiveOp_9ac88113545a0a3ec925f82b1167737d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b5d5b2483abbdf6f0c72379f92925368(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 192, 288], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e061998d06205e4f99bd1202bd6bb114(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de007bd9a6ceecee7266f8c81ecb4d7c
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f8e2dc07edfa99a53f8b4224f798d8ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[68, 68], dtype='float32'),
            paddle.static.InputSpec(shape=[68, 68], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3dde0d1425551a91ebae0bfc1a0d75bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8e2dc07edfa99a53f8b4224f798d8ab
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

class PrimitiveOp_ccead173ab5dc2a7a8b05130fcaeff4f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[34, 34], dtype='float32'),
            paddle.static.InputSpec(shape=[34, 34], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_86690e4a23c10710502f19ad4e2f19e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccead173ab5dc2a7a8b05130fcaeff4f
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

class PrimitiveOp_1febda3a454828d94ffc748cf89b0a08(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[17, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef330fbb9b0bd025328a7ebcf9019ece(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1febda3a454828d94ffc748cf89b0a08
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
class TestPrimitiveOp_376821aa5dbc46bd375f122fe71447eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12460564821958542, 0.22093810141086578, 0.22483277320861816, 0.3253820240497589]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ef8c2878092d3cbb853a14c2d58ae1d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 36, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 36, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d682f7195787490ff249f1494355d0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef8c2878092d3cbb853a14c2d58ae1d9
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
class TestPrimitiveOp_7140e540c850cc1ae5872af22c014961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1457086f8118f6ae44c94f8b23f05743
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

class PrimitiveOp_60649a0a13347a8bfb315367c178d719(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e4f42f414a63e386f3d1dda92eda8838(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60649a0a13347a8bfb315367c178d719
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ca4bf23f83d81476d78fe7e486bd318b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22656188905239105, 0.04892861843109131, 0.18918175995349884, 0.1263914257287979]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_24de5209ef1769607e77367b7529f83f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([2335, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2335, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2335, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2335, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a46a4f8d2384f759945daff456cefcbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([2986, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2986, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2986, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2986, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e3765eb371e4cb536e8d826f8c461264(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([3783, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3783, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3783, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3783, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_50b11169d450c2e8042c278fb9a30ee4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7638bc76aa3f42d0111457b599af47a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 16, 16], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f6e7168907dce2d12d67d4fbce67080e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f773928d2200a734db0124a0490e22aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1457086f8118f6ae44c94f8b23f05743
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

class PrimitiveOp_f3be89c0f277058286c2fbf26294dd39(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 60, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e903aa8ad1302fdfacbf8ac819f259d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3be89c0f277058286c2fbf26294dd39
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
class TestPrimitiveOp_4300d7297bfb51346fc8301b61699730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60649a0a13347a8bfb315367c178d719
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_49fb5a5c53121cbbb077d9fc04b4a1f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 8, 8], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a083461fb796b4c517d430f84e62c368(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a725b002c841ebd7195bbdc91c89a0c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a083461fb796b4c517d430f84e62c368
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 120, 200], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a960e4682b5a2a5cd89abd5a92f7ec0c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68a484dfb118a85792a5944760044614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a960e4682b5a2a5cd89abd5a92f7ec0c
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

class PrimitiveOp_b174ab41add4daa07f58a0cded01a098(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f69fe0dca67c823514b6686719aad70d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b174ab41add4daa07f58a0cded01a098
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

class PrimitiveOp_2c0afdcbc127a986d0493e5e53870e69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ad4ee6922c92d223e5a85d91b92d75ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c0afdcbc127a986d0493e5e53870e69
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

class PrimitiveOp_55304ee151add2b9bba22e1d623bdd68(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9f06df5683fd3fbc3918ebd050c18780(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55304ee151add2b9bba22e1d623bdd68
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

class PrimitiveOp_117b0b37907192d9279457a2b34eae69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d672b9189d715a0db51923590733cf4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_117b0b37907192d9279457a2b34eae69
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

class PrimitiveOp_c5ae14ceab59c880c270d07936a63665(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
        return input_0

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
class TestPrimitiveOp_d63d90da3d59f84d4bfcae533bc0add3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5ae14ceab59c880c270d07936a63665
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
class TestPrimitiveOp_d2d9fb0efc0941188e0fa11827d94709(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_eaa0b3ce2a1281c6e4185372b36da52e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 20, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 20, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_019177d611734a910ae150bc64680359(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eaa0b3ce2a1281c6e4185372b36da52e
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
class TestPrimitiveOp_1ff139d5e53b954fa27f5eaa97d74e14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1a77cb9f3adb8f9d9229879f5d13bfc1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e85f8c49fcfc1674f75cea147fa8b7a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a77cb9f3adb8f9d9229879f5d13bfc1
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
class TestPrimitiveOp_7b863a86ec53ab56f01ee47f912f9365(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8681c1a1016d2414d6f73dfd91b7dc98
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c9b45179d377f41c8513985e207940e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 180, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 180, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1c442672e4d473cb753011571d4b82c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9b45179d377f41c8513985e207940e1
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
class TestPrimitiveOp_0a48a52c405fa6653ef50e8770d92a29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.353889137506485, 0.04948345944285393, 0.21553945541381836, 0.37403661012649536]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68594b557b2db00ca3dd682d3e60e2e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2c7087483d6a7edb614a72ef92dfc338(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b1002cf5bd85949783ef2c25a960e4
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_37ce8b3b946ed32fa01cf8a02c1ce028(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([2041, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2041, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2041, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2041, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d4f109297c9f109d6ad3d8aeec19c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1457086f8118f6ae44c94f8b23f05743
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

class PrimitiveOp_ac91c718cafc4d70d922717cfe40d71f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 196, 4, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 196, 4, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 196, 4, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0f03d9df43b037b3cb01aeda9ca8e4f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac91c718cafc4d70d922717cfe40d71f
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 4, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196, 4, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196, 4, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9a9a2119522180997a70051854b22786(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[72, 72], dtype='float32'),
            paddle.static.InputSpec(shape=[72, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_62fccef12a052df23954020672006d85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a9a2119522180997a70051854b22786
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

class PrimitiveOp_ea4e995872996f08c2e0ddf127ce8f8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[36, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a7f813f2defd56a4ec6ed0e7bd1eba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4e995872996f08c2e0ddf127ce8f8d
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

class PrimitiveOp_dcefb85597ca23d8362580812deb084a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[18, 18], dtype='float32'),
            paddle.static.InputSpec(shape=[18, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c3ed92d5802b6502c303f20d596a8a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcefb85597ca23d8362580812deb084a
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

class PrimitiveOp_50bc6f8d04b6f3f17ca313dd335cf748(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1a4dd60c6b56635d4c9571f0f8abad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50bc6f8d04b6f3f17ca313dd335cf748
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

class PrimitiveOp_138f0ee51b3230e45d32ed40b249ddc0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 12, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 16, 12, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 16, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9003d27dd0c23eb731b000709f87b3f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_138f0ee51b3230e45d32ed40b249ddc0
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 12, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 16, 12, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 16, 12, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4440d5093b18b5508df98b435bc89b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 18, 18], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_419eabf62d96aab28e5ef4fbde7c66c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 14, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_599c1fdebbd10a68e3db9872c334fee3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([4210, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4210, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4210, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4210, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9473ca49bccc6b8354b6bb2a87271762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12450958043336868, 0.052685387432575226, 0.3968353867530823, 0.15485253930091858], [0.34951451420783997, 0.32297182083129883, 0.38014039397239685, 0.47485774755477905], [0.2576924264431, 0.4425520598888397, 0.14566507935523987, 0.4091264307498932], [0.19692909717559814, 0.3446423411369324, 0.3547402620315552, 0.12531565129756927], [0.4130346477031708, 0.48923182487487793, 0.30839577317237854, 0.05351301282644272], [0.2585386633872986, 0.3128417432308197, 0.34252235293388367, 0.39456647634506226], [0.3814266622066498, 0.10656841099262238, 0.28748100996017456, 0.2259080708026886]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f6120a79a2c08f7e28137f51058e2b24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.13486644625663757, 0.05036383494734764, 0.17426297068595886, 0.035235773772001266], [0.45595043897628784, 0.45600008964538574, 0.41477057337760925, 0.39209648966789246], [0.254535436630249, 0.358079731464386, 0.4156394898891449, 0.2574434280395508], [0.13503143191337585, 0.07170978933572769, 0.31542617082595825, 0.4614073634147644], [0.17103426158428192, 0.2569636106491089, 0.24876902997493744, 0.303025484085083], [0.4615570306777954, 0.3139835000038147, 0.48396337032318115, 0.15067026019096375]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b389237df845b14ae35a31353be929d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a083461fb796b4c517d430f84e62c368
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_df0a61246424ee1fb2573328ee1fa968(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 49, 16, None], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 49, 16, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c4d58ff896ab581cd9c89a9a50f42b0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df0a61246424ee1fb2573328ee1fa968
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 49, 16, 64], dtype='float32', min=0, max=0.5),
        ]


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