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
class PrimitiveOp_8296ddf5dde359ab3b0063eb7fd007ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bd026fa34fef33f4b28dc758807641c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8296ddf5dde359ab3b0063eb7fd007ab
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_718ac333040c9e7448c99e6c6ff82432(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.5'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9a04d0a3a99b7d4743b2e52f6910f335(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_718ac333040c9e7448c99e6c6ff82432
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6c125219b83095616d325578e2a0ddb4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d0a7f56f2a2aa28040ae33aa31b8237d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c125219b83095616d325578e2a0ddb4
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_228122174c598f8dc815ca4e7538c6a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('13'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d50564d01b29e19eb653d9c97cd99ef2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228122174c598f8dc815ca4e7538c6a7
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4f74e394d53bc8e5dd51dd888fc7911c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('80'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2071b583d72ece297e30a7f79b6280b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f74e394d53bc8e5dd51dd888fc7911c
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4e43ddbe204e78347ce9fd2eff44de0d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2dbff49f36eff568bb193d29140f42c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e43ddbe204e78347ce9fd2eff44de0d
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9fe6a97c79e09223b8c3947b8ca61459(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('-1'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_61a77784dc7c853c4738da7a23243258(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fe6a97c79e09223b8c3947b8ca61459
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ffe004a7015f6802b1d404d16e96f13e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('16'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8cc107c2c0b624b0fbf129f801b4d857(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffe004a7015f6802b1d404d16e96f13e
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6132065deb1757bfba0d50f60217bb12(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('36'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b82c0d9b32c6de662b9e893f7f44192(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6132065deb1757bfba0d50f60217bb12
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f6464be8f7d5497a6abf9a7b4a15b496(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('1'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ba1dfca264be74526f70e5ce0adbedd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6464be8f7d5497a6abf9a7b4a15b496
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3b53bc55b2cc7750fa20479dc13dc2d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('512'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a5d26ea89c529d657d65adf1780370c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b53bc55b2cc7750fa20479dc13dc2d1
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8566514a1b08a7da77b101ca95aa72ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('128'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eec9a5c3a6684440a8f8cc26c33da81f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8566514a1b08a7da77b101ca95aa72ab
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b2180adf119fe4519e8057e9578fa966(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5442030f43afeecf7272f60764aadcb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2180adf119fe4519e8057e9578fa966
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5f489a747ae50e35db1a3a8581c6fe60(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('17'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9164bb0aab3230f5a7d10ee7feb039bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f489a747ae50e35db1a3a8581c6fe60
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d03c57ca0930a9cacae86c7ad99457b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('3'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_536565d1ac2d6a5063422beff8f7a570(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d03c57ca0930a9cacae86c7ad99457b4
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6a847d486b3f88267393cc2d0b94a5a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([247, 1], float('64'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7c4fa35c3fb600d5610ae60d268e6ccf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a847d486b3f88267393cc2d0b94a5a8
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d1dbc18f365c04b051e2dffddda0728e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('2'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7e6f4555e7eafedf9b0505cbf68d717c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1dbc18f365c04b051e2dffddda0728e
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f10c82f931fa2cd4fa491138e0b4d87e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('4'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c980e05eda3eae447ac78de8d85cd23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f10c82f931fa2cd4fa491138e0b4d87e
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ad78a44ee580f12defb47a6eb86ed765(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('72'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fad6b7c930182e290bf3c14d49d5ac16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad78a44ee580f12defb47a6eb86ed765
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8626c5435cce0c299f48bddba1d64960(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65e4791a8cedf02bd70c1b16119c3cf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8626c5435cce0c299f48bddba1d64960
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0c878e0effc6dfc75a6260fee1dc8ef7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4fe8cf774fae41cbfbfe8f3e73341a3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c878e0effc6dfc75a6260fee1dc8ef7
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f8ef53fe51575205bda06dfc41777322(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f253a0b3456d44b837ab1a0b75f86703(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8ef53fe51575205bda06dfc41777322
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c8e4e571722040817bb90b08d6cddd48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('192'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3b2c2f6ed032a2bb259c4e730274562f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e4e571722040817bb90b08d6cddd48
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d76bec2c81784d529441321aae3b6d60(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('64'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bbd909be7145b7c11c17f5e3c6eadfbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76bec2c81784d529441321aae3b6d60
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f3027bfbad0c7b7e251d41aeb91ba577(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([2304, 1], float('16'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_84fea42d92d9b74548ed7c70f1004e4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3027bfbad0c7b7e251d41aeb91ba577
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3971982df61108c11063398afaf9bb48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c301654a231d909a9f30a41c2a18a130(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3971982df61108c11063398afaf9bb48
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9668b53de7c6e6a3f6d0cd654763063e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([551, 80], float('0'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e9eb66a34718ea0ea16e04f4ba7b7bd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9668b53de7c6e6a3f6d0cd654763063e
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ab8482e590eb65cbfb60c1497a095f2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('0'), paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_05ca7287536affa00e97ac5a4a159b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab8482e590eb65cbfb60c1497a095f2c
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0f25eb0ac92f607cb15b94c24c429b02(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('0'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ed4238c89f97c34b8e83fa14e7a320eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f25eb0ac92f607cb15b94c24c429b02
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4ce1514d91b0bcfed3f1a9bd8f5a85be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('15.99'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea134bdd9a9cfeb78f5a0f51ffb29fb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ce1514d91b0bcfed3f1a9bd8f5a85be
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b87aecc173c35dcc0b9d5125a10bf345(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_44af5907ca247dbd4503cbf0df47ff81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b87aecc173c35dcc0b9d5125a10bf345
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5696f7fd621aa8af98ec6a6ececcd301(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('1'), paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_27e83fc040ff8225b39754bb722e60a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5696f7fd621aa8af98ec6a6ececcd301
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_62d945b416212f4816a842348e8c0ef3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6aee4f5f711d7f4c9644ff82cc0d7d32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62d945b416212f4816a842348e8c0ef3
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_76dc498077f1ea6728541e3eb5bdb3f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('2'), paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4803c65bdffb386577024dcc38a8e9c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76dc498077f1ea6728541e3eb5bdb3f9
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_10ee9f2031d8a930979e83c88b981edf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('256'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0407d8e1deaa95cb3b01eb72fecdbf8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10ee9f2031d8a930979e83c88b981edf
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6ea5f97066dcc32367c854ce67f80a3e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('56'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_102a17b8cf71b48b29616cf677aa844f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ea5f97066dcc32367c854ce67f80a3e
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0e890d521eea1e876d1fcd132612387f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8c328067dec217f4b99fb448709bf6bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e890d521eea1e876d1fcd132612387f
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1840421833a761eb81d538547e0da5f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('116'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0ed753ac62ee651dc0976be88af6e56b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1840421833a761eb81d538547e0da5f2
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2b089b4350b9b3a50ee132aa96ea19b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.0833333'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_720b659c5e231ffec3b150690749ecd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b089b4350b9b3a50ee132aa96ea19b4
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_57cf6ec532c708114ca7ecc221964f13(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('3.40282e+38'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_adbed5bceb68a397e4235cc4d8d90133(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57cf6ec532c708114ca7ecc221964f13
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_12ecf54f3c014ef517fa4fdd91e8e3c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('-1'), paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2fa0fb11eb8ebc1ecf8cdb722aa71fee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12ecf54f3c014ef517fa4fdd91e8e3c7
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2a92569727b26e68b241e0c02068997d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([70, 81], float('0'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7137a223d345c268597526d59e098b18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a92569727b26e68b241e0c02068997d
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_382ad4450f3c4636098998a7c5977299(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('80'), paddle.int64, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7cd754d152c6f893f76d89ac4f80ccab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382ad4450f3c4636098998a7c5977299
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_21c397a0ca774e304960f232e5c4607a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('32'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ab3f8cb8299d08d17db90a584ccc860f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21c397a0ca774e304960f232e5c4607a
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5e4d4a691f14b6203ee37a16f73b277f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_099fc2e6128aa7486fb1dca850781fa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e4d4a691f14b6203ee37a16f73b277f
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_abdb1292f59a37a8d9c030b0ae32c7ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1.11111'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a30db4488b7689d2d752bc83db22d6b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abdb1292f59a37a8d9c030b0ae32c7ee
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_08efd8d9ded18c8bde6582aaf6a122bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('12'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fb0da155a8fdce16d64f326592eafe6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08efd8d9ded18c8bde6582aaf6a122bb
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7e10cc7542b9d8cdfe4635f3f4bd5d5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('768'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ff222f791aa2581bece1caa3db776109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e10cc7542b9d8cdfe4635f3f4bd5d5b
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f25df6d580b5b740f3f0a6dcf6c11df0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('32'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_64e2c1e4a1592e9ebfe65eefa8c47bad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f25df6d580b5b740f3f0a6dcf6c11df0
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_126c2e3c0f356507e0e43e929cef43d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('28'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a2a68ea0bbae38d53fbd3cd2ab1000d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_126c2e3c0f356507e0e43e929cef43d9
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bdfd35aea60a6ef984bf174f98af0603(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.2'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f3c0034b3df593440dec2b35871f4247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdfd35aea60a6ef984bf174f98af0603
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_dd8bc863a28537f66740649de2020e8f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('24'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd5babd971a0bfe2cb1ae0ba43736e10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd8bc863a28537f66740649de2020e8f
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e99841e88a2f66e92972dbc49d5b4b45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1e-10'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a18efe34a8e76219dc9361530e348657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e99841e88a2f66e92972dbc49d5b4b45
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b8b6f65bbd5eb011c6fe70dfde2f19f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1.09589'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b1574e7d0b537f8f27977f800652f49f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8b6f65bbd5eb011c6fe70dfde2f19f6
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_08e68d4e5a19b48a5bc455123814b7c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_49c7733d1ae8e7b3c415496bfa9ce1db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08e68d4e5a19b48a5bc455123814b7c3
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5b3b02fd5ff9c7c6874f4e9c91d7eeb4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('48'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ed4713cc5e8ee428eb37af67cbdeb76c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b3b02fd5ff9c7c6874f4e9c91d7eeb4
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2e5c3aaa1d50364a730db08b40b71dac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('-1'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f5e90146cdee7be4c1a51322964a75c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e5c3aaa1d50364a730db08b40b71dac
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4d9f3db78b7fb0210a3e3618d3e2e8a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('8'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dc8b68f3e3cb364274cb072af35331cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d9f3db78b7fb0210a3e3618d3e2e8a6
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_70aaf8a819834ebe2b7bd0a9d3220207(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0db907d8e59d9b66b34ee45e9a6c1c7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70aaf8a819834ebe2b7bd0a9d3220207
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f998502ca4fa7e9f98ae21f2e1be5e07(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('25'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a09ede017bd9a01cc18847c2bdde579(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f998502ca4fa7e9f98ae21f2e1be5e07
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6f8fc4e6dfed0d05100a9b164b035c48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('25'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8ef1f5dd0c1b814f117d05d2fa26a952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f8fc4e6dfed0d05100a9b164b035c48
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_544ca9c631d41e43f73b48cabafd10cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e466d863ef0141f5379e1abb337c5a91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_544ca9c631d41e43f73b48cabafd10cc
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9a3950e68120eec1eea8e83b280486ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('8'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e9ca3d457491c2d79219d26c659bd111(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a3950e68120eec1eea8e83b280486ae
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7697515ac7b448ef44e06b55a2388018(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('24'), paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a09001c56147fd892a0f640c5479d1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7697515ac7b448ef44e06b55a2388018
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c902553683ad43d31d5f48c1cd3ab5e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('38'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5da8a37f307e82440f601180ca4a9319(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c902553683ad43d31d5f48c1cd3ab5e1
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e3c7ee2534698876ca07abbf9a55bf16(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('96'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8c7f5584237287e688b19fbb12770626(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3c7ee2534698876ca07abbf9a55bf16
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_54f769a3e5d2f8c7256b927573bd8c32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('384'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_80002db6859f8402da771c1750962487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54f769a3e5d2f8c7256b927573bd8c32
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_236bdff9a34cbb9f2a80566ae579e949(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('-50'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_43f5ea2b7a7e5b40e4cb9ae1e8eda110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_236bdff9a34cbb9f2a80566ae579e949
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_263c7a9132fb61ee4588cd758ce56c8b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('81'), paddle.int64, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4fb6cb019b6c356b7b0a777dbea395a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_263c7a9132fb61ee4588cd758ce56c8b
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_14b9fc91bd13088e3163a18363fbf5b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.05'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2bc592dc1cade157c84243329b8b6214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14b9fc91bd13088e3163a18363fbf5b8
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_da1557849b559f4321e9cbe3eb4151c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('14'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ae2514a9a61e6119f5803bbbdd86dc2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1557849b559f4321e9cbe3eb4151c0
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_baf713c33e8210b6da24fc0698ae0d37(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.75'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_812a0aa828f001dcf2812e1ac52924ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_baf713c33e8210b6da24fc0698ae0d37
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c308d756480f793cb8c21638868c1ed9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('160'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b28cc03b759b9edb0c23ffedc37160c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c308d756480f793cb8c21638868c1ed9
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_01c9548cb85f26211be7c499ffe95e2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('14'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6dfd2cb48dc332729872163d2eeedc9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01c9548cb85f26211be7c499ffe95e2b
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_232989b342366aa6d60b6c120fbad134(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('19'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cfebeeb64c6fd8e1783e556551688db0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_232989b342366aa6d60b6c120fbad134
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_80348b491f31f9d1fded5bdc2b85b4df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('38'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2af26ff6585bd99f34f53eb98f0eabca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80348b491f31f9d1fded5bdc2b85b4df
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f12b754b8eb8775307745116fcff666b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.25'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d583d963763e959838143ac20ac6b818(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f12b754b8eb8775307745116fcff666b
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cff94d5339c086731327f19f26ff3ef8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('300'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_01978a3d6121631fd5cd1810447a7090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cff94d5339c086731327f19f26ff3ef8
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_acba41949fe95fd49af3c50559a0d95e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('128'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f0c55c121db6ffb391b4f1da6cbf200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acba41949fe95fd49af3c50559a0d95e
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_893e8eff1be5c1cbc7a0c3608f552c95(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('16'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a4a1fc0a0d0607af63118d30586c071(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_893e8eff1be5c1cbc7a0c3608f552c95
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7e4a6ade9bf8f6f74335d6e89b7c7220(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('9'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_629a553aad4cd4719db53c9ce96f7278(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e4a6ade9bf8f6f74335d6e89b7c7220
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7f23e0bf9e2be05c0eb07593a386cbb3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 501, 30], float('0'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b6e0b1a3e7d4537e2d1e8fb63e4aa938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f23e0bf9e2be05c0eb07593a386cbb3
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0653449704799d66e7d72a6911006956(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('4.13517'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_338eff07d7840abd48009387e7978a13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0653449704799d66e7d72a6911006956
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_45ebefb234e37e36882d863814bd7d5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('100'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_28bd5b5a95fa5287cb20fb39a773c3ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45ebefb234e37e36882d863814bd7d5c
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9d9654cf09e0d1e7abf614ea3a8e2e3f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1.02564'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_84a5a2bb50e4d178e87b68da38b46e67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d9654cf09e0d1e7abf614ea3a8e2e3f
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_666c16e786577544e968bfd2b753680e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('2.5'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_899200d31a4df881b311d928eb3564b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_666c16e786577544e968bfd2b753680e
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_086ceb7d833e00a498753c9de0203335(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6c943b31cf878d8afdcf12b7994a7724(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_086ceb7d833e00a498753c9de0203335
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_de8dca561c48c8631e926f56b0f05dd4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1.17647'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5821f8f68179d91d7702b5c59ef5a658(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de8dca561c48c8631e926f56b0f05dd4
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cd33f9277e655e8c0fe394b0de5266de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('19'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e7c2c0e5af968361c898221517955d6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd33f9277e655e8c0fe394b0de5266de
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_33fa0e0f9fe033a5eca850cb8f509651(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('40'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2871e79353ea98449e63a4427563515c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33fa0e0f9fe033a5eca850cb8f509651
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7239d44be66d08bbd09110405c2d1585(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([8816, 80], float('0'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d15b5c1c9f76bcbacc93fea541864ee0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7239d44be66d08bbd09110405c2d1585
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_349cc85ef3dd595d3c5402ed39ea43f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('10'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e16359747ef68ea276bf79e406af812(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349cc85ef3dd595d3c5402ed39ea43f4
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c79cd8fb11b0d920cb87c8a62e7e8aa4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([950, 81], float('0'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3e42b4d345700c046d3bc7b3123f4df1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c79cd8fb11b0d920cb87c8a62e7e8aa4
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_dd0540064fde0301494094e7019eae80(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([2204, 80], float('0'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d47a0ae39d6f6062051db478f4c06621(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd0540064fde0301494094e7019eae80
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bb5b20489d8739dd13374bc6a4b43a8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('7'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_31ce29282f751962fa0c83f57f84012a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb5b20489d8739dd13374bc6a4b43a8d
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bbeee861b35b2362a72c85232edbd5b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1.0101'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9537c176d5ed8cd85a673f6742894edf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbeee861b35b2362a72c85232edbd5b6
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e2ff18306aca375b12243899b8484d86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f229e57fcad0da687c64fb041360410a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2ff18306aca375b12243899b8484d86
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a5aa80f2b8c952a22b2ed3d0a730b1d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('144'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb4f1adeea27e0bc57000f5b2d07e125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5aa80f2b8c952a22b2ed3d0a730b1d3
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c1f4c1b7e5cabe902be9fb6a4ae6979c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1.08108'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bfcc876ab986f1b281dde926ddf2ecbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1f4c1b7e5cabe902be9fb6a4ae6979c
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_872bf74daa5627b3732a02219e2d61b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.00390625'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b0f9e42e1be207ab4636731417605b20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_872bf74daa5627b3732a02219e2d61b7
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8b007dc37c0eece03188408c2fb3b8d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('81'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_041ce97998c1e268c2c8d55f6a709c4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b007dc37c0eece03188408c2fb3b8d7
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1b88f619a5840e6ba67d5c7ad8caa058(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('68'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47ed5310e39d4d5bd6b78f79767de3f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b88f619a5840e6ba67d5c7ad8caa058
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_76b82327400f7822e44141b3dadb9b08(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a11411d50ce9d2581590c34068b4a25a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76b82327400f7822e44141b3dadb9b08
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e1f599441096ffd76cadb75cd71b4041(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1.14286'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_426cb0bc3e19ec20c2a9ae7ebaa2b40d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1f599441096ffd76cadb75cd71b4041
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bfcdf000d5f8c33e2d58a366cb60212a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('80'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6cf052959b51d371920b95c5069d7bdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfcdf000d5f8c33e2d58a366cb60212a
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_01d742139288b82103836fdfa634fa48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('0.111111'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5bdf65850b0e2d16b530df5cfe25e324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01d742139288b82103836fdfa634fa48
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a5e4fd46dfed526e220cd0e8689bb5dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('76'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1eeb9f1d4c87c4e5d99ac9f8f06848e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e4fd46dfed526e220cd0e8689bb5dc
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f598ff7481851ce779e7051a5a211fdc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('30'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_49c0aace9464f94a3ed98a78defedf4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f598ff7481851ce779e7051a5a211fdc
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7af8eaa2b61f7d86a6082406f0e50b84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('64'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_24f5d5c475a988f8ce352f668ff4919a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7af8eaa2b61f7d86a6082406f0e50b84
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_19c7dc9e9fda21e9d9ef0916b5030337(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('1280'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_80f8e9c8d7ba9327f31f88549fb54de9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19c7dc9e9fda21e9d9ef0916b5030337
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cff6c6f2448fb5a49ac7d2a8b5537a33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9479def229ca657aca8908758d390942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cff6c6f2448fb5a49ac7d2a8b5537a33
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bfbffbb5be3676433c0d34dae3560ea0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('320'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bc440e2156f954376c8d75ea83397c39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfbffbb5be3676433c0d34dae3560ea0
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_07575e2561be91871e95263f94f85724(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([4096, 1], float('8'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bb6e54776d9b72ea4c8c1f4ab32dbc40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07575e2561be91871e95263f94f85724
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6a8763cfb3502d0d0d5de65d216b7464(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('2'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b396f0697f42e1d34d7f3bc94c682ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a8763cfb3502d0d0d5de65d216b7464
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f5afa4c9eafd96028edd5f9f9ecea68b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([256, 1], float('32'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bb43828dee602b21cf8ae8681720af52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5afa4c9eafd96028edd5f9f9ecea68b
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b23a57ed6dbbeb327d8a5d0f4ec863be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([950, 80], float('0'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7d52c6a6d6cdbbde72556fc26d8a93e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b23a57ed6dbbeb327d8a5d0f4ec863be
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f615965278dc9af6d2e25ac3666d2044(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([3800, 81], float('0'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bad93338734e46c3a299b8d224596d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f615965278dc9af6d2e25ac3666d2044
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_daeb066e5aef4d6b00feca471823b82b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([950, 1], float('32'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3ad72a48990a7d402bf89f800d7e9fdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_daeb066e5aef4d6b00feca471823b82b
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ba129f523ab6f051ea9af8c3a6e5d37c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('1'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f2a0aa025188c62a32d0c0fe552c23d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba129f523ab6f051ea9af8c3a6e5d37c
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4523236fd15af779b77c7ce8da89f4a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([289, 1], float('32'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_316089fe012c1af9805df7e2cb996b16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4523236fd15af779b77c7ce8da89f4a2
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_678bd77b9191eb9d0d031c923a142f94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f4106f0448a34d1c7d51a394aeb7a8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_678bd77b9191eb9d0d031c923a142f94
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_62fb0266ddd57ed4b4f8e3f2cdc8be2e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.01'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ec594cc9a37e6fda15c38c7d306dcaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62fb0266ddd57ed4b4f8e3f2cdc8be2e
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e8d230e339404e37264ccaadd88b0a9d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([5184, 1], float('8'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9f430cf5ff6c4fe881f2841a281fb2e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8d230e339404e37264ccaadd88b0a9d
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3d9dd2a8a5dadd8e9bdbf6308f0e29be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('6'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e39cbbb6b73482281dcb05d28d3d5efe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d9dd2a8a5dadd8e9bdbf6308f0e29be
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2e03375bef6a268f8e0fdca1487a0d84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([2], float('0'), paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7cdf41c42123a9dad23defb4265e3192(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e03375bef6a268f8e0fdca1487a0d84
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5c0ea9eff8b032afdc2f80f82bf6c8db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([247, 80], float('0'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7bac907dc7ff87addc69bb88dc7830d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c0ea9eff8b032afdc2f80f82bf6c8db
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9b0f3bdf14be1f1d3b907a97f9cbd908(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('-6'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2bfe370aa044ba77f06fbeece7aabf1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b0f3bdf14be1f1d3b907a97f9cbd908
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7907ce5e4a7cb737f51ab8b8cc33a7eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('36'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c0dc413a9dd01f2f675b6f485ea50e57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7907ce5e4a7cb737f51ab8b8cc33a7eb
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_15f41f7951663f4c309e69c8378602ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('38'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7011e8e4713d4f2d86de0581566119f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15f41f7951663f4c309e69c8378602ab
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_40e1f243fd849b0510f5e22af806838a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('20'), paddle.int64, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a20985736ffbc9d53cde82911185242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40e1f243fd849b0510f5e22af806838a
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b3060a38c18bc81bbbc5155172cd068f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([196, 1], float('32'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8dc22e320b0598148e06cdfceb639803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3060a38c18bc81bbbc5155172cd068f
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f1080f78bb85e7054f618b0f01a934ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([15200, 1], float('8'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf00607ff535d5a72a0f7d2bdce12069(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1080f78bb85e7054f618b0f01a934ca
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_46c7dcda056769669f559378c1221930(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([9216, 1], float('8'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b77cc84c3a8cd761e38fad714ac869fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46c7dcda056769669f559378c1221930
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7f04ded8501f83a85be132206c0b5770(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('-2'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3bd9d1239af9903bfc8184be46718b18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f04ded8501f83a85be132206c0b5770
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e823b3b1b17822517590b3d9608923b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('30'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_78b93ca59997a880a1372bff4e7ab802(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e823b3b1b17822517590b3d9608923b3
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ab3d351ca3f8e03a4037d580f941c771(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('232'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2db6a2d87a715fd9228f3764e3632b6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab3d351ca3f8e03a4037d580f941c771
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c1ebcd949015d28d98346486debf817a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1600, 1], float('16'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3aa801fea5e69101675d0a8233e0791e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1ebcd949015d28d98346486debf817a
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f0fbab453d3c01d1e1726ce2ebe30f88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1.19403'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_090d5e07633e7ba122585b0d7688cc39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0fbab453d3c01d1e1726ce2ebe30f88
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_04d47c7fd1431833d2620d954911d709(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([40, 80], float('0'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2879f09f438c31f2b1629ba20df5d3d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04d47c7fd1431833d2620d954911d709
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cf2b2acc1180db637bb17ae20c934226(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('150'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_06e16efa28c634a3794b72b0f38058b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf2b2acc1180db637bb17ae20c934226
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a4721f88d4163ab39f4eea3229dd2bdf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('18'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_76f394d29b4c3a9e2fc105453f7e8c46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4721f88d4163ab39f4eea3229dd2bdf
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_66fba478c9fd9b68d9f210a318da1013(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([6400, 1], float('8'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_58934a8af579b1b0ba121b2be7088966(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66fba478c9fd9b68d9f210a318da1013
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_47d9319bf1cb2925125edfcfede8c66f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1156, 1], float('16'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1fa801b59cb5d56240e6d4cb7a70363d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47d9319bf1cb2925125edfcfede8c66f
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_acd89ac4414e4de8a85c61e7bdce612f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('152'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f9a251d56ee8965eb610ec71e4338097(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acd89ac4414e4de8a85c61e7bdce612f
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cf74e15f9a14ec8df738f9a6d09a47bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([15200, 81], float('0'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e71d37d6d7b57600880dfe257b588053(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf74e15f9a14ec8df738f9a6d09a47bf
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_045ab350b6a32c3f2f6ad0541ef64dda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a6fcc4cda5f2125053a2941090bf3c50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_045ab350b6a32c3f2f6ad0541ef64dda
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a3fc62061cee7be1ce8ce298e6274acc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([70, 1], float('128'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8a002d439d58ae20896dbf5a6b1518e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3fc62061cee7be1ce8ce298e6274acc
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ac02948dc6f5f36e0d59942aaecd7eb4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('96'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0237ab33c4d4627d566b003ed693ea4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac02948dc6f5f36e0d59942aaecd7eb4
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fc536081c5158a7cdade32bbaa1ff3a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('24'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e2e25822ee8e8fbfedf82806ae3e5c48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc536081c5158a7cdade32bbaa1ff3a6
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_83d588b3c3333402b38c188658430027(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([150, 80], float('0'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_159f332cb3b79d8a61f279f12f6585a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83d588b3c3333402b38c188658430027
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ff0663f962c5a64ee9f6a0b3514e26a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1296, 1], float('16'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d9f4ca7c419aab0943bb429608f93ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff0663f962c5a64ee9f6a0b3514e26a9
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a22a2a43500e3844f04d1b57507d9062(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('21'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a571624f40947aa495ad9b0bab692eca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a22a2a43500e3844f04d1b57507d9062
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d6cda2f0ee2e0c7ff6a39f63608acfb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_082106d4a4ade001417cc2468235515c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6cda2f0ee2e0c7ff6a39f63608acfb7
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_152a1e3f2223141915690f47f4f13c74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([4624, 1], float('8'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ad5d83a830dfd9f5993547b9cd152983(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_152a1e3f2223141915690f47f4f13c74
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cfbe94b34fe6b89fd94abbd6588e7c02(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('56'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2ebc5848a88ea35098342b0d4d8784a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfbe94b34fe6b89fd94abbd6588e7c02
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_87412ecc4a2b39c43e202a2d76dec92d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('20'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f7c718a8d2ca2c7d52863a37cf7138c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87412ecc4a2b39c43e202a2d76dec92d
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_87eed3c185658b28773951e4cff2dde5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('7'), paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c014c7b138b026c9198b8551e3d123f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87eed3c185658b28773951e4cff2dde5
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_61923fef77192d1897e6eec2f0b4242d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([324, 1], float('32'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_579253b049cf80727f7ef8c4fcba2c9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61923fef77192d1897e6eec2f0b4242d
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cdb9c6e37a1ad099fb02aab0dd9fb177(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('96'), paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d1f80f61182a06becf58be845c287a1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdb9c6e37a1ad099fb02aab0dd9fb177
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f253157df38fcfa727590b68b668e4d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([15200, 80], float('0'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4fc6072af1cb8946ba9286d04d060311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f253157df38fcfa727590b68b668e4d9
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_714da24ea57ea077f61e44fb9eaead8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('28'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ab81132ee1f84348ab9405b651154f0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714da24ea57ea077f61e44fb9eaead8e
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b5b1b266389d5bcd3afc3712d045af48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('50'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5028679af16bbc87538f1f5ac0891791(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5b1b266389d5bcd3afc3712d045af48
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_acbeed34561b9349030f6528fbecbbe5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 256], float('0'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_da4332405f0512be511e76956afaa22e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acbeed34561b9349030f6528fbecbbe5
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d2413253aa63670d3d1346d1d6f94d12(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('100'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bddeb994f497b72621b23963a8a0d6b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2413253aa63670d3d1346d1d6f94d12
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8134952c4a4e9002a9ea7d5ea899b6f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('7'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_84ef42c6d2877cf1f04515847113ef93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8134952c4a4e9002a9ea7d5ea899b6f2
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_99bd4562489f357244575c550349b73b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([3136, 1], float('8'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a51dbb24d7f973dddbae951657a9c76b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bd4562489f357244575c550349b73b
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b5c8adf6225c8db9feca49a81ac085a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('6.28319'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6378e391667e41f2cf355e6e3345c6a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5c8adf6225c8db9feca49a81ac085a6
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_13c1100d8c75d133facc38afe52933e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('36'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8363337bfceb4a9dcbdfe7889ba1a36c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13c1100d8c75d133facc38afe52933e6
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5191092bbd44e5eb3c3e9797ca10485f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 500, 1], float('0'), paddle.int64, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e23cdd8a6e6a06737330baffcc19803d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5191092bbd44e5eb3c3e9797ca10485f
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8dbbbacbde5f109432bda6f80f5a8bf8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('-1e+10'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e98363713ec017ecf7721253c5ec03ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dbbbacbde5f109432bda6f80f5a8bf8
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_dfffe9ca7420769f32ee3992df292928(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('34'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8ab73aecee2297b7c67235599d861aa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dfffe9ca7420769f32ee3992df292928
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d5b9d1d84a3f10bbefd624c03999fc62(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('48'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_10e12b7f159da945374be6bf5b176d59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5b9d1d84a3f10bbefd624c03999fc62
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9834278516a7083804562a30c4a15061(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('21'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_835a546be48a66a8f371bb3e5165d300(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9834278516a7083804562a30c4a15061
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5c24e0761c0b69f050eae7f6cac6753e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('25'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7b79e98af52bd1ab4b81af8436c515fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c24e0761c0b69f050eae7f6cac6753e
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_daa4cf185a1a405cda07b76b40be34cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1.05263'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1149a01c6a955e4b1d099b2b13acbb4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_daa4cf185a1a405cda07b76b40be34cc
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7c8d399b5c374f660d415f57f849f39c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([576, 1], float('32'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_50bdabb907c534efa6304545d894dca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c8d399b5c374f660d415f57f849f39c
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0fc6bb519cbfa3c9f81e14fc23a13e2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('15'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_29e4fe97e677c8afd2e7f3570ead5f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fc6bb519cbfa3c9f81e14fc23a13e2c
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_15bf176535d2630da9853a6457043f1d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([247, 81], float('0'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b80267da280d957d87feef6d925f215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15bf176535d2630da9853a6457043f1d
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e73693ce569a85d8384e37834f8fcc4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('-2'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dc9ce2652a2626d7544f09c9c8ccc29f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73693ce569a85d8384e37834f8fcc4c
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8ef070682d7b8817922a17fe489271ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.0909091'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c29af94b4b7bfcf25b9e9aecef8bbdb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ef070682d7b8817922a17fe489271ee
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2619f6925bdd1e5b0e9bf8eb0aa60e5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([400, 1], float('32'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_63132f1a124ab64b29e5763bcf251a99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2619f6925bdd1e5b0e9bf8eb0aa60e5c
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_45844bca6febeb2106216c64f707415d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('20'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3ea9113c7d653b4e86efaa7c4bf0ddd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45844bca6febeb2106216c64f707415d
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6699bc9704e725492fffa3458ffc334f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('1025'), paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2d480cff818affa03ff7e1187dc657bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6699bc9704e725492fffa3458ffc334f
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0b4903a908dc04b63996ec2711b67a74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('464'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_87dcb0aad53113d7325703657d72bd87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b4903a908dc04b63996ec2711b67a74
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e7ff74b631d6bb0b961307534164b9ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.405285'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f7683ea1f119e2d418042a09ed76dc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7ff74b631d6bb0b961307534164b9ee
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_74e8f29c5e4d33361199b1e287b7d579(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([784, 1], float('16'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe527abd7d1bffd523fb1753d81315c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74e8f29c5e4d33361199b1e287b7d579
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6ff6eb911004dbbcdd89b22fb36d2e9d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([3800, 80], float('0'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_33f11e74154d4d6e4e0a29e10254428d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ff6eb911004dbbcdd89b22fb36d2e9d
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c8ee718376401cf5836947468b4456dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([70, 80], float('0'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9caab1d4029d5b81fc8aae09f55d4e67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8ee718376401cf5836947468b4456dd
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2a42690a8575d87771800bfb573b8318(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('72'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fd8b215687e814599ab599db2bb8d606(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a42690a8575d87771800bfb573b8318
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c98b8da2038dfc804181eed3c5061cd1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('58'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e26e44c9f125c343dcb8d915ee0c1765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c98b8da2038dfc804181eed3c5061cd1
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_db697d842c95523d54f62e752b5c60b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.0015625'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_17048d4c30b8427535541b9dc59eedde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db697d842c95523d54f62e752b5c60b5
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6ba5bdd08d42ebebdfe3b89caf34c3a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 501, 4], float('0'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e7e53d180f4086623f385c72d251dcc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ba5bdd08d42ebebdfe3b89caf34c3a3
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4a94b53e1da5b2d120f94a560bec135b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('17'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7e394a371726f65107c4962957d5a98f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a94b53e1da5b2d120f94a560bec135b
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c3612a3a867fdb5ca27c5b197a55e9ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([3800, 1], float('16'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2501f034a3023febb49f905f5ec5d0c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3612a3a867fdb5ca27c5b197a55e9ec
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_11304d3cbbdb196aca0fafaa7229a5e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('15'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5fc4db0a03212f417338c35977e1f676(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11304d3cbbdb196aca0fafaa7229a5e1
    def get_inputs(self):
        return [
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_228b7ebd6d346b8c50e27b1a676043ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1024, 1], float('16'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8371f310145d87f833c67f94e104c18b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_228b7ebd6d346b8c50e27b1a676043ee
    def get_inputs(self):
        return [
        ]


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