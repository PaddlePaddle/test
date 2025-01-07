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

class PrimitiveOp_212f5717a6d19317cb79ef5885451912(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('192'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e89f1b523aa592158e3aa17e9e5767c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_212f5717a6d19317cb79ef5885451912
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

class PrimitiveOp_892aa6fdb53cddb086de55ebec43b8a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 3], float('0'), paddle.float64, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_69bc2d8ad54eb2ff0044cdfa215651b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_892aa6fdb53cddb086de55ebec43b8a0
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

class PrimitiveOp_f0e68dba167d108e9b3598a95df993d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c267fcb13fba36f4b69ca0a2d07eca9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0e68dba167d108e9b3598a95df993d4
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

class PrimitiveOp_c7efbbf8b5f7fe8f23796008dd0393e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ab772cfe8db05db5a00b22eae6873e89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7efbbf8b5f7fe8f23796008dd0393e0
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

class PrimitiveOp_01410cc183085d07fc00672f9f48a9f0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a6f342ce300e884871b538f7d3804e9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01410cc183085d07fc00672f9f48a9f0
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

class PrimitiveOp_5c5af60162b7a461e8e178b38fd203c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b7481c8679349d917f861640717085de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c5af60162b7a461e8e178b38fd203c5
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

class PrimitiveOp_f68daac7cccc907ccc756ce4b26975db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0c0047366af7ad02de8376b7eb15b2a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f68daac7cccc907ccc756ce4b26975db
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

class PrimitiveOp_e69121b8cd0da02312380c0ff6e332c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_66bce8791df667edfffc0188280c9bdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e69121b8cd0da02312380c0ff6e332c3
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

class PrimitiveOp_98e030319e21b81318af30777c68ff8a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1'), paddle.float16, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b7b8a89dc8ef603f388a802fd086d8b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98e030319e21b81318af30777c68ff8a
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

class PrimitiveOp_a815726f5f8be9ddfaad889074983039(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('7'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34d19b292fb704f84f47359add70c481(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a815726f5f8be9ddfaad889074983039
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

class PrimitiveOp_b24e650b2dd63f476878427b84d16741(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_03100bffa9d93d6d1eb6ff5945106531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b24e650b2dd63f476878427b84d16741
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

class PrimitiveOp_62a175a5c80832c2f6b3e80404550e7b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d019458c50b0f1b20c78f075feae4551(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62a175a5c80832c2f6b3e80404550e7b
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

class PrimitiveOp_816fdcec44b98cfee881610c6664c6c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c22a7a7bd24b5df50fdeb8264e08396d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_816fdcec44b98cfee881610c6664c6c8
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

class PrimitiveOp_efe317dc7250c2c6c3e743b72e4deb9f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_19bbb740c1b44c382b274072a675e006(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efe317dc7250c2c6c3e743b72e4deb9f
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

class PrimitiveOp_9694c5e30f233d6ad2fcc2efcb482b09(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_60ff7efa442ad0f1117d56ba0d52064a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9694c5e30f233d6ad2fcc2efcb482b09
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

class PrimitiveOp_1dd77b524009836eb01e75a9adfb4b1d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_71b12948b6e380dc824a365883fda908(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1dd77b524009836eb01e75a9adfb4b1d
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

class PrimitiveOp_287fcbebe7c4936f1dfc52f57cd65dea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('26'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32dac9810a6a930e2c8658d68963d27e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_287fcbebe7c4936f1dfc52f57cd65dea
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

class PrimitiveOp_5e799df0472160eb0007d5080868837f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1271ab533b5b14420e7cac9faa396e0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e799df0472160eb0007d5080868837f
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

class PrimitiveOp_1490f1fb6d1891d3fc8095d073bf4f12(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_421e0d940d0218db0de81160a066f6a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1490f1fb6d1891d3fc8095d073bf4f12
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

class PrimitiveOp_34ea4723fbded6e67742bf2df5f2a767(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('768'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c1f4f29c8c7a367572139381caf8a777(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34ea4723fbded6e67742bf2df5f2a767
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

class PrimitiveOp_d5db7a753e048b50598821858503e7f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a46f1b9699e533360f5e3e7147f3f3de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5db7a753e048b50598821858503e7f9
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

class PrimitiveOp_8b78076bbd1d529a975bb468edd0a46e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([2000, 9], float('0'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77b5f350b3c44c613dcb449988a60ba4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b78076bbd1d529a975bb468edd0a46e
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

class PrimitiveOp_1a6548ed7eeb1b5cfbcd512e778ac2e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('2'), paddle.float64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4b790112a74543fccbf563d97af8f16d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a6548ed7eeb1b5cfbcd512e778ac2e5
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

class PrimitiveOp_d54c4625389fdf7f474672cef80e37e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('91'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e328f20a87d38d34c3105b4e2ae11c04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d54c4625389fdf7f474672cef80e37e1
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

class PrimitiveOp_4af097647e7fa811a036cfec0d75a022(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2d443f371ddfbb2801c96268c9337172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4af097647e7fa811a036cfec0d75a022
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

class PrimitiveOp_fd563aeb894770095a4d8a7be946f6a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fd03f6ef648f5d3a301d850e5ede4e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd563aeb894770095a4d8a7be946f6a6
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

class PrimitiveOp_ab2d1d4dcfe21d435511a88236370d3f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('4'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5533692a1bb48e9705600124f5fe8b9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab2d1d4dcfe21d435511a88236370d3f
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

class PrimitiveOp_79b80b2f4388141935ef8e22a3cde4e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('96'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a97d375fd693134ec9fce3a2dca974fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79b80b2f4388141935ef8e22a3cde4e6
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

class PrimitiveOp_7ea218f273d15439a226ac342f6d9f54(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('75'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1fb94c4a1eb90cb2432e11b335f3177e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ea218f273d15439a226ac342f6d9f54
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

class PrimitiveOp_9a312d7d2baf49a6e75b7bd2a3eb39ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.408248'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ef232cecd7d7e0dce2517ac1ba70691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a312d7d2baf49a6e75b7bd2a3eb39ad
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

class PrimitiveOp_5af66642177fb642246439cee79875cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2fda629fa606cd6e76a1127984b8ce4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5af66642177fb642246439cee79875cd
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

class PrimitiveOp_4838c6dce3db5063b6ed86f31e63dcc9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b78089d3f941730bcaace8ca25196f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4838c6dce3db5063b6ed86f31e63dcc9
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

class PrimitiveOp_18b96b7555da59088189b735112d1a9c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bfc14410db09e995eff0a19f5398291a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18b96b7555da59088189b735112d1a9c
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

class PrimitiveOp_c9e537c62b98718e36ee82f5b812cb9c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 6], float('0'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_31492bb2ce44579ad577ffe78c662297(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9e537c62b98718e36ee82f5b812cb9c
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

class PrimitiveOp_9ed37cb6b257e599f7f5b482616c6c84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('48'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f3e9631e54e3754336af4a32fc1bf6e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ed37cb6b257e599f7f5b482616c6c84
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

class PrimitiveOp_202258866afd7a351a012f30307b3cfe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('26'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_483c946ea6fa273b32d2ed5f71674e12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202258866afd7a351a012f30307b3cfe
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

class PrimitiveOp_b5dbb17e86077f798e6a0b6999747c96(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('576'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a7dd1e02f8965ec14e0636e292ea58eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5dbb17e86077f798e6a0b6999747c96
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

class PrimitiveOp_6aaacf6d4aebce325244f321cfdaf64a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('10'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0c01084af22d1bc6a76b97d840e2066d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aaacf6d4aebce325244f321cfdaf64a
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

class PrimitiveOp_128fccacfb7d8cf5248f98681686af00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1.702'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_087f2bdaa50b27fa98835f2cc87c9576(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_128fccacfb7d8cf5248f98681686af00
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

class PrimitiveOp_b4b66b37b7c11295b9f175c5b9e6baa9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46bcbf234b68ffd94a02e5b73cab36ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4b66b37b7c11295b9f175c5b9e6baa9
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

class PrimitiveOp_9f33c23efcde3846b8c3cc7824109e73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 4], float('0'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_966c30677846a298ab74e61436046204(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f33c23efcde3846b8c3cc7824109e73
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

class PrimitiveOp_865fa8c5dcc47a331c342aee8620f874(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.204124'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_559e92d7107d548bd12d79a5a8897719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_865fa8c5dcc47a331c342aee8620f874
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

class PrimitiveOp_ef77397a681d05254cfb2a183415e302(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a089c09c6186214b5e303b0dfb99e478(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef77397a681d05254cfb2a183415e302
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

class PrimitiveOp_07d27a2638f6d707277212823c5a1f0e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('3.14159'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_71b7feee37f74dce5f1e94b57a0d78fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07d27a2638f6d707277212823c5a1f0e
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

class PrimitiveOp_6ad3d156862b70ed6f9637c904794898(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1e-07'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f0de3b235c423dede59b49ebf76a940d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ad3d156862b70ed6f9637c904794898
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

class PrimitiveOp_1f3b6506a0ceadb449453e1f7d49dbe4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('640'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_23160eb45c112217f88da7343c63feb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f3b6506a0ceadb449453e1f7d49dbe4
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

class PrimitiveOp_b95f90c68ef1347d435678e0fb940175(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.333333'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3db5ec85179c2626f0b3058fe3a5f2af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b95f90c68ef1347d435678e0fb940175
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

class PrimitiveOp_5b0647bd111fb263416489a99685e04a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b54da4875490d41708c2d03100576f7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b0647bd111fb263416489a99685e04a
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

class PrimitiveOp_72f917db0d1fe31ffc2d32242e8bbe14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('784'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f372e1d2aa8f76b1772ddbf798aa3ab6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72f917db0d1fe31ffc2d32242e8bbe14
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

class PrimitiveOp_c7881c9c8a31af817d4e45b9b8db8b26(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1772ab3cb0d3f32cf485939e446649de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7881c9c8a31af817d4e45b9b8db8b26
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

class PrimitiveOp_b1ec24dc68f030195db887de97dd6e45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('-1'), paddle.float16, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b7cb42774c1e0453c79ad816a8585ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1ec24dc68f030195db887de97dd6e45
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

class PrimitiveOp_bfc4e2516ba5d143aadbd329d4e78471(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('2304'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fa3c084020635e4d903b2ab8a08ba184(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfc4e2516ba5d143aadbd329d4e78471
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

class PrimitiveOp_5194026ebc7ed5a982445d141eb874d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('-100'), paddle.float64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cc320ce3ca24a560bd68cd1682ca6f75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5194026ebc7ed5a982445d141eb874d2
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

class PrimitiveOp_0914745bd003abe880bcfbe848799e1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('0'), paddle.float16, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3fde827a865435f2fce26ec61f93fdcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0914745bd003abe880bcfbe848799e1b
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

class PrimitiveOp_7590156445f9d1ce732b6a39bdf77e74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.0031348'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_09309e6c52c1593fae27aa2f973ebc26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7590156445f9d1ce732b6a39bdf77e74
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

class PrimitiveOp_d0ad9af5e069dc9ba3253cbe17323918(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6ec903fc8d0783b063a7c68d1932d69c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0ad9af5e069dc9ba3253cbe17323918
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

class PrimitiveOp_3b06791f087600e7b9e0b98bd95c7053(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1769c3f8cdfd56b960c20f6bd60d94ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b06791f087600e7b9e0b98bd95c7053
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

class PrimitiveOp_8901d552498200a3fe3ec7eed0b2a2e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ba6ff88b2ca14aa8d67b346c4816b6e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8901d552498200a3fe3ec7eed0b2a2e3
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

class PrimitiveOp_058bcfd862d97f053cee964d600f87d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('350'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf6b8c6b36ff6cdaa1cf0a751d23a2c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_058bcfd862d97f053cee964d600f87d0
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

class PrimitiveOp_0e8f94b76d8e9c43d71e3ca38d5b7b9c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a439d79aea43067746910540d1cb4a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e8f94b76d8e9c43d71e3ca38d5b7b9c
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

class PrimitiveOp_37f3947666c182f0a5be44db04321d1e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('2000'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4bd01ae5337522876333bf06a084f19b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37f3947666c182f0a5be44db04321d1e
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

class PrimitiveOp_c3819ec1621ae6c1558e0e60cc640f8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('8.92857e-05'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e6812e2c1a5ca9e0a811054129a4602d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3819ec1621ae6c1558e0e60cc640f8e
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

class PrimitiveOp_3c812ebc0183a9b7aa9db832af274c48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('-3.40282e+38'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_635015d73ee2132f0e93f7b24f63de70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c812ebc0183a9b7aa9db832af274c48
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

class PrimitiveOp_8959def050a8a2aedd2a6bc0a5752560(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('-1'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_691d6cb812eb7ed34ee1526d61eb679d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8959def050a8a2aedd2a6bc0a5752560
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

class PrimitiveOp_6700a28128c27a12077da10de3bb3dec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('175'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8861d6d133b4f6e54b99392d728a3279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6700a28128c27a12077da10de3bb3dec
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

class PrimitiveOp_be2d19bfc5abfff69bda8a5503d52dbc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.000178571'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_45727429846ab239c325c8def694312b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2d19bfc5abfff69bda8a5503d52dbc
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

class PrimitiveOp_5daef799e9e59f2173893182043fe3ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('4'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a70a540da3fc92ecec1e20831f96b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5daef799e9e59f2173893182043fe3ba
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

class PrimitiveOp_c45cffdee58a7941cdd86ed7b9be7d29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('80'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_06afaf82f2a6254273b9e5651ad2f873(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c45cffdee58a7941cdd86ed7b9be7d29
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

class PrimitiveOp_2777b13cdaaf9b9b311c3480c37e0730(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('72'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_29ba27da5a052c683046214dfe288d9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2777b13cdaaf9b9b311c3480c37e0730
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

class PrimitiveOp_532475190902cbfce077e38db759d92d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1024'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3a8dc1251f34fbac521fd858f79ee649(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_532475190902cbfce077e38db759d92d
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

class PrimitiveOp_77b3f90d0dce9f83414223a2b864a39c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('2'), paddle.float64, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_df829089ddafe964230be9a009e1533f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77b3f90d0dce9f83414223a2b864a39c
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

class PrimitiveOp_cd7809b1cfae864c5a7c1ee191177985(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('60'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4b65b53a15d9daf3bf2d939497b5ac67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd7809b1cfae864c5a7c1ee191177985
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

class PrimitiveOp_41b7b762baaa380fe321d2512c9ce92b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([3200, 1], float('1'), paddle.float64, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ff4e1fec95db6568d6e6b8ea4c20fdb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41b7b762baaa380fe321d2512c9ce92b
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

class PrimitiveOp_38511ad5245677501c975cb6f05ec084(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('-13.8155'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8bd021d0da400f5dc7daf49bcbc81857(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38511ad5245677501c975cb6f05ec084
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

class PrimitiveOp_0bc24ca55765815d8e28a9bb1166e602(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('192'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d7ed44c5b2a0132160c9ed634f1f4e08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bc24ca55765815d8e28a9bb1166e602
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

class PrimitiveOp_f8287a279c9ce5b764d4509fca903eb1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('112'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b5c64095dc19c4880409b5b837faf91a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8287a279c9ce5b764d4509fca903eb1
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

class PrimitiveOp_445a6351bdc8e2f1b3a5f19393643cac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('13.8155'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_00d91cdc14cdffcece1bfa2e33d71652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445a6351bdc8e2f1b3a5f19393643cac
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

class PrimitiveOp_184a192558d84a22687b6412566d3173(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('96'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_37effc8c5d73c43809cb3cea80abd298(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_184a192558d84a22687b6412566d3173
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

class PrimitiveOp_d7a00f52ff28759366de4bdd1e7906d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.000177557'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e897b455acdd044ca787c5620ea1e27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7a00f52ff28759366de4bdd1e7906d1
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

class PrimitiveOp_ccb00156bb3dd1b891ce6140d15c5eb8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('20'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe5bd75c35fcec8f2275877932fd293b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccb00156bb3dd1b891ce6140d15c5eb8
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

class PrimitiveOp_63fc9e56d7b0282615b834b19bf13c1a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('32'), paddle.float64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_645ec30940c462895e6caa5a4362a128(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63fc9e56d7b0282615b834b19bf13c1a
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

class PrimitiveOp_51f7cd5bf70766b198384774a920992d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('3136'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b128aa98c9ade1804d00a469673abde1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51f7cd5bf70766b198384774a920992d
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

class PrimitiveOp_80b698a08ad47173aad33969e710e1ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.0666667'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_02e4bb605a91d5861e01b7c418af7bb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80b698a08ad47173aad33969e710e1ff
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

class PrimitiveOp_e3530fbddea55a818bcff7dfa373e5b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 32, 180, 320], float('0'), paddle.float16, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_61ba65f033f163501a094f477a9b6d20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3530fbddea55a818bcff7dfa373e5b6
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

class PrimitiveOp_21d9a7fc8c411ac341119da8ed1184b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('14'), paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_566317227ff8b28d8189aa91300e82c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21d9a7fc8c411ac341119da8ed1184b8
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

class PrimitiveOp_0fd46740ac9f8ee443e5b86a76dbe239(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('70'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f11b3355bc6c59243f94b4ec5375e931(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fd46740ac9f8ee443e5b86a76dbe239
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

class PrimitiveOp_bdeaa77fed9821f851422ce860450e49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('1'), paddle.int64, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1d812ce9480f10d4c4179d5a65ca09d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdeaa77fed9821f851422ce860450e49
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

class PrimitiveOp_3d672c671c2022ba522ebd66923a7cc3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('96'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f06333c893cd4bc551708d19fceadd8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d672c671c2022ba522ebd66923a7cc3
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

class PrimitiveOp_7514b7dda43fb375ecb02f24034eb27d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('22.6274'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32712b5d45b3ffbb54e994b6697bd45c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7514b7dda43fb375ecb02f24034eb27d
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

class PrimitiveOp_341e4a2a256940eb7f9fddd97484541d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([26], float('-inf'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_124b2f793c934240ec54e89d91036988(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_341e4a2a256940eb7f9fddd97484541d
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

class PrimitiveOp_4282747c1e84ae88813c88e2b4ab3216(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('2000'), paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d1bc72691e50708c54e04475d43346f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4282747c1e84ae88813c88e2b4ab3216
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

class PrimitiveOp_98903ba97881ac6a782d020ca418c327(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([2000, 5], float('0'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f0f8a89c1ede0c8031b1d40a6ae5a4be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98903ba97881ac6a782d020ca418c327
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

class PrimitiveOp_825c26d16fcc5f4c35acbebb1187f583(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('100'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b9e5f2b7925f59f017246b17f3a4ef83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_825c26d16fcc5f4c35acbebb1187f583
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

class PrimitiveOp_6eea14b1c8312ddbd83228e1702fe2f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 256, 28, 28], float('0'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2809931f4546b2d86ce9500a39d642f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6eea14b1c8312ddbd83228e1702fe2f1
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

class PrimitiveOp_777777d0fd6a2826c84be63e5a3533bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 20], float('1'), paddle.float64, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d0144a545f255dd622914dbc6ec4e832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777777d0fd6a2826c84be63e5a3533bb
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

class PrimitiveOp_cf2fb69895ccf1dbdecc554abfe064d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('88'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fa7c8527d5d7adec94489afd2451d334(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf2fb69895ccf1dbdecc554abfe064d8
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

class PrimitiveOp_e0adeac9a5b3dac4fba426bdd74fe7cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([2, 3], float('0'), paddle.float64, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3e6d7f8c47c843448b4aa47d3fda8322(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0adeac9a5b3dac4fba426bdd74fe7cf
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

class PrimitiveOp_47cb7cb5f1a5f339cdc45f545826c6ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 1], float('0'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0178f9aea2737a3bee92ab81922c874a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47cb7cb5f1a5f339cdc45f545826c6ec
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

class PrimitiveOp_e56fbee32706807f4abcacffbf847f24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('6'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8746d1181951a829ad0510da6455dab5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e56fbee32706807f4abcacffbf847f24
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

class PrimitiveOp_a69af0311d788543266f5c96c27d52f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('9'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d26b971954904b59a437ccf9c6e6a7d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a69af0311d788543266f5c96c27d52f7
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

class PrimitiveOp_f2f64d36fc84f3fcf8f04415044340d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('-inf'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_21f966e18696a98be2d7de8fdb627594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f2f64d36fc84f3fcf8f04415044340d3
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

class PrimitiveOp_3cf40280d9541509bd37532bffe5a6c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('16'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a93e01b4ee40ba34d0a192627ad6f31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cf40280d9541509bd37532bffe5a6c0
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

class PrimitiveOp_7048aee557196fb39ed4b50648472393(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('3.14159'), paddle.float16, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bb7295024e90009ec7aedee1b73b9209(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7048aee557196fb39ed4b50648472393
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

class PrimitiveOp_58517b06790a7bcd91d9f79fede49d85(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 32, 180, 320], float('0'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_98c2c134ebb208161cb6c548ad6081aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58517b06790a7bcd91d9f79fede49d85
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

class PrimitiveOp_b384d23fa0c7bc3dab079f018e1796f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('501'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_802821d38da66df983400fbb0791385e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b384d23fa0c7bc3dab079f018e1796f9
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

class PrimitiveOp_ea1ea09126365ff97d32aeea0f937a98(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([20, 1], float('1'), paddle.float64, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b63f1ab0f3ee53975794f7f3b8c0acd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea1ea09126365ff97d32aeea0f937a98
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

class PrimitiveOp_3ce03cec47716f0b7fce2c5820cc23ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('18'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c3657f54a622dc166e7045572ebab7d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ce03cec47716f0b7fce2c5820cc23ae
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

class PrimitiveOp_9e7eb85e136bccf8e43ced3a41e9a556(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('320'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_21a9c8093d7477a33a536719d29af6a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e7eb85e136bccf8e43ced3a41e9a556
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

class PrimitiveOp_7d0176b258e2ba204f9bb3290bc82bf1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('4'), paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7e8edd2702effa58d139cebfacf5acc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d0176b258e2ba204f9bb3290bc82bf1
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

class PrimitiveOp_f181e134202366a013361714ddc0f27f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1280'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_903380626726466fc6ba38ed08c7a78a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f181e134202366a013361714ddc0f27f
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

class PrimitiveOp_bee61ca257139a3d1c88c960f6d78239(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 80, 28, 28], float('0'), paddle.float16, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4345cb92ebdb37a9aeabff29799593ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bee61ca257139a3d1c88c960f6d78239
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

class PrimitiveOp_36704f2ffa0c131055a022e82b54486d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('2'), paddle.float16, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_96c8409179245c23e2d125c7abd90f26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36704f2ffa0c131055a022e82b54486d
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

class PrimitiveOp_dcac572044d8d21d04ec06fd0955e240(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 5], float('0'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d3b6fa976201280c479b8bb2d971a91e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcac572044d8d21d04ec06fd0955e240
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

class PrimitiveOp_05787dfaea1e924c2a64c3c575b528fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.0126582'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_236451ee0211428d11c7f938002f25a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05787dfaea1e924c2a64c3c575b528fc
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

class PrimitiveOp_ace07ed75034c1001387bf78ac5354d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.00558659'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a2580c3d6f61bcd41c4750e5ea2e8de3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ace07ed75034c1001387bf78ac5354d8
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

class PrimitiveOp_cfd7dc459cc5e2704a72a42a9ab2de35(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('0.5'), paddle.float16, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_badfadd2a5addecc5e9526b4ce420318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfd7dc459cc5e2704a72a42a9ab2de35
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

class PrimitiveOp_678ff66a47a8e3abd952f437e8d59d52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('256'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_536ef9574a8102fe2ee657eb97523a54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_678ff66a47a8e3abd952f437e8d59d52
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

class PrimitiveOp_070c9419ca42f68ee1012a6898a278a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('2'), paddle.float16, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_84ece2ae71ffafa02aee8e137a863cdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_070c9419ca42f68ee1012a6898a278a5
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

class PrimitiveOp_f8d48f378e89bbc889c03fdf6e0d7b3d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('-1'), paddle.float64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1024b496562139d18aca24b2ac5ac18b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8d48f378e89bbc889c03fdf6e0d7b3d
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

class PrimitiveOp_85c95959631d3adda955c76736300aa3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('26'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dbcf603a9b883e53e0d6cd61036a5c44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85c95959631d3adda955c76736300aa3
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

class PrimitiveOp_517539a2a211d0fdf2d2e1061de3d7aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([2000], float('0'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68720d31dffa0d6b1c3b96a3dd06af60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_517539a2a211d0fdf2d2e1061de3d7aa
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

class PrimitiveOp_7a5614432ce9207b381be900eec520c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1e+07'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a51b9b023e8e4b71faa6e071192eeb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a5614432ce9207b381be900eec520c8
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

class PrimitiveOp_449dceabd8048ff8e6da402db62a2804(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 2], float('0'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d9a57d5367c1e3851d78bc84c7a615c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_449dceabd8048ff8e6da402db62a2804
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

class PrimitiveOp_98b64ab24bf6e21d3bc733897963e1cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('16'), paddle.float16, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b68a6c1d8b1b05eea072c91039aafa32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98b64ab24bf6e21d3bc733897963e1cf
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

class PrimitiveOp_b893b1a7bb4a1d5bcb2479139b838e4a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('128'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_857bbdc334f3af417796fade17a7f005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b893b1a7bb4a1d5bcb2479139b838e4a
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

class PrimitiveOp_a826e3ffad3b004d3706efeaf70d5eb8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0'), paddle.float16, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_01cc8cc4a9ca526c433105fc2cb2352f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a826e3ffad3b004d3706efeaf70d5eb8
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

class PrimitiveOp_26084978640f2a6303f2a61b51cb3cac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('120'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_285c56766fd8e2b4165f03966e28cbf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26084978640f2a6303f2a61b51cb3cac
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

class PrimitiveOp_16c0ff9fc5bc24dc0dd8e7c9a6dc5009(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('56'), paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a6213ee5b1fd0d28c9ffce41ac5b1d17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16c0ff9fc5bc24dc0dd8e7c9a6dc5009
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

class PrimitiveOp_9ff4909b71923c6ec31e4dab5c8b9957(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([10], float('1'), paddle.float64, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_142809de605af651a4bf3750060a1c33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ff4909b71923c6ec31e4dab5c8b9957
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

class PrimitiveOp_93a16c8f11403d95dc1cafb87a25c8e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('-32'), paddle.float64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1daf7a57b010b109fd0a26b106328cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93a16c8f11403d95dc1cafb87a25c8e2
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

class PrimitiveOp_d034a0d080c52b79761cdde76a563e15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('0'), paddle.bool, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ee55caf3563ee9814e1c1e91617dac95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d034a0d080c52b79761cdde76a563e15
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

class PrimitiveOp_7075013c5b5a6db6c024c8798bc01acb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.00628931'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd1f9592b5370d7c4a6cd1ae968496f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7075013c5b5a6db6c024c8798bc01acb
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

class PrimitiveOp_dfaff812d76d8d703d2d9103fd408e0a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.0883883'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e16f96b578853723db30d517690552ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dfaff812d76d8d703d2d9103fd408e0a
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

class PrimitiveOp_71ece4593dfb12f19cde3357f290cafc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 256, 14, 14], float('0'), paddle.float16, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d3e26ea76067e4e51bbc9a765e01b591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71ece4593dfb12f19cde3357f290cafc
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

class PrimitiveOp_8e7c79b71907cbdf1e6303548fe0d1a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('100'), paddle.float64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1c41891250e9841d55faf3872a7af465(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e7c79b71907cbdf1e6303548fe0d1a5
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

class PrimitiveOp_21cac0ee8dc8483fa138781d3e47430d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('32'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f71daecea6517c050a5e63c37f3dd8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21cac0ee8dc8483fa138781d3e47430d
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

class PrimitiveOp_f957149aed78151938b5a4adc872a28b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('85'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac7460ff3913a77a18dd1f1fd81ef032(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f957149aed78151938b5a4adc872a28b
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

class PrimitiveOp_ebc4e708aee3d71e8dd885e16c60cf21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([2000], float('0'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3364de13f99e8a824bfa0baebea32b70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebc4e708aee3d71e8dd885e16c60cf21
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

class PrimitiveOp_cd32ca15de34bc73e285246e4d35b8f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('3.14159'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8120ec1dda7ec6658c15c72614b95734(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd32ca15de34bc73e285246e4d35b8f8
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

class PrimitiveOp_4f970ae35222759d5acf509c4bed9f20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('40'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46d57f081b22b9c98f75a16456da9dc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f970ae35222759d5acf509c4bed9f20
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

class PrimitiveOp_6bb7729bd49f2fe340fb7f8a7dc2a0e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('1'), paddle.bool, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dbf2f4cdc29626ed95b1a876a85d028b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bb7729bd49f2fe340fb7f8a7dc2a0e4
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

class PrimitiveOp_40348899109439c41e10a572ff17591c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.0105263'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_56dd5f8242baf4a9293845b0c6797e7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40348899109439c41e10a572ff17591c
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

class PrimitiveOp_e58c52f0adb37111feae36de0d78c10f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 1], float('0'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_81d0e47beb3093693021d5e03ddebf88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e58c52f0adb37111feae36de0d78c10f
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

class PrimitiveOp_c1293eb6925a48b066f0213ec2d28cfb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 48, 256, 512], float('0'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_23c2abc26a123b03c147a582c3f09291(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1293eb6925a48b066f0213ec2d28cfb
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

class PrimitiveOp_db81043d2cf1e19429e87802b2acf29c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('48'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_117b6759684034100eab3d84e2e56797(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db81043d2cf1e19429e87802b2acf29c
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

class PrimitiveOp_825bb99a66a42822d0c1caba407a688d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.144338'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07d6ca355636cd797018f9c4ace2a8ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_825bb99a66a42822d0c1caba407a688d
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

class PrimitiveOp_9368b173cc2f17a019f8811a8fac0c4e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.9375'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1de997479b4c5438455789427e16864d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9368b173cc2f17a019f8811a8fac0c4e
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

class PrimitiveOp_118317c87f7c183294a90394f022ddde(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.129099'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2eb9d505b3a74b12e5f152a1a73ad2e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_118317c87f7c183294a90394f022ddde
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

class PrimitiveOp_906bed397f275a3d719be1d823ae3e91(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('1'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e7f595820f4ca377ed3a04a0fd89ee5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_906bed397f275a3d719be1d823ae3e91
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

class PrimitiveOp_9acae3adbf6635be98328dc5d747cc16(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1000], float('0'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d245f3f3073c3940364be75810034962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9acae3adbf6635be98328dc5d747cc16
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

class PrimitiveOp_125dc9b7dbe179e45b57a253d8e73787(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('64'), paddle.float16, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_568365a4dc972171513ff3006e5dd3f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_125dc9b7dbe179e45b57a253d8e73787
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

class PrimitiveOp_216af61262d7395a26c53c5da5490bdc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('16'), paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5ad09e6c4c944c8ad55c3e484989e293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_216af61262d7395a26c53c5da5490bdc
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

class PrimitiveOp_a925c1e20768322fdaeb359a62a91f0e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('3'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_31b891e56f8c2b77cd9d70927f4b72cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a925c1e20768322fdaeb359a62a91f0e
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

class PrimitiveOp_7ce43e244de07002abcede0580a0f884(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('6400'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e568bee4e6e7cf25a5e513c1dc129f63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce43e244de07002abcede0580a0f884
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

class PrimitiveOp_54c5737692d983955bdd7084438cca23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('9216'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d86ad744fc9c513ca0bc3eb41a7f3877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54c5737692d983955bdd7084438cca23
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

class PrimitiveOp_f54942d1af00faca88a9330bcc08dc3b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.0333333'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c8811877b6dd1fc7e3624bc7b9ebfa0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f54942d1af00faca88a9330bcc08dc3b
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

class PrimitiveOp_b6ee33736b7b88747ff22b0ad343f206(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('80'), paddle.float16, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c48ca9c476b47508b924415ef4e500a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6ee33736b7b88747ff22b0ad343f206
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

class PrimitiveOp_3de0902e231d3ce44ceff5eb1971ad39(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('26'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f49b70912712a14fe705f6389c099fd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3de0902e231d3ce44ceff5eb1971ad39
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

class PrimitiveOp_727174fbafee912258417adc6586a92d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1'), paddle.float64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_67b9578696595dfa9c56a7527731e97a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_727174fbafee912258417adc6586a92d
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

class PrimitiveOp_8e9142d04d2e7d386c9ced9bad6f61ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([3], float('0'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_57745b20e7d0dd6c7abc87dde5f8dcfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e9142d04d2e7d386c9ced9bad6f61ac
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

class PrimitiveOp_b39b9af5fdd980f0245033df7d080702(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 1, 1], float('-1'), paddle.float16, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6fcc82801de0133a628f0d6a3cccb52b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b39b9af5fdd980f0245033df7d080702
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

class PrimitiveOp_37059026fe9f1551edfc67d653bd57aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 256, 28, 28], float('0'), paddle.float16, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0280696fa5056259a05e0bb7efb82610(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37059026fe9f1551edfc67d653bd57aa
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

class PrimitiveOp_8c0dd7ba510acbd9d91257c3ccf66bcf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 9], float('0'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_566cb6b2a0a5ece5ee65440f1736f3e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c0dd7ba510acbd9d91257c3ccf66bcf
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

class PrimitiveOp_39ebd5881e86f72c3254fadbaa1ea731(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('40'), paddle.float16, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_939420bc43dd16bcebd8d68624b72d99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39ebd5881e86f72c3254fadbaa1ea731
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

class PrimitiveOp_3b0558e8486e224fc277040870abf987(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('13312'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5bcfa6dc5780993c14658a4ff5ec1c60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b0558e8486e224fc277040870abf987
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

class PrimitiveOp_5ff2599aa9491cc32038ed5efc62a2ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('64'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eba8a4d6706298eb3b43dea30e2442a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ff2599aa9491cc32038ed5efc62a2ac
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

class PrimitiveOp_33ba65a5710bb83a51e947103cda4a1f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([26], float('-inf'), paddle.float16, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3b824e61ad6c6cbc6c802cfaacf7cdf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33ba65a5710bb83a51e947103cda4a1f
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

class PrimitiveOp_429e117a840a4752e374132e7744d58f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('0.5'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_504d559d480f03191c62d652931b2f7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_429e117a840a4752e374132e7744d58f
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

class PrimitiveOp_32f996816ee8e5e552dc6b35e6532f5e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('2'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c573cfcf3567658de52b46c09ee8e00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32f996816ee8e5e552dc6b35e6532f5e
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

class PrimitiveOp_4502138816fd29c446dacd517473454d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('32'), paddle.float16, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_029543476bb444a456a115129433e562(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4502138816fd29c446dacd517473454d
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

class PrimitiveOp_ce297dcd0faf473d39e0ce63d27ce973(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 2], float('0'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_59c292c52d9086cf08431a292b47cbb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce297dcd0faf473d39e0ce63d27ce973
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

class PrimitiveOp_cde87449b73f70672163f4941a5e471b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('0'), paddle.bool, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf443467e30f74abf712e5cf42004f22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cde87449b73f70672163f4941a5e471b
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

class PrimitiveOp_4cc9b1d13b3552eb543f9e239170369e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('160'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b94be926f229cdd0595585d3a39bac7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cc9b1d13b3552eb543f9e239170369e
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

class PrimitiveOp_167c633832fa109f292e776d418fad7e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.0052356'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_63f78e7b23443b2a6ebcac197f0b1b53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_167c633832fa109f292e776d418fad7e
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

class PrimitiveOp_685b30b2ccfeb748ce519e93b3f08ca3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('8'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9138b54e243209f570d5b852bfdc7d99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685b30b2ccfeb748ce519e93b3f08ca3
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

class PrimitiveOp_5b2efdba9f4a1d90e34c9110851392d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.0212766'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5f4897fffad1d2aa0e1baacf188f95b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b2efdba9f4a1d90e34c9110851392d7
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

class PrimitiveOp_fd0298790299bc81ede9a7bd126b9b2e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('2048'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aa439c87bc98fbd2e2319db7d302e855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd0298790299bc81ede9a7bd126b9b2e
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

class PrimitiveOp_da80ed31bdf33faf6a9857ca18b2aefe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([2000], float('0'), paddle.float16, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a10435188590f738bcf90cdc5dc4b165(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da80ed31bdf33faf6a9857ca18b2aefe
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

class PrimitiveOp_35bbf7406b0b0cb9148a4f96cce86e18(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('20'), paddle.float16, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf72433829731b2032a542f4826be3c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35bbf7406b0b0cb9148a4f96cce86e18
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

class PrimitiveOp_57e0c8a4198b6b8a741489fcc9d5e9d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('8'), paddle.float16, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec4d3f224d110b2431ddbab31a5d30af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e0c8a4198b6b8a741489fcc9d5e9d4
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

class PrimitiveOp_355fd36548b6d817fb335b84e6643501(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 4], float('0'), paddle.float16, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c4187a1bc39abb63fb606728d256c02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_355fd36548b6d817fb335b84e6643501
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

class PrimitiveOp_d0554a71feb173a2140efa6e4475318f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 1, 2, 180, 320], float('0'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2518d012ee0df751ffda1b76344b4d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0554a71feb173a2140efa6e4475318f
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

class PrimitiveOp_15e520241fa35691c6fbcc1a07878fe8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.0441942'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0eb4bed0f0a0d74ecc850d8209f29920(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e520241fa35691c6fbcc1a07878fe8
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

class PrimitiveOp_ae56094bd4e8b632d6e00364ff3c408f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1600'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_12de04d9d072a86b2a07552000480ddf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae56094bd4e8b632d6e00364ff3c408f
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

class PrimitiveOp_808fcd9ebad67ad028bfebf2d45588d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1000], float('0'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_35533cecafa84eceba623ef6fd18c496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_808fcd9ebad67ad028bfebf2d45588d6
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

class PrimitiveOp_338d770fafbe1603df85eb9d1457c8d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('64'), paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7c25a2c4229f1d872ec7d2cfab7cc867(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_338d770fafbe1603df85eb9d1457c8d1
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

class PrimitiveOp_1b2c11065b9259d7e04748c3f830e493(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1000, 4], float('0'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_94f7120ab5f5483e31f816888464cf39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b2c11065b9259d7e04748c3f830e493
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

class PrimitiveOp_88ff73ba7afeed45614bab1159c2cc9d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 10], float('0'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b4f5f6e9d597e2e5ec3d667c80fc1c7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88ff73ba7afeed45614bab1159c2cc9d
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

class PrimitiveOp_ce04457864c8f99d28f87bb5ef7807e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 1, 1], float('0'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c555e13d0238bcfa9ea5f5c95e6b0df4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce04457864c8f99d28f87bb5ef7807e2
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

class PrimitiveOp_e5a8da703f1f67af75102a596a4c23cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('255'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_466219a63838871e6e9b295aa23e5efe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5a8da703f1f67af75102a596a4c23cd
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

class PrimitiveOp_ee218b5373d306e6c902cd80f29dc607(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('54'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_170ef17cdcf90cedda38eccdf3111e01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee218b5373d306e6c902cd80f29dc607
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

class PrimitiveOp_89dd8e992ff1b55b03b602a58f3cc09c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1000'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_95cf6cf4b6bc474e108f0fd67861704c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89dd8e992ff1b55b03b602a58f3cc09c
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

class PrimitiveOp_15fcd8374e72b31bc2531dddf2ad5056(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('180'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_117552fa0fc48cd635d96421e95a83ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15fcd8374e72b31bc2531dddf2ad5056
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

class PrimitiveOp_ba6775e2206fa93b6da07abad0fe85f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('400'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a6506d7caff7e02e6c2b2e7576389033(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba6775e2206fa93b6da07abad0fe85f3
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

class PrimitiveOp_65c21abed78c4d5285242da19c7125f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('4'), paddle.float16, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_facb0536c7021bc1db7ecd621a9ae9ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65c21abed78c4d5285242da19c7125f2
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

class PrimitiveOp_1d30164d0e66179593ce17738774c5eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('1000'), paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_74670dbc22833d5fc7aa6f4545a66966(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d30164d0e66179593ce17738774c5eb
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

class PrimitiveOp_207727900305fed320c0da9a0bf7e474(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('0.166667'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b1304e1642b62efcd22a4c5f996e298(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_207727900305fed320c0da9a0bf7e474
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

class PrimitiveOp_3fa83b43f9e29feee5f95df26cf0a3dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('0.01'), paddle.float16, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a4863e9ca9d62dd50435281df47936d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3fa83b43f9e29feee5f95df26cf0a3dd
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

class PrimitiveOp_dc81edb0f26e661792bf72655ee2a16a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('30'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_948ffa72a3ec067078ec0dce970017a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc81edb0f26e661792bf72655ee2a16a
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

class PrimitiveOp_8545297a7703bede5d5f449299b88da4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('2.14748e+09'), paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e86d99bb7fb1e82840b5ae18757bf080(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8545297a7703bede5d5f449299b88da4
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

class PrimitiveOp_88bbc2376842acf268af64d2c70094c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('0.01'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14ac61daa5a264a4e9eaf5fce568641b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88bbc2376842acf268af64d2c70094c0
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

class PrimitiveOp_a942de5809137167ce78996190f3648e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('0'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_87bc0e9588f2456b43abeb13f4838642(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a942de5809137167ce78996190f3648e
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

class PrimitiveOp_31f8dc51490f349998d9896dc49700fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 1, 1], float('-1'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_763925f0625c6651fbc5ce1d62474bca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31f8dc51490f349998d9896dc49700fc
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

class PrimitiveOp_fcdf6979f6a1aad4d3ba3e9159144111(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('17'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2d3d530cbdd6528e6b933a159acd6a2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fcdf6979f6a1aad4d3ba3e9159144111
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

class PrimitiveOp_59d7b2b3ced31fe973cd11757b1f158b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 64, 128, 256], float('0'), paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bcd10d58c2ca265cacb4cd1c33b82294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59d7b2b3ced31fe973cd11757b1f158b
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

class PrimitiveOp_32b8305ebc386f65165a8e17243b56cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 80, 28, 28], float('0'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef2dbb63f3d3f0760ec4b384dbc2c2ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32b8305ebc386f65165a8e17243b56cd
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

class PrimitiveOp_caee9b2d94e7982745bfb65eed27d44e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 256, 14, 14], float('0'), paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d5497fb760d3280bd246e400918cf6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_caee9b2d94e7982745bfb65eed27d44e
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

class PrimitiveOp_9c6f259487fd7aecefb720ebe1257d21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], float('1536'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_434e68ea0603c150af69f5cdf69dddb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c6f259487fd7aecefb720ebe1257d21
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

class PrimitiveOp_8a4c0a9ba102fd998070d3efc721852a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], float('512'), paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38c2036b7a50dddfe0d234497ae3557d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a4c0a9ba102fd998070d3efc721852a
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

class PrimitiveOp_0fa164ca39fa8d94ddeb06625cab8e3b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 64, 128, 256], float('0'), paddle.float16, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_428fc81c81f93cb6192e9931a5ffa750(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fa164ca39fa8d94ddeb06625cab8e3b
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

class PrimitiveOp_5761efab18dd20c1af17edd370ce959e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 48, 256, 512], float('0'), paddle.float16, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3905949f66245993292140aeaf39c9d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5761efab18dd20c1af17edd370ce959e
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