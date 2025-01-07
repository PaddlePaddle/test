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
class PrimitiveOp_e3198402da2b742230b9736bcad023da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [32, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77c4eee26fd78130bedbefd69e19519c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3198402da2b742230b9736bcad023da
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 100, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([512, 128, 32, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_006b5b98b706a326aa0b5eec8d00110e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 3, 3], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f004baaee1964ec219acc88934a7f770(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_006b5b98b706a326aa0b5eec8d00110e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 4, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 3, 1, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7e0d67bfac727891aeb70c8b1260e3e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [2, 3, 3], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2df322bdf34a11ec0bf076de0b74ee90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e0d67bfac727891aeb70c8b1260e3e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 32, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 3, 5, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f3d78f5aa88c6674a43896ed5cf24f48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [8, 1, 1], [2, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4d30ab3a6be1fbfbf53a8adeeeb505cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3d78f5aa88c6674a43896ed5cf24f48
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 32, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 8, 5, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c1cf31f62c30d384819c2a669733632d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b6de4f0dd179a6a5c7e279b470892acb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1cf31f62c30d384819c2a669733632d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 80, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bb82a6ef1eb4d809d246405f77b6d19b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1cf31f62c30d384819c2a669733632d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 80, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_92f3080fb4f4f7e6606a97d9f124e279(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_98fc27a946b8e768ebd86f89f13b81d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92f3080fb4f4f7e6606a97d9f124e279
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 4, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e9aadbf06a171f69dcff8be62076ab3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1cf31f62c30d384819c2a669733632d
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 4, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e45512509d304d4a0af9b91b60897d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1cf31f62c30d384819c2a669733632d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 4, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f7ac25beb020c46e363555fb835e198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1cf31f62c30d384819c2a669733632d
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 32, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 8, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c5bd00ba7b3546d5697590f2022337d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8344edd37319eb755ad8bc8f20329564(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5bd00ba7b3546d5697590f2022337d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 32, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 8, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_48a4d41c59d0965ecc2123776c189568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92f3080fb4f4f7e6606a97d9f124e279
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 32, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 8, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f48708a649cb851e5e8dbee6f5b55b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5bd00ba7b3546d5697590f2022337d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 32, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_27c304f46cf56f7cbc414d2c800e4f90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3d78f5aa88c6674a43896ed5cf24f48
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 32, 5, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_12ecba074b7759ba72c742299057b117(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0e63cc408769a670d12693b94666b56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12ecba074b7759ba72c742299057b117
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 4, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 320, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_78148f9cdb34674f1b93c71066b6d792(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1cf31f62c30d384819c2a669733632d
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 4, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 320, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9d44baac84ccfcc25dcff61c88b29981(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d2e52a34f8ec75641514a04968c68e8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d44baac84ccfcc25dcff61c88b29981
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_21a020d36e3b31d0cf160415cd75e2f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1cf31f62c30d384819c2a669733632d
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b4d423ddea627a111b9a768b130a9930(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1cf31f62c30d384819c2a669733632d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 512, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec148e4f2acee45b19bf6924aac8f13c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92f3080fb4f4f7e6606a97d9f124e279
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_487cea5d5eaa85afd3d8c6222ce712c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12ecba074b7759ba72c742299057b117
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 32, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2ee624154d051a636e0f98fc52aa18ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5bd00ba7b3546d5697590f2022337d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 32, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cefac97b69325d6490b98572d656233f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d44baac84ccfcc25dcff61c88b29981
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 32, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 16, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8ed5ac45fce786a05b227955a7add618(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1cf31f62c30d384819c2a669733632d
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 32, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_479d62df4839d925750ac364fbdd9376(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5bd00ba7b3546d5697590f2022337d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 64, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_82e452742709c8bffcd195c36608bc23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92f3080fb4f4f7e6606a97d9f124e279
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 32, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 16, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e8830730eb2be3b07e7c98c447df654c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3d78f5aa88c6674a43896ed5cf24f48
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 64, 5, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_59c2cf9b59188849f11c6bf193352a56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12ecba074b7759ba72c742299057b117
    def get_inputs(self):
        return [
            paddle.uniform([1, 640, 4, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 640, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_88131d8abfa4765a9b697ea6e5e76577(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5bd00ba7b3546d5697590f2022337d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 640, 4, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 640, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_690415a7305f4930a1fa4f2dddf4847d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d44baac84ccfcc25dcff61c88b29981
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 4, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_83fbcb69b87a795d5b9ff0bc78577b8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1cf31f62c30d384819c2a669733632d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 4, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f06e124f3eff7ac068f436d776344f51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5bd00ba7b3546d5697590f2022337d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 4, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0d29deb2e615df7b8798848ab91293c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92f3080fb4f4f7e6606a97d9f124e279
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 4, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8ec6462c4544a6cb16c80223e2184ed4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12ecba074b7759ba72c742299057b117
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 64, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_58257e2ab356f7c148d23fc40ae79a35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5bd00ba7b3546d5697590f2022337d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 64, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6d5361b234e123777c5bfbd714c4299c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d44baac84ccfcc25dcff61c88b29981
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07f523f5fcf01bff7fc283f4bbdda8d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1cf31f62c30d384819c2a669733632d
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e90e1c7d1cce8860be63971634f45eac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5bd00ba7b3546d5697590f2022337d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 128, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f320dbe7daf7cf8d77161d6d19612bc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92f3080fb4f4f7e6606a97d9f124e279
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_975e5905c2188f5b0848aa2761150618(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3d78f5aa88c6674a43896ed5cf24f48
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 5, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_472f9f10cd2b9e30daa641f2c43dc609(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12ecba074b7759ba72c742299057b117
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 4, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1280, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aeed3368e4a044d86d3a9c64a2a14630(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5bd00ba7b3546d5697590f2022337d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 4, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 1280, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_01e82bc70e0d3b302f2fbf09f0693cdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d44baac84ccfcc25dcff61c88b29981
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4626cf12c4e45e282fd5e4a63d73c05a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1cf31f62c30d384819c2a669733632d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 512, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6c2b42e5540bfff09e276823a3b4be76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5bd00ba7b3546d5697590f2022337d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 4, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 2048, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ee0a7523b5844202b8b35ed9fd74e703(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92f3080fb4f4f7e6606a97d9f124e279
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a512ac94445e2a0a349e546b6180c0fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12ecba074b7759ba72c742299057b117
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f7da408ee30c18311e0b786a60895a80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5bd00ba7b3546d5697590f2022337d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 128, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6d92bd7abaf534313105676210ace842(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d44baac84ccfcc25dcff61c88b29981
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_da29314f1d14ed6730d1b4c4ae50d839(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1cf31f62c30d384819c2a669733632d
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3a798569695dfa376954c5683bed2898(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5bd00ba7b3546d5697590f2022337d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d609e6ce398bd57726b27739d94ae485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92f3080fb4f4f7e6606a97d9f124e279
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cd218e611a8b9edf7362a037176a4c63(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 3, 3], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_08808731ac70aac767b51eacd11bf3b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd218e611a8b9edf7362a037176a4c63
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 4, 256, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([64, 3, 1, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5320b71006599eaeaa56d0233ef41a05(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [2, 3, 3], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_33c6bf431bc94507d3aad8758717114d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5320b71006599eaeaa56d0233ef41a05
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 32, 256, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([8, 3, 5, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_88ca3762adb71e6da46ac03b3b5102bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [8, 1, 1], [2, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e4f054c370513182cb86681f8ee8635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88ca3762adb71e6da46ac03b3b5102bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 32, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([16, 8, 5, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_369fd5b2876f89c93bb014cf364a0bc6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d1fba19067bbb8e247b2eb4ee917f57a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_369fd5b2876f89c93bb014cf364a0bc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([256, 80, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_63ebb93a464a64b413d8358f5fc79c02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_369fd5b2876f89c93bb014cf364a0bc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64, 80, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_54f1e3b5f5095e69948824df66dfcefd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d26b83e7bf1a0b12474edc8258cf759f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54f1e3b5f5095e69948824df66dfcefd
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 4, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64, 64, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e60337db64d1c735b953fe35ba19230f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_369fd5b2876f89c93bb014cf364a0bc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 4, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7aef8ced0327e67282b90e6fed7d2dfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_369fd5b2876f89c93bb014cf364a0bc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 4, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64, 256, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9ed0f82aa369febeda056ae3ff898b60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_369fd5b2876f89c93bb014cf364a0bc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 32, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([32, 8, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9d1ff7c863ce79b6ed0880829351549f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_969aabbb3afd8e50a63fdcd3c9db7180(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1ff7c863ce79b6ed0880829351549f
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 32, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([8, 8, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_355aa684f474ccb8b276ec7f4e018893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54f1e3b5f5095e69948824df66dfcefd
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 32, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([8, 8, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e11f419e392cdc2cf05bcc751f9c1464(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1ff7c863ce79b6ed0880829351549f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([8, 32, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5a78459b8c92cb21afdc2f9319bc317c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88ca3762adb71e6da46ac03b3b5102bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64, 32, 5, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_60a3df5a2d44007f59833a3cf16e2581(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fed1462d7a7d4a5214210d8df5adbd67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60a3df5a2d44007f59833a3cf16e2581
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 4, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([512, 320, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6de798117184dd8ecbba041a19797cb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_369fd5b2876f89c93bb014cf364a0bc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 4, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([128, 320, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5c3807994fe4aed646979a4a4890d349(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_575188547bcbb028623131e2984f7d51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c3807994fe4aed646979a4a4890d349
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([128, 128, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fa887eb9e98ce8793f821eb6146807c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_369fd5b2876f89c93bb014cf364a0bc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([512, 128, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c5c5b1307a189c3a5f0ff91db9db8e8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_369fd5b2876f89c93bb014cf364a0bc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([128, 512, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fa1de96b9f26721ed93165a37b48d9ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54f1e3b5f5095e69948824df66dfcefd
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([128, 128, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dcea25a9538b2c81d38b61418d899568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60a3df5a2d44007f59833a3cf16e2581
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64, 32, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_72b49230c92fa59e72083942b01695f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1ff7c863ce79b6ed0880829351549f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([16, 32, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6bae919f01bbb78004840e6f5755256b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c3807994fe4aed646979a4a4890d349
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 32, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([16, 16, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f6c3ebf156e28aa3f7c674b51e9e59e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_369fd5b2876f89c93bb014cf364a0bc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 32, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([64, 16, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec7eb13fdb3d8b8862c7fbd29a8d2eda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1ff7c863ce79b6ed0880829351549f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([16, 64, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2ddd4d59f2aa628644f333f776358e25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54f1e3b5f5095e69948824df66dfcefd
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 32, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([16, 16, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a9d729d1fcb3dae4c42b322f1fe7b8d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88ca3762adb71e6da46ac03b3b5102bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([128, 64, 5, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9638a5f1d29e86942de94ec43e7e14b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60a3df5a2d44007f59833a3cf16e2581
    def get_inputs(self):
        return [
            paddle.uniform([1, 640, 4, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([1024, 640, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_353047ce57fa8352cacf42cfcbf011d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1ff7c863ce79b6ed0880829351549f
    def get_inputs(self):
        return [
            paddle.uniform([1, 640, 4, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([256, 640, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f3a360e4c07a8ad2fc01204c9b739c30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c3807994fe4aed646979a4a4890d349
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 4, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c4c14eee60244cb37fd6cfe8feb7e827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_369fd5b2876f89c93bb014cf364a0bc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 4, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([1024, 256, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_87a2f39e2adde4bc21e20646c363e482(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1ff7c863ce79b6ed0880829351549f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 4, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([256, 1024, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73c30f501a4510e46b2cf67cc82efe2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54f1e3b5f5095e69948824df66dfcefd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 4, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bc42a405e91f24d0532d8dec0af04ed2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60a3df5a2d44007f59833a3cf16e2581
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([128, 64, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bd365e05d019ccf1f5f5e579a63c4a67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1ff7c863ce79b6ed0880829351549f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([32, 64, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_39d49ec5815c68bd8f4bc1c5448eee38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c3807994fe4aed646979a4a4890d349
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([32, 32, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_773a2cb7ff8e26796df05993542f0544(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_369fd5b2876f89c93bb014cf364a0bc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_67398327c4b76e2c50546400de33ff5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1ff7c863ce79b6ed0880829351549f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([32, 128, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9923ac37d31628fd9806dd225f1f3eea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54f1e3b5f5095e69948824df66dfcefd
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([32, 32, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7452f32f60cebafefd93ee05f6cbae9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88ca3762adb71e6da46ac03b3b5102bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([256, 128, 5, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1524e6e5b9c8843f80c6a39b90cb0221(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60a3df5a2d44007f59833a3cf16e2581
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 4, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([2048, 1280, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f2839eafd39c5061acc4eb3d26a5531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1ff7c863ce79b6ed0880829351549f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 4, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([512, 1280, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_977ed113ba0bec8cc0419a586f0e8588(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c3807994fe4aed646979a4a4890d349
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([512, 512, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aee1aee338bccfd908756a8d9f8cf81f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_369fd5b2876f89c93bb014cf364a0bc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4, 8, 8], dtype='float16', min=0, max=0.5),
            paddle.uniform([2048, 512, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4d331f592062d489d021816103b7415f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1ff7c863ce79b6ed0880829351549f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 4, 8, 8], dtype='float16', min=0, max=0.5),
            paddle.uniform([512, 2048, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1cdfb8e78c8808cc141ce5b2d3707432(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54f1e3b5f5095e69948824df66dfcefd
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4, 8, 8], dtype='float16', min=0, max=0.5),
            paddle.uniform([512, 512, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a4526106e4275d2e8da8b87569382fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60a3df5a2d44007f59833a3cf16e2581
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([256, 128, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f07aa5661747749b808886237dcea7d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1ff7c863ce79b6ed0880829351549f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([64, 128, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a7cf69dd912454a88e795ff7476b63ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c3807994fe4aed646979a4a4890d349
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([64, 64, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7b3cf50da83544fd1eb88dd57039888f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_369fd5b2876f89c93bb014cf364a0bc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 8, 8], dtype='float16', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_51c5055db2159013d1ff239f8fc0901d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1ff7c863ce79b6ed0880829351549f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 8, 8], dtype='float16', min=0, max=0.5),
            paddle.uniform([64, 256, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5425e063fb391389412427aecc34c192(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54f1e3b5f5095e69948824df66dfcefd
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 8, 8], dtype='float16', min=0, max=0.5),
            paddle.uniform([64, 64, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6ded225482d953b5824ae5f4d5faa90e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [32, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb430e325f061f10b4653a0704ff0e96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ded225482d953b5824ae5f4d5faa90e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 100, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_46da6dcbdf5f5683389094ea66db959e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [32, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, 100, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[512, 128, 32, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_23de55a6561c1ccbf8fb2df7f11369d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46da6dcbdf5f5683389094ea66db959e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 100, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([512, 128, 32, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1c8c1c9bad72ea18fe8164dd9e27b766(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 3, 3], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 4, 256, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 3, 1, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_53d95f651cfa54a6306bacc21151e26e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c8c1c9bad72ea18fe8164dd9e27b766
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 4, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 3, 1, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f1fa5313c044d256b253c76d200ebbbf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [2, 3, 3], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 32, 256, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[8, 3, 5, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ee2ba0b99769623ac2fce54bbb632cef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1fa5313c044d256b253c76d200ebbbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 32, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 3, 5, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_00a8caad68a6f7ad7da9e1fbf6837876(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [8, 1, 1], [2, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 32, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[16, 8, 5, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c72df0a50c43966ad2779952431a3821(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a8caad68a6f7ad7da9e1fbf6837876
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 32, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 8, 5, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5545c522f80130f47f87562ebd9f29d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, 4, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 80, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4ad1d4902241ceac03fa9581a5f69b14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5545c522f80130f47f87562ebd9f29d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 80, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_38faea778e1c095c7ac80742e6c5e1c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, 4, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 80, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b0c7d3fe849084ba1d3744b5d0a5a704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38faea778e1c095c7ac80742e6c5e1c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 80, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1bd7a1a6d0f3066614a10ea219ef6fe7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 4, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e2bfc5195ebefd6eb5199b9cdfc21dd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bd7a1a6d0f3066614a10ea219ef6fe7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 4, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_34f0c3fad62fa9da44e930705f4aeaef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 4, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 64, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1193ed367dee43371fa2d2f273d35001(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34f0c3fad62fa9da44e930705f4aeaef
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 4, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b9aae958e1f1be381e0b72dd031d6352(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 4, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 256, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_23fd964aeee154f7b5edae80ce819440(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9aae958e1f1be381e0b72dd031d6352
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 4, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5d5ca28cdf68ca7215d3a9e9aa2d3d5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 32, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 8, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eadc9e97317835a92ff93584a4ec6f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d5ca28cdf68ca7215d3a9e9aa2d3d5c
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 32, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 8, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_eb17db6a9ebab9b4784f01d53cede6e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 32, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[8, 8, 3, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f82896f1aaaf43aa43cfad1b60376f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb17db6a9ebab9b4784f01d53cede6e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 32, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 8, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_63acf6bb1bc6449afd7f4c3a898fe682(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 32, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[8, 8, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_39b6167e08a6b7dc62a0b5ae22bdeb6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63acf6bb1bc6449afd7f4c3a898fe682
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 32, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 8, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_32541a690827e35c3ae80cd5e2f8b83b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 32, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[8, 32, 3, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_12e52407807186fffc99d7580f21a18f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32541a690827e35c3ae80cd5e2f8b83b
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 32, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_70b06a50e3050a97da4d26c324921660(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [8, 1, 1], [2, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 32, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 32, 5, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a6bc09335240567c4ce0f4078790b77a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70b06a50e3050a97da4d26c324921660
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 32, 5, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_07202448b8b973b971c42fa0966a684a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 320, 4, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 320, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c4c28106d1fc2a2fc3719c354bde1be2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07202448b8b973b971c42fa0966a684a
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 4, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 320, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e245a9f418c601bf7358a0471ecee141(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 320, 4, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 320, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f06211df3779606d2265b3620965041f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e245a9f418c601bf7358a0471ecee141
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 4, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 320, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_57315b0d6dbd3483e2572957e3cfe80c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 4, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c3225a0089b13a2243c3c4f8a13776b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57315b0d6dbd3483e2572957e3cfe80c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_48d954134e2a3c894562652a3bb41ce9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 4, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 128, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f66252b851f4b6b3d8dc36e54d48ddcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48d954134e2a3c894562652a3bb41ce9
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_58b93f1b663db8889a069b953222f453(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 4, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 512, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_805195c751b4a2298a8d7db0d0da27fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58b93f1b663db8889a069b953222f453
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 512, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cbada0b8065c53161a3ac9ad154d5ff3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 4, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7c3b1b40e7dffa38083aea1e176bba16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbada0b8065c53161a3ac9ad154d5ff3
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_51e919c960f21d768f91e0c1bef80fb2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 32, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 32, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47fc7efbd7052334c5d7afde99679d31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51e919c960f21d768f91e0c1bef80fb2
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 32, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2faa519d7b348d24f7e8f6c793bb7c2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 32, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[16, 32, 3, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1f60878bd75c4b445d3ebcb5b8bede5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2faa519d7b348d24f7e8f6c793bb7c2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 32, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_25368d9654cbbdcba22b436008633ed8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 32, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[16, 16, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b8d9f1592aa67f72a4adaed590bf644b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25368d9654cbbdcba22b436008633ed8
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 32, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 16, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d7a5bb40b07a75b344edc830fa0e4eb5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 32, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 16, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8ef1b95366dd742d06b599e6df8560a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7a5bb40b07a75b344edc830fa0e4eb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 32, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_86f1c947e1fc1444dc742d7dfd95907d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 32, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[16, 64, 3, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a6061c24ac42266a727b7187bcec8f60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86f1c947e1fc1444dc742d7dfd95907d
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 64, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bd75c7cd55a61921100673ffd6778acd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 32, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[16, 16, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2d66e2a905ca79841b488fd493e7ce6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd75c7cd55a61921100673ffd6778acd
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 32, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 16, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e703cb3795fc47648ce2884882adeff6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [8, 1, 1], [2, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 32, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 64, 5, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b425f8b78758d6cc239489157fc5b0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e703cb3795fc47648ce2884882adeff6
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 64, 5, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e72df9809e27932952a1c399f907adce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 640, 4, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 640, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_302bf10d543142fe36a640e3ba9ac895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e72df9809e27932952a1c399f907adce
    def get_inputs(self):
        return [
            paddle.uniform([1, 640, 4, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 640, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1b4e5c0b7a97fda36495bb5348f5b121(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 640, 4, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 640, 3, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5927d4f5371fdfede3e1f47021857375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b4e5c0b7a97fda36495bb5348f5b121
    def get_inputs(self):
        return [
            paddle.uniform([1, 640, 4, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 640, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d78a58d9c03ef009b13aaec929394c46(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 4, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4bd114ad15585fa6aedfd45416ca746f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d78a58d9c03ef009b13aaec929394c46
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 4, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_107cae1e0df39b81388d034ad6e90d07(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 4, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 256, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_668ea71bf489de2ef86b555c83ecadfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_107cae1e0df39b81388d034ad6e90d07
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 4, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_06d412b07b436e0a2a463d59fd5f9f55(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, 4, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 1024, 3, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2236cb3ce9c516090516d6d32e8e9fff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06d412b07b436e0a2a463d59fd5f9f55
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 4, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_95faf26b3cdccfa96329afdaf7dc110e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 4, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e26d0cfef7b02eecc97ef85ab1c487aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95faf26b3cdccfa96329afdaf7dc110e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 4, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_27c955a2c98db9cdd7389e0a1721e534(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 32, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 64, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32c436a1259bb383cb1d437aa5a3a88d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c955a2c98db9cdd7389e0a1721e534
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 64, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_625a9c1a79126263de7d87011d462bff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 32, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 64, 3, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8af1e163855b9dd8fe791266a7a1b8fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_625a9c1a79126263de7d87011d462bff
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 64, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_25056092947d17f628801a8d39150154(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 32, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 32, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6c1f7b410561eafd3d79094f2da24de4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25056092947d17f628801a8d39150154
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b30c4503a3fa755b22681913b5ef3f06(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 32, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 32, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1e8a974a776d6e1a889d847b0cdfe768(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b30c4503a3fa755b22681913b5ef3f06
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6deeb80a884d27e723888e41cc62c0e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 32, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 128, 3, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_62964899871261c9bf4403759ac7c157(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6deeb80a884d27e723888e41cc62c0e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 128, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_15dad10f74ce122a06a31ea77726bfcd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 32, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 32, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e8ce0ce3fa5b232e06084bacee27d605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15dad10f74ce122a06a31ea77726bfcd
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6a4047ec437638b82513dfe300c43231(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [8, 1, 1], [2, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 32, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 128, 5, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5ad6a4b47dbf2314971c80f63de43919(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a4047ec437638b82513dfe300c43231
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 5, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a8c2c16750df709af2e59cd654baeb5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1280, 4, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[2048, 1280, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b7089e81153e3820a83d054bc5f6d989(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8c2c16750df709af2e59cd654baeb5c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 4, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1280, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_efd5f5656e1fe8e8e97c001a246ea0e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1280, 4, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 1280, 3, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_de5087a763f43ab5c88751a68c22a810(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efd5f5656e1fe8e8e97c001a246ea0e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 4, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 1280, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cf5bb85ae40f87d329803b5bed035418(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 4, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a2250300d84ed09679305fc4f142f112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf5bb85ae40f87d329803b5bed035418
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a4bf1d9ef3e493e972162e6364a86a30(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 4, 8, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[2048, 512, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f75b2597cbdd863918a95e55ca52de98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4bf1d9ef3e493e972162e6364a86a30
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 512, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1258acdbb0797853e38e3a6c23994fe3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2048, 4, 8, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 2048, 3, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c222f4a52516a9cca45b3214705e8ba4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1258acdbb0797853e38e3a6c23994fe3
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 4, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 2048, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1f8eb37c0db7d4f013a45ff23b73f72b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 4, 8, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7ae0e4764af65339ad2a96380f4212a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f8eb37c0db7d4f013a45ff23b73f72b
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0e2f26ef6397048d3b4c61e8ceb6dafe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 32, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 128, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_96fe6fd5bf797df753769c96a0da8cab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2f26ef6397048d3b4c61e8ceb6dafe
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_92fa627080fae1d2293d21d12a8a7671(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 32, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 128, 3, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a872151e5e8ae9ffd1d30d6e66a5e00d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92fa627080fae1d2293d21d12a8a7671
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 128, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4905a00aca8f89ea56c0222838327f25(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 32, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_385ae933eb9a0648a4668a7d80ccf403(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4905a00aca8f89ea56c0222838327f25
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_551919a9c84b038979d5ee7844b83f78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 32, 8, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 64, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32c028cd50c4fbab7e9a6e4a10ab74e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_551919a9c84b038979d5ee7844b83f78
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b64baafd1bc68799bed7db736305de6a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 32, 8, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 256, 3, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9fd524d438d9ab9960866849fbd9694e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b64baafd1bc68799bed7db736305de6a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 3, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_eb62471ffbbf18cd5c757d39bb6b4972(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 32, 8, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_23e1b4fe747c7fdfb6ffda58e2d5bb23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb62471ffbbf18cd5c757d39bb6b4972
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_484c3fc7ea8151f9cd6425eac470fc28(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 3, 3], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 4, 256, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[64, 3, 1, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2c10da86a8f7951aa2f1fc4a2f91f6a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_484c3fc7ea8151f9cd6425eac470fc28
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 4, 256, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([64, 3, 1, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a97c6f9f05395792a9d2b63425706ee6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [2, 3, 3], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 32, 256, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[8, 3, 5, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd4dc6dd9b182623a35d732a68de00f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a97c6f9f05395792a9d2b63425706ee6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 32, 256, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([8, 3, 5, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_662d96d7a569a1aad2d774e8e1a6a943(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [8, 1, 1], [2, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 32, 64, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[16, 8, 5, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_26d6c442f5041a69816037596a0b26aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_662d96d7a569a1aad2d774e8e1a6a943
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 32, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([16, 8, 5, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b1388078a63f513964a64713ca35a9b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, 4, 64, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[256, 80, 1, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_05c857626004b2e9bde3b0e694d5db13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1388078a63f513964a64713ca35a9b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([256, 80, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7119143ae19d0563b9a6f02d28e60918(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, 4, 64, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[64, 80, 1, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2bb6300b9787725fc762c8b003e1d7b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7119143ae19d0563b9a6f02d28e60918
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64, 80, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_dfb33d77b735fda01836f7dfa6da62ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 4, 64, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[64, 64, 1, 3, 3], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dc7e7f36fe3d6b3f69c09db241bc4916(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dfb33d77b735fda01836f7dfa6da62ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 4, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64, 64, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d48e8dd5db91a34a078763828c7e5c93(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 4, 64, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[256, 64, 1, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_766e7364c0e244b8cdd1b3d1a9d6f114(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d48e8dd5db91a34a078763828c7e5c93
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 4, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e2d308cb25f6d3d4828f3a3dcb4bc330(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 4, 64, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[64, 256, 1, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_17e4d4c283bfe1c0cb61aae2daa21e78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2d308cb25f6d3d4828f3a3dcb4bc330
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 4, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64, 256, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fa1e9c23056b47b7fa0dc660138c0bbb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 32, 64, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[32, 8, 1, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e88c35feeea22892461861ff37b21489(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa1e9c23056b47b7fa0dc660138c0bbb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 32, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([32, 8, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_361c3c04c615ed5d286e44fc719072fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 32, 64, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[8, 8, 3, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2b37d7ae4fc6e6b5f954f97f121e4940(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_361c3c04c615ed5d286e44fc719072fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 32, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([8, 8, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cb5eaf610724234ee9644d6c951e9255(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 32, 64, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[8, 8, 1, 3, 3], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_99499bbc99290f6c1f949f6f49b5c998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb5eaf610724234ee9644d6c951e9255
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 32, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([8, 8, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0a988200bb1407e0cd79d970c79ce3ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 32, 64, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[8, 32, 3, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_852b7c7d046a526d1ebdbe5f4d4c57d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a988200bb1407e0cd79d970c79ce3ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([8, 32, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_80d52dbd99d1fb1d810e37d2c3ba7777(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [8, 1, 1], [2, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 32, 64, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[64, 32, 5, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9b96154d08ce8ef4a28082a39d46eed2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80d52dbd99d1fb1d810e37d2c3ba7777
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64, 32, 5, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3eeed9c1bd21501ba374ee13ea170d7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 320, 4, 64, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[512, 320, 1, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f732aab4ae780ddc09ee63dfb681d3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3eeed9c1bd21501ba374ee13ea170d7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 4, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([512, 320, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_53107a4faf9567179af35a1a14e47523(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 320, 4, 64, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[128, 320, 1, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bebdf2e9273056c0641bc42c6896995e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53107a4faf9567179af35a1a14e47523
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 4, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([128, 320, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9f72955788f852914229a5f3b7823db9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 4, 64, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[128, 128, 1, 3, 3], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cd28a283d3a00eaae0d68809b7f07098(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f72955788f852914229a5f3b7823db9
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([128, 128, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_db2f46347557dcd82d6e69e087a4422d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 4, 32, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[512, 128, 1, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_17da5e030c2941198fb680571bcbba21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db2f46347557dcd82d6e69e087a4422d
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([512, 128, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_38ca4b487d71731f1865a1dd8f322edf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 4, 32, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[128, 512, 1, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a7053abb846541586146fbb4076cb56b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38ca4b487d71731f1865a1dd8f322edf
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([128, 512, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5b15965a243f71dfff3e0d34b6a9ef0c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 4, 32, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[128, 128, 1, 3, 3], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68dbe56c3f998a7aff309f1aea0915f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b15965a243f71dfff3e0d34b6a9ef0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([128, 128, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a5437b6fbbb6c470ea9b82d3e892fa9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 32, 64, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[64, 32, 1, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_baf4e36b8e400ed4343202c3233c5784(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5437b6fbbb6c470ea9b82d3e892fa9b
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64, 32, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_888645844ec20c53c6608ad7a3426da8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 32, 64, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[16, 32, 3, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_873693b55ff90010807d221a7ebb9bca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_888645844ec20c53c6608ad7a3426da8
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([16, 32, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cc769b00d5c2d10f7b395358c5533f67(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 32, 64, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[16, 16, 1, 3, 3], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f8c751126117006b10b506d02dfb2dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc769b00d5c2d10f7b395358c5533f67
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 32, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([16, 16, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_619cbc363afc96dab76c9e04a04744e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 32, 32, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[64, 16, 1, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3fa1c82ca92f3fecca3a32ca4955fe88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_619cbc363afc96dab76c9e04a04744e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 32, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([64, 16, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bac6b6db15967b1b2b7c09cca8038fac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 32, 32, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[16, 64, 3, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_86bfa0b7200533227f576df9b9d09921(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bac6b6db15967b1b2b7c09cca8038fac
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([16, 64, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_767ba2bca7be000d53093df4197b99e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 32, 32, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[16, 16, 1, 3, 3], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_805c7061a875e5e1768744b2047d3dd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_767ba2bca7be000d53093df4197b99e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 32, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([16, 16, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_882d5e8f4ed8c2e6c5b36aef6ec566e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [8, 1, 1], [2, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 32, 32, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[128, 64, 5, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_628eedd320982476a9c7aed8ae416aa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_882d5e8f4ed8c2e6c5b36aef6ec566e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([128, 64, 5, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b9504d865bc294afc05f42d7e3799195(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 640, 4, 32, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[1024, 640, 1, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4e692dc5f60dd9ddf6979da6db5774ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9504d865bc294afc05f42d7e3799195
    def get_inputs(self):
        return [
            paddle.uniform([1, 640, 4, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([1024, 640, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_04be1010c3d13eaa94848ec62357326e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 640, 4, 32, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[256, 640, 3, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11595e251726b0483e551f75a2ad9435(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04be1010c3d13eaa94848ec62357326e
    def get_inputs(self):
        return [
            paddle.uniform([1, 640, 4, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([256, 640, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1b715fa35d195813d857af33d69f7925(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 4, 32, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[256, 256, 1, 3, 3], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bbdbe6a21ef9070cc878003980d01692(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b715fa35d195813d857af33d69f7925
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 4, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d13a46ab7f9141f69e033e7571d553b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 4, 16, 16], dtype='float16'),
            paddle.static.InputSpec(shape=[1024, 256, 1, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_adb342d289d585baceca4df2d6996ab4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d13a46ab7f9141f69e033e7571d553b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 4, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([1024, 256, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_52d2ffc30a920966eb49555755f0f54e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, 4, 16, 16], dtype='float16'),
            paddle.static.InputSpec(shape=[256, 1024, 3, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8250f8dee027cd2287a34683dd646d66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52d2ffc30a920966eb49555755f0f54e
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 4, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([256, 1024, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f8106c82542e37174ff331d183e98d05(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 4, 16, 16], dtype='float16'),
            paddle.static.InputSpec(shape=[256, 256, 1, 3, 3], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5387878472df9edc3dff55997e0478fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8106c82542e37174ff331d183e98d05
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 4, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_76f52c5a7bf8901eaaa98b1bc978ae69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 32, 32, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[128, 64, 1, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4b53c99c001228ca8a82630e60f0d39f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76f52c5a7bf8901eaaa98b1bc978ae69
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([128, 64, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f4a35550398d2147372598474dce394a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 32, 32, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[32, 64, 3, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef8f0cbd3c46fbbe3d1548c14916c496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a35550398d2147372598474dce394a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([32, 64, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f70d161af7ea965b7e334d3c932ea7f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 32, 32, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[32, 32, 1, 3, 3], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2cfb5ac7be2fa9dea89384c5a1d653c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f70d161af7ea965b7e334d3c932ea7f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([32, 32, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b11bca3e52e8c9fbbb3ed443318e8459(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 32, 16, 16], dtype='float16'),
            paddle.static.InputSpec(shape=[128, 32, 1, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_791595f1339aeb98d8a70a5db962cc53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b11bca3e52e8c9fbbb3ed443318e8459
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7ce369412f06a55f120cefaccc7fe630(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 32, 16, 16], dtype='float16'),
            paddle.static.InputSpec(shape=[32, 128, 3, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4004dfd8bbbaf215d8c6829e28aeabc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce369412f06a55f120cefaccc7fe630
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([32, 128, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6682afba19a1ff01cf6fdcfdc096bbfd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 32, 16, 16], dtype='float16'),
            paddle.static.InputSpec(shape=[32, 32, 1, 3, 3], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8716ed7a23ae10682869f3d4adda2f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6682afba19a1ff01cf6fdcfdc096bbfd
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([32, 32, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8cfd5ccfc4550fc292820530559601ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [8, 1, 1], [2, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 32, 16, 16], dtype='float16'),
            paddle.static.InputSpec(shape=[256, 128, 5, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ddf8d9aa6c9191e4074b5641c52c46f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cfd5ccfc4550fc292820530559601ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([256, 128, 5, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ecfa2bb7d3935236f4297fbdf1d6cde6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1280, 4, 16, 16], dtype='float16'),
            paddle.static.InputSpec(shape=[2048, 1280, 1, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4bbb8bbf0cd311f0418700271f3fa596(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecfa2bb7d3935236f4297fbdf1d6cde6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 4, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([2048, 1280, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_62e25c398463d78de0459b44ce7fa1c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1280, 4, 16, 16], dtype='float16'),
            paddle.static.InputSpec(shape=[512, 1280, 3, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_39a50923d16876a8c8aa0fa294f1c724(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e25c398463d78de0459b44ce7fa1c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 4, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([512, 1280, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_87bc59ea3986984039b8c1ac05a1d351(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 4, 16, 16], dtype='float16'),
            paddle.static.InputSpec(shape=[512, 512, 1, 3, 3], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3098e0894a7c7f920340434f242546ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87bc59ea3986984039b8c1ac05a1d351
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([512, 512, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ba937c803540e0fe3e0ded83f8da4245(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 4, 8, 8], dtype='float16'),
            paddle.static.InputSpec(shape=[2048, 512, 1, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d0317c98729547c448f8948d9cbb9e46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba937c803540e0fe3e0ded83f8da4245
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4, 8, 8], dtype='float16', min=0, max=0.5),
            paddle.uniform([2048, 512, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fcebb066808e7bc1f98fb0b8f152be22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2048, 4, 8, 8], dtype='float16'),
            paddle.static.InputSpec(shape=[512, 2048, 3, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4086ad38117c4234708cc6239863f288(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fcebb066808e7bc1f98fb0b8f152be22
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 4, 8, 8], dtype='float16', min=0, max=0.5),
            paddle.uniform([512, 2048, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a3842a4f1bac177127a2ce324ee8b505(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 4, 8, 8], dtype='float16'),
            paddle.static.InputSpec(shape=[512, 512, 1, 3, 3], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_de379e5301cf4e56eb409ff253cec12c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3842a4f1bac177127a2ce324ee8b505
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4, 8, 8], dtype='float16', min=0, max=0.5),
            paddle.uniform([512, 512, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b93aece2d777e886eaaf0ca55554e143(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 32, 16, 16], dtype='float16'),
            paddle.static.InputSpec(shape=[256, 128, 1, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2a85acbffb1bf0bf2f2d8ec2cb9986c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93aece2d777e886eaaf0ca55554e143
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([256, 128, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_18fb23a5747625fa6d9e55d3b51cb458(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 32, 16, 16], dtype='float16'),
            paddle.static.InputSpec(shape=[64, 128, 3, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e084f967ec6ef7fb0d1d2a01f871f989(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18fb23a5747625fa6d9e55d3b51cb458
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([64, 128, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_789d19b7a6d493bbae5871c00b3b83b0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 2, 2], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 32, 16, 16], dtype='float16'),
            paddle.static.InputSpec(shape=[64, 64, 1, 3, 3], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b76d106db558303eb348b0851533d185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_789d19b7a6d493bbae5871c00b3b83b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.uniform([64, 64, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d30bce5696c7a68cfbd014b6c8744d33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 32, 8, 8], dtype='float16'),
            paddle.static.InputSpec(shape=[256, 64, 1, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c3ca8ebfeea2d6be62bb7b1f52e3f9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d30bce5696c7a68cfbd014b6c8744d33
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 8, 8], dtype='float16', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b34465a9f49d6fe421d29561195bd90b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [1, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 32, 8, 8], dtype='float16'),
            paddle.static.InputSpec(shape=[64, 256, 3, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7552b13c5a275814c9e970ebe3351c7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b34465a9f49d6fe421d29561195bd90b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 8, 8], dtype='float16', min=0, max=0.5),
            paddle.uniform([64, 256, 3, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0a7d45d1e19d37c49cdb3b8b52e7af1a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [1, 1, 1], [0, 1, 1], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 32, 8, 8], dtype='float16'),
            paddle.static.InputSpec(shape=[64, 64, 1, 3, 3], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_23ec7aa547acd1b8cad994f3934ce960(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a7d45d1e19d37c49cdb3b8b52e7af1a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 8, 8], dtype='float16', min=0, max=0.5),
            paddle.uniform([64, 64, 1, 3, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_21e0c118ed07a0dd50a9f60f9dfe4c72(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.conv3d(input_0, input_1, [32, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, 100, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 128, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e3235036ecef9b23e5d2865c582e57fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21e0c118ed07a0dd50a9f60f9dfe4c72
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 100, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


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