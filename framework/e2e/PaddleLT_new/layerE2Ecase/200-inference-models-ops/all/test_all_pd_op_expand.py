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
class PrimitiveOp_da7411ae6f86e433c921c8c98cbdce93(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([3], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([2], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ae569f5c2cf6ad7d7f2d98cae1f7273(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da7411ae6f86e433c921c8c98cbdce93
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2541094422340393, 0.14051038026809692], [0.11752574145793915, 0.19737613201141357], [0.3536222279071808, 0.37550926208496094]], dtype='float32').reshape([3, 2]),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
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

class PrimitiveOp_d85f2e75678eaedaf6258b088fb64044(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_835e3ca03b11ccd42b7c94a52c4b50dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d85f2e75678eaedaf6258b088fb64044
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 384], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c528b7b557ade8187a93c041a5b88b9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d85f2e75678eaedaf6258b088fb64044
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 768], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_06bc7ad7f8c354364143cd565486227d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([26], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e45c868d2057e057abe47ba0ffc28864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06bc7ad7f8c354364143cd565486227d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]], dtype='int64').reshape([1, 26]),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([26], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3701a4a088b26f46b4c481115b5a5f03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array(100, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array(168, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2fd3340a351ab68dc0ef5438ccc2a48e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3701a4a088b26f46b4c481115b5a5f03
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 168], dtype='int64'), 'int64'),
            paddle.to_tensor(100, dtype='int32').reshape([]),
            paddle.to_tensor(168, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1bf74f0c47d066ccffc77131e06e3946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3701a4a088b26f46b4c481115b5a5f03
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[100, 1], dtype='int64'), 'int64'),
            paddle.to_tensor(100, dtype='int32').reshape([]),
            paddle.to_tensor(168, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ab1224bd4cfb389d192edc0703b958cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array(50, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array(84, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8a0aa779f5b1b61f4d0ff5dfd463692f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab1224bd4cfb389d192edc0703b958cd
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 84], dtype='int64'), 'int64'),
            paddle.to_tensor(50, dtype='int32').reshape([]),
            paddle.to_tensor(84, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4ce84f0d978c95cde4efec25de8a7611(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab1224bd4cfb389d192edc0703b958cd
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[50, 1], dtype='int64'), 'int64'),
            paddle.to_tensor(50, dtype='int32').reshape([]),
            paddle.to_tensor(84, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_830e3d01d68b21bbbaa3311c89bb3ffb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array(25, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array(42, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_64de05433420a4a47dfe47717cb838c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_830e3d01d68b21bbbaa3311c89bb3ffb
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 42], dtype='int64'), 'int64'),
            paddle.to_tensor(25, dtype='int32').reshape([]),
            paddle.to_tensor(42, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f42e2751b1f5069169522424627cc5dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_830e3d01d68b21bbbaa3311c89bb3ffb
    def get_inputs(self):
        return [
            paddle.to_tensor([[0], [32], [64], [96], [128], [160], [192], [224], [256], [288], [320], [352], [384], [416], [448], [480], [512], [544], [576], [608], [640], [672], [704], [736], [768]], dtype='int64').reshape([25, 1]),
            paddle.to_tensor(25, dtype='int32').reshape([]),
            paddle.to_tensor(42, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e626c3c032ffc691ae401b4c82624ec7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array(13, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array(21, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c61e55682579a31326d749bb18848cd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e626c3c032ffc691ae401b4c82624ec7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280]], dtype='int64').reshape([1, 21]),
            paddle.to_tensor(13, dtype='int32').reshape([]),
            paddle.to_tensor(21, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_212ddbb5329bd85720c70adbd41fc390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e626c3c032ffc691ae401b4c82624ec7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0], [64], [128], [192], [256], [320], [384], [448], [512], [576], [640], [704], [768]], dtype='int64').reshape([13, 1]),
            paddle.to_tensor(13, dtype='int32').reshape([]),
            paddle.to_tensor(21, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fe37e3bff3705701357a2d1ca91b5ee6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array(7, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array(11, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b90e526fd2dd508ef8b7dbb51f380b7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe37e3bff3705701357a2d1ca91b5ee6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280]], dtype='int64').reshape([1, 11]),
            paddle.to_tensor(7, dtype='int32').reshape([]),
            paddle.to_tensor(11, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ccd4ca862592b925e4b55cb6f130aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe37e3bff3705701357a2d1ca91b5ee6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0], [128], [256], [384], [512], [640], [768]], dtype='int64').reshape([7, 1]),
            paddle.to_tensor(7, dtype='int32').reshape([]),
            paddle.to_tensor(11, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_653aadb0c610cf7454f9d95b088f954e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b6f51376cdd7a98fd46ba4b39f5985df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_653aadb0c610cf7454f9d95b088f954e
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d4112095a88f161fada201ae46b46275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_653aadb0c610cf7454f9d95b088f954e
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f701ec8a598a929d9057843424d9f7af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 80, 80, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a343ea37d163ab32ea385b0b162dc227(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f701ec8a598a929d9057843424d9f7af
    def get_inputs(self):
        return [
            paddle.uniform([80, 80, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 3, 80, 80, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ffac6b51a91ac77a972489bd2dbf75cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 80, 80, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c5e1cead24dcc9a75f3cc4e917abad94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffac6b51a91ac77a972489bd2dbf75cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.1090087890625, 0.27099609375]]], [[[0.35498046875, 0.097412109375]]], [[[0.0457763671875, 0.383544921875]]]]], dtype='float16').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 80, 80, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2c0bb69cd75aa2a0fb18c0f02fb9af6e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 40, 40, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f05dcc6110708adc21a21dc26f70ab7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c0bb69cd75aa2a0fb18c0f02fb9af6e
    def get_inputs(self):
        return [
            paddle.uniform([40, 40, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 3, 40, 40, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_65e82737129f5ff1d5edec467c511c55(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 40, 40, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2c2f52387acfc4c977bd23166a33b559(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65e82737129f5ff1d5edec467c511c55
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.083740234375, 0.464599609375]]], [[[0.17041015625, 0.462890625]]], [[[0.0198974609375, 0.40380859375]]]]], dtype='float16').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 40, 40, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9ad1122bd8e713189fdec2e0e12377c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 20, 20, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_269cc63e2f72a7a395dbb30078221b59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ad1122bd8e713189fdec2e0e12377c2
    def get_inputs(self):
        return [
            paddle.uniform([20, 20, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 3, 20, 20, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8eba4165a62fb3bb41200aba6f02de32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 20, 20, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a33a15be04c0c00051451c1fe395b51c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8eba4165a62fb3bb41200aba6f02de32
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.037811279296875, 0.1580810546875]]], [[[0.058929443359375, 0.175048828125]]], [[[0.01407623291015625, 0.3212890625]]]]], dtype='float16').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 20, 20, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_aea5b74822a641a85c6867f77afb1b66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 80, 80, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7e2859887940faa5feabe6e76caa2dc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aea5b74822a641a85c6867f77afb1b66
    def get_inputs(self):
        return [
            paddle.uniform([80, 80, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 3, 80, 80, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ee21004f7cca4e47442bb7ec98ee4585(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 80, 80, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5a379ee4b385bfda758f6bc95dd5d812(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee21004f7cca4e47442bb7ec98ee4585
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.2704720199108124, 0.07836049795150757]]], [[[0.05318314954638481, 0.18742601573467255]]], [[[0.016026077792048454, 0.07621552050113678]]]]], dtype='float32').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 80, 80, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_49c335aee2f5e66b99c7439ccada3063(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 40, 40, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a6dc8b4aa79950087ff7a6c1e713afc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49c335aee2f5e66b99c7439ccada3063
    def get_inputs(self):
        return [
            paddle.uniform([40, 40, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 3, 40, 40, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3057e0955b51dcd688b2e0c49690fd9d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 40, 40, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d9c900dc1845277ef3154fa945384d03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3057e0955b51dcd688b2e0c49690fd9d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.00928601622581482, 0.40804237127304077]]], [[[0.4149647355079651, 0.14376626908779144]]], [[[0.4760911762714386, 0.4059772193431854]]]]], dtype='float32').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 40, 40, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5f8e349549466c8fbac20c02609943f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 20, 20, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9131fc61bc15e9f69b8d4d40a9ff428f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f8e349549466c8fbac20c02609943f3
    def get_inputs(self):
        return [
            paddle.uniform([20, 20, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 3, 20, 20, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_375e542547f226333b7f36e4c07566a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 20, 20, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a318d749d5b2a5ca7ef2a8c2888aaa68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_375e542547f226333b7f36e4c07566a8
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.1855674833059311, 0.28713515400886536]]], [[[0.013776411302387714, 0.49110886454582214]]], [[[0.001819793600589037, 0.2800101637840271]]]]], dtype='float32').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 20, 20, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e598280f71476eef41506e8304217307(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([3], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([2], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f73c3faf88a5cc1e1499b61fd9d2ab19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e598280f71476eef41506e8304217307
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1458740234375, 0.1572265625], [0.092529296875, 0.08453369140625], [0.0440673828125, 0.27197265625]], dtype='float16').reshape([3, 2]),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
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

class PrimitiveOp_cac61380137923b5a8c8fe3ce66aab3f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([3], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([2], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_baeed896a7a2135dd7adac15171145f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cac61380137923b5a8c8fe3ce66aab3f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2541094422340393, 0.14051038026809692], [0.11752574145793915, 0.19737613201141357], [0.3536222279071808, 0.37550926208496094]], dtype='float32').reshape([3, 2]),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
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

class PrimitiveOp_7f6062fb530a190b11c93b8785a850fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 384], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5951704ba40e7876295dfd3dae9168b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f6062fb530a190b11c93b8785a850fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 384], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8fd3607e0074cf5ba3ae6281af6f292a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 768], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea62744834286431e80d8f19613ca712(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fd3607e0074cf5ba3ae6281af6f292a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 768], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_eeed62168899c06f183f169287247be4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([26], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 26], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7b23440d20e2f3f2d38aaa0aaa7e7939(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeed62168899c06f183f169287247be4
    def get_inputs(self):
        return [
            paddle.to_tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]], dtype='int64').reshape([1, 26]),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([26], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0815988b584cc6f62e5379287fc68426(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array(100, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array(168, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_753d2da1d91bd9e23e5f38305b08aca7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0815988b584cc6f62e5379287fc68426
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 168], dtype='int64'), 'int64'),
            paddle.to_tensor(100, dtype='int32').reshape([]),
            paddle.to_tensor(168, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_351b205e4c30d06037163323e807eff5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array(100, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array(168, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c2384a1fc4849bc78c64a0e2b48f3d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_351b205e4c30d06037163323e807eff5
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[100, 1], dtype='int64'), 'int64'),
            paddle.to_tensor(100, dtype='int32').reshape([]),
            paddle.to_tensor(168, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3750a4a60b7e45434a7484abfffefca4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array(50, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array(84, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a60fbc929acc9743db4c00dcf13a41ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3750a4a60b7e45434a7484abfffefca4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 84], dtype='int64'), 'int64'),
            paddle.to_tensor(50, dtype='int32').reshape([]),
            paddle.to_tensor(84, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c3eda226ca53aa0876c979603c214ba7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array(50, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array(84, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_116fc1e3fb9dc3967c2c2434d4cb296d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3eda226ca53aa0876c979603c214ba7
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[50, 1], dtype='int64'), 'int64'),
            paddle.to_tensor(50, dtype='int32').reshape([]),
            paddle.to_tensor(84, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8face9c9d0f224b9ea55444b9220abb5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array(25, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array(42, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f3180924c9392768d8aebdcb036e451(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8face9c9d0f224b9ea55444b9220abb5
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 42], dtype='int64'), 'int64'),
            paddle.to_tensor(25, dtype='int32').reshape([]),
            paddle.to_tensor(42, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_94275548893e85ff6b69e0b8ccea7eea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array(25, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array(42, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e29213b0edeefe7363027a8baac5e1ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94275548893e85ff6b69e0b8ccea7eea
    def get_inputs(self):
        return [
            paddle.to_tensor([[0], [32], [64], [96], [128], [160], [192], [224], [256], [288], [320], [352], [384], [416], [448], [480], [512], [544], [576], [608], [640], [672], [704], [736], [768]], dtype='int64').reshape([25, 1]),
            paddle.to_tensor(25, dtype='int32').reshape([]),
            paddle.to_tensor(42, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_785fe356d6930aaedbfec7f9008a69a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array(13, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array(21, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0388ced0a667135ea66b4ad200c95ca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_785fe356d6930aaedbfec7f9008a69a8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280]], dtype='int64').reshape([1, 21]),
            paddle.to_tensor(13, dtype='int32').reshape([]),
            paddle.to_tensor(21, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0187636bb06f040ab1d83d8a070698e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array(13, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array(21, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_87d311a151694f333b63a63cab266394(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0187636bb06f040ab1d83d8a070698e1
    def get_inputs(self):
        return [
            paddle.to_tensor([[0], [64], [128], [192], [256], [320], [384], [448], [512], [576], [640], [704], [768]], dtype='int64').reshape([13, 1]),
            paddle.to_tensor(13, dtype='int32').reshape([]),
            paddle.to_tensor(21, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8f517b43f69112852b2a581eca32c99e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array(7, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array(11, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e231f94fde9af74238082a135167639d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f517b43f69112852b2a581eca32c99e
    def get_inputs(self):
        return [
            paddle.to_tensor([[0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280]], dtype='int64').reshape([1, 11]),
            paddle.to_tensor(7, dtype='int32').reshape([]),
            paddle.to_tensor(11, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8153a9ea093dfeed6a617bfed419a71d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array(7, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array(11, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fd5d08246fdec3ef6a18d4d0a5455d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8153a9ea093dfeed6a617bfed419a71d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0], [128], [256], [384], [512], [640], [768]], dtype='int64').reshape([7, 1]),
            paddle.to_tensor(7, dtype='int32').reshape([]),
            paddle.to_tensor(11, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e3a38e17d38cf0c57f77de3adac19b30(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ee99e5d1cc276ba9d470bd302778a06a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3a38e17d38cf0c57f77de3adac19b30
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_831159e7c1875b5d21e8977585cdf666(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c956256432ba65f3bdf45c448b28994(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_831159e7c1875b5d21e8977585cdf666
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6bdce6b84bfd67002cf04115b98a5369(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 80, 80, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80, 80, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_31180049b1595e94f0319f5afc7332ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bdce6b84bfd67002cf04115b98a5369
    def get_inputs(self):
        return [
            paddle.uniform([80, 80, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 3, 80, 80, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_60bdd1c1ad4a1da7abcb6ae8962feb89(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 80, 80, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 1, 1, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a4eec4dfc7e695546e841ee2b6e22692(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bdd1c1ad4a1da7abcb6ae8962feb89
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.1090087890625, 0.27099609375]]], [[[0.35498046875, 0.097412109375]]], [[[0.0457763671875, 0.383544921875]]]]], dtype='float16').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 80, 80, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cc3a7fa1f6597a9d865009233b5588b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 40, 40, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40, 40, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_559bd8186c2182c9ed52f90a28a3b19b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc3a7fa1f6597a9d865009233b5588b5
    def get_inputs(self):
        return [
            paddle.uniform([40, 40, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 3, 40, 40, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2a85327b217add3a90bfccdb1e71f0f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 40, 40, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 1, 1, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a2cbc8e07bfb89bb86c67c6dabaecaad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a85327b217add3a90bfccdb1e71f0f7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.083740234375, 0.464599609375]]], [[[0.17041015625, 0.462890625]]], [[[0.0198974609375, 0.40380859375]]]]], dtype='float16').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 40, 40, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1c42319fbe57fddaf3568ae679d76a8a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 20, 20, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 20, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1dcefd634df17f6e347548385c5a9fa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c42319fbe57fddaf3568ae679d76a8a
    def get_inputs(self):
        return [
            paddle.uniform([20, 20, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 3, 20, 20, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9783df7d56d0d4bdb33e509130bb1b73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 20, 20, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 1, 1, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4374319ba8ba8d6f3890921c6a27f554(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9783df7d56d0d4bdb33e509130bb1b73
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.037811279296875, 0.1580810546875]]], [[[0.058929443359375, 0.175048828125]]], [[[0.01407623291015625, 0.3212890625]]]]], dtype='float16').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 20, 20, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4a81d798c54b24f330fc800fee557d31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 80, 80, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80, 80, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_111aa266cc5dd64605acb7846be2d9ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a81d798c54b24f330fc800fee557d31
    def get_inputs(self):
        return [
            paddle.uniform([80, 80, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 3, 80, 80, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_51a703effd6103998d40491866e9d675(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 80, 80, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 1, 1, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_78ae9559f72e1972c8a04f2f9270a6b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51a703effd6103998d40491866e9d675
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.2704720199108124, 0.07836049795150757]]], [[[0.05318314954638481, 0.18742601573467255]]], [[[0.016026077792048454, 0.07621552050113678]]]]], dtype='float32').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 80, 80, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d9ff143e88931cc7fb0b952581ff1de6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 40, 40, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40, 40, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_922275da1e4ae3a338403d2854a843c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9ff143e88931cc7fb0b952581ff1de6
    def get_inputs(self):
        return [
            paddle.uniform([40, 40, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 3, 40, 40, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_281a3dea7e0f1eb8358122ab9b25d10e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 40, 40, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 1, 1, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e3d0f40b744af234b7e233bbf3ceedc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_281a3dea7e0f1eb8358122ab9b25d10e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.00928601622581482, 0.40804237127304077]]], [[[0.4149647355079651, 0.14376626908779144]]], [[[0.4760911762714386, 0.4059772193431854]]]]], dtype='float32').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 40, 40, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_da75383804267717813edb3dc1903e62(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 20, 20, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 20, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_215864605da4de03c1e3d595b915be2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da75383804267717813edb3dc1903e62
    def get_inputs(self):
        return [
            paddle.uniform([20, 20, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 3, 20, 20, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_62b8c045348ee71c475646ecabe0a43c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 20, 20, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 1, 1, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a45a957f95d4e74da82e00e2a64e8233(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b8c045348ee71c475646ecabe0a43c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.1855674833059311, 0.28713515400886536]]], [[[0.013776411302387714, 0.49110886454582214]]], [[[0.001819793600589037, 0.2800101637840271]]]]], dtype='float32').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 20, 20, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_912c7e8fc0dcee8c183b5b5a96b546e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([3], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([2], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, [x.reshape([]) for x in input_1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d0a46be0189390fa5cd9bb3c23a703d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_912c7e8fc0dcee8c183b5b5a96b546e8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1458740234375, 0.1572265625], [0.092529296875, 0.08453369140625], [0.0440673828125, 0.27197265625]], dtype='float16').reshape([3, 2]),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
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


if __name__ == '__main__':
    unittest.main()