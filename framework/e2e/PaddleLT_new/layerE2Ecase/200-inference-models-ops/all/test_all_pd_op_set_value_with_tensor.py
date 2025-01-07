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
class PrimitiveOp_b95ea6dc5d2f4684da7451afbc08f5d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2_0, arg_3_0, arg_4):
        arg_2_0 = paddle._C_ops.full_int_array(0, paddle.int64, paddle.core.CPUPlace())
        arg_3_0 = paddle._C_ops.full_int_array(1, paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = [arg_2_0]
        input_3 = [arg_3_0]
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor(input_0, input_1, [x.reshape([]) for x in input_2], [x.reshape([]) for x in input_3], input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6c1aaae4099b7188d6902343f0270d5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b95ea6dc5d2f4684da7451afbc08f5d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[16.287872314453125, 17.632444381713867, 16.034217834472656, 16.862110137939453, 16.98337745666504, 16.920209884643555, 17.048248291015625, 16.583837509155273, 18.13858413696289, 16.5045223236084, 16.764514923095703, 15.302421569824219, 16.991336822509766, 15.412895202636719, 17.656911849975586, 15.998936653137207, 17.28907585144043, 15.693737030029297, 17.184268951416016, 16.401487350463867, 17.41071319580078, 17.454984664916992, 17.847900390625, 16.6573486328125, 17.9927921295166, 17.93208885192871, 16.386157989501953, 16.252670288085938, 15.35257339477539, 16.72439956665039]], dtype='float32').reshape([1, 30]),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_23fe72332df9c8354bb59d9b33311bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b95ea6dc5d2f4684da7451afbc08f5d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1.0, 1.0, 1.0, 1.0]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f3050a4e6adcd4135ceca58230a832fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2_0, arg_3_0, arg_4):
        arg_2_0 = paddle._C_ops.full_int_array(0, paddle.int64, paddle.core.CPUPlace())
        arg_3_0 = paddle._C_ops.full_int_array(1, paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = [arg_2_0]
        input_3 = [arg_3_0]
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor(input_0, input_1, [x.reshape([]) for x in input_2], [x.reshape([]) for x in input_3], input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6241809bf7c61d08b36b31bb627ff2a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3050a4e6adcd4135ceca58230a832fc
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3ef688abd085c317135c01f171908b6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b95ea6dc5d2f4684da7451afbc08f5d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0cc14ddc2719a50056881f0f5065a97b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3050a4e6adcd4135ceca58230a832fc
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 30], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[16.671875, 16.71875, 15.609375, 15.0703125, 15.5625, 16.4375, 15.40625, 17.375, 15.4765625, 16.359375, 15.265625, 17.265625, 16.21875, 16.09375, 16.46875, 16.21875, 16.765625, 16.59375, 16.0, 15.2734375, 16.75, 15.7890625, 16.046875, 16.046875, 16.015625, 15.4296875, 15.1640625, 16.875, 16.40625, 15.1484375]], dtype='float16').reshape([1, 30]),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cbabeb3bf0e45816044e768eaa996f00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3050a4e6adcd4135ceca58230a832fc
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 4], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[1.0, 1.0, 1.0, 1.0]], dtype='float16').reshape([1, 4]),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_39c02607fd8728dec0ce863667698299(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2_0, arg_3_0, arg_4):
        arg_2_0 = paddle._C_ops.full_int_array(0, paddle.int64, paddle.core.CPUPlace())
        arg_3_0 = paddle._C_ops.full_int_array(1, paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = [arg_2_0]
        input_3 = [arg_3_0]
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor(input_0, input_1, [x.reshape([]) for x in input_2], [x.reshape([]) for x in input_3], input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 501, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a66fb43fef0e645399c4752c866fccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39c02607fd8728dec0ce863667698299
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[16.287872314453125, 17.632444381713867, 16.034217834472656, 16.862110137939453, 16.98337745666504, 16.920209884643555, 17.048248291015625, 16.583837509155273, 18.13858413696289, 16.5045223236084, 16.764514923095703, 15.302421569824219, 16.991336822509766, 15.412895202636719, 17.656911849975586, 15.998936653137207, 17.28907585144043, 15.693737030029297, 17.184268951416016, 16.401487350463867, 17.41071319580078, 17.454984664916992, 17.847900390625, 16.6573486328125, 17.9927921295166, 17.93208885192871, 16.386157989501953, 16.252670288085938, 15.35257339477539, 16.72439956665039]], dtype='float32').reshape([1, 30]),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c5592b15a22829d9651292832d6ce6f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2_0, arg_3_0, arg_4):
        arg_2_0 = paddle._C_ops.full_int_array(0, paddle.int64, paddle.core.CPUPlace())
        arg_3_0 = paddle._C_ops.full_int_array(1, paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = [arg_2_0]
        input_3 = [arg_3_0]
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor(input_0, input_1, [x.reshape([]) for x in input_2], [x.reshape([]) for x in input_3], input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 501, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f8815e9f9b80625712195cbf1b7af06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5592b15a22829d9651292832d6ce6f8
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1.0, 1.0, 1.0, 1.0]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_76e55ec357020ce8600a8f32123da13d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2_0, arg_3_0, arg_4):
        arg_2_0 = paddle._C_ops.full_int_array(0, paddle.int64, paddle.core.CPUPlace())
        arg_3_0 = paddle._C_ops.full_int_array(1, paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = [arg_2_0]
        input_3 = [arg_3_0]
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor(input_0, input_1, [x.reshape([]) for x in input_2], [x.reshape([]) for x in input_3], input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 501, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e7c8d13c616a53d3f98b73c1ff411ddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76e55ec357020ce8600a8f32123da13d
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b0d198b1ac70f2a133e3d234178ebb0a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2_0, arg_3_0, arg_4):
        arg_2_0 = paddle._C_ops.full_int_array(0, paddle.int64, paddle.core.CPUPlace())
        arg_3_0 = paddle._C_ops.full_int_array(1, paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = [arg_2_0]
        input_3 = [arg_3_0]
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor(input_0, input_1, [x.reshape([]) for x in input_2], [x.reshape([]) for x in input_3], input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 501, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2af98e2118c48c8cad68750058ec828a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0d198b1ac70f2a133e3d234178ebb0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1dc9792b06e9ccb78cefb6a6f5eeb793(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2_0, arg_3_0, arg_4):
        arg_2_0 = paddle._C_ops.full_int_array(0, paddle.int64, paddle.core.CPUPlace())
        arg_3_0 = paddle._C_ops.full_int_array(1, paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = [arg_2_0]
        input_3 = [arg_3_0]
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor(input_0, input_1, [x.reshape([]) for x in input_2], [x.reshape([]) for x in input_3], input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 501, 30], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 30], dtype='float16'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b6857e5503ad8c4cd5e6fd25ac911a5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1dc9792b06e9ccb78cefb6a6f5eeb793
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 30], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[16.671875, 16.71875, 15.609375, 15.0703125, 15.5625, 16.4375, 15.40625, 17.375, 15.4765625, 16.359375, 15.265625, 17.265625, 16.21875, 16.09375, 16.46875, 16.21875, 16.765625, 16.59375, 16.0, 15.2734375, 16.75, 15.7890625, 16.046875, 16.046875, 16.015625, 15.4296875, 15.1640625, 16.875, 16.40625, 15.1484375]], dtype='float16').reshape([1, 30]),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_881f6c7fd75ab92258b2cebd654f138f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2_0, arg_3_0, arg_4):
        arg_2_0 = paddle._C_ops.full_int_array(0, paddle.int64, paddle.core.CPUPlace())
        arg_3_0 = paddle._C_ops.full_int_array(1, paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = [arg_2_0]
        input_3 = [arg_3_0]
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor(input_0, input_1, [x.reshape([]) for x in input_2], [x.reshape([]) for x in input_3], input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 501, 4], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float16'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f27c7e2344aff0d7fcc91594ceb85f9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_881f6c7fd75ab92258b2cebd654f138f
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 4], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[1.0, 1.0, 1.0, 1.0]], dtype='float16').reshape([1, 4]),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


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