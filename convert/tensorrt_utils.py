from contextlib import contextmanager
import os
from time import time
import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import torch

import tensorrt as trt
from tensorrt import ICudaEngine, ILayer, INetworkDefinition, Logger, Runtime
from tensorrt.tensorrt import Builder, IBuilderConfig, IElementWiseLayer, IOptimizationProfile, IReduceLayer


@dataclass
class TensorRTShape:
    """
    Store input shapes for TensorRT build.
    3 shapes per input tensor are required (as tuple of integers):

    * minimum input shape
    * optimal size used for the benchmarks during building
    * maximum input shape

    Set input name to None for default shape.
    """

    min_shape: List[int]
    optimal_shape: List[int]
    max_shape: List[int]
    input_name: Optional[str]

    def check_validity(self) -> None:
        """
        Basic checks of provided shapes
        """
        assert len(self.min_shape) == len(self.optimal_shape) == len(self.max_shape)
        assert len(self.min_shape) > 0
        assert self.min_shape[0] > 0 and self.optimal_shape[0] > 0 and self.max_shape[0] > 0
        assert self.input_name is not None

    def make_copy(self, input_name: str) -> "TensorRTShape":
        """
        Make a copy of the current instance, with a different input name.
        :param input_name: new input name to use
        :return: a copy of the current shape with a different name
        """
        instance_copy = dataclasses.replace(self)
        instance_copy.input_name = input_name
        return instance_copy

    def generate_multiple_shapes(self, input_names: List[str]) -> List["TensorRTShape"]:
        """
        Generate multiple shapes when only a single default one is defined.
        :param input_names: input names used by the model
        :return: a list of shapes
        """
        assert self.input_name is None, f"input name is not None: {self.input_name}"
        result = list()
        for name in input_names:
            shape = self.make_copy(input_name=name)
            result.append(shape)
        return result


def fix_fp16_network(network_definition: INetworkDefinition,
                     fp16_banned_ops = None) -> INetworkDefinition:
    """
    Mixed precision on TensorRT can generate scores very far from Pytorch because of some operator being saturated.
    Indeed, FP16 can't store very large and very small numbers like FP32.
    Here, we search for some patterns of operators to keep in FP32, in most cases, it is enough to fix the inference
    and don't hurt performances.
    :param network_definition: graph generated by TensorRT after parsing ONNX file (during the model building)
    :return: patched network definition
    """
    # search for patterns which may overflow in FP16 precision, we force FP32 precisions for those nodes
    for layer_index in range(network_definition.num_layers - 1):
        layer: ILayer = network_definition.get_layer(layer_index)
        next_layer: ILayer = network_definition.get_layer(layer_index + 1)
        print("layer name:{}; type:{}".format(layer.name, layer.type))

        if fp16_banned_ops is not None:
            for cand in fp16_banned_ops:
                if layer.name.find(cand) != -1:
                    layer.precision = trt.DataType.FLOAT
                    layer.set_output_type(index=0, dtype=trt.DataType.FLOAT)

        # POW operation usually followed by mean reduce(transformer-deploy's strategy)
        if layer.type == trt.LayerType.ELEMENTWISE and next_layer.type == trt.LayerType.REDUCE:
            # casting to get access to op attribute
            layer.__class__ = IElementWiseLayer
            next_layer.__class__ = IReduceLayer
            if layer.op == trt.ElementWiseOperation.POW:
                layer.precision = trt.DataType.FLOAT
                next_layer.precision = trt.DataType.FLOAT
            layer.set_output_type(index=0, dtype=trt.DataType.FLOAT)
            next_layer.set_output_type(index=0, dtype=trt.DataType.FLOAT)
    return network_definition


def build_engine(
    runtime: Runtime,
    onnx_file_path: str,
    logger: Logger,
    fp16: bool,
    int8: bool,
    workspace_size: Optional[int] = None,
    fp16_fix: Callable[[INetworkDefinition], INetworkDefinition] = fix_fp16_network,
    fp16_banned_ops = None,
    timing_cache=None,
    **kwargs,
) -> ICudaEngine:
    """
    Convert ONNX file to TensorRT engine.
    It supports dynamic shape, however it's advised to keep sequence length fix as it hurts performance otherwise.
    Dynamic batch size doesn't hurt performance and is highly advised.
    Batch size can provided through different ways:
    * **min_shape**, **optimal_shape**, **max_shape**: for simple case, 3 tuples of int when all
    input tensors have the same shape
    * **input_shapes**: a list of TensorRTShape with names if there are several input tensors with different shapes
    **TIP**: minimum batch size should be 1 in most cases.
    :param runtime: global variable shared accross inference call / model building
    :param onnx_file_path: path to the ONNX file
    :param logger: specific logger to TensorRT
    :param workspace_size: GPU memory to use during the building, more is always better.
        If there is not enough memory, some optimization may fail, and the whole conversion process will crash.
    :param fp16: enable FP16 precision, it usually provide a 20-30% boost compared to ONNX Runtime.
    :param int8: enable INT-8 quantization, best performance but model should have been quantized.
    :param fp16_fix: a function to set FP32 precision on some nodes to fix FP16 overflow
    :return: TensorRT engine to use during inference
    """
    # default input shape
    if "min_shape" in kwargs and "optimal_shape" in kwargs and "max_shape" in kwargs:
        default_shape = TensorRTShape(
            min_shape=kwargs["min_shape"],
            optimal_shape=kwargs["optimal_shape"],
            max_shape=kwargs["max_shape"],
            input_name=None,
        )
        input_shapes = [default_shape]
    else:
        assert "input_shapes" in kwargs, "missing input shapes"
        input_shapes: List[TensorRTShape] = kwargs["input_shapes"]

    builder: Builder = trt.Builder(logger)
    config: IBuilderConfig = builder.create_builder_config()
    if workspace_size is not None:
        config.max_workspace_size = workspace_size
        # config.set_memory_pool_limit(trt.tensorrt.MemoryPoolType.DLA_GLOBAL_DRAM, workspace_size)
    config.set_tactic_sources(
        tactic_sources=1 << int(trt.TacticSource.CUBLAS)
        | 1 << int(trt.TacticSource.CUBLAS_LT)
        | 1 << int(trt.TacticSource.CUDNN)  # trt advised to use cuDNN for transfo architecture
    )
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    timing_cache_available = int(trt.__version__[0]) >= 8 and timing_cache != None
    # load global timing cache
    if timing_cache_available:
        if os.path.exists(timing_cache):
            with open(timing_cache, "rb") as f:
                cache = config.create_timing_cache(f.read())
                config.set_timing_cache(cache, ignore_mismatch=False)
        else:
            cache = config.create_timing_cache(b"")
            config.set_timing_cache(cache, ignore_mismatch=False)

    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network_def = builder.create_network(explicit_batch)
    # https://github.com/NVIDIA/TensorRT/issues/1196 (sometimes big diff in output when using FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    logger.log(msg="parsing TensorRT model", severity=trt.ILogger.INFO)

    with trt.OnnxParser(network_def, logger) as parser:
        with open(onnx_file_path, "rb") as f:
            # file path needed for models with external dataformat
            # https://github.com/onnx/onnx-tensorrt/issues/818
            parsed = parser.parse(f.read(), onnx_file_path)
            for i in range(parser.num_errors):
                print("TensorRT ONNX parser error:", parser.get_error(i))

    profile: IOptimizationProfile = builder.create_optimization_profile()
    # duplicate default shape (one for each input)
    if len(input_shapes) == 1 and input_shapes[0].input_name is None:
        names = [network_def.get_input(num_input).name for num_input in range(network_def.num_inputs)]
        input_shapes = input_shapes[0].generate_multiple_shapes(input_names=names)

    for shape in input_shapes:
        shape.check_validity()
        profile.set_shape(
            input=shape.input_name,
            min=shape.min_shape,
            opt=shape.optimal_shape,
            max=shape.max_shape,
        )
    if "shape_tensors" in kwargs:
        for shape in kwargs["shape_tensors"]:
            profile.set_shape_input(
                input=shape.input_name,
                min=shape.min_shape,
                opt=shape.optimal_shape,
                max=shape.max_shape,
            )
    config.add_optimization_profile(profile)
    if fp16:
        network_def = fp16_fix(network_def, fp16_banned_ops)

    logger.log(msg="building engine. depending on model size this may take a while", severity=trt.ILogger.WARNING)
    start = time()
    # trt_engine = builder.build_serialized_network(network_def, config)
    # engine: ICudaEngine = runtime.deserialize_cuda_engine(trt_engine)
    engine = builder.build_engine(network_def, config=config)
    logger.log(msg=f"building engine took {time() - start:4.1f} seconds", severity=trt.ILogger.WARNING)
    assert engine is not None, "error during engine generation, check error messages above :-("
    # save global timing cache
    if timing_cache_available:
        cache = config.get_timing_cache()
        with cache.serialize() as buffer:
            with open(timing_cache, "wb") as f:
                f.write(buffer)
                f.flush()
                os.fsync(f)
    return engine


@contextmanager
def track_infer_time(buffer: List[int]) -> None:
    """
    A context manager to perform latency measures
    :param buffer: a List where to save latencies for each input
    """
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    buffer.append(end - start)


TRT_LOGGER = trt.Logger()


def get_binding_idxs(engine: trt.ICudaEngine, profile_index: int):
    """
    Calculate start/end binding indices for current context's profile
    https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#opt_profiles_bindings
    :param engine: TensorRT engine generated during the model building
    :param profile_index: profile to use (several profiles can be set during building)
    :return: input and output tensor indexes
    """
    num_bindings_per_profile = engine.num_bindings // engine.num_optimization_profiles
    start_binding = profile_index * num_bindings_per_profile
    end_binding = start_binding + num_bindings_per_profile  # Separate input and output binding indices for convenience
    input_binding_idxs: List[int] = []
    output_binding_idxs: List[int] = []
    for binding_index in range(start_binding, end_binding):
        if engine.binding_is_input(binding_index):
            input_binding_idxs.append(binding_index)
        else:
            output_binding_idxs.append(binding_index)
    return input_binding_idxs, output_binding_idxs


def get_output_tensors(
    context: trt.IExecutionContext,
    host_inputs: List[torch.Tensor],
    input_binding_idxs: List[int],
    output_binding_idxs: List[int],
):
    """
    Reserve memory in GPU for input and output tensors.
    :param context: TensorRT context shared accross inference steps
    :param host_inputs: input tensor
    :param input_binding_idxs: indexes of each input vector (should be the same than during building)
    :param output_binding_idxs: indexes of each output vector (should be the same than during building)
    :return: tensors where output will be stored
    """
    # explicitly set dynamic input shapes, so dynamic output shapes can be computed internally
    for host_input, binding_index in zip(host_inputs, input_binding_idxs):
        input_name = context.engine.get_binding_name(binding_index)
        context.set_binding_shape(binding_index, tuple(host_input.shape))
    # assert context.all_binding_shapes_specified
    device_outputs: Dict[str, torch.Tensor] = dict()
    for binding_index in output_binding_idxs:
        # TensorRT computes output shape based on input shape provided above
        output_shape = context.get_binding_shape(binding=binding_index)
        output_name = context.engine.get_binding_name(index=binding_index)
        # allocate buffers to hold output results
        device_outputs[output_name] = torch.empty(tuple(output_shape), device="cuda")
    return device_outputs


class TensorRTModel(object):
    def __init__(self, engine_path):
        print(f'load engine_path is {engine_path}')
        self.engine = self.load_engine(engine_path)
        profile_index = 0
        self.context = self.engine.create_execution_context()
        self.context.set_optimization_profile_async(
            profile_index=profile_index, stream_handle=torch.cuda.current_stream().cuda_stream
        )
        self.input_binding_idxs, self.output_binding_idxs = get_binding_idxs(self.engine, profile_index)

    def load_engine(self, engine_file_path):
        assert os.path.exists(engine_file_path)
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def __call__(self, inputs, time_buffer=None):
        input_tensors: List[torch.Tensor] = list()
        for i in range(self.context.engine.num_bindings):
            if not self.context.engine.binding_is_input(index=i):
                continue
            tensor_name = self.context.engine.get_binding_name(i)
            assert tensor_name in inputs, f"input not provided: {tensor_name}"
            tensor = inputs[tensor_name]
            assert isinstance(tensor, torch.Tensor), f"unexpected tensor class: {type(tensor)}"
            assert tensor.device.type == "cuda", f"unexpected device type (trt only works on CUDA): {tensor.device.type}"
            # warning: small changes in output if int64 is used instead of int32
            if tensor.dtype in [torch.int64, torch.long]:
                # logging.warning(f"using {tensor.dtype} instead of int32 for {tensor_name}, will be casted to int32")
                tensor = tensor.type(torch.int32)
            input_tensors.append(tensor)

        # calculate input shape, bind it, allocate GPU memory for the output
        outputs: Dict[str, torch.Tensor] = get_output_tensors(
            self.context, input_tensors, self.input_binding_idxs, self.output_binding_idxs
        )
        bindings = [int(i.data_ptr()) for i in input_tensors + list(outputs.values())]
        if time_buffer is None:
            self.context.execute_v2(bindings=bindings)
        else:
            with track_infer_time(time_buffer):
                self.context.execute_v2(bindings=bindings)

        torch.cuda.current_stream().synchronize()  # sync all CUDA ops

        return outputs