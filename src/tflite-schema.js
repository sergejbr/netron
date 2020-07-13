const $root = flatbuffers.get('tflite');

$root.tflite = $root.tflite || {};

$root.tflite.TensorType = {
    FLOAT32: 0,
    FLOAT16: 1,
    INT32: 2,
    UINT8: 3,
    INT64: 4,
    STRING: 5,
    BOOL: 6,
    INT16: 7,
    COMPLEX64: 8,
    INT8: 9,
    FLOAT64: 10,
    COMPLEX128: 11
};

$root.tflite.CustomQuantization = class CustomQuantization {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get custom() {
        const offset = this._reader.offset(this._offset, 4);
        // TODO
        return undefined;
    }
};

$root.tflite.QuantizationParameters = class QuantizationParameters {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get min() {
        const offset = this._reader.offset(this._offset, 4);
        // TODO
        return undefined;
    }

    get max() {
        const offset = this._reader.offset(this._offset, 6);
        // TODO
        return undefined;
    }

    get scale() {
        const offset = this._reader.offset(this._offset, 8);
        // TODO
        return undefined;
    }

    get zero_point() {
        const offset = this._reader.offset(this._offset, 10);
        // TODO
        return undefined;
    }

    get details() {
        const offset = this._reader.offset(this._offset, 12);
        // TODO
        return undefined;
    }

    get quantized_dimension() {
        const offset = this._reader.offset(this._offset, 14);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }
};

$root.tflite.DimensionType = {
    DENSE: 0,
    SPARSE_CSR: 1
};

$root.tflite.Int32Vector = class Int32Vector {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get values() {
        const offset = this._reader.offset(this._offset, 4);
        // TODO
        return undefined;
    }
};

$root.tflite.Uint16Vector = class Uint16Vector {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get values() {
        const offset = this._reader.offset(this._offset, 4);
        // TODO
        return undefined;
    }
};

$root.tflite.Uint8Vector = class Uint8Vector {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get values() {
        const offset = this._reader.offset(this._offset, 4);
        // TODO
        return undefined;
    }
};

$root.tflite.DimensionMetadata = class DimensionMetadata {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get format() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get dense_size() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get array_segments() {
        const offset = this._reader.offset(this._offset, 8);
        // TODO
        return undefined;
    }

    get array_indices() {
        const offset = this._reader.offset(this._offset, 10);
        // TODO
        return undefined;
    }
};

$root.tflite.SparsityParameters = class SparsityParameters {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get traversal_order() {
        const offset = this._reader.offset(this._offset, 4);
        // TODO
        return undefined;
    }

    get block_map() {
        const offset = this._reader.offset(this._offset, 6);
        // TODO
        return undefined;
    }

    get dim_metadata() {
        const offset = this._reader.offset(this._offset, 8);
        const length = offset ? this._reader.length(this._offset + offset) : 0;
        const vector = [];
        for (let i = 0; i < length; i++) {
            vector.push(new $root.tflite.DimensionMetadata(this._reader, this._reader.indirect(this._reader.vector(this._offset + offset) + i * 4)));
        }
        return vector;
    }
};

$root.tflite.Tensor = class Tensor {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get shape() {
        const offset = this._reader.offset(this._offset, 4);
        // TODO
        return undefined;
    }

    get type() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get buffer() {
        const offset = this._reader.offset(this._offset, 8);
        return offset ? this._reader.uint32(this._offset + offset) : 0;
    }

    get name() {
        const offset = this._reader.offset(this._offset, 10);
        return offset ? this._reader.string(this._offset + offset) : null;
    }

    get quantization() {
        const offset = this._reader.offset(this._offset, 12);
        // TODO
        return undefined;
    }

    get is_variable() {
        const offset = this._reader.offset(this._offset, 14);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }

    get sparsity() {
        const offset = this._reader.offset(this._offset, 16);
        // TODO
        return undefined;
    }

    get shape_signature() {
        const offset = this._reader.offset(this._offset, 18);
        // TODO
        return undefined;
    }
};

$root.tflite.BuiltinOperator = {
    ADD: 0,
    AVERAGE_POOL_2D: 1,
    CONCATENATION: 2,
    CONV_2D: 3,
    DEPTHWISE_CONV_2D: 4,
    DEPTH_TO_SPACE: 5,
    DEQUANTIZE: 6,
    EMBEDDING_LOOKUP: 7,
    FLOOR: 8,
    FULLY_CONNECTED: 9,
    HASHTABLE_LOOKUP: 10,
    L2_NORMALIZATION: 11,
    L2_POOL_2D: 12,
    LOCAL_RESPONSE_NORMALIZATION: 13,
    LOGISTIC: 14,
    LSH_PROJECTION: 15,
    LSTM: 16,
    MAX_POOL_2D: 17,
    MUL: 18,
    RELU: 19,
    RELU_N1_TO_1: 20,
    RELU6: 21,
    RESHAPE: 22,
    RESIZE_BILINEAR: 23,
    RNN: 24,
    SOFTMAX: 25,
    SPACE_TO_DEPTH: 26,
    SVDF: 27,
    TANH: 28,
    CONCAT_EMBEDDINGS: 29,
    SKIP_GRAM: 30,
    CALL: 31,
    CUSTOM: 32,
    EMBEDDING_LOOKUP_SPARSE: 33,
    PAD: 34,
    UNIDIRECTIONAL_SEQUENCE_RNN: 35,
    GATHER: 36,
    BATCH_TO_SPACE_ND: 37,
    SPACE_TO_BATCH_ND: 38,
    TRANSPOSE: 39,
    MEAN: 40,
    SUB: 41,
    DIV: 42,
    SQUEEZE: 43,
    UNIDIRECTIONAL_SEQUENCE_LSTM: 44,
    STRIDED_SLICE: 45,
    BIDIRECTIONAL_SEQUENCE_RNN: 46,
    EXP: 47,
    TOPK_V2: 48,
    SPLIT: 49,
    LOG_SOFTMAX: 50,
    DELEGATE: 51,
    BIDIRECTIONAL_SEQUENCE_LSTM: 52,
    CAST: 53,
    PRELU: 54,
    MAXIMUM: 55,
    ARG_MAX: 56,
    MINIMUM: 57,
    LESS: 58,
    NEG: 59,
    PADV2: 60,
    GREATER: 61,
    GREATER_EQUAL: 62,
    LESS_EQUAL: 63,
    SELECT: 64,
    SLICE: 65,
    SIN: 66,
    TRANSPOSE_CONV: 67,
    SPARSE_TO_DENSE: 68,
    TILE: 69,
    EXPAND_DIMS: 70,
    EQUAL: 71,
    NOT_EQUAL: 72,
    LOG: 73,
    SUM: 74,
    SQRT: 75,
    RSQRT: 76,
    SHAPE: 77,
    POW: 78,
    ARG_MIN: 79,
    FAKE_QUANT: 80,
    REDUCE_PROD: 81,
    REDUCE_MAX: 82,
    PACK: 83,
    LOGICAL_OR: 84,
    ONE_HOT: 85,
    LOGICAL_AND: 86,
    LOGICAL_NOT: 87,
    UNPACK: 88,
    REDUCE_MIN: 89,
    FLOOR_DIV: 90,
    REDUCE_ANY: 91,
    SQUARE: 92,
    ZEROS_LIKE: 93,
    FILL: 94,
    FLOOR_MOD: 95,
    RANGE: 96,
    RESIZE_NEAREST_NEIGHBOR: 97,
    LEAKY_RELU: 98,
    SQUARED_DIFFERENCE: 99,
    MIRROR_PAD: 100,
    ABS: 101,
    SPLIT_V: 102,
    UNIQUE: 103,
    CEIL: 104,
    REVERSE_V2: 105,
    ADD_N: 106,
    GATHER_ND: 107,
    COS: 108,
    WHERE: 109,
    RANK: 110,
    ELU: 111,
    REVERSE_SEQUENCE: 112,
    MATRIX_DIAG: 113,
    QUANTIZE: 114,
    MATRIX_SET_DIAG: 115,
    ROUND: 116,
    HARD_SWISH: 117,
    IF: 118,
    WHILE: 119,
    NON_MAX_SUPPRESSION_V4: 120,
    NON_MAX_SUPPRESSION_V5: 121,
    SCATTER_ND: 122,
    SELECT_V2: 123,
    DENSIFY: 124,
    SEGMENT_SUM: 125,
    BATCH_MATMUL: 126
};

$root.tflite.Padding = {
    SAME: 0,
    VALID: 1
};

$root.tflite.ActivationFunctionType = {
    NONE: 0,
    RELU: 1,
    RELU_N1_TO_1: 2,
    RELU6: 3,
    TANH: 4,
    SIGN_BIT: 5
};

$root.tflite.Conv2DOptions = class Conv2DOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get padding() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get stride_w() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get stride_h() {
        const offset = this._reader.offset(this._offset, 8);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get fused_activation_function() {
        const offset = this._reader.offset(this._offset, 10);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get dilation_w_factor() {
        const offset = this._reader.offset(this._offset, 12);
        return offset ? this._reader.int32(this._offset + offset) : 1;
    }

    get dilation_h_factor() {
        const offset = this._reader.offset(this._offset, 14);
        return offset ? this._reader.int32(this._offset + offset) : 1;
    }
};

$root.tflite.Pool2DOptions = class Pool2DOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get padding() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get stride_w() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get stride_h() {
        const offset = this._reader.offset(this._offset, 8);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get filter_width() {
        const offset = this._reader.offset(this._offset, 10);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get filter_height() {
        const offset = this._reader.offset(this._offset, 12);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get fused_activation_function() {
        const offset = this._reader.offset(this._offset, 14);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }
};

$root.tflite.DepthwiseConv2DOptions = class DepthwiseConv2DOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get padding() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get stride_w() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get stride_h() {
        const offset = this._reader.offset(this._offset, 8);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get depth_multiplier() {
        const offset = this._reader.offset(this._offset, 10);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get fused_activation_function() {
        const offset = this._reader.offset(this._offset, 12);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get dilation_w_factor() {
        const offset = this._reader.offset(this._offset, 14);
        return offset ? this._reader.int32(this._offset + offset) : 1;
    }

    get dilation_h_factor() {
        const offset = this._reader.offset(this._offset, 16);
        return offset ? this._reader.int32(this._offset + offset) : 1;
    }
};

$root.tflite.ConcatEmbeddingsOptions = class ConcatEmbeddingsOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get num_channels() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get num_columns_per_channel() {
        const offset = this._reader.offset(this._offset, 6);
        // TODO
        return undefined;
    }

    get embedding_dim_per_channel() {
        const offset = this._reader.offset(this._offset, 8);
        // TODO
        return undefined;
    }
};

$root.tflite.LSHProjectionType = {
    UNKNOWN: 0,
    SPARSE: 1,
    DENSE: 2
};

$root.tflite.LSHProjectionOptions = class LSHProjectionOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get type() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }
};

$root.tflite.SVDFOptions = class SVDFOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get rank() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get fused_activation_function() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get asymmetric_quantize_inputs() {
        const offset = this._reader.offset(this._offset, 8);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }
};

$root.tflite.RNNOptions = class RNNOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get fused_activation_function() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get asymmetric_quantize_inputs() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }
};

$root.tflite.SequenceRNNOptions = class SequenceRNNOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get time_major() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }

    get fused_activation_function() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get asymmetric_quantize_inputs() {
        const offset = this._reader.offset(this._offset, 8);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }
};

$root.tflite.BidirectionalSequenceRNNOptions = class BidirectionalSequenceRNNOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get time_major() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }

    get fused_activation_function() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get merge_outputs() {
        const offset = this._reader.offset(this._offset, 8);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }

    get asymmetric_quantize_inputs() {
        const offset = this._reader.offset(this._offset, 10);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }
};

$root.tflite.FullyConnectedOptionsWeightsFormat = {
    DEFAULT: 0,
    SHUFFLED4x16INT8: 1
};

$root.tflite.FullyConnectedOptions = class FullyConnectedOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get fused_activation_function() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get weights_format() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get keep_num_dims() {
        const offset = this._reader.offset(this._offset, 8);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }

    get asymmetric_quantize_inputs() {
        const offset = this._reader.offset(this._offset, 10);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }
};

$root.tflite.SoftmaxOptions = class SoftmaxOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get beta() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.float32(this._offset + offset) : 0;
    }
};

$root.tflite.ConcatenationOptions = class ConcatenationOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get axis() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get fused_activation_function() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }
};

$root.tflite.AddOptions = class AddOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get fused_activation_function() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }
};

$root.tflite.MulOptions = class MulOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get fused_activation_function() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }
};

$root.tflite.L2NormOptions = class L2NormOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get fused_activation_function() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }
};

$root.tflite.LocalResponseNormalizationOptions = class LocalResponseNormalizationOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get radius() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get bias() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.float32(this._offset + offset) : 0;
    }

    get alpha() {
        const offset = this._reader.offset(this._offset, 8);
        return offset ? this._reader.float32(this._offset + offset) : 0;
    }

    get beta() {
        const offset = this._reader.offset(this._offset, 10);
        return offset ? this._reader.float32(this._offset + offset) : 0;
    }
};

$root.tflite.LSTMKernelType = {
    FULL: 0,
    BASIC: 1
};

$root.tflite.LSTMOptions = class LSTMOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get fused_activation_function() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get cell_clip() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.float32(this._offset + offset) : 0;
    }

    get proj_clip() {
        const offset = this._reader.offset(this._offset, 8);
        return offset ? this._reader.float32(this._offset + offset) : 0;
    }

    get kernel_type() {
        const offset = this._reader.offset(this._offset, 10);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get asymmetric_quantize_inputs() {
        const offset = this._reader.offset(this._offset, 12);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }
};

$root.tflite.UnidirectionalSequenceLSTMOptions = class UnidirectionalSequenceLSTMOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get fused_activation_function() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get cell_clip() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.float32(this._offset + offset) : 0;
    }

    get proj_clip() {
        const offset = this._reader.offset(this._offset, 8);
        return offset ? this._reader.float32(this._offset + offset) : 0;
    }

    get time_major() {
        const offset = this._reader.offset(this._offset, 10);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }

    get asymmetric_quantize_inputs() {
        const offset = this._reader.offset(this._offset, 12);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }
};

$root.tflite.BidirectionalSequenceLSTMOptions = class BidirectionalSequenceLSTMOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get fused_activation_function() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get cell_clip() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.float32(this._offset + offset) : 0;
    }

    get proj_clip() {
        const offset = this._reader.offset(this._offset, 8);
        return offset ? this._reader.float32(this._offset + offset) : 0;
    }

    get merge_outputs() {
        const offset = this._reader.offset(this._offset, 10);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }

    get time_major() {
        const offset = this._reader.offset(this._offset, 12);
        return offset ? this._reader.bool(this._offset + offset) : true;
    }

    get asymmetric_quantize_inputs() {
        const offset = this._reader.offset(this._offset, 14);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }
};

$root.tflite.ResizeBilinearOptions = class ResizeBilinearOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get new_height() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get new_width() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get align_corners() {
        const offset = this._reader.offset(this._offset, 8);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }

    get half_pixel_centers() {
        const offset = this._reader.offset(this._offset, 10);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }
};

$root.tflite.ResizeNearestNeighborOptions = class ResizeNearestNeighborOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get align_corners() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }

    get half_pixel_centers() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }
};

$root.tflite.CallOptions = class CallOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get subgraph() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.uint32(this._offset + offset) : 0;
    }
};

$root.tflite.PadOptions = class PadOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.PadV2Options = class PadV2Options {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.ReshapeOptions = class ReshapeOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get new_shape() {
        const offset = this._reader.offset(this._offset, 4);
        // TODO
        return undefined;
    }
};

$root.tflite.SpaceToBatchNDOptions = class SpaceToBatchNDOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.BatchToSpaceNDOptions = class BatchToSpaceNDOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.SkipGramOptions = class SkipGramOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get ngram_size() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get max_skip_size() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get include_all_ngrams() {
        const offset = this._reader.offset(this._offset, 8);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }
};

$root.tflite.SpaceToDepthOptions = class SpaceToDepthOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get block_size() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }
};

$root.tflite.DepthToSpaceOptions = class DepthToSpaceOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get block_size() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }
};

$root.tflite.SubOptions = class SubOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get fused_activation_function() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }
};

$root.tflite.DivOptions = class DivOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get fused_activation_function() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }
};

$root.tflite.TopKV2Options = class TopKV2Options {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.CombinerType = {
    SUM: 0,
    MEAN: 1,
    SQRTN: 2
};

$root.tflite.EmbeddingLookupSparseOptions = class EmbeddingLookupSparseOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get combiner() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }
};

$root.tflite.GatherOptions = class GatherOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get axis() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }
};

$root.tflite.TransposeOptions = class TransposeOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.ExpOptions = class ExpOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.CosOptions = class CosOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.ReducerOptions = class ReducerOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get keep_dims() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }
};

$root.tflite.SqueezeOptions = class SqueezeOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get squeeze_dims() {
        const offset = this._reader.offset(this._offset, 4);
        // TODO
        return undefined;
    }
};

$root.tflite.SplitOptions = class SplitOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get num_splits() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }
};

$root.tflite.SplitVOptions = class SplitVOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get num_splits() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }
};

$root.tflite.StridedSliceOptions = class StridedSliceOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get begin_mask() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get end_mask() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get ellipsis_mask() {
        const offset = this._reader.offset(this._offset, 8);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get new_axis_mask() {
        const offset = this._reader.offset(this._offset, 10);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get shrink_axis_mask() {
        const offset = this._reader.offset(this._offset, 12);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }
};

$root.tflite.LogSoftmaxOptions = class LogSoftmaxOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.CastOptions = class CastOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get in_data_type() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get out_data_type() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }
};

$root.tflite.DequantizeOptions = class DequantizeOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.MaximumMinimumOptions = class MaximumMinimumOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.TileOptions = class TileOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.ArgMaxOptions = class ArgMaxOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get output_type() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }
};

$root.tflite.ArgMinOptions = class ArgMinOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get output_type() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }
};

$root.tflite.GreaterOptions = class GreaterOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.GreaterEqualOptions = class GreaterEqualOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.LessOptions = class LessOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.LessEqualOptions = class LessEqualOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.NegOptions = class NegOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.SelectOptions = class SelectOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.SliceOptions = class SliceOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.TransposeConvOptions = class TransposeConvOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get padding() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get stride_w() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get stride_h() {
        const offset = this._reader.offset(this._offset, 8);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }
};

$root.tflite.ExpandDimsOptions = class ExpandDimsOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.SparseToDenseOptions = class SparseToDenseOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get validate_indices() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }
};

$root.tflite.EqualOptions = class EqualOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.NotEqualOptions = class NotEqualOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.ShapeOptions = class ShapeOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get out_type() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }
};

$root.tflite.RankOptions = class RankOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.PowOptions = class PowOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.FakeQuantOptions = class FakeQuantOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get min() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.float32(this._offset + offset) : 0;
    }

    get max() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.float32(this._offset + offset) : 0;
    }

    get num_bits() {
        const offset = this._reader.offset(this._offset, 8);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get narrow_range() {
        const offset = this._reader.offset(this._offset, 10);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }
};

$root.tflite.PackOptions = class PackOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get values_count() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get axis() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }
};

$root.tflite.LogicalOrOptions = class LogicalOrOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.OneHotOptions = class OneHotOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get axis() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }
};

$root.tflite.AbsOptions = class AbsOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.HardSwishOptions = class HardSwishOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.LogicalAndOptions = class LogicalAndOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.LogicalNotOptions = class LogicalNotOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.UnpackOptions = class UnpackOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get num() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get axis() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }
};

$root.tflite.FloorDivOptions = class FloorDivOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.SquareOptions = class SquareOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.ZerosLikeOptions = class ZerosLikeOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.FillOptions = class FillOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.FloorModOptions = class FloorModOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.RangeOptions = class RangeOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.LeakyReluOptions = class LeakyReluOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get alpha() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.float32(this._offset + offset) : 0;
    }
};

$root.tflite.SquaredDifferenceOptions = class SquaredDifferenceOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.MirrorPadMode = {
    REFLECT: 0,
    SYMMETRIC: 1
};

$root.tflite.MirrorPadOptions = class MirrorPadOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get mode() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }
};

$root.tflite.UniqueOptions = class UniqueOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get idx_out_type() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 2;
    }
};

$root.tflite.ReverseV2Options = class ReverseV2Options {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.AddNOptions = class AddNOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.GatherNdOptions = class GatherNdOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.WhereOptions = class WhereOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.ReverseSequenceOptions = class ReverseSequenceOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get seq_dim() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get batch_dim() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }
};

$root.tflite.MatrixDiagOptions = class MatrixDiagOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.QuantizeOptions = class QuantizeOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.MatrixSetDiagOptions = class MatrixSetDiagOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.IfOptions = class IfOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get then_subgraph_index() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get else_subgraph_index() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }
};

$root.tflite.WhileOptions = class WhileOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get cond_subgraph_index() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get body_subgraph_index() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }
};

$root.tflite.NonMaxSuppressionV4Options = class NonMaxSuppressionV4Options {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.NonMaxSuppressionV5Options = class NonMaxSuppressionV5Options {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.ScatterNdOptions = class ScatterNdOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.SelectV2Options = class SelectV2Options {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.DensifyOptions = class DensifyOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.SegmentSumOptions = class SegmentSumOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.BatchMatMulOptions = class BatchMatMulOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get adj_x() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }

    get adj_y() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.bool(this._offset + offset) : false;
    }
};

$root.tflite.OperatorCode = class OperatorCode {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get builtin_code() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get custom_code() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.string(this._offset + offset) : null;
    }

    get version() {
        const offset = this._reader.offset(this._offset, 8);
        return offset ? this._reader.int32(this._offset + offset) : 1;
    }
};

$root.tflite.CustomOptionsFormat = {
    FLEXBUFFERS: 0
};

$root.tflite.Operator = class Operator {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get opcode_index() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.uint32(this._offset + offset) : 0;
    }

    get inputs() {
        const offset = this._reader.offset(this._offset, 6);
        // TODO
        return undefined;
    }

    get outputs() {
        const offset = this._reader.offset(this._offset, 8);
        // TODO
        return undefined;
    }

    get builtin_options() {
        const offset = this._reader.offset(this._offset, 10);
        // TODO
        return undefined;
    }

    get custom_options() {
        const offset = this._reader.offset(this._offset, 12);
        // TODO
        return undefined;
    }

    get custom_options_format() {
        const offset = this._reader.offset(this._offset, 14);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get mutating_variable_inputs() {
        const offset = this._reader.offset(this._offset, 16);
        // TODO
        return undefined;
    }

    get intermediates() {
        const offset = this._reader.offset(this._offset, 18);
        // TODO
        return undefined;
    }
};

$root.tflite.SubGraph = class SubGraph {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get tensors() {
        const offset = this._reader.offset(this._offset, 4);
        const length = offset ? this._reader.length(this._offset + offset) : 0;
        const vector = [];
        for (let i = 0; i < length; i++) {
            vector.push(new $root.tflite.Tensor(this._reader, this._reader.indirect(this._reader.vector(this._offset + offset) + i * 4)));
        }
        return vector;
    }

    get inputs() {
        const offset = this._reader.offset(this._offset, 6);
        // TODO
        return undefined;
    }

    get outputs() {
        const offset = this._reader.offset(this._offset, 8);
        // TODO
        return undefined;
    }

    get operators() {
        const offset = this._reader.offset(this._offset, 10);
        const length = offset ? this._reader.length(this._offset + offset) : 0;
        const vector = [];
        for (let i = 0; i < length; i++) {
            vector.push(new $root.tflite.Operator(this._reader, this._reader.indirect(this._reader.vector(this._offset + offset) + i * 4)));
        }
        return vector;
    }

    get name() {
        const offset = this._reader.offset(this._offset, 12);
        return offset ? this._reader.string(this._offset + offset) : null;
    }
};

$root.tflite.Buffer = class Buffer {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get data() {
        const offset = this._reader.offset(this._offset, 4);
        // TODO
        return undefined;
    }
};

$root.tflite.Metadata = class Metadata {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get name() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.string(this._offset + offset) : null;
    }

    get buffer() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.uint32(this._offset + offset) : 0;
    }
};

$root.tflite.Model = class Model {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    static create(reader) {
        return new $root.tflite.Model(reader, reader.int32(reader.position) + reader.position);
    }

    static identifier(reader) {
        return reader.identifier('TFL3');
    }

    get version() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.uint32(this._offset + offset) : 0;
    }

    get operator_codes() {
        const offset = this._reader.offset(this._offset, 6);
        const length = offset ? this._reader.length(this._offset + offset) : 0;
        const vector = [];
        for (let i = 0; i < length; i++) {
            vector.push(new $root.tflite.OperatorCode(this._reader, this._reader.indirect(this._reader.vector(this._offset + offset) + i * 4)));
        }
        return vector;
    }

    get subgraphs() {
        const offset = this._reader.offset(this._offset, 8);
        const length = offset ? this._reader.length(this._offset + offset) : 0;
        const vector = [];
        for (let i = 0; i < length; i++) {
            vector.push(new $root.tflite.SubGraph(this._reader, this._reader.indirect(this._reader.vector(this._offset + offset) + i * 4)));
        }
        return vector;
    }

    get description() {
        const offset = this._reader.offset(this._offset, 10);
        return offset ? this._reader.string(this._offset + offset) : null;
    }

    get buffers() {
        const offset = this._reader.offset(this._offset, 12);
        const length = offset ? this._reader.length(this._offset + offset) : 0;
        const vector = [];
        for (let i = 0; i < length; i++) {
            vector.push(new $root.tflite.Buffer(this._reader, this._reader.indirect(this._reader.vector(this._offset + offset) + i * 4)));
        }
        return vector;
    }

    get metadata_buffer() {
        const offset = this._reader.offset(this._offset, 14);
        // TODO
        return undefined;
    }

    get metadata() {
        const offset = this._reader.offset(this._offset, 16);
        const length = offset ? this._reader.length(this._offset + offset) : 0;
        const vector = [];
        for (let i = 0; i < length; i++) {
            vector.push(new $root.tflite.Metadata(this._reader, this._reader.indirect(this._reader.vector(this._offset + offset) + i * 4)));
        }
        return vector;
    }
};


$root.tflite = $root.tflite || {};

$root.tflite.AssociatedFileType = {
    UNKNOWN: 0,
    DESCRIPTIONS: 1,
    TENSOR_AXIS_LABELS: 2,
    TENSOR_VALUE_LABELS: 3,
    TENSOR_AXIS_SCORE_CALIBRATION: 4,
    VOCABULARY: 5
};

$root.tflite.AssociatedFile = class AssociatedFile {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get name() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.string(this._offset + offset) : null;
    }

    get description() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.string(this._offset + offset) : null;
    }

    get type() {
        const offset = this._reader.offset(this._offset, 8);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get locale() {
        const offset = this._reader.offset(this._offset, 10);
        return offset ? this._reader.string(this._offset + offset) : null;
    }
};

$root.tflite.FeatureProperties = class FeatureProperties {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }
};

$root.tflite.ColorSpaceType = {
    UNKNOWN: 0,
    RGB: 1,
    GRAYSCALE: 2
};

$root.tflite.ImageSize = class ImageSize {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get width() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.uint32(this._offset + offset) : 0;
    }

    get height() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.uint32(this._offset + offset) : 0;
    }
};

$root.tflite.ImageProperties = class ImageProperties {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get color_space() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get default_size() {
        const offset = this._reader.offset(this._offset, 6);
        // TODO
        return undefined;
    }
};

$root.tflite.BoundingBoxType = {
    UNKNOWN: 0,
    BOUNDARIES: 1,
    UPPER_LEFT: 2,
    CENTER: 3
};

$root.tflite.CoordinateType = {
    RATIO: 0,
    PIXEL: 1
};

$root.tflite.BoundingBoxProperties = class BoundingBoxProperties {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get index() {
        const offset = this._reader.offset(this._offset, 4);
        // TODO
        return undefined;
    }

    get type() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get coordinate_type() {
        const offset = this._reader.offset(this._offset, 8);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }
};

$root.tflite.ValueRange = class ValueRange {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get min() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }

    get max() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.int32(this._offset + offset) : 0;
    }
};

$root.tflite.Content = class Content {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get content_properties() {
        const offset = this._reader.offset(this._offset, 4);
        // TODO
        return undefined;
    }

    get range() {
        const offset = this._reader.offset(this._offset, 6);
        // TODO
        return undefined;
    }
};

$root.tflite.NormalizationOptions = class NormalizationOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get mean() {
        const offset = this._reader.offset(this._offset, 4);
        // TODO
        return undefined;
    }

    get std() {
        const offset = this._reader.offset(this._offset, 6);
        // TODO
        return undefined;
    }
};

$root.tflite.ScoreTransformationType = {
    IDENTITY: 0,
    LOG: 1,
    INVERSE_LOGISTIC: 2
};

$root.tflite.ScoreCalibrationOptions = class ScoreCalibrationOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get score_transformation() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.int8(this._offset + offset) : 0;
    }

    get default_score() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.float32(this._offset + offset) : 0;
    }
};

$root.tflite.ScoreThresholdingOptions = class ScoreThresholdingOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get global_score_threshold() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.float32(this._offset + offset) : 0;
    }
};

$root.tflite.BertTokenizerOptions = class BertTokenizerOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get vocab_file() {
        const offset = this._reader.offset(this._offset, 4);
        const length = offset ? this._reader.length(this._offset + offset) : 0;
        const vector = [];
        for (let i = 0; i < length; i++) {
            vector.push(new $root.tflite.AssociatedFile(this._reader, this._reader.indirect(this._reader.vector(this._offset + offset) + i * 4)));
        }
        return vector;
    }
};

$root.tflite.SentencePieceTokenizerOptions = class SentencePieceTokenizerOptions {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get sentencePiece_model() {
        const offset = this._reader.offset(this._offset, 4);
        const length = offset ? this._reader.length(this._offset + offset) : 0;
        const vector = [];
        for (let i = 0; i < length; i++) {
            vector.push(new $root.tflite.AssociatedFile(this._reader, this._reader.indirect(this._reader.vector(this._offset + offset) + i * 4)));
        }
        return vector;
    }

    get vocab_file() {
        const offset = this._reader.offset(this._offset, 6);
        const length = offset ? this._reader.length(this._offset + offset) : 0;
        const vector = [];
        for (let i = 0; i < length; i++) {
            vector.push(new $root.tflite.AssociatedFile(this._reader, this._reader.indirect(this._reader.vector(this._offset + offset) + i * 4)));
        }
        return vector;
    }
};

$root.tflite.ProcessUnit = class ProcessUnit {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get options() {
        const offset = this._reader.offset(this._offset, 4);
        // TODO
        return undefined;
    }
};

$root.tflite.Stats = class Stats {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get max() {
        const offset = this._reader.offset(this._offset, 4);
        // TODO
        return undefined;
    }

    get min() {
        const offset = this._reader.offset(this._offset, 6);
        // TODO
        return undefined;
    }
};

$root.tflite.TensorGroup = class TensorGroup {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get name() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.string(this._offset + offset) : null;
    }

    get tensor_names() {
        const offset = this._reader.offset(this._offset, 6);
        // TODO
        return undefined;
    }
};

$root.tflite.TensorMetadata = class TensorMetadata {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get name() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.string(this._offset + offset) : null;
    }

    get description() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.string(this._offset + offset) : null;
    }

    get dimension_names() {
        const offset = this._reader.offset(this._offset, 8);
        // TODO
        return undefined;
    }

    get content() {
        const offset = this._reader.offset(this._offset, 10);
        // TODO
        return undefined;
    }

    get process_units() {
        const offset = this._reader.offset(this._offset, 12);
        const length = offset ? this._reader.length(this._offset + offset) : 0;
        const vector = [];
        for (let i = 0; i < length; i++) {
            vector.push(new $root.tflite.ProcessUnit(this._reader, this._reader.indirect(this._reader.vector(this._offset + offset) + i * 4)));
        }
        return vector;
    }

    get stats() {
        const offset = this._reader.offset(this._offset, 14);
        // TODO
        return undefined;
    }

    get associated_files() {
        const offset = this._reader.offset(this._offset, 16);
        const length = offset ? this._reader.length(this._offset + offset) : 0;
        const vector = [];
        for (let i = 0; i < length; i++) {
            vector.push(new $root.tflite.AssociatedFile(this._reader, this._reader.indirect(this._reader.vector(this._offset + offset) + i * 4)));
        }
        return vector;
    }
};

$root.tflite.SubGraphMetadata = class SubGraphMetadata {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    get name() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.string(this._offset + offset) : null;
    }

    get description() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.string(this._offset + offset) : null;
    }

    get input_tensor_metadata() {
        const offset = this._reader.offset(this._offset, 8);
        const length = offset ? this._reader.length(this._offset + offset) : 0;
        const vector = [];
        for (let i = 0; i < length; i++) {
            vector.push(new $root.tflite.TensorMetadata(this._reader, this._reader.indirect(this._reader.vector(this._offset + offset) + i * 4)));
        }
        return vector;
    }

    get output_tensor_metadata() {
        const offset = this._reader.offset(this._offset, 10);
        const length = offset ? this._reader.length(this._offset + offset) : 0;
        const vector = [];
        for (let i = 0; i < length; i++) {
            vector.push(new $root.tflite.TensorMetadata(this._reader, this._reader.indirect(this._reader.vector(this._offset + offset) + i * 4)));
        }
        return vector;
    }

    get associated_files() {
        const offset = this._reader.offset(this._offset, 12);
        const length = offset ? this._reader.length(this._offset + offset) : 0;
        const vector = [];
        for (let i = 0; i < length; i++) {
            vector.push(new $root.tflite.AssociatedFile(this._reader, this._reader.indirect(this._reader.vector(this._offset + offset) + i * 4)));
        }
        return vector;
    }

    get input_process_units() {
        const offset = this._reader.offset(this._offset, 14);
        const length = offset ? this._reader.length(this._offset + offset) : 0;
        const vector = [];
        for (let i = 0; i < length; i++) {
            vector.push(new $root.tflite.ProcessUnit(this._reader, this._reader.indirect(this._reader.vector(this._offset + offset) + i * 4)));
        }
        return vector;
    }

    get output_process_units() {
        const offset = this._reader.offset(this._offset, 16);
        const length = offset ? this._reader.length(this._offset + offset) : 0;
        const vector = [];
        for (let i = 0; i < length; i++) {
            vector.push(new $root.tflite.ProcessUnit(this._reader, this._reader.indirect(this._reader.vector(this._offset + offset) + i * 4)));
        }
        return vector;
    }

    get input_tensor_groups() {
        const offset = this._reader.offset(this._offset, 18);
        const length = offset ? this._reader.length(this._offset + offset) : 0;
        const vector = [];
        for (let i = 0; i < length; i++) {
            vector.push(new $root.tflite.TensorGroup(this._reader, this._reader.indirect(this._reader.vector(this._offset + offset) + i * 4)));
        }
        return vector;
    }

    get output_tensor_groups() {
        const offset = this._reader.offset(this._offset, 20);
        const length = offset ? this._reader.length(this._offset + offset) : 0;
        const vector = [];
        for (let i = 0; i < length; i++) {
            vector.push(new $root.tflite.TensorGroup(this._reader, this._reader.indirect(this._reader.vector(this._offset + offset) + i * 4)));
        }
        return vector;
    }
};

$root.tflite.ModelMetadata = class ModelMetadata {

    constructor(reader, offset) {
        this._reader = reader;
        this._offset = offset;
    }

    static create(reader) {
        return new $root.tflite.ModelMetadata(reader, reader.int32(reader.position) + reader.position);
    }

    static identifier(reader) {
        return reader.identifier('M001');
    }

    get name() {
        const offset = this._reader.offset(this._offset, 4);
        return offset ? this._reader.string(this._offset + offset) : null;
    }

    get description() {
        const offset = this._reader.offset(this._offset, 6);
        return offset ? this._reader.string(this._offset + offset) : null;
    }

    get version() {
        const offset = this._reader.offset(this._offset, 8);
        return offset ? this._reader.string(this._offset + offset) : null;
    }

    get subgraph_metadata() {
        const offset = this._reader.offset(this._offset, 10);
        const length = offset ? this._reader.length(this._offset + offset) : 0;
        const vector = [];
        for (let i = 0; i < length; i++) {
            vector.push(new $root.tflite.SubGraphMetadata(this._reader, this._reader.indirect(this._reader.vector(this._offset + offset) + i * 4)));
        }
        return vector;
    }

    get author() {
        const offset = this._reader.offset(this._offset, 12);
        return offset ? this._reader.string(this._offset + offset) : null;
    }

    get license() {
        const offset = this._reader.offset(this._offset, 14);
        return offset ? this._reader.string(this._offset + offset) : null;
    }

    get associated_files() {
        const offset = this._reader.offset(this._offset, 16);
        const length = offset ? this._reader.length(this._offset + offset) : 0;
        const vector = [];
        for (let i = 0; i < length; i++) {
            vector.push(new $root.tflite.AssociatedFile(this._reader, this._reader.indirect(this._reader.vector(this._offset + offset) + i * 4)));
        }
        return vector;
    }

    get min_parser_version() {
        const offset = this._reader.offset(this._offset, 18);
        return offset ? this._reader.string(this._offset + offset) : null;
    }
};
