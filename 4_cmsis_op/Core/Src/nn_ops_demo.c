#include "nn_ops_demo.h"    // 当这个文件同文件函数的时候就会需要了
#include "arm_nnsupportfunctions.h"
#include "stdio.h"

void show_vector(q7_t* vec, const int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", vec[i]);
    }
    printf("\r\n");
}

void pooling_demo() {
    q7_t in_vec[] = {-1, -1, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 5, 5, 4, 4,
                    7, 7, 0, 0, 0, 0, 0, 0, -1, -1, -2, -2, 0, 0, 9, 9};
    q7_t out_vec[20] = {0};

    printf("Input vector: ");
    show_vector(in_vec, sizeof(in_vec));

    arm_maxpool_q7_HWC(in_vec, 4, 2, 2, 0, 2, 2, NULL, out_vec);

    printf("Output vector: ");
    show_vector(out_vec, sizeof(out_vec));
}

void relu_demo() {
    q7_t in_vec[] = {-1, -1, 0, 1, 2, -3};

    printf("Input vector: ");
    show_vector(in_vec, sizeof(in_vec));

    arm_relu_q7(in_vec, sizeof(in_vec));

    printf("Output vector: ");
    show_vector(in_vec, sizeof(in_vec));
}

void conv_demo() {
    /****************** 例1输入设置 ******************/
    // 输入设置
    // q7_t in_vec[] = {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    //             1, 2, 1, 2, 1, 2, 1, 2, 1, 2};
    // q7_t weight[] = {2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 3, 1, 3, 1,
    //             3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1};
    // q7_t bias[] = {0, 0};        // 有几个卷积核就有几个bias

    /****************** 例2输入设置 ******************/
    // 输入设置
    q7_t in_vec[] = {1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 8, 1, 9, 1, 1, 1, 2, 1,
                3, 1, 4, 1, 5, 1, 6, 1, 7, 1};
    q7_t weight[] = {2, -1, 2, -2, 2, -3, 2, -4, 2, -5, 2, -6, 2, -7, 2, -8, 2,
                -9, 1, 1, 2, 1, 3, 1, 1, 1, 2, 1, 3, 1, 1, 1, 2, 1, 3, 1};
    q7_t bias[] = {0, 0};        // 有几个卷积核就有几个bias

    /****************** 共用部分 ******************/
    // 输出和暂存设置
    q7_t out_vec[20] = {0};
    q7_t buffer_a[80] = {0};

    // 网络参数设置
    const uint16_t input_dim = 4, input_channels = 2;
    const uint16_t output_dim = 2, output_channels = 2;
    const uint16_t kernel_dim = 3;
    const uint16_t padding = 0, stride = 1;

    // 量化参数
    const uint16_t bias_lshift = 0, output_rshift = 0;
    
    arm_status ret = arm_convolve_HWC_q7_basic(
        in_vec, input_dim, input_channels,
        weight, output_channels, kernel_dim,
        padding, stride, bias,
        bias_lshift, output_rshift,
        out_vec, output_dim, (q15_t*)buffer_a, NULL
    );
    if (ret != ARM_MATH_SUCCESS) {
        printf("Error\r\n");
        return;
    }

    printf("Output vector: ");
    show_vector(out_vec, sizeof(out_vec));
}

void dense_demo() {
    // Y = WX

    // 输入设置
    q7_t in_vec[] = {-1, -2, 1, 2, 3};
    q7_t weight[] = {2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2};
    q7_t bias[] = {0, 0, 0};

    // 网络参数设置
    const uint16_t vector_dim = 5;
    const uint16_t weight_rows = 3;

    // 量化参数
    const uint16_t bias_lshift = 0, output_rshift = 0;

    // 输出和暂存设置
    q7_t out_vec[16] = {0};
    q7_t vector_buffer[80] = {0};

    arm_status ret = arm_fully_connected_q7(
        in_vec, weight,
        vector_dim, weight_rows,
        bias_lshift, output_rshift,
        bias,
        out_vec, (q15_t*)vector_buffer
    );
    if (ret != ARM_MATH_SUCCESS) {
        printf("Error\r\n");
        return;
    }

    printf("Output vector: ");
    show_vector(out_vec, sizeof(out_vec));
}

void basic_func_demo() {
    q31_t m1 = 1;
    q31_t m2 = 100;
    q31_t result = arm_nn_doubling_high_mult_no_sat(m1, m2);
    printf("%d\r\n", result);
}

void pooling_s8_demo() {
    // The function returns required buffer size in bytes
    // int32_t buffer_size = arm_avgpool_s8_get_buffer_size(2, 2);
    // printf("%d\r\n", buffer_size);   // 8

    // 构造context (max pool里根本不会用到)
    cmsis_nn_context context;
    // q7_t buffer[8] = {0};
    // context.buf = buffer;
    // context.size = buffer_size;

    // 构造参数
    cmsis_nn_pool_params pool_params;
    cmsis_nn_tile stride, padding;
    cmsis_nn_activation activation_range;
    stride.w = 2, stride.h = 2;
    padding.w = 0, padding.h = 0;
    activation_range.min = -128, activation_range.max = 127;
    pool_params.stride = stride;
    pool_params.padding = padding;
    pool_params.activation = activation_range;

    // 维度信息
    cmsis_nn_dims input_dims, filter_dims, output_dims;
    input_dims.h = 4, input_dims.w = 4, input_dims.c = 2;
    filter_dims.h = 2, filter_dims.w = 2;
    output_dims.h = 2, output_dims.w = 2, output_dims.c = 2;

    // 数据
    q7_t input_data[] = {-1, -1, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 5, 5, 4, 4,
                    7, 7, 0, 0, 0, 0, 0, 0, -1, -1, -2, -2, 0, 0, 9, 9};
    q7_t output_data[20] = {0};

    arm_status ret = arm_max_pool_s8(&context, &pool_params,
                        &input_dims, input_data, &filter_dims,
                        &output_dims, output_data);
    if (ret != ARM_MATH_SUCCESS) {
        printf("Error\r\n");
        return;
    }

    show_vector(output_data, sizeof(output_data));
}

void dense_s8_demo() {
    // 维度信息
    cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;
    input_dims.n = 2, input_dims.h = 1, input_dims.w = 5, input_dims.c = 1;
    filter_dims.n = 5, filter_dims.c = 3;
    bias_dims.c = 3;
    output_dims.n = 2, output_dims.c = 3;

    // context
    // The function returns required buffer size in bytes
    int32_t buffer_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    // printf("%d\r\n", buffer_size);   // 0，不需要额外的内存

    // 量化参数
    cmsis_nn_fc_params fc_params;
    fc_params.input_offset = 1;     // 这就是直接在input上+1
    fc_params.filter_offset = 0;
    fc_params.output_offset = 2;    // 这就是在最后的结果上直接+2
    cmsis_nn_activation activation_range;
    activation_range.min = -128;
    activation_range.max = 127;
    fc_params.activation = activation_range;

    cmsis_nn_per_tensor_quant_params quant_params;
    // 无量化版本
    // quant_params.multiplier = (1 << 31) - 1;
    // quant_params.shift = 0;

    // 量化版本
    quant_params.multiplier = 1717986918;
    quant_params.shift = -4;


    // 数据
    q7_t input_data[] = {1, 2, 1, 1, 1, -1, -2, -1, -1, -1};
    // q7_t filter_data[] = {10, 10, 30, 10, 10, 10, 20, 20, 20, 20, 5, 20, 30, 5, 30};
    q7_t filter_data[] = {10, 10, 20, 20, 30, 10, 10, 20, 5, 5, 30, 10, 20, 20, 30};
    int32_t bias_data[] = {0, 0, 0};
    q7_t output_data[10] = {0};
    // memset(output_data, 1, sizeof(output_data));

    arm_status ret = arm_fully_connected_s8(NULL, &fc_params, &quant_params,
                        &input_dims, input_data, &filter_dims,
                        filter_data, &bias_dims, bias_data,
                        &output_dims, output_data);
    if (ret != ARM_MATH_SUCCESS) {
        printf("Error\r\n");
        return;
    }

    show_vector(output_data, sizeof(output_data));
}

void vec_mat_mult_s8_demo() {
    // 输入设置
    q7_t lhs[] = {1, 2, 1, 1, 1};
    // q7_t rhs[] = {10, 10, 30, 10, 10, 10, 20, 20, 20, 20, 5, 20, 30, 5, 30};
    q7_t rhs[] = {10, 10, 20, 20, 30, 10, 10, 20, 5, 5, 30, 10, 20, 20, 30};
    q31_t bias[] = {0, 0, 0};
    q7_t dst[10] = {0};

    int32_t lhs_offset = 0, rhs_offset = 0, dst_offset = 0;
    int32_t multiplier = (1 << 31) - 1, shift = 0;    // 无量化版本
    // int32_t multiplier = 1717986918, shift = -4;    // 量化版本
    int32_t rhs_cols = 5, rhs_rows = 3;
    int32_t activation_min = -128, activation_max = 127;

    arm_status ret = arm_nn_vec_mat_mult_t_s8(lhs, rhs, bias, dst,
                        lhs_offset, rhs_offset, dst_offset,
                        multiplier, shift, rhs_cols, rhs_rows,
                        activation_min, activation_max);
    if (ret != ARM_MATH_SUCCESS) {
        printf("Error\r\n");
        return;
    }

    show_vector(dst, sizeof(dst));
}

void erf_demo(float32_t* data, float32_t* output) {
    for (int i = 0; i < 4; i++) {
        output[i] = erff(data[i]);  // erff用于浮点32位的，erf用于64位的
    }
}