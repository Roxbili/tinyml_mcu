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
    uint16_t out_shift = 9;
    uint16_t new_shift = NN_ROUND(out_shift);
    printf("%d\r\n", new_shift);
}