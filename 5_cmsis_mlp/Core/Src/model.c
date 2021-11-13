#include "arm_nnsupportfunctions.h"
#include "stdio.h"

#include "model.h"    // 当这个文件调用同文件函数的时候就会需要了
#include "weights.h"

/**
 *  @ingroup groupNN
 */

  /**
   * @brief       mlp network
   * @param[in]       input_data   输入数据，但是大小需要设置成整个网络最大的input
   * @param[in,out]   output_data  输出数据，设置成整个网络最大的那个       
   * @return      None
   *
   * @details
   *
   * None
   *
   */
void mlp(q7_t *buffer1, q7_t *buffer2) {
    /*********** 网络通用参数信息 ************/
    cmsis_nn_activation activation_range;
    activation_range.min = -128;
    activation_range.max = 127;

    /*********** dense_1 ************/
    // 维度信息
    cmsis_nn_dims input_dims_dense_1, filter_dims_dense_1, bias_dims_dense_1, output_dims_dense_1;
    input_dims_dense_1.n = 1, input_dims_dense_1.h = 1, input_dims_dense_1.w = 784, input_dims_dense_1.c = 1;
    filter_dims_dense_1.n = 784, filter_dims_dense_1.c = 100;
    bias_dims_dense_1.c = 100;
    output_dims_dense_1.n = 1, output_dims_dense_1.c = 100;

    // context
    // The function returns required buffer size in bytes
    // int32_t buffer_size = arm_fully_connected_s8_get_buffer_size(&filter_dims_dense_1);
    // printf("%d\r\n", buffer_size);   // 0，不需要额外的内存

    // 量化参数
    cmsis_nn_fc_params fc_params_dense_1;
    fc_params_dense_1.input_offset = 128;     // 这就是直接在input上+1
    fc_params_dense_1.filter_offset = 0;
    fc_params_dense_1.output_offset = -128;    // 这就是在最后的结果上直接+2
    fc_params_dense_1.activation = activation_range;

    cmsis_nn_per_tensor_quant_params quant_params_dense_1;
    quant_params_dense_1.multiplier = 1459603623;
    quant_params_dense_1.shift = -10;

    /*********** dense_2 ************/
    // 维度信息
    cmsis_nn_dims input_dims_dense_2, filter_dims_dense_2, bias_dims_dense_2, output_dims_dense_2;
    input_dims_dense_2.n = 1, input_dims_dense_2.h = 1, input_dims_dense_2.w = 100, input_dims_dense_2.c = 1;
    filter_dims_dense_2.n = 100, filter_dims_dense_2.c = 10;
    bias_dims_dense_2.c = 10;
    output_dims_dense_2.n = 1, output_dims_dense_2.c = 10;

    // context
    // The function returns required buffer size in bytes
    // int32_t buffer_size = arm_fully_connected_s8_get_buffer_size(&filter_dims_dense_2);
    // printf("%d\r\n", buffer_size);   // 0，不需要额外的内存

    // 量化参数
    cmsis_nn_fc_params fc_params_dense_2;
    fc_params_dense_2.input_offset = 128;     // 这就是直接在input上+1
    fc_params_dense_2.filter_offset = 0;
    fc_params_dense_2.output_offset = 21;    // 这就是在最后的结果上直接+2
    fc_params_dense_2.activation = activation_range;

    cmsis_nn_per_tensor_quant_params quant_params_dense_2;
    quant_params_dense_2.multiplier = 1863683814;
    quant_params_dense_2.shift = -9;



    /*********** dense_1 inference ************/
    arm_status ret = arm_fully_connected_s8(NULL, &fc_params_dense_1, &quant_params_dense_1,
                        &input_dims_dense_1, buffer1, &filter_dims_dense_1,
                        filter_data_dense_1, &bias_dims_dense_1, bias_data_dense_1,
                        &output_dims_dense_1, buffer2);
    if (ret != ARM_MATH_SUCCESS) {
        printf("Error\r\n");
        return;
    }

    // memcpy(input_data, output_data, output_dims_dense_1.n * output_dims_dense_1.c); // 不可使用，复制并不是完全正确

    /*********** dense_2 inference ************/
    ret = arm_fully_connected_s8(NULL, &fc_params_dense_2, &quant_params_dense_2,
                        &input_dims_dense_2, buffer2, &filter_dims_dense_2,
                        filter_data_dense_2, &bias_dims_dense_2, bias_data_dense_2,
                        &output_dims_dense_2, buffer1);
    if (ret != ARM_MATH_SUCCESS) {
        printf("Error\r\n");
        return;
    }
}