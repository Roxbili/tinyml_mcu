#include "arm_nnfunctions.h"

/**
 * @brief 获得指定size长度的vector中最大值对应的index
 *
 * @param vec 向量
 * @param size 向量长度
 * @param show_vector 是否打印该向量，1打印，其他值不打印
 * @return 该向量最大的数值的索引值
 *   @retval None
 */
int get_vector_max_index(q7_t* vec, const int size, const int show_vector);