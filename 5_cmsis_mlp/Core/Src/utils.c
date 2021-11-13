#include "arm_nnfunctions.h"
#include "stdio.h"

int get_vector_max_index(q7_t* vec, const int size, const int show_vector) {
    q7_t max_num = -128;
    int max_index = -1;

    for (int i = 0; i < size; i++) {
        if (vec[i] > max_num) {
            max_num = vec[i];
            max_index = i;
        }
        if (show_vector == 1)
            printf("%d ", vec[i]);
    }
    if (show_vector == 1)
        printf("\r\n");
    return max_index;
}