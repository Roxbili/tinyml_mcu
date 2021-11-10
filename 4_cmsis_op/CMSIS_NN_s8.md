# CMSIS_NN_s8

针对CMSIS_NN_s8接口算子的尝试。


## 量化简单推导：

基础量化公式如下：
```
real_value = (quant_value - zero_point) * scale
```

### TFLite量化规范
input/output:
```
dtype: int8
scale: yes
zero_point: [-128, 127]
```

filter:
```
dtype: int8
scale: yes
zero_point: 0
```

bias:
```
dtype: int32
scale: yes, bias_scale = input_scale * filter_scale (为了方便运算才这么设置的)
zero_point: 0
```

### 推导
```
output_r = input_r * filter_r + bias_r
         = input_scale * (input_q - input_zero_point) *
           filter_scale * filter_q +
           bias_scale * bias_q
         = input_scale * filter_scale * (input_q - input_zero_point) * filter_q + bias_scale * bias_q

output_q = output_r / output_scale + output_zero_point
```

由于`bias_scale = input_scale * filter_scale`，则：
令`scale = input_scale * filter_scale / output_scale`，则上式变为：
```
output_q = scale * [(input_q - input_zero_point) * filter_q + bias_q] + output_zero_point
```

此时，对上式进行简化：
```
output_q = scale * value + output_zero_point
```

由于`scale`是一个浮点数，在MCU中运算不方便，因此转换为：
```
output_q = (val * multiplier) / (2^shift) + output_zero_point
```

其中，`multiplier`和`shift`都是整型，实现了浮点数到整型的转换。

### 注意！
CMSIS-NE底层调用的反量化函数`arm_nn_requantize`有三个参数：`val, multipler, shift`。

其中shift已经内置了-31位，即结果至少是右移31位。
(想要补正只能把`multipler`设置成2^31，偏移量设置成0应该就行)

也就是说，上面的公式实际上是实现成了如下形式：
```
output_q = (val * multiplier) / (2^(shift + 31)) + output_zero_point
```

***(比较让人迷惑的就是不知道为什么要这么做)***

---------------------------------------

## Pooling

这里使用max pool方法展示结果。

输入输出顺序：`(N, H, W, C)`，其中N在函数说明中会写明是否有用。

这里以2通道为例子

输入：
```
channel 1:

-1  0  1  2
 2  3  5  4
 7  0  0  0
-1 -2  0  9


channel 2:

-1  0  1  2
 2  3  5  4
 7  0  0  0
-1 -2  0  9
```

此时`input_data`需要构建成：
```c
q7_t input_data[] = {-1, -1, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 5, 5, 4, 4,
                7, 7, 0, 0, 0, 0, 0, 0, -1, -1, -2, -2, 0, 0, 9, 9};
```

输出：
```
output_vec: 3, 3, 5, 5, 7, 7, 9, 9
```

---------------------------------------

## Dense

由于`arm_nn_requantize`的存在，里面已经内置了一个-31的偏移量，因此输入参数的时候需要注意。

输入和权重：
```
input:

1  -1
2  -2
1  -1
1  -1
1  -1


weight:

10  10  20  20  30
10  10  20  5   5
30  10  20  20  30
```

### 无量化
量化参数：
```
multipler: 2^31 - 1 
shift: 0
```

预期得到的结果：
```
100  -100
60   -60
120  -120
```

### 量化
量化参数：
```
scale: 0.05
multipler: 1717986918 
shift: -4
```

预期得到的结果：
```
5  -5
3  -3
6  -6
```

### 接口参数补充说明

参考官方接口参数即可，但是以下参数和官方描述不一致：

- filter_data: 顺序应该是`[C, N]`，即在上面的样例中是`[3, 5]`，官方说明文档有误。