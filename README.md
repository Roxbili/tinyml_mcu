# STM32cubeMX
在选择板子的时候最好选到STM32F746G-Discovery，创建工程后再把pinOUT清空。

UART可以参考YouTube那个印度人的教学视频。


------------------------------------------------------------------

# Keil

## 调试技巧
编译器优化级别高的时候，编译器为了优化，会优化变量内存分配，导致调试的时候无法看到局部变量的值(`not in scope`)

调整编译器优化级别便于调试：
1. options for target...
2. C/C++栏目
3. Opitmization: 从Level 3调整至Level 0

## CMSIS
手动添加CMSIS记得把path加到c++的项目里面，还需要把cubeMX生成的CMSIS路径都给换到自己的CMSIS库中，可能因为版本不一致导致include的问题。

但是CMSIS/Device/ST在原始的库下不存在，是从cubeMX复制过去的。

## stdio
需要开启Compiler -> I/O -> stdout(user)

然后创建以下函数，printf底层接口是调用这个函数实现的：
```c
int stdout_putchar(int ch)
{
    HAL_UART_Transmit(&huart1, (uint8_t *)&ch, 1, 0xFFFF);
    return ch;
}
```

## 添加外部文件
添加外部文件后需要去Keil中选择Manage Item，把`.c`文件添加进去才能正常编译，`.h`文件不用手动添加。



------------------------------------------------------------------

# CMSIS-NN

分为q7接口和s8接口两部分。

q7接口采用Qm.n定点量化方式，和TFLite per-tensor量化方式不一致，放弃尝试。
文档[CMSIS_NN_q7](4_cmsis_op/CMSIS_NN_q7.md)。

s8接口继续探索，文档[CMSIS_NN_s8](4_cmsis_op/CMSIS_NN_s8.md)。

------------------------------------------------------------------

# Vscode
Vscode上面的一些配置

## includePath
创建config：
```
command + shift + p: c/c++ edit config
```

然后修改里面的path，path可以参考keil，创建好的文件在`.vscode/c_cpp_properties.json`中

## SFTP
主要是windows侧要用Freesshd开启SFTP服务，然后vscode和windows传输的时候若启用`useTempFile`字段，那么将会造成temp file最后有个.new的后缀名无法删除，因为没有进一步修改文件的权限。
所以这个字段不能开。

------------------------------------------------------------------

# TODO

TFLite针对weight是零点为0，但是针对input、output零点范围是[-128,127]的（[链接](https://tensorflow.google.cn/lite/performance/quantization_spec?hl=zh-cn)）

因此，需要修改卷积啥的接口，在里面添加对应的零点偏移量化处理，让推理能够正常进行。实际上应该加一个input_zero_point和一个output_zero_point就行。