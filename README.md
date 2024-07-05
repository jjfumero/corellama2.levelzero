
## Compile OpenCL Kernel to SPIR-V 

Use CLANG from the Intel/LLVM Fork:

```bash
clang -cc1 -triple spir kernels.cl -O0 -finclude-default-header -emit-llvm-bc -o kernels.bc
llvm-spirv kernels.bc -o kernels.spv
```

## Dissasemble the SPIR-V Code

```bash
spirv-dis kernels.spv
```
