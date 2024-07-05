
clang -cc1 -triple spir copyData.cl -O0 -finclude-default-header -emit-llvm-bc -o file.bc
llvm-spirv file.bc -o file.spv


clang -cc1 -triple spir kernels.cl -O0 -finclude-default-header -emit-llvm-bc -o kernels.bc
llvm-spirv kernels.bc -o kernels.spv

