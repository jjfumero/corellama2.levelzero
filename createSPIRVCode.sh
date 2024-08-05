#! /usr/bin/env bash 

clang -cc1 -triple spir copyData.cl -O0 -finclude-default-header -emit-llvm-bc -o file.bc
llvm-spirv file.bc -o file.spv
retval=$?
if [ $retval -ne 0 ]; then
	echo "creating spir-v binary: file.spv ........... [failed]" 
else
	echo "creating spir-v binary: file.spv ........... [ok]" 
fi

clang -cc1 -triple spir kernels.cl -O0 -finclude-default-header -emit-llvm-bc -o kernels.bc
llvm-spirv kernels.bc -o kernels.spv
retval=$?
if [ $retval -ne 0 ]; then
	echo "creating spir-v binary: kernels.spv ........ [failed]" 
else
	echo "creating spir-v binary: kernels.spv ........ [ok]" 
fi


rm file.bc
rm kernels.bc
