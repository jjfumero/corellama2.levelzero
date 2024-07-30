# Core LLama2.c Math Functions in LevelZero & SPIR-V

**This repo shows a PoC** with the core math functions in Llama2.c written for OpenCL and compiled to SPIR-V binary.
The resulting SPIR-V kernels are dispatched using the Intel Level Zero API for Intel GPUs. 
Data is shared using Level Zero Shared Memory and Panama Segments. 

## Build

### 1) Build Intel/LLVM (If needed)

#### Compile OpenCL Kernel to SPIR-V 

Use CLANG from the Intel/LLVM Fork: 

```bash
git clone https://github.com/intel/llvm 
cd llvm 
python buildbot/configure.py
python buildbot/compile.py

## Export PATH to LLVM
export PATH=/home/juan/repos/SPIRV/llvm/build/bin:$PATH
```

Compile the OpenCL C kernels to SPIR-V using LLVM:

```bash
./createSPIRVCode.sh
```

### 2) Build Level Zero JNI Dependency

#### 2.1 Compile Intel Level Zero Library

```bash
cd $LEVEL_ZERO_ROOT
git clone https://github.com/oneapi-src/level-zero.git
cd level-zero
mkdir build
cd build
cmake ..
cmake --build . --config Release
cmake --build . --config Release --target package
```

#### 2.2 Compile the JNI Library    

a) Compile the native code: 

```bash
cd $LEVEL_ZERO_JNI
git clone https://github.com/beehive-lab/levelzero-jni
export ZE_SHARED_LOADER="$LEVEL_ZERO_ROOT/build/lib/libze_loader.so"
export CPLUS_INCLUDE_PATH=$LEVEL_ZERO_ROOT/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=$LEVEL_ZERO_ROOT/include:$CPLUS_INCLUDE_PATH
cd levelzero-jni/levelZeroLib
mkdir build
cd build
cmake ..
make
```

b) Compile the Java library:

```bash
cd $LEVEL_ZERO_JNI
mvn clean install
```

### 3) Compile `corellama2.levelzero` Java PoC Library

```bash
git clone https://github.com/jjfumero/corellama2.levelzero
mvn clean package
```

## Run

```bash
/home/juan/repos/SPIRV/levelzero-jni/levelZeroLib/build/
java -Djava.library.path=$LEVEL_ZERO_JNI/levelZeroLib/build/ \
  -cp target/corellama2.levelzero-1.0-SNAPSHOT.jar:$HOME/.m2/repository/beehive-lab/beehive-levelzero-jni/0.1.3/beehive-levelzero-jni-0.1.3.jar \
  --enable-native-access=ALL-UNNAMED \
  --enable-preview \
  computellms.CoreLLama2LZ
``` 

