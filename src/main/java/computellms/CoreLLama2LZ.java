/*
 * MIT License
 *
 * Copyright (c) 2024, Juan Fumero.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
package computellms;

import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroKernel;

import java.lang.foreign.ValueLayout;

/**
 * Level Zero Implementation of the Compute Llama2.c kernels using the Level Zero JNI library.
 *
 * <p>
 *  Next goal: imitate the OpenCL C++ program to dispatch the first kernel.
 * </p>
 *
 */
public class CoreLLama2LZ {

    public static final int ELEMENTS = 4096;
    public static final int GROUP_SIZE = 256;

    public static final boolean TESTING = false;

    public void testingKernel() {

        ComputeBundle computeBundle = new ComputeBundle();
        computeBundle.initializeLevelZeroPlatform("file.spv");

        LevelZeroKernel kernel = computeBundle.createKernel("copyData");

        // Data Initialization
        int numElements = 1024;
        MemBundle input = computeBundle.allocateSharedWithSegment(numElements);
        MemBundle output = computeBundle.allocateSharedWithSegment(numElements);

        computeBundle.testingInitData(input.segment, numElements);

        computeBundle.runKernelTesting(kernel, numElements, input.buffer, output.buffer);

        computeBundle.print(output.segment);
        boolean isCorrect = computeBundle.testingCheckResult(output.segment, numElements);
        if (isCorrect) {
            System.out.println("Result is correct");
        } else {
            System.out.println("Result is wrong");
        }
    }

    public void runRMSNorm(ComputeBundle computeBundle, MemBundle dOutput, MemBundle dX, MemBundle dWeight, final int numElements) {

        LevelZeroKernel kernel1 = computeBundle.createKernel("rmsnormReduction");
        LevelZeroKernel kernel2 = computeBundle.createKernel("rmsnormNormalization");

        computeBundle.runRMSNorm1(kernel1, numElements, GROUP_SIZE, dOutput.buffer, dX.buffer);

        int numGroups = numElements / GROUP_SIZE;
        var val = dOutput.segment.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
        for (int i = 1; i < numGroups; i++) {
            val += dOutput.segment.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        System.out.println(val);
        float ss = val + 1e-5f;
        ss = (float) (1.0 / Math.sqrt(ss));
        System.out.println("VALUE: " + ss);

        computeBundle.runRMSNorm2(kernel2, numElements, ss, dOutput.buffer, dX.buffer, dWeight.buffer);
    }

    private void runSoftMax(ComputeBundle computeBundle, MemBundle dOutput, MemBundle dX, final int numElements) {

        // This operation is composed of three kernels
        LevelZeroKernel kernel1 = computeBundle.createKernel("softMaxReduction");
        LevelZeroKernel kernel2 = computeBundle.createKernel("softMaxExpAndSum");
        LevelZeroKernel kernel3 = computeBundle.createKernel("softMaxNormalization");

        computeBundle.runSoftMax1(kernel1, numElements, GROUP_SIZE, dOutput.buffer, dX.buffer);

        int numGroups = numElements / GROUP_SIZE;
        var maxValue = dOutput.segment.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
        for (int i = 1; i < numGroups; i++) {
            maxValue = Math.max(maxValue, dOutput.segment.getAtIndex(ValueLayout.JAVA_FLOAT, i));
        }
        System.out.println("VALUE-Max: " + maxValue);

        computeBundle.runSoftMax2(kernel2, numElements, GROUP_SIZE, dOutput.buffer, dX.buffer, maxValue);

        // final reduction
        var valSum = dOutput.segment.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
        for (int i = 1; i < numGroups; i++) {
            valSum += dOutput.segment.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        System.out.println("VALUE-Sum (2): " + valSum);

        computeBundle.runSoftMax3(kernel3, numElements, dX.buffer, valSum);
    }

    private void runMatMul(ComputeBundle computeBundle, MemBundle dXout, MemBundle dX, MemBundle dW, final int numElements) {
        LevelZeroKernel kernel1 = computeBundle.createKernel("matMul");
        computeBundle.runMatMul(kernel1, numElements, dXout.buffer, dX.buffer, dW.buffer, numElements);
        System.out.println("Mat-mul finished");
    }


    private static void computeLlama2CoreMethods() {
        CoreLLama2LZ coreLLama2LZ = new CoreLLama2LZ();
        ComputeBundle computeBundle = new ComputeBundle();
        computeBundle.initializeLevelZeroPlatform("kernels.spv");

        // Data Initialization
        int numElements = ELEMENTS;
        MemBundle dOutput = computeBundle.allocateSharedWithSegment(numElements);
        MemBundle dX = computeBundle.allocateSharedWithSegment(numElements);
        computeBundle.init1DRandom(dX.segment, numElements);
        MemBundle dWeight = computeBundle.allocateSharedWithSegment(numElements);
        computeBundle.init1DRandom(dWeight.segment, numElements);

        // rmsNorm
        coreLLama2LZ.runRMSNorm(computeBundle, dOutput, dX, dWeight, numElements);

        // softmax
        coreLLama2LZ.runSoftMax(computeBundle, dOutput, dX, numElements);

        // Matrix-Vector matMul
        MemBundle dXout = computeBundle.allocateSharedWithSegment(numElements);
        MemBundle dW = computeBundle.allocateSharedWithSegment(numElements * numElements);
        computeBundle.init1DRandom(dXout.segment, numElements);

        computeBundle.init2DRandom(dW.segment, numElements);

        coreLLama2LZ.runMatMul(computeBundle, dXout, dX, dW, numElements);
    }

    public static void main(String[] args) {
        if (TESTING) {
            // Kernel just for testing
            CoreLLama2LZ coreLLama2LZ = new CoreLLama2LZ();
            coreLLama2LZ.testingKernel();
        } else {
            computeLlama2CoreMethods();
        }
    }
}