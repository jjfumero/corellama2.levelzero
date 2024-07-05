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

    public void testingKernel() {

        ComputeBundle computeBundle = new ComputeBundle();
        computeBundle.initializeLevelZeroPlatform("file.spv");

        LevelZeroKernel kernel = computeBundle.createKernel("copyData");

        // Data Initialization
        int numElements = 1024;
        ComputeBundle.MemBundle input = computeBundle.allocateSharedWithSegment(numElements);
        ComputeBundle.MemBundle output = computeBundle.allocateSharedWithSegment(numElements);

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

    public void runRMSNorm() {
        ComputeBundle computeBundle = new ComputeBundle();
        computeBundle.initializeLevelZeroPlatform("kernels.spv");

        LevelZeroKernel kernel = computeBundle.createKernel("rmsnormReduction");

        // Data Initialization
        int numElements = 4;
        ComputeBundle.MemBundle dOutput = computeBundle.allocateSharedWithSegment(numElements);
        ComputeBundle.MemBundle dX = computeBundle.allocateSharedWithSegment(numElements);

        computeBundle.init(dX.segment, numElements);

        computeBundle.runRMSNorm1(kernel, numElements, 4, dOutput.buffer, dX.buffer);

        int numGroups = numElements / 4;
        var val = dOutput.segment.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
        for (int i = 1; i < numGroups; i++) {
            val += dOutput.segment.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        System.out.println(val);
        float ss = val + 1e-5f;
        ss = (float) (1.0 / Math.sqrt(ss));
        System.out.println("VALUE: " + ss);

    }


    public static void main(String[] args) {

       CoreLLama2LZ coreLLama2LZ = new CoreLLama2LZ();

       // Kernel just for testing
       //coreLLama2LZ.testingKernel();

       // rmsNorm
       coreLLama2LZ.runRMSNorm();;
    }
}