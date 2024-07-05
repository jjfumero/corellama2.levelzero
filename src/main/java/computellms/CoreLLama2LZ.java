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

/**
 * Level Zero Implementation of the Compute Llama2.c kernels using the Level Zero JNI library.
 *
 * <p>
 *  Next goal: imitate the OpenCL C++ program to dispatch the first kernel.
 * </p>
 *
 */
public class CoreLLama2LZ {
    public static void main(String[] args) {

        System.out.println("CoreLlama2.levelZero");

        ComputeBundle computeBundle = new ComputeBundle();
        computeBundle.initializeLevelZeroPlatform();

        LevelZeroKernel kernel = computeBundle.createKernel("copyData");

        // Data Initialization
        int numElements = 1024;
        ComputeBundle.MemBundle input = computeBundle.allocateSharedWithSegment(numElements);
        ComputeBundle.MemBundle output = computeBundle.allocateSharedWithSegment(numElements);

        computeBundle.initialzeHostData(input.segment, numElements);

        computeBundle.run(kernel, numElements, input.buffer, output.buffer);

        // print output
        System.out.println("Final result: ");
        computeBundle.print(output.segment);
        boolean isCorrect = computeBundle.checkFinalResult(output.segment, numElements);
        if (isCorrect) {
            System.out.println("Result is correct");
        } else {
            System.out.println("Result is wrong");
        }

    }
}