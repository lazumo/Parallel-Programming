# HW1: Denoising with OpenMP

This project demonstrates the implementation of image denoising using OpenMP for parallelization. The main principle of denoising is to average the brightness (明度) of each pixel with its neighbors.

## File Structure

- `hw1.cpp`: The main source code file implementing the denoising algorithm.

## Key Modifications

1. **Summed Area Table**
   - Implemented a summed area table (also known as an integral image) to achieve constant-time access to the sum of subarray elements.
   - The summed area table allows for efficient computation of the average brightness of a pixel's neighborhood.

2. **OpenMP Parallelization**
   - Utilized the `#pragma omp parallel for collapse(2)` directive to distribute the workload across all available CPU cores.
   - The `collapse(2)` clause is used to parallelize nested loops and improve the granularity of parallelization.

## Usage

1. Compile the `hw1.cpp` file with OpenMP support:
   ```
   g++ -std=c++11 -O3 -lpng -lpthread -fopenmp hw1.cpp -o hw1
   ```

2. Run the compiled executable:
   ```
   srun  ./hw1 path/to/input.png path/to/output.png
   ```
   - `<input_image>`: The path to the input image file.
   - `<output_image>`: The path to save the denoised output image.

## Requirements

- C++ compiler with OpenMP support
- OpenCV library (for image I/O)

Here's a formatted version of your README file for the HW2 project:

# HW2: 3D Mandelbrot Set Computation with MPI

This project implements the computation of the 3D Mandelbrot set using the Message Passing Interface (MPI) for parallel processing. The main challenge in this task is achieving proper load balancing, as the time required to process each pixel varies. To address this, a master-slave model is implemented to facilitate dynamic load distribution among the available processes.

## File Structure

- `hw2.cpp`: The main source code file implementing the 3D Mandelbrot set computation using MPI.

## Implementation Details

1. **Master-Slave Model**
   - The master process is responsible for distributing tasks to the slave processes and collecting the results.
   - Slave processes receive tasks from the master, perform the computation for the assigned pixels, and send the results back to the master.
   - The master process dynamically assigns tasks to the slaves based on their availability, ensuring efficient load balancing.

2. **Dynamic Load Distribution**
   - The master process maintains a queue of tasks, representing the pixels to be computed.
   - Slave processes request tasks from the master when they become idle.
   - The master process assigns tasks to the slaves dynamically, taking into account the varying computation time for each pixel.
   - This dynamic load distribution ensures that all processes are kept busy and the workload is evenly distributed.

## Requirements

- C++ compiler with MPI support
- MPI library
Here's a formatted version of your README file for the HW3 project:

# HW3: Edge Detection with CUDA on RTX 4090
![Uploading report.out.png…]()


This project implements edge detection using CUDA on an NVIDIA RTX 4090 GPU. By leveraging the understanding of GPU architecture, various optimization techniques are employed to achieve high performance in the class.

## File Structure

- `hw3.cu`: The main source code file implementing edge detection using CUDA.

## Optimization Techniques

1. **Double to Float Conversion**
   - The input image data is converted from double precision to single precision floating-point format.
   - This optimization reduces memory bandwidth usage and improves computational efficiency on the GPU.

2. **Loop Unrolling**
   - Loops in the CUDA kernel are unrolled to reduce loop overhead and increase instruction-level parallelism.
   - Unrolling loops allows for better utilization of GPU resources and improves overall performance.

3. **Shared Memory Utilization**
   - Shared memory is used to store frequently accessed data, such as image tiles, during the edge detection process.
   - By copying data from global memory to shared memory, the kernel can achieve faster memory access and reduce latency.

4. **Coalesced Memory Access**
   - The CUDA kernel is designed to ensure coalesced memory access patterns.
   - By accessing memory in a contiguous and aligned manner, the kernel can maximize memory bandwidth utilization and minimize memory transaction overhead.

5. **Proper Task Distribution**
   - The workload is distributed among CUDA threads and blocks in a way that maximizes parallelism and minimizes idle time.
   - The task distribution takes into account the concept of coalesced memory access to ensure efficient utilization of GPU resources.

## Usage

1. Compile the `hw3.cu` file with CUDA support:
   ```
   nvcc hw3.cu -o hw3
   ```

2. Run the compiled executable:
   ```
   ./hw3 <input_image> <output_image>
   ```
   - `<input_image>`: The path to the input image file.
   - `<output_image>`: The path to save the edge-detected output image.

## Requirements

- NVIDIA CUDA Toolkit
- NVIDIA RTX 4090 GPU (or compatible GPU)
