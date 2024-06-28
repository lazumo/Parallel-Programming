#include <iostream>
#include <cstdlib>
#include <string>
#include <utility>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <openacc.h>
#include "mnist/mnist_reader.hpp"

using namespace std;

#define RESULT_FILE "result.txt"
#define MNIST_DATA_LOCATION "/tmp/dataset-nthu-ipc24/share/hw6/testcases/MNIST"
#define WEIGHT_ROOT "/tmp/dataset-nthu-ipc24/share/hw6/cnn_weights"

/* Model architecture:
 *  Conv1: (Cin, Cout, H, W) = (1, 16, 5, 5)
 *  Conv1: (Cin, Cout, H, W) = (16, 32, 5, 5)
 *  FC1: 1568 * 512
 *  FC2: 512 * 10
 */
#define IMAGE_SIZE 28
#define C1 16
#define C2 32
#define K 5
#define FC1 1568 // 32*7*7
#define FC2 512
#define OUT 10

void Padding2D(float *A, float *B, int n, int size) {
    int p = (K - 1) / 2;
    int padded_size = size + 2 * p;
    int size_size = size * size;
    int padded_size_padded_size = padded_size * padded_size;

    #pragma acc parallel loop present(A[0:n * size_size], B[0:n * padded_size_padded_size])
    for (int i = 0; i < n; i++) {
        float *A_ptr = A + i * size_size;
        float *B_ptr = B + i * padded_size_padded_size;

        // Fill top and bottom edges
        #pragma acc loop vector
        for (int j = 0; j < p; j++) {
            #pragma acc loop vector
            for (int k = 0; k < padded_size; k++) {
                B_ptr[j * padded_size + k] = 0;
                B_ptr[(padded_size - 1 - j) * padded_size + k] = 0;
            }
        }

        // Fill left and right edges and copy inner elements
        #pragma acc loop vector
        for (int j = p; j < size + p; j++) {
            int j_padded_size = j * padded_size;
            #pragma acc loop vector
            for (int k = 0; k < p; k++) {
                B_ptr[j_padded_size + k] = 0;
                B_ptr[j_padded_size + padded_size - 1 - k] = 0;
            }
            #pragma acc loop vector
            for (int k = p; k < size + p; k++) {
                B_ptr[j_padded_size + k] = A_ptr[(j - p) * size + (k - p)];
            }
        }
    }
}

void Conv2D(float *A, float *B, float *C, float *D, int n, int cin, int cout, int size) {
    int padded_size = size + K - 1;

    #pragma acc parallel loop collapse(2) gang present(A[0:n * cin * padded_size * padded_size], B[0:cout * cin * K * K], C[0:cout], D[0:n * cout * size * size])
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < cout; j++) {
            #pragma acc loop collapse(2) vector
            for (int x = 0; x < size; x++) {
                for (int y = 0; y < size; y++) {
                    float sum = C[j];

                    #pragma acc loop collapse(3) reduction(+:sum)
                    for (int k = 0; k < cin; k++) {
                        for (int kx = 0; kx < K; kx++) {
                            for (int ky = 0; ky < K; ky++) {
                                sum += B[(j * cin + k) * K * K + kx * K + ky] * A[(i * cin + k) * padded_size * padded_size + (x + kx) * padded_size + (y + ky)];
                            }
                        }
                    }

                    D[(i * cout + j) * size * size + x * size + y] = sum;
                }
            }
        }
    }
}

void ReLU(float *A, int n) {
    #pragma acc parallel loop present(A[0:n])
    for (int i = 0; i < n; i++) {
        A[i] = fmaxf(0.0f, A[i]);
    }
}

void MaxPool2D(float *A, float *B, int n, int size) {
    int pool_size = size / 2;

    #pragma acc parallel loop collapse(3) gang vector present(A[0:n * size * size], B[0:n * pool_size * pool_size])
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < pool_size; j++) {
            for (int k = 0; k < pool_size; k++) {
                int idx = i * size * size + (2 * j) * size + (2 * k);
                float max_val = A[idx];

                max_val = fmaxf(max_val, A[idx + 1]);
                max_val = fmaxf(max_val, A[idx + size]);
                max_val = fmaxf(max_val, A[idx + size + 1]);

                B[i * pool_size * pool_size + j * pool_size + k] = max_val;
            }
        }
    }
}

void ConvolutionLayer(float *A, float *B, float *C, float *D, int n, int cin, int cout, int size) {
    float *padded_input = new float[n * cin * (size + K - 1) * (size + K - 1)];
    float *conv_output = new float[n * cout * size * size];

    #pragma acc data present(A[0:n * cin * size * size], D[0:n * cout * size / 2 * size / 2]) present(B[0:cout * cin * K * K], C[0:cout]) create(padded_input[0:n * cin * (size + K - 1) * (size + K - 1)], conv_output[0:n * cout * size * size])
    {
        Padding2D(A, padded_input, n * cin, size);
        Conv2D(padded_input, B, C, conv_output, n, cin, cout, size);
        ReLU(conv_output, n * cout * size * size);
        MaxPool2D(conv_output, D, n * cout, size);
    }

    delete[] padded_input;
    delete[] conv_output;
}

void LinearLayer(float *A, float *B, float *C, float *D, int n, int k, int m) {
    #pragma acc kernels copyin(A[0:n*k], B[0:k*m], C[0:m]) copyout(D[0:n*m])
    {
        #pragma acc loop independent
        for (int i = 0; i < n; i++) {
            #pragma acc loop independent
            for (int j = 0; j < m; j++) {
                float sum = C[j];
                #pragma acc loop reduction(+:sum)
                for (int a = 0; a < k; a++) {
                    sum += A[i * k + a] * B[a * m + j];
                }
                D[i * m + j] = sum;
            }
        }
    }
}

void Sigmoid(float *A, int n, int m) {
    #pragma acc parallel loop present(A[0:n * m])
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A[i * m + j] = 1.0f / (1.0f + expf(-A[i * m + j]));
        }
    }
}

void Argmax(float *A, int *D, int n, int m) {
    #pragma acc parallel loop present(A[0:n * m], D[0:n])
    for (int i = 0; i < n; i++) {
        float mx = A[i * m];
        int index = 0;

        #pragma acc loop reduction(max:mx) reduction(max:index)
        for (int j = 1; j < m; j++) {
            if (mx < A[i * m + j]) {
                mx = A[i * m + j];
                index = j;
            }
        }
        D[i] = index;
    }
}

void my_cnn(float *training_images_flat, int num_images,
            float *conv1_weight, float *conv1_bias, float *conv2_weight, float *conv2_bias, 
            float *fc1_weight, float *fc1_bias, float *fc2_weight, float *fc2_bias,
            int *result) {
    int num_pixels = IMAGE_SIZE * IMAGE_SIZE;

    float *conv1_output = new float[num_images * C1 * IMAGE_SIZE / 2 * IMAGE_SIZE / 2];
    float *conv2_output = new float[num_images * C2 * IMAGE_SIZE / 4 * IMAGE_SIZE / 4];
    float *fc1_output = new float[num_images * FC2];
    float *fc2_output = new float[num_images * OUT];

    #pragma acc data create(conv1_output[0:num_images * C1 * IMAGE_SIZE / 2 * IMAGE_SIZE / 2], conv2_output[0:num_images * C2 * IMAGE_SIZE / 4 * IMAGE_SIZE / 4], fc1_output[0:num_images * FC2], fc2_output[0:num_images * OUT]) copyout(result[0:num_images])
    #pragma acc data copyin(training_images_flat[0:num_images * num_pixels])
    #pragma acc data copyin(conv1_weight[0:1 * C1 * K * K], conv1_bias[0:C1])
    #pragma acc data copyin(conv2_weight[0:C1 * C2 * K * K], conv2_bias[0:C2])
    #pragma acc data copyin(fc1_weight[0:FC1 * FC2], fc1_bias[0:FC2])
    #pragma acc data copyin(fc2_weight[0:FC2 * OUT], fc2_bias[0:OUT])
    {
        ConvolutionLayer(training_images_flat, conv1_weight, conv1_bias, conv1_output, num_images, 1, C1, IMAGE_SIZE);
        ConvolutionLayer(conv1_output, conv2_weight, conv2_bias, conv2_output, num_images, C1, C2, IMAGE_SIZE / 2);
        LinearLayer(conv2_output, fc1_weight, fc1_bias, fc1_output, num_images, FC1, FC2);
        LinearLayer(fc1_output, fc2_weight, fc2_bias, fc2_output, num_images, FC2, OUT);
        Argmax(fc2_output, result, num_images, OUT);
    }

    delete[] conv1_output;
    delete[] conv2_output;
    delete[] fc1_output;
    delete[] fc2_output;
}


/* Read neural network's weight from file (in binary format)
 */
void read_weight(float *array, string filename, int num_floats) {
    string full_filename = string(WEIGHT_ROOT) + '/' + filename;
    std::cout << "Reading file: " << full_filename << std::endl;

    ifstream file(full_filename, ios::in | ios::binary);
    if (!file) {
        std::cerr << "error reading file: " << full_filename << std::endl;
        exit(1);
    }
    file.read((char *)array, num_floats * sizeof(float));
}

/* Write predicted result to file
 */
void write_predict(int *result, int n, char *filename) {
    std::ofstream file(filename, std::ofstream::out);
    for (int i = 0; i < n; i++) {
        file << result[i] << '\n';
    }
    file.close();
}

/* Print an image
 * Usage: print_img(training_images[i])
 */
void print_img(float *img) {
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            if (img[i * 28 + j] > 0.5) {
                std::cout << 'x';
            }
            else {
                std::cout << ' ';
            }
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {
    auto read_start = std::chrono::steady_clock::now();
    // std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    /* Load MNIST data
     */
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset;
    if (argc == 1) {
        dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);
    }else {
        dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(argv[1]);
    }

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    // std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    // std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    // std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

    //int num_train_images = dataset.training_images.size();
    int num_train_images = 30000; // due to CT memory limit
    int num_pixels = dataset.training_images.front().size(); // should be 28*28 = 784

    /* Convert 60000 training images from [0, 255] to [0, 1]
     * We will first generate another 2D array by `new`
     */

    /* training_images_flat[i*num_pixels + j] == training_images[i][j]
     * j-th pixel in i-th image
     */
    float *training_images_flat = new float[num_train_images * num_pixels];

    float **training_images = new float *[num_train_images];
    for (int i = 0; i < num_train_images; i++) {
        training_images[i] = training_images_flat + i * num_pixels;
    }

    for (int i = 0; i < num_train_images; i++) {
        for (int j = 0; j < num_pixels; j++) {
            training_images[i][j] = (float)(dataset.training_images[i][j]) / 255.0;
        }
    }

    /* Print first image */
    // print_img(training_images[0]);

    /* Load matrices' weight from binary file
     * You can print the binary file by: `od -f weights/conv1_bias`
     * https://stackoverflow.com/questions/36791622/how-to-print-float-value-from-binary-file-in-shell
     */
    float *conv1_weight = new float[C1 * 1 * K * K];
    float *conv1_bias = new float[C1];
    float *conv2_weight = new float[C2 * C1 * K * K];
    float *conv2_bias = new float[C2];
    float *fc1_weight = new float[FC1 * FC2];
    float *fc1_bias = new float[FC2];
    float *fc2_weight = new float[FC2 * OUT];
    float *fc2_bias = new float[OUT];
    read_weight(conv1_weight, "conv1_weight", C1 * 1 * K * K);
    read_weight(conv1_bias, "conv1_bias", C1);
    read_weight(conv2_weight, "conv2_weight", C2 * C1 * K * K);
    read_weight(conv2_bias, "conv2_bias", C2);
    read_weight(fc1_weight, "fc1_weight", FC1 * FC2);
    read_weight(fc1_bias, "fc1_bias", OUT);
    read_weight(fc2_weight, "fc2_weight", FC2 * OUT);
    read_weight(fc2_bias, "fc2_bias", OUT);

    auto read_end = std::chrono::steady_clock::now();

    /* Inference */
    int *result = new int[num_train_images];
    my_cnn(training_images_flat, num_train_images,
           conv1_weight, conv1_bias, conv2_weight, conv2_bias, 
           fc1_weight, fc1_bias, fc2_weight, fc2_bias, result);
    auto inference_end = std::chrono::steady_clock::now();

    /* Calculate accuracy */
    int correct = 0;
    int total = 0;
    for (int i = 0; i < num_train_images; i++) {
        if ((int)result[i] == (int)dataset.training_labels[i]) {
            correct++;
        }
        total++;
    }
    std::cout << "\nInference accuracy: " << (double)correct / (double)total * 100.0 << "%\n";
    if (argc < 3) {
        write_predict(result, num_train_images, RESULT_FILE);
    }else {
        write_predict(result, num_train_images, argv[2]);
    }

    auto acc_end = std::chrono::steady_clock::now();

    std::cout << std::setprecision(5) << std::fixed;
    std::cout << "\n-----     STATS     -----\n";
    std::cout << "Time for reading MNIST data & weights: " << std::chrono::duration_cast<std::chrono::milliseconds>(read_end - read_start).count() << " m.s.\n";
    std::cout << "Time for inferencing                 : " << std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - read_end).count() << " m.s.\n";
    std::cout << "Time for calculating accuracy        : " << std::chrono::duration_cast<std::chrono::milliseconds>(acc_end - inference_end).count() << " m.s.\n";
    std::cout << "----- END OF STATS  -----\n";

    delete[] result;
    delete[] conv1_weight;
    delete[] conv1_bias;
    delete[] conv2_weight;
    delete[] conv2_bias;
    delete[] fc1_weight;
    delete[] fc1_bias;
    delete[] fc2_weight;
    delete[] fc2_bias;
    delete[] training_images_flat;
    delete[] training_images;
    return 0;
}