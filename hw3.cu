#include <png.h>
#include <zlib.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <time.h>

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define SCALE 8

clock_t start, end;
// clang-format off
__constant__ int mask[MASK_N][MASK_X][MASK_Y] = {
    {{ -1, -4, -6, -4, -1},
     { -2, -8,-12, -8, -2},
     {  0,  0,  0,  0,  0},
     {  2,  8, 12,  8,  2},
     {  1,  4,  6,  4,  1}},
    {{ -1, -2,  0,  2,  1},
     { -4, -8,  0,  8,  4},
     { -6,-12,  0, 12,  6},
     { -4, -8,  0,  8,  4},
     { -1, -2,  0,  2,  1}}
};
// clang-format on

int read_png(const char* filename, unsigned char** image, unsigned* height, unsigned* width,
    unsigned* channels) {
    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);

    if (!png_check_sig(sig, 8)) return 1; /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr) return 4; /* out of memory */


    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4; /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32 i, rowbytes;
    png_bytep row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int)png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char*)malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0; i < *height; ++i) {
        row_pointers[i] = *image + i * rowbytes;
    }

    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width,
    const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__global__ void sobel(unsigned char* s, unsigned char* t, unsigned height, unsigned width, unsigned channels) {
    __shared__ unsigned char shared_img[32][32][3];
    //__shared__ int shared_mask[MASK_N][MASK_X][MASK_Y];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * 28 + tx -2 ;
    int y = blockIdx.y * 28 + ty -2 ;

/*
    if (tx < MASK_X && ty < MASK_Y) {
        for (int i = 0; i < MASK_N; ++i) {
            shared_mask[i][tx][ty] = mask[i][tx][ty];
        }
    }*/

    // Load source image into shared memory
    if (x < width && y < height) {
        shared_img[ty][tx][0] = s[channels * (width * y + x) + 0];
        shared_img[ty][tx][1] = s[channels * (width * y + x) + 1];
        shared_img[ty][tx][2] = s[channels * (width * y + x) + 2];
    } else {
        shared_img[ty][tx][0] = 0;
        shared_img[ty][tx][1] = 0;
        shared_img[ty][tx][2] = 0;
    }
    
    __syncthreads();

    // Perform Sobel edge detection
    if (tx >= 2 && tx < 30 && ty >= 2 && ty < 30) {
        float val[6] = {0.0};
        #pragma unroll
        for (int v = -2; v <= 2; ++v) {
            #pragma unroll
            for (int u = -2; u <= 2; ++u) {
                val[0] += shared_img[ty+v][tx+u][2] * mask[0][u+2][v+2];
                val[1] += shared_img[ty+v][tx+u][1] * mask[0][u+2][v+2];
                val[2] += shared_img[ty+v][tx+u][0] * mask[0][u+2][v+2];
                val[3] += shared_img[ty+v][tx+u][2] * mask[1][u+2][v+2];
                val[4] += shared_img[ty+v][tx+u][1] * mask[1][u+2][v+2];
                val[5] += shared_img[ty+v][tx+u][0] * mask[1][u+2][v+2];
            }
        }

        float totalR = sqrt(val[0] * val[0] + val[3] * val[3]) / SCALE;
        float totalG = sqrt(val[1] * val[1] + val[4] * val[4]) / SCALE;
        float totalB = sqrt(val[2] * val[2] + val[5] * val[5]) / SCALE;

        unsigned char cR = min(totalR, 255.0f);
        unsigned char cG = min(totalG, 255.0f);
        unsigned char cB = min(totalB, 255.0f);

        if (x < width && y < height) {
            int  tmp = channels * (width * y + x);
            t[tmp + 2] = cR;
            t[tmp + 1] = cG;
            t[tmp + 0] = cB;
        }
    }
}

int main(int argc, char** argv) {
    
    assert(argc == 3);

    unsigned height, width, channels;
    unsigned char* src_img = NULL;

    read_png(argv[1], &src_img, &height, &width, &channels);
    assert(channels == 3);

    unsigned char* dst_img =
        (unsigned char*)malloc(height * width * channels * sizeof(unsigned char));

    unsigned char* d_src_img;
    unsigned char* d_dst_img;
    cudaMalloc((void**)&d_src_img, height * width * channels * sizeof(unsigned char));
    cudaMalloc((void**)&d_dst_img, height * width * channels * sizeof(unsigned char));

    // Copy source image to device memory
    cudaMemcpy(d_src_img, src_img, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(32, 32);
    dim3 gridDim((width + 27) / 28, (height + 27) / 28);
    start = clock();
    // Launch kernel
    sobel<<<gridDim, blockDim>>>(d_src_img, d_dst_img, height, width, channels);

    end = clock();
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("CPU執行時間: %f 秒\n", cpu_time_used);

    // Copy result from device memory to host
    cudaMemcpy(dst_img, d_dst_img, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_src_img);
    cudaFree(d_dst_img);

    write_png(argv[2], dst_img, height, width, channels);

    free(src_img);
    free(dst_img);




    return 0;
}
