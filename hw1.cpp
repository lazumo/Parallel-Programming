#include <iostream>
#include <vector>
#include <algorithm>
#include <png.h>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <omp.h>
// #include <pthread.h>

struct RGB {
    int r, g, b;
};

int computeSum(const std::vector<std::vector<int>>& prefixSum, int x1, int y1, int x2, int y2) {
    double totalSum = prefixSum[x2][y2];
    double topArea = (x1 > 0) ? prefixSum[x1 - 1][y2] : 0;
    double leftArea = (y1 > 0) ? prefixSum[x2][y1 - 1] : 0;
    double overlapArea = (x1 > 0 && y1 > 0) ? prefixSum[x1 - 1][y1 - 1] : 0;

    return totalSum - topArea - leftArea + overlapArea;
}

void calculatePrefixSum(const std::vector<std::vector<RGB>>& inputImage, 
                        std::vector<std::vector<int>>& prefixSum,
                        int channel,
                        int height,
                        int width) {
    if(channel == 0){
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                prefixSum[i][j] = inputImage[i][j].r;
                if (i > 0) prefixSum[i][j] += prefixSum[i - 1][j];
                if (j > 0) prefixSum[i][j] += prefixSum[i][j - 1];
                if (i > 0 && j > 0) prefixSum[i][j] -= prefixSum[i - 1][j - 1];
            }
        }
    }

    if(channel == 1){
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                prefixSum[i][j] = inputImage[i][j].g;
                if (i > 0) prefixSum[i][j] += prefixSum[i - 1][j];
                if (j > 0) prefixSum[i][j] += prefixSum[i][j - 1];
                if (i > 0 && j > 0) prefixSum[i][j] -= prefixSum[i - 1][j - 1];
            }
        }
    }

    if(channel == 2){
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                prefixSum[i][j] = inputImage[i][j].b;
                if (i > 0) prefixSum[i][j] += prefixSum[i - 1][j];
                if (j > 0) prefixSum[i][j] += prefixSum[i][j - 1];
                if (i > 0 && j > 0) prefixSum[i][j] -= prefixSum[i - 1][j - 1];
            }
        }
    }

}

double calculateLuminance(const RGB& pixel) {
    return 0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b;
}



int determineKernelSize(double brightness) {
    return brightness > 128 ? 10 : 5;
}

void applyFilterToChannel(
    const std::vector<std::vector<int>>& input, 
    std::vector<std::vector<int>>& output, 
    const std::vector<std::vector<int>>& kernelSizes, 
    int height,
    int width
) {
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            int kernelSize = kernelSizes[x][y];
            int kernelRadius = kernelSize / 2;
            double sum = 0.0;
            double filteredPixel = 0.0;

            for (int i = -kernelRadius; i <= kernelRadius; i++) {
                for (int j = -kernelRadius; j <= kernelRadius; j++) {
                    int pixelX = std::min(std::max(x + i, 0), height - 1);
                    int pixelY = std::min(std::max(y + j, 0), width - 1);
                    filteredPixel += input[pixelX][pixelY];
                    sum += 1.0;
                }
            }

            output[x][y] = static_cast<int>(filteredPixel / sum);
        }
    }
}

void myApplyFilterToChannel(
    //const std::vector<std::vector<int>>& input, 
    const std::vector<std::vector<int>>& prefixSum, 
    std::vector<std::vector<int>>& output, 
    const std::vector<std::vector<int>>& kernelSizes, 
    int height,
    int width
){
    #pragma omp parallel for collapse(2) 
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            int kernelSize = kernelSizes[x][y];
            int kernelRadius = kernelSize >> 1;
            double sum = 0.0;
            double filteredPixel = 0.0;
            double bordercheck;

            sum =  kernelSize == 10 ? 121  :  25;
            int x1 = std::max(x - kernelRadius, 0);
            int y1 = std::max(y - kernelRadius, 0);
            int x2 = std::min(x + kernelRadius, height - 1);
            int y2 = std::min(y + kernelRadius, width - 1);

            bordercheck  =  (x2 - x1 + 1) * (y2 - y1 + 1);

            if(bordercheck!=sum){
                
                for (int i = -kernelRadius; i <= kernelRadius; i++) {
                    for (int j = -kernelRadius; j <= kernelRadius; j++) {
                        int pixelX = std::min(std::max(x + i, 0), height - 1);
                        int pixelY = std::min(std::max(y + j, 0), width - 1);
                        filteredPixel += computeSum(prefixSum, pixelX, pixelY,pixelX, pixelY);
                    }
                }
            }
            else{
                filteredPixel = computeSum(prefixSum, x1, y1,x2,y2);
            }
            output[x][y] = static_cast<int>(filteredPixel / sum);
        }
    }
}

void adaptiveFilterRGB(
    const std::vector<std::vector<RGB>>& inputImage,
    std::vector<std::vector<RGB>>& outputImage,
    int height, 
    int width
) {
    //std::vector<std::vector<int>> redChannel(height, std::vector<int>(width));
    //std::vector<std::vector<int>> greenChannel(height, std::vector<int>(width));
    //std::vector<std::vector<int>> blueChannel(height, std::vector<int>(width));

    /*
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            redChannel[x][y] = inputImage[x][y].r;
            greenChannel[x][y] = inputImage[x][y].g;
            blueChannel[x][y] = inputImage[x][y].b;
        }
    }*/

    std::vector<std::vector<int>> redPrefix(height, std::vector<int>(width));
    std::vector<std::vector<int>> greenPrefix(height, std::vector<int>(width));
    std::vector<std::vector<int>> bluePrefix(height, std::vector<int>(width));

    #pragma omp parallel sections
    {
		#pragma omp section
		{
            calculatePrefixSum(inputImage, redPrefix,0,height,width);
		}
		#pragma omp section
		{
            calculatePrefixSum(inputImage, greenPrefix,1,height,width);
		}
		#pragma omp section
		{
            calculatePrefixSum(inputImage, bluePrefix,2,height,width);
		}

    }
    
    std::vector<std::vector<int>> kernelSizes(height, std::vector<int>(width));
    #pragma omp parallel for collapse(2)
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            double brightness = calculateLuminance(inputImage[x][y]);
            kernelSizes[x][y] = determineKernelSize(brightness);
        }
    }

    //std::vector<std::vector<int>> tempRed(height, std::vector<int>(width));
    //std::vector<std::vector<int>> tempGreen(height, std::vector<int>(width));
    //std::vector<std::vector<int>> tempBlue(height, std::vector<int>(width));

    
    std::vector<std::vector<int>> mytempRed(height, std::vector<int>(width));
    std::vector<std::vector<int>> mytempGreen(height, std::vector<int>(width));
    std::vector<std::vector<int>> mytempBlue(height, std::vector<int>(width));

    //applyFilterToChannel(redChannel, tempRed, kernelSizes, height, width);
    //applyFilterToChannel(greenChannel, tempGreen, kernelSizes, height, width);
    //applyFilterToChannel(blueChannel, tempBlue, kernelSizes, height, width);
    myApplyFilterToChannel(/*redChannel,*/ redPrefix,mytempRed, kernelSizes, height, width);
    myApplyFilterToChannel(/*greenChannel,*/ greenPrefix,mytempGreen, kernelSizes, height, width);
    myApplyFilterToChannel(/*blueChannel,*/ bluePrefix,mytempBlue, kernelSizes, height, width);
    /*
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            printf("Origin RGB / My RGB: %d \n",tempRed[x][y]==mytempRed[x][y] && tempGreen[x][y]==mytempGreen[x][y] && tempBlue[x][y]==mytempBlue[x][y]);
        }
    }*/
    #pragma omp parallel for collapse(2)
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            outputImage[x][y].r = mytempRed[x][y];
            outputImage[x][y].g = mytempGreen[x][y];
            outputImage[x][y].b = mytempBlue[x][y];
        }
    }
}



void read_png_file(char* file_name, std::vector<std::vector<RGB>>& image) {
    FILE *fp = fopen(file_name, "rb");
    if (!fp) {
        std::cerr << "Error: Cannot open file " << file_name << std::endl;
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        std::cerr << "Error: Cannot create PNG read structure" << std::endl;
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        std::cerr << "Error: Cannot create PNG info structure" << std::endl;
        png_destroy_read_struct(&png, nullptr, nullptr);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png))) {
        std::cerr << "Error during PNG creation" << std::endl;
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if(bit_depth == 16)
        png_set_strip_16(png);

    if(color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if(png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    if(color_type == PNG_COLOR_TYPE_RGB ||
       color_type == PNG_COLOR_TYPE_GRAY ||
       color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    if(color_type == PNG_COLOR_TYPE_GRAY ||
       color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);

    #pragma omp parallel for
    for(int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
    }

    png_read_image(png, row_pointers);

    fclose(fp);

    image.resize(height, std::vector<RGB>(width));
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for (int x = 0; x < width; x++) {
            png_bytep px = &(row[x * 4]);
            image[y][x].r = px[0];
            image[y][x].g = px[1];
            image[y][x].b = px[2];
        }
        free(row_pointers[y]);
    }
    free(row_pointers);

    png_destroy_read_struct(&png, &info, nullptr);
}

void write_png_file(char* file_name, std::vector<std::vector<RGB>>& image) {
    
    int width = image[0].size();
    int height = image.size();

    FILE *fp = fopen(file_name, "wb");
    if (!fp) {
        std::cerr << "Error: Cannot open file " << file_name << std::endl;
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        std::cerr << "Error: Cannot create PNG write structure" << std::endl;
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        std::cerr << "Error: Cannot create PNG info structure" << std::endl;
        png_destroy_write_struct(&png, nullptr);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png))) {
        std::cerr << "Error during PNG creation" << std::endl;
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    //modified
    png_set_compression_level(png,0);
    png_set_filter(png, PNG_FILTER_TYPE_BASE, PNG_FILTER_NONE);


    png_init_io(png, fp);

    png_set_IHDR(
        png,
        info,
        width, height,
        8,
        PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);

    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    

    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
        for (int x = 0; x < width; x++) {
            row_pointers[y][x * 3] = image[y][x].r;
            row_pointers[y][x * 3 + 1] = image[y][x].g;
            row_pointers[y][x * 3 + 2] = image[y][x].b;
        }
    }

    png_write_image(png, row_pointers);
    png_write_end(png, nullptr);

    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);

    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <inputfile.png> <outputfile.png>" << std::endl;
        return -1;
    }
    //omp_set_num_threads(16);
    // auto start_all = std::chrono::high_resolution_clock::now();

    char* input_file = argv[1];
    char* output_file = argv[2];

    std::vector<std::vector<RGB>> inputImage;
    //double start_time = omp_get_wtime();

    read_png_file(input_file, inputImage);

    //double end_time = omp_get_wtime();
    //printf("read_png_file time: %f seconds\n", end_time - start_time);

    int height = inputImage.size();
    int width = inputImage[0].size();

    std::vector<std::vector<RGB>> outputImage(height, std::vector<RGB>(width));

    // auto start = std::chrono::high_resolution_clock::now();
    //start_time = omp_get_wtime();
    adaptiveFilterRGB(inputImage, outputImage, height, width);
    //end_time = omp_get_wtime();
    //printf("adaptiveFilterRGB time: %f seconds\n", end_time - start_time);
    // adaptiveFilterRGB_parallel(inputImage, outputImage, height, width);

    // auto end = std::chrono::high_resolution_clock::now();

    // std::chrono::duration<double> elapsed_seconds = end - start;
    // std::cout << "Main Program Time: " << elapsed_seconds.count() * 1000.0 << " ms" << std::endl;
    //start_time = omp_get_wtime();
    write_png_file(output_file, outputImage);
    //end_time = omp_get_wtime();
    //printf("write_png_file time: %f seconds\n", end_time - start_time);

    // auto end_all = std::chrono::high_resolution_clock::now();
    // elapsed_seconds = end_all - start_all;
    // std::cout << "Total Program Time: " << elapsed_seconds.count() * 1000.0 << " ms" << std::endl;

    return 0;
}
                /*
                x1 = x - kernelRadius;
                y1 = y - kernelRadius;
                x2 = x + kernelRadius;
                y2 = y + kernelRadius;

                if(x1 <  0){
                    filteredPixel += (0 - x1) * computeSum(prefixSum, 0, std::max(0,y1),0,std::min(y2,width-1));
                }
                if(x2 >  height - 1){
                    filteredPixel +=(x2 - (height - 1) ) * computeSum(prefixSum, height - 1,std::max(0,y1), height - 1,std::min(y2,width-1));
                }
                if(y1 <  0){
                    filteredPixel +=  (0 - y1) * computeSum(prefixSum, std::max(0,x1),0, std::min(x2,height-1),0); 
                }
                if(y2 >  width - 1){
                    filteredPixel += (y2 - (width - 1) ) * computeSum(prefixSum, std::max(0,x1), width - 1, std::min(x2,height-1), width - 1); 
                }
                //lefttop
                if(x1 < 0  &&  y1 < 0){
                    filteredPixel += (0-x1)*(0-y1)*computeSum(prefixSum, 0,0, 0,0); 
                }
                //righttop
                if(x1 < 0  &&  y2 > width - 1){
                    filteredPixel += (0-x1)*(y2-(width - 1))*computeSum(prefixSum, 0,width - 1, 0,width - 1); 
                }
                //rightbotton
                if(x2 > height-1 &&  y2 > width - 1)
                    filteredPixel += (x2-(height-1))*(y2-(width - 1))*computeSum(prefixSum, height-1,width - 1, height-1,width - 1); 
                //left
                if(x2 > height-1 &&  y1 < 0)
                    filteredPixel += (x2-(height-1))*(0-y1)*computeSum(prefixSum, height-1,0, height-1,0); 
                
                */