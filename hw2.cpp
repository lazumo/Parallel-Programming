#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <omp.h>

#include <lodepng.h>

#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#define pi 3.1415926535897932384626433832795

typedef glm::dvec2 vec2;
typedef glm::dvec3 vec3;
typedef glm::dvec4 vec4;
typedef glm::dmat3 mat3;

unsigned int num_threads;
unsigned int width;
unsigned int height;
vec2 iResolution;

int AA = 2;

double power = 8.0;
double md_iter = 24;
double ray_step = 10000;
double shadow_step = 1500;
double step_limiter = 0.2;
double ray_multiplier = 0.1;
double bailout = 2.0;
double eps = 0.0005;
double FOV = 1.5;
double far_plane = 100.;

vec3 camera_pos;
vec3 target_pos;

unsigned char* raw_image;
unsigned char** image;

void write_png(const char* filename) {
    unsigned error = lodepng_encode32_file(filename, raw_image, width, height);

    if (error) printf("png error %u: %s\n", error, lodepng_error_text(error));
}

double md(vec3 p, double& trap) {
    vec3 v = p;
    double dr = 1.;             // |v'|
    double r = glm::length(v);  // r = |v| = sqrt(x^2 + y^2 + z^2)
    trap = r;

    for (int i = 0; i < md_iter; ++i) {
        double theta = glm::atan(v.y, v.x) * power;
        double phi = glm::asin(v.z / r) * power;
        dr = power * glm::pow(r, power - 1.) * dr + 1.;
        v = p + glm::pow(r, power) *
                    vec3(cos(theta) * cos(phi), cos(phi) * sin(theta), -sin(phi));  // update vk+1

        // orbit trap for coloring
        trap = glm::min(trap, r);

        r = glm::length(v);      // update r
        if (r > bailout) break;  // if escaped
    }
    return 0.5 * log(r) * r / dr;  // mandelbulb's DE function
}

// scene mapping
double map(vec3 p, double& trap, int& ID) {
    vec2 rt = vec2(cos(pi / 2.), sin(pi / 2.));
    vec3 rp = mat3(1., 0., 0., 0., rt.x, -rt.y, 0., rt.y, rt.x) *
              p;  // rotation matrix, rotate 90 deg (pi/2) along the X-axis
    ID = 1;
    return md(rp, trap);
}

// dummy function
// becase we dont need to know the ordit trap or the object ID when we are calculating the surface
// normal
double map(vec3 p) {
    double dmy;  // dummy
    int dmy2;    // dummy2
    return map(p, dmy, dmy2);
}

// simple palette function (borrowed from Inigo Quilez)
// see: https://www.shadertoy.com/view/ll2GD3
vec3 pal(double t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * glm::cos(2. * pi * (c * t + d));
}

// second march: cast shadow
// also borrowed from Inigo Quilez
// see: http://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm
double softshadow(vec3 ro, vec3 rd, double k) {
    double res = 1.0;
    double t = 0.;  // total distance
    for (int i = 0; i < shadow_step; ++i) {
        double h = map(ro + rd * t);
        res = glm::min(
            res, k * h / t);  // closer to the objects, k*h/t terms will produce darker shadow
        if (res < 0.02) return 0.02;
        t += glm::clamp(h, .001, step_limiter);  // move ray
    }
    return glm::clamp(res, .02, 1.);
}

// use gradient to calc surface normal
vec3 calcNor(vec3 p) {
    vec2 e = vec2(eps, 0.);
    return normalize(vec3(map(p + e.xyy()) - map(p - e.xyy()),  // dx
        map(p + e.yxy()) - map(p - e.yxy()),                    // dy
        map(p + e.yyx()) - map(p - e.yyx())                     // dz
        ));
}

// first march: find object's surface
double trace(vec3 ro, vec3 rd, double& trap, int& ID) {
    double t = 0;    // total distance
    double len = 0;  // current distance

    for (int i = 0; i < ray_step; ++i) {
        len = map(ro + rd * t, trap,
            ID);  // get minimum distance from current ray position to the object's surface
        if (glm::abs(len) < eps || t > far_plane) break;
        t += len * ray_multiplier;
    }
    return t < far_plane
               ? t
               : -1.;  // if exceeds the far plane then return -1 which means the ray missed a shot
}
int main(int argc, char** argv) {
    assert(argc == 11);

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    num_threads = atoi(argv[1]);
    camera_pos = vec3(atof(argv[2]), atof(argv[3]), atof(argv[4]));
    target_pos = vec3(atof(argv[5]), atof(argv[6]), atof(argv[7]));
    width = atoi(argv[8]);
    height = atoi(argv[9]);

    iResolution = vec2(width, height);

    raw_image = new unsigned char[width * height * 4];
    image = new unsigned char*[height];

    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < height; ++i) {
        image[i] = raw_image + i * width * 4;
    }

    if(size > 2){
        if (rank == 0) {
            // Master process
            int next_row = 0;
            int rows_per_request = 10; 
            int jobRequest[4];
            int start_row;
            int num_of_row;
            MPI_Status status;
            int slave_rank;
            int hasProduct;
            int terminated = 0;
            int myWork  =  2;
            while (next_row < height) {

                MPI_Recv(jobRequest, 4, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
                slave_rank = jobRequest[0];
                hasProduct =  jobRequest[1];
                start_row = jobRequest[2];
                num_of_row = jobRequest[3];

                if(hasProduct){
                    MPI_Recv(raw_image + start_row * width * 4, num_of_row * width * 4, MPI_UNSIGNED_CHAR, slave_rank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

                int rows_to_send = std::min(rows_per_request, static_cast<int>(height - next_row));
                int row_range[2] = {next_row, rows_to_send};
                MPI_Send(row_range, 2, MPI_INT, slave_rank, 1, MPI_COMM_WORLD);
                //MPI_Send(&next_row, 1, MPI_INT, slave_rank, 1, MPI_COMM_WORLD);
                //MPI_Send(&rows_to_send, 1, MPI_INT, slave_rank, 2, MPI_COMM_WORLD);


                next_row += rows_to_send;

                if(next_row>=height)
                    break;
                start_row =  next_row;

                #pragma omp parallel for num_threads(num_threads) collapse(2) schedule(dynamic)
                    for (int i = start_row; i < start_row + myWork; ++i) {
                        for (int j = 0; j < width; ++j) {
                            vec4 fcol(0.);
                            for (int m = 0; m < AA; ++m) {
                                for (int n = 0; n < AA; ++n) {
                                    vec2 p = vec2(j, i) + vec2(m, n) / (double)AA;

                                    vec2 uv = (-iResolution.xy() + 2. * p) / iResolution.y;
                                    uv.y *= -1;

                                    vec3 ro = camera_pos;
                                    vec3 ta = target_pos;
                                    vec3 cf = glm::normalize(ta - ro);
                                    vec3 cs = glm::normalize(glm::cross(cf, vec3(0., 1., 0.)));
                                    vec3 cu = glm::normalize(glm::cross(cs, cf));
                                    vec3 rd = glm::normalize(uv.x * cs + uv.y * cu + FOV * cf);

                                    double trap;
                                    int objID;
                                    double d = trace(ro, rd, trap, objID);

                                    vec3 col(0.);
                                    vec3 sd = glm::normalize(camera_pos);
                                    vec3 sc = vec3(1., .9, .717);

                                    if (d < 0.) {
                                        col = vec3(0.);
                                    } else {
                                        vec3 pos = ro + rd * d;
                                        vec3 nr = calcNor(pos);
                                        vec3 hal = glm::normalize(sd - rd);

                                        col = pal(trap - .4, vec3(.5), vec3(.5), vec3(1.), vec3(.0, .1, .2));
                                        vec3 ambc = vec3(0.3);
                                        double gloss = 32.;

                                        double amb = (0.7 + 0.3 * nr.y) * (0.2 + 0.8 * glm::clamp(0.05 * log(trap), 0.0, 1.0));
                                        double sdw = softshadow(pos + .001 * nr, sd, 16.);
                                        double dif = glm::clamp(glm::dot(sd, nr), 0., 1.) * sdw;
                                        double spe = glm::pow(glm::clamp(glm::dot(nr, hal), 0., 1.), gloss) * dif;

                                        vec3 lin(0.);
                                        lin += ambc * (.05 + .95 * amb);
                                        lin += sc * dif * 0.8;
                                        col *= lin;

                                        col = glm::pow(col, vec3(.7, .9, 1.));
                                        col += spe * 0.8;
                                    }

                                    col = glm::clamp(glm::pow(col, vec3(.4545)), 0., 1.);

                                    #pragma omp critical
                                    fcol += vec4(col, 1.);
                                }
                            }

                            fcol /= (double)(AA * AA);
                            fcol *= 255.0;
                            image[i][4 * j + 0] = (unsigned char)fcol.r;
                            image[i][4 * j + 1] = (unsigned char)fcol.g;
                            image[i][4 * j + 2] = (unsigned char)fcol.b;
                            image[i][4 * j + 3] = 255;
                        }
                    }
                    next_row += myWork;
            }

            // Terminate slave processes
            for (int i = 1; i < size; ++i) {

                MPI_Recv(jobRequest, 4, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
                slave_rank = jobRequest[0];
                hasProduct =  jobRequest[1];
                start_row = jobRequest[2];
                num_of_row = jobRequest[3];

                if(hasProduct){
                    MPI_Recv(raw_image + start_row * width * 4, num_of_row * width * 4, MPI_UNSIGNED_CHAR, slave_rank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

                int terminate = -1;
                int row_range[2] = {terminate, 0};
                MPI_Send(&terminate, 1, MPI_INT, slave_rank, 1, MPI_COMM_WORLD);
            }
        } else {
            int hasProduct = 0;
            int start_row;
            int num_rows;
            int row_range[2];
            //int jobRequest[4];
            // Slave processes
            while (true) {
                
                int  jobRequest[4] = {rank,hasProduct,start_row,num_rows};
                MPI_Send(jobRequest, 4, MPI_INT, 0, 0, MPI_COMM_WORLD);
                if(hasProduct){
                    MPI_Send(raw_image + start_row * width * 4, num_rows * width * 4, MPI_UNSIGNED_CHAR, 0, 3, MPI_COMM_WORLD);
                }
                MPI_Recv(row_range, 2, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                
                if (row_range[0] == -1) {
                    break;
                }
                start_row =  row_range[0];
                num_rows  =  row_range[1];

                //MPI_Recv(&num_rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    #pragma omp parallel for num_threads(num_threads) collapse(2) schedule(dynamic)
                    for (int i = start_row; i < start_row + num_rows; ++i) {
                        for (int j = 0; j < width; ++j) {
                            vec4 fcol(0.);
                            for (int m = 0; m < AA; ++m) {
                                for (int n = 0; n < AA; ++n) {
                                    vec2 p = vec2(j, i) + vec2(m, n) / (double)AA;

                                    vec2 uv = (-iResolution.xy() + 2. * p) / iResolution.y;
                                    uv.y *= -1;

                                    vec3 ro = camera_pos;
                                    vec3 ta = target_pos;
                                    vec3 cf = glm::normalize(ta - ro);
                                    vec3 cs = glm::normalize(glm::cross(cf, vec3(0., 1., 0.)));
                                    vec3 cu = glm::normalize(glm::cross(cs, cf));
                                    vec3 rd = glm::normalize(uv.x * cs + uv.y * cu + FOV * cf);

                                    double trap;
                                    int objID;
                                    double d = trace(ro, rd, trap, objID);

                                    vec3 col(0.);
                                    vec3 sd = glm::normalize(camera_pos);
                                    vec3 sc = vec3(1., .9, .717);

                                    if (d < 0.) {
                                        col = vec3(0.);
                                    } else {
                                        vec3 pos = ro + rd * d;
                                        vec3 nr = calcNor(pos);
                                        vec3 hal = glm::normalize(sd - rd);

                                        col = pal(trap - .4, vec3(.5), vec3(.5), vec3(1.), vec3(.0, .1, .2));
                                        vec3 ambc = vec3(0.3);
                                        double gloss = 32.;

                                        double amb = (0.7 + 0.3 * nr.y) * (0.2 + 0.8 * glm::clamp(0.05 * log(trap), 0.0, 1.0));
                                        double sdw = softshadow(pos + .001 * nr, sd, 16.);
                                        double dif = glm::clamp(glm::dot(sd, nr), 0., 1.) * sdw;
                                        double spe = glm::pow(glm::clamp(glm::dot(nr, hal), 0., 1.), gloss) * dif;

                                        vec3 lin(0.);
                                        lin += ambc * (.05 + .95 * amb);
                                        lin += sc * dif * 0.8;
                                        col *= lin;

                                        col = glm::pow(col, vec3(.7, .9, 1.));
                                        col += spe * 0.8;
                                    }

                                    col = glm::clamp(glm::pow(col, vec3(.4545)), 0., 1.);

                                    #pragma omp critical
                                    fcol += vec4(col, 1.);
                                }
                            }

                            fcol /= (double)(AA * AA);
                            fcol *= 255.0;
                            image[i][4 * j + 0] = (unsigned char)fcol.r;
                            image[i][4 * j + 1] = (unsigned char)fcol.g;
                            image[i][4 * j + 2] = (unsigned char)fcol.b;
                            image[i][4 * j + 3] = 255;
                        }
                    }
                    hasProduct  =  1;

                }
            }

            if (rank == 0) {
                write_png(argv[10]);
                delete[] raw_image;
                delete[] image;
            }

            MPI_Finalize();

            return 0;
    }
    else{
        int chunk_size = height / size;
            int start = rank * chunk_size;
            int end = (rank == size - 1) ? height : (rank + 1) * chunk_size;
            #pragma omp parallel for num_threads(num_threads) collapse(2) schedule(dynamic)
            for (int i = start; i < end; ++i) {
                for (int j = 0; j < width; ++j) {
                    vec4 fcol(0.);
                    for (int m = 0; m < AA; ++m) {
                        for (int n = 0; n < AA; ++n) {
                            vec2 p = vec2(j, i) + vec2(m, n) / (double)AA;

                            vec2 uv = (-iResolution.xy() + 2. * p) / iResolution.y;
                            uv.y *= -1;

                            vec3 ro = camera_pos;
                            vec3 ta = target_pos;
                            vec3 cf = glm::normalize(ta - ro);
                            vec3 cs = glm::normalize(glm::cross(cf, vec3(0., 1., 0.)));
                            vec3 cu = glm::normalize(glm::cross(cs, cf));
                            vec3 rd = glm::normalize(uv.x * cs + uv.y * cu + FOV * cf);

                            double trap;
                            int objID;
                            double d = trace(ro, rd, trap, objID);

                            vec3 col(0.);
                            vec3 sd = glm::normalize(camera_pos);
                            vec3 sc = vec3(1., .9, .717);

                            if (d < 0.) {
                                col = vec3(0.);
                            } else {
                                vec3 pos = ro + rd * d;
                                vec3 nr = calcNor(pos);
                                vec3 hal = glm::normalize(sd - rd);

                                col = pal(trap - .4, vec3(.5), vec3(.5), vec3(1.), vec3(.0, .1, .2));
                                vec3 ambc = vec3(0.3);
                                double gloss = 32.;

                                double amb = (0.7 + 0.3 * nr.y) * (0.2 + 0.8 * glm::clamp(0.05 * log(trap), 0.0, 1.0));
                                double sdw = softshadow(pos + .001 * nr, sd, 16.);
                                double dif = glm::clamp(glm::dot(sd, nr), 0., 1.) * sdw;
                                double spe = glm::pow(glm::clamp(glm::dot(nr, hal), 0., 1.), gloss) * dif;

                                vec3 lin(0.);
                                lin += ambc * (.05 + .95 * amb);
                                lin += sc * dif * 0.8;
                                col *= lin;

                                col = glm::pow(col, vec3(.7, .9, 1.));
                                col += spe * 0.8;
                            }

                            col = glm::clamp(glm::pow(col, vec3(.4545)), 0., 1.);

                            #pragma omp critical
                            fcol += vec4(col, 1.);
                        }
                    }

                    fcol /= (double)(AA * AA);
                    fcol *= 255.0;
                    image[i][4 * j + 0] = (unsigned char)fcol.r;
                    image[i][4 * j + 1] = (unsigned char)fcol.g;
                    image[i][4 * j + 2] = (unsigned char)fcol.b;
                    image[i][4 * j + 3] = 255;
                }
            }

            MPI_Gather(rank == 0 ? MPI_IN_PLACE : raw_image + start * width * 4, chunk_size * width * 4, MPI_UNSIGNED_CHAR,
                    raw_image, chunk_size * width * 4, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                write_png(argv[10]);
                delete[] raw_image;
                delete[] image;
            }

            MPI_Finalize();

            return 0;
    }

    
}