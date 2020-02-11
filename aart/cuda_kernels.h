#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

extern void add_with_cuda(const float* A, const float* B, float* C, int numElements);
extern void set_color_with_cuda(cv::InputArray input, cv::OutputArray output);

#endif
