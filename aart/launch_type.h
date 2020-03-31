#ifndef LAUNCH_TYPE_H
#define LAUNCH_TYPE_H

enum class launch_t
{
	cpu, // Use basic implementation
#ifdef AART_CUDA
	cuda // Use CUDA-accelerated version
#endif // AART_CUDA
};

#endif
