#include <cuda.h>
#include "nn_kernels.h"

__global__ void fill_ones_kernel(float *arr, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n)
		arr[idx] = 1.0f;
}

__global__ void active_kernel(float *in, float *out, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n)
		out[idx] = tanhf(in[idx]);
}

__global__ void active_prime_kernel(float *in, float *out, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n)
	{
		float t = tanhf(in[idx]);
		out[idx] = 1 - t * t;
	}
}

//c = a - b
__global__ void subtract_kernel(float *a, float *b, float *c, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n)
		c[idx] = a[idx] - b[idx];
}

//c = a .* b
__global__ void mul_kernel(float *a, float *b, float *c, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n)
		c[idx] = a[idx] * b[idx];
}
void fill_ones(float *d_arr, int n)
{
	int block_size = 1024;
	int grid_size = (n + block_size - 1) / block_size;
	fill_ones_kernel<<<grid_size, block_size>>>(d_arr, n);
}

void active(float *d_in, float *d_out, int n)
{
	int block_size = 1024;
	int grid_size = (n + block_size - 1) / block_size;
	active_kernel<<<grid_size, block_size>>>(d_in, d_out, n);
}

void active_prime(float *d_in, float *d_out, int n)
{
	int block_size = 1024;
	int grid_size = (n + block_size - 1) / block_size;
	active_prime_kernel<<<grid_size, block_size>>>(d_in, d_out, n);
}

void square_loss_prime(float *d_Ypred, float *d_Ytrue, float *d_err, int n_rows, int n_cols)
{
	int n = n_rows * n_cols;
	int block_size = 1024;
	int grid_size = (n + block_size - 1) / block_size;
	//err = Y_pred - Y_true
	subtract_kernel<<<grid_size, block_size>>>(d_Ypred, d_Ytrue, d_err, n);
}

void element_mul(float *d_a, float *d_b, float *d_c, int n_rows, int n_cols)
{
	int n = n_rows * n_cols;
	int block_size = 1024;
	int grid_size = (n + block_size - 1) / block_size;
	//c = a .* b
	mul_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
}
