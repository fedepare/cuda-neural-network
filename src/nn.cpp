#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <random>
#include <sys/time.h>
using namespace std;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "helper_cuda.h"

#include "utils.h"
#include "nn_kernels.h"
const int N = 30;

cublasHandle_t handle;
default_random_engine gen;
normal_distribution<float> gaussian(0, 1e-2);

float *d_W1, *d_b1, *d_W2, *d_b2;

void init_gaussian(float *arr, int n)
{
	for (int i = 0; i < n; i++)
		arr[i] = gaussian(gen);
}

void init_gaussian_device(float *d_arr, int n)
{
	float *arr = new float[n];
	init_gaussian(arr, n);
	checkCudaErrors(
			cudaMemcpy(d_arr, arr, sizeof(float) * n, cudaMemcpyHostToDevice));
	delete[] arr;
}

void init_rnd()
{
	checkCudaErrors(cudaMalloc(&d_W1, N * sizeof(float))); //W1(N*1)
	checkCudaErrors(cudaMalloc(&d_b1, N * sizeof(float))); //b1(N*1)
	checkCudaErrors(cudaMalloc(&d_W2, N * sizeof(float))); //W2(1*N)
	checkCudaErrors(cudaMalloc(&d_b2, sizeof(float)));	 	//b2(1*1)

	init_gaussian_device(d_W1, N);
	init_gaussian_device(d_b1, N);
	init_gaussian_device(d_W2, N);
	init_gaussian_device(d_b2, 1);
}
//read a column major matrix from a row major file
void read_mat_trans(const char *fname, float *A, int n_row, int n_col)
{
	FILE *fin = fopen(fname, "r");
	for (int i = 0; i < n_row; i++)
		for (int j = 0; j < n_col; j++)
			fscanf(fin, "%f", &A[j * n_row + i]);
	fclose(fin);
}

void init_from_file()
{
	checkCudaErrors(cudaMalloc(&d_W1, N * sizeof(float))); //W1(N*1)
	checkCudaErrors(cudaMalloc(&d_b1, N * sizeof(float))); //b1(N*1)
	checkCudaErrors(cudaMalloc(&d_W2, N * sizeof(float))); //W2(1*N)
	checkCudaErrors(cudaMalloc(&d_b2, sizeof(float)));	 	//b2(1*1)

	float *matT = new float[N];
	read_mat_trans("W1.txt", matT, N, 1);
	checkCudaErrors(
			cudaMemcpy(d_W1, matT, N * sizeof(float), cudaMemcpyHostToDevice));

	read_mat_trans("b1.txt", matT, N, 1);
	checkCudaErrors(
			cudaMemcpy(d_b1, matT, N * sizeof(float), cudaMemcpyHostToDevice));

	read_mat_trans("W2.txt", matT, 1, N);
	checkCudaErrors(
			cudaMemcpy(d_W2, matT, N * sizeof(float), cudaMemcpyHostToDevice));

	read_mat_trans("b2.txt", matT, 1, 1);
	checkCudaErrors(
			cudaMemcpy(d_b2, matT, 1 * sizeof(float), cudaMemcpyHostToDevice));

	delete[] matT;
}

void finish()
{
	checkCudaErrors(cudaFree(d_W1));
	checkCudaErrors(cudaFree(d_W2));
	checkCudaErrors(cudaFree(d_b1));
	checkCudaErrors(cudaFree(d_b2));
}

void linspace(float *arr, int n, float low, float high)
{
	float step = (high - low) / (n - 1);
	for (int i = 0; i < n; i++)
		arr[i] = low + step * i;
}

//X(1*n_batch) z2(N*n_batch) a2(N*n_batch) z3(1*n_batch) a3(1*n_batch) ones(1*n_batch)
void forward(float *d_X, int n_batch, float *d_z2, float *d_a2, float *d_z3,
		float *d_a3, float *d_ones)
{
	float alpha = 1, beta = 0;
	timeval t1, t2;
	//z2 = W1*X
	checkCublasErrors(
			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, n_batch, 1, &alpha, d_W1, N, d_X, 1, &beta, d_z2, N));
	//z2 += b1*ones
	checkCublasErrors(
			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, n_batch, 1, &alpha, d_b1, N, d_ones, 1, &alpha, d_z2, N));

	//a2 = f(z2)
	active(d_z2, d_a2, N * n_batch);

	//z3 = W2*a2
	checkCublasErrors(
			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, n_batch, N, &alpha, d_W2, 1, d_a2, N, &beta, d_z3, 1));
	//z3 += b2*ones
	checkCublasErrors(
			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, n_batch, 1, &alpha, d_b2, 1, d_ones, 1, &alpha, d_z3, 1));

	//a3 = f(z3)
	active(d_z3, d_a3, 1 * n_batch);
}

void test()
{
	const int n_batch = 1000;
	float *X = new float[n_batch], *Y = new float[n_batch];
	float *d_X, *d_Ytest;

	linspace(X, n_batch, 0, 2 * M_PI);
	checkCudaErrors(cudaMalloc(&d_X, n_batch * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_Ytest, n_batch * sizeof(float)));

	float *d_z2, *d_z3, *d_a2, *d_ones;

	checkCudaErrors(cudaMalloc(&d_z2, N * n_batch * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_a2, N * n_batch * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_z3, 1 * n_batch * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_ones, 1 * n_batch * sizeof(float)));

	checkCudaErrors(
			cudaMemcpy(d_X, X, n_batch * sizeof(float),
					cudaMemcpyHostToDevice));

	fill_ones(d_ones, n_batch);

	//Y_test = a3
	forward(d_X, n_batch, d_z2, d_a2, d_z3, d_Ytest, d_ones);

	checkCudaErrors(
			cudaMemcpy(Y, d_Ytest, n_batch * sizeof(float),
					cudaMemcpyDeviceToHost));

	FILE *fout = fopen("output.txt", "w");
	for (int i = 0; i < n_batch; i++)
	{
		printf("%f ", Y[i]);
		fprintf(fout, "%f\n", Y[i]);
	}
	printf("\n finish print Y\n");
	fclose(fout);
	checkCudaErrors(cudaFree(d_X));
	checkCudaErrors(cudaFree(d_Ytest));
	delete[] X;
	delete[] Y;
}

void read_data_device(float *d_X, float *d_Y, int n_batch)
{
	float *X, *Y;
	X = new float[n_batch];
	Y = new float[n_batch];

	FILE *fin = fopen("data.txt", "r");
	for (int i = 0; i < n_batch; i++)
		fscanf(fin, "%f%f", &X[i], &Y[i]);

	checkCudaErrors(
			cudaMemcpy(d_X, X, n_batch * sizeof(float),
					cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy(d_Y, Y, n_batch * sizeof(float),
					cudaMemcpyHostToDevice));

	fclose(fin);
	delete[] X;
	delete[] Y;
}

//X(1*n_batch) Y(1*n_batch) z2(N*n_batch) a2(N*n_batch) z3(1*n_batch) a3(1*n_batch) ones(1*n_batch)
//delta2(N*n_batch) delta3(1*n_batch) err(1*n_batch) grad_W2(N*1) grad_b2(N*1) grad_W3(1*N) grad_b3(1*1)
void gradient_descent(float *d_X, float *d_Ytrain, int n_batch, float *d_z2,
		float *d_a2, float *d_z3, float *d_a3, float *d_ones, float *d_delta2,
		float *d_delta3, float *d_err, float *d_grad_W1, float *d_grad_b1,
		float *d_grad_W2, float *d_grad_b2, float *d_ap_z2, float *d_ap_z3,
		float *d_prod_W2T_delta3)
{
	timeval t1, t2;
	gettimeofday(&t1, NULL);
	const int MAX_IT = 100000;
	float rate = 1e-4, alpha = 1, beta = 0, alpha_rate = -rate;
	for (int i = 1; i <= MAX_IT; i++)
	{
		forward(d_X, n_batch, d_z2, d_a2, d_z3, d_a3, d_ones);

		// err = a3 - y_train
		square_loss_prime(d_a3, d_Ytrain, d_err, 1, n_batch);
		// ap_z3 = f'(z3)
		active_prime(d_z3, d_ap_z3, 1 * n_batch);

		// delta3 = err .* ap_z3 = err .* f'(z3)
		element_mul(d_err, d_ap_z3, d_delta3, 1, n_batch);

		// ap_z2 = f'(z2)
		active_prime(d_z2, d_ap_z2, N * n_batch);
//		out_device_mat(d_ap_z2, N, n_batch);

		// prod_W2T_delta3 = W2.T * delta3
		checkCublasErrors(
				cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, n_batch, 1, &alpha, d_W2, 1, d_delta3, 1, &beta, d_prod_W2T_delta3, N));

		// delta2 = prod_W2T_delta3 .* ap_z2 = (W2.T * delta3) .* f'(z2)
		element_mul(d_prod_W2T_delta3, d_ap_z2, d_delta2, N, n_batch);

		// grad_W2 = delta3 * a2.T
		checkCublasErrors(
				cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 1, N, n_batch, &alpha, d_delta3, 1, d_a2, N, &beta, d_grad_W2, 1));

		// grad_b2 = sum(delta3, 'row') = delta3 * ones
		checkCublasErrors(
				cublasSgemv(handle, CUBLAS_OP_N, 1, n_batch, &alpha, d_delta3, 1, d_ones, 1, &beta, d_grad_b2, 1));

		// grad_W1 = delta2 * X.T
		checkCublasErrors(
				cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, 1, n_batch, &alpha, d_delta2, N, d_X, 1, &beta, d_grad_W1, N));

		// grad_b1 = sum(delta2, 'row') = delta2 * ones
		checkCublasErrors(
				cublasSgemv(handle, CUBLAS_OP_N, N, n_batch, &alpha, d_delta2, N, d_ones, 1, &beta, d_grad_b1, 1));

		// W1 = W1 - rate * grad_W1
		checkCublasErrors(
				cublasSaxpy(handle, N, &alpha_rate, d_grad_W1, 1, d_W1, 1));

		// b1 = b1 - rate * grad_b1
		checkCublasErrors(
				cublasSaxpy(handle, N, &alpha_rate, d_grad_b1, 1, d_b1, 1));

		//W2 = W2 - rate * grad_W2
		checkCublasErrors(
				cublasSaxpy(handle, N, &alpha_rate, d_grad_W2, 1, d_W2, 1));

		//b2 = b2 - rate * grad_b2
		checkCublasErrors(
				cublasSaxpy(handle, 1, &alpha_rate, d_grad_b2, 1, d_b2, 1));

		if (i % 1000 == 999)
		{
			gettimeofday(&t2, NULL);
			float mean_err = 1e10;
			checkCublasErrors(cublasSnrm2(handle, n_batch, d_err, 1, &mean_err));
			mean_err /= n_batch;
			printf("epoch %d error: %f time elpased %ld\n", i + 1, mean_err, (t2.tv_sec- t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec);
			gettimeofday(&t1, NULL);
		}
//		printf("grad_b1");
//		out_device_mat(d_grad_b1, 1, 1);
//		printf("b1");
//		out_device_mat(d_b1, N, 1);
//		printf("delta2");
//		out_device_mat(d_delta2, N, n_batch);
//		printf("b2");
//		out_device_mat(d_b2, 1, 1);
	}

}

void train()
{
	const int n_batch = 1000;
	//X(1*n_batch) Y(1*n_batch) z2(N*n_batch) a2(N*n_batch) z3(1*n_batch) a3(1*n_batch) ones(1*n_batch)
	float *d_z2, *d_z3, *d_a2, *d_ones, *d_a3, *d_X, *d_Ytrain;
	//delta2(N*n_batch) delta3(1*n_batch) err(1*n_batch) grad_W1(N*1) grad_b1(N*1) grad_W2(1*N) grad_b2(1*1)
	float *d_delta2, *d_delta3, *d_err, *d_grad_W1, *d_grad_b1, *d_grad_W2,
			*d_grad_b2;

	//activation prime of z2(N*n_batch) and z3(1*n_batch)
	float *d_ap_z2, *d_ap_z3;

	//temporary variable to store W2.T*delta3, size z2(N*n_batch)
	float *d_prod_W2T_delta3;

	checkCudaErrors(cudaMalloc(&d_X, 1 * n_batch * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_Ytrain, 1 * n_batch * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_z2, N * n_batch * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_a2, N * n_batch * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_z3, 1 * n_batch * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_a3, 1 * n_batch * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_ones, 1 * n_batch * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_delta2, N * n_batch * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_delta3, 1 * n_batch * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_err, 1 * n_batch * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_grad_W1, N * 1 * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_grad_b1, N * 1 * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_grad_W2, 1 * N * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_grad_b2, 1 * 1 * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_ap_z2, N * n_batch * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_ap_z3, 1 * n_batch * sizeof(float)));
	checkCudaErrors(
			cudaMalloc(&d_prod_W2T_delta3, N * n_batch * sizeof(float)));

	fill_ones(d_ones, n_batch);
	read_data_device(d_X, d_Ytrain, n_batch);

	gradient_descent(d_X, d_Ytrain, n_batch, d_z2, d_a2, d_z3, d_a3, d_ones,
			d_delta2, d_delta3, d_err, d_grad_W1, d_grad_b1, d_grad_W2,
			d_grad_b2, d_ap_z2, d_ap_z3, d_prod_W2T_delta3);

	checkCudaErrors(cudaFree(d_X));
	checkCudaErrors(cudaFree(d_Ytrain));
	checkCudaErrors(cudaFree(d_z2));
	checkCudaErrors(cudaFree(d_a2));
	checkCudaErrors(cudaFree(d_z3));
	checkCudaErrors(cudaFree(d_a3));
	checkCudaErrors(cudaFree(d_ones));
	checkCudaErrors(cudaFree(d_delta2));
	checkCudaErrors(cudaFree(d_delta3));
	checkCudaErrors(cudaFree(d_err));
	checkCudaErrors(cudaFree(d_grad_W1));
	checkCudaErrors(cudaFree(d_grad_b1));
	checkCudaErrors(cudaFree(d_grad_W2));
	checkCudaErrors(cudaFree(d_grad_b2));
	checkCudaErrors(cudaFree(d_ap_z2));
	checkCudaErrors(cudaFree(d_ap_z3));
	checkCudaErrors(cudaFree(d_prod_W2T_delta3));
}

int main()
{
	checkCublasErrors(cublasCreate(&handle));
	init_rnd();
//	init_from_file();
//	out_device_mat(d_W1, N, 1);
//	out_device_mat(d_W2, 1, N);
//	out_device_mat(d_b1, N, 1);
//	out_device_mat(d_b2, 1, 1);
	train();
	test();
	finish();
	return 0;
}
