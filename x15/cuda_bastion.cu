/*
 * Bastion cudaport by sp-hash@github
 * <Windows exe file for sale>
 */

#include "cuda_helper.h"
#include <stdint.h>
#include <memory.h>

__constant__ uint32_t c_PaddedMessage80[20]; // padded message (80 bytes + padding?)
static uint32_t *d_found[MAX_GPUS];

__host__
void bastion_setBlock_80(void *pdata)
{
	unsigned char PaddedMessage[4*8];
	memcpy(PaddedMessage, pdata, 4 * 8);
	cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, 4 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

__global__
void bastion_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint32_t target, uint32_t *d_found)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
	}
}

__host__ void bastion_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t target, uint32_t *h_found)
{
	const uint32_t threadsperblock = 1;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	cudaMemset(d_found[thr_id], 0xffffffff, 2 * sizeof(uint32_t));

	bastion_gpu_hash_80 << <grid, block >> >(threads, startNounce, target,d_found[thr_id] );
	cudaMemcpy(h_found, d_found[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}