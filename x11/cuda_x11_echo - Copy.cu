#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

#include "cuda_x11_aes.cu"

static uint2 *d_nonce[MAX_GPUS];
static uint32_t *d_found[MAX_GPUS];

__device__ __forceinline__ void AES_2ROUND(
	const uint32_t*const __restrict__ sharedMemory,
	uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3,
	const uint32_t k0)
{
	uint32_t y0, y1, y2, y3;
	aes_round(sharedMemory,
		x0, x1, x2, x3,
		k0,
		y0, y1, y2, y3);

	aes_round(sharedMemory,
		y0, y1, y2, y3,
		x0, x1, x2, x3);
}

__device__ __forceinline__ void cuda_echo_round(
	const uint32_t *const __restrict__ sharedMemory, uint32_t *const __restrict__  hash)
{
	uint32_t h[16];
	const uint32_t P[48] = {
		0xe7e9f5f5,
		0xf5e7e9f5,
		0xb3b36b23,
		0xb3dbe7af,

		0xa4213d7e,
		0xf5e7e9f5,
		0xb3b36b23,
		0xb3dbe7af,
		//8-12
		0x01425eb8,
		0xf5e7e9f5,
		0xb3b36b23,
		0xb3dbe7af,

		0x65978b09,
		0xf5e7e9f5,
		0xb3b36b23,
		0xb3dbe7af,

		//21-25
		0x2cb6b661,
		0x6b23b3b3,
		0xcf93a7cf,
		0x9d9d3751,

		0x9ac2dea3,
		0xf5e7e9f5,
		0xb3b36b23,
		0xb3dbe7af,

		//34-38
		0x579f9f33,
		0xfbfbfbfb,
		0xfbfbfbfb,
		0xefefd3c7,

		0xdbfde1dd,
		0xf5e7e9f5,
		0xb3b36b23,
		0xb3dbe7af,

		0x34514d9e,
		0xf5e7e9f5,
		0xb3b36b23,
		0xb3dbe7af,


		0xb134347e,
		0xea6f7e7e,
		0xbd7731bd,
		0x8a8a1968,

		0x14b8a457,
		0xf5e7e9f5,
		0xb3b36b23,
		0xb3dbe7af,

		0x265f4382,
		0xf5e7e9f5,
		0xb3b36b23,
		0xb3dbe7af
		//58-61
	};
	uint32_t k0;

#pragma unroll
	for (int i = 0; i < 16; i++)
	{
		h[i] = hash[i];
	}

	k0 = 512 + 8;

#pragma unroll
	for (int idx = 0; idx < 16; idx+= 4)
	{
		AES_2ROUND(sharedMemory,
			h[idx + 0], h[idx + 1], h[idx + 2], h[idx + 3], k0++);
	}
	k0 += 4;

	uint32_t W[64];

#pragma unroll
	for (int i = 0; i < 4; i++) 
	{
		uint32_t a = P[i];
		uint32_t b = P[i + 4];
		uint32_t c = h[i + 8];
		uint32_t d = P[i + 8];
		
		uint32_t ab = a ^ b;
		uint32_t bc = b ^ c;
		uint32_t cd = c ^ d;


		uint32_t t = (ab & 0x80808080);
		uint32_t t2 = (bc & 0x80808080);
		uint32_t t3 = (cd & 0x80808080);

		uint32_t abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		uint32_t bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		uint32_t cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);

		W[0 + i] = abx ^ bc ^ d;
		W[0 + i + 4] = bcx ^ a ^ cd;
		W[0 + i + 8] = cdx ^ ab ^ d;
		W[0 + i + 12] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = P[12 + i];
		b = h[i + 4]; 
		c = P[12 + i + 4];
		d = P[12 + i + 8];

		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;


		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);

		abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);

		W[16 + i] = abx ^ bc ^ d;
		W[16 + i + 4] = bcx ^ a ^ cd;
		W[16 + i + 8] = cdx ^ ab ^ d;
		W[16 + i + 12] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = h[i];
		b = P[24 + i + 0];
		c = P[24 + i + 4];
		d = P[24 + i + 8];

		 ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;


		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);

		abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
	    cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);

		W[32 + i] = abx ^ bc ^ d;
		W[32 + i + 4] = bcx ^ a ^ cd;
		W[32 + i + 8] = cdx ^ ab ^ d;
		W[32 + i + 12] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = P[36 + i ];
		b = P[36 + i +4 ];
		c = P[36 + i + 8];
		d = h[i + 12];

		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;

		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);

		abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);

		W[48 + i] = abx ^ bc ^ d;
		W[48 + i + 4] = bcx ^ a ^ cd;
		W[48 + i + 8] = cdx ^ ab ^ d;
		W[48 + i + 12] = abx ^ bcx ^ cdx ^ ab ^ c;

	}

	for (int k = 1; k < 10; k++)
	{

		// Big Sub Words
		#pragma unroll
		for (int idx = 0; idx < 64; idx+=16)
		{
			AES_2ROUND(sharedMemory,
				W[idx + 0], W[idx + 1], W[idx + 2], W[idx + 3],
				k0++);
			AES_2ROUND(sharedMemory,
				W[idx + 4], W[idx + 5], W[idx + 6], W[idx + 7],
				k0++);
			AES_2ROUND(sharedMemory,
				W[idx + 8], W[idx + 9], W[idx + 10], W[idx + 11],
				k0++);
			AES_2ROUND(sharedMemory,
				W[idx + 12], W[idx + 13], W[idx + 14], W[idx + 15],
				k0++);

		}

		// Shift Rows
#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			uint32_t t;

			/// 1, 5, 9, 13
			t = W[4 + i];
			W[4 + i] = W[20 + i];
			W[20 + i] = W[36 + i];
			W[36 + i] = W[52 + i];
			W[52 + i] = t;

			// 2, 6, 10, 14
			t = W[8 + i];
			W[8 + i] = W[40 + i];
			W[40 + i] = t;
			t = W[24 + i];
			W[24 + i] = W[56 + i];
			W[56 + i] = t;

			// 15, 11, 7, 3
			t = W[60 + i];
			W[60 + i] = W[44 + i];
			W[44 + i] = W[28 + i];
			W[28 + i] = W[12 + i];
			W[12 + i] = t;
		}

		// Mix Columns
#pragma unroll
		for (int i = 0; i < 4; i++) // Schleife über je 2*uint32_t
		{
#pragma unroll
			for (int idx = 0; idx < 64; idx += 16) // Schleife über die elemnte
			{

				uint32_t a = W[idx + i];
				uint32_t b = W[idx + i + 4];
				uint32_t c = W[idx + i + 8];
				uint32_t d = W[idx + i + 12];

				uint32_t ab = a ^ b;
				uint32_t bc = b ^ c;
				uint32_t cd = c ^ d;

				uint32_t t, t2, t3;
				t = (ab & 0x80808080);
				t2 = (bc & 0x80808080);
				t3 = (cd & 0x80808080);

				uint32_t abx = ((t >> 7) * 27 ^ ((ab^t) << 1));
				uint32_t bcx = ((t2 >> 7) * 27 ^ ((bc^t2) << 1));
				uint32_t cdx = ((t3 >> 7) * 27 ^ ((cd^t3) << 1));

				W[idx + i] = abx ^ bc ^ d;
				W[idx + i + 4] = bcx ^ a ^ cd;
				W[idx + i + 8] = cdx ^ ab ^ d;
				W[idx + i + 12] = abx ^ bcx ^ cdx ^ ab ^ c;
			}
		}
	}

#pragma unroll
	for (int i = 0; i<16; i += 4)
	{
		W[i] ^= W[32 + i] ^ 512;
		W[i + 1] ^= W[32 + i + 1];
		W[i + 2] ^= W[32 + i + 2];
		W[i + 3] ^= W[32 + i + 3];
	}

#pragma unroll
	for (int i = 0; i<16; i++)
		hash[i] ^= W[i];
}

/*
__device__ __forceinline__
void echo_gpu_init_128(uint32_t *const __restrict__ sharedMemory)
{
	if (threadIdx.x < 128) 
	{
		sharedMemory[threadIdx.x] = d_AES0[threadIdx.x];
		sharedMemory[threadIdx.x + 256] = d_AES1[threadIdx.x];
		sharedMemory[threadIdx.x + 512] = d_AES2[threadIdx.x];
		sharedMemory[threadIdx.x + 768] = d_AES3[threadIdx.x];

		sharedMemory[threadIdx.x + 64 * 2] = d_AES0[threadIdx.x + 64 * 2];
		sharedMemory[threadIdx.x + 64 * 2 + 256] = d_AES1[threadIdx.x + 64 * 2];
		sharedMemory[threadIdx.x + 64 * 2 + 512] = d_AES2[threadIdx.x + 64 * 2];
		sharedMemory[threadIdx.x + 64 * 2 + 768] = d_AES3[threadIdx.x + 64 * 2];
	}
}
*/


__device__ __forceinline__
void echo_gpu_init(uint32_t *const __restrict__ sharedMemory)
{
	/* each thread startup will fill a uint32 */
	if (threadIdx.x < 256) 
	{
		sharedMemory[threadIdx.x] = d_AES0[threadIdx.x];
		sharedMemory[threadIdx.x + 256] = ROL8(sharedMemory[threadIdx.x]);
		sharedMemory[threadIdx.x + 512] = ROL16(sharedMemory[threadIdx.x]);
		sharedMemory[threadIdx.x + 768] = ROL24(sharedMemory[threadIdx.x]);
	}
}

__global__ __launch_bounds__(256, 4)
void x11_echo512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint64_t *const __restrict__ g_hash)
{
	__shared__ __align__(128) uint32_t sharedMemory[1024];

	echo_gpu_init(sharedMemory);

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = (startNounce + thread);
        int hashPosition = nounce - startNounce;
        uint32_t *Hash = (uint32_t*)&g_hash[hashPosition<<3];
		cuda_echo_round(sharedMemory, Hash);
    }
}

// Setup-Funktionen
__host__ void x11_echo512_cpu_init(int thr_id, uint32_t threads)
{
	cudaMalloc(&d_nonce[thr_id], sizeof(uint2));
	CUDA_SAFE_CALL(cudaMalloc(&(d_found[thr_id]), 2 * sizeof(uint32_t)));
}

__host__ void x11_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash)
{
	uint32_t threadsperblock = 256;
    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    x11_echo512_gpu_hash_64<<<grid, block>>>(threads, startNounce, (uint64_t*)d_hash);
}

__host__ void x11_echo512_cpu_free(int32_t thr_id)
{
	cudaFreeHost(&d_nonce[thr_id]);
}


__constant__ uint32_t P[48] = {
	0xe7e9f5f5,
	0xf5e7e9f5,
	0xb3b36b23,
	0xb3dbe7af,

	0xa4213d7e,
	0xf5e7e9f5,
	0xb3b36b23,
	0xb3dbe7af,
	//8-12
	0x01425eb8,
	0xf5e7e9f5,
	0xb3b36b23,
	0xb3dbe7af,

	0x65978b09,
	0xf5e7e9f5,
	0xb3b36b23,
	0xb3dbe7af,

	//21-25
	0x2cb6b661,
	0x6b23b3b3,
	0xcf93a7cf,
	0x9d9d3751,

	0x9ac2dea3,
	0xf5e7e9f5,
	0xb3b36b23,
	0xb3dbe7af,

	//34-38
	0x579f9f33,
	0xfbfbfbfb,
	0xfbfbfbfb,
	0xefefd3c7,

	0xdbfde1dd,
	0xf5e7e9f5,
	0xb3b36b23,
	0xb3dbe7af,

	0x34514d9e,
	0xf5e7e9f5,
	0xb3b36b23,
	0xb3dbe7af,


	0xb134347e,
	0xea6f7e7e,
	0xbd7731bd,
	0x8a8a1968,

	0x14b8a457,
	0xf5e7e9f5,
	0xb3b36b23,
	0xb3dbe7af,

	0x265f4382,
	0xf5e7e9f5,
	0xb3b36b23,
	0xb3dbe7af
	//58-61
};

__global__
void x11_echo512_gpu_hash_64_final(uint32_t threads, uint32_t startNounce, const uint64_t *const __restrict__ g_hash, uint32_t *const __restrict__ d_found, uint32_t target)
{

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{

		__shared__ __align__(128) uint32_t sharedMemory[1024];
		echo_gpu_init(sharedMemory);

		uint32_t nounce = (startNounce + thread);

		int hashPosition = nounce - startNounce;
		uint32_t *Hash = (uint32_t*)&g_hash[hashPosition *8];

		uint32_t h[16];

#pragma unroll 16
		for (int i = 0; i < 16; i++)
		{
			h[i] = Hash[i];
		}
		uint32_t backup = h[7];

		AES_2ROUND(sharedMemory,
				h[0 + 0], h[0 + 1], h[0 + 2], h[0 + 3], 512 + 8);
		AES_2ROUND(sharedMemory,
				h[4 + 0], h[4 + 1], h[4 + 2], h[4 + 3], 512 + 9);
		AES_2ROUND(sharedMemory,
				h[8 + 0], h[8 + 1], h[8 + 2], h[8 + 3], 512 + 10);
		AES_2ROUND(sharedMemory,
				h[12 + 0], h[12 + 1], h[12 + 2], h[12 + 3], 512 + 11);

		uint32_t W[64];

		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			const uint32_t a = P[i];
			const uint32_t a2 = P[12 + i];
			const uint32_t a3 = h[i];
			const uint32_t a4 = P[36 + i];
			const uint32_t b = P[i + 4];
			const uint32_t b2 = h[i + 4];
			const uint32_t b3 = P[24 + i + 0];
			const uint32_t b4 = P[36 + i + 4];
			const uint32_t c = h[i + 8];
			const uint32_t c2 = P[12 + i + 4];
			const uint32_t c3 = P[24 + i + 4];
			const uint32_t c4 = P[36 + i + 8];
			const uint32_t d = P[i + 8];
			const uint32_t d2 = P[12 + i + 8];
			const uint32_t d3 = P[24 + i + 8];
			const uint32_t d4 = h[i + 12];

			const uint32_t ab = a ^ b;
			const uint32_t ab2 = a2 ^ b2;
			const uint32_t ab3 = a3 ^ b3;
			const uint32_t ab4 = a4 ^ b4;
			const uint32_t bc = b ^ c;
			const uint32_t bc2 = b2 ^ c2;
			const uint32_t bc3 = b3 ^ c3;
			const uint32_t bc4 = b4 ^ c4;
			const uint32_t cd = c ^ d;
			const uint32_t cd2 = c2 ^ d2;
			const uint32_t cd3 = c3 ^ d3;
			const uint32_t cd4 = c4 ^ d4;

			const uint32_t t = (ab & 0x80808080);
			const uint32_t ta2 = (ab2 & 0x80808080);
			const uint32_t ta3 = (ab3 & 0x80808080);
			const uint32_t t4 = (ab4 & 0x80808080);
			const uint32_t t2 = (bc & 0x80808080);
			const uint32_t t22 = (bc2 & 0x80808080);
			const uint32_t t23 = (bc3 & 0x80808080);
			const uint32_t t24 = (bc4 & 0x80808080);
			const uint32_t t3 = (cd & 0x80808080);
			const uint32_t t32 = (cd2 & 0x80808080);
			const uint32_t t33 = (cd3 & 0x80808080);
			const uint32_t t34 = (cd4 & 0x80808080);

			const uint32_t abx = (t >> 7) * 27 ^ ((ab^t) << 1);
			const uint32_t abx2 = (ta2 >> 7) * 27 ^ ((ab2^ta2) << 1);
			const uint32_t abx3 = (ta3 >> 7) * 27 ^ ((ab3^ta3) << 1);
			const uint32_t abx4 = (t4 >> 7) * 27 ^ ((ab4^t4) << 1);
			const uint32_t bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
			const uint32_t bcx2 = (t22 >> 7) * 27 ^ ((bc2^t22) << 1);
			const uint32_t bcx3 = (t23 >> 7) * 27 ^ ((bc3^t23) << 1);
			const uint32_t bcx4 = (t24 >> 7) * 27 ^ ((bc4^t24) << 1);
			const uint32_t cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);
			const uint32_t cdx2 = (t32 >> 7) * 27 ^ ((cd2^t32) << 1);
			const uint32_t cdx3 = (t33 >> 7) * 27 ^ ((cd3^t33) << 1);
			const uint32_t cdx4 = (t34 >> 7) * 27 ^ ((cd4^t34) << 1);

			W[0 + i] = abx ^ bc ^ d;
			W[0 + i + 4] = bcx ^ a ^ cd;
			W[0 + i + 8] = cdx ^ ab ^ d;
			W[0 + i + 12] = abx ^ bcx ^ cdx ^ ab ^ c;
			W[16 + i] = abx2 ^ bc2 ^ d2;
			W[16 + i + 4] = bcx2 ^ a2 ^ cd2;
			W[16 + i + 8] = cdx2 ^ ab2 ^ d2;
			W[16 + i + 12] = abx2 ^ bcx2 ^ cdx2 ^ ab2 ^ c2;
			W[32 + i] = abx3 ^ bc3 ^ d3;
			W[32 + i + 4] = bcx3 ^ a3 ^ cd3;
			W[32 + i + 8] = cdx3 ^ ab3 ^ d3;
			W[32 + i + 12] = abx3 ^ bcx3 ^ cdx3 ^ ab3 ^ c3;
			W[48 + i] = abx4 ^ bc4 ^ d4;
			W[48 + i + 4] = bcx4 ^ a4 ^ cd4;
			W[48 + i + 8] = cdx4 ^ ab4 ^ d4;
			W[48 + i + 12] = abx4 ^ bcx4 ^ cdx4 ^ ab4 ^ c4;}

		uint32_t k0 = 512 + 16;

		for (int k = 1; k < 8; k++)
		{

			// Big Sub Words

		#pragma unroll 4
			for (int idx = 0; idx < 64; idx += 16)
			{
				AES_2ROUND(sharedMemory,
					W[idx + 0], W[idx + 1], W[idx + 2], W[idx + 3],
					k0++);
				AES_2ROUND(sharedMemory,
					W[idx + 4], W[idx + 5], W[idx + 6], W[idx + 7],
					k0++);
				AES_2ROUND(sharedMemory,
					W[idx + 8], W[idx + 9], W[idx + 10], W[idx + 11],
					k0++);
				AES_2ROUND(sharedMemory,
					W[idx + 12], W[idx + 13], W[idx + 14], W[idx + 15],
					k0++);

			}
			// Shift Rows
#pragma unroll 4
			for (int i = 0; i < 4; i++)
			{

				/// 1, 5, 9, 13
				uint32_t t = W[4 + i];
				W[4 + i] = W[20 + i];
				W[20 + i] = W[36 + i];
				W[36 + i] = W[52 + i];
				W[52 + i] = t;

				// 2, 6, 10, 14
				t = W[8 + i];
				W[8 + i] = W[40 + i];
				W[40 + i] = t;
				t = W[24 + i];
				W[24 + i] = W[56 + i];
				W[56 + i] = t;

				// 15, 11, 7, 3
				t = W[60 + i];
				W[60 + i] = W[44 + i];
				W[44 + i] = W[28 + i];
				W[28 + i] = W[12 + i];
				W[12 + i] = t;
			}

			// Mix Columns
#pragma unroll
			for (int i = 0; i < 4; i++) // Schleife über je 2*uint32_t
			{
#pragma unroll
				for (int idx = 0; idx < 64; idx += 16) // Schleife über die elemnte
				{

					const uint32_t a = W[idx + i];
					const uint32_t b = W[idx + i + 4];
					const uint32_t c = W[idx + i + 8];
					const uint32_t d = W[idx + i + 12];

					const uint32_t ab = a ^ b;
					const uint32_t bc = b ^ c;
					const uint32_t cd = c ^ d;

					const uint32_t t = (ab & 0x80808080);
					const uint32_t t2 = (bc & 0x80808080);
					const uint32_t t3 = (cd & 0x80808080);

					const uint32_t abx = (t >> 7) * 27 ^ ((ab^t) << 1);
					const uint32_t bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
					const uint32_t cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);

					W[idx + i] = abx ^ bc ^ d;
					W[idx + i + 4] = bcx ^ a ^ cd;
					W[idx + i + 8] = cdx ^ ab ^ d;
					W[idx + i + 12] = abx ^ bcx ^ cdx ^ ab ^ c;

				}
			}
		}

		// Big Sub Words

#pragma unroll 4
		for (int idx = 0; idx < 64; idx += 16)
		{
			AES_2ROUND(sharedMemory,
				W[idx + 0], W[idx + 1], W[idx + 2], W[idx + 3],
				k0++);
			AES_2ROUND(sharedMemory,
				W[idx + 4], W[idx + 5], W[idx + 6], W[idx + 7],
				k0++);
			AES_2ROUND(sharedMemory,
				W[idx + 8], W[idx + 9], W[idx + 10], W[idx + 11],
				k0++);
			AES_2ROUND(sharedMemory,
				W[idx + 12], W[idx + 13], W[idx + 14], W[idx + 15],
				k0++);

		}
/*
		// Shift Rows
#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{

			/// 1, 5, 9, 13
			uint32_t t = W[4 + i];
			W[4 + i] = W[20 + i];
			W[20 + i] = W[36 + i];
			W[36 + i] = W[52 + i];
			W[52 + i] = t;

			// 2, 6, 10, 14
			t = W[8 + i];
			W[8 + i] = W[40 + i];
			W[40 + i] = t;
			t = W[24 + i];
			W[24 + i] = W[56 + i];
			W[56 + i] = t;

			// 15, 11, 7, 3
			t = W[60 + i];
			W[60 + i] = W[44 + i];
			W[44 + i] = W[28 + i];
			W[28 + i] = W[12 + i];
			W[12 + i] = t;
		}
*/
		uint32_t a, b, c, d, ab, bc, cd, t, t2, t3, abx, bcx, cdx;

//		t = W[4];
		//W[4] = W[20];
		//W[20] = W[36];
		//W[36] = W[52];
//		W[52] = t;

//		t = W[8];
//		W[8] = W[40];
//		W[40] = t;
//		t = W[24];
//		W[24] = W[56];
//		W[56] = t;
//		t = W[60];
		//W[60] = W[44];
//		W[44] = W[28];
//		W[28] = W[12];
//		W[12] = t;


//		t = W[5];
	//	W[5] = W[21];
//		W[21] = W[37];
//		W[37] = W[53];
//		W[53] = t;

//		t = W[9];
//		W[9] = W[41];
//		W[41] = t;
//		t = W[25];
//		W[25] = W[57];
//		W[57] = t;

		t = W[61];
		W[61] = W[45];
		W[45] = W[29];
		W[29] = W[13];
		W[13] = t;


		t = W[6];
		W[6] = W[22];
		W[22] = W[38];
		W[38] = W[54];
		W[54] = t;
		t = W[10];
		W[10] = W[42];
		W[42] = t;
		t = W[26];
		W[26] = W[58];
		W[58] = t;
		t = W[62];
		W[62] = W[46];
		W[46] = W[30];
		W[30] = W[14];
		W[14] = t;


		t = W[7];
		W[7] = W[23];
		W[23] = W[39];
		W[39] = W[55];
		W[55] = t;
		t = W[11];
		W[11] = W[43];
		W[43] = t;
		t = W[27];
		W[27] = W[59];
		W[59] = t;
		t = W[63];
		W[63] = W[47];
		W[47] = W[31];
		W[31] = W[15];
		W[15] = t;


		a = W[0];
		b = W[20];
		c = W[40];
		d = W[60];
		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;
		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);
		abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		//	bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);
		W[0] = abx ^ bc ^ d;
		//	W[7] = bcx ^ a ^ cd;
		W[40] = cdx ^ ab ^ d;
		//	W[15] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = W[16];
		b = W[36];
		c = W[56];
		d = W[12];
		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;
		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);
		abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);
		//	W[19] = abx ^ bc ^ d;
		W[20] = bcx ^ a ^ cd;
		//	W[27] = cdx ^ ab ^ d;
		W[12] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = W[32];
		b = W[52];
		c = W[8];
		d = W[28];
		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;
		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);
		abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		//bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);
		W[32] = abx ^ bc ^ d;
		//	W[39] = bcx ^ a ^ cd;
		W[8] = cdx ^ ab ^ d;
		//	W[47] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = W[48];
		b = W[4];
		c = W[24];
		d = W[44];
		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;
		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);
		abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);
		//	W[51] = abx ^ bc ^ d;
		W[52] = bcx ^ a ^ cd;
		//	W[59] = cdx ^ ab ^ d;
		W[44] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = W[1];
		b = W[21];
		c = W[41];
		d = W[13];
		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;
		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);
		abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		//	bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);
		W[1] = abx ^ bc ^ d;
		//	W[7] = bcx ^ a ^ cd;
		W[41] = cdx ^ ab ^ d;
		//	W[15] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = W[17];
		b = W[37];
		c = W[57];
		d = W[29];
		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;
		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);
		abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);
		//	W[19] = abx ^ bc ^ d;
		W[37] = bcx ^ a ^ cd;
		//	W[27] = cdx ^ ab ^ d;
		W[29] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = W[33];
		b = W[53];
		c = W[9];
		d = W[45];
		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;
		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);
		abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		//bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);
		W[33] = abx ^ bc ^ d;
		//	W[39] = bcx ^ a ^ cd;
		W[9] = cdx ^ ab ^ d;
		//	W[47] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = W[49];
		b = W[5];
		c = W[25];
		d = W[61];
		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;
		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);
		abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);
		//	W[51] = abx ^ bc ^ d;
		W[5] = bcx ^ a ^ cd;
		//	W[59] = cdx ^ ab ^ d;
		W[61] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = W[2];
		b = W[6];
		c = W[10];
		d = W[14];
		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;
		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);
		abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		//	bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);
		W[2] = abx ^ bc ^ d;
		//	W[7] = bcx ^ a ^ cd;
		W[10] = cdx ^ ab ^ d;
		//	W[15] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = W[18];
		b = W[22];
		c = W[26];
		d = W[30];
		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;
		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);
		abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);
		//	W[19] = abx ^ bc ^ d;
		W[22] = bcx ^ a ^ cd;
		//	W[27] = cdx ^ ab ^ d;
		W[30] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = W[34];
		b = W[38];
		c = W[42];
		d = W[46];
		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;
		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);
		abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		//bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);
		W[35-1] = abx ^ bc ^ d;
		//	W[39] = bcx ^ a ^ cd;
		W[43-1] = cdx ^ ab ^ d;
		//	W[47] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = W[50];
		b = W[54];
		c = W[58];
		d = W[62];
		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;
		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);
		abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);
		//	W[51] = abx ^ bc ^ d;
		W[54] = bcx ^ a ^ cd;
		//	W[59] = cdx ^ ab ^ d;
		W[62] = abx ^ bcx ^ cdx ^ ab ^ c;


		a = W[3];
		b = W[7];
		c = W[11];
		d = W[15];
		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;
		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);
		abx = (t >> 7) * 27 ^ ((ab^t) << 1);
	//	bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);
		W[3] = abx ^ bc ^ d;
	//	W[7] = bcx ^ a ^ cd;
		W[11] = cdx ^ ab ^ d;
	//	W[15] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = W[19];
		b = W[23];
		c = W[27];
		d = W[31];
		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;
		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);
		abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);
	//	W[19] = abx ^ bc ^ d;
		W[23] = bcx ^ a ^ cd;
	//	W[27] = cdx ^ ab ^ d;
		W[31] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = W[35];
		b = W[39];
		c = W[43];
		d = W[47];
		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;
		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);
		abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		//bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);
		W[35] = abx ^ bc ^ d;
	//	W[39] = bcx ^ a ^ cd;
		W[43] = cdx ^ ab ^ d;
	//	W[47] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = W[51];
		b = W[55];
		c = W[59];
		d = W[63];
		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;
		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);
		abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);
	//	W[51] = abx ^ bc ^ d;
		W[55] = bcx ^ a ^ cd;
	//	W[59] = cdx ^ ab ^ d;
		W[63] = abx ^ bcx ^ cdx ^ ab ^ c;

		//3, 11, 23, 31, 35, 43, 55, 63

		AES_2ROUND(sharedMemory,
			W[0], W[1], W[2], W[3],
			512+(9*16));
		AES_2ROUND(sharedMemory,
			W[40], W[41], W[10], W[11],
			512 + (9 * 16)+2);
		AES_2ROUND(sharedMemory,
			W[20], W[37], W[22], W[23],
			512 + (9 * 16)+5);
		AES_2ROUND(sharedMemory,
			W[12], W[29], W[30], W[31],
			512 + (9 * 16) + 7);
		AES_2ROUND(sharedMemory,
			W[32], W[33], W[34], W[35],
			512 + (9 * 16) + 8);
		AES_2ROUND(sharedMemory,
			W[8], W[9], W[42], W[43],
			512 + (9 * 16) + 10);
		AES_2ROUND(sharedMemory,
			W[52], W[5], W[54], W[55],
			512 + (9 * 16) + 13);
		AES_2ROUND(sharedMemory,
			W[44], W[61], W[62], W[63],
			512 + (9 * 16) + 15);

		bc = W[23] ^ W[43];
		t2 = (bc & 0x80808080);
		uint32_t test = (t2 >> 7) * 27 ^ ((bc^t2) << 1) ^ W[3] ^ W[43] ^ W[63];
		bc = W[55] ^ W[11];
		t2 = (bc & 0x80808080);
		test ^= (t2 >> 7) * 27 ^ ((bc^t2) << 1) ^ W[35] ^ W[11] ^ W[31] ^ backup;
		if (test <= target)
		{
			uint32_t tmp = atomicExch(&(d_found[0]), nounce);
			if (tmp != 0xffffffff)
				d_found[1] = tmp;
		}
	}
}

const uint32_t threadsperblock = 256;
__host__ void x11_echo512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, uint32_t target, uint32_t *h_found)
{


	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	cudaMemset(d_found[thr_id], 0xffffffff, 2*sizeof(uint32_t));

	x11_echo512_gpu_hash_64_final << <grid, block>> >(threads, startNounce, (uint64_t*)d_hash, d_found[thr_id], target);
	cudaMemcpy(h_found, d_found[thr_id], 2*sizeof(uint32_t), cudaMemcpyDeviceToHost);
}
