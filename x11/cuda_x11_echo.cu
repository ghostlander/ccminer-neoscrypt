#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"
#include "cuda_vector.h"

#include "cuda_x11_aes.cu"

static uint2 *d_nonce[MAX_GPUS];
static uint32_t *d_found[MAX_GPUS];

__device__ __forceinline__ uint32_t mul27(const uint32_t x)
{
//	uint32_t result = (x << 5) - (x + x + x + x + x);
	uint32_t result = (x *27);

	//	uint32_t result;
	//	asm("mul24.lo.u32 %0,%1,%2; \n\t" : "=r"(result): "r"(x) , "r"(y));
	return result;
}

__device__ __forceinline__ void AES_2ROUND(
	const uint32_t*const __restrict__ sharedMemory,
	uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3,
	const uint32_t k0)
{
	uint32_t f[4]=
	{
		x0,x1,x2,x3
	};

	aes_round(sharedMemory,k0,(uint8_t *) &f[0]);
	aes_round(sharedMemory, (uint8_t *)&f[0]);

	x0 = f[0];
	x1 = f[1];
	x2 = f[2];
	x3 = f[3];

//	aes_round(sharedMemory,f[0], f[1], f[2], f[3],x0, x1, x2, x3);
}

__device__ __forceinline__ void cuda_echo_round(
	const uint32_t *const __restrict__ sharedMemory, uint32_t *const __restrict__  hash)
{
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
	uint32_t a, b, c, d, ab, bc, cd, t, t2, t3,abx,bcx,cdx;

	uint32_t h[16];
	uint28 *phash = (uint28*)hash;
	uint28 *outpt = (uint28*)h;
	outpt[0] = phash[0];
	outpt[1] = phash[1];

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
		 a = P[i];
		 b = P[i + 4];
		 c = h[i + 8];
		 d = P[i + 8];
		
		 ab = a ^ b;
		 bc = b ^ c;
		 cd = c ^ d;


		 t = (ab & 0x80808080);
		 t2 = (bc & 0x80808080);
		 t3 = (cd & 0x80808080);

		 abx = mul27(t >> 7) ^ ((ab^t) << 1);
		 bcx = mul27(t2 >> 7) ^ ((bc^t2) << 1);
		 cdx = mul27(t3 >> 7) ^ ((cd^t3) << 1);

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

		abx = mul27(t >> 7) ^ ((ab^t) << 1);
		bcx = mul27(t2 >> 7) ^ ((bc^t2) << 1);
		cdx = mul27(t3 >> 7) ^ ((cd^t3) << 1);

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

		abx = mul27(t >> 7) ^ ((ab^t) << 1);
		bcx = mul27(t2 >> 7) ^ ((bc^t2) << 1);
		cdx = mul27(t3 >> 7) ^ ((cd^t3) << 1);

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

		abx = mul27(t >> 7) ^ ((ab^t) << 1);
		bcx = mul27(t2 >> 7) ^ ((bc^t2) << 1);
		cdx = mul27(t3 >> 7) ^ ((cd^t3) << 1);

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

				a = W[idx + i];
				 b = W[idx + i + 4];
				 c = W[idx + i + 8];
				 d = W[idx + i + 12];

				 ab = a ^ b;
				 bc = b ^ c;
				 cd = c ^ d;

				t = (ab & 0x80808080);
				t2 = (bc & 0x80808080);
				t3 = (cd & 0x80808080);

				 abx = (mul27(t >> 7)  ^ ((ab^t) << 1));
				 bcx = (mul27(t2 >> 7) ^ ((bc^t2) << 1));
				 cdx = (mul27(t3 >> 7) ^ ((cd^t3) << 1));

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


__device__ __forceinline__
void echo_gpu_init_128(uint32_t *const __restrict__ sharedMemory)
{
	if (threadIdx.x < 128) 
	{
		sharedMemory[threadIdx.x] = d_AES0[threadIdx.x];
		sharedMemory[threadIdx.x + 128] = d_AES0[threadIdx.x + 128];
		sharedMemory[threadIdx.x + 256] = ROL8(sharedMemory[threadIdx.x]);
		sharedMemory[threadIdx.x + 256 + 128] = ROL8(sharedMemory[threadIdx.x + 128]);
		sharedMemory[threadIdx.x + 512] = ROL16(sharedMemory[threadIdx.x]);
		sharedMemory[threadIdx.x + 512 + 128] = ROL16(sharedMemory[threadIdx.x + 128]);
		sharedMemory[threadIdx.x + 768] = ROL24(sharedMemory[threadIdx.x]);
		sharedMemory[threadIdx.x + 768 + 128] = ROL24(sharedMemory[threadIdx.x + 128]);
	}
}


/*__device__ __forceinline__
void echo_gpu_init(uint32_t *const __restrict__ sharedMemory)
{
	if (threadIdx.x < 256) 
	{
		sharedMemory[threadIdx.x] = d_AES0[threadIdx.x];
		sharedMemory[threadIdx.x + 256] = ROL8(sharedMemory[threadIdx.x]);
		sharedMemory[threadIdx.x + 512] = ROL16(sharedMemory[threadIdx.x]);
		sharedMemory[threadIdx.x + 768] = ROL24(sharedMemory[threadIdx.x]);
	}
}
*/

__global__
__launch_bounds__(256,2)
void x11_echo512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint64_t *const __restrict__ g_hash)
{
	__shared__ __align__(128) uint32_t sharedMemory[1024];

	echo_gpu_init_128(sharedMemory);

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
	uint32_t threadsperblock = 128;
    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    x11_echo512_gpu_hash_64<<<grid, block>>>(threads, startNounce, (uint64_t*)d_hash);
}

__host__ void x11_echo512_cpu_free(int32_t thr_id)
{
	cudaFreeHost(&d_nonce[thr_id]);
}

/*
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
*/
__global__
__launch_bounds__(128, 4)
void x11_echo512_gpu_hash_64_final(uint32_t threads, uint32_t startNounce, const uint64_t *const __restrict__ g_hash, uint32_t *const __restrict__ d_found, uint32_t target)
{
	uint32_t P[48] = {
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
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{

		__shared__ __align__(128) uint32_t sharedMemory[1024];
		echo_gpu_init_128(sharedMemory);

		uint32_t nounce = (startNounce + thread);

		int hashPosition = nounce - startNounce;
		uint32_t *Hash = (uint32_t*)&g_hash[hashPosition * 8];

		uint32_t h[16];
		uint28 *phash = (uint28*)Hash;
		uint28 *outpt = (uint28*)h;
		outpt[0] = phash[0];
		outpt[1] = phash[1];

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
		uint32_t abx, abx2, abx3, abx4,bcx,bcx2,bcx3,bcx4,cdx,cdx2,cdx3,cdx4;
		//		#pragma unroll
//		for (int i = 0; i < 4; i++)
//		{
		uint32_t i = 0;
			abx = mul27(((P[i] ^ P[i + 4]) & 0x80808080) >> 7) ^ ((P[i] ^ P[i + 4] ^ ((P[i] ^ P[i + 4]) & 0x80808080)) << 1);
			abx2 = mul27(((P[12 + i] ^ h[i + 4]) & 0x80808080) >> 7) ^ (((P[12 + i] ^ h[i + 4]) ^ ((P[12 + i] ^ h[i + 4]) & 0x80808080)) << 1);
			abx3 = mul27(((h[i] ^ P[24 + i + 0]) & 0x80808080) >> 7) ^ (((h[i] ^ P[24 + i + 0]) ^ ((h[i] ^ P[24 + i + 0]) & 0x80808080)) << 1);
			abx4 = mul27(((P[36 + i] ^ P[36 + i + 4]) & 0x80808080) >> 7) ^ (((P[36 + i] ^ P[36 + i + 4]) ^ ((P[36 + i] ^ P[36 + i + 4]) & 0x80808080)) << 1);
			bcx = mul27(((P[i + 4] ^ h[i + 8]) & 0x80808080) >> 7) ^ ((P[i + 4] ^ h[i + 8] ^ ((P[i + 4] ^ h[i + 8]) & 0x80808080)) << 1);
			bcx2 = mul27(((h[i + 4] ^ P[12 + i + 4]) & 0x80808080) >> 7) ^ ((h[i + 4] ^ P[12 + i + 4] ^ ((h[i + 4] ^ P[12 + i + 4]) & 0x80808080)) << 1);
			bcx3 = mul27(((P[24 + i + 0] ^ P[24 + i + 4]) & 0x80808080) >> 7) ^ (((P[24 + i + 0] ^ P[24 + i + 4]) ^ ((P[24 + i + 0] ^ P[24 + i + 4]) & 0x80808080)) << 1);
			bcx4 = mul27(((P[36 + i + 4] ^ P[36 + i + 8]) & 0x80808080) >> 7) ^ (((P[36 + i + 4] ^ P[36 + i + 8]) ^ ((P[36 + i + 4] ^ P[36 + i + 8]) & 0x80808080)) << 1);
			cdx = mul27(((h[i + 8] ^ P[i + 8]) & 0x80808080) >> 7) ^ (((h[i + 8] ^ P[i + 8]) ^ ((h[i + 8] ^ P[i + 8]) & 0x80808080)) << 1);
			cdx2 = mul27(((P[12 + i + 4] ^ P[12 + i + 8]) & 0x80808080) >> 7) ^ (((P[12 + i + 4] ^ P[12 + i + 8]) ^ ((P[12 + i + 4] ^ P[12 + i + 8]) & 0x80808080)) << 1);
			cdx3 = mul27(((P[24 + i + 4] ^ P[24 + i + 8]) & 0x80808080) >> 7) ^ (((P[24 + i + 4] ^ P[24 + i + 8]) ^ ((P[24 + i + 4] ^ P[24 + i + 8]) & 0x80808080)) << 1);
			cdx4 = mul27(((P[36 + i + 8] ^ h[i + 12]) & 0x80808080) >> 7) ^ (((P[36 + i + 8] ^ h[i + 12]) ^ ((P[36 + i + 8] ^ h[i + 12]) & 0x80808080)) << 1);

			W[0 + i] = abx ^ P[i + 4] ^ h[i + 8] ^ P[i + 8];
			W[0 + i + 4] = bcx ^ P[i] ^ h[i + 8] ^ P[i + 8];
			W[0 + i + 8] = cdx ^  P[i] ^ P[i + 4] ^ P[i + 8];
			W[0 + i + 12] = abx ^ bcx ^ cdx ^  P[i] ^ P[i + 4] ^ h[i + 8];
			W[16 + i] = abx2 ^ h[i + 4] ^ P[12 + i + 4] ^ P[12 + i + 8];
			W[16 + i + 4] = bcx2 ^ P[12 + i] ^ P[12 + i + 4] ^ P[12 + i + 8];
			W[16 + i + 8] = cdx2 ^ P[12 + i] ^ h[i + 4] ^ P[12 + i + 8];
			W[16 + i + 12] = abx2 ^ bcx2 ^ cdx2 ^ P[12 + i] ^ h[i + 4] ^ P[12 + i + 4];
			W[32 + i] = abx3 ^ P[24 + i + 0] ^ P[24 + i + 4] ^ P[24 + i + 8];
			W[32 + i + 4] = bcx3 ^ h[i] ^ P[24 + i + 4] ^ P[24 + i + 8];
			W[32 + i + 8] = cdx3 ^ h[i] ^ P[24 + i + 0] ^ P[24 + i + 8];
			W[32 + i + 12] = abx3 ^ bcx3 ^ cdx3 ^ h[i] ^ P[24 + i + 0] ^ P[24 + i + 4];
			W[48 + i] = abx4 ^ P[36 + i + 4] ^ P[36 + i + 8] ^ h[i + 12];
			W[48 + i + 4] = bcx4 ^ P[36 + i] ^ P[36 + i + 8] ^ h[i + 12];
			W[48 + i + 8] = cdx4 ^ P[36 + i] ^ P[36 + i + 4] ^ h[i + 12];
			W[48 + i + 12] = abx4 ^ bcx4 ^ cdx4 ^ P[36 + i] ^ P[36 + i + 4] ^ P[36 + i + 8];

			i = 1;

			abx = mul27(((P[i] ^ P[i + 4]) & 0x80808080) >> 7) ^ ((P[i] ^ P[i + 4] ^ ((P[i] ^ P[i + 4]) & 0x80808080)) << 1);
			abx2 = mul27(((P[12 + i] ^ h[i + 4]) & 0x80808080) >> 7) ^ (((P[12 + i] ^ h[i + 4]) ^ ((P[12 + i] ^ h[i + 4]) & 0x80808080)) << 1);
			abx3 = mul27(((h[i] ^ P[24 + i + 0]) & 0x80808080) >> 7) ^ (((h[i] ^ P[24 + i + 0]) ^ ((h[i] ^ P[24 + i + 0]) & 0x80808080)) << 1);
			abx4 = mul27(((P[36 + i] ^ P[36 + i + 4]) & 0x80808080) >> 7) ^ (((P[36 + i] ^ P[36 + i + 4]) ^ ((P[36 + i] ^ P[36 + i + 4]) & 0x80808080)) << 1);
			bcx = mul27(((P[i + 4] ^ h[i + 8]) & 0x80808080) >> 7) ^ ((P[i + 4] ^ h[i + 8] ^ ((P[i + 4] ^ h[i + 8]) & 0x80808080)) << 1);
			bcx2 = mul27(((h[i + 4] ^ P[12 + i + 4]) & 0x80808080) >> 7) ^ ((h[i + 4] ^ P[12 + i + 4] ^ ((h[i + 4] ^ P[12 + i + 4]) & 0x80808080)) << 1);
			bcx3 = mul27(((P[24 + i + 0] ^ P[24 + i + 4]) & 0x80808080) >> 7) ^ (((P[24 + i + 0] ^ P[24 + i + 4]) ^ ((P[24 + i + 0] ^ P[24 + i + 4]) & 0x80808080)) << 1);
			bcx4 = mul27(((P[36 + i + 4] ^ P[36 + i + 8]) & 0x80808080) >> 7) ^ (((P[36 + i + 4] ^ P[36 + i + 8]) ^ ((P[36 + i + 4] ^ P[36 + i + 8]) & 0x80808080)) << 1);
			cdx = mul27(((h[i + 8] ^ P[i + 8]) & 0x80808080) >> 7) ^ (((h[i + 8] ^ P[i + 8]) ^ ((h[i + 8] ^ P[i + 8]) & 0x80808080)) << 1);
			cdx2 = mul27(((P[12 + i + 4] ^ P[12 + i + 8]) & 0x80808080) >> 7) ^ (((P[12 + i + 4] ^ P[12 + i + 8]) ^ ((P[12 + i + 4] ^ P[12 + i + 8]) & 0x80808080)) << 1);
			cdx3 = mul27(((P[24 + i + 4] ^ P[24 + i + 8]) & 0x80808080) >> 7) ^ (((P[24 + i + 4] ^ P[24 + i + 8]) ^ ((P[24 + i + 4] ^ P[24 + i + 8]) & 0x80808080)) << 1);
			cdx4 = mul27(((P[36 + i + 8] ^ h[i + 12]) & 0x80808080) >> 7) ^ (((P[36 + i + 8] ^ h[i + 12]) ^ ((P[36 + i + 8] ^ h[i + 12]) & 0x80808080)) << 1);

			W[0 + i] = abx ^ P[i + 4] ^ h[i + 8] ^ P[i + 8];
			W[0 + i + 4] = bcx ^ P[i] ^ h[i + 8] ^ P[i + 8];
			W[0 + i + 8] = cdx ^  P[i] ^ P[i + 4] ^ P[i + 8];
			W[0 + i + 12] = abx ^ bcx ^ cdx ^  P[i] ^ P[i + 4] ^ h[i + 8];
			W[16 + i] = abx2 ^ h[i + 4] ^ P[12 + i + 4] ^ P[12 + i + 8];
			W[16 + i + 4] = bcx2 ^ P[12 + i] ^ P[12 + i + 4] ^ P[12 + i + 8];
			W[16 + i + 8] = cdx2 ^ P[12 + i] ^ h[i + 4] ^ P[12 + i + 8];
			W[16 + i + 12] = abx2 ^ bcx2 ^ cdx2 ^ P[12 + i] ^ h[i + 4] ^ P[12 + i + 4];
			W[32 + i] = abx3 ^ P[24 + i + 0] ^ P[24 + i + 4] ^ P[24 + i + 8];
			W[32 + i + 4] = bcx3 ^ h[i] ^ P[24 + i + 4] ^ P[24 + i + 8];
			W[32 + i + 8] = cdx3 ^ h[i] ^ P[24 + i + 0] ^ P[24 + i + 8];
			W[32 + i + 12] = abx3 ^ bcx3 ^ cdx3 ^ h[i] ^ P[24 + i + 0] ^ P[24 + i + 4];
			W[48 + i] = abx4 ^ P[36 + i + 4] ^ P[36 + i + 8] ^ h[i + 12];
			W[48 + i + 4] = bcx4 ^ P[36 + i] ^ P[36 + i + 8] ^ h[i + 12];
			W[48 + i + 8] = cdx4 ^ P[36 + i] ^ P[36 + i + 4] ^ h[i + 12];
			W[48 + i + 12] = abx4 ^ bcx4 ^ cdx4 ^ P[36 + i] ^ P[36 + i + 4] ^ P[36 + i + 8];

			i = 2;

			abx = mul27(((P[i] ^ P[i + 4]) & 0x80808080) >> 7) ^ ((P[i] ^ P[i + 4] ^ ((P[i] ^ P[i + 4]) & 0x80808080)) << 1);
			abx2 = mul27(((P[12 + i] ^ h[i + 4]) & 0x80808080) >> 7) ^ (((P[12 + i] ^ h[i + 4]) ^ ((P[12 + i] ^ h[i + 4]) & 0x80808080)) << 1);
			abx3 = mul27(((h[i] ^ P[24 + i + 0]) & 0x80808080) >> 7) ^ (((h[i] ^ P[24 + i + 0]) ^ ((h[i] ^ P[24 + i + 0]) & 0x80808080)) << 1);
			abx4 = mul27(((P[36 + i] ^ P[36 + i + 4]) & 0x80808080) >> 7) ^ (((P[36 + i] ^ P[36 + i + 4]) ^ ((P[36 + i] ^ P[36 + i + 4]) & 0x80808080)) << 1);
			bcx = mul27(((P[i + 4] ^ h[i + 8]) & 0x80808080) >> 7) ^ ((P[i + 4] ^ h[i + 8] ^ ((P[i + 4] ^ h[i + 8]) & 0x80808080)) << 1);
			bcx2 = mul27(((h[i + 4] ^ P[12 + i + 4]) & 0x80808080) >> 7) ^ ((h[i + 4] ^ P[12 + i + 4] ^ ((h[i + 4] ^ P[12 + i + 4]) & 0x80808080)) << 1);
			bcx3 = mul27(((P[24 + i + 0] ^ P[24 + i + 4]) & 0x80808080) >> 7) ^ (((P[24 + i + 0] ^ P[24 + i + 4]) ^ ((P[24 + i + 0] ^ P[24 + i + 4]) & 0x80808080)) << 1);
			bcx4 = mul27(((P[36 + i + 4] ^ P[36 + i + 8]) & 0x80808080) >> 7) ^ (((P[36 + i + 4] ^ P[36 + i + 8]) ^ ((P[36 + i + 4] ^ P[36 + i + 8]) & 0x80808080)) << 1);
			cdx = mul27(((h[i + 8] ^ P[i + 8]) & 0x80808080) >> 7) ^ (((h[i + 8] ^ P[i + 8]) ^ ((h[i + 8] ^ P[i + 8]) & 0x80808080)) << 1);
			cdx2 = mul27(((P[12 + i + 4] ^ P[12 + i + 8]) & 0x80808080) >> 7) ^ (((P[12 + i + 4] ^ P[12 + i + 8]) ^ ((P[12 + i + 4] ^ P[12 + i + 8]) & 0x80808080)) << 1);
			cdx3 = mul27(((P[24 + i + 4] ^ P[24 + i + 8]) & 0x80808080) >> 7) ^ (((P[24 + i + 4] ^ P[24 + i + 8]) ^ ((P[24 + i + 4] ^ P[24 + i + 8]) & 0x80808080)) << 1);
			cdx4 = mul27(((P[36 + i + 8] ^ h[i + 12]) & 0x80808080) >> 7) ^ (((P[36 + i + 8] ^ h[i + 12]) ^ ((P[36 + i + 8] ^ h[i + 12]) & 0x80808080)) << 1);

			W[0 + i] = abx ^ P[i + 4] ^ h[i + 8] ^ P[i + 8];
			W[0 + i + 4] = bcx ^ P[i] ^ h[i + 8] ^ P[i + 8];
			W[0 + i + 8] = cdx ^  P[i] ^ P[i + 4] ^ P[i + 8];
			W[0 + i + 12] = abx ^ bcx ^ cdx ^  P[i] ^ P[i + 4] ^ h[i + 8];
			W[16 + i] = abx2 ^ h[i + 4] ^ P[12 + i + 4] ^ P[12 + i + 8];
			W[16 + i + 4] = bcx2 ^ P[12 + i] ^ P[12 + i + 4] ^ P[12 + i + 8];
			W[16 + i + 8] = cdx2 ^ P[12 + i] ^ h[i + 4] ^ P[12 + i + 8];
			W[16 + i + 12] = abx2 ^ bcx2 ^ cdx2 ^ P[12 + i] ^ h[i + 4] ^ P[12 + i + 4];
			W[32 + i] = abx3 ^ P[24 + i + 0] ^ P[24 + i + 4] ^ P[24 + i + 8];
			W[32 + i + 4] = bcx3 ^ h[i] ^ P[24 + i + 4] ^ P[24 + i + 8];
			W[32 + i + 8] = cdx3 ^ h[i] ^ P[24 + i + 0] ^ P[24 + i + 8];
			W[32 + i + 12] = abx3 ^ bcx3 ^ cdx3 ^ h[i] ^ P[24 + i + 0] ^ P[24 + i + 4];
			W[48 + i] = abx4 ^ P[36 + i + 4] ^ P[36 + i + 8] ^ h[i + 12];
			W[48 + i + 4] = bcx4 ^ P[36 + i] ^ P[36 + i + 8] ^ h[i + 12];
			W[48 + i + 8] = cdx4 ^ P[36 + i] ^ P[36 + i + 4] ^ h[i + 12];
			W[48 + i + 12] = abx4 ^ bcx4 ^ cdx4 ^ P[36 + i] ^ P[36 + i + 4] ^ P[36 + i + 8];

			i = 3;

			abx = mul27(((P[i] ^ P[i + 4]) & 0x80808080) >> 7) ^ ((P[i] ^ P[i + 4] ^ ((P[i] ^ P[i + 4]) & 0x80808080)) << 1);
			abx2 = mul27(((P[12 + i] ^ h[i + 4]) & 0x80808080) >> 7) ^ (((P[12 + i] ^ h[i + 4]) ^ ((P[12 + i] ^ h[i + 4]) & 0x80808080)) << 1);
			abx3 = mul27(((h[i] ^ P[24 + i + 0]) & 0x80808080) >> 7) ^ (((h[i] ^ P[24 + i + 0]) ^ ((h[i] ^ P[24 + i + 0]) & 0x80808080)) << 1);
			abx4 = mul27(((P[36 + i] ^ P[36 + i + 4]) & 0x80808080) >> 7) ^ (((P[36 + i] ^ P[36 + i + 4]) ^ ((P[36 + i] ^ P[36 + i + 4]) & 0x80808080)) << 1);
			bcx = mul27(((P[i + 4] ^ h[i + 8]) & 0x80808080) >> 7) ^ ((P[i + 4] ^ h[i + 8] ^ ((P[i + 4] ^ h[i + 8]) & 0x80808080)) << 1);
			bcx2 = mul27(((h[i + 4] ^ P[12 + i + 4]) & 0x80808080) >> 7) ^ ((h[i + 4] ^ P[12 + i + 4] ^ ((h[i + 4] ^ P[12 + i + 4]) & 0x80808080)) << 1);
			bcx3 = mul27(((P[24 + i + 0] ^ P[24 + i + 4]) & 0x80808080) >> 7) ^ (((P[24 + i + 0] ^ P[24 + i + 4]) ^ ((P[24 + i + 0] ^ P[24 + i + 4]) & 0x80808080)) << 1);
			bcx4 = mul27(((P[36 + i + 4] ^ P[36 + i + 8]) & 0x80808080) >> 7) ^ (((P[36 + i + 4] ^ P[36 + i + 8]) ^ ((P[36 + i + 4] ^ P[36 + i + 8]) & 0x80808080)) << 1);
			cdx = mul27(((h[i + 8] ^ P[i + 8]) & 0x80808080) >> 7) ^ (((h[i + 8] ^ P[i + 8]) ^ ((h[i + 8] ^ P[i + 8]) & 0x80808080)) << 1);
			cdx2 = mul27(((P[12 + i + 4] ^ P[12 + i + 8]) & 0x80808080) >> 7) ^ (((P[12 + i + 4] ^ P[12 + i + 8]) ^ ((P[12 + i + 4] ^ P[12 + i + 8]) & 0x80808080)) << 1);
			cdx3 = mul27(((P[24 + i + 4] ^ P[24 + i + 8]) & 0x80808080) >> 7) ^ (((P[24 + i + 4] ^ P[24 + i + 8]) ^ ((P[24 + i + 4] ^ P[24 + i + 8]) & 0x80808080)) << 1);
			cdx4 = mul27(((P[36 + i + 8] ^ h[i + 12]) & 0x80808080) >> 7) ^ (((P[36 + i + 8] ^ h[i + 12]) ^ ((P[36 + i + 8] ^ h[i + 12]) & 0x80808080)) << 1);

			W[0 + i] = abx ^ P[i + 4] ^ h[i + 8] ^ P[i + 8];
			W[0 + i + 4] = bcx ^ P[i] ^ h[i + 8] ^ P[i + 8];
			W[0 + i + 8] = cdx ^  P[i] ^ P[i + 4] ^ P[i + 8];
			W[0 + i + 12] = abx ^ bcx ^ cdx ^  P[i] ^ P[i + 4] ^ h[i + 8];
			W[16 + i] = abx2 ^ h[i + 4] ^ P[12 + i + 4] ^ P[12 + i + 8];
			W[16 + i + 4] = bcx2 ^ P[12 + i] ^ P[12 + i + 4] ^ P[12 + i + 8];
			W[16 + i + 8] = cdx2 ^ P[12 + i] ^ h[i + 4] ^ P[12 + i + 8];
			W[16 + i + 12] = abx2 ^ bcx2 ^ cdx2 ^ P[12 + i] ^ h[i + 4] ^ P[12 + i + 4];
			W[32 + i] = abx3 ^ P[24 + i + 0] ^ P[24 + i + 4] ^ P[24 + i + 8];
			W[32 + i + 4] = bcx3 ^ h[i] ^ P[24 + i + 4] ^ P[24 + i + 8];
			W[32 + i + 8] = cdx3 ^ h[i] ^ P[24 + i + 0] ^ P[24 + i + 8];
			W[32 + i + 12] = abx3 ^ bcx3 ^ cdx3 ^ h[i] ^ P[24 + i + 0] ^ P[24 + i + 4];
			W[48 + i] = abx4 ^ P[36 + i + 4] ^ P[36 + i + 8] ^ h[i + 12];
			W[48 + i + 4] = bcx4 ^ P[36 + i] ^ P[36 + i + 8] ^ h[i + 12];
			W[48 + i + 8] = cdx4 ^ P[36 + i] ^ P[36 + i + 4] ^ h[i + 12];
			W[48 + i + 12] = abx4 ^ bcx4 ^ cdx4 ^ P[36 + i] ^ P[36 + i + 4] ^ P[36 + i + 8];

			//		}

		uint32_t k0 = 512 + 16;
		uint32_t t, t2, t3;
		uint32_t a, b, c, d;
		uint32_t ab, bc, cd;
//		uint32_t abx, bcx, cdx;

		for (int k = 1; k < 9; k++)
		{

				AES_2ROUND(sharedMemory, W[0 + 4], W[0 + 5], W[0 + 6], W[0 + 7], k0 + 1);
				AES_2ROUND(sharedMemory, W[16 + 4], W[16 + 5], W[16 + 6], W[16 + 7], k0 + 5);
				AES_2ROUND(sharedMemory, W[32 + 4], W[32 + 5], W[32 + 6], W[32 + 7], k0 + 9);
				AES_2ROUND(sharedMemory, W[48 + 4], W[48 + 5], W[48 + 6], W[48 + 7], k0 + 13);

			/// 1, 5, 9, 13
				 t = W[4 + 0];
				W[4 + 0] = W[20 + 0];
				W[20 + 0] = W[36 + 0];
				W[36 + 0] = W[52 + 0];
				W[52 + 0] = t;

				AES_2ROUND(sharedMemory, W[0 + 8], W[0 + 9], W[0 + 10], W[0 + 11], k0 + 2);
				AES_2ROUND(sharedMemory, W[32 + 8], W[32 + 9], W[32 + 10], W[32 + 11], k0 + 10);
				AES_2ROUND(sharedMemory, W[16 + 8], W[16 + 9], W[16 + 10], W[16 + 11], k0 + 6);
				AES_2ROUND(sharedMemory, W[48 + 8], W[48 + 9], W[48 + 10], W[48 + 11], k0 + 14);

				// 2, 6, 10, 14
				t = W[8 + 0];
				W[8 + 0] = W[40 + 0];
				W[40 + 0] = t;
				t = W[24 + 0];
				W[24 + 0] = W[56 + 0];
				W[56 + 0] = t;

				AES_2ROUND(sharedMemory, W[48 + 12], W[48 + 13], W[48 + 14], W[48 + 15], k0 + 15);
				AES_2ROUND(sharedMemory, W[32 + 12], W[32 + 13], W[32 + 14], W[32 + 15], k0 + 11);
				AES_2ROUND(sharedMemory, W[16 + 12], W[16 + 13], W[16 + 14], W[16 + 15], k0 + 7);
				AES_2ROUND(sharedMemory, W[0 + 12], W[0 + 13], W[0 + 14], W[0 + 15], k0 + 3);


				// 15, 11, 7, 3
				t = W[60 + 0];
				W[60 + 0] = W[44 + 0];
				W[44 + 0] = W[28 + 0];
				W[28 + 0] = W[12 + 0];
				W[12 + 0] = t;


				/// 1, 5, 9, 13
				t = W[4 + 1];
				W[4 + 1] = W[20 + 1];
				W[20 + 1] = W[36 + 1];
				W[36 + 1] = W[52 + 1];
				W[52 + 1] = t;

				AES_2ROUND(sharedMemory, W[0 + 0], W[0 + 1], W[0 + 2], W[0 + 3], k0);

				// 2, 6, 10, 14
				t = W[8 + 1];
				W[8 + 1] = W[40 + 1];
				W[40 + 1] = t;
				t = W[24 + 1];
				W[24 + 1] = W[56 + 1];
				W[56 + 1] = t;


				// 15, 11, 7, 3
				t = W[60 + 1];
				W[60 + 1] = W[44 + 1];
				W[44 + 1] = W[28 + 1];
				W[28 + 1] = W[12 + 1];
				W[12 + 1] = t;

				AES_2ROUND(sharedMemory, W[16 + 0], W[16 + 1], W[16 + 2], W[16 + 3], k0 + 4);

				/// 1, 5, 9, 13
				t = W[4 + 2];
				W[4 + 2] = W[20 + 2];
				W[20 + 2] = W[36 + 2];
				W[36 + 2] = W[52 + 2];
				W[52 + 2] = t;

				// 2, 6, 10, 14
				t = W[8 + 2];
				W[8 + 2] = W[40 + 2];
				W[40 + 2] = t;
				t = W[24 + 2];
				W[24 + 2] = W[56 + 2];
				W[56 + 2] = t;

				AES_2ROUND(sharedMemory, W[32 + 0], W[32 + 1], W[32 + 2], W[32 + 3], k0 + 8);

				// 15, 11, 7, 3
				t = W[60 + 2];
				W[60 + 2] = W[44 + 2];
				W[44 + 2] = W[28 + 2];
				W[28 + 2] = W[12 + 2];
				W[12 + 2] = t;


				/// 1, 5, 9, 13
				t = W[4 + 3];
				W[4 + 3] = W[20 + 3];
				W[20 + 3] = W[36 + 3];
				W[36 + 3] = W[52 + 3];
				W[52 + 3] = t;

				AES_2ROUND(sharedMemory, W[48 + 0], W[48 + 1], W[48 + 2], W[48 + 3], k0 + 12);

				// 2, 6, 10, 14
				t = W[8 + 3];
				W[8 + 3] = W[40 + 3];
				W[40 + 3] = t;
				t = W[24 + 3];
				W[24 + 3] = W[56 + 3];
				W[56 + 3] = t;

				// 15, 11, 7, 3
				t = W[60 + 3];
				W[60 + 3] = W[44 + 3];
				W[44 + 3] = W[28 + 3];
				W[28 + 3] = W[12 + 3];
				W[12 + 3] = t;


				k0 = k0 + 16;


			// Mix Columns
			#pragma unroll
			for (int i = 0; i < 4; i++) // Schleife über je 2*uint32_t
			{
				#pragma unroll
				for (int idx = 0; idx < 64; idx += 16) // Schleife über die elemnte
				{


					a = W[idx + i];
					b = W[idx + i + 4];
					c = W[idx + i + 8];
					d = W[idx + i + 12];

					ab = a ^ b;
					bc = b ^ c;
					cd = c ^ d;

					t = (ab & 0x80808080);
					t2 = (bc & 0x80808080);
					t3 = (cd & 0x80808080);



					abx = mul27(t >> 7) ^ ((ab^t) << 1);
					bcx = mul27(t2 >> 7) ^ ((bc^t2) << 1);
					cdx = mul27(t3 >> 7) ^ ((cd^t3) << 1);

					W[idx + i] = abx ^ bc ^ d;
					W[idx + i + 4] = bcx ^ a ^ cd;
					W[idx + i + 8] = cdx ^ ab ^ d;
					W[idx + i + 12] = abx ^ bcx ^ cdx ^ ab ^ c;

				}
			}
		}

		//3, 11, 23, 31, 35, 43, 55, 63

		AES_2ROUND(sharedMemory,
			W[0 + 0], W[0 + 1], W[0 + 2], W[0 + 3],
			512 + (9 * 16));
		AES_2ROUND(sharedMemory,
			W[0 + 8], W[0 + 9], W[0 + 10], W[0 + 11],
			512 + (9 * 16) + 2);
		AES_2ROUND(sharedMemory,
			W[16 + 4], W[16 + 5], W[16 + 6], W[16 + 7],
			512 + (9 * 16) + 5);
		AES_2ROUND(sharedMemory,
			W[16 + 12], W[16 + 13], W[16 + 14], W[16 + 15],
			512 + (9 * 16) + 7);
		AES_2ROUND(sharedMemory,
			W[32 + 0], W[32 + 1], W[32 + 2], W[32 + 3],
			512 + (9 * 16) + 8);
		AES_2ROUND(sharedMemory,
			W[32 + 8], W[32 + 9], W[32 + 10], W[32 + 11],
			512 + (9 * 16) + 10);
		AES_2ROUND(sharedMemory,
			W[48 + 4], W[48 + 5], W[48 + 6], W[48 + 7],
			512 + (9 * 16) + 13);

		AES_2ROUND(sharedMemory,
			W[60], W[61], W[62], W[63],
			512 + (9 * 16) + 15);

		bc = W[23] ^ W[43];
		t2 = (bc & 0x80808080);
		uint32_t test = mul27(t2 >> 7) ^ ((bc^t2) << 1) ^ W[3] ^ W[43] ^ W[63];
		bc = W[55] ^ W[11];
		t2 = (bc & 0x80808080);
		test ^= mul27(t2 >> 7) ^ ((bc^t2) << 1) ^ W[35] ^ W[11] ^ W[31] ^ backup;
		if (test <= target)
		{
			uint32_t tmp = atomicCAS(d_found, 0xffffffff, nounce);
			if (tmp != 0xffffffff)
				d_found[1] = nounce;
		}
	}
}
const uint32_t threadsperblock = 128;
__host__ void x11_echo512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, uint32_t target, uint32_t *h_found)
{


	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	cudaMemset(d_found[thr_id], 0xffffffff, 2*sizeof(uint32_t));

	x11_echo512_gpu_hash_64_final << <grid, block>> >(threads, startNounce, (uint64_t*)d_hash, d_found[thr_id], target);
	cudaMemcpy(h_found, d_found[thr_id], 2*sizeof(uint32_t), cudaMemcpyDeviceToHost);
}
