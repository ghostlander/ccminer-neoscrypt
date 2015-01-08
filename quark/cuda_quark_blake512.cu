#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

#define ROTR(x,n) ROTR64(x,n)

#define USE_SHUFFLE 0

// die Message it Padding zur Berechnung auf der GPU
__constant__ uint64_t c_PaddedMessage80[16]; // padded message (80 bytes + padding)

// ---------------------------- BEGIN CUDA quark_blake512 functions ------------------------------------

__constant__ uint8_t c_sigma[16][16] =
{
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 }
};

__device__ __constant__
const uint64_t c_u512[16] =
{
	0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL,
	0xa4093822299f31d0ULL, 0x082efa98ec4e6c89ULL,
	0x452821e638d01377ULL, 0xbe5466cf34e90c6cULL,
	0xc0ac29b7c97c50ddULL, 0x3f84d5b5b5470917ULL,
	0x9216d5d98979fb1bULL, 0xd1310ba698dfb5acULL,
	0x2ffd72dbd01adfb7ULL, 0xb8e1afed6a267e96ULL,
	0xba7c9045f12c7f99ULL, 0x24a19947b3916cf7ULL,
	0x0801f2e2858efc16ULL, 0x636920d871574e69ULL
};

#define G(a,b,c,d,x) { \
	uint8_t idx1 = c_sigma[i][x]; \
	uint8_t idx2 = c_sigma[i][x+1]; \
	v[a] += vectorize(block[idx1] ^ c_u512[idx2]) + v[b]; \
	v[d] = SWAPDWORDS2( v[d] ^ v[a]); \
	v[c] += v[d]; \
	v[b] = ROR2(v[b] ^ v[c], 25); \
	v[a] += vectorize(block[idx2] ^ c_u512[idx1]) + v[b]; \
	v[d] = ROR2(v[d] ^ v[a],16); \
	v[c] += v[d]; \
	v[b] = ROR2(v[b] ^ v[c], 11); \
  }

#define Gprecalc(a,b,c,d,idx1,idx2) { \
	v[a] += vectorize(block[idx2] ^ u512[idx1]) + v[b]; \
	v[d] = SWAPDWORDS2( v[d] ^ v[a]); \
	v[c] += v[d]; \
	v[b] = ROR2(v[b] ^ v[c], 25); \
	v[a] += vectorize(block[idx1] ^ u512[idx2]) + v[b]; \
	v[d] = ROR2(v[d] ^ v[a],16); \
	v[c] += v[d]; \
	v[b] = ROR2(v[b] ^ v[c], 11); \
	}

__global__ 

#if __CUDA_ARCH__ > 500
	__launch_bounds__(256, 4)
#else
	__launch_bounds__(64, 16)
#endif
void quark_blake512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *const __restrict__ g_nonceVector, uint64_t *const __restrict__ g_hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

#if USE_SHUFFLE
	const int warpID = threadIdx.x & 0x0F; // 16 warps
	const int warpBlockID = (thread + 15)>>4; // aufrunden auf volle Warp-Blöcke
	const int maxHashPosition = thread<<3;
#endif

#if USE_SHUFFLE
	if (warpBlockID < ( (threads+15)>>4 ))
#else
	if (thread < threads)
#endif
	{
		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		int hashPosition = nounce - startNounce;
		uint64_t *inpHash = &g_hash[hashPosition<<3]; // hashPosition * 8

		// 128 Bytes
		uint64_t block[16];

		// Message for first round
		#pragma unroll 8
		for (int i=0; i < 8; ++i)
			block[i] = cuda_swab64(inpHash[i]);
		block[ 8] = 0x8000000000000000;
		block[ 9] = 0;
		block[10] = 0;
		block[11] = 0;
		block[12] = 0;
		block[13] = 1;
		block[14] = 0;
		block[15] = 0x0000000000000200;

		register uint2 v[16];

		const uint2 h[8] =
		{
				{ 0xf3bcc908UL, 0x6a09e667UL },
				{ 0x84caa73bUL, 0xbb67ae85UL },
				{ 0xfe94f82bUL, 0x3c6ef372UL },
				{ 0x5f1d36f1UL, 0xa54ff53aUL },
				{ 0xade682d1UL, 0x510e527fUL },
				{ 0x2b3e6c1fUL, 0x9b05688cUL },
				{ 0xfb41bd6bUL, 0x1f83d9abUL },
				{ 0x137e2179UL, 0x5be0cd19UL }
		};

#pragma unroll 8
		for (int i = 0; i < 8; i++)
			v[i] = h[i];
		v[8] = vectorize(c_u512[0]);
		v[9] = vectorize(c_u512[1]);
		v[10] = vectorize(c_u512[2]);
		v[11] = vectorize(c_u512[3]);
		v[12] = vectorize(c_u512[4] ^ 512);
		v[13] = vectorize(c_u512[5] ^ 512);
		v[14] = vectorize(c_u512[6]);
		v[15] = vectorize(c_u512[7]);

#pragma unroll 2
		for (int i = 0; i < 16; i++)
		{
			G(0, 4, 8, 12, 0);
			G(1, 5, 9, 13, 2);
			G(2, 6, 10, 14, 4);
			G(3, 7, 11, 15, 6);
			G(0, 5, 10, 15, 8);
			G(1, 6, 11, 12, 10);
			G(2, 7, 8, 13, 12);
			G(3, 4, 9, 14, 14);
		}


		uint64_t *outHash = &g_hash[8 * hashPosition];

		outHash[0] = devectorizeswap(h[0] ^ v[0] ^ v[8]);
		outHash[1] = devectorizeswap(h[1] ^ v[1] ^ v[9]);
		outHash[2] = devectorizeswap(h[2] ^ v[2] ^ v[10]);
		outHash[3] = devectorizeswap(h[3] ^ v[3] ^ v[11]);
		outHash[4] = devectorizeswap(h[4] ^ v[4] ^ v[12]);
		outHash[5] = devectorizeswap(h[5] ^ v[5] ^ v[13]);
		outHash[6] = devectorizeswap(h[6] ^ v[6] ^ v[14]);
		outHash[7] = devectorizeswap(h[7] ^ v[7] ^ v[15]);
	}
}

__global__ 
#if __CUDA_ARCH__ > 500
__launch_bounds__(256, 4)
#else
__launch_bounds__(64, 16)
#endif
void quark_blake512_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint32_t *outputHash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = startNounce + thread;

		uint64_t block[16];

		// Message für die erste Runde in Register holen
#pragma unroll 16
		for (int i = 0; i < 16; ++i)
			block[i] = c_PaddedMessage80[i];
		// The test Nonce
		((uint32_t*)block)[18] = nounce;

		register uint2 v[16];
		register uint64_t u512[16] =
		{
			0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL,
			0xa4093822299f31d0ULL, 0x082efa98ec4e6c89ULL,
			0x452821e638d01377ULL, 0xbe5466cf34e90c6cULL,
			0xc0ac29b7c97c50ddULL, 0x3f84d5b5b5470917ULL,
			0x9216d5d98979fb1bULL, 0xd1310ba698dfb5acULL,
			0x2ffd72dbd01adfb7ULL, 0xb8e1afed6a267e96ULL,
			0xba7c9045f12c7f99ULL, 0x24a19947b3916cf7ULL,
			0x0801f2e2858efc16ULL, 0x636920d871574e69ULL
		};

		const uint2 h[8] = {
				{ 0xf3bcc908UL,0x6a09e667UL },
				{ 0x84caa73bUL ,0xbb67ae85UL },
				{ 0xfe94f82bUL,0x3c6ef372UL },
				{ 0x5f1d36f1UL,0xa54ff53aUL },
				{ 0xade682d1UL,0x510e527fUL },
				{ 0x2b3e6c1fUL,0x9b05688cUL },
				{ 0xfb41bd6bUL,0x1f83d9abUL },
				{ 0x137e2179UL,0x5be0cd19UL }
		};

		#pragma unroll 8
		for (int i = 0; i < 8; i++)
			v[i] = h[i];
		v[8] = vectorize(u512[0]);
		v[9] = vectorize(u512[1]);
		v[10] = vectorize(u512[2]);
		v[11] = vectorize(u512[3]);
		v[12] = vectorize(u512[4] ^ 640);
		v[13] = vectorize(u512[5] ^ 640);
		v[14] = vectorize(u512[6]);
		v[15] = vectorize(u512[7]);

		Gprecalc(0, 4, 8, 12, 0x1, 0x0)
		Gprecalc(1, 5, 9, 13, 0x3, 0x2)
		Gprecalc(2, 6, 10, 14, 0x5, 0x4)
		Gprecalc(3, 7, 11, 15, 0x7, 0x6)
		Gprecalc(0, 5, 10, 15, 0x9, 0x8)
		Gprecalc(1, 6, 11, 12, 0xb, 0xa)
		Gprecalc(2, 7, 8, 13, 0xd, 0xc)
		Gprecalc(3, 4, 9, 14, 0xf, 0xe)

		Gprecalc(0, 4, 8, 12, 0xa, 0xe)
		Gprecalc(1, 5, 9, 13, 0x8, 0x4)
		Gprecalc(2, 6, 10, 14, 0xf, 0x9)
		Gprecalc(3, 7, 11, 15, 0x6, 0xd)
		Gprecalc(0, 5, 10, 15, 0xc, 0x1)
		Gprecalc(1, 6, 11, 12, 0x2, 0x0)
		Gprecalc(2, 7, 8, 13, 0x7, 0xb)
		Gprecalc(3, 4, 9, 14, 0x3, 0x5)

		Gprecalc(0, 4, 8, 12, 0x8, 0xb)
		Gprecalc(1, 5, 9, 13, 0x0, 0xc)
		Gprecalc(2, 6, 10, 14, 0x2, 0x5)
		Gprecalc(3, 7, 11, 15, 0xd, 0xf)
		Gprecalc(0, 5, 10, 15, 0xe, 0xa)
		Gprecalc(1, 6, 11, 12, 0x6, 0x3)
		Gprecalc(2, 7, 8, 13, 0x1, 0x7)
		Gprecalc(3, 4, 9, 14, 0x4, 0x9)

		Gprecalc(0, 4, 8, 12, 0x9, 0x7)
		Gprecalc(1, 5, 9, 13, 0x1, 0x3)
		Gprecalc(2, 6, 10, 14, 0xc, 0xd)
		Gprecalc(3, 7, 11, 15, 0xe, 0xb)
		Gprecalc(0, 5, 10, 15, 0x6, 0x2)
		Gprecalc(1, 6, 11, 12, 0xa, 0x5)
		Gprecalc(2, 7, 8, 13, 0x0, 0x4)
		Gprecalc(3, 4, 9, 14, 0x8, 0xf)

		Gprecalc(0, 4, 8, 12, 0x0, 0x9)
		Gprecalc(1, 5, 9, 13, 0x7, 0x5)
		Gprecalc(2, 6, 10, 14, 0x4, 0x2)
		Gprecalc(3, 7, 11, 15, 0xf, 0xa)
		Gprecalc(0, 5, 10, 15, 0x1, 0xe)
		Gprecalc(1, 6, 11, 12, 0xc, 0xb)
		Gprecalc(2, 7, 8, 13, 0x8, 0x6)
		Gprecalc(3, 4, 9, 14, 0xd, 0x3)
		
		Gprecalc(0, 4, 8, 12, 0xc, 0x2)
		Gprecalc(1, 5, 9, 13, 0xa, 0x6)
		Gprecalc(2, 6, 10, 14, 0xb, 0x0)
		Gprecalc(3, 7, 11, 15, 0x3, 0x8)
		Gprecalc(0, 5, 10, 15, 0xd, 0x4)
		Gprecalc(1, 6, 11, 12, 0x5, 0x7)
		Gprecalc(2, 7, 8, 13, 0xe, 0xf)
		Gprecalc(3, 4, 9, 14, 0x9, 0x1)

		Gprecalc(0, 4, 8, 12, 0x5, 0xc)
		Gprecalc(1, 5, 9, 13, 0xf, 0x1)
		Gprecalc(2, 6, 10, 14, 0xd, 0xe)
		Gprecalc(3, 7, 11, 15, 0xa, 0x4)
		Gprecalc(0, 5, 10, 15, 0x7, 0x0)
		Gprecalc(1, 6, 11, 12, 0x3, 0x6)
		Gprecalc(2, 7, 8, 13, 0x2, 0x9)
		Gprecalc(3, 4, 9, 14, 0xb, 0x8)

		Gprecalc(0, 4, 8, 12, 0xb, 0xd)
		Gprecalc(1, 5, 9, 13, 0xe, 0x7)
		Gprecalc(2, 6, 10, 14, 0x1, 0xc)
		Gprecalc(3, 7, 11, 15, 0x9, 0x3)
		Gprecalc(0, 5, 10, 15, 0x0, 0x5)
		Gprecalc(1, 6, 11, 12, 0x4, 0xf)
		Gprecalc(2, 7, 8, 13, 0x6, 0x8)
		Gprecalc(3, 4, 9, 14, 0xa, 0x2)

		Gprecalc(0, 4, 8, 12, 0xf, 0x6)
		Gprecalc(1, 5, 9, 13, 0x9, 0xe)
		Gprecalc(2, 6, 10, 14, 0x3, 0xb)
		Gprecalc(3, 7, 11, 15, 0x8, 0x0)
		Gprecalc(0, 5, 10, 15, 0x2, 0xc)
		Gprecalc(1, 6, 11, 12, 0x7, 0xd)
		Gprecalc(2, 7, 8, 13, 0x4, 0x1)
		Gprecalc(3, 4, 9, 14, 0x5, 0xa)

		Gprecalc(0, 4, 8, 12, 0x2, 0xa)
		Gprecalc(1, 5, 9, 13, 0x4, 0x8)
		Gprecalc(2, 6, 10, 14, 0x6, 0x7)
		Gprecalc(3, 7, 11, 15, 0x5, 0x1)
		Gprecalc(0, 5, 10, 15, 0xb, 0xf)
		Gprecalc(1, 6, 11, 12, 0xe, 0x9)
		Gprecalc(2, 7, 8, 13, 0xc, 0x3)
		Gprecalc(3, 4, 9, 14, 0x0, 0xd)
		
		Gprecalc(0, 4, 8, 12, 0x1, 0x0)
		Gprecalc(1, 5, 9, 13, 0x3, 0x2)
		Gprecalc(2, 6, 10, 14, 0x5, 0x4)
		Gprecalc(3, 7, 11, 15, 0x7, 0x6)
		Gprecalc(0, 5, 10, 15, 0x9, 0x8)
		Gprecalc(1, 6, 11, 12, 0xb, 0xa)
		Gprecalc(2, 7, 8, 13, 0xd, 0xc)
		Gprecalc(3, 4, 9, 14, 0xf, 0xe)

		Gprecalc(0, 4, 8, 12, 0xa, 0xe)
		Gprecalc(1, 5, 9, 13, 0x8, 0x4)
		Gprecalc(2, 6, 10, 14, 0xf, 0x9)
		Gprecalc(3, 7, 11, 15, 0x6, 0xd)
		Gprecalc(0, 5, 10, 15, 0xc, 0x1)
		Gprecalc(1, 6, 11, 12, 0x2, 0x0)
		Gprecalc(2, 7, 8, 13, 0x7, 0xb)
		Gprecalc(3, 4, 9, 14, 0x3, 0x5)

		Gprecalc(0, 4, 8, 12, 0x8, 0xb)
		Gprecalc(1, 5, 9, 13, 0x0, 0xc)
		Gprecalc(2, 6, 10, 14, 0x2, 0x5)
		Gprecalc(3, 7, 11, 15, 0xd, 0xf)
		Gprecalc(0, 5, 10, 15, 0xe, 0xa)
		Gprecalc(1, 6, 11, 12, 0x6, 0x3)
		Gprecalc(2, 7, 8, 13, 0x1, 0x7)
		Gprecalc(3, 4, 9, 14, 0x4, 0x9)

		Gprecalc(0, 4, 8, 12, 0x9, 0x7)
		Gprecalc(1, 5, 9, 13, 0x1, 0x3)
		Gprecalc(2, 6, 10, 14, 0xc, 0xd)
		Gprecalc(3, 7, 11, 15, 0xe, 0xb)
		Gprecalc(0, 5, 10, 15, 0x6, 0x2)
		Gprecalc(1, 6, 11, 12, 0xa, 0x5)
		Gprecalc(2, 7, 8, 13, 0x0, 0x4)
		Gprecalc(3, 4, 9, 14, 0x8, 0xf)

		Gprecalc(0, 4, 8, 12, 0x0, 0x9)
		Gprecalc(1, 5, 9, 13, 0x7, 0x5)
		Gprecalc(2, 6, 10, 14, 0x4, 0x2)
		Gprecalc(3, 7, 11, 15, 0xf, 0xa)
		Gprecalc(0, 5, 10, 15, 0x1, 0xe)
		Gprecalc(1, 6, 11, 12, 0xc, 0xb)
		Gprecalc(2, 7, 8, 13, 0x8, 0x6)
		Gprecalc(3, 4, 9, 14, 0xd, 0x3)
		
		Gprecalc(0, 4, 8, 12, 0xc, 0x2)
		Gprecalc(1, 5, 9, 13, 0xa, 0x6)
		Gprecalc(2, 6, 10, 14, 0xb, 0x0)
		Gprecalc(3, 7, 11, 15, 0x3, 0x8)
		Gprecalc(0, 5, 10, 15, 0xd, 0x4)
		Gprecalc(1, 6, 11, 12, 0x5, 0x7)
		Gprecalc(2, 7, 8, 13, 0xe, 0xf)
		Gprecalc(3, 4, 9, 14, 0x9, 0x1)

		uint64_t *outHash = (uint64_t *)outputHash + 8 * thread;
		outHash[0] = devectorizeswap(h[0] ^ v[0] ^ v[8]);
		outHash[1] = devectorizeswap(h[1] ^ v[1] ^ v[9]);
		outHash[2] = devectorizeswap(h[2] ^ v[2] ^ v[10]);
		outHash[3] = devectorizeswap(h[3] ^ v[3] ^ v[11]);
		outHash[4] = devectorizeswap(h[4] ^ v[4] ^ v[12]);
		outHash[5] = devectorizeswap(h[5] ^ v[5] ^ v[13]);
		outHash[6] = devectorizeswap(h[6] ^ v[6] ^ v[14]);
		outHash[7] = devectorizeswap(h[7] ^ v[7] ^ v[15]);
	}
}


// ---------------------------- END CUDA quark_blake512 functions ------------------------------------


// Blake512 für 80 Byte grosse Eingangsdaten
__host__ void quark_blake512_cpu_setBlock_80(void *pdata)
{
	// Message mit Padding bereitstellen
	// lediglich die korrekte Nonce ist noch ab Byte 76 einzusetzen.
	unsigned char PaddedMessage[128];
	memcpy(PaddedMessage, pdata, 80);
	memset(PaddedMessage+80, 0, 48);
	PaddedMessage[80] = 0x80;
	PaddedMessage[111] = 1;
	PaddedMessage[126] = 0x02;
	PaddedMessage[127] = 0x80;
	for (int i = 0; i < 16; i++)
		((uint64_t*)PaddedMessage)[i] = cuda_swab64(((uint64_t*)PaddedMessage)[i]);
	CUDA_SAFE_CALL(
		cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, 16*sizeof(uint64_t), 0, cudaMemcpyHostToDevice)
	);
}


__host__ void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_outputHash, int order)
{
	const uint32_t threadsperblock = 32;
	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
	quark_blake512_gpu_hash_64<<<grid, block>>>(threads, startNounce, d_nonceVector, (uint64_t*)d_outputHash);
//	MyStreamSynchronize(NULL, order, thr_id);
}

__host__ void quark_blake512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash, int order)
{
	const uint32_t threadsperblock = 64;
	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	quark_blake512_gpu_hash_80<<<grid, block>>>(threads, startNounce, d_outputHash);

}
