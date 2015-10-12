/**
 * Blake-256 Cuda Kernel (Tested on SM 5.0)
 *
 * Tanguy Pruvot - Nov. 2014
 */
extern "C" {
#include "sph/sph_blake.h"
}

#include "cuda_helper.h"
#include "cuda_vector.h"
#include <memory.h>

#define UINT2(x,y) make_uint2(x,y)


//static __device__ uint64_t cuda_swab32ll(uint64_t x) {
//	return MAKE_ULONGLONG(cuda_swab32(_LOWORD(x)), cuda_swab32(_HIWORD(x)));
//}

__constant__ static uint32_t  c_data[3];
__constant__ static uint28 Hostprecalc[2];

//__constant__ static uint8_t sigma[16][16];
static uint8_t  c_sigma[16][16] = {
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

static const uint32_t  c_IV256[8] = {
	0x6A09E667, 0xBB67AE85,
	0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C,
	0x1F83D9AB, 0x5BE0CD19
};

__device__ __constant__ static uint32_t cpu_h[8];

//__device__ __constant__ static  uint32_t  u256[16];
static const uint32_t  c_u256[16] = {
	0x243F6A88, 0x85A308D3,
	0x13198A2E, 0x03707344,
	0xA4093822, 0x299F31D0,
	0x082EFA98, 0xEC4E6C89,
	0x452821E6, 0x38D01377,
	0xBE5466CF, 0x34E90C6C,
	0xC0AC29B7, 0xC97C50DD,
	0x3F84D5B5, 0xB5470917
};

#define GS2(a,b,c,d,x) { \
	const uint8_t idx1 = sigma[r][x]; \
	const uint8_t idx2 = sigma[r][x+1]; \
	v[a] += (m[idx1] ^ u256[idx2]) + v[b]; \
	v[d] = __byte_perm(v[d] ^ v[a],0, 0x1032); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 12); \
\
	v[a] += (m[idx2] ^ u256[idx1]) + v[b]; \
	v[d] = __byte_perm(v[d] ^ v[a],0, 0x0321); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 7); \
}

#define hostGS(a,b,c,d,x) { \
	const uint8_t idx1 = c_sigma[r][x]; \
	const uint8_t idx2 = c_sigma[r][x+1]; \
	v[a] += (m[idx1] ^ c_u256[idx2]) + v[b]; \
	v[d] = ROTR32(v[d] ^ v[a], 16); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 12); \
\
	v[a] += (m[idx2] ^ c_u256[idx1]) + v[b]; \
	v[d] = ROTR32(v[d] ^ v[a], 8); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 7); \
	}

#define GSPREC(a,b,c,d,x,y) { \
	v[a] += (m[x] ^ u256[y]) + v[b]; \
	v[d] = __byte_perm(v[d] ^ v[a],0, 0x1032); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 12); \
	v[a] += (m[y] ^ u256[x]) + v[b]; \
	v[d] = __byte_perm(v[d] ^ v[a],0, 0x0321); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 7); \
		}

#define GSPRECHOST(a,b,c,d,x,y) { \
	v[a] += (m2[x] ^ u256[y]) + v[b]; \
	v[d] = SPH_ROTR32(v[d] ^ v[a],16); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 12); \
	v[a] += (m2[y] ^ u256[x]) + v[b]; \
	v[d] = SPH_ROTR32(v[d] ^ v[a], 8); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 7); \
			}



__constant__ uint64_t keccak_round_constants[24] = {
	0x0000000000000001ull, 0x0000000000008082ull,
	0x800000000000808aull, 0x8000000080008000ull,
	0x000000000000808bull, 0x0000000080000001ull,
	0x8000000080008081ull, 0x8000000000008009ull,
	0x000000000000008aull, 0x0000000000000088ull,
	0x0000000080008009ull, 0x000000008000000aull,
	0x000000008000808bull, 0x800000000000008bull,
	0x8000000000008089ull, 0x8000000000008003ull,
	0x8000000000008002ull, 0x8000000000000080ull,
	0x000000000000800aull, 0x800000008000000aull,
	0x8000000080008081ull, 0x8000000000008080ull,
	0x0000000080000001ull, 0x8000000080008008ull
};

__constant__ uint2 keccak_round_constants35[24] = {
	{ 0x00000001ul, 0x00000000 }, { 0x00008082ul, 0x00000000 },
	{ 0x0000808aul, 0x80000000 }, { 0x80008000ul, 0x80000000 },
	{ 0x0000808bul, 0x00000000 }, { 0x80000001ul, 0x00000000 },
	{ 0x80008081ul, 0x80000000 }, { 0x00008009ul, 0x80000000 },
	{ 0x0000008aul, 0x00000000 }, { 0x00000088ul, 0x00000000 },
	{ 0x80008009ul, 0x00000000 }, { 0x8000000aul, 0x00000000 },
	{ 0x8000808bul, 0x00000000 }, { 0x0000008bul, 0x80000000 },
	{ 0x00008089ul, 0x80000000 }, { 0x00008003ul, 0x80000000 },
	{ 0x00008002ul, 0x80000000 }, { 0x00000080ul, 0x80000000 },
	{ 0x0000800aul, 0x00000000 }, { 0x8000000aul, 0x80000000 },
	{ 0x80008081ul, 0x80000000 }, { 0x00008080ul, 0x80000000 },
	{ 0x80000001ul, 0x00000000 }, { 0x80008008ul, 0x80000000 }
};

__host__ __forceinline__
static void blake256_compress1st(uint32_t *h, const uint32_t *block, const uint32_t T0)
{
	uint32_t m[16];
	uint32_t v[16];
	
	for (int i = 0; i < 16; i++) {
		m[i] = block[i];
	}

	for (int i = 0; i < 8; i++)
		v[i] = h[i];

	v[8] = c_u256[0];
	v[9] = c_u256[1];
	v[10] = c_u256[2];
	v[11] = c_u256[3];

	v[12] = c_u256[4] ^ T0;
	v[13] = c_u256[5] ^ T0;
	v[14] = c_u256[6];
	v[15] = c_u256[7];

	for (int r = 0; r < 14; r++) {
		/* column step */
		hostGS(0, 4, 0x8, 0xC, 0x0);
		hostGS(1, 5, 0x9, 0xD, 0x2);
		hostGS(2, 6, 0xA, 0xE, 0x4);
		hostGS(3, 7, 0xB, 0xF, 0x6);
		/* diagonal step */
		hostGS(0, 5, 0xA, 0xF, 0x8);
		hostGS(1, 6, 0xB, 0xC, 0xA);
		hostGS(2, 7, 0x8, 0xD, 0xC);
		hostGS(3, 4, 0x9, 0xE, 0xE);
	}

	h[0] ^= v[0] ^ v[8];
	h[1] ^= v[1] ^ v[9];
	h[2] ^= v[2] ^ v[10];
	h[3] ^= v[3] ^ v[11];
	h[4] ^= v[4] ^ v[12];
	h[5] ^= v[5] ^ v[13];
	h[6] ^= v[6] ^ v[14];
	h[7] ^= v[7] ^ v[15];



	const uint32_t T02 = 640;
//	uint32_t v[16];

	const uint32_t c_Padding[12] = {
		0x80000000, 0, 0, 0,
		0, 0, 0, 0,
		0, 1, 0, 640
	};

	const uint32_t  u256[16] =
	{
		0x243F6A88, 0x85A308D3,
		0x13198A2E, 0x03707344,
		0xA4093822, 0x299F31D0,
		0x082EFA98, 0xEC4E6C89,
		0x452821E6, 0x38D01377,
		0xBE5466CF, 0x34E90C6C,
		0xC0AC29B7, 0xC97C50DD,
		0x3F84D5B5, 0xB5470917
	};

	uint32_t m2[16] =
	{
		block[16], block[17], block[18], 0,
		c_Padding[0], c_Padding[1], c_Padding[2], c_Padding[3],
		c_Padding[4], c_Padding[5], c_Padding[6], c_Padding[7],
		c_Padding[8], c_Padding[9], c_Padding[10], c_Padding[11]
	};

	for (int i = 0; i < 8; i++)
		v[i] = h[i];

	v[8] = u256[0];
	v[9] = u256[1];
	v[10] = u256[2];
	v[11] = u256[3];
	v[12] = u256[4] ^ T02;
	v[13] = u256[5] ^ T02;
	v[14] = u256[6];
	v[15] = u256[7];

	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },

	GSPRECHOST(0, 4, 0x8, 0xC, 0, 1);


//	GSPRECHOST(1, 5, 0x9, 0xD, 2, 3);
	GSPRECHOST(2, 6, 0xA, 0xE, 4, 5);
	GSPRECHOST(3, 7, 0xB, 0xF, 6, 7);

	v[0] += (m2[8] ^ u256[9]);
	v[2] += (m2[12] ^ u256[13]) + v[7];

	v[3] += (m2[14] ^ u256[15]) + v[4];
	v[0xe] = ROTR32(v[0xe] ^ v[3], 16);

	cudaMemcpyToSymbol(Hostprecalc,v, sizeof(v), 0, cudaMemcpyHostToDevice);

}
#define bitselect(a, b, c) ((a) ^ ((c) & ((b) ^ (a))))

__global__
void blakeKeccak256_gpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint32_t * Hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;
		uint32_t h[8];
#pragma unroll 8
		for (int i = 0; i < 8; i++) { h[i] = cpu_h[i]; }

		uint32_t v[16];

		const uint32_t c_Padding[12] = {
			0x80000000, 0, 0, 0,
			0, 0, 0, 0,
			0, 1, 0, 640
		};

		const uint32_t  u256[16] =
		{
			0x243F6A88, 0x85A308D3,
			0x13198A2E, 0x03707344,
			0xA4093822, 0x299F31D0,
			0x082EFA98, 0xEC4E6C89,
			0x452821E6, 0x38D01377,
			0xBE5466CF, 0x34E90C6C,
			0xC0AC29B7, 0xC97C50DD,
			0x3F84D5B5, 0xB5470917
		};

		uint32_t m[16] =
		{
			c_data[0], c_data[1], c_data[2], nonce,
			c_Padding[0], c_Padding[1], c_Padding[2], c_Padding[3],
			c_Padding[4], c_Padding[5], c_Padding[6], c_Padding[7],
			c_Padding[8], c_Padding[9], c_Padding[10], c_Padding[11]
		};
	
		uint28 *outpt = (uint28*)v;
		outpt[0] = Hostprecalc[0];
		outpt[1] = Hostprecalc[1];
	

		//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },

//		GSPREC(0, 4, 0x8, 0xC, 0, 1);
		GSPREC(1, 5, 0x9, 0xD, 2, 3);
//		GSPREC(2, 6, 0xA, 0xE, 4, 5);
//		GSPREC(3, 7, 0xB, 0xF, 6, 7);

//		GSPREC(0, 5, 0xA, 0xF, 8, 9);
		v[0] += v[5];
		v[0xf] = __byte_perm(v[0xf] ^ v[0],0, 0x1032); 
		v[0xa] += v[0xf]; 
		v[5] = SPH_ROTR32(v[5] ^ v[0xa], 12); 
		v[0] += (m[9] ^ u256[8]) + v[5]; 
		v[0xf] = __byte_perm(v[0xf] ^ v[0],0, 0x0321); 
		v[0xa] += v[0xf]; 
		v[5] = SPH_ROTR32(v[5] ^ v[0xa], 7); 

		GSPREC(1, 6, 0xB, 0xC, 10, 11);
		
//		GSPREC(2, 7, 0x8, 0xD, 12, 13);
		v[0xD] = __byte_perm(v[0xD] ^ v[2],0, 0x1032);
		v[0x8] += v[0xD];
		v[7] = SPH_ROTR32(v[7] ^ v[8], 12);
		v[2] += (m[13] ^ u256[12]) + v[7];
		v[0xD] = __byte_perm(v[0xD] ^ v[2],0, 0x0321);
		v[0x8] += v[0xD]; 
		v[7] = SPH_ROTR32(v[7] ^ v[8], 7);

	//	GSPREC(3, 4, 0x9, 0xE, 14, 15);
		v[0x9] += v[0xe];
		v[4] = SPH_ROTR32(v[4] ^ v[9], 12);
		v[3] += (m[15] ^ u256[14]) + v[4];
		v[0xe] = __byte_perm(v[0xe] ^ v[3],0, 0x0321);
		v[0x9] += v[0xe];
		v[4] = SPH_ROTR32(v[4] ^ v[0x9], 7);

		//	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
		GSPREC(0, 4, 0x8, 0xC, 14, 10);
		GSPREC(1, 5, 0x9, 0xD, 4, 8);
		GSPREC(2, 6, 0xA, 0xE, 9, 15);
		GSPREC(3, 7, 0xB, 0xF, 13, 6);
		GSPREC(0, 5, 0xA, 0xF, 1, 12);
		GSPREC(1, 6, 0xB, 0xC, 0, 2);
		GSPREC(2, 7, 0x8, 0xD, 11, 7);
		GSPREC(3, 4, 0x9, 0xE, 5, 3);
		//	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
		GSPREC(0, 4, 0x8, 0xC, 11, 8);
		GSPREC(1, 5, 0x9, 0xD, 12, 0);
		GSPREC(2, 6, 0xA, 0xE, 5, 2);
		GSPREC(3, 7, 0xB, 0xF, 15, 13);
		GSPREC(0, 5, 0xA, 0xF, 10, 14);
		GSPREC(1, 6, 0xB, 0xC, 3, 6);
		GSPREC(2, 7, 0x8, 0xD, 7, 1);
		GSPREC(3, 4, 0x9, 0xE, 9, 4);
		//	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
		GSPREC(0, 4, 0x8, 0xC, 7, 9);
		GSPREC(1, 5, 0x9, 0xD, 3, 1);
		GSPREC(2, 6, 0xA, 0xE, 13, 12);
		GSPREC(3, 7, 0xB, 0xF, 11, 14);
		GSPREC(0, 5, 0xA, 0xF, 2, 6);
		GSPREC(1, 6, 0xB, 0xC, 5, 10);
		GSPREC(2, 7, 0x8, 0xD, 4, 0);
		GSPREC(3, 4, 0x9, 0xE, 15, 8);

		//	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
		GSPREC(0, 4, 0x8, 0xC, 9, 0);
		GSPREC(1, 5, 0x9, 0xD, 5, 7);
		GSPREC(2, 6, 0xA, 0xE, 2, 4);
		GSPREC(3, 7, 0xB, 0xF, 10, 15);
		GSPREC(0, 5, 0xA, 0xF, 14, 1);
		GSPREC(1, 6, 0xB, 0xC, 11, 12);
		GSPREC(2, 7, 0x8, 0xD, 6, 8);
		GSPREC(3, 4, 0x9, 0xE, 3, 13);
		//	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
		GSPREC(0, 4, 0x8, 0xC, 2, 12);
		GSPREC(1, 5, 0x9, 0xD, 6, 10);
		GSPREC(2, 6, 0xA, 0xE, 0, 11);
		GSPREC(3, 7, 0xB, 0xF, 8, 3);
		GSPREC(0, 5, 0xA, 0xF, 4, 13);
		GSPREC(1, 6, 0xB, 0xC, 7, 5);
		GSPREC(2, 7, 0x8, 0xD, 15, 14);
		GSPREC(3, 4, 0x9, 0xE, 1, 9);

		//	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
		GSPREC(0, 4, 0x8, 0xC, 12, 5);
		GSPREC(1, 5, 0x9, 0xD, 1, 15);
		GSPREC(2, 6, 0xA, 0xE, 14, 13);
		GSPREC(3, 7, 0xB, 0xF, 4, 10);
		GSPREC(0, 5, 0xA, 0xF, 0, 7);
		GSPREC(1, 6, 0xB, 0xC, 6, 3);
		GSPREC(2, 7, 0x8, 0xD, 9, 2);
		GSPREC(3, 4, 0x9, 0xE, 8, 11);

		//	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
		GSPREC(0, 4, 0x8, 0xC, 13, 11);
		GSPREC(1, 5, 0x9, 0xD, 7, 14);
		GSPREC(2, 6, 0xA, 0xE, 12, 1);
		GSPREC(3, 7, 0xB, 0xF, 3, 9);
		GSPREC(0, 5, 0xA, 0xF, 5, 0);
		GSPREC(1, 6, 0xB, 0xC, 15, 4);
		GSPREC(2, 7, 0x8, 0xD, 8, 6);
		GSPREC(3, 4, 0x9, 0xE, 2, 10);
		//	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
		GSPREC(0, 4, 0x8, 0xC, 6, 15);
		GSPREC(1, 5, 0x9, 0xD, 14, 9);
		GSPREC(2, 6, 0xA, 0xE, 11, 3);
		GSPREC(3, 7, 0xB, 0xF, 0, 8);
		GSPREC(0, 5, 0xA, 0xF, 12, 2);
		GSPREC(1, 6, 0xB, 0xC, 13, 7);
		GSPREC(2, 7, 0x8, 0xD, 1, 4);
		GSPREC(3, 4, 0x9, 0xE, 10, 5);
		//	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
		GSPREC(0, 4, 0x8, 0xC, 10, 2);
		GSPREC(1, 5, 0x9, 0xD, 8, 4);
		GSPREC(2, 6, 0xA, 0xE, 7, 6);
		GSPREC(3, 7, 0xB, 0xF, 1, 5);
		GSPREC(0, 5, 0xA, 0xF, 15, 11);
		GSPREC(1, 6, 0xB, 0xC, 9, 14);
		GSPREC(2, 7, 0x8, 0xD, 3, 12);
		GSPREC(3, 4, 0x9, 0xE, 13, 0);
		//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
		GSPREC(0, 4, 0x8, 0xC, 0, 1);
		GSPREC(1, 5, 0x9, 0xD, 2, 3);
		GSPREC(2, 6, 0xA, 0xE, 4, 5);
		GSPREC(3, 7, 0xB, 0xF, 6, 7);
		GSPREC(0, 5, 0xA, 0xF, 8, 9);
		GSPREC(1, 6, 0xB, 0xC, 10, 11);
		GSPREC(2, 7, 0x8, 0xD, 12, 13);
		GSPREC(3, 4, 0x9, 0xE, 14, 15);

		//	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
		GSPREC(0, 4, 0x8, 0xC, 14, 10);
		GSPREC(1, 5, 0x9, 0xD, 4, 8);
		GSPREC(2, 6, 0xA, 0xE, 9, 15);
		GSPREC(3, 7, 0xB, 0xF, 13, 6);
		GSPREC(0, 5, 0xA, 0xF, 1, 12);
		GSPREC(1, 6, 0xB, 0xC, 0, 2);
		GSPREC(2, 7, 0x8, 0xD, 11, 7);
		GSPREC(3, 4, 0x9, 0xE, 5, 3);

		//	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
		GSPREC(0, 4, 0x8, 0xC, 11, 8);
		GSPREC(1, 5, 0x9, 0xD, 12, 0);
		GSPREC(2, 6, 0xA, 0xE, 5, 2);
		GSPREC(3, 7, 0xB, 0xF, 15, 13);
		GSPREC(0, 5, 0xA, 0xF, 10, 14);
		GSPREC(1, 6, 0xB, 0xC, 3, 6);
		GSPREC(2, 7, 0x8, 0xD, 7, 1);
		GSPREC(3, 4, 0x9, 0xE, 9, 4);
		//	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
		GSPREC(0, 4, 0x8, 0xC, 7, 9);
		GSPREC(1, 5, 0x9, 0xD, 3, 1);
		GSPREC(2, 6, 0xA, 0xE, 13, 12);
		GSPREC(3, 7, 0xB, 0xF, 11, 14);
		GSPREC(0, 5, 0xA, 0xF, 2, 6);
		GSPREC(1, 6, 0xB, 0xC, 5, 10);
		GSPREC(2, 7, 0x8, 0xD, 4, 0);
		GSPREC(3, 4, 0x9, 0xE, 15, 8);



		uint2 s[25] = {0};
		s[0].x = cuda_swab32(h[0] ^ v[0] ^ v[8]);
		s[0].y = cuda_swab32(h[1] ^ v[1] ^ v[9]);
		s[1].x = cuda_swab32(h[2] ^ v[2] ^ v[10]);
		s[1].y = cuda_swab32(h[3] ^ v[3] ^ v[11]);
		s[2].x = cuda_swab32(h[4] ^ v[4] ^ v[12]);
		s[2].y = cuda_swab32(h[5] ^ v[5] ^ v[13]);
		s[3].x = cuda_swab32(h[6] ^ v[6] ^ v[14]);
		s[3].y = cuda_swab32(h[7] ^ v[7] ^ v[15]);
		s[4] = vectorizelow(1);

		s[16] = vectorizehigh(0x80000000);
		uint2 bc[5], tmpxor[5], tmp1, tmp2;
		//	uint2 s[25];

//#pragma unroll
//		for (uint32_t x = 0; x < 5; x++)
//			tmpxor[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20];

		tmpxor[0] = s[0];
		tmpxor[1] = s[1] ^ s[16];
		tmpxor[2] = s[2];
		tmpxor[3] = s[3];
		tmpxor[4] = s[4];


		bc[0] = tmpxor[0] ^ ROL2(tmpxor[2], 1);
		bc[1] = tmpxor[1] ^ ROL2(tmpxor[3], 1);
		bc[2] = tmpxor[2] ^ ROL2(tmpxor[4], 1);
		bc[3] = tmpxor[3] ^ ROL2(tmpxor[0], 1);
		bc[4] = tmpxor[4] ^ ROL2(tmpxor[1], 1);

		tmp1 = s[1] ^ bc[0];

		s[0] ^= bc[4];
		s[1] = ROL2(s[6] ^ bc[0], 44);
		s[6] = ROL2(s[9] ^ bc[3], 20);
		s[9] = ROL2(s[22] ^ bc[1], 61);
		s[22] = ROL2(s[14] ^ bc[3], 39);
		s[14] = ROL2(s[20] ^ bc[4], 18);
		s[20] = ROL2(s[2] ^ bc[1], 62);
		s[2] = ROL2(s[12] ^ bc[1], 43);
		s[12] = ROL2(s[13] ^ bc[2], 25);
		s[13] = ROL8(s[19] ^ bc[3]);
		s[19] = ROR8(s[23] ^ bc[2]);
		s[23] = ROL2(s[15] ^ bc[4], 41);
		s[15] = ROL2(s[4] ^ bc[3], 27);
		s[4] = ROL2(s[24] ^ bc[3], 14);
		s[24] = ROL2(s[21] ^ bc[0], 2);
		s[21] = ROL2(s[8] ^ bc[2], 55);
		s[8] = ROL2(s[16] ^ bc[0], 45);
		s[16] = ROL2(s[5] ^ bc[4], 36);
		s[5] = ROL2(s[3] ^ bc[2], 28);
		s[3] = ROL2(s[18] ^ bc[2], 21);
		s[18] = ROL2(s[17] ^ bc[1], 15);
		s[17] = ROL2(s[11] ^ bc[0], 10);
		s[11] = ROL2(s[7] ^ bc[1], 6);
		s[7] = ROL2(s[10] ^ bc[4], 3);
		s[10] = ROL2(tmp1, 1);

		tmp1 = s[0]; tmp2 = s[1]; s[0] = bitselect(s[0] ^ s[2], s[0], s[1]); s[1] = bitselect(s[1] ^ s[3], s[1], s[2]); s[2] = bitselect(s[2] ^ s[4], s[2], s[3]); s[3] = bitselect(s[3] ^ tmp1, s[3], s[4]); s[4] = bitselect(s[4] ^ tmp2, s[4], tmp1);
		s[0].x ^= 1;
		tmp1 = s[5]; tmp2 = s[6]; s[5] = bitselect(s[5] ^ s[7], s[5], s[6]); s[6] = bitselect(s[6] ^ s[8], s[6], s[7]); s[7] = bitselect(s[7] ^ s[9], s[7], s[8]); s[8] = bitselect(s[8] ^ tmp1, s[8], s[9]); s[9] = bitselect(s[9] ^ tmp2, s[9], tmp1);
		tmp1 = s[10]; tmp2 = s[11]; s[10] = bitselect(s[10] ^ s[12], s[10], s[11]); s[11] = bitselect(s[11] ^ s[13], s[11], s[12]); s[12] = bitselect(s[12] ^ s[14], s[12], s[13]); s[13] = bitselect(s[13] ^ tmp1, s[13], s[14]); s[14] = bitselect(s[14] ^ tmp2, s[14], tmp1);
		tmp1 = s[15]; tmp2 = s[16]; s[15] = bitselect(s[15] ^ s[17], s[15], s[16]); s[16] = bitselect(s[16] ^ s[18], s[16], s[17]); s[17] = bitselect(s[17] ^ s[19], s[17], s[18]); s[18] = bitselect(s[18] ^ tmp1, s[18], s[19]); s[19] = bitselect(s[19] ^ tmp2, s[19], tmp1);
		tmp1 = s[20]; tmp2 = s[21]; s[20] = bitselect(s[20] ^ s[22], s[20], s[21]); s[21] = bitselect(s[21] ^ s[23], s[21], s[22]); s[22] = bitselect(s[22] ^ s[24], s[22], s[23]); s[23] = bitselect(s[23] ^ tmp1, s[23], s[24]); s[24] = bitselect(s[24] ^ tmp2, s[24], tmp1);


#pragma unroll 1
		for (int i = 1; i < 24; i++)
		{
			#pragma unroll
			for (uint32_t x = 0; x < 5; x++)
				tmpxor[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20];

			bc[0] = tmpxor[0] ^ ROL2(tmpxor[2], 1);
			bc[1] = tmpxor[1] ^ ROL2(tmpxor[3], 1);
			bc[2] = tmpxor[2] ^ ROL2(tmpxor[4], 1);
			bc[3] = tmpxor[3] ^ ROL2(tmpxor[0], 1);
			bc[4] = tmpxor[4] ^ ROL2(tmpxor[1], 1);

			tmp1 = s[1] ^ bc[0];

			s[0] ^= bc[4];
			s[1] = ROL2(s[6] ^ bc[0], 44);
			s[6] = ROL2(s[9] ^ bc[3], 20);
			s[9] = ROL2(s[22] ^ bc[1], 61);
			s[22] = ROL2(s[14] ^ bc[3], 39);
			s[14] = ROL2(s[20] ^ bc[4], 18);
			s[20] = ROL2(s[2] ^ bc[1], 62);
			s[2] = ROL2(s[12] ^ bc[1], 43);
			s[12] = ROL2(s[13] ^ bc[2], 25);
			s[13] = ROL8(s[19] ^ bc[3]);
			s[19] = ROR8(s[23] ^ bc[2]);
			s[23] = ROL2(s[15] ^ bc[4], 41);
			s[15] = ROL2(s[4] ^ bc[3], 27);
			s[4] = ROL2(s[24] ^ bc[3], 14);
			s[24] = ROL2(s[21] ^ bc[0], 2);
			s[21] = ROL2(s[8] ^ bc[2], 55);
			s[8] = ROL2(s[16] ^ bc[0], 45);
			s[16] = ROL2(s[5] ^ bc[4], 36);
			s[5] = ROL2(s[3] ^ bc[2], 28);
			s[3] = ROL2(s[18] ^ bc[2], 21);
			s[18] = ROL2(s[17] ^ bc[1], 15);
			s[17] = ROL2(s[11] ^ bc[0], 10);
			s[11] = ROL2(s[7] ^ bc[1], 6);
			s[7] = ROL2(s[10] ^ bc[4], 3);
			s[10] = ROL2(tmp1, 1);

			tmp1 = s[0]; tmp2 = s[1]; s[0] = bitselect(s[0] ^ s[2], s[0], s[1]); s[1] = bitselect(s[1] ^ s[3], s[1], s[2]); s[2] = bitselect(s[2] ^ s[4], s[2], s[3]); s[3] = bitselect(s[3] ^ tmp1, s[3], s[4]); s[4] = bitselect(s[4] ^ tmp2, s[4], tmp1);
			s[0] ^= keccak_round_constants35[i];
			tmp1 = s[5]; tmp2 = s[6]; s[5] = bitselect(s[5] ^ s[7], s[5], s[6]); s[6] = bitselect(s[6] ^ s[8], s[6], s[7]); s[7] = bitselect(s[7] ^ s[9], s[7], s[8]); s[8] = bitselect(s[8] ^ tmp1, s[8], s[9]); s[9] = bitselect(s[9] ^ tmp2, s[9], tmp1);
			tmp1 = s[10]; tmp2 = s[11]; s[10] = bitselect(s[10] ^ s[12], s[10], s[11]); s[11] = bitselect(s[11] ^ s[13], s[11], s[12]); s[12] = bitselect(s[12] ^ s[14], s[12], s[13]); s[13] = bitselect(s[13] ^ tmp1, s[13], s[14]); s[14] = bitselect(s[14] ^ tmp2, s[14], tmp1);
			tmp1 = s[15]; tmp2 = s[16]; s[15] = bitselect(s[15] ^ s[17], s[15], s[16]); s[16] = bitselect(s[16] ^ s[18], s[16], s[17]); s[17] = bitselect(s[17] ^ s[19], s[17], s[18]); s[18] = bitselect(s[18] ^ tmp1, s[18], s[19]); s[19] = bitselect(s[19] ^ tmp2, s[19], tmp1);
			tmp1 = s[20]; tmp2 = s[21]; s[20] = bitselect(s[20] ^ s[22], s[20], s[21]); s[21] = bitselect(s[21] ^ s[23], s[21], s[22]); s[22] = bitselect(s[22] ^ s[24], s[22], s[23]); s[23] = bitselect(s[23] ^ tmp1, s[23], s[24]); s[24] = bitselect(s[24] ^ tmp2, s[24], tmp1);
		}

		uint64_t *outputHash = (uint64_t *)Hash;
#pragma unroll 4
		for (int i = 0; i<4; i++)
			outputHash[i*threads + thread] = devectorize(s[i]);
	}



}


__global__ 
void blake256_gpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint32_t * Hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;
		uint32_t h[8];
		//		uint32_t input[4];
		const uint32_t T0 = 640;
#pragma unroll 8
		for (int i = 0; i<8; i++) { h[i] = cpu_h[i]; }

		uint32_t v[16];

		const uint32_t c_Padding[12] = {
			0x80000000, 0, 0, 0,
			0, 0, 0, 0,
			0, 1, 0, 640
		};

		const uint32_t  u256[16] =
		{
			0x243F6A88, 0x85A308D3,
			0x13198A2E, 0x03707344,
			0xA4093822, 0x299F31D0,
			0x082EFA98, 0xEC4E6C89,
			0x452821E6, 0x38D01377,
			0xBE5466CF, 0x34E90C6C,
			0xC0AC29B7, 0xC97C50DD,
			0x3F84D5B5, 0xB5470917
		};

		uint32_t m[16] =
		{
			c_data[0], c_data[1], c_data[ 2], nonce,
			c_Padding[0], c_Padding[1], c_Padding[2], c_Padding[3],
			c_Padding[4], c_Padding[5], c_Padding[6], c_Padding[7],
			c_Padding[8], c_Padding[9], c_Padding[10], c_Padding[11]
		};

#pragma unroll 8
		for (int i = 0; i < 8; i++)
			v[i] = h[i];

		v[8] = u256[0];
		v[9] = u256[1];
		v[10] = u256[2];
		v[11] = u256[3];
		v[12] = u256[4] ^ T0;
		v[13] = u256[5] ^ T0;
		v[14] = u256[6];
		v[15] = u256[7];

		//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
		GSPREC(0, 4, 0x8, 0xC, 0, 1);
		GSPREC(1, 5, 0x9, 0xD, 2, 3);
		GSPREC(2, 6, 0xA, 0xE, 4, 5);
		GSPREC(3, 7, 0xB, 0xF, 6, 7);

		GSPREC(0, 5, 0xA, 0xF, 8, 9);
		GSPREC(1, 6, 0xB, 0xC, 10, 11);
		GSPREC(2, 7, 0x8, 0xD, 12, 13);
		GSPREC(3, 4, 0x9, 0xE, 14, 15);
		//	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
		GSPREC(0, 4, 0x8, 0xC, 14, 10);
		GSPREC(1, 5, 0x9, 0xD, 4, 8);
		GSPREC(2, 6, 0xA, 0xE, 9, 15);
		GSPREC(3, 7, 0xB, 0xF, 13, 6);
		GSPREC(0, 5, 0xA, 0xF, 1, 12);
		GSPREC(1, 6, 0xB, 0xC, 0, 2);
		GSPREC(2, 7, 0x8, 0xD, 11, 7);
		GSPREC(3, 4, 0x9, 0xE, 5, 3);
		//	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
		GSPREC(0, 4, 0x8, 0xC, 11, 8);
		GSPREC(1, 5, 0x9, 0xD, 12, 0);
		GSPREC(2, 6, 0xA, 0xE, 5, 2);
		GSPREC(3, 7, 0xB, 0xF, 15, 13);
		GSPREC(0, 5, 0xA, 0xF, 10, 14);
		GSPREC(1, 6, 0xB, 0xC, 3, 6);
		GSPREC(2, 7, 0x8, 0xD, 7, 1);
		GSPREC(3, 4, 0x9, 0xE, 9, 4);
		//	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
		GSPREC(0, 4, 0x8, 0xC, 7, 9);
		GSPREC(1, 5, 0x9, 0xD, 3, 1);
		GSPREC(2, 6, 0xA, 0xE, 13, 12);
		GSPREC(3, 7, 0xB, 0xF, 11, 14);
		GSPREC(0, 5, 0xA, 0xF, 2, 6);
		GSPREC(1, 6, 0xB, 0xC, 5, 10);
		GSPREC(2, 7, 0x8, 0xD, 4, 0);
		GSPREC(3, 4, 0x9, 0xE, 15, 8);

		//	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
		GSPREC(0, 4, 0x8, 0xC, 9, 0);
		GSPREC(1, 5, 0x9, 0xD, 5, 7);
		GSPREC(2, 6, 0xA, 0xE, 2, 4);
		GSPREC(3, 7, 0xB, 0xF, 10, 15);
		GSPREC(0, 5, 0xA, 0xF, 14, 1);
		GSPREC(1, 6, 0xB, 0xC, 11, 12);
		GSPREC(2, 7, 0x8, 0xD, 6, 8);
		GSPREC(3, 4, 0x9, 0xE, 3, 13);
		//	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
		GSPREC(0, 4, 0x8, 0xC, 2, 12);
		GSPREC(1, 5, 0x9, 0xD, 6, 10);
		GSPREC(2, 6, 0xA, 0xE, 0, 11);
		GSPREC(3, 7, 0xB, 0xF, 8, 3);
		GSPREC(0, 5, 0xA, 0xF, 4, 13);
		GSPREC(1, 6, 0xB, 0xC, 7, 5);
		GSPREC(2, 7, 0x8, 0xD, 15, 14);
		GSPREC(3, 4, 0x9, 0xE, 1, 9);

		//	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
		GSPREC(0, 4, 0x8, 0xC, 12, 5);
		GSPREC(1, 5, 0x9, 0xD, 1, 15);
		GSPREC(2, 6, 0xA, 0xE, 14, 13);
		GSPREC(3, 7, 0xB, 0xF, 4, 10);
		GSPREC(0, 5, 0xA, 0xF, 0, 7);
		GSPREC(1, 6, 0xB, 0xC, 6, 3);
		GSPREC(2, 7, 0x8, 0xD, 9, 2);
		GSPREC(3, 4, 0x9, 0xE, 8, 11);

		//	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
		GSPREC(0, 4, 0x8, 0xC, 13, 11);
		GSPREC(1, 5, 0x9, 0xD, 7, 14);
		GSPREC(2, 6, 0xA, 0xE, 12, 1);
		GSPREC(3, 7, 0xB, 0xF, 3, 9);
		GSPREC(0, 5, 0xA, 0xF, 5, 0);
		GSPREC(1, 6, 0xB, 0xC, 15, 4);
		GSPREC(2, 7, 0x8, 0xD, 8, 6);
		GSPREC(3, 4, 0x9, 0xE, 2, 10);
		//	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
		GSPREC(0, 4, 0x8, 0xC, 6, 15);
		GSPREC(1, 5, 0x9, 0xD, 14, 9);
		GSPREC(2, 6, 0xA, 0xE, 11, 3);
		GSPREC(3, 7, 0xB, 0xF, 0, 8);
		GSPREC(0, 5, 0xA, 0xF, 12, 2);
		GSPREC(1, 6, 0xB, 0xC, 13, 7);
		GSPREC(2, 7, 0x8, 0xD, 1, 4);
		GSPREC(3, 4, 0x9, 0xE, 10, 5);
		//	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
		GSPREC(0, 4, 0x8, 0xC, 10, 2);
		GSPREC(1, 5, 0x9, 0xD, 8, 4);
		GSPREC(2, 6, 0xA, 0xE, 7, 6);
		GSPREC(3, 7, 0xB, 0xF, 1, 5);
		GSPREC(0, 5, 0xA, 0xF, 15, 11);
		GSPREC(1, 6, 0xB, 0xC, 9, 14);
		GSPREC(2, 7, 0x8, 0xD, 3, 12);
		GSPREC(3, 4, 0x9, 0xE, 13, 0);
		//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
		GSPREC(0, 4, 0x8, 0xC, 0, 1);
		GSPREC(1, 5, 0x9, 0xD, 2, 3);
		GSPREC(2, 6, 0xA, 0xE, 4, 5);
		GSPREC(3, 7, 0xB, 0xF, 6, 7);
		GSPREC(0, 5, 0xA, 0xF, 8, 9);
		GSPREC(1, 6, 0xB, 0xC, 10, 11);
		GSPREC(2, 7, 0x8, 0xD, 12, 13);
		GSPREC(3, 4, 0x9, 0xE, 14, 15);

		//	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
		GSPREC(0, 4, 0x8, 0xC, 14, 10);
		GSPREC(1, 5, 0x9, 0xD, 4, 8);
		GSPREC(2, 6, 0xA, 0xE, 9, 15);
		GSPREC(3, 7, 0xB, 0xF, 13, 6);
		GSPREC(0, 5, 0xA, 0xF, 1, 12);
		GSPREC(1, 6, 0xB, 0xC, 0, 2);
		GSPREC(2, 7, 0x8, 0xD, 11, 7);
		GSPREC(3, 4, 0x9, 0xE, 5, 3);

		//	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
		GSPREC(0, 4, 0x8, 0xC, 11, 8);
		GSPREC(1, 5, 0x9, 0xD, 12, 0);
		GSPREC(2, 6, 0xA, 0xE, 5, 2);
		GSPREC(3, 7, 0xB, 0xF, 15, 13);
		GSPREC(0, 5, 0xA, 0xF, 10, 14);
		GSPREC(1, 6, 0xB, 0xC, 3, 6);
		GSPREC(2, 7, 0x8, 0xD, 7, 1);
		GSPREC(3, 4, 0x9, 0xE, 9, 4);
		//	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
		GSPREC(0, 4, 0x8, 0xC, 7, 9);
		GSPREC(1, 5, 0x9, 0xD, 3, 1);
		GSPREC(2, 6, 0xA, 0xE, 13, 12);
		GSPREC(3, 7, 0xB, 0xF, 11, 14);
		GSPREC(0, 5, 0xA, 0xF, 2, 6);
		GSPREC(1, 6, 0xB, 0xC, 5, 10);
		GSPREC(2, 7, 0x8, 0xD, 4, 0);
		GSPREC(3, 4, 0x9, 0xE, 15, 8);

		h[0] = cuda_swab32(h[0] ^ v[0] ^ v[8]);
		h[1] = cuda_swab32(h[1] ^ v[1] ^ v[9]);
		h[2] = cuda_swab32(h[2] ^ v[2] ^ v[10]);
		h[3] = cuda_swab32(h[3] ^ v[3] ^ v[11]);
		h[4] = cuda_swab32(h[4] ^ v[4] ^ v[12]);
		h[5] = cuda_swab32(h[5] ^ v[5] ^ v[13]);
		h[6] = cuda_swab32(h[6] ^ v[6] ^ v[14]);
		h[7] = cuda_swab32(h[7] ^ v[7] ^ v[15]);

		Hash[((0 * threads) + thread)*2] = (h[0]);
		Hash[((0 * threads) + thread) * 2 + 1] = (h[1]);
		Hash[((1 * threads) + thread) * 2] = (h[2]);
		Hash[((1 * threads) + thread) * 2 + 1] = (h[3]);
		Hash[((2 * threads) + thread) * 2] = (h[4]);
		Hash[((2 * threads) + thread) * 2 + 1] = (h[5]);
		Hash[((3 * threads) + thread) * 2] = (h[6]);
		Hash[((3 * threads) + thread) * 2 + 1] = (h[7]);
	}
}

__host__
void blake256_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash)
{
	const uint32_t threadsperblock = 64;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	blake256_gpu_hash_80 <<<grid, block>>> (threads, startNonce,(uint32_t *)Hash);
}

__host__
void blake256_cpu_setBlock_80(uint32_t *pdata)
{
	uint32_t h[8];
	uint32_t data[20];
	memcpy(data, pdata, 80);
	for (int i = 0; i<8; i++) {
		h[i] = c_IV256[i];
	}
	blake256_compress1st(h, pdata, 512);

	cudaMemcpyToSymbol(cpu_h, h, sizeof(h), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_data, &data[16], 3*4, 0, cudaMemcpyHostToDevice);
}

__host__
void blakeKeccak256_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	blakeKeccak256_gpu_hash_80 << <grid, block >> > (threads, startNonce, (uint32_t *)Hash);
}
