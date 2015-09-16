// Neoscrypt Kernel by djm34 enchanced by Sp_ and Pallas (@bitcointalk and github)

#include <stdio.h>
#include <memory.h>
#include "cuda_vector.h" 
 
//extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

__device__ uint4 *W;
uint32_t *d_NNonce[MAX_GPUS];
__constant__ uint32_t pTarget[1];
__constant__ uint32_t input_init[16];
__constant__ uint32_t c_data[20];

#define BLAKE2S_BLOCK_SIZE    64U 
#define BLAKE2S_OUT_SIZE      32U
#define BLAKE2S_KEY_SIZE      32U
#define FASTKDF_BUFFER_SIZE  256U


static const __constant__ uint8 BLAKE2S_IV_Vec =
{
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

static const uint8 BLAKE2S_IV_Vechost =
{
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

static const __constant__ uint32_t BLAKE2S_SIGMA[10][16] =
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
};

static const uint32_t BLAKE2S_SIGMA_host[10][16] =
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
};


#define BLAKE_G(idx0, idx1, a, b, c, d, key) { \
	a += key[BLAKE2S_SIGMA[idx0][idx1]] + b; \
	d = __byte_perm(d^a, 0, 0x1032); \
	c += d; \
	b = rotateR(b^c, 12); \
	a += key[BLAKE2S_SIGMA[idx0][idx1+1]] + b; \
	d = __byte_perm(d^a, 0, 0x0321); \
	c += d; \
	b = rotateR(b^c, 7); \
} 

#define BLAKE_G_PRE(idx0,idx1, a, b, c, d, key) { \
	a += key[idx0] + b; \
	d = __byte_perm(d^a, 0, 0x1032); \
	c += d; \
	b = rotateR(b^c, 12); \
	a += key[idx1] + b; \
	d = __byte_perm(d^a, 0, 0x0321); \
	c += d; \
	b = rotateR(b^c, 7); \
} 

#define BLAKE_Ghost(idx0, idx1, a, b, c, d, key) { \
	idx = BLAKE2S_SIGMA_host[idx0][idx1]; a += key[idx]; \
	a += b; d = ROTR32(d^a,16); \
	c += d; b = ROTR32(b^c, 12); \
	idx = BLAKE2S_SIGMA_host[idx0][idx1+1]; a += key[idx]; \
	a += b; d = ROTR32(d^a,8); \
	c += d; b = ROTR32(b^c, 7); \
} 


static __forceinline__ __device__ void Blake2S(uint32_t* inout, const uint32_t* TheKey)
{
	uint16 V;
	uint8 tmpblock;
 					
	V.hi = BLAKE2S_IV_Vec; 
  V.lo = V.hi;
	V.lo.s0 ^= 0x01012020;

	// Copy input block for later
	tmpblock = V.lo;

	V.hi.s4 ^= BLAKE2S_BLOCK_SIZE;

//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	BLAKE_G_PRE(0, 1, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(2, 3, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(4, 5, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(6, 7, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(8, 9, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(10,11, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(12, 13, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(14, 15, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

//	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	BLAKE_G_PRE(14, 10, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(4, 8, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(9, 15, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(13, 6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(1, 12, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(0, 2, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(11, 7, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(5, 3, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

//	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	BLAKE_G_PRE(11, 8, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(12, 0, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(5, 2, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(15, 13, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(10, 14, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(3, 6, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(7, 1, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(9, 4, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

//	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	BLAKE_G_PRE(7, 9, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(3, 1, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(13, 12, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(11, 14, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(2, 6, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(5, 10, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(4, 0, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(15, 8, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

//	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	BLAKE_G_PRE(9, 0, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(5, 7, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(2, 4, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(10, 15, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(14, 1, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(11, 12, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(6, 8, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(3, 13, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

//	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	BLAKE_G_PRE(2, 12, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(6, 10, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(0, 11, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(8, 3, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(4, 13, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(7, 5, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(15, 14, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(1, 9, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

//	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	BLAKE_G_PRE(12, 5, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(1, 15, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(14, 13, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(4, 10, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(0, 7, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(6, 3, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(9, 2, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(8, 11, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

//	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	BLAKE_G_PRE(13, 11, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(7, 14, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(12, 1, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(3, 9, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(5, 0, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(15, 4, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(8, 6, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(2, 10, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

//	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	BLAKE_G_PRE(6, 15, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(14, 9, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(11, 3, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(0, 8, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(12, 2, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(13, 7, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(1, 4, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(10, 5, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

//	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
	BLAKE_G_PRE(10, 2, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(8, 4, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(7, 6, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(1, 5, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(15, 11, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(9, 14, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(3, 12, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(13, 0, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	V.lo ^= V.hi ^ tmpblock;
	V.hi = BLAKE2S_IV_Vec;
	tmpblock = V.lo;

	V.hi.s4 ^= 128;
	V.hi.s6 ^= 0xFFFFFFFF;

	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	BLAKE_G_PRE(0, 1, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(2, 3, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(4, 5, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(6, 7, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(8, 9, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(10, 11, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(12, 13, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(14, 15, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

	//		{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	BLAKE_G_PRE(14, 10, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(4, 8, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(9, 15, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(13, 6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(1, 12, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(0, 2, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(11, 7, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(5, 3, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

	//		{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	BLAKE_G_PRE(11, 8, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(12, 0, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(5, 2, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(15, 13, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(10, 14, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(3, 6, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(7, 1, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(9, 4, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

	//		{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	BLAKE_G_PRE(7, 9, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(3, 1, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(13, 12, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(11, 14, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(2, 6, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(5, 10, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(4, 0, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(15, 8, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

//	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	BLAKE_G_PRE(9, 0, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(5, 7, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(2, 4, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(10, 15, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(14, 1, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(11, 12, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(6, 8, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(3, 13, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);
/*
//	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	BLAKE_G_PRE(2, 12, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(6, 10, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(0, 11, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(8, 3, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(4, 13, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(7, 5, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(15, 14, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(1, 9, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

//	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	BLAKE_G_PRE(12, 5, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(1, 15, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(14, 13, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(4, 10, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(0, 7, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(6, 3, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(9, 2, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(8, 11, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

//	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	BLAKE_G_PRE(13, 11, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(7, 14, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(12, 1, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(3, 9, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(5, 0, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(15, 4, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(8, 6, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(2, 10, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

//	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	BLAKE_G_PRE(6, 15, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(14, 9, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(11, 3, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(0, 8, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(12, 2, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(13, 7, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(1, 4, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(10, 5, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

//	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
	BLAKE_G_PRE(10, 2, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(8, 4, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(7, 6, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(1, 5, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(15, 11, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(9, 14, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(3, 12, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(13, 0, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);
*/
	#pragma nounroll
	for (int x = 5; x < 10; ++x)
	{
		BLAKE_G(x, 0x00, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
		BLAKE_G(x, 0x02, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
		BLAKE_G(x, 0x04, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
		BLAKE_G(x, 0x06, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
		BLAKE_G(x, 0x08, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
		BLAKE_G(x, 0x0A, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
		BLAKE_G(x, 0x0C, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
		BLAKE_G(x, 0x0E, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);
	}

	((uint8*)inout)[0] = V.lo ^ V.hi ^ tmpblock;
}


static __forceinline__ __host__ void Blake2Shost(uint32_t * inout, const uint32_t * inkey)
{
	uint16 V;
	uint32_t idx;
	uint8 tmpblock;

	V.hi = BLAKE2S_IV_Vechost;
	V.lo = BLAKE2S_IV_Vechost;
	V.lo.s0 ^= 0x01012020;

	tmpblock = V.lo;

	V.hi.s4 ^= BLAKE2S_BLOCK_SIZE;

	for (int x = 0; x < 10; ++x)
	{
		BLAKE_Ghost(x, 0x00, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inkey);
		BLAKE_Ghost(x, 0x02, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inkey);
		BLAKE_Ghost(x, 0x04, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inkey);
		BLAKE_Ghost(x, 0x06, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inkey);
		BLAKE_Ghost(x, 0x08, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inkey);
		BLAKE_Ghost(x, 0x0A, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inkey);
		BLAKE_Ghost(x, 0x0C, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inkey);
		BLAKE_Ghost(x, 0x0E, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inkey);
	}

	V.lo ^= V.hi;
	V.lo ^= tmpblock;

	V.hi = BLAKE2S_IV_Vechost;
	tmpblock = V.lo;

	V.hi.s4 ^= 128;
	V.hi.s6 = ~V.hi.s6;

	for (int x = 0; x < 10; ++x)
	{
		BLAKE_Ghost(x, 0x00, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
		BLAKE_Ghost(x, 0x02, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
		BLAKE_Ghost(x, 0x04, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
		BLAKE_Ghost(x, 0x06, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
		BLAKE_Ghost(x, 0x08, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
		BLAKE_Ghost(x, 0x0A, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
		BLAKE_Ghost(x, 0x0C, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
		BLAKE_Ghost(x, 0x0E, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);
	}

	V.lo ^= V.hi ^ tmpblock;

	((uint8*)inout)[0] = V.lo;
}


static __forceinline__ __device__ void shift256R2_final(uint32_t *ret, const uint8 &vec4, const uint32_t shift)
{
	uint32_t truc = 0, truc2 = cuda_swab32(vec4.s7), truc3 = 0;
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(truc) : "r"(truc3), "r"(truc2), "r"(shift));
	ret[8] = cuda_swab32(truc);
	truc3 = cuda_swab32(vec4.s6);
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(truc) : "r"(truc2), "r"(truc3), "r"(shift));
	ret[7] = cuda_swab32(truc);
}


static __forceinline__ __device__ uint32_t fastkdf(const uint32_t *password, uint8_t *output, const uint32_t *salt)
{ 
	uint8_t bufidx, A[320], B[288];
	((uintx64*)A)[0] = ((uintx64*)password)[0];	// 256 bits
	((uint816*)A)[4] = ((uint816*)A)[0];	// 64 bits
	uint32_t input[BLAKE2S_BLOCK_SIZE/4];
	uint32_t key[BLAKE2S_BLOCK_SIZE / 4]={0};
	
	if (salt != NULL) {
		((uintx64*)B)[0] = ((uintx64*)salt)[0];
	 	((uint48 *)B)[8] = ((uint48 *)B)[0];
		((uint816*)input)[0] = ((uint816*)A)[0];
		((uint48*)key)[0] = ((uint48*)B)[0];
	} else {
		((uintx64*)B)[0] = ((uintx64*)A)[0];
		((uint48 *)B)[8] = ((uint48 *)A)[0];	// 32 bits
		((uint816*)input)[0] = ((uint816*)input_init)[0];
		((uint48*)key)[0] = ((uint48*)input)[0];
	}
	
	#pragma nounroll
	for (int i = 0; i < 32; ++i)
	{
		if (salt != NULL) Blake2S((uint32_t*)input, key);
		
		//const uchar4 bufhelper = ((uchar4*)input)[0] + ((uchar4*)input)[1] + ((uchar4*)input)[2] + ((uchar4*)input)[3] + ((uchar4*)input)[4] + ((uchar4*)input)[5] + ((uchar4*)input)[6] + ((uchar4*)input)[7];
		uchar4 bufhelper = ((uchar4*)input)[0];
		#pragma unroll
		for (int x = 1; x < BLAKE2S_OUT_SIZE / 4; ++x) bufhelper += ((uchar4*)input)[x];
		bufidx = bufhelper.x + bufhelper.y + bufhelper.z + bufhelper.w;
		uint32_t shifted[9];
  
		if (i == 31 && salt != NULL) {
			shift256R2_final(shifted, ((uint8*)input)[0], (bufidx & 3) << 3);
			((uint32_t *)B)[7 + (bufidx >> 2)] ^= shifted[7];
			((uint32_t *)B)[8 + (bufidx >> 2)] ^= shifted[8];
			break;
		}

		shift256R2(shifted, ((uint8*)input)[0], (bufidx & 3) << 3);
		#pragma unroll
		for (int k = 0; k < 9; ++k) ((uint32_t *)B)[k + (bufidx >> 2)] ^= shifted[k];

		if (bufidx < BLAKE2S_KEY_SIZE)                          ((uint8*)B)[8] = ((uint8*)B)[0];
		else if (bufidx > FASTKDF_BUFFER_SIZE-BLAKE2S_OUT_SIZE) ((uint8*)B)[0] = ((uint8*)B)[8];

		if (i == 31) break;

		#pragma unroll
		for (int k = 0; k < BLAKE2S_BLOCK_SIZE / 4; k++) {
			((uchar4*)(input))[k] =
				make_uchar4((A + bufidx)[4 * k], (A + bufidx)[4 * k + 1], (A + bufidx)[4 * k + 2], (A + bufidx)[4 * k + 3]);
		}
		#pragma unroll
		for (int k = 0; k < BLAKE2S_KEY_SIZE / 4; k++) {
			((uchar4*)(key))[k] =
				make_uchar4((B + bufidx)[4 * k], (B + bufidx)[4 * k + 1], (B + bufidx)[4 * k + 2], (B + bufidx)[4 * k + 3]);
		} 
		
		if (salt == NULL) Blake2S((uint32_t*)input, key);
	}

	if (salt != NULL) {
		uchar4 unfucked[1];
		unfucked[0] = make_uchar4(B[28 + bufidx], B[29 + bufidx], B[30 + bufidx], B[31 + bufidx]);
		return ((uint32_t*)unfucked)[0] ^ ((uint32_t*)A)[7];
	} else {
		#pragma nounroll
		for (int i = 0; i < FASTKDF_BUFFER_SIZE / 4; ++i) {
			((uchar4*)output)[i] =
				make_uchar4(B[(uint8_t)(4 * i + bufidx)], B[(uint8_t)(4 * i + 1 + bufidx)], B[(uint8_t)(4 * i + 2 + bufidx)], B[(uint8_t)(4 * i + 3 + bufidx)]) ^ ((uchar4*)A)[i];
		}
		return 0;
	}
}

 
#define SALSA(a,b,c,d) { \
    b^=rotate(a+d,  7); \
    c^=rotate(b+a,  9); \
    d^=rotate(c+b, 13); \
    a^=rotate(d+c, 18); \
}

#define SALSA_CORE(state) { \
	SALSA(state.s0,state.s4,state.s8,state.sc); \
	SALSA(state.s5,state.s9,state.sd,state.s1); \
	SALSA(state.sa,state.se,state.s2,state.s6); \
	SALSA(state.sf,state.s3,state.s7,state.sb); \
	SALSA(state.s0,state.s1,state.s2,state.s3); \
	SALSA(state.s5,state.s6,state.s7,state.s4); \
	SALSA(state.sa,state.sb,state.s8,state.s9); \
	SALSA(state.sf,state.sc,state.sd,state.se); \
} 

#define CHACHA_STEP(a,b,c,d) { \
	a += b; d = __byte_perm(d^a,0,0x1032); \
	c += d; b = rotate(b^c, 12); \
	a += b; d = __byte_perm(d^a,0,0x2103); \
	c += d; b = rotate(b^c, 7); \
}

#define CHACHA_CORE_PARALLEL(state)	 { \
	CHACHA_STEP(state.lo.s0, state.lo.s4, state.hi.s0, state.hi.s4); \
	CHACHA_STEP(state.lo.s1, state.lo.s5, state.hi.s1, state.hi.s5); \
	CHACHA_STEP(state.lo.s2, state.lo.s6, state.hi.s2, state.hi.s6); \
	CHACHA_STEP(state.lo.s3, state.lo.s7, state.hi.s3, state.hi.s7); \
	CHACHA_STEP(state.lo.s0, state.lo.s5, state.hi.s2, state.hi.s7); \
	CHACHA_STEP(state.lo.s1, state.lo.s6, state.hi.s3, state.hi.s4); \
	CHACHA_STEP(state.lo.s2, state.lo.s7, state.hi.s0, state.hi.s5); \
	CHACHA_STEP(state.lo.s3, state.lo.s4, state.hi.s1, state.hi.s6); \
}


__forceinline__ __device__ uint16 salsa_small_scalar_rnd(const uint16 &X)
{
	uint16 state = X;
	#pragma nounroll
	for (int i = 0; i < 10; ++i) SALSA_CORE(state);
	return(X + state);
}


__device__ __forceinline__ uint16 chacha_small_parallel_rnd(const uint16 &X)
{ 
	uint16 state = X;
	#pragma nounroll
	for (int i = 0; i < 10; ++i) CHACHA_CORE_PARALLEL(state);
	return(X + state);
}


static __device__ __forceinline__ void neoscrypt_chacha(uint16 *XV)
{
	XV[0] ^= XV[3];
	XV[0] = chacha_small_parallel_rnd(XV[0]); XV[1] ^= XV[0];
  uint16 temp = chacha_small_parallel_rnd(XV[1]); XV[2] ^= temp;
	XV[1] = chacha_small_parallel_rnd(XV[2]); XV[3] ^= XV[1];
	XV[3] = chacha_small_parallel_rnd(XV[3]);
  XV[2] = temp;
}


static __device__ __forceinline__ void neoscrypt_salsa(uint16 *XV)
{
	XV[0] ^= XV[3];
	XV[0] = salsa_small_scalar_rnd(XV[0]); XV[1] ^= XV[0];
	uint16 temp = salsa_small_scalar_rnd(XV[1]); XV[2] ^= temp;
	XV[1] = salsa_small_scalar_rnd(XV[2]); XV[3] ^= XV[1];
	XV[3] = salsa_small_scalar_rnd(XV[3]);
  XV[2] = temp;
}   

 
#define SHIFT 130

__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(128, 2)
#else
__launch_bounds__(128, 3)
#endif
void neoscrypt_gpu_hash_k0(int stratum, uint32_t startNonce)
{
	const int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const int shift = SHIFT * 16 * thread;
	const uint32_t nonce = startNonce + thread;
	
	uint16 X[4];
	uint32_t data[80];

	#pragma unroll
	for (int i = 0; i <  5; i++) ((uint4*)data)[i] = ((uint4 *)c_data)[i];	//ld.local.v4
	data[19] = (stratum) ? cuda_swab32(nonce) : nonce;
	#pragma unroll
	for (int i = 5; i < 20; i++) ((uint4*)data)[i] = ((uint4 *)data)[i % 5];

	fastkdf(data, (uint8_t*)X, NULL);	//256
	((uintx64 *)(W + shift))[0] = ((uintx64 *)X)[0];
//	((ulonglong16 *)(W + shift))[0] = ((ulonglong16 *)X)[0];
}


__global__ __launch_bounds__(128, 2) void neoscrypt_gpu_hash_k01()
{
	const int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const int shift = SHIFT * 16 * thread;
	uint16 X[4];
	((uintx64 *)X)[0]= __ldg32(&(W + shift)[0]);

	#pragma nounroll
	for (int i = 0; i < 128; ++i)
	{			
		neoscrypt_chacha(X);
//		((ulonglong16 *)(W + shift))[i + 1] = ((ulonglong16 *)X)[0];
		((uintx64 *)(W + shift))[i + 1] = ((uintx64 *)X)[0];
	}
}


__global__ __launch_bounds__(128, 2) void neoscrypt_gpu_hash_k2()
{
	const int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const int shift = SHIFT * 16 * thread;
	uint16 X[4];
	((uintx64 *)X)[0] = __ldg32(&(W + shift)[2048]);
	
	#pragma nounroll 
	for (int t = 0; t < 128; t++)
	{
		((uintx64 *)X)[0] ^= __ldg32(&(W + shift)[(X[3].lo.s0 & 0x7F) << 4]);
		neoscrypt_chacha(X);
	}
	((uintx64 *)(W + shift))[129] = ((uintx64*)X)[0];  // best checked
}


__global__ __launch_bounds__(128, 2) void neoscrypt_gpu_hash_k3()
{
	const int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const int shift = SHIFT * 16 * thread;
	uint16 Z[4];
	
	((uintx64*)Z)[0] = __ldg32(&(W + shift)[0]);

	#pragma nounroll 
	for (int i = 0; i < 128; ++i)
	{
		neoscrypt_salsa(Z);
//		((ulonglong16 *)(W + shift))[i + 1] = ((ulonglong16 *)Z)[0];
		((uintx64 *)(W + shift))[i + 1] = ((uintx64 *)Z)[0];
	}
}


__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(128, 3)
#else
__launch_bounds__(32, 12)
#endif
void neoscrypt_gpu_hash_k4(int stratum, uint32_t startNonce, uint32_t *nonceVector)
{
	const int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint32_t nonce = startNonce + thread;
	const int shift = SHIFT * 16 * thread;
	uint16 Z[4]; 
	uint32_t data[80];

	#pragma unroll
	for (int i = 0; i <  5; i++) ((uint4*)data)[i] = ((uint4 *)c_data)[i];
	data[19] = (stratum) ? cuda_swab32(nonce) : nonce;
	#pragma unroll
	for (int i = 5; i < 20; i++) ((uint4*)data)[i] = ((uint4 *)data)[i % 5];

	((uintx64 *)Z)[0] = __ldg32(&(W + shift)[2048]);
	#pragma nounroll
	for (int t = 0; t < 128; t++)
	{
		((uintx64 *)Z)[0] ^= __ldg32(&(W + shift)[(Z[3].lo.s0 & 0x7F) << 4]);
		neoscrypt_salsa(Z);
	}
	((uintx64 *)Z)[0] ^= __ldg32(&(W + shift)[2064]);
	
	if (fastkdf(data, NULL, (uint32_t*)Z) <= pTarget[0]) atomicCAS(&nonceVector[0], 0xffffffff, nonce);	//32
}


void neoscrypt_cpu_init(int thr_id, uint32_t *hash)
{
//	cudaMemcpyToSymbol(BLAKE2S_SIGMA, BLAKE2S_SIGMA_host, sizeof(BLAKE2S_SIGMA_host), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(W, &hash, sizeof(hash), 0, cudaMemcpyHostToDevice);
	cudaMalloc(&d_NNonce[thr_id], sizeof(uint32_t));
} 


__host__ uint32_t neoscrypt_cpu_hash_k4(int stratum, int thr_id, int threads, uint32_t startNounce, const int threadsperblock)
{
	uint32_t result;
	cudaMemset(d_NNonce[thr_id], 0xffffffff, sizeof(uint32_t));

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
 	
	neoscrypt_gpu_hash_k0  << <grid, block >> >(stratum, startNounce);  //b
	neoscrypt_gpu_hash_k01 << <grid, block >> >();  //b
	neoscrypt_gpu_hash_k2  << <grid, block >> >();  //a
	neoscrypt_gpu_hash_k3  << <grid, block >> >();  //b
	neoscrypt_gpu_hash_k4  << <grid, block >> >(stratum, startNounce, d_NNonce[thr_id]);  //a

//	MyStreamSynchronize(NULL, order, thr_id);
	cudaMemcpy(&result, d_NNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
//	cudaDeviceReset();
	
	return result;
}


__host__ void neoscrypt_setBlockTarget(uint32_t *pdata, const void *target)
{
	uint32_t input[16], key[16] = {0};

	((uint16*)input)[0] = ((uint16*)pdata)[0];
	((uint8*)key)[0] = ((uint8*)pdata)[0];
	Blake2Shost(input, key);

	cudaMemcpyToSymbol(pTarget, ((uint32_t*) target) + 7, sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(input_init, input, 16 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_data, pdata, 10 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
}

