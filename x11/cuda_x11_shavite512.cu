#include "cuda_helper.h"
#include <memory.h> // memcpy()

#define TPB 128

// aus heavy.cu
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

__constant__ uint32_t c_PaddedMessage80[32]; // padded message (80 bytes + padding)

#include "cuda_x11_aes.cu"

__device__ __forceinline__
static void AES_ROUND_NOKEY(
	const uint32_t* __restrict__ sharedMemory,
	uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3)
{
	uint32_t y0, y1, y2, y3;
	aes_round(sharedMemory,
		x0, x1, x2, x3,
		y0, y1, y2, y3);

	x0 = y0;
	x1 = y1;
	x2 = y2;
	x3 = y3;
}

__device__ __forceinline__
static void KEY_EXPAND_ELT(
	const uint32_t* __restrict__ sharedMemory,
	uint32_t &k0, uint32_t &k1, uint32_t &k2, uint32_t &k3)
{
	uint32_t y0, y1, y2, y3;
	aes_round(sharedMemory,
		k0, k1, k2, k3,
		y0, y1, y2, y3);

	k0 = y1;
	k1 = y2;
	k2 = y3;
	k3 = y0;
}

__device__ __forceinline__
static void c512(const uint32_t*const __restrict__ sharedMemory, uint32_t *const __restrict__  state, uint32_t *const __restrict__  msg, const uint32_t count)
{
	uint32_t p0, p1, p2, p3, p4, p5, p6, p7;
	uint32_t p8, p9, pA, pB, pC, pD, pE, pF;
	uint32_t x0, x1, x2, x3;
	uint32_t rk[32];
	uint32_t i;
	const uint32_t counter = count;

	p0 = state[0x0];
	p1 = state[0x1];
	p2 = state[0x2];
	p3 = state[0x3];
	p4 = state[0x4];
	p5 = state[0x5];
	p6 = state[0x6];
	p7 = state[0x7];
	p8 = state[0x8];
	p9 = state[0x9];
	pA = state[0xA];
	pB = state[0xB];
	pC = state[0xC];
	pD = state[0xD];
	pE = state[0xE];
	pF = state[0xF];

	x0 = p4;
	x1 = p5;
	x2 = p6;
	x3 = p7;	
#pragma unroll
	for (i=0;i<16;i+=4)
	{
		rk[i] = msg[i];
		x0 ^= msg[i];
		rk[i + 1] = msg[i + 1];
		x1 ^= msg[i+1];
		rk[i + 2] = msg[i + 2];
		x2 ^= msg[i + 2];
		rk[i + 3] = msg[i + 3];
		x3 ^= msg[i + 3];
		AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	}

	p0 ^= x0;
	p1 ^= x1;
	p2 ^= x2;
	p3 ^= x3;
	if (count == 512)
	{
		rk[16] = 0x80U;
		x0 = pC ^ 0x80U;
		rk[17] = 0;
		x1 = pD;
		rk[18] = 0;
		x2 = pE;
		rk[19] = 0;
		x3 = pF;
		AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
		rk[20] = 0;
		rk[21] = 0;
		rk[22] = 0;
		rk[23] = 0;
		AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
		rk[24] = 0;
		rk[25] = 0;
		rk[26] = 0;
		rk[27] = 0x02000000U;
		x3 ^= 0x02000000U;
		AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
		rk[28] = 0;
		rk[29] = 0;
		rk[30] = 0;
		rk[31] = 0x02000000;
		x3 ^= 0x02000000;
		AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	}
	else
	{
		x0 = pC;
		x1 = pD;
		x2 = pE;
		x3 = pF;

		for (i = 16; i<32; i += 4)
		{
			rk[i] = msg[i];
			x0 ^= msg[i];
			rk[i+1] = msg[i+1];
			x1 ^= msg[i + 1];
			rk[i + 2] = msg[i + 2];
			x2 ^= msg[i + 2];
			rk[i + 3] = msg[i + 3];
			x3 ^= msg[i + 3];
			AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
		}
	}
	p8 ^= x0;
	p9 ^= x1;
	pA ^= x2;
	pB ^= x3;

	// 1
	KEY_EXPAND_ELT(sharedMemory, rk[0], rk[1], rk[2], rk[3]);

	rk[0] ^= rk[28];
	rk[1] ^= rk[29];
	rk[2] ^= rk[30];
	rk[3] ^= rk[31];
	rk[0] ^= counter;
	rk[3] ^= 0xFFFFFFFF;
	x0 = p0 ^ rk[0];
	x1 = p1 ^ rk[1];
	x2 = p2 ^ rk[2];
	x3 = p3 ^ rk[3];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[4], rk[5], rk[6], rk[7]);
	rk[4] ^= rk[0];
	rk[5] ^= rk[1];
	rk[6] ^= rk[2];
	rk[7] ^= rk[3];
	x0 ^= rk[4];
	x1 ^= rk[5];
	x2 ^= rk[6];
	x3 ^= rk[7];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[8], rk[9], rk[10], rk[11]);
	rk[8] ^= rk[4];
	rk[9] ^= rk[5];
	rk[10] ^= rk[6];
	rk[11] ^= rk[7];
	x0 ^= rk[8];
	x1 ^= rk[9];
	x2 ^= rk[10];
	x3 ^= rk[11];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[12], rk[13], rk[14], rk[15]);
	rk[12] ^= rk[8];
	rk[13] ^= rk[9];
	rk[14] ^= rk[10];
	rk[15] ^= rk[11];
	x0 ^= rk[12];
	x1 ^= rk[13];
	x2 ^= rk[14];
	x3 ^= rk[15];

	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	pC ^= x0;
	pD ^= x1;
	pE ^= x2;
	pF ^= x3;


	KEY_EXPAND_ELT(sharedMemory, rk[16], rk[17], rk[18], rk[19]);
	rk[16] ^= rk[12];
	rk[17] ^= rk[13];
	rk[18] ^= rk[14];
	rk[19] ^= rk[15];
	x0 = p8 ^ rk[16];
	x1 = p9 ^ rk[17];
	x2 = pA ^ rk[18];
	x3 = pB ^ rk[19];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[20], rk[21], rk[22], rk[23]);
	rk[20] ^= rk[16];
	rk[21] ^= rk[17];
	rk[22] ^= rk[18];
	rk[23] ^= rk[19];
	x0 ^= rk[20];
	x1 ^= rk[21];
	x2 ^= rk[22];
	x3 ^= rk[23];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[24], rk[25], rk[26], rk[27]);
	rk[24] ^= rk[20];
	rk[25] ^= rk[21];
	rk[26] ^= rk[22];
	rk[27] ^= rk[23];
	x0 ^= rk[24];
	x1 ^= rk[25];
	x2 ^= rk[26];
	x3 ^= rk[27];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[28], rk[29], rk[30], rk[31]);
	rk[28] ^= rk[24];
	rk[29] ^= rk[25];
	rk[30] ^= rk[26];
	rk[31] ^= rk[27];
	x0 ^= rk[28];
	x1 ^= rk[29];
	x2 ^= rk[30];
	x3 ^= rk[31];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p4 ^= x0;
	p5 ^= x1;
	p6 ^= x2;
	p7 ^= x3;

	rk[0] ^= rk[25];
	x0 = pC ^ rk[0];
	rk[1] ^= rk[26];
	x1 = pD ^ rk[1];
	rk[2] ^= rk[27];
	x2 = pE ^ rk[2];
	rk[3] ^= rk[28];
	x3 = pF ^ rk[3];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[4] ^= rk[29];
	x0 ^= rk[4];
	rk[5] ^= rk[30];
	x1 ^= rk[5];
	rk[6] ^= rk[31];
	x2 ^= rk[6];
	rk[7] ^= rk[0];
	x3 ^= rk[7];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[8] ^= rk[1];
	x0 ^= rk[8];
	rk[9] ^= rk[2];
	x1 ^= rk[9];
	rk[10] ^= rk[3];
	x2 ^= rk[10];
	rk[11] ^= rk[4];
	x3 ^= rk[11];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[12] ^= rk[5];
	x0 ^= rk[12];
	rk[13] ^= rk[6];
	x1 ^= rk[13];
	rk[14] ^= rk[7];
	x2 ^= rk[14];
	rk[15] ^= rk[8];
	x3 ^= rk[15];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p8 ^= x0;
	p9 ^= x1;
	pA ^= x2;
	pB ^= x3;
	rk[16] ^= rk[9];
	x0 = p4 ^ rk[16];
	rk[17] ^= rk[10];
	x1 = p5 ^ rk[17];
	rk[18] ^= rk[11];
	x2 = p6 ^ rk[18];
	rk[19] ^= rk[12];
	x3 = p7 ^ rk[19];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[20] ^= rk[13];
	x0 ^= rk[20];
	rk[21] ^= rk[14];
	x1 ^= rk[21];
	rk[22] ^= rk[15];
	x2 ^= rk[22];
	rk[23] ^= rk[16];
	x3 ^= rk[23];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[24] ^= rk[17];
	x0 ^= rk[24];
	rk[25] ^= rk[18];
	x1 ^= rk[25];
	rk[26] ^= rk[19];
	x2 ^= rk[26];
	rk[27] ^= rk[20];
	x3 ^= rk[27];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[28] ^= rk[21];
	x0 ^= rk[28];
	rk[29] ^= rk[22];
	x1 ^= rk[29];
	rk[30] ^= rk[23];
	x2 ^= rk[30];
	rk[31] ^= rk[24];
	x3 ^= rk[31];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p0 ^= x0;
	p1 ^= x1;
	p2 ^= x2;
	p3 ^= x3;
	/* round 3, 7, 11 */
	KEY_EXPAND_ELT(sharedMemory, rk[0], rk[1], rk[2], rk[3]);
	rk[0] ^= rk[28];
	rk[1] ^= rk[29];
	rk[2] ^= rk[30];
	rk[3] ^= rk[31];
	x0 = p8 ^ rk[0];
	x1 = p9 ^ rk[1];
	x2 = pA ^ rk[2];
	x3 = pB ^ rk[3];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[4], rk[5], rk[6], rk[7]);
	rk[4] ^= rk[0];
	rk[5] ^= rk[1];
	rk[6] ^= rk[2];
	rk[7] ^= rk[3];
	x0 ^= rk[4];
	x1 ^= rk[5];
	x2 ^= rk[6];
	x3 ^= rk[7];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[8], rk[9], rk[10], rk[11]);
	rk[8] ^= rk[4];
	rk[9] ^= rk[5];
	rk[10] ^= rk[6];
	rk[11] ^= rk[7];
	x0 ^= rk[8];
	x1 ^= rk[9];
	x2 ^= rk[10];
	x3 ^= rk[11];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[12], rk[13], rk[14], rk[15]);
	rk[12] ^= rk[8];
	rk[13] ^= rk[9];
	rk[14] ^= rk[10];
	rk[15] ^= rk[11];
	x0 ^= rk[12];
	x1 ^= rk[13];
	x2 ^= rk[14];
	x3 ^= rk[15];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p4 ^= x0;
	p5 ^= x1;
	p6 ^= x2;
	p7 ^= x3;
	KEY_EXPAND_ELT(sharedMemory, rk[16], rk[17], rk[18], rk[19]);
	rk[16] ^= rk[12];
	rk[17] ^= rk[13];
	rk[18] ^= rk[14];
	rk[19] ^= rk[15];
	x0 = p0 ^ rk[16];
	x1 = p1 ^ rk[17];
	x2 = p2 ^ rk[18];
	x3 = p3 ^ rk[19];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[20], rk[21], rk[22], rk[23]);
	rk[20] ^= rk[16];
	rk[21] ^= rk[17];
	rk[22] ^= rk[18];
	rk[23] ^= rk[19];
	x0 ^= rk[20];
	x1 ^= rk[21];
	x2 ^= rk[22];
	x3 ^= rk[23];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[24], rk[25], rk[26], rk[27]);
	rk[24] ^= rk[20];
	rk[25] ^= rk[21];
	rk[26] ^= rk[22];
	rk[27] ^= rk[23];
	x0 ^= rk[24];
	x1 ^= rk[25];
	x2 ^= rk[26];
	x3 ^= rk[27];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[28], rk[29], rk[30], rk[31]);
	rk[28] ^= rk[24];
	rk[29] ^= rk[25];
	rk[30] ^= rk[26];
	rk[31] ^= rk[27];
	x0 ^= rk[28];
	x1 ^= rk[29];
	x2 ^= rk[30];
	x3 ^= rk[31];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	pC ^= x0;
	pD ^= x1;
	pE ^= x2;
	pF ^= x3;
	/* round 4, 8, 12 */
	rk[0] ^= rk[25];
	x0 = p4 ^ rk[0];
	rk[1] ^= rk[26];
	x1 = p5 ^ rk[1];
	rk[2] ^= rk[27];
	x2 = p6 ^ rk[2];
	rk[3] ^= rk[28];
	x3 = p7 ^ rk[3];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[4] ^= rk[29];
	x0 ^= rk[4];
	rk[5] ^= rk[30];
	x1 ^= rk[5];
	rk[6] ^= rk[31];
	x2 ^= rk[6];
	rk[7] ^= rk[0];
	x3 ^= rk[7];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[8] ^= rk[1];
	x0 ^= rk[8];
	rk[9] ^= rk[2];
	x1 ^= rk[9];
	rk[10] ^= rk[3];
	x2 ^= rk[10];
	rk[11] ^= rk[4];
	x3 ^= rk[11];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[12] ^= rk[5];
	x0 ^= rk[12];
	rk[13] ^= rk[6];
	x1 ^= rk[13];
	rk[14] ^= rk[7];
	x2 ^= rk[14];
	rk[15] ^= rk[8];
	x3 ^= rk[15];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);

	p0 ^= x0;
	p1 ^= x1;
	p2 ^= x2;
	p3 ^= x3;
	rk[16] ^= rk[9];
	x0 = pC ^ rk[16];
	rk[17] ^= rk[10];
	x1 = pD ^ rk[17];
	rk[18] ^= rk[11];
	x2 = pE ^ rk[18];
	rk[19] ^= rk[12];
	x3 = pF ^ rk[19];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[20] ^= rk[13];
	x0 ^= rk[20];
	rk[21] ^= rk[14];
	x1 ^= rk[21];
	rk[22] ^= rk[15];
	x2 ^= rk[22];
	rk[23] ^= rk[16];
	x3 ^= rk[23];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[24] ^= rk[17];
	x0 ^= rk[24];
	rk[25] ^= rk[18];
	x1 ^= rk[25];
	rk[26] ^= rk[19];
	x2 ^= rk[26];
	rk[27] ^= rk[20];
	x3 ^= rk[27];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[28] ^= rk[21];
	x0 ^= rk[28];
	rk[29] ^= rk[22];
	x1 ^= rk[29];
	rk[30] ^= rk[23];
	x2 ^= rk[30];
	rk[31] ^= rk[24];
	x3 ^= rk[31];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p8 ^= x0;
	p9 ^= x1;
	pA ^= x2;
	pB ^= x3;

	// 2
	KEY_EXPAND_ELT(sharedMemory, rk[0], rk[1], rk[2], rk[3]);
	rk[0] ^= rk[28];
	rk[1] ^= rk[29];
	rk[2] ^= rk[30];
	rk[3] ^= rk[31];
	x0 = p0 ^ rk[0];
	x1 = p1 ^ rk[1];
	x2 = p2 ^ rk[2];
	x3 = p3 ^ rk[3];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[4], rk[5], rk[6], rk[7]);
	rk[4] ^= rk[0];
	rk[5] ^= rk[1];
	rk[6] ^= rk[2];
	rk[7] ^= rk[3];
	rk[7] ^= SPH_T32(~counter);
	x0 ^= rk[4];
	x1 ^= rk[5];
	x2 ^= rk[6];
	x3 ^= rk[7];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[8], rk[9], rk[10], rk[11]);
	rk[8] ^= rk[4];
	rk[9] ^= rk[5];
	rk[10] ^= rk[6];
	rk[11] ^= rk[7];
	x0 ^= rk[8];
	x1 ^= rk[9];
	x2 ^= rk[10];
	x3 ^= rk[11];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[12], rk[13], rk[14], rk[15]);
	rk[12] ^= rk[8];
	rk[13] ^= rk[9];
	rk[14] ^= rk[10];
	rk[15] ^= rk[11];
	x0 ^= rk[12];
	x1 ^= rk[13];
	x2 ^= rk[14];
	x3 ^= rk[15];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	pC ^= x0;
	pD ^= x1;
	pE ^= x2;
	pF ^= x3;
	KEY_EXPAND_ELT(sharedMemory, rk[16], rk[17], rk[18], rk[19]);
	rk[16] ^= rk[12];
	rk[17] ^= rk[13];
	rk[18] ^= rk[14];
	rk[19] ^= rk[15];
	x0 = p8 ^ rk[16];
	x1 = p9 ^ rk[17];
	x2 = pA ^ rk[18];
	x3 = pB ^ rk[19];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[20], rk[21], rk[22], rk[23]);
	rk[20] ^= rk[16];
	rk[21] ^= rk[17];
	rk[22] ^= rk[18];
	rk[23] ^= rk[19];
	x0 ^= rk[20];
	x1 ^= rk[21];
	x2 ^= rk[22];
	x3 ^= rk[23];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[24], rk[25], rk[26], rk[27]);
	rk[24] ^= rk[20];
	rk[25] ^= rk[21];
	rk[26] ^= rk[22];
	rk[27] ^= rk[23];
	x0 ^= rk[24];
	x1 ^= rk[25];
	x2 ^= rk[26];
	x3 ^= rk[27];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[28], rk[29], rk[30], rk[31]);
	rk[28] ^= rk[24];
	rk[29] ^= rk[25];
	rk[30] ^= rk[26];
	rk[31] ^= rk[27];
	x0 ^= rk[28];
	x1 ^= rk[29];
	x2 ^= rk[30];
	x3 ^= rk[31];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p4 ^= x0;
	p5 ^= x1;
	p6 ^= x2;
	p7 ^= x3;

	rk[0] ^= rk[25];
	x0 = pC ^ rk[0];
	rk[1] ^= rk[26];
	x1 = pD ^ rk[1];
	rk[2] ^= rk[27];
	x2 = pE ^ rk[2];
	rk[3] ^= rk[28];
	x3 = pF ^ rk[3];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[4] ^= rk[29];
	x0 ^= rk[4];
	rk[5] ^= rk[30];
	x1 ^= rk[5];
	rk[6] ^= rk[31];
	x2 ^= rk[6];
	rk[7] ^= rk[0];
	x3 ^= rk[7];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[8] ^= rk[1];
	x0 ^= rk[8];
	rk[9] ^= rk[2];
	x1 ^= rk[9];
	rk[10] ^= rk[3];
	x2 ^= rk[10];
	rk[11] ^= rk[4];
	x3 ^= rk[11];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[12] ^= rk[5];
	x0 ^= rk[12];
	rk[13] ^= rk[6];
	x1 ^= rk[13];
	rk[14] ^= rk[7];
	x2 ^= rk[14];
	rk[15] ^= rk[8];
	x3 ^= rk[15];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p8 ^= x0;
	p9 ^= x1;
	pA ^= x2;
	pB ^= x3;
	rk[16] ^= rk[9];
	x0 = p4 ^ rk[16];
	rk[17] ^= rk[10];
	x1 = p5 ^ rk[17];
	rk[18] ^= rk[11];
	x2 = p6 ^ rk[18];
	rk[19] ^= rk[12];
	x3 = p7 ^ rk[19];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[20] ^= rk[13];
	x0 ^= rk[20];
	rk[21] ^= rk[14];
	x1 ^= rk[21];
	rk[22] ^= rk[15];
	x2 ^= rk[22];
	rk[23] ^= rk[16];
	x3 ^= rk[23];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[24] ^= rk[17];
	x0 ^= rk[24];
	rk[25] ^= rk[18];
	x1 ^= rk[25];
	rk[26] ^= rk[19];
	x2 ^= rk[26];
	rk[27] ^= rk[20];
	x3 ^= rk[27];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[28] ^= rk[21];
	x0 ^= rk[28];
	rk[29] ^= rk[22];
	x1 ^= rk[29];
	rk[30] ^= rk[23];
	x2 ^= rk[30];
	rk[31] ^= rk[24];
	x3 ^= rk[31];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p0 ^= x0;
	p1 ^= x1;
	p2 ^= x2;
	p3 ^= x3;
	/* round 3, 7, 11 */
	KEY_EXPAND_ELT(sharedMemory, rk[0], rk[1], rk[2], rk[3]);
	rk[0] ^= rk[28];
	rk[1] ^= rk[29];
	rk[2] ^= rk[30];
	rk[3] ^= rk[31];
	x0 = p8 ^ rk[0];
	x1 = p9 ^ rk[1];
	x2 = pA ^ rk[2];
	x3 = pB ^ rk[3];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[4], rk[5], rk[6], rk[7]);
	rk[4] ^= rk[0];
	rk[5] ^= rk[1];
	rk[6] ^= rk[2];
	rk[7] ^= rk[3];
	x0 ^= rk[4];
	x1 ^= rk[5];
	x2 ^= rk[6];
	x3 ^= rk[7];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[8], rk[9], rk[10], rk[11]);
	rk[8] ^= rk[4];
	rk[9] ^= rk[5];
	rk[10] ^= rk[6];
	rk[11] ^= rk[7];
	x0 ^= rk[8];
	x1 ^= rk[9];
	x2 ^= rk[10];
	x3 ^= rk[11];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[12], rk[13], rk[14], rk[15]);
	rk[12] ^= rk[8];
	rk[13] ^= rk[9];
	rk[14] ^= rk[10];
	rk[15] ^= rk[11];
	x0 ^= rk[12];
	x1 ^= rk[13];
	x2 ^= rk[14];
	x3 ^= rk[15];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p4 ^= x0;
	p5 ^= x1;
	p6 ^= x2;
	p7 ^= x3;
	KEY_EXPAND_ELT(sharedMemory, rk[16], rk[17], rk[18], rk[19]);
	rk[16] ^= rk[12];
	rk[17] ^= rk[13];
	rk[18] ^= rk[14];
	rk[19] ^= rk[15];
	x0 = p0 ^ rk[16];
	x1 = p1 ^ rk[17];
	x2 = p2 ^ rk[18];
	x3 = p3 ^ rk[19];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[20], rk[21], rk[22], rk[23]);
	rk[20] ^= rk[16];
	rk[21] ^= rk[17];
	rk[22] ^= rk[18];
	rk[23] ^= rk[19];
	x0 ^= rk[20];
	x1 ^= rk[21];
	x2 ^= rk[22];
	x3 ^= rk[23];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[24], rk[25], rk[26], rk[27]);
	rk[24] ^= rk[20];
	rk[25] ^= rk[21];
	rk[26] ^= rk[22];
	rk[27] ^= rk[23];
	x0 ^= rk[24];
	x1 ^= rk[25];
	x2 ^= rk[26];
	x3 ^= rk[27];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[28], rk[29], rk[30], rk[31]);
	rk[28] ^= rk[24];
	rk[29] ^= rk[25];
	rk[30] ^= rk[26];
	rk[31] ^= rk[27];
	x0 ^= rk[28];
	x1 ^= rk[29];
	x2 ^= rk[30];
	x3 ^= rk[31];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	pC ^= x0;
	pD ^= x1;
	pE ^= x2;
	pF ^= x3;
	/* round 4, 8, 12 */
	rk[0] ^= rk[25];
	x0 = p4 ^ rk[0];
	rk[1] ^= rk[26];
	x1 = p5 ^ rk[1];
	rk[2] ^= rk[27];
	x2 = p6 ^ rk[2];
	rk[3] ^= rk[28];
	x3 = p7 ^ rk[3];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[4] ^= rk[29];
	x0 ^= rk[4];
	rk[5] ^= rk[30];
	x1 ^= rk[5];
	rk[6] ^= rk[31];
	x2 ^= rk[6];
	rk[7] ^= rk[0];
	x3 ^= rk[7];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[8] ^= rk[1];
	x0 ^= rk[8];
	rk[9] ^= rk[2];
	x1 ^= rk[9];
	rk[10] ^= rk[3];
	x2 ^= rk[10];
	rk[11] ^= rk[4];
	x3 ^= rk[11];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[12] ^= rk[5];
	x0 ^= rk[12];
	rk[13] ^= rk[6];
	x1 ^= rk[13];
	rk[14] ^= rk[7];
	x2 ^= rk[14];
	rk[15] ^= rk[8];
	x3 ^= rk[15];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p0 ^= x0;
	p1 ^= x1;
	p2 ^= x2;
	p3 ^= x3;
	rk[16] ^= rk[9];
	x0 = pC ^ rk[16];
	rk[17] ^= rk[10];
	x1 = pD ^ rk[17];
	rk[18] ^= rk[11];
	x2 = pE ^ rk[18];
	rk[19] ^= rk[12];
	x3 = pF ^ rk[19];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[20] ^= rk[13];
	x0 ^= rk[20];
	rk[21] ^= rk[14];
	x1 ^= rk[21];
	rk[22] ^= rk[15];
	x2 ^= rk[22];
	rk[23] ^= rk[16];
	x3 ^= rk[23];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[24] ^= rk[17];
	x0 ^= rk[24];
	rk[25] ^= rk[18];
	x1 ^= rk[25];
	rk[26] ^= rk[19];
	x2 ^= rk[26];
	rk[27] ^= rk[20];
	x3 ^= rk[27];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[28] ^= rk[21];
	x0 ^= rk[28];
	rk[29] ^= rk[22];
	x1 ^= rk[29];
	rk[30] ^= rk[23];
	x2 ^= rk[30];
	rk[31] ^= rk[24];
	x3 ^= rk[31];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p8 ^= x0;
	p9 ^= x1;
	pA ^= x2;
	pB ^= x3;

	// 3
	KEY_EXPAND_ELT(sharedMemory, rk[0], rk[1], rk[2], rk[3]);
	rk[0] ^= rk[28];
	rk[1] ^= rk[29];
	rk[2] ^= rk[30];
	rk[3] ^= rk[31];
	x0 = p0 ^ rk[0];
	x1 = p1 ^ rk[1];
	x2 = p2 ^ rk[2];
	x3 = p3 ^ rk[3];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[4], rk[5], rk[6], rk[7]);
	rk[4] ^= rk[0];
	rk[5] ^= rk[1];
	rk[6] ^= rk[2];
	rk[7] ^= rk[3];
	x0 ^= rk[4];
	x1 ^= rk[5];
	x2 ^= rk[6];
	x3 ^= rk[7];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[8], rk[9], rk[10], rk[11]);
	rk[8] ^= rk[4];
	rk[9] ^= rk[5];
	rk[10] ^= rk[6];
	rk[11] ^= rk[7];
	x0 ^= rk[8];
	x1 ^= rk[9];
	x2 ^= rk[10];
	x3 ^= rk[11];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[12], rk[13], rk[14], rk[15]);
	rk[12] ^= rk[8];
	rk[13] ^= rk[9];
	rk[14] ^= rk[10];
	rk[15] ^= rk[11];
	x0 ^= rk[12];
	x1 ^= rk[13];
	x2 ^= rk[14];
	x3 ^= rk[15];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	pC ^= x0;
	pD ^= x1;
	pE ^= x2;
	pF ^= x3;
	KEY_EXPAND_ELT(sharedMemory, rk[16], rk[17], rk[18], rk[19]);
	rk[16] ^= rk[12];
	rk[17] ^= rk[13];
	rk[18] ^= rk[14];
	rk[19] ^= rk[15];
	x0 = p8 ^ rk[16];
	x1 = p9 ^ rk[17];
	x2 = pA ^ rk[18];
	x3 = pB ^ rk[19];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[20], rk[21], rk[22], rk[23]);
	rk[20] ^= rk[16];
	rk[21] ^= rk[17];
	rk[22] ^= rk[18];
	rk[23] ^= rk[19];
	x0 ^= rk[20];
	x1 ^= rk[21];
	x2 ^= rk[22];
	x3 ^= rk[23];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[24], rk[25], rk[26], rk[27]);
	rk[24] ^= rk[20];
	rk[25] ^= rk[21];
	rk[26] ^= rk[22];
	rk[27] ^= rk[23];
	x0 ^= rk[24];
	x1 ^= rk[25];
	x2 ^= rk[26];
	x3 ^= rk[27];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[28], rk[29], rk[30], rk[31]);
	rk[28] ^= rk[24];
	rk[29] ^= rk[25];
	rk[30] ^= rk[26];
	rk[31] ^= rk[27];
	rk[30] ^= counter;
	rk[31] ^= 0xFFFFFFFF;
	x0 ^= rk[28];
	x1 ^= rk[29];
	x2 ^= rk[30];
	x3 ^= rk[31];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p4 ^= x0;
	p5 ^= x1;
	p6 ^= x2;
	p7 ^= x3;

	rk[0] ^= rk[25];
	x0 = pC ^ rk[0];
	rk[1] ^= rk[26];
	x1 = pD ^ rk[1];
	rk[2] ^= rk[27];
	x2 = pE ^ rk[2];
	rk[3] ^= rk[28];
	x3 = pF ^ rk[3];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[4] ^= rk[29];
	x0 ^= rk[4];
	rk[5] ^= rk[30];
	x1 ^= rk[5];
	rk[6] ^= rk[31];
	x2 ^= rk[6];
	rk[7] ^= rk[0];
	x3 ^= rk[7];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[8] ^= rk[1];
	x0 ^= rk[8];
	rk[9] ^= rk[2];
	x1 ^= rk[9];
	rk[10] ^= rk[3];
	x2 ^= rk[10];
	rk[11] ^= rk[4];
	x3 ^= rk[11];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[12] ^= rk[5];
	x0 ^= rk[12];
	rk[13] ^= rk[6];
	x1 ^= rk[13];
	rk[14] ^= rk[7];
	x2 ^= rk[14];
	rk[15] ^= rk[8];
	x3 ^= rk[15];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p8 ^= x0;
	p9 ^= x1;
	pA ^= x2;
	pB ^= x3;
	rk[16] ^= rk[9];
	x0 = p4 ^ rk[16];
	rk[17] ^= rk[10];
	x1 = p5 ^ rk[17];
	rk[18] ^= rk[11];
	x2 = p6 ^ rk[18];
	rk[19] ^= rk[12];
	x3 = p7 ^ rk[19];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[20] ^= rk[13];
	x0 ^= rk[20];
	rk[21] ^= rk[14];
	x1 ^= rk[21];
	rk[22] ^= rk[15];
	x2 ^= rk[22];
	rk[23] ^= rk[16];
	x3 ^= rk[23];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[24] ^= rk[17];
	x0 ^= rk[24];
	rk[25] ^= rk[18];
	x1 ^= rk[25];
	rk[26] ^= rk[19];
	x2 ^= rk[26];
	rk[27] ^= rk[20];
	x3 ^= rk[27];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[28] ^= rk[21];
	x0 ^= rk[28];
	rk[29] ^= rk[22];
	x1 ^= rk[29];
	rk[30] ^= rk[23];
	x2 ^= rk[30];
	rk[31] ^= rk[24];
	x3 ^= rk[31];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p0 ^= x0;
	p1 ^= x1;
	p2 ^= x2;
	p3 ^= x3;


	/* round 3, 7, 11 */
	KEY_EXPAND_ELT(sharedMemory, rk[0], rk[1], rk[2], rk[3]);
	rk[0] ^= rk[28];
	rk[1] ^= rk[29];
	rk[2] ^= rk[30];
	rk[3] ^= rk[31];
	x0 = p8 ^ rk[0];
	x1 = p9 ^ rk[1];
	x2 = pA ^ rk[2];
	x3 = pB ^ rk[3];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[4], rk[5], rk[6], rk[7]);
	rk[4] ^= rk[0];
	rk[5] ^= rk[1];
	rk[6] ^= rk[2];
	rk[7] ^= rk[3];
	x0 ^= rk[4];
	x1 ^= rk[5];
	x2 ^= rk[6];
	x3 ^= rk[7];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[8], rk[9], rk[10], rk[11]);
	rk[8] ^= rk[4];
	rk[9] ^= rk[5];
	rk[10] ^= rk[6];
	rk[11] ^= rk[7];
	x0 ^= rk[8];
	x1 ^= rk[9];
	x2 ^= rk[10];
	x3 ^= rk[11];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[12], rk[13], rk[14], rk[15]);
	rk[12] ^= rk[8];
	rk[13] ^= rk[9];
	rk[14] ^= rk[10];
	rk[15] ^= rk[11];
	x0 ^= rk[12];
	x1 ^= rk[13];
	x2 ^= rk[14];
	x3 ^= rk[15];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p4 ^= x0;
	p5 ^= x1;
	p6 ^= x2;
	p7 ^= x3;
	KEY_EXPAND_ELT(sharedMemory, rk[16], rk[17], rk[18], rk[19]);
	rk[16] ^= rk[12];
	rk[17] ^= rk[13];
	rk[18] ^= rk[14];
	rk[19] ^= rk[15];
	x0 = p0 ^ rk[16];
	x1 = p1 ^ rk[17];
	x2 = p2 ^ rk[18];
	x3 = p3 ^ rk[19];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[20], rk[21], rk[22], rk[23]);
	rk[20] ^= rk[16];
	rk[21] ^= rk[17];
	rk[22] ^= rk[18];
	rk[23] ^= rk[19];
	x0 ^= rk[20];
	x1 ^= rk[21];
	x2 ^= rk[22];
	x3 ^= rk[23];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[24], rk[25], rk[26], rk[27]);
	rk[24] ^= rk[20];
	rk[25] ^= rk[21];
	rk[26] ^= rk[22];
	rk[27] ^= rk[23];
	x0 ^= rk[24];
	x1 ^= rk[25];
	x2 ^= rk[26];
	x3 ^= rk[27];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[28], rk[29], rk[30], rk[31]);
	rk[28] ^= rk[24];
	rk[29] ^= rk[25];
	rk[30] ^= rk[26];
	rk[31] ^= rk[27];
	x0 ^= rk[28];
	x1 ^= rk[29];
	x2 ^= rk[30];
	x3 ^= rk[31];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	pC ^= x0;
	pD ^= x1;
	pE ^= x2;
	pF ^= x3;
	/* round 4, 8, 12 */
	rk[0] ^= rk[25];
	x0 = p4 ^ rk[0];
	rk[1] ^= rk[26];
	x1 = p5 ^ rk[1];
	rk[2] ^= rk[27];
	x2 = p6 ^ rk[2];
	rk[3] ^= rk[28];
	x3 = p7 ^ rk[3];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[4] ^= rk[29];
	x0 ^= rk[4];
	rk[5] ^= rk[30];
	x1 ^= rk[5];
	rk[6] ^= rk[31];
	x2 ^= rk[6];
	rk[7] ^= rk[0];
	x3 ^= rk[7];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[8] ^= rk[1];
	x0 ^= rk[8];
	rk[9] ^= rk[2];
	x1 ^= rk[9];
	rk[10] ^= rk[3];
	x2 ^= rk[10];
	rk[11] ^= rk[4];
	x3 ^= rk[11];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[12] ^= rk[5];
	x0 ^= rk[12];
	rk[13] ^= rk[6];
	x1 ^= rk[13];
	rk[14] ^= rk[7];
	x2 ^= rk[14];
	rk[15] ^= rk[8];
	x3 ^= rk[15];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p0 ^= x0;
	p1 ^= x1;
	p2 ^= x2;
	p3 ^= x3;
	rk[16] ^= rk[9];
	x0 = pC ^ rk[16];
	rk[17] ^= rk[10];
	x1 = pD ^ rk[17];
	rk[18] ^= rk[11];
	x2 = pE ^ rk[18];
	rk[19] ^= rk[12];
	x3 = pF ^ rk[19];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[20] ^= rk[13];
	x0 ^= rk[20];
	rk[21] ^= rk[14];
	x1 ^= rk[21];
	rk[22] ^= rk[15];
	x2 ^= rk[22];
	rk[23] ^= rk[16];
	x3 ^= rk[23];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[24] ^= rk[17];
	x0 ^= rk[24];
	rk[25] ^= rk[18];
	x1 ^= rk[25];
	rk[26] ^= rk[19];
	x2 ^= rk[26];
	rk[27] ^= rk[20];
	x3 ^= rk[27];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk[28] ^= rk[21];
	x0 ^= rk[28];
	rk[29] ^= rk[22];
	x1 ^= rk[29];
	rk[30] ^= rk[23];
	x2 ^= rk[30];
	rk[31] ^= rk[24];
	x3 ^= rk[31];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p8 ^= x0;
	p9 ^= x1;
	pA ^= x2;
	pB ^= x3;

	/* round 13 */
	KEY_EXPAND_ELT(sharedMemory, rk[0], rk[1], rk[2], rk[3]);
	rk[0] ^= rk[28];
	rk[1] ^= rk[29];
	rk[2] ^= rk[30];
	rk[3] ^= rk[31];
	x0 = p0 ^ rk[0];
	x1 = p1 ^ rk[1];
	x2 = p2 ^ rk[2];
	x3 = p3 ^ rk[3];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[4], rk[5], rk[6], rk[7]);
	rk[4] ^= rk[0];
	rk[5] ^= rk[1];
	rk[6] ^= rk[2];
	rk[7] ^= rk[3];
	x0 ^= rk[4];
	x1 ^= rk[5];
	x2 ^= rk[6];
	x3 ^= rk[7];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[8], rk[9], rk[10], rk[11]);
	rk[8] ^= rk[4];
	rk[9] ^= rk[5];
	rk[10] ^= rk[6];
	rk[11] ^= rk[7];
	x0 ^= rk[8];
	x1 ^= rk[9];
	x2 ^= rk[10];
	x3 ^= rk[11];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[12], rk[13], rk[14], rk[15]);
	rk[12] ^= rk[8];
	rk[13] ^= rk[9];
	rk[14] ^= rk[10];
	rk[15] ^= rk[11];
	x0 ^= rk[12];
	x1 ^= rk[13];
	x2 ^= rk[14];
	x3 ^= rk[15];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	pC ^= x0;
	pD ^= x1;
	pE ^= x2;
	pF ^= x3;
	KEY_EXPAND_ELT(sharedMemory, rk[16], rk[17], rk[18], rk[19]);
	rk[16] ^= rk[12];
	rk[17] ^= rk[13];
	rk[18] ^= rk[14];
	rk[19] ^= rk[15];
	x0 = p8 ^ rk[16];
	x1 = p9 ^ rk[17];
	x2 = pA ^ rk[18];
	x3 = pB ^ rk[19];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[20], rk[21], rk[22], rk[23]);
	rk[20] ^= rk[16];
	rk[21] ^= rk[17];
	rk[22] ^= rk[18];
	rk[23] ^= rk[19];
	x0 ^= rk[20];
	x1 ^= rk[21];
	x2 ^= rk[22];
	x3 ^= rk[23];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[24], rk[25], rk[26], rk[27]);
	rk[24] ^= rk[20];
	rk[25] ^= rk[21] ^ counter;
	rk[26] ^= rk[22];
	rk[27] ^= rk[23] ^ 0xFFFFFFFF;
	x0 ^= rk[24];
	x1 ^= rk[25];
	x2 ^= rk[26];
	x3 ^= rk[27];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk[28], rk[29], rk[30], rk[31]);
	rk[28] ^= rk[24];
	rk[29] ^= rk[25];
	rk[30] ^= rk[26];
	rk[31] ^= rk[27];
	x0 ^= rk[28];
	x1 ^= rk[29];
	x2 ^= rk[30];
	x3 ^= rk[31];
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p4 ^= x0;
	p5 ^= x1;
	p6 ^= x2;
	p7 ^= x3;
	state[0x0] ^= p8;
	state[0x1] ^= p9;
	state[0x2] ^= pA;
	state[0x3] ^= pB;
	state[0x4] ^= pC;
	state[0x5] ^= pD;
	state[0x6] ^= pE;
	state[0x7] ^= pF;
	state[0x8] ^= p0;
	state[0x9] ^= p1;
	state[0xA] ^= p2;
	state[0xB] ^= p3;
	state[0xC] ^= p4;
	state[0xD] ^= p5;
	state[0xE] ^= p6;
	state[0xF] ^= p7;
}

__device__ __forceinline__
void shavite_gpu_init(uint32_t *sharedMemory)
{
	/* each thread startup will fill a uint32 */
	if (threadIdx.x < 128) {
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

// __launch_bounds__(TPB, 8)
// GPU Hash
__global__ __launch_bounds__(TPB, 8)
void x11_shavite512_gpu_hash_64(int threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
	__shared__ uint32_t sharedMemory[1024];

	shavite_gpu_init(sharedMemory);

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		int hashPosition = nounce - startNounce;
		uint32_t *Hash = (uint32_t*)&g_hash[hashPosition<<3];

		// kopiere init-state

		uint32_t msg[32] =
		{
			Hash[0], Hash[1], Hash[2], Hash[3], Hash[4], Hash[5], Hash[6], Hash[7], Hash[8], Hash[9], Hash[10], Hash[11], Hash[12], Hash[13], Hash[14], Hash[15],
			0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02000000, 0, 0, 0, 0x02000000
		};

		/*
		// nachricht laden
		uint32_t msg[32];

		// fülle die Nachricht mit 64-byte (vorheriger Hash)
		#pragma unroll 16
		for(int i=0;i<16;i++)
			msg[i] = Hash[i];

		// Nachrichtenende
		msg[16] = 0x80;
		#pragma unroll 10
		for(int i=17;i<27;i++)
			msg[i] = 0;

		msg[27] = 0x02000000;
		msg[28] = 0;
		msg[29] = 0;
		msg[30] = 0;
		msg[31] = 0x02000000;
*/
		uint32_t state[16] = {
			SPH_C32(0x72FCCDD8), SPH_C32(0x79CA4727), SPH_C32(0x128A077B), SPH_C32(0x40D55AEC),
			SPH_C32(0xD1901A06), SPH_C32(0x430AE307), SPH_C32(0xB29F5CD1), SPH_C32(0xDF07FBFC),
			SPH_C32(0x8E45D73D), SPH_C32(0x681AB538), SPH_C32(0xBDE86578), SPH_C32(0xDD577E47),
			SPH_C32(0xE275EADE), SPH_C32(0x502D9FCD), SPH_C32(0xB9357178), SPH_C32(0x022A4B9A)
		};
		c512(sharedMemory, state, msg, 512);

		#pragma unroll 16
		for(int i=0;i<16;i++)
			Hash[i] = state[i];
	}
}

__global__ __launch_bounds__(TPB, 8)
void x11_shavite512_gpu_hash_80(int threads, uint32_t startNounce, void *outputHash)
{
	__shared__ uint32_t sharedMemory[1024];

	shavite_gpu_init(sharedMemory);

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nounce = startNounce + thread;

		// kopiere init-state
		uint32_t state[16] = {
			SPH_C32(0x72FCCDD8), SPH_C32(0x79CA4727), SPH_C32(0x128A077B), SPH_C32(0x40D55AEC),
			SPH_C32(0xD1901A06), SPH_C32(0x430AE307), SPH_C32(0xB29F5CD1), SPH_C32(0xDF07FBFC),
			SPH_C32(0x8E45D73D), SPH_C32(0x681AB538), SPH_C32(0xBDE86578), SPH_C32(0xDD577E47),
			SPH_C32(0xE275EADE), SPH_C32(0x502D9FCD), SPH_C32(0xB9357178), SPH_C32(0x022A4B9A)
		};

		uint32_t msg[32];

		#pragma unroll 32
		for(int i=0;i<32;i++) {
			msg[i] = c_PaddedMessage80[i];
		}
		msg[19] = cuda_swab32(nounce);
		msg[20] = 0x80;
		msg[27] = 0x2800000;
		msg[31] = 0x2000000;

		c512(sharedMemory, state, msg, 640);

		uint32_t *outHash = (uint32_t *)outputHash + 16 * thread;

		#pragma unroll 16
		for(int i=0;i<16;i++)
			outHash[i] = state[i];

	} //thread < threads
}

__host__ void x11_shavite512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const int threadsperblock = TPB;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 4096;

	x11_shavite512_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
	MyStreamSynchronize(NULL, order, thr_id);
}

__host__ void x11_shavite512_cpu_hash_80(int thr_id, int threads, uint32_t startNounce, uint32_t *d_outputHash, int order)
{
	const int threadsperblock = TPB;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 4096;

	x11_shavite512_gpu_hash_80<<<grid, block, shared_size>>>(threads, startNounce, d_outputHash);
	MyStreamSynchronize(NULL, order, thr_id);
}

__host__ void x11_shavite512_setBlock_80(void *pdata)
{
	// Message mit Padding bereitstellen
	// lediglich die korrekte Nonce ist noch ab Byte 76 einzusetzen.
	unsigned char PaddedMessage[128];
	memcpy(PaddedMessage, pdata, 80);
	memset(PaddedMessage+80, 0, 48);

	cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, 32*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

