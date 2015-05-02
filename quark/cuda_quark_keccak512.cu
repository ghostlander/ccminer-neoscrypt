#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

#ifdef _MSC_VER
#define UINT2(x,y) { x, y }
#else
#define UINT2(x,y) (uint2) { x, y }
#endif

__constant__ uint2 c_keccak_round_constants35[24] = {
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
#define bitselect(a, b, c) ((a) ^ ((c) & ((b) ^ (a))))

__global__  __launch_bounds__(128, 4)
void quark_keccak512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint2 *g_hash, uint32_t *g_nonceVector)
{
    uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

        int hashPosition = nounce - startNounce;
        uint2 *inpHash = &g_hash[8 * hashPosition];

        uint2 s[25];
		uint2 bc[5], tmpxor[5], tmp1, tmp2;

		tmpxor[0] = inpHash[0] ^ inpHash[5];
		tmpxor[1] = inpHash[1] ^ inpHash[6];
		tmpxor[2] = inpHash[2] ^ inpHash[7];
		tmpxor[3] = inpHash[3] ^ make_uint2(0x1, 0x80000000);
		tmpxor[4] = inpHash[4];

		bc[0] = tmpxor[0] ^ ROL2(tmpxor[2], 1);
		bc[1] = tmpxor[1] ^ ROL2(tmpxor[3], 1);
		bc[2] = tmpxor[2] ^ ROL2(tmpxor[4], 1);
		bc[3] = tmpxor[3] ^ ROL2(tmpxor[0], 1);
		bc[4] = tmpxor[4] ^ ROL2(tmpxor[1], 1);

		s[0] = inpHash[0] ^ bc[4];
		s[1] = ROL2(inpHash[6] ^ bc[0], 44);
		s[6] = ROL2(bc[3], 20);
		s[9] = ROL2(bc[1], 61);
		s[22] = ROL2(bc[3], 39);
		s[14] = ROL2(bc[4], 18);
		s[20] = ROL2(inpHash[2] ^ bc[1], 62);
		s[2] = ROL2(bc[1], 43);
		s[12] = ROL2(bc[2], 25);
		s[13] = ROL2(bc[3], 8);
		s[19] = ROL2(bc[2], 56);
		s[23] = ROL2(bc[4], 41);
		s[15] = ROL2(inpHash[4] ^ bc[3], 27);
		s[4] = ROL2(bc[3], 14);
		s[24] = ROL2(bc[0], 2);
		s[21] = ROL2(make_uint2(0x1, 0x80000000) ^ bc[2], 55);
		s[8] = ROL2(bc[0], 45);
		s[16] = ROL2(inpHash[5] ^ bc[4], 36);
		s[5] = ROL2(inpHash[3] ^ bc[2], 28);
		s[3] = ROL2(bc[2], 21);
		s[18] = ROL2(bc[1], 15);
		s[17] = ROL2(bc[0], 10);
		s[11] = ROL2(inpHash[7] ^ bc[1], 6);
		s[7] = ROL2(bc[4], 3);
		s[10] = ROL2(inpHash[1] ^ bc[0], 1);

		tmp1 = s[0]; tmp2 = s[1]; s[0] = bitselect(s[0] ^ s[2], s[0], s[1]); s[1] = bitselect(s[1] ^ s[3], s[1], s[2]); s[2] = bitselect(s[2] ^ s[4], s[2], s[3]); s[3] = bitselect(s[3] ^ tmp1, s[3], s[4]); s[4] = bitselect(s[4] ^ tmp2, s[4], tmp1);
		tmp1 = s[5]; tmp2 = s[6]; s[5] = bitselect(s[5] ^ s[7], s[5], s[6]); s[6] = bitselect(s[6] ^ s[8], s[6], s[7]); s[7] = bitselect(s[7] ^ s[9], s[7], s[8]); s[8] = bitselect(s[8] ^ tmp1, s[8], s[9]); s[9] = bitselect(s[9] ^ tmp2, s[9], tmp1);
		tmp1 = s[10]; tmp2 = s[11]; s[10] = bitselect(s[10] ^ s[12], s[10], s[11]); s[11] = bitselect(s[11] ^ s[13], s[11], s[12]); s[12] = bitselect(s[12] ^ s[14], s[12], s[13]); s[13] = bitselect(s[13] ^ tmp1, s[13], s[14]); s[14] = bitselect(s[14] ^ tmp2, s[14], tmp1);
		tmp1 = s[15]; tmp2 = s[16]; s[15] = bitselect(s[15] ^ s[17], s[15], s[16]); s[16] = bitselect(s[16] ^ s[18], s[16], s[17]); s[17] = bitselect(s[17] ^ s[19], s[17], s[18]); s[18] = bitselect(s[18] ^ tmp1, s[18], s[19]); s[19] = bitselect(s[19] ^ tmp2, s[19], tmp1);
		tmp1 = s[20]; tmp2 = s[21]; s[20] = bitselect(s[20] ^ s[22], s[20], s[21]); s[21] = bitselect(s[21] ^ s[23], s[21], s[22]); s[22] = bitselect(s[22] ^ s[24], s[22], s[23]); s[23] = bitselect(s[23] ^ tmp1, s[23], s[24]); s[24] = bitselect(s[24] ^ tmp2, s[24], tmp1);
		s[0].x ^= 1;

#pragma unroll 2
		for (int i = 1; i < 24; ++i)
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
			s[13] = ROL2(s[19] ^ bc[3], 8);
			s[19] = ROL2(s[23] ^ bc[2], 56);
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
			tmp1 = s[5]; tmp2 = s[6]; s[5] = bitselect(s[5] ^ s[7], s[5], s[6]); s[6] = bitselect(s[6] ^ s[8], s[6], s[7]); s[7] = bitselect(s[7] ^ s[9], s[7], s[8]); s[8] = bitselect(s[8] ^ tmp1, s[8], s[9]); s[9] = bitselect(s[9] ^ tmp2, s[9], tmp1);
			tmp1 = s[10]; tmp2 = s[11]; s[10] = bitselect(s[10] ^ s[12], s[10], s[11]); s[11] = bitselect(s[11] ^ s[13], s[11], s[12]); s[12] = bitselect(s[12] ^ s[14], s[12], s[13]); s[13] = bitselect(s[13] ^ tmp1, s[13], s[14]); s[14] = bitselect(s[14] ^ tmp2, s[14], tmp1);
			tmp1 = s[15]; tmp2 = s[16]; s[15] = bitselect(s[15] ^ s[17], s[15], s[16]); s[16] = bitselect(s[16] ^ s[18], s[16], s[17]); s[17] = bitselect(s[17] ^ s[19], s[17], s[18]); s[18] = bitselect(s[18] ^ tmp1, s[18], s[19]); s[19] = bitselect(s[19] ^ tmp2, s[19], tmp1);
			tmp1 = s[20]; tmp2 = s[21]; s[20] = bitselect(s[20] ^ s[22], s[20], s[21]); s[21] = bitselect(s[21] ^ s[23], s[21], s[22]); s[22] = bitselect(s[22] ^ s[24], s[22], s[23]); s[23] = bitselect(s[23] ^ tmp1, s[23], s[24]); s[24] = bitselect(s[24] ^ tmp2, s[24], tmp1);
			s[0] ^= c_keccak_round_constants35[i];
		}

#pragma unroll
        for(int i=0;i<8;i++)
			inpHash[i] = s[i];
    }
}

__global__  __launch_bounds__(192, 4)
void quark_keccak512_gpu_hash_64_final(uint32_t threads, uint32_t startNounce, uint2 *g_hash, uint32_t *g_nonceVector)
{
    const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		int hashPosition = nounce - startNounce;
		uint2 *inpHash = &g_hash[8 * hashPosition];

		uint2 s[25];
		uint2 bc[5], tmpxor[5], tmp1, tmp2;

		tmpxor[0] = inpHash[0] ^ inpHash[5];
		tmpxor[1] = inpHash[1] ^ inpHash[6];
		tmpxor[2] = inpHash[2] ^ inpHash[7];
		tmpxor[3] = inpHash[3] ^ make_uint2(0x1, 0x80000000);
		tmpxor[4] = inpHash[4];

		bc[0] = tmpxor[0] ^ ROL2(tmpxor[2], 1);
		bc[1] = tmpxor[1] ^ ROL2(tmpxor[3], 1);
		bc[2] = tmpxor[2] ^ ROL2(tmpxor[4], 1);
		bc[3] = tmpxor[3] ^ ROL2(tmpxor[0], 1);
		bc[4] = tmpxor[4] ^ ROL2(tmpxor[1], 1);

		s[0] = inpHash[0] ^ bc[4];
		s[1] = ROL2(inpHash[6] ^ bc[0], 44);
		s[6] = ROL2(bc[3], 20);
		s[9] = ROL2(bc[1], 61);
		s[22] = ROL2(bc[3], 39);
		s[14] = ROL2(bc[4], 18);
		s[20] = ROL2(inpHash[2] ^ bc[1], 62);
		s[2] = ROL2(bc[1], 43);
		s[12] = ROL2(bc[2], 25);
		s[13] = ROL2(bc[3], 8);
		s[19] = ROL2(bc[2], 56);
		s[23] = ROL2(bc[4], 41);
		s[15] = ROL2(inpHash[4] ^ bc[3], 27);
		s[4] = ROL2(bc[3], 14);
		s[24] = ROL2(bc[0], 2);
		s[21] = ROL2(make_uint2(0x1, 0x80000000) ^ bc[2], 55);
		s[8] = ROL2(bc[0], 45);
		s[16] = ROL2(inpHash[5] ^ bc[4], 36);
		s[5] = ROL2(inpHash[3] ^ bc[2], 28);
		s[3] = ROL2(bc[2], 21);
		s[18] = ROL2(bc[1], 15);
		s[17] = ROL2(bc[0], 10);
		s[11] = ROL2(inpHash[7] ^ bc[1], 6);
		s[7] = ROL2(bc[4], 3);
		s[10] = ROL2(inpHash[1] ^ bc[0], 1);

		tmp1 = s[0]; tmp2 = s[1]; s[0] = bitselect(s[0] ^ s[2], s[0], s[1]); s[1] = bitselect(s[1] ^ s[3], s[1], s[2]); s[2] = bitselect(s[2] ^ s[4], s[2], s[3]); s[3] = bitselect(s[3] ^ tmp1, s[3], s[4]); s[4] = bitselect(s[4] ^ tmp2, s[4], tmp1);
		tmp1 = s[5]; tmp2 = s[6]; s[5] = bitselect(s[5] ^ s[7], s[5], s[6]); s[6] = bitselect(s[6] ^ s[8], s[6], s[7]); s[7] = bitselect(s[7] ^ s[9], s[7], s[8]); s[8] = bitselect(s[8] ^ tmp1, s[8], s[9]); s[9] = bitselect(s[9] ^ tmp2, s[9], tmp1);
		tmp1 = s[10]; tmp2 = s[11]; s[10] = bitselect(s[10] ^ s[12], s[10], s[11]); s[11] = bitselect(s[11] ^ s[13], s[11], s[12]); s[12] = bitselect(s[12] ^ s[14], s[12], s[13]); s[13] = bitselect(s[13] ^ tmp1, s[13], s[14]); s[14] = bitselect(s[14] ^ tmp2, s[14], tmp1);
		tmp1 = s[15]; tmp2 = s[16]; s[15] = bitselect(s[15] ^ s[17], s[15], s[16]); s[16] = bitselect(s[16] ^ s[18], s[16], s[17]); s[17] = bitselect(s[17] ^ s[19], s[17], s[18]); s[18] = bitselect(s[18] ^ tmp1, s[18], s[19]); s[19] = bitselect(s[19] ^ tmp2, s[19], tmp1);
		tmp1 = s[20]; tmp2 = s[21]; s[20] = bitselect(s[20] ^ s[22], s[20], s[21]); s[21] = bitselect(s[21] ^ s[23], s[21], s[22]); s[22] = bitselect(s[22] ^ s[24], s[22], s[23]); s[23] = bitselect(s[23] ^ tmp1, s[23], s[24]); s[24] = bitselect(s[24] ^ tmp2, s[24], tmp1);
		s[0].x ^= 1;

#pragma unroll 2
		for (int i = 1; i < 23; i++)
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
			s[13] = ROL2(s[19] ^ bc[3], 8);
			s[19] = ROL2(s[23] ^ bc[2], 56);
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
			tmp1 = s[5]; tmp2 = s[6]; s[5] = bitselect(s[5] ^ s[7], s[5], s[6]); s[6] = bitselect(s[6] ^ s[8], s[6], s[7]); s[7] = bitselect(s[7] ^ s[9], s[7], s[8]); s[8] = bitselect(s[8] ^ tmp1, s[8], s[9]); s[9] = bitselect(s[9] ^ tmp2, s[9], tmp1);
			tmp1 = s[10]; tmp2 = s[11]; s[10] = bitselect(s[10] ^ s[12], s[10], s[11]); s[11] = bitselect(s[11] ^ s[13], s[11], s[12]); s[12] = bitselect(s[12] ^ s[14], s[12], s[13]); s[13] = bitselect(s[13] ^ tmp1, s[13], s[14]); s[14] = bitselect(s[14] ^ tmp2, s[14], tmp1);
			tmp1 = s[15]; tmp2 = s[16]; s[15] = bitselect(s[15] ^ s[17], s[15], s[16]); s[16] = bitselect(s[16] ^ s[18], s[16], s[17]); s[17] = bitselect(s[17] ^ s[19], s[17], s[18]); s[18] = bitselect(s[18] ^ tmp1, s[18], s[19]); s[19] = bitselect(s[19] ^ tmp2, s[19], tmp1);
			tmp1 = s[20]; tmp2 = s[21]; s[20] = bitselect(s[20] ^ s[22], s[20], s[21]); s[21] = bitselect(s[21] ^ s[23], s[21], s[22]); s[22] = bitselect(s[22] ^ s[24], s[22], s[23]); s[23] = bitselect(s[23] ^ tmp1, s[23], s[24]); s[24] = bitselect(s[24] ^ tmp2, s[24], tmp1);
			s[0] ^= c_keccak_round_constants35[i];
		}
		uint2 t[5];
		t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		s[0] ^= t[4] ^ ROL2(t[1], 1);
		s[18] ^= t[2] ^ ROL2(t[4], 1);
		s[24] ^= t[3] ^ ROL2(t[0], 1);

		s[3] = ROL2(s[18], 21) ^ ((~ROL2(s[24], 14)) & s[0]);

		inpHash[3] = s[3];
	}
}

__host__ void quark_keccak512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash)
{
    const uint32_t threadsperblock = 128;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    quark_keccak512_gpu_hash_64<<<grid, block>>>(threads, startNounce, (uint2 *)d_hash, d_nonceVector);
}

__host__ void quark_keccak512_cpu_hash_64_final(uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 32;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	quark_keccak512_gpu_hash_64_final << <grid, block >> >(threads, startNounce, (uint2 *)d_hash, d_nonceVector);
}
