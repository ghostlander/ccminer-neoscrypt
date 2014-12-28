#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

// aus heavy.cu
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

// die Message it Padding zur Berechnung auf der GPU
__constant__ uint64_t c_PaddedMessage80[16]; // padded message (80 bytes + padding)

//#define SHL(x, n)            ((x) << (n))
//#define SHR(x, n)            ((x) >> (n))
#define SHR(x, n) SHR2(x, n) 
#define SHL(x, n) SHL2(x, n) 

#undef ROTL64
#define ROTL64 ROL2


#define CONST_EXP2    q[i+0] + ROTL64(q[i+1], 5)  + q[i+2] + ROTL64(q[i+3], 11) + \
                    q[i+4] + ROTL64(q[i+5], 27) + q[i+6] + SWAPDWORDS2(q[i+7]) + \
                    q[i+8] + ROTL64(q[i+9], 37) + q[i+10] + ROTL64(q[i+11], 43) + \
                    q[i+12] + ROTL64(q[i+13], 53) + (SHR(q[i+14],1) ^ q[i+14]) + (SHR(q[i+15],2) ^ q[i+15])

__device__ void Compression512_64_first(uint2 *msg, uint2 *hash)
{
	// Compression ref. implementation
	uint2 q[32];
	uint2 tmp;

	tmp = (msg[5] ^ hash[5]) - (msg[7] ^ hash[7]) + (hash[10]) + (hash[13]) + (hash[14]);
	q[0] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + hash[1];
	tmp = (msg[6] ^ hash[6]) - (msg[8] ^ hash[8]) + (hash[11]) + (hash[14]) - (msg[15] ^ hash[15]);
	q[1] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + hash[2];
	tmp = (msg[0] ^ hash[0]) + (msg[7] ^ hash[7]) + (hash[9]) - (hash[12]) + (msg[15] ^ hash[15]);
	q[2] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + hash[3];
	tmp = (msg[0] ^ hash[0]) - (msg[1] ^ hash[1]) + (msg[8] ^ hash[8]) - (hash[10]) + (hash[13]);
	q[3] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + hash[4];
	tmp = (msg[1] ^ hash[1]) + (msg[2] ^ hash[2]) + (hash[9]) - (hash[11]) - (hash[14]);
	q[4] = (SHR(tmp, 1) ^ tmp) + hash[5];
	tmp = (msg[3] ^ hash[3]) - (msg[2] ^ hash[2]) + (hash[10]) - (hash[12]) + (msg[15] ^ hash[15]);
	q[5] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + hash[6];
	tmp = (msg[4] ^ hash[4]) - (msg[0] ^ hash[0]) - (msg[3] ^ hash[3]) - (hash[11]) + (hash[13]);
	q[6] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + hash[7];
	tmp = (msg[1] ^ hash[1]) - (msg[4] ^ hash[4]) - (msg[5] ^ hash[5]) - (hash[12]) - (hash[14]);
	q[7] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + hash[8];
	tmp = (msg[2] ^ hash[2]) - (msg[5] ^ hash[5]) - (msg[6] ^ hash[6]) + (hash[13]) - (msg[15] ^ hash[15]);
	q[8] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + hash[9];
	tmp = (msg[0] ^ hash[0]) - (msg[3] ^ hash[3]) + (msg[6] ^ hash[6]) - (msg[7] ^ hash[7]) + (hash[14]);
	q[9] = (SHR(tmp, 1) ^ tmp) + hash[10];
	tmp = (msg[8] ^ hash[8]) - (msg[1] ^ hash[1]) - (msg[4] ^ hash[4]) - (msg[7] ^ hash[7]) + (msg[15] ^ hash[15]);
	q[10] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + hash[11];
	tmp = (msg[8] ^ hash[8]) - (msg[0] ^ hash[0]) - (msg[2] ^ hash[2]) - (msg[5] ^ hash[5]) + (hash[9]);
	q[11] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + hash[12];
	tmp = (msg[1] ^ hash[1]) + (msg[3] ^ hash[3]) - (msg[6] ^ hash[6]) - (hash[9]) + (hash[10]);
	q[12] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + hash[13];
	tmp = (msg[2] ^ hash[2]) + (msg[4] ^ hash[4]) + (msg[7] ^ hash[7]) + (hash[10]) + (hash[11]);
	q[13] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + hash[14];
	tmp = (msg[3] ^ hash[3]) - (msg[5] ^ hash[5]) + (msg[8] ^ hash[8]) - (hash[11]) - (hash[12]);
	q[14] = (SHR(tmp, 1) ^ tmp) + hash[15];
	tmp = (msg[12] ^ hash[12]) - (msg[4] ^ hash[4]) - (msg[6] ^ hash[6]) - (hash[9]) + (hash[13]);
	q[15] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + hash[0];

	// Expand 1
#pragma unroll 2
	for (int i = 0; i<2; i++)
	{
		q[i + 16] =
			(SHR(q[i], 1) ^ SHL(q[i], 2) ^ ROTL64(q[i], 13) ^ ROTL64(q[i], 43)) +
			(SHR(q[i + 1], 2) ^ SHL(q[i + 1], 1) ^ ROTL64(q[i + 1], 19) ^ ROTL64(q[i + 1], 53)) +
			(SHR(q[i + 2], 2) ^ SHL(q[i + 2], 2) ^ ROTL64(q[i + 2], 28) ^ ROTL64(q[i + 2], 59)) +
			(SHR(q[i + 3], 1) ^ SHL(q[i + 3], 3) ^ ROTL64(q[i + 3], 4) ^ ROTL64(q[i + 3], 37)) +
			(SHR(q[i + 4], 1) ^ SHL(q[i + 4], 2) ^ ROTL64(q[i + 4], 13) ^ ROTL64(q[i + 4], 43)) +
			(SHR(q[i + 5], 2) ^ SHL(q[i + 5], 1) ^ ROTL64(q[i + 5], 19) ^ ROTL64(q[i + 5], 53)) +
			(SHR(q[i + 6], 2) ^ SHL(q[i + 6], 2) ^ ROTL64(q[i + 6], 28) ^ ROTL64(q[i + 6], 59)) +
			(SHR(q[i + 7], 1) ^ SHL(q[i + 7], 3) ^ ROTL64(q[i + 7], 4) ^ ROTL64(q[i + 7], 37)) +
			(SHR(q[i + 8], 1) ^ SHL(q[i + 8], 2) ^ ROTL64(q[i + 8], 13) ^ ROTL64(q[i + 8], 43)) +
			(SHR(q[i + 9], 2) ^ SHL(q[i + 9], 1) ^ ROTL64(q[i + 9], 19) ^ ROTL64(q[i + 9], 53)) +
			(SHR(q[i + 10], 2) ^ SHL(q[i + 10], 2) ^ ROTL64(q[i + 10], 28) ^ ROTL64(q[i + 10], 59)) +
			(SHR(q[i + 11], 1) ^ SHL(q[i + 11], 3) ^ ROTL64(q[i + 11], 4) ^ ROTL64(q[i + 11], 37)) +
			(SHR(q[i + 12], 1) ^ SHL(q[i + 12], 2) ^ ROTL64(q[i + 12], 13) ^ ROTL64(q[i + 12], 43)) +
			(SHR(q[i + 13], 2) ^ SHL(q[i + 13], 1) ^ ROTL64(q[i + 13], 19) ^ ROTL64(q[i + 13], 53)) +
			(SHR(q[i + 14], 2) ^ SHL(q[i + 14], 2) ^ ROTL64(q[i + 14], 28) ^ ROTL64(q[i + 14], 59)) +
			(SHR(q[i + 15], 1) ^ SHL(q[i + 15], 3) ^ ROTL64(q[i + 15], 4) ^ ROTL64(q[i + 15], 37)) +
			((vectorize((i + 16)*(0x0555555555555555ull)) + ROTL64(msg[i], i + 1) +
			ROTL64(msg[i + 3], i + 4)) ^ hash[i + 7]);
	}

#pragma unroll 4
	for (int i = 2; i<6; i++) {
		q[i + 16] = CONST_EXP2 +
			((vectorize((i + 16)*(0x0555555555555555ull)) + ROTL64(msg[i], i + 1) +
			ROTL64(msg[i + 3], i + 4) - ROTL64(msg[i + 10], i + 11)) ^ hash[i + 7]);
	}
#pragma unroll 3
	for (int i = 6; i<9; i++) {
		q[i + 16] = CONST_EXP2 +
			((vectorize((i + 16)*(0x0555555555555555ull)) + ROTL64(msg[i], i + 1) -
			ROTL64(msg[i - 6], (i - 6) + 1)) ^ hash[i + 7]);
	}
#pragma unroll 4
	for (int i = 9; i<13; i++) {
		q[i + 16] = CONST_EXP2 +
			((vectorize((i + 16)*(0x0555555555555555ull)) +
			ROTL64(msg[i + 3], i + 4) - ROTL64(msg[i - 6], (i - 6) + 1)) ^ hash[i - 9]);
	}
#pragma unroll 3
	for (int i = 13; i<16; i++) {
		q[i + 16] = CONST_EXP2 +
			((vectorize((i + 16)*(0x0555555555555555ull)) + ROTL64(msg[i], i + 1) +
			ROTL64(msg[i - 13], (i - 13) + 1) - ROTL64(msg[i - 6], (i - 6) + 1)) ^ hash[i - 9]);
	}

	uint2 XL64 = q[16] ^ q[17] ^ q[18] ^ q[19] ^ q[20] ^ q[21] ^ q[22] ^ q[23];
	uint2 XH64 = XL64^q[24] ^ q[25] ^ q[26] ^ q[27] ^ q[28] ^ q[29] ^ q[30] ^ q[31];

	hash[0] = (SHL(XH64, 5) ^ SHR(q[16], 5) ^ msg[0]) + (XL64    ^ q[24] ^ q[0]);
	hash[1] = (SHR(XH64, 7) ^ SHL(q[17], 8) ^ msg[1]) + (XL64    ^ q[25] ^ q[1]);
	hash[2] = (SHR(XH64, 5) ^ SHL(q[18], 5) ^ msg[2]) + (XL64    ^ q[26] ^ q[2]);
	hash[3] = (SHR(XH64, 1) ^ SHL(q[19], 5) ^ msg[3]) + (XL64    ^ q[27] ^ q[3]);
	hash[4] = (SHR(XH64, 3) ^ q[20] ^ msg[4]) + (XL64    ^ q[28] ^ q[4]);
	hash[5] = (SHL(XH64, 6) ^ SHR(q[21], 6) ^ msg[5]) + (XL64    ^ q[29] ^ q[5]);
	hash[6] = (SHR(XH64, 4) ^ SHL(q[22], 6) ^ msg[6]) + (XL64    ^ q[30] ^ q[6]);
	hash[7] = (SHR(XH64, 11) ^ SHL(q[23], 2) ^ msg[7]) + (XL64    ^ q[31] ^ q[7]);

	hash[8] = ROTL64(hash[4], 9) + (XH64     ^     q[24] ^ msg[8]) + (SHL(XL64, 8) ^ q[23] ^ q[8]);
	hash[9] = ROTL64(hash[5], 10) + (XH64     ^     q[25]) + (SHR(XL64, 6) ^ q[16] ^ q[9]);
	hash[10] = ROTL64(hash[6], 11) + (XH64     ^     q[26]) + (SHL(XL64, 6) ^ q[17] ^ q[10]);
	hash[11] = ROTL64(hash[7], 12) + (XH64     ^     q[27]) + (SHL(XL64, 4) ^ q[18] ^ q[11]);
	hash[12] = ROTL64(hash[0], 13) + (XH64     ^     q[28]) + (SHR(XL64, 3) ^ q[19] ^ q[12]);
	hash[13] = ROTL64(hash[1], 14) + (XH64     ^     q[29]) + (SHR(XL64, 4) ^ q[20] ^ q[13]);
	hash[14] = ROTL64(hash[2], 15) + (XH64     ^     q[30]) + (SHR(XL64, 7) ^ q[21] ^ q[14]);
	hash[15] = ROTL64(hash[3], 16) + (XH64     ^     q[31] ^ msg[15]) + (SHR(XL64, 2) ^ q[22] ^ q[15]);
}

__device__ void Compression512(uint2 *msg, uint2 *hash)
{
    // Compression ref. implementation
	uint2 q[32];
	uint2 tmp;

    tmp = (msg[ 5] ^ hash[ 5]) - (msg[ 7] ^ hash[ 7]) + (msg[10] ^ hash[10]) + (msg[13] ^ hash[13]) + (msg[14] ^ hash[14]);
    q[0] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp,  4) ^ ROTL64(tmp, 37)) + hash[1];
    tmp = (msg[ 6] ^ hash[ 6]) - (msg[ 8] ^ hash[ 8]) + (msg[11] ^ hash[11]) + (msg[14] ^ hash[14]) - (msg[15] ^ hash[15]);
    q[1] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + hash[2];
    tmp = (msg[ 0] ^ hash[ 0]) + (msg[ 7] ^ hash[ 7]) + (msg[ 9] ^ hash[ 9]) - (msg[12] ^ hash[12]) + (msg[15] ^ hash[15]);
    q[2] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + hash[3];
    tmp = (msg[ 0] ^ hash[ 0]) - (msg[ 1] ^ hash[ 1]) + (msg[ 8] ^ hash[ 8]) - (msg[10] ^ hash[10]) + (msg[13] ^ hash[13]);
    q[3] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + hash[4];
    tmp = (msg[ 1] ^ hash[ 1]) + (msg[ 2] ^ hash[ 2]) + (msg[ 9] ^ hash[ 9]) - (msg[11] ^ hash[11]) - (msg[14] ^ hash[14]);
    q[4] = (SHR(tmp, 1) ^ tmp) + hash[5];
    tmp = (msg[ 3] ^ hash[ 3]) - (msg[ 2] ^ hash[ 2]) + (msg[10] ^ hash[10]) - (msg[12] ^ hash[12]) + (msg[15] ^ hash[15]);
    q[5] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp,  4) ^ ROTL64(tmp, 37)) + hash[6];
    tmp = (msg[ 4] ^ hash[ 4]) - (msg[ 0] ^ hash[ 0]) - (msg[ 3] ^ hash[ 3]) - (msg[11] ^ hash[11]) + (msg[13] ^ hash[13]);
    q[6] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + hash[7];
    tmp = (msg[ 1] ^ hash[ 1]) - (msg[ 4] ^ hash[ 4]) - (msg[ 5] ^ hash[ 5]) - (msg[12] ^ hash[12]) - (msg[14] ^ hash[14]);
    q[7] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + hash[8];
    tmp = (msg[ 2] ^ hash[ 2]) - (msg[ 5] ^ hash[ 5]) - (msg[ 6] ^ hash[ 6]) + (msg[13] ^ hash[13]) - (msg[15] ^ hash[15]);
    q[8] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + hash[9];
    tmp = (msg[ 0] ^ hash[ 0]) - (msg[ 3] ^ hash[ 3]) + (msg[ 6] ^ hash[ 6]) - (msg[ 7] ^ hash[ 7]) + (msg[14] ^ hash[14]);
    q[9] = (SHR(tmp, 1) ^ tmp) + hash[10];
    tmp = (msg[ 8] ^ hash[ 8]) - (msg[ 1] ^ hash[ 1]) - (msg[ 4] ^ hash[ 4]) - (msg[ 7] ^ hash[ 7]) + (msg[15] ^ hash[15]);
    q[10] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp,  4) ^ ROTL64(tmp, 37)) + hash[11];
    tmp = (msg[ 8] ^ hash[ 8]) - (msg[ 0] ^ hash[ 0]) - (msg[ 2] ^ hash[ 2]) - (msg[ 5] ^ hash[ 5]) + (msg[ 9] ^ hash[ 9]);
    q[11] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + hash[12];
    tmp = (msg[ 1] ^ hash[ 1]) + (msg[ 3] ^ hash[ 3]) - (msg[ 6] ^ hash[ 6]) - (msg[ 9] ^ hash[ 9]) + (msg[10] ^ hash[10]);
    q[12] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + hash[13];
    tmp = (msg[ 2] ^ hash[ 2]) + (msg[ 4] ^ hash[ 4]) + (msg[ 7] ^ hash[ 7]) + (msg[10] ^ hash[10]) + (msg[11] ^ hash[11]);
    q[13] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + hash[14];
    tmp = (msg[ 3] ^ hash[ 3]) - (msg[ 5] ^ hash[ 5]) + (msg[ 8] ^ hash[ 8]) - (msg[11] ^ hash[11]) - (msg[12] ^ hash[12]);
    q[14] = (SHR(tmp, 1) ^ tmp) + hash[15];
    tmp = (msg[12] ^ hash[12]) - (msg[ 4] ^ hash[ 4]) - (msg[ 6] ^ hash[ 6]) - (msg[ 9] ^ hash[ 9]) + (msg[13] ^ hash[13]);
    q[15] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + hash[0];

    // Expand 1
#pragma unroll 2
    for(int i=0;i<2;i++)
    {
        q[i+16] =
        (SHR(q[i], 1) ^ SHL(q[i], 2) ^ ROTL64(q[i], 13) ^ ROTL64(q[i], 43)) +
        (SHR(q[i+1], 2) ^ SHL(q[i+1], 1) ^ ROTL64(q[i+1], 19) ^ ROTL64(q[i+1], 53)) +
        (SHR(q[i+2], 2) ^ SHL(q[i+2], 2) ^ ROTL64(q[i+2], 28) ^ ROTL64(q[i+2], 59)) +
        (SHR(q[i+3], 1) ^ SHL(q[i+3], 3) ^ ROTL64(q[i+3],  4) ^ ROTL64(q[i+3], 37)) +
        (SHR(q[i+4], 1) ^ SHL(q[i+4], 2) ^ ROTL64(q[i+4], 13) ^ ROTL64(q[i+4], 43)) +
        (SHR(q[i+5], 2) ^ SHL(q[i+5], 1) ^ ROTL64(q[i+5], 19) ^ ROTL64(q[i+5], 53)) +
        (SHR(q[i+6], 2) ^ SHL(q[i+6], 2) ^ ROTL64(q[i+6], 28) ^ ROTL64(q[i+6], 59)) +
        (SHR(q[i+7], 1) ^ SHL(q[i+7], 3) ^ ROTL64(q[i+7],  4) ^ ROTL64(q[i+7], 37)) +
        (SHR(q[i+8], 1) ^ SHL(q[i+8], 2) ^ ROTL64(q[i+8], 13) ^ ROTL64(q[i+8], 43)) +
        (SHR(q[i+9], 2) ^ SHL(q[i+9], 1) ^ ROTL64(q[i+9], 19) ^ ROTL64(q[i+9], 53)) +
        (SHR(q[i+10], 2) ^ SHL(q[i+10], 2) ^ ROTL64(q[i+10], 28) ^ ROTL64(q[i+10], 59)) +
        (SHR(q[i+11], 1) ^ SHL(q[i+11], 3) ^ ROTL64(q[i+11],  4) ^ ROTL64(q[i+11], 37)) +
        (SHR(q[i+12], 1) ^ SHL(q[i+12], 2) ^ ROTL64(q[i+12], 13) ^ ROTL64(q[i+12], 43)) +
        (SHR(q[i+13], 2) ^ SHL(q[i+13], 1) ^ ROTL64(q[i+13], 19) ^ ROTL64(q[i+13], 53)) +
        (SHR(q[i+14], 2) ^ SHL(q[i+14], 2) ^ ROTL64(q[i+14], 28) ^ ROTL64(q[i+14], 59)) +
        (SHR(q[i+15], 1) ^ SHL(q[i+15], 3) ^ ROTL64(q[i+15],  4) ^ ROTL64(q[i+15], 37)) +
        ((    vectorize((i+16)*(0x0555555555555555ull)) + ROTL64(msg[i], i+1) +
            ROTL64(msg[i+3], i+4) - ROTL64(msg[i+10], i+11) ) ^ hash[i+7]);
    }

#pragma unroll 4
    for(int i=2;i<6;i++) {
        q[i+16] = CONST_EXP2 + 
        ((    vectorize((i+16)*(0x0555555555555555ull)) + ROTL64(msg[i], i+1) +
            ROTL64(msg[i+3], i+4) - ROTL64(msg[i+10], i+11) ) ^ hash[i+7]);
    }
#pragma unroll 3
    for(int i=6;i<9;i++) {
        q[i+16] = CONST_EXP2 + 
			((vectorize((i + 16)*(0x0555555555555555ull)) + ROTL64(msg[i], i + 1) +
            ROTL64(msg[i+3], i+4) - ROTL64(msg[i-6], (i-6)+1) ) ^ hash[i+7]);
    }
#pragma unroll 4
    for(int i=9;i<13;i++) {
        q[i+16] = CONST_EXP2 + 
			((vectorize((i + 16)*(0x0555555555555555ull)) + ROTL64(msg[i], i + 1) +
            ROTL64(msg[i+3], i+4) - ROTL64(msg[i-6], (i-6)+1) ) ^ hash[i-9]);
    }
#pragma unroll 3
    for(int i=13;i<16;i++) {
        q[i+16] = CONST_EXP2 + 
			((vectorize((i + 16)*(0x0555555555555555ull)) + ROTL64(msg[i], i + 1) +
            ROTL64(msg[i-13], (i-13)+1) - ROTL64(msg[i-6], (i-6)+1) ) ^ hash[i-9]);
    }

    uint2 XL64 = q[16]^q[17]^q[18]^q[19]^q[20]^q[21]^q[22]^q[23];
	uint2 XH64 = XL64^q[24] ^ q[25] ^ q[26] ^ q[27] ^ q[28] ^ q[29] ^ q[30] ^ q[31];

    hash[0] =                       (SHL(XH64, 5) ^ SHR(q[16],5) ^ msg[ 0]) + (    XL64    ^ q[24] ^ q[ 0]);
    hash[1] =                       (SHR(XH64, 7) ^ SHL(q[17],8) ^ msg[ 1]) + (    XL64    ^ q[25] ^ q[ 1]);
    hash[2] =                       (SHR(XH64, 5) ^ SHL(q[18],5) ^ msg[ 2]) + (    XL64    ^ q[26] ^ q[ 2]);
    hash[3] =                       (SHR(XH64, 1) ^ SHL(q[19],5) ^ msg[ 3]) + (    XL64    ^ q[27] ^ q[ 3]);
    hash[4] =                       (SHR(XH64, 3) ^     q[20]    ^ msg[ 4]) + (    XL64    ^ q[28] ^ q[ 4]);
    hash[5] =                       (SHL(XH64, 6) ^ SHR(q[21],6) ^ msg[ 5]) + (    XL64    ^ q[29] ^ q[ 5]);
    hash[6] =                       (SHR(XH64, 4) ^ SHL(q[22],6) ^ msg[ 6]) + (    XL64    ^ q[30] ^ q[ 6]);
    hash[7] =                       (SHR(XH64,11) ^ SHL(q[23],2) ^ msg[ 7]) + (    XL64    ^ q[31] ^ q[ 7]);

    hash[ 8] = ROTL64(hash[4], 9) + (    XH64     ^     q[24]    ^ msg[ 8]) + (SHL(XL64,8) ^ q[23] ^ q[ 8]);
    hash[ 9] = ROTL64(hash[5],10) + (    XH64     ^     q[25]    ^ msg[ 9]) + (SHR(XL64,6) ^ q[16] ^ q[ 9]);
    hash[10] = ROTL64(hash[6],11) + (    XH64     ^     q[26]    ^ msg[10]) + (SHL(XL64,6) ^ q[17] ^ q[10]);
    hash[11] = ROTL64(hash[7],12) + (    XH64     ^     q[27]    ^ msg[11]) + (SHL(XL64,4) ^ q[18] ^ q[11]);
    hash[12] = ROTL64(hash[0],13) + (    XH64     ^     q[28]    ^ msg[12]) + (SHR(XL64,3) ^ q[19] ^ q[12]);
    hash[13] = ROTL64(hash[1],14) + (    XH64     ^     q[29]    ^ msg[13]) + (SHR(XL64,4) ^ q[20] ^ q[13]);
    hash[14] = ROTL64(hash[2],15) + (    XH64     ^     q[30]    ^ msg[14]) + (SHR(XL64,7) ^ q[21] ^ q[14]);
	hash[15] = ROTL64(hash[3],16) + (XH64     ^     q[31] ^ msg[15]) + (SHR(XL64, 2) ^ q[22] ^ q[15]);
}

__global__ __launch_bounds__(256, 2)
void quark_bmw512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
    uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

        int hashPosition = nounce - startNounce;
        uint64_t *inpHash = &g_hash[8 * hashPosition];

        // Init
		uint2 h[16] = {
			{ 0x84858687UL, 0x80818283UL },
			{ 0x8C8D8E8FUL, 0x88898A8BUL },
			{ 0x94959697UL, 0x90919293UL },
			{ 0x9C9D9E9FUL, 0x98999A9BUL },
			{ 0xA4A5A6A7UL, 0xA0A1A2A3UL },
			{ 0xACADAEAFUL, 0xA8A9AAABUL },
			{ 0xB4B5B6B7UL, 0xB0B1B2B3UL },
			{ 0xBCBDBEBFUL, 0xB8B9BABBUL },
			{ 0xC4C5C6C7UL, 0xC0C1C2C3UL, },
			{ 0xCCCDCECFUL, 0xC8C9CACBUL, },
			{ 0xD4D5D6D7UL, 0xD0D1D2D3UL },
			{ 0xDCDDDEDFUL, 0xD8D9DADBUL },
			{ 0xE4E5E6E7UL, 0xE0E1E2E3UL },
			{ 0xECEDEEEFUL, 0xE8E9EAEBUL },
			{ 0xF4F5F6F7UL, 0xF0F1F2F3UL },
			{ 0xFCFDFEFFUL, 0xF8F9FAFBUL }
		};
        // Nachricht kopieren (Achtung, die Nachricht hat 64 Byte,
        // BMW arbeitet mit 128 Byte!!!
		uint2 message[16];
#pragma unroll 8
        for(int i=0;i<8;i++)
			message[i] = vectorize(inpHash[i]);
#pragma unroll 6
        for(int i=9;i<15;i++)
            message[i] = make_uint2(0,0);

        // Padding einfügen (Byteorder?!?)
		message[8] = make_uint2(0x80,0);
        // Länge (in Bits, d.h. 64 Byte * 8 = 512 Bits
		message[15] = make_uint2(512,0);

        // Compression 1
        Compression512_64_first(message, h);

        // Final
#pragma unroll 16
        for(int i=0;i<16;i++)
		{
			message[i].y = 0xaaaaaaaa;
			message[i].x = 0xaaaaaaa0ul + (uint32_t)i;
		}
        Compression512(h, message);

        // fertig
        uint64_t *outpHash = &g_hash[8 * hashPosition];

#pragma unroll 8
        for(int i=0;i<8;i++)
            outpHash[i] = devectorize(message[i+8]);
    }
}

__global__ __launch_bounds__(256, 2)
void quark_bmw512_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
{
    uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = startNounce + thread;

        // Init
		uint2 h[16] = {
			{ 0x84858687UL, 0x80818283UL },
			{ 0x8C8D8E8FUL, 0x88898A8BUL },
			{ 0x94959697UL, 0x90919293UL },
			{ 0x9C9D9E9FUL, 0x98999A9BUL },
			{ 0xA4A5A6A7UL, 0xA0A1A2A3UL },
			{ 0xACADAEAFUL, 0xA8A9AAABUL },
			{ 0xB4B5B6B7UL, 0xB0B1B2B3UL },
			{ 0xBCBDBEBFUL, 0xB8B9BABBUL },
			{ 0xC4C5C6C7UL, 0xC0C1C2C3UL, },
			{ 0xCCCDCECFUL, 0xC8C9CACBUL, },
			{ 0xD4D5D6D7UL, 0xD0D1D2D3UL },
			{ 0xDCDDDEDFUL, 0xD8D9DADBUL },
			{ 0xE4E5E6E7UL, 0xE0E1E2E3UL },
			{ 0xECEDEEEFUL, 0xE8E9EAEBUL },
			{ 0xF4F5F6F7UL, 0xF0F1F2F3UL },
			{ 0xFCFDFEFFUL, 0xF8F9FAFBUL }
		};
        // Nachricht kopieren (Achtung, die Nachricht hat 64 Byte,
        // BMW arbeitet mit 128 Byte!!!
		uint2 message[16];
#pragma unroll 16
        for(int i=0;i<16;i++)
			message[i] = vectorize(c_PaddedMessage80[i]);

        // die Nounce durch die thread-spezifische ersetzen
		message[9].x = cuda_swab32(nounce);	//REPLACE_HIWORD(message[9], cuda_swab32(nounce));
        // Compression 1
        Compression512(message, h);

        // Final
#pragma unroll 16
        for(int i=0;i<16;i++)
            message[i] = vectorize(0xaaaaaaaaaaaaaaa0ull + (uint64_t)i);

        Compression512(h, message);

        // fertig
        uint64_t *outpHash = &g_hash[8 * thread];

#pragma unroll 8
        for(int i=0;i<8;i++)
            outpHash[i] = devectorize(message[i+8]);
    }
}

// Setup-Funktionen
__host__ void quark_bmw512_cpu_init(int thr_id, uint32_t threads)
{
}

// Bmw512 für 80 Byte grosse Eingangsdaten
__host__ void quark_bmw512_cpu_setBlock_80(void *pdata)
{
	// Message mit Padding bereitstellen
	// lediglich die korrekte Nonce ist noch ab Byte 76 einzusetzen.
	unsigned char PaddedMessage[128];
	memcpy(PaddedMessage, pdata, 80);
	memset(PaddedMessage+80, 0, 48);
	uint64_t *message = (uint64_t*)PaddedMessage;
	// Padding einfügen (Byteorder?!?)
	message[10] = SPH_C64(0x80);
	// Länge (in Bits, d.h. 80 Byte * 8 = 640 Bits
	message[15] = SPH_C64(640);

	// die Message zur Berechnung auf der GPU
	cudaMemcpyToSymbol( c_PaddedMessage80, PaddedMessage, 16*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
}

__host__ void quark_bmw512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
    const uint32_t threadsperblock = 32;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    quark_bmw512_gpu_hash_64<<<grid, block>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
    MyStreamSynchronize(NULL, order, thr_id);
}

__host__ void quark_bmw512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order)
{
    const uint32_t threadsperblock = 128;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    quark_bmw512_gpu_hash_80<<<grid, block>>>(threads, startNounce, (uint64_t*)d_hash);
    MyStreamSynchronize(NULL, order, thr_id);
}

