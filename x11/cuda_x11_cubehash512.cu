#include "cuda_helper.h"

#define CUBEHASH_ROUNDS 16 /* this is r for CubeHashr/b */
#define CUBEHASH_BLOCKBYTES 32 /* this is b for CubeHashr/b */

#if __CUDA_ARCH__ < 350
#define LROT(x,bits) ((x << bits) | (x >> (32 - bits)))
#else
#define LROT(x, bits) __funnelshift_l(x, x, bits)
#endif

#define ROTATEUPWARDS7(a)  LROT(a,7)
#define ROTATEUPWARDS11(a) LROT(a,11)

#define SWAP(a,b) { uint32_t u = a; a = b; b = u; }

__device__ __forceinline__ void rrounds(uint32_t x[2][2][2][2][2])
{
	int r;
	int j;
	int k;
	int l;
	int m;

	#pragma unroll 2
	for (r = 0; r < CUBEHASH_ROUNDS; ++r) {

		/* "add x_0jklm into x_1jklmn modulo 2^32" */
#pragma unroll 2
		for (j = 0; j < 2; ++j)
#pragma unroll 2
			for (k = 0; k < 2; ++k)
#pragma unroll 2
				for (l = 0; l < 2; ++l)
#pragma unroll 2
					for (m = 0; m < 2; ++m)
						x[1][j][k][l][m] += x[0][j][k][l][m];

		/* "rotate x_0jklm upwards by 7 bits" */
#pragma unroll 2
		for (j = 0; j < 2; ++j)
#pragma unroll 2
			for (k = 0; k < 2; ++k)
#pragma unroll 2
				for (l = 0; l < 2; ++l)
#pragma unroll 2
					for (m = 0; m < 2; ++m)
						x[0][j][k][l][m] = ROTATEUPWARDS7(x[0][j][k][l][m]);

		/* "swap x_00klm with x_01klm" */
#pragma unroll 2
		for (k = 0; k < 2; ++k)
#pragma unroll 2
			for (l = 0; l < 2; ++l)
#pragma unroll 2
				for (m = 0; m < 2; ++m)
					SWAP(x[0][0][k][l][m], x[0][1][k][l][m])

					/* "xor x_1jklm into x_0jklm" */
#pragma unroll 2
					for (j = 0; j < 2; ++j)
#pragma unroll 2
						for (k = 0; k < 2; ++k)
#pragma unroll 2
							for (l = 0; l < 2; ++l)
#pragma unroll 2
								for (m = 0; m < 2; ++m)
									x[0][j][k][l][m] ^= x[1][j][k][l][m];

		/* "swap x_1jk0m with x_1jk1m" */
#pragma unroll 2
		for (j = 0; j < 2; ++j)
#pragma unroll 2
			for (k = 0; k < 2; ++k)
#pragma unroll 2
				for (m = 0; m < 2; ++m)
					SWAP(x[1][j][k][0][m], x[1][j][k][1][m])

					/* "add x_0jklm into x_1jklm modulo 2^32" */
#pragma unroll 2
					for (j = 0; j < 2; ++j)
#pragma unroll 2
						for (k = 0; k < 2; ++k)
#pragma unroll 2
							for (l = 0; l < 2; ++l)
#pragma unroll 2
								for (m = 0; m < 2; ++m)
									x[1][j][k][l][m] += x[0][j][k][l][m];

		/* "rotate x_0jklm upwards by 11 bits" */
#pragma unroll 2
		for (j = 0; j < 2; ++j)
#pragma unroll 2
			for (k = 0; k < 2; ++k)
#pragma unroll 2
				for (l = 0; l < 2; ++l)
#pragma unroll 2
					for (m = 0; m < 2; ++m)
						x[0][j][k][l][m] = ROTATEUPWARDS11(x[0][j][k][l][m]);

		/* "swap x_0j0lm with x_0j1lm" */
#pragma unroll 2
		for (j = 0; j < 2; ++j)
#pragma unroll 2
			for (l = 0; l < 2; ++l)
#pragma unroll 2
				for (m = 0; m < 2; ++m)
					SWAP(x[0][j][0][l][m], x[0][j][1][l][m])

					/* "xor x_1jklm into x_0jklm" */
#pragma unroll 2
					for (j = 0; j < 2; ++j)
#pragma unroll 2
						for (k = 0; k < 2; ++k)
#pragma unroll 2
							for (l = 0; l < 2; ++l)
#pragma unroll 2
								for (m = 0; m < 2; ++m)
									x[0][j][k][l][m] ^= x[1][j][k][l][m];

		/* "swap x_1jkl0 with x_1jkl1" */
#pragma unroll 2
		for (j = 0; j < 2; ++j)
#pragma unroll 2
			for (k = 0; k < 2; ++k)
#pragma unroll 2
				for (l = 0; l < 2; ++l)
					SWAP(x[1][j][k][l][0], x[1][j][k][l][1])

	}
}

__device__ __forceinline__ void block_tox(const uint32_t *in, uint32_t x[2][2][2][2][2])
{
	x[0][0][0][0][0] ^= in[0];
	x[0][0][0][0][1] ^= in[1];
	x[0][0][0][1][0] ^= in[2];
	x[0][0][0][1][1] ^= in[3];
	x[0][0][1][0][0] ^= in[4];
	x[0][0][1][0][1] ^= in[5];
	x[0][0][1][1][0] ^= in[6];
	x[0][0][1][1][1] ^= in[7];
}

__device__ __forceinline__ void hash_fromx(uint32_t *out, uint32_t x[2][2][2][2][2])
{
	out[0] = x[0][0][0][0][0];
	out[1] = x[0][0][0][0][1];
	out[2] = x[0][0][0][1][0];
	out[3] = x[0][0][0][1][1];
	out[4] = x[0][0][1][0][0];
	out[5] = x[0][0][1][0][1];
	out[6] = x[0][0][1][1][0];
	out[7] = x[0][0][1][1][1];

	out[8] = x[0][1][0][0][0];
	out[9] = x[0][1][0][0][1];
	out[10] = x[0][1][0][1][0];
	out[11] = x[0][1][0][1][1];
	out[12] = x[0][1][1][0][0];
	out[13] = x[0][1][1][0][1];
	out[14] = x[0][1][1][1][0];
	out[15] = x[0][1][1][1][1];

}

void __device__ __forceinline__ Update32(uint32_t x[2][2][2][2][2], const uint32_t *data)
{
	/* "xor the block into the first b bytes of the state" */
	/* "and then transform the state invertibly through r identical rounds" */
	block_tox(data, x);
	rrounds(x);
}

void __device__ __forceinline__ Update32_const(uint32_t x[2][2][2][2][2])
{
	x[0][0][0][0][0] ^= 0x80;
	rrounds(x);
}



void __device__ __forceinline__ Final(uint32_t x[2][2][2][2][2], uint32_t *hashval)
{
	int i;

	/* "the integer 1 is xored into the last state word x_11111" */
	x[1][1][1][1][1] ^= 1;

	/* "the state is then transformed invertibly through 10r identical rounds" */
	#pragma unroll 2
	for (i = 0; i < 10; ++i) rrounds(x);

	/* "output the first h/8 bytes of the state" */
	hash_fromx(hashval, x);
}



/***************************************************/
// Die Hash-Funktion
__global__	__launch_bounds__(256,5)
void x11_cubehash512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *g_hash)
{
    uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = (startNounce + thread);

        int hashPosition = nounce - startNounce;
        uint32_t *Hash = &g_hash[16 * hashPosition];

		uint32_t x[2][2][2][2][2] =
		{
			0x2AEA2A61, 0x50F494D4, 0x2D538B8B,
			0x4167D83E, 0x3FEE2313, 0xC701CF8C,
			0xCC39968E, 0x50AC5695, 0x4D42C787,
			0xA647A8B3, 0x97CF0BEF, 0x825B4537,
			0xEEF864D2, 0xF22090C4, 0xD0E5CD33,
			0xA23911AE, 0xFCD398D9, 0x148FE485,
			0x1B017BEF, 0xB6444532, 0x6A536159,
			0x2FF5781C, 0x91FA7934, 0x0DBADEA9,
			0xD65C8A2B, 0xA5A70E75, 0xB1C62456,
			0xBC796576, 0x1921C8F7, 0xE7989AF1,
			0x7795D246, 0xD43E3B44
		};
		x[0][0][0][0][0] ^= Hash[0];
		x[0][0][0][0][1] ^= Hash[1];
		x[0][0][0][1][0] ^= Hash[2];
		x[0][0][0][1][1] ^= Hash[3];
		x[0][0][1][0][0] ^= Hash[4];
		x[0][0][1][0][1] ^= Hash[5];
		x[0][0][1][1][0] ^= Hash[6];
		x[0][0][1][1][1] ^= Hash[7];

		rrounds(x);
		Update32(x, &Hash[8]);
		Update32_const(x);
		Final(x, Hash);
    }
}


__host__
void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 256;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    x11_cubehash512_gpu_hash_64<<<grid, block>>>(threads, startNounce, d_hash);
}

