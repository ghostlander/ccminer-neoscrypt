#include <memory.h>

#include "cuda_helper.h"

#define TPB 160



static __device__ __forceinline__
void Gfunc_v35(uint2 & a, uint2 &b, uint2 &c, uint2 &d)
{
	a += b; d = SWAPDWORDS2(d ^ a);
	c += d; b = ROR24(b ^ c);
	a += b; d = ROR16(d ^ a);
	c += d; b = ROR2(b ^ c, 63);
}
static __device__ __forceinline__ void round_lyra_v35(uint2 *s)
{
	Gfunc_v35(s[0], s[4], s[8],  s[12]);
	Gfunc_v35(s[1], s[5], s[9],  s[13]);
	Gfunc_v35(s[2], s[6], s[10], s[14]);
	Gfunc_v35(s[3], s[7], s[11], s[15]);
	Gfunc_v35(s[0], s[5], s[10], s[15]);
	Gfunc_v35(s[1], s[6], s[11], s[12]);
	Gfunc_v35(s[2], s[7], s[8],  s[13]);
	Gfunc_v35(s[3], s[4], s[9],  s[14]);
}

__device__ __forceinline__ void reduceDuplexRowSetup(const int rowIn, const int rowInOut, const int rowOut, uint2 state[16], uint2 Matrix[96][8])
{ 
#if __CUDA_ARCH__ > 500
	#pragma unroll
	for (int i = 0; i < 8 * 12; i += 12)
#else
	for (int i = 0; i < 8 * 12; i += 12)
#endif	
	{
		for (int j = 0; j < 12; j++)
			state[j] ^= Matrix[i + j][rowIn] + Matrix[i + j][rowInOut];
		round_lyra_v35(state);
		#pragma unroll
		for (int j = 0; j < 12; j++)
			Matrix[j + 84 - i][rowOut] = Matrix[i + j][rowIn] ^ state[j];
		Matrix[0 + i][rowInOut] ^= state[11];
		Matrix[1 + i][rowInOut] ^= state[0];
		Matrix[2 + i][rowInOut] ^= state[1];
		Matrix[3 + i][rowInOut] ^= state[2];
		Matrix[4 + i][rowInOut] ^= state[3];
		Matrix[5 + i][rowInOut] ^= state[4];
		Matrix[6 + i][rowInOut] ^= state[5];
		Matrix[7 + i][rowInOut] ^= state[6];
		Matrix[8 + i][rowInOut] ^= state[7];
		Matrix[9 + i][rowInOut] ^= state[8];
		Matrix[10 + i][rowInOut] ^= state[9];
		Matrix[11 + i][rowInOut] ^= state[10];
	}


}


__device__ __forceinline__ void reduceDuplexRowt(const int rowIn,  const int rowOut, uint2 state[16], uint2 Matrix[96][8])
{
	uint32_t rowInOut = state[0].x & 7;
	for (int i = 0; i < 8; i++) 
	{
		#pragma unroll
		for (int j = 0; j < 12; j++)
			state[j] ^= Matrix[12 * i + j][rowIn] + Matrix[12 * i + j][rowInOut];
		round_lyra_v35(state);
		#pragma unroll
		for (int j = 0; j < 12; j++)
			Matrix[j + 12 * i][rowOut] ^= state[j];
		Matrix[0 + 12 * i][rowInOut] ^= state[11];
		Matrix[1 + 12 * i][rowInOut] ^= state[0];
		Matrix[2 + 12 * i][rowInOut] ^= state[1];
		Matrix[3 + 12 * i][rowInOut] ^= state[2];
		Matrix[4 + 12 * i][rowInOut] ^= state[3];
		Matrix[5 + 12 * i][rowInOut] ^= state[4];
		Matrix[6 + 12 * i][rowInOut] ^= state[5];
		Matrix[7 + 12 * i][rowInOut] ^= state[6];
		Matrix[8 + 12 * i][rowInOut] ^= state[7];
		Matrix[9 + 12 * i][rowInOut] ^= state[8];
		Matrix[10+ 12 * i][rowInOut] ^= state[9];
		Matrix[11+ 12 * i][rowInOut] ^= state[10]; 
	}
}

__global__	__launch_bounds__(TPB, 1)
void lyra2_gpu_hash_32(uint32_t threads, uint32_t startNounce, uint64_t *outputHash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{

		const uint2 blake2b_IV[8] = {
			{ 0xf3bcc908, 0x6a09e667 },
			{ 0x84caa73b, 0xbb67ae85 },
			{ 0xfe94f82b, 0x3c6ef372 },
			{ 0x5f1d36f1, 0xa54ff53a },
			{ 0xade682d1, 0x510e527f },
			{ 0x2b3e6c1f, 0x9b05688c },
			{ 0xfb41bd6b, 0x1f83d9ab },
			{ 0x137e2179, 0x5be0cd19 }
		};


		register uint2 state[16];
		#pragma unroll
		for (int i = 0; i<4; i++) 
		{ 
			LOHI(state[i].x, state[i].y, outputHash[threads*i + thread]); 
		} //password
		#pragma unroll
		for (int i = 0; i<4; i++) { state[i + 4] = state[i]; } //salt
		#pragma unroll
		for (int i = 0; i<8; i++) { state[i + 8] = blake2b_IV[i]; }

		// blake2blyra x2
		//#pragma unroll 24
		for (int i = 0; i<24; i++) { round_lyra_v35(state); } //because 12 is not enough

		uint2 __align__(16) Matrix[96][8]; // not cool

		// reducedSqueezeRow0
		#pragma unroll
		for (int i = 0; i < 8*12; i+=12)
		{
			#pragma unroll 12
			for (int j = 0; j<12; j++) { Matrix[j + 84 - i][0] = state[j]; }
			round_lyra_v35(state);
		}

		// reducedSqueezeRow1
//		#pragma unroll 	
		for (int i = 0; i < 8*12; i+=12)
		{
			#pragma unroll 12
			for (int j = 0; j<12; j++) { state[j] ^= Matrix[j + i][0]; }
			round_lyra_v35(state);
			#pragma unroll 12
			for (int j = 0; j<12; j++) { Matrix[j + 84 - i][1] = Matrix[j + i][0] ^ state[j]; }
		}

		reduceDuplexRowSetup(1, 0, 2,state, Matrix);
		reduceDuplexRowSetup(2, 1, 3, state, Matrix);
		reduceDuplexRowSetup(3, 0, 4, state, Matrix);
		reduceDuplexRowSetup(4, 3, 5, state, Matrix);
		reduceDuplexRowSetup(5, 2, 6, state, Matrix);
		reduceDuplexRowSetup(6, 1, 7, state, Matrix);

		reduceDuplexRowt(7, 0,state, Matrix);
		reduceDuplexRowt(0, 3, state, Matrix);
		reduceDuplexRowt(3, 6, state, Matrix);
		reduceDuplexRowt(6, 1, state, Matrix);
		reduceDuplexRowt(1, 4, state, Matrix);
		reduceDuplexRowt(4, 7, state, Matrix);
		reduceDuplexRowt(7, 2, state, Matrix);
		uint32_t rowa = state[0].x & 7;
		reduceDuplexRowt(2, 5, state, Matrix);

		state[0] ^= Matrix[0][rowa];
		state[1] ^= Matrix[1][rowa];
		state[2] ^= Matrix[2][rowa];
		state[3] ^= Matrix[3][rowa];
		state[4] ^= Matrix[4][rowa];
		state[5] ^= Matrix[5][rowa];
		state[6] ^= Matrix[6][rowa];
		state[7] ^= Matrix[7][rowa];
		state[8] ^= Matrix[8][rowa];
		state[9] ^= Matrix[9][rowa];
		state[10] ^= Matrix[10][rowa];
		state[11] ^= Matrix[11][rowa];

		#pragma unroll
		for (int i = 0; i < 12; i++)
		{
			round_lyra_v35(state);
		}

		#pragma unroll
		for (int i = 0; i<4; i++) {
			outputHash[threads*i + thread] = devectorize(state[i]);
		} //password

	} //thread
}

__host__
void lyra2_cpu_init(int thr_id, uint32_t threads)
{
	//not used
}

__host__
void lyra2_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash)
{
	dim3 grid((threads + TPB - 1) / TPB);
	dim3 block(TPB);

	lyra2_gpu_hash_32 <<<grid, block>>> (threads, startNounce, d_outputHash);
}

