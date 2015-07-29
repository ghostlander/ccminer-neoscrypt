/*
 * Axiomcoin cudaport by sp-hash@github
 */

#include "cuda_helper.h"
#include <stdint.h>
#include <memory.h>


__constant__ uint32_t c_PaddedMessage80[20]; // padded message (80 bytes + padding?)
static uint32_t *d_found[MAX_GPUS];

#define sM    16

#define O1   13
#define O2    9
#define O3    6

#define INPUT_BLOCK_ADD  \
		B0 = B0 + M0; \
		B1 = B1 + M1; \
		B2 = B2 + M2; \
		B3 = B3 + M3; \
		B4 = B4 + M4; \
		B5 = B5 + M5; \
		B6 = B6 + M6; \
		B7 = B7 + M7; \
		B8 = B8 + M8; \
		B9 = B9 + M9; \
		BA = BA + MA; \
		BB = BB + MB; \
		BC = BC + MC; \
		BD = BD + MD; \
		BE = BE + ME; \
		BF = BF + MF; \

#define INPUT_BLOCK_SUB \
		C0 = C0 - M0; \
		C1 = C1 - M1; \
		C2 = C2 - M2; \
		C3 = C3 - M3; \
		C4 = C4 - M4; \
		C5 = C5 - M5; \
		C6 = C6 - M6; \
		C7 = C7 - M7; \
		C8 = C8 - M8; \
		C9 = C9 - M9; \
		CA = CA - MA; \
		CB = CB - MB; \
		CC = CC - MC; \
		CD = CD - MD; \
		CE = CE - ME; \
		CF = CF - MF; \

#define XOR_W  \
		A00 ^= Wlow; \
		A01 ^= Whigh; \

#define SWAP(v1, v2) \
		v1^=v2;\
		v2 ^= v1;\
		v1 ^= v2;

#define SWAP_BC   \
		SWAP(B0, C0); \
		SWAP(B1, C1); \
		SWAP(B2, C2); \
		SWAP(B3, C3); \
		SWAP(B4, C4); \
		SWAP(B5, C5); \
		SWAP(B6, C6); \
		SWAP(B7, C7); \
		SWAP(B8, C8); \
		SWAP(B9, C9); \
		SWAP(BA, CA); \
		SWAP(BB, CB); \
		SWAP(BC, CC); \
		SWAP(BD, CD); \
		SWAP(BE, CE); \
		SWAP(BF, CF); \

#define PERM_ELT(xa0, xa1, xb0, xb1, xb2, xb3, xc, xm) \
		xa0 = ((xa0 \
			^ (ROTL32(xa1, 15) * 5U) \
			^ xc) * 3U) \
			^ xb1 ^ (xb2 & ~xb3) ^ xm; \
		xb0 = (~(ROTL32(xb0, 1) ^ xa0)); \

#define PERM_STEP_0 \
		PERM_ELT(A00, A0B, B0, BD, B9, B6, C8, M0); \
		PERM_ELT(A01, A00, B1, BE, BA, B7, C7, M1); \
		PERM_ELT(A02, A01, B2, BF, BB, B8, C6, M2); \
		PERM_ELT(A03, A02, B3, B0, BC, B9, C5, M3); \
		PERM_ELT(A04, A03, B4, B1, BD, BA, C4, M4); \
		PERM_ELT(A05, A04, B5, B2, BE, BB, C3, M5); \
		PERM_ELT(A06, A05, B6, B3, BF, BC, C2, M6); \
		PERM_ELT(A07, A06, B7, B4, B0, BD, C1, M7); \
		PERM_ELT(A08, A07, B8, B5, B1, BE, C0, M8); \
		PERM_ELT(A09, A08, B9, B6, B2, BF, CF, M9); \
		PERM_ELT(A0A, A09, BA, B7, B3, B0, CE, MA); \
		PERM_ELT(A0B, A0A, BB, B8, B4, B1, CD, MB); \
		PERM_ELT(A00, A0B, BC, B9, B5, B2, CC, MC); \
		PERM_ELT(A01, A00, BD, BA, B6, B3, CB, MD); \
		PERM_ELT(A02, A01, BE, BB, B7, B4, CA, ME); \
		PERM_ELT(A03, A02, BF, BC, B8, B5, C9, MF); \

#define PERM_STEP_1 \
		PERM_ELT(A04, A03, B0, BD, B9, B6, C8, M0); \
		PERM_ELT(A05, A04, B1, BE, BA, B7, C7, M1); \
		PERM_ELT(A06, A05, B2, BF, BB, B8, C6, M2); \
		PERM_ELT(A07, A06, B3, B0, BC, B9, C5, M3); \
		PERM_ELT(A08, A07, B4, B1, BD, BA, C4, M4); \
		PERM_ELT(A09, A08, B5, B2, BE, BB, C3, M5); \
		PERM_ELT(A0A, A09, B6, B3, BF, BC, C2, M6); \
		PERM_ELT(A0B, A0A, B7, B4, B0, BD, C1, M7); \
		PERM_ELT(A00, A0B, B8, B5, B1, BE, C0, M8); \
		PERM_ELT(A01, A00, B9, B6, B2, BF, CF, M9); \
		PERM_ELT(A02, A01, BA, B7, B3, B0, CE, MA); \
		PERM_ELT(A03, A02, BB, B8, B4, B1, CD, MB); \
		PERM_ELT(A04, A03, BC, B9, B5, B2, CC, MC); \
		PERM_ELT(A05, A04, BD, BA, B6, B3, CB, MD); \
		PERM_ELT(A06, A05, BE, BB, B7, B4, CA, ME); \
		PERM_ELT(A07, A06, BF, BC, B8, B5, C9, MF); \

#define PERM_STEP_2 \
		PERM_ELT(A08, A07, B0, BD, B9, B6, C8, M0); \
		PERM_ELT(A09, A08, B1, BE, BA, B7, C7, M1); \
		PERM_ELT(A0A, A09, B2, BF, BB, B8, C6, M2); \
		PERM_ELT(A0B, A0A, B3, B0, BC, B9, C5, M3); \
		PERM_ELT(A00, A0B, B4, B1, BD, BA, C4, M4); \
		PERM_ELT(A01, A00, B5, B2, BE, BB, C3, M5); \
		PERM_ELT(A02, A01, B6, B3, BF, BC, C2, M6); \
		PERM_ELT(A03, A02, B7, B4, B0, BD, C1, M7); \
		PERM_ELT(A04, A03, B8, B5, B1, BE, C0, M8); \
		PERM_ELT(A05, A04, B9, B6, B2, BF, CF, M9); \
		PERM_ELT(A06, A05, BA, B7, B3, B0, CE, MA); \
		PERM_ELT(A07, A06, BB, B8, B4, B1, CD, MB); \
		PERM_ELT(A08, A07, BC, B9, B5, B2, CC, MC); \
		PERM_ELT(A09, A08, BD, BA, B6, B3, CB, MD); \
		PERM_ELT(A0A, A09, BE, BB, B7, B4, CA, ME); \
		PERM_ELT(A0B, A0A, BF, BC, B8, B5, C9, MF); \

#define APPLY_P  \
		B0 = ROTL32(B0, 17); \
		B1 = ROTL32(B1, 17); \
		B2 = ROTL32(B2, 17); \
		B3 = ROTL32(B3, 17); \
		B4 = ROTL32(B4, 17); \
		B5 = ROTL32(B5, 17); \
		B6 = ROTL32(B6, 17); \
		B7 = ROTL32(B7, 17); \
		B8 = ROTL32(B8, 17); \
		B9 = ROTL32(B9, 17); \
		BA = ROTL32(BA, 17); \
		BB = ROTL32(BB, 17); \
		BC = ROTL32(BC, 17); \
		BD = ROTL32(BD, 17); \
		BE = ROTL32(BE, 17); \
		BF = ROTL32(BF, 17); \
		PERM_STEP_0; \
		PERM_STEP_1; \
		PERM_STEP_2; \
		A0B = (A0B + C6); \
		A0A = (A0A + C5); \
		A09 = (A09 + C4); \
		A08 = (A08 + C3); \
		A07 = (A07 + C2); \
		A06 = (A06 + C1); \
		A05 = (A05 + C0); \
		A04 = (A04 + CF); \
		A03 = (A03 + CE); \
		A02 = (A02 + CD); \
		A01 = (A01 + CC); \
		A00 = (A00 + CB); \
		A0B = (A0B + CA); \
		A0A = (A0A + C9); \
		A09 = (A09 + C8); \
		A08 = (A08 + C7); \
		A07 = (A07 + C6); \
		A06 = (A06 + C5); \
		A05 = (A05 + C4); \
		A04 = (A04 + C3); \
		A03 = (A03 + C2); \
		A02 = (A02 + C1); \
		A01 = (A01 + C0); \
		A00 = (A00 + CF); \
		A0B = (A0B + CE); \
		A0A = (A0A + CD); \
		A09 = (A09 + CC); \
		A08 = (A08 + CB); \
		A07 = (A07 + CA); \
		A06 = (A06 + C9); \
		A05 = (A05 + C8); \
		A04 = (A04 + C7); \
		A03 = (A03 + C6); \
		A02 = (A02 + C5); \
		A01 = (A01 + C4); \
		A00 = (A00 + C3); \

#define APPLY_P_FINAL  \
		B0 = ROTL32(B0, 17); \
		B1 = ROTL32(B1, 17); \
		B2 = ROTL32(B2, 17); \
		B3 = ROTL32(B3, 17); \
		B4 = ROTL32(B4, 17); \
		B5 = ROTL32(B5, 17); \
		B6 = ROTL32(B6, 17); \
		B7 = ROTL32(B7, 17); \
		B8 = ROTL32(B8, 17); \
		B9 = ROTL32(B9, 17); \
		BA = ROTL32(BA, 17); \
		BB = ROTL32(BB, 17); \
		BC = ROTL32(BC, 17); \
		BD = ROTL32(BD, 17); \
		BE = ROTL32(BE, 17); \
		BF = ROTL32(BF, 17); \
		PERM_STEP_0; \
		PERM_STEP_1; \
		PERM_STEP_2; \

#define INCR_W if ((Wlow = (Wlow + 1)) == 0) \
			Whigh = (Whigh + 1); \
	
static __device__ void axiom_shabal256_gpu_hash_64(uint32_t *g_hash)
{
	const uint32_t A_init_256[] = 
	{
		0x52F84552, 0xE54B7999, 0x2D8EE3EC, 0xB9645191,
		0xE0078B86, 0xBB7C44C9, 0xD2B5C1CA, 0xB0D2EB8C,
		0x14CE5A45, 0x22AF50DC, 0xEFFDBC6B, 0xEB21B74A
	};

	const uint32_t B_init_256[] = {
		0xB555C6EE, 0x3E710596, 0xA72A652F, 0x9301515F,
		0xDA28C1FA, 0x696FD868, 0x9CB6BF72, 0x0AFE4002,
		0xA6E03615, 0x5138C1D4, 0xBE216306, 0xB38B8890,
		0x3EA8B96B, 0x3299ACE4, 0x30924DD4, 0x55CB34A5
	};

	const uint32_t C_init_256[] = {
		0xB405F031, 0xC4233EBA, 0xB3733979, 0xC0DD9D55,
		0xC51C28AE, 0xA327B8E1, 0x56C56167, 0xED614433,
		0x88B59D60, 0x60E2CEBA, 0x758B4B8B, 0x83E82A7F,
		0xBC968828, 0xE6E00BF7, 0xBA839E55, 0x9B491C60
	};

	uint32_t *Hash = &g_hash[0]; // [hashPosition * 8]
	uint32_t A00 = A_init_256[0], A01 = A_init_256[1], A02 = A_init_256[2], A03 = A_init_256[3],
		A04 = A_init_256[4], A05 = A_init_256[5], A06 = A_init_256[6], A07 = A_init_256[7],
		A08 = A_init_256[8], A09 = A_init_256[9], A0A = A_init_256[10], A0B = A_init_256[11];
	uint32_t B0 = B_init_256[0], B1 = B_init_256[1], B2 = B_init_256[2], B3 = B_init_256[3],
			B4 = B_init_256[4], B5 = B_init_256[5], B6 = B_init_256[6], B7 = B_init_256[7],
			B8 = B_init_256[8], B9 = B_init_256[9], BA = B_init_256[10], BB = B_init_256[11],
			BC = B_init_256[12], BD = B_init_256[13], BE = B_init_256[14], BF = B_init_256[15];
		uint32_t C0 = C_init_256[0], C1 = C_init_256[1], C2 = C_init_256[2], C3 = C_init_256[3],
			C4 = C_init_256[4], C5 = C_init_256[5], C6 = C_init_256[6], C7 = C_init_256[7],
			C8 = C_init_256[8], C9 = C_init_256[9], CA = C_init_256[10], CB = C_init_256[11],
			CC = C_init_256[12], CD = C_init_256[13], CE = C_init_256[14], CF = C_init_256[15];
		uint32_t M0, M1, M2, M3, M4, M5, M6, M7, M8, M9, MA, MB, MC, MD, ME, MF;

		M0 = Hash[0];
		M1 = Hash[1];
		M2 = Hash[2];
		M3 = Hash[3];
		M4 = Hash[4];
		M5 = Hash[5];
		M6 = Hash[6];
		M7 = Hash[7];

		M8 = 0;
		M9 = 0;
		MA = 0;
		MB = 0;
		MC = 0;
		MD = 0;
		ME = 0;
		MF = 0;

		INPUT_BLOCK_ADD;
		A00 ^= 1;
		APPLY_P;
		INPUT_BLOCK_SUB;
		SWAP_BC;

		M0 = 0x80;
		M1 = M2 = M3 = M4 = M5 = M6 = M7 = M8 = M9 = MA = MB = MC = MD = ME = MF = 0;

		INPUT_BLOCK_ADD;
		A00 ^= 2;
		APPLY_P;

		SWAP_BC;
		A00 ^= 2;
		APPLY_P;

		SWAP_BC;
		A00 ^= 2;
		APPLY_P;

		SWAP_BC;
		A00 ^= 2;
		APPLY_P_FINAL;

	Hash[0] = B0;
	Hash[1] = B1;
	Hash[2] = B2;
	Hash[3] = B3;
	Hash[4] = B4;
	Hash[5] = B5;
	Hash[6] = B6;
	Hash[7] = B7;
}

__host__
void axiom_setBlock_80(void *pdata)
{
	unsigned char PaddedMessage[4*8];
	memcpy(PaddedMessage, pdata, 4 * 8);
	cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, 4 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

__global__
__launch_bounds__(256, 1)
void axiom_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint32_t target, uint32_t *d_found)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		int N = 65536;
		uint32_t hash[65536][8];
		int R = 2;
		uint32_t nounce = (startNounce + thread);

		hash[0][0] = c_PaddedMessage80[0];
		hash[0][1] = c_PaddedMessage80[1];
		hash[0][2] = c_PaddedMessage80[2];
		hash[0][3] = c_PaddedMessage80[3];
		hash[0][4] = c_PaddedMessage80[4];
		hash[0][5] = c_PaddedMessage80[5];
		hash[0][6] = c_PaddedMessage80[6];
		hash[0][7] = c_PaddedMessage80[7] ^ nounce;

		for (int i = 0; i < N-1; i++)
		{
			axiom_shabal256_gpu_hash_64(&hash[i][0]);
			hash[i + 1][0] = hash[i][0];
			hash[i + 1][1] = hash[i][1];
			hash[i + 1][2] = hash[i][2];
			hash[i + 1][3] = hash[i][3];
			hash[i + 1][4] = hash[i][4];
			hash[i + 1][5] = hash[i][5];
			hash[i + 1][6] = hash[i][6];
			hash[i + 1][7] = hash[i][7];
		}
		axiom_shabal256_gpu_hash_64(&hash[N-1][0]);

		for (int r = 1; r < R; r++)
		{
			for (int b = 0; b < N; b++)
			{
				int p = b > 0 ? b - 1 : N - 1;
				int q = hash[p][0] % (N - 1);
				int j = (b + q) % N;
				axiom_shabal256_gpu_hash_64(&hash[p][0]);
				axiom_shabal256_gpu_hash_64(&hash[j][0]);
			}
		}
		if (hash[N - 1][7] <= target)
		{
			uint32_t tmp = atomicExch(&(d_found[0]), nounce);
			if (tmp != 0xffffffff)
				d_found[1] = tmp;
		}
	}
}

__host__ void axiom_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t target, uint32_t *h_found)
{
	const uint32_t threadsperblock = 1;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	cudaMemset(d_found[thr_id], 0xffffffff, 2 * sizeof(uint32_t));

	axiom_gpu_hash_80 << <grid, block >> >(threads, startNounce, target,d_found[thr_id] );
	cudaMemcpy(h_found, d_found[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}