#include <stdint.h>
#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h" 
#define TPBf 128

static __constant__ uint64_t c_PaddedMessage80[16]; // padded message (80 bytes + padding)
__constant__ uint64_t precalcvalues[9];

// Take a look at: https://www.schneier.com/skein1.3.pdf

#define SHL(x, n)			((x) << (n))
#define SHR(x, n)			((x) >> (n))

static uint32_t *d_nonce[MAX_GPUS];

/*
 * M9_ ## s ## _ ## i  evaluates to s+i mod 9 (0 <= s <= 18, 0 <= i <= 7).
 */

#define M9_0_0    0
#define M9_0_1    1
#define M9_0_2    2
#define M9_0_3    3
#define M9_0_4    4
#define M9_0_5    5
#define M9_0_6    6
#define M9_0_7    7

#define M9_1_0    1
#define M9_1_1    2
#define M9_1_2    3
#define M9_1_3    4
#define M9_1_4    5
#define M9_1_5    6
#define M9_1_6    7
#define M9_1_7    8

#define M9_2_0    2
#define M9_2_1    3
#define M9_2_2    4
#define M9_2_3    5
#define M9_2_4    6
#define M9_2_5    7
#define M9_2_6    8
#define M9_2_7    0

#define M9_3_0    3
#define M9_3_1    4
#define M9_3_2    5
#define M9_3_3    6
#define M9_3_4    7
#define M9_3_5    8
#define M9_3_6    0
#define M9_3_7    1

#define M9_4_0    4
#define M9_4_1    5
#define M9_4_2    6
#define M9_4_3    7
#define M9_4_4    8
#define M9_4_5    0
#define M9_4_6    1
#define M9_4_7    2

#define M9_5_0    5
#define M9_5_1    6
#define M9_5_2    7
#define M9_5_3    8
#define M9_5_4    0
#define M9_5_5    1
#define M9_5_6    2
#define M9_5_7    3

#define M9_6_0    6
#define M9_6_1    7
#define M9_6_2    8
#define M9_6_3    0
#define M9_6_4    1
#define M9_6_5    2
#define M9_6_6    3
#define M9_6_7    4

#define M9_7_0    7
#define M9_7_1    8
#define M9_7_2    0
#define M9_7_3    1
#define M9_7_4    2
#define M9_7_5    3
#define M9_7_6    4
#define M9_7_7    5

#define M9_8_0    8
#define M9_8_1    0
#define M9_8_2    1
#define M9_8_3    2
#define M9_8_4    3
#define M9_8_5    4
#define M9_8_6    5
#define M9_8_7    6

#define M9_9_0    0
#define M9_9_1    1
#define M9_9_2    2
#define M9_9_3    3
#define M9_9_4    4
#define M9_9_5    5
#define M9_9_6    6
#define M9_9_7    7

#define M9_10_0   1
#define M9_10_1   2
#define M9_10_2   3
#define M9_10_3   4
#define M9_10_4   5
#define M9_10_5   6
#define M9_10_6   7
#define M9_10_7   8

#define M9_11_0   2
#define M9_11_1   3
#define M9_11_2   4
#define M9_11_3   5
#define M9_11_4   6
#define M9_11_5   7
#define M9_11_6   8
#define M9_11_7   0

#define M9_12_0   3
#define M9_12_1   4
#define M9_12_2   5
#define M9_12_3   6
#define M9_12_4   7
#define M9_12_5   8
#define M9_12_6   0
#define M9_12_7   1

#define M9_13_0   4
#define M9_13_1   5
#define M9_13_2   6
#define M9_13_3   7
#define M9_13_4   8
#define M9_13_5   0
#define M9_13_6   1
#define M9_13_7   2

#define M9_14_0   5
#define M9_14_1   6
#define M9_14_2   7
#define M9_14_3   8
#define M9_14_4   0
#define M9_14_5   1
#define M9_14_6   2
#define M9_14_7   3

#define M9_15_0   6
#define M9_15_1   7
#define M9_15_2   8
#define M9_15_3   0
#define M9_15_4   1
#define M9_15_5   2
#define M9_15_6   3
#define M9_15_7   4

#define M9_16_0   7
#define M9_16_1   8
#define M9_16_2   0
#define M9_16_3   1
#define M9_16_4   2
#define M9_16_5   3
#define M9_16_6   4
#define M9_16_7   5

#define M9_17_0   8
#define M9_17_1   0
#define M9_17_2   1
#define M9_17_3   2
#define M9_17_4   3
#define M9_17_5   4
#define M9_17_6   5
#define M9_17_7   6

#define M9_18_0   0
#define M9_18_1   1
#define M9_18_2   2
#define M9_18_3   3
#define M9_18_4   4
#define M9_18_5   5
#define M9_18_6   6
#define M9_18_7   7

/*
 * M3_ ## s ## _ ## i  evaluates to s+i mod 3 (0 <= s <= 18, 0 <= i <= 1).
 */

#define M3_0_0    0
#define M3_0_1    1
#define M3_1_0    1
#define M3_1_1    2
#define M3_2_0    2
#define M3_2_1    0
#define M3_3_0    0
#define M3_3_1    1
#define M3_4_0    1
#define M3_4_1    2
#define M3_5_0    2
#define M3_5_1    0
#define M3_6_0    0
#define M3_6_1    1
#define M3_7_0    1
#define M3_7_1    2
#define M3_8_0    2
#define M3_8_1    0
#define M3_9_0    0
#define M3_9_1    1
#define M3_10_0   1
#define M3_10_1   2
#define M3_11_0   2
#define M3_11_1   0
#define M3_12_0   0
#define M3_12_1   1
#define M3_13_0   1
#define M3_13_1   2
#define M3_14_0   2
#define M3_14_1   0
#define M3_15_0   0
#define M3_15_1   1
#define M3_16_0   1
#define M3_16_1   2
#define M3_17_0   2
#define M3_17_1   0
#define M3_18_0   0
#define M3_18_1   1

#define XCAT(x, y)     XCAT_(x, y)
#define XCAT_(x, y)    x ## y

#define SKBI(k, s, i)   XCAT(k, XCAT(XCAT(XCAT(M9_, s), _), i))
#define SKBT(t, s, v)   XCAT(t, XCAT(XCAT(XCAT(M3_, s), _), v))

#define TFBIG_KINIT(k0, k1, k2, k3, k4, k5, k6, k7, k8, t0, t1, t2) { \
		k8 = ((k0 ^ k1) ^ (k2 ^ k3)) ^ ((k4 ^ k5) ^ (k6 ^ k7)) \
			^ make_uint2( 0xA9FC1A22UL,0x1BD11BDA); \
		t2 = t0 ^ t1; \
	}
//vectorize(0x1BD11BDAA9FC1A22ULL);
#define TFBIG_ADDKEY(w0, w1, w2, w3, w4, w5, w6, w7, k, t, s) { \
		w0 = (w0 + SKBI(k, s, 0)); \
		w1 = (w1 + SKBI(k, s, 1)); \
		w2 = (w2 + SKBI(k, s, 2)); \
		w3 = (w3 + SKBI(k, s, 3)); \
		w4 = (w4 + SKBI(k, s, 4)); \
		w5 = (w5 + SKBI(k, s, 5) + SKBT(t, s, 0)); \
		w6 = (w6 + SKBI(k, s, 6) + SKBT(t, s, 1)); \
		w7 = (w7 + SKBI(k, s, 7) + vectorizelow(s)); \
	}

#define TFBIG_MIX(x0, x1, rc) { \
		x0 = x0 + x1; \
		x1 = ROL2(x1, rc) ^ x0; \
	}

#define TFBIG_MIX8(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
		TFBIG_MIX(w0, w1, rc0); \
		TFBIG_MIX(w2, w3, rc1); \
		TFBIG_MIX(w4, w5, rc2); \
		TFBIG_MIX(w6, w7, rc3); \
	}

#define TFBIG_4e(s)  { \
		TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 46, 36, 19, 37); \
		TFBIG_MIX8(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 33, 27, 14, 42); \
		TFBIG_MIX8(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 17, 49, 36, 39); \
		TFBIG_MIX8(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3], 44,  9, 54, 56); \
	}

#define TFBIG_4o(s)  { \
		TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 39, 30, 34, 24); \
		TFBIG_MIX8(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 13, 50, 10, 17); \
		TFBIG_MIX8(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 25, 29, 39, 43); \
		TFBIG_MIX8(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3],  8, 35, 56, 22); \
	}

/* uint2 variant for SM3.2+ */

#define TFBIG_KINIT_UI2(k0, k1, k2, k3, k4, k5, k6, k7, k8, t0, t1, t2) { \
		k8 = ((k0 ^ k1) ^ (k2 ^ k3)) ^ ((k4 ^ k5) ^ (k6 ^ k7)) \
			^ vectorize(SPH_C64(0x1BD11BDAA9FC1A22)); \
		t2 = t0 ^ t1; \
		}

#define TFBIG_ADDKEY_UI2(w0, w1, w2, w3, w4, w5, w6, w7, k, t, s) { \
		w0 = (w0 + SKBI(k, s, 0)); \
		w1 = (w1 + SKBI(k, s, 1)); \
		w2 = (w2 + SKBI(k, s, 2)); \
		w3 = (w3 + SKBI(k, s, 3)); \
		w4 = (w4 + SKBI(k, s, 4)); \
		w5 = (w5 + SKBI(k, s, 5) + SKBT(t, s, 0)); \
		w6 = (w6 + SKBI(k, s, 6) + SKBT(t, s, 1)); \
		w7 = (w7 + SKBI(k, s, 7) + vectorize(s)); \
		}

#define TFBIG_ADDKEY_PRE(w0, w1, w2, w3, w4, w5, w6, w7, k, t, s) { \
		w0 = (w0 + SKBI(k, s, 0)); \
		w1 = (w1 + SKBI(k, s, 1)); \
		w2 = (w2 + SKBI(k, s, 2)); \
		w3 = (w3 + SKBI(k, s, 3)); \
		w4 = (w4 + SKBI(k, s, 4)); \
		w5 = (w5 + SKBI(k, s, 5) + SKBT(t, s, 0)); \
		w6 = (w6 + SKBI(k, s, 6) + SKBT(t, s, 1)); \
		w7 = (w7 + SKBI(k, s, 7) + (s)); \
				}

#define TFBIG_MIX_UI2(x0, x1, rc) { \
		x0 = x0 + x1; \
		x1 = ROL2(x1, rc) ^ x0; \
		}

#define TFBIG_MIX_PRE(x0, x1, rc) { \
		x0 = x0 + x1; \
		x1 = ROTL64(x1, rc) ^ x0; \
				}

#define TFBIG_MIX8_UI2(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
		TFBIG_MIX_UI2(w0, w1, rc0); \
		TFBIG_MIX_UI2(w2, w3, rc1); \
		TFBIG_MIX_UI2(w4, w5, rc2); \
		TFBIG_MIX_UI2(w6, w7, rc3); \
		}

#define TFBIG_MIX8_PRE(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
		TFBIG_MIX_PRE(w0, w1, rc0); \
		TFBIG_MIX_PRE(w2, w3, rc1); \
		TFBIG_MIX_PRE(w4, w5, rc2); \
		TFBIG_MIX_PRE(w6, w7, rc3); \
				}

#define TFBIG_4e_UI2(s)  { \
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 46, 36, 19, 37); \
		TFBIG_MIX8_UI2(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 33, 27, 14, 42); \
		TFBIG_MIX8_UI2(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 17, 49, 36, 39); \
		TFBIG_MIX8_UI2(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3], 44,  9, 54, 56); \
		}

#define TFBIG_4e_PRE(s)  { \
		TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 46, 36, 19, 37); \
		TFBIG_MIX8_PRE(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 33, 27, 14, 42); \
		TFBIG_MIX8_PRE(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 17, 49, 36, 39); \
		TFBIG_MIX8_PRE(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3], 44,  9, 54, 56); \
				}

#define TFBIG_4o_UI2(s)  { \
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 39, 30, 34, 24); \
		TFBIG_MIX8_UI2(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 13, 50, 10, 17); \
		TFBIG_MIX8_UI2(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 25, 29, 39, 43); \
		TFBIG_MIX8_UI2(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3],  8, 35, 56, 22); \
		}

#define TFBIG_4o_PRE(s)  { \
		TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 39, 30, 34, 24); \
		TFBIG_MIX8_PRE(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 13, 50, 10, 17); \
		TFBIG_MIX8_PRE(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 25, 29, 39, 43); \
		TFBIG_MIX8_PRE(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3],  8, 35, 56, 22); \
		}

__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(448, 2)
#else
__launch_bounds__(128,10)
#endif
void quark_skein512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint64_t * const __restrict__ g_hash, const uint32_t *const __restrict__ g_nonceVector)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// Skein
		uint2 skein_p[8], h[9];

		const uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		const int hashPosition = nounce - startNounce;
		uint64_t *Hash = &g_hash[8 * hashPosition];

		h[0] = skein_p[0] = vectorize(Hash[0]);
		h[1] = skein_p[1] = vectorize(Hash[1]);
		h[2] = skein_p[2] = vectorize(Hash[2]);
		h[3] = skein_p[3] = vectorize(Hash[3]);
		h[4] = skein_p[4] = vectorize(Hash[4]);
		h[5] = skein_p[5] = vectorize(Hash[5]);
		h[6] = skein_p[6] = vectorize(Hash[6]);
		h[7] = skein_p[7] = vectorize(Hash[7]);

		skein_p[0] += vectorize(0x4903ADFF749C51CEULL);
		skein_p[1] += vectorize(0x0D95DE399746DF03ULL);
		skein_p[2] += vectorize(0x8FD1934127C79BCEULL);
		skein_p[3] += vectorize(0x9A255629FF352CB1ULL);
		skein_p[4] += vectorize(0x5DB62599DF6CA7B0ULL);
		skein_p[5] += vectorize(0xEABE394CA9D5C434ULL);
		skein_p[6] += vectorize(0x891112C71A75B523ULL);
		skein_p[7] += vectorize(0xAE18A40B660FCC33ULL);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 46) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 36) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 19) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 37) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 33) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 27) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 14) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 42) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 17) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 49) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 36) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 39) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 44) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 9) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 54) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 56) ^ skein_p[4];
		skein_p[0] += vectorize(0x0D95DE399746DF03ULL);
		skein_p[1] += vectorize(0x8FD1934127C79BCEULL);
		skein_p[2] += vectorize(0x9A255629FF352CB1ULL);
		skein_p[3] += vectorize(0x5DB62599DF6CA7B0ULL);
		skein_p[4] += vectorize(0xEABE394CA9D5C3F4ULL);
		skein_p[5] += vectorize(0x891112C71A75B523ULL);
		skein_p[6] += vectorize(0x9E18A40B660FCC73ULL);
		skein_p[7] += vectorize(0xcab2076d98173ec4ULL+1);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 39) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 30) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 34) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 24) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 13) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 50) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 10) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 17) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 25) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 29) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 39) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 43) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 8) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 35) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 56) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 22) ^ skein_p[4];
		skein_p[0] += vectorize(0x8FD1934127C79BCEULL);
		skein_p[1] += vectorize(0x9A255629FF352CB1ULL);
		skein_p[2] += vectorize(0x5DB62599DF6CA7B0ULL);
		skein_p[3] += vectorize(0xEABE394CA9D5C3F4ULL);
		skein_p[4] += vectorize(0x991112C71A75B523ULL);
		skein_p[5] += vectorize(0x9E18A40B660FCC73ULL);
		skein_p[6] += vectorize(0xCAB2076D98173F04ULL);
		skein_p[7] += vectorize(0x4903ADFF749C51D0ULL);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 46) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 36) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 19) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 37) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 33) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 27) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 14) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 42) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 17) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 49) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 36) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 39) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 44) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 9) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 54) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 56) ^ skein_p[4];
		skein_p[0] += vectorize(0x9A255629FF352CB1ULL);
		skein_p[1] += vectorize(0x5DB62599DF6CA7B0ULL);
		skein_p[2] += vectorize(0xEABE394CA9D5C3F4ULL);
		skein_p[3] += vectorize(0x991112C71A75B523ULL);
		skein_p[4] += vectorize(0xAE18A40B660FCC33ULL);
		skein_p[5] += vectorize(0xcab2076d98173f04ULL);
		skein_p[6] += vectorize(0x3903ADFF749C51CEULL);
		skein_p[7] += vectorize(0x0D95DE399746DF03ULL+3);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 39) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 30) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 34) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 24) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 13) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 50) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 10) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 17) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 25) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 29) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 39) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 43) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 8) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 35) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 56) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 22) ^ skein_p[4];
		skein_p[0] += vectorize(0x5DB62599DF6CA7B0ULL);
		skein_p[1] += vectorize(0xEABE394CA9D5C3F4ULL);
		skein_p[2] += vectorize(0x991112C71A75B523ULL);
		skein_p[3] += vectorize(0xAE18A40B660FCC33ULL);
		skein_p[4] += vectorize(0xcab2076d98173ec4ULL);
		skein_p[5] += vectorize(0x3903ADFF749C51CEULL);
		skein_p[6] += vectorize(0xFD95DE399746DF43ULL);
		skein_p[7] += vectorize(0x8FD1934127C79BD2ULL);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 46) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 36) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 19) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 37) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 33) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 27) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 14) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 42) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 17) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 49) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 36) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 39) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 44) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 9) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 54) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 56) ^ skein_p[4];
		skein_p[0] += vectorize(0xEABE394CA9D5C3F4ULL);
		skein_p[1] += vectorize(0x991112C71A75B523ULL);
		skein_p[2] += vectorize(0xAE18A40B660FCC33ULL);
		skein_p[3] += vectorize(0xcab2076d98173ec4ULL);
		skein_p[4] += vectorize(0x4903ADFF749C51CEULL);
		skein_p[5] += vectorize(0x0D95DE399746DF03ULL + 0xf000000000000040ULL);
		skein_p[6] += vectorize(0x8FD1934127C79BCEULL + 0x0000000000000040ULL);
		skein_p[7] += vectorize(0x9A255629FF352CB1ULL + 5);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 39) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 30) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 34) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 24) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 13) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 50) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 10) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 17) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 25) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 29) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 39) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 43) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 8) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 35) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 56) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 22) ^ skein_p[4];
		skein_p[0] += vectorize(0x991112C71A75B523ULL);
		skein_p[1] += vectorize(0xAE18A40B660FCC33ULL);
		skein_p[2] += vectorize(0xcab2076d98173ec4ULL);
		skein_p[3] += vectorize(0x4903ADFF749C51CEULL);
		skein_p[4] += vectorize(0x0D95DE399746DF03ULL);
		skein_p[5] += vectorize(0x8FD1934127C79BCEULL + 0x0000000000000040ULL);
		skein_p[6] += vectorize(0x8A255629FF352CB1ULL);
		skein_p[7] += vectorize(0x5DB62599DF6CA7B0ULL + 6);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 46) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 36) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 19) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 37) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 33) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 27) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 14) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 42) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 17) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 49) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 36) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 39) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 44) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 9) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 54) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 56) ^ skein_p[4];
		skein_p[0] += vectorize(0xAE18A40B660FCC33ULL);
		skein_p[1] += vectorize(0xcab2076d98173ec4ULL);
		skein_p[2] += vectorize(0x4903ADFF749C51CEULL);
		skein_p[3] += vectorize(0x0D95DE399746DF03ULL);
		skein_p[4] += vectorize(0x8FD1934127C79BCEULL);
		skein_p[5] += vectorize(0x8A255629FF352CB1ULL);
		skein_p[6] += vectorize(0x4DB62599DF6CA7F0ULL);
		skein_p[7] += vectorize(0xEABE394CA9D5C3F4ULL + 7);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 39) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 30) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 34) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 24) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 13) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 50) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 10) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 17) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 25) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 29) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 39) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 43) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 8) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 35) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 56) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 22) ^ skein_p[4];
		skein_p[0] += vectorize(0xcab2076d98173ec4ULL);
		skein_p[1] += vectorize(0x4903ADFF749C51CEULL);
		skein_p[2] += vectorize(0x0D95DE399746DF03ULL);
		skein_p[3] += vectorize(0x8FD1934127C79BCEULL);
		skein_p[4] += vectorize(0x9A255629FF352CB1ULL);
		skein_p[5] += vectorize(0x4DB62599DF6CA7F0ULL);
		skein_p[6] += vectorize(0xEABE394CA9D5C434ULL);
		skein_p[7] += vectorize(0x991112C71A75B52BULL);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 46) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 36) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 19) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 37) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 33) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 27) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 14) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 42) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 17) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 49) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 36) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 39) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 44) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 9) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 54) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 56) ^ skein_p[4];
		skein_p[0] += vectorize(0x4903ADFF749C51CEULL);
		skein_p[1] += vectorize(0x0D95DE399746DF03ULL);
		skein_p[2] += vectorize(0x8FD1934127C79BCEULL);
		skein_p[3] += vectorize(0x9A255629FF352CB1ULL);
		skein_p[4] += vectorize(0x5DB62599DF6CA7B0ULL);
		skein_p[5] += vectorize(0xEABE394CA9D5C434ULL);
		skein_p[6] += vectorize(0x891112C71A75B523ULL);
		skein_p[7] += vectorize(0xAE18A40B660FCC33ULL + 9);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 39) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 30) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 34) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 24) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 13) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 50) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 10) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 17) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 25) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 29) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 39) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 43) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 8) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 35) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 56) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 22) ^ skein_p[4];
		skein_p[0] += vectorize(0x0D95DE399746DF03ULL);
		skein_p[1] += vectorize(0x8FD1934127C79BCEULL);
		skein_p[2] += vectorize(0x9A255629FF352CB1ULL);
		skein_p[3] += vectorize(0x5DB62599DF6CA7B0ULL);
		skein_p[4] += vectorize(0xEABE394CA9D5C3F4ULL);
		skein_p[5] += vectorize(0x891112C71A75B523ULL);
		skein_p[6] += vectorize(0x9E18A40B660FCC73ULL);
		skein_p[7] += vectorize(0xcab2076d98173eceULL);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 46) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 36) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 19) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 37) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 33) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 27) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 14) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 42) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 17) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 49) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 36) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 39) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 44) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 9) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 54) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 56) ^ skein_p[4];
		skein_p[0] += vectorize(0x8FD1934127C79BCEULL);
		skein_p[1] += vectorize(0x9A255629FF352CB1ULL);
		skein_p[2] += vectorize(0x5DB62599DF6CA7B0ULL);
		skein_p[3] += vectorize(0xEABE394CA9D5C3F4ULL);
		skein_p[4] += vectorize(0x991112C71A75B523ULL);
		skein_p[5] += vectorize(0x9E18A40B660FCC73ULL);
		skein_p[6] += vectorize(0xcab2076d98173ec4ULL + 0x0000000000000040ULL);
		skein_p[7] += vectorize(0x4903ADFF749C51CEULL + 11);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 39) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 30) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 34) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 24) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 13) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 50) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 10) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 17) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 25) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 29) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 39) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 43) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 8) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 35) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 56) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 22) ^ skein_p[4];
		skein_p[0] += vectorize(0x9A255629FF352CB1ULL);
		skein_p[1] += vectorize(0x5DB62599DF6CA7B0ULL);
		skein_p[2] += vectorize(0xEABE394CA9D5C3F4ULL);
		skein_p[3] += vectorize(0x991112C71A75B523ULL);
		skein_p[4] += vectorize(0xAE18A40B660FCC33ULL);
		skein_p[5] += vectorize(0xcab2076d98173ec4ULL + 0x0000000000000040ULL);
		skein_p[6] += vectorize(0x3903ADFF749C51CEULL);
		skein_p[7] += vectorize(0x0D95DE399746DF03ULL + 12);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 46) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 36) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 19) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 37) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 33) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 27) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 14) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 42) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 17) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 49) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 36) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 39) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 44) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 9) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 54) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 56) ^ skein_p[4];
		skein_p[0] += vectorize(0x5DB62599DF6CA7B0ULL);
		skein_p[1] += vectorize(0xEABE394CA9D5C3F4ULL);
		skein_p[2] += vectorize(0x991112C71A75B523ULL);
		skein_p[3] += vectorize(0xAE18A40B660FCC33ULL);
		skein_p[4] += vectorize(0xcab2076d98173ec4ULL);
		skein_p[5] += vectorize(0x3903ADFF749C51CEULL);
		skein_p[6] += vectorize(0x0D95DE399746DF03ULL + 0xf000000000000040ULL);
		skein_p[7] += vectorize(0x8FD1934127C79BCEULL + 13);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 39) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 30) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 34) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 24) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 13) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 50) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 10) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 17) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 25) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 29) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 39) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 43) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 8) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 35) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 56) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 22) ^ skein_p[4];
		skein_p[0] += vectorize(0xEABE394CA9D5C3F4ULL);
		skein_p[1] += vectorize(0x991112C71A75B523ULL);
		skein_p[2] += vectorize(0xAE18A40B660FCC33ULL);
		skein_p[3] += vectorize(0xcab2076d98173ec4ULL);
		skein_p[4] += vectorize(0x4903ADFF749C51CEULL);
		skein_p[5] += vectorize(0x0D95DE399746DF03ULL + 0xf000000000000040ULL);
		skein_p[6] += vectorize(0x8FD1934127C79BCEULL + 0x0000000000000040ULL);
		skein_p[7] += vectorize(0x9A255629FF352CB1ULL + 14);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 46) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 36) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 19) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 37) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 33) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 27) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 14) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 42) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 17) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 49) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 36) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 39) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 44) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 9) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 54) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 56) ^ skein_p[4];
		skein_p[0] += vectorize(0x991112C71A75B523ULL);
		skein_p[1] += vectorize(0xAE18A40B660FCC33ULL);
		skein_p[2] += vectorize(0xcab2076d98173ec4ULL);
		skein_p[3] += vectorize(0x4903ADFF749C51CEULL);
		skein_p[4] += vectorize(0x0D95DE399746DF03ULL);
		skein_p[5] += vectorize(0x8FD1934127C79BCEULL + 0x0000000000000040ULL);
		skein_p[6] += vectorize(0x8A255629FF352CB1ULL);
		skein_p[7] += vectorize(0x5DB62599DF6CA7B0ULL + 15);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 39) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 30) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 34) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 24) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 13) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 50) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 10) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 17) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 25) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 29) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 39) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 43) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 8) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 35) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 56) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 22) ^ skein_p[4];
		skein_p[0] += vectorize(0xAE18A40B660FCC33ULL);
		skein_p[1] += vectorize(0xcab2076d98173ec4ULL);
		skein_p[2] += vectorize(0x4903ADFF749C51CEULL);
		skein_p[3] += vectorize(0x0D95DE399746DF03ULL);
		skein_p[4] += vectorize(0x8FD1934127C79BCEULL);
		skein_p[5] += vectorize(0x8A255629FF352CB1ULL);
		skein_p[6] += vectorize(0x4DB62599DF6CA7F0ULL);
		skein_p[7] += vectorize(0xEABE394CA9D5C3F4ULL +16ULL);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 46) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 36) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 19) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 37) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 33) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 27) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 14) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 42) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 17) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 49) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 36) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 39) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 44) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 9) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 54) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 56) ^ skein_p[4];
		skein_p[0] += vectorize(0xcab2076d98173ec4ULL);
		skein_p[1] += vectorize(0x4903ADFF749C51CEULL);
		skein_p[2] += vectorize(0x0D95DE399746DF03ULL);
		skein_p[3] += vectorize(0x8FD1934127C79BCEULL);
		skein_p[4] += vectorize(0x9A255629FF352CB1ULL);
		skein_p[5] += vectorize(0x4DB62599DF6CA7F0ULL);
		skein_p[6] += vectorize(0xEABE394CA9D5C3F4ULL + 0x0000000000000040ULL);
		skein_p[7] += vectorize(0x991112C71A75B523ULL + 17);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 39) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 30) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 34) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 24) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 13) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 50) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 10) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 17) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 25) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 29) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 39) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 43) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 8) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 35) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 56) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 22) ^ skein_p[4];
		skein_p[0] += vectorize(0x4903ADFF749C51CEULL);
		skein_p[1] += vectorize(0x0D95DE399746DF03ULL);
		skein_p[2] += vectorize(0x8FD1934127C79BCEULL);
		skein_p[3] += vectorize(0x9A255629FF352CB1ULL);
		skein_p[4] += vectorize(0x5DB62599DF6CA7B0ULL);
		skein_p[5] += vectorize(0xEABE394CA9D5C3F4ULL + 0x0000000000000040ULL);
		skein_p[6] += vectorize(0x891112C71A75B523ULL);
		skein_p[7] += vectorize(0xAE18A40B660FCC33ULL + 18);

#define h0 skein_p[0]
#define h1 skein_p[1]
#define h2 skein_p[2]
#define h3 skein_p[3]
#define h4 skein_p[4]
#define h5 skein_p[5]
#define h6 skein_p[6]
#define h7 skein_p[7]
		h0 ^= h[0];
		h1 ^= h[1];
		h2 ^= h[2];
		h3 ^= h[3];
		h4 ^= h[4];
		h5 ^= h[5];
		h6 ^= h[6];
		h7 ^= h[7];

		uint2 skein_h8 = h0 ^ h1 ^ h2 ^ h3 ^ h4 ^ h5 ^ h6 ^ h7 ^ vectorize(0x1BD11BDAA9FC1A22ULL);

		uint2 hash64[8];

		hash64[0] = (h0);
//		hash64[1] = (h1);
		hash64[2] = (h2);
//		hash64[3] = (h3);
		hash64[4] = (h4);
		hash64[5] = (h5 + vectorizelow(8ULL));
		hash64[6] = (h6 + vectorize(0xff00000000000000ULL));
//		hash64[7] = (h7);

		hash64[0] += h1;
		hash64[1] = ROL2(h1, 46) ^ hash64[0];
		hash64[2] += h3;
		hash64[3] = ROL2(h3, 36) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] += h7;
		hash64[7] = ROL2(h7, 37) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 56) ^ hash64[4];
		hash64[0] = (hash64[0] + h1);
		hash64[1] = (hash64[1] + h2);
		hash64[2] = (hash64[2] + h3);
		hash64[3] = (hash64[3] + h4);
		hash64[4] = (hash64[4] + h5);
		hash64[5] = (hash64[5] + h6 + vectorize(0xff00000000000000ULL));
		hash64[6] = (hash64[6] + h7 + vectorize(0xff00000000000008ULL));
		hash64[7] = (hash64[7] + skein_h8 + vectorizelow(1));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 39) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 30) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 34) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 24) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 13) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 50) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 10) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 17) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 25) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 29) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 39) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 43) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 8) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 35) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 56) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 22) ^ hash64[4];
		hash64[0] = (hash64[0] + h2);
		hash64[1] = (hash64[1] + h3);
		hash64[2] = (hash64[2] + h4);
		hash64[3] = (hash64[3] + h5);
		hash64[4] = (hash64[4] + h6);
		hash64[5] = (hash64[5] + h7 + vectorize(0xff00000000000008ULL));
		hash64[6] = (hash64[6] + skein_h8 + vectorizelow(8ULL));
		hash64[7] = (hash64[7] + h0 + vectorize(2));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 46) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 36) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 37) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 56) ^ hash64[4];
		hash64[0] = (hash64[0] + h3);
		hash64[1] = (hash64[1] + h4);
		hash64[2] = (hash64[2] + h5);
		hash64[3] = (hash64[3] + h6);
		hash64[4] = (hash64[4] + h7);
		hash64[5] = (hash64[5] + skein_h8 + vectorizelow(8));
		hash64[6] = (hash64[6] + h0 + vectorize(0xff00000000000000ULL));
		hash64[7] = (hash64[7] + h1 + vectorizelow(3));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 39) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 30) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 34) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 24) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 13) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 50) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 10) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 17) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 25) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 29) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 39) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 43) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 8) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 35) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 56) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 22) ^ hash64[4];
		hash64[0] = (hash64[0] + h4);
		hash64[1] = (hash64[1] + h5);
		hash64[2] = (hash64[2] + h6);
		hash64[3] = (hash64[3] + h7);
		hash64[4] = (hash64[4] + skein_h8);
		hash64[5] = (hash64[5] + h0 + vectorize(0xff00000000000000ULL));
		hash64[6] = (hash64[6] + h1 + vectorize(0xff00000000000008ULL));
		hash64[7] = (hash64[7] + h2 + vectorizelow(4));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 46) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 36) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 37) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 56) ^ hash64[4];
		hash64[0] = (hash64[0] + h5);
		hash64[1] = (hash64[1] + h6);
		hash64[2] = (hash64[2] + h7);
		hash64[3] = (hash64[3] + skein_h8);
		hash64[4] = (hash64[4] + h0);
		hash64[5] = (hash64[5] + h1 + vectorize(0xff00000000000008ULL));
		hash64[6] = (hash64[6] + h2 + vectorizelow(8ULL));
		hash64[7] = (hash64[7] + h3 + vectorizelow(5));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 39) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 30) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 34) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 24) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 13) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 50) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 10) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 17) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 25) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 29) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 39) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 43) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 8) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 35) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 56) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 22) ^ hash64[4];
		hash64[0] = (hash64[0] + h6);
		hash64[1] = (hash64[1] + h7);
		hash64[2] = (hash64[2] + skein_h8);
		hash64[3] = (hash64[3] + h0);
		hash64[4] = (hash64[4] + h1);
		hash64[5] = (hash64[5] + h2 + vectorizelow(8ULL));
		hash64[6] = (hash64[6] + h3 + vectorize(0xff00000000000000ULL));
		hash64[7] = (hash64[7] + h4 + vectorizelow(6));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 46) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 36) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 37) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 56) ^ hash64[4];
		hash64[0] = (hash64[0] + h7);
		hash64[1] = (hash64[1] + skein_h8);
		hash64[2] = (hash64[2] + h0);
		hash64[3] = (hash64[3] + h1);
		hash64[4] = (hash64[4] + h2);
		hash64[5] = (hash64[5] + h3 + vectorize(0xff00000000000000ULL));
		hash64[6] = (hash64[6] + h4 + vectorize(0xff00000000000008ULL));
		hash64[7] = (hash64[7] + h5 + vectorizelow(7));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 39) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 30) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 34) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 24) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 13) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 50) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 10) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 17) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 25) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 29) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 39) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 43) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 8) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 35) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 56) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 22) ^ hash64[4];
		hash64[0] = (hash64[0] + skein_h8);
		hash64[1] = (hash64[1] + h0);
		hash64[2] = (hash64[2] + h1);
		hash64[3] = (hash64[3] + h2);
		hash64[4] = (hash64[4] + h3);
		hash64[5] = (hash64[5] + h4 + vectorize(0xff00000000000008ULL));
		hash64[6] = (hash64[6] + h5 + vectorizelow(8));
		hash64[7] = (hash64[7] + h6 + vectorizelow(8));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 46) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 36) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 37) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 56) ^ hash64[4];
		hash64[0] = (hash64[0] + h0);
		hash64[1] = (hash64[1] + h1);
		hash64[2] = (hash64[2] + h2);
		hash64[3] = (hash64[3] + h3);
		hash64[4] = (hash64[4] + h4);
		hash64[5] = (hash64[5] + h5 + vectorizelow(8));
		hash64[6] = (hash64[6] + h6 + vectorize(0xff00000000000000ULL));
		hash64[7] = (hash64[7] + h7 + vectorizelow(9));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 39) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 30) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 34) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 24) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 13) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 50) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 10) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 17) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 25) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 29) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 39) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 43) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 8) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 35) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 56) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 22) ^ hash64[4];

		hash64[0] = (hash64[0] + h1);
		hash64[1] = (hash64[1] + h2);
		hash64[2] = (hash64[2] + h3);
		hash64[3] = (hash64[3] + h4);
		hash64[4] = (hash64[4] + h5);
		hash64[5] = (hash64[5] + h6 + vectorize(0xff00000000000000ULL));
		hash64[6] = (hash64[6] + h7 + vectorize(0xff00000000000008ULL));
		hash64[7] = (hash64[7] + skein_h8 + (vectorizelow(10)));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 46) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 36) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 37) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 56) ^ hash64[4];
		hash64[0] = (hash64[0] + h2);
		hash64[1] = (hash64[1] + h3);
		hash64[2] = (hash64[2] + h4);
		hash64[3] = (hash64[3] + h5);
		hash64[4] = (hash64[4] + h6);
		hash64[5] = (hash64[5] + h7 + vectorize(0xff00000000000008ULL));
		hash64[6] = (hash64[6] + skein_h8 + vectorizelow(8ULL));
		hash64[7] = (hash64[7] + h0 + vectorizelow(11));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 39) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 30) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 34) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 24) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 13) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 50) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 10) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 17) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 25) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 29) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 39) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 43) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 8) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 35) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 56) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 22) ^ hash64[4];
		hash64[0] = (hash64[0] + h3);
		hash64[1] = (hash64[1] + h4);
		hash64[2] = (hash64[2] + h5);
		hash64[3] = (hash64[3] + h6);
		hash64[4] = (hash64[4] + h7);
		hash64[5] = (hash64[5] + skein_h8 + vectorizelow(8));
		hash64[6] = (hash64[6] + h0 + vectorize(0xff00000000000000ULL));
		hash64[7] = (hash64[7] + h1 + vectorizelow(12));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 46) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 36) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 37) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 56) ^ hash64[4];
		hash64[0] = (hash64[0] + h4);
		hash64[1] = (hash64[1] + h5);
		hash64[2] = (hash64[2] + h6);
		hash64[3] = (hash64[3] + h7);
		hash64[4] = (hash64[4] + skein_h8);
		hash64[5] = (hash64[5] + h0 + vectorize(0xff00000000000000ULL));
		hash64[6] = (hash64[6] + h1 + vectorize(0xff00000000000008ULL));
		hash64[7] = (hash64[7] + h2 + vectorizelow(13));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 39) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 30) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 34) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 24) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 13) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 50) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 10) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 17) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 25) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 29) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 39) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 43) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 8) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 35) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 56) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 22) ^ hash64[4];
		hash64[0] = (hash64[0] + h5);
		hash64[1] = (hash64[1] + h6);
		hash64[2] = (hash64[2] + h7);
		hash64[3] = (hash64[3] + skein_h8);
		hash64[4] = (hash64[4] + h0);
		hash64[5] = (hash64[5] + h1 + vectorize(0xff00000000000008ULL));
		hash64[6] = (hash64[6] + h2 + vectorizelow(8ULL));
		hash64[7] = (hash64[7] + h3 + vectorizelow(14));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 46) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 36) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 37) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 56) ^ hash64[4];
		hash64[0] = (hash64[0] + h6);
		hash64[1] = (hash64[1] + h7);
		hash64[2] = (hash64[2] + skein_h8);
		hash64[3] = (hash64[3] + h0);
		hash64[4] = (hash64[4] + h1);
		hash64[5] = (hash64[5] + h2 + vectorizelow(8ULL));
		hash64[6] = (hash64[6] + h3 + vectorize(0xff00000000000000ULL));
		hash64[7] = (hash64[7] + h4 + vectorizelow(15));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 39) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 30) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 34) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 24) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 13) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 50) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 10) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 17) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 25) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 29) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 39) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 43) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 8) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 35) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 56) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 22) ^ hash64[4];
		hash64[0] = (hash64[0] + h7);
		hash64[1] = (hash64[1] + skein_h8);
		hash64[2] = (hash64[2] + h0);
		hash64[3] = (hash64[3] + h1);
		hash64[4] = (hash64[4] + h2);
		hash64[5] = (hash64[5] + h3 + vectorize(0xff00000000000000ULL));
		hash64[6] = (hash64[6] + h4 + vectorize(0xff00000000000008ULL));
		hash64[7] = (hash64[7] + h5 + vectorizelow(16));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 46) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 36) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 37) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 56) ^ hash64[4];
		hash64[0] = (hash64[0] + skein_h8);
		hash64[1] = (hash64[1] + h0);
		hash64[2] = (hash64[2] + h1);
		hash64[3] = (hash64[3] + h2);
		hash64[4] = (hash64[4] + h3);
		hash64[5] = (hash64[5] + h4 + vectorize(0xff00000000000008ULL));
		hash64[6] = (hash64[6] + h5 + vectorizelow(8ULL));
		hash64[7] = (hash64[7] + h6 + vectorizelow(17));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 39) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 30) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 34) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 24) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 13) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 50) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 10) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 17) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 25) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 29) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 39) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 43) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 8) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 35) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 56) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 22) ^ hash64[4];


		Hash[0] = devectorize(hash64[0] + h0);
		Hash[1] = devectorize(hash64[1] + h1);
		Hash[2] = devectorize(hash64[2] + h2);
		Hash[3] = devectorize(hash64[3] + h3);
		Hash[4] = devectorize(hash64[4] + h4);
		Hash[5] = devectorize(hash64[5] + h5)+ 8;
		Hash[6] = devectorize(hash64[6] + h6)+ 0xff00000000000000ULL;
		Hash[7] = devectorize(hash64[7] + h7)+ 18;

#undef h0
#undef h1
#undef h2
#undef h3
#undef h4
#undef h5
#undef h6
#undef h7
	}
}

__global__ 
__launch_bounds__(128, 10)
void quark_skein512_gpu_hash_64_final(const uint32_t threads, const uint32_t startNounce, uint64_t * const __restrict__ g_hash, const uint32_t *g_nonceVector, uint32_t *d_nonce, uint32_t target)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// Skein
		uint2 p[8];
		uint2 h0, h1, h2, h3, h4, h5, h6, h7, h8;
		uint2 t0, t1, t2;

		const uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		const int hashPosition = nounce - startNounce;
		const uint64_t *const inpHash = &g_hash[8 * hashPosition];

		h0 = make_uint2(0x749C51CEull, 0x4903ADFF);
		h1 = make_uint2(0x9746DF03ull, 0x0D95DE39);
		h2 = make_uint2(0x27C79BCEull, 0x8FD19341);
		h3 = make_uint2(0xFF352CB1ull, 0x9A255629);
		h4 = make_uint2(0xDF6CA7B0ull, 0x5DB62599);
		h5 = make_uint2(0xA9D5C3F4ull, 0xEABE394C);
		h6 = make_uint2(0x1A75B523ull, 0x991112C7);
		h7 = make_uint2(0x660FCC33ull, 0xAE18A40B);

		// 1. Runde -> etype = 480, ptr = 64, bcount = 0, data = msg
#pragma unroll 8
		for (int i = 0; i<8; i++)
			p[i] = vectorize(inpHash[i]);

		t0 = vectorizelow(64); // ptr
		t1 = vectorize(480ull << 55); // etype
		TFBIG_KINIT(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);
		TFBIG_4e(0);
		TFBIG_4o(1);
		TFBIG_4e(2);
		TFBIG_4o(3);
		TFBIG_4e(4);
		TFBIG_4o(5);
		TFBIG_4e(6);
		TFBIG_4o(7);
		TFBIG_4e(8);
		TFBIG_4o(9);
		TFBIG_4e(10);
		TFBIG_4o(11);
		TFBIG_4e(12);
		TFBIG_4o(13);
		TFBIG_4e(14);
		TFBIG_4o(15);
		TFBIG_4e(16);
		TFBIG_4o(17);
		TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

		h0 = vectorize(inpHash[0]) ^ p[0];
		h1 = vectorize(inpHash[1]) ^ p[1];
		h2 = vectorize(inpHash[2]) ^ p[2];
		h3 = vectorize(inpHash[3]) ^ p[3];
		h4 = vectorize(inpHash[4]) ^ p[4];
		h5 = vectorize(inpHash[5]) ^ p[5];
		h6 = vectorize(inpHash[6]) ^ p[6];
		h7 = vectorize(inpHash[7]) ^ p[7];

		// 2. Runde -> etype = 510, ptr = 8, bcount = 0, data = 0
#pragma unroll 8
		for (int i = 0; i<8; i++)
			p[i] = make_uint2(0, 0);

		t0 = vectorizelow(8); // ptr
		t1 = vectorize(510ull << 55); // etype
		TFBIG_KINIT(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);
		TFBIG_4e(0);
		TFBIG_4o(1);
		TFBIG_4e(2);
		TFBIG_4o(3);
		TFBIG_4e(4);
		TFBIG_4o(5);
		TFBIG_4e(6);
		TFBIG_4o(7);
		TFBIG_4e(8);
		TFBIG_4o(9);
		TFBIG_4e(10);
		TFBIG_4o(11);
		TFBIG_4e(12);
		TFBIG_4o(13);
		TFBIG_4e(14);
		TFBIG_4o(15);
		TFBIG_4e(16);
		TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 17); 
		p[0] = p[0] + p[1];
		p[1] = ROL2(p[1], 39) ^ p[0];
		p[2] = p[2] + p[3];
		p[3] = ROL2(p[3], 30) ^ p[2];
		p[4] = p[4] + p[5];
		p[5] = ROL2(p[5], 34) ^ p[4];
		p[6] = p[6] + p[7];
		p[7] = ROL2(p[7], 24) ^ p[6];
		p[1] = ROL2(p[1], 13) ^ (p[2] + p[1]);
		p[3] = ROL2(p[3], 17) ^ (p[0] + p[3]);
		p[3] = ROL2(p[3], 29) ^ (p[6] + p[5] + p[3]);
		p[3] = (ROL2(p[3], 22) ^ (p[4] + p[7] + p[1] + p[3])) + h3;

		if (p[3].y <= target)
		{
			uint32_t tmp = atomicExch(&d_nonce[0], nounce);
			if (tmp != 0xffffffff)
				d_nonce[1] = tmp;
		}
	}
}


__host__ void quark_skein512_cpu_init(int thr_id)
{
	cudaMalloc(&d_nonce[thr_id], 2*sizeof(uint32_t));
}

__host__ void quark_skein512_setTarget(const void *ptarget)
{
}
__host__ void quark_skein512_cpu_free(int32_t thr_id)
{
	cudaFreeHost(&d_nonce[thr_id]);
}

__global__ __launch_bounds__(128, 10)
void skein512_gpu_hash_close(uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint2 t0 = vectorizelow(8); // extra
		uint2 t1 = vectorize(0xFF00000000000000ull); // etype
		uint2 t2 = vectorize(0xB000000000000050ull);

		uint64_t *state = &g_hash[8 * thread];
		uint2 h0 = vectorize(state[0]);
		uint2 h1 = vectorize(state[1]);
		uint2 h2 = vectorize(state[2]);
		uint2 h3 = vectorize(state[3]);
		uint2 h4 = vectorize(state[4]);
		uint2 h5 = vectorize(state[5]);
		uint2 h6 = vectorize(state[6]);
		uint2 h7 = vectorize(state[7]);
		uint2 h8;
		TFBIG_KINIT_UI2(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);

		uint2 p[8] = { 0 };
		//#pragma unroll 8
		//for (int i = 0; i<8; i++)
		//	p[i] = make_uint2(0, 0);

		TFBIG_4e_UI2(0);
		TFBIG_4o_UI2(1);
		TFBIG_4e_UI2(2);
		TFBIG_4o_UI2(3);
		TFBIG_4e_UI2(4);
		TFBIG_4o_UI2(5);
		TFBIG_4e_UI2(6);
		TFBIG_4o_UI2(7);
		TFBIG_4e_UI2(8);
		TFBIG_4o_UI2(9);
		TFBIG_4e_UI2(10);
		TFBIG_4o_UI2(11);
		TFBIG_4e_UI2(12);
		TFBIG_4o_UI2(13);
		TFBIG_4e_UI2(14);
		TFBIG_4o_UI2(15);
		TFBIG_4e_UI2(16);
		TFBIG_4o_UI2(17);
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

		uint64_t *outpHash = state;
		#pragma unroll 8
		for (int i = 0; i < 8; i++)
			outpHash[i] = devectorize(p[i]);
	}
}

__global__ __launch_bounds__(128, 10)
void skein512_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint64_t *output64)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint2 h0, h1, h2, h3, h4, h5, h6, h7, h8;
		uint2 t0, t1, t2;
		uint2 p[8];

		h0 = vectorize(precalcvalues[0]);
		h1 = vectorize(precalcvalues[1]);
		h2 = vectorize(precalcvalues[2]);
		h3 = vectorize(precalcvalues[3]);
		h4 = vectorize(precalcvalues[4]);
		h5 = vectorize(precalcvalues[5]);
		h6 = vectorize(precalcvalues[6]);
		h7 = vectorize(precalcvalues[7]);
		t2 = vectorize(precalcvalues[8]);

		const uint2 nounce2 = make_uint2(_LOWORD(c_PaddedMessage80[9]), cuda_swab32(startNounce + thread));

		// skein_big_close -> etype = 0x160, ptr = 16, bcount = 1, extra = 16
		p[0] = vectorize(c_PaddedMessage80[8]);
		p[1] = nounce2;

		#pragma unroll
		for (int i = 2; i < 8; i++)
			p[i] = make_uint2(0,0);

		t0 = vectorizelow(0x50ull); // SPH_T64(bcount << 6) + (sph_u64)(extra);
		t1 = vectorize(0xB000000000000000ull); // (bcount >> 58) + ((sph_u64)(etype) << 55);
		TFBIG_KINIT_UI2(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);
		TFBIG_4e_UI2(0);
		TFBIG_4o_UI2(1);
		TFBIG_4e_UI2(2);
		TFBIG_4o_UI2(3);
		TFBIG_4e_UI2(4);
		TFBIG_4o_UI2(5);
		TFBIG_4e_UI2(6);
		TFBIG_4o_UI2(7);
		TFBIG_4e_UI2(8);
		TFBIG_4o_UI2(9);
		TFBIG_4e_UI2(10);
		TFBIG_4o_UI2(11);
		TFBIG_4e_UI2(12);
		TFBIG_4o_UI2(13);
		TFBIG_4e_UI2(14);
		TFBIG_4o_UI2(15);
		TFBIG_4e_UI2(16);
		TFBIG_4o_UI2(17);
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

		uint64_t *outpHash = &output64[thread * 8];
		outpHash[0] = c_PaddedMessage80[8] ^ devectorize(p[0]);
		outpHash[1] = devectorize(nounce2 ^ p[1]);
		outpHash[2] = devectorize(p[2]);
		outpHash[3] = devectorize(p[3]);
		outpHash[4] = devectorize(p[4]);
		outpHash[5] = devectorize(p[5]);
		outpHash[6] = devectorize(p[6]);
		outpHash[7] = devectorize(p[7]);
	}
}

#if __CUDA_ARCH__ > 500
#define tp 448
#else
#define tp 128
#endif

__host__
void quark_skein512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash)
{
	dim3 grid((threads + tp - 1) / tp);
	dim3 block(tp);
	quark_skein512_gpu_hash_64 << <grid, block >> >(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);

}

__host__
void quark_skein512_cpu_hash_64_quark(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash)
{
	int t = 128;
	dim3 grid((threads + t - 1) / t);
	dim3 block(t);
	quark_skein512_gpu_hash_64 << <grid, block >> >(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);

}

__host__
void quark_skein512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, uint32_t *h_nonce, uint32_t target)
{
	dim3 grid((threads + TPBf - 1) / TPBf);
	dim3 block(TPBf);

	cudaMemset(d_nonce[thr_id], 0xff, 2 * sizeof(uint32_t));

	quark_skein512_gpu_hash_64_final << <grid, block >> >(threads, startNounce, (uint64_t*)d_hash, d_nonceVector, d_nonce[thr_id], target);
	CUDA_SAFE_CALL(cudaMemcpy(h_nonce, d_nonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	/* skeincoin */
}

static uint64_t PaddedMessage[16];

static void precalc()
{
	uint64_t h0, h1, h2, h3, h4, h5, h6, h7, h8;
	uint64_t t0, t1, t2;

	h0 = 0x4903ADFF749C51CEull;
	h1 = 0x0D95DE399746DF03ull;
	h2 = 0x8FD1934127C79BCEull;
	h3 = 0x9A255629FF352CB1ull;
	h4 = 0x5DB62599DF6CA7B0ull;
	h5 = 0xEABE394CA9D5C3F4ull;
	h6 = 0x991112C71A75B523ull;
	h7 = 0xAE18A40B660FCC33ull;
	h8 = h0 ^ h1 ^ h2 ^ h3 ^ h4 ^ h5 ^ h6 ^ h7 ^ SPH_C64(0x1BD11BDAA9FC1A22);

	t0 = 64; // ptr
	t1 = 0x7000000000000000ull;
	t2 = 0x7000000000000040ull;

	uint64_t p[8];
	for (int i = 0; i<8; i++)
		p[i] = PaddedMessage[i];

	TFBIG_4e_PRE(0);
	TFBIG_4o_PRE(1);
	TFBIG_4e_PRE(2);
	TFBIG_4o_PRE(3);
	TFBIG_4e_PRE(4);
	TFBIG_4o_PRE(5);
	TFBIG_4e_PRE(6);
	TFBIG_4o_PRE(7);
	TFBIG_4e_PRE(8);
	TFBIG_4o_PRE(9);
	TFBIG_4e_PRE(10);
	TFBIG_4o_PRE(11);
	TFBIG_4e_PRE(12);
	TFBIG_4o_PRE(13);
	TFBIG_4e_PRE(14);
	TFBIG_4o_PRE(15);
	TFBIG_4e_PRE(16);
	TFBIG_4o_PRE(17);
	TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

	uint64_t buffer[9];

	buffer[0] = PaddedMessage[0] ^ p[0];
	buffer[1] = PaddedMessage[1] ^ p[1];
	buffer[2] = PaddedMessage[2] ^ p[2];
	buffer[3] = PaddedMessage[3] ^ p[3];
	buffer[4] = PaddedMessage[4] ^ p[4];
	buffer[5] = PaddedMessage[5] ^ p[5];
	buffer[6] = PaddedMessage[6] ^ p[6];
	buffer[7] = PaddedMessage[7] ^ p[7];
	buffer[8] = t2;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(precalcvalues, buffer, sizeof(buffer), 0, cudaMemcpyHostToDevice));
}

__host__
void skein512_cpu_setBlock_80(void *pdata)
{
	memcpy(&PaddedMessage[0], pdata, 80);
	memset(PaddedMessage + 10, 0, 48);

	CUDA_SAFE_CALL(
		cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, sizeof(PaddedMessage), 0, cudaMemcpyHostToDevice)
	);
	precalc();
}

__host__
void skein512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int swap)
{
	const uint32_t threadsperblock = 128;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	// hash function is cut in 2 parts
	skein512_gpu_hash_80 <<< grid, block >>> (threads, startNounce, (uint64_t*)d_hash);
	skein512_gpu_hash_close <<< grid, block >>> (threads, startNounce, (uint64_t*)d_hash);
}

