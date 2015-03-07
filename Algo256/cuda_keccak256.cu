#include "miner.h"
#include <stdint.h>
#include <memory.h>
#include "cuda_helper.h"

#define UINT2(x,y) make_uint2(x,y)

static uint32_t *d_KNonce[MAX_GPUS];

__constant__ uint32_t pTarget[8];
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


__constant__ uint2 c_PaddedMessage80[10]; // padded message (80 bytes + padding?)

#if __CUDA_ARCH__ >= 350
__device__ __forceinline__
static void keccak_blockv35_32(uint2 *s)
{
	int i;
	uint2 t1, t[5], u[5], v, w;

	t1 = s[1] ^ s[16];

	/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
	u[0] = s[4] ^ ROL2(t1, 1);
	u[1] = s[0] ^ ROL2(s[2], 1);
	u[2] = t1 ^ ROL2(s[3], 1);
	u[3] = s[2] ^ ROL2(s[4], 1);
	u[4] = s[3] ^ ROL2(s[0], 1);

	/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
	s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
	s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
	s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
	s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
	s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

	/* rho pi: b[..] = rotl(a[..], ..) */
	v = s[1];
	s[1] = ROL2(s[6], 44);
	s[6] = ROL2(s[9], 20);
	s[9] = ROL2(s[22], 61);
	s[22] = ROL2(s[14], 39);
	s[14] = ROL2(s[20], 18);
	s[20] = ROL2(s[2], 62);
	s[2] = ROL2(s[12], 43);
	s[12] = ROL2(s[13], 25);
	s[13] = ROL2(s[19], 8);
	s[19] = ROL2(s[23], 56);
	s[23] = ROL2(s[15], 41);
	s[15] = ROL2(s[4], 27);
	s[4] = ROL2(s[24], 14);
	s[24] = ROL2(s[21], 2);
	s[21] = ROL2(s[8], 55);
	s[8] = ROL2(s[16], 45);
	s[16] = ROL2(s[5], 36);
	s[5] = ROL2(s[3], 28);
	s[3] = ROL2(s[18], 21);
	s[18] = ROL2(s[17], 15);
	s[17] = ROL2(s[11], 10);
	s[11] = ROL2(s[7], 6);
	s[7] = ROL2(s[10], 3);
	s[10] = ROL2(v, 1);

	/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
	v = s[0]; w = s[1]; s[0] ^= (~w) & s[2]; s[1] ^= (~s[2]) & s[3]; s[2] ^= (~s[3]) & s[4]; s[3] ^= (~s[4]) & v; s[4] ^= (~v) & w;
	v = s[5]; w = s[6]; s[5] ^= (~w) & s[7]; s[6] ^= (~s[7]) & s[8]; s[7] ^= (~s[8]) & s[9]; s[8] ^= (~s[9]) & v; s[9] ^= (~v) & w;
	v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
	v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
	v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

	/* iota: a[0,0] ^= round constant */
	s[0] = s[0]^1;	//vectorize(keccak_round_constants[0]);

#pragma unroll
	for (i = 1; i < 24; i++) {
		/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		u[0] = t[4] ^ ROL2(t[1], 1);
		u[1] = t[0] ^ ROL2(t[2], 1);
		u[2] = t[1] ^ ROL2(t[3], 1);
		u[3] = t[2] ^ ROL2(t[4], 1);
		u[4] = t[3] ^ ROL2(t[0], 1);

		/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
		s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
		s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
		s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
		s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
		s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[1];
		s[1] = ROL2(s[6], 44);
		s[6] = ROL2(s[9], 20);
		s[9] = ROL2(s[22], 61);
		s[22] = ROL2(s[14], 39);
		s[14] = ROL2(s[20], 18);
		s[20] = ROL2(s[2], 62);
		s[2] = ROL2(s[12], 43);
		s[12] = ROL2(s[13], 25);
		s[13] = ROL2(s[19], 8);
		s[19] = ROL2(s[23], 56);
		s[23] = ROL2(s[15], 41);
		s[15] = ROL2(s[4], 27);
		s[4] = ROL2(s[24], 14);
		s[24] = ROL2(s[21], 2);
		s[21] = ROL2(s[8], 55);
		s[8] = ROL2(s[16], 45);
		s[16] = ROL2(s[5], 36);
		s[5] = ROL2(s[3], 28);
		s[3] = ROL2(s[18], 21);
		s[18] = ROL2(s[17], 15);
		s[17] = ROL2(s[11], 10);
		s[11] = ROL2(s[7], 6);
		s[7] = ROL2(s[10], 3);
		s[10] = ROL2(v, 1);

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		v = s[0]; w = s[1]; s[0] ^= (~w) & s[2]; s[1] ^= (~s[2]) & s[3]; s[2] ^= (~s[3]) & s[4]; s[3] ^= (~s[4]) & v; s[4] ^= (~v) & w;
		v = s[5]; w = s[6]; s[5] ^= (~w) & s[7]; s[6] ^= (~s[7]) & s[8]; s[7] ^= (~s[8]) & s[9]; s[8] ^= (~s[9]) & v; s[9] ^= (~v) & w;
		v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
		v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
		v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

		/* iota: a[0,0] ^= round constant */
		s[0] ^= keccak_round_constants35[i]; //vectorize(keccak_round_constants[i]);
	}
}
#else

__device__ __forceinline__
static void keccak_blockv30_32(uint64_t *s, const uint64_t *keccak_round_constants)
{
	int i;
	uint64_t t1, t[5], u[5], v, w;

	/* absorb input */
	/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
	t1 = s[1] ^ s[16];

	/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
	u[0] = s[4] ^ ROTL64(t1, 1);
	u[1] = s[0] ^ ROTL64(s[2], 1);
	u[2] = t1 ^ ROTL64(s[3], 1);
	u[3] = s[2] ^ ROTL64(s[4], 1);
	u[4] = s[3] ^ ROTL64(s[0], 1);

	/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
	s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
	s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
	s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
	s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
	s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

	/* rho pi: b[..] = rotl(a[..], ..) */
	v = s[1];
	s[1] = ROTL64(s[6], 44);
	s[6] = ROTL64(s[9], 20);
	s[9] = ROTL64(s[22], 61);
	s[22] = ROTL64(s[14], 39);
	s[14] = ROTL64(s[20], 18);
	s[20] = ROTL64(s[2], 62);
	s[2] = ROTL64(s[12], 43);
	s[12] = ROTL64(s[13], 25);
	s[13] = ROTL64(s[19], 8);
	s[19] = ROTL64(s[23], 56);
	s[23] = ROTL64(s[15], 41);
	s[15] = ROTL64(s[4], 27);
	s[4] = ROTL64(s[24], 14);
	s[24] = ROTL64(s[21], 2);
	s[21] = ROTL64(s[8], 55);
	s[8] = ROTL64(s[16], 45);
	s[16] = ROTL64(s[5], 36);
	s[5] = ROTL64(s[3], 28);
	s[3] = ROTL64(s[18], 21);
	s[18] = ROTL64(s[17], 15);
	s[17] = ROTL64(s[11], 10);
	s[11] = ROTL64(s[7], 6);
	s[7] = ROTL64(s[10], 3);
	s[10] = ROTL64(v, 1);

	/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
	v = s[0]; w = s[1]; s[0] ^= (~w) & s[2]; s[1] ^= (~s[2]) & s[3]; s[2] ^= (~s[3]) & s[4]; s[3] ^= (~s[4]) & v; s[4] ^= (~v) & w;
	v = s[5]; w = s[6]; s[5] ^= (~w) & s[7]; s[6] ^= (~s[7]) & s[8]; s[7] ^= (~s[8]) & s[9]; s[8] ^= (~s[9]) & v; s[9] ^= (~v) & w;
	v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
	v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
	v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

	/* iota: a[0,0] ^= round constant */
	s[0] ^= 1;//keccak_round_constants[0];

	for (i = 1; i < 24; i++) {
		/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		u[0] = t[4] ^ ROTL64(t[1], 1);
		u[1] = t[0] ^ ROTL64(t[2], 1);
		u[2] = t[1] ^ ROTL64(t[3], 1);
		u[3] = t[2] ^ ROTL64(t[4], 1);
		u[4] = t[3] ^ ROTL64(t[0], 1);

		/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
		s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
		s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
		s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
		s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
		s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[1];
		s[1] = ROTL64(s[6], 44);
		s[6] = ROTL64(s[9], 20);
		s[9] = ROTL64(s[22], 61);
		s[22] = ROTL64(s[14], 39);
		s[14] = ROTL64(s[20], 18);
		s[20] = ROTL64(s[2], 62);
		s[2] = ROTL64(s[12], 43);
		s[12] = ROTL64(s[13], 25);
		s[13] = ROTL64(s[19], 8);
		s[19] = ROTL64(s[23], 56);
		s[23] = ROTL64(s[15], 41);
		s[15] = ROTL64(s[4], 27);
		s[4] = ROTL64(s[24], 14);
		s[24] = ROTL64(s[21], 2);
		s[21] = ROTL64(s[8], 55);
		s[8] = ROTL64(s[16], 45);
		s[16] = ROTL64(s[5], 36);
		s[5] = ROTL64(s[3], 28);
		s[3] = ROTL64(s[18], 21);
		s[18] = ROTL64(s[17], 15);
		s[17] = ROTL64(s[11], 10);
		s[11] = ROTL64(s[7], 6);
		s[7] = ROTL64(s[10], 3);
		s[10] = ROTL64(v, 1);

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		v = s[0]; w = s[1]; s[0] ^= (~w) & s[2]; s[1] ^= (~s[2]) & s[3]; s[2] ^= (~s[3]) & s[4]; s[3] ^= (~s[4]) & v; s[4] ^= (~v) & w;
		v = s[5]; w = s[6]; s[5] ^= (~w) & s[7]; s[6] ^= (~s[7]) & s[8]; s[7] ^= (~s[8]) & s[9]; s[8] ^= (~s[9]) & v; s[9] ^= (~v) & w;
		v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
		v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
		v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

		/* iota: a[0,0] ^= round constant */
		s[0] ^= keccak_round_constants[i];
	}
}
#endif

#define bitselect(a, b, c) ((a) ^ ((c) & ((b) ^ (a))))

__global__ __launch_bounds__(128,5)
void keccak256_gpu_hash_80(uint32_t threads, uint32_t startNounce, void *const __restrict__ outputHash, uint32_t *const __restrict__ resNounce)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nounce = startNounce + thread;
		uint2 bc[5], tmpxor[5], tmp1, tmp2;
		uint2 s[25];
		
		s[9] = make_uint2(c_PaddedMessage80[9].x, cuda_swab32(nounce));
		s[10] = make_uint2(1, 0);
		s[16] = make_uint2(0, 0x80000000);

		tmpxor[0] = c_PaddedMessage80[0] ^ c_PaddedMessage80[5] ^ s[10];
		tmpxor[1] = c_PaddedMessage80[1] ^ c_PaddedMessage80[6] ^ s[16];
		tmpxor[2] = c_PaddedMessage80[2] ^ c_PaddedMessage80[7];
		tmpxor[3] = c_PaddedMessage80[3] ^ c_PaddedMessage80[8];
		tmpxor[4] = c_PaddedMessage80[4] ^ s[9];

		bc[0] = tmpxor[0] ^ ROL2(tmpxor[2], 1);
		bc[1] = tmpxor[1] ^ ROL2(tmpxor[3], 1);
		bc[2] = tmpxor[2] ^ ROL2(tmpxor[4], 1);
		bc[3] = tmpxor[3] ^ ROL2(tmpxor[0], 1);
		bc[4] = tmpxor[4] ^ ROL2(tmpxor[1], 1);

		tmp1 = c_PaddedMessage80[1] ^ bc[0];

		s[0] = c_PaddedMessage80[0] ^ bc[4];
		s[1] = ROL2(c_PaddedMessage80[6] ^ bc[0], 44);
		s[6] = ROL2(s[9] ^ bc[3], 20);
		s[9] = ROL2(bc[1], 61);
		s[22] = ROL2(bc[3], 39);
		s[14] = ROL2(bc[4], 18);
		s[20] = ROL2(c_PaddedMessage80[2] ^ bc[1], 62);
		s[2] = ROL2(bc[1], 43);
		s[12] = ROL2(bc[2], 25);
		s[13] = ROL2(bc[3], 8);
		s[19] = ROL2(bc[2], 56);
		s[23] = ROL2(bc[4], 41);
		s[15] = ROL2(c_PaddedMessage80[4] ^ bc[3], 27);
		s[4] = ROL2(bc[3], 14);
		s[24] = ROL2(bc[0], 2);
		s[21] = ROL2(c_PaddedMessage80[8] ^ bc[2], 55);
		s[8] = ROL2(s[16] ^ bc[0], 45);
		s[16] = ROL2(c_PaddedMessage80[5] ^ bc[4], 36);
		s[5] = ROL2(c_PaddedMessage80[3] ^ bc[2], 28);
		s[3] = ROL2( bc[2], 21);
		s[18] = ROL2(bc[1], 15);
		s[17] = ROL2(bc[0], 10);
		s[11] = ROL2(c_PaddedMessage80[7] ^ bc[1], 6);
		s[7] = ROL2(s[10] ^ bc[4], 3);
		s[10] = ROL2(tmp1, 1);

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
			s[0] ^= keccak_round_constants35[i];
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


		if (devectorize(s[3]) <= ((uint64_t*)pTarget)[3])
		{
			uint32_t tmp = atomicCAS(resNounce, 0xffffffff, nounce);
			if (tmp != 0xffffffff)
				resNounce[1] = nounce;
		}
	}
}

__host__
void keccak256_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash, uint32_t *h_nounce)
{
	cudaMemset(d_KNonce[thr_id], 0xff, 4*sizeof(uint32_t));
	const uint32_t threadsperblock = 64;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	keccak256_gpu_hash_80<<<grid, block>>>(threads, startNounce, d_outputHash, d_KNonce[thr_id]);
	//MyStreamSynchronize(NULL, order, thr_id);
	CUDA_SAFE_CALL(cudaMemcpy(h_nounce, d_KNonce[thr_id], 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

__global__ __launch_bounds__(256,3)
void keccak256_gpu_hash_32(uint32_t threads, uint32_t startNounce, uint64_t *outputHash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
#if __CUDA_ARCH__ >= 350 /* tpr: to double check if faster on SM5+ */
		uint2 keccak_gpu_state[25];
		#pragma unroll 25
		for (int i = 0; i<25; i++) {
			if (i<4) keccak_gpu_state[i] = vectorize(outputHash[i*threads+thread]);
			else     keccak_gpu_state[i] = UINT2(0, 0);
		}
		keccak_gpu_state[4]  = UINT2(1, 0);
		keccak_gpu_state[16] = UINT2(0, 0x80000000);
		keccak_blockv35_32(keccak_gpu_state);

		#pragma unroll 4
		for (int i=0; i<4; i++)
			outputHash[i*threads+thread] = devectorize(keccak_gpu_state[i]);
#else
		uint64_t keccak_gpu_state[25];
		#pragma unroll 25
		for (int i = 0; i<25; i++) {
			if (i<4)
				keccak_gpu_state[i] = outputHash[i*threads+thread];
			else
				keccak_gpu_state[i] = 0;
		}
		keccak_gpu_state[4]  = 0x0000000000000001;
		keccak_gpu_state[16] = 0x8000000000000000;

		keccak_blockv30_32(keccak_gpu_state, keccak_round_constants);
		#pragma unroll 4
		for (int i = 0; i<4; i++)
			outputHash[i*threads + thread] = keccak_gpu_state[i];
#endif
	}
}

__host__
void keccak256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	keccak256_gpu_hash_32 <<<grid, block>>> (threads, startNounce, d_outputHash);
}

__host__
void keccak256_setBlock_80(void *pdata,const void *pTargetIn)
{
	unsigned char PaddedMessage[80];
	memcpy(PaddedMessage, pdata, 80);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(pTarget, pTargetIn, 8*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, 10*sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
}

__host__
void keccak256_cpu_init(int thr_id, uint32_t threads)
{
	CUDA_SAFE_CALL(cudaMalloc(&d_KNonce[thr_id], 4*sizeof(uint32_t)));
}