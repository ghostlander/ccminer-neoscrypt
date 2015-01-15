#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 300
/**
 * __shfl() returns the value of var held by the thread whose ID is given by srcLane.
 * If srcLane is outside the range 0..width-1, the thread's own value of var is returned.
 */
#undef __shfl
#define __shfl(var, srcLane, width) (uint32_t)(var)
#endif

#define merge8(z,x,y)\
		z=__byte_perm(x, y, 0x5140); \

#define SWAP8(x,y)\
		x=__byte_perm(x, y, 0x5410); \
		y=__byte_perm(x, y, 0x7632);

#define SWAP4(x,y)\
		t = (y<<4); \
		t = (x ^ t); \
		t = 0xf0f0f0f0UL & t; \
		x = (x ^ t); \
		t=  t>>4;\
		y=  y ^ t;

#define SWAP2(x,y)\
		t = (y<<2); \
		t = (x ^ t); \
		t = 0xccccccccUL & t; \
		x = (x ^ t); \
		t=  t>>2;\
		y=  y ^ t;

#define SWAP1(x,y)\
		t = (y+y); \
		t = (x ^ t); \
		t = 0xaaaaaaaaUL & t; \
		x = (x ^ t); \
		t=  t>>1;\
		y=  y ^ t;


__device__ __forceinline__
void to_bitslice_quad(uint32_t *const __restrict__ input, uint32_t *const __restrict__ output)
{
    uint32_t other[8];
	uint32_t d[8];
	uint32_t t;
    const unsigned int n = threadIdx.x & 3;

    #pragma unroll
    for (int i = 0; i < 8; i++) 
	{
        input[i] = __shfl((int)input[i], n ^ (3*(n >=1 && n <=2)), 4);
        other[i] = __shfl((int)input[i], (threadIdx.x + 1) & 3, 4);
        input[i] = __shfl((int)input[i], threadIdx.x & 2, 4);
        other[i] = __shfl((int)other[i], threadIdx.x & 2, 4);
        if (threadIdx.x & 1) {
            input[i] = __byte_perm(input[i], 0, 0x1032);
            other[i] = __byte_perm(other[i], 0, 0x1032);
        }
    }

	merge8(d[0], input[0], input[4]);
	merge8(d[1], other[0], other[4]);
	merge8(d[2], input[1], input[5]);
	merge8(d[3], other[1], other[5]);
	merge8(d[4], input[2], input[6]);
	merge8(d[5], other[2], other[6]);
	merge8(d[6], input[3], input[7]);
	merge8(d[7], other[3], other[7]);

	SWAP1(d[0], d[1]);
	SWAP1(d[2], d[3]);
	SWAP1(d[4], d[5]);
	SWAP1(d[6], d[7]);

	SWAP2(d[0], d[2]);
	SWAP2(d[1], d[3]);
	SWAP2(d[4], d[6]);
	SWAP2(d[5], d[7]);

	SWAP4(d[0], d[4]);
	SWAP4(d[1], d[5]);
	SWAP4(d[2], d[6]);
	SWAP4(d[3], d[7]);

	output[0] = d[0];
	output[1] = d[1];
	output[2] = d[2];
	output[3] = d[3];
	output[4] = d[4];
	output[5] = d[5];
	output[6] = d[6];
	output[7] = d[7];
}

__device__ __forceinline__
void from_bitslice_quad(const uint32_t *const __restrict__ input, uint32_t *const __restrict__ output)
{

	uint32_t d[8];

	d[0] = __byte_perm(input[0], 0, 0x2310);
	d[1] = __byte_perm(input[1], 0, 0x2310);
	d[2] = __byte_perm(input[2], 0, 0x2310);
	d[3] = __byte_perm(input[3], 0, 0x2310);
	d[4] = __byte_perm(input[4], 0, 0x2310);
	d[5] = __byte_perm(input[5], 0, 0x2310);
	d[6] = __byte_perm(input[6], 0, 0x2310);
	d[7] = __byte_perm(input[7], 0, 0x2310);


	output[0] = ((d[0] & 0x00010100) >> 8);
	output[0] |= ((d[1] & 0x00010100) >> 7);
	output[0] |= ((d[2] & 0x00010100) >> 6);
	output[0] |= ((d[3] & 0x00010100) >> 5);
	output[0] |= ((d[4] & 0x00010100) >> 4);
	output[0] |= ((d[5] & 0x00010100) >> 3);
	output[0] |= ((d[6] & 0x00010100) >> 2);
	output[0] |= ((d[7] & 0x00010100) >> 1);
	output[2] = ((d[0] &  0x00020200) >> 9);
	output[2] |= ((d[1] & 0x00020200) >> 8);
	output[2] |= ((d[2] & 0x00020200) >> 7);
	output[2] |= ((d[3] & 0x00020200) >> 6);
	output[2] |= ((d[4] & 0x00020200) >> 5);
	output[2] |= ((d[5] & 0x00020200) >> 4);
	output[2] |= ((d[6] & 0x00020200) >> 3);
	output[2] |= ((d[7] & 0x00020200) >> 2);
	output[4] = ((d[0]  & 0x00040400) >> 10);
	output[4] |= ((d[1] & 0x00040400) >> 9);
	output[4] |= ((d[2] & 0x00040400) >> 8);
	output[4] |= ((d[3] & 0x00040400) >> 7);
	output[4] |= ((d[4] & 0x00040400) >> 6);
	output[4] |= ((d[5] & 0x00040400) >> 5);
	output[4] |= ((d[6] & 0x00040400) >> 4);
	output[4] |= ((d[7] & 0x00040400) >> 3);
	output[6] = ((d[0]  & 0x00080800) >> 11);
	output[6] |= ((d[1] & 0x00080800) >> 10);
	output[6] |= ((d[2] & 0x00080800) >> 9);
	output[6] |= ((d[3] & 0x00080800) >> 8);
	output[6] |= ((d[4] & 0x00080800) >> 7);
	output[6] |= ((d[5] & 0x00080800) >> 6);
	output[6] |= ((d[6] & 0x00080800) >> 5);
	output[6] |= ((d[7] & 0x00080800) >> 4);
	output[8] = ((d[0]  & 0x00101000) >> 12);
	output[8] |= ((d[1] & 0x00101000) >> 11);
	output[8] |= ((d[2] & 0x00101000) >> 10);
	output[8] |= ((d[3] & 0x00101000) >> 9);
	output[8] |= ((d[4] & 0x00101000) >> 8);
	output[8] |= ((d[5] & 0x00101000) >> 7);
	output[8] |= ((d[6] & 0x00101000) >> 6);
	output[8] |= ((d[7] & 0x00101000) >> 5);
	output[10] = ((d[0]  & 0x00202000) >> 13);
	output[10] |= ((d[1] & 0x00202000) >> 12);
	output[10] |= ((d[2] & 0x00202000) >> 11);
	output[10] |= ((d[3] & 0x00202000) >> 10);
	output[10] |= ((d[4] & 0x00202000) >> 9);
	output[10] |= ((d[5] & 0x00202000) >> 8);
	output[10] |= ((d[6] & 0x00202000) >> 7);
	output[10] |= ((d[7] & 0x00202000) >> 6);
	output[12] = ((d[0]  & 0x00404000) >> 14);
	output[12] |= ((d[1] & 0x00404000) >> 13);
	output[12] |= ((d[2] & 0x00404000) >> 12);
	output[12] |= ((d[3] & 0x00404000) >> 11);
	output[12] |= ((d[4] & 0x00404000) >> 10);
	output[12] |= ((d[5] & 0x00404000) >> 9);
	output[12] |= ((d[6] & 0x00404000) >> 8);
	output[12] |= ((d[7] & 0x00404000) >> 7);
	output[14] = ((d[0]  & 0x00808000) >> 15);
	output[14] |= ((d[1] & 0x00808000) >> 14);
	output[14] |= ((d[2] & 0x00808000) >> 13);
	output[14] |= ((d[3] & 0x00808000) >> 12);
	output[14] |= ((d[4] & 0x00808000) >> 11);
	output[14] |= ((d[5] & 0x00808000) >> 10);
	output[14] |= ((d[6] & 0x00808000) >> 9);
	output[14] |= ((d[7] & 0x00808000) >> 8);

#pragma unroll 8
    for (int i = 0; i < 16; i+=2) {
        if (threadIdx.x & 1) output[i] = __byte_perm(output[i], 0, 0x1032);
        output[i] = __byte_perm(output[i], __shfl((int)output[i], (threadIdx.x+1)&3, 4), 0x7610);
        output[i+1] = __shfl((int)output[i], (threadIdx.x+2)&3, 4);
        if (threadIdx.x & 3) output[i] = output[i+1] = 0;
    }
}
