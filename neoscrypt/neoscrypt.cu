extern "C"
{
#include "sph/neoscrypt.h"
}

#include "cuda_helper.h"
#include "miner.h"


static uint32_t *d_hash1[MAX_GPUS];
static uint32_t *d_hash2[MAX_GPUS]; // 2 streams

static uint32_t *t_hash1[MAX_GPUS];
static uint32_t *t_hash2[MAX_GPUS]; // 2 streams
static uint32_t *test[MAX_GPUS]; // 2 streams

static uint32_t *b_hash[MAX_GPUS];

extern void neoscrypt_setBlockTarget(uint32_t * data, const void *ptarget);
extern void neoscrypt_cpu_init(int thr_id, int threads, uint32_t* hash);
extern void neoscrypt_cpu_init_2stream(int thr_id, int threads, uint32_t* d_hash1, uint32_t* d_hash2, uint32_t* t_hash1, uint32_t* t_hash2, uint32_t* t_hash3, uint32_t* b_hash);


extern uint32_t neoscrypt_cpu_hash_k4(int stratum, int thr_id, int threads, uint32_t startNounce, int order);
extern uint32_t neoscrypt_cpu_hash_k4_2stream(int stratum, int thr_id, int threads, uint32_t startNounce, int order);

extern "C" int scanhash_neoscrypt(int stratum, int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x01ff;

	//	const int throughput = gpus_intensity[thr_id] ? 256 * 64 * gpus_intensity[thr_id] : 256 * 64 * 3.5;
	int intensity = (256 * 64 * 2);

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device_map[thr_id]);
	if (strstr(props.name, "970"))
	{
		intensity = (256 * 64 * 5);
	}
	else if (strstr(props.name, "980"))
	{
#if CUDART_VERSION >= 7000
		intensity = (256 * 64 * 5);
#else
		intensity = (256 * 64 * 4);
#endif
	}
	else if (strstr(props.name, "750 Ti"))
	{
#if CUDART_VERSION >= 7000
		intensity = (256 * 64 * 3.5);
#else
		intensity = (256 * 64 * 3);
#endif
	}
	else if (strstr(props.name, "750"))
	{
		intensity = ((256 * 64 * 3.5) / 2);
	}
	else if (strstr(props.name, "960"))
	{
		intensity = (256 * 64 * 3.5);
	}

	uint32_t throughput = device_intensity(device_map[thr_id], __func__, intensity) / 2;

	// throughput = min(throughput, (max_nonce - first_nonce));


	static bool init[MAX_GPUS] = { 0 };
	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		//		cudaDeviceReset();
		//		cudaSetDeviceFlags(cudaStreamNonBlocking);
//		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#if CUDART_VERSION >= 7000
		CUDA_SAFE_CALL(cudaMalloc(&d_hash1[thr_id], 32 * 130 * sizeof(uint64_t) * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&d_hash2[thr_id], 32 * 130 * sizeof(uint64_t) * throughput));
#else
		CUDA_SAFE_CALL(cudaMalloc(&d_hash1[thr_id], 32 * 130 * sizeof(uint64_t) * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&d_hash2[thr_id], 32 * 130 * sizeof(uint64_t) * throughput));
#endif

		CUDA_SAFE_CALL(cudaMalloc(&t_hash1[thr_id], 32 * sizeof(uint64_t) * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&t_hash2[thr_id], 32 * sizeof(uint64_t) * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&test[thr_id], 32 * sizeof(uint64_t) * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&b_hash[thr_id], 128 * sizeof(uint32_t) * throughput));



		neoscrypt_cpu_init_2stream(thr_id, throughput, d_hash1[thr_id], d_hash2[thr_id], t_hash1[thr_id], t_hash2[thr_id], test[thr_id], b_hash[thr_id]);
		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	if (stratum) {
		for (int k = 0; k < 20; k++)
			be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);
	}
	else {
		for (int k = 0; k < 20; k++)
			endiandata[k] = pdata[k];
	}



	neoscrypt_setBlockTarget(endiandata, ptarget);

	do {
		int order = 0;
		uint32_t foundNonce = neoscrypt_cpu_hash_k4_2stream(stratum, thr_id, throughput, pdata[19], order++);
		//		foundNonce = 10 + pdata[19];
		if (foundNonce != 0xffffffff)
		{
			if (opt_benchmark)
				applog(LOG_INFO, "GPU #%d Found nounce %08x", thr_id, foundNonce);

			uint32_t vhash64[8];

			if (stratum) {
				be32enc(&endiandata[19], foundNonce);
			}
			else {
				endiandata[19] = foundNonce;
			}
			neoscrypt((unsigned char*)endiandata, (unsigned char*)vhash64, 0x80000620);

			if (vhash64[7] <= ptarget[7]) { // && fulltest(vhash64, ptarget)) {
				pdata[19] = foundNonce;
				*hashes_done = foundNonce - first_nonce + 1;
				return 1;
			}
			else {
				*hashes_done = foundNonce - first_nonce + 1; // keeps hashrate calculation happy
				applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU!", thr_id, foundNonce);
			}

		}

		pdata[19] += throughput;
	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
