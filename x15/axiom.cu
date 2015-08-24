/*
 * Axiom routine (sp)
 */
extern "C"
{
#include "sph/sph_shabal.h"
#include "miner.h"

}
#include "cuda_helper.h"
static uint32_t foundnonces[MAX_GPUS][2];


extern void axiom_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t target, uint32_t *h_found);

extern void axiom_setBlock_80(void *pdata);

static bool init[MAX_GPUS] = { 0 };
uint32_t _ALIGN(128) M[65536][8];

extern "C" void axiomhash(void *output, const void *input)
{
	sph_shabal256_context ctx;
	const int N = 65536;

	
	sph_shabal256_init((void *)&ctx);
	sph_shabal256((void *)&ctx, input, 80);
	sph_shabal256_close((void *)&ctx, M[0]);

	for(int i = 1; i < N; i++) {
		sph_shabal256((void *)&ctx, M[i - 1], 32);
		sph_shabal256_close((void *)&ctx, M[i]);
	}

	for(int b = 0; b < N; b++)
	{
		const int p = b > 0 ? b - 1 : 0xFFFF;
		const int q = M[p][0] % 0xFFFF;
		const int j = (b + q) % N;

		//sph_shabal256_init(&ctx);
#if 0
		sph_shabal256(&ctx, M[p], 32);
		sph_shabal256(&ctx, M[j], 32);
#else
		uint8_t _ALIGN(128) hash[64];
		memcpy(hash, M[p], 32);
		memcpy(&hash[32], M[j], 32);
		sph_shabal256(&ctx, hash, 64);
#endif
		sph_shabal256_close(&ctx, M[b]);
	}
	memcpy(output, M[N-1], 32);
}


extern "C" int scanhash_axiom(int thr_id, uint32_t *pdata, uint32_t *ptarget, uint32_t max_nonce, unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	uint32_t _ALIGN(128) endiandata[20];
	uint32_t throughput = device_intensity(device_map[thr_id], __func__, (1 << 20));
	throughput = min(throughput, max_nonce - first_nonce);
	if (opt_benchmark)
		ptarget[7] = 0xffffff;

	uint32_t _ALIGN(128) hash64[8];

	const uint32_t Htarg = ptarget[7];

	uint32_t n = first_nonce;
	for (int k = 0; k < 20; k++)
	{
		be32enc(&endiandata[k], pdata[k]);
	}

	if (!init[thr_id])
	{
		CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		if (opt_n_gputhreads == 1)
		{
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		}
		init[thr_id] = true;
	//	cuda_check_cpu_setTarget(ptarget);
	//	axiom_setBlock_80((void*)endiandata);
		
	}

	do 
	{
/*		axiom_cpu_hash_80(thr_id, throughput, pdata[19], ptarget[7], foundnonces[thr_id]);
		if (foundnonces[thr_id][0] != 0xffffffff && foundnonces[thr_id][0]!=0)
		{
			int res = 1;
			*hashes_done = pdata[19] - first_nonce + throughput;
			if (foundnonces[thr_id][1] != 0xffffffff)
			{
				pdata[21] = foundnonces[thr_id][1];
				res++;
				if (opt_benchmark)
					applog(LOG_INFO, "GPU #%d Found second nounce %08x", thr_id, foundnonces[thr_id][1]);
			}
			pdata[19] = foundnonces[thr_id][0];
			if (opt_benchmark)
				applog(LOG_INFO, "GPU #%d Found nounce %08x", thr_id, foundnonces[thr_id][0]);
			return res;
		}		
		pdata[19] += throughput;
		*/
			be32enc(&endiandata[19], n);
			axiomhash(hash64, endiandata);
			if (hash64[7] < Htarg && fulltest(hash64, ptarget)) 
			{
				if (opt_benchmark)
					applog(LOG_INFO, "GPU #%d Found nounce %08x", thr_id, n);

				*hashes_done = n - first_nonce + 1;
				pdata[19] = n;
				return true;
			}
			n++;
			*hashes_done = n;
	} while (n < max_nonce && !work_restart[thr_id].restart);

	//	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));


	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
