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

extern "C" int scanhash_axiom(int thr_id, uint32_t *pdata, uint32_t *ptarget, uint32_t max_nonce, unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	uint32_t endiandata[20];
	uint32_t throughput = device_intensity(device_map[thr_id], __func__, (1 << 20));
	throughput = min(throughput, max_nonce - first_nonce);
	if (opt_benchmark)
		ptarget[7] = 0xf;

	if (!init[thr_id])
	{
		CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		if (opt_n_gputhreads == 1)
		{
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		}
		init[thr_id] = true;
		cuda_check_cpu_setTarget(ptarget);
		for (int k = 0; k < 20; k++)
		{
			be32enc(&endiandata[k], pdata[k]);
		}
		axiom_setBlock_80((void*)endiandata);
	}

	do 
	{
		axiom_cpu_hash_80(thr_id, throughput, pdata[19], ptarget[7], foundnonces[thr_id]);
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
	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));
	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
