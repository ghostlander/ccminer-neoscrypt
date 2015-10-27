/*
 * Bastion routine ((sp) 27-oct-2015
 */
#include "uint256.h"

extern "C"
{
#include "sph/sph_shabal.h"
#include "sph/sph_echo.h"
#include "sph/sph_luffa.h"
#include "sph/sph_whirlpool.h"
#include "sph/sph_skein.h"
#include "sph/sph_hamsi.h"
#include "sph/sph_fugue.h"
#include "sph/hefty1.h"
}
extern "C"
{
#include "miner.h"
}
#include "cuda_helper.h"
static uint32_t foundnonces[MAX_GPUS][2];


//extern void bastion_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t target, uint32_t *h_found);
//extern void bastion_setBlock_80(void *pdata);

static bool init[MAX_GPUS] = { 0 };
uint32_t _ALIGN(128) M[65536][8];

#define BASTION_BLKHDR_SZ		80

extern "C" void bastionhash(uint32_t *hash, const unsigned char *input)
{
	sph_echo512_context ctx_echo;
	sph_luffa512_context ctx_luffa;
	sph_fugue512_context ctx_fugue;
	sph_whirlpool_context ctx_whirlpool;
	sph_shabal512_context ctx_shabal;
	sph_skein512_context ctx_skein;
	sph_hamsi512_context ctx_hamsi;
	static unsigned char pblank[1];

	uint32_t mask = 8;
	uint32_t zero = 0;

	HEFTY1((const unsigned char*)&input[0], BASTION_BLKHDR_SZ, (unsigned char*)&hash[0]);

	sph_luffa512_init(&ctx_luffa);
	sph_luffa512(&ctx_luffa, static_cast<const void*>(&hash[0]), 64);
	sph_luffa512_close(&ctx_luffa, static_cast<void*>(&hash[1]));

	if ((hash[1] & mask) != zero)
	{
		sph_fugue512_init(&ctx_fugue);
		sph_fugue512(&ctx_fugue, static_cast<const void*>(&hash[1]), 64);
		sph_fugue512_close(&ctx_fugue, static_cast<void*>(&hash[2]));
	}
	else
	{
		sph_skein512_init(&ctx_skein);
		sph_skein512(&ctx_skein, static_cast<const void*>(&hash[1]), 64);
		sph_skein512_close(&ctx_skein, static_cast<void*>(&hash[2]));
	}


	sph_whirlpool_init(&ctx_whirlpool);
	sph_whirlpool(&ctx_whirlpool, static_cast<const void*>(&hash[2]), 64);
	sph_whirlpool_close(&ctx_whirlpool, static_cast<void*>(&hash[3]));

	sph_fugue512_init(&ctx_fugue);
	sph_fugue512(&ctx_fugue, static_cast<const void*>(&hash[3]), 64);
	sph_fugue512_close(&ctx_fugue, static_cast<void*>(&hash[4]));

	if ((hash[4] & mask) != zero)
	{
		sph_echo512_init(&ctx_echo);
		sph_echo512(&ctx_echo, static_cast<const void*>(&hash[4]), 64);
		sph_echo512_close(&ctx_echo, static_cast<void*>(&hash[5]));
	}
	else
	{
		sph_luffa512_init(&ctx_luffa);
		sph_luffa512(&ctx_luffa, static_cast<const void*>(&hash[4]), 64);
		sph_luffa512_close(&ctx_luffa, static_cast<void*>(&hash[5]));
	}

	sph_shabal512_init(&ctx_shabal);
	sph_shabal512(&ctx_shabal, static_cast<const void*>(&hash[5]), 64);
	sph_shabal512_close(&ctx_shabal, static_cast<void*>(&hash[6]));

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, static_cast<const void*>(&hash[6]), 64);
	sph_skein512_close(&ctx_skein, static_cast<void*>(&hash[7]));

	if ((hash[7] & mask) != zero)
	{
		sph_shabal512_init(&ctx_shabal);
		sph_shabal512(&ctx_shabal, static_cast<const void*>(&hash[7]), 64);
		sph_shabal512_close(&ctx_shabal, static_cast<void*>(&hash[8]));
	}
	else
	{
		sph_whirlpool_init(&ctx_whirlpool);
		sph_whirlpool(&ctx_whirlpool, static_cast<const void*>(&hash[7]), 64);
		sph_whirlpool_close(&ctx_whirlpool, static_cast<void*>(&hash[8]));
	}

	sph_shabal512_init(&ctx_shabal);
	sph_shabal512(&ctx_shabal, static_cast<const void*>(&hash[8]), 64);
	sph_shabal512_close(&ctx_shabal, static_cast<void*>(&hash[9]));

	if ((hash[9] & mask) != zero)
	{
		sph_hamsi512_init(&ctx_hamsi);
		sph_hamsi512(&ctx_hamsi, static_cast<const void*>(&hash[9]), 64);
		sph_hamsi512_close(&ctx_hamsi, static_cast<void*>(&hash[10]));
	}
	else
	{
		sph_luffa512_init(&ctx_luffa);
		sph_luffa512(&ctx_luffa, static_cast<const void*>(&hash[9]), 64);
		sph_luffa512_close(&ctx_luffa, static_cast<void*>(&hash[10]));
	}
}

extern "C" int scanhash_bastion(int thr_id, uint32_t *pdata, uint32_t *ptarget, uint32_t max_nonce, unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	uint32_t _ALIGN(128) endiandata[20];
	uint32_t throughput = device_intensity(device_map[thr_id], __func__, (1 << 20));
	throughput = min(throughput, max_nonce - first_nonce);
	if (opt_benchmark)
		ptarget[7] = 0xffff;

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
		if (!opt_cpumining) cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		if (opt_n_gputhreads == 1)
		{
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		}
		init[thr_id] = true;
	//	cuda_check_cpu_setTarget(ptarget);
	//	bastion_setBlock_80((void*)endiandata);
		
	}

	do 
	{
/*		bastion_cpu_hash_80(thr_id, throughput, pdata[19], ptarget[7], foundnonces[thr_id]);
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
			bastionhash(hash64,(const unsigned char *) endiandata);
			if (hash64[7] < Htarg && fulltest(hash64, ptarget)) 
			{
				if (opt_benchmark)
					applog(LOG_INFO, "GPU #%d Found nounce %08x", thr_id, n);

				*hashes_done = n - first_nonce;
				pdata[19] = n;
				return true;
			}
			n++;
			*hashes_done = n;
	} while (n < max_nonce && !scan_abort_flag && !work_restart[thr_id].restart);

	//	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));


	*hashes_done = pdata[19] - first_nonce;
	return 0;
}
