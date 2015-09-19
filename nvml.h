/*
 * A trivial little dlopen()-based wrapper library for the
 * NVIDIA NVML library, to allow runtime discovery of NVML on an
 * arbitrary system.  This is all very hackish and simple-minded, but
 * it serves my immediate needs in the short term until NVIDIA provides
 * a static NVML wrapper library themselves, hopefully in
 * CUDA 6.5 or maybe sometime shortly after.
 *
 * This trivial code is made available under the "new" 3-clause BSD license,
 * and/or any of the GPL licenses you prefer.
 * Feel free to use the code and modify as you see fit.
 *
 * John E. Stone - john.stone@gmail.com
 *
 */
#ifdef USE_WRAPNVML

#include "miner.h"

typedef void * nvmlDevice_t;

/* our own version of the PCI info struct */
typedef struct {
	char bus_id_str[16];             /* string form of bus info */
	uint32_t domain;
	uint32_t bus;
	uint32_t device;
	uint32_t pci_device_id;      /* combined device and vendor id */
	uint32_t pci_subsystem_id;
	uint32_t res0;               /* NVML internal use only */
	uint32_t res1;
	uint32_t res2;
	uint32_t res3;
} nvmlPciInfo_t;

enum nvmlEnableState_t {
	NVML_FEATURE_DISABLED = 0,
	NVML_FEATURE_ENABLED = 1,
	NVML_FEATURE_UNKNOWN = 2
};

enum nvmlRestrictedAPI_t {
	NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS = 0,
	NVML_RESTRICTED_API_SET_AUTO_BOOSTED_CLOCKS = 1,
	NVML_RESTRICTED_API_COUNT = 2
};

enum nvmlReturn_t {
	NVML_SUCCESS = 0,
	NVML_ERROR_UNINITIALIZED = 1,
	NVML_ERROR_INVALID_ARGUMENT = 2,
	NVML_ERROR_NOT_SUPPORTED = 3,
	NVML_ERROR_NO_PERMISSION = 4,
	NVML_ERROR_ALREADY_INITIALIZED = 5,
	NVML_ERROR_NOT_FOUND = 6,
	NVML_ERROR_INSUFFICIENT_SIZE = 7,
	NVML_ERROR_INSUFFICIENT_POWER = 8,
	NVML_ERROR_DRIVER_NOT_LOADED = 9,
	NVML_ERROR_TIMEOUT = 10,
	NVML_ERROR_UNKNOWN = 999
};

enum nvmlClockType_t {
	NVML_CLOCK_GRAPHICS = 0,
	NVML_CLOCK_SM = 1,
	NVML_CLOCK_MEM = 2
};

#define NVML_DEVICE_SERIAL_BUFFER_SIZE 30
#define NVML_DEVICE_UUID_BUFFER_SIZE 80
#define NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE 32

/*
 * Handle to hold the function pointers for the entry points we need,
 * and the shared library itself.
 */
typedef struct {
	void *nvml_dll;
	int nvml_gpucount;
	int cuda_gpucount;
	uint32_t *nvml_pci_domain_id;
	uint32_t *nvml_pci_bus_id;
	uint32_t *nvml_pci_device_id;
	uint32_t *nvml_pci_subsys_id;
	int *nvml_cuda_device_id;          /* map NVML dev to CUDA dev */
	int *cuda_nvml_device_id;          /* map CUDA dev to NVML dev */
	nvmlDevice_t *devs;
	nvmlEnableState_t *app_clocks;
	nvmlReturn_t (*nvmlInit)(void);
	nvmlReturn_t (*nvmlDeviceGetCount)(int *);
	nvmlReturn_t (*nvmlDeviceGetHandleByIndex)(int, nvmlDevice_t *);
	nvmlReturn_t (*nvmlDeviceGetAPIRestriction)(nvmlDevice_t, nvmlRestrictedAPI_t, nvmlEnableState_t *);
	nvmlReturn_t (*nvmlDeviceSetAPIRestriction)(nvmlDevice_t, nvmlRestrictedAPI_t, nvmlEnableState_t);
	nvmlReturn_t (*nvmlDeviceGetDefaultApplicationsClock)(nvmlDevice_t, nvmlClockType_t, uint32_t *);
	nvmlReturn_t (*nvmlDeviceGetApplicationsClock)(nvmlDevice_t, nvmlClockType_t, uint32_t *);
	nvmlReturn_t (*nvmlDeviceSetApplicationsClocks)(nvmlDevice_t, uint32_t, uint32_t);
	nvmlReturn_t (*nvmlDeviceGetClockInfo)(nvmlDevice_t, nvmlClockType_t, uint32_t *);
	nvmlReturn_t (*nvmlDeviceGetPciInfo)(nvmlDevice_t, nvmlPciInfo_t *);
	nvmlReturn_t (*nvmlDeviceGetName)(nvmlDevice_t, char *, int);
	nvmlReturn_t (*nvmlDeviceGetTemperature)(nvmlDevice_t, int, uint32_t *);
	nvmlReturn_t (*nvmlDeviceGetFanSpeed)(nvmlDevice_t, uint32_t *);
	nvmlReturn_t (*nvmlDeviceGetPerformanceState)(nvmlDevice_t, int *); /* enum */
	nvmlReturn_t (*nvmlDeviceGetPowerUsage)(nvmlDevice_t, uint32_t *);
	nvmlReturn_t (*nvmlDeviceGetSerial)(nvmlDevice_t, char *serial, uint32_t len);
	nvmlReturn_t (*nvmlDeviceGetUUID)(nvmlDevice_t, char *uuid, uint32_t len);
	nvmlReturn_t (*nvmlDeviceGetVbiosVersion)(nvmlDevice_t, char *version, uint32_t len);
	nvmlReturn_t (*nvmlSystemGetDriverVersion)(char *version, uint32_t len);
	char* (*nvmlErrorString)(nvmlReturn_t);
	nvmlReturn_t (*nvmlShutdown)(void);
} nvml_handle;


nvml_handle * nvml_create();
int nvml_destroy(nvml_handle *nvmlh);

/*
 * Query the number of GPUs seen by NVML
 */
int nvml_get_gpucount(nvml_handle *nvmlh, int *gpucount);

/*
 * Query the number of GPUs seen by CUDA
 */
int cuda_get_gpucount(nvml_handle *nvmlh, int *gpucount);


/*
 * query the name of the GPU model from the CUDA device ID
 *
 */
int nvml_get_gpu_name(nvml_handle *nvmlh,
                           int gpuindex,
                           char *namebuf,
                           int bufsize);

/*
 * Query the current GPU temperature (Celsius), from the CUDA device ID
 */
int nvml_get_tempC(nvml_handle *nvmlh,
                        int gpuindex, uint32_t *tempC);

/*
 * Query the current GPU fan speed (percent) from the CUDA device ID
 */
int nvml_get_fanpcnt(nvml_handle *nvmlh,
                          int gpuindex, uint32_t *fanpcnt);
                          
/*
 * Query the current graphics and memory clocks from the CUDA device ID
 */
int nvml_get_current_clocks(nvml_handle *nvmlh, int cudaindex, uint32_t *graphics_clock, uint32_t *mem_clock);

/*
 * Query the current GPU power usage in millwatts from the CUDA device ID
 *
 * This feature is only available on recent GPU generations and may be
 * limited in some cases only to Tesla series GPUs.
 * If the query is run on an unsupported GPU, this routine will return -1.
 */
int nvml_get_power_usage(nvml_handle *nvmlh,
                              int gpuindex,
                              uint32_t *milliwatts);

/* api functions */

uint32_t gpu_fanpercent(struct cgpu_info *gpu);
uint32_t gpu_fanrpm(struct cgpu_info *gpu);
float gpu_temp(struct cgpu_info *gpu);
uint32_t gpu_power(struct cgpu_info *gpu);
uint32_t gpu_usage(struct cgpu_info *gpu);
int gpu_pstate(struct cgpu_info *gpu);
int gpu_busid(struct cgpu_info *gpu);

/* pid/vid, sn and bios rev */
int gpu_info(struct cgpu_info *gpu);

/* nvapi functions */
#ifdef WIN32
int nvapi_init();
#endif

#endif /* USE_WRAPNVML */
