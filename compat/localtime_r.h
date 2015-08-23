#pragma once

#ifdef WIN32
#include <time.h>
#define localtime_r(src, dst) localtime_s(dst, src)
#endif

