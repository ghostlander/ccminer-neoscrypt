#pragma once

#ifdef WIN32
#include <Windows.h>
// #include <Processthreadsapi.h>  XXX: For Win8+ ???

#define gettid() GetCurrentThreadId()

#else
#include <sys/syscall.h>
#include <unistd.h>

#define gettid() syscall(SYS_gettid)

#endif

