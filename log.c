
#include "log.h"
#include "miner.h" /* TODO: Drop this header */
#include "compat/localtime_r.h"
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdarg.h>
#include <pthread.h>

extern char* format_hash(char* buf, uchar *hash);

static pthread_mutex_t  applog_lock = PTHREAD_MUTEX_INITIALIZER;

void applog(int prio, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);

#ifdef HAVE_SYSLOG_H
	if (use_syslog) {
		va_list ap2;
		char *buf;
		int len;

		/* custom colors to syslog prio */
		if (prio > LOG_DEBUG) {
			switch (prio) {
				case LOG_BLUE: prio = LOG_NOTICE; break;
			}
		}

		va_copy(ap2, ap);
		len = vsnprintf(NULL, 0, fmt, ap2) + 1;
		va_end(ap2);
		buf = (char*) alloca(len);
		if (vsnprintf(buf, len, fmt, ap) >= 0)
			syslog(prio, "%s", buf);
	}
#else
	if (0) {}
#endif
	else {
		const char* color = "";
		char* hdr     = NULL;
		int   hdr_len = 0;
		struct tm tm;
		time_t now = time(NULL);

		localtime_r(&now, &tm);
		
		if (use_colors) {
			switch (prio) {
				case LOG_ERR:     color = CL_RED; break;
				case LOG_WARNING: color = CL_YLW; break;
				case LOG_NOTICE:  color = CL_WHT; break;
				case LOG_INFO:    color = ""; break;
				case LOG_DEBUG:   color = CL_GRY; break;

				case LOG_BLUE:
					prio = LOG_NOTICE;
					color = CL_CYN;
					break;
			}
		} else if (prio == LOG_BLUE) {
			prio = LOG_NOTICE;
		}
#define HDR_FMT "[%d-%02d-%02d %02d:%02d:%02d]%s "

		hdr_len = snprintf(hdr, hdr_len, HDR_FMT,
			tm.tm_year + 1900,
			tm.tm_mon + 1,
			tm.tm_mday,
			tm.tm_hour,
			tm.tm_min,
			tm.tm_sec,
			color);

		if (hdr_len > 0) {
			hdr_len += 1; /* Make room for NUL terminal */
			if (NULL != (hdr = alloca(hdr_len * sizeof(char)))) {
				char* msg     = NULL;
				int   msg_len = 0;
				va_list ap2;

				va_copy(ap2, ap);
				msg_len = vsnprintf(msg, msg_len, fmt, ap2);
				va_end(ap2);

				if (msg_len > 0) {
					msg_len += 1; /* Make room for NUL terminal */

					if (NULL != (msg = alloca(msg_len * sizeof(char)))) {
						const char* eol = "\n";

						if (use_colors)
							eol = CL_N "\n";

						snprintf(hdr, hdr_len, HDR_FMT,
							tm.tm_year + 1900,
							tm.tm_mon + 1,
							tm.tm_mday,
							tm.tm_hour,
							tm.tm_min,
							tm.tm_sec,
							color);
						vsnprintf(msg, msg_len, fmt, ap);

						pthread_mutex_lock(&applog_lock);
						fwrite(hdr, sizeof(char), hdr_len, stderr);
						fwrite(msg, sizeof(char), msg_len, stderr);
						fwrite(eol, sizeof(char), strlen(eol), stderr);
						pthread_mutex_unlock(&applog_lock);

						fflush(stderr);
					}
				}
			}
		}
	}
	va_end(ap);
}

/* to debug diff in data */
extern void applog_compare_hash(uchar *hash, uchar *hash2)
{
	char s[256] = "";
	int len = 0;
	for (int i=0; i < 32; i += 4) {
		const char *color = memcmp(hash+i, hash2+i, 4) ? CL_WHT : CL_GRY;
		len += sprintf(s+len, "%s%02x%02x%02x%02x " CL_GRY, color,
			hash[i], hash[i+1], hash[i+2], hash[i+3]);
		s[len] = '\0';
	}
	applog(LOG_DEBUG, "%s", s);
}

extern void applog_hash(uchar *hash)
{
	char s[128] = {'\0'};
	applog(LOG_DEBUG, "%s", format_hash(s, hash));
}

