#ifndef REPORT_ONCE_H
#define REPORT_ONCE_H 1

#include "knr-compat.h"
#if defined(__cplusplus)
extern "C" {
#endif

#ifndef RO_EXTERN
#define RO_EXTERN extern
#endif

#define _P(x) x

RO_EXTERN void reportonce_files _P((int));
RO_EXTERN void reportonce_accumulate _P((int file, int line, int exception));
RO_EXTERN void reportonce_summary _P((void));
RO_EXTERN void reportonce_reset _P((void));
RO_EXTERN void reportonce_ehsfid _P((int *g_ehfid, char *routine, char *filename));

RO_EXTERN void reportonce_set_output_file _P((char *output_filename));
RO_EXTERN void reportonce_set_raw_output _P((FILE *outfile));

RO_EXTERN char *reportonce_get_filename _P((int file_id));
RO_EXTERN char *reportonce_get_routine_name _P((int file_id));

#if defined(__cplusplus)
}
#endif

#endif /* REPORT_ONCE_H */
