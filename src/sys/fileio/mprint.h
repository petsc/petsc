
#if !defined(__MPRINT_H)
#define __MPRINT_H

#include <petscsys.h>             /*I    "petscsys.h"   I*/
#include <stdarg.h>
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif


/* ----------------------------------------------------------------------- */
typedef struct _PrintfQueue *PrintfQueue;
struct _PrintfQueue {
  char       *string;
  int         size;
  PrintfQueue next;
};
extern PrintfQueue petsc_printfqueue,petsc_printfqueuebase;
extern int         petsc_printfqueuelength;
extern FILE        *petsc_printfqueuefile;

#endif
