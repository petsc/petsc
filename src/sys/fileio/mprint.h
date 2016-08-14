
#if !defined(__MPRINT_H)
#define __MPRINT_H

#include <petscsys.h>             /*I    "petscsys.h"   I*/
#include <petsc/private/petscimpl.h>

/* ----------------------------------------------------------------------- */
typedef struct _PrintfQueue *PrintfQueue;
struct _PrintfQueue {
  char        *string;
  int         size;
  PrintfQueue next;
};
extern PrintfQueue petsc_printfqueue,petsc_printfqueuebase;
extern int         petsc_printfqueuelength;

#endif
