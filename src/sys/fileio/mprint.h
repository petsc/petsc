#pragma once

#include <petscsys.h> /*I    "petscsys.h"   I*/
#include <petsc/private/petscimpl.h>

/* ----------------------------------------------------------------------- */
typedef struct _PrintfQueue *PrintfQueue;
struct _PrintfQueue {
  char       *string;
  size_t      size;
  PrintfQueue next;
};

PETSC_INTERN PrintfQueue petsc_printfqueue;
PETSC_INTERN PrintfQueue petsc_printfqueuebase;
PETSC_INTERN int         petsc_printfqueuelength;
