
#ifndef __VECPTHREADIMPL
#define __VECPTHREADIMPL

#include <petscsys.h>
#include <private/vecimpl.h>

typedef struct {
  VECHEADER
  PetscInt nthreads;  /* Number of threads */
  PetscInt *arrindex; /* starting array indices for each thread */
  PetscInt *nelem;    /* Number of array elements assigned to each thread */
}Vec_SeqPthread;

#endif
