
#ifndef __THREADCOMMIMPL_H
#define __THREADCOMMIMPL_H

#include <petscthreadcomm.h>

typedef struct _PetscThreadCommOps *PetscThreadCommOps;
struct _PetscThreadCommOps {
  PetscErrorCode (*destroy)(PetscThreadComm);
  PetscErrorCode (*runkernel)(PetscThreadComm,PetscErrorCode (*)(void*),void**);
  PetscErrorCode (*view)(PetscThreadComm,PetscViewer);
};

struct _p_PetscThreadComm{
  PETSCHEADER          (struct _PetscThreadCommOps);
  PetscInt             nworkThreads; /* Number of threads in the pool */
  PetscInt             *affinities;  /* Thread affinity */
  void                 *data;    /* implementation specific data */
};

#endif
