
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

/* Common data for all kernels */
typedef struct {
  PetscScalar   *x,*y,*w;
  PetscInt      n;
  PetscScalar   result;
  PetscScalar   alpha;
  NormType      typeUse;
  Vec*          yvec;
  PetscInt      nvec;
  PetscScalar*  results;
  PetscInt      gind;
  PetscInt      localind;
  PetscReal     localmax;
  PetscReal     localmin;
  PetscRandom   rand;
  const PetscScalar*  amult;   /* multipliers */
  PetscInt      istart;
} Kernel_Data;

Kernel_Data *kerneldatap;
Kernel_Data **pdata;
PetscInt    vecs_created=0;


#endif
