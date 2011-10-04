
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

extern void* VecDot_Kernel(void*);
extern void* VecScale_Kernel(void*);
extern void* VecAXPY_Kernel(void*);
extern void* VecAYPX_Kernel(void*);
extern void* VecWAXPY_Kernel(void*);
extern void* VecNorm_Kernel(void*);
extern void* VecMDot_Kernel(void*);
extern void* VecMax_Kernel(void*);
extern void* VecMin_Kernel(void*);
extern void* VecPointwiseMult_Kernel(void*);
extern void* VecPointwiseDivide_Kernel(void*);
extern void* VecSwap_Kernel(void*);
extern void* VecSetRandom_Kernel(void*);
extern void* VecCopy_Kernel(void*);
extern void* VecMAXPY_Kernel(void*);
extern void* VecSet_Kernel(void*);

#endif
