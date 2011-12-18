
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
  PetscScalar   beta;
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

extern PetscErrorCode VecNorm_SeqPThread(Vec,NormType,PetscReal*);
extern PetscErrorCode VecDot_SeqPThread(Vec,Vec,PetscScalar*);
extern PetscErrorCode VecScale_SeqPThread(Vec,PetscScalar);
extern PetscErrorCode VecMDot_SeqPThread(Vec,PetscInt,const Vec[],PetscScalar*);
extern PetscErrorCode VecMax_SeqPThread(Vec,PetscInt*,PetscReal*);
extern PetscErrorCode VecMin_SeqPThread(Vec,PetscInt*,PetscReal*);
extern PetscErrorCode VecPointwiseMult_SeqPThread(Vec,Vec,Vec);
extern PetscErrorCode VecPointwiseDivide_SeqPThread(Vec,Vec,Vec);
extern PetscErrorCode VecSwap_SeqPThread(Vec,Vec);
extern PetscErrorCode VecSetRandom_SeqPThread(Vec,PetscRandom);
extern PetscErrorCode VecCopy_SeqPThread(Vec,Vec);
extern PetscErrorCode VecAXPY_SeqPThread(Vec,PetscScalar,Vec);
extern PetscErrorCode VecAYPX_SeqPThread(Vec,PetscScalar,Vec);
extern PetscErrorCode VecWAXPY_SeqPThread(Vec,PetscScalar,Vec,Vec);
extern PetscErrorCode VecMAXPY_SeqPThread(Vec,PetscInt,const PetscScalar[],Vec*);
extern PetscErrorCode VecSet_SeqPThread(Vec,PetscScalar);
extern PetscErrorCode VecSetFromOptions_SeqPThread(Vec);

EXTERN_C_BEGIN
extern PetscErrorCode VecCreate_SeqPThread(Vec);
EXTERN_C_END

#endif
