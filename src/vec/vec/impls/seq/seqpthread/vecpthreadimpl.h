
#ifndef __VECPTHREADIMPL
#define __VECPTHREADIMPL

#include <petscsys.h>
#include <petsc-private/vecimpl.h>

/* Common data for all kernels */
typedef struct {
  Vec           X;
  PetscInt      thread_id;
  PetscScalar   *x,*y,*w;
  PetscScalar   *y0,*y1,*y2,*y3;
  PetscScalar   result,result0,result1,result2,result3;
  PetscScalar   alpha;
  PetscScalar   beta;
  NormType      typeUse;
  Vec*          yvec;
  PetscInt      nvec;
  PetscScalar*  results;
  PetscInt      localind;
  PetscReal     localmax;
  PetscReal     localmin;
  PetscRandom   rand;
  const PetscScalar*  amult;   /* multipliers */
} Vec_KernelData;

extern Vec_KernelData *vec_kerneldatap;
extern Vec_KernelData **vec_pdata;

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

EXTERN_C_BEGIN
extern PetscErrorCode VecCreate_SeqPThread(Vec);
EXTERN_C_END

#endif
