#ifndef _VHYP_H
#define _VHYP_H

#include <petsc/private/petschypre.h>
#include <HYPRE_IJ_mv.h>
#include <_hypre_IJ_mv.h>

struct VecHYPRE_IJVector_ {
  HYPRE_IJVector ij;
  /* Support for push/pop of PETSc's Vec memory into a ParVector */
  Vec            pvec;
  HYPRE_Complex *hv;
  PetscErrorCode (*restore)(Vec, PetscScalar **);
};
typedef struct VecHYPRE_IJVector_ *VecHYPRE_IJVector;

PETSC_EXTERN PetscErrorCode VecHYPRE_IJVectorCreate(PetscLayout, VecHYPRE_IJVector *);
PETSC_EXTERN PetscErrorCode VecHYPRE_IJVectorDestroy(VecHYPRE_IJVector *);
PETSC_EXTERN PetscErrorCode VecHYPRE_IJVectorCopy(Vec, VecHYPRE_IJVector);
PETSC_EXTERN PetscErrorCode VecHYPRE_IJVectorPushVecRead(VecHYPRE_IJVector, Vec);
PETSC_EXTERN PetscErrorCode VecHYPRE_IJVectorPushVecWrite(VecHYPRE_IJVector, Vec);
PETSC_EXTERN PetscErrorCode VecHYPRE_IJVectorPushVec(VecHYPRE_IJVector, Vec);
PETSC_EXTERN PetscErrorCode VecHYPRE_IJVectorPopVec(VecHYPRE_IJVector);
PETSC_EXTERN PetscErrorCode VecHYPRE_IJBindToCPU(VecHYPRE_IJVector, PetscBool);

#endif
