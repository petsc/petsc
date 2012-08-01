/* "Unintrusive" algebraic DM for KKT systems defined by a matrix. */
#if !defined(__PETSCDMAKKT_H)
#define __PETSCDMAKKT_H

#include <petscdm.h>

PETSC_EXTERN PetscErrorCode DMAKKTSetDM(DM,DM);
PETSC_EXTERN PetscErrorCode DMAKKTGetDM(DM,DM*);
PETSC_EXTERN PetscErrorCode DMAKKTSetMatrix(DM,Mat);
PETSC_EXTERN PetscErrorCode DMAKKTGetMatrix(DM,Mat*);
PETSC_EXTERN PetscErrorCode DMAKKTSetFieldDecompositionName(DM,const char*);
PETSC_EXTERN PetscErrorCode DMAKKTGetFieldDecompositionName(DM,char**);
PETSC_EXTERN PetscErrorCode DMAKKTSetFieldDecomposition(DM,PetscInt, const char* const*, IS[], DM[]);
PETSC_EXTERN PetscErrorCode DMAKKTGetFieldDecomposition(DM,PetscInt*, char***, IS**, DM**);

PETSC_EXTERN PetscErrorCode DMAKKTSetDuplicateMat(DM,PetscBool);
PETSC_EXTERN PetscErrorCode DMAKKTGetDuplicateMat(DM,PetscBool*);
PETSC_EXTERN PetscErrorCode DMAKKTSetDetectSaddlePoint(DM,PetscBool);
PETSC_EXTERN PetscErrorCode DMAKKTGetDetectSaddlePoint(DM,PetscBool*);


#endif
