/* "Unintrusive" algebraic DM for KKT systems defined by a matrix. */
#if !defined(__PETSCDMAKKT_H)
#define __PETSCDMAKKT_H

#include "petscdm.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscErrorCode   DMAKKTSetDM(DM,DM);
extern PetscErrorCode   DMAKKTGetDM(DM,DM*);
extern PetscErrorCode   DMAKKTSetMatrix(DM,Mat);
extern PetscErrorCode   DMAKKTGetMatrix(DM,Mat*);
extern PetscErrorCode   DMAKKTSetDecompositionName(DM,const char*);
extern PetscErrorCode   DMAKKTGetDecompositionName(DM,char**);
extern PetscErrorCode   DMAKKTSetDecomposition(DM,PetscInt, const char* const*, IS[], DM[]);
extern PetscErrorCode   DMAKKTGetDecomposition(DM,PetscInt*, char***, IS**, DM**);

extern PetscErrorCode   DMAKKTSetDuplicateMat(DM,PetscBool);
extern PetscErrorCode   DMAKKTGetDuplicateMat(DM,PetscBool*);
extern PetscErrorCode   DMAKKTSetDetectSaddlePoint(DM,PetscBool);
extern PetscErrorCode   DMAKKTGetDetectSaddlePoint(DM,PetscBool*);


PETSC_EXTERN_CXX_END
#endif
