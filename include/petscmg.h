/*
      Structure used for Multigrid preconditioners 
*/
#if !defined(__PETSCMG_H)
#define __PETSCMG_H
#include "petscksp.h"
PETSC_EXTERN_CXX_BEGIN

/*E
    MGType - Determines the type of multigrid method that is run.

   Level: beginner

   Values:
+  MGMULTIPLICATIVE (default) - traditional V or W cycle as determined by MGSetCycles()
.  MGADDITIVE - the additive multigrid preconditioner where all levels are
                smoothed before updating the residual
.  MGFULL - same as multiplicative except one also performs grid sequencing, 
            that is starts on the coarsest grid, performs a cycle, interpolates
            to the next, performs a cycle etc
-  MGKASKADE - like full multigrid except one never goes back to a coarser level
               from a finer

.seealso: MGSetType()

E*/
typedef enum { MGMULTIPLICATIVE,MGADDITIVE,MGFULL,MGKASKADE } MGType;
#define MGCASCADE MGKASKADE;

#define MG_V_CYCLE     1
#define MG_W_CYCLE     2

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MGSetType(PC,MGType);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MGSetLevels(PC,PetscInt,MPI_Comm*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MGGetLevels(PC,PetscInt*);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MGSetNumberSmoothUp(PC,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MGSetNumberSmoothDown(PC,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MGSetCycles(PC,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MGSetCyclesOnLevel(PC,PetscInt,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MGSetGalerkin(PC);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MGGetSmoother(PC,PetscInt,KSP*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MGGetSmootherDown(PC,PetscInt,KSP*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MGGetSmootherUp(PC,PetscInt,KSP*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MGGetCoarseSolve(PC,KSP*);


EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MGSetRhs(PC,PetscInt,Vec);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MGSetX(PC,PetscInt,Vec);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MGSetR(PC,PetscInt,Vec);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MGSetRestriction(PC,PetscInt,Mat);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MGSetInterpolate(PC,PetscInt,Mat);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MGSetResidual(PC,PetscInt,PetscErrorCode (*)(Mat,Vec,Vec,Vec),Mat);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MGDefaultResidual(Mat,Vec,Vec,Vec);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MGCheck(PC);

PETSC_EXTERN_CXX_END
#endif
