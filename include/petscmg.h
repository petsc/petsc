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
+  MGMULTIPLICATIVE (default) - traditional V or W cycle as determined by PCMGSetCycles()
.  MGADDITIVE - the additive multigrid preconditioner where all levels are
                smoothed before updating the residual
.  MGFULL - same as multiplicative except one also performs grid sequencing, 
            that is starts on the coarsest grid, performs a cycle, interpolates
            to the next, performs a cycle etc
-  MGKASKADE - like full multigrid except one never goes back to a coarser level
               from a finer

.seealso: PCMGSetType()

E*/
typedef enum { MGMULTIPLICATIVE,MGADDITIVE,MGFULL,MGKASKADE } MGType;
#define MGCASCADE MGKASKADE;

#define MG_V_CYCLE     1
#define MG_W_CYCLE     2

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetType(PC,MGType);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetLevels(PC,PetscInt,MPI_Comm*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGGetLevels(PC,PetscInt*);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetNumberSmoothUp(PC,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetNumberSmoothDown(PC,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetCycles(PC,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetCyclesOnLevel(PC,PetscInt,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetGalerkin(PC);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGGetSmoother(PC,PetscInt,KSP*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGGetSmootherDown(PC,PetscInt,KSP*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGGetSmootherUp(PC,PetscInt,KSP*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGGetCoarseSolve(PC,KSP*);


EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetRhs(PC,PetscInt,Vec);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetX(PC,PetscInt,Vec);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetR(PC,PetscInt,Vec);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetRestriction(PC,PetscInt,Mat);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetInterpolate(PC,PetscInt,Mat);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetResidual(PC,PetscInt,PetscErrorCode (*)(Mat,Vec,Vec,Vec),Mat);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGDefaultResidual(Mat,Vec,Vec,Vec);

PETSC_EXTERN_CXX_END
#endif
