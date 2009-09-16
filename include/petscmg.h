/*
      Structure used for Multigrid preconditioners 
*/
#if !defined(__PETSCMG_H)
#define __PETSCMG_H
#include "petscksp.h"
PETSC_EXTERN_CXX_BEGIN

/*E
    PCMGType - Determines the type of multigrid method that is run.

   Level: beginner

   Values:
+  PC_MG_MULTIPLICATIVE (default) - traditional V or W cycle as determined by PCMGSetCycles()
.  PC_MG_ADDITIVE - the additive multigrid preconditioner where all levels are
                smoothed before updating the residual. This only uses the 
                down smoother, in the preconditioner the upper smoother is ignored
.  PC_MG_FULL - same as multiplicative except one also performs grid sequencing, 
            that is starts on the coarsest grid, performs a cycle, interpolates
            to the next, performs a cycle etc
-  PC_MG_KASKADE - like full multigrid except one never goes back to a coarser level
               from a finer

.seealso: PCMGSetType()

E*/
typedef enum { PC_MG_MULTIPLICATIVE,PC_MG_ADDITIVE,PC_MG_FULL,PC_MG_KASKADE } PCMGType;
extern const char *PCMGTypes[];
#define PC_MG_CASCADE PC_MG_KASKADE;

/*E
    PCMGCycleType - Use V-cycle or W-cycle

   Level: beginner

   Values:
+  PC_MG_V_CYCLE
-  PC_MG_W_CYCLE

.seealso: PCMGSetCycleType()

E*/
typedef enum { PC_MG_CYCLE_V = 1,PC_MG_CYCLE_W = 2 } PCMGCycleType;
extern const char *PCMGCycleTypes[];

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetType(PC,PCMGType);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetLevels(PC,PetscInt,MPI_Comm*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGGetLevels(PC,PetscInt*);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetNumberSmoothUp(PC,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetNumberSmoothDown(PC,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetCycleType(PC,PCMGCycleType);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetCycleTypeOnLevel(PC,PetscInt,PCMGCycleType);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetCyclesOnLevel(PC,PetscInt,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGMultiplicativeSetCycles(PC,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetGalerkin(PC);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGGetGalerkin(PC,PetscTruth*);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGGetSmoother(PC,PetscInt,KSP*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGGetSmootherDown(PC,PetscInt,KSP*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGGetSmootherUp(PC,PetscInt,KSP*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGGetCoarseSolve(PC,KSP*);


EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetRhs(PC,PetscInt,Vec);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetX(PC,PetscInt,Vec);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetR(PC,PetscInt,Vec);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetRestriction(PC,PetscInt,Mat);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetInterpolation(PC,PetscInt,Mat);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetResidual(PC,PetscInt,PetscErrorCode (*)(Mat,Vec,Vec,Vec),Mat);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCMGDefaultResidual(Mat,Vec,Vec,Vec);

#include "petscda.h"
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCSetDA(PC,DA);

/*E
    PCExoticType - Face based or wirebasket based coarse grid space

   Level: beginner

.seealso: PCExoticSetType(), PCEXOTIC
E*/ 
typedef enum { PC_EXOTIC_FACE,PC_EXOTIC_WIREBASKET } PCExoticType;
extern const char *PCExoticTypes[];
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCExoticSetType(PC,PCExoticType);


PETSC_EXTERN_CXX_END
#endif
