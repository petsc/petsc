/*
      Structure used for Multigrid preconditioners 
*/
#if !defined(__PETSCPCMG_H)
#define __PETSCPCMG_H
#include "petscksp.h"

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
            to the next, performs a cycle etc. This is much like the F-cycle presented in "Multigrid" by Trottenberg, Oosterlee, Schuller page 49, but that
            algorithm supports smoothing on before the restriction on each level in the initial restriction to the coarsest stage. In addition that algorithm
            calls the V-cycle only on the coarser level and has a post-smoother instead.
-  PC_MG_KASKADE - like full multigrid except one never goes back to a coarser level
               from a finer

.seealso: PCMGSetType()

E*/
typedef enum { PC_MG_MULTIPLICATIVE,PC_MG_ADDITIVE,PC_MG_FULL,PC_MG_KASKADE } PCMGType;
PETSC_EXTERN const char *PCMGTypes[];
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
PETSC_EXTERN const char *PCMGCycleTypes[];

PETSC_EXTERN PetscErrorCode PCMGSetType(PC,PCMGType);
PETSC_EXTERN PetscErrorCode PCMGSetLevels(PC,PetscInt,MPI_Comm*);
PETSC_EXTERN PetscErrorCode PCMGGetLevels(PC,PetscInt*);

PETSC_EXTERN PetscErrorCode PCMGSetNumberSmoothUp(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCMGSetNumberSmoothDown(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCMGSetCycleType(PC,PCMGCycleType);
PETSC_EXTERN PetscErrorCode PCMGSetCycleTypeOnLevel(PC,PetscInt,PCMGCycleType);
PETSC_EXTERN PetscErrorCode PCMGSetCyclesOnLevel(PC,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode PCMGMultiplicativeSetCycles(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCMGSetGalerkin(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCMGGetGalerkin(PC,PetscBool*);

PETSC_EXTERN PetscErrorCode PCMGGetSmoother(PC,PetscInt,KSP*);
PETSC_EXTERN PetscErrorCode PCMGGetSmootherDown(PC,PetscInt,KSP*);
PETSC_EXTERN PetscErrorCode PCMGGetSmootherUp(PC,PetscInt,KSP*);
PETSC_EXTERN PetscErrorCode PCMGGetCoarseSolve(PC,KSP*);


PETSC_EXTERN PetscErrorCode PCMGSetRhs(PC,PetscInt,Vec);
PETSC_EXTERN PetscErrorCode PCMGSetX(PC,PetscInt,Vec);
PETSC_EXTERN PetscErrorCode PCMGSetR(PC,PetscInt,Vec);

PETSC_EXTERN PetscErrorCode PCMGSetRestriction(PC,PetscInt,Mat);
PETSC_EXTERN PetscErrorCode PCMGGetRestriction(PC,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode PCMGSetInterpolation(PC,PetscInt,Mat);
PETSC_EXTERN PetscErrorCode PCMGGetInterpolation(PC,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode PCMGSetRScale(PC,PetscInt,Vec);
PETSC_EXTERN PetscErrorCode PCMGGetRScale(PC,PetscInt,Vec*);
PETSC_EXTERN PetscErrorCode PCMGSetResidual(PC,PetscInt,PetscErrorCode (*)(Mat,Vec,Vec,Vec),Mat);
PETSC_EXTERN PetscErrorCode PCMGDefaultResidual(Mat,Vec,Vec,Vec);

/*E
    PCExoticType - Face based or wirebasket based coarse grid space

   Level: beginner

.seealso: PCExoticSetType(), PCEXOTIC
E*/ 
typedef enum { PC_EXOTIC_FACE,PC_EXOTIC_WIREBASKET } PCExoticType;
PETSC_EXTERN const char *PCExoticTypes[];
PETSC_EXTERN PetscErrorCode PCExoticSetType(PC,PCExoticType);


#endif
