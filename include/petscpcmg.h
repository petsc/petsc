/*
      Structure used for Multigrid preconditioners 
*/
#if !defined(__PETSCPCMG_H)
#define __PETSCPCMG_H
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
            to the next, performs a cycle etc. This is much like the F-cycle presented in "Multigrid" by Trottenberg, Oosterlee, Schuller page 49, but that
            algorithm supports smoothing on before the restriction on each level in the initial restriction to the coarsest stage. In addition that algorithm
            calls the V-cycle only on the coarser level and has a post-smoother instead.
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

extern PetscErrorCode  PCMGSetType(PC,PCMGType);
extern PetscErrorCode  PCMGSetLevels(PC,PetscInt,MPI_Comm*);
extern PetscErrorCode  PCMGGetLevels(PC,PetscInt*);

extern PetscErrorCode  PCMGSetNumberSmoothUp(PC,PetscInt);
extern PetscErrorCode  PCMGSetNumberSmoothDown(PC,PetscInt);
extern PetscErrorCode  PCMGSetCycleType(PC,PCMGCycleType);
extern PetscErrorCode  PCMGSetCycleTypeOnLevel(PC,PetscInt,PCMGCycleType);
extern PetscErrorCode  PCMGSetCyclesOnLevel(PC,PetscInt,PetscInt);
extern PetscErrorCode  PCMGMultiplicativeSetCycles(PC,PetscInt);
extern PetscErrorCode  PCMGSetGalerkin(PC,PetscBool);
extern PetscErrorCode  PCMGGetGalerkin(PC,PetscBool*);

extern PetscErrorCode  PCMGGetSmoother(PC,PetscInt,KSP*);
extern PetscErrorCode  PCMGGetSmootherDown(PC,PetscInt,KSP*);
extern PetscErrorCode  PCMGGetSmootherUp(PC,PetscInt,KSP*);
extern PetscErrorCode  PCMGGetCoarseSolve(PC,KSP*);


extern PetscErrorCode  PCMGSetRhs(PC,PetscInt,Vec);
extern PetscErrorCode  PCMGSetX(PC,PetscInt,Vec);
extern PetscErrorCode  PCMGSetR(PC,PetscInt,Vec);

extern PetscErrorCode  PCMGSetRestriction(PC,PetscInt,Mat);
extern PetscErrorCode  PCMGGetRestriction(PC,PetscInt,Mat*);
extern PetscErrorCode  PCMGSetInterpolation(PC,PetscInt,Mat);
extern PetscErrorCode  PCMGGetInterpolation(PC,PetscInt,Mat*);
extern PetscErrorCode  PCMGSetRScale(PC,PetscInt,Vec);
extern PetscErrorCode  PCMGGetRScale(PC,PetscInt,Vec*);
extern PetscErrorCode  PCMGSetResidual(PC,PetscInt,PetscErrorCode (*)(Mat,Vec,Vec,Vec),Mat);
extern PetscErrorCode  PCMGDefaultResidual(Mat,Vec,Vec,Vec);

/*E
    PCExoticType - Face based or wirebasket based coarse grid space

   Level: beginner

.seealso: PCExoticSetType(), PCEXOTIC
E*/ 
typedef enum { PC_EXOTIC_FACE,PC_EXOTIC_WIREBASKET } PCExoticType;
extern const char *PCExoticTypes[];
extern PetscErrorCode  PCExoticSetType(PC,PCExoticType);


PETSC_EXTERN_CXX_END
#endif
