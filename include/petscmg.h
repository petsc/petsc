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

EXTERN PetscErrorCode MGSetType(PC,MGType);
EXTERN PetscErrorCode MGCheck(PC);
EXTERN PetscErrorCode MGSetLevels(PC,int,MPI_Comm*);
EXTERN PetscErrorCode MGGetLevels(PC,int*);

EXTERN PetscErrorCode MGSetNumberSmoothUp(PC,int);
EXTERN PetscErrorCode MGSetNumberSmoothDown(PC,int);
EXTERN PetscErrorCode MGSetCycles(PC,int);
EXTERN PetscErrorCode MGSetCyclesOnLevel(PC,int,int);

EXTERN PetscErrorCode MGGetSmoother(PC,int,KSP*);
EXTERN PetscErrorCode MGGetSmootherDown(PC,int,KSP*);
EXTERN PetscErrorCode MGGetSmootherUp(PC,int,KSP*);
EXTERN PetscErrorCode MGGetCoarseSolve(PC,KSP*);

EXTERN PetscErrorCode MGSetRhs(PC,int,Vec);
EXTERN PetscErrorCode MGSetX(PC,int,Vec);
EXTERN PetscErrorCode MGSetR(PC,int,Vec);

EXTERN PetscErrorCode MGSetRestriction(PC,int,Mat);
EXTERN PetscErrorCode MGSetInterpolate(PC,int,Mat);
EXTERN PetscErrorCode MGSetResidual(PC,int,int (*)(Mat,Vec,Vec,Vec),Mat);
EXTERN PetscErrorCode MGDefaultResidual(Mat,Vec,Vec,Vec);


PETSC_EXTERN_CXX_END
#endif
