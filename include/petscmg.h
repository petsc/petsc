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

EXTERN int MGSetType(PC,MGType);
EXTERN int MGCheck(PC);
EXTERN int MGSetLevels(PC,int,MPI_Comm*);
EXTERN int MGGetLevels(PC,int*);

EXTERN int MGSetNumberSmoothUp(PC,int);
EXTERN int MGSetNumberSmoothDown(PC,int);
EXTERN int MGSetCycles(PC,int);
EXTERN int MGSetCyclesOnLevel(PC,int,int);

EXTERN int MGGetSmoother(PC,int,KSP*);
EXTERN int MGGetSmootherDown(PC,int,KSP*);
EXTERN int MGGetSmootherUp(PC,int,KSP*);
EXTERN int MGGetCoarseSolve(PC,KSP*);

EXTERN int MGSetRhs(PC,int,Vec);
EXTERN int MGSetX(PC,int,Vec);
EXTERN int MGSetR(PC,int,Vec);

EXTERN int MGSetRestriction(PC,int,Mat);
EXTERN int MGSetInterpolate(PC,int,Mat);
EXTERN int MGSetResidual(PC,int,int (*)(Mat,Vec,Vec,Vec),Mat);
EXTERN int MGDefaultResidual(Mat,Vec,Vec,Vec);


PETSC_EXTERN_CXX_END
#endif
