/* $Id: mg.h,v 1.7 1995/06/07 16:32:01 bsmith Exp curfman $ */
/*
      Structure used for Multigrid code 
*/
#if !defined(__MG_PACKAGE)
#define __MG_PACKAGE
#include "sles.h"

/*  Possible Multigrid Variants */
typedef enum { MGMULTIPLICATIVE, MGADDITIVE, MGFULL, MGKASKADE } MGMethod;

#define MG_V_CYCLE     1
#define MG_W_CYCLE     2

extern int MGSetMethod(PC,MGMethod);
extern int MGCheck(PC);
extern int MGSetLevels(PC,int);

extern int MGSetNumberSmoothUp(PC,int);
extern int MGSetNumberSmoothDown(PC,int);
extern int MGSetCycles(PC,int);
extern int MGSetCyclesOnLevel(PC,int,int);

extern int MGGetSmoother(PC,int,SLES*);
extern int MGGetSmootherDown(PC,int,SLES*);
extern int MGGetSmootherUp(PC,int,SLES*);
extern int MGGetCoarseSolve(PC,SLES*);

extern int MGSetRhs(PC,int,Vec);
extern int MGSetX(PC,int,Vec);
extern int MGSetR(PC,int,Vec);

extern int MGSetRestriction(PC,int,Mat);
extern int MGSetInterpolate(PC,int,Mat);
extern int MGSetResidual(PC,int,int (*)(Mat,Vec,Vec,Vec),Mat);
extern int MGDefaultResidual(Mat,Vec,Vec,Vec);

#endif

