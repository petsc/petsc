/* $Id: petscmg.h,v 1.18 2000/05/08 15:09:50 balay Exp bsmith $ */
/*
      Structure used for Multigrid preconditioners 
*/
#if !defined(__PETSCMG_H)
#define __PETSCMG_H
#include "petscsles.h"

/*  Possible Multigrid Variants */
typedef enum { MGMULTIPLICATIVE,MGADDITIVE,MGFULL,MGKASKADE } MGType;
#define MGCASCADE MGKASKADE;

#define MG_V_CYCLE     1
#define MG_W_CYCLE     2

EXTERN int MGSetType(PC,MGType);
EXTERN int MGCheck(PC);
EXTERN int MGSetLevels(PC,int);
EXTERN int MGGetLevels(PC,int*);

EXTERN int MGSetNumberSmoothUp(PC,int);
EXTERN int MGSetNumberSmoothDown(PC,int);
EXTERN int MGSetCycles(PC,int);
EXTERN int MGSetCyclesOnLevel(PC,int,int);

EXTERN int MGGetSmoother(PC,int,SLES*);
EXTERN int MGGetSmootherDown(PC,int,SLES*);
EXTERN int MGGetSmootherUp(PC,int,SLES*);
EXTERN int MGGetCoarseSolve(PC,SLES*);

EXTERN int MGSetRhs(PC,int,Vec);
EXTERN int MGSetX(PC,int,Vec);
EXTERN int MGSetR(PC,int,Vec);

EXTERN int MGSetRestriction(PC,int,Mat);
EXTERN int MGSetInterpolate(PC,int,Mat);
EXTERN int MGSetResidual(PC,int,int (*)(Mat,Vec,Vec,Vec),Mat);
EXTERN int MGDefaultResidual(Mat,Vec,Vec,Vec);


#endif

