
/* $Id: dfvec.h,v 1.12 1996/04/15 03:06:33 curfman Exp curfman $ */

/* This file declares some utility routines for manipulating discrete functions,
   or vectors that are associated with grids, possibly with multiple degrees of
   freedom per node).  DFVec = Discrete Function */

#ifndef __DISCRETE_FUNCTION
#define __DISCRETE_FUNCTION

#include "vec.h"
#include "da.h"

typedef enum {DFVEC_SEQGEN,DFVEC_MPIGEN,DFVEC_SEQREG,DFVEC_MPIREG} DFVecType;

typedef enum {ORDER_1,ORDER_2} DFVecComponentOrdering;
  /* ORDER_1 - ordering by interlacing components at each grid point
     ORDER_2 - ordering by segregating unknowns according to type
          (1 component for whole grid, then the next component, etc.) */

extern int DFVEC_COOKIE;

typedef struct _DFVec* DFVec;

extern int DFVecCreateDA(MPI_Comm,char**,DA,DFVec*);
extern int DFVecCreateGeneral(MPI_Comm,DFVecType,int,int,DFVecComponentOrdering,char**,int,int,int,DFVec*);
extern int DFVecDuplicate(DFVec,DFVec*);
extern int DFVecDestroy(DFVec);
extern int DFVecGetComponentVectors(Vec,DFVec,Vec**);
extern int DFVecAssembleFullVector(Vec*,DFVec,Vec);
extern int DFVecGetInfo(DFVec,DFVecType*,int*,int*,DFVecComponentOrdering*,int*,int*,int*);
extern int DFVecView(DFVec,Viewer);
extern int DFVecDrawContours(Vec,DFVec,int,int);
extern int DFVecRefineVector(Vec,DFVec,Vec*,DFVec*);
extern int DFVecSetCoordinates(DFVec,double*,double*,double*);
extern int DFVecGetCoordinates(DFVec,double**,double**,double**);
extern int DFVecRefineCoordinates(DFVec,double**,double**,double**);

#endif
