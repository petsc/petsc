
/* $Id: dfvec.h,v 1.12 1996/04/15 03:06:33 curfman Exp curfman $ */

/* This file declares some utility routines for manipulating discrete functions,
   or vectors that are associated with grids, possibly with multiple degrees of
   freedom per node).  DF = Discrete Function */

#ifndef __DISCRETE_FUNCTION
#define __DISCRETE_FUNCTION

#include "vec.h"
#include "da.h"

typedef enum {DF_SEQGEN,DF_MPIGEN,DF_SEQREG,DF_MPIREG} DFType;

typedef enum {ORDER_1,ORDER_2} DFComponentOrdering;
  /* ORDER_1 - ordering by interlacing components at each grid point
     ORDER_2 - ordering by segregating unknowns according to type
          (1 component for whole grid, then the next component, etc.) */

extern int DF_COOKIE;

typedef struct _DF* DF;
#define DFVec Vec

extern int DFCreateDA(MPI_Comm,char**,DA,DF*);
extern int DFCreateGeneral(MPI_Comm,DFType,int,int,DFComponentOrdering,char**,int,int,int,DF*);
extern int DFGetInfo(DF,DFType*,int*,int*,DFComponentOrdering*,int*,int*,int*);
extern int DFDuplicate(DF,DF*);
extern int DFDestroy(DF);
extern int DFSetCoordinates(DF,int,int,int,double*,double*,double*);
extern int DFGetCoordinates(DF,int*,int*,int*,double**,double**,double**);

extern int DFVecAssociate(DF,Vec);
extern int DFVecGetDFContext(Vec,DF*);
extern int DFGetLocalDF(DF,DF*);
extern int DFVecGetComponentVectors(DFVec,int*,DFVec**);
extern int DFVecAssembleFullVector(Vec*,DFVec);
extern int DFVecView(DFVec,Viewer);
extern int DFVecDrawContours(DFVec,int,int);
extern int DFVecRefineVector(DFVec,DFVec*);

#endif
