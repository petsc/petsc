
/* $Id: dfunc.h,v 1.10 1996/03/20 21:16:32 curfman Exp curfman $ */

/* This file declares some utility routines for manipulating discrete functions,
   or vectors that are associated with grids, possibly with multiple degrees of
   freedom per node).  DF = Discrete Function */

#ifndef __DISCRETE_FUNCTION
#define __DISCRETE_FUNCTION

#include "vec.h"
#include "da.h"

typedef enum {DFSEQ,DFMPI} GridType;

typedef enum {ORDER_1,ORDER_2} GridComponentOrdering;
  /* ORDER_1 - ordering by interlacing components at each grid point
     ORDER_2 - ordering by segregating unknowns according to type
          (1 component for whole grid, then the next component, etc.) */

extern int DF_COOKIE;

typedef struct _DF* DF;

extern int DFCreateMPI(MPI_Comm,int,int,GridComponentOrdering,char**,DA,DF*);
extern int DFCreateSeq(MPI_Comm,int,int,GridComponentOrdering,char**,int,int,int,DF*);
extern int DFDuplicate(DF,DF*);
extern int DFDestroy(DF);
extern int DFGetComponentVectors(Vec,DF,Vec**);
extern int DFAssembleFullVector(Vec*,DF,Vec);
extern int DFGetInfo(DF,GridType*,int*,int*,GridComponentOrdering*,int*,int*,int*);
extern int DFView(DF,Viewer);
extern int DFDrawContours(Vec,DF,int,int);
extern int DFRefineVector(Vec,DF,Vec*,DF*);
extern int DFSetCoordinates(DF,double*,double*,double*);
extern int DFGetCoordinates(DF,double**,double**,double**);
extern int DFRefineCoordinates(DF,double**,double**,double**);

#endif
