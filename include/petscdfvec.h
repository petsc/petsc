
/* $Id: dfunc.h,v 1.11 1996/04/14 14:03:54 curfman Exp curfman $ */

/* This file declares some utility routines for manipulating discrete functions,
   or vectors that are associated with grids, possibly with multiple degrees of
   freedom per node).  DF = Discrete Function */

#ifndef __DISCRETE_FUNCTION
#define __DISCRETE_FUNCTION

#include "vec.h"
#include "da.h"

typedef enum {DFSEQ_GEN,DFMPI_GEN,DFSEQ_REG,DFMPI_REG} DFType;

typedef enum {ORDER_1,ORDER_2} DFComponentOrdering;
  /* ORDER_1 - ordering by interlacing components at each grid point
     ORDER_2 - ordering by segregating unknowns according to type
          (1 component for whole grid, then the next component, etc.) */

extern int DF_COOKIE;

typedef struct _DF* DF;

extern int DFCreateDA(MPI_Comm,char**,DA,DF*);
extern int DFCreateGeneral(MPI_Comm,DFType,int,int,DFComponentOrdering,char**,int,int,int,DF*);
extern int DFDuplicate(DF,DF*);
extern int DFDestroy(DF);
extern int DFGetComponentVectors(Vec,DF,Vec**);
extern int DFAssembleFullVector(Vec*,DF,Vec);
extern int DFGetInfo(DF,DFType*,int*,int*,DFComponentOrdering*,int*,int*,int*);
extern int DFView(DF,Viewer);
extern int DFDrawContours(Vec,DF,int,int);
extern int DFRefineVector(Vec,DF,Vec*,DF*);
extern int DFSetCoordinates(DF,double*,double*,double*);
extern int DFGetCoordinates(DF,double**,double**,double**);
extern int DFRefineCoordinates(DF,double**,double**,double**);

#endif
