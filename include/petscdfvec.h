
/* $Id: vgrid.h,v 1.9 1996/02/27 07:06:28 curfman Exp curfman $ */

/* This file declares some utility routines for manipulating vectors that are
   associated with multicomponent problems on grids.  VGI = VecGridInfo */

#ifndef __VEC_GRID_UTILS
#define __VEC_GRID_UTILS

#include "vec.h"
#include "da.h"

typedef enum {VGISEQ,VGIMPI} GridType;

typedef enum {ORDER_1,ORDER_2} GridComponentOrdering;
  /* ORDER_1 - ordering by interlacing components at each grid point
     ORDER_2 - ordering by segregating unknowns according to type
          (1 component for whole grid, then the next component, etc.) */

extern int VECGRID_COOKIE;

typedef struct _VGI* VGI;

extern int VGICreateMPI(MPI_Comm,int,int,GridComponentOrdering,char**,DA,VGI*);
extern int VGICreateSeq(MPI_Comm,int,int,GridComponentOrdering,char**,int,int,int,VGI*);
extern int VGIDuplicate(VGI,VGI*);
extern int VGIDestroy(VGI);
extern int VGIGetComponentVectors(Vec,VGI,Vec**);
extern int VGIAssembleFullVector(Vec*,VGI,Vec);
extern int VGIGetInfo(VGI,GridType*,int*,int*,GridComponentOrdering*,int*,int*,int*);
extern int VGIView(VGI,Viewer);
extern int VGIDrawContours(Vec,VGI,int,int);
extern int VGIRefineVector(Vec,VGI,Vec*,VGI*);
extern int VGISetCoordinates(VGI,double*,double*,double*);
extern int VGIGetCoordinates(VGI,double**,double**,double**);
extern int VGIRefineCoordinates(VGI,double**,double**,double**);

#endif
