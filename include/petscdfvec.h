
/* $Id: vgrid.h,v 1.7 1996/02/26 02:44:30 curfman Exp curfman $ */

/* This file declares some utility routines for manipulating vectors that are
   associated with multicomponent problems on grids.  VGI = VecGridInfo */

#ifndef __VEC_GRID_UTILS
#define __VEC_GRID_UTILS

#include "vec.h"
#include "da.h"

typedef enum {STRUCT_SEQ,STRUCT_MPI,UNSTRUCT_SEQ,UNSTRUCT_MPI} GridType;

typedef enum {ORDER_1,ORDER_2} GridComponentOrdering;
  /* ORDER_1 - ordering by interlacing components at each grid point
     ORDER_2 - ordering by segregating unknowns according to type
          (1 component for whole grid, then the next component, etc.) */

extern int VECGRID_COOKIE;

typedef struct _VGI* VGI;

extern int VGICreate(MPI_Comm,GridType,int,int,GridComponentOrdering,char**,int,int,int,VGI*);
extern int VGIDuplicate(VGI,VGI*);
extern int VGIDestroy(VGI);
extern int VGIGetComponentVectors(Vec,VGI,Vec**);
extern int VGIAssembleFullVector(Vec*,VGI,Vec);
extern int VGIGetInfo(VGI,GridType*,int*,int*,GridComponentOrdering*,int*,int*,int*);
extern int VGIDrawContours(Vec,VGI,int,int);
extern int VGIRefineVector(Vec,VGI,DA,Vec*,VGI*);
extern int VGISetCoordinates(VGI,Scalar*,Scalar*,Scalar*);
extern int VGIGetCoordinates(VGI,Scalar**,Scalar**,Scalar**);
extern int VGIRefineCoordinates(VGI,DA,Scalar**,Scalar**,Scalar**);

#endif
