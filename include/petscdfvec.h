
/* $Id: vec.h,v 1.45 1996/01/26 04:35:46 bsmith Exp $ */

/* This file declares some utility routins for manipulating vectors that are
   associated with grids */

#ifndef __VEC_GRID_UTILS
#define __VEC_GRID_UTILS

#include "vec.h"

typedef enum {STRUCTURED,UNSTRUCTURED} GridType;

typedef enum {ORDER_1,ORDER_2} GridComponentOrdering;
  /* ORDER_1 - ordering by components most rapidly 
     ORDER_2 - ordering 1 component for whole grid, then the next component, etc. */

#define VECGRID_COOKIE         PETSC_COOKIE+40 /* need to set range for petsc cookies */

typedef struct _VecGridInfo* VecGridInfo;

extern int VecGridInfoCreate(MPI_Comm,GridType,int,VecGridInfo*);
extern int VecGridInfoDestroy(VecGridInfo);
extern int VecGridInfoSetComponents(VecGridInfo,int,GridComponentOrdering,char**);
extern int VecGridInfoSetCoordinates(VecGridInfo,double*,double*,double*,int,int,int);
extern int VecGridInfoGetComponentVecs(Vec v,VecGridInfo vgrid,Vec **vcomp);
extern int VecGridInfoDrawContours(Vec,VecGridInfo,int,int);

#endif
