
/* $Id: vgrid.h,v 1.1 1996/01/28 18:13:42 curfman Exp curfman $ */

/* This file declares some utility routins for manipulating vectors that are
   associated with grids */

#ifndef __VEC_GRID_UTILS
#define __VEC_GRID_UTILS

#include "vec.h"

typedef enum {STRUCTURED,UNSTRUCTURED} GridType;

typedef enum {ORDER_1,ORDER_2} GridComponentOrdering;
  /* ORDER_1 - ordering by interlacing components at each grid point
     ORDER_2 - ordering by segregating unknowns according to type
          (1 component for whole grid, then the next component, etc.) */

#define VECGRID_COOKIE         PETSC_COOKIE+40 /* need to set range for petsc cookies */

typedef struct _VecGridInfo* VecGridInfo;

extern int VecGridInfoCreate(MPI_Comm,GridType,int,VecGridInfo*);
extern int VecGridInfoDestroy(VecGridInfo);
extern int VecGridInfoSetComponents(VecGridInfo,int,GridComponentOrdering,char**);
extern int VecGridInfoSetCoordinates(VecGridInfo,double*,double*,double*,int,int,int);
extern int VecGridInfoGetComponentVecs(Vec v,VecGridInfo vgrid,Vec **vcomp);
extern int VecGridInfoDrawContours(Vec,VecGridInfo,int,int);

#endif
