
/* $Id: vgrid.h,v 1.4 1996/02/09 00:28:00 curfman Exp bsmith $ */

/* This file declares some utility routines for manipulating vectors that are
   associated with multicomponent problems on grids */

#ifndef __VEC_GRID_UTILS
#define __VEC_GRID_UTILS

#include "vec.h"

typedef enum {STRUCTURED,UNSTRUCTURED} GridType;

typedef enum {ORDER_1,ORDER_2} GridComponentOrdering;
  /* ORDER_1 - ordering by interlacing components at each grid point
     ORDER_2 - ordering by segregating unknowns according to type
          (1 component for whole grid, then the next component, etc.) */

extern int VECGRID_COOKIE;

typedef struct _VecGridInfo* VecGridInfo;

extern int VecGridInfoCreate(MPI_Comm,GridType,int,VecGridInfo*);
extern int VecGridInfoDuplicate(VecGridInfo,VecGridInfo*);
extern int VecGridInfoDestroy(VecGridInfo);
extern int VecGridInfoSetComponents(VecGridInfo,int,GridComponentOrdering,char**);
extern int VecGridInfoSetCoordinates(VecGridInfo,double*,double*,double*,int,int,int);
extern int VecGridInfoGetComponentVecs(Vec,VecGridInfo,Vec**);
extern int VecGridInfoAssembleGlobalVec(Vec*,VecGridInfo,Vec);
extern int VecGridInfoDrawContours(Vec,VecGridInfo,int,int);
extern int VecGridInfoRefine(Vec,VecGridInfo,int,Vec*,VecGridInfo*);

#endif
