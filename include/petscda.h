/* $Id: da.h,v 1.47 1999/03/31 03:46:07 bsmith Exp bsmith $ */

/*
      Regular array object, for easy parallelism of simple grid 
   problems on regular distributed arrays.
*/
#if !defined(__DA_H)
#define __DA_H
#include "vec.h"
#include "ao.h"

#define DA_COOKIE PETSC_COOKIE+14

typedef struct _p_DA* DA;
typedef enum { DA_STENCIL_STAR, DA_STENCIL_BOX } DAStencilType;
typedef enum { DA_NONPERIODIC, DA_XPERIODIC, DA_YPERIODIC, DA_XYPERIODIC,
               DA_XYZPERIODIC, DA_XZPERIODIC, DA_YZPERIODIC,DA_ZPERIODIC} 
               DAPeriodicType;
typedef enum { DA_X, DA_Y, DA_Z } DADirection;

extern int   DACreate1d(MPI_Comm,DAPeriodicType,int,int,int,int*,DA *);
extern int   DACreate2d(MPI_Comm,DAPeriodicType,DAStencilType,int,int,int,int,int,int,int*,int*,DA *);
extern int   DACreate3d(MPI_Comm,DAPeriodicType,DAStencilType, 
                        int,int,int,int,int,int,int,int,int *,int *,int *,DA *);
extern int   DADestroy(DA);
extern int   DAView(DA,Viewer);

extern int   DAPrintHelp(DA);

extern int   DAGlobalToLocalBegin(DA,Vec, InsertMode,Vec);
extern int   DAGlobalToLocalEnd(DA,Vec, InsertMode,Vec);
extern int   DAGlobalToNaturalBegin(DA,Vec, InsertMode,Vec);
extern int   DAGlobalToNaturalEnd(DA,Vec, InsertMode,Vec);
extern int   DANaturalToGlobalBegin(DA,Vec, InsertMode,Vec);
extern int   DANaturalToGlobalEnd(DA,Vec, InsertMode,Vec);
extern int   DALocalToLocalBegin(DA,Vec, InsertMode,Vec);
extern int   DALocalToLocalEnd(DA,Vec, InsertMode,Vec);
extern int   DALocalToGlobal(DA,Vec, InsertMode,Vec);
extern int   DAGetOwnershipRange(DA,int **,int **,int **);
extern int   DACreateGlobalVector(DA,Vec *);
extern int   DACreateNaturalVector(DA,Vec *);
extern int   DACreateLocalVector(DA,Vec *);
extern int   DALoad(Viewer,int,int,int,DA *);
extern int   DAGetCorners(DA,int*,int*,int*,int*,int*,int*);
extern int   DAGetGhostCorners(DA,int*,int*,int*,int*,int*,int*);
extern int   DAGetInfo(DA,int*,int*,int*,int*,int*,int*,int*,int*,int*,DAPeriodicType*,DAStencilType*);
extern int   DAGetProcessorSubset(DA,DADirection,int,MPI_Comm*);
extern int   DARefine(DA,DA*);

extern int   DAGlobalToNaturalAllCreate(DA,VecScatter*);
extern int   DANaturalAllToGlobalCreate(DA,VecScatter*);

extern int   DAGetGlobalIndices(DA,int*,int**);
extern int   DAGetISLocalToGlobalMapping(DA,ISLocalToGlobalMapping*);

extern int   DAGetScatter(DA,VecScatter*,VecScatter*,VecScatter*);

extern int   DAGetAO(DA,AO*);
extern int   DASetCoordinates(DA,Vec); 
extern int   DAGetCoordinates(DA,Vec *);
extern int   DACreateUniformCoordinates(DA,double,double,double,double,double,double);
extern int   DASetFieldName(DA,int,const char[]);
extern int   DAGetFieldName(DA,int,char **);

#include "mat.h"
extern int   DAGetColoring(DA,ISColoring *,Mat *);

#endif
