/* $Id: petscda.h,v 1.53 2000/05/08 15:09:50 balay Exp bsmith $ */

/*
      Regular array object, for easy parallelism of simple grid 
   problems on regular distributed arrays.
*/
#if !defined(__PETSCDA_H)
#define __PETSCDA_H
#include "petscvec.h"
#include "petscao.h"

#define DA_COOKIE PETSC_COOKIE+14

typedef struct _p_DA* DA;
typedef enum { DA_STENCIL_STAR,DA_STENCIL_BOX } DAStencilType;
typedef enum { DA_NONPERIODIC,DA_XPERIODIC,DA_YPERIODIC,DA_XYPERIODIC,
               DA_XYZPERIODIC,DA_XZPERIODIC,DA_YZPERIODIC,DA_ZPERIODIC} 
               DAPeriodicType;
typedef enum { DA_X,DA_Y,DA_Z } DADirection;

EXTERN int   DACreate1d(MPI_Comm,DAPeriodicType,int,int,int,int*,DA *);
EXTERN int   DACreate2d(MPI_Comm,DAPeriodicType,DAStencilType,int,int,int,int,int,int,int*,int*,DA *);
EXTERN int   DACreate3d(MPI_Comm,DAPeriodicType,DAStencilType,
                        int,int,int,int,int,int,int,int,int *,int *,int *,DA *);
EXTERN int   DADestroy(DA);
EXTERN int   DAView(DA,Viewer);

EXTERN int   DAPrintHelp(DA);

EXTERN int   DAGlobalToLocalBegin(DA,Vec,InsertMode,Vec);
EXTERN int   DAGlobalToLocalEnd(DA,Vec,InsertMode,Vec);
EXTERN int   DAGlobalToNaturalBegin(DA,Vec,InsertMode,Vec);
EXTERN int   DAGlobalToNaturalEnd(DA,Vec,InsertMode,Vec);
EXTERN int   DANaturalToGlobalBegin(DA,Vec,InsertMode,Vec);
EXTERN int   DANaturalToGlobalEnd(DA,Vec,InsertMode,Vec);
EXTERN int   DALocalToLocalBegin(DA,Vec,InsertMode,Vec);
EXTERN int   DALocalToLocalEnd(DA,Vec,InsertMode,Vec);
EXTERN int   DALocalToGlobal(DA,Vec,InsertMode,Vec);
EXTERN int   DAGetOwnershipRange(DA,int **,int **,int **);
EXTERN int   DACreateGlobalVector(DA,Vec *);
EXTERN int   DACreateNaturalVector(DA,Vec *);
EXTERN int   DACreateLocalVector(DA,Vec *);
EXTERN int   DALoad(Viewer,int,int,int,DA *);
EXTERN int   DAGetCorners(DA,int*,int*,int*,int*,int*,int*);
EXTERN int   DAGetGhostCorners(DA,int*,int*,int*,int*,int*,int*);
EXTERN int   DAGetInfo(DA,int*,int*,int*,int*,int*,int*,int*,int*,int*,DAPeriodicType*,DAStencilType*);
EXTERN int   DAGetProcessorSubset(DA,DADirection,int,MPI_Comm*);
EXTERN int   DARefine(DA,DA*);

EXTERN int   DAGlobalToNaturalAllCreate(DA,VecScatter*);
EXTERN int   DANaturalAllToGlobalCreate(DA,VecScatter*);

EXTERN int   DAGetGlobalIndices(DA,int*,int**);
EXTERN int   DAGetISLocalToGlobalMapping(DA,ISLocalToGlobalMapping*);

EXTERN int   DAGetScatter(DA,VecScatter*,VecScatter*,VecScatter*);

EXTERN int   DAGetAO(DA,AO*);
EXTERN int   DASetCoordinates(DA,Vec); 
EXTERN int   DAGetCoordinates(DA,Vec *);
EXTERN int   DASetUniformCoordinates(DA,double,double,double,double,double,double);
EXTERN int   DASetFieldName(DA,int,const char[]);
EXTERN int   DAGetFieldName(DA,int,char **);

EXTERN int   DAVecGetArray(DA,Vec,void **);
EXTERN int   DAVecRestoreArray(DA,Vec,void **);

#include "petscmat.h"
EXTERN int   DAGetColoring(DA,ISColoring *,Mat *);
EXTERN int   DAGetInterpolation(DA,DA,Mat*,Vec*);

#include "petscpf.h"
EXTERN int DACreatePF(DA,PF*);

#endif
