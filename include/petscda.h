/* $Id: snes.h,v 1.17 1995/06/02 21:05:19 bsmith Exp $ */

/*
      Regular array object, for easy parallism of simple grid 
   problems on regular distributed arrays.
*/
#if !defined(__DA_PACKAGE)
#define __DA_PACKAGE
#include "petsc.h"
#include "vec.h"

#define DA_COOKIE PETSC_COOKIE+14

typedef struct _DA* DA;

extern int   DACreate2d(MPI_Comm,int,int,int,int,int,int,DA *);
extern int   DADestroy(DA);
extern int   DAView(DA,Viewer);
extern int   DAGlobalToLocalBegin(DA,Vec, InsertMode,Vec);
extern int   DAGlobalToLocalEnd(DA,Vec, InsertMode,Vec);
extern int   DALocalToGlobal(DA,Vec, InsertMode,Vec);

extern int   DAGetDistributedVector(DA,Vec*);
extern int   DAGetLocalVector(DA,Vec*);
extern int   DAGetCorners(DA,int*,int*,int*,int*,int*,int*);
extern int   DAGetGhostCorners(DA,int*,int*,int*,int*,int*,int*);

extern int   DAGetGlobalIndices(DA,int*,int**);
extern int   DAGetScatterCtx(DA,VecScatterCtx*,VecScatterCtx*);


#endif
