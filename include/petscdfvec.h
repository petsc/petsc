
/* $Id: dfvec.h,v 1.17 1996/04/17 23:01:59 curfman Exp curfman $ */

/* This file declares some utility routines for manipulating discrete functions,
   or vectors that are associated with grids, possibly with multiple degrees of
   freedom per node).  DF = Discrete Function */

#ifndef __DISCRETE_FUNCTION
#define __DISCRETE_FUNCTION

#include "vec.h"

typedef enum {DF_SEQGEN,DF_MPIGEN,DF_SEQREG,DF_MPIREG} DFType;

typedef enum {ORDER_1,ORDER_2} DFComponentOrdering;
  /* ORDER_1 - ordering by interlacing components at each grid point
     ORDER_2 - ordering by segregating unknowns according to type
          (1 component for whole grid, then the next component, etc.) */

#define DF_COOKIE PETSC_COOKIE+20

typedef struct _DF* DF;
#define DFVec Vec

/* These routines manipulate the DFVec objects (vectors that are discrete functions). */
extern int DFVecShellAssociate(DF,Vec);
extern int DFVecGetDFShell(Vec,DF*);
extern int DFVecGetComponentVectors(DFVec,int*,DFVec**);
extern int DFVecAssembleFullVector(Vec*,DFVec);
extern int DFVecView(DFVec,Viewer);
extern int DFVecDrawTensorContoursX(DFVec,int,int);
extern int DFVecDrawTensorSurfaceContoursVRML(DFVec);
extern int DFVecRefineVector(DFVec,DFVec*);
extern int DFVecCopy(DFVec,DFVec);

/* These routines manipulate the DF shell context. The interface for creating the
   shells and using these routines will change in the near future */
extern int DFShellCreate(MPI_Comm,DFType,int,int,DFComponentOrdering,char**,int,int,int,DF*);
extern int DFShellGetInfo(DF,DFType*,int*,int*,DFComponentOrdering*,int*,int*,int*);
extern int DFShellDuplicate(DF,DF*);
extern int DFShellDestroy(DF);
extern int DFShellGetLocalDFShell(DF,DF*);
extern int DFShellSetCoordinates(DF,int,int,int,double*,double*,double*);
extern int DFShellGetCoordinates(DF,int*,int*,int*,double**,double**,double**);

#endif
