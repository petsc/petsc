/* $Id: is.h,v 1.31 1996/09/28 20:21:29 curfman Exp bsmith $ */

/*
   An index set is a generalization of a subset of integers.  Index sets
   are used for defining scatters and gathers.
*/
#if !defined(__IS_PACKAGE)
#define __IS_PACKAGE
#include "petsc.h"

#define IS_COOKIE PETSC_COOKIE+2

typedef struct _IS* IS;

/*
    Default index set data structures that PETSc provides.
*/
typedef enum {IS_GENERAL=0, IS_STRIDE=1, IS_BLOCK = 2} ISType;
extern int   ISCreateGeneral(MPI_Comm,int,int *,IS *);
extern int   ISCreateBlock(MPI_Comm,int,int,int *,IS *);
extern int   ISCreateStride(MPI_Comm,int,int,int,IS *);

extern int   ISDestroy(IS);

extern int   ISSetPermutation(IS);
extern int   ISPermutation(IS,PetscTruth*); 
extern int   ISSetIdentity(IS);
extern int   ISIdentity(IS,PetscTruth*);

extern int   ISGetIndices(IS,int **);
extern int   ISRestoreIndices(IS,int **);
extern int   ISGetSize(IS,int *);
extern int   ISInvertPermutation(IS,IS*);
extern int   ISView(IS,Viewer);
extern int   ISEqual(IS, IS, PetscTruth *);
extern int   ISSort(IS);
extern int   ISSorted(IS, PetscTruth *);

extern int   ISBlock(IS,PetscTruth*);
extern int   ISBlockGetIndices(IS,int **);
extern int   ISBlockRestoreIndices(IS,int **);
extern int   ISBlockGetSize(IS,int *);
extern int   ISBlockGetBlockSize(IS,int *);

extern int   ISStride(IS,PetscTruth*);
extern int   ISStrideGetInfo(IS,int *,int*);

/*
     ISColorings are sets of IS's that define a coloring
   of the underlying indices
*/
struct _ISColoring {
  int      n;
  IS       *is;
  MPI_Comm comm;
};
typedef struct _ISColoring* ISColoring;

extern int ISColoringDestroy(ISColoring);
extern int ISColoringView(ISColoring,Viewer);
extern int ISColoringCreate(MPI_Comm,int,int*,ISColoring*);

#endif




