/* $Id: is.h,v 1.30 1996/08/15 12:51:50 bsmith Exp curfman $ */

/*
   An index set is a generalization of a subset of integers.  Index sets
   are used for defining scatters and gathers.
*/
#if !defined(__IS_PACKAGE)
#define __IS_PACKAGE
#include "petsc.h"

typedef enum {IS_GENERAL=0, IS_STRIDE=1, IS_BLOCK = 2} ISType;

#define IS_COOKIE PETSC_COOKIE+2

typedef struct _IS* IS;

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

#endif


