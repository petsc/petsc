/* $Id: is.h,v 1.32 1996/11/07 15:12:51 bsmith Exp bsmith $ */

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
   ISLocalToGlobalMappings are mappings from an arbitrary
  local ordering from 0 to n-1 to a global PETSc ordering 
  used by a vector or matrix
*/
struct _ISLocalToGlobalMapping{
  int n;
  int *indices;
  int refcnt;
};
typedef struct _ISLocalToGlobalMapping* ISLocalToGlobalMapping;

extern int ISLocalToGlobalMappingCreate(int, int*, ISLocalToGlobalMapping*);
extern int ISLocalToGlobalMappingDestroy(ISLocalToGlobalMapping);
#define ISLocalToGlobalMappingApply(mp,N,in,out) \
  {\
   int _i,*_idx = mp->indices; \
   for ( _i=0; _i<N; _i++ ) { \
     out[_i] = _idx[in[_i]]; \
   }\
  }
#define ISLocalToGlobalMappingReference(mp) mp->refcnt++;
extern int ISLocalToGlobalMappingApplyIS(ISLocalToGlobalMapping,IS,IS*);

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




