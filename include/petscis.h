/* $Id: petscis.h,v 1.55 2000/05/25 22:37:09 bsmith Exp bsmith $ */

/*
   An index set is a generalization of a subset of integers.  Index sets
   are used for defining scatters and gathers.
*/
#if !defined(__PETSCIS_H)
#define __PETSCIS_H
#include "petsc.h"

#define IS_COOKIE PETSC_COOKIE+2

typedef struct _p_IS* IS;

/*
    Default index set data structures that PETSc provides.
*/
typedef enum {IS_GENERAL=0,IS_STRIDE=1,IS_BLOCK = 2} ISType;
EXTERN int   ISCreateGeneral(MPI_Comm,int,const int[],IS *);
EXTERN int   ISCreateBlock(MPI_Comm,int,int,const int[],IS *);
EXTERN int   ISCreateStride(MPI_Comm,int,int,int,IS *);

EXTERN int   ISDestroy(IS);

EXTERN int   ISSetPermutation(IS);
EXTERN int   ISPermutation(IS,PetscTruth*); 
EXTERN int   ISSetIdentity(IS);
EXTERN int   ISIdentity(IS,PetscTruth*);

EXTERN int   ISGetIndices(IS,int *[]);
EXTERN int   ISRestoreIndices(IS,int *[]);
EXTERN int   ISGetSize(IS,int *);
EXTERN int   ISGetLocalSize(IS,int *);
EXTERN int   ISInvertPermutation(IS,int,IS*);
EXTERN int   ISView(IS,Viewer);
EXTERN int   ISEqual(IS,IS,PetscTruth *);
EXTERN int   ISSort(IS);
EXTERN int   ISSorted(IS,PetscTruth *);
EXTERN int   ISDifference(IS,IS,IS*);
EXTERN int   ISSum(IS,IS,IS*);

EXTERN int   ISBlock(IS,PetscTruth*);
EXTERN int   ISBlockGetIndices(IS,int *[]);
EXTERN int   ISBlockRestoreIndices(IS,int *[]);
EXTERN int   ISBlockGetSize(IS,int *);
EXTERN int   ISBlockGetBlockSize(IS,int *);

EXTERN int   ISStride(IS,PetscTruth*);
EXTERN int   ISStrideGetInfo(IS,int *,int*);

EXTERN int   ISStrideToGeneral(IS);

EXTERN int   ISDuplicate(IS,IS*);
EXTERN int   ISAllGather(IS,IS*);

/* --------------------------------------------------------------------------*/

/*
   ISLocalToGlobalMappings are mappings from an arbitrary
  local ordering from 0 to n-1 to a global PETSc ordering 
  used by a vector or matrix.

   Note: mapping from Local to Global is scalable; but Global
  to Local may not be if the range of global values represented locally
  is very large.

   Note: the ISLocalToGlobalMapping is actually a private object; it is included
  here for the MACRO ISLocalToGlobalMappingApply() to allow it to be inlined since
  it is used so often.
*/
#define IS_LTOGM_COOKIE PETSC_COOKIE+12

struct _p_ISLocalToGlobalMapping{
  PETSCHEADER(int)
  int n;                  /* number of local indices */
  int *indices;           /* global index of each local index */
  int globalstart;        /* first global referenced in indices */
  int globalend;          /* last + 1 global referenced in indices */
  int *globals;           /* local index for each global index between start and end */
};
typedef struct _p_ISLocalToGlobalMapping* ISLocalToGlobalMapping;
typedef enum {IS_GTOLM_MASK,IS_GTOLM_DROP} ISGlobalToLocalMappingType;

EXTERN int ISLocalToGlobalMappingCreate(MPI_Comm,int,const int[],ISLocalToGlobalMapping*);
EXTERN int ISLocalToGlobalMappingCreateIS(IS,ISLocalToGlobalMapping *);
EXTERN int ISLocalToGlobalMappingView(ISLocalToGlobalMapping,Viewer);
EXTERN int ISLocalToGlobalMappingDestroy(ISLocalToGlobalMapping);
EXTERN int ISLocalToGlobalMappingApplyIS(ISLocalToGlobalMapping,IS,IS*);
EXTERN int ISGlobalToLocalMappingApply(ISLocalToGlobalMapping,ISGlobalToLocalMappingType,int,const int[],int*,int[]);
EXTERN int ISLocalToGlobalMappingGetSize(ISLocalToGlobalMapping,int*);
EXTERN int ISLocalToGlobalMappingGetInfo(ISLocalToGlobalMapping,int*,int**,int**,int***);
EXTERN int ISLocalToGlobalMappingRestoreInfo(ISLocalToGlobalMapping,int*,int**,int**,int***);

#define ISLocalToGlobalMappingApply(mapping,N,in,out) 0;\
{\
  int _i,*_idx = (mapping)->indices,_Nmax = (mapping)->n;\
  for (_i=0; _i<N; _i++) {\
    if ((in)[_i] < 0) {(out)[_i] = (in)[_i]; continue;}\
    if ((in)[_i] >= _Nmax) SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,1,"Local index %d too large %d (max) at %d",(in)[_i],_Nmax,_i);\
    (out)[_i] = _idx[(in)[_i]];\
  }\
}

/* --------------------------------------------------------------------------*/

/*
     ISColorings are sets of IS's that define a coloring
   of the underlying indices
*/
struct _p_ISColoring {
  int      n;
  IS       *is;
  MPI_Comm comm;
};
typedef struct _p_ISColoring* ISColoring;

EXTERN int ISColoringCreate(MPI_Comm,int,const int[],ISColoring*);
EXTERN int ISColoringDestroy(ISColoring);
EXTERN int ISColoringView(ISColoring,Viewer);
EXTERN int ISColoringGetIS(ISColoring,int*,IS*[]);
EXTERN int ISColoringRestoreIS(ISColoring,IS*[]);

/* --------------------------------------------------------------------------*/

EXTERN int ISPartitioningToNumbering(IS,IS*);
EXTERN int ISPartitioningCount(IS,int[]);

#endif




