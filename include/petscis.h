/* $Id: is.h,v 1.50 2000/01/11 21:04:04 bsmith Exp bsmith $ */

/*
   An index set is a generalization of a subset of integers.  Index sets
   are used for defining scatters and gathers.
*/
#if !defined(__IS_H)
#define __IS_H
#include "petsc.h"

#define IS_COOKIE PETSC_COOKIE+2

typedef struct _p_IS* IS;

/*
    Default index set data structures that PETSc provides.
*/
typedef enum {IS_GENERAL=0,IS_STRIDE=1,IS_BLOCK = 2} ISType;
extern int   ISCreateGeneral(MPI_Comm,int,const int[],IS *);
extern int   ISCreateBlock(MPI_Comm,int,int,const int[],IS *);
extern int   ISCreateStride(MPI_Comm,int,int,int,IS *);

extern int   ISDestroy(IS);

extern int   ISSetPermutation(IS);
extern int   ISPermutation(IS,PetscTruth*); 
extern int   ISSetIdentity(IS);
extern int   ISIdentity(IS,PetscTruth*);

extern int   ISGetIndices(IS,int *[]);
extern int   ISRestoreIndices(IS,int *[]);
extern int   ISGetSize(IS,int *);
extern int   ISInvertPermutation(IS,int,IS*);
extern int   ISView(IS,Viewer);
extern int   ISEqual(IS,IS,PetscTruth *);
extern int   ISSort(IS);
extern int   ISSorted(IS,PetscTruth *);
extern int   ISDifference(IS,IS,IS*);
extern int   ISSum(IS,IS,IS*);

extern int   ISBlock(IS,PetscTruth*);
extern int   ISBlockGetIndices(IS,int *[]);
extern int   ISBlockRestoreIndices(IS,int *[]);
extern int   ISBlockGetSize(IS,int *);
extern int   ISBlockGetBlockSize(IS,int *);

extern int   ISStride(IS,PetscTruth*);
extern int   ISStrideGetInfo(IS,int *,int*);

extern int   ISStrideToGeneral(IS);

extern int   ISDuplicate(IS,IS*);
extern int   ISAllGather(IS,IS*);

/* --------------------------------------------------------------------------*/

/*
   ISLocalToGlobalMappings are mappings from an arbitrary
  local ordering from 0 to n-1 to a global PETSc ordering 
  used by a vector or matrix.

   Note: mapping from Local to Global is scalable; but Global
  to Local may not be if the range of global values represented locally
  is very large.
*/
#define IS_LTOGM_COOKIE PETSC_COOKIE+12
typedef struct _p_ISLocalToGlobalMapping* ISLocalToGlobalMapping;
typedef enum {IS_GTOLM_MASK,IS_GTOLM_DROP} ISGlobalToLocalMappingType;

extern int ISLocalToGlobalMappingCreate(MPI_Comm,int,const int[],ISLocalToGlobalMapping*);
extern int ISLocalToGlobalMappingCreateIS(IS,ISLocalToGlobalMapping *);
extern int ISLocalToGlobalMappingView(ISLocalToGlobalMapping,Viewer);
extern int ISLocalToGlobalMappingDestroy(ISLocalToGlobalMapping);
extern int ISLocalToGlobalMappingApply(ISLocalToGlobalMapping,int,const int[],int[]);
extern int ISLocalToGlobalMappingApplyIS(ISLocalToGlobalMapping,IS,IS*);
extern int ISGlobalToLocalMappingApply(ISLocalToGlobalMapping,ISGlobalToLocalMappingType,
                                       int,const int[],int*,int[]);

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

extern int ISColoringCreate(MPI_Comm,int,const int[],ISColoring*);
extern int ISColoringDestroy(ISColoring);
extern int ISColoringView(ISColoring,Viewer);
extern int ISColoringGetIS(ISColoring,int*,IS*[]);

/* --------------------------------------------------------------------------*/

extern int ISPartitioningToNumbering(IS,IS*);
extern int ISPartitioningCount(IS,int[]);

#endif




