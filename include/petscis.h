/* $Id: petscis.h,v 1.63 2001/06/21 21:15:49 bsmith Exp $ */

/*
   An index set is a generalization of a subset of integers.  Index sets
   are used for defining scatters and gathers.
*/
#if !defined(__PETSCIS_H)
#define __PETSCIS_H
#include "petsc.h"
PETSC_EXTERN_CXX_BEGIN

extern int IS_COOKIE;

/*S
     IS - Abstract PETSc object that indexing.

   Level: beginner

  Concepts: indexing, stride

.seealso:  ISCreateGeneral(), ISCreateBlock(), ISCreateStride(), ISGetIndices(), ISDestroy()
S*/
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
EXTERN int   ISView(IS,PetscViewer);
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
EXTERN int   ISAllGatherIndices(MPI_Comm,int,const int[],int*,int*[]);

/* --------------------------------------------------------------------------*/
extern int IS_LTOGM_COOKIE;

/*S
   ISLocalToGlobalMapping - mappings from an arbitrary
      local ordering from 0 to n-1 to a global PETSc ordering 
      used by a vector or matrix.

   Level: intermediate

   Note: mapping from Local to Global is scalable; but Global
  to Local may not be if the range of global values represented locally
  is very large.

   Note: the ISLocalToGlobalMapping is actually a private object; it is included
  here for the MACRO ISLocalToGlobalMappingApply() to allow it to be inlined since
  it is used so often.

.seealso:  ISLocalToGlobalMappingCreate()
S*/
struct _p_ISLocalToGlobalMapping{
  PETSCHEADER(int)
  int n;                  /* number of local indices */
  int *indices;           /* global index of each local index */
  int globalstart;        /* first global referenced in indices */
  int globalend;          /* last + 1 global referenced in indices */
  int *globals;           /* local index for each global index between start and end */
};
typedef struct _p_ISLocalToGlobalMapping* ISLocalToGlobalMapping;

/*E
    ISGlobalToLocalMappingType - Indicates if missing global indices are 

   IS_GTOLM_MASK - missing global indices are replaced with -1
   IS_GTOLM_DROP - missing global indices are dropped

   Level: beginner

.seealso: ISGlobalToLocalMappingApply()

E*/
typedef enum {IS_GTOLM_MASK,IS_GTOLM_DROP} ISGlobalToLocalMappingType;

EXTERN int ISLocalToGlobalMappingCreate(MPI_Comm,int,const int[],ISLocalToGlobalMapping*);
EXTERN int ISLocalToGlobalMappingCreateNC(MPI_Comm,int,const int[],ISLocalToGlobalMapping*);
EXTERN int ISLocalToGlobalMappingCreateIS(IS,ISLocalToGlobalMapping *);
EXTERN int ISLocalToGlobalMappingView(ISLocalToGlobalMapping,PetscViewer);
EXTERN int ISLocalToGlobalMappingDestroy(ISLocalToGlobalMapping);
EXTERN int ISLocalToGlobalMappingApplyIS(ISLocalToGlobalMapping,IS,IS*);
EXTERN int ISGlobalToLocalMappingApply(ISLocalToGlobalMapping,ISGlobalToLocalMappingType,int,const int[],int*,int[]);
EXTERN int ISLocalToGlobalMappingGetSize(ISLocalToGlobalMapping,int*);
EXTERN int ISLocalToGlobalMappingGetInfo(ISLocalToGlobalMapping,int*,int*[],int*[],int**[]);
EXTERN int ISLocalToGlobalMappingRestoreInfo(ISLocalToGlobalMapping,int*,int*[],int*[],int**[]);
EXTERN int ISLocalToGlobalMappingBlock(ISLocalToGlobalMapping,int,ISLocalToGlobalMapping*);

#define ISLocalToGlobalMappingApply(mapping,N,in,out) 0;\
{\
  int _i,*_idx = (mapping)->indices,_Nmax = (mapping)->n;\
  for (_i=0; _i<N; _i++) {\
    if ((in)[_i] < 0) {(out)[_i] = (in)[_i]; continue;}\
    if ((in)[_i] >= _Nmax) SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,"Local index %d too large %d (max) at %d",(in)[_i],_Nmax,_i);\
    (out)[_i] = _idx[(in)[_i]];\
  }\
}

/* --------------------------------------------------------------------------*/
/*E
    ISColoringType - determines if the coloring is for the entire parallel grid/graph/matrix
                     or for just the local ghosted portion

    Level: beginner

$   IS_COLORING_LOCAL - does not include the colors for ghost points
$   IS_COLORING_GHOSTED - includes colors for ghost points

.seealso: DAGetColoring()
E*/
typedef enum {IS_COLORING_LOCAL,IS_COLORING_GHOSTED} ISColoringType;

#define MPIU_COLORING_VALUE MPI_CHAR
#define IS_COLORING_MAX     255
typedef unsigned char ISColoringValue;
EXTERN int ISAllGatherColors(MPI_Comm,int,ISColoringValue*,int*,ISColoringValue*[]);

/*S
     ISColoring - sets of IS's that define a coloring
              of the underlying indices

   Level: intermediate

    Notes:
        One should not access the *is records below directly because they may not yet 
    have been created. One should use ISColoringGetIS() to make sure they are 
    created when needed.

.seealso:  ISColoringCreate(), ISColoringGetIS(), ISColoringView(), ISColoringGetIS()
S*/
struct _p_ISColoring {
  int             refct;
  int             n;                /* number of colors */
  IS              *is;              /* for each color indicates columns */
  MPI_Comm        comm;
  ISColoringValue *colors;          /* for each column indicates color */
  int             N;                /* number of columns */
  ISColoringType  ctype;
};
typedef struct _p_ISColoring* ISColoring;

EXTERN int ISColoringCreate(MPI_Comm,int,const ISColoringValue[],ISColoring*);
EXTERN int ISColoringDestroy(ISColoring);
EXTERN int ISColoringView(ISColoring,PetscViewer);
EXTERN int ISColoringGetIS(ISColoring,int*,IS*[]);
EXTERN int ISColoringRestoreIS(ISColoring,IS*[]);
#define ISColoringReference(coloring) ((coloring)->refct++,0)
#define ISColoringSetType(coloring,type) ((coloring)->ctype = type,0)

/* --------------------------------------------------------------------------*/

EXTERN int ISPartitioningToNumbering(IS,IS*);
EXTERN int ISPartitioningCount(IS,int[]);

PETSC_EXTERN_CXX_END
#endif
