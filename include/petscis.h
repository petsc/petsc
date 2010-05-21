/*
   An index set is a generalization of a subset of integers.  Index sets
   are used for defining scatters and gathers.
*/
#if !defined(__PETSCIS_H)
#define __PETSCIS_H
#include "petscsys.h"
PETSC_EXTERN_CXX_BEGIN

extern PETSCVEC_DLLEXPORT PetscCookie IS_COOKIE;

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISInitializePackage(const char[]);

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
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISCreateGeneral(MPI_Comm,PetscInt,const PetscInt[],IS *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISCreateGeneralNC(MPI_Comm,PetscInt,const PetscInt[],IS *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISCreateGeneralWithArray(MPI_Comm,PetscInt,PetscInt[],IS *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISCreateBlock(MPI_Comm,PetscInt,PetscInt,const PetscInt[],IS *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISCreateStride(MPI_Comm,PetscInt,PetscInt,PetscInt,IS *);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISDestroy(IS);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISSetPermutation(IS);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISPermutation(IS,PetscTruth*); 
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISSetIdentity(IS);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISIdentity(IS,PetscTruth*);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISGetIndices(IS,const PetscInt *[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISRestoreIndices(IS,const PetscInt *[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISGetSize(IS,PetscInt *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISGetLocalSize(IS,PetscInt *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISInvertPermutation(IS,PetscInt,IS*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISView(IS,PetscViewer);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISEqual(IS,IS,PetscTruth *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISSort(IS);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISSorted(IS,PetscTruth *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISDifference(IS,IS,IS*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISSum(IS,IS,IS*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISExpand(IS,IS,IS*);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISBlock(IS,PetscTruth*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISBlockGetIndices(IS,const PetscInt *[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISBlockRestoreIndices(IS,const PetscInt *[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISBlockGetLocalSize(IS,PetscInt *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISBlockGetSize(IS,PetscInt *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISBlockGetBlockSize(IS,PetscInt *);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISStride(IS,PetscTruth*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISStrideGetInfo(IS,PetscInt *,PetscInt*);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISStrideToGeneral(IS);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISDuplicate(IS,IS*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISCopy(IS,IS);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISAllGather(IS,IS*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISComplement(IS,PetscInt,PetscInt,IS*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISAllGatherIndices(MPI_Comm,PetscInt,const PetscInt[],PetscInt*,PetscInt*[]);

/* --------------------------------------------------------------------------*/
extern PETSCVEC_DLLEXPORT PetscCookie IS_LTOGM_COOKIE;

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
  PETSCHEADER(int);
  PetscInt n;                  /* number of local indices */
  PetscInt *indices;           /* global index of each local index */
  PetscInt globalstart;        /* first global referenced in indices */
  PetscInt globalend;          /* last + 1 global referenced in indices */
  PetscInt *globals;           /* local index for each global index between start and end */
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

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingCreate(MPI_Comm,PetscInt,const PetscInt[],ISLocalToGlobalMapping*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingCreateNC(MPI_Comm,PetscInt,const PetscInt[],ISLocalToGlobalMapping*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingCreateIS(IS,ISLocalToGlobalMapping *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingView(ISLocalToGlobalMapping,PetscViewer);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingDestroy(ISLocalToGlobalMapping);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingApplyIS(ISLocalToGlobalMapping,IS,IS*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISGlobalToLocalMappingApply(ISLocalToGlobalMapping,ISGlobalToLocalMappingType,PetscInt,const PetscInt[],PetscInt*,PetscInt[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingGetSize(ISLocalToGlobalMapping,PetscInt*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingGetInfo(ISLocalToGlobalMapping,PetscInt*,PetscInt*[],PetscInt*[],PetscInt**[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingRestoreInfo(ISLocalToGlobalMapping,PetscInt*,PetscInt*[],PetscInt*[],PetscInt**[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingBlock(ISLocalToGlobalMapping,PetscInt,ISLocalToGlobalMapping*);

PETSC_STATIC_INLINE PetscErrorCode ISLocalToGlobalMappingApply(ISLocalToGlobalMapping mapping,PetscInt N,const PetscInt in[],PetscInt out[])
{
  PetscInt i,*idx = mapping->indices,Nmax = mapping->n;
  PetscFunctionBegin;
  for (i=0; i<N; i++) {
    if (in[i] < 0) {out[i] = in[i]; continue;}
    if (in[i] >= Nmax) SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i],Nmax,i);
    out[i] = idx[in[i]];
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------*/
/*E
    ISColoringType - determines if the coloring is for the entire parallel grid/graph/matrix
                     or for just the local ghosted portion

    Level: beginner

$   IS_COLORING_GLOBAL - does not include the colors for ghost points, this is used when the function 
$                        is called synchronously in parallel. This requires generating a "parallel coloring".
$   IS_COLORING_GHOSTED - includes colors for ghost points, this is used when the function can be called
$                         seperately on individual processes with the ghost points already filled in. Does not
$                         require a "parallel coloring", rather each process colors its local + ghost part.
$                         Using this can result in much less parallel communication. In the paradigm of 
$                         DAGetLocalVector() and DAGetGlobalVector() this could be called IS_COLORING_LOCAL

.seealso: DAGetColoring()
E*/
typedef enum {IS_COLORING_GLOBAL,IS_COLORING_GHOSTED} ISColoringType;
extern const char *ISColoringTypes[];
typedef unsigned PETSC_IS_COLOR_VALUE_TYPE ISColoringValue;
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISAllGatherColors(MPI_Comm,PetscInt,ISColoringValue*,PetscInt*,ISColoringValue*[]);

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
struct _n_ISColoring {
  PetscInt        refct;
  PetscInt        n;                /* number of colors */
  IS              *is;              /* for each color indicates columns */
  MPI_Comm        comm;
  ISColoringValue *colors;          /* for each column indicates color */
  PetscInt        N;                /* number of columns */
  ISColoringType  ctype;
};
typedef struct _n_ISColoring* ISColoring;

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISColoringCreate(MPI_Comm,PetscInt,PetscInt,const ISColoringValue[],ISColoring*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISColoringDestroy(ISColoring);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISColoringView(ISColoring,PetscViewer);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISColoringGetIS(ISColoring,PetscInt*,IS*[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISColoringRestoreIS(ISColoring,IS*[]);
#define ISColoringReference(coloring) ((coloring)->refct++,0)
#define ISColoringSetType(coloring,type) ((coloring)->ctype = type,0)

/* --------------------------------------------------------------------------*/

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISPartitioningToNumbering(IS,IS*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISPartitioningCount(IS,PetscInt,PetscInt[]);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISCompressIndicesGeneral(PetscInt,PetscInt,PetscInt,const IS[],IS[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISCompressIndicesSorted(PetscInt,PetscInt,PetscInt,const IS[],IS[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISExpandIndicesGeneral(PetscInt,PetscInt,PetscInt,const IS[],IS[]);

PETSC_EXTERN_CXX_END
#endif
