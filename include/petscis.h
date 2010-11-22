/*
   An index set is a generalization of a subset of integers.  Index sets
   are used for defining scatters and gathers.
*/
#if !defined(__PETSCIS_H)
#define __PETSCIS_H
#include "petscsys.h"
PETSC_EXTERN_CXX_BEGIN

#define IS_FILE_CLASSID 1211218
extern PETSCVEC_DLLEXPORT PetscClassId IS_CLASSID;

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISInitializePackage(const char[]);

/*S
     IS - Abstract PETSc object that allows indexing.

   Level: beginner

  Concepts: indexing, stride

.seealso:  ISCreateGeneral(), ISCreateBlock(), ISCreateStride(), ISGetIndices(), ISDestroy()
S*/
typedef struct _p_IS* IS;

/*E
    ISType - String with the name of a PETSc vector or the creation function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:myveccreate()

   Level: beginner

.seealso: ISSetType(), IS
E*/
#define ISType char*
#define ISGENERAL      "general"
#define ISSTRIDE       "stride"
#define ISBLOCK        "block"

/* Dynamic creation and loading functions */
extern PetscFList ISList;
extern PetscBool  ISRegisterAllCalled;
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISSetType(IS, const ISType);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISGetType(IS, const ISType *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISRegister(const char[],const char[],const char[],PetscErrorCode (*)(IS));
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISRegisterAll(const char []);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISRegisterDestroy(void);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISCreate(MPI_Comm,IS*);

/*MC
  ISRegisterDynamic - Adds a new vector component implementation

  Synopsis:
  PetscErrorCode ISRegisterDynamic(const char *name, const char *path, const char *func_name, PetscErrorCode (*create_func)(IS))

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
. path        - The path (either absolute or relative) of the library containing this routine
. func_name   - The name of routine to create method context
- create_func - The creation routine itself

  Notes:
  ISRegisterDynamic() may be called multiple times to add several user-defined vectors

  If dynamic libraries are used, then the fourth input argument (routine_create) is ignored.

  Sample usage:
.vb
    ISRegisterDynamic("my_is","/home/username/my_lib/lib/libO/solaris/libmy.a", "MyIStorCreate", MyIStorCreate);
.ve

  Then, your vector type can be chosen with the procedural interface via
.vb
    ISCreate(MPI_Comm, IS *);
    ISSetType(IS,"my_vector_name");
.ve
   or at runtime via the option
.vb
    -vec_type my_vector_name
.ve

  Notes: $PETSC_ARCH occuring in pathname will be replaced with appropriate values.
         If your function is not being put into a shared library then use ISRegister() instead
        
  Level: advanced

.keywords: IS, register
.seealso: ISRegisterAll(), ISRegisterDestroy(), ISRegister()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define ISRegisterDynamic(a,b,c,d) ISRegister(a,b,c,0)
#else
#define ISRegisterDynamic(a,b,c,d) ISRegister(a,b,c,d)
#endif

/*
    Default index set data structures that PETSc provides.
*/
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISCreateGeneral(MPI_Comm,PetscInt,const PetscInt[],PetscCopyMode,IS *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISGeneralSetIndices(IS,PetscInt,const PetscInt[],PetscCopyMode);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISCreateBlock(MPI_Comm,PetscInt,PetscInt,const PetscInt[],PetscCopyMode,IS *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISBlockSetIndices(IS,PetscInt,PetscInt,const PetscInt[],PetscCopyMode);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISCreateStride(MPI_Comm,PetscInt,PetscInt,PetscInt,IS *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISStrideSetStride(IS,PetscInt,PetscInt,PetscInt);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISDestroy(IS);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISSetPermutation(IS);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISPermutation(IS,PetscBool *); 
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISSetIdentity(IS);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISIdentity(IS,PetscBool *);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISGetIndices(IS,const PetscInt *[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISRestoreIndices(IS,const PetscInt *[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISGetTotalIndices(IS,const PetscInt *[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISRestoreTotalIndices(IS,const PetscInt *[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISGetNonlocalIndices(IS,const PetscInt *[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISRestoreNonlocalIndices(IS,const PetscInt *[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISGetNonlocalIS(IS, IS *is);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISRestoreNonlocalIS(IS, IS *is);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISGetSize(IS,PetscInt *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISGetLocalSize(IS,PetscInt *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISInvertPermutation(IS,PetscInt,IS*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISView(IS,PetscViewer);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISEqual(IS,IS,PetscBool  *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISSort(IS);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISSorted(IS,PetscBool  *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISDifference(IS,IS,IS*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISSum(IS,IS,IS*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISExpand(IS,IS,IS*);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISBlockGetIndices(IS,const PetscInt *[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISBlockRestoreIndices(IS,const PetscInt *[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISBlockGetLocalSize(IS,PetscInt *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISBlockGetSize(IS,PetscInt *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISGetBlockSize(IS,PetscInt*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISSetBlockSize(IS,PetscInt);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISStrideGetInfo(IS,PetscInt *,PetscInt*);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISToGeneral(IS);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISDuplicate(IS,IS*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISCopy(IS,IS);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISAllGather(IS,IS*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISComplement(IS,PetscInt,PetscInt,IS*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT   ISOnComm(IS,MPI_Comm,PetscCopyMode,IS*);

/* --------------------------------------------------------------------------*/
extern PETSCVEC_DLLEXPORT PetscClassId IS_LTOGM_CLASSID;

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

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingCreate(MPI_Comm,PetscInt,const PetscInt[],PetscCopyMode,ISLocalToGlobalMapping*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingCreateIS(IS,ISLocalToGlobalMapping *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingView(ISLocalToGlobalMapping,PetscViewer);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingDestroy(ISLocalToGlobalMapping);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingApplyIS(ISLocalToGlobalMapping,IS,IS*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISGlobalToLocalMappingApply(ISLocalToGlobalMapping,ISGlobalToLocalMappingType,PetscInt,const PetscInt[],PetscInt*,PetscInt[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingGetSize(ISLocalToGlobalMapping,PetscInt*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingGetInfo(ISLocalToGlobalMapping,PetscInt*,PetscInt*[],PetscInt*[],PetscInt**[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingRestoreInfo(ISLocalToGlobalMapping,PetscInt*,PetscInt*[],PetscInt*[],PetscInt**[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingGetIndices(ISLocalToGlobalMapping,const PetscInt**);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingRestoreIndices(ISLocalToGlobalMapping,const PetscInt**);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingBlock(ISLocalToGlobalMapping,PetscInt,ISLocalToGlobalMapping*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT ISLocalToGlobalMappingUnBlock(ISLocalToGlobalMapping,PetscInt,ISLocalToGlobalMapping*);

PETSC_STATIC_INLINE PetscErrorCode ISLocalToGlobalMappingApply(ISLocalToGlobalMapping mapping,PetscInt N,const PetscInt in[],PetscInt out[])
{
  PetscInt       i,Nmax = mapping->n;
  const PetscInt *idx = mapping->indices;
  PetscFunctionBegin;
  for (i=0; i<N; i++) {
    if (in[i] < 0) {out[i] = in[i]; continue;}
    if (in[i] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i],Nmax,i);
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
$                         DMGetLocalVector() and DMGetGlobalVector() this could be called IS_COLORING_LOCAL

.seealso: DMGetColoring()
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
