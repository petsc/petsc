/*
   An index set is a generalization of a subset of integers.  Index sets
   are used for defining scatters and gathers.
*/
#if !defined(__PETSCIS_H)
#define __PETSCIS_H
#include "petscsys.h"
#include "petscsf.h"
PETSC_EXTERN_CXX_BEGIN

#define IS_FILE_CLASSID 1211218
extern  PetscClassId IS_CLASSID;

extern PetscErrorCode  ISInitializePackage(const char[]);

/*S
     IS - Abstract PETSc object that allows indexing.

   Level: beginner

  Concepts: indexing, stride

.seealso:  ISCreateGeneral(), ISCreateBlock(), ISCreateStride(), ISGetIndices(), ISDestroy()
S*/
typedef struct _p_IS* IS;

/*J
    ISType - String with the name of a PETSc vector or the creation function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:myveccreate()

   Level: beginner

.seealso: ISSetType(), IS
J*/
#define ISType char*
#define ISGENERAL      "general"
#define ISSTRIDE       "stride"
#define ISBLOCK        "block"

/* Dynamic creation and loading functions */
extern PetscFList ISList;
extern PetscBool  ISRegisterAllCalled;
extern PetscErrorCode  ISSetType(IS, const ISType);
extern PetscErrorCode  ISGetType(IS, const ISType *);
extern PetscErrorCode  ISRegister(const char[],const char[],const char[],PetscErrorCode (*)(IS));
extern PetscErrorCode  ISRegisterAll(const char []);
extern PetscErrorCode  ISRegisterDestroy(void);
extern PetscErrorCode  ISCreate(MPI_Comm,IS*);

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
    ISRegisterDynamic("my_is_name","/home/username/my_lib/lib/libO/solaris/libmy.a", "MyISCreate", MyISCreate);
.ve

  Then, your vector type can be chosen with the procedural interface via
.vb
    ISCreate(MPI_Comm, IS *);
    ISSetType(IS,"my_is_name");
.ve
   or at runtime via the option
.vb
    -is_type my_is_name
.ve

  Notes: $PETSC_ARCH occuring in pathname will be replaced with appropriate values.
         If your function is not being put into a shared library then use ISRegister() instead

  This is no ISSetFromOptions() and the current implementations do not have a way to dynamically determine type, so
  dynamic registration of custom IS types will be of limited use to users.

  Level: developer

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
extern PetscErrorCode    ISCreateGeneral(MPI_Comm,PetscInt,const PetscInt[],PetscCopyMode,IS *);
extern PetscErrorCode    ISGeneralSetIndices(IS,PetscInt,const PetscInt[],PetscCopyMode);
extern PetscErrorCode    ISCreateBlock(MPI_Comm,PetscInt,PetscInt,const PetscInt[],PetscCopyMode,IS *);
extern PetscErrorCode    ISBlockSetIndices(IS,PetscInt,PetscInt,const PetscInt[],PetscCopyMode);
extern PetscErrorCode    ISCreateStride(MPI_Comm,PetscInt,PetscInt,PetscInt,IS *);
extern PetscErrorCode    ISStrideSetStride(IS,PetscInt,PetscInt,PetscInt);

extern PetscErrorCode    ISDestroy(IS*);
extern PetscErrorCode    ISSetPermutation(IS);
extern PetscErrorCode    ISPermutation(IS,PetscBool *); 
extern PetscErrorCode    ISSetIdentity(IS);
extern PetscErrorCode    ISIdentity(IS,PetscBool *);
extern PetscErrorCode    ISContiguousLocal(IS,PetscInt,PetscInt,PetscInt*,PetscBool*);

extern PetscErrorCode    ISGetIndices(IS,const PetscInt *[]);
extern PetscErrorCode    ISRestoreIndices(IS,const PetscInt *[]);
extern PetscErrorCode    ISGetTotalIndices(IS,const PetscInt *[]);
extern PetscErrorCode    ISRestoreTotalIndices(IS,const PetscInt *[]);
extern PetscErrorCode    ISGetNonlocalIndices(IS,const PetscInt *[]);
extern PetscErrorCode    ISRestoreNonlocalIndices(IS,const PetscInt *[]);
extern PetscErrorCode    ISGetNonlocalIS(IS, IS *is);
extern PetscErrorCode    ISRestoreNonlocalIS(IS, IS *is);
extern PetscErrorCode    ISGetSize(IS,PetscInt *);
extern PetscErrorCode    ISGetLocalSize(IS,PetscInt *);
extern PetscErrorCode    ISInvertPermutation(IS,PetscInt,IS*);
extern PetscErrorCode    ISView(IS,PetscViewer);
extern PetscErrorCode    ISEqual(IS,IS,PetscBool  *);
extern PetscErrorCode    ISSort(IS);
extern PetscErrorCode    ISSorted(IS,PetscBool  *);
extern PetscErrorCode    ISDifference(IS,IS,IS*);
extern PetscErrorCode    ISSum(IS,IS,IS*);
extern PetscErrorCode    ISExpand(IS,IS,IS*);

extern PetscErrorCode    ISBlockGetIndices(IS,const PetscInt *[]);
extern PetscErrorCode    ISBlockRestoreIndices(IS,const PetscInt *[]);
extern PetscErrorCode    ISBlockGetLocalSize(IS,PetscInt *);
extern PetscErrorCode    ISBlockGetSize(IS,PetscInt *);
extern PetscErrorCode    ISGetBlockSize(IS,PetscInt*);
extern PetscErrorCode    ISSetBlockSize(IS,PetscInt);

extern PetscErrorCode    ISStrideGetInfo(IS,PetscInt *,PetscInt*);

extern PetscErrorCode    ISToGeneral(IS);

extern PetscErrorCode    ISDuplicate(IS,IS*);
extern PetscErrorCode    ISCopy(IS,IS);
extern PetscErrorCode    ISAllGather(IS,IS*);
extern PetscErrorCode    ISComplement(IS,PetscInt,PetscInt,IS*);
extern PetscErrorCode    ISConcatenate(MPI_Comm,PetscInt,const IS[],IS*);
extern PetscErrorCode    ISListToColoring(MPI_Comm,PetscInt, IS[],IS*,IS*);
extern PetscErrorCode    ISColoringToList(IS, IS, PetscInt*, IS *[]);
extern PetscErrorCode    ISOnComm(IS,MPI_Comm,PetscCopyMode,IS*);

/* --------------------------------------------------------------------------*/
extern  PetscClassId IS_LTOGM_CLASSID;

/*S
   ISLocalToGlobalMapping - mappings from an arbitrary
      local ordering from 0 to n-1 to a global PETSc ordering 
      used by a vector or matrix.

   Level: intermediate

   Note: mapping from Local to Global is scalable; but Global
  to Local may not be if the range of global values represented locally
  is very large.

   Note: the ISLocalToGlobalMapping is actually a private object; it is included
  here for the inline function ISLocalToGlobalMappingApply() to allow it to be inlined since
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

extern PetscErrorCode  ISLocalToGlobalMappingCreate(MPI_Comm,PetscInt,const PetscInt[],PetscCopyMode,ISLocalToGlobalMapping*);
extern PetscErrorCode  ISLocalToGlobalMappingCreateIS(IS,ISLocalToGlobalMapping *);
extern PetscErrorCode  ISLocalToGlobalMappingCreateSF(PetscSF,PetscInt,ISLocalToGlobalMapping*);
extern PetscErrorCode  ISLocalToGlobalMappingView(ISLocalToGlobalMapping,PetscViewer);
extern PetscErrorCode  ISLocalToGlobalMappingDestroy(ISLocalToGlobalMapping*);
extern PetscErrorCode  ISLocalToGlobalMappingApplyIS(ISLocalToGlobalMapping,IS,IS*);
extern PetscErrorCode  ISGlobalToLocalMappingApply(ISLocalToGlobalMapping,ISGlobalToLocalMappingType,PetscInt,const PetscInt[],PetscInt*,PetscInt[]);
extern PetscErrorCode  ISLocalToGlobalMappingGetSize(ISLocalToGlobalMapping,PetscInt*);
extern PetscErrorCode  ISLocalToGlobalMappingGetInfo(ISLocalToGlobalMapping,PetscInt*,PetscInt*[],PetscInt*[],PetscInt**[]);
extern PetscErrorCode  ISLocalToGlobalMappingRestoreInfo(ISLocalToGlobalMapping,PetscInt*,PetscInt*[],PetscInt*[],PetscInt**[]);
extern PetscErrorCode  ISLocalToGlobalMappingGetIndices(ISLocalToGlobalMapping,const PetscInt**);
extern PetscErrorCode  ISLocalToGlobalMappingRestoreIndices(ISLocalToGlobalMapping,const PetscInt**);
extern PetscErrorCode  ISLocalToGlobalMappingBlock(ISLocalToGlobalMapping,PetscInt,ISLocalToGlobalMapping*);
extern PetscErrorCode  ISLocalToGlobalMappingUnBlock(ISLocalToGlobalMapping,PetscInt,ISLocalToGlobalMapping*);
extern PetscErrorCode  ISLocalToGlobalMappingConcatenate(MPI_Comm,PetscInt,const ISLocalToGlobalMapping[],ISLocalToGlobalMapping*);

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingApply"
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

.seealso: DMCreateColoring()
E*/
typedef enum {IS_COLORING_GLOBAL,IS_COLORING_GHOSTED} ISColoringType;
extern const char *ISColoringTypes[];
typedef unsigned PETSC_IS_COLOR_VALUE_TYPE ISColoringValue;
extern PetscErrorCode  ISAllGatherColors(MPI_Comm,PetscInt,ISColoringValue*,PetscInt*,ISColoringValue*[]);

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

extern PetscErrorCode  ISColoringCreate(MPI_Comm,PetscInt,PetscInt,const ISColoringValue[],ISColoring*);
extern PetscErrorCode  ISColoringDestroy(ISColoring*);
extern PetscErrorCode  ISColoringView(ISColoring,PetscViewer);
extern PetscErrorCode  ISColoringGetIS(ISColoring,PetscInt*,IS*[]);
extern PetscErrorCode  ISColoringRestoreIS(ISColoring,IS*[]);
#define ISColoringReference(coloring) ((coloring)->refct++,0)
#define ISColoringSetType(coloring,type) ((coloring)->ctype = type,0)

/* --------------------------------------------------------------------------*/

extern PetscErrorCode  ISPartitioningToNumbering(IS,IS*);
extern PetscErrorCode  ISPartitioningCount(IS,PetscInt,PetscInt[]);

extern PetscErrorCode  ISCompressIndicesGeneral(PetscInt,PetscInt,PetscInt,PetscInt,const IS[],IS[]);
extern PetscErrorCode  ISCompressIndicesSorted(PetscInt,PetscInt,PetscInt,const IS[],IS[]);
extern PetscErrorCode  ISExpandIndicesGeneral(PetscInt,PetscInt,PetscInt,PetscInt,const IS[],IS[]);

PETSC_EXTERN_CXX_END

/* Reset __FUNCT__ in case the user does not define it themselves */
#undef __FUNCT__
#define __FUNCT__ "User provided function"

#endif
