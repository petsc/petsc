/* 
    Defines the vector component of PETSc. Vectors generally represent 
  degrees of freedom for finite element/finite difference functions
  on a grid. They have more mathematical structure then simple arrays.
*/

#ifndef __PETSCVEC_H 
#define __PETSCVEC_H
#include "petscis.h"
#include "petscsys.h"
PETSC_EXTERN_CXX_BEGIN

/*S
     PetscMap - Abstract PETSc object that defines the layout of vector and
  matrices across processors

   Level: advanced

   Notes:
    Does not play a role in the PETSc design, can be ignored

  Concepts: parallel decomposition

.seealso:  PetscMapCreateMPI()
S*/
typedef struct _p_PetscMap*         PetscMap;

#define MAP_SEQ "seq"
#define MAP_MPI "mpi"
#define PetscMapType char*

/* Logging support */
extern int MAP_COOKIE;

EXTERN PetscErrorCode PetscMapCreate(MPI_Comm,PetscMap*);
EXTERN PetscErrorCode PetscMapCreateMPI(MPI_Comm,int,int,PetscMap*);
EXTERN PetscErrorCode PetscMapSetFromOptions(PetscMap);
EXTERN PetscErrorCode PetscMapPrintHelp(PetscMap);
EXTERN PetscErrorCode PetscMapDestroy(PetscMap);

EXTERN PetscErrorCode PetscMapSetLocalSize(PetscMap,int);
EXTERN PetscErrorCode PetscMapGetLocalSize(PetscMap,int *);
EXTERN PetscErrorCode PetscMapSetSize(PetscMap,int);
EXTERN PetscErrorCode PetscMapGetSize(PetscMap,int *);
EXTERN PetscErrorCode PetscMapGetLocalRange(PetscMap,int *,int *);
EXTERN PetscErrorCode PetscMapGetGlobalRange(PetscMap,int *[]);

/* Dynamic creation and loading functions */
extern PetscFList PetscMapList;
extern PetscTruth PetscMapRegisterAllCalled;
EXTERN PetscErrorCode PetscMapSetType(PetscMap, const PetscMapType);
EXTERN PetscErrorCode PetscMapGetType(PetscMap, PetscMapType *);
EXTERN PetscErrorCode PetscMapRegister(const char[],const char[],const char[],int(*)(PetscMap));
EXTERN PetscErrorCode PetscMapRegisterAll(const char []);
EXTERN PetscErrorCode PetscMapRegisterDestroy(void);
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PetscMapRegisterDynamic(a,b,c,d) PetscMapRegister(a,b,c,0)
#else
#define PetscMapRegisterDynamic(a,b,c,d) PetscMapRegister(a,b,c,d)
#endif

/*S
     Vec - Abstract PETSc vector object

   Level: beginner

  Concepts: field variables, unknowns, arrays

.seealso:  VecCreate(), VecType, VecSetType()
S*/
typedef struct _p_Vec*         Vec;

/*S
     VecScatter - Object used to manage communication of data
       between vectors in parallel. Manages both scatters and gathers

   Level: beginner

  Concepts: scatter

.seealso:  VecScatterCreate(), VecScatterBegin(), VecScatterEnd()
S*/
typedef struct _p_VecScatter*  VecScatter;

/*E
    VecType - String with the name of a PETSc vector or the creation function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:myveccreate()

   Level: beginner

.seealso: VecSetType(), Vec
E*/
#define VECSEQ         "seq"
#define VECMPI         "mpi"
#define VECFETI        "feti"
#define VECSHARED      "shared"
#define VECESI         "esi"
#define VECPETSCESI    "petscesi"
#define VecType char*

/* Logging support */
#define    VEC_FILE_COOKIE 1211214
extern int VEC_COOKIE;
extern int VEC_SCATTER_COOKIE;
extern int VEC_View, VEC_Max, VEC_Min, VEC_DotBarrier, VEC_Dot, VEC_MDotBarrier, VEC_MDot, VEC_TDot, VEC_MTDot, VEC_NormBarrier;
extern int VEC_Norm, VEC_Normalize, VEC_Scale, VEC_Copy, VEC_Set, VEC_AXPY, VEC_AYPX, VEC_WAXPY, VEC_MAXPY, VEC_Swap, VEC_AssemblyBegin;
extern int VEC_AssemblyEnd, VEC_PointwiseMult, VEC_SetValues, VEC_Load, VEC_ScatterBarrier, VEC_ScatterBegin, VEC_ScatterEnd;
extern int VEC_SetRandom, VEC_ReduceArithmetic, VEC_ReduceBarrier, VEC_ReduceCommunication;

EXTERN PetscErrorCode VecInitializePackage(char *);

EXTERN PetscErrorCode VecCreate(MPI_Comm,Vec *);
EXTERN PetscErrorCode VecCreateSeq(MPI_Comm,int,Vec*);
EXTERN PetscErrorCode VecCreateMPI(MPI_Comm,int,int,Vec*);
EXTERN PetscErrorCode VecCreateSeqWithArray(MPI_Comm,int,const PetscScalar[],Vec*);
EXTERN PetscErrorCode VecCreateMPIWithArray(MPI_Comm,int,int,const PetscScalar[],Vec*);
EXTERN PetscErrorCode VecCreateShared(MPI_Comm,int,int,Vec*);
EXTERN PetscErrorCode VecSetFromOptions(Vec);
EXTERN PetscErrorCode VecPrintHelp(Vec);
EXTERN PetscErrorCode VecDestroy(Vec);

EXTERN PetscErrorCode VecSetSizes(Vec,int,int);

EXTERN PetscErrorCode VecDot(Vec,Vec,PetscScalar*);
EXTERN PetscErrorCode VecTDot(Vec,Vec,PetscScalar*);  
EXTERN PetscErrorCode VecMDot(int,Vec,const Vec[],PetscScalar*);
EXTERN PetscErrorCode VecMTDot(int,Vec,const Vec[],PetscScalar*); 

/*E
    NormType - determines what type of norm to compute

    Level: beginner

.seealso: VecNorm(), VecNormBegin(), VecNormEnd(), MatNorm()
E*/
typedef enum {NORM_1=1,NORM_2=2,NORM_FROBENIUS=3,NORM_INFINITY=4,NORM_1_AND_2=5} NormType;
#define NORM_MAX NORM_INFINITY

/*MC
     NORM_1 - the one norm, ||v|| = sum_i | v_i |. ||A|| = max_j || v_*j ||, maximum column sum

   Level: beginner

.seealso:  NormType, MatNorm(), VecNorm(), VecNormBegin(), VecNormEnd(), NORM_2, NORM_FROBENIUS, 
           NORM_INFINITY, NORM_1_AND_2

M*/

/*MC
     NORM_2 - the two norm, ||v|| = sqrt(sum_i (v_i)^2) (vectors only)

   Level: beginner

.seealso:  NormType, MatNorm(), VecNorm(), VecNormBegin(), VecNormEnd(), NORM_1, NORM_FROBENIUS, 
           NORM_INFINITY, NORM_1_AND_2

M*/

/*MC
     NORM_FROBENIUS - ||A|| = sqrt(sum_ij (A_ij)^2), same as NORM_2 for vectors

   Level: beginner

.seealso:  NormType, MatNorm(), VecNorm(), VecNormBegin(), VecNormEnd(), NORM_1, NORM_2, 
           NORM_INFINITY, NORM_1_AND_2

M*/

/*MC
     NORM_INFINITY - ||v|| = max_i |v_i|. ||A|| = max_i || v_i* ||, maximum row sum

   Level: beginner

.seealso:  NormType, MatNorm(), VecNorm(), VecNormBegin(), VecNormEnd(), NORM_1, NORM_2, 
           NORM_FROBINIUS, NORM_1_AND_2

M*/

/*MC
     NORM_1_AND_2 - computes both the 1 and 2 norm of a vector

   Level: beginner

.seealso:  NormType, MatNorm(), VecNorm(), VecNormBegin(), VecNormEnd(), NORM_1, NORM_2, 
           NORM_FROBINIUS, NORM_INFINITY

M*/

/*MC
     NORM_MAX - see NORM_INFINITY

   Level: beginner

M*/

EXTERN PetscErrorCode VecNorm(Vec,NormType,PetscReal *);
EXTERN PetscErrorCode VecNormComposedDataID(NormType,int*);
EXTERN PetscErrorCode VecNormalize(Vec,PetscReal *);
EXTERN PetscErrorCode VecSum(Vec,PetscScalar*);
EXTERN PetscErrorCode VecMax(Vec,int*,PetscReal *);
EXTERN PetscErrorCode VecMin(Vec,int*,PetscReal *);
EXTERN PetscErrorCode VecScale(const PetscScalar *a,Vec v);
EXTERN PetscErrorCode VecCopy(Vec,Vec);        
EXTERN PetscErrorCode VecSetRandom(PetscRandom,Vec);
EXTERN PetscErrorCode VecSet(const PetscScalar*,Vec);
EXTERN PetscErrorCode VecSwap(Vec,Vec);
EXTERN PetscErrorCode VecAXPY(const PetscScalar*,Vec,Vec);  
EXTERN PetscErrorCode VecAXPBY(const PetscScalar*,const PetscScalar *,Vec,Vec);  
EXTERN PetscErrorCode VecMAXPY(int,const PetscScalar*,Vec,Vec*);
EXTERN PetscErrorCode VecAYPX(const PetscScalar*,Vec,Vec);
EXTERN PetscErrorCode VecWAXPY(const PetscScalar*,Vec,Vec,Vec);
EXTERN PetscErrorCode VecPointwiseMult(Vec,Vec,Vec);    
EXTERN PetscErrorCode VecPointwiseDivide(Vec,Vec,Vec);    
EXTERN PetscErrorCode VecMaxPointwiseDivide(Vec,Vec,PetscReal*);    
EXTERN PetscErrorCode VecShift(const PetscScalar*,Vec);
EXTERN PetscErrorCode VecReciprocal(Vec);
EXTERN PetscErrorCode VecPermute(Vec, IS, PetscTruth);
EXTERN PetscErrorCode VecSqrt(Vec);
EXTERN PetscErrorCode VecAbs(Vec);
EXTERN PetscErrorCode VecDuplicate(Vec,Vec*);          
EXTERN PetscErrorCode VecDuplicateVecs(Vec,int,Vec*[]);         
EXTERN PetscErrorCode VecDestroyVecs(const Vec[],int); 
EXTERN PetscErrorCode VecGetPetscMap(Vec,PetscMap*);

EXTERN PetscErrorCode VecStrideNormAll(Vec,NormType,PetscReal*);
EXTERN PetscErrorCode VecStrideMaxAll(Vec,int *,PetscReal *);
EXTERN PetscErrorCode VecStrideMinAll(Vec,int *,PetscReal *);
EXTERN PetscErrorCode VecStrideScaleAll(Vec,PetscScalar*);

EXTERN PetscErrorCode VecStrideNorm(Vec,int,NormType,PetscReal*);
EXTERN PetscErrorCode VecStrideMax(Vec,int,int *,PetscReal *);
EXTERN PetscErrorCode VecStrideMin(Vec,int,int *,PetscReal *);
EXTERN PetscErrorCode VecStrideScale(Vec,int,PetscScalar*);
EXTERN PetscErrorCode VecStrideGather(Vec,int,Vec,InsertMode);
EXTERN PetscErrorCode VecStrideScatter(Vec,int,Vec,InsertMode);
EXTERN PetscErrorCode VecStrideGatherAll(Vec,Vec*,InsertMode);
EXTERN PetscErrorCode VecStrideScatterAll(Vec*,Vec,InsertMode);

EXTERN PetscErrorCode VecSetValues(Vec,int,const int[],const PetscScalar[],InsertMode);
EXTERN PetscErrorCode VecAssemblyBegin(Vec);
EXTERN PetscErrorCode VecAssemblyEnd(Vec);
EXTERN PetscErrorCode VecStashSetInitialSize(Vec,int,int);
EXTERN PetscErrorCode VecStashView(Vec,PetscViewer);
EXTERN PetscErrorCode VecStashGetInfo(Vec,int*,int*,int*,int*);

extern int         VecSetValue_Row;
extern PetscScalar VecSetValue_Value;
/*MC
   VecSetValue - Set a single entry into a vector.

   Synopsis:
   int VecSetValue(Vec v,int row,PetscScalar value, InsertMode mode);

   Not Collective

   Input Parameters:
+  v - the vector
.  row - the row location of the entry
.  value - the value to insert
-  mode - either INSERT_VALUES or ADD_VALUES

   Notes:
   For efficiency one should use VecSetValues() and set several or 
   many values simultaneously if possible.

   These values may be cached, so VecAssemblyBegin() and VecAssemblyEnd() 
   MUST be called after all calls to VecSetValues() have been completed.

   VecSetValues() uses 0-based indices in Fortran as well as in C.

   Level: beginner

.seealso: VecSetValues(), VecAssemblyBegin(), VecAssemblyEnd(), VecSetValuesBlockedLocal(), VecSetValueLocal()
M*/
#define VecSetValue(v,i,va,mode) (VecSetValue_Row = i, VecSetValue_Value = va, VecSetValues(v,1,&VecSetValue_Row,&VecSetValue_Value,mode))

/*MC
   VecSetValueLocal - Set a single entry into a vector using the local numbering

   Synopsis:
   int VecSetValueLocal(Vec v,int row,PetscScalar value, InsertMode mode);

   Not Collective

   Input Parameters:
+  v - the vector
.  row - the row location of the entry
.  value - the value to insert
-  mode - either INSERT_VALUES or ADD_VALUES

   Notes:
   For efficiency one should use VecSetValues() and set several or 
   many values simultaneously if possible.

   These values may be cached, so VecAssemblyBegin() and VecAssemblyEnd() 
   MUST be called after all calls to VecSetValues() have been completed.

   VecSetValues() uses 0-based indices in Fortran as well as in C.

   Level: beginner

.seealso: VecSetValues(), VecAssemblyBegin(), VecAssemblyEnd(), VecSetValuesBlockedLocal(), VecSetValue()
M*/
#define VecSetValueLocal(v,i,va,mode) (VecSetValue_Row = i,VecSetValue_Value = va,VecSetValuesLocal(v,1,&VecSetValue_Row,&VecSetValue_Value,mode))

EXTERN PetscErrorCode VecSetBlockSize(Vec,int);
EXTERN PetscErrorCode VecGetBlockSize(Vec,int*);
EXTERN PetscErrorCode VecSetValuesBlocked(Vec,int,const int[],const PetscScalar[],InsertMode);

/* Dynamic creation and loading functions */
extern PetscFList VecList;
extern PetscTruth VecRegisterAllCalled;
EXTERN PetscErrorCode VecSetType(Vec, const VecType);
EXTERN PetscErrorCode VecGetType(Vec, VecType *);
EXTERN PetscErrorCode VecRegister(const char[],const char[],const char[],int(*)(Vec));
EXTERN PetscErrorCode VecRegisterAll(const char []);
EXTERN PetscErrorCode VecRegisterDestroy(void);

/*MC
  VecRegisterDynamic - Adds a new vector component implementation

  Synopsis:
  VecRegisterDynamic(char *name, char *path, char *func_name, int (*create_func)(Vec))

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
. path        - The path (either absolute or relative) of the library containing this routine
. func_name   - The name of routine to create method context
- create_func - The creation routine itself

  Notes:
  VecRegisterDynamic() may be called multiple times to add several user-defined vectors

  If dynamic libraries are used, then the fourth input argument (routine_create) is ignored.

  Sample usage:
.vb
    VecRegisterDynamic("my_vec","/home/username/my_lib/lib/libO/solaris/libmy.a", "MyVectorCreate", MyVectorCreate);
.ve

  Then, your vector type can be chosen with the procedural interface via
.vb
    VecCreate(MPI_Comm, Vec *);
    VecSetType(Vec,"my_vector_name");
.ve
   or at runtime via the option
.vb
    -vec_type my_vector_name
.ve

  Notes: $PETSC_ARCH and $BOPT occuring in pathname will be replaced with appropriate values.
         If your function is not being put into a shared library then use VecRegister() instead
        
  Level: advanced

.keywords: Vec, register
.seealso: VecRegisterAll(), VecRegisterDestroy(), VecRegister()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define VecRegisterDynamic(a,b,c,d) VecRegister(a,b,c,0)
#else
#define VecRegisterDynamic(a,b,c,d) VecRegister(a,b,c,d)
#endif


EXTERN PetscErrorCode VecScatterCreate(Vec,IS,Vec,IS,VecScatter *);
EXTERN PetscErrorCode VecScatterPostRecvs(Vec,Vec,InsertMode,ScatterMode,VecScatter);
EXTERN PetscErrorCode VecScatterBegin(Vec,Vec,InsertMode,ScatterMode,VecScatter);
EXTERN PetscErrorCode VecScatterEnd(Vec,Vec,InsertMode,ScatterMode,VecScatter); 
EXTERN PetscErrorCode VecScatterDestroy(VecScatter);
EXTERN PetscErrorCode VecScatterCopy(VecScatter,VecScatter *);
EXTERN PetscErrorCode VecScatterView(VecScatter,PetscViewer);
EXTERN PetscErrorCode VecScatterRemap(VecScatter,int *,int*);

typedef enum {PIPELINE_DOWN=0,PIPELINE_UP=1} PipelineDirection;
typedef enum {PIPELINE_NONE=1,PIPELINE_SEQUENTIAL=2,
	      PIPELINE_REDBLACK=3,PIPELINE_MULTICOLOR=4} PipelineType;

typedef struct _p_VecPipeline*  VecPipeline;

EXTERN PetscErrorCode VecPipelineCreate(MPI_Comm,Vec,IS,Vec,IS,VecPipeline *);
EXTERN PetscErrorCode VecPipelineSetType(VecPipeline,PipelineType,PetscObject);
EXTERN PetscErrorCode VecPipelineSetup(VecPipeline);
EXTERN PetscErrorCode VecPipelineBegin(Vec,Vec,InsertMode,ScatterMode,PipelineDirection,VecPipeline);
EXTERN PetscErrorCode VecPipelineEnd(Vec,Vec,InsertMode,ScatterMode,PipelineDirection,VecPipeline); 
EXTERN PetscErrorCode VecPipelineView(VecPipeline,PetscViewer);
EXTERN PetscErrorCode VecPipelineDestroy(VecPipeline);

EXTERN PetscErrorCode VecGetArray_Private(Vec,PetscScalar*[]);
EXTERN PetscErrorCode VecRestoreArray_Private(Vec,PetscScalar*[]);
EXTERN PetscErrorCode VecGetArray4d(Vec,int,int,int,int,int,int,int,int,PetscScalar****[]);
EXTERN PetscErrorCode VecRestoreArray4d(Vec,int,int,int,int,int,int,int,int,PetscScalar****[]);
EXTERN PetscErrorCode VecGetArray3d(Vec,int,int,int,int,int,int,PetscScalar***[]);
EXTERN PetscErrorCode VecRestoreArray3d(Vec,int,int,int,int,int,int,PetscScalar***[]);
EXTERN PetscErrorCode VecGetArray2d(Vec,int,int,int,int,PetscScalar**[]);
EXTERN PetscErrorCode VecRestoreArray2d(Vec,int,int,int,int,PetscScalar**[]);
EXTERN PetscErrorCode VecGetArray1d(Vec,int,int,PetscScalar *[]);
EXTERN PetscErrorCode VecRestoreArray1d(Vec,int,int,PetscScalar *[]);

EXTERN PetscErrorCode VecPlaceArray(Vec,const PetscScalar[]);
EXTERN PetscErrorCode VecResetArray(Vec);
EXTERN PetscErrorCode VecReplaceArray(Vec,const PetscScalar[]);
EXTERN PetscErrorCode VecGetArrays(const Vec[],int,PetscScalar**[]);
EXTERN PetscErrorCode VecRestoreArrays(const Vec[],int,PetscScalar**[]);

EXTERN PetscErrorCode VecValid(Vec,PetscTruth*);
EXTERN PetscErrorCode VecView(Vec,PetscViewer);
EXTERN PetscErrorCode VecViewFromOptions(Vec, char *);
EXTERN PetscErrorCode VecEqual(Vec,Vec,PetscTruth*);
EXTERN PetscErrorCode VecLoad(PetscViewer,const VecType,Vec*);
EXTERN PetscErrorCode VecLoadIntoVector(PetscViewer,Vec);

EXTERN PetscErrorCode VecGetSize(Vec,int*);
EXTERN PetscErrorCode VecGetLocalSize(Vec,int*);
EXTERN PetscErrorCode VecGetOwnershipRange(Vec,int*,int*);

EXTERN PetscErrorCode VecSetLocalToGlobalMapping(Vec,ISLocalToGlobalMapping);
EXTERN PetscErrorCode VecSetValuesLocal(Vec,int,const int[],const PetscScalar[],InsertMode);
EXTERN PetscErrorCode VecSetLocalToGlobalMappingBlock(Vec,ISLocalToGlobalMapping);
EXTERN PetscErrorCode VecSetValuesBlockedLocal(Vec,int,const int[],const PetscScalar[],InsertMode);

EXTERN PetscErrorCode VecDotBegin(Vec,Vec,PetscScalar *);
EXTERN PetscErrorCode VecDotEnd(Vec,Vec,PetscScalar *);
EXTERN PetscErrorCode VecTDotBegin(Vec,Vec,PetscScalar *);
EXTERN PetscErrorCode VecTDotEnd(Vec,Vec,PetscScalar *);
EXTERN PetscErrorCode VecNormBegin(Vec,NormType,PetscReal *);
EXTERN PetscErrorCode VecNormEnd(Vec,NormType,PetscReal *);

typedef enum {VEC_IGNORE_OFF_PROC_ENTRIES,VEC_TREAT_OFF_PROC_ENTRIES} VecOption;
EXTERN PetscErrorCode VecSetOption(Vec,VecOption);

/*
   Expose VecGetArray()/VecRestoreArray() to users. Allows this to work without any function
   call overhead on any 'native' Vecs.
*/
#include "vecimpl.h"

EXTERN PetscErrorCode VecContourScale(Vec,PetscReal,PetscReal);

/*
    These numbers need to match the entries in 
  the function table in vecimpl.h
*/
typedef enum { VECOP_VIEW = 32,
               VECOP_LOADINTOVECTOR = 38
             } VecOperation;
EXTERN PetscErrorCode VecSetOperation(Vec,VecOperation,void(*)(void));

/*
     Routines for dealing with ghosted vectors:
  vectors with ghost elements at the end of the array.
*/
EXTERN PetscErrorCode VecCreateGhost(MPI_Comm,int,int,int,const int[],Vec*);  
EXTERN PetscErrorCode VecCreateGhostWithArray(MPI_Comm,int,int,int,const int[],const PetscScalar[],Vec*);  
EXTERN PetscErrorCode VecCreateGhostBlock(MPI_Comm,int,int,int,int,const int[],Vec*);  
EXTERN PetscErrorCode VecCreateGhostBlockWithArray(MPI_Comm,int,int,int,int,const int[],const PetscScalar[],Vec*);  
EXTERN PetscErrorCode VecGhostGetLocalForm(Vec,Vec*);
EXTERN PetscErrorCode VecGhostRestoreLocalForm(Vec,Vec*);
EXTERN PetscErrorCode VecGhostUpdateBegin(Vec,InsertMode,ScatterMode);
EXTERN PetscErrorCode VecGhostUpdateEnd(Vec,InsertMode,ScatterMode);

EXTERN PetscErrorCode VecConjugate(Vec);

EXTERN PetscErrorCode VecScatterCreateToAll(Vec,VecScatter*,Vec*);
EXTERN PetscErrorCode VecScatterCreateToZero(Vec,VecScatter*,Vec*);

EXTERN PetscErrorCode VecESISetType(Vec,const char*);
EXTERN PetscErrorCode VecESISetFromOptions(Vec);

EXTERN PetscErrorCode PetscViewerMathematicaGetVector(PetscViewer, Vec);
EXTERN PetscErrorCode PetscViewerMathematicaPutVector(PetscViewer, Vec);

/*S
     Vecs - Collection of vectors where the data for the vectors is stored in 
            one continquous memory

   Level: advanced

   Notes:
    Temporary construct for handling multiply right hand side solves

    This is faked by storing a single vector that has enough array space for 
    n vectors

  Concepts: parallel decomposition

S*/
        struct _p_Vecs  {int n; Vec v;};
typedef struct _p_Vecs* Vecs;
#define VecsDestroy(x)            (VecDestroy((x)->v)         || PetscFree(x))
#define VecsCreateSeq(comm,p,m,x) (PetscNew(struct _p_Vecs,x) || VecCreateSeq(comm,p*m,&(*(x))->v) || (-1 == ((*(x))->n = (m))))
#define VecsCreateSeqWithArray(comm,p,m,a,x) (PetscNew(struct _p_Vecs,x) || VecCreateSeqWithArray(comm,p*m,a,&(*(x))->v) || (-1 == ((*(x))->n = (m))))
#define VecsDuplicate(x,y)        (PetscNew(struct _p_Vecs,y) || VecDuplicate(x->v,&(*(y))->v) || (-1 == ((*(y))->n = (x)->n)))


PETSC_EXTERN_CXX_END
#endif
