/* $Id: petscvec.h,v 1.127 2001/09/11 16:31:30 bsmith Exp $ */
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

EXTERN int PetscMapCreate(MPI_Comm,PetscMap*);
EXTERN int PetscMapCreateMPI(MPI_Comm,int,int,PetscMap*);
EXTERN int PetscMapSetFromOptions(PetscMap);
EXTERN int PetscMapPrintHelp(PetscMap);
EXTERN int PetscMapDestroy(PetscMap);

EXTERN int PetscMapSetLocalSize(PetscMap,int);
EXTERN int PetscMapGetLocalSize(PetscMap,int *);
EXTERN int PetscMapSetSize(PetscMap,int);
EXTERN int PetscMapGetSize(PetscMap,int *);
EXTERN int PetscMapGetLocalRange(PetscMap,int *,int *);
EXTERN int PetscMapGetGlobalRange(PetscMap,int *[]);

/* Dynamic creation and loading functions */
extern PetscFList PetscMapList;
extern PetscTruth PetscMapRegisterAllCalled;
EXTERN int PetscMapSetType(PetscMap, const PetscMapType);
EXTERN int PetscMapGetType(PetscMap, PetscMapType *);
EXTERN int PetscMapRegister(const char[],const char[],const char[],int(*)(PetscMap));
EXTERN int PetscMapRegisterAll(const char []);
EXTERN int PetscMapRegisterDestroy(void);
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

EXTERN int VecInitializePackage(char *);

EXTERN int VecCreate(MPI_Comm,Vec *);
EXTERN int VecCreateSeq(MPI_Comm,int,Vec*);
EXTERN int VecCreateMPI(MPI_Comm,int,int,Vec*);
EXTERN int VecCreateSeqWithArray(MPI_Comm,int,const PetscScalar[],Vec*);
EXTERN int VecCreateMPIWithArray(MPI_Comm,int,int,const PetscScalar[],Vec*);
EXTERN int VecCreateShared(MPI_Comm,int,int,Vec*);
EXTERN int VecSetFromOptions(Vec);
EXTERN int VecPrintHelp(Vec);
EXTERN int VecDestroy(Vec);

EXTERN int VecSetSizes(Vec,int,int);

EXTERN int VecDot(Vec,Vec,PetscScalar*);
EXTERN int VecTDot(Vec,Vec,PetscScalar*);  
EXTERN int VecMDot(int,Vec,const Vec[],PetscScalar*);
EXTERN int VecMTDot(int,Vec,const Vec[],PetscScalar*); 

/*E
    NormType - determines what type of norm to compute

    Level: beginner

.seealso: VecNorm(), VecNormBegin(), VecNormEnd(), MatNorm()
E*/
typedef enum {NORM_1=1,NORM_2=2,NORM_FROBENIUS=3,NORM_INFINITY=4,NORM_1_AND_2=5} NormType;
#define NORM_MAX NORM_INFINITY

EXTERN int VecNorm(Vec,NormType,PetscReal *);
EXTERN int VecNormComposedDataID(NormType,int*);
EXTERN int VecNormalize(Vec,PetscReal *);
EXTERN int VecSum(Vec,PetscScalar*);
EXTERN int VecMax(Vec,int*,PetscReal *);
EXTERN int VecMin(Vec,int*,PetscReal *);
EXTERN int VecScale(const PetscScalar *a,Vec v);
EXTERN int VecCopy(Vec,Vec);        
EXTERN int VecSetRandom(PetscRandom,Vec);
EXTERN int VecSet(const PetscScalar*,Vec);
EXTERN int VecSwap(Vec,Vec);
EXTERN int VecAXPY(const PetscScalar*,Vec,Vec);  
EXTERN int VecAXPBY(const PetscScalar*,const PetscScalar *,Vec,Vec);  
EXTERN int VecMAXPY(int,const PetscScalar*,Vec,Vec*);
EXTERN int VecAYPX(const PetscScalar*,Vec,Vec);
EXTERN int VecWAXPY(const PetscScalar*,Vec,Vec,Vec);
EXTERN int VecPointwiseMult(Vec,Vec,Vec);    
EXTERN int VecPointwiseDivide(Vec,Vec,Vec);    
EXTERN int VecMaxPointwiseDivide(Vec,Vec,PetscReal*);    
EXTERN int VecShift(const PetscScalar*,Vec);
EXTERN int VecReciprocal(Vec);
EXTERN int VecPermute(Vec, IS, PetscTruth);
EXTERN int VecSqrt(Vec);
EXTERN int VecAbs(Vec);
EXTERN int VecDuplicate(Vec,Vec*);          
EXTERN int VecDuplicateVecs(Vec,int,Vec*[]);         
EXTERN int VecDestroyVecs(const Vec[],int); 
EXTERN int VecGetPetscMap(Vec,PetscMap*);

EXTERN int VecStrideNormAll(Vec,NormType,PetscReal*);
EXTERN int VecStrideMaxAll(Vec,int *,PetscReal *);
EXTERN int VecStrideMinAll(Vec,int *,PetscReal *);
EXTERN int VecStrideScaleAll(Vec,PetscScalar*);

EXTERN int VecStrideNorm(Vec,int,NormType,PetscReal*);
EXTERN int VecStrideMax(Vec,int,int *,PetscReal *);
EXTERN int VecStrideMin(Vec,int,int *,PetscReal *);
EXTERN int VecStrideScale(Vec,int,PetscScalar*);
EXTERN int VecStrideGather(Vec,int,Vec,InsertMode);
EXTERN int VecStrideScatter(Vec,int,Vec,InsertMode);
EXTERN int VecStrideGatherAll(Vec,Vec*,InsertMode);
EXTERN int VecStrideScatterAll(Vec*,Vec,InsertMode);

EXTERN int VecSetValues(Vec,int,const int[],const PetscScalar[],InsertMode);
EXTERN int VecAssemblyBegin(Vec);
EXTERN int VecAssemblyEnd(Vec);
EXTERN int VecStashSetInitialSize(Vec,int,int);
EXTERN int VecStashView(Vec,PetscViewer);
EXTERN int VecStashGetInfo(Vec,int*,int*,int*,int*);

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

EXTERN int VecSetBlockSize(Vec,int);
EXTERN int VecGetBlockSize(Vec,int*);
EXTERN int VecSetValuesBlocked(Vec,int,const int[],const PetscScalar[],InsertMode);

/* Dynamic creation and loading functions */
extern PetscFList VecList;
extern PetscTruth VecRegisterAllCalled;
EXTERN int VecSetType(Vec, const VecType);
EXTERN int VecGetType(Vec, VecType *);
EXTERN int VecRegister(const char[],const char[],const char[],int(*)(Vec));
EXTERN int VecRegisterAll(const char []);
EXTERN int VecRegisterDestroy(void);

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


EXTERN int VecScatterCreate(Vec,IS,Vec,IS,VecScatter *);
EXTERN int VecScatterPostRecvs(Vec,Vec,InsertMode,ScatterMode,VecScatter);
EXTERN int VecScatterBegin(Vec,Vec,InsertMode,ScatterMode,VecScatter);
EXTERN int VecScatterEnd(Vec,Vec,InsertMode,ScatterMode,VecScatter); 
EXTERN int VecScatterDestroy(VecScatter);
EXTERN int VecScatterCopy(VecScatter,VecScatter *);
EXTERN int VecScatterView(VecScatter,PetscViewer);
EXTERN int VecScatterRemap(VecScatter,int *,int*);

typedef enum {PIPELINE_DOWN=0,PIPELINE_UP=1} PipelineDirection;
typedef enum {PIPELINE_NONE=1,PIPELINE_SEQUENTIAL=2,
	      PIPELINE_REDBLACK=3,PIPELINE_MULTICOLOR=4} PipelineType;

typedef struct _p_VecPipeline*  VecPipeline;

EXTERN int VecPipelineCreate(MPI_Comm,Vec,IS,Vec,IS,VecPipeline *);
EXTERN int VecPipelineSetType(VecPipeline,PipelineType,PetscObject);
EXTERN int VecPipelineSetup(VecPipeline);
EXTERN int VecPipelineBegin(Vec,Vec,InsertMode,ScatterMode,PipelineDirection,VecPipeline);
EXTERN int VecPipelineEnd(Vec,Vec,InsertMode,ScatterMode,PipelineDirection,VecPipeline); 
EXTERN int VecPipelineView(VecPipeline,PetscViewer);
EXTERN int VecPipelineDestroy(VecPipeline);

EXTERN int VecGetArray_Private(Vec,PetscScalar*[]);
EXTERN int VecRestoreArray_Private(Vec,PetscScalar*[]);
EXTERN int VecGetArray4d(Vec,int,int,int,int,int,int,int,int,PetscScalar****[]);
EXTERN int VecRestoreArray4d(Vec,int,int,int,int,int,int,int,int,PetscScalar****[]);
EXTERN int VecGetArray3d(Vec,int,int,int,int,int,int,PetscScalar***[]);
EXTERN int VecRestoreArray3d(Vec,int,int,int,int,int,int,PetscScalar***[]);
EXTERN int VecGetArray2d(Vec,int,int,int,int,PetscScalar**[]);
EXTERN int VecRestoreArray2d(Vec,int,int,int,int,PetscScalar**[]);
EXTERN int VecGetArray1d(Vec,int,int,PetscScalar *[]);
EXTERN int VecRestoreArray1d(Vec,int,int,PetscScalar *[]);

EXTERN int VecPlaceArray(Vec,const PetscScalar[]);
EXTERN int VecResetArray(Vec);
EXTERN int VecReplaceArray(Vec,const PetscScalar[]);
EXTERN int VecGetArrays(const Vec[],int,PetscScalar**[]);
EXTERN int VecRestoreArrays(const Vec[],int,PetscScalar**[]);

EXTERN int VecValid(Vec,PetscTruth*);
EXTERN int VecView(Vec,PetscViewer);
EXTERN int VecViewFromOptions(Vec, char *);
EXTERN int VecEqual(Vec,Vec,PetscTruth*);
EXTERN int VecLoad(PetscViewer,const VecType,Vec*);
EXTERN int VecLoadIntoVector(PetscViewer,Vec);

EXTERN int VecGetSize(Vec,int*);
EXTERN int VecGetLocalSize(Vec,int*);
EXTERN int VecGetOwnershipRange(Vec,int*,int*);

EXTERN int VecSetLocalToGlobalMapping(Vec,ISLocalToGlobalMapping);
EXTERN int VecSetValuesLocal(Vec,int,const int[],const PetscScalar[],InsertMode);
EXTERN int VecSetLocalToGlobalMappingBlock(Vec,ISLocalToGlobalMapping);
EXTERN int VecSetValuesBlockedLocal(Vec,int,const int[],const PetscScalar[],InsertMode);

EXTERN int VecDotBegin(Vec,Vec,PetscScalar *);
EXTERN int VecDotEnd(Vec,Vec,PetscScalar *);
EXTERN int VecTDotBegin(Vec,Vec,PetscScalar *);
EXTERN int VecTDotEnd(Vec,Vec,PetscScalar *);
EXTERN int VecNormBegin(Vec,NormType,PetscReal *);
EXTERN int VecNormEnd(Vec,NormType,PetscReal *);

typedef enum {VEC_IGNORE_OFF_PROC_ENTRIES,VEC_TREAT_OFF_PROC_ENTRIES} VecOption;
EXTERN int VecSetOption(Vec,VecOption);

/*
   Expose VecGetArray()/VecRestoreArray() to users. Allows this to work without any function
   call overhead on any 'native' Vecs.
*/
#include "vecimpl.h"

EXTERN int VecContourScale(Vec,PetscReal,PetscReal);

/*
    These numbers need to match the entries in 
  the function table in vecimpl.h
*/
typedef enum { VECOP_VIEW = 32,
               VECOP_LOADINTOVECTOR = 38
             } VecOperation;
EXTERN int VecSetOperation(Vec,VecOperation,void(*)(void));

/*
     Routines for dealing with ghosted vectors:
  vectors with ghost elements at the end of the array.
*/
EXTERN int VecCreateGhost(MPI_Comm,int,int,int,const int[],Vec*);  
EXTERN int VecCreateGhostWithArray(MPI_Comm,int,int,int,const int[],const PetscScalar[],Vec*);  
EXTERN int VecCreateGhostBlock(MPI_Comm,int,int,int,int,const int[],Vec*);  
EXTERN int VecCreateGhostBlockWithArray(MPI_Comm,int,int,int,int,const int[],const PetscScalar[],Vec*);  
EXTERN int VecGhostGetLocalForm(Vec,Vec*);
EXTERN int VecGhostRestoreLocalForm(Vec,Vec*);
EXTERN int VecGhostUpdateBegin(Vec,InsertMode,ScatterMode);
EXTERN int VecGhostUpdateEnd(Vec,InsertMode,ScatterMode);

EXTERN int VecConjugate(Vec);

EXTERN int VecConvertMPIToSeqAll(Vec vin,Vec *vout);
EXTERN int VecConvertMPIToMPIZero(Vec vin,Vec *vout);


EXTERN int VecESISetType(Vec,const char*);
EXTERN int VecESISetFromOptions(Vec);

EXTERN int PetscViewerMathematicaGetVector(PetscViewer, Vec);
EXTERN int PetscViewerMathematicaPutVector(PetscViewer, Vec);

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
