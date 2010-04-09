/* 
    Defines the vector component of PETSc. Vectors generally represent 
  degrees of freedom for finite element/finite difference functions
  on a grid. They have more mathematical structure then simple arrays.
*/

#ifndef __PETSCVEC_H 
#define __PETSCVEC_H
#include "petscis.h"

PETSC_EXTERN_CXX_BEGIN

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
#define VecType char*
#define VECSEQ         "seq"
#define VECMPI         "mpi"
#define VECFETI        "feti"
#define VECSHARED      "shared"
#define VECSIEVE       "sieve"

/* Logging support */
#define    VEC_FILE_COOKIE 1211214
extern PETSCVEC_DLLEXPORT PetscCookie VEC_COOKIE;
extern PETSCVEC_DLLEXPORT PetscCookie VEC_SCATTER_COOKIE;

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecInitializePackage(const char[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecFinalizePackage(void);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecCreate(MPI_Comm,Vec*);
PetscPolymorphicSubroutine(VecCreate,(Vec *x),(PETSC_COMM_SELF,x))
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecCreateSeq(MPI_Comm,PetscInt,Vec*);
PetscPolymorphicSubroutine(VecCreateSeq,(PetscInt n,Vec *x),(PETSC_COMM_SELF,n,x))
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecCreateMPI(MPI_Comm,PetscInt,PetscInt,Vec*);
PetscPolymorphicSubroutine(VecCreateMPI,(PetscInt n,PetscInt N,Vec *x),(PETSC_COMM_WORLD,n,N,x))
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecCreateSeqWithArray(MPI_Comm,PetscInt,const PetscScalar[],Vec*);
PetscPolymorphicSubroutine(VecCreateSeqWithArray,(PetscInt n,PetscScalar s[],Vec *x),(PETSC_COMM_SELF,n,s,x))
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecCreateMPIWithArray(MPI_Comm,PetscInt,PetscInt,const PetscScalar[],Vec*);
PetscPolymorphicSubroutine(VecCreateMPIWithArray,(PetscInt n,PetscInt N,PetscScalar s[],Vec *x),(PETSC_COMM_WORLD,n,N,s,x))
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecCreateShared(MPI_Comm,PetscInt,PetscInt,Vec*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecSetFromOptions(Vec);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecSetUp(Vec);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecDestroy(Vec);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecZeroEntries(Vec);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecSetOptionsPrefix(Vec,const char[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecAppendOptionsPrefix(Vec,const char[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecGetOptionsPrefix(Vec,const char*[]);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecSetSizes(Vec,PetscInt,PetscInt);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecDotNorm2(Vec,Vec,PetscScalar*,PetscScalar*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecDot(Vec,Vec,PetscScalar*);
PetscPolymorphicFunction(VecDot,(Vec x,Vec y),(x,y,&s),PetscScalar,s)
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecTDot(Vec,Vec,PetscScalar*);  
PetscPolymorphicFunction(VecTDot,(Vec x,Vec y),(x,y,&s),PetscScalar,s)
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecMDot(Vec,PetscInt,const Vec[],PetscScalar[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecMTDot(Vec,PetscInt,const Vec[],PetscScalar[]); 

/*E
    NormType - determines what type of norm to compute

    Level: beginner

.seealso: VecNorm(), VecNormBegin(), VecNormEnd(), MatNorm()
E*/
typedef enum {NORM_1=0,NORM_2=1,NORM_FROBENIUS=2,NORM_INFINITY=3,NORM_1_AND_2=4} NormType;
extern const char *NormTypes[];
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

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecNorm(Vec,NormType,PetscReal *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecNormAvailable(Vec,NormType,PetscTruth*,PetscReal *);
PetscPolymorphicSubroutine(VecNorm,(Vec x,PetscReal *r),(x,NORM_2,r))
PetscPolymorphicFunction(VecNorm,(Vec x,NormType t),(x,t,&r),PetscReal,r)
PetscPolymorphicFunction(VecNorm,(Vec x),(x,NORM_2,&r),PetscReal,r)
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecNormalize(Vec,PetscReal *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecSum(Vec,PetscScalar*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecMax(Vec,PetscInt*,PetscReal *);
PetscPolymorphicSubroutine(VecMax,(Vec x,PetscReal *r),(x,PETSC_NULL,r))
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecMin(Vec,PetscInt*,PetscReal *);
PetscPolymorphicSubroutine(VecMin,(Vec x,PetscReal *r),(x,PETSC_NULL,r))
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecScale(Vec,PetscScalar);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecCopy(Vec,Vec);        
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecSetRandom(Vec,PetscRandom);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecSet(Vec,PetscScalar);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecSwap(Vec,Vec);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecAXPY(Vec,PetscScalar,Vec);  
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecAXPBY(Vec,PetscScalar,PetscScalar,Vec);  
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecMAXPY(Vec,PetscInt,const PetscScalar[],Vec[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecAYPX(Vec,PetscScalar,Vec);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecWAXPY(Vec,PetscScalar,Vec,Vec);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecAXPBYPCZ(Vec,PetscScalar,PetscScalar,PetscScalar,Vec,Vec);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecPointwiseMax(Vec,Vec,Vec);    
PetscPolymorphicSubroutine(VecPointwiseMax,(Vec x,Vec y),(x,y,y))
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecPointwiseMaxAbs(Vec,Vec,Vec);    
PetscPolymorphicSubroutine(VecPointwiseMaxAbs,(Vec x,Vec y),(x,y,y))
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecPointwiseMin(Vec,Vec,Vec);    
PetscPolymorphicSubroutine(VecPointwiseMin,(Vec x,Vec y),(x,y,y))
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecPointwiseMult(Vec,Vec,Vec);    
PetscPolymorphicSubroutine(VecPointwiseMult,(Vec x,Vec y),(x,x,y))
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecPointwiseDivide(Vec,Vec,Vec);    
PetscPolymorphicSubroutine(VecPointwiseDivide,(Vec x,Vec y),(x,x,y))
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecMaxPointwiseDivide(Vec,Vec,PetscReal*);    
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecShift(Vec,PetscScalar);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecReciprocal(Vec);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecPermute(Vec, IS, PetscTruth);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecSqrt(Vec);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecLog(Vec);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecExp(Vec);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecAbs(Vec);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecDuplicate(Vec,Vec*);          
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecDuplicateVecs(Vec,PetscInt,Vec*[]);         
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecDestroyVecs(Vec[],PetscInt); 
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecStrideNormAll(Vec,NormType,PetscReal[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecStrideMaxAll(Vec,PetscInt [],PetscReal []);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecStrideMinAll(Vec,PetscInt [],PetscReal []);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecStrideScaleAll(Vec,PetscScalar[]);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecStrideNorm(Vec,PetscInt,NormType,PetscReal*);
PetscPolymorphicFunction(VecStrideNorm,(Vec x,PetscInt i),(x,i,NORM_2,&r),PetscReal,r)
PetscPolymorphicFunction(VecStrideNorm,(Vec x,PetscInt i,NormType t),(x,i,t,&r),PetscReal,r)
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecStrideMax(Vec,PetscInt,PetscInt *,PetscReal *);
PetscPolymorphicFunction(VecStrideMax,(Vec x,PetscInt i),(x,i,PETSC_NULL,&r),PetscReal,r)
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecStrideMin(Vec,PetscInt,PetscInt *,PetscReal *);
PetscPolymorphicFunction(VecStrideMin,(Vec x,PetscInt i),(x,i,PETSC_NULL,&r),PetscReal,r)
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecStrideScale(Vec,PetscInt,PetscScalar);


EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecStrideGather(Vec,PetscInt,Vec,InsertMode);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecStrideScatter(Vec,PetscInt,Vec,InsertMode);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecStrideGatherAll(Vec,Vec[],InsertMode);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecStrideScatterAll(Vec[],Vec,InsertMode);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecSetValues(Vec,PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecGetValues(Vec,PetscInt,const PetscInt[],PetscScalar[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecAssemblyBegin(Vec);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecAssemblyEnd(Vec);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecStashSetInitialSize(Vec,PetscInt,PetscInt);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecStashView(Vec,PetscViewer);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecStashGetInfo(Vec,PetscInt*,PetscInt*,PetscInt*,PetscInt*);

/*MC
   VecSetValue - Set a single entry into a vector.

   Synopsis:
   PetscErrorCode VecSetValue(Vec v,int row,PetscScalar value, InsertMode mode);

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
PETSC_STATIC_INLINE PetscErrorCode VecSetValue(Vec v,PetscInt i,PetscScalar va,InsertMode mode) {return VecSetValues(v,1,&i,&va,mode);}


EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecSetBlockSize(Vec,PetscInt);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecGetBlockSize(Vec,PetscInt*);
PetscPolymorphicFunction(VecGetBlockSize,(Vec x),(x,&i),PetscInt,i)
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecSetValuesBlocked(Vec,PetscInt,const PetscInt[],const PetscScalar[],InsertMode);

/* Dynamic creation and loading functions */
extern PetscFList VecList;
extern PetscTruth VecRegisterAllCalled;
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecSetType(Vec, const VecType);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecGetType(Vec, const VecType *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecRegister(const char[],const char[],const char[],PetscErrorCode (*)(Vec));
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecRegisterAll(const char []);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecRegisterDestroy(void);

/*MC
  VecRegisterDynamic - Adds a new vector component implementation

  Synopsis:
  PetscErrorCode VecRegisterDynamic(const char *name, const char *path, const char *func_name, PetscErrorCode (*create_func)(Vec))

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

  Notes: $PETSC_ARCH occuring in pathname will be replaced with appropriate values.
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


EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecScatterCreate(Vec,IS,Vec,IS,VecScatter *);
PetscPolymorphicFunction(VecScatterCreate,(Vec x,IS is1,Vec y,IS is2),(x,is1,y,is2,&s),VecScatter,s)
PetscPolymorphicSubroutine(VecScatterCreate,(Vec x,IS is,Vec y,VecScatter *s),(x,is,y,PETSC_NULL,s))
PetscPolymorphicFunction(VecScatterCreate,(Vec x,IS is,Vec y),(x,is,y,PETSC_NULL,&s),VecScatter,s)
PetscPolymorphicSubroutine(VecScatterCreate,(Vec x,Vec y,IS is,VecScatter *s),(x,PETSC_NULL,y,is,s))
PetscPolymorphicFunction(VecScatterCreate,(Vec x,Vec y,IS is),(x,PETSC_NULL,y,is,&s),VecScatter,s)
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecScatterCreateEmpty(MPI_Comm,VecScatter *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecScatterCreateLocal(VecScatter,PetscInt,const PetscInt[],const PetscInt[],const PetscInt[],PetscInt,const PetscInt[],const PetscInt[],const PetscInt[],PetscInt);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecScatterBegin(VecScatter,Vec,Vec,InsertMode,ScatterMode);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecScatterEnd(VecScatter,Vec,Vec,InsertMode,ScatterMode); 
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecScatterDestroy(VecScatter);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecScatterCopy(VecScatter,VecScatter *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecScatterView(VecScatter,PetscViewer);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecScatterRemap(VecScatter,PetscInt *,PetscInt*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecScatterGetMerged(VecScatter,PetscTruth*);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecGetArray_Private(Vec,PetscScalar*[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecRestoreArray_Private(Vec,PetscScalar*[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecGetArray4d(Vec,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscScalar****[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecRestoreArray4d(Vec,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscScalar****[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecGetArray3d(Vec,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscScalar***[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecRestoreArray3d(Vec,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscScalar***[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecGetArray2d(Vec,PetscInt,PetscInt,PetscInt,PetscInt,PetscScalar**[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecRestoreArray2d(Vec,PetscInt,PetscInt,PetscInt,PetscInt,PetscScalar**[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecGetArray1d(Vec,PetscInt,PetscInt,PetscScalar *[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecRestoreArray1d(Vec,PetscInt,PetscInt,PetscScalar *[]);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecPlaceArray(Vec,const PetscScalar[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecResetArray(Vec);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecReplaceArray(Vec,const PetscScalar[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecGetArrays(const Vec[],PetscInt,PetscScalar**[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecRestoreArrays(const Vec[],PetscInt,PetscScalar**[]);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecValid(Vec,PetscTruth*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecView(Vec,PetscViewer);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecViewFromOptions(Vec, const char *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecEqual(Vec,Vec,PetscTruth*);
PetscPolymorphicFunction(VecEqual,(Vec x,Vec y),(x,y,&s),PetscTruth,s)
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecLoad(PetscViewer,const VecType,Vec*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecLoadIntoVector(PetscViewer,Vec);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecGetSize(Vec,PetscInt*);
PetscPolymorphicFunction(VecGetSize,(Vec x),(x,&s),PetscInt,s)
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecGetLocalSize(Vec,PetscInt*);
PetscPolymorphicFunction(VecGetLocalSize,(Vec x),(x,&s),PetscInt,s)
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecGetOwnershipRange(Vec,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecGetOwnershipRanges(Vec,const PetscInt *[]);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecSetLocalToGlobalMapping(Vec,ISLocalToGlobalMapping);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecSetValuesLocal(Vec,PetscInt,const PetscInt[],const PetscScalar[],InsertMode);

/*MC
   VecSetValueLocal - Set a single entry into a vector using the local numbering

   Synopsis:
   PetscErrorCode VecSetValueLocal(Vec v,int row,PetscScalar value, InsertMode mode);

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
PETSC_STATIC_INLINE PetscErrorCode VecSetValueLocal(Vec v,PetscInt i,PetscScalar va,InsertMode mode) {return VecSetValuesLocal(v,1,&i,&va,mode);}

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecSetLocalToGlobalMappingBlock(Vec,ISLocalToGlobalMapping);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecSetValuesBlockedLocal(Vec,PetscInt,const PetscInt[],const PetscScalar[],InsertMode);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecDotBegin(Vec,Vec,PetscScalar *);
PetscPolymorphicSubroutine(VecDotBegin,(Vec x,Vec y),(x,y,PETSC_NULL))
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecDotEnd(Vec,Vec,PetscScalar *);
PetscPolymorphicFunction(VecDotEnd,(Vec x,Vec y),(x,y,&s),PetscScalar,s)
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecTDotBegin(Vec,Vec,PetscScalar *);
PetscPolymorphicSubroutine(VecTDotBegin,(Vec x,Vec y),(x,y,PETSC_NULL))
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecTDotEnd(Vec,Vec,PetscScalar *);
PetscPolymorphicFunction(VecTDotEnd,(Vec x,Vec y),(x,y,&s),PetscScalar,s)
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecNormBegin(Vec,NormType,PetscReal *);
PetscPolymorphicSubroutine(VecNormBegin,(Vec x,NormType t),(x,t,PETSC_NULL))
PetscPolymorphicSubroutine(VecNormBegin,(Vec x),(x,NORM_2,PETSC_NULL))
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecNormEnd(Vec,NormType,PetscReal *);
PetscPolymorphicFunction(VecNormEnd,(Vec x,NormType t),(x,t,&s),PetscReal,s)
PetscPolymorphicFunction(VecNormEnd,(Vec x),(x,NORM_2,&s),PetscReal,s)

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecMDotBegin(Vec,PetscInt,const Vec[],PetscScalar[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecMDotEnd(Vec,PetscInt,const Vec[],PetscScalar[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecMTDotBegin(Vec,PetscInt,const Vec[],PetscScalar[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecMTDotEnd(Vec,PetscInt,const Vec[],PetscScalar[]);


typedef enum {VEC_IGNORE_OFF_PROC_ENTRIES,VEC_IGNORE_NEGATIVE_INDICES} VecOption;
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecSetOption(Vec,VecOption,PetscTruth);

/*
   Expose VecGetArray()/VecRestoreArray() to users. Allows this to work without any function
   call overhead on any 'native' Vecs.
*/

#include "private/vecimpl.h"

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecContourScale(Vec,PetscReal,PetscReal);

/*
    These numbers need to match the entries in 
  the function table in vecimpl.h
*/
typedef enum { VECOP_VIEW = 33, VECOP_LOADINTOVECTOR = 41} VecOperation;
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecSetOperation(Vec,VecOperation,void(*)(void));

/*
     Routines for dealing with ghosted vectors:
  vectors with ghost elements at the end of the array.
*/
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecCreateGhost(MPI_Comm,PetscInt,PetscInt,PetscInt,const PetscInt[],Vec*);  
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecCreateGhostWithArray(MPI_Comm,PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscScalar[],Vec*);  
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecCreateGhostBlock(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],Vec*);  
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecCreateGhostBlockWithArray(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscScalar[],Vec*);  
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecGhostGetLocalForm(Vec,Vec*);
PetscPolymorphicFunction(VecGhostGetLocalForm,(Vec x),(x,&s),Vec,s)
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecGhostRestoreLocalForm(Vec,Vec*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecGhostUpdateBegin(Vec,InsertMode,ScatterMode);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecGhostUpdateEnd(Vec,InsertMode,ScatterMode);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecConjugate(Vec);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecScatterCreateToAll(Vec,VecScatter*,Vec*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecScatterCreateToZero(Vec,VecScatter*,Vec*);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PetscViewerMathematicaGetVector(PetscViewer, Vec);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PetscViewerMathematicaPutVector(PetscViewer, Vec);

/*S
     Vecs - Collection of vectors where the data for the vectors is stored in 
            one contiguous memory

   Level: advanced

   Notes:
    Temporary construct for handling multiply right hand side solves

    This is faked by storing a single vector that has enough array space for 
    n vectors

  Concepts: parallel decomposition

S*/
        struct _n_Vecs  {PetscInt n; Vec v;};
typedef struct _n_Vecs* Vecs;
#define VecsDestroy(x)            (VecDestroy((x)->v)         || PetscFree(x))
#define VecsCreateSeq(comm,p,m,x) (PetscNew(struct _n_Vecs,x) || VecCreateSeq(comm,p*m,&(*(x))->v) || (-1 == ((*(x))->n = (m))))
#define VecsCreateSeqWithArray(comm,p,m,a,x) (PetscNew(struct _n_Vecs,x) || VecCreateSeqWithArray(comm,p*m,a,&(*(x))->v) || (-1 == ((*(x))->n = (m))))
#define VecsDuplicate(x,y)        (PetscNew(struct _n_Vecs,y) || VecDuplicate(x->v,&(*(y))->v) || (-1 == ((*(y))->n = (x)->n)))


PETSC_EXTERN_CXX_END
#endif
