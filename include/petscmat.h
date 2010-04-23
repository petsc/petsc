/*
     Include file for the matrix component of PETSc
*/
#ifndef __PETSCMAT_H
#define __PETSCMAT_H
#include "petscvec.h"
PETSC_EXTERN_CXX_BEGIN

/*S
     Mat - Abstract PETSc matrix object

   Level: beginner

  Concepts: matrix; linear operator

.seealso:  MatCreate(), MatType, MatSetType()
S*/
typedef struct _p_Mat*           Mat;

/*E
    MatType - String with the name of a PETSc matrix or the creation function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:mymatcreate()

   Level: beginner

.seealso: MatSetType(), Mat, MatSolverPackage
E*/
#define MatType char*
#define MATSAME            "same"
#define MATSEQMAIJ         "seqmaij"
#define MATMPIMAIJ         "mpimaij"
#define MATMAIJ            "maij"
#define MATIS              "is"
#define MATSEQAIJ          "seqaij"
#define MATMPIAIJ          "mpiaij"
#define MATAIJ             "aij"
#define MATSHELL           "shell"
#define MATSEQDENSE        "seqdense"
#define MATMPIDENSE        "mpidense"
#define MATDENSE           "dense"
#define MATSEQBAIJ         "seqbaij"
#define MATMPIBAIJ         "mpibaij"
#define MATBAIJ            "baij"
#define MATMPIADJ          "mpiadj"
#define MATSEQSBAIJ        "seqsbaij"
#define MATMPISBAIJ        "mpisbaij"
#define MATSBAIJ           "sbaij"
#define MATDAAD            "daad"
#define MATMFFD            "mffd"
#define MATNORMAL          "normal"
#define MATLRC             "lrc"
#define MATSEQCSRPERM      "seqcsrperm"
#define MATMPICSRPERM      "mpicsrperm"
#define MATCSRPERM         "csrperm"
#define MATSEQCRL          "seqcrl"
#define MATMPICRL          "mpicrl"
#define MATCRL             "crl"
#define MATSCATTER         "scatter"
#define MATBLOCKMAT        "blockmat"
#define MATCOMPOSITE       "composite"
#define MATSEQFFTW         "seqfftw"
#define MATTRANSPOSEMAT    "transpose"
#define MATSCHURCOMPLEMENT "schurcomplement"
#define MATPYTHON          "python"
#define MATHYPRESTRUCT     "hyprestruct"
#define MATHYPRESSTRUCT    "hypresstruct"
#define MATSUBMATRIX       "submatrix"

#ifdef PETSC_USE_MATFWK
#define MATFWK           "matfwk"
#endif

/*E
    MatSolverPackage - String with the name of a PETSc matrix solver type. 

    For example: "petsc" indicates what PETSc provides, "superlu" indicates either 
       SuperLU or SuperLU_Dist etc.


   Level: beginner

.seealso: MatGetFactor(), Mat, MatSetType(), MatType
E*/
#define MatSolverPackage char*
#define MAT_SOLVER_SPOOLES      "spooles"
#define MAT_SOLVER_SUPERLU      "superlu"
#define MAT_SOLVER_SUPERLU_DIST "superlu_dist"
#define MAT_SOLVER_UMFPACK      "umfpack"
#define MAT_SOLVER_ESSL         "essl"
#define MAT_SOLVER_LUSOL        "lusol"
#define MAT_SOLVER_MUMPS        "mumps"
#define MAT_SOLVER_PASTIX       "pastix"
#define MAT_SOLVER_DSCPACK      "dscpack"
#define MAT_SOLVER_MATLAB       "matlab"
#define MAT_SOLVER_PETSC        "petsc"
#define MAT_SOLVER_PLAPACK      "plapack"
#define MAT_SOLVER_BAS          "bas"

/*E
    MatFactorType - indicates what type of factorization is requested

    Level: beginner

   Any additions/changes here MUST also be made in include/finclude/petscmat.h

.seealso: MatSolverPackage, MatGetFactor()
E*/
typedef enum {MAT_FACTOR_NONE, MAT_FACTOR_LU, MAT_FACTOR_CHOLESKY, MAT_FACTOR_ILU, MAT_FACTOR_ICC,MAT_FACTOR_ILUDT} MatFactorType;
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetFactor(Mat,const MatSolverPackage,MatFactorType,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetFactorAvailable(Mat,const MatSolverPackage,MatFactorType,PetscTruth*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatFactorGetSolverPackage(Mat,const MatSolverPackage*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetFactorType(Mat,MatFactorType*);

/* Logging support */
#define    MAT_FILE_COOKIE 1211216    /* used to indicate matrices in binary files */
extern PetscCookie PETSCMAT_DLLEXPORT MAT_COOKIE;
extern PetscCookie PETSCMAT_DLLEXPORT MAT_FDCOLORING_COOKIE;
extern PetscCookie PETSCMAT_DLLEXPORT MAT_PARTITIONING_COOKIE;
extern PetscCookie PETSCMAT_DLLEXPORT MAT_NULLSPACE_COOKIE;
extern PetscCookie PETSCMAT_DLLEXPORT MATMFFD_COOKIE;

/*E
    MatReuse - Indicates if matrices obtained from a previous call to MatGetSubMatrices()
     or MatGetSubMatrix() are to be reused to store the new matrix values.

    Level: beginner

   Any additions/changes here MUST also be made in include/finclude/petscmat.h

.seealso: MatGetSubMatrices(), MatGetSubMatrix(), MatDestroyMatrices(), MatConvert()
E*/
typedef enum {MAT_INITIAL_MATRIX,MAT_REUSE_MATRIX,MAT_IGNORE_MATRIX} MatReuse;

/*E
    MatGetSubMatrixOption - Indicates if matrices obtained from a call to MatGetSubMatrices()
     include the matrix values. Currently it is only used by MatGetSeqNonzerostructure().

    Level: beginner

.seealso: MatGetSeqNonzerostructure()
E*/
typedef enum {MAT_DO_NOT_GET_VALUES,MAT_GET_VALUES} MatGetSubMatrixOption;

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatInitializePackage(const char[]);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate(MPI_Comm,Mat*);
PetscPolymorphicFunction(MatCreate,(MPI_Comm comm),(comm,&A),Mat,A)
PetscPolymorphicFunction(MatCreate,(),(PETSC_COMM_WORLD,&A),Mat,A)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetSizes(Mat,PetscInt,PetscInt,PetscInt,PetscInt);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetType(Mat,const MatType);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetFromOptions(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetUpPreallocation(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatRegisterAll(const char[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatRegister(const char[],const char[],const char[],PetscErrorCode(*)(Mat));

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetOptionsPrefix(Mat,const char[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatAppendOptionsPrefix(Mat,const char[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetOptionsPrefix(Mat,const char*[]);

/*MC
   MatRegisterDynamic - Adds a new matrix type

   Synopsis:
   PetscErrorCode MatRegisterDynamic(const char *name,const char *path,const char *name_create,PetscErrorCode (*routine_create)(Mat))

   Not Collective

   Input Parameters:
+  name - name of a new user-defined matrix type
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Notes:
   MatRegisterDynamic() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   MatRegisterDynamic("my_mat",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyMatCreate",MyMatCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     MatSetType(Mat,"my_mat")
   or at runtime via the option
$     -mat_type my_mat

   Level: advanced

   Notes: ${PETSC_ARCH} occuring in pathname will be replaced with appropriate values.
         If your function is not being put into a shared library then use VecRegister() instead

.keywords: Mat, register

.seealso: MatRegisterAll(), MatRegisterDestroy()

M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define MatRegisterDynamic(a,b,c,d) MatRegister(a,b,c,0)
#else
#define MatRegisterDynamic(a,b,c,d) MatRegister(a,b,c,d)
#endif

extern PetscTruth MatRegisterAllCalled;
extern PetscFList MatList;
extern PetscFList MatColoringList;
extern PetscFList MatPartitioningList;

/*E
    MatStructure - Indicates if the matrix has the same nonzero structure

    Level: beginner

   Any additions/changes here MUST also be made in include/finclude/petscmat.h

.seealso: MatCopy(), KSPSetOperators(), PCSetOperators()
E*/
typedef enum {SAME_NONZERO_PATTERN,DIFFERENT_NONZERO_PATTERN,SAME_PRECONDITIONER,SUBSET_NONZERO_PATTERN} MatStructure;

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateSeqDense(MPI_Comm,PetscInt,PetscInt,PetscScalar[],Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateMPIDense(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscScalar[],Mat*); 
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateSeqAIJ(MPI_Comm,PetscInt,PetscInt,PetscInt,const PetscInt[],Mat*);
PetscPolymorphicFunction(MatCreateSeqAIJ,(PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[]),(PETSC_COMM_SELF,m,n,nz,nnz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateSeqAIJ,(PetscInt m,PetscInt n,PetscInt nz),(PETSC_COMM_SELF,m,n,nz,PETSC_NULL,&A),Mat,A)
PetscPolymorphicFunction(MatCreateSeqAIJ,(PetscInt m,PetscInt n,const PetscInt nnz[]),(PETSC_COMM_SELF,m,n,0,nnz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateSeqAIJ,(PetscInt m,PetscInt n),(PETSC_COMM_SELF,m,n,0,PETSC_NULL,&A),Mat,A)
PetscPolymorphicSubroutine(MatCreateSeqAIJ,(PetscInt m,PetscInt n,PetscInt nz,Mat *A),(PETSC_COMM_SELF,m,n,nz,PETSC_NULL,A))
PetscPolymorphicSubroutine(MatCreateSeqAIJ,(PetscInt m,PetscInt n,const PetscInt nnz[],Mat *A),(PETSC_COMM_SELF,m,n,0,nnz,A))
PetscPolymorphicSubroutine(MatCreateSeqAIJ,(PetscInt m,PetscInt n,Mat *A),(PETSC_COMM_SELF,m,n,0,PETSC_NULL,A))
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateMPIAIJ(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Mat*); 
PetscPolymorphicFunction(MatCreateMPIAIJ,(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt nz,const PetscInt nnz[],PetscInt onz,const PetscInt onnz[]),(comm,m,n,M,N,nz,nnz,onz,onnz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateMPIAIJ,(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt nz,PetscInt nnz),(comm,m,n,M,N,nz,PETSC_NULL,nnz,PETSC_NULL,&A),Mat,A)
PetscPolymorphicFunction(MatCreateMPIAIJ,(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,const PetscInt nnz[],const PetscInt onz[]),(comm,m,n,M,N,0,nnz,0,onz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateMPIAIJ,(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N),(comm,m,n,M,N,0,PETSC_NULL,0,PETSC_NULL,&A),Mat,A)
PetscPolymorphicSubroutine(MatCreateMPIAIJ,(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt nz,PetscInt nnz,Mat *A),(comm,m,n,M,N,nz,PETSC_NULL,nnz,PETSC_NULL,A))
PetscPolymorphicSubroutine(MatCreateMPIAIJ,(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,const PetscInt nnz[],const PetscInt onz[],Mat *A),(comm,m,n,M,N,0,nnz,0,onz,A))
PetscPolymorphicSubroutine(MatCreateMPIAIJ,(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,Mat *A),(comm,m,n,M,N,0,PETSC_NULL,0,PETSC_NULL,A))
PetscPolymorphicFunction(MatCreateMPIAIJ,(PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt nz,const PetscInt nnz[],PetscInt onz,const PetscInt onnz[]),(PETSC_COMM_WORLD,m,n,M,N,nz,nnz,onz,onnz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateMPIAIJ,(PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt nz,PetscInt nnz),(PETSC_COMM_WORLD,m,n,M,N,nz,PETSC_NULL,nnz,PETSC_NULL,&A),Mat,A)
PetscPolymorphicFunction(MatCreateMPIAIJ,(PetscInt m,PetscInt n,PetscInt M,PetscInt N,const PetscInt nnz[],const PetscInt onz[]),(PETSC_COMM_WORLD,m,n,M,N,0,nnz,0,onz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateMPIAIJ,(PetscInt m,PetscInt n,PetscInt M,PetscInt N),(PETSC_COMM_WORLD,m,n,M,N,0,PETSC_NULL,0,PETSC_NULL,&A),Mat,A)
PetscPolymorphicSubroutine(MatCreateMPIAIJ,(PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt nz,PetscInt nnz,Mat *A),(PETSC_COMM_WORLD,m,n,M,N,nz,PETSC_NULL,nnz,PETSC_NULL,A))
PetscPolymorphicSubroutine(MatCreateMPIAIJ,(PetscInt m,PetscInt n,PetscInt M,PetscInt N,const PetscInt nnz[],const PetscInt onz[],Mat *A),(PETSC_COMM_WORLD,m,n,M,N,0,nnz,0,onz,A))
PetscPolymorphicSubroutine(MatCreateMPIAIJ,(PetscInt m,PetscInt n,PetscInt M,PetscInt N,Mat *A),(PETSC_COMM_WORLD,m,n,M,N,0,PETSC_NULL,0,PETSC_NULL,A))
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateMPIAIJWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[],Mat *);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateMPIAIJWithSplitArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt[],PetscInt[],PetscScalar[],PetscInt[],PetscInt[],PetscScalar[],Mat*);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateSeqBAIJ(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],Mat*); 
PetscPolymorphicFunction(MatCreateSeqBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[]),(PETSC_COMM_SELF,bs,m,n,nz,nnz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateSeqBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt nz),(PETSC_COMM_SELF,bs,m,n,nz,PETSC_NULL,&A),Mat,A)
PetscPolymorphicFunction(MatCreateSeqBAIJ,(PetscInt bs,PetscInt m,PetscInt n,const PetscInt nnz[]),(PETSC_COMM_SELF,bs,m,n,0,nnz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateSeqBAIJ,(PetscInt bs,PetscInt m,PetscInt n),(PETSC_COMM_SELF,bs,m,n,0,PETSC_NULL,&A),Mat,A)
PetscPolymorphicSubroutine(MatCreateSeqBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt nz,Mat *A),(PETSC_COMM_SELF,bs,m,n,nz,PETSC_NULL,A))
PetscPolymorphicSubroutine(MatCreateSeqBAIJ,(PetscInt bs,PetscInt m,PetscInt n,const PetscInt nnz[],Mat *A),(PETSC_COMM_SELF,bs,m,n,0,nnz,A))
PetscPolymorphicSubroutine(MatCreateSeqBAIJ,(PetscInt bs,PetscInt m,PetscInt n,Mat *A),(PETSC_COMM_SELF,bs,m,n,0,PETSC_NULL,A))
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateMPIBAIJ(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Mat*);
PetscPolymorphicFunction(MatCreateMPIBAIJ,(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt nz,const PetscInt nnz[],PetscInt onz,const PetscInt onnz[]),(comm,bs,m,n,M,N,nz,nnz,onz,onnz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateMPIBAIJ,(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt nz,PetscInt nnz),(comm,bs,m,n,M,N,nz,PETSC_NULL,nnz,PETSC_NULL,&A),Mat,A)
PetscPolymorphicFunction(MatCreateMPIBAIJ,(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,const PetscInt nnz[],const PetscInt onz[]),(comm,bs,m,n,M,N,0,nnz,0,onz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateMPIBAIJ,(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N),(comm,bs,m,n,M,N,0,PETSC_NULL,0,PETSC_NULL,&A),Mat,A)
PetscPolymorphicSubroutine(MatCreateMPIBAIJ,(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt nz,PetscInt nnz,Mat *A),(comm,bs,m,n,M,N,nz,PETSC_NULL,nnz,PETSC_NULL,A))
PetscPolymorphicSubroutine(MatCreateMPIBAIJ,(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,const PetscInt nnz[],const PetscInt onz[],Mat *A),(comm,bs,m,n,M,N,0,nnz,0,onz,A))
PetscPolymorphicSubroutine(MatCreateMPIBAIJ,(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,Mat *A),(comm,bs,m,n,M,N,0,PETSC_NULL,0,PETSC_NULL,A))
PetscPolymorphicFunction(MatCreateMPIBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt nz,const PetscInt nnz[],PetscInt onz,const PetscInt onnz[]),(PETSC_COMM_WORLD,bs,m,n,M,N,nz,nnz,onz,onnz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateMPIBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt nz,PetscInt nnz),(PETSC_COMM_WORLD,bs,m,n,M,N,nz,PETSC_NULL,nnz,PETSC_NULL,&A),Mat,A)
PetscPolymorphicFunction(MatCreateMPIBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,const PetscInt nnz[],const PetscInt onz[]),(PETSC_COMM_WORLD,bs,m,n,M,N,0,nnz,0,onz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateMPIBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N),(PETSC_COMM_WORLD,bs,m,n,M,N,0,PETSC_NULL,0,PETSC_NULL,&A),Mat,A)
PetscPolymorphicSubroutine(MatCreateMPIBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt nz,PetscInt nnz,Mat *A),(PETSC_COMM_WORLD,bs,m,n,M,N,nz,PETSC_NULL,nnz,PETSC_NULL,A))
PetscPolymorphicSubroutine(MatCreateMPIBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,const PetscInt nnz[],const PetscInt onz[],Mat *A),(PETSC_COMM_WORLD,bs,m,n,M,N,0,nnz,0,onz,A))
PetscPolymorphicSubroutine(MatCreateMPIBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,Mat *A),(PETSC_COMM_WORLD,bs,m,n,M,N,0,PETSC_NULL,0,PETSC_NULL,A))
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateMPIAdj(MPI_Comm,PetscInt,PetscInt,PetscInt[],PetscInt[],PetscInt[],Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateSeqSBAIJ(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],Mat*); 
PetscPolymorphicFunction(MatCreateSeqSBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[]),(PETSC_COMM_SELF,bs,m,n,nz,nnz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateSeqSBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt nz),(PETSC_COMM_SELF,bs,m,n,nz,PETSC_NULL,&A),Mat,A)
PetscPolymorphicFunction(MatCreateSeqSBAIJ,(PetscInt bs,PetscInt m,PetscInt n,const PetscInt nnz[]),(PETSC_COMM_SELF,bs,m,n,0,nnz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateSeqSBAIJ,(PetscInt bs,PetscInt m,PetscInt n),(PETSC_COMM_SELF,bs,m,n,0,PETSC_NULL,&A),Mat,A)
PetscPolymorphicSubroutine(MatCreateSeqSBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt nz,Mat *A),(PETSC_COMM_SELF,bs,m,n,nz,PETSC_NULL,A))
PetscPolymorphicSubroutine(MatCreateSeqSBAIJ,(PetscInt bs,PetscInt m,PetscInt n,const PetscInt nnz[],Mat *A),(PETSC_COMM_SELF,bs,m,n,0,nnz,A))
PetscPolymorphicSubroutine(MatCreateSeqSBAIJ,(PetscInt bs,PetscInt m,PetscInt n,Mat *A),(PETSC_COMM_SELF,bs,m,n,0,PETSC_NULL,A))
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateMPISBAIJ(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Mat*);
PetscPolymorphicFunction(MatCreateMPISBAIJ,(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt nz,const PetscInt nnz[],PetscInt onz,const PetscInt onnz[]),(comm,bs,m,n,M,N,nz,nnz,onz,onnz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateMPISBAIJ,(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt nz,PetscInt nnz),(comm,bs,m,n,M,N,nz,PETSC_NULL,nnz,PETSC_NULL,&A),Mat,A)
PetscPolymorphicFunction(MatCreateMPISBAIJ,(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,const PetscInt nnz[],const PetscInt onz[]),(comm,bs,m,n,M,N,0,nnz,0,onz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateMPISBAIJ,(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N),(comm,bs,m,n,M,N,0,PETSC_NULL,0,PETSC_NULL,&A),Mat,A)
PetscPolymorphicSubroutine(MatCreateMPISBAIJ,(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt nz,PetscInt nnz,Mat *A),(comm,bs,m,n,M,N,nz,PETSC_NULL,nnz,PETSC_NULL,A))
PetscPolymorphicSubroutine(MatCreateMPISBAIJ,(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,const PetscInt nnz[],const PetscInt onz[],Mat *A),(comm,bs,m,n,M,N,0,nnz,0,onz,A))
PetscPolymorphicSubroutine(MatCreateMPISBAIJ,(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,Mat *A),(comm,bs,m,n,M,N,0,PETSC_NULL,0,PETSC_NULL,A))
PetscPolymorphicFunction(MatCreateMPISBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt nz,const PetscInt nnz[],PetscInt onz,const PetscInt onnz[]),(PETSC_COMM_WORLD,bs,m,n,M,N,nz,nnz,onz,onnz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateMPISBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt nz,PetscInt nnz),(PETSC_COMM_WORLD,bs,m,n,M,N,nz,PETSC_NULL,nnz,PETSC_NULL,&A),Mat,A)
PetscPolymorphicFunction(MatCreateMPISBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,const PetscInt nnz[],const PetscInt onz[]),(PETSC_COMM_WORLD,bs,m,n,M,N,0,nnz,0,onz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateMPISBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N),(PETSC_COMM_WORLD,bs,m,n,M,N,0,PETSC_NULL,0,PETSC_NULL,&A),Mat,A)
PetscPolymorphicSubroutine(MatCreateMPISBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt nz,PetscInt nnz,Mat *A),(PETSC_COMM_WORLD,bs,m,n,M,N,nz,PETSC_NULL,nnz,PETSC_NULL,A))
PetscPolymorphicSubroutine(MatCreateMPISBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,const PetscInt nnz[],const PetscInt onz[],Mat *A),(PETSC_COMM_WORLD,bs,m,n,M,N,0,nnz,0,onz,A))
PetscPolymorphicSubroutine(MatCreateMPISBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,Mat *A),(PETSC_COMM_WORLD,bs,m,n,M,N,0,PETSC_NULL,0,PETSC_NULL,A))
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateShell(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,void *,Mat*);
PetscPolymorphicFunction(MatCreateShell,(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,void *ctx),(comm,m,n,M,N,ctx,&A),Mat,A)
PetscPolymorphicFunction(MatCreateShell,(PetscInt m,PetscInt n,PetscInt M,PetscInt N,void *ctx),(PETSC_COMM_WORLD,m,n,M,N,ctx,&A),Mat,A)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateAdic(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,void (*)(void),Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateNormal(Mat,Mat*);
PetscPolymorphicFunction(MatCreateNormal,(Mat mat),(mat,&A),Mat,A)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateLRC(Mat,Mat,Mat,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateIS(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,ISLocalToGlobalMapping,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateSeqCRL(MPI_Comm,PetscInt,PetscInt,PetscInt,const PetscInt[],Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateMPICRL(MPI_Comm,PetscInt,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateScatter(MPI_Comm,VecScatter,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatScatterSetVecScatter(Mat,VecScatter);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatScatterGetVecScatter(Mat,VecScatter*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateBlockMat(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCompositeAddMat(Mat,Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCompositeMerge(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateComposite(MPI_Comm,PetscInt,const Mat*,Mat*);
typedef enum {MAT_COMPOSITE_ADDITIVE,MAT_COMPOSITE_MULTIPLICATIVE} MatCompositeType;
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCompositeSetType(Mat,MatCompositeType);

#ifdef PETSC_USE_MATFWK
typedef enum {MATFWK_ADDITIVE, MATFWK_MULTIPLICATIVE} MatFwkType;
typedef enum {MATFWK_MERGED, MATFWK_SPLIT} MatFwkMode;

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateFwk(MPI_Comm,Mat *);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatFwkSetType(Mat,MatFwkType);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatFwkGetType(Mat,MatFwkType*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatFwkSetMode(Mat,MatFwkMode);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatFwkGetMode(Mat,MatFwkMode*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatFwkAddBlock(Mat, IS, IS, Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatFwkBlockSetMat(Mat, Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatFwkBlockGetMat(Mat, Mat*);
#endif

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateSeqFFTW(MPI_Comm,PetscInt,const PetscInt[],Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateTranspose(Mat,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateSubMatrix(Mat,IS,IS,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSubMatrixUpdate(Mat,Mat,IS,IS);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreatePython(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,const char[],Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPythonSetType(Mat,const char[]);


EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetUp(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatDestroy(Mat);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatConjugate(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatRealPart(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatImaginaryPart(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetDiagonalBlock(Mat,PetscTruth*,MatReuse,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetTrace(Mat,PetscScalar*);

/* ------------------------------------------------------------*/
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetValues(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetValuesBlocked(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetValuesRow(Mat,PetscInt,const PetscScalar[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetValuesRowLocal(Mat,PetscInt,const PetscScalar[]);

/*S
     MatStencil - Data structure (C struct) for storing information about a single row or
        column of a matrix as index on an associated grid.

   Level: beginner

  Concepts: matrix; linear operator

.seealso:  MatSetValuesStencil(), MatSetStencil(), MatSetValuesBlockStencil()
S*/
typedef struct {
  PetscInt k,j,i,c;
} MatStencil;

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetValuesStencil(Mat,PetscInt,const MatStencil[],PetscInt,const MatStencil[],const PetscScalar[],InsertMode);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetValuesBlockedStencil(Mat,PetscInt,const MatStencil[],PetscInt,const MatStencil[],const PetscScalar[],InsertMode);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetStencil(Mat,PetscInt,const PetscInt[],const PetscInt[],PetscInt);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetColoring(Mat,ISColoring);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetValuesAdic(Mat,void*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetValuesAdifor(Mat,PetscInt,void*);

/*E
    MatAssemblyType - Indicates if the matrix is now to be used, or if you plan 
     to continue to add values to it

    Level: beginner

.seealso: MatAssemblyBegin(), MatAssemblyEnd()
E*/
typedef enum {MAT_FLUSH_ASSEMBLY=1,MAT_FINAL_ASSEMBLY=0} MatAssemblyType;
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatAssemblyBegin(Mat,MatAssemblyType);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatAssemblyEnd(Mat,MatAssemblyType);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatAssembled(Mat,PetscTruth*);



/*E
    MatOption - Options that may be set for a matrix and its behavior or storage

    Level: beginner

   Any additions/changes here MUST also be made in include/finclude/petscmat.h

.seealso: MatSetOption()
E*/
typedef enum {MAT_ROW_ORIENTED,MAT_NEW_NONZERO_LOCATIONS,      
              MAT_SYMMETRIC,          
              MAT_STRUCTURALLY_SYMMETRIC,
              MAT_NEW_DIAGONALS,MAT_IGNORE_OFF_PROC_ENTRIES,
              MAT_NEW_NONZERO_LOCATION_ERR,
              MAT_NEW_NONZERO_ALLOCATION_ERR,MAT_USE_HASH_TABLE,
              MAT_KEEP_NONZERO_PATTERN,MAT_IGNORE_ZERO_ENTRIES,
              MAT_USE_INODES,
              MAT_HERMITIAN,
              MAT_SYMMETRY_ETERNAL,
              MAT_USE_COMPRESSEDROW,
              MAT_IGNORE_LOWER_TRIANGULAR,MAT_ERROR_LOWER_TRIANGULAR,
              MAT_GETROW_UPPERTRIANGULAR,MAT_UNUSED_NONZERO_LOCATION_ERR,
              NUM_MAT_OPTIONS} MatOption;
extern const char *MatOptions[];
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetOption(Mat,MatOption,PetscTruth);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetType(Mat,const MatType*);
PetscPolymorphicFunction(MatGetType,(Mat mat),(mat,&t),const MatType,t)

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetValues(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],PetscScalar[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetRow(Mat,PetscInt,PetscInt *,const PetscInt *[],const PetscScalar*[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatRestoreRow(Mat,PetscInt,PetscInt *,const PetscInt *[],const PetscScalar*[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetRowUpperTriangular(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatRestoreRowUpperTriangular(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetColumn(Mat,PetscInt,PetscInt *,const PetscInt *[],const PetscScalar*[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatRestoreColumn(Mat,PetscInt,PetscInt *,const PetscInt *[],const PetscScalar*[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetColumnVector(Mat,Vec,PetscInt);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetArray(Mat,PetscScalar *[]);
PetscPolymorphicFunction(MatGetArray,(Mat mat),(mat,&a),PetscScalar*,a)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatRestoreArray(Mat,PetscScalar *[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetBlockSize(Mat,PetscInt *);
PetscPolymorphicFunction(MatGetBlockSize,(Mat mat),(mat,&a),PetscInt,a)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetBlockSize(Mat,PetscInt);


EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMult(Mat,Vec,Vec);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMultDiagonalBlock(Mat,Vec,Vec);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMultAdd(Mat,Vec,Vec,Vec);
PetscPolymorphicSubroutine(MatMultAdd,(Mat A,Vec x,Vec y),(A,x,y,y))
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMultTranspose(Mat,Vec,Vec);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMultHermitianTranspose(Mat,Vec,Vec);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatIsTranspose(Mat,Mat,PetscReal,PetscTruth*);
PetscPolymorphicFunction(MatIsTranspose,(Mat A,Mat B,PetscReal tol),(A,B,tol,&t),PetscTruth,t)
PetscPolymorphicFunction(MatIsTranspose,(Mat A,Mat B),(A,B,0,&t),PetscTruth,t)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatIsHermitianTranspose(Mat,Mat,PetscReal,PetscTruth*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMultTransposeAdd(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMultHermitianTransposeAdd(Mat,Vec,Vec,Vec);
PetscPolymorphicSubroutine(MatMultTransposeAdd,(Mat A,Vec x,Vec y),(A,x,y,y))
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMultConstrained(Mat,Vec,Vec);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMultTransposeConstrained(Mat,Vec,Vec);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMatSolve(Mat,Mat,Mat);

/*E
    MatDuplicateOption - Indicates if a duplicated sparse matrix should have
  its numerical values copied over or just its nonzero structure.

    Level: beginner

   Any additions/changes here MUST also be made in include/finclude/petscmat.h

$   MAT_SHARE_NONZERO_PATTERN - the i and j arrays in the new matrix will be shared with the original matrix
$                               this also triggers the MAT_DO_NOT_COPY_VALUES option. This is used when you
$                               have several matrices with the same nonzero pattern.

.seealso: MatDuplicate()
E*/
typedef enum {MAT_DO_NOT_COPY_VALUES,MAT_COPY_VALUES,MAT_SHARE_NONZERO_PATTERN} MatDuplicateOption;

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatConvert(Mat,const MatType,MatReuse,Mat*);
PetscPolymorphicFunction(MatConvert,(Mat A,const MatType t),(A,t,MAT_INITIAL_MATRIX,&a),Mat,a)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatDuplicate(Mat,MatDuplicateOption,Mat*);
PetscPolymorphicFunction(MatDuplicate,(Mat A,MatDuplicateOption o),(A,o,&a),Mat,a)
PetscPolymorphicFunction(MatDuplicate,(Mat A),(A,MAT_COPY_VALUES,&a),Mat,a)


EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCopy(Mat,Mat,MatStructure);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatView(Mat,PetscViewer);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatIsSymmetric(Mat,PetscReal,PetscTruth*);
PetscPolymorphicFunction(MatIsSymmetric,(Mat A,PetscReal tol),(A,tol,&t),PetscTruth,t)
PetscPolymorphicFunction(MatIsSymmetric,(Mat A),(A,0,&t),PetscTruth,t)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatIsStructurallySymmetric(Mat,PetscTruth*);
PetscPolymorphicFunction(MatIsStructurallySymmetric,(Mat A),(A,&t),PetscTruth,t)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatIsHermitian(Mat,PetscReal,PetscTruth*);
PetscPolymorphicFunction(MatIsHermitian,(Mat A),(A,0,&t),PetscTruth,t)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatIsSymmetricKnown(Mat,PetscTruth*,PetscTruth*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatIsHermitianKnown(Mat,PetscTruth*,PetscTruth*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMissingDiagonal(Mat,PetscTruth *,PetscInt *);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatLoad(PetscViewer,const MatType,Mat*);
PetscPolymorphicFunction(MatLoad,(PetscViewer v,const MatType t),(v,t,&a),Mat,a)

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetRowIJ(Mat,PetscInt,PetscTruth,PetscTruth,PetscInt*,PetscInt *[],PetscInt *[],PetscTruth *);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatRestoreRowIJ(Mat,PetscInt,PetscTruth,PetscTruth,PetscInt *,PetscInt *[],PetscInt *[],PetscTruth *);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetColumnIJ(Mat,PetscInt,PetscTruth,PetscTruth,PetscInt*,PetscInt *[],PetscInt *[],PetscTruth *);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatRestoreColumnIJ(Mat,PetscInt,PetscTruth,PetscTruth,PetscInt *,PetscInt *[],PetscInt *[],PetscTruth *);

/*S
     MatInfo - Context of matrix information, used with MatGetInfo()

   In Fortran this is simply a double precision array of dimension MAT_INFO_SIZE

   Level: intermediate

  Concepts: matrix^nonzero information

.seealso:  MatGetInfo(), MatInfoType
S*/
typedef struct {
  PetscLogDouble block_size;                         /* block size */
  PetscLogDouble nz_allocated,nz_used,nz_unneeded;   /* number of nonzeros */
  PetscLogDouble memory;                             /* memory allocated */
  PetscLogDouble assemblies;                         /* number of matrix assemblies called */
  PetscLogDouble mallocs;                            /* number of mallocs during MatSetValues() */
  PetscLogDouble fill_ratio_given,fill_ratio_needed; /* fill ratio for LU/ILU */
  PetscLogDouble factor_mallocs;                     /* number of mallocs during factorization */
} MatInfo;

/*E
    MatInfoType - Indicates if you want information about the local part of the matrix,
     the entire parallel matrix or the maximum over all the local parts.

    Level: beginner

   Any additions/changes here MUST also be made in include/finclude/petscmat.h

.seealso: MatGetInfo(), MatInfo
E*/
typedef enum {MAT_LOCAL=1,MAT_GLOBAL_MAX=2,MAT_GLOBAL_SUM=3} MatInfoType;
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetInfo(Mat,MatInfoType,MatInfo*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatValid(Mat,PetscTruth*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetDiagonal(Mat,Vec);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetRowMax(Mat,Vec,PetscInt[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetRowMin(Mat,Vec,PetscInt[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetRowMaxAbs(Mat,Vec,PetscInt[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetRowMinAbs(Mat,Vec,PetscInt[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetRowSum(Mat,Vec);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatTranspose(Mat,MatReuse,Mat*);
PetscPolymorphicFunction(MatTranspose,(Mat A),(A,MAT_INITIAL_MATRIX,&t),Mat,t)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatHermitianTranspose(Mat,MatReuse,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPermute(Mat,IS,IS,Mat *);
PetscPolymorphicFunction(MatPermute,(Mat A,IS is1,IS is2),(A,is1,is2,&t),Mat,t)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPermuteSparsify(Mat,PetscInt,PetscReal,PetscReal,IS,IS,Mat *);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatDiagonalScale(Mat,Vec,Vec);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatDiagonalSet(Mat,Vec,InsertMode);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatEqual(Mat,Mat,PetscTruth*);
PetscPolymorphicFunction(MatEqual,(Mat A,Mat B),(A,B,&t),PetscTruth,t)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMultEqual(Mat,Mat,PetscInt,PetscTruth*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMultAddEqual(Mat,Mat,PetscInt,PetscTruth*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMultTransposeEqual(Mat,Mat,PetscInt,PetscTruth*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMultTransposeAddEqual(Mat,Mat,PetscInt,PetscTruth*);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatNorm(Mat,NormType,PetscReal *);
PetscPolymorphicFunction(MatNorm,(Mat A,NormType t),(A,t,&n),PetscReal,n)
EXTERN PetscErrorCode MatGetColumnNorms(Mat,NormType,PetscReal *);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatZeroEntries(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatZeroRows(Mat,PetscInt,const PetscInt [],PetscScalar);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatZeroRowsIS(Mat,IS,PetscScalar);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatZeroColumns(Mat,PetscInt,const PetscInt [],const PetscScalar*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatZeroColumnsIS(Mat,IS,const PetscScalar*);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatUseScaledForm(Mat,PetscTruth);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatScaleSystem(Mat,Vec,Vec);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatUnScaleSystem(Mat,Vec,Vec);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetSize(Mat,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetLocalSize(Mat,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetOwnershipRange(Mat,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetOwnershipRanges(Mat,const PetscInt**);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetOwnershipRangeColumn(Mat,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetOwnershipRangesColumn(Mat,const PetscInt**);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetSubMatrices(Mat,PetscInt,const IS[],const IS[],MatReuse,Mat *[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatDestroyMatrices(PetscInt,Mat *[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetSubMatrix(Mat,IS,IS,MatReuse,Mat *);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetSeqNonzeroStructure(Mat,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatDestroySeqNonzeroStructure(Mat*); 

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMerge(MPI_Comm,Mat,PetscInt,MatReuse,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMerge_SeqsToMPI(MPI_Comm,Mat,PetscInt,PetscInt,MatReuse,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMerge_SeqsToMPISymbolic(MPI_Comm,Mat,PetscInt,PetscInt,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMerge_SeqsToMPINumeric(Mat,Mat); 
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatDestroy_MPIAIJ_SeqsToMPI(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetLocalMat(Mat,MatReuse,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetLocalMatCondensed(Mat,MatReuse,IS*,IS*,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetBrowsOfAcols(Mat,Mat,MatReuse,IS*,IS*,PetscInt*,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetBrowsOfAoCols(Mat,Mat,MatReuse,PetscInt**,PetscInt**,MatScalar**,Mat*);
#if defined (PETSC_USE_CTABLE)
#include "petscctable.h"
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetCommunicationStructs(Mat, Vec *, PetscTable *, VecScatter *);
#else
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetCommunicationStructs(Mat, Vec *, PetscInt *[], VecScatter *);
#endif
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetGhosts(Mat, PetscInt *,const PetscInt *[]);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatIncreaseOverlap(Mat,PetscInt,IS[],PetscInt);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMatMult(Mat,Mat,MatReuse,PetscReal,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMatMultSymbolic(Mat,Mat,PetscReal,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMatMultNumeric(Mat,Mat,Mat);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPtAP(Mat,Mat,MatReuse,PetscReal,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPtAPSymbolic(Mat,Mat,PetscReal,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPtAPNumeric(Mat,Mat,Mat);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMatMultTranspose(Mat,Mat,MatReuse,PetscReal,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMatMultTransposeSymbolic(Mat,Mat,PetscReal,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMatMultTransposeNumeric(Mat,Mat,Mat);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatAXPY(Mat,PetscScalar,Mat,MatStructure);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatAYPX(Mat,PetscScalar,Mat,MatStructure);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatScale(Mat,PetscScalar);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatShift(Mat,PetscScalar);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetLocalToGlobalMapping(Mat,ISLocalToGlobalMapping);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetLocalToGlobalMappingBlock(Mat,ISLocalToGlobalMapping);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatZeroRowsLocal(Mat,PetscInt,const PetscInt [],PetscScalar);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatZeroRowsLocalIS(Mat,IS,PetscScalar);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetValuesLocal(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetValuesBlockedLocal(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatStashSetInitialSize(Mat,PetscInt,PetscInt);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatStashGetInfo(Mat,PetscInt*,PetscInt*,PetscInt*,PetscInt*);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatInterpolate(Mat,Vec,Vec);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatInterpolateAdd(Mat,Vec,Vec,Vec);
PetscPolymorphicSubroutine(MatInterpolateAdd,(Mat A,Vec x,Vec y),(A,x,y,y))
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatRestrict(Mat,Vec,Vec);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetVecs(Mat,Vec*,Vec*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetRedundantMatrix(Mat,PetscInt,MPI_Comm,PetscInt,MatReuse,Mat*);

/*MC
   MatSetValue - Set a single entry into a matrix.

   Not collective

   Input Parameters:
+  m - the matrix
.  row - the row location of the entry
.  col - the column location of the entry
.  value - the value to insert
-  mode - either INSERT_VALUES or ADD_VALUES

   Notes: 
   For efficiency one should use MatSetValues() and set several or many
   values simultaneously if possible.

   Level: beginner

.seealso: MatSetValues(), MatSetValueLocal()
M*/
PETSC_STATIC_INLINE PetscErrorCode MatSetValue(Mat v,PetscInt i,PetscInt j,PetscScalar va,InsertMode mode) {return MatSetValues(v,1,&i,1,&j,&va,mode);}

PETSC_STATIC_INLINE PetscErrorCode MatGetValue(Mat v,PetscInt i,PetscInt j,PetscScalar *va) {return MatGetValues(v,1,&i,1,&j,va);}

PETSC_STATIC_INLINE PetscErrorCode MatSetValueLocal(Mat v,PetscInt i,PetscInt j,PetscScalar va,InsertMode mode) {return MatSetValuesLocal(v,1,&i,1,&j,&va,mode);}

/*MC
   MatPreallocateInitialize - Begins the block of code that will count the number of nonzeros per
       row in a matrix providing the data that one can use to correctly preallocate the matrix.

   Synopsis:
   PetscErrorCode MatPreallocateInitialize(MPI_Comm comm, PetscInt nrows, PetscInt ncols, PetscInt *dnz, PetscInt *onz)

   Collective on MPI_Comm

   Input Parameters:
+  comm - the communicator that will share the eventually allocated matrix
.  nrows - the number of LOCAL rows in the matrix
-  ncols - the number of LOCAL columns in the matrix

   Output Parameters:
+  dnz - the array that will be passed to the matrix preallocation routines
-  ozn - the other array passed to the matrix preallocation routines


   Level: intermediate

   Notes:
   See the chapter in the users manual on performance for more details

   Do not malloc or free dnz and onz, that is handled internally by these routines

   Use MatPreallocateInitializeSymmetric() for symmetric matrices (MPISBAIJ matrices)

   This is a MACRO not a function because it has a leading { that is closed by PetscPreallocateFinalize().

  Concepts: preallocation^Matrix

.seealso: MatPreallocateFinalize(), MatPreallocateSet(), MatPreallocateSymmetricSet(), MatPreallocateSetLocal(),
          MatPreallocateInitializeSymmetric(), MatPreallocateSymmetricSetLocal()
M*/
#define MatPreallocateInitialize(comm,nrows,ncols,dnz,onz) 0; \
{ \
  PetscErrorCode _4_ierr; PetscInt __nrows = (nrows),__ctmp = (ncols),__rstart,__start,__end; \
  _4_ierr = PetscMalloc2(__nrows,PetscInt,&dnz,__nrows,PetscInt,&onz);CHKERRQ(_4_ierr); \
  _4_ierr = PetscMemzero(dnz,__nrows*sizeof(PetscInt));CHKERRQ(_4_ierr);\
  _4_ierr = PetscMemzero(onz,__nrows*sizeof(PetscInt));CHKERRQ(_4_ierr);\
  _4_ierr = MPI_Scan(&__ctmp,&__end,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(_4_ierr); __start = __end - __ctmp;\
  _4_ierr = MPI_Scan(&__nrows,&__rstart,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(_4_ierr); __rstart = __rstart - __nrows;

/*MC
   MatPreallocateSymmetricInitialize - Begins the block of code that will count the number of nonzeros per
       row in a matrix providing the data that one can use to correctly preallocate the matrix.

   Synopsis:
   PetscErrorCode MatPreallocateSymmetricInitialize(MPI_Comm comm, PetscInt nrows, PetscInt ncols, PetscInt *dnz, PetscInt *onz)

   Collective on MPI_Comm

   Input Parameters:
+  comm - the communicator that will share the eventually allocated matrix
.  nrows - the number of LOCAL rows in the matrix
-  ncols - the number of LOCAL columns in the matrix

   Output Parameters:
+  dnz - the array that will be passed to the matrix preallocation routines
-  ozn - the other array passed to the matrix preallocation routines


   Level: intermediate

   Notes:
   See the chapter in the users manual on performance for more details

   Do not malloc or free dnz and onz, that is handled internally by these routines

   This is a MACRO not a function because it has a leading { that is closed by PetscPreallocateFinalize().

  Concepts: preallocation^Matrix

.seealso: MatPreallocateFinalize(), MatPreallocateSet(), MatPreallocateSymmetricSet(), MatPreallocateSetLocal(),
          MatPreallocateInitialize(), MatPreallocateSymmetricSetLocal()
M*/
#define MatPreallocateSymmetricInitialize(comm,nrows,ncols,dnz,onz) 0; \
{ \
  PetscErrorCode _4_ierr; PetscInt __nrows = (nrows),__ctmp = (ncols),__rstart,__end; \
  _4_ierr = PetscMalloc2(__nrows,PetscInt,&dnz,__nrows,PetscInt,&onz);CHKERRQ(_4_ierr); \
  _4_ierr = PetscMemzero(dnz,__nrows*sizeof(PetscInt));CHKERRQ(_4_ierr);\
  _4_ierr = PetscMemzero(onz,__nrows*sizeof(PetscInt));CHKERRQ(_4_ierr);\
  _4_ierr = MPI_Scan(&__ctmp,&__end,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(_4_ierr);\
  _4_ierr = MPI_Scan(&__nrows,&__rstart,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(_4_ierr); __rstart = __rstart - __nrows;

/*MC
   MatPreallocateSetLocal - Indicates the locations (rows and columns) in the matrix where nonzeros will be
       inserted using a local number of the rows and columns

   Synopsis:
   PetscErrorCode MatPreallocateSetLocal(ISLocalToGlobalMappping map,PetscInt nrows, PetscInt *rows,PetscInt ncols, PetscInt *cols,PetscInt *dnz, PetscInt *onz)

   Not Collective

   Input Parameters:
+  map - the mapping between local numbering and global numbering
.  nrows - the number of rows indicated
.  rows - the indices of the rows 
.  ncols - the number of columns in the matrix
.  cols - the columns indicated
.  dnz - the array that will be passed to the matrix preallocation routines
-  ozn - the other array passed to the matrix preallocation routines


   Level: intermediate

   Notes:
   See the chapter in the users manual on performance for more details

   Do not malloc or free dnz and onz, that is handled internally by these routines

  Concepts: preallocation^Matrix

.seealso: MatPreallocateFinalize(), MatPreallocateSet(), MatPreallocateSymmetricSet(), MatPreallocateInitialize(),
          MatPreallocateInitialize(), MatPreallocateSymmetricSetLocal()
M*/
#define MatPreallocateSetLocal(map,nrows,rows,ncols,cols,dnz,onz) 0;\
{\
  PetscInt __l;\
  _4_ierr = ISLocalToGlobalMappingApply(map,nrows,rows,rows);CHKERRQ(_4_ierr);\
  _4_ierr = ISLocalToGlobalMappingApply(map,ncols,cols,cols);CHKERRQ(_4_ierr);\
  for (__l=0;__l<nrows;__l++) {\
    _4_ierr = MatPreallocateSet((rows)[__l],ncols,cols,dnz,onz);CHKERRQ(_4_ierr);\
  }\
}
    
/*MC
   MatPreallocateSymmetricSetLocal - Indicates the locations (rows and columns) in the matrix where nonzeros will be
       inserted using a local number of the rows and columns

   Synopsis:
   PetscErrorCode MatPreallocateSymmetricSetLocal(ISLocalToGlobalMappping map,PetscInt nrows, PetscInt *rows,PetscInt ncols, PetscInt *cols,PetscInt *dnz, PetscInt *onz)

   Not Collective

   Input Parameters:
+  map - the mapping between local numbering and global numbering
.  nrows - the number of rows indicated
.  rows - the indices of the rows 
.  ncols - the number of columns in the matrix
.  cols - the columns indicated
.  dnz - the array that will be passed to the matrix preallocation routines
-  ozn - the other array passed to the matrix preallocation routines


   Level: intermediate

   Notes:
   See the chapter in the users manual on performance for more details

   Do not malloc or free dnz and onz that is handled internally by these routines

  Concepts: preallocation^Matrix

.seealso: MatPreallocateFinalize(), MatPreallocateSet(), MatPreallocateSymmetricSet(), MatPreallocateInitialize(),
          MatPreallocateInitialize(), MatPreallocateSymmetricSetLocal(), MatPreallocateSetLocal()
M*/
#define MatPreallocateSymmetricSetLocal(map,nrows,rows,ncols,cols,dnz,onz) 0;\
{\
  PetscInt __l;\
  _4_ierr = ISLocalToGlobalMappingApply(map,nrows,rows,rows);CHKERRQ(_4_ierr);\
  _4_ierr = ISLocalToGlobalMappingApply(map,ncols,cols,cols);CHKERRQ(_4_ierr);\
  for (__l=0;__l<nrows;__l++) {\
    _4_ierr = MatPreallocateSymmetricSet((rows)[__l],ncols,cols,dnz,onz);CHKERRQ(_4_ierr);\
  }\
}

/*MC
   MatPreallocateSet - Indicates the locations (rows and columns) in the matrix where nonzeros will be
       inserted using a local number of the rows and columns

   Synopsis:
   PetscErrorCode MatPreallocateSet(PetscInt nrows, PetscInt *rows,PetscInt ncols, PetscInt *cols,PetscInt *dnz, PetscInt *onz)

   Not Collective

   Input Parameters:
+  row - the row
.  ncols - the number of columns in the matrix
-  cols - the columns indicated

   Output Parameters:
+  dnz - the array that will be passed to the matrix preallocation routines
-  ozn - the other array passed to the matrix preallocation routines


   Level: intermediate

   Notes:
   See the chapter in the users manual on performance for more details

   Do not malloc or free dnz and onz that is handled internally by these routines

   This is a MACRO not a function because it uses variables declared in MatPreallocateInitialize().

  Concepts: preallocation^Matrix

.seealso: MatPreallocateFinalize(), MatPreallocateSet(), MatPreallocateSymmetricSet(), MatPreallocateInitialize(),
          MatPreallocateInitialize(), MatPreallocateSymmetricSetLocal(), MatPreallocateSetLocal()
M*/
#define MatPreallocateSet(row,nc,cols,dnz,onz) 0;\
{ PetscInt __i; \
  if (row < __rstart) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Trying to set preallocation for row %D less than first local row %D",row,__rstart);\
  if (row >= __rstart+__nrows) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Trying to set preallocation for row %D greater than last local row %D",row,__rstart+__nrows-1);\
  for (__i=0; __i<nc; __i++) {\
    if ((cols)[__i] < __start || (cols)[__i] >= __end) onz[row - __rstart]++; \
    else dnz[row - __rstart]++;\
  }\
}

/*MC
   MatPreallocateSymmetricSet - Indicates the locations (rows and columns) in the matrix where nonzeros will be
       inserted using a local number of the rows and columns

   Synopsis:
   PetscErrorCode MatPreallocateSymmetricSet(PetscInt nrows, PetscInt *rows,PetscInt ncols, PetscInt *cols,PetscInt *dnz, PetscInt *onz)

   Not Collective

   Input Parameters:
+  nrows - the number of rows indicated
.  rows - the indices of the rows 
.  ncols - the number of columns in the matrix
.  cols - the columns indicated
.  dnz - the array that will be passed to the matrix preallocation routines
-  ozn - the other array passed to the matrix preallocation routines


   Level: intermediate

   Notes:
   See the chapter in the users manual on performance for more details

   Do not malloc or free dnz and onz that is handled internally by these routines

   This is a MACRO not a function because it uses variables declared in MatPreallocateInitialize().

  Concepts: preallocation^Matrix

.seealso: MatPreallocateFinalize(), MatPreallocateSet(), MatPreallocateSymmetricSet(), MatPreallocateInitialize(),
          MatPreallocateInitialize(), MatPreallocateSymmetricSetLocal(), MatPreallocateSetLocal()
M*/
#define MatPreallocateSymmetricSet(row,nc,cols,dnz,onz) 0;\
{ PetscInt __i; \
  for (__i=0; __i<nc; __i++) {\
    if (cols[__i] >= __end) onz[row - __rstart]++; \
    else if (cols[__i] >= row) dnz[row - __rstart]++;\
  }\
}

/*MC
   MatPreallocateLocation -  An alternative to MatPreallocationSet() that puts the nonzero locations into the matrix if it exists

   Synopsis:
   PetscErrorCode MatPreallocateLocations(Mat A,PetscInt row,PetscInt ncols,PetscInt *cols,PetscInt *dnz,PetscInt *onz)

   Not Collective

   Input Parameters:
.  A - matrix
.  row - row where values exist (must be local to this process)
.  ncols - number of columns
.  cols - columns with nonzeros
.  dnz - the array that will be passed to the matrix preallocation routines
-  ozn - the other array passed to the matrix preallocation routines


   Level: intermediate

   Notes:
   See the chapter in the users manual on performance for more details

   Do not malloc or free dnz and onz that is handled internally by these routines

   This is a MACRO not a function because it uses a bunch of variables private to the MatPreallocation.... routines.

  Concepts: preallocation^Matrix

.seealso: MatPreallocateInitialize(), MatPreallocateSet(), MatPreallocateSymmetricSet(), MatPreallocateSetLocal(),
          MatPreallocateSymmetricInitialize(), MatPreallocateSymmetricSetLocal()
M*/
#define MatPreallocateLocation(A,row,ncols,cols,dnz,onz) 0;if (A) {ierr = MatSetValues(A,1,&row,ncols,cols,PETSC_NULL,INSERT_VALUES);CHKERRQ(ierr);} else {ierr =  MatPreallocateSet(row,ncols,cols,dnz,onz);CHKERRQ(ierr);}


/*MC
   MatPreallocateFinalize - Ends the block of code that will count the number of nonzeros per
       row in a matrix providing the data that one can use to correctly preallocate the matrix.

   Synopsis:
   PetscErrorCode MatPreallocateFinalize(PetscInt *dnz, PetscInt *onz)

   Collective on MPI_Comm

   Input Parameters:
+  dnz - the array that was be passed to the matrix preallocation routines
-  ozn - the other array passed to the matrix preallocation routines


   Level: intermediate

   Notes:
   See the chapter in the users manual on performance for more details

   Do not malloc or free dnz and onz that is handled internally by these routines

   This is a MACRO not a function because it closes the { started in MatPreallocateInitialize().

  Concepts: preallocation^Matrix

.seealso: MatPreallocateInitialize(), MatPreallocateSet(), MatPreallocateSymmetricSet(), MatPreallocateSetLocal(),
          MatPreallocateSymmetricInitialize(), MatPreallocateSymmetricSetLocal()
M*/
#define MatPreallocateFinalize(dnz,onz) 0;_4_ierr = PetscFree2(dnz,onz);CHKERRQ(_4_ierr);}



/* Routines unique to particular data structures */
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatShellGetContext(Mat,void **);
PetscPolymorphicFunction(MatShellGetContext,(Mat A),(A,&t),void*,t)

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatInodeAdjustForInodes(Mat,IS*,IS*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatInodeGetInodeSizes(Mat,PetscInt *,PetscInt *[],PetscInt *);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSeqAIJSetColumnIndices(Mat,PetscInt[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSeqBAIJSetColumnIndices(Mat,PetscInt[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateSeqAIJWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt[],PetscInt[],PetscScalar[],Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateSeqBAIJWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt[],PetscInt[],PetscScalar[],Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateSeqSBAIJWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt[],PetscInt[],PetscScalar[],Mat*);

#define MAT_SKIP_ALLOCATION -4

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSeqBAIJSetPreallocation(Mat,PetscInt,PetscInt,const PetscInt[]);
PetscPolymorphicSubroutine(MatSeqBAIJSetPreallocation,(Mat A,PetscInt bs,const PetscInt nnz[]),(A,bs,0,nnz))
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSeqSBAIJSetPreallocation(Mat,PetscInt,PetscInt,const PetscInt[]);
PetscPolymorphicSubroutine(MatSeqSBAIJSetPreallocation,(Mat A,PetscInt bs,const PetscInt nnz[]),(A,bs,0,nnz))
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSeqAIJSetPreallocation(Mat,PetscInt,const PetscInt[]);
PetscPolymorphicSubroutine(MatSeqAIJSetPreallocation,(Mat A,const PetscInt nnz[]),(A,0,nnz))

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMPIBAIJSetPreallocation(Mat,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);
PetscPolymorphicSubroutine(MatMPIBAIJSetPreallocation,(Mat A,PetscInt bs,const PetscInt nnz[],const PetscInt onz[]),(A,bs,0,nnz,0,onz))
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMPISBAIJSetPreallocation(Mat,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMPIAIJSetPreallocation(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSeqAIJSetPreallocationCSR(Mat,const PetscInt [],const PetscInt [],const PetscScalar []);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSeqBAIJSetPreallocationCSR(Mat,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMPIAIJSetPreallocationCSR(Mat,const PetscInt[],const PetscInt[],const PetscScalar[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMPIBAIJSetPreallocationCSR(Mat,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMPIAdjSetPreallocation(Mat,PetscInt[],PetscInt[],PetscInt[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMPIDenseSetPreallocation(Mat,PetscScalar[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSeqDenseSetPreallocation(Mat,PetscScalar[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMPIAIJGetSeqAIJ(Mat,Mat*,Mat*,PetscInt*[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMPIBAIJGetSeqBAIJ(Mat,Mat*,Mat*,PetscInt*[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatAdicSetLocalFunction(Mat,void (*)(void));

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSeqDenseSetLDA(Mat,PetscInt);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatDenseGetLocalMatrix(Mat,Mat*);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatStoreValues(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatRetrieveValues(Mat);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatDAADSetCtx(Mat,void*);

/* 
  These routines are not usually accessed directly, rather solving is 
  done through the KSP and PC interfaces.
*/

/*E
    MatOrderingType - String with the name of a PETSc matrix ordering or the creation function
       with an optional dynamic library name, for example 
       http://www.mcs.anl.gov/petsc/lib.a:orderingcreate()

   Level: beginner

   Cannot use const because the PC objects manipulate the string

.seealso: MatGetOrdering()
E*/
#define MatOrderingType char*
#define MATORDERING_NATURAL     "natural"
#define MATORDERING_ND          "nd"
#define MATORDERING_1WD         "1wd"
#define MATORDERING_RCM         "rcm"
#define MATORDERING_QMD         "qmd"
#define MATORDERING_ROWLENGTH   "rowlength"
#define MATORDERING_DSC_ND      "dsc_nd"         /* these three are only for DSCPACK, see its documentation for details */
#define MATORDERING_DSC_MMD     "dsc_mmd"
#define MATORDERING_DSC_MDF     "dsc_mdf"
#define MATORDERING_CONSTRAINED "constrained"
#define MATORDERING_IDENTITY    "identity"
#define MATORDERING_REVERSE     "reverse"
#define MATORDERING_FLOW        "flow"
#define MATORDERING_AMD         "amd"

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetOrdering(Mat,const MatOrderingType,IS*,IS*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetOrderingList(PetscFList *list);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatOrderingRegister(const char[],const char[],const char[],PetscErrorCode(*)(Mat,const MatOrderingType,IS*,IS*));

/*MC
   MatOrderingRegisterDynamic - Adds a new sparse matrix ordering to the matrix package. 

   Synopsis:
   PetscErrorCode MatOrderingRegisterDynamic(const char *name_ordering,const char *path,const char *name_create,PetscErrorCode (*routine_create)(MatOrdering))

   Not Collective

   Input Parameters:
+  sname - name of ordering (for example MATORDERING_ND)
.  path - location of library where creation routine is 
.  name - name of function that creates the ordering type,a string
-  function - function pointer that creates the ordering

   Level: developer

   If dynamic libraries are used, then the fourth input argument (function)
   is ignored.

   Sample usage:
.vb
   MatOrderingRegisterDynamic("my_order",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyOrder",MyOrder);
.ve

   Then, your partitioner can be chosen with the procedural interface via
$     MatOrderingSetType(part,"my_order)
   or at runtime via the option
$     -pc_factor_mat_ordering_type my_order

   ${PETSC_ARCH} occuring in pathname will be replaced with appropriate values.

.keywords: matrix, ordering, register

.seealso: MatOrderingRegisterDestroy(), MatOrderingRegisterAll()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define MatOrderingRegisterDynamic(a,b,c,d) MatOrderingRegister(a,b,c,0)
#else
#define MatOrderingRegisterDynamic(a,b,c,d) MatOrderingRegister(a,b,c,d)
#endif

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatOrderingRegisterDestroy(void);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatOrderingRegisterAll(const char[]);
extern PetscTruth MatOrderingRegisterAllCalled;
extern PetscFList MatOrderingList;

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatReorderForNonzeroDiagonal(Mat,PetscReal,IS,IS);

/*S
    MatFactorShiftType - Numeric Shift.

   Level: beginner

S*/
typedef enum {MAT_SHIFT_NONE,MAT_SHIFT_NONZERO,MAT_SHIFT_POSITIVE_DEFINITE,MAT_SHIFT_INBLOCKS} MatFactorShiftType;
extern const char *MatFactorShiftTypes[];

/*S 
   MatFactorInfo - Data passed into the matrix factorization routines

   In Fortran these are simply double precision arrays of size MAT_FACTORINFO_SIZE, that is use
$     MatFactorInfo  info(MAT_FACTORINFO_SIZE)

   Notes: These are not usually directly used by users, instead use PC type of LU, ILU, CHOLESKY or ICC.

      You can use MatFactorInfoInitialize() to set default values.

   Level: developer

.seealso: MatLUFactorSymbolic(), MatILUFactorSymbolic(), MatCholeskyFactorSymbolic(), MatICCFactorSymbolic(), MatICCFactor(), 
          MatFactorInfoInitialize()

S*/
typedef struct {
  PetscReal     diagonal_fill;  /* force diagonal to fill in if initially not filled */
  PetscReal     usedt;
  PetscReal     dt;             /* drop tolerance */
  PetscReal     dtcol;          /* tolerance for pivoting */
  PetscReal     dtcount;        /* maximum nonzeros to be allowed per row */
  PetscReal     fill;           /* expected fill, nonzeros in factored matrix/nonzeros in original matrix */
  PetscReal     levels;         /* ICC/ILU(levels) */ 
  PetscReal     pivotinblocks;  /* for BAIJ and SBAIJ matrices pivot in factorization on blocks, default 1.0 
                                   factorization may be faster if do not pivot */
  PetscReal     zeropivot;      /* pivot is called zero if less than this */
  PetscReal     shifttype;      /* type of shift added to matrix factor to prevent zero pivots */
  PetscReal     shiftamount;     /* how large the shift is */
} MatFactorInfo;

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatFactorInfoInitialize(MatFactorInfo*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCholeskyFactor(Mat,IS,const MatFactorInfo*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCholeskyFactorSymbolic(Mat,Mat,IS,const MatFactorInfo*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCholeskyFactorNumeric(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatLUFactor(Mat,IS,IS,const MatFactorInfo*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatILUFactor(Mat,IS,IS,const MatFactorInfo*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatLUFactorSymbolic(Mat,Mat,IS,IS,const MatFactorInfo*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatILUFactorSymbolic(Mat,Mat,IS,IS,const MatFactorInfo*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatICCFactorSymbolic(Mat,Mat,IS,const MatFactorInfo*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatICCFactor(Mat,IS,const MatFactorInfo*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatLUFactorNumeric(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetInertia(Mat,PetscInt*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSolve(Mat,Vec,Vec);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatForwardSolve(Mat,Vec,Vec);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatBackwardSolve(Mat,Vec,Vec);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSolveAdd(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSolveTranspose(Mat,Vec,Vec);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSolveTransposeAdd(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSolves(Mat,Vecs,Vecs);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSetUnfactored(Mat);

/*E
    MatSORType - What type of (S)SOR to perform

    Level: beginner

   May be bitwise ORd together

   Any additions/changes here MUST also be made in include/finclude/petscmat.h

   MatSORType may be bitwise ORd together, so do not change the numbers 

.seealso: MatSOR()
E*/
typedef enum {SOR_FORWARD_SWEEP=1,SOR_BACKWARD_SWEEP=2,SOR_SYMMETRIC_SWEEP=3,
              SOR_LOCAL_FORWARD_SWEEP=4,SOR_LOCAL_BACKWARD_SWEEP=8,
              SOR_LOCAL_SYMMETRIC_SWEEP=12,SOR_ZERO_INITIAL_GUESS=16,
              SOR_EISENSTAT=32,SOR_APPLY_UPPER=64,SOR_APPLY_LOWER=128} MatSORType;
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSOR(Mat,Vec,PetscReal,MatSORType,PetscReal,PetscInt,PetscInt,Vec);

/* 
    These routines are for efficiently computing Jacobians via finite differences.
*/

/*E
    MatColoringType - String with the name of a PETSc matrix coloring or the creation function
       with an optional dynamic library name, for example 
       http://www.mcs.anl.gov/petsc/lib.a:coloringcreate()

   Level: beginner

.seealso: MatGetColoring()
E*/
#define MatColoringType char*
#define MATCOLORING_NATURAL "natural"
#define MATCOLORING_SL      "sl"
#define MATCOLORING_LF      "lf"
#define MATCOLORING_ID      "id"

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetColoring(Mat,const MatColoringType,ISColoring*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatColoringRegister(const char[],const char[],const char[],PetscErrorCode(*)(Mat,MatColoringType,ISColoring *));

/*MC
   MatColoringRegisterDynamic - Adds a new sparse matrix coloring to the 
                               matrix package. 

   Synopsis:
   PetscErrorCode MatColoringRegisterDynamic(const char *name_coloring,const char *path,const char *name_create,PetscErrorCode (*routine_create)(MatColoring))

   Not Collective

   Input Parameters:
+  sname - name of Coloring (for example MATCOLORING_SL)
.  path - location of library where creation routine is 
.  name - name of function that creates the Coloring type, a string
-  function - function pointer that creates the coloring

   Level: developer

   If dynamic libraries are used, then the fourth input argument (function)
   is ignored.

   Sample usage:
.vb
   MatColoringRegisterDynamic("my_color",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyColor",MyColor);
.ve

   Then, your partitioner can be chosen with the procedural interface via
$     MatColoringSetType(part,"my_color")
   or at runtime via the option
$     -mat_coloring_type my_color

   $PETSC_ARCH occuring in pathname will be replaced with appropriate values.

.keywords: matrix, Coloring, register

.seealso: MatColoringRegisterDestroy(), MatColoringRegisterAll()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define MatColoringRegisterDynamic(a,b,c,d) MatColoringRegister(a,b,c,0)
#else
#define MatColoringRegisterDynamic(a,b,c,d) MatColoringRegister(a,b,c,d)
#endif

extern PetscTruth MatColoringRegisterAllCalled;

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatColoringRegisterAll(const char[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatColoringRegisterDestroy(void);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatColoringPatch(Mat,PetscInt,PetscInt,ISColoringValue[],ISColoring*);

/*S
     MatFDColoring - Object for computing a sparse Jacobian via finite differences
        and coloring

   Level: beginner

  Concepts: coloring, sparse Jacobian, finite differences

.seealso:  MatFDColoringCreate()
S*/
typedef struct _p_MatFDColoring* MatFDColoring;

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatFDColoringCreate(Mat,ISColoring,MatFDColoring *);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatFDColoringDestroy(MatFDColoring);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatFDColoringView(MatFDColoring,PetscViewer);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatFDColoringSetFunction(MatFDColoring,PetscErrorCode (*)(void),void*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatFDColoringGetFunction(MatFDColoring,PetscErrorCode (**)(void),void**);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatFDColoringSetParameters(MatFDColoring,PetscReal,PetscReal);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatFDColoringSetFromOptions(MatFDColoring);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatFDColoringApply(Mat,MatFDColoring,Vec,MatStructure*,void *);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatFDColoringApplyTS(Mat,MatFDColoring,PetscReal,Vec,MatStructure*,void *);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatFDColoringSetF(MatFDColoring,Vec);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatFDColoringGetPerturbedColumns(MatFDColoring,PetscInt*,PetscInt*[]);
/* 
    These routines are for partitioning matrices: currently used only 
  for adjacency matrix, MatCreateMPIAdj().
*/

/*S
     MatPartitioning - Object for managing the partitioning of a matrix or graph

   Level: beginner

  Concepts: partitioning

.seealso:  MatPartitioningCreate(), MatPartitioningType
S*/
typedef struct _p_MatPartitioning* MatPartitioning;

/*E
    MatPartitioningType - String with the name of a PETSc matrix partitioning or the creation function
       with an optional dynamic library name, for example 
       http://www.mcs.anl.gov/petsc/lib.a:partitioningcreate()

   Level: beginner

.seealso: MatPartitioningCreate(), MatPartitioning
E*/
#define MatPartitioningType char*
#define MAT_PARTITIONING_CURRENT  "current"
#define MAT_PARTITIONING_SQUARE   "square"
#define MAT_PARTITIONING_PARMETIS "parmetis"
#define MAT_PARTITIONING_CHACO    "chaco"
#define MAT_PARTITIONING_JOSTLE   "jostle"
#define MAT_PARTITIONING_PARTY    "party"
#define MAT_PARTITIONING_SCOTCH   "scotch"


EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningCreate(MPI_Comm,MatPartitioning*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningSetType(MatPartitioning,const MatPartitioningType);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningSetNParts(MatPartitioning,PetscInt);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningSetAdjacency(MatPartitioning,Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningSetVertexWeights(MatPartitioning,const PetscInt[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningSetPartitionWeights(MatPartitioning,const PetscReal []);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningApply(MatPartitioning,IS*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningDestroy(MatPartitioning);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningRegister(const char[],const char[],const char[],PetscErrorCode (*)(MatPartitioning));

/*MC
   MatPartitioningRegisterDynamic - Adds a new sparse matrix partitioning to the 
   matrix package. 

   Synopsis:
   PetscErrorCode MatPartitioningRegisterDynamic(const char *name_partitioning,const char *path,const char *name_create,PetscErrorCode (*routine_create)(MatPartitioning))

   Not Collective

   Input Parameters:
+  sname - name of partitioning (for example MAT_PARTITIONING_CURRENT) or parmetis
.  path - location of library where creation routine is 
.  name - name of function that creates the partitioning type, a string
-  function - function pointer that creates the partitioning type

   Level: developer

   If dynamic libraries are used, then the fourth input argument (function)
   is ignored.

   Sample usage:
.vb
   MatPartitioningRegisterDynamic("my_part",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyPartCreate",MyPartCreate);
.ve

   Then, your partitioner can be chosen with the procedural interface via
$     MatPartitioningSetType(part,"my_part")
   or at runtime via the option
$     -mat_partitioning_type my_part

   $PETSC_ARCH occuring in pathname will be replaced with appropriate values.

.keywords: matrix, partitioning, register

.seealso: MatPartitioningRegisterDestroy(), MatPartitioningRegisterAll()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define MatPartitioningRegisterDynamic(a,b,c,d) MatPartitioningRegister(a,b,c,0)
#else
#define MatPartitioningRegisterDynamic(a,b,c,d) MatPartitioningRegister(a,b,c,d)
#endif

extern PetscTruth MatPartitioningRegisterAllCalled;

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningRegisterAll(const char[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningRegisterDestroy(void);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningView(MatPartitioning,PetscViewer);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningSetFromOptions(MatPartitioning);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningGetType(MatPartitioning,const MatPartitioningType*);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningParmetisSetCoarseSequential(MatPartitioning);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningParmetisGetEdgeCut(MatPartitioning, PetscInt *);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningJostleSetCoarseLevel(MatPartitioning,PetscReal);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningJostleSetCoarseSequential(MatPartitioning);

typedef enum {MP_CHACO_MULTILEVEL_KL,MP_CHACO_SPECTRAL,MP_CHACO_LINEAR,MP_CHACO_RANDOM, MP_CHACO_SCATTERED} MPChacoGlobalType;
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningChacoSetGlobal(MatPartitioning, MPChacoGlobalType);
typedef enum { MP_CHACO_KERNIGHAN_LIN, MP_CHACO_NONE } MPChacoLocalType;
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningChacoSetLocal(MatPartitioning, MPChacoLocalType);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningChacoSetCoarseLevel(MatPartitioning,PetscReal);
typedef enum { MP_CHACO_LANCZOS, MP_CHACO_RQI_SYMMLQ } MPChacoEigenType;
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningChacoSetEigenSolver(MatPartitioning,MPChacoEigenType);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningChacoSetEigenTol(MatPartitioning, PetscReal);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningChacoSetEigenNumber(MatPartitioning, PetscInt);

#define MP_PARTY_OPT "opt"
#define MP_PARTY_LIN "lin"
#define MP_PARTY_SCA "sca"
#define MP_PARTY_RAN "ran"
#define MP_PARTY_GBF "gbf"
#define MP_PARTY_GCF "gcf"
#define MP_PARTY_BUB "bub"
#define MP_PARTY_DEF "def"
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningPartySetGlobal(MatPartitioning, const char*);
#define MP_PARTY_HELPFUL_SETS "hs"
#define MP_PARTY_KERNIGHAN_LIN "kl"
#define MP_PARTY_NONE "no"
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningPartySetLocal(MatPartitioning, const char*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningPartySetCoarseLevel(MatPartitioning,PetscReal);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningPartySetBipart(MatPartitioning,PetscTruth);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningPartySetMatchOptimization(MatPartitioning,PetscTruth);

typedef enum { MP_SCOTCH_GREEDY, MP_SCOTCH_GPS, MP_SCOTCH_GR_GPS } MPScotchGlobalType;
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningScotchSetArch(MatPartitioning,const char*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningScotchSetMultilevel(MatPartitioning);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningScotchSetGlobal(MatPartitioning,MPScotchGlobalType);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningScotchSetCoarseLevel(MatPartitioning,PetscReal);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningScotchSetHostList(MatPartitioning,const char*);
typedef enum { MP_SCOTCH_KERNIGHAN_LIN, MP_SCOTCH_NONE } MPScotchLocalType;
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningScotchSetLocal(MatPartitioning,MPScotchLocalType);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningScotchSetMapping(MatPartitioning);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningScotchSetStrategy(MatPartitioning,char*);

EXTERN PetscErrorCode MatMeshToVertexGraph(Mat,PetscInt,Mat*);
EXTERN PetscErrorCode MatMeshToCellGraph(Mat,PetscInt,Mat*);

/*
    If you add entries here you must also add them to finclude/petscmat.h
*/
typedef enum { MATOP_SET_VALUES=0,
               MATOP_GET_ROW=1,
               MATOP_RESTORE_ROW=2,
               MATOP_MULT=3,
               MATOP_MULT_ADD=4,
               MATOP_MULT_TRANSPOSE=5,
               MATOP_MULT_TRANSPOSE_ADD=6,
               MATOP_SOLVE=7,
               MATOP_SOLVE_ADD=8,
               MATOP_SOLVE_TRANSPOSE=9,
               MATOP_SOLVE_TRANSPOSE_ADD=10,
               MATOP_LUFACTOR=11,
               MATOP_CHOLESKYFACTOR=12,
               MATOP_SOR=13,
               MATOP_TRANSPOSE=14,
               MATOP_GETINFO=15,
               MATOP_EQUAL=16,
               MATOP_GET_DIAGONAL=17,
               MATOP_DIAGONAL_SCALE=18,
               MATOP_NORM=19,
               MATOP_ASSEMBLY_BEGIN=20,
               MATOP_ASSEMBLY_END=21,
               MATOP_SET_OPTION=22,
               MATOP_ZERO_ENTRIES=23,
               MATOP_ZERO_ROWS=24,
               MATOP_LUFACTOR_SYMBOLIC=25,
               MATOP_LUFACTOR_NUMERIC=26,
               MATOP_CHOLESKY_FACTOR_SYMBOLIC=27,
               MATOP_CHOLESKY_FACTOR_NUMERIC=28,
               MATOP_SETUP_PREALLOCATION=29,
               MATOP_ILUFACTOR_SYMBOLIC=30,
               MATOP_ICCFACTOR_SYMBOLIC=31,
               MATOP_GET_ARRAY=32,
               MATOP_RESTORE_ARRAY=33,
               MATOP_DUPLICATE=34,
               MATOP_FORWARD_SOLVE=35,
               MATOP_BACKWARD_SOLVE=36,
               MATOP_ILUFACTOR=37,
               MATOP_ICCFACTOR=38,
               MATOP_AXPY=39,
               MATOP_GET_SUBMATRICES=40,
               MATOP_INCREASE_OVERLAP=41,
               MATOP_GET_VALUES=42,
               MATOP_COPY=43,
               MATOP_GET_ROW_MAX=44,
               MATOP_SCALE=45,
               MATOP_SHIFT=46,
               MATOP_DIAGONAL_SET=47,
               MATOP_ILUDT_FACTOR=48,
               MATOP_SET_BLOCK_SIZE=49,
               MATOP_GET_ROW_IJ=50,
               MATOP_RESTORE_ROW_IJ=51,
               MATOP_GET_COLUMN_IJ=52,
               MATOP_RESTORE_COLUMN_IJ=53,
               MATOP_FDCOLORING_CREATE=54,
               MATOP_COLORING_PATCH=55,
               MATOP_SET_UNFACTORED=56,
               MATOP_PERMUTE=57,
               MATOP_SET_VALUES_BLOCKED=58,
               MATOP_GET_SUBMATRIX=59,
               MATOP_DESTROY=60,
               MATOP_VIEW=61,
               MATOP_CONVERT_FROM=62,
               MATOP_USE_SCALED_FORM=63,
               MATOP_SCALE_SYSTEM=64,
               MATOP_UNSCALE_SYSTEM=65,
               MATOP_SET_LOCAL_TO_GLOBAL_MAP=66,
               MATOP_SET_VALUES_LOCAL=67,
               MATOP_ZERO_ROWS_LOCAL=68,
               MATOP_GET_ROW_MAX_ABS=69,
               MATOP_GET_ROW_MIN_ABS=70,
               MATOP_CONVERT=71,
               MATOP_SET_COLORING=72,
               MATOP_SET_VALUES_ADIC=73,
               MATOP_SET_VALUES_ADIFOR=74,
               MATOP_FD_COLORING_APPLY=75,
               MATOP_SET_FROM_OPTIONS=76,
               MATOP_MULT_CON=77,
               MATOP_MULT_TRANSPOSE_CON=78,
               MATOP_PERMUTE_SPARSIFY=79,
               MATOP_MULT_MULTIPLE=80,
               MATOP_SOLVE_MULTIPLE=81,
               MATOP_GET_INERTIA=82,
               MATOP_LOAD=83,
               MATOP_IS_SYMMETRIC=84,
               MATOP_IS_HERMITIAN=85,
               MATOP_IS_STRUCTURALLY_SYMMETRIC=86,
               MATOP_DUMMY=87,
               MATOP_GET_VECS=88,
               MATOP_MAT_MULT=89,
               MATOP_MAT_MULT_SYMBOLIC=90,
               MATOP_MAT_MULT_NUMERIC=91,
               MATOP_PTAP=92,
               MATOP_PTAP_SYMBOLIC=93,
               MATOP_PTAP_NUMERIC=94,
               MATOP_MAT_MULTTRANSPOSE=95,
               MATOP_MAT_MULTTRANSPOSE_SYM=96,
               MATOP_MAT_MULTTRANSPOSE_NUM=97,
               MATOP_PTAP_SYMBOLIC_SEQAIJ=98,
               MATOP_PTAP_NUMERIC_SEQAIJ=99,
               MATOP_PTAP_SYMBOLIC_MPIAIJ=100,
               MATOP_PTAP_NUMERIC_MPIAIJ=101,
               MATOP_CONJUGATE=102,
               MATOP_SET_SIZES=103,
               MATOP_SET_VALUES_ROW=104,
               MATOP_REAL_PART=105,
               MATOP_IMAG_PART=106,
               MATOP_GET_ROW_UTRIANGULAR=107,
               MATOP_RESTORE_ROW_UTRIANGULAR=108,
               MATOP_MATSOLVE=109,
               MATOP_GET_REDUNDANTMATRIX=110,
               MATOP_GET_ROW_MIN=111,
               MATOP_GET_COLUMN_VEC=112,
               MATOP_MISSING_DIAGONAL=113,
               MATOP_MATGETSEQNONZEROSTRUCTURE=114,
               MATOP_CREATE=115,
               MATOP_GET_GHOSTS=116,
               MATOP_MULT_DIAGONAL_BLOCK=119,
               MATOP_HERMITIANTRANSPOSE=120,
               MATOP_MULTHERMITIANTRANSPOSE=121,
               MATOP_MULTHERMITIANTRANSPOSEADD=122
             } MatOperation;
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatHasOperation(Mat,MatOperation,PetscTruth*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatShellSetOperation(Mat,MatOperation,void(*)(void));
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatShellGetOperation(Mat,MatOperation,void(**)(void));
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatShellSetContext(Mat,void*);

/*
   Codes for matrices stored on disk. By default they are
 stored in a universal format. By changing the format with 
 PetscViewerSetFormat(viewer,PETSC_VIEWER_NATIVE); the matrices will
 be stored in a way natural for the matrix, for example dense matrices
 would be stored as dense. Matrices stored this way may only be
 read into matrices of the same time.
*/
#define MATRIX_BINARY_FORMAT_DENSE -1

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMPIBAIJSetHashTableFactor(Mat,PetscReal);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatISGetLocalMat(Mat,Mat*);

/*S
     MatNullSpace - Object that removes a null space from a vector, i.e.
         orthogonalizes the vector to a subsapce

   Level: advanced

  Concepts: matrix; linear operator, null space

  Users manual sections:
.   sec_singular

.seealso:  MatNullSpaceCreate()
S*/
typedef struct _p_MatNullSpace* MatNullSpace;

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatNullSpaceCreate(MPI_Comm,PetscTruth,PetscInt,const Vec[],MatNullSpace*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatNullSpaceSetFunction(MatNullSpace,PetscErrorCode (*)(MatNullSpace,Vec,void*),void*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatNullSpaceDestroy(MatNullSpace);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatNullSpaceRemove(MatNullSpace,Vec,Vec*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatNullSpaceAttach(Mat,MatNullSpace);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatNullSpaceTest(MatNullSpace,Mat,PetscTruth *);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatReorderingSeqSBAIJ(Mat,IS);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMPISBAIJSetHashTableFactor(Mat,PetscReal);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSeqSBAIJSetColumnIndices(Mat,PetscInt *);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSeqBAIJInvertBlockDiagonal(Mat);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateMAIJ(Mat,PetscInt,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMAIJRedimension(Mat,PetscInt,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMAIJGetAIJ(Mat,Mat*);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatComputeExplicitOperator(Mat,Mat*);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatDiagonalScaleLocal(Mat,Vec);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateMFFD(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetBase(Mat,Vec,Vec);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetFunction(Mat,PetscErrorCode(*)(void*,Vec,Vec),void*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetFunctioni(Mat,PetscErrorCode (*)(void*,PetscInt,Vec,PetscScalar*));
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetFunctioniBase(Mat,PetscErrorCode (*)(void*,Vec));
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDAddNullSpace(Mat,MatNullSpace);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetHHistory(Mat,PetscScalar[],PetscInt);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDResetHHistory(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetFunctionError(Mat,PetscReal);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetPeriod(Mat,PetscInt);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDGetH(Mat,PetscScalar *);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetOptionsPrefix(Mat,const char[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetFromOptions(Mat);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDCheckPositivity(void*,Vec,Vec,PetscScalar*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetCheckh(Mat,PetscErrorCode (*)(void*,Vec,Vec,PetscScalar*),void*);

/*S
    MatMFFD - A data structured used to manage the computation of the h differencing parameter for matrix-free 
              Jacobian vector products

    Notes: MATMFFD is a specific MatType which uses the MatMFFD data structure

           MatMFFD*() methods actually take the Mat as their first argument. Not a MatMFFD data structure

    Level: developer

.seealso: MATMFFD, MatCreateMFFD(), MatMFFDSetFuction(), MatMFFDSetType(), MatMFFDRegister()
S*/
typedef struct _p_MatMFFD* MatMFFD;

/*E
    MatMFFDType - algorithm used to compute the h used in computing matrix-vector products via differencing of the function

   Level: beginner

.seealso: MatMFFDSetType(), MatMFFDRegister()
E*/
#define MatMFFDType char*
#define MATMFFD_DS  "ds"
#define MATMFFD_WP  "wp"

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetType(Mat,const MatMFFDType);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDRegister(const char[],const char[],const char[],PetscErrorCode (*)(MatMFFD));

/*MC
   MatMFFDRegisterDynamic - Adds a method to the MatMFFD registry.

   Synopsis:
   PetscErrorCode MatMFFDRegisterDynamic(const char *name_solver,const char *path,const char *name_create,PetscErrorCode (*routine_create)(MatMFFD))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined compute-h module
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Level: developer

   Notes:
   MatMFFDRegisterDynamic() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   MatMFFDRegisterDynamic("my_h",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyHCreate",MyHCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     MatMFFDSetType(mfctx,"my_h")
   or at runtime via the option
$     -snes_mf_type my_h

.keywords: MatMFFD, register

.seealso: MatMFFDRegisterAll(), MatMFFDRegisterDestroy()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define MatMFFDRegisterDynamic(a,b,c,d) MatMFFDRegister(a,b,c,0)
#else
#define MatMFFDRegisterDynamic(a,b,c,d) MatMFFDRegister(a,b,c,d)
#endif

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDRegisterAll(const char[]);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDRegisterDestroy(void);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDDSSetUmin(Mat,PetscReal);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDWPSetComputeNormU(Mat,PetscTruth);


EXTERN PetscErrorCode PETSCMAT_DLLEXPORT PetscViewerMathematicaPutMatrix(PetscViewer, PetscInt, PetscInt, PetscReal *);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT PetscViewerMathematicaPutCSRMatrix(PetscViewer, PetscInt, PetscInt, PetscInt *, PetscInt *, PetscReal *);

/* 
   PETSc interface to MUMPS 
*/
#ifdef PETSC_HAVE_MUMPS
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMumpsSetIcntl(Mat,PetscInt,PetscInt);
#endif

PETSC_EXTERN_CXX_END
#endif
