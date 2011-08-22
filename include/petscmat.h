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
#define MATMAIJ            "maij"
#define MATSEQMAIJ         "seqmaij"
#define MATMPIMAIJ         "mpimaij"
#define MATIS              "is"
#define MATAIJ             "aij"
#define MATSEQAIJ          "seqaij"
#define MATSEQAIJPTHREAD   "seqaijpthread"
#define MATAIJPTHREAD      "aijpthread"
#define MATMPIAIJ          "mpiaij"
#define MATAIJCRL          "aijcrl"
#define MATSEQAIJCRL       "seqaijcrl"
#define MATMPIAIJCRL       "mpiaijcrl"
#define MATAIJCUSP         "aijcusp"
#define MATSEQAIJCUSP      "seqaijcusp"
#define MATMPIAIJCUSP      "mpiaijcusp"
#define MATAIJPERM         "aijperm"
#define MATSEQAIJPERM      "seqaijperm"
#define MATMPIAIJPERM      "mpiaijperm"
#define MATSHELL           "shell"
#define MATDENSE           "dense"
#define MATSEQDENSE        "seqdense"
#define MATMPIDENSE        "mpidense"
#define MATBAIJ            "baij"
#define MATSEQBAIJ         "seqbaij"
#define MATMPIBAIJ         "mpibaij"
#define MATMPIADJ          "mpiadj"
#define MATSBAIJ           "sbaij"
#define MATSEQSBAIJ        "seqsbaij"
#define MATMPISBAIJ        "mpisbaij"

#define MATSEQBSTRM        "seqbstrm"
#define MATMPIBSTRM        "mpibstrm"
#define MATBSTRM           "bstrm"
#define MATSEQSBSTRM       "seqsbstrm"
#define MATMPISBSTRM       "mpisbstrm"
#define MATSBSTRM          "sbstrm"

#define MATDAAD            "daad"
#define MATMFFD            "mffd"
#define MATNORMAL          "normal"
#define MATLRC             "lrc"
#define MATSCATTER         "scatter"
#define MATBLOCKMAT        "blockmat"
#define MATCOMPOSITE       "composite"
#define MATFFT             "fft"
#define MATFFTW            "fftw"
#define MATSEQCUFFT        "seqcufft"
#define MATTRANSPOSEMAT    "transpose"
#define MATSCHURCOMPLEMENT "schurcomplement"
#define MATPYTHON          "python"
#define MATHYPRESTRUCT     "hyprestruct"
#define MATHYPRESSTRUCT    "hypresstruct"
#define MATSUBMATRIX       "submatrix"
#define MATLOCALREF        "localref"
#define MATNEST            "nest"

/*E
    MatSolverPackage - String with the name of a PETSc matrix solver type. 

    For example: "petsc" indicates what PETSc provides, "superlu" indicates either 
       SuperLU or SuperLU_Dist etc.


   Level: beginner

.seealso: MatGetFactor(), Mat, MatSetType(), MatType
E*/
#define MatSolverPackage char*
#define MATSOLVERSPOOLES      "spooles"
#define MATSOLVERSUPERLU      "superlu"
#define MATSOLVERSUPERLU_DIST "superlu_dist"
#define MATSOLVERUMFPACK      "umfpack"
#define MATSOLVERCHOLMOD      "cholmod"
#define MATSOLVERESSL         "essl"
#define MATSOLVERLUSOL        "lusol"
#define MATSOLVERMUMPS        "mumps"
#define MATSOLVERPASTIX       "pastix"
#define MATSOLVERDSCPACK      "dscpack"
#define MATSOLVERMATLAB       "matlab"
#define MATSOLVERPETSC        "petsc"
#define MATSOLVERPLAPACK      "plapack"
#define MATSOLVERBAS          "bas"

#define MATSOLVERBSTRM        "bstrm"
#define MATSOLVERSBSTRM       "sbstrm"

/*E
    MatFactorType - indicates what type of factorization is requested

    Level: beginner

   Any additions/changes here MUST also be made in include/finclude/petscmat.h

.seealso: MatSolverPackage, MatGetFactor()
E*/
typedef enum {MAT_FACTOR_NONE, MAT_FACTOR_LU, MAT_FACTOR_CHOLESKY, MAT_FACTOR_ILU, MAT_FACTOR_ICC,MAT_FACTOR_ILUDT} MatFactorType;
extern const char *const MatFactorTypes[];

extern PetscErrorCode  MatGetFactor(Mat,const MatSolverPackage,MatFactorType,Mat*);
extern PetscErrorCode  MatGetFactorAvailable(Mat,const MatSolverPackage,MatFactorType,PetscBool *);
extern PetscErrorCode  MatFactorGetSolverPackage(Mat,const MatSolverPackage*);
extern PetscErrorCode  MatGetFactorType(Mat,MatFactorType*);

/* Logging support */
#define    MAT_FILE_CLASSID 1211216    /* used to indicate matrices in binary files */
extern PetscClassId  MAT_CLASSID;
extern PetscClassId  MAT_FDCOLORING_CLASSID;
extern PetscClassId  MAT_PARTITIONING_CLASSID;
extern PetscClassId  MAT_NULLSPACE_CLASSID;
extern PetscClassId  MATMFFD_CLASSID;

/*E
    MatReuse - Indicates if matrices obtained from a previous call to MatGetSubMatrices()
     or MatGetSubMatrix() are to be reused to store the new matrix values. For MatConvert() is used to indicate
     that the input matrix is to be replaced with the converted matrix.

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

extern PetscErrorCode  MatInitializePackage(const char[]);

extern PetscErrorCode  MatCreate(MPI_Comm,Mat*);
PetscPolymorphicFunction(MatCreate,(MPI_Comm comm),(comm,&A),Mat,A)
PetscPolymorphicFunction(MatCreate,(),(PETSC_COMM_WORLD,&A),Mat,A)
extern PetscErrorCode  MatSetSizes(Mat,PetscInt,PetscInt,PetscInt,PetscInt);
extern PetscErrorCode  MatSetType(Mat,const MatType);
extern PetscErrorCode  MatSetFromOptions(Mat);
extern PetscErrorCode  MatSetUpPreallocation(Mat);
extern PetscErrorCode  MatRegisterAll(const char[]);
extern PetscErrorCode  MatRegister(const char[],const char[],const char[],PetscErrorCode(*)(Mat));
extern PetscErrorCode  MatRegisterBaseName(const char[],const char[],const char[]);
extern PetscErrorCode  MatSetOptionsPrefix(Mat,const char[]);
extern PetscErrorCode  MatAppendOptionsPrefix(Mat,const char[]);
extern PetscErrorCode  MatGetOptionsPrefix(Mat,const char*[]);

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

extern PetscBool  MatRegisterAllCalled;
extern PetscFList MatList;
extern PetscFList MatColoringList;
extern PetscFList MatPartitioningList;

/*E
    MatStructure - Indicates if the matrix has the same nonzero structure

    Level: beginner

   Any additions/changes here MUST also be made in include/finclude/petscmat.h

.seealso: MatCopy(), KSPSetOperators(), PCSetOperators()
E*/
typedef enum {DIFFERENT_NONZERO_PATTERN,SUBSET_NONZERO_PATTERN,SAME_NONZERO_PATTERN,SAME_PRECONDITIONER} MatStructure;

extern PetscErrorCode  MatCreateSeqDense(MPI_Comm,PetscInt,PetscInt,PetscScalar[],Mat*);
extern PetscErrorCode  MatCreateMPIDense(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscScalar[],Mat*); 
extern PetscErrorCode  MatCreateSeqAIJ(MPI_Comm,PetscInt,PetscInt,PetscInt,const PetscInt[],Mat*);
PetscPolymorphicFunction(MatCreateSeqAIJ,(PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[]),(PETSC_COMM_SELF,m,n,nz,nnz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateSeqAIJ,(PetscInt m,PetscInt n,PetscInt nz),(PETSC_COMM_SELF,m,n,nz,PETSC_NULL,&A),Mat,A)
PetscPolymorphicFunction(MatCreateSeqAIJ,(PetscInt m,PetscInt n,const PetscInt nnz[]),(PETSC_COMM_SELF,m,n,0,nnz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateSeqAIJ,(PetscInt m,PetscInt n),(PETSC_COMM_SELF,m,n,0,PETSC_NULL,&A),Mat,A)
PetscPolymorphicSubroutine(MatCreateSeqAIJ,(PetscInt m,PetscInt n,PetscInt nz,Mat *A),(PETSC_COMM_SELF,m,n,nz,PETSC_NULL,A))
PetscPolymorphicSubroutine(MatCreateSeqAIJ,(PetscInt m,PetscInt n,const PetscInt nnz[],Mat *A),(PETSC_COMM_SELF,m,n,0,nnz,A))
PetscPolymorphicSubroutine(MatCreateSeqAIJ,(PetscInt m,PetscInt n,Mat *A),(PETSC_COMM_SELF,m,n,0,PETSC_NULL,A))
extern PetscErrorCode  MatCreateMPIAIJ(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Mat*); 
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
extern PetscErrorCode  MatCreateMPIAIJWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[],Mat *);
extern PetscErrorCode  MatCreateMPIAIJWithSplitArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt[],PetscInt[],PetscScalar[],PetscInt[],PetscInt[],PetscScalar[],Mat*);

extern PetscErrorCode  MatCreateSeqBAIJ(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],Mat*); 
PetscPolymorphicFunction(MatCreateSeqBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[]),(PETSC_COMM_SELF,bs,m,n,nz,nnz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateSeqBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt nz),(PETSC_COMM_SELF,bs,m,n,nz,PETSC_NULL,&A),Mat,A)
PetscPolymorphicFunction(MatCreateSeqBAIJ,(PetscInt bs,PetscInt m,PetscInt n,const PetscInt nnz[]),(PETSC_COMM_SELF,bs,m,n,0,nnz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateSeqBAIJ,(PetscInt bs,PetscInt m,PetscInt n),(PETSC_COMM_SELF,bs,m,n,0,PETSC_NULL,&A),Mat,A)
PetscPolymorphicSubroutine(MatCreateSeqBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt nz,Mat *A),(PETSC_COMM_SELF,bs,m,n,nz,PETSC_NULL,A))
PetscPolymorphicSubroutine(MatCreateSeqBAIJ,(PetscInt bs,PetscInt m,PetscInt n,const PetscInt nnz[],Mat *A),(PETSC_COMM_SELF,bs,m,n,0,nnz,A))
PetscPolymorphicSubroutine(MatCreateSeqBAIJ,(PetscInt bs,PetscInt m,PetscInt n,Mat *A),(PETSC_COMM_SELF,bs,m,n,0,PETSC_NULL,A))
extern PetscErrorCode  MatCreateMPIBAIJ(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Mat*);
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
extern PetscErrorCode  MatCreateMPIBAIJWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[],Mat*);

extern PetscErrorCode  MatCreateMPIAdj(MPI_Comm,PetscInt,PetscInt,PetscInt[],PetscInt[],PetscInt[],Mat*);
extern PetscErrorCode  MatCreateSeqSBAIJ(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],Mat*); 
PetscPolymorphicFunction(MatCreateSeqSBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[]),(PETSC_COMM_SELF,bs,m,n,nz,nnz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateSeqSBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt nz),(PETSC_COMM_SELF,bs,m,n,nz,PETSC_NULL,&A),Mat,A)
PetscPolymorphicFunction(MatCreateSeqSBAIJ,(PetscInt bs,PetscInt m,PetscInt n,const PetscInt nnz[]),(PETSC_COMM_SELF,bs,m,n,0,nnz,&A),Mat,A)
PetscPolymorphicFunction(MatCreateSeqSBAIJ,(PetscInt bs,PetscInt m,PetscInt n),(PETSC_COMM_SELF,bs,m,n,0,PETSC_NULL,&A),Mat,A)
PetscPolymorphicSubroutine(MatCreateSeqSBAIJ,(PetscInt bs,PetscInt m,PetscInt n,PetscInt nz,Mat *A),(PETSC_COMM_SELF,bs,m,n,nz,PETSC_NULL,A))
PetscPolymorphicSubroutine(MatCreateSeqSBAIJ,(PetscInt bs,PetscInt m,PetscInt n,const PetscInt nnz[],Mat *A),(PETSC_COMM_SELF,bs,m,n,0,nnz,A))
PetscPolymorphicSubroutine(MatCreateSeqSBAIJ,(PetscInt bs,PetscInt m,PetscInt n,Mat *A),(PETSC_COMM_SELF,bs,m,n,0,PETSC_NULL,A))

extern PetscErrorCode  MatCreateMPISBAIJ(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Mat*);
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
extern PetscErrorCode  MatCreateMPISBAIJWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[],Mat *);
extern PetscErrorCode  MatMPISBAIJSetPreallocationCSR(Mat,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[]);

extern PetscErrorCode  MatCreateShell(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,void *,Mat*);
PetscPolymorphicFunction(MatCreateShell,(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,void *ctx),(comm,m,n,M,N,ctx,&A),Mat,A)
PetscPolymorphicFunction(MatCreateShell,(PetscInt m,PetscInt n,PetscInt M,PetscInt N,void *ctx),(PETSC_COMM_WORLD,m,n,M,N,ctx,&A),Mat,A)
extern PetscErrorCode  MatCreateAdic(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,void (*)(void),Mat*);
extern PetscErrorCode  MatCreateNormal(Mat,Mat*);
PetscPolymorphicFunction(MatCreateNormal,(Mat mat),(mat,&A),Mat,A)
extern PetscErrorCode  MatCreateLRC(Mat,Mat,Mat,Mat*);
extern PetscErrorCode  MatCreateIS(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,ISLocalToGlobalMapping,Mat*);
extern PetscErrorCode  MatCreateSeqAIJCRL(MPI_Comm,PetscInt,PetscInt,PetscInt,const PetscInt[],Mat*);
extern PetscErrorCode  MatCreateMPIAIJCRL(MPI_Comm,PetscInt,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Mat*);

extern PetscErrorCode  MatCreateSeqBSTRM(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],Mat*);
extern PetscErrorCode  MatCreateMPIBSTRM(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Mat*);
extern PetscErrorCode  MatCreateSeqSBSTRM(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],Mat*);
extern PetscErrorCode  MatCreateMPISBSTRM(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Mat*);

extern PetscErrorCode  MatCreateScatter(MPI_Comm,VecScatter,Mat*);
extern PetscErrorCode  MatScatterSetVecScatter(Mat,VecScatter);
extern PetscErrorCode  MatScatterGetVecScatter(Mat,VecScatter*);
extern PetscErrorCode  MatCreateBlockMat(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,Mat*);
extern PetscErrorCode  MatCompositeAddMat(Mat,Mat);
extern PetscErrorCode  MatCompositeMerge(Mat);
extern PetscErrorCode  MatCreateComposite(MPI_Comm,PetscInt,const Mat*,Mat*);
typedef enum {MAT_COMPOSITE_ADDITIVE,MAT_COMPOSITE_MULTIPLICATIVE} MatCompositeType;
extern PetscErrorCode  MatCompositeSetType(Mat,MatCompositeType);

extern PetscErrorCode  MatCreateFFT(MPI_Comm,PetscInt,const PetscInt[],const MatType,Mat*);
extern PetscErrorCode  MatCreateSeqCUFFT(MPI_Comm,PetscInt,const PetscInt[],Mat*);

extern PetscErrorCode  MatCreateTranspose(Mat,Mat*);
extern PetscErrorCode  MatCreateSubMatrix(Mat,IS,IS,Mat*);
extern PetscErrorCode  MatSubMatrixUpdate(Mat,Mat,IS,IS);
extern PetscErrorCode  MatCreateLocalRef(Mat,IS,IS,Mat*);

extern PetscErrorCode  MatPythonSetType(Mat,const char[]);

extern PetscErrorCode  MatSetUp(Mat);
extern PetscErrorCode  MatDestroy(Mat*);

extern PetscErrorCode  MatConjugate(Mat);
extern PetscErrorCode  MatRealPart(Mat);
extern PetscErrorCode  MatImaginaryPart(Mat);
extern PetscErrorCode  MatGetDiagonalBlock(Mat,Mat*);
extern PetscErrorCode  MatGetTrace(Mat,PetscScalar*);
extern PetscErrorCode  MatInvertBlockDiagonal(Mat,PetscScalar **);

/* ------------------------------------------------------------*/
extern PetscErrorCode  MatSetValues(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
extern PetscErrorCode  MatSetValuesBlocked(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
extern PetscErrorCode  MatSetValuesRow(Mat,PetscInt,const PetscScalar[]);
extern PetscErrorCode  MatSetValuesRowLocal(Mat,PetscInt,const PetscScalar[]);
extern PetscErrorCode  MatSetValuesBatch(Mat,PetscInt,PetscInt,PetscInt[],const PetscScalar[]);

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

extern PetscErrorCode  MatSetValuesStencil(Mat,PetscInt,const MatStencil[],PetscInt,const MatStencil[],const PetscScalar[],InsertMode);
extern PetscErrorCode  MatSetValuesBlockedStencil(Mat,PetscInt,const MatStencil[],PetscInt,const MatStencil[],const PetscScalar[],InsertMode);
extern PetscErrorCode  MatSetStencil(Mat,PetscInt,const PetscInt[],const PetscInt[],PetscInt);

extern PetscErrorCode  MatSetColoring(Mat,ISColoring);
extern PetscErrorCode  MatSetValuesAdic(Mat,void*);
extern PetscErrorCode  MatSetValuesAdifor(Mat,PetscInt,void*);

/*E
    MatAssemblyType - Indicates if the matrix is now to be used, or if you plan 
     to continue to add values to it

    Level: beginner

.seealso: MatAssemblyBegin(), MatAssemblyEnd()
E*/
typedef enum {MAT_FLUSH_ASSEMBLY=1,MAT_FINAL_ASSEMBLY=0} MatAssemblyType;
extern PetscErrorCode  MatAssemblyBegin(Mat,MatAssemblyType);
extern PetscErrorCode  MatAssemblyEnd(Mat,MatAssemblyType);
extern PetscErrorCode  MatAssembled(Mat,PetscBool *);



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
              MAT_CHECK_COMPRESSED_ROW,
              MAT_IGNORE_LOWER_TRIANGULAR,MAT_ERROR_LOWER_TRIANGULAR,
              MAT_GETROW_UPPERTRIANGULAR,MAT_UNUSED_NONZERO_LOCATION_ERR,
              MAT_SPD,MAT_NO_OFF_PROC_ENTRIES,MAT_NO_OFF_PROC_ZERO_ROWS,
              NUM_MAT_OPTIONS} MatOption;
extern const char *MatOptions[];
extern PetscErrorCode  MatSetOption(Mat,MatOption,PetscBool );
extern PetscErrorCode  MatGetType(Mat,const MatType*);
PetscPolymorphicFunction(MatGetType,(Mat mat),(mat,&t),const MatType,t)

extern PetscErrorCode  MatGetValues(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],PetscScalar[]);
extern PetscErrorCode  MatGetRow(Mat,PetscInt,PetscInt *,const PetscInt *[],const PetscScalar*[]);
extern PetscErrorCode  MatRestoreRow(Mat,PetscInt,PetscInt *,const PetscInt *[],const PetscScalar*[]);
extern PetscErrorCode  MatGetRowUpperTriangular(Mat);
extern PetscErrorCode  MatRestoreRowUpperTriangular(Mat);
extern PetscErrorCode  MatGetColumn(Mat,PetscInt,PetscInt *,const PetscInt *[],const PetscScalar*[]);
extern PetscErrorCode  MatRestoreColumn(Mat,PetscInt,PetscInt *,const PetscInt *[],const PetscScalar*[]);
extern PetscErrorCode  MatGetColumnVector(Mat,Vec,PetscInt);
extern PetscErrorCode  MatGetArray(Mat,PetscScalar *[]);
PetscPolymorphicFunction(MatGetArray,(Mat mat),(mat,&a),PetscScalar*,a)
extern PetscErrorCode  MatRestoreArray(Mat,PetscScalar *[]);
extern PetscErrorCode  MatGetBlockSize(Mat,PetscInt *);
PetscPolymorphicFunction(MatGetBlockSize,(Mat mat),(mat,&a),PetscInt,a)
extern PetscErrorCode  MatSetBlockSize(Mat,PetscInt);


extern PetscErrorCode  MatMult(Mat,Vec,Vec);
extern PetscErrorCode  MatMultDiagonalBlock(Mat,Vec,Vec);
extern PetscErrorCode  MatMultAdd(Mat,Vec,Vec,Vec);
PetscPolymorphicSubroutine(MatMultAdd,(Mat A,Vec x,Vec y),(A,x,y,y))
extern PetscErrorCode  MatMultTranspose(Mat,Vec,Vec);
extern PetscErrorCode  MatMultHermitianTranspose(Mat,Vec,Vec);
extern PetscErrorCode  MatIsTranspose(Mat,Mat,PetscReal,PetscBool *);
PetscPolymorphicFunction(MatIsTranspose,(Mat A,Mat B,PetscReal tol),(A,B,tol,&t),PetscBool ,t)
PetscPolymorphicFunction(MatIsTranspose,(Mat A,Mat B),(A,B,0,&t),PetscBool ,t)
extern PetscErrorCode  MatIsHermitianTranspose(Mat,Mat,PetscReal,PetscBool *);
extern PetscErrorCode  MatMultTransposeAdd(Mat,Vec,Vec,Vec);
extern PetscErrorCode  MatMultHermitianTransposeAdd(Mat,Vec,Vec,Vec);
PetscPolymorphicSubroutine(MatMultTransposeAdd,(Mat A,Vec x,Vec y),(A,x,y,y))
extern PetscErrorCode  MatMultConstrained(Mat,Vec,Vec);
extern PetscErrorCode  MatMultTransposeConstrained(Mat,Vec,Vec);
extern PetscErrorCode  MatMatSolve(Mat,Mat,Mat);

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

extern PetscErrorCode  MatConvert(Mat,const MatType,MatReuse,Mat*);
PetscPolymorphicFunction(MatConvert,(Mat A,const MatType t),(A,t,MAT_INITIAL_MATRIX,&a),Mat,a)
extern PetscErrorCode  MatDuplicate(Mat,MatDuplicateOption,Mat*);
PetscPolymorphicFunction(MatDuplicate,(Mat A,MatDuplicateOption o),(A,o,&a),Mat,a)
PetscPolymorphicFunction(MatDuplicate,(Mat A),(A,MAT_COPY_VALUES,&a),Mat,a)


extern PetscErrorCode  MatCopy(Mat,Mat,MatStructure);
extern PetscErrorCode  MatView(Mat,PetscViewer);
extern PetscErrorCode  MatIsSymmetric(Mat,PetscReal,PetscBool *);
PetscPolymorphicFunction(MatIsSymmetric,(Mat A,PetscReal tol),(A,tol,&t),PetscBool ,t)
PetscPolymorphicFunction(MatIsSymmetric,(Mat A),(A,0,&t),PetscBool ,t)
extern PetscErrorCode  MatIsStructurallySymmetric(Mat,PetscBool *);
PetscPolymorphicFunction(MatIsStructurallySymmetric,(Mat A),(A,&t),PetscBool ,t)
extern PetscErrorCode  MatIsHermitian(Mat,PetscReal,PetscBool *);
PetscPolymorphicFunction(MatIsHermitian,(Mat A),(A,0,&t),PetscBool ,t)
extern PetscErrorCode  MatIsSymmetricKnown(Mat,PetscBool *,PetscBool *);
extern PetscErrorCode  MatIsHermitianKnown(Mat,PetscBool *,PetscBool *);
extern PetscErrorCode  MatMissingDiagonal(Mat,PetscBool  *,PetscInt *);
extern PetscErrorCode  MatLoad(Mat, PetscViewer);

extern PetscErrorCode  MatGetRowIJ(Mat,PetscInt,PetscBool ,PetscBool ,PetscInt*,PetscInt *[],PetscInt *[],PetscBool  *);
extern PetscErrorCode  MatRestoreRowIJ(Mat,PetscInt,PetscBool ,PetscBool ,PetscInt *,PetscInt *[],PetscInt *[],PetscBool  *);
extern PetscErrorCode  MatGetColumnIJ(Mat,PetscInt,PetscBool ,PetscBool ,PetscInt*,PetscInt *[],PetscInt *[],PetscBool  *);
extern PetscErrorCode  MatRestoreColumnIJ(Mat,PetscInt,PetscBool ,PetscBool ,PetscInt *,PetscInt *[],PetscInt *[],PetscBool  *);

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
extern PetscErrorCode  MatGetInfo(Mat,MatInfoType,MatInfo*);
extern PetscErrorCode  MatGetDiagonal(Mat,Vec);
extern PetscErrorCode  MatGetRowMax(Mat,Vec,PetscInt[]);
extern PetscErrorCode  MatGetRowMin(Mat,Vec,PetscInt[]);
extern PetscErrorCode  MatGetRowMaxAbs(Mat,Vec,PetscInt[]);
extern PetscErrorCode  MatGetRowMinAbs(Mat,Vec,PetscInt[]);
extern PetscErrorCode  MatGetRowSum(Mat,Vec);
extern PetscErrorCode  MatTranspose(Mat,MatReuse,Mat*);
PetscPolymorphicFunction(MatTranspose,(Mat A),(A,MAT_INITIAL_MATRIX,&t),Mat,t)
extern PetscErrorCode  MatHermitianTranspose(Mat,MatReuse,Mat*);
extern PetscErrorCode  MatPermute(Mat,IS,IS,Mat *);
PetscPolymorphicFunction(MatPermute,(Mat A,IS is1,IS is2),(A,is1,is2,&t),Mat,t)
extern PetscErrorCode  MatDiagonalScale(Mat,Vec,Vec);
extern PetscErrorCode  MatDiagonalSet(Mat,Vec,InsertMode);
extern PetscErrorCode  MatEqual(Mat,Mat,PetscBool *);
PetscPolymorphicFunction(MatEqual,(Mat A,Mat B),(A,B,&t),PetscBool ,t)
extern PetscErrorCode  MatMultEqual(Mat,Mat,PetscInt,PetscBool *);
extern PetscErrorCode  MatMultAddEqual(Mat,Mat,PetscInt,PetscBool *);
extern PetscErrorCode  MatMultTransposeEqual(Mat,Mat,PetscInt,PetscBool *);
extern PetscErrorCode  MatMultTransposeAddEqual(Mat,Mat,PetscInt,PetscBool *);

extern PetscErrorCode  MatNorm(Mat,NormType,PetscReal *);
PetscPolymorphicFunction(MatNorm,(Mat A,NormType t),(A,t,&n),PetscReal,n)
extern PetscErrorCode  MatGetColumnNorms(Mat,NormType,PetscReal *);
extern PetscErrorCode  MatZeroEntries(Mat);
extern PetscErrorCode  MatZeroRows(Mat,PetscInt,const PetscInt [],PetscScalar,Vec,Vec);
extern PetscErrorCode  MatZeroRowsIS(Mat,IS,PetscScalar,Vec,Vec);
extern PetscErrorCode  MatZeroRowsStencil(Mat,PetscInt,const MatStencil [],PetscScalar,Vec,Vec);
extern PetscErrorCode  MatZeroRowsColumns(Mat,PetscInt,const PetscInt [],PetscScalar,Vec,Vec);
extern PetscErrorCode  MatZeroRowsColumnsIS(Mat,IS,PetscScalar,Vec,Vec);

extern PetscErrorCode  MatUseScaledForm(Mat,PetscBool );
extern PetscErrorCode  MatScaleSystem(Mat,Vec,Vec);
extern PetscErrorCode  MatUnScaleSystem(Mat,Vec,Vec);

extern PetscErrorCode  MatGetSize(Mat,PetscInt*,PetscInt*);
extern PetscErrorCode  MatGetLocalSize(Mat,PetscInt*,PetscInt*);
extern PetscErrorCode  MatGetOwnershipRange(Mat,PetscInt*,PetscInt*);
extern PetscErrorCode  MatGetOwnershipRanges(Mat,const PetscInt**);
extern PetscErrorCode  MatGetOwnershipRangeColumn(Mat,PetscInt*,PetscInt*);
extern PetscErrorCode  MatGetOwnershipRangesColumn(Mat,const PetscInt**);

extern PetscErrorCode  MatGetSubMatrices(Mat,PetscInt,const IS[],const IS[],MatReuse,Mat *[]);
extern PetscErrorCode  MatGetSubMatricesParallel(Mat,PetscInt,const IS[],const IS[],MatReuse,Mat *[]);
extern PetscErrorCode  MatDestroyMatrices(PetscInt,Mat *[]);
extern PetscErrorCode  MatGetSubMatrix(Mat,IS,IS,MatReuse,Mat *);
extern PetscErrorCode  MatGetLocalSubMatrix(Mat,IS,IS,Mat*);
extern PetscErrorCode  MatRestoreLocalSubMatrix(Mat,IS,IS,Mat*);
extern PetscErrorCode  MatGetSeqNonzeroStructure(Mat,Mat*);
extern PetscErrorCode  MatDestroySeqNonzeroStructure(Mat*); 

extern PetscErrorCode  MatMerge(MPI_Comm,Mat,PetscInt,MatReuse,Mat*);
extern PetscErrorCode  MatMerge_SeqsToMPI(MPI_Comm,Mat,PetscInt,PetscInt,MatReuse,Mat*);
extern PetscErrorCode  MatMerge_SeqsToMPISymbolic(MPI_Comm,Mat,PetscInt,PetscInt,Mat*);
extern PetscErrorCode  MatMerge_SeqsToMPINumeric(Mat,Mat); 
extern PetscErrorCode  MatMPIAIJGetLocalMat(Mat,MatReuse,Mat*);
extern PetscErrorCode  MatMPIAIJGetLocalMatCondensed(Mat,MatReuse,IS*,IS*,Mat*);
extern PetscErrorCode  MatGetBrowsOfAcols(Mat,Mat,MatReuse,IS*,IS*,PetscInt*,Mat*);
extern PetscErrorCode  MatGetBrowsOfAoCols(Mat,Mat,MatReuse,PetscInt**,PetscInt**,MatScalar**,Mat*);
#if defined (PETSC_USE_CTABLE)
#include "petscctable.h"
extern PetscErrorCode  MatGetCommunicationStructs(Mat, Vec *, PetscTable *, VecScatter *);
#else
extern PetscErrorCode  MatGetCommunicationStructs(Mat, Vec *, PetscInt *[], VecScatter *);
#endif
extern PetscErrorCode  MatGetGhosts(Mat, PetscInt *,const PetscInt *[]);

extern PetscErrorCode  MatIncreaseOverlap(Mat,PetscInt,IS[],PetscInt);

extern PetscErrorCode  MatMatMult(Mat,Mat,MatReuse,PetscReal,Mat*);
extern PetscErrorCode  MatMatMultSymbolic(Mat,Mat,PetscReal,Mat*);
extern PetscErrorCode  MatMatMultNumeric(Mat,Mat,Mat);

extern PetscErrorCode  MatPtAP(Mat,Mat,MatReuse,PetscReal,Mat*);
extern PetscErrorCode  MatPtAPSymbolic(Mat,Mat,PetscReal,Mat*);
extern PetscErrorCode  MatPtAPNumeric(Mat,Mat,Mat);

extern PetscErrorCode  MatMatMultTranspose(Mat,Mat,MatReuse,PetscReal,Mat*);
extern PetscErrorCode  MatMatMultTransposeSymbolic(Mat,Mat,PetscReal,Mat*);
extern PetscErrorCode  MatMatMultTransposeNumeric(Mat,Mat,Mat);

extern PetscErrorCode  MatAXPY(Mat,PetscScalar,Mat,MatStructure);
extern PetscErrorCode  MatAYPX(Mat,PetscScalar,Mat,MatStructure);

extern PetscErrorCode  MatScale(Mat,PetscScalar);
extern PetscErrorCode  MatShift(Mat,PetscScalar);

extern PetscErrorCode  MatSetLocalToGlobalMapping(Mat,ISLocalToGlobalMapping,ISLocalToGlobalMapping);
extern PetscErrorCode  MatSetLocalToGlobalMappingBlock(Mat,ISLocalToGlobalMapping,ISLocalToGlobalMapping);
extern PetscErrorCode  MatGetLocalToGlobalMapping(Mat,ISLocalToGlobalMapping*,ISLocalToGlobalMapping*);
extern PetscErrorCode  MatGetLocalToGlobalMappingBlock(Mat,ISLocalToGlobalMapping*,ISLocalToGlobalMapping*);
extern PetscErrorCode  MatZeroRowsLocal(Mat,PetscInt,const PetscInt [],PetscScalar,Vec,Vec);
extern PetscErrorCode  MatZeroRowsLocalIS(Mat,IS,PetscScalar,Vec,Vec);
extern PetscErrorCode  MatZeroRowsColumnsLocal(Mat,PetscInt,const PetscInt [],PetscScalar,Vec,Vec);
extern PetscErrorCode  MatZeroRowsColumnsLocalIS(Mat,IS,PetscScalar,Vec,Vec);
extern PetscErrorCode  MatSetValuesLocal(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
extern PetscErrorCode  MatSetValuesBlockedLocal(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);

extern PetscErrorCode  MatStashSetInitialSize(Mat,PetscInt,PetscInt);
extern PetscErrorCode  MatStashGetInfo(Mat,PetscInt*,PetscInt*,PetscInt*,PetscInt*);

extern PetscErrorCode  MatInterpolate(Mat,Vec,Vec);
extern PetscErrorCode  MatInterpolateAdd(Mat,Vec,Vec,Vec);
PetscPolymorphicSubroutine(MatInterpolateAdd,(Mat A,Vec x,Vec y),(A,x,y,y))
extern PetscErrorCode  MatRestrict(Mat,Vec,Vec);
extern PetscErrorCode  MatGetVecs(Mat,Vec*,Vec*);
extern PetscErrorCode  MatGetRedundantMatrix(Mat,PetscInt,MPI_Comm,PetscInt,MatReuse,Mat*);
extern PetscErrorCode  MatGetMultiProcBlock(Mat,MPI_Comm,Mat*);
extern PetscErrorCode  MatFindZeroDiagonals(Mat,IS*);

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
    See the <A href="../../docs/manual.pdf#nameddest=ch_performance">Hints for Performance Improvment</A> chapter in the users manual for more details.

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
    See the <A href="../../docs/manual.pdf#nameddest=ch_performance">Hints for Performance Improvment</A> chapter in the users manual for more details.

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
+  map - the row mapping from local numbering to global numbering
.  nrows - the number of rows indicated
.  rows - the indices of the rows
.  cmap - the column mapping from local to global numbering
.  ncols - the number of columns in the matrix
.  cols - the columns indicated
.  dnz - the array that will be passed to the matrix preallocation routines
-  ozn - the other array passed to the matrix preallocation routines


   Level: intermediate

   Notes:
    See the <A href="../../docs/manual.pdf#nameddest=ch_performance">Hints for Performance Improvment</A> chapter in the users manual for more details.

   Do not malloc or free dnz and onz, that is handled internally by these routines

  Concepts: preallocation^Matrix

.seealso: MatPreallocateFinalize(), MatPreallocateSet(), MatPreallocateSymmetricSet(), MatPreallocateInitialize(),
          MatPreallocateInitialize(), MatPreallocateSymmetricSetLocal()
M*/
#define MatPreallocateSetLocal(rmap,nrows,rows,cmap,ncols,cols,dnz,onz) 0; \
{\
  PetscInt __l;\
  _4_ierr = ISLocalToGlobalMappingApply(rmap,nrows,rows,rows);CHKERRQ(_4_ierr);\
  _4_ierr = ISLocalToGlobalMappingApply(cmap,ncols,cols,cols);CHKERRQ(_4_ierr);\
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
    See the <A href="../../docs/manual.pdf#nameddest=ch_performance">Hints for Performance Improvment</A> chapter in the users manual for more details.

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
    See the <A href="../../docs/manual.pdf#nameddest=ch_performance">Hints for Performance Improvment</A> chapter in the users manual for more details.

   Do not malloc or free dnz and onz that is handled internally by these routines

   This is a MACRO not a function because it uses variables declared in MatPreallocateInitialize().

  Concepts: preallocation^Matrix

.seealso: MatPreallocateFinalize(), MatPreallocateSet(), MatPreallocateSymmetricSet(), MatPreallocateInitialize(),
          MatPreallocateInitialize(), MatPreallocateSymmetricSetLocal(), MatPreallocateSetLocal()
M*/
#define MatPreallocateSet(row,nc,cols,dnz,onz) 0;\
{ PetscInt __i; \
  if (row < __rstart) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Trying to set preallocation for row %D less than first local row %D",row,__rstart);\
  if (row >= __rstart+__nrows) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Trying to set preallocation for row %D greater than last local row %D",row,__rstart+__nrows-1);\
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
    See the <A href="../../docs/manual.pdf#nameddest=ch_performance">Hints for Performance Improvment</A> chapter in the users manual for more details.

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
    See the <A href="../../docs/manual.pdf#nameddest=ch_performance">Hints for Performance Improvment</A> chapter in the users manual for more details.

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
    See the <A href="../../docs/manual.pdf#nameddest=ch_performance">Hints for Performance Improvment</A> chapter in the users manual for more details.

   Do not malloc or free dnz and onz that is handled internally by these routines

   This is a MACRO not a function because it closes the { started in MatPreallocateInitialize().

  Concepts: preallocation^Matrix

.seealso: MatPreallocateInitialize(), MatPreallocateSet(), MatPreallocateSymmetricSet(), MatPreallocateSetLocal(),
          MatPreallocateSymmetricInitialize(), MatPreallocateSymmetricSetLocal()
M*/
#define MatPreallocateFinalize(dnz,onz) 0;_4_ierr = PetscFree2(dnz,onz);CHKERRQ(_4_ierr);}



/* Routines unique to particular data structures */
extern PetscErrorCode  MatShellGetContext(Mat,void *);
PetscPolymorphicFunction(MatShellGetContext,(Mat A),(A,&t),void*,t)

extern PetscErrorCode  MatInodeAdjustForInodes(Mat,IS*,IS*);
extern PetscErrorCode  MatInodeGetInodeSizes(Mat,PetscInt *,PetscInt *[],PetscInt *);

extern PetscErrorCode  MatSeqAIJSetColumnIndices(Mat,PetscInt[]);
extern PetscErrorCode  MatSeqBAIJSetColumnIndices(Mat,PetscInt[]);
extern PetscErrorCode  MatCreateSeqAIJWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt[],PetscInt[],PetscScalar[],Mat*);
extern PetscErrorCode  MatCreateSeqBAIJWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt[],PetscInt[],PetscScalar[],Mat*);
extern PetscErrorCode  MatCreateSeqSBAIJWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt[],PetscInt[],PetscScalar[],Mat*);

#define MAT_SKIP_ALLOCATION -4

extern PetscErrorCode  MatSeqBAIJSetPreallocation(Mat,PetscInt,PetscInt,const PetscInt[]);
PetscPolymorphicSubroutine(MatSeqBAIJSetPreallocation,(Mat A,PetscInt bs,const PetscInt nnz[]),(A,bs,0,nnz))
extern PetscErrorCode  MatSeqSBAIJSetPreallocation(Mat,PetscInt,PetscInt,const PetscInt[]);
PetscPolymorphicSubroutine(MatSeqSBAIJSetPreallocation,(Mat A,PetscInt bs,const PetscInt nnz[]),(A,bs,0,nnz))
extern PetscErrorCode  MatSeqAIJSetPreallocation(Mat,PetscInt,const PetscInt[]);
PetscPolymorphicSubroutine(MatSeqAIJSetPreallocation,(Mat A,const PetscInt nnz[]),(A,0,nnz))

extern PetscErrorCode  MatMPIBAIJSetPreallocation(Mat,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);
PetscPolymorphicSubroutine(MatMPIBAIJSetPreallocation,(Mat A,PetscInt bs,const PetscInt nnz[],const PetscInt onz[]),(A,bs,0,nnz,0,onz))
extern PetscErrorCode  MatMPISBAIJSetPreallocation(Mat,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);
extern PetscErrorCode  MatMPIAIJSetPreallocation(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);
extern PetscErrorCode  MatSeqAIJSetPreallocationCSR(Mat,const PetscInt [],const PetscInt [],const PetscScalar []);
extern PetscErrorCode  MatSeqBAIJSetPreallocationCSR(Mat,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[]);
extern PetscErrorCode  MatMPIAIJSetPreallocationCSR(Mat,const PetscInt[],const PetscInt[],const PetscScalar[]);
extern PetscErrorCode  MatMPIBAIJSetPreallocationCSR(Mat,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[]);
extern PetscErrorCode  MatMPIAdjSetPreallocation(Mat,PetscInt[],PetscInt[],PetscInt[]);
extern PetscErrorCode  MatMPIDenseSetPreallocation(Mat,PetscScalar[]);
extern PetscErrorCode  MatSeqDenseSetPreallocation(Mat,PetscScalar[]);
extern PetscErrorCode  MatMPIAIJGetSeqAIJ(Mat,Mat*,Mat*,PetscInt*[]);
extern PetscErrorCode  MatMPIBAIJGetSeqBAIJ(Mat,Mat*,Mat*,PetscInt*[]);
extern PetscErrorCode  MatAdicSetLocalFunction(Mat,void (*)(void));

extern PetscErrorCode  MatSeqDenseSetLDA(Mat,PetscInt);
extern PetscErrorCode  MatDenseGetLocalMatrix(Mat,Mat*);

extern PetscErrorCode  MatStoreValues(Mat);
extern PetscErrorCode  MatRetrieveValues(Mat);

extern PetscErrorCode  MatDAADSetCtx(Mat,void*);

extern PetscErrorCode  MatFindNonzeroRows(Mat,IS*);
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
#define MATORDERINGNATURAL     "natural"
#define MATORDERINGND          "nd"
#define MATORDERING1WD         "1wd"
#define MATORDERINGRCM         "rcm"
#define MATORDERINGQMD         "qmd"
#define MATORDERINGROWLENGTH   "rowlength"
#define MATORDERINGDSC_ND      "dsc_nd"         /* these three are only for DSCPACK, see its documentation for details */
#define MATORDERINGDSC_MMD     "dsc_mmd"
#define MATORDERINGDSC_MDF     "dsc_mdf"
#define MATORDERINGAMD         "amd"            /* only works if UMFPACK is installed with PETSc */

extern PetscErrorCode  MatGetOrdering(Mat,const MatOrderingType,IS*,IS*);
extern PetscErrorCode  MatGetOrderingList(PetscFList *list);
extern PetscErrorCode  MatOrderingRegister(const char[],const char[],const char[],PetscErrorCode(*)(Mat,const MatOrderingType,IS*,IS*));

/*MC
   MatOrderingRegisterDynamic - Adds a new sparse matrix ordering to the matrix package. 

   Synopsis:
   PetscErrorCode MatOrderingRegisterDynamic(const char *name_ordering,const char *path,const char *name_create,PetscErrorCode (*routine_create)(MatOrdering))

   Not Collective

   Input Parameters:
+  sname - name of ordering (for example MATORDERINGND)
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

extern PetscErrorCode  MatOrderingRegisterDestroy(void);
extern PetscErrorCode  MatOrderingRegisterAll(const char[]);
extern PetscBool  MatOrderingRegisterAllCalled;
extern PetscFList MatOrderingList;

extern PetscErrorCode  MatReorderForNonzeroDiagonal(Mat,PetscReal,IS,IS);

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

extern PetscErrorCode  MatFactorInfoInitialize(MatFactorInfo*);
extern PetscErrorCode  MatCholeskyFactor(Mat,IS,const MatFactorInfo*);
extern PetscErrorCode  MatCholeskyFactorSymbolic(Mat,Mat,IS,const MatFactorInfo*);
extern PetscErrorCode  MatCholeskyFactorNumeric(Mat,Mat,const MatFactorInfo*);
extern PetscErrorCode  MatLUFactor(Mat,IS,IS,const MatFactorInfo*);
extern PetscErrorCode  MatILUFactor(Mat,IS,IS,const MatFactorInfo*);
extern PetscErrorCode  MatLUFactorSymbolic(Mat,Mat,IS,IS,const MatFactorInfo*);
extern PetscErrorCode  MatILUFactorSymbolic(Mat,Mat,IS,IS,const MatFactorInfo*);
extern PetscErrorCode  MatICCFactorSymbolic(Mat,Mat,IS,const MatFactorInfo*);
extern PetscErrorCode  MatICCFactor(Mat,IS,const MatFactorInfo*);
extern PetscErrorCode  MatLUFactorNumeric(Mat,Mat,const MatFactorInfo*);
extern PetscErrorCode  MatGetInertia(Mat,PetscInt*,PetscInt*,PetscInt*);
extern PetscErrorCode  MatSolve(Mat,Vec,Vec);
extern PetscErrorCode  MatForwardSolve(Mat,Vec,Vec);
extern PetscErrorCode  MatBackwardSolve(Mat,Vec,Vec);
extern PetscErrorCode  MatSolveAdd(Mat,Vec,Vec,Vec);
extern PetscErrorCode  MatSolveTranspose(Mat,Vec,Vec);
extern PetscErrorCode  MatSolveTransposeAdd(Mat,Vec,Vec,Vec);
extern PetscErrorCode  MatSolves(Mat,Vecs,Vecs);

extern PetscErrorCode  MatSetUnfactored(Mat);

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
extern PetscErrorCode  MatSOR(Mat,Vec,PetscReal,MatSORType,PetscReal,PetscInt,PetscInt,Vec);

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
#define MATCOLORINGNATURAL "natural"
#define MATCOLORINGSL      "sl"
#define MATCOLORINGLF      "lf"
#define MATCOLORINGID      "id"

extern PetscErrorCode  MatGetColoring(Mat,const MatColoringType,ISColoring*);
extern PetscErrorCode  MatColoringRegister(const char[],const char[],const char[],PetscErrorCode(*)(Mat,MatColoringType,ISColoring *));

/*MC
   MatColoringRegisterDynamic - Adds a new sparse matrix coloring to the 
                               matrix package. 

   Synopsis:
   PetscErrorCode MatColoringRegisterDynamic(const char *name_coloring,const char *path,const char *name_create,PetscErrorCode (*routine_create)(MatColoring))

   Not Collective

   Input Parameters:
+  sname - name of Coloring (for example MATCOLORINGSL)
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

extern PetscBool  MatColoringRegisterAllCalled;

extern PetscErrorCode  MatColoringRegisterAll(const char[]);
extern PetscErrorCode  MatColoringRegisterDestroy(void);
extern PetscErrorCode  MatColoringPatch(Mat,PetscInt,PetscInt,ISColoringValue[],ISColoring*);

/*S
     MatFDColoring - Object for computing a sparse Jacobian via finite differences
        and coloring

   Level: beginner

  Concepts: coloring, sparse Jacobian, finite differences

.seealso:  MatFDColoringCreate()
S*/
typedef struct _p_MatFDColoring* MatFDColoring;

extern PetscErrorCode  MatFDColoringCreate(Mat,ISColoring,MatFDColoring *);
extern PetscErrorCode  MatFDColoringDestroy(MatFDColoring*);
extern PetscErrorCode  MatFDColoringView(MatFDColoring,PetscViewer);
extern PetscErrorCode  MatFDColoringSetFunction(MatFDColoring,PetscErrorCode (*)(void),void*);
extern PetscErrorCode  MatFDColoringGetFunction(MatFDColoring,PetscErrorCode (**)(void),void**);
extern PetscErrorCode  MatFDColoringSetParameters(MatFDColoring,PetscReal,PetscReal);
extern PetscErrorCode  MatFDColoringSetFromOptions(MatFDColoring);
extern PetscErrorCode  MatFDColoringApply(Mat,MatFDColoring,Vec,MatStructure*,void *);
extern PetscErrorCode  MatFDColoringApplyTS(Mat,MatFDColoring,PetscReal,Vec,MatStructure*,void *);
extern PetscErrorCode  MatFDColoringSetF(MatFDColoring,Vec);
extern PetscErrorCode  MatFDColoringGetPerturbedColumns(MatFDColoring,PetscInt*,PetscInt*[]);
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
#define MATPARTITIONINGCURRENT  "current"
#define MATPARTITIONINGSQUARE   "square"
#define MATPARTITIONINGPARMETIS "parmetis"
#define MATPARTITIONINGCHACO    "chaco"
#define MATPARTITIONINGPARTY    "party"
#define MATPARTITIONINGPTSCOTCH "ptscotch"


extern PetscErrorCode  MatPartitioningCreate(MPI_Comm,MatPartitioning*);
extern PetscErrorCode  MatPartitioningSetType(MatPartitioning,const MatPartitioningType);
extern PetscErrorCode  MatPartitioningSetNParts(MatPartitioning,PetscInt);
extern PetscErrorCode  MatPartitioningSetAdjacency(MatPartitioning,Mat);
extern PetscErrorCode  MatPartitioningSetVertexWeights(MatPartitioning,const PetscInt[]);
extern PetscErrorCode  MatPartitioningSetPartitionWeights(MatPartitioning,const PetscReal []);
extern PetscErrorCode  MatPartitioningApply(MatPartitioning,IS*);
extern PetscErrorCode  MatPartitioningDestroy(MatPartitioning*);

extern PetscErrorCode  MatPartitioningRegister(const char[],const char[],const char[],PetscErrorCode (*)(MatPartitioning));

/*MC
   MatPartitioningRegisterDynamic - Adds a new sparse matrix partitioning to the 
   matrix package. 

   Synopsis:
   PetscErrorCode MatPartitioningRegisterDynamic(const char *name_partitioning,const char *path,const char *name_create,PetscErrorCode (*routine_create)(MatPartitioning))

   Not Collective

   Input Parameters:
+  sname - name of partitioning (for example MATPARTITIONINGCURRENT) or parmetis
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

extern PetscBool  MatPartitioningRegisterAllCalled;

extern PetscErrorCode  MatPartitioningRegisterAll(const char[]);
extern PetscErrorCode  MatPartitioningRegisterDestroy(void);

extern PetscErrorCode  MatPartitioningView(MatPartitioning,PetscViewer);
extern PetscErrorCode  MatPartitioningSetFromOptions(MatPartitioning);
extern PetscErrorCode  MatPartitioningGetType(MatPartitioning,const MatPartitioningType*);

extern PetscErrorCode  MatPartitioningParmetisSetCoarseSequential(MatPartitioning);
extern PetscErrorCode  MatPartitioningParmetisGetEdgeCut(MatPartitioning, PetscInt *);

typedef enum { MP_CHACO_MULTILEVEL=1,MP_CHACO_SPECTRAL=2,MP_CHACO_LINEAR=4,MP_CHACO_RANDOM=5,MP_CHACO_SCATTERED=6 } MPChacoGlobalType;
extern const char *MPChacoGlobalTypes[];
typedef enum { MP_CHACO_KERNIGHAN=1,MP_CHACO_NONE=2 } MPChacoLocalType;
extern const char *MPChacoLocalTypes[];
typedef enum { MP_CHACO_LANCZOS=0,MP_CHACO_RQI=1 } MPChacoEigenType;
extern const char *MPChacoEigenTypes[];

extern PetscErrorCode  MatPartitioningChacoSetGlobal(MatPartitioning,MPChacoGlobalType);
extern PetscErrorCode  MatPartitioningChacoGetGlobal(MatPartitioning,MPChacoGlobalType*);
extern PetscErrorCode  MatPartitioningChacoSetLocal(MatPartitioning,MPChacoLocalType);
extern PetscErrorCode  MatPartitioningChacoGetLocal(MatPartitioning,MPChacoLocalType*);
extern PetscErrorCode  MatPartitioningChacoSetCoarseLevel(MatPartitioning,PetscReal);
extern PetscErrorCode  MatPartitioningChacoSetEigenSolver(MatPartitioning,MPChacoEigenType);
extern PetscErrorCode  MatPartitioningChacoGetEigenSolver(MatPartitioning,MPChacoEigenType*);
extern PetscErrorCode  MatPartitioningChacoSetEigenTol(MatPartitioning,PetscReal);
extern PetscErrorCode  MatPartitioningChacoGetEigenTol(MatPartitioning,PetscReal*);
extern PetscErrorCode  MatPartitioningChacoSetEigenNumber(MatPartitioning,PetscInt);
extern PetscErrorCode  MatPartitioningChacoGetEigenNumber(MatPartitioning,PetscInt*);

#define MP_PARTY_OPT "opt"
#define MP_PARTY_LIN "lin"
#define MP_PARTY_SCA "sca"
#define MP_PARTY_RAN "ran"
#define MP_PARTY_GBF "gbf"
#define MP_PARTY_GCF "gcf"
#define MP_PARTY_BUB "bub"
#define MP_PARTY_DEF "def"
extern PetscErrorCode  MatPartitioningPartySetGlobal(MatPartitioning,const char*);
#define MP_PARTY_HELPFUL_SETS "hs"
#define MP_PARTY_KERNIGHAN_LIN "kl"
#define MP_PARTY_NONE "no"
extern PetscErrorCode  MatPartitioningPartySetLocal(MatPartitioning,const char*);
extern PetscErrorCode  MatPartitioningPartySetCoarseLevel(MatPartitioning,PetscReal);
extern PetscErrorCode  MatPartitioningPartySetBipart(MatPartitioning,PetscBool);
extern PetscErrorCode  MatPartitioningPartySetMatchOptimization(MatPartitioning,PetscBool);

typedef enum { MP_PTSCOTCH_QUALITY,MP_PTSCOTCH_SPEED,MP_PTSCOTCH_BALANCE,MP_PTSCOTCH_SAFETY,MP_PTSCOTCH_SCALABILITY } MPPTScotchStrategyType;
extern const char *MPPTScotchStrategyTypes[];

extern PetscErrorCode MatPartitioningPTScotchSetImbalance(MatPartitioning,PetscReal);
extern PetscErrorCode MatPartitioningPTScotchGetImbalance(MatPartitioning,PetscReal*);
extern PetscErrorCode MatPartitioningPTScotchSetStrategy(MatPartitioning,MPPTScotchStrategyType);
extern PetscErrorCode MatPartitioningPTScotchGetStrategy(MatPartitioning,MPPTScotchStrategyType*);

extern PetscErrorCode MatMeshToVertexGraph(Mat,PetscInt,Mat*);
extern PetscErrorCode MatMeshToCellGraph(Mat,PetscInt,Mat*);

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
               MATOP_GET_LOCALSUBMATRIX=117,
               MATOP_RESTORE_LOCALSUBMATRIX=118,
               MATOP_MULT_DIAGONAL_BLOCK=119,
               MATOP_HERMITIANTRANSPOSE=120,
               MATOP_MULTHERMITIANTRANSPOSE=121,
               MATOP_MULTHERMITIANTRANSPOSEADD=122,
               MATOP_GETMULTIPROCBLOCK=123,
               MATOP_GETCOLUMNNORMS=125,
	       MATOP_GET_SUBMATRICES_PARALLEL=128,
               MATOP_SET_VALUES_BATCH=129
             } MatOperation;
extern PetscErrorCode  MatHasOperation(Mat,MatOperation,PetscBool *);
extern PetscErrorCode  MatShellSetOperation(Mat,MatOperation,void(*)(void));
extern PetscErrorCode  MatShellGetOperation(Mat,MatOperation,void(**)(void));
extern PetscErrorCode  MatShellSetContext(Mat,void*);

/*
   Codes for matrices stored on disk. By default they are
   stored in a universal format. By changing the format with 
   PetscViewerSetFormat(viewer,PETSC_VIEWER_NATIVE); the matrices will
   be stored in a way natural for the matrix, for example dense matrices
   would be stored as dense. Matrices stored this way may only be
   read into matrices of the same type.
*/
#define MATRIX_BINARY_FORMAT_DENSE -1

extern PetscErrorCode  MatMPIBAIJSetHashTableFactor(Mat,PetscReal);
extern PetscErrorCode  MatISGetLocalMat(Mat,Mat*);

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

extern PetscErrorCode  MatNullSpaceCreate(MPI_Comm,PetscBool ,PetscInt,const Vec[],MatNullSpace*);
extern PetscErrorCode  MatNullSpaceSetFunction(MatNullSpace,PetscErrorCode (*)(MatNullSpace,Vec,void*),void*);
extern PetscErrorCode  MatNullSpaceDestroy(MatNullSpace*);
extern PetscErrorCode  MatNullSpaceRemove(MatNullSpace,Vec,Vec*);
extern PetscErrorCode  MatNullSpaceAttach(Mat,MatNullSpace);
extern PetscErrorCode  MatNullSpaceTest(MatNullSpace,Mat,PetscBool  *);
extern PetscErrorCode  MatNullSpaceView(MatNullSpace,PetscViewer);

extern PetscErrorCode  MatReorderingSeqSBAIJ(Mat,IS);
extern PetscErrorCode  MatMPISBAIJSetHashTableFactor(Mat,PetscReal);
extern PetscErrorCode  MatSeqSBAIJSetColumnIndices(Mat,PetscInt *);
extern PetscErrorCode  MatSeqBAIJInvertBlockDiagonal(Mat);

extern PetscErrorCode  MatCreateMAIJ(Mat,PetscInt,Mat*);
extern PetscErrorCode  MatMAIJRedimension(Mat,PetscInt,Mat*);
extern PetscErrorCode  MatMAIJGetAIJ(Mat,Mat*);

extern PetscErrorCode  MatComputeExplicitOperator(Mat,Mat*);

extern PetscErrorCode  MatDiagonalScaleLocal(Mat,Vec);

extern PetscErrorCode  MatCreateMFFD(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,Mat*);
extern PetscErrorCode  MatMFFDSetBase(Mat,Vec,Vec);
extern PetscErrorCode  MatMFFDSetFunction(Mat,PetscErrorCode(*)(void*,Vec,Vec),void*);
extern PetscErrorCode  MatMFFDSetFunctioni(Mat,PetscErrorCode (*)(void*,PetscInt,Vec,PetscScalar*));
extern PetscErrorCode  MatMFFDSetFunctioniBase(Mat,PetscErrorCode (*)(void*,Vec));
extern PetscErrorCode  MatMFFDAddNullSpace(Mat,MatNullSpace);
extern PetscErrorCode  MatMFFDSetHHistory(Mat,PetscScalar[],PetscInt);
extern PetscErrorCode  MatMFFDResetHHistory(Mat);
extern PetscErrorCode  MatMFFDSetFunctionError(Mat,PetscReal);
extern PetscErrorCode  MatMFFDSetPeriod(Mat,PetscInt);
extern PetscErrorCode  MatMFFDGetH(Mat,PetscScalar *);
extern PetscErrorCode  MatMFFDSetOptionsPrefix(Mat,const char[]);
extern PetscErrorCode  MatMFFDCheckPositivity(void*,Vec,Vec,PetscScalar*);
extern PetscErrorCode  MatMFFDSetCheckh(Mat,PetscErrorCode (*)(void*,Vec,Vec,PetscScalar*),void*);

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

extern PetscErrorCode  MatMFFDSetType(Mat,const MatMFFDType);
extern PetscErrorCode  MatMFFDRegister(const char[],const char[],const char[],PetscErrorCode (*)(MatMFFD));

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

extern PetscErrorCode  MatMFFDRegisterAll(const char[]);
extern PetscErrorCode  MatMFFDRegisterDestroy(void);
extern PetscErrorCode  MatMFFDDSSetUmin(Mat,PetscReal);
extern PetscErrorCode  MatMFFDWPSetComputeNormU(Mat,PetscBool );


extern PetscErrorCode  PetscViewerMathematicaPutMatrix(PetscViewer, PetscInt, PetscInt, PetscReal *);
extern PetscErrorCode  PetscViewerMathematicaPutCSRMatrix(PetscViewer, PetscInt, PetscInt, PetscInt *, PetscInt *, PetscReal *);

/* 
   PETSc interface to MUMPS 
*/
#ifdef PETSC_HAVE_MUMPS
extern PetscErrorCode  MatMumpsSetIcntl(Mat,PetscInt,PetscInt);
#endif

/* 
   PETSc interface to SUPERLU
*/
#ifdef PETSC_HAVE_SUPERLU
extern PetscErrorCode  MatSuperluSetILUDropTol(Mat,PetscReal);
#endif

#if defined(PETSC_HAVE_CUSP)
extern PetscErrorCode  MatCreateSeqAIJCUSP(MPI_Comm,PetscInt,PetscInt,PetscInt,const PetscInt[],Mat*);
extern PetscErrorCode  MatCreateMPIAIJCUSP(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Mat*);
#endif

/* 
   PETSc interface to FFTW
*/
#if defined(PETSC_HAVE_FFTW)
extern PetscErrorCode VecScatterPetscToFFTW(Mat,Vec,Vec);
extern PetscErrorCode VecScatterFFTWToPetsc(Mat,Vec,Vec);
extern PetscErrorCode MatGetVecsFFTW(Mat,Vec*,Vec*,Vec*);
#endif

extern PetscErrorCode MatCreateNest(MPI_Comm,PetscInt,const IS[],PetscInt,const IS[],const Mat[],Mat*);
extern PetscErrorCode MatNestGetSize(Mat,PetscInt*,PetscInt*);
extern PetscErrorCode MatNestGetSubMats(Mat,PetscInt*,PetscInt*,Mat***);
extern PetscErrorCode MatNestGetSubMat(Mat,PetscInt,PetscInt,Mat*);
extern PetscErrorCode MatNestSetVecType(Mat,const VecType);
extern PetscErrorCode MatNestSetSubMats(Mat,PetscInt,const IS[],PetscInt,const IS[],const Mat[]);
extern PetscErrorCode MatNestSetSubMat(Mat,PetscInt,PetscInt,Mat);

PETSC_EXTERN_CXX_END
#endif
