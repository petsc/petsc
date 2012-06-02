/*
     Include file for the matrix component of PETSc
*/
#ifndef __PETSCMAT_H
#define __PETSCMAT_H
#include "petscvec.h"

/*S
     Mat - Abstract PETSc matrix object

   Level: beginner

  Concepts: matrix; linear operator

.seealso:  MatCreate(), MatType, MatSetType()
S*/
typedef struct _p_Mat*           Mat;

/*J
    MatType - String with the name of a PETSc matrix or the creation function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:mymatcreate()

   Level: beginner

.seealso: MatSetType(), Mat, MatSolverPackage
J*/
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
#define MATIJ              "ij"

/*J
    MatSolverPackage - String with the name of a PETSc matrix solver type. 

    For example: "petsc" indicates what PETSc provides, "superlu" indicates either 
       SuperLU or SuperLU_Dist etc.


   Level: beginner

.seealso: MatGetFactor(), Mat, MatSetType(), MatType
J*/
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
PETSC_EXTERN const char *const MatFactorTypes[];

PETSC_EXTERN PetscErrorCode MatGetFactor(Mat,const MatSolverPackage,MatFactorType,Mat*);
PETSC_EXTERN PetscErrorCode MatGetFactorAvailable(Mat,const MatSolverPackage,MatFactorType,PetscBool *);
PETSC_EXTERN PetscErrorCode MatFactorGetSolverPackage(Mat,const MatSolverPackage*);
PETSC_EXTERN PetscErrorCode MatGetFactorType(Mat,MatFactorType*);

/* Logging support */
#define    MAT_FILE_CLASSID 1211216    /* used to indicate matrices in binary files */
PETSC_EXTERN PetscClassId MAT_CLASSID;
PETSC_EXTERN PetscClassId MAT_FDCOLORING_CLASSID;
PETSC_EXTERN PetscClassId MAT_TRANSPOSECOLORING_CLASSID;
PETSC_EXTERN PetscClassId MAT_PARTITIONING_CLASSID;
PETSC_EXTERN PetscClassId MAT_COARSEN_CLASSID;
PETSC_EXTERN PetscClassId MAT_NULLSPACE_CLASSID;
PETSC_EXTERN PetscClassId MATMFFD_CLASSID;

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

PETSC_EXTERN PetscErrorCode MatInitializePackage(const char[]);

PETSC_EXTERN PetscErrorCode MatCreate(MPI_Comm,Mat*);
PETSC_EXTERN PetscErrorCode MatSetSizes(Mat,PetscInt,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode MatSetType(Mat,const MatType);
PETSC_EXTERN PetscErrorCode MatSetFromOptions(Mat);
PETSC_EXTERN PetscErrorCode MatRegisterAll(const char[]);
PETSC_EXTERN PetscErrorCode MatRegister(const char[],const char[],const char[],PetscErrorCode(*)(Mat));
PETSC_EXTERN PetscErrorCode MatRegisterBaseName(const char[],const char[],const char[]);
PETSC_EXTERN PetscErrorCode MatSetOptionsPrefix(Mat,const char[]);
PETSC_EXTERN PetscErrorCode MatAppendOptionsPrefix(Mat,const char[]);
PETSC_EXTERN PetscErrorCode MatGetOptionsPrefix(Mat,const char*[]);

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

PETSC_EXTERN PetscBool MatRegisterAllCalled;
PETSC_EXTERN PetscFList MatList;
PETSC_EXTERN PetscFList MatColoringList;
PETSC_EXTERN PetscFList MatPartitioningList;
PETSC_EXTERN PetscFList MatCoarsenList;

/*E
    MatStructure - Indicates if the matrix has the same nonzero structure

    Level: beginner

   Any additions/changes here MUST also be made in include/finclude/petscmat.h

.seealso: MatCopy(), KSPSetOperators(), PCSetOperators()
E*/
typedef enum {DIFFERENT_NONZERO_PATTERN,SUBSET_NONZERO_PATTERN,SAME_NONZERO_PATTERN,SAME_PRECONDITIONER} MatStructure;

PETSC_EXTERN PetscErrorCode MatCreateSeqDense(MPI_Comm,PetscInt,PetscInt,PetscScalar[],Mat*);
PETSC_EXTERN PetscErrorCode MatCreateDense(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscScalar[],Mat*); 
PETSC_EXTERN PetscErrorCode MatCreateSeqAIJ(MPI_Comm,PetscInt,PetscInt,PetscInt,const PetscInt[],Mat*);
PETSC_EXTERN PetscErrorCode MatCreateAIJ(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Mat*); 
PETSC_EXTERN PetscErrorCode MatCreateMPIAIJWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[],Mat *);
PETSC_EXTERN PetscErrorCode MatCreateMPIAIJWithSplitArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt[],PetscInt[],PetscScalar[],PetscInt[],PetscInt[],PetscScalar[],Mat*);

PETSC_EXTERN PetscErrorCode MatCreateSeqBAIJ(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],Mat*); 
PETSC_EXTERN PetscErrorCode MatCreateBAIJ(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Mat*);
PETSC_EXTERN PetscErrorCode MatCreateMPIBAIJWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[],Mat*);

PETSC_EXTERN PetscErrorCode MatCreateMPIAdj(MPI_Comm,PetscInt,PetscInt,PetscInt[],PetscInt[],PetscInt[],Mat*);
PETSC_EXTERN PetscErrorCode MatCreateSeqSBAIJ(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],Mat*); 

PETSC_EXTERN PetscErrorCode MatCreateSBAIJ(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Mat*);
PETSC_EXTERN PetscErrorCode MatCreateMPISBAIJWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[],Mat *);
PETSC_EXTERN PetscErrorCode MatMPISBAIJSetPreallocationCSR(Mat,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatXAIJSetPreallocation(Mat,PetscInt,const PetscInt*,const PetscInt*,const PetscInt*,const PetscInt*);

PETSC_EXTERN PetscErrorCode MatCreateShell(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,void *,Mat*);
PETSC_EXTERN PetscErrorCode MatCreateAdic(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,void (*)(void),Mat*);
PETSC_EXTERN PetscErrorCode MatCreateNormal(Mat,Mat*);
PETSC_EXTERN PetscErrorCode MatCreateLRC(Mat,Mat,Mat,Mat*);
PETSC_EXTERN PetscErrorCode MatCreateIS(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,ISLocalToGlobalMapping,Mat*);
PETSC_EXTERN PetscErrorCode MatCreateSeqAIJCRL(MPI_Comm,PetscInt,PetscInt,PetscInt,const PetscInt[],Mat*);
PETSC_EXTERN PetscErrorCode MatCreateMPIAIJCRL(MPI_Comm,PetscInt,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Mat*);

PETSC_EXTERN PetscErrorCode MatCreateSeqBSTRM(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],Mat*);
PETSC_EXTERN PetscErrorCode MatCreateMPIBSTRM(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Mat*);
PETSC_EXTERN PetscErrorCode MatCreateSeqSBSTRM(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],Mat*);
PETSC_EXTERN PetscErrorCode MatCreateMPISBSTRM(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Mat*);

PETSC_EXTERN PetscErrorCode MatCreateScatter(MPI_Comm,VecScatter,Mat*);
PETSC_EXTERN PetscErrorCode MatScatterSetVecScatter(Mat,VecScatter);
PETSC_EXTERN PetscErrorCode MatScatterGetVecScatter(Mat,VecScatter*);
PETSC_EXTERN PetscErrorCode MatCreateBlockMat(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,Mat*);
PETSC_EXTERN PetscErrorCode MatCompositeAddMat(Mat,Mat);
PETSC_EXTERN PetscErrorCode MatCompositeMerge(Mat);
PETSC_EXTERN PetscErrorCode MatCreateComposite(MPI_Comm,PetscInt,const Mat*,Mat*);
typedef enum {MAT_COMPOSITE_ADDITIVE,MAT_COMPOSITE_MULTIPLICATIVE} MatCompositeType;
PETSC_EXTERN PetscErrorCode MatCompositeSetType(Mat,MatCompositeType);

PETSC_EXTERN PetscErrorCode MatCreateFFT(MPI_Comm,PetscInt,const PetscInt[],const MatType,Mat*);
PETSC_EXTERN PetscErrorCode MatCreateSeqCUFFT(MPI_Comm,PetscInt,const PetscInt[],Mat*);

PETSC_EXTERN PetscErrorCode MatCreateTranspose(Mat,Mat*);
PETSC_EXTERN PetscErrorCode MatCreateSubMatrix(Mat,IS,IS,Mat*);
PETSC_EXTERN PetscErrorCode MatSubMatrixUpdate(Mat,Mat,IS,IS);
PETSC_EXTERN PetscErrorCode MatCreateLocalRef(Mat,IS,IS,Mat*);

PETSC_EXTERN PetscErrorCode MatPythonSetType(Mat,const char[]);

PETSC_EXTERN PetscErrorCode MatSetUp(Mat);
PETSC_EXTERN PetscErrorCode MatDestroy(Mat*);

PETSC_EXTERN PetscErrorCode MatConjugate(Mat);
PETSC_EXTERN PetscErrorCode MatRealPart(Mat);
PETSC_EXTERN PetscErrorCode MatImaginaryPart(Mat);
PETSC_EXTERN PetscErrorCode MatGetDiagonalBlock(Mat,Mat*);
PETSC_EXTERN PetscErrorCode MatGetTrace(Mat,PetscScalar*);
PETSC_EXTERN PetscErrorCode MatInvertBlockDiagonal(Mat,const PetscScalar **);

/* ------------------------------------------------------------*/
PETSC_EXTERN PetscErrorCode MatSetValues(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
PETSC_EXTERN PetscErrorCode MatSetValuesBlocked(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
PETSC_EXTERN PetscErrorCode MatSetValuesRow(Mat,PetscInt,const PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatSetValuesRowLocal(Mat,PetscInt,const PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatSetValuesBatch(Mat,PetscInt,PetscInt,PetscInt[],const PetscScalar[]);

/*S
     MatStencil - Data structure (C struct) for storing information about a single row or
        column of a matrix as indexed on an associated grid.

   Fortran usage is different, see MatSetValuesStencil() for details.

   Level: beginner

  Concepts: matrix; linear operator

.seealso:  MatSetValuesStencil(), MatSetStencil(), MatSetValuesBlockedStencil()
S*/
typedef struct {
  PetscInt k,j,i,c;
} MatStencil;

PETSC_EXTERN PetscErrorCode MatSetValuesStencil(Mat,PetscInt,const MatStencil[],PetscInt,const MatStencil[],const PetscScalar[],InsertMode);
PETSC_EXTERN PetscErrorCode MatSetValuesBlockedStencil(Mat,PetscInt,const MatStencil[],PetscInt,const MatStencil[],const PetscScalar[],InsertMode);
PETSC_EXTERN PetscErrorCode MatSetStencil(Mat,PetscInt,const PetscInt[],const PetscInt[],PetscInt);

PETSC_EXTERN PetscErrorCode MatSetColoring(Mat,ISColoring);
PETSC_EXTERN PetscErrorCode MatSetValuesAdic(Mat,void*);
PETSC_EXTERN PetscErrorCode MatSetValuesAdifor(Mat,PetscInt,void*);

/*E
    MatAssemblyType - Indicates if the matrix is now to be used, or if you plan 
     to continue to add values to it

    Level: beginner

.seealso: MatAssemblyBegin(), MatAssemblyEnd()
E*/
typedef enum {MAT_FLUSH_ASSEMBLY=1,MAT_FINAL_ASSEMBLY=0} MatAssemblyType;
PETSC_EXTERN PetscErrorCode MatAssemblyBegin(Mat,MatAssemblyType);
PETSC_EXTERN PetscErrorCode MatAssemblyEnd(Mat,MatAssemblyType);
PETSC_EXTERN PetscErrorCode MatAssembled(Mat,PetscBool *);



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
PETSC_EXTERN const char *MatOptions[];
PETSC_EXTERN PetscErrorCode MatSetOption(Mat,MatOption,PetscBool );
PETSC_EXTERN PetscErrorCode MatGetType(Mat,const MatType*);

PETSC_EXTERN PetscErrorCode MatGetValues(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatGetRow(Mat,PetscInt,PetscInt *,const PetscInt *[],const PetscScalar*[]);
PETSC_EXTERN PetscErrorCode MatRestoreRow(Mat,PetscInt,PetscInt *,const PetscInt *[],const PetscScalar*[]);
PETSC_EXTERN PetscErrorCode MatGetRowUpperTriangular(Mat);
PETSC_EXTERN PetscErrorCode MatRestoreRowUpperTriangular(Mat);
PETSC_EXTERN PetscErrorCode MatGetColumn(Mat,PetscInt,PetscInt *,const PetscInt *[],const PetscScalar*[]);
PETSC_EXTERN PetscErrorCode MatRestoreColumn(Mat,PetscInt,PetscInt *,const PetscInt *[],const PetscScalar*[]);
PETSC_EXTERN PetscErrorCode MatGetColumnVector(Mat,Vec,PetscInt);
PETSC_EXTERN PetscErrorCode MatGetArray(Mat,PetscScalar *[]);
PETSC_EXTERN PetscErrorCode MatRestoreArray(Mat,PetscScalar *[]);
PETSC_EXTERN PetscErrorCode MatGetBlockSize(Mat,PetscInt *);
PETSC_EXTERN PetscErrorCode MatSetBlockSize(Mat,PetscInt);
PETSC_EXTERN PetscErrorCode MatGetBlockSizes(Mat,PetscInt *,PetscInt *);
PETSC_EXTERN PetscErrorCode MatSetBlockSizes(Mat,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode MatSetNThreads(Mat,PetscInt);
PETSC_EXTERN PetscErrorCode MatGetNThreads(Mat,PetscInt*);

PETSC_EXTERN PetscErrorCode MatMult(Mat,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatMultDiagonalBlock(Mat,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatMultAdd(Mat,Vec,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatMultTranspose(Mat,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatMultHermitianTranspose(Mat,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatIsTranspose(Mat,Mat,PetscReal,PetscBool *);
PETSC_EXTERN PetscErrorCode MatIsHermitianTranspose(Mat,Mat,PetscReal,PetscBool *);
PETSC_EXTERN PetscErrorCode MatMultTransposeAdd(Mat,Vec,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatMultHermitianTransposeAdd(Mat,Vec,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatMultConstrained(Mat,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatMultTransposeConstrained(Mat,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatMatSolve(Mat,Mat,Mat);

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

PETSC_EXTERN PetscErrorCode MatConvert(Mat,const MatType,MatReuse,Mat*);
PETSC_EXTERN PetscErrorCode MatDuplicate(Mat,MatDuplicateOption,Mat*);


PETSC_EXTERN PetscErrorCode MatCopy(Mat,Mat,MatStructure);
PETSC_EXTERN PetscErrorCode MatView(Mat,PetscViewer);
PETSC_EXTERN PetscErrorCode MatIsSymmetric(Mat,PetscReal,PetscBool *);
PETSC_EXTERN PetscErrorCode MatIsStructurallySymmetric(Mat,PetscBool *);
PETSC_EXTERN PetscErrorCode MatIsHermitian(Mat,PetscReal,PetscBool *);
PETSC_EXTERN PetscErrorCode MatIsSymmetricKnown(Mat,PetscBool *,PetscBool *);
PETSC_EXTERN PetscErrorCode MatIsHermitianKnown(Mat,PetscBool *,PetscBool *);
PETSC_EXTERN PetscErrorCode MatMissingDiagonal(Mat,PetscBool  *,PetscInt *);
PETSC_EXTERN PetscErrorCode MatLoad(Mat, PetscViewer);

PETSC_EXTERN PetscErrorCode MatGetRowIJ(Mat,PetscInt,PetscBool ,PetscBool ,PetscInt*,PetscInt *[],PetscInt *[],PetscBool  *);
PETSC_EXTERN PetscErrorCode MatRestoreRowIJ(Mat,PetscInt,PetscBool ,PetscBool ,PetscInt *,PetscInt *[],PetscInt *[],PetscBool  *);
PETSC_EXTERN PetscErrorCode MatGetColumnIJ(Mat,PetscInt,PetscBool ,PetscBool ,PetscInt*,PetscInt *[],PetscInt *[],PetscBool  *);
PETSC_EXTERN PetscErrorCode MatRestoreColumnIJ(Mat,PetscInt,PetscBool ,PetscBool ,PetscInt *,PetscInt *[],PetscInt *[],PetscBool  *);

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
PETSC_EXTERN PetscErrorCode MatGetInfo(Mat,MatInfoType,MatInfo*);
PETSC_EXTERN PetscErrorCode MatGetDiagonal(Mat,Vec);
PETSC_EXTERN PetscErrorCode MatGetRowMax(Mat,Vec,PetscInt[]);
PETSC_EXTERN PetscErrorCode MatGetRowMin(Mat,Vec,PetscInt[]);
PETSC_EXTERN PetscErrorCode MatGetRowMaxAbs(Mat,Vec,PetscInt[]);
PETSC_EXTERN PetscErrorCode MatGetRowMinAbs(Mat,Vec,PetscInt[]);
PETSC_EXTERN PetscErrorCode MatGetRowSum(Mat,Vec);
PETSC_EXTERN PetscErrorCode MatTranspose(Mat,MatReuse,Mat*);
PETSC_EXTERN PetscErrorCode MatHermitianTranspose(Mat,MatReuse,Mat*);
PETSC_EXTERN PetscErrorCode MatPermute(Mat,IS,IS,Mat *);
PETSC_EXTERN PetscErrorCode MatDiagonalScale(Mat,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatDiagonalSet(Mat,Vec,InsertMode);
PETSC_EXTERN PetscErrorCode MatEqual(Mat,Mat,PetscBool *);
PETSC_EXTERN PetscErrorCode MatMultEqual(Mat,Mat,PetscInt,PetscBool *);
PETSC_EXTERN PetscErrorCode MatMultAddEqual(Mat,Mat,PetscInt,PetscBool *);
PETSC_EXTERN PetscErrorCode MatMultTransposeEqual(Mat,Mat,PetscInt,PetscBool *);
PETSC_EXTERN PetscErrorCode MatMultTransposeAddEqual(Mat,Mat,PetscInt,PetscBool *);

PETSC_EXTERN PetscErrorCode MatNorm(Mat,NormType,PetscReal *);
PETSC_EXTERN PetscErrorCode MatGetColumnNorms(Mat,NormType,PetscReal *);
PETSC_EXTERN PetscErrorCode MatZeroEntries(Mat);
PETSC_EXTERN PetscErrorCode MatZeroRows(Mat,PetscInt,const PetscInt [],PetscScalar,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatZeroRowsIS(Mat,IS,PetscScalar,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatZeroRowsStencil(Mat,PetscInt,const MatStencil [],PetscScalar,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatZeroRowsColumnsStencil(Mat,PetscInt,const MatStencil[],PetscScalar,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatZeroRowsColumns(Mat,PetscInt,const PetscInt [],PetscScalar,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatZeroRowsColumnsIS(Mat,IS,PetscScalar,Vec,Vec);

PETSC_EXTERN PetscErrorCode MatGetSize(Mat,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode MatGetLocalSize(Mat,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode MatGetOwnershipRange(Mat,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode MatGetOwnershipRanges(Mat,const PetscInt**);
PETSC_EXTERN PetscErrorCode MatGetOwnershipRangeColumn(Mat,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode MatGetOwnershipRangesColumn(Mat,const PetscInt**);

PETSC_EXTERN PetscErrorCode MatGetSubMatrices(Mat,PetscInt,const IS[],const IS[],MatReuse,Mat *[]);
PETSC_EXTERN PetscErrorCode MatGetSubMatricesParallel(Mat,PetscInt,const IS[],const IS[],MatReuse,Mat *[]);
PETSC_EXTERN PetscErrorCode MatDestroyMatrices(PetscInt,Mat *[]);
PETSC_EXTERN PetscErrorCode MatGetSubMatrix(Mat,IS,IS,MatReuse,Mat *);
PETSC_EXTERN PetscErrorCode MatGetLocalSubMatrix(Mat,IS,IS,Mat*);
PETSC_EXTERN PetscErrorCode MatRestoreLocalSubMatrix(Mat,IS,IS,Mat*);
PETSC_EXTERN PetscErrorCode MatGetSeqNonzeroStructure(Mat,Mat*);
PETSC_EXTERN PetscErrorCode MatDestroySeqNonzeroStructure(Mat*); 

PETSC_EXTERN PetscErrorCode MatCreateMPIAIJConcatenateSeqAIJ(MPI_Comm,Mat,PetscInt,MatReuse,Mat*);
PETSC_EXTERN PetscErrorCode MatCreateMPIAIJConcatenateSeqAIJSymbolic(MPI_Comm,Mat,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode MatCreateMPIAIJConcatenateSeqAIJNumeric(MPI_Comm,Mat,PetscInt,Mat);
PETSC_EXTERN PetscErrorCode MatCreateMPIAIJSumSeqAIJ(MPI_Comm,Mat,PetscInt,PetscInt,MatReuse,Mat*);
PETSC_EXTERN PetscErrorCode MatCreateMPIAIJSumSeqAIJSymbolic(MPI_Comm,Mat,PetscInt,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode MatCreateMPIAIJSumSeqAIJNumeric(Mat,Mat); 
PETSC_EXTERN PetscErrorCode MatMPIAIJGetLocalMat(Mat,MatReuse,Mat*);
PETSC_EXTERN PetscErrorCode MatMPIAIJGetLocalMatCondensed(Mat,MatReuse,IS*,IS*,Mat*);
PETSC_EXTERN PetscErrorCode MatGetBrowsOfAcols(Mat,Mat,MatReuse,IS*,IS*,Mat*);
#if defined (PETSC_USE_CTABLE)
PETSC_EXTERN PetscErrorCode MatGetCommunicationStructs(Mat, Vec *, PetscTable *, VecScatter *);
#else
PETSC_EXTERN PetscErrorCode MatGetCommunicationStructs(Mat, Vec *, PetscInt *[], VecScatter *);
#endif
PETSC_EXTERN PetscErrorCode MatGetGhosts(Mat, PetscInt *,const PetscInt *[]);

PETSC_EXTERN PetscErrorCode MatIncreaseOverlap(Mat,PetscInt,IS[],PetscInt);

PETSC_EXTERN PetscErrorCode MatMatMult(Mat,Mat,MatReuse,PetscReal,Mat*);
PETSC_EXTERN PetscErrorCode MatMatMultSymbolic(Mat,Mat,PetscReal,Mat*);
PETSC_EXTERN PetscErrorCode MatMatMultNumeric(Mat,Mat,Mat);

PETSC_EXTERN PetscErrorCode MatPtAP(Mat,Mat,MatReuse,PetscReal,Mat*);
PETSC_EXTERN PetscErrorCode MatPtAPSymbolic(Mat,Mat,PetscReal,Mat*);
PETSC_EXTERN PetscErrorCode MatPtAPNumeric(Mat,Mat,Mat);
PETSC_EXTERN PetscErrorCode MatRARt(Mat,Mat,MatReuse,PetscReal,Mat*);
PETSC_EXTERN PetscErrorCode MatRARtSymbolic(Mat,Mat,PetscReal,Mat*);
PETSC_EXTERN PetscErrorCode MatRARtNumeric(Mat,Mat,Mat);

PETSC_EXTERN PetscErrorCode MatTransposeMatMult(Mat,Mat,MatReuse,PetscReal,Mat*);
PETSC_EXTERN PetscErrorCode MatTransposetMatMultSymbolic(Mat,Mat,PetscReal,Mat*);
PETSC_EXTERN PetscErrorCode MatTransposetMatMultNumeric(Mat,Mat,Mat);
PETSC_EXTERN PetscErrorCode MatMatTransposeMult(Mat,Mat,MatReuse,PetscReal,Mat*);
PETSC_EXTERN PetscErrorCode MatMatTransposeMultSymbolic(Mat,Mat,PetscReal,Mat*);
PETSC_EXTERN PetscErrorCode MatMatTransposeMultNumeric(Mat,Mat,Mat);

PETSC_EXTERN PetscErrorCode MatAXPY(Mat,PetscScalar,Mat,MatStructure);
PETSC_EXTERN PetscErrorCode MatAYPX(Mat,PetscScalar,Mat,MatStructure);

PETSC_EXTERN PetscErrorCode MatScale(Mat,PetscScalar);
PETSC_EXTERN PetscErrorCode MatShift(Mat,PetscScalar);

PETSC_EXTERN PetscErrorCode MatSetLocalToGlobalMapping(Mat,ISLocalToGlobalMapping,ISLocalToGlobalMapping);
PETSC_EXTERN PetscErrorCode MatSetLocalToGlobalMappingBlock(Mat,ISLocalToGlobalMapping,ISLocalToGlobalMapping);
PETSC_EXTERN PetscErrorCode MatGetLocalToGlobalMapping(Mat,ISLocalToGlobalMapping*,ISLocalToGlobalMapping*);
PETSC_EXTERN PetscErrorCode MatGetLocalToGlobalMappingBlock(Mat,ISLocalToGlobalMapping*,ISLocalToGlobalMapping*);
PETSC_EXTERN PetscErrorCode MatZeroRowsLocal(Mat,PetscInt,const PetscInt [],PetscScalar,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatZeroRowsLocalIS(Mat,IS,PetscScalar,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatZeroRowsColumnsLocal(Mat,PetscInt,const PetscInt [],PetscScalar,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatZeroRowsColumnsLocalIS(Mat,IS,PetscScalar,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatSetValuesLocal(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
PETSC_EXTERN PetscErrorCode MatSetValuesBlockedLocal(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);

PETSC_EXTERN PetscErrorCode MatStashSetInitialSize(Mat,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode MatStashGetInfo(Mat,PetscInt*,PetscInt*,PetscInt*,PetscInt*);

PETSC_EXTERN PetscErrorCode MatInterpolate(Mat,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatInterpolateAdd(Mat,Vec,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatRestrict(Mat,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatGetVecs(Mat,Vec*,Vec*);
PETSC_EXTERN PetscErrorCode MatGetRedundantMatrix(Mat,PetscInt,MPI_Comm,PetscInt,MatReuse,Mat*);
PETSC_EXTERN PetscErrorCode MatGetMultiProcBlock(Mat,MPI_Comm,MatReuse,Mat*);
PETSC_EXTERN PetscErrorCode MatFindZeroDiagonals(Mat,IS*);

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
  _4_ierr = PetscMemzero(onz,__nrows*sizeof(PetscInt));CHKERRQ(_4_ierr); __start = 0; __end = __start; \
  _4_ierr = MPI_Scan(&__ctmp,&__end,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(_4_ierr); __start = __end - __ctmp;\
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
          MatPreallocateSymmetricSetLocal()
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
          MatPreallocateSymmetricSetLocal()
M*/
#define MatPreallocateFinalize(dnz,onz) 0;_4_ierr = PetscFree2(dnz,onz);CHKERRQ(_4_ierr);}



/* Routines unique to particular data structures */
PETSC_EXTERN PetscErrorCode MatShellGetContext(Mat,void *);

PETSC_EXTERN PetscErrorCode MatInodeAdjustForInodes(Mat,IS*,IS*);
PETSC_EXTERN PetscErrorCode MatInodeGetInodeSizes(Mat,PetscInt *,PetscInt *[],PetscInt *);

PETSC_EXTERN PetscErrorCode MatSeqAIJSetColumnIndices(Mat,PetscInt[]);
PETSC_EXTERN PetscErrorCode MatSeqBAIJSetColumnIndices(Mat,PetscInt[]);
PETSC_EXTERN PetscErrorCode MatCreateSeqAIJWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt[],PetscInt[],PetscScalar[],Mat*);
PETSC_EXTERN PetscErrorCode MatCreateSeqBAIJWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt[],PetscInt[],PetscScalar[],Mat*);
PETSC_EXTERN PetscErrorCode MatCreateSeqSBAIJWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt[],PetscInt[],PetscScalar[],Mat*);
PETSC_EXTERN PetscErrorCode MatCreateSeqAIJFromTriple(MPI_Comm,PetscInt,PetscInt,PetscInt[],PetscInt[],PetscScalar[],Mat*,PetscInt,PetscBool);

#define MAT_SKIP_ALLOCATION -4

PETSC_EXTERN PetscErrorCode MatSeqBAIJSetPreallocation(Mat,PetscInt,PetscInt,const PetscInt[]);
PETSC_EXTERN PetscErrorCode MatSeqSBAIJSetPreallocation(Mat,PetscInt,PetscInt,const PetscInt[]);
PETSC_EXTERN PetscErrorCode MatSeqAIJSetPreallocation(Mat,PetscInt,const PetscInt[]);

PETSC_EXTERN PetscErrorCode MatMPIBAIJSetPreallocation(Mat,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);
PETSC_EXTERN PetscErrorCode MatMPISBAIJSetPreallocation(Mat,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);
PETSC_EXTERN PetscErrorCode MatMPIAIJSetPreallocation(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);
PETSC_EXTERN PetscErrorCode MatSeqAIJSetPreallocationCSR(Mat,const PetscInt [],const PetscInt [],const PetscScalar []);
PETSC_EXTERN PetscErrorCode MatSeqBAIJSetPreallocationCSR(Mat,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatMPIAIJSetPreallocationCSR(Mat,const PetscInt[],const PetscInt[],const PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatMPIBAIJSetPreallocationCSR(Mat,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatMPIAdjSetPreallocation(Mat,PetscInt[],PetscInt[],PetscInt[]);
PETSC_EXTERN PetscErrorCode MatMPIDenseSetPreallocation(Mat,PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatSeqDenseSetPreallocation(Mat,PetscScalar[]);
PETSC_EXTERN PetscErrorCode MatMPIAIJGetSeqAIJ(Mat,Mat*,Mat*,PetscInt*[]);
PETSC_EXTERN PetscErrorCode MatMPIBAIJGetSeqBAIJ(Mat,Mat*,Mat*,PetscInt*[]);
PETSC_EXTERN PetscErrorCode MatAdicSetLocalFunction(Mat,void (*)(void));
PETSC_EXTERN PetscErrorCode MatMPIAdjCreateNonemptySubcommMat(Mat,Mat*);

PETSC_EXTERN PetscErrorCode MatSeqDenseSetLDA(Mat,PetscInt);
PETSC_EXTERN PetscErrorCode MatDenseGetLocalMatrix(Mat,Mat*);

PETSC_EXTERN PetscErrorCode MatStoreValues(Mat);
PETSC_EXTERN PetscErrorCode MatRetrieveValues(Mat);

PETSC_EXTERN PetscErrorCode MatDAADSetCtx(Mat,void*);

PETSC_EXTERN PetscErrorCode MatFindNonzeroRows(Mat,IS*);
/* 
  These routines are not usually accessed directly, rather solving is 
  done through the KSP and PC interfaces.
*/

/*J
    MatOrderingType - String with the name of a PETSc matrix ordering or the creation function
       with an optional dynamic library name, for example 
       http://www.mcs.anl.gov/petsc/lib.a:orderingcreate()

   Level: beginner

   Cannot use const because the PC objects manipulate the string

.seealso: MatGetOrdering()
J*/
#define MatOrderingType char*
#define MATORDERINGNATURAL     "natural"
#define MATORDERINGND          "nd"
#define MATORDERING1WD         "1wd"
#define MATORDERINGRCM         "rcm"
#define MATORDERINGQMD         "qmd"
#define MATORDERINGROWLENGTH   "rowlength"
#define MATORDERINGAMD         "amd"            /* only works if UMFPACK is installed with PETSc */

PETSC_EXTERN PetscErrorCode MatGetOrdering(Mat,const MatOrderingType,IS*,IS*);
PETSC_EXTERN PetscErrorCode MatGetOrderingList(PetscFList *list);
PETSC_EXTERN PetscErrorCode MatOrderingRegister(const char[],const char[],const char[],PetscErrorCode(*)(Mat,const MatOrderingType,IS*,IS*));

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

PETSC_EXTERN PetscErrorCode MatOrderingRegisterDestroy(void);
PETSC_EXTERN PetscErrorCode MatOrderingRegisterAll(const char[]);
PETSC_EXTERN PetscBool MatOrderingRegisterAllCalled;
PETSC_EXTERN PetscFList MatOrderingList;

PETSC_EXTERN PetscErrorCode MatReorderForNonzeroDiagonal(Mat,PetscReal,IS,IS);

/*S
    MatFactorShiftType - Numeric Shift.

   Level: beginner

S*/
typedef enum {MAT_SHIFT_NONE,MAT_SHIFT_NONZERO,MAT_SHIFT_POSITIVE_DEFINITE,MAT_SHIFT_INBLOCKS} MatFactorShiftType;
PETSC_EXTERN const char *MatFactorShiftTypes[];

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

PETSC_EXTERN PetscErrorCode MatFactorInfoInitialize(MatFactorInfo*);
PETSC_EXTERN PetscErrorCode MatCholeskyFactor(Mat,IS,const MatFactorInfo*);
PETSC_EXTERN PetscErrorCode MatCholeskyFactorSymbolic(Mat,Mat,IS,const MatFactorInfo*);
PETSC_EXTERN PetscErrorCode MatCholeskyFactorNumeric(Mat,Mat,const MatFactorInfo*);
PETSC_EXTERN PetscErrorCode MatLUFactor(Mat,IS,IS,const MatFactorInfo*);
PETSC_EXTERN PetscErrorCode MatILUFactor(Mat,IS,IS,const MatFactorInfo*);
PETSC_EXTERN PetscErrorCode MatLUFactorSymbolic(Mat,Mat,IS,IS,const MatFactorInfo*);
PETSC_EXTERN PetscErrorCode MatILUFactorSymbolic(Mat,Mat,IS,IS,const MatFactorInfo*);
PETSC_EXTERN PetscErrorCode MatICCFactorSymbolic(Mat,Mat,IS,const MatFactorInfo*);
PETSC_EXTERN PetscErrorCode MatICCFactor(Mat,IS,const MatFactorInfo*);
PETSC_EXTERN PetscErrorCode MatLUFactorNumeric(Mat,Mat,const MatFactorInfo*);
PETSC_EXTERN PetscErrorCode MatGetInertia(Mat,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode MatSolve(Mat,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatForwardSolve(Mat,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatBackwardSolve(Mat,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatSolveAdd(Mat,Vec,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatSolveTranspose(Mat,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatSolveTransposeAdd(Mat,Vec,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatSolves(Mat,Vecs,Vecs);

PETSC_EXTERN PetscErrorCode MatSetUnfactored(Mat);

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
PETSC_EXTERN PetscErrorCode MatSOR(Mat,Vec,PetscReal,MatSORType,PetscReal,PetscInt,PetscInt,Vec);

/* 
    These routines are for efficiently computing Jacobians via finite differences.
*/

/*J
    MatColoringType - String with the name of a PETSc matrix coloring or the creation function
       with an optional dynamic library name, for example 
       http://www.mcs.anl.gov/petsc/lib.a:coloringcreate()

   Level: beginner

.seealso: MatGetColoring()
J*/
#define MatColoringType char*
#define MATCOLORINGNATURAL "natural"
#define MATCOLORINGSL      "sl"
#define MATCOLORINGLF      "lf"
#define MATCOLORINGID      "id"

PETSC_EXTERN PetscErrorCode MatGetColoring(Mat,const MatColoringType,ISColoring*);
PETSC_EXTERN PetscErrorCode MatColoringRegister(const char[],const char[],const char[],PetscErrorCode(*)(Mat,MatColoringType,ISColoring *));

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

PETSC_EXTERN PetscBool MatColoringRegisterAllCalled;

PETSC_EXTERN PetscErrorCode MatColoringRegisterAll(const char[]);
PETSC_EXTERN PetscErrorCode MatColoringRegisterDestroy(void);
PETSC_EXTERN PetscErrorCode MatColoringPatch(Mat,PetscInt,PetscInt,ISColoringValue[],ISColoring*);

/*S
     MatFDColoring - Object for computing a sparse Jacobian via finite differences
        and coloring

   Level: beginner

  Concepts: coloring, sparse Jacobian, finite differences

.seealso:  MatFDColoringCreate()
S*/
typedef struct _p_MatFDColoring* MatFDColoring;

PETSC_EXTERN PetscErrorCode MatFDColoringCreate(Mat,ISColoring,MatFDColoring *);
PETSC_EXTERN PetscErrorCode MatFDColoringDestroy(MatFDColoring*);
PETSC_EXTERN PetscErrorCode MatFDColoringView(MatFDColoring,PetscViewer);
PETSC_EXTERN PetscErrorCode MatFDColoringSetFunction(MatFDColoring,PetscErrorCode (*)(void),void*);
PETSC_EXTERN PetscErrorCode MatFDColoringGetFunction(MatFDColoring,PetscErrorCode (**)(void),void**);
PETSC_EXTERN PetscErrorCode MatFDColoringSetParameters(MatFDColoring,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode MatFDColoringSetFromOptions(MatFDColoring);
PETSC_EXTERN PetscErrorCode MatFDColoringApply(Mat,MatFDColoring,Vec,MatStructure*,void *);
PETSC_EXTERN PetscErrorCode MatFDColoringSetF(MatFDColoring,Vec);
PETSC_EXTERN PetscErrorCode MatFDColoringGetPerturbedColumns(MatFDColoring,PetscInt*,PetscInt*[]);

/*S
     MatTransposeColoring - Object for computing a sparse matrix product C=A*B^T via coloring

   Level: beginner

  Concepts: coloring, sparse matrix product

.seealso:  MatTransposeColoringCreate()
S*/
typedef struct _p_MatTransposeColoring* MatTransposeColoring;

PETSC_EXTERN PetscErrorCode MatTransposeColoringCreate(Mat,ISColoring,MatTransposeColoring *);
PETSC_EXTERN PetscErrorCode MatTransColoringApplySpToDen(MatTransposeColoring,Mat,Mat);
PETSC_EXTERN PetscErrorCode MatTransColoringApplyDenToSp(MatTransposeColoring,Mat,Mat);
PETSC_EXTERN PetscErrorCode MatTransposeColoringDestroy(MatTransposeColoring*);

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

/*J
    MatPartitioningType - String with the name of a PETSc matrix partitioning or the creation function
       with an optional dynamic library name, for example 
       http://www.mcs.anl.gov/petsc/lib.a:partitioningcreate()

   Level: beginner
dm
.seealso: MatPartitioningCreate(), MatPartitioning
J*/
#define MatPartitioningType char*
#define MATPARTITIONINGCURRENT  "current"
#define MATPARTITIONINGSQUARE   "square"
#define MATPARTITIONINGPARMETIS "parmetis"
#define MATPARTITIONINGCHACO    "chaco"
#define MATPARTITIONINGPARTY    "party"
#define MATPARTITIONINGPTSCOTCH "ptscotch"


PETSC_EXTERN PetscErrorCode MatPartitioningCreate(MPI_Comm,MatPartitioning*);
PETSC_EXTERN PetscErrorCode MatPartitioningSetType(MatPartitioning,const MatPartitioningType);
PETSC_EXTERN PetscErrorCode MatPartitioningSetNParts(MatPartitioning,PetscInt);
PETSC_EXTERN PetscErrorCode MatPartitioningSetAdjacency(MatPartitioning,Mat);
PETSC_EXTERN PetscErrorCode MatPartitioningSetVertexWeights(MatPartitioning,const PetscInt[]);
PETSC_EXTERN PetscErrorCode MatPartitioningSetPartitionWeights(MatPartitioning,const PetscReal []);
PETSC_EXTERN PetscErrorCode MatPartitioningApply(MatPartitioning,IS*);
PETSC_EXTERN PetscErrorCode MatPartitioningDestroy(MatPartitioning*);

PETSC_EXTERN PetscErrorCode MatPartitioningRegister(const char[],const char[],const char[],PetscErrorCode (*)(MatPartitioning));

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

PETSC_EXTERN PetscBool MatPartitioningRegisterAllCalled;

PETSC_EXTERN PetscErrorCode MatPartitioningRegisterAll(const char[]);
PETSC_EXTERN PetscErrorCode MatPartitioningRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode MatPartitioningView(MatPartitioning,PetscViewer);
PETSC_EXTERN PetscErrorCode MatPartitioningSetFromOptions(MatPartitioning);
PETSC_EXTERN PetscErrorCode MatPartitioningGetType(MatPartitioning,const MatPartitioningType*);

PETSC_EXTERN PetscErrorCode MatPartitioningParmetisSetCoarseSequential(MatPartitioning);
PETSC_EXTERN PetscErrorCode MatPartitioningParmetisGetEdgeCut(MatPartitioning, PetscInt *);

typedef enum { MP_CHACO_MULTILEVEL=1,MP_CHACO_SPECTRAL=2,MP_CHACO_LINEAR=4,MP_CHACO_RANDOM=5,MP_CHACO_SCATTERED=6 } MPChacoGlobalType;
PETSC_EXTERN const char *MPChacoGlobalTypes[];
typedef enum { MP_CHACO_KERNIGHAN=1,MP_CHACO_NONE=2 } MPChacoLocalType;
PETSC_EXTERN const char *MPChacoLocalTypes[];
typedef enum { MP_CHACO_LANCZOS=0,MP_CHACO_RQI=1 } MPChacoEigenType;
PETSC_EXTERN const char *MPChacoEigenTypes[];

PETSC_EXTERN PetscErrorCode MatPartitioningChacoSetGlobal(MatPartitioning,MPChacoGlobalType);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoGetGlobal(MatPartitioning,MPChacoGlobalType*);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoSetLocal(MatPartitioning,MPChacoLocalType);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoGetLocal(MatPartitioning,MPChacoLocalType*);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoSetCoarseLevel(MatPartitioning,PetscReal);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoSetEigenSolver(MatPartitioning,MPChacoEigenType);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoGetEigenSolver(MatPartitioning,MPChacoEigenType*);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoSetEigenTol(MatPartitioning,PetscReal);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoGetEigenTol(MatPartitioning,PetscReal*);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoSetEigenNumber(MatPartitioning,PetscInt);
PETSC_EXTERN PetscErrorCode MatPartitioningChacoGetEigenNumber(MatPartitioning,PetscInt*);

#define MP_PARTY_OPT "opt"
#define MP_PARTY_LIN "lin"
#define MP_PARTY_SCA "sca"
#define MP_PARTY_RAN "ran"
#define MP_PARTY_GBF "gbf"
#define MP_PARTY_GCF "gcf"
#define MP_PARTY_BUB "bub"
#define MP_PARTY_DEF "def"
PETSC_EXTERN PetscErrorCode MatPartitioningPartySetGlobal(MatPartitioning,const char*);
#define MP_PARTY_HELPFUL_SETS "hs"
#define MP_PARTY_KERNIGHAN_LIN "kl"
#define MP_PARTY_NONE "no"
PETSC_EXTERN PetscErrorCode MatPartitioningPartySetLocal(MatPartitioning,const char*);
PETSC_EXTERN PetscErrorCode MatPartitioningPartySetCoarseLevel(MatPartitioning,PetscReal);
PETSC_EXTERN PetscErrorCode MatPartitioningPartySetBipart(MatPartitioning,PetscBool);
PETSC_EXTERN PetscErrorCode MatPartitioningPartySetMatchOptimization(MatPartitioning,PetscBool);

typedef enum { MP_PTSCOTCH_QUALITY,MP_PTSCOTCH_SPEED,MP_PTSCOTCH_BALANCE,MP_PTSCOTCH_SAFETY,MP_PTSCOTCH_SCALABILITY } MPPTScotchStrategyType;
PETSC_EXTERN const char *MPPTScotchStrategyTypes[];

PETSC_EXTERN PetscErrorCode MatPartitioningPTScotchSetImbalance(MatPartitioning,PetscReal);
PETSC_EXTERN PetscErrorCode MatPartitioningPTScotchGetImbalance(MatPartitioning,PetscReal*);
PETSC_EXTERN PetscErrorCode MatPartitioningPTScotchSetStrategy(MatPartitioning,MPPTScotchStrategyType);
PETSC_EXTERN PetscErrorCode MatPartitioningPTScotchGetStrategy(MatPartitioning,MPPTScotchStrategyType*);

/* 
    These routines are for coarsening matrices: 
*/

/*S
     MatCoarsen - Object for managing the coarsening of a graph (symmetric matrix)

   Level: beginner

  Concepts: coarsen

.seealso:  MatCoarsenCreate), MatCoarsenType
S*/
typedef struct _p_MatCoarsen* MatCoarsen;

/*J
    MatCoarsenType - String with the name of a PETSc matrix coarsen or the creation function
       with an optional dynamic library name, for example 
       http://www.mcs.anl.gov/petsc/lib.a:coarsencreate()

   Level: beginner
dm
.seealso: MatCoarsenCreate(), MatCoarsen
J*/
#define MatCoarsenType char*
#define MATCOARSENMIS  "mis"
#define MATCOARSENHEM  "hem"

/* linked list for aggregates */
typedef struct _PetscCDIntNd{
  struct _PetscCDIntNd *next;
  PetscInt   gid;
}PetscCDIntNd;

/* only used by node pool */
typedef struct _PetscCDArrNd{
  struct _PetscCDArrNd *next;
  struct _PetscCDIntNd *array; 
}PetscCDArrNd;

typedef struct _PetscCoarsenData{
  /* node pool */
  PetscCDArrNd  pool_list;
  PetscCDIntNd *new_node;
  PetscInt   new_left;
  PetscInt   chk_sz;
  PetscCDIntNd *extra_nodes;
  /* Array of lists */
  PetscCDIntNd **array;
  PetscInt size;
  /* cache a Mat for communication data */
  Mat mat;
  /* cache IS of removed equations */
  IS removedIS;
}PetscCoarsenData;

PETSC_EXTERN PetscErrorCode MatCoarsenCreate(MPI_Comm,MatCoarsen*);
PETSC_EXTERN PetscErrorCode MatCoarsenSetType(MatCoarsen,const MatCoarsenType);
PETSC_EXTERN PetscErrorCode MatCoarsenSetAdjacency(MatCoarsen,Mat);
PETSC_EXTERN PetscErrorCode MatCoarsenSetGreedyOrdering(MatCoarsen,const IS);
PETSC_EXTERN PetscErrorCode MatCoarsenSetStrictAggs(MatCoarsen,PetscBool);
PETSC_EXTERN PetscErrorCode MatCoarsenSetVerbose(MatCoarsen,PetscInt);
PETSC_EXTERN PetscErrorCode MatCoarsenGetData( MatCoarsen, PetscCoarsenData ** );
PETSC_EXTERN PetscErrorCode MatCoarsenApply(MatCoarsen);
PETSC_EXTERN PetscErrorCode MatCoarsenDestroy(MatCoarsen*);

PETSC_EXTERN PetscErrorCode MatCoarsenRegister(const char[],const char[],const char[],PetscErrorCode (*)(MatCoarsen));

/*MC
   MatCoarsenRegisterDynamic - Adds a new sparse matrix coarsen to the 
   matrix package. 

   Synopsis:
   PetscErrorCode MatCoarsenRegisterDynamic(const char *name_coarsen,const char *path,const char *name_create,PetscErrorCode (*routine_create)(MatCoarsen))

   Not Collective

   Input Parameters:
+  sname - name of coarsen (for example MATCOARSENMIS) 
.  path - location of library where creation routine is 
.  name - name of function that creates the coarsen type, a string
-  function - function pointer that creates the coarsen type

   Level: developer

   If dynamic libraries are used, then the fourth input argument (function)
   is ignored.

   Sample usage:
.vb
   MatCoarsenRegisterDynamic("my_agg",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyAggCreate",MyAggCreate);
.ve

   Then, your aggregator can be chosen with the procedural interface via
$     MatCoarsenSetType(agg,"my_agg")
   or at runtime via the option
$     -mat_coarsen_type my_agg

   $PETSC_ARCH occuring in pathname will be replaced with appropriate values.

.keywords: matrix, coarsen, register

.seealso: MatCoarsenRegisterDestroy(), MatCoarsenRegisterAll()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define MatCoarsenRegisterDynamic(a,b,c,d) MatCoarsenRegister(a,b,c,0)
#else
#define MatCoarsenRegisterDynamic(a,b,c,d) MatCoarsenRegister(a,b,c,d)
#endif

PETSC_EXTERN PetscBool MatCoarsenRegisterAllCalled;

PETSC_EXTERN PetscErrorCode MatCoarsenRegisterAll(const char[]);
PETSC_EXTERN PetscErrorCode MatCoarsenRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode MatCoarsenView(MatCoarsen,PetscViewer);
PETSC_EXTERN PetscErrorCode MatCoarsenSetFromOptions(MatCoarsen);
PETSC_EXTERN PetscErrorCode MatCoarsenGetType(MatCoarsen,const MatCoarsenType*);


PETSC_EXTERN PetscErrorCode MatMeshToVertexGraph(Mat,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode MatMeshToCellGraph(Mat,PetscInt,Mat*);

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
               MATOP_SET_VALUES_BATCH=129,
               MATOP_TRANSPOSEMATMULT=130,
               MATOP_TRANSPOSEMATMULT_SYMBOLIC=131,
               MATOP_TRANSPOSEMATMULT_NUMERIC=132,
               MATOP_TRANSPOSECOLORING_CREATE=133,
               MATOP_TRANSCOLORING_APPLY_SPTODEN=134,
               MATOP_TRANSCOLORING_APPLY_DENTOSP=135,
               MATOP_RARt=136,
               MATOP_RARt_SYMBOLIC=137,
               MATOP_RARt_NUMERIC=138,
               MATOP_SET_BLOCK_SIZES=139
             } MatOperation;
PETSC_EXTERN PetscErrorCode MatHasOperation(Mat,MatOperation,PetscBool *);
PETSC_EXTERN PetscErrorCode MatShellSetOperation(Mat,MatOperation,void(*)(void));
PETSC_EXTERN PetscErrorCode MatShellGetOperation(Mat,MatOperation,void(**)(void));
PETSC_EXTERN PetscErrorCode MatShellSetContext(Mat,void*);

/*
   Codes for matrices stored on disk. By default they are
   stored in a universal format. By changing the format with 
   PetscViewerSetFormat(viewer,PETSC_VIEWER_NATIVE); the matrices will
   be stored in a way natural for the matrix, for example dense matrices
   would be stored as dense. Matrices stored this way may only be
   read into matrices of the same type.
*/
#define MATRIX_BINARY_FORMAT_DENSE -1

PETSC_EXTERN PetscErrorCode MatMPIBAIJSetHashTableFactor(Mat,PetscReal);
PETSC_EXTERN PetscErrorCode MatISGetLocalMat(Mat,Mat*);
PETSC_EXTERN PetscErrorCode MatISSetLocalMat(Mat,Mat);

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

PETSC_EXTERN PetscErrorCode MatNullSpaceCreate(MPI_Comm,PetscBool ,PetscInt,const Vec[],MatNullSpace*);
PETSC_EXTERN PetscErrorCode MatNullSpaceSetFunction(MatNullSpace,PetscErrorCode (*)(MatNullSpace,Vec,void*),void*);
PETSC_EXTERN PetscErrorCode MatNullSpaceDestroy(MatNullSpace*);
PETSC_EXTERN PetscErrorCode MatNullSpaceRemove(MatNullSpace,Vec,Vec*);
PETSC_EXTERN PetscErrorCode MatGetNullSpace(Mat, MatNullSpace *);
PETSC_EXTERN PetscErrorCode MatSetNullSpace(Mat,MatNullSpace);
PETSC_EXTERN PetscErrorCode MatSetNearNullSpace(Mat,MatNullSpace);
PETSC_EXTERN PetscErrorCode MatGetNearNullSpace(Mat,MatNullSpace*);
PETSC_EXTERN PetscErrorCode MatNullSpaceTest(MatNullSpace,Mat,PetscBool  *);
PETSC_EXTERN PetscErrorCode MatNullSpaceView(MatNullSpace,PetscViewer);
PETSC_EXTERN PetscErrorCode MatNullSpaceGetVecs(MatNullSpace,PetscBool*,PetscInt*,const Vec**);
PETSC_EXTERN PetscErrorCode MatNullSpaceCreateRigidBody(Vec,MatNullSpace*);

PETSC_EXTERN PetscErrorCode MatReorderingSeqSBAIJ(Mat,IS);
PETSC_EXTERN PetscErrorCode MatMPISBAIJSetHashTableFactor(Mat,PetscReal);
PETSC_EXTERN PetscErrorCode MatSeqSBAIJSetColumnIndices(Mat,PetscInt *);
PETSC_EXTERN PetscErrorCode MatSeqBAIJInvertBlockDiagonal(Mat);

PETSC_EXTERN PetscErrorCode MatCreateMAIJ(Mat,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode MatMAIJRedimension(Mat,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode MatMAIJGetAIJ(Mat,Mat*);

PETSC_EXTERN PetscErrorCode MatComputeExplicitOperator(Mat,Mat*);

PETSC_EXTERN PetscErrorCode MatDiagonalScaleLocal(Mat,Vec);

PETSC_EXTERN PetscErrorCode MatCreateMFFD(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode MatMFFDSetBase(Mat,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatMFFDSetFunction(Mat,PetscErrorCode(*)(void*,Vec,Vec),void*);
PETSC_EXTERN PetscErrorCode MatMFFDSetFunctioni(Mat,PetscErrorCode (*)(void*,PetscInt,Vec,PetscScalar*));
PETSC_EXTERN PetscErrorCode MatMFFDSetFunctioniBase(Mat,PetscErrorCode (*)(void*,Vec));
PETSC_EXTERN PetscErrorCode MatMFFDAddNullSpace(Mat,MatNullSpace);
PETSC_EXTERN PetscErrorCode MatMFFDSetHHistory(Mat,PetscScalar[],PetscInt);
PETSC_EXTERN PetscErrorCode MatMFFDResetHHistory(Mat);
PETSC_EXTERN PetscErrorCode MatMFFDSetFunctionError(Mat,PetscReal);
PETSC_EXTERN PetscErrorCode MatMFFDSetPeriod(Mat,PetscInt);
PETSC_EXTERN PetscErrorCode MatMFFDGetH(Mat,PetscScalar *);
PETSC_EXTERN PetscErrorCode MatMFFDSetOptionsPrefix(Mat,const char[]);
PETSC_EXTERN PetscErrorCode MatMFFDCheckPositivity(void*,Vec,Vec,PetscScalar*);
PETSC_EXTERN PetscErrorCode MatMFFDSetCheckh(Mat,PetscErrorCode (*)(void*,Vec,Vec,PetscScalar*),void*);

/*S
    MatMFFD - A data structured used to manage the computation of the h differencing parameter for matrix-free 
              Jacobian vector products

    Notes: MATMFFD is a specific MatType which uses the MatMFFD data structure

           MatMFFD*() methods actually take the Mat as their first argument. Not a MatMFFD data structure

    Level: developer

.seealso: MATMFFD, MatCreateMFFD(), MatMFFDSetFuction(), MatMFFDSetType(), MatMFFDRegister()
S*/
typedef struct _p_MatMFFD* MatMFFD;

/*J
    MatMFFDType - algorithm used to compute the h used in computing matrix-vector products via differencing of the function

   Level: beginner

.seealso: MatMFFDSetType(), MatMFFDRegister()
J*/
#define MatMFFDType char*
#define MATMFFD_DS  "ds"
#define MATMFFD_WP  "wp"

PETSC_EXTERN PetscErrorCode MatMFFDSetType(Mat,const MatMFFDType);
PETSC_EXTERN PetscErrorCode MatMFFDRegister(const char[],const char[],const char[],PetscErrorCode (*)(MatMFFD));

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

PETSC_EXTERN PetscErrorCode MatMFFDRegisterAll(const char[]);
PETSC_EXTERN PetscErrorCode MatMFFDRegisterDestroy(void);
PETSC_EXTERN PetscErrorCode MatMFFDDSSetUmin(Mat,PetscReal);
PETSC_EXTERN PetscErrorCode MatMFFDWPSetComputeNormU(Mat,PetscBool );


PETSC_EXTERN PetscErrorCode PetscViewerMathematicaPutMatrix(PetscViewer, PetscInt, PetscInt, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscViewerMathematicaPutCSRMatrix(PetscViewer, PetscInt, PetscInt, PetscInt *, PetscInt *, PetscReal *);

/* 
   PETSc interface to MUMPS 
*/
#ifdef PETSC_HAVE_MUMPS
PETSC_EXTERN PetscErrorCode MatMumpsSetIcntl(Mat,PetscInt,PetscInt);
#endif

/* 
   PETSc interface to SUPERLU
*/
#ifdef PETSC_HAVE_SUPERLU
PETSC_EXTERN PetscErrorCode MatSuperluSetILUDropTol(Mat,PetscReal);
#endif

#if defined(PETSC_HAVE_CUSP)
PETSC_EXTERN PetscErrorCode MatCreateSeqAIJCUSP(MPI_Comm,PetscInt,PetscInt,PetscInt,const PetscInt[],Mat*);
PETSC_EXTERN PetscErrorCode MatCreateAIJCUSP(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Mat*);
#endif

/* 
   PETSc interface to FFTW
*/
#if defined(PETSC_HAVE_FFTW)
PETSC_EXTERN PetscErrorCode VecScatterPetscToFFTW(Mat,Vec,Vec);
PETSC_EXTERN PetscErrorCode VecScatterFFTWToPetsc(Mat,Vec,Vec);
PETSC_EXTERN PetscErrorCode MatGetVecsFFTW(Mat,Vec*,Vec*,Vec*);
#endif

PETSC_EXTERN PetscErrorCode MatCreateNest(MPI_Comm,PetscInt,const IS[],PetscInt,const IS[],const Mat[],Mat*);
PETSC_EXTERN PetscErrorCode MatNestGetSize(Mat,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode MatNestGetISs(Mat,IS[],IS[]);
PETSC_EXTERN PetscErrorCode MatNestGetLocalISs(Mat,IS[],IS[]);
PETSC_EXTERN PetscErrorCode MatNestGetSubMats(Mat,PetscInt*,PetscInt*,Mat***);
PETSC_EXTERN PetscErrorCode MatNestGetSubMat(Mat,PetscInt,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode MatNestSetVecType(Mat,const VecType);
PETSC_EXTERN PetscErrorCode MatNestSetSubMats(Mat,PetscInt,const IS[],PetscInt,const IS[],const Mat[]);
PETSC_EXTERN PetscErrorCode MatNestSetSubMat(Mat,PetscInt,PetscInt,Mat);

/* 
 MatIJ: 
 An unweighted directed pseudograph
 An interpretation of this matrix as a (pseudo)graph allows us to define additional operations on it:
 A MatIJ can act on sparse arrays: arrays of indices, or index arrays of integers, scalars, or integer-scalar pairs
 by mapping the indices to the indices connected to them by the (pseudo)graph ed
 */
typedef enum {MATIJ_LOCAL, MATIJ_GLOBAL} MatIJIndexType; 
PETSC_EXTERN PetscErrorCode MatIJSetMultivalued(Mat, PetscBool);
PETSC_EXTERN PetscErrorCode MatIJGetMultivalued(Mat, PetscBool*);
PETSC_EXTERN PetscErrorCode MatIJSetEdges(Mat, PetscInt, const PetscInt*, const PetscInt*);
PETSC_EXTERN PetscErrorCode MatIJGetEdges(Mat, PetscInt *, PetscInt **, PetscInt **);
PETSC_EXTERN PetscErrorCode MatIJSetEdgesIS(Mat, IS, IS);
PETSC_EXTERN PetscErrorCode MatIJGetEdgesIS(Mat, IS*, IS*);
PETSC_EXTERN PetscErrorCode MatIJGetRowSizes(Mat, MatIJIndexType, PetscInt, const PetscInt *, PetscInt **);
PETSC_EXTERN PetscErrorCode MatIJGetMinRowSize(Mat, PetscInt *);
PETSC_EXTERN PetscErrorCode MatIJGetMaxRowSize(Mat, PetscInt *);
PETSC_EXTERN PetscErrorCode MatIJGetSupport(Mat,  PetscInt *, PetscInt **);
PETSC_EXTERN PetscErrorCode MatIJGetSupportIS(Mat, IS *);
PETSC_EXTERN PetscErrorCode MatIJGetImage(Mat, PetscInt*, PetscInt**);
PETSC_EXTERN PetscErrorCode MatIJGetImageIS(Mat, IS *);
PETSC_EXTERN PetscErrorCode MatIJGetSupportSize(Mat, PetscInt *);
PETSC_EXTERN PetscErrorCode MatIJGetImageSize(Mat, PetscInt *);

PETSC_EXTERN PetscErrorCode MatIJBinRenumber(Mat, Mat*);

PETSC_EXTERN PetscErrorCode MatIJMap(Mat, MatIJIndexType, PetscInt,const PetscInt*,const PetscInt*,const PetscScalar*, MatIJIndexType,PetscInt*,PetscInt**,PetscInt**,PetscScalar**,PetscInt**);
PETSC_EXTERN PetscErrorCode MatIJBin(Mat, MatIJIndexType, PetscInt,const PetscInt*,const PetscInt*,const PetscScalar*,PetscInt*,PetscInt**,PetscInt**,PetscScalar**,PetscInt**);
PETSC_EXTERN PetscErrorCode MatIJBinMap(Mat,Mat, MatIJIndexType,PetscInt,const PetscInt*,const PetscInt*,const PetscScalar*,MatIJIndexType,PetscInt*,PetscInt**,PetscInt**,PetscScalar**,PetscInt**);

#endif
