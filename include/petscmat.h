/* $Id: petscmat.h,v 1.192 2000/05/17 02:45:02 bsmith Exp bsmith $ */
/*
     Include file for the matrix component of PETSc
*/
#ifndef __PETSCMAT_H
#define __PETSCMAT_H
#include "petscvec.h"

#define MAT_COOKIE         PETSC_COOKIE+5

typedef struct _p_Mat*           Mat;

#define MAX_MATRIX_TYPES 14
/*
   The default matrix data storage formats and routines to create them.
  
   MATLASTTYPE is "end-of-list" marker that can be used to check that
   MAX_MATRIX_TYPES is large enough.  The rule is 
   MAX_MATRIX_TYPES >= MATLASTTYPE .  

   To do: add a test program that checks the consistency of these values.
*/
typedef enum { MATSAME=-1,  MATSEQDENSE, MATSEQAIJ,   MATMPIAIJ,   MATSHELL, 
               MATMPIROWBS, MATSEQBDIAG, MATMPIBDIAG, MATMPIDENSE, MATSEQBAIJ,
               MATMPIBAIJ,  MATMPICSN,   MATSEQCSN,   MATMPIADJ, MATSEQSBAIJ,
               MATMPISBAIJ, MATLASTTYPE } MatType;

EXTERN int MatCreate(MPI_Comm,int,int,int,int,Mat*);
EXTERN int MatCreateSeqDense(MPI_Comm,int,int,Scalar*,Mat*);
EXTERN int MatCreateMPIDense(MPI_Comm,int,int,int,int,Scalar*,Mat*); 
EXTERN int MatCreateSeqAIJ(MPI_Comm,int,int,int,int*,Mat*);
EXTERN int MatCreateMPIAIJ(MPI_Comm,int,int,int,int,int,int*,int,int*,Mat*); 
EXTERN int MatCreateMPIRowbs(MPI_Comm,int,int,int,int*,void*,Mat*); 
EXTERN int MatCreateSeqBDiag(MPI_Comm,int,int,int,int,int*,Scalar**,Mat*); 
EXTERN int MatCreateMPIBDiag(MPI_Comm,int,int,int,int,int,int*,Scalar**,Mat*); 
EXTERN int MatCreateSeqBAIJ(MPI_Comm,int,int,int,int,int*,Mat*); 
EXTERN int MatCreateMPIBAIJ(MPI_Comm,int,int,int,int,int,int,int*,int,int*,Mat*);
EXTERN int MatCreateMPIAdj(MPI_Comm,int,int,int*,int*,int *,Mat*);
EXTERN int MatCreateSeqSBAIJ(MPI_Comm,int,int,int,int,int*,Mat*); 
EXTERN int MatCreateMPISBAIJ(MPI_Comm,int,int,int,int,int,int,int*,int,int*,Mat*);


EXTERN int MatDestroy(Mat);

EXTERN int MatCreateShell(MPI_Comm,int,int,int,int,void *,Mat*);
EXTERN int MatShellGetContext(Mat,void **);

EXTERN int MatPrintHelp(Mat);
EXTERN int MatGetMaps(Mat,Map*,Map*);

/* ------------------------------------------------------------*/
EXTERN int MatSetValues(Mat,int,int*,int,int*,Scalar*,InsertMode);
EXTERN int MatSetValuesBlocked(Mat,int,int*,int,int*,Scalar*,InsertMode);

typedef enum {MAT_FLUSH_ASSEMBLY=1,MAT_FINAL_ASSEMBLY=0} MatAssemblyType;
EXTERN int MatAssemblyBegin(Mat,MatAssemblyType);
EXTERN int MatAssemblyEnd(Mat,MatAssemblyType);
EXTERN int MatAssembled(Mat,PetscTruth*);

#define MatSetValue(v,i,j,va,mode) \
{int _ierr,_row = i,_col = j; Scalar _va = va; \
  _ierr = MatSetValues(v,1,&_row,1,&_col,&_va,mode);CHKERRQ(_ierr); \
}
#define MatGetValue(v,i,j,va) \
{int _ierr,_row = i,_col = j; \
  _ierr = MatGetValues(v,1,&_row,1,&_col,&va);CHKERRQ(_ierr); \
}
/*
   Any additions/changes here MUST also be made in include/finclude/petscmat.h
*/
typedef enum {MAT_ROW_ORIENTED=1,MAT_COLUMN_ORIENTED=2,MAT_ROWS_SORTED=4,
              MAT_COLUMNS_SORTED=8,MAT_NO_NEW_NONZERO_LOCATIONS=16,
              MAT_YES_NEW_NONZERO_LOCATIONS=32,MAT_SYMMETRIC=64,
              MAT_STRUCTURALLY_SYMMETRIC=65,MAT_NO_NEW_DIAGONALS=66,
              MAT_YES_NEW_DIAGONALS=67,MAT_INODE_LIMIT_1=68,MAT_INODE_LIMIT_2=69,
              MAT_INODE_LIMIT_3=70,MAT_INODE_LIMIT_4=71,MAT_INODE_LIMIT_5=72,
              MAT_IGNORE_OFF_PROC_ENTRIES=73,MAT_ROWS_UNSORTED=74,
              MAT_COLUMNS_UNSORTED=75,MAT_NEW_NONZERO_LOCATION_ERR=76,
              MAT_NEW_NONZERO_ALLOCATION_ERR=77,MAT_USE_HASH_TABLE=78,
              MAT_KEEP_ZEROED_ROWS=79,MAT_IGNORE_ZERO_ENTRIES=80,MAT_USE_INODES=81,
              MAT_DO_NOT_USE_INODES} MatOption;
EXTERN int MatSetOption(Mat,MatOption);
EXTERN int MatGetType(Mat,MatType*,char**);
EXTERN int MatGetTypeFromOptions(MPI_Comm,char*,MatType*,PetscTruth*);

EXTERN int MatGetValues(Mat,int,int*,int,int*,Scalar*);
EXTERN int MatGetRow(Mat,int,int *,int **,Scalar**);
EXTERN int MatRestoreRow(Mat,int,int *,int **,Scalar**);
EXTERN int MatGetColumn(Mat,int,int *,int **,Scalar**);
EXTERN int MatRestoreColumn(Mat,int,int *,int **,Scalar**);
EXTERN int MatGetColumnVector(Mat,Vec,int);
EXTERN int MatGetArray(Mat,Scalar **);
EXTERN int MatRestoreArray(Mat,Scalar **);
EXTERN int MatGetBlockSize(Mat,int *);

EXTERN int MatMult(Mat,Vec,Vec);
EXTERN int MatMultAdd(Mat,Vec,Vec,Vec);
EXTERN int MatMultTranspose(Mat,Vec,Vec);
EXTERN int MatMultTransposeAdd(Mat,Vec,Vec,Vec);

typedef enum {MAT_DO_NOT_COPY_VALUES,MAT_COPY_VALUES} MatDuplicateOption;

EXTERN int MatConvert(Mat,MatType,Mat*);
EXTERN int MatDuplicate(Mat,MatDuplicateOption,Mat*);
EXTERN int MatConvertRegister(MatType,MatType,int (*)(Mat,MatType,Mat*));
EXTERN int MatConvertRegisterAll(void);

typedef enum {SAME_NONZERO_PATTERN,DIFFERENT_NONZERO_PATTERN,SAME_PRECONDITIONER} MatStructure;

EXTERN int MatCopy(Mat,Mat,MatStructure);
EXTERN int MatView(Mat,Viewer);
EXTERN int MatLoad(Viewer,MatType,Mat*);
EXTERN int MatLoadRegister(MatType,int (*)(Viewer,MatType,Mat*));
EXTERN int MatLoadRegisterAll(void);

EXTERN int MatGetRowIJ(Mat,int,PetscTruth,int*,int **,int **,PetscTruth *);
EXTERN int MatRestoreRowIJ(Mat,int,PetscTruth,int *,int **,int **,PetscTruth *);
EXTERN int MatGetColumnIJ(Mat,int,PetscTruth,int*,int **,int **,PetscTruth *);
EXTERN int MatRestoreColumnIJ(Mat,int,PetscTruth,int *,int **,int **,PetscTruth *);

/* 
   Context of matrix information, used with MatGetInfo()
   Note: If any entries are added to this context, be sure
         to adjust MAT_INFO_SIZE in finclude/petscmat.h
 */
typedef struct {
  PLogDouble rows_global,columns_global;         /* number of global rows and columns */
  PLogDouble rows_local,columns_local;           /* number of local rows and columns */
  PLogDouble block_size;                          /* block size */
  PLogDouble nz_allocated,nz_used,nz_unneeded;  /* number of nonzeros */
  PLogDouble memory;                              /* memory allocated */
  PLogDouble assemblies;                          /* number of matrix assemblies */
  PLogDouble mallocs;                             /* number of mallocs during MatSetValues() */
  PLogDouble fill_ratio_given,fill_ratio_needed; /* fill ratio for LU/ILU */
  PLogDouble factor_mallocs;                      /* number of mallocs during factorization */
} MatInfo;

typedef enum {MAT_LOCAL=1,MAT_GLOBAL_MAX=2,MAT_GLOBAL_SUM=3} MatInfoType;
EXTERN int MatGetInfo(Mat,MatInfoType,MatInfo*);
EXTERN int MatValid(Mat,PetscTruth*);
EXTERN int MatGetDiagonal(Mat,Vec);
EXTERN int MatTranspose(Mat,Mat*);
EXTERN int MatPermute(Mat,IS,IS,Mat *);
EXTERN int MatDiagonalScale(Mat,Vec,Vec);
EXTERN int MatDiagonalShift(Mat,Vec);
EXTERN int MatEqual(Mat,Mat,PetscTruth*);

EXTERN int MatNorm(Mat,NormType,double *);
EXTERN int MatZeroEntries(Mat);
EXTERN int MatZeroRows(Mat,IS,Scalar*);
EXTERN int MatZeroColumns(Mat,IS,Scalar*);

EXTERN int MatUseScaledForm(Mat,PetscTruth);
EXTERN int MatScaleSystem(Mat,Vec,Vec);
EXTERN int MatUnScaleSystem(Mat,Vec,Vec);

EXTERN int MatGetSize(Mat,int*,int*);
EXTERN int MatGetLocalSize(Mat,int*,int*);
EXTERN int MatGetOwnershipRange(Mat,int*,int*);

typedef enum {MAT_INITIAL_MATRIX,MAT_REUSE_MATRIX} MatReuse;
EXTERN int MatGetSubMatrices(Mat,int,IS *,IS *,MatReuse,Mat **);
EXTERN int MatDestroyMatrices(int,Mat **);
EXTERN int MatGetSubMatrix(Mat,IS,IS,int,MatReuse,Mat *);

EXTERN int MatIncreaseOverlap(Mat,int,IS *,int);

EXTERN int MatAXPY(Scalar *,Mat,Mat);
EXTERN int MatAYPX(Scalar *,Mat,Mat);
EXTERN int MatCompress(Mat);

EXTERN int MatScale(Scalar *,Mat);
EXTERN int MatShift(Scalar *,Mat);

EXTERN int MatSetLocalToGlobalMapping(Mat,ISLocalToGlobalMapping);
EXTERN int MatSetLocalToGlobalMappingBlock(Mat,ISLocalToGlobalMapping);
EXTERN int MatZeroRowsLocal(Mat,IS,Scalar*);
EXTERN int MatSetValuesLocal(Mat,int,int*,int,int*,Scalar*,InsertMode);
EXTERN int MatSetValuesBlockedLocal(Mat,int,int*,int,int*,Scalar*,InsertMode);

EXTERN int MatSetStashInitialSize(Mat,int,int);

EXTERN int MatInterpolateAdd(Mat,Vec,Vec,Vec);
EXTERN int MatInterpolate(Mat,Vec,Vec);
EXTERN int MatRestrict(Mat,Vec,Vec);

/*
      These three macros MUST be used together. The third one closes the open { of the first one
*/
#define MatPreallocateInitialize(comm,nrows,ncols,dnz,onz) \
{ \
  int __ierr,__tmp = (nrows),__ctmp = (ncols),__rstart,__start,__end; \
  dnz = (int*)PetscMalloc(2*__tmp*sizeof(int));CHKPTRQ(dnz);onz = dnz + __tmp;\
  __ierr = PetscMemzero(dnz,2*__tmp*sizeof(int));CHKERRQ(__ierr);\
  __ierr = MPI_Scan(&__ctmp,&__end,1,MPI_INT,MPI_SUM,comm);CHKERRQ(__ierr); __start = __end - __ctmp;\
  __ierr = MPI_Scan(&__tmp,&__rstart,1,MPI_INT,MPI_SUM,comm);CHKERRQ(__ierr); __rstart = __rstart - __tmp;

#define MatPreallocateSet(row,nc,cols,dnz,onz)\
{ int __i; \
  for (__i=0; __i<nc; __i++) {\
    if (cols[__i] < __start || cols[__i] >= __end) onz[row - __rstart]++; \
  }\
  dnz[row - __rstart] = nc - onz[row - __rstart];\
}

#define MatPreallocateFinalize(dnz,onz) __ierr = PetscFree(dnz);CHKERRQ(__ierr);}

/* Routines unique to particular data structures */
EXTERN int MatBDiagGetData(Mat,int*,int*,int**,int**,Scalar***);
EXTERN int MatSeqAIJSetColumnIndices(Mat,int *);
EXTERN int MatSeqBAIJSetColumnIndices(Mat,int *);
EXTERN int MatCreateSeqAIJWithArrays(MPI_Comm,int,int,int*,int*,Scalar *,Mat*);

EXTERN int MatStoreValues(Mat);
EXTERN int MatRetrieveValues(Mat);

/* 
  These routines are not usually accessed directly, rather solving is 
  done through the SLES, KSP and PC interfaces.
*/

typedef char* MatOrderingType;
#define MATORDERING_NATURAL   "natural"
#define MATORDERING_ND        "nd"
#define MATORDERING_1WD       "1wd"
#define MATORDERING_RCM       "rcm"
#define MATORDERING_QMD       "qmd"
#define MATORDERING_ROWLENGTH "rowlength"

EXTERN int MatGetOrdering(Mat,MatOrderingType,IS*,IS*);
EXTERN int MatOrderingRegister(char*,char*,char*,int(*)(Mat,MatOrderingType,IS*,IS*));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define MatOrderingRegisterDynamic(a,b,c,d) MatOrderingRegister(a,b,c,0)
#else
#define MatOrderingRegisterDynamic(a,b,c,d) MatOrderingRegister(a,b,c,d)
#endif
EXTERN int        MatOrderingRegisterDestroy(void);
EXTERN int        MatOrderingRegisterAll(char*);
extern PetscTruth MatOrderingRegisterAllCalled;

EXTERN int MatReorderForNonzeroDiagonal(Mat,double,IS,IS);

EXTERN int MatCholeskyFactor(Mat,IS,double);
EXTERN int MatCholeskyFactorSymbolic(Mat,IS,double,Mat*);
EXTERN int MatCholeskyFactorNumeric(Mat,Mat*);

/* 
   Context of matrix information, used with MatILUFactor() and MatILUFactorSymbolic()
   of MatLUFactor() and MatLUFactorSymbolic()

   Note: If any entries are added to this context, be sure
         to adjust MAT_ILUINFO_SIZE in finclude/petscmat.h and/or
         to adjust MAT_LUINFO_SIZE  in finclude/petscmat.h

   Note: The integer values below are passed in double to allow easy use from Fortran
 */
typedef struct {
  double     levels;  /* ILU(levels) */ 
  double     fill;    /* expected fill; nonzeros in factored matrix/nonzeros in original matrix*/
  double     diagonal_fill;  /* force diagonal to fill in if initially not filled */

  double     dt;             /* drop tolerance */
  double     dtcol;          /* tolerance for pivoting */
  double     dtcount;        /* maximum nonzeros to be allowed per row */
} MatILUInfo;

typedef struct {
  double     fill;    /* expected fill; nonzeros in factored matrix/nonzeros in original matrix*/
  double     dtcol;   /* tolerance for pivoting; pivot if off_diagonal*dtcol > diagonal */
} MatLUInfo;

EXTERN int MatLUFactor(Mat,IS,IS,MatLUInfo*);
EXTERN int MatILUFactor(Mat,IS,IS,MatILUInfo*);
EXTERN int MatLUFactorSymbolic(Mat,IS,IS,MatLUInfo*,Mat*);
EXTERN int MatILUFactorSymbolic(Mat,IS,IS,MatILUInfo*,Mat*);
EXTERN int MatIncompleteCholeskyFactorSymbolic(Mat,IS,double,int,Mat*);
EXTERN int MatLUFactorNumeric(Mat,Mat*);
EXTERN int MatILUDTFactor(Mat,MatILUInfo*,IS,IS,Mat *);

EXTERN int MatSolve(Mat,Vec,Vec);
EXTERN int MatForwardSolve(Mat,Vec,Vec);
EXTERN int MatBackwardSolve(Mat,Vec,Vec);
EXTERN int MatSolveAdd(Mat,Vec,Vec,Vec);
EXTERN int MatSolveTranspose(Mat,Vec,Vec);
EXTERN int MatSolveTransposeAdd(Mat,Vec,Vec,Vec);

EXTERN int MatSetUnfactored(Mat);

/*  MatSORType may be bitwise ORd together, so do not change the numbers */

typedef enum {SOR_FORWARD_SWEEP=1,SOR_BACKWARD_SWEEP=2,SOR_SYMMETRIC_SWEEP=3,
              SOR_LOCAL_FORWARD_SWEEP=4,SOR_LOCAL_BACKWARD_SWEEP=8,
              SOR_LOCAL_SYMMETRIC_SWEEP=12,SOR_ZERO_INITIAL_GUESS=16,
              SOR_EISENSTAT=32,SOR_APPLY_UPPER=64,SOR_APPLY_LOWER=128} MatSORType;
EXTERN int MatRelax(Mat,Vec,double,MatSORType,double,int,Vec);

/* 
    These routines are for efficiently computing Jacobians via finite differences.
*/

typedef char* MatColoringType;
#define MATCOLORING_NATURAL "natural"
#define MATCOLORING_SL      "sl"
#define MATCOLORING_LF      "lf"
#define MATCOLORING_ID      "id"

EXTERN int MatGetColoring(Mat,MatColoringType,ISColoring*);
EXTERN int MatColoringRegister(char*,char*,char*,int(*)(Mat,MatColoringType,ISColoring *));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define MatColoringRegisterDynamic(a,b,c,d) MatColoringRegister(a,b,c,0)
#else
#define MatColoringRegisterDynamic(a,b,c,d) MatColoringRegister(a,b,c,d)
#endif
EXTERN int        MatColoringRegisterAll(char *);
extern PetscTruth MatColoringRegisterAllCalled;
EXTERN int        MatColoringRegisterDestroy(void);
EXTERN int MatColoringPatch(Mat,int,int *,ISColoring*);

/*
    Data structures used to compute Jacobian vector products 
  efficiently using finite differences.
*/
#define MAT_FDCOLORING_COOKIE PETSC_COOKIE + 23

typedef struct _p_MatFDColoring *MatFDColoring;

EXTERN int MatFDColoringCreate(Mat,ISColoring,MatFDColoring *);
EXTERN int MatFDColoringDestroy(MatFDColoring);
EXTERN int MatFDColoringView(MatFDColoring,Viewer);
EXTERN int MatFDColoringSetFunction(MatFDColoring,int (*)(void),void*);
EXTERN int MatFDColoringSetParameters(MatFDColoring,double,double);
EXTERN int MatFDColoringSetFrequency(MatFDColoring,int);
EXTERN int MatFDColoringGetFrequency(MatFDColoring,int*);
EXTERN int MatFDColoringSetFromOptions(MatFDColoring);
EXTERN int MatFDColoringPrintHelp(MatFDColoring);
EXTERN int MatFDColoringApply(Mat,MatFDColoring,Vec,MatStructure*,void *);
EXTERN int MatFDColoringApplyTS(Mat,MatFDColoring,double,Vec,MatStructure*,void *);

/* 
    These routines are for partitioning matrices: currently used only 
  for adjacency matrix, MatCreateMPIAdj().
*/
#define MATPARTITIONING_COOKIE PETSC_COOKIE + 25

typedef struct _p_MatPartitioning *MatPartitioning;
typedef char* MatPartitioningType;
#define MATPARTITIONING_CURRENT  "current"
#define MATPARTITIONING_PARMETIS "parmetis"

EXTERN int MatPartitioningCreate(MPI_Comm,MatPartitioning*);
EXTERN int MatPartitioningSetType(MatPartitioning,MatPartitioningType);
EXTERN int MatPartitioningSetAdjacency(MatPartitioning,Mat);
EXTERN int MatPartitioningSetVertexWeights(MatPartitioning,int*);
EXTERN int MatPartitioningApply(MatPartitioning,IS*);
EXTERN int MatPartitioningDestroy(MatPartitioning);

EXTERN int MatPartitioningRegister(char*,char*,char*,int(*)(MatPartitioning));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define MatPartitioningRegisterDynamic(a,b,c,d) MatPartitioningRegister(a,b,c,0)
#else
#define MatPartitioningRegisterDynamic(a,b,c,d) MatPartitioningRegister(a,b,c,d)
#endif

EXTERN int        MatPartitioningRegisterAll(char *);
extern PetscTruth MatPartitioningRegisterAllCalled;
EXTERN int        MatPartitioningRegisterDestroy(void);

EXTERN int MatPartitioningView(MatPartitioning,Viewer);
EXTERN int MatPartitioningSetFromOptions(MatPartitioning);
EXTERN int MatPartitioningPrintHelp(MatPartitioning);
EXTERN int MatPartitioningGetType(MatPartitioning,MatPartitioningType*);

EXTERN int MatPartitioningParmetisSetCoarseSequential(MatPartitioning);

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
               MATOP_RELAX=13,
               MATOP_TRANSPOSE=14,
               MATOP_GETINFO=15,
               MATOP_EQUAL=16,
               MATOP_GET_DIAGONAL=17,
               MATOP_DIAGONAL_SCALE=18,
               MATOP_NORM=19,
               MATOP_ASSEMBLY_BEGIN=20,
               MATOP_ASSEMBLY_END=21,
               MATOP_COMPRESS=22,
               MATOP_SET_OPTION=23,
               MATOP_ZERO_ENTRIES=24,
               MATOP_ZERO_ROWS=25,
               MATOP_LUFACTOR_SYMBOLIC=26,
               MATOP_LUFACTOR_NUMERIC=27,
               MATOP_CHOLESKY_FACTOR_SYMBOLIC=28,
               MATOP_CHOLESKY_FACTOR_NUMERIC=29,
               MATOP_GET_SIZE=30,
               MATOP_GET_LOCAL_SIZE=31,
               MATOP_GET_OWNERSHIP_RANGE=32,
               MATOP_ILUFACTOR_SYMBOLIC=33,
               MATOP_INCOMPLETECHOLESKYFACTOR_SYMBOLIC=34,
               MATOP_GET_ARRAY=35,
               MATOP_RESTORE_ARRAY=36,

               MATOP_CONVERT_SAME_TYPE=37,
               MATOP_FORWARD_SOLVE=38,
               MATOP_BACKWARD_SOLVE=39,
               MATOP_ILUFACTOR=40,
               MATOP_INCOMPLETECHOLESKYFACTOR=41,
               MATOP_AXPY=42,
               MATOP_GET_SUBMATRICES=43,
               MATOP_INCREASE_OVERLAP=44,
               MATOP_GET_VALUES=45,
               MATOP_COPY=46,
               MATOP_PRINT_HELP=47,
               MATOP_SCALE=48,
               MATOP_SHIFT=49,
               MATOP_DIAGONAL_SHIFT=50,
               MATOP_ILUDT_FACTOR=51,
               MATOP_GET_BLOCK_SIZE=52,
               MATOP_GET_ROW_IJ=53,
               MATOP_RESTORE_ROW_IJ=54,
               MATOP_GET_COLUMN_IJ=55,
               MATOP_RESTORE_COLUMN_IJ=56,
               MATOP_FDCOLORING_CREATE=57,
               MATOP_COLORING_PATCH=58,
               MATOP_SET_UNFACTORED=59,
               MATOP_PERMUTE=60,
               MATOP_SET_VALUES_BLOCKED=61,
               MATOP_DESTROY=250,
               MATOP_VIEW=251
             } MatOperation;
EXTERN int MatHasOperation(Mat,MatOperation,PetscTruth*);
EXTERN int MatShellSetOperation(Mat,MatOperation,void *);
EXTERN int MatShellGetOperation(Mat,MatOperation,void **);

/*
   Codes for matrices stored on disk. By default they are
 stored in a universal format. By changing the format with 
 ViewerSetFormat(viewer,VIEWER_FORMAT_BINARY_NATIVE); the matrices will
 be stored in a way natural for the matrix, for example dense matrices
 would be stored as dense. Matrices stored this way may only be
 read into matrices of the same time.
*/
#define MATRIX_BINARY_FORMAT_DENSE -1

/*
     New matrix classes not yet distributed 
*/
/*
    MatAIJIndices is a data structure for storing the nonzero location information
  for sparse matrices. Several matrices with identical nonzero structure can share
  the same MatAIJIndices.
*/ 
typedef struct _p_MatAIJIndices* MatAIJIndices;

EXTERN int MatCreateAIJIndices(int,int,int*,int*,PetscTruth,MatAIJIndices*);
EXTERN int MatCreateAIJIndicesEmpty(int,int,int*,PetscTruth,MatAIJIndices*);
EXTERN int MatAttachAIJIndices(MatAIJIndices,MatAIJIndices*);
EXTERN int MatDestroyAIJIndices(MatAIJIndices);
EXTERN int MatCopyAIJIndices(MatAIJIndices,MatAIJIndices*);
EXTERN int MatValidateAIJIndices(int,MatAIJIndices);
EXTERN int MatShiftAIJIndices(MatAIJIndices);
EXTERN int MatShrinkAIJIndices(MatAIJIndices);
EXTERN int MatTransposeAIJIndices(MatAIJIndices,MatAIJIndices*);

EXTERN int MatCreateSeqCSN(MPI_Comm,int,int,int*,int,Mat*);
EXTERN int MatCreateSeqCSN_Single(MPI_Comm,int,int,int*,int,Mat*);
EXTERN int MatCreateSeqCSNWithPrecision(MPI_Comm,int,int,int*,int,ScalarPrecision,Mat*);

EXTERN int MatCreateSeqCSNIndices(MPI_Comm,MatAIJIndices,int,Mat *);
EXTERN int MatCreateSeqCSNIndices_Single(MPI_Comm,MatAIJIndices,int,Mat *);
EXTERN int MatCreateSeqCSNIndicesWithPrecision(MPI_Comm,MatAIJIndices,int,ScalarPrecision,Mat *);

EXTERN int MatMPIBAIJSetHashTableFactor(Mat,double);
EXTERN int MatSeqAIJGetInodeSizes(Mat,int *,int *[],int *);

typedef char* MATType;
EXTERN int MATCreate(MPI_Comm,int,int,int,int,Mat*);
EXTERN int MatSetType(Mat,MATType);
EXTERN int MatRegisterAll(char*);
EXTERN int MatRegister(char*,char*,char*,int(*)(Mat));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define MatRegisterDynamic(a,b,c,d) MatRegister(a,b,c,0)
#else
#define MatRegisterDynamic(a,b,c,d) MatRegister(a,b,c,d)
#endif

EXTERN int MatCreateMAIJ(Mat,int,Mat*);

#define MATSEQMAIJ "seqaij"
#endif



