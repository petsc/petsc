/* $Id: mat.h,v 1.186 2000/01/13 22:20:22 bsmith Exp bsmith $ */
/*
     Include file for the matrix component of PETSc
*/
#ifndef __MAT_H
#define __MAT_H
#include "vec.h"

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
               MATMPIBAIJ,  MATMPICSN,   MATSEQCSN,   MATMPICSR, MATSEQSBAIJ,
               MATLASTTYPE } MatType;

extern int MatCreate(MPI_Comm,int,int,int,int,Mat*);
extern int MatCreateSeqDense(MPI_Comm,int,int,Scalar*,Mat*);
extern int MatCreateMPIDense(MPI_Comm,int,int,int,int,Scalar*,Mat*); 
extern int MatCreateSeqAIJ(MPI_Comm,int,int,int,int*,Mat*);
extern int MatCreateMPIAIJ(MPI_Comm,int,int,int,int,int,int*,int,int*,Mat*); 
extern int MatCreateMPIRowbs(MPI_Comm,int,int,int,int*,void*,Mat*); 
extern int MatCreateSeqBDiag(MPI_Comm,int,int,int,int,int*,Scalar**,Mat*); 
extern int MatCreateMPIBDiag(MPI_Comm,int,int,int,int,int,int*,Scalar**,Mat*); 
extern int MatCreateSeqBAIJ(MPI_Comm,int,int,int,int,int*,Mat*); 
extern int MatCreateMPIBAIJ(MPI_Comm,int,int,int,int,int,int,int*,int,int*,Mat*);
extern int MatCreateMPICSR(MPI_Comm,int,int,int*,int*,int *,Mat*);

extern int MatDestroy(Mat);

extern int MatCreateShell(MPI_Comm,int,int,int,int,void *,Mat*);
extern int MatShellGetContext(Mat,void **);

extern int MatPrintHelp(Mat);
extern int MatGetMaps(Mat,Map*,Map*);

/* ------------------------------------------------------------*/
extern int MatSetValues(Mat,int,int*,int,int*,Scalar*,InsertMode);
extern int MatSetValuesBlocked(Mat,int,int*,int,int*,Scalar*,InsertMode);

typedef enum {MAT_FLUSH_ASSEMBLY=1,MAT_FINAL_ASSEMBLY=0} MatAssemblyType;
extern int MatAssemblyBegin(Mat,MatAssemblyType);
extern int MatAssemblyEnd(Mat,MatAssemblyType);
extern int MatAssembled(Mat,PetscTruth*);

#define MatSetValue(v,i,j,va,mode) \
{int _ierr,_row = i,_col = j; Scalar _va = va; \
  _ierr = MatSetValues(v,1,&_row,1,&_col,&_va,mode);CHKERRQ(_ierr); \
}
#define MatGetValue(v,i,j,va) \
{int _ierr,_row = i,_col = j; \
  _ierr = MatGetValues(v,1,&_row,1,&_col,&va);CHKERRQ(_ierr); \
}
/*
   Any additions/changes here MUST also be made in include/finclude/mat.h
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
extern int MatSetOption(Mat,MatOption);
extern int MatGetType(Mat,MatType*,char**);
extern int MatGetTypeFromOptions(MPI_Comm,char*,MatType*,PetscTruth*);

extern int MatGetValues(Mat,int,int*,int,int*,Scalar*);
extern int MatGetRow(Mat,int,int *,int **,Scalar**);
extern int MatRestoreRow(Mat,int,int *,int **,Scalar**);
extern int MatGetColumn(Mat,int,int *,int **,Scalar**);
extern int MatRestoreColumn(Mat,int,int *,int **,Scalar**);
extern int MatGetColumnVector(Mat,Vec,int);
extern int MatGetArray(Mat,Scalar **);
extern int MatRestoreArray(Mat,Scalar **);
extern int MatGetBlockSize(Mat,int *);

extern int MatMult(Mat,Vec,Vec);
extern int MatMultAdd(Mat,Vec,Vec,Vec);
extern int MatMultTranspose(Mat,Vec,Vec);
extern int MatMultTransposeAdd(Mat,Vec,Vec,Vec);

typedef enum {MAT_DO_NOT_COPY_VALUES,MAT_COPY_VALUES} MatDuplicateOption;

extern int MatConvert(Mat,MatType,Mat*);
extern int MatDuplicate(Mat,MatDuplicateOption,Mat*);
extern int MatConvertRegister(MatType,MatType,int (*)(Mat,MatType,Mat*));
extern int MatConvertRegisterAll(void);

typedef enum {SAME_NONZERO_PATTERN,DIFFERENT_NONZERO_PATTERN,SAME_PRECONDITIONER} MatStructure;

extern int MatCopy(Mat,Mat,MatStructure);
extern int MatView(Mat,Viewer);
extern int MatLoad(Viewer,MatType,Mat*);
extern int MatLoadRegister(MatType,int (*)(Viewer,MatType,Mat*));
extern int MatLoadRegisterAll(void);

extern int MatGetRowIJ(Mat,int,PetscTruth,int*,int **,int **,PetscTruth *);
extern int MatRestoreRowIJ(Mat,int,PetscTruth,int *,int **,int **,PetscTruth *);
extern int MatGetColumnIJ(Mat,int,PetscTruth,int*,int **,int **,PetscTruth *);
extern int MatRestoreColumnIJ(Mat,int,PetscTruth,int *,int **,int **,PetscTruth *);

/* 
   Context of matrix information, used with MatGetInfo()
   Note: If any entries are added to this context, be sure
         to adjust MAT_INFO_SIZE in finclude/mat.h
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
extern int MatGetInfo(Mat,MatInfoType,MatInfo*);
extern int MatValid(Mat,PetscTruth*);
extern int MatGetDiagonal(Mat,Vec);
extern int MatTranspose(Mat,Mat*);
extern int MatPermute(Mat,IS,IS,Mat *);
extern int MatDiagonalScale(Mat,Vec,Vec);
extern int MatDiagonalShift(Mat,Vec);
extern int MatEqual(Mat,Mat,PetscTruth*);

extern int MatNorm(Mat,NormType,double *);
extern int MatZeroEntries(Mat);
extern int MatZeroRows(Mat,IS,Scalar*);
extern int MatZeroColumns(Mat,IS,Scalar*);

extern int MatUseScaledForm(Mat,PetscTruth);
extern int MatScaleSystem(Mat,Vec,Vec);
extern int MatUnScaleSystem(Mat,Vec,Vec);

extern int MatGetSize(Mat,int*,int*);
extern int MatGetLocalSize(Mat,int*,int*);
extern int MatGetOwnershipRange(Mat,int*,int*);

typedef enum {MAT_INITIAL_MATRIX,MAT_REUSE_MATRIX} MatReuse;
extern int MatGetSubMatrices(Mat,int,IS *,IS *,MatReuse,Mat **);
extern int MatDestroyMatrices(int,Mat **);
extern int MatGetSubMatrix(Mat,IS,IS,int,MatReuse,Mat *);

extern int MatIncreaseOverlap(Mat,int,IS *,int);

extern int MatAXPY(Scalar *,Mat,Mat);
extern int MatAYPX(Scalar *,Mat,Mat);
extern int MatCompress(Mat);

extern int MatScale(Scalar *,Mat);
extern int MatShift(Scalar *,Mat);

extern int MatSetLocalToGlobalMapping(Mat,ISLocalToGlobalMapping);
extern int MatSetLocalToGlobalMappingBlock(Mat,ISLocalToGlobalMapping);
extern int MatZeroRowsLocal(Mat,IS,Scalar*);
extern int MatSetValuesLocal(Mat,int,int*,int,int*,Scalar*,InsertMode);
extern int MatSetValuesBlockedLocal(Mat,int,int*,int,int*,Scalar*,InsertMode);

extern int MatSetStashInitialSize(Mat,int,int);

extern int MatInterpolateAdd(Mat,Vec,Vec,Vec);
extern int MatInterpolate(Mat,Vec,Vec);
extern int MatRestrict(Mat,Vec,Vec);

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
extern int MatBDiagGetData(Mat,int*,int*,int**,int**,Scalar***);
extern int MatSeqAIJSetColumnIndices(Mat,int *);
extern int MatSeqBAIJSetColumnIndices(Mat,int *);
extern int MatCreateSeqAIJWithArrays(MPI_Comm,int,int,int*,int*,Scalar *,Mat*);

extern int MatStoreValues(Mat);
extern int MatRetrieveValues(Mat);

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

extern int MatGetOrdering(Mat,MatOrderingType,IS*,IS*);
extern int MatOrderingRegister(char*,char*,char*,int(*)(Mat,MatOrderingType,IS*,IS*));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define MatOrderingRegisterDynamic(a,b,c,d) MatOrderingRegister(a,b,c,0)
#else
#define MatOrderingRegisterDynamic(a,b,c,d) MatOrderingRegister(a,b,c,d)
#endif
extern int        MatOrderingRegisterDestroy(void);
extern int        MatOrderingRegisterAll(char*);
extern PetscTruth MatOrderingRegisterAllCalled;

extern int MatReorderForNonzeroDiagonal(Mat,double,IS,IS);

extern int MatCholeskyFactor(Mat,IS,double);
extern int MatCholeskyFactorSymbolic(Mat,IS,double,Mat*);
extern int MatCholeskyFactorNumeric(Mat,Mat*);

/* 
   Context of matrix information, used with MatILUFactor() and MatILUFactorSymbolic()
   Note: If any entries are added to this context, be sure
         to adjust MAT_ILUINFO_SIZE in finclude/mat.h

   Note: The integer values below are passed in double to allow easy use from 
         Fortran
 */
typedef struct {
  double     levels;  /* ILU(levels) */ 
  double     fill;    /* expected fill; nonzeros in factored matrix/nonzeros in original matrix*/
  double     diagonal_fill;  /* force diagonal to fill in if initially not filled */

  double     dt;             /* drop tolerance */
  double     dtcol;          /* tolerance for pivoting */
  double     dtcount;        /* maximum nonzeros to be allowed per row */
} MatILUInfo;

extern int MatLUFactor(Mat,IS,IS,double);
extern int MatILUFactor(Mat,IS,IS,MatILUInfo*);
extern int MatLUFactorSymbolic(Mat,IS,IS,double,Mat*);
extern int MatILUFactorSymbolic(Mat,IS,IS,MatILUInfo*,Mat*);
extern int MatIncompleteCholeskyFactorSymbolic(Mat,IS,double,int,Mat*);
extern int MatLUFactorNumeric(Mat,Mat*);
extern int MatILUDTFactor(Mat,MatILUInfo*,IS,IS,Mat *);

extern int MatSolve(Mat,Vec,Vec);
extern int MatForwardSolve(Mat,Vec,Vec);
extern int MatBackwardSolve(Mat,Vec,Vec);
extern int MatSolveAdd(Mat,Vec,Vec,Vec);
extern int MatSolveTranspose(Mat,Vec,Vec);
extern int MatSolveTransposeAdd(Mat,Vec,Vec,Vec);

extern int MatSetUnfactored(Mat);

/*  MatSORType may be bitwise ORd together, so do not change the numbers */

typedef enum {SOR_FORWARD_SWEEP=1,SOR_BACKWARD_SWEEP=2,SOR_SYMMETRIC_SWEEP=3,
              SOR_LOCAL_FORWARD_SWEEP=4,SOR_LOCAL_BACKWARD_SWEEP=8,
              SOR_LOCAL_SYMMETRIC_SWEEP=12,SOR_ZERO_INITIAL_GUESS=16,
              SOR_EISENSTAT=32,SOR_APPLY_UPPER=64,SOR_APPLY_LOWER=128} MatSORType;
extern int MatRelax(Mat,Vec,double,MatSORType,double,int,Vec);

/* 
    These routines are for efficiently computing Jacobians via finite differences.
*/

typedef char* MatColoringType;
#define MATCOLORING_NATURAL "natural"
#define MATCOLORING_SL      "sl"
#define MATCOLORING_LF      "lf"
#define MATCOLORING_ID      "id"

extern int MatGetColoring(Mat,MatColoringType,ISColoring*);
extern int MatColoringRegister(char*,char*,char*,int(*)(Mat,MatColoringType,ISColoring *));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define MatColoringRegisterDynamic(a,b,c,d) MatColoringRegister(a,b,c,0)
#else
#define MatColoringRegisterDynamic(a,b,c,d) MatColoringRegister(a,b,c,d)
#endif
extern int        MatColoringRegisterAll(char *);
extern PetscTruth MatColoringRegisterAllCalled;
extern int        MatColoringRegisterDestroy(void);
extern int MatColoringPatch(Mat,int,int *,ISColoring*);

/*
    Data structures used to compute Jacobian vector products 
  efficiently using finite differences.
*/
#define MAT_FDCOLORING_COOKIE PETSC_COOKIE + 23

typedef struct _p_MatFDColoring *MatFDColoring;

extern int MatFDColoringCreate(Mat,ISColoring,MatFDColoring *);
extern int MatFDColoringDestroy(MatFDColoring);
extern int MatFDColoringView(MatFDColoring,Viewer);
extern int MatFDColoringSetFunction(MatFDColoring,int (*)(void),void*);
extern int MatFDColoringSetParameters(MatFDColoring,double,double);
extern int MatFDColoringSetFrequency(MatFDColoring,int);
extern int MatFDColoringGetFrequency(MatFDColoring,int*);
extern int MatFDColoringSetFromOptions(MatFDColoring);
extern int MatFDColoringPrintHelp(MatFDColoring);
extern int MatFDColoringApply(Mat,MatFDColoring,Vec,MatStructure*,void *);
extern int MatFDColoringApplyTS(Mat,MatFDColoring,double,Vec,MatStructure*,void *);

/* 
    These routines are for partitioning matrices: currently used only 
  for adjacency matrix, MatCreateMPICSR().
*/
#define MATPARTITIONING_COOKIE PETSC_COOKIE + 25

typedef struct _p_MatPartitioning *MatPartitioning;
typedef char* MatPartitioningType;
#define MATPARTITIONING_CURRENT  "current"
#define MATPARTITIONING_PARMETIS "parmetis"

extern int MatPartitioningCreate(MPI_Comm,MatPartitioning*);
extern int MatPartitioningSetType(MatPartitioning,MatPartitioningType);
extern int MatPartitioningSetAdjacency(MatPartitioning,Mat);
extern int MatPartitioningSetVertexWeights(MatPartitioning,int*);
extern int MatPartitioningApply(MatPartitioning,IS*);
extern int MatPartitioningDestroy(MatPartitioning);

extern int MatPartitioningRegister(char*,char*,char*,int(*)(MatPartitioning));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define MatPartitioningRegisterDynamic(a,b,c,d) MatPartitioningRegister(a,b,c,0)
#else
#define MatPartitioningRegisterDynamic(a,b,c,d) MatPartitioningRegister(a,b,c,d)
#endif

extern int        MatPartitioningRegisterAll(char *);
extern PetscTruth MatPartitioningRegisterAllCalled;
extern int        MatPartitioningRegisterDestroy(void);

extern int MatPartitioningView(MatPartitioning,Viewer);
extern int MatPartitioningSetFromOptions(MatPartitioning);
extern int MatPartitioningPrintHelp(MatPartitioning);
extern int MatPartitioningGetType(MatPartitioning,MatPartitioningType*);

extern int MatPartitioningParmetisSetCoarseSequential(MatPartitioning);

/*
    If you add entries here you must also add them to finclude/mat.h
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
extern int MatHasOperation(Mat,MatOperation,PetscTruth*);
extern int MatShellSetOperation(Mat,MatOperation,void *);
extern int MatShellGetOperation(Mat,MatOperation,void **);

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

extern int MatCreateAIJIndices(int,int,int*,int*,PetscTruth,MatAIJIndices*);
extern int MatCreateAIJIndicesEmpty(int,int,int*,PetscTruth,MatAIJIndices*);
extern int MatAttachAIJIndices(MatAIJIndices,MatAIJIndices*);
extern int MatDestroyAIJIndices(MatAIJIndices);
extern int MatCopyAIJIndices(MatAIJIndices,MatAIJIndices*);
extern int MatValidateAIJIndices(int,MatAIJIndices);
extern int MatShiftAIJIndices(MatAIJIndices);
extern int MatShrinkAIJIndices(MatAIJIndices);
extern int MatTransposeAIJIndices(MatAIJIndices,MatAIJIndices*);

extern int MatCreateSeqCSN(MPI_Comm,int,int,int*,int,Mat*);
extern int MatCreateSeqCSN_Single(MPI_Comm,int,int,int*,int,Mat*);
extern int MatCreateSeqCSNWithPrecision(MPI_Comm,int,int,int*,int,ScalarPrecision,Mat*);

extern int MatCreateSeqCSNIndices(MPI_Comm,MatAIJIndices,int,Mat *);
extern int MatCreateSeqCSNIndices_Single(MPI_Comm,MatAIJIndices,int,Mat *);
extern int MatCreateSeqCSNIndicesWithPrecision(MPI_Comm,MatAIJIndices,int,ScalarPrecision,Mat *);

extern int MatMPIBAIJSetHashTableFactor(Mat,double);
extern int MatSeqAIJGetInodeSizes(Mat,int *,int *[],int *);


#endif



