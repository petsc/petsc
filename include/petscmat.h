/* $Id: mat.h,v 1.127 1997/03/01 16:00:14 bsmith Exp bsmith $ */
/*
     Include file for the matrix component of PETSc
*/
#ifndef __MAT_PACKAGE
#define __MAT_PACKAGE
#include "vec.h"

#define MAT_COOKIE         PETSC_COOKIE+5

typedef struct _Mat*           Mat;

#define MAX_MATRIX_TYPES 12
/*
   The default matrix data storage formats and routines to create them.
*/
typedef enum { MATSAME=-1,  MATSEQDENSE, MATSEQAIJ,   MATMPIAIJ,   MATSHELL, 
               MATMPIROWBS, MATSEQBDIAG, MATMPIBDIAG, MATMPIDENSE, MATSEQBAIJ,
               MATMPIBAIJ,  MATMPICSN,   MATSEQCSN} MatType;

extern int MatCreate(MPI_Comm,int,int,Mat*);
extern int MatCreateSeqDense(MPI_Comm,int,int,Scalar*,Mat*);
extern int MatCreateMPIDense(MPI_Comm,int,int,int,int,Scalar*,Mat*); 
extern int MatCreateSeqAIJ(MPI_Comm,int,int,int,int*,Mat*);
extern int MatCreateMPIAIJ(MPI_Comm,int,int,int,int,int,int*,int,int*,Mat*); 
extern int MatCreateMPIRowbs(MPI_Comm,int,int,int,int*,void*,Mat*); 
extern int MatCreateSeqBDiag(MPI_Comm,int,int,int,int,int*,Scalar**,Mat*); 
extern int MatCreateMPIBDiag(MPI_Comm,int,int,int,int,int,int*,Scalar**,Mat*); 
extern int MatCreateSeqBAIJ(MPI_Comm,int,int,int,int,int*,Mat*); 
extern int MatCreateMPIBAIJ(MPI_Comm,int,int,int,int,int,int,int*,int,int*,Mat*);

extern int MatDestroy(Mat);

extern int MatCreateShell(MPI_Comm,int,int,int,int,void *,Mat*);
extern int MatShellGetContext(Mat,void **);

extern int MatPrintHelp(Mat);

/* ------------------------------------------------------------*/
extern int MatSetValues(Mat,int,int*,int,int*,Scalar*,InsertMode);
extern int MatSetValuesBlocked(Mat,int,int*,int,int*,Scalar*,InsertMode);

typedef enum {MAT_FLUSH_ASSEMBLY=1,MAT_FINAL_ASSEMBLY=0} MatAssemblyType;
extern int MatAssemblyBegin(Mat,MatAssemblyType);
extern int MatAssemblyEnd(Mat,MatAssemblyType);
#define MatSetValue(v,i,j,va,mode) \
{int _ierr,_row = i,_col = j; Scalar _va = va; \
  _ierr = MatSetValues(v,1,&_row,1,&_col,&_va,mode);CHKERRQ(_ierr); \
}

typedef enum {MAT_ROW_ORIENTED=1,MAT_COLUMN_ORIENTED=2,MAT_ROWS_SORTED=4,
              MAT_COLUMNS_SORTED=8,MAT_NO_NEW_NONZERO_LOCATIONS=16,
              MAT_YES_NEW_NONZERO_LOCATIONS=32,MAT_SYMMETRIC=64,
              MAT_STRUCTURALLY_SYMMETRIC,MAT_NO_NEW_DIAGONALS,
              MAT_YES_NEW_DIAGONALS,MAT_INODE_LIMIT_1,MAT_INODE_LIMIT_2,
              MAT_INODE_LIMIT_3,MAT_INODE_LIMIT_4,MAT_INODE_LIMIT_5,
              MAT_IGNORE_OFF_PROCESSOR_ENTRIES,MAT_ROWS_UNSORTED,
              MAT_COLUMNS_UNSORTED,MAT_NEW_NONZERO_LOCATION_ERROR} MatOption;
extern int MatSetOption(Mat,MatOption);
extern int MatGetType(Mat,MatType*,char**);
extern int MatGetTypeFromOptions(MPI_Comm,char*,MatType*,int*);

extern int MatGetValues(Mat,int,int*,int,int*,Scalar*);
extern int MatGetRow(Mat,int,int *,int **,Scalar**);
extern int MatRestoreRow(Mat,int,int *,int **,Scalar**);
extern int MatGetColumn(Mat,int,int *,int **,Scalar**);
extern int MatRestoreColumn(Mat,int,int *,int **,Scalar**);
extern int MatGetArray(Mat,Scalar **);
extern int MatRestoreArray(Mat,Scalar **);
extern int MatGetBlockSize(Mat,int *);

extern int MatMult(Mat,Vec,Vec);
extern int MatMultAdd(Mat,Vec,Vec,Vec);
extern int MatMultTrans(Mat,Vec,Vec);
extern int MatMultTransAdd(Mat,Vec,Vec,Vec);

extern int MatConvert(Mat,MatType,Mat*);
extern int MatConvertRegister(MatType,MatType,int (*)(Mat,MatType,Mat*));
extern int MatConvertRegisterAll();

extern int MatCopy(Mat,Mat);
extern int MatView(Mat,Viewer);
extern int MatLoad(Viewer,MatType,Mat*);
extern int MatLoadRegister(MatType,int (*)(Viewer,MatType,Mat*));
extern int MatLoadRegisterAll();

extern int MatGetRowIJ(Mat,int,PetscTruth,int*,int **,int **,PetscTruth *);
extern int MatRestoreRowIJ(Mat,int,PetscTruth,int *,int **,int **,PetscTruth *);
extern int MatGetColumnIJ(Mat,int,PetscTruth,int*,int **,int **,PetscTruth *);
extern int MatRestoreColumnIJ(Mat,int,PetscTruth,int *,int **,int **,PetscTruth *);

/* 
   Context of matrix information, used with MatGetInfo()
   Note: If any entries are added to this context, be sure
         to adjust MAT_INFO_SIZE in FINCLUDE/mat.h
 */
typedef struct {
  double rows_global, columns_global;         /* number of global rows and columns */
  double rows_local, columns_local;           /* number of local rows and columns */
  double block_size;                          /* block size */
  double nz_allocated, nz_used, nz_unneeded;  /* number of nonzeros */
  double memory;                              /* memory allocated */
  double assemblies;                          /* number of matrix assemblies */
  double mallocs;                             /* number of mallocs during MatSetValues() */
  double fill_ratio_given, fill_ratio_needed; /* fill ration for LU/ILU */
  double factor_mallocs;                      /* number of mallocs during factorization */
} MatInfo;

typedef enum {MAT_LOCAL=1,MAT_GLOBAL_MAX=2,MAT_GLOBAL_SUM=3} MatInfoType;
extern int MatGetInfo(Mat,MatInfoType,MatInfo*);
extern int MatValid(Mat,PetscTruth*);
extern int MatGetDiagonal(Mat,Vec);
extern int MatTranspose(Mat,Mat*);
extern int MatPermute(Mat,IS,IS,Mat *);
extern int MatDiagonalScale(Mat,Vec,Vec);
extern int MatDiagonalShift(Mat,Vec);
extern int MatEqual(Mat,Mat, PetscTruth*);

extern int MatNorm(Mat,NormType,double *);
extern int MatZeroEntries(Mat);
extern int MatZeroRows(Mat,IS,Scalar*);
extern int MatZeroColumns(Mat,IS,Scalar*);

extern int MatGetSize(Mat,int*,int*);
extern int MatGetLocalSize(Mat,int*,int*);
extern int MatGetOwnershipRange(Mat,int*,int*);

typedef enum {MAT_INITIAL_MATRIX, MAT_REUSE_MATRIX} MatGetSubMatrixCall;
extern int MatGetSubMatrices(Mat,int,IS *,IS *,MatGetSubMatrixCall,Mat **);
extern int MatDestroyMatrices(int, Mat **);
extern int MatIncreaseOverlap(Mat,int,IS *,int);

extern int MatAXPY(Scalar *,Mat,Mat);
extern int MatCompress(Mat);

extern int MatScale(Scalar *,Mat);
extern int MatShift(Scalar *,Mat);

extern int MatSetLocalToGlobalMapping(Mat, int,int *);
extern int MatZeroRowsLocal(Mat,IS,Scalar*);
extern int MatSetValuesLocal(Mat,int,int*,int,int*,Scalar*,InsertMode);

/* Routines unique to particular data structures */
extern int MatBDiagGetData(Mat,int*,int*,int**,int**,Scalar***);

/* 
  These routines are not usually accessed directly, rather solving is 
  done through the SLES, KSP and PC interfaces.
*/

typedef enum {ORDER_NATURAL=0,ORDER_ND=1,ORDER_1WD=2,ORDER_RCM=3,
              ORDER_QMD=4,ORDER_ROWLENGTH=5,ORDER_FLOW,ORDER_NEW} MatReordering;
extern int MatGetReordering(Mat,MatReordering,IS*,IS*);
extern int MatGetReorderingTypeFromOptions(char *,MatReordering*);
extern int MatReorderingRegister(MatReordering,MatReordering*,char*,int(*)(Mat,MatReordering,IS*,IS*));
extern int MatReorderingGetName(MatReordering,char **);
extern int MatReorderingRegisterDestroy();
extern int MatReorderingRegisterAll();
extern int MatReorderingRegisterAllCalled;

extern int MatReorderForNonzeroDiagonal(Mat,double,IS,IS);

extern int MatCholeskyFactor(Mat,IS,double);
extern int MatCholeskyFactorSymbolic(Mat,IS,double,Mat*);
extern int MatCholeskyFactorNumeric(Mat,Mat*);

extern int MatLUFactor(Mat,IS,IS,double);
extern int MatILUFactor(Mat,IS,IS,double,int);
extern int MatLUFactorSymbolic(Mat,IS,IS,double,Mat*);
extern int MatILUFactorSymbolic(Mat,IS,IS,double,int,Mat*);
extern int MatIncompleteCholeskyFactorSymbolic(Mat,IS,double,int,Mat*);
extern int MatLUFactorNumeric(Mat,Mat*);
extern int MatILUDTFactor(Mat,double,int,IS,IS,Mat *);

extern int MatSolve(Mat,Vec,Vec);
extern int MatForwardSolve(Mat,Vec,Vec);
extern int MatBackwardSolve(Mat,Vec,Vec);
extern int MatSolveAdd(Mat,Vec,Vec,Vec);
extern int MatSolveTrans(Mat,Vec,Vec);
extern int MatSolveTransAdd(Mat,Vec,Vec,Vec);

extern int MatSetUnfactored(Mat);

typedef enum {SOR_FORWARD_SWEEP=1,SOR_BACKWARD_SWEEP=2,SOR_SYMMETRIC_SWEEP=3,
              SOR_LOCAL_FORWARD_SWEEP=4,SOR_LOCAL_BACKWARD_SWEEP=8,
              SOR_LOCAL_SYMMETRIC_SWEEP=12,SOR_ZERO_INITIAL_GUESS=16,
              SOR_EISENSTAT=32,SOR_APPLY_UPPER=64,SOR_APPLY_LOWER=128} MatSORType;
extern int MatRelax(Mat,Vec,double,MatSORType,double,int,Vec);

/* 
    These routines are for efficiently computing Jacobians via finite differences.
*/
typedef enum {COLORING_NATURAL, COLORING_SL, COLORING_LF, COLORING_ID,
              COLORING_NEW} MatColoring;
extern int MatGetColoring(Mat,MatColoring,ISColoring*);
extern int MatGetColoringTypeFromOptions(char *,MatColoring*);
extern int MatColoringRegister(MatColoring,MatColoring*,char*,int(*)(Mat,MatColoring,ISColoring *));
extern int MatColoringRegisterAll();
extern int MatColoringRegisterAllCalled;
extern int MatColoringRegisterDestroy();
extern int MatColoringPatch(Mat,int,int *,ISColoring*);

/*
    Data structures used to compute Jacobian vector products 
  efficiently using finite differences.
*/
#define MAT_FDCOLORING_COOKIE PETSC_COOKIE + 22

typedef struct _MatFDColoring *MatFDColoring;

extern int MatFDColoringCreate(Mat,ISColoring,MatFDColoring *);
extern int MatFDColoringDestroy(MatFDColoring);
extern int MatFDColoringView(MatFDColoring,Viewer);
extern int MatFDColoringSetParameters(MatFDColoring,double,double);
extern int MatFDColoringSetFromOptions(MatFDColoring);
extern int MatFDColoringPrintHelp(MatFDColoring);
extern int MatFDColoringApply(Mat,MatFDColoring,Vec,Vec,Vec,Vec,int (*)(void *,Vec,Vec,void*),
                              void *,void *);

/*
    If you add entries here you must also add them to FINCLUDE/mat.h
*/
typedef enum { MATOP_SET_VALUES=0,
               MATOP_GET_ROW=1,
               MATOP_RESTORE_ROW=2,
               MATOP_MULT=3,
               MATOP_MULT_ADD=4,
               MATOP_MULT_TRANS=5,
               MATOP_MULT_TRANS_ADD=6,
               MATOP_SOLVE=7,
               MATOP_SOLVE_ADD=8,
               MATOP_SOLVE_TRANS=9,
               MATOP_SOLVE_TRANS_ADD=10,
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

               MATOP_CONVERT_SAME_TYPE=39,
               MATOP_FORWARD_SOLVE=40,
               MATOP_BACKWARD_SOLVE=41,
               MATOP_ILUFACTOR=42,
               MATOP_INCOMPLETECHOLESKYFACTOR=43,
               MATOP_AXPY=44,
               MATOP_GET_SUBMATRICES=45,
               MATOP_INCREASE_OVERLAP=46,
               MATOP_GET_VALUES=47,
               MATOP_COPY=48,
               MATOP_PRINT_HELP=49,
               MATOP_SCALE=50,
               MATOP_SHIFT=51,
               MATOP_DIAGONAL_SHIFT=52,
               MATOP_ILUDT_FACTOR=53,
               MATOP_GET_BLOCK_SIZE=54,
               MATOP_GET_ROW_IJ=55,
               MATOP_RESTORE_ROW_IJ=56,
               MATOP_GET_COLUMN_IJ=57,
               MATOP_RESTORE_COLUMN_IJ=58,
               MATOP_FDCOLORING_CREATE=59,
               MATOP_DESTROY=250,
               MATOP_VIEW=251
             } MatOperation;
extern int MatHasOperation(Mat,MatOperation,PetscTruth*);
extern int MatShellSetOperation(Mat,MatOperation,void *);

/*
   Codes for matrices stored on disk. By default they are
 stored in a universal format. By changing the format with 
 ViewerSetFormat(viewer,VIEWER_FORMAT_BINARY_NATIVE); the matrices will
 be stored in a way natural for the matrix, for example dense matrices
 would be stored as dense. Matrices stored this way may only be
 read into matrices of the same time.
*/
#define MATRIX_BINARY_FORMAT_DENSE -1

#endif



