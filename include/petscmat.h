/* $Id: mat.h,v 1.113 1996/08/22 22:25:21 curfman Exp bsmith $ */
/*
     Include file for the matrix component of PETSc
*/
#ifndef __MAT_PACKAGE
#define __MAT_PACKAGE
#include "vec.h"

#define MAT_COOKIE         PETSC_COOKIE+5

typedef struct _Mat*           Mat;

typedef enum { MATSAME=-1, MATSEQDENSE, MATSEQAIJ, MATMPIAIJ, MATSHELL, 
               MATMPIROWBS, MATSEQBDIAG, MATMPIBDIAG,
               MATMPIDENSE, MATSEQBAIJ, MATMPIBAIJ} MatType;

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
typedef enum {MAT_FLUSH_ASSEMBLY=1,MAT_FINAL_ASSEMBLY=0} MatAssemblyType;
extern int MatAssemblyBegin(Mat,MatAssemblyType);
extern int MatAssemblyEnd(Mat,MatAssemblyType);

typedef enum {MAT_ROW_ORIENTED=1,MAT_COLUMN_ORIENTED=2,MAT_ROWS_SORTED=4,
              MAT_COLUMNS_SORTED=8,MAT_NO_NEW_NONZERO_LOCATIONS=16,
              MAT_YES_NEW_NONZERO_LOCATIONS=32,MAT_SYMMETRIC=64,
              MAT_STRUCTURALLY_SYMMETRIC,MAT_NO_NEW_DIAGONALS,
              MAT_YES_NEW_DIAGONALS,MAT_INODE_LIMIT_1,MAT_INODE_LIMIT_2,
              MAT_INODE_LIMIT_3,MAT_INODE_LIMIT_4,MAT_INODE_LIMIT_5} MatOption;
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
extern int MatCopy(Mat,Mat);
extern int MatView(Mat,Viewer);
extern int MatLoad(Viewer,MatType,Mat*);

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

/* Routines unique to particular data structures */
extern int MatBDiagGetData(Mat,int*,int*,int**,int**,Scalar***);

/* 
  These routines are not usually accessed directly, rather solving is 
  done through the SLES, KSP and PC interfaces.
*/

typedef enum {ORDER_NATURAL=0,ORDER_ND=1,ORDER_1WD=2,
              ORDER_RCM=3,ORDER_QMD=4,ORDER_ROWLENGTH=5,ORDER_FLOW,
              ORDER_APPLICATION_1,ORDER_APPLICATION_2} MatReordering;
extern int MatGetReordering(Mat,MatReordering,IS*,IS*);
extern int MatGetReorderingTypeFromOptions(char *,MatReordering*);
extern int MatReorderingRegister(MatReordering *,char*,int (*)(Mat,MatReordering,IS*,IS*));
extern int MatReorderingRegisterAll();
extern int MatReorderingRegisterDestroy();
extern int MatReorderingGetName(MatReordering,char **);

typedef enum {COLORING_NATURAL, COLORING_SL, COLORING_LD, COLORING_IF,
              COLORING_APPLICATION_1,COLORING_APPLICATION_2} MatColoring;
extern int MatGetColoring(Mat,MatColoring,int *,IS**);
extern int MatGetColoringTypeFromOptions(char *,MatColoring*);
extern int MatColoringRegister(MatColoring *,char*,int (*)(Mat,MatColoring,int*,IS**));
extern int MatColoringRegisterAll();
extern int MatColoringRegisterDestroy();

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

typedef enum {SOR_FORWARD_SWEEP=1,SOR_BACKWARD_SWEEP=2,SOR_SYMMETRIC_SWEEP=3,
              SOR_LOCAL_FORWARD_SWEEP=4,SOR_LOCAL_BACKWARD_SWEEP=8,
              SOR_LOCAL_SYMMETRIC_SWEEP=12,SOR_ZERO_INITIAL_GUESS=16,
              SOR_EISENSTAT=32,SOR_APPLY_UPPER=64,SOR_APPLY_LOWER=128
              } MatSORType;
extern int MatRelax(Mat,Vec,double,MatSORType,double,int,Vec);

/*
    If you add entries here you must also add them to FINCLUDE/mat.h
*/
typedef enum { MAT_SET_VALUES=0,
               MAT_GET_ROW=1,
               MAT_RESTORE_ROW=2,
               MAT_MULT=3,
               MAT_MULT_ADD=4,
               MAT_MULT_TRANS=5,
               MAT_MULT_TRANS_ADD=6,
               MAT_SOLVE=7,
               MAT_SOLVE_ADD=8,
               MAT_SOLVE_TRANS=9,
               MAT_SOLVE_TRANS_ADD=10,
               MAT_LUFACTOR=11,
               MAT_CHOLESKYFACTOR=12,
               MAT_RELAX=13,
               MAT_TRANSPOSE=14,
               MAT_GETINFO=15,
               MAT_EQUAL=16,
               MAT_GET_DIAGONAL=17, 
               MAT_DIAGONAL_SCALE=18,
               MAT_NORM=19,
               MAT_ASSEMBLY_BEGIN=20,
               MAT_ASSEMBLY_END=21,
               MAT_COMPRESS=22,
               MAT_SET_OPTION=23,
               MAT_ZERO_ENTRIES=24,
               MAT_ZERO_ROWS=25,
               MAT_LUFACTOR_SYMBOLIC=26,
               MAT_LUFACTOR_NUMERIC=27,
               MAT_CHOLESKY_FACTOR_SYMBOLIC=28,
               MAT_CHOLESKY_FACTOR_NUMERIC=29,
               MAT_GET_SIZE=30,
               MAT_GET_LOCAL_SIZE=31,
               MAT_GET_OWNERSHIP_RANGE=32,
               MAT_ILUFACTOR_SYMBOLIC=33,
               MAT_INCOMPLETECHOLESKYFACTOR_SYMBOLIC=34,
               MAT_GET_ARRAY=35,
               MAT_RESTORE_ARRAY=36,
               MAT_CONVERT=37,
               MAT_GET_SUBMATRIX=38,
               MAT_GET_SUBMATRIX_INPLACE=39,
               MAT_CONVERT_SAME_TYPE=40,
               MAT_FORWARD_SOLVE=41,
               MAT_BACKWARD_SOLVE=42,
               MAT_ILUFACTOR=43,
               MAT_INCOMPLETECHOLESKYFACTOR=44,
               MAT_AXPY=45,
               MAT_GET_SUBMATRICES=46,
               MAT_INCREASE_OVERLAP=47,
               MAT_GET_VALUES=48,
               MAT_COPY=49,
               MAT_PRINT_HELP=50,
               MAT_SCALE=51,
               MAT_SHIFT=52,
               MAT_DIAGONAL_SHIFT=53,
               MAT_ILUDT_FACTOR=54,
               MAT_GET_BLOCK_SIZE=55,
               MAT_DESTROY=250,
               MAT_VIEW=251
             } MatOperation;
extern int MatHasOperation(Mat,MatOperation,PetscTruth*);
extern int MatShellSetOperation(Mat,MatOperation,void *);

/*
   Codes for matrices stored on disk. By default they are
 stored in a universal format. By changing the format with 
 ViewerSetFormat(viewer,BINARY_FORMAT_NATIVE); the matrices will
 be stored in a way natural for the matrix, for example dense matrices
 would be stored as dense. Matrices stored this way may only be
 read into matrices of the same time.
*/
#define MATRIX_BINARY_FORMAT_DENSE -1

#endif



