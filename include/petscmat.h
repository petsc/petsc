/* $Id: mat.h,v 1.89 1996/02/08 18:28:49 bsmith Exp bsmith $ */
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
               MATMPIDENSE, MATSETBAIJ } MatType;

extern int MatCreate(MPI_Comm,int,int,Mat*);
extern int MatCreateSeqDense(MPI_Comm,int,int,Scalar*,Mat*);
extern int MatCreateMPIDense(MPI_Comm,int,int,int,int,Scalar*,Mat*); 
extern int MatCreateSeqAIJ(MPI_Comm,int,int,int,int*,Mat*);
extern int MatCreateMPIAIJ(MPI_Comm,int,int,int,int,int,int*,int,int*,Mat*); 
extern int MatCreateMPIRowbs(MPI_Comm,int,int,int,int*,void*,Mat*); 
extern int MatCreateSeqBDiag(MPI_Comm,int,int,int,int,int*,Scalar**,Mat*); 
extern int MatCreateMPIBDiag(MPI_Comm,int,int,int,int,int,int*,Scalar**,Mat*); 
extern int MatCreateSeqBAIJ(MPI_Comm,int,int,int,int,int*,Mat*); 

extern int MatDestroy(Mat);

extern int MatCreateShell(MPI_Comm,int,int,void *,Mat*);
extern int MatShellGetContext(Mat,void **);
extern int MatShellSetMult(Mat,int (*)(void*,Vec,Vec));
extern int MatShellSetDestroy(Mat,int (*)(void*));
extern int MatShellSetMultTrans(Mat,int (*)(void*,Vec,Vec));
extern int MatShellSetMultTransAdd(Mat,int (*)(void*,Vec,Vec,Vec));
  
extern int MatPrintHelp(Mat);

/* ------------------------------------------------------------*/
extern int MatSetValues(Mat,int,int*,int,int*,Scalar*,InsertMode);
typedef enum {FLUSH_ASSEMBLY=1,FINAL_ASSEMBLY=0} MatAssemblyType;
extern int MatAssemblyBegin(Mat,MatAssemblyType);
extern int MatAssemblyEnd(Mat,MatAssemblyType);

typedef enum {ROW_ORIENTED=1,COLUMN_ORIENTED=2,ROWS_SORTED=4,
              COLUMNS_SORTED=8,NO_NEW_NONZERO_LOCATIONS=16,
              YES_NEW_NONZERO_LOCATIONS=32,SYMMETRIC_MATRIX=64,
              STRUCTURALLY_SYMMETRIC_MATRIX,NO_NEW_DIAGONALS,
              YES_NEW_DIAGONALS,INODE_LIMIT_1,INODE_LIMIT_2,
              INODE_LIMIT_3,INODE_LIMIT_4,INODE_LIMIT_5} MatOption;
extern int MatSetOption(Mat,MatOption);
extern int MatGetType(Mat,MatType*,char**);
extern int MatGetFormatFromOptions(MPI_Comm,char*,MatType*,int*);
extern int MatGetValues(Mat,int,int*,int,int*,Scalar*);
extern int MatGetRow(Mat,int,int *,int **,Scalar**);
extern int MatRestoreRow(Mat,int,int *,int **,Scalar**);
extern int MatGetCol(Mat,int,int *,int **,Scalar**);
extern int MatRestoreCol(Mat,int,int *,int **,Scalar**);
extern int MatGetArray(Mat,Scalar **);

extern int MatMult(Mat,Vec,Vec);
extern int MatMultAdd(Mat,Vec,Vec,Vec);
extern int MatMultTrans(Mat,Vec,Vec);
extern int MatMultTransAdd(Mat,Vec,Vec,Vec);

extern int MatConvert(Mat,MatType,Mat*);
extern int MatCopy(Mat,Mat);
extern int MatView(Mat,Viewer);
extern int MatLoad(Viewer,MatType,Mat*);

typedef enum {MAT_LOCAL=1,MAT_GLOBAL_MAX=2,MAT_GLOBAL_SUM=3} MatInfoType;
extern int MatGetInfo(Mat,MatInfoType,int*,int*,int*);
extern int MatValidMatrix(Mat);
extern int MatGetDiagonal(Mat,Vec);
extern int MatTranspose(Mat,Mat*);
extern int MatDiagonalScale(Mat,Vec,Vec);
extern int MatEqual(Mat,Mat);

extern int MatNorm(Mat,NormType,double *);
extern int MatZeroEntries(Mat);
extern int MatZeroRows(Mat,IS,Scalar*);
extern int MatZeroColumns(Mat,IS,Scalar*);

extern int MatGetSize(Mat,int*,int*);
extern int MatGetLocalSize(Mat,int*,int*);
extern int MatGetOwnershipRange(Mat,int*,int*);

typedef enum {MAT_INITIAL_MATRIX, MAT_REUSE_MATRIX} MatGetSubMatrixCall;
extern int MatGetSubMatrix(Mat,IS,IS,MatGetSubMatrixCall,Mat*);
extern int MatGetSubMatrixInPlace(Mat,IS,IS);
extern int MatGetSubMatrices(Mat,int,IS *,IS *,MatGetSubMatrixCall,Mat **);
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
              ORDER_RCM=3,ORDER_QMD=4,ORDER_APPLICATION_1,
              ORDER_APPLICATION_2} MatOrdering;
extern int MatGetReordering(Mat,MatOrdering,IS*,IS*);
extern int MatGetReorderingTypeFromOptions(char *,MatOrdering*);
extern int MatReorderForNonzeroDiagonal(Mat,double,IS,IS);
extern int MatReorderingRegister(MatOrdering,char*,int (*)(int*,int*,int*,int*,int*));
extern int MatReorderingRegisterAll();
extern int MatReorderingRegisterDestroy();

extern int MatLUFactor(Mat,IS,IS,double);
extern int MatILUFactor(Mat,IS,IS,double,int);
extern int MatCholeskyFactor(Mat,IS,double);
extern int MatLUFactorSymbolic(Mat,IS,IS,double,Mat*);
extern int MatILUFactorSymbolic(Mat,IS,IS,double,int,Mat*);
extern int MatCholeskyFactorSymbolic(Mat,IS,double,Mat*);
extern int MatIncompleteCholeskyFactorSymbolic(Mat,IS,double,int,Mat*);
extern int MatLUFactorNumeric(Mat,Mat*);
extern int MatCholeskyFactorNumeric(Mat,Mat*);

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

typedef enum { MAT_GET_DIAGONAL=17 } MatOperation;
extern int MatHasOperation(Mat,MatOperation,PetscTruth*);

/*  Not currently supported! 
#define MAT_SCATTER_COOKIE PETSC_COOKIE+15
typedef struct _MatScatter* MatScatter;

extern int MatScatterBegin(Mat,Mat,InsertMode,MatScatter);
extern int MatScatterEnd(Mat,Mat,InsertMode,MatScatter);
extern int MatScatterCreate(Mat,IS,IS,Mat,IS,IS,MatScatter*);
extern int MatScatterDestroy(MatScatter);
*/

#endif



