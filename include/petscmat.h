/*
     Include file for the matrix component of PETSc
*/
#ifndef __MAT_PACKAGE
#define __MAT_PACKAGE
#include "vec.h"

#define MAT_COOKIE PETSC_COOKIE+5

typedef struct _Mat*           Mat;
typedef struct _MatScatterCtx* MatScatterCtx;

typedef enum { MATDENSE, MATAIJ, MATMPIAIJ, MATSHELL, MATROW, 
               MATMPIROW } MATTYPE;

extern int MatCreateSequentialDense(int,int,Mat*);
extern int MatCreateSequentialAIJ(int,int,int,int *,Mat*);
extern int MatCreateMPIAIJ(MPI_Comm,int,int,int,int,int,int*,int,int*,Mat*); 
extern int MatCreateSequentialRow(int,int,int,int *,Mat*);
extern int MatCreateMPIRow(MPI_Comm,int,int,int,int,int,int*,int,int*,Mat*); 

extern int MatShellCreate(int,int,void *,Mat*);
extern int MatShellSetMult(Mat,int (*)(void*,Vec,Vec));
extern int MatShellSetMultTrans(Mat,int (*)(void*,Vec,Vec));
extern int MatShellSetMultTransAdd(Mat,int (*)(void*,Vec,Vec,Vec));
  
/* ------------------------------------------------------------*/
extern int  MatValidMatrix(Mat);

extern int MatSetValues(Mat,int,int*,int,int*,Scalar*,InsertMode);
extern int MatBeginAssembly(Mat);
extern int MatEndAssembly(Mat);
extern int MatSetOption(Mat,int);
#define ROW_ORIENTED              1 
#define COLUMN_ORIENTED           2
#define ROWS_SORTED               4
#define COLUMNS_SORTED            8
#define NO_NEW_NONZERO_LOCATIONS  16
#define YES_NEW_NONZERO_LOCATIONS 32

extern int MatGetValues(Mat,Scalar*,int,int*,int,int*);
extern int MatGetRow(Mat,int,int *,int **,Scalar**);
extern int MatRestoreRow(Mat,int,int *,int **,Scalar**);
extern int MatGetCol(Mat,int,int *,int **,Scalar**);
extern int MatRestoreCol(Mat,int,int *,int **,Scalar**);
extern int MatGetArray(Mat,Scalar **);
extern int MatMult(Mat,Vec,Vec);
extern int MatMultAdd(Mat,Vec,Vec,Vec);
extern int MatMultTrans(Mat,Vec,Vec);
extern int MatMultTransAdd(Mat,Vec,Vec,Vec);

#define ORDER_NATURAL 0
#define ORDER_ND      1
#define ORDER_1WD     2
#define ORDER_RCM     3
#define ORDER_QMD     4
extern int MatGetReordering(Mat,int,IS*,IS*);

extern int MatLUFactor(Mat,IS,IS);
extern int MatCholeskyFactor(Mat,IS);
extern int MatLUFactorSymbolic(Mat,IS,IS,Mat*);
extern int MatILUFactorSymbolic(Mat,IS,IS,int,Mat*);
extern int MatCholeskyFactorSymbolic(Mat,IS,Mat*);
extern int MatIncompleteCholeskyFactorSymbolic(Mat,IS,int,Mat*);
extern int MatLUFactorNumeric(Mat,Mat*);
extern int MatCholeskyFactorNumeric(Mat,Mat*);

extern int MatSolve(Mat,Vec,Vec);
extern int MatSolveAdd(Mat,Vec,Vec,Vec);
extern int MatSolveTrans(Mat,Vec,Vec);
extern int MatSolveTransAdd(Mat,Vec,Vec,Vec);

#define SOR_FORWARD_SWEEP            1
#define SOR_BACKWARD_SWEEP           2
#define SOR_SYMMETRIC_SWEEP          3
#define SOR_LOCAL_FORWARD_SWEEP      4
#define SOR_LOCAL_BACKWARD_SWEEP     8
#define SOR_LOCAL_SYMMETRIC_SWEEP    12
#define SOR_ZERO_INITIAL_GUESS       16
#define SOR_EISENSTAT                32
#define SOR_APPLY_UPPER              64
#define SOR_APPLY_LOWER              128
extern int MatRelax(Mat,Vec,double,int,double,int,Vec);

extern int MatCopy(Mat,Mat*);
extern int MatView(Mat,Viewer);
#include <stdio.h>
extern int MatPrintMatlab(Mat,FILE*,char *);
extern int MatNonZeros(Mat,int*);
extern int MatMemoryUsed(Mat,int*);
extern int MatGetDiagonal(Mat,Vec);
extern int MatTranspose(Mat);
extern int MatScale(Mat,Vec,Vec);
extern int MatShrink(Mat,int,int*,int,int*);
extern int MatEqual(Mat,Mat);
extern int MatScatterBegin(Mat,IS,IS,Mat,IS,IS,InsertMode,MatScatterCtx*);
extern int MatScatterEnd(Mat,IS,IS,Mat,IS,IS,InsertMode,MatScatterCtx*);
extern int MatReOrder(Mat,IS,IS);

#define NORM_1         1
#define NORM_2         2
#define NORM_FROBENIUS 3
#define NORM_INFINITY  4
extern int MatNorm(Mat,int,double *);

extern int MatZeroEntries(Mat);
extern int MatZeroRows(Mat,IS,Scalar*);
extern int MatZeroColumns(Mat,IS,Scalar*);

extern int MatCompress(Mat);
extern int MatDestroy(Mat);

extern int MatGetSize(Mat,int*,int*);
extern int MatGetLocalSize(Mat,int*,int*);
extern int MatGetOwnershipRange(Mat,int*,int*);

extern int MatCreateInitialMatrix(int,int,Mat*);




#endif


