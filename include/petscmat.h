/*
     Include file for the matrix component of PETSc
*/
#ifndef __MAT 
#define __MAT
#include "vec.h"

typedef struct _Mat*           Mat;
typedef struct _MatScatterCtx* MatScatterCtx;


extern int MatCreateSequentialDense(int,int,Mat*);
extern int MatCreateSequentialAIJ(int,int,int,int *,Mat*);

extern int MatShellCreate(int,int,void *,Mat*);
extern int MatShellSetMult(Mat,int (*)(void*,Vec,Vec));
extern int MatShellSetMultTrans(Mat,int (*)(void*,Vec,Vec));

  
/* ------------------------------------------------------------*/
extern int  MatValidMatrix(Mat);

extern int MatInsertValues(Mat,Scalar*,int,int*,int,int*);
extern int MatAddValues(Mat,Scalar*,int,int*,int,int*);
extern int MatBeginAssembly(Mat);
extern int MatEndAssembly(Mat);
extern int MatSetInsertOption(Mat,int);
#define ROW_ORIENTED              1 
#define COLUMN_ORIENTED           2
#define ROWS_SORTED               3
#define COLUMNS_SORTED            4
#define NO_NEW_NONZERO_LOCATIONS  5
#define YES_NEW_NONZERO_LOCATIONS 6

extern int MatGetValues(Mat,Scalar*,int,int*,int,int*);
extern int MatGetRow(Mat,int,int *,int **,Scalar**);
extern int MatRestoreRow(Mat,int,int *,int **,Scalar**);
extern int MatGetCol(Mat,int,int *,int **,Scalar**);
extern int MatRestoreCol(Mat,int,int *,int **,Scalar**);

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
extern int MatCholeskyFactorSymbolic(Mat,IS,Mat*);
extern int MatLUFactorNumeric(Mat,Mat);
extern int MatCholeskyFactorNumeric(Mat,Mat);

extern int MatSolve(Mat,Vec,Vec);
extern int MatSolveAdd(Mat,Vec,Vec,Vec);
extern int MatSolveTran(Mat,Vec,Vec);
extern int MatSolveTranAdd(Mat,Vec,Vec,Vec);

#define SOR_FORWARD_SWEEP      1
#define SOR_BACKWARD_SWEEP     2
#define SOR_SYMMETRIC_SWEEP    3
#define SOR_ZERO_INITIAL_GUESS 4
extern int MatRelax(Mat,Vec,double,int,IS,int,Vec);

extern int MatCopy(Mat,Mat*);
extern int MatView(Mat,Viewer);
#include <stdio.h>
extern int MatPrintMatlab(Mat,FILE*,char *);
extern int  MatNonZeros(Mat,int*);
extern int  MatMemoryUsed(Mat,int*);
extern int MatGetDiagonal(Mat,Vec);
extern int MatTranspose(Mat);
extern int MatScale(Mat,Vec,Vec);
extern int MatShrink(Mat,int,int*,int,int*);
extern int  MatEqual(Mat,Mat);
extern int  MatScatterBegin(Mat,IS,IS,Mat,IS,IS,MatScatterCtx*);
extern int  MatScatterEnd(Mat,IS,IS,Mat,IS,IS,MatScatterCtx*);
extern int  MatScatterAddBegin(Mat,IS,IS,Mat,IS,IS,MatScatterCtx*);
extern int  MatScatterAddEnd(Mat,IS,IS,Mat,IS,IS,MatScatterCtx*);
extern int MatReOrder(Mat,IS,IS);

#define NORM_1         1
#define NORM_2         2
#define NORM_FROBENIUS 3
#define NORM_INFINITY  4
extern int MatNorm(Mat,int,double *);

extern int MatZeroEntries(Mat);
extern int MatZeroRows(Mat);

extern int MatCompress(Mat);
extern int MatDestroy(Mat);

extern int MatCreateInitialMatrix(int,int,Mat*);



#endif


