/*
     Include file for the matrix component of PETSc
*/
#ifndef __MAT_PACKAGE 
#define __MAT_PACKAGE

#if !defined(IS_PACKAGE)
  is.h must be included before mat.h
#endif
#if !defined(VEC_COMPONENT)
  vec.h must be included before mat.h
#endif

typedef struct _Mat*           Mat;
typedef double*                MatScalar;
typedef struct _MatScatterCtx* MatScatterCtx;

extern int MatCreateSequentialDense   ANSI_ARGS((int,int,Mat *));
extern int MatCreateSequentialAIJ     ANSI_ARGS((int,int,Mat *));
  
extern int  MatValidMatrix            ANSI_ARGS((Mat));

extern int MatInsertValues      ANSI_ARGS((Mat,MatScalar,int,int*,int,int*));
extern int MatAddValues         ANSI_ARGS((Mat,MatScalar,int,int*,int,int*));
extern int MatBeginAssembly     ANSI_ARGS((Mat));
extern int MatEndAssembly       ANSI_ARGS((Mat));

extern int MatGetValues         ANSI_ARGS((Mat,MatScalar,int,int*,int,int*));
extern int MatGetRow            ANSI_ARGS((Mat,int,int *,int **,MatScalar *));
extern int MatRestoreRow        ANSI_ARGS((Mat,int,int *,int **,MatScalar *));
extern int MatGetCol            ANSI_ARGS((Mat,int,int *,int **,MatScalar *));
extern int MatRestoreCol        ANSI_ARGS((Mat,int,int *,int **,MatScalar *));

extern int MatMult         ANSI_ARGS((Mat,Vec,Vec));
extern int MatMultAdd      ANSI_ARGS((Mat,Vec,Vec,Vec));
extern int MatMultTran     ANSI_ARGS((Mat,Vec,Vec));
extern int MatMultTranAdd  ANSI_ARGS((Mat,Vec,Vec,Vec));

extern int MatLUFactor           ANSI_ARGS((Mat));
extern int MatCholeskyFactor     ANSI_ARGS((Mat));
extern int MatSolve              ANSI_ARGS((Mat,Vec,Vec));
extern int MatSolveAdd           ANSI_ARGS((Mat,Vec,Vec,Vec));
extern int MatSolveTran          ANSI_ARGS((Mat,Vec,Vec));
extern int MatSolveTranAdd       ANSI_ARGS((Mat,Vec,Vec,Vec));

extern int MatRelax         ANSI_ARGS((Mat,Vec,double,int,Vec));
extern int MatRelaxForward  ANSI_ARGS((Mat,Vec,double,Vec));
extern int MatRelaxBackward ANSI_ARGS((Mat,Vec,double,Vec));

extern int MatCopy          ANSI_ARGS((Mat,Mat *));
extern int MatView         ANSI_ARGS((Mat,void*));
extern int MatPrintMatlab  ANSI_ARGS((Mat,FILE*,char *));
extern int  MatNonZeros     ANSI_ARGS((Mat));
extern int  MatMemoryUsed   ANSI_ARGS((Mat));
extern int MatGetDiagonal  ANSI_ARGS((Mat,Vec *));
extern int MatTranspose    ANSI_ARGS((Mat));
extern int MatScale        ANSI_ARGS((Mat,Vec,Vec));
extern int  MatShrink       ANSI_ARGS((Mat,int,int*,int,int*));
extern int  MatEqual        ANSI_ARGS((Mat,Mat));
extern int  MatGetSubMatrix ANSI_ARGS((Mat,IS,IS,Mat *));
extern int MatSetSubMatrix ANSI_ARGS((Mat,Mat,IS,IS));
extern int MatReOrder      ANSI_ARGS((Mat,IS,IS));

#define NORM_1         1
#define NORM_2         2
#define NORM_FROBENIUS 3
#define NORM_INFINITY  4
extern int MatNorm         ANSI_ARGS((Mat,int,double *));

extern int MatDestroy      ANSI_ARGS((Mat));
#endif


