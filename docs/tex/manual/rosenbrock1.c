#include "tao.h"

typedef struct {
  int n;             /* dimension */
  PetscReal alpha;   /* condition parameter */
} AppCtx;

/* -------------- User-defined routines ---------- */
PetscErrorCode FormFunctionGradient(TaoSolver,Vec,PetscReal*,Vec,void*);
PetscErrorCode FormHessian(TaoSolver,Vec,Mat*,Mat*,MatStructure*,void*);

int main(int argc,char **argv)
{
  int        info;      /* used to check for functions returning nonzeros */
  Vec        x;         /* solution vector */
  Mat        H;         /* Hessian matrix */
  TaoSolver  tao;       /* TaoSolver context */
  AppCtx     user;      /* user-defined application context */

  /* Initialize TAO and PETSc */
  PetscInitialize(&argc,&argv,(char *)0,0);
  TaoInitialize(&argc,&argv,(char*)0,0);

  /* Initialize problem parameters */
  user.n = 2; user.alpha = 99.0;

  /* Allocate vectors for the solution and gradient */
  info = VecCreateSeq(PETSC_COMM_SELF,user.n,&x); CHKERRQ(info);
  info = MatCreateSeqBAIJ(PETSC_COMM_SELF,2,user.n,user.n,1,PETSC_NULL,&H); 
  CHKERRQ(info);

  /* Create TAO solver with desired solution method */
  info = TaoCreate(PETSC_COMM_SELF,&tao); CHKERRQ(info);
  info = TaoSetType(tao,"tao_lmvm"); CHKERRQ(info);

  /* Set solution vec and an initial guess */
  info = VecSet(x, 0); CHKERRQ(info);
  info = TaoSetInitialVector(tao,x); CHKERRQ(info); 

  /* Set routines for function, gradient, hessian evaluation */
  info = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,&user); 
  CHKERRQ(info);
  info = TaoSetHessianRoutine(tao,H,H,FormHessian,&user); CHKERRQ(info);
    
  /* Check for TAO command line options */
  info = TaoSetFromOptions(tao); CHKERRQ(info);

  /* SOLVE THE APPLICATION */
  info = TaoSolve(tao); CHKERRQ(info);

  /* Free TAO data structures */
  info = TaoDestroy(&tao); CHKERRQ(info);

  /* Free PETSc data structures */
  info = VecDestroy(&x); CHKERRQ(info);
  info = MatDestroy(&H); CHKERRQ(info);

  TaoFinalize();
  PetscFinalize();
  return 0;
}
