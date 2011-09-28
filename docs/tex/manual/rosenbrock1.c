#include "tao.h"

typedef struct {
  PetscIng n;             /* dimension */
  PetscReal alpha;   /* condition parameter */
} AppCtx;

/* -------------- User-defined routines ---------- */
PetscErrorCode FormFunctionGradient(TaoSolver,Vec,PetscReal*,Vec,void*);
PetscErrorCode FormHessian(TaoSolver,Vec,Mat*,Mat*,MatStructure*,void*);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;  /* used to check for functions returning nonzeros */
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
  ierr = VecCreateSeq(PETSC_COMM_SELF,user.n,&x); CHKERRQ(ierr);
  ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF,2,user.n,user.n,1,PETSC_NULL,&H); 
  CHKERRQ(ierr);

  /* Create TAO solver with desired solution method */
  ierr = TaoCreate(PETSC_COMM_SELF,&tao); CHKERRQ(ierr);
  ierr = TaoSetType(tao,"tao_lmvm"); CHKERRQ(ierr);

  /* Set solution vec and an initial guess */
  ierr = VecSet(x, 0); CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao,x); CHKERRQ(ierr); 

  /* Set routines for function, gradient, hessian evaluation */
  ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,&user); 
  CHKERRQ(ierr);
  ierr = TaoSetHessianRoutine(tao,H,H,FormHessian,&user); CHKERRQ(ierr);
    
  /* Check for TAO command line options */
  ierr = TaoSetFromOptions(tao); CHKERRQ(ierr);

  /* SOLVE THE APPLICATION */
  ierr = TaoSolve(tao); CHKERRQ(ierr);

  /* Free TAO data structures */
  ierr = TaoDestroy(&tao); CHKERRQ(ierr);

  /* Free PETSc data structures */
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = MatDestroy(&H); CHKERRQ(ierr);

  TaoFinalize();
  PetscFinalize();
  return 0;
}
