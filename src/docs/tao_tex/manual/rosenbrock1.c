#include "petsctao.h"

typedef struct {
  PetscInt  n;       /* dimension */
  PetscReal alpha;   /* condition parameter */
} AppCtx;

/* -------------- User-defined routines ---------- */
PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);
PetscErrorCode FormHessian(Tao,Vec,Mat,Mat,void*);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;  /* used to check for functions returning nonzeros */
  Vec            x;         /* solution vector */
  Mat            H;         /* Hessian matrix */
  Tao            tao;       /* Tao context */
  AppCtx         user;      /* user-defined application context */

  ierr = PetscInitialize(&argc,&argv,(char*)0,0);if (ierr) return ierr;

  /* Initialize problem parameters */
  user.n = 2; user.alpha = 99.0;

  /* Allocate vectors for the solution and gradient */
  ierr = VecCreateSeq(PETSC_COMM_SELF,user.n,&x);CHKERRQ(ierr);
  ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF,2,user.n,user.n,1,NULL,&H);

  /* Create TAO solver with desired solution method */
  ierr = TaoCreate(PETSC_COMM_SELF,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOLMVM);CHKERRQ(ierr);

  /* Set solution vec and an initial guess */
  ierr = VecSet(x, 0);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao,x);CHKERRQ(ierr);

  /* Set routines for function, gradient, hessian evaluation */
  ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,&user);
  ierr = TaoSetHessianRoutine(tao,H,H,FormHessian,&user);CHKERRQ(ierr);

  /* Check for TAO command line options */
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

  /* Solve the application */
  ierr = TaoSolve(tao);CHKERRQ(ierr);

  /* Free data structures */
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = MatDestroy(&H);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
