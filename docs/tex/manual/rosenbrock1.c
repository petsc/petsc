#include "tao.h"
/* -------------- User-defined constructs ---------- */
typedef struct {
  int n;          /* dimension */
  double alpha;   /* condition parameter */
} AppCtx;
int FormFunctionGradient(TAO_APPLICATION,Vec,double*,Vec,void*);
int FormHessian(TAO_APPLICATION,Vec,Mat*,Mat*,MatStructure*,void*);

int main(int argc,char **argv)
{
  int        info;                  /* used to check for functions returning nonzeros */
  double     zero=0.0;
  Vec        x;                     /* solution vector */
  Mat        H;                     /* Hessian matrix */
  TAO_SOLVER tao;                   /* TAO_SOLVER solver context */
  TAO_APPLICATION taoapp;           /* TAO application context */
  AppCtx     user;                  /* user-defined application context */

  /* Initialize TAO and PETSc */
  PetscInitialize(&argc,&argv,(char *)0,0);
  TaoInitialize(&argc,&argv,(char *)0,0);
  user.n = 2; user.alpha = 99.0;

  /* Allocate vectors for the solution and gradient */
  info = VecCreateSeq(PETSC_COMM_SELF,user.n,&x); CHKERRQ(info);
  info = MatCreateSeqBDiag(PETSC_COMM_SELF,user.n,user.n,0,2,0,0,&H);CHKERRQ(info);

  /* Create TAO solver with desired solution method */
  info = TaoCreate(PETSC_COMM_SELF,"tao_lmvm",&tao); CHKERRQ(info);
  info = TaoApplicationCreate(PETSC_COMM_SELF,&taoapp); CHKERRQ(info);

  /* Set solution vec and an initial guess */
  info = VecSet(&zero,x); CHKERRQ(info);
  info = TaoAppSetInitialSolutionVec(taoapp,x); CHKERRQ(info); 

  /* Set routines for function, gradient, hessian evaluation */
  info = TaoAppSetObjectiveAndGradientRoutine(taoapp,FormFunctionGradient,(void *)&user); 
  CHKERRQ(info);
  info = TaoAppSetHessianMat(taoapp,H,H); CHKERRQ(info);
  info = TaoAppSetHessianRoutine(taoapp,FormHessian,(void *)&user); CHKERRQ(info);

  /* SOLVE THE APPLICATION */
  info = TaoSolveApplication(taoapp,tao); CHKERRQ(info);

  /* Free TAO data structures */
  info = TaoDestroy(tao); CHKERRQ(info);
  info = TaoAppDestroy(taoapp); CHKERRQ(info);

  /* Free PETSc data structures */
  info = VecDestroy(x); CHKERRQ(info);
  info = MatDestroy(H); CHKERRQ(info);

  /* Finalize TAO */
  TaoFinalize();
  PetscFinalize();
  return 0;
}
