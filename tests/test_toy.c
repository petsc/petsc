#include "taosolver.h"
typedef struct {
  PetscInt n;
  PetscInt m;
  IS ais;
} AppCtx;

PetscErrorCode FormFunction(TaoSolver, Vec, PetscReal*, void*);
PetscErrorCode FormGradient(TaoSolver, Vec, Vec, void*);
PetscErrorCode FormFunctionGradient(TaoSolver, Vec, PetscReal*, Vec, void*);
PetscErrorCode FormJacobianState(TaoSolver, Vec, Mat*, Mat*, MatStructure*,void*);
PetscErrorCode FormJacobianDesign(TaoSolver, Vec, Mat*, Mat*, MatStructure*,void*);
PetscErrorCode FormConstraints(TaoSolver, Vec, Vec, void*);
PetscErrorCode FormHessian(TaoSolver, Vec, Mat*, Mat*, MatStructure*, void*);
static  char help[]="Demonstrates use of the TAO package to solve \n";

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  Vec x;
  Vec c;
  Mat Js;
  Mat Jd;
  TaoSolver tao;
  TaoSolverTerminationReason reason;
  AppCtx user;
  PetscInt idx[] = {0,1};

  PetscInitialize(&argc, &argv, (char*)0,help);
  TaoInitialize(&argc, &argv, (char*)0,help);

  ierr = PetscPrintf(PETSC_COMM_SELF,"\n --- Toy Problem for testing RSQN ---\n");
  user.n = 3;
  user.m = 2;

  ierr = ISCreateGeneral(PETSC_COMM_SELF,user.m,idx,PETSC_COPY_VALUES,&user.ais); CHKERRQ(ierr);
  
  ierr = VecCreateSeq(PETSC_COMM_SELF,user.n,&x); CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,user.m,&c); CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, user.m, 2, 2, PETSC_NULL,&Js); CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, user.m, 1, 1, PETSC_NULL,&Jd); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Js,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Js,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Jd,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jd,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  /* Create TAO solver and set desired solution method */
  ierr = TaoSolverCreate(PETSC_COMM_SELF,&tao); CHKERRQ(ierr);
  ierr = TaoSolverSetType(tao,"tao_rsqn"); CHKERRQ(ierr);

  /* Set solution vector with an initial guess */
  ierr = VecSet(c, 0); CHKERRQ(ierr);
  ierr = VecSet(x, 0); CHKERRQ(ierr);

  ierr = TaoSolverSetInitialVector(tao,x); CHKERRQ(ierr);
  ierr = TaoSolverSetObjectiveRoutine(tao, FormFunction, (void *)&user); CHKERRQ(ierr);
  ierr = TaoSolverSetGradientRoutine(tao, FormGradient, (void *)&user); CHKERRQ(ierr);
  ierr = TaoSolverSetConstraintsRoutine(tao, c, FormConstraints, (void *)&user); CHKERRQ(ierr);

  ierr = TaoSolverSetJacobianStateRoutine(tao, Js, Js, FormJacobianState, (void *)&user); CHKERRQ(ierr);
  ierr = TaoSolverSetJacobianDesignRoutine(tao, Jd, Jd, FormJacobianDesign, (void *)&user); CHKERRQ(ierr);
  //ierr = TaoSolverSetHessianRoutine(tao, H, H, FormHessian,  (void *)&user); CHKERRQ(ierr);
  ierr = TaoSolverRSQNSetStateIS(tao,user.ais); CHKERRQ(ierr);
  ierr = TaoSolverSetFromOptions(tao); CHKERRQ(ierr);

  /* SOLVE THE APPLICATION */
  ierr = TaoSolverSolve(tao);  CHKERRQ(ierr);

  ierr = TaoSolverGetConvergedReason(tao,&reason); CHKERRQ(ierr);

  if (reason < 0)
  {
    PetscPrintf(MPI_COMM_SELF, "Try a different TAO method. RSQN failed.\n");
  }
  else
  {
    PetscPrintf(MPI_COMM_SELF, "Optimization terminated with status %2d.\n", reason);
  }


  /* Free TAO data structures */
  ierr = TaoSolverDestroy(tao); CHKERRQ(ierr);

  /* Free PETSc data structures */
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = VecDestroy(&c); CHKERRQ(ierr);
  ierr = MatDestroy(&Js); CHKERRQ(ierr);
  ierr = MatDestroy(&Jd); CHKERRQ(ierr);
  ierr = ISDestroy(&user.ais); CHKERRQ(ierr);

  /* Finalize TAO, PETSc */
  TaoFinalize();
  PetscFinalize();

  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunction"
/* 
   FormFunction - Evaluates the function, f(X).

   Input Parameters:
.  taoapp - the TAO_APPLICATION context
.  X   - the input vector 
.  ptr - optional user-defined context, as set by TaoSetFunction()

   Output Parameters:
.  f    - the newly evaluated function
*/
PetscErrorCode FormFunction(TaoSolver tao,Vec X,PetscReal *f,void *ptr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecDot(X, X, f); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormGradient"
/*  
    FormGradient - Evaluates the gradient, G(X).              

    Input Parameters:
.   taoapp  - the TAO_APPLICATION context
.   X    - input vector
.   ptr  - optional user-defined context
    
    Output Parameters:
.   G - vector containing the newly evaluated gradient
*/
PetscErrorCode FormGradient(TaoSolver tao,Vec X,Vec G,void *ptr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecCopy(X, G); CHKERRQ(ierr);
  ierr = VecScale(G, 2); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "FormFunctionGradient"
PetscErrorCode FormFunctionGradient(TaoSolver tao, Vec X, PetscScalar *f, Vec G, void *ptr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecDot(X,X,f); CHKERRQ(ierr);
  ierr = VecCopy(X,G); CHKERRQ(ierr);
  ierr = VecScale(G,2); CHKERRQ(ierr);
  PetscFunctionReturn(0);

}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormJacobianState"
PetscErrorCode FormJacobianState(TaoSolver tao, Vec X, Mat *tH, Mat* tHPre, MatStructure* flag, void *ptr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  *flag = DIFFERENT_NONZERO_PATTERN;

  ierr = MatSetValue(*tH, 0, 0,  1.0, INSERT_VALUES); CHKERRQ(ierr);
  ierr = MatSetValue(*tH, 0, 1,    0, INSERT_VALUES); CHKERRQ(ierr);

  ierr = MatSetValue(*tH, 1, 0,    0, INSERT_VALUES); CHKERRQ(ierr);
  ierr = MatSetValue(*tH, 1, 1,  1.0, INSERT_VALUES); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*tH, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*tH, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormJacobianDesign"
PetscErrorCode FormJacobianDesign(TaoSolver tao, Vec X, Mat *tH, Mat* tHPre, MatStructure* flag, void *ptr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  *flag = DIFFERENT_NONZERO_PATTERN;

  ierr = MatSetValue(*tH, 0, 0, -1.0, INSERT_VALUES); CHKERRQ(ierr);

  ierr = MatSetValue(*tH, 1, 0, -1.0, INSERT_VALUES); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*tH, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*tH, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormHessian"
PetscErrorCode FormHessian(TaoSolver tao, Vec X, Mat *tH, Mat* tHPre, MatStructure* flag, void *ptr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  *flag = SAME_NONZERO_PATTERN;

  ierr = MatSetValue(*tH, 0, 0, 2.0, INSERT_VALUES); CHKERRQ(ierr);
  ierr = MatSetValue(*tH, 1, 1, 2.0, INSERT_VALUES); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*tH, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*tH, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormConstraints"
PetscErrorCode FormConstraints(TaoSolver tao, Vec X, Vec C, void *)
{
   PetscErrorCode ierr;
   PetscScalar *a;
   PetscScalar *b;
   PetscFunctionBegin;
   ierr = VecGetArray(X,&a); CHKERRQ(ierr);
   ierr = VecGetArray(C,&b); CHKERRQ(ierr);
   b[0] = a[0]-a[2] -1;
   b[1] = a[1]-a[2] -1;
   ierr = VecRestoreArray(C, &b); CHKERRQ(ierr);
   ierr = VecRestoreArray(X, &a); CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

