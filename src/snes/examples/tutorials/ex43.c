
static char help[] = "Newton's method to solve a many-variable system that comes from the 2 variable Rosenbrock function + trivial.\n\n";

/* 

./ex43 -snes_monitor_range -snes_max_it 1000 -snes_rtol 1.e-14 -n 10 -snes_converged_reason -sub_snes_monito -sub_snes_mf -sub_snes_converged_reason -sub_snes_rtol 1.e-10 -sub_snes_max_it 1000 -sub_snes_monitor -snes_max_it 500

  Accelerates Newton's method by solving a small problem defined by those elements with large residual plus one level of overlap
*/
#include "petscsnes.h"

extern PetscErrorCode FormJacobian1(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode FormFunction1(SNES,Vec,Vec,void*);
extern PetscErrorCode FormFunctionSub(SNES,Vec,Vec,void*);

typedef struct {
  PetscInt   n;
  Vec        xwork,fwork;
  VecScatter scatter;
  SNES       snes;
} SubCtx;

#undef __FUNCT__
#define __FUNCT__ "FormFunctionSub"
PetscErrorCode FormFunctionSub(SNES snes,Vec x,Vec f,void *ictx)
{
  PetscErrorCode ierr;
  SubCtx         *ctx = (SubCtx*) ictx;

  PetscFunctionBegin;
  ierr = VecScatterBegin(ctx->scatter,x,ctx->xwork,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatter,x,ctx->xwork,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = FormFunction1(ctx->snes,ctx->xwork,ctx->fwork,0);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatter,ctx->fwork,f,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatter,ctx->fwork,f,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SolveSubproblem"
PetscErrorCode SolveSubproblem(SNES snes)
{
  PetscErrorCode ierr;
  Vec            residual,solution;
  PetscReal      rmax;
  PetscInt       i,n,cnt,*indices;
  PetscScalar    *r;
  IS             is;
  SNES           snessub;
  Vec            x,f;
  SubCtx         ctx;
  Mat            mat;

  PetscFunctionBegin;
  ctx.snes = snes;
  ierr = SNESGetSolution(snes,&solution);CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,&residual,0,0);CHKERRQ(ierr);
  ierr = VecNorm(residual,NORM_INFINITY,&rmax);CHKERRQ(ierr);
  ierr = VecGetLocalSize(residual,&n);CHKERRQ(ierr);
  ierr = VecGetArray(residual,&r);CHKERRQ(ierr);
  cnt  = 0;
  for (i=0; i<n; i++) {
    if (PetscAbsScalar(r[i]) > .2*rmax ) cnt++;
  }
  ierr = PetscMalloc(cnt*sizeof(PetscInt),&indices);CHKERRQ(ierr);
  cnt  = 0;
  for (i=0; i<n; i++) {
    if (PetscAbsScalar(r[i]) > .2*rmax ) indices[cnt++] = i;
  }
  if (cnt > .2*n) PetscFunctionReturn(0);

  printf("number in subproblem %d\n",cnt);CHKERRQ(ierr);
  ierr = VecRestoreArray(residual,&r);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,cnt,indices,&is);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);

  ierr = SNESGetJacobian(snes,0,&mat,0,0);CHKERRQ(ierr);
  ierr = MatIncreaseOverlap(mat,1,&is,1);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is,&cnt);CHKERRQ(ierr);
  printf("number in subproblem %d\n",cnt);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,cnt,cnt);CHKERRQ(ierr);
  ierr = VecSetType(x,VECSEQ);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&f);CHKERRQ(ierr);
  ierr = VecScatterCreate(solution,is,x,PETSC_NULL,&ctx.scatter);CHKERRQ(ierr);
  ierr = ISDestroy(is);CHKERRQ(ierr);

  ierr = VecDuplicate(solution,&ctx.xwork);CHKERRQ(ierr);
  ierr = VecCopy(solution,ctx.xwork);CHKERRQ(ierr);
  ierr = VecDuplicate(residual,&ctx.fwork);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx.scatter,solution,x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx.scatter,solution,x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD,&snessub);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snessub,"sub_");CHKERRQ(ierr);
  ierr = SNESSetFunction(snessub,f,FormFunctionSub,(void*)&ctx);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snessub);CHKERRQ(ierr);
  ierr = SNESSolve(snessub,PETSC_NULL,x);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx.scatter,x,solution,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx.scatter,x,solution,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  ierr = VecDestroy(ctx.xwork);CHKERRQ(ierr);
  ierr = VecDestroy(ctx.fwork);CHKERRQ(ierr);
  ierr = VecScatterDestroy(ctx.scatter);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(f);CHKERRQ(ierr);
  ierr = SNESDestroy(snessub);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscErrorCode PETSCSNES_DLLEXPORT SNESMonitorRange_Private(SNES,PetscInt,PetscReal*);
static PetscInt CountGood = 0;

#undef __FUNCT__  
#define __FUNCT__ "MonitorRange"
PetscErrorCode PETSCSNES_DLLEXPORT MonitorRange(SNES snes,PetscInt it,PetscReal rnorm,void *dummy)
{
  PetscErrorCode          ierr;
  PetscReal               perc,rel;

  ierr = SNESMonitorRange_Private(snes,it,&perc);CHKERRQ(ierr);
  if (perc < .20) CountGood++;
  else CountGood = 0;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  SNES           snes;         /* nonlinear solver context */
  Vec            x,r;          /* solution, residual vectors */
  Mat            J;            /* Jacobian matrix */
  PetscErrorCode ierr;
  PetscInt       its;
  PetscScalar    *xx;
  PetscInt       i,n = 0;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix and vector data structures; set corresponding routines
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create vectors for solution and nonlinear function
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,2+n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);

  /*
     Create Jacobian matrix data structure
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,2+n,2+n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);

  /* 
     Set function evaluation routine and vector.
  */
  ierr = SNESSetFunction(snes,r,FormFunction1,PETSC_NULL);CHKERRQ(ierr);

  /* 
     Set Jacobian matrix data structure and Jacobian evaluation routine
  */
  ierr = SNESSetJacobian(snes,J,J,FormJacobian1,PETSC_NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecSet(x,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  xx[0] = -1.2; xx[1] = 1.0;
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);

  /*
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */

  ierr = SNESMonitorSet(snes,MonitorRange,0,0);CHKERRQ(ierr);
  ierr = SNESSetTolerances(snes,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,1,PETSC_DEFAULT);CHKERRQ(ierr);
  for (i=0; i<100; i++) { 
    ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);
    if (CountGood > 0) {
       ierr = SolveSubproblem(snes);CHKERRQ(ierr);
       CountGood = 0;
    }
  }
  ierr = VecView(x,0);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"number of Newton iterations = %D\n\n",its);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = VecDestroy(x);CHKERRQ(ierr); ierr = VecDestroy(r);CHKERRQ(ierr);
  ierr = MatDestroy(J);CHKERRQ(ierr); ierr = SNESDestroy(snes);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunction1"
/* 
   FormFunction1 - Evaluates nonlinear function, F(x).

   Input Parameters:
.  snes - the SNES context
.  x    - input vector
.  ctx  - optional user-defined context

   Output Parameter:
.  f - function vector
 */
PetscErrorCode FormFunction1(SNES snes,Vec x,Vec f,void *ctx)
{
  PetscErrorCode ierr;
  PetscScalar    *xx,*ff;
  PetscInt       i,n;

  /*
    Get pointers to vector data.
    - For default PETSc vectors, VecGetArray() returns a pointer to
    the data array.  Otherwise, the routine is implementation dependent.
    - You MUST call VecRestoreArray() when you no longer need access to
    the array.
  */
  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(f,&ff);CHKERRQ(ierr);

  /* Compute function */
  ff[0] = -2.0 + 2.0*xx[0] + 400.0*xx[0]*xx[0]*xx[0] - 400.0*xx[0]*xx[1] - xx[2];
  ff[1] = -200.0*xx[0]*xx[0] + 200.0*xx[1];

  for (i=2; i<n; i++) {
    ff[i] = xx[i] - xx[0] + .2*xx[1];
  }

  /* Restore vectors */
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&ff);CHKERRQ(ierr); 
  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormJacobian1"
/*
   FormJacobian1 - Evaluates Jacobian matrix.

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  dummy - optional user-defined context (not used here)

   Output Parameters:
.  jac - Jacobian matrix
.  B - optionally different preconditioning matrix
.  flag - flag indicating matrix structure
*/
PetscErrorCode FormJacobian1(SNES snes,Vec x,Mat *jac,Mat *B,MatStructure *flag,void *dummy)
{
  PetscScalar    *xx,A[4];
  PetscErrorCode ierr;
  PetscInt       idx[2] = {0,1},i,n;

  /*
     Get pointer to vector data
  */
  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);

  /*
     Compute Jacobian entries and insert into matrix.
      - Since this is such a small problem, we set all entries for
        the matrix at once.
  */
  A[0] = 2.0 + 1200.0*xx[0]*xx[0] - 400.0*xx[1]; A[1] = -400*xx[0];
  A[2] = -400*xx[0]; A[3] = 200;
  ierr = MatSetValues(*B,2,idx,2,idx,A,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(*B,0,2,-1.0,INSERT_VALUES);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;

  for (i=2; i<n; i++) {
    ierr = MatSetValue(*B,i,i,1.0,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(*B,i,0,-1.0,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(*B,i,1,.2,INSERT_VALUES);CHKERRQ(ierr);
  }
  /*
     Restore vector
  */
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);

  /* 
     Assemble matrix
  */
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*jac != *B){
    ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  return 0;
}

