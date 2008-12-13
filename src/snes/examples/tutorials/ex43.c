
static char help[] = "Newton's method to solve a many-variable system that comes from the 2 variable Rosenbrock function + trivial.\n\n";

/* 

./ex43 -snes_monitor_range -snes_max_it 1000 -snes_rtol 1.e-14 -n 10 -snes_converged_reason -sub_snes_monito -sub_snes_mf -sub_snes_converged_reason -sub_snes_rtol 1.e-10 -sub_snes_max_it 1000 -sub_snes_monitor 

  Accelerates Newton's method by solving a small problem defined by those elements with large residual plus one level of overlap

  This is a toy code for playing around

  Counts residual entries as small if they are less then .2 times the maximum
  Decides to solve a reduced problem if the number of large entries is less than 20 percent of all entries (and this has been true for criteria_reduce iterations)
*/
#include "ex43-44.h"


extern PetscErrorCode FormJacobian1(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode FormFunction1(SNES,Vec,Vec,void*);

typedef struct {
  PetscInt   n,p;
} Ctx;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  SNES                snes;         /* nonlinear solver context */
  Vec                 x,r;          /* solution, residual vectors */
  Mat                 J;            /* Jacobian matrix */
  PetscErrorCode      ierr;
  PetscScalar         *xx;
  PetscInt            i,max_snes_solves = 20,snes_steps_per_solve = 2 ,criteria_reduce = 1;
  Ctx                 ctx;
  SNESConvergedReason reason;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ctx.n = 0;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&ctx.n,PETSC_NULL);CHKERRQ(ierr);
  ctx.p = 0;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-p",&ctx.p,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-max_snes_solves",&max_snes_solves,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-snes_steps_per_solve",&snes_steps_per_solve,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-criteria_reduce",&ctx.p,PETSC_NULL);CHKERRQ(ierr);

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
  ierr = VecSetSizes(x,PETSC_DECIDE,2+ctx.n+ctx.p);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);

  /*
     Create Jacobian matrix data structure
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,2+ctx.p+ctx.n,2+ctx.p+ctx.n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);

  /* 
     Set function evaluation routine and vector.
  */
  ierr = SNESSetFunction(snes,r,FormFunction1,(void*)&ctx);CHKERRQ(ierr);

  /* 
     Set Jacobian matrix data structure and Jacobian evaluation routine
  */
  ierr = SNESSetJacobian(snes,J,J,FormJacobian1,(void*)&ctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecSet(x,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  xx[0] = -1.2; 
  for (i=1; i<ctx.p+2; i++) {
    xx[i] = 1.0;
  }
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);

  /*
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */

  ierr = SNESMonitorSet(snes,MonitorRange,0,0);CHKERRQ(ierr);
  ierr = SNESSetTolerances(snes,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,snes_steps_per_solve,PETSC_DEFAULT);CHKERRQ(ierr);
  for (i=0; i<max_snes_solves; i++) { 
    ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes,&reason);CHKERRQ(ierr);
    if (reason && reason != SNES_DIVERGED_MAX_IT) break;
    if (CountGood > criteria_reduce) {
      ierr = SolveSubproblem(snes);CHKERRQ(ierr);
       CountGood = 0;
    }
  }

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
PetscErrorCode FormFunction1(SNES snes,Vec x,Vec f,void *ictx)
{
  PetscErrorCode ierr;
  PetscScalar    *xx,*ff;
  PetscInt       i;
  Ctx            *ctx = (Ctx*)ictx;

  /*
    Get pointers to vector data.
    - For default PETSc vectors, VecGetArray() returns a pointer to
    the data array.  Otherwise, the routine is implementation dependent.
    - You MUST call VecRestoreArray() when you no longer need access to
    the array.
  */
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(f,&ff);CHKERRQ(ierr);

  /* Compute function */
  ff[0] = -2.0 + 2.0*xx[0] + 400.0*xx[0]*xx[0]*xx[0] - 400.0*xx[0]*xx[1];
  for (i=1; i<1+ctx->p; i++) {
    ff[i] = -2.0 + 2.0*xx[i] + 400.0*xx[i]*xx[i]*xx[i] - 400.0*xx[i]*xx[i+1] + 200.0*(xx[i] - xx[i-1]*xx[i-1]);
  }
  ff[ctx->p+1] = -200.0*xx[ctx->p]*xx[ctx->p] + 200.0*xx[ctx->p+1];
  for (i=ctx->p+2; i<2+ctx->p+ctx->n; i++) {
    ff[i] = xx[i] - xx[0] + .7*xx[1] - .2*xx[i-1]*xx[i-1];
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
PetscErrorCode FormJacobian1(SNES snes,Vec x,Mat *jac,Mat *B,MatStructure *flag,void *ictx)
{
  PetscScalar    *xx;
  PetscErrorCode ierr;
  PetscInt       i;
  Ctx            *ctx = (Ctx*)ictx;

  ierr = MatZeroEntries(*B);CHKERRQ(ierr);
  /*
     Get pointer to vector data
  */
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);

  /*
     Compute Jacobian entries and insert into matrix.
      - Since this is such a small problem, we set all entries for
        the matrix at once.
  */
  ierr = MatSetValue(*B,0,0, 2.0 + 1200.0*xx[0]*xx[0] - 400.0*xx[1],ADD_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(*B,0,1,-400.0*xx[0] ,ADD_VALUES);CHKERRQ(ierr);

  for (i=1; i<ctx->p+1; i++) {
    ierr = MatSetValue(*B,i,i-1, -400.0*xx[i-1] ,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(*B,i,i, 2.0 + 1200.0*xx[i]*xx[i] - 400.0*xx[i+1] + 200.0,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(*B,i,i+1,-400.0*xx[i] ,ADD_VALUES);CHKERRQ(ierr);
  }

  ierr = MatSetValue(*B,ctx->p+1,ctx->p, -400.0*xx[ctx->p] ,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue(*B,ctx->p+1,ctx->p+1,200 ,ADD_VALUES);CHKERRQ(ierr);

  *flag = SAME_NONZERO_PATTERN;

  for (i=ctx->p+2; i<2+ctx->p+ctx->n; i++) {
    ierr = MatSetValue(*B,i,i,1.0,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(*B,i,0,-1.0,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(*B,i,1,.7,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(*B,i,i-1,-.4*xx[i-1],ADD_VALUES);CHKERRQ(ierr);
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

