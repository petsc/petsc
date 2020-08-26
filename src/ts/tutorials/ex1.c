
static char help[] ="Solves the time independent Bratu problem using pseudo-timestepping.";

/*
   Concepts: TS^pseudo-timestepping
   Concepts: pseudo-timestepping
   Concepts: TS^nonlinear problems
   Processors: 1

*/

/* ------------------------------------------------------------------------

    This code demonstrates how one may solve a nonlinear problem
    with pseudo-timestepping. In this simple example, the pseudo-timestep
    is the same for all grid points, i.e., this is equivalent to using
    the backward Euler method with a variable timestep.

    Note: This example does not require pseudo-timestepping since it
    is an easy nonlinear problem, but it is included to demonstrate how
    the pseudo-timestepping may be done.

    See snes/tutorials/ex4.c[ex4f.F] and
    snes/tutorials/ex5.c[ex5f.F] where the problem is described
    and solved using Newton's method alone.

  ----------------------------------------------------------------------------- */
/*
    Include "petscts.h" to use the PETSc timestepping routines. Note that
    this file automatically includes "petscsys.h" and other lower-level
    PETSc include files.
*/
#include <petscts.h>

/*
  Create an application context to contain data needed by the
  application-provided call-back routines, FormJacobian() and
  FormFunction().
*/
typedef struct {
  PetscReal param;          /* test problem parameter */
  PetscInt  mx;             /* Discretization in x-direction */
  PetscInt  my;             /* Discretization in y-direction */
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode  FormJacobian(TS,PetscReal,Vec,Mat,Mat,void*), FormFunction(TS,PetscReal,Vec,Vec,void*), FormInitialGuess(Vec,AppCtx*);

int main(int argc,char **argv)
{
  TS             ts;                 /* timestepping context */
  Vec            x,r;               /* solution, residual vectors */
  Mat            J;                  /* Jacobian matrix */
  AppCtx         user;               /* user-defined work context */
  PetscInt       its,N;                /* iterations for convergence */
  PetscErrorCode ierr;
  PetscReal      param_max = 6.81,param_min = 0.,dt;
  PetscReal      ftime;
  PetscMPIInt    size;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only");

  user.mx    = 4;
  user.my    = 4;
  user.param = 6.0;

  /*
     Allow user to set the grid dimensions and nonlinearity parameter at run-time
  */
  PetscOptionsGetInt(NULL,NULL,"-mx",&user.mx,NULL);
  PetscOptionsGetInt(NULL,NULL,"-my",&user.my,NULL);
  N  = user.mx*user.my;
  dt = .5/PetscMax(user.mx,user.my);
  PetscOptionsGetReal(NULL,NULL,"-param",&user.param,NULL);
  if (user.param >= param_max || user.param <= param_min) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Parameter is out of range");

  /*
      Create vectors to hold the solution and function value
  */
  ierr = VecCreateSeq(PETSC_COMM_SELF,N,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);

  /*
    Create matrix to hold Jacobian. Preallocate 5 nonzeros per row
    in the sparse matrix. Note that this is not the optimal strategy; see
    the Performance chapter of the users manual for information on
    preallocating memory in sparse matrices.
  */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,5,0,&J);CHKERRQ(ierr);

  /*
     Create timestepper context
  */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);

  /*
     Tell the timestepper context where to compute solutions
  */
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);

  /*
     Provide the call-back for the nonlinear function we are
     evaluating. Thus whenever the timestepping routines need the
     function they will call this routine. Note the final argument
     is the application context used by the call-back functions.
  */
  ierr = TSSetRHSFunction(ts,NULL,FormFunction,&user);CHKERRQ(ierr);

  /*
     Set the Jacobian matrix and the function used to compute
     Jacobians.
  */
  ierr = TSSetRHSJacobian(ts,J,J,FormJacobian,&user);CHKERRQ(ierr);

  /*
       Form the initial guess for the problem
  */
  ierr = FormInitialGuess(x,&user);CHKERRQ(ierr);

  /*
       This indicates that we are using pseudo timestepping to
     find a steady state solution to the nonlinear problem.
  */
  ierr = TSSetType(ts,TSPSEUDO);CHKERRQ(ierr);

  /*
       Set the initial time to start at (this is arbitrary for
     steady state problems); and the initial timestep given above
  */
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);

  /*
      Set a large number of timesteps and final duration time
     to insure convergence to steady state.
  */
  ierr = TSSetMaxSteps(ts,10000);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,1e12);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);

  /*
      Use the default strategy for increasing the timestep
  */
  ierr = TSPseudoSetTimeStep(ts,TSPseudoTimeStepDefault,0);CHKERRQ(ierr);

  /*
      Set any additional options from the options database. This
     includes all options for the nonlinear and linear solvers used
     internally the timestepping routines.
  */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSetUp(ts);CHKERRQ(ierr);

  /*
      Perform the solve. This is where the timestepping takes place.
  */
  ierr = TSSolve(ts,x);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);

  /*
      Get the number of steps
  */
  ierr = TSGetStepNumber(ts,&its);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of pseudo timesteps = %D final time %4.2e\n",its,(double)ftime);CHKERRQ(ierr);

  /*
     Free the data structures constructed above
  */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
/* ------------------------------------------------------------------ */
/*           Bratu (Solid Fuel Ignition) Test Problem                 */
/* ------------------------------------------------------------------ */

/* --------------------  Form initial approximation ----------------- */

PetscErrorCode FormInitialGuess(Vec X,AppCtx *user)
{
  PetscInt       i,j,row,mx,my;
  PetscErrorCode ierr;
  PetscReal      one = 1.0,lambda;
  PetscReal      temp1,temp,hx,hy;
  PetscScalar    *x;

  mx     = user->mx;
  my     = user->my;
  lambda = user->param;

  hx = one / (PetscReal)(mx-1);
  hy = one / (PetscReal)(my-1);

  ierr  = VecGetArray(X,&x);CHKERRQ(ierr);
  temp1 = lambda/(lambda + one);
  for (j=0; j<my; j++) {
    temp = (PetscReal)(PetscMin(j,my-j-1))*hy;
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        x[row] = 0.0;
        continue;
      }
      x[row] = temp1*PetscSqrtReal(PetscMin((PetscReal)(PetscMin(i,mx-i-1))*hx,temp));
    }
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  return 0;
}
/* --------------------  Evaluate Function F(x) --------------------- */

PetscErrorCode FormFunction(TS ts,PetscReal t,Vec X,Vec F,void *ptr)
{
  AppCtx            *user = (AppCtx*)ptr;
  PetscErrorCode    ierr;
  PetscInt          i,j,row,mx,my;
  PetscReal         two = 2.0,one = 1.0,lambda;
  PetscReal         hx,hy,hxdhy,hydhx;
  PetscScalar       ut,ub,ul,ur,u,uxx,uyy,sc,*f;
  const PetscScalar *x;

  mx     = user->mx;
  my     = user->my;
  lambda = user->param;

  hx    = one / (PetscReal)(mx-1);
  hy    = one / (PetscReal)(my-1);
  sc    = hx*hy;
  hxdhy = hx/hy;
  hydhx = hy/hx;

  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  for (j=0; j<my; j++) {
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        f[row] = x[row];
        continue;
      }
      u      = x[row];
      ub     = x[row - mx];
      ul     = x[row - 1];
      ut     = x[row + mx];
      ur     = x[row + 1];
      uxx    = (-ur + two*u - ul)*hydhx;
      uyy    = (-ut + two*u - ub)*hxdhy;
      f[row] = -uxx + -uyy + sc*lambda*PetscExpScalar(u);
    }
  }
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  return 0;
}
/* --------------------  Evaluate Jacobian F'(x) -------------------- */

/*
   Calculate the Jacobian matrix J(X,t).

   Note: We put the Jacobian in the preconditioner storage B instead of J. This
   way we can give the -snes_mf_operator flag to check our work. This replaces
   J with a finite difference approximation, using our analytic Jacobian B for
   the preconditioner.
*/
PetscErrorCode FormJacobian(TS ts,PetscReal t,Vec X,Mat J,Mat B,void *ptr)
{
  AppCtx            *user = (AppCtx*)ptr;
  PetscInt          i,j,row,mx,my,col[5];
  PetscErrorCode    ierr;
  PetscScalar       two = 2.0,one = 1.0,lambda,v[5],sc;
  const PetscScalar *x;
  PetscReal         hx,hy,hxdhy,hydhx;


  mx     = user->mx;
  my     = user->my;
  lambda = user->param;

  hx    = 1.0 / (PetscReal)(mx-1);
  hy    = 1.0 / (PetscReal)(my-1);
  sc    = hx*hy;
  hxdhy = hx/hy;
  hydhx = hy/hx;

  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  for (j=0; j<my; j++) {
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        ierr = MatSetValues(B,1,&row,1,&row,&one,INSERT_VALUES);CHKERRQ(ierr);
        continue;
      }
      v[0] = hxdhy; col[0] = row - mx;
      v[1] = hydhx; col[1] = row - 1;
      v[2] = -two*(hydhx + hxdhy) + sc*lambda*PetscExpScalar(x[row]); col[2] = row;
      v[3] = hydhx; col[3] = row + 1;
      v[4] = hxdhy; col[4] = row + mx;
      ierr = MatSetValues(B,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J != B) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  return 0;
}


/*TEST

    test:
      args: -ksp_gmres_cgs_refinement_type refine_always -snes_type newtonls -ts_monitor_pseudo -snes_atol 1.e-7 -ts_pseudo_frtol 1.e-5 -ts_view draw:tikz:fig.tex

    test:
      suffix: 2
      args: -ts_monitor_pseudo -ts_pseudo_frtol 1.e-5

TEST*/
