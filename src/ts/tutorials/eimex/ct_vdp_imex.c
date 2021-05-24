/*
 * ex_vdp.c
 *
 *  Created on: Jun 1, 2012
 *      Author: Hong Zhang
 */
static char help[] = "Solves the van der Pol equation. \n Input parameters include:\n";

/*
 * Processors:1
 */

/*
 * This program solves the van der Pol equation
 * y' = z                               (1)
 * z' = (((1-y^2)*z-y)/eps              (2)
 * on the domain 0<=x<=0.5, with the initial conditions
 * y(0) = 2,
 * z(0) = -2/3 + 10/81*eps - 292/2187*eps^2-1814/19683*eps^3
 * IMEX schemes are applied to the splitted equation
 * [y'] = [z]  + [0                 ]
 * [z']   [0]    [(((1-y^2)*z-y)/eps]
 *
 * F(x)= [z]
 *       [0]
 *
 * G(x)= [y'] -   [0                 ]
 *       [z']     [(((1-y^2)*z-y)/eps]
 *
 * JG(x) =  G_x + a G_xdot
 */

#include <petscdmda.h>
#include <petscts.h>

typedef struct _User *User;
struct _User {
  PetscReal mu;  /*stiffness control coefficient: epsilon*/
};

static PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode IFunction(TS,PetscReal,Vec,Vec,Vec,void*);
static PetscErrorCode IJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);

int main(int argc, char **argv)
{
  TS                ts;
  Vec               x; /*solution vector*/
  Mat               A; /*Jacobian*/
  PetscInt          steps,mx,eimex_rowcol[2],two;
  PetscErrorCode    ierr;
  PetscScalar       *x_ptr;
  PetscReal         ftime,dt,norm;
  Vec               ref;
  struct _User      user;       /* user-defined work context */
  PetscViewer       viewer;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  /* Initialize user application context */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"van der Pol options","");
  user.mu      = 1e0;
  ierr = PetscOptionsReal("-eps","Stiffness controller","",user.mu,&user.mu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
   ierr = PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL);CHKERRQ(ierr);
   */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Create necessary matrix and vectors, solve same ODE on every process
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&x,NULL);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&ref,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(ref,&x_ptr);CHKERRQ(ierr);
  /*
   * [0,1], mu=10^-3
   */
  /*
   x_ptr[0] = -1.8881254106283;
   x_ptr[1] =  0.7359074233370;*/

  /*
   * [0,0.5],mu=10^-3
   */
  /*
   x_ptr[0] = 1.596980778659137;
   x_ptr[1] = -1.029103015879544;
   */
  /*
   * [0,0.5],mu=1
   */
  x_ptr[0] = 1.619084329683235;
  x_ptr[1] = -0.803530465176385;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Create timestepping solver context
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSEIMEX);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,&user);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,IFunction,&user);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,A,A,IJacobian,&user);CHKERRQ(ierr);

  dt    = 0.00001;
  ftime = 1.1;
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecGetArray(x,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2.;
  x_ptr[1] = -2./3. + 10./81.*(user.mu) - 292./2187.* (user.mu) * (user.mu)
    -1814./19683.*(user.mu)*(user.mu)*(user.mu);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);
  ierr = VecGetSize(x,&mx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,x);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);

  ierr = VecAXPY(x,-1.0,ref);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);

  eimex_rowcol[0] = 0; eimex_rowcol[1] = 0; two = 2;
  ierr = PetscOptionsGetIntArray(NULL,NULL,"-ts_eimex_row_col",eimex_rowcol,&two,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"order %11s %18s %37s\n","dt","norm","final solution components 0 and 1");CHKERRQ(ierr);
  ierr = VecGetArray(x,&x_ptr);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"(%D,%D) %10.8f %18.15f %18.15f %18.15f\n",eimex_rowcol[0],eimex_rowcol[1],(double)dt,(double)norm,(double)PetscRealPart(x_ptr[0]),(double)PetscRealPart(x_ptr[1]));CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&x_ptr);CHKERRQ(ierr);

  /* Write line in convergence log */
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer,PETSCVIEWERASCII);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer,FILE_MODE_APPEND);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer,"eimex_nonstiff_vdp.txt");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%D %D %10.8f %18.15f\n",eimex_rowcol[0],eimex_rowcol[1],(double)dt,(double)norm);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&ref);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ptr)
{
  PetscErrorCode    ierr;
  PetscScalar       *f;
  const PetscScalar *x;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = x[1];
  f[1] = 0.0;
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ptr)
{
  User              user = (User)ptr;
  PetscScalar       *f;
  const PetscScalar *x,*xdot;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = xdot[0];
  f[1] = xdot[1]-((1.-x[0]*x[0])*x[1]-x[0])/user->mu;
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobian(TS  ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,void *ptr)
{
  PetscErrorCode    ierr;
  User              user = (User)ptr;
  PetscReal         mu = user->mu;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *x;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  J[0][0] = a;
  J[0][1] = 0;
  J[1][0] = (2.*x[0]*x[1]+1.)/mu;
  J[1][1] = a - (1. - x[0]*x[0])/mu;
  ierr = MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*TEST

   test:
     args: -ts_type eimex -ts_adapt_type none  -pc_type lu -ts_dt 0.01 -ts_max_time 10 -ts_eimex_row_col 3,3 -ts_monitor_lg_solution
     requires: x

   test:
     suffix: adapt
     args: -ts_type eimex -ts_adapt_type none  -pc_type lu -ts_dt 0.01 -ts_max_time 10 -ts_eimex_order_adapt -ts_eimex_max_rows 7 -ts_monitor_lg_solution
     requires: x

   test:
     suffix: loop
     args: -ts_type eimex  -ts_adapt_type none  -pc_type lu -ts_dt {{0.005 0.001 0.0005}separate output} -ts_max_steps {{100 500 1000}separate output} -ts_eimex_row_col {{1,1 2,1 3,1 2,2 3,2 3,3}separate output}
     requires: x

 TEST*/
