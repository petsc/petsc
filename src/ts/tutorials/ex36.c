
static char help[] = "Transistor amplifier.\n";

/*F
 ` This example illustrates the implementation of an implicit DAE index-1 of form M y'=f(t,y) with singular mass matrix, where

     [ -C1  C1           ]
     [  C1 -C1           ]
  M =[        -C2        ]; Ck = k * 1e-06
     [            -C3  C3]
     [             C3 -C3]


        [ -(U(t) - y[0])/1000                    ]
        [ -6/R + y[1]/4500 + 0.01 * h(y[1]-y[2]) ]
f(t,y)= [ y[2]/R - h(y[1]-y[2]) ]
        [ (y[3]-6)/9000 + 0.99 * h([y1]-y[2]) ]
        [ y[4]/9000 ]

U(t) = 0.4 * Sin(200 Pi t); h[V] = 1e-06 * Exp(V/0.026 - 1) `

  Useful options: -ts_monitor_lg_solution -ts_monitor_lg_timestep -lg_indicate_data_points 0
F*/

/*
   Include "petscts.h" so that we can use TS solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/
#include <petscts.h>

FILE *gfilepointer_data,*gfilepointer_info;

/* Defines the source  */
PetscErrorCode Ue(PetscScalar t,PetscScalar *U)
{
  PetscFunctionBegin;
  * U = 0.4*PetscSinReal(200*PETSC_PI*t);
  PetscFunctionReturn(0);
}

/*
     Defines the DAE passed to the time solver
*/
static PetscErrorCode IFunctionImplicit(TS ts,PetscReal t,Vec Y,Vec Ydot,Vec F,void *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *y,*ydot;
  PetscScalar       *f;

  PetscFunctionBegin;
  /*  The next three lines allow us to access the entries of the vectors directly */
  ierr = VecGetArrayRead(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Ydot,&ydot);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(F,&f);CHKERRQ(ierr);

  f[0] = ydot[0]/1.e6 - ydot[1]/1.e6 - PetscSinReal(200*PETSC_PI*t)/2500. + y[0]/1000.;
  f[1] = -ydot[0]/1.e6 + ydot[1]/1.e6 - 0.0006666766666666667 +  PetscExpReal((500*(y[1] - y[2]))/13.)/1.e8 + y[1]/4500.;
  f[2] = ydot[2]/500000. + 1.e-6 -  PetscExpReal((500*(y[1] - y[2]))/13.)/1.e6 + y[2]/9000.;
  f[3] = (3*ydot[3])/1.e6 - (3*ydot[4])/1.e6 - 0.0006676566666666666 + (99* PetscExpReal((500*(y[1] - y[2]))/13.))/1.e8 + y[3]/9000.;
  f[4] = (3*ydot[4])/1.e6 - (3*ydot[3])/1.e6 + y[4]/9000.;

  ierr = VecRestoreArrayRead(Y,&y);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Ydot,&ydot);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Defines the Jacobian of the ODE passed to the ODE solver. See TSSetIJacobian() for the meaning of a and the Jacobian.
*/
static PetscErrorCode IJacobianImplicit(TS ts,PetscReal t,Vec Y,Vec Ydot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          rowcol[] = {0,1,2,3,4};
  const PetscScalar *y,*ydot;
  PetscScalar       J[5][5];

  PetscFunctionBegin;
  ierr = VecGetArrayRead(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Ydot,&ydot);CHKERRQ(ierr);

  ierr = PetscMemzero(J,sizeof(J));CHKERRQ(ierr);

  J[0][0]= a/1.e6 + 0.001;
  J[0][1]= -a/1.e6;
  J[1][0]= -a/1.e6;
  J[1][1]= a/1.e6 + 0.00022222222222222223 +  PetscExpReal((500*(y[1] - y[2]))/13.)/2.6e6;
  J[1][2]= -PetscExpReal((500*(y[1] - y[2]))/13.)/2.6e6;
  J[2][1]= -PetscExpReal((500*(y[1] - y[2]))/13.)/26000.;
  J[2][2]= a/500000 + 0.00011111111111111112 +  PetscExpReal((500*(y[1] - y[2]))/13.)/26000.;
  J[3][1]= (99*PetscExpReal((500*(y[1] - y[2]))/13.))/2.6e6;
  J[3][2]= (-99*PetscExpReal((500*(y[1] - y[2]))/13.))/2.6e6;
  J[3][3]= (3*a)/1.e6 + 0.00011111111111111112;
  J[3][4]= -(3*a)/1.e6;
  J[4][3]= -(3*a)/1.e6;
  J[4][4]= (3*a)/1.e6 + 0.00011111111111111112 ;


  ierr = MatSetValues(B,5,rowcol,5,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(Y,&y);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Ydot,&ydot);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;            /* ODE integrator */
  Vec            Y;             /* solution will be stored here */
  Mat            A;             /* Jacobian matrix */
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n = 5;
  PetscScalar    *y;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for sequential runs");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&Y,NULL);CHKERRQ(ierr);

  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  y[0] = 0.0;
  y[1] = 3.0;
  y[2] = y[1];
  y[3] = 6.0;
  y[4] = 0.0;
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  /* Must use ARKIMEX with fully implicit stages since mass matrix is not the indentity */
  ierr = TSARKIMEXSetType(ts,TSARKIMEXPRSSP2);CHKERRQ(ierr);
  ierr = TSSetEquationType(ts,TS_EQ_DAE_IMPLICIT_INDEX1);CHKERRQ(ierr);
  /*ierr = TSSetType(ts,TSROSW);CHKERRQ(ierr);*/
  ierr = TSSetIFunction(ts,NULL,IFunctionImplicit,NULL);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,A,A,IJacobianImplicit,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetSolution(ts,Y);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetMaxTime(ts,0.15);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,.001);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Do time stepping
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,Y);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&Y);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
    build:
      requires: !single !complex
    test:
      args: -ts_monitor

TEST*/
