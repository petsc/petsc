
static char help[] = "Transistor amplifier (autonomous).\n";

/*F
  M y'=f(y)

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
  U=0.4*sin(200*pi*t);
  PetscFunctionReturn(0);
  }*/

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
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  f[0]= PetscSinReal(200*PETSC_PI*y[5])/2500. - y[0]/1000. - ydot[0]/1.e6 + ydot[1]/1.e6;
  f[1]=0.0006666766666666667 -  PetscExpReal((500*(y[1] - y[2]))/13.)/1.e8 - y[1]/4500. + ydot[0]/1.e6 - ydot[1]/1.e6;
  f[2]=-1.e-6 +  PetscExpReal((500*(y[1] - y[2]))/13.)/1.e6 - y[2]/9000. - ydot[2]/500000.;
  f[3]=0.0006676566666666666 - (99* PetscExpReal((500*(y[1] - y[2]))/13.))/1.e8 - y[3]/9000. - (3*ydot[3])/1.e6 + (3*ydot[4])/1.e6;
  f[4]=-y[4]/9000. + (3*ydot[3])/1.e6 - (3*ydot[4])/1.e6;
  f[5]=-1 + ydot[5];

  ierr = VecRestoreArrayRead(Y,&y);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Ydot,&ydot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Defines the Jacobian of the ODE passed to the ODE solver. See TSSetIJacobian() for the meaning of a and the Jacobian.
*/
static PetscErrorCode IJacobianImplicit(TS ts,PetscReal t,Vec Y,Vec Ydot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       rowcol[] = {0,1,2,3,4,5};
  const PetscScalar    *y,*ydot;
  PetscScalar    J[6][6];

  PetscFunctionBegin;
  ierr    = VecGetArrayRead(Y,&y);CHKERRQ(ierr);
  ierr    = VecGetArrayRead(Ydot,&ydot);CHKERRQ(ierr);

  ierr = PetscMemzero(J,sizeof(J));CHKERRQ(ierr);

  J[0][0]=-0.001 - a/1.e6;
  J[0][1]=a/1.e6;
  J[0][5]=(2*PETSC_PI* PetscCosReal(200*PETSC_PI*y[5]))/25.;
  J[1][0]=a/1.e6;
  J[1][1]=-0.00022222222222222223 - a/1.e6 -  PetscExpReal((500*(y[1] - y[2]))/13.)/2.6e6;
  J[1][2]= PetscExpReal((500*(y[1] - y[2]))/13.)/2.6e6;
  J[2][1]= PetscExpReal((500*(y[1] - y[2]))/13.)/26000.;
  J[2][2]=-0.00011111111111111112 - a/500000. -  PetscExpReal((500*(y[1] - y[2]))/13.)/26000.;
  J[3][1]=(-99* PetscExpReal((500*(y[1] - y[2]))/13.))/2.6e6;
  J[3][2]=(99* PetscExpReal((500*(y[1] - y[2]))/13.))/2.6e6;
  J[3][3]=-0.00011111111111111112 - (3*a)/1.e6;
  J[3][4]=(3*a)/1.e6;
  J[4][3]=(3*a)/1.e6;
  J[4][4]=-0.00011111111111111112 - (3*a)/1.e6;
  J[5][5]=a;

  ierr    = MatSetValues(B,6,rowcol,6,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);

  ierr    = VecRestoreArrayRead(Y,&y);CHKERRQ(ierr);
  ierr    = VecRestoreArrayRead(Ydot,&ydot);CHKERRQ(ierr);

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
  PetscInt       n = 6;
  PetscScalar    *y;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscAssertFalse(size > 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for sequential runs");

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
  y[5] = 0.0;
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSSetEquationType(ts,TS_EQ_DAE_IMPLICIT_INDEX1);CHKERRQ(ierr);
  ierr = TSARKIMEXSetFullyImplicit(ts,PETSC_TRUE);CHKERRQ(ierr);
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
     Do Time stepping
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
