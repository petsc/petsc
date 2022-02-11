
static char help[] = "Transistor amplifier (semi-explicit).\n";

/*F
  [I 0] [y'] = f(t,y,z)
  [0 0] [z'] = g(t,y,z)
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
static PetscErrorCode IFunctionSemiExplicit(TS ts,PetscReal t,Vec Y,Vec Ydot,Vec F,void *ctx)
{
  PetscErrorCode     ierr;
  const PetscScalar  *y,*ydot;
  PetscScalar        *f;

  PetscFunctionBegin;
  /*  The next three lines allow us to access the entries of the vectors directly */
  ierr = VecGetArrayRead(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Ydot,&ydot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  f[0]=-400* PetscSinReal(200*PETSC_PI*t) + 1000*y[3] + ydot[0];
  f[1]=0.5 - 1/(2.* PetscExpReal((500*(y[0] + y[1] - y[3]))/13.)) + (500*y[1])/9. + ydot[1];
  f[2]=-222.5522222222222 + 33/(100.* PetscExpReal((500*(y[0] + y[1] - y[3]))/13.)) + (1000*y[4])/27. + ydot[2];
  f[3]=0.0006666766666666667 - 1/(1.e8* PetscExpReal((500*(y[0] + y[1] - y[3]))/13.)) +  PetscSinReal(200*PETSC_PI*t)/2500. + y[0]/4500. - (11*y[3])/9000.;
  f[4]=0.0006676566666666666 - 99/(1.e8* PetscExpReal((500*(y[0] + y[1] - y[3]))/13.)) + y[2]/9000. - y[4]/4500.;

  ierr = VecRestoreArrayRead(Y,&y);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Ydot,&ydot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Defines the Jacobian of the ODE passed to the ODE solver. See TSSetIJacobian() for the meaning of a and the Jacobian.
*/
static PetscErrorCode IJacobianSemiExplicit(TS ts,PetscReal t,Vec Y,Vec Ydot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscErrorCode     ierr;
  PetscInt           rowcol[] = {0,1,2,3,4};
  const PetscScalar  *y,*ydot;
  PetscScalar        J[5][5];

  PetscFunctionBegin;
  ierr    = VecGetArrayRead(Y,&y);CHKERRQ(ierr);
  ierr    = VecGetArrayRead(Ydot,&ydot);CHKERRQ(ierr);

  ierr = PetscMemzero(J,sizeof(J));CHKERRQ(ierr);

  J[0][0]=a;
  J[0][3]=1000;
  J[1][0]=250/(13.* PetscExpReal((500*(y[0] + y[1] - y[3]))/13.));
  J[1][1]=55.55555555555556 + a + 250/(13.* PetscExpReal((500*(y[0] + y[1] - y[3]))/13.));
  J[1][3]=-250/(13.* PetscExpReal((500*(y[0] + y[1] - y[3]))/13.));
  J[2][0]=-165/(13.* PetscExpReal((500*(y[0] + y[1] - y[3]))/13.));
  J[2][1]=-165/(13.* PetscExpReal((500*(y[0] + y[1] - y[3]))/13.));
  J[2][2]=a;
  J[2][3]=165/(13.* PetscExpReal((500*(y[0] + y[1] - y[3]))/13.));
  J[2][4]=37.03703703703704;
  J[3][0]=0.00022222222222222223 + 1/(2.6e6* PetscExpReal((500*(y[0] + y[1] - y[3]))/13.));
  J[3][1]=1/(2.6e6* PetscExpReal((500*(y[0] + y[1] - y[3]))/13.));
  J[3][3]=-0.0012222222222222222 - 1/(2.6e6* PetscExpReal((500*(y[0] + y[1] - y[3]))/13.));
  J[4][0]=99/(2.6e6* PetscExpReal((500*(y[0] + y[1] - y[3]))/13.));
  J[4][1]=99/(2.6e6* PetscExpReal((500*(y[0] + y[1] - y[3]))/13.));
  J[4][2]=0.00011111111111111112;
  J[4][3]=-99/(2.6e6* PetscExpReal((500*(y[0] + y[1] - y[3]))/13.));
  J[4][4]=-0.00022222222222222223;

  ierr    = MatSetValues(B,5,rowcol,5,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);

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
  PetscInt       n = 5;
  PetscScalar    *y;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscCheckFalse(size > 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for sequential runs");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&Y,NULL);CHKERRQ(ierr);

  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  y[0] = -3.0;
  y[1] =  3.0;
  y[2] =  6.0;
  y[3] =  0.0;
  y[4] =  6.0;
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
  ierr = TSSetIFunction(ts,NULL,IFunctionSemiExplicit,NULL);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,A,A,IJacobianSemiExplicit,NULL);CHKERRQ(ierr);

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
