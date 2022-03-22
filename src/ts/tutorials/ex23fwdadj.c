static char help[] = "A toy example for testing forward and adjoint sensitivity analysis of an implicit ODE with a paramerized mass matrice.\n";

/*
  This example solves the simple ODE
    c x' = b x, x(0) = a,
  whose analytical solution is x(T)=a*exp(b/c*T), and calculates the derivative of x(T) w.r.t. c (by default) or w.r.t. b (can be enabled with command line option -der 2).

*/

#include <petscts.h>

typedef struct _n_User *User;
struct _n_User {
  PetscReal a;
  PetscReal b;
  PetscReal c;
  /* Sensitivity analysis support */
  PetscInt  steps;
  PetscReal ftime;
  Mat       Jac;                    /* Jacobian matrix */
  Mat       Jacp;                   /* JacobianP matrix */
  Vec       x;
  Mat       sp;                     /* forward sensitivity variables */
  Vec       lambda[1];              /* adjoint sensitivity variables */
  Vec       mup[1];                 /* adjoint sensitivity variables */
  PetscInt  der;
};

static PetscErrorCode IFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscErrorCode    ierr;
  User              user = (User)ctx;
  const PetscScalar *x,*xdot;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(F,&f);CHKERRQ(ierr);
  f[0] = user->c*xdot[0] - user->b*x[0];
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscErrorCode    ierr;
  User              user     = (User)ctx;
  PetscInt          rowcol[] = {0};
  PetscScalar       J[1][1];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr    = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  J[0][0] = user->c*a - user->b*1.0;
  ierr    = MatSetValues(B,1,rowcol,1,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr    = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobianP(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal shift,Mat A,void *ctx)
{
  User              user = (User)ctx;
  PetscInt          row[] = {0},col[]={0};
  PetscScalar       J[1][1];
  const PetscScalar *x,*xdot;
  PetscReal         dt;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr    = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr    = VecGetArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr    = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  if (user->der == 1) J[0][0] = xdot[0];
  if (user->der == 2) J[0][0] = -x[0];
  ierr    = MatSetValues(A,1,row,1,col,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr    = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;
  PetscScalar    *x_ptr;
  PetscMPIInt    size;
  struct _n_User user;
  PetscInt       rows,cols;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  user.a           = 2.0;
  user.b           = 4.0;
  user.c           = 3.0;
  user.steps       = 0;
  user.ftime       = 1.0;
  user.der         = 1;
  ierr = PetscOptionsGetInt(NULL,NULL,"-der",&user.der,NULL);CHKERRQ(ierr);

  rows = 1;
  cols = 1;
  ierr = MatCreate(PETSC_COMM_WORLD,&user.Jac);CHKERRQ(ierr);
  ierr = MatSetSizes(user.Jac,PETSC_DECIDE,PETSC_DECIDE,1,1);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user.Jac);CHKERRQ(ierr);
  ierr = MatSetUp(user.Jac);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.Jac,&user.x,NULL);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,IFunction,&user);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,user.Jac,user.Jac,IJacobian,&user);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,user.ftime);CHKERRQ(ierr);

  ierr = VecGetArrayWrite(user.x,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = user.a;
  ierr = VecRestoreArrayWrite(user.x,&x_ptr);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.001);CHKERRQ(ierr);

  /* Set up forward sensitivity */
  ierr = MatCreate(PETSC_COMM_WORLD,&user.Jacp);CHKERRQ(ierr);
  ierr = MatSetSizes(user.Jacp,PETSC_DECIDE,PETSC_DECIDE,rows,cols);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user.Jacp);CHKERRQ(ierr);
  ierr = MatSetUp(user.Jacp);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,rows,cols,NULL,&user.sp);CHKERRQ(ierr);
  ierr = MatZeroEntries(user.sp);CHKERRQ(ierr);
  ierr = TSForwardSetSensitivities(ts,cols,user.sp);CHKERRQ(ierr);
  ierr = TSSetIJacobianP(ts,user.Jacp,IJacobianP,&user);CHKERRQ(ierr);

  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSolve(ts,user.x);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&user.ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&user.steps);CHKERRQ(ierr);
  ierr = VecGetArray(user.x,&x_ptr);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n ode solution %g\n",(double)PetscRealPart(x_ptr[0]));CHKERRQ(ierr);
  ierr = VecRestoreArray(user.x,&x_ptr);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n analytical solution %g\n",(double)user.a*PetscExpReal(user.b/user.c*user.ftime));CHKERRQ(ierr);

  if (user.der == 1) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n analytical derivative w.r.t. c %g\n",(double)-user.a*user.ftime*user.b/(user.c*user.c)*PetscExpReal(user.b/user.c*user.ftime));CHKERRQ(ierr);
  }
  if (user.der == 2) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n analytical derivative w.r.t. b %g\n",user.a*user.ftime/user.c*PetscExpReal(user.b/user.c*user.ftime));CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n forward sensitivity:\n");CHKERRQ(ierr);
  ierr = MatView(user.sp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatCreateVecs(user.Jac,&user.lambda[0],NULL);CHKERRQ(ierr);
  /* Set initial conditions for the adjoint integration */
  ierr = VecGetArrayWrite(user.lambda[0],&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 1.0;
  ierr = VecRestoreArrayWrite(user.lambda[0],&x_ptr);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.Jacp,&user.mup[0],NULL);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(user.mup[0],&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 0.0;
  ierr = VecRestoreArrayWrite(user.mup[0],&x_ptr);CHKERRQ(ierr);

  ierr = TSSetCostGradients(ts,1,user.lambda,user.mup);CHKERRQ(ierr);
  ierr = TSAdjointSolve(ts);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n adjoint sensitivity:\n");CHKERRQ(ierr);
  ierr = VecView(user.mup[0],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(&user.Jac);CHKERRQ(ierr);
  ierr = MatDestroy(&user.sp);CHKERRQ(ierr);
  ierr = MatDestroy(&user.Jacp);CHKERRQ(ierr);
  ierr = VecDestroy(&user.x);CHKERRQ(ierr);
  ierr = VecDestroy(&user.lambda[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&user.mup[0]);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return(ierr);
}

/*TEST

    test:
      args: -ts_type beuler

    test:
      suffix: 2
      args: -ts_type cn
      output_file: output/ex23fwdadj_1.out

TEST*/
