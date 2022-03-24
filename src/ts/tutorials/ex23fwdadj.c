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
  User              user = (User)ctx;
  const PetscScalar *x,*xdot;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArrayRead(Xdot,&xdot));
  CHKERRQ(VecGetArrayWrite(F,&f));
  f[0] = user->c*xdot[0] - user->b*x[0];
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArrayRead(Xdot,&xdot));
  CHKERRQ(VecRestoreArrayWrite(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,void *ctx)
{
  User              user     = (User)ctx;
  PetscInt          rowcol[] = {0};
  PetscScalar       J[1][1];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(X,&x));
  J[0][0] = user->c*a - user->b*1.0;
  CHKERRQ(MatSetValues(B,1,rowcol,1,rowcol,&J[0][0],INSERT_VALUES));
  CHKERRQ(VecRestoreArrayRead(X,&x));

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (A != B) {
    CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
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

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArrayRead(Xdot,&xdot));
  CHKERRQ(TSGetTimeStep(ts,&dt));
  if (user->der == 1) J[0][0] = xdot[0];
  if (user->der == 2) J[0][0] = -x[0];
  CHKERRQ(MatSetValues(A,1,row,1,col,&J[0][0],INSERT_VALUES));
  CHKERRQ(VecRestoreArrayRead(X,&x));

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;
  PetscScalar    *x_ptr;
  PetscMPIInt    size;
  struct _n_User user;
  PetscInt       rows,cols;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));

  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  user.a           = 2.0;
  user.b           = 4.0;
  user.c           = 3.0;
  user.steps       = 0;
  user.ftime       = 1.0;
  user.der         = 1;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-der",&user.der,NULL));

  rows = 1;
  cols = 1;
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user.Jac));
  CHKERRQ(MatSetSizes(user.Jac,PETSC_DECIDE,PETSC_DECIDE,1,1));
  CHKERRQ(MatSetFromOptions(user.Jac));
  CHKERRQ(MatSetUp(user.Jac));
  CHKERRQ(MatCreateVecs(user.Jac,&user.x,NULL));

  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetType(ts,TSBEULER));
  CHKERRQ(TSSetIFunction(ts,NULL,IFunction,&user));
  CHKERRQ(TSSetIJacobian(ts,user.Jac,user.Jac,IJacobian,&user));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  CHKERRQ(TSSetMaxTime(ts,user.ftime));

  CHKERRQ(VecGetArrayWrite(user.x,&x_ptr));
  x_ptr[0] = user.a;
  CHKERRQ(VecRestoreArrayWrite(user.x,&x_ptr));
  CHKERRQ(TSSetTimeStep(ts,0.001));

  /* Set up forward sensitivity */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user.Jacp));
  CHKERRQ(MatSetSizes(user.Jacp,PETSC_DECIDE,PETSC_DECIDE,rows,cols));
  CHKERRQ(MatSetFromOptions(user.Jacp));
  CHKERRQ(MatSetUp(user.Jacp));
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,rows,cols,NULL,&user.sp));
  CHKERRQ(MatZeroEntries(user.sp));
  CHKERRQ(TSForwardSetSensitivities(ts,cols,user.sp));
  CHKERRQ(TSSetIJacobianP(ts,user.Jacp,IJacobianP,&user));

  CHKERRQ(TSSetSaveTrajectory(ts));
  CHKERRQ(TSSetFromOptions(ts));

  CHKERRQ(TSSolve(ts,user.x));
  CHKERRQ(TSGetSolveTime(ts,&user.ftime));
  CHKERRQ(TSGetStepNumber(ts,&user.steps));
  CHKERRQ(VecGetArray(user.x,&x_ptr));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n ode solution %g\n",(double)PetscRealPart(x_ptr[0])));
  CHKERRQ(VecRestoreArray(user.x,&x_ptr));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n analytical solution %g\n",(double)user.a*PetscExpReal(user.b/user.c*user.ftime)));

  if (user.der == 1) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n analytical derivative w.r.t. c %g\n",(double)-user.a*user.ftime*user.b/(user.c*user.c)*PetscExpReal(user.b/user.c*user.ftime)));
  }
  if (user.der == 2) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n analytical derivative w.r.t. b %g\n",user.a*user.ftime/user.c*PetscExpReal(user.b/user.c*user.ftime)));
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n forward sensitivity:\n"));
  CHKERRQ(MatView(user.sp,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatCreateVecs(user.Jac,&user.lambda[0],NULL));
  /* Set initial conditions for the adjoint integration */
  CHKERRQ(VecGetArrayWrite(user.lambda[0],&x_ptr));
  x_ptr[0] = 1.0;
  CHKERRQ(VecRestoreArrayWrite(user.lambda[0],&x_ptr));
  CHKERRQ(MatCreateVecs(user.Jacp,&user.mup[0],NULL));
  CHKERRQ(VecGetArrayWrite(user.mup[0],&x_ptr));
  x_ptr[0] = 0.0;
  CHKERRQ(VecRestoreArrayWrite(user.mup[0],&x_ptr));

  CHKERRQ(TSSetCostGradients(ts,1,user.lambda,user.mup));
  CHKERRQ(TSAdjointSolve(ts));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n adjoint sensitivity:\n"));
  CHKERRQ(VecView(user.mup[0],PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatDestroy(&user.Jac));
  CHKERRQ(MatDestroy(&user.sp));
  CHKERRQ(MatDestroy(&user.Jacp));
  CHKERRQ(VecDestroy(&user.x));
  CHKERRQ(VecDestroy(&user.lambda[0]));
  CHKERRQ(VecDestroy(&user.mup[0]));
  CHKERRQ(TSDestroy(&ts));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

    test:
      args: -ts_type beuler

    test:
      suffix: 2
      args: -ts_type cn
      output_file: output/ex23fwdadj_1.out

TEST*/
