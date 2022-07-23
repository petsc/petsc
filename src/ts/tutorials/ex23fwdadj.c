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
  PetscCall(VecGetArrayRead(X,&x));
  PetscCall(VecGetArrayRead(Xdot,&xdot));
  PetscCall(VecGetArrayWrite(F,&f));
  f[0] = user->c*xdot[0] - user->b*x[0];
  PetscCall(VecRestoreArrayRead(X,&x));
  PetscCall(VecRestoreArrayRead(Xdot,&xdot));
  PetscCall(VecRestoreArrayWrite(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,void *ctx)
{
  User              user     = (User)ctx;
  PetscInt          rowcol[] = {0};
  PetscScalar       J[1][1];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X,&x));
  J[0][0] = user->c*a - user->b*1.0;
  PetscCall(MatSetValues(B,1,rowcol,1,rowcol,&J[0][0],INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(X,&x));

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
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
  PetscCall(VecGetArrayRead(X,&x));
  PetscCall(VecGetArrayRead(Xdot,&xdot));
  PetscCall(TSGetTimeStep(ts,&dt));
  if (user->der == 1) J[0][0] = xdot[0];
  if (user->der == 2) J[0][0] = -x[0];
  PetscCall(MatSetValues(A,1,row,1,col,&J[0][0],INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(X,&x));

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;
  PetscScalar    *x_ptr;
  PetscMPIInt    size;
  struct _n_User user;
  PetscInt       rows,cols;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  user.a           = 2.0;
  user.b           = 4.0;
  user.c           = 3.0;
  user.steps       = 0;
  user.ftime       = 1.0;
  user.der         = 1;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-der",&user.der,NULL));

  rows = 1;
  cols = 1;
  PetscCall(MatCreate(PETSC_COMM_WORLD,&user.Jac));
  PetscCall(MatSetSizes(user.Jac,PETSC_DECIDE,PETSC_DECIDE,1,1));
  PetscCall(MatSetFromOptions(user.Jac));
  PetscCall(MatSetUp(user.Jac));
  PetscCall(MatCreateVecs(user.Jac,&user.x,NULL));

  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetType(ts,TSBEULER));
  PetscCall(TSSetIFunction(ts,NULL,IFunction,&user));
  PetscCall(TSSetIJacobian(ts,user.Jac,user.Jac,IJacobian,&user));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetMaxTime(ts,user.ftime));

  PetscCall(VecGetArrayWrite(user.x,&x_ptr));
  x_ptr[0] = user.a;
  PetscCall(VecRestoreArrayWrite(user.x,&x_ptr));
  PetscCall(TSSetTimeStep(ts,0.001));

  /* Set up forward sensitivity */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&user.Jacp));
  PetscCall(MatSetSizes(user.Jacp,PETSC_DECIDE,PETSC_DECIDE,rows,cols));
  PetscCall(MatSetFromOptions(user.Jacp));
  PetscCall(MatSetUp(user.Jacp));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,rows,cols,NULL,&user.sp));
  PetscCall(MatZeroEntries(user.sp));
  PetscCall(TSForwardSetSensitivities(ts,cols,user.sp));
  PetscCall(TSSetIJacobianP(ts,user.Jacp,IJacobianP,&user));

  PetscCall(TSSetSaveTrajectory(ts));
  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSSolve(ts,user.x));
  PetscCall(TSGetSolveTime(ts,&user.ftime));
  PetscCall(TSGetStepNumber(ts,&user.steps));
  PetscCall(VecGetArray(user.x,&x_ptr));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n ode solution %g\n",(double)PetscRealPart(x_ptr[0])));
  PetscCall(VecRestoreArray(user.x,&x_ptr));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n analytical solution %g\n",(double)(user.a*PetscExpReal(user.b/user.c*user.ftime))));

  if (user.der == 1) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n analytical derivative w.r.t. c %g\n",(double)(-user.a*user.ftime*user.b/(user.c*user.c)*PetscExpReal(user.b/user.c*user.ftime))));
  }
  if (user.der == 2) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n analytical derivative w.r.t. b %g\n",(double)(user.a*user.ftime/user.c*PetscExpReal(user.b/user.c*user.ftime))));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n forward sensitivity:\n"));
  PetscCall(MatView(user.sp,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatCreateVecs(user.Jac,&user.lambda[0],NULL));
  /* Set initial conditions for the adjoint integration */
  PetscCall(VecGetArrayWrite(user.lambda[0],&x_ptr));
  x_ptr[0] = 1.0;
  PetscCall(VecRestoreArrayWrite(user.lambda[0],&x_ptr));
  PetscCall(MatCreateVecs(user.Jacp,&user.mup[0],NULL));
  PetscCall(VecGetArrayWrite(user.mup[0],&x_ptr));
  x_ptr[0] = 0.0;
  PetscCall(VecRestoreArrayWrite(user.mup[0],&x_ptr));

  PetscCall(TSSetCostGradients(ts,1,user.lambda,user.mup));
  PetscCall(TSAdjointSolve(ts));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n adjoint sensitivity:\n"));
  PetscCall(VecView(user.mup[0],PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&user.Jac));
  PetscCall(MatDestroy(&user.sp));
  PetscCall(MatDestroy(&user.Jacp));
  PetscCall(VecDestroy(&user.x));
  PetscCall(VecDestroy(&user.lambda[0]));
  PetscCall(VecDestroy(&user.mup[0]));
  PetscCall(TSDestroy(&ts));

  PetscCall(PetscFinalize());
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
