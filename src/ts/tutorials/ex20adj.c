static char help[] = "Performs adjoint sensitivity analysis for the van der Pol equation.\n";

/* ------------------------------------------------------------------------

   This program solves the van der Pol DAE ODE equivalent
      [ u_1' ] = [          u_2                ]  (2)
      [ u_2' ]   [ \mu ((1 - u_1^2) u_2 - u_1) ]
   on the domain 0 <= x <= 1, with the boundary conditions
       u_1(0) = 2, u_2(0) = - 2/3 +10/(81*\mu) - 292/(2187*\mu^2),
   and
       \mu = 10^6 ( y'(0) ~ -0.6666665432100101).,
   and computes the sensitivities of the final solution w.r.t. initial conditions and parameter \mu with the implicit theta method and its discrete adjoint.

   Notes:
   This code demonstrates the TSAdjoint interface to a DAE system.

   The user provides the implicit right-hand-side function
   [ F(u',u,t) ] = [u' - f(u,t)] = [ u_1'] - [        u_2             ]
                                   [ u_2']   [ \mu ((1-u_1^2)u_2-u_1) ]

   and the Jacobian of F (from the PETSc user manual)

              dF   dF
   J(F) = a * -- + --
              du'  du

   and the JacobianP of the explicit right-hand side of (2) f(u,t) ( which is equivalent to -F(0,u,t)).
   df   [       0               ]
   -- = [                       ]
   dp   [ (1 - u_1^2) u_2 - u_1 ].

   See ex20.c for more details on the Jacobian.

  ------------------------------------------------------------------------- */
#include <petscts.h>
#include <petsctao.h>

typedef struct _n_User *User;
struct _n_User {
  PetscReal mu;
  PetscReal next_output;

  /* Sensitivity analysis support */
  PetscInt  steps;
  PetscReal ftime;
  Mat       A;                   /* Jacobian matrix */
  Mat       Jacp;                /* JacobianP matrix */
  Vec       U,lambda[2],mup[2];  /* adjoint variables */
};

/* ----------------------- Explicit form of the ODE  -------------------- */

static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec U,Vec F,void *ctx)
{
  User              user = (User)ctx;
  PetscScalar       *f;
  const PetscScalar *u;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArray(F,&f));
  f[0] = u[1];
  f[1] = user->mu*((1.-u[0]*u[0])*u[1]-u[0]);
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec U,Mat A,Mat B,void *ctx)
{
  User              user = (User)ctx;
  PetscReal         mu   = user->mu;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U,&u));
  J[0][0] = 0;
  J[1][0] = -mu*(2.0*u[1]*u[0]+1.);
  J[0][1] = 1.0;
  J[1][1] = mu*(1.0-u[0]*u[0]);
  PetscCall(MatSetValues(A,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscFunctionReturn(0);
}

/* ----------------------- Implicit form of the ODE  -------------------- */

static PetscErrorCode IFunction(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,void *ctx)
{
  User              user = (User)ctx;
  const PetscScalar *u,*udot;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArrayRead(Udot,&udot));
  PetscCall(VecGetArray(F,&f));
  f[0] = udot[0] - u[1];
  f[1] = udot[1] - user->mu*((1.0-u[0]*u[0])*u[1] - u[0]);
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArrayRead(Udot,&udot));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat B,void *ctx)
{
  User              user     = (User)ctx;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U,&u));

  J[0][0] = a;     J[0][1] =  -1.0;
  J[1][0] = user->mu*(2.0*u[0]*u[1] + 1.0);   J[1][1] = a - user->mu*(1.0-u[0]*u[0]);

  PetscCall(MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(U,&u));

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (B && A != B) {
    PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobianP(TS ts,PetscReal t,Vec U,Mat A,void *ctx)
{
  PetscInt          row[] = {0,1},col[]={0};
  PetscScalar       J[2][1];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U,&u));
  J[0][0] = 0;
  J[1][0] = (1.-u[0]*u[0])*u[1]-u[0];
  PetscCall(MatSetValues(A,2,row,1,col,&J[0][0],INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscFunctionReturn(0);
}

/* Monitor timesteps and use interpolation to output at integer multiples of 0.1 */
static PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec U,void *ctx)
{
  const PetscScalar *u;
  PetscReal         tfinal, dt;
  User              user = (User)ctx;
  Vec               interpolatedU;

  PetscFunctionBeginUser;
  PetscCall(TSGetTimeStep(ts,&dt));
  PetscCall(TSGetMaxTime(ts,&tfinal));

  while (user->next_output <= t && user->next_output <= tfinal) {
    PetscCall(VecDuplicate(U,&interpolatedU));
    PetscCall(TSInterpolate(ts,user->next_output,interpolatedU));
    PetscCall(VecGetArrayRead(interpolatedU,&u));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[%g] %D TS %g (dt = %g) X %g %g\n",
                        (double)user->next_output,step,(double)t,(double)dt,(double)PetscRealPart(u[0]),
                        (double)PetscRealPart(u[1])));
    PetscCall(VecRestoreArrayRead(interpolatedU,&u));
    PetscCall(VecDestroy(&interpolatedU));
    user->next_output += 0.1;
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;
  PetscBool      monitor = PETSC_FALSE,implicitform = PETSC_TRUE;
  PetscScalar    *x_ptr,*y_ptr,derp;
  PetscMPIInt    size;
  struct _n_User user;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.next_output = 0.0;
  user.mu          = 1.0e3;
  user.steps       = 0;
  user.ftime       = 0.5;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-mu",&user.mu,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-implicitform",&implicitform,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors, solve same ODE on every process
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&user.A));
  PetscCall(MatSetSizes(user.A,PETSC_DECIDE,PETSC_DECIDE,2,2));
  PetscCall(MatSetFromOptions(user.A));
  PetscCall(MatSetUp(user.A));
  PetscCall(MatCreateVecs(user.A,&user.U,NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&user.Jacp));
  PetscCall(MatSetSizes(user.Jacp,PETSC_DECIDE,PETSC_DECIDE,2,1));
  PetscCall(MatSetFromOptions(user.Jacp));
  PetscCall(MatSetUp(user.Jacp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetEquationType(ts,TS_EQ_ODE_EXPLICIT)); /* less Jacobian evaluations when adjoint BEuler is used, otherwise no effect */
  if (implicitform) {
    PetscCall(TSSetIFunction(ts,NULL,IFunction,&user));
    PetscCall(TSSetIJacobian(ts,user.A,user.A,IJacobian,&user));
    PetscCall(TSSetType(ts,TSCN));
  } else {
    PetscCall(TSSetRHSFunction(ts,NULL,RHSFunction,&user));
    PetscCall(TSSetRHSJacobian(ts,user.A,user.A,RHSJacobian,&user));
    PetscCall(TSSetType(ts,TSRK));
  }
  PetscCall(TSSetRHSJacobianP(ts,user.Jacp,RHSJacobianP,&user));
  PetscCall(TSSetMaxTime(ts,user.ftime));
  PetscCall(TSSetTimeStep(ts,0.001));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  if (monitor) PetscCall(TSMonitorSet(ts,Monitor,&user,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecGetArray(user.U,&x_ptr));
  x_ptr[0] = 2.0;
  x_ptr[1] = -2.0/3.0 + 10.0/(81.0*user.mu) - 292.0/(2187.0*user.mu*user.mu);
  PetscCall(VecRestoreArray(user.U,&x_ptr));
  PetscCall(TSSetTimeStep(ts,0.001));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Save trajectory of solution so that TSAdjointSolve() may be used
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetSaveTrajectory(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSSolve(ts,user.U));
  PetscCall(TSGetSolveTime(ts,&user.ftime));
  PetscCall(TSGetStepNumber(ts,&user.steps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreateVecs(user.A,&user.lambda[0],NULL));
  /* Set initial conditions for the adjoint integration */
  PetscCall(VecGetArray(user.lambda[0],&y_ptr));
  y_ptr[0] = 1.0; y_ptr[1] = 0.0;
  PetscCall(VecRestoreArray(user.lambda[0],&y_ptr));
  PetscCall(MatCreateVecs(user.A,&user.lambda[1],NULL));
  PetscCall(VecGetArray(user.lambda[1],&y_ptr));
  y_ptr[0] = 0.0; y_ptr[1] = 1.0;
  PetscCall(VecRestoreArray(user.lambda[1],&y_ptr));

  PetscCall(MatCreateVecs(user.Jacp,&user.mup[0],NULL));
  PetscCall(VecGetArray(user.mup[0],&x_ptr));
  x_ptr[0] = 0.0;
  PetscCall(VecRestoreArray(user.mup[0],&x_ptr));
  PetscCall(MatCreateVecs(user.Jacp,&user.mup[1],NULL));
  PetscCall(VecGetArray(user.mup[1],&x_ptr));
  x_ptr[0] = 0.0;
  PetscCall(VecRestoreArray(user.mup[1],&x_ptr));

  PetscCall(TSSetCostGradients(ts,2,user.lambda,user.mup));

  PetscCall(TSAdjointSolve(ts));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt initial conditions: d[y(tf)]/d[y0]  d[y(tf)]/d[z0]\n"));
  PetscCall(VecView(user.lambda[0],PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt initial conditions: d[z(tf)]/d[y0]  d[z(tf)]/d[z0]\n"));
  PetscCall(VecView(user.lambda[1],PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecGetArray(user.mup[0],&x_ptr));
  PetscCall(VecGetArray(user.lambda[0],&y_ptr));
  derp = y_ptr[1]*(-10.0/(81.0*user.mu*user.mu)+2.0*292.0/(2187.0*user.mu*user.mu*user.mu))+x_ptr[0];
  PetscCall(VecRestoreArray(user.mup[0],&x_ptr));
  PetscCall(VecRestoreArray(user.lambda[0],&y_ptr));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt parameters: d[y(tf)]/d[mu]\n%g\n",(double)PetscRealPart(derp)));

  PetscCall(VecGetArray(user.mup[1],&x_ptr));
  PetscCall(VecGetArray(user.lambda[1],&y_ptr));
  derp = y_ptr[1]*(-10.0/(81.0*user.mu*user.mu)+2.0*292.0/(2187.0*user.mu*user.mu*user.mu))+x_ptr[0];
  PetscCall(VecRestoreArray(user.mup[1],&x_ptr));
  PetscCall(VecRestoreArray(user.lambda[1],&y_ptr));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n sensivitity wrt parameters: d[z(tf)]/d[mu]\n%g\n",(double)PetscRealPart(derp)));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&user.A));
  PetscCall(MatDestroy(&user.Jacp));
  PetscCall(VecDestroy(&user.U));
  PetscCall(VecDestroy(&user.lambda[0]));
  PetscCall(VecDestroy(&user.lambda[1]));
  PetscCall(VecDestroy(&user.mup[0]));
  PetscCall(VecDestroy(&user.mup[1]));
  PetscCall(TSDestroy(&ts));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      requires: revolve
      args: -monitor 0 -ts_type theta -ts_theta_endpoint -ts_theta_theta 0.5 -viewer_binary_skip_info -ts_dt 0.001 -mu 100000

    test:
      suffix: 2
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_solution_only

    test:
      suffix: 3
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_solution_only 0
      output_file: output/ex20adj_2.out

    test:
      suffix: 4
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_stride 5 -ts_trajectory_solution_only -ts_trajectory_save_stack
      output_file: output/ex20adj_2.out

    test:
      suffix: 5
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_stride 5 -ts_trajectory_solution_only 0 -ts_trajectory_save_stack
      output_file: output/ex20adj_2.out

    test:
      suffix: 6
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_stride 5 -ts_trajectory_solution_only -ts_trajectory_save_stack 0
      output_file: output/ex20adj_2.out

    test:
      suffix: 7
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_stride 5 -ts_trajectory_solution_only 0 -ts_trajectory_save_stack 0
      output_file: output/ex20adj_2.out

    test:
      suffix: 8
      requires: revolve !cams
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 5 -ts_trajectory_solution_only -ts_trajectory_monitor
      output_file: output/ex20adj_3.out

    test:
      suffix: 9
      requires: revolve !cams
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 5 -ts_trajectory_solution_only 0 -ts_trajectory_monitor
      output_file: output/ex20adj_4.out

    test:
      requires: revolve
      suffix: 10
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 5 -ts_trajectory_revolve_online -ts_trajectory_solution_only
      output_file: output/ex20adj_2.out

    test:
      requires: revolve
      suffix: 11
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 5 -ts_trajectory_revolve_online -ts_trajectory_solution_only 0
      output_file: output/ex20adj_2.out

    test:
      suffix: 12
      requires: revolve
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_max_cps_disk 8 -ts_trajectory_solution_only
      output_file: output/ex20adj_2.out

    test:
      suffix: 13
      requires: revolve
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_max_cps_disk 8 -ts_trajectory_solution_only 0
      output_file: output/ex20adj_2.out

    test:
      suffix: 14
      requires: revolve
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_stride 5 -ts_trajectory_solution_only -ts_trajectory_save_stack
      output_file: output/ex20adj_2.out

    test:
      suffix: 15
      requires: revolve
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_stride 5 -ts_trajectory_solution_only -ts_trajectory_save_stack 0
      output_file: output/ex20adj_2.out

    test:
      suffix: 16
      requires: revolve
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_stride 5 -ts_trajectory_solution_only 0 -ts_trajectory_save_stack
      output_file: output/ex20adj_2.out

    test:
      suffix: 17
      requires: revolve
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_stride 5 -ts_trajectory_solution_only 0 -ts_trajectory_save_stack 0
      output_file: output/ex20adj_2.out

    test:
      suffix: 18
      requires: revolve
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_max_cps_disk 8 -ts_trajectory_stride 5 -ts_trajectory_solution_only -ts_trajectory_save_stack
      output_file: output/ex20adj_2.out

    test:
      suffix: 19
      requires: revolve
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_max_cps_disk 8 -ts_trajectory_stride 5 -ts_trajectory_solution_only 0 -ts_trajectory_save_stack
      output_file: output/ex20adj_2.out

    test:
      suffix: 20
      requires: revolve
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_max_cps_disk 8 -ts_trajectory_solution_only 0
      output_file: output/ex20adj_2.out

    test:
      suffix: 21
      requires: revolve
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_max_cps_disk 8 -ts_trajectory_stride 5 -ts_trajectory_solution_only 0 -ts_trajectory_save_stack 0
      output_file: output/ex20adj_2.out

    test:
      suffix: 22
      args: -ts_type beuler -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_solution_only
      output_file: output/ex20adj_2.out

    test:
      suffix: 23
      requires: cams
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_units_ram 5 -ts_trajectory_solution_only -ts_trajectory_monitor -ts_trajectory_memory_type cams
      output_file: output/ex20adj_5.out

    test:
      suffix: 24
      requires: cams
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_units_ram 5 -ts_trajectory_solution_only 0 -ts_trajectory_monitor -ts_trajectory_memory_type cams
      output_file: output/ex20adj_6.out

TEST*/
