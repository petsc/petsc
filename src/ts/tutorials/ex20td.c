static char help[] = "Performs adjoint sensitivity analysis for a van der Pol like \n\
equation with time dependent parameters using two approaches :  \n\
track  : track only local sensitivities at each adjoint step \n\
         and accumulate them in a global array \n\
global : track parameters at all timesteps together \n\
Choose one of the two at runtime by -sa_method {track,global}. \n";

/*
   Simple example to demonstrate TSAdjoint capabilities for time dependent params
   without integral cost terms using either a tracking or global method.

   Modify the Van Der Pol Eq to :
   [u1'] = [mu1(t)*u1]
   [u2'] = [mu2(t)*((1-u1^2)*u2-u1)]
   (with initial conditions & params independent)

   Define uref to be solution with initail conditions (2,-2/3), mu=(1,1e3)
   - u_ref : (1.5967,-1.02969)

   Define const function as cost = 2-norm(u - u_ref);

   Initialization for the adjoint TS :
   - dcost/dy|final_time = 2*(u-u_ref)|final_time
   - dcost/dp|final_time = 0

   The tracking method only tracks local sensitivity at each time step
   and accumulates these sensitivities in a global array. Since the structure
   of the equations being solved at each time step does not change, the jacobian
   wrt parameters is defined analogous to constant RHSJacobian for a liner
   TSSolve and the size of the jacP is independent of the number of time
   steps. Enable this mode of adjoint analysis by -sa_method track.

   The global method combines the parameters at all timesteps and tracks them
   together. Thus, the columns of the jacP matrix are filled dependent upon the
   time step. Also, the dimensions of the jacP matrix now depend upon the number
   of time steps. Enable this mode of adjoint analysis by -sa_method global.

   Since the equations here have parameters at predefined time steps, this
   example should be run with non adaptive time stepping solvers only. This
   can be ensured by -ts_adapt_type none (which is the default behavior only
   for certain TS solvers like TSCN. If using an explicit TS like TSRK,
   please be sure to add the aforementioned option to disable adaptive
   timestepping.)
*/

/*
   Include "petscts.h" so that we can use TS solvers.  Note that this file
   automatically includes:
     petscsys.h    - base PETSc routines   petscvec.h  - vectors
     petscmat.h    - matrices
     petscis.h     - index sets            petscksp.h  - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h   - preconditioners
     petscksp.h    - linear solvers        petscsnes.h - nonlinear solvers
*/
#include <petscts.h>

extern PetscErrorCode RHSFunction(TS ,PetscReal ,Vec ,Vec ,void *);
extern PetscErrorCode RHSJacobian(TS ,PetscReal ,Vec ,Mat ,Mat ,void *);
extern PetscErrorCode RHSJacobianP_track(TS ,PetscReal ,Vec ,Mat ,void *);
extern PetscErrorCode RHSJacobianP_global(TS ,PetscReal ,Vec ,Mat ,void *);
extern PetscErrorCode Monitor(TS ,PetscInt ,PetscReal ,Vec ,void *);
extern PetscErrorCode AdjointMonitor(TS ,PetscInt ,PetscReal ,Vec ,PetscInt ,Vec *, Vec *,void *);

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/

typedef struct {
 /*------------- Forward solve data structures --------------*/
  PetscInt  max_steps;     /* number of steps to run ts for */
  PetscReal final_time;    /* final time to integrate to*/
  PetscReal time_step;     /* ts integration time step */
  Vec       mu1;           /* time dependent params */
  Vec       mu2;           /* time dependent params */
  Vec       U;             /* solution vector */
  Mat       A;             /* Jacobian matrix */

  /*------------- Adjoint solve data structures --------------*/
  Mat       Jacp;          /* JacobianP matrix */
  Vec       lambda;        /* adjoint variable */
  Vec       mup;           /* adjoint variable */

  /*------------- Global accumation vecs for monitor based tracking --------------*/
  Vec       sens_mu1;      /* global sensitivity accumulation */
  Vec       sens_mu2;      /* global sensitivity accumulation */
  PetscInt  adj_idx;       /* to keep track of adjoint solve index */
} AppCtx;

typedef enum {SA_TRACK, SA_GLOBAL} SAMethod;
static const char *const SAMethods[] = {"TRACK","GLOBAL","SAMethod","SA_",0};

/* ----------------------- Explicit form of the ODE  -------------------- */

PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec U,Vec F,void *ctx)
{
  AppCtx            *user = (AppCtx*) ctx;
  PetscScalar       *f;
  PetscInt          curr_step;
  const PetscScalar *u;
  const PetscScalar *mu1;
  const PetscScalar *mu2;

  PetscFunctionBeginUser;
  PetscCall(TSGetStepNumber(ts,&curr_step));
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArrayRead(user->mu1,&mu1));
  PetscCall(VecGetArrayRead(user->mu2,&mu2));
  PetscCall(VecGetArray(F,&f));
  f[0] = mu1[curr_step]*u[1];
  f[1] = mu2[curr_step]*((1.-u[0]*u[0])*u[1]-u[0]);
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArrayRead(user->mu1,&mu1));
  PetscCall(VecRestoreArrayRead(user->mu2,&mu2));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec U,Mat A,Mat B,void *ctx)
{
  AppCtx            *user = (AppCtx*) ctx;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  PetscInt          curr_step;
  const PetscScalar *u;
  const PetscScalar *mu1;
  const PetscScalar *mu2;

  PetscFunctionBeginUser;
  PetscCall(TSGetStepNumber(ts,&curr_step));
  PetscCall(VecGetArrayRead(user->mu1,&mu1));
  PetscCall(VecGetArrayRead(user->mu2,&mu2));
  PetscCall(VecGetArrayRead(U,&u));
  J[0][0] = 0;
  J[1][0] = -mu2[curr_step]*(2.0*u[1]*u[0]+1.);
  J[0][1] = mu1[curr_step];
  J[1][1] = mu2[curr_step]*(1.0-u[0]*u[0]);
  PetscCall(MatSetValues(A,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArrayRead(user->mu1,&mu1));
  PetscCall(VecRestoreArrayRead(user->mu2,&mu2));
  PetscFunctionReturn(0);
}

/* ------------------ Jacobian wrt parameters for tracking method ------------------ */

PetscErrorCode RHSJacobianP_track(TS ts,PetscReal t,Vec U,Mat A,void *ctx)
{
  PetscInt          row[] = {0,1},col[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U,&u));
  J[0][0] = u[1];
  J[1][0] = 0;
  J[0][1] = 0;
  J[1][1] = (1.-u[0]*u[0])*u[1]-u[0];
  PetscCall(MatSetValues(A,2,row,2,col,&J[0][0],INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscFunctionReturn(0);
}

/* ------------------ Jacobian wrt parameters for global method ------------------ */

PetscErrorCode RHSJacobianP_global(TS ts,PetscReal t,Vec U,Mat A,void *ctx)
{
  PetscInt          row[] = {0,1},col[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u;
  PetscInt          curr_step;

  PetscFunctionBeginUser;
  PetscCall(TSGetStepNumber(ts,&curr_step));
  PetscCall(VecGetArrayRead(U,&u));
  J[0][0] = u[1];
  J[1][0] = 0;
  J[0][1] = 0;
  J[1][1] = (1.-u[0]*u[0])*u[1]-u[0];
  col[0] = (curr_step)*2;
  col[1] = (curr_step)*2+1;
  PetscCall(MatSetValues(A,2,row,2,col,&J[0][0],INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscFunctionReturn(0);
}

/* Dump solution to console if called */
PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec U,void *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n Solution at time %e is \n", (double)t));
  PetscCall(VecView(U,PETSC_VIEWER_STDOUT_WORLD));
  PetscFunctionReturn(0);
}

/* Customized adjoint monitor to keep track of local
   sensitivities by storing them in a global sensitivity array.
   Note : This routine is only used for the tracking method. */
PetscErrorCode AdjointMonitor(TS ts,PetscInt steps,PetscReal time,Vec u,PetscInt numcost,Vec *lambda, Vec *mu,void *ctx)
{
  AppCtx            *user = (AppCtx*) ctx;
  PetscInt          curr_step;
  PetscScalar       *sensmu1_glob;
  PetscScalar       *sensmu2_glob;
  const PetscScalar *sensmu_loc;

  PetscFunctionBeginUser;
  PetscCall(TSGetStepNumber(ts,&curr_step));
  /* Note that we skip the first call to the monitor in the adjoint
     solve since the sensitivities are already set (during
     initialization of adjoint vectors).
     We also note that each indvidial TSAdjointSolve calls the monitor
     twice, once at the step it is integrating from and once at the step
     it integrates to. Only the second call is useful for transferring
     local sensitivities to the global array. */
  if (curr_step == user->adj_idx) {
    PetscFunctionReturn(0);
  } else {
    PetscCall(VecGetArrayRead(*mu,&sensmu_loc));
    PetscCall(VecGetArray(user->sens_mu1,&sensmu1_glob));
    PetscCall(VecGetArray(user->sens_mu2,&sensmu2_glob));
    sensmu1_glob[curr_step] = sensmu_loc[0];
    sensmu2_glob[curr_step] = sensmu_loc[1];
    PetscCall(VecRestoreArray(user->sens_mu1,&sensmu1_glob));
    PetscCall(VecRestoreArray(user->sens_mu2,&sensmu2_glob));
    PetscCall(VecRestoreArrayRead(*mu,&sensmu_loc));
    PetscFunctionReturn(0);
  }
}

int main(int argc,char **argv)
{
  TS             ts;
  AppCtx         user;
  PetscScalar    *x_ptr,*y_ptr,*u_ptr;
  PetscMPIInt    size;
  PetscBool      monitor = PETSC_FALSE;
  SAMethod       sa = SA_GLOBAL;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"SA analysis options.","");{
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL));
  PetscCall(PetscOptionsEnum("-sa_method","Sensitivity analysis method (track or global)","",SAMethods,(PetscEnum)sa,(PetscEnum*)&sa,NULL));
  }
  PetscOptionsEnd();

  user.final_time = 0.1;
  user.max_steps  = 5;
  user.time_step  = user.final_time/user.max_steps;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create necessary matrix and vectors for forward solve.
     Create Jacp matrix for adjoint solve.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD,user.max_steps,&user.mu1));
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD,user.max_steps,&user.mu2));
  PetscCall(VecSet(user.mu1,1.25));
  PetscCall(VecSet(user.mu2,1.0e2));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      For tracking method : create the global sensitivity array to
      accumulate sensitivity with respect to parameters at each step.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (sa == SA_TRACK) {
    PetscCall(VecCreateSeq(PETSC_COMM_WORLD,user.max_steps,&user.sens_mu1));
    PetscCall(VecCreateSeq(PETSC_COMM_WORLD,user.max_steps,&user.sens_mu2));
  }

  PetscCall(MatCreate(PETSC_COMM_WORLD,&user.A));
  PetscCall(MatSetSizes(user.A,PETSC_DECIDE,PETSC_DECIDE,2,2));
  PetscCall(MatSetFromOptions(user.A));
  PetscCall(MatSetUp(user.A));
  PetscCall(MatCreateVecs(user.A,&user.U,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Note that the dimensions of the Jacp matrix depend upon the
      sensitivity analysis method being used !
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&user.Jacp));
  if (sa == SA_TRACK) {
    PetscCall(MatSetSizes(user.Jacp,PETSC_DECIDE,PETSC_DECIDE,2,2));
  }
  if (sa == SA_GLOBAL) {
    PetscCall(MatSetSizes(user.Jacp,PETSC_DECIDE,PETSC_DECIDE,2,user.max_steps*2));
  }
  PetscCall(MatSetFromOptions(user.Jacp));
  PetscCall(MatSetUp(user.Jacp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetEquationType(ts,TS_EQ_ODE_EXPLICIT));
  PetscCall(TSSetType(ts,TSCN));

  PetscCall(TSSetRHSFunction(ts,NULL,RHSFunction,&user));
  PetscCall(TSSetRHSJacobian(ts,user.A,user.A,RHSJacobian,&user));
  if (sa == SA_TRACK) {
    PetscCall(TSSetRHSJacobianP(ts,user.Jacp,RHSJacobianP_track,&user));
  }
  if (sa == SA_GLOBAL) {
    PetscCall(TSSetRHSJacobianP(ts,user.Jacp,RHSJacobianP_global,&user));
  }

  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetMaxTime(ts,user.final_time));
  PetscCall(TSSetTimeStep(ts,user.final_time/user.max_steps));

  if (monitor) {
    PetscCall(TSMonitorSet(ts,Monitor,&user,NULL));
  }
  if (sa == SA_TRACK) {
    PetscCall(TSAdjointMonitorSet(ts,AdjointMonitor,&user,NULL));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecGetArray(user.U,&x_ptr));
  x_ptr[0] = 2.0;
  x_ptr[1] = -2.0/3.0;
  PetscCall(VecRestoreArray(user.U,&x_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Save trajectory of solution so that TSAdjointSolve() may be used
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetSaveTrajectory(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Execute forward model and print solution.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts,user.U));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n Solution of forward TS :\n"));
  PetscCall(VecView(user.U,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n Forward TS solve successful! Adjoint run begins!\n"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here! Create adjoint vectors.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreateVecs(user.A,&user.lambda,NULL));
  PetscCall(MatCreateVecs(user.Jacp,&user.mup,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions for the adjoint vector
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecGetArray(user.U,&u_ptr));
  PetscCall(VecGetArray(user.lambda,&y_ptr));
  y_ptr[0] = 2*(u_ptr[0] - 1.5967);
  y_ptr[1] = 2*(u_ptr[1] - -(1.02969));
  PetscCall(VecRestoreArray(user.lambda,&y_ptr));
  PetscCall(VecRestoreArray(user.U,&y_ptr));
  PetscCall(VecSet(user.mup,0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set number of cost functions.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetCostGradients(ts,1,&user.lambda,&user.mup));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     The adjoint vector mup has to be reset for each adjoint step when
     using the tracking method as we want to treat the parameters at each
     time step one at a time and prevent accumulation of the sensitivities
     from parameters at previous time steps.
     This is not necessary for the global method as each time dependent
     parameter is treated as an independent parameter.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (sa == SA_TRACK) {
    for (user.adj_idx=user.max_steps; user.adj_idx>0; user.adj_idx--) {
      PetscCall(VecSet(user.mup,0));
      PetscCall(TSAdjointSetSteps(ts, 1));
      PetscCall(TSAdjointSolve(ts));
    }
  }
  if (sa == SA_GLOBAL) {
    PetscCall(TSAdjointSolve(ts));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Dispaly adjoint sensitivities wrt parameters and initial conditions
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (sa == SA_TRACK) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt  mu1: d[cost]/d[mu1]\n"));
    PetscCall(VecView(user.sens_mu1,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt  mu2: d[cost]/d[mu2]\n"));
    PetscCall(VecView(user.sens_mu2,PETSC_VIEWER_STDOUT_WORLD));
  }

  if (sa == SA_GLOBAL) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt  params: d[cost]/d[p], where p refers to \nthe interlaced vector made by combining mu1,mu2\n"));
    PetscCall(VecView(user.mup,PETSC_VIEWER_STDOUT_WORLD));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt initial conditions: d[cost]/d[u(t=0)]\n"));
  PetscCall(VecView(user.lambda,PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space!
     All PETSc objects should be destroyed when they are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&user.A));
  PetscCall(MatDestroy(&user.Jacp));
  PetscCall(VecDestroy(&user.U));
  PetscCall(VecDestroy(&user.lambda));
  PetscCall(VecDestroy(&user.mup));
  PetscCall(VecDestroy(&user.mu1));
  PetscCall(VecDestroy(&user.mu2));
  if (sa == SA_TRACK) {
    PetscCall(VecDestroy(&user.sens_mu1));
    PetscCall(VecDestroy(&user.sens_mu2));
  }
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return(0);
}

/*TEST

  test:
    requires: !complex
    suffix : track
    args : -sa_method track

  test:
    requires: !complex
    suffix : global
    args : -sa_method global

TEST*/
