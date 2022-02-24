static char help[] = "Performs adjoint sensitivity analysis for a van der Pol like \n\
equation with time dependent parameters using two approaches :  \n\
track  : track only local sensitivities at each adjoint step \n\
         and accumulate them in a global array \n\
global : track parameters at all timesteps together \n\
Choose one of the two at runtime by -sa_method {track,global}. \n";

/*
  Concepts: TS^adjoint for time dependent parameters
  Concepts: TS^Customized adjoint monitor based sensitivity tracking
  Concepts: TS^All at once approach to sensitivity tracking
  Processors: 1
*/

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
  CHKERRQ(TSGetStepNumber(ts,&curr_step));
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(user->mu1,&mu1));
  CHKERRQ(VecGetArrayRead(user->mu2,&mu2));
  CHKERRQ(VecGetArray(F,&f));
  f[0] = mu1[curr_step]*u[1];
  f[1] = mu2[curr_step]*((1.-u[0]*u[0])*u[1]-u[0]);
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(user->mu1,&mu1));
  CHKERRQ(VecRestoreArrayRead(user->mu2,&mu2));
  CHKERRQ(VecRestoreArray(F,&f));
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
  CHKERRQ(TSGetStepNumber(ts,&curr_step));
  CHKERRQ(VecGetArrayRead(user->mu1,&mu1));
  CHKERRQ(VecGetArrayRead(user->mu2,&mu2));
  CHKERRQ(VecGetArrayRead(U,&u));
  J[0][0] = 0;
  J[1][0] = -mu2[curr_step]*(2.0*u[1]*u[0]+1.);
  J[0][1] = mu1[curr_step];
  J[1][1] = mu2[curr_step]*(1.0-u[0]*u[0]);
  CHKERRQ(MatSetValues(A,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(user->mu1,&mu1));
  CHKERRQ(VecRestoreArrayRead(user->mu2,&mu2));
  PetscFunctionReturn(0);
}

/* ------------------ Jacobian wrt parameters for tracking method ------------------ */

PetscErrorCode RHSJacobianP_track(TS ts,PetscReal t,Vec U,Mat A,void *ctx)
{
  PetscInt          row[] = {0,1},col[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(U,&u));
  J[0][0] = u[1];
  J[1][0] = 0;
  J[0][1] = 0;
  J[1][1] = (1.-u[0]*u[0])*u[1]-u[0];
  CHKERRQ(MatSetValues(A,2,row,2,col,&J[0][0],INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(VecRestoreArrayRead(U,&u));
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
  CHKERRQ(TSGetStepNumber(ts,&curr_step));
  CHKERRQ(VecGetArrayRead(U,&u));
  J[0][0] = u[1];
  J[1][0] = 0;
  J[0][1] = 0;
  J[1][1] = (1.-u[0]*u[0])*u[1]-u[0];
  col[0] = (curr_step)*2;
  col[1] = (curr_step)*2+1;
  CHKERRQ(MatSetValues(A,2,row,2,col,&J[0][0],INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(VecRestoreArrayRead(U,&u));
  PetscFunctionReturn(0);
}

/* Dump solution to console if called */
PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec U,void *ctx)
{
  PetscFunctionBeginUser;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n Solution at time %e is \n", t));
  CHKERRQ(VecView(U,PETSC_VIEWER_STDOUT_WORLD));
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
  CHKERRQ(TSGetStepNumber(ts,&curr_step));
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
    CHKERRQ(VecGetArrayRead(*mu,&sensmu_loc));
    CHKERRQ(VecGetArray(user->sens_mu1,&sensmu1_glob));
    CHKERRQ(VecGetArray(user->sens_mu2,&sensmu2_glob));
    sensmu1_glob[curr_step] = sensmu_loc[0];
    sensmu2_glob[curr_step] = sensmu_loc[1];
    CHKERRQ(VecRestoreArray(user->sens_mu1,&sensmu1_glob));
    CHKERRQ(VecRestoreArray(user->sens_mu2,&sensmu2_glob));
    CHKERRQ(VecRestoreArrayRead(*mu,&sensmu_loc));
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
  PetscErrorCode ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"SA analysis options.","");CHKERRQ(ierr);{
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL));
  CHKERRQ(PetscOptionsEnum("-sa_method","Sensitivity analysis method (track or global)","",SAMethods,(PetscEnum)sa,(PetscEnum*)&sa,NULL));
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  user.final_time = 0.1;
  user.max_steps  = 5;
  user.time_step  = user.final_time/user.max_steps;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create necessary matrix and vectors for forward solve.
     Create Jacp matrix for adjoint solve.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecCreateSeq(PETSC_COMM_WORLD,user.max_steps,&user.mu1));
  CHKERRQ(VecCreateSeq(PETSC_COMM_WORLD,user.max_steps,&user.mu2));
  CHKERRQ(VecSet(user.mu1,1.25));
  CHKERRQ(VecSet(user.mu2,1.0e2));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      For tracking method : create the global sensitivity array to
      accumulate sensitivity with respect to parameters at each step.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (sa == SA_TRACK) {
    CHKERRQ(VecCreateSeq(PETSC_COMM_WORLD,user.max_steps,&user.sens_mu1));
    CHKERRQ(VecCreateSeq(PETSC_COMM_WORLD,user.max_steps,&user.sens_mu2));
  }

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user.A));
  CHKERRQ(MatSetSizes(user.A,PETSC_DECIDE,PETSC_DECIDE,2,2));
  CHKERRQ(MatSetFromOptions(user.A));
  CHKERRQ(MatSetUp(user.A));
  CHKERRQ(MatCreateVecs(user.A,&user.U,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Note that the dimensions of the Jacp matrix depend upon the
      sensitivity analysis method being used !
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user.Jacp));
  if (sa == SA_TRACK) {
    CHKERRQ(MatSetSizes(user.Jacp,PETSC_DECIDE,PETSC_DECIDE,2,2));
  }
  if (sa == SA_GLOBAL) {
    CHKERRQ(MatSetSizes(user.Jacp,PETSC_DECIDE,PETSC_DECIDE,2,user.max_steps*2));
  }
  CHKERRQ(MatSetFromOptions(user.Jacp));
  CHKERRQ(MatSetUp(user.Jacp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetEquationType(ts,TS_EQ_ODE_EXPLICIT));
  CHKERRQ(TSSetType(ts,TSCN));

  CHKERRQ(TSSetRHSFunction(ts,NULL,RHSFunction,&user));
  CHKERRQ(TSSetRHSJacobian(ts,user.A,user.A,RHSJacobian,&user));
  if (sa == SA_TRACK) {
    CHKERRQ(TSSetRHSJacobianP(ts,user.Jacp,RHSJacobianP_track,&user));
  }
  if (sa == SA_GLOBAL) {
    CHKERRQ(TSSetRHSJacobianP(ts,user.Jacp,RHSJacobianP_global,&user));
  }

  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  CHKERRQ(TSSetMaxTime(ts,user.final_time));
  CHKERRQ(TSSetTimeStep(ts,user.final_time/user.max_steps));

  if (monitor) {
    CHKERRQ(TSMonitorSet(ts,Monitor,&user,NULL));
  }
  if (sa == SA_TRACK) {
    CHKERRQ(TSAdjointMonitorSet(ts,AdjointMonitor,&user,NULL));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecGetArray(user.U,&x_ptr));
  x_ptr[0] = 2.0;
  x_ptr[1] = -2.0/3.0;
  CHKERRQ(VecRestoreArray(user.U,&x_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Save trajectory of solution so that TSAdjointSolve() may be used
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetSaveTrajectory(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Execute forward model and print solution.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,user.U));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n Solution of forward TS :\n"));
  CHKERRQ(VecView(user.U,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n Forward TS solve successfull! Adjoint run begins!\n"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here! Create adjoint vectors.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreateVecs(user.A,&user.lambda,NULL));
  CHKERRQ(MatCreateVecs(user.Jacp,&user.mup,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions for the adjoint vector
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecGetArray(user.U,&u_ptr));
  CHKERRQ(VecGetArray(user.lambda,&y_ptr));
  y_ptr[0] = 2*(u_ptr[0] - 1.5967);
  y_ptr[1] = 2*(u_ptr[1] - -(1.02969));
  CHKERRQ(VecRestoreArray(user.lambda,&y_ptr));
  CHKERRQ(VecRestoreArray(user.U,&y_ptr));
  CHKERRQ(VecSet(user.mup,0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set number of cost functions.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetCostGradients(ts,1,&user.lambda,&user.mup));

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
      CHKERRQ(VecSet(user.mup,0));
      CHKERRQ(TSAdjointSetSteps(ts, 1));
      CHKERRQ(TSAdjointSolve(ts));
    }
  }
  if (sa == SA_GLOBAL) {
    CHKERRQ(TSAdjointSolve(ts));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Dispaly adjoint sensitivities wrt parameters and initial conditions
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (sa == SA_TRACK) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt  mu1: d[cost]/d[mu1]\n"));
    CHKERRQ(VecView(user.sens_mu1,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt  mu2: d[cost]/d[mu2]\n"));
    CHKERRQ(VecView(user.sens_mu2,PETSC_VIEWER_STDOUT_WORLD));
  }

  if (sa == SA_GLOBAL) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt  params: d[cost]/d[p], where p refers to \n\
                    the interlaced vector made by combining mu1,mu2\n");CHKERRQ(ierr);
    CHKERRQ(VecView(user.mup,PETSC_VIEWER_STDOUT_WORLD));
  }

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt initial conditions: d[cost]/d[u(t=0)]\n"));
  CHKERRQ(VecView(user.lambda,PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space!
     All PETSc objects should be destroyed when they are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatDestroy(&user.A));
  CHKERRQ(MatDestroy(&user.Jacp));
  CHKERRQ(VecDestroy(&user.U));
  CHKERRQ(VecDestroy(&user.lambda));
  CHKERRQ(VecDestroy(&user.mup));
  CHKERRQ(VecDestroy(&user.mu1));
  CHKERRQ(VecDestroy(&user.mu2));
  if (sa == SA_TRACK) {
    CHKERRQ(VecDestroy(&user.sens_mu1));
    CHKERRQ(VecDestroy(&user.sens_mu2));
  }
  CHKERRQ(TSDestroy(&ts));
  ierr = PetscFinalize();
  return(ierr);
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
