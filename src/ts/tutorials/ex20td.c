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
  PetscErrorCode    ierr;
  AppCtx            *user = (AppCtx*) ctx;
  PetscScalar       *f;
  PetscInt          curr_step;
  const PetscScalar *u;
  const PetscScalar *mu1;
  const PetscScalar *mu2;

  PetscFunctionBeginUser;
  ierr = TSGetStepNumber(ts,&curr_step);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->mu1,&mu1);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->mu2,&mu2);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = mu1[curr_step]*u[1];
  f[1] = mu2[curr_step]*((1.-u[0]*u[0])*u[1]-u[0]);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(user->mu1,&mu1);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(user->mu2,&mu2);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec U,Mat A,Mat B,void *ctx)
{
  PetscErrorCode    ierr;
  AppCtx            *user = (AppCtx*) ctx;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  PetscInt          curr_step;
  const PetscScalar *u;
  const PetscScalar *mu1;
  const PetscScalar *mu2;

  PetscFunctionBeginUser;
  ierr = TSGetStepNumber(ts,&curr_step);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->mu1,&mu1);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->mu2,&mu2);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  J[0][0] = 0;
  J[1][0] = -mu2[curr_step]*(2.0*u[1]*u[0]+1.);
  J[0][1] = mu1[curr_step];
  J[1][1] = mu2[curr_step]*(1.0-u[0]*u[0]);
  ierr = MatSetValues(A,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(user->mu1,&mu1);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(user->mu2,&mu2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------ Jacobian wrt parameters for tracking method ------------------ */

PetscErrorCode RHSJacobianP_track(TS ts,PetscReal t,Vec U,Mat A,void *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          row[] = {0,1},col[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  J[0][0] = u[1];
  J[1][0] = 0;
  J[0][1] = 0;
  J[1][1] = (1.-u[0]*u[0])*u[1]-u[0];
  ierr = MatSetValues(A,2,row,2,col,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------ Jacobian wrt parameters for global method ------------------ */

PetscErrorCode RHSJacobianP_global(TS ts,PetscReal t,Vec U,Mat A,void *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          row[] = {0,1},col[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u;
  PetscInt          curr_step;

  PetscFunctionBeginUser;
  ierr = TSGetStepNumber(ts,&curr_step);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  J[0][0] = u[1];
  J[1][0] = 0;
  J[0][1] = 0;
  J[1][1] = (1.-u[0]*u[0])*u[1]-u[0];
  col[0] = (curr_step)*2;
  col[1] = (curr_step)*2+1;
  ierr = MatSetValues(A,2,row,2,col,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Dump solution to console if called */
PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec U,void *ctx)
{
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n Solution at time %e is \n", t);CHKERRQ(ierr);
  ierr = VecView(U,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Customized adjoint monitor to keep track of local
   sensitivities by storing them in a global sensitivity array.
   Note : This routine is only used for the tracking method. */
PetscErrorCode AdjointMonitor(TS ts,PetscInt steps,PetscReal time,Vec u,PetscInt numcost,Vec *lambda, Vec *mu,void *ctx)
{
  PetscErrorCode    ierr;
  AppCtx            *user = (AppCtx*) ctx;
  PetscInt          curr_step;
  PetscScalar       *sensmu1_glob;
  PetscScalar       *sensmu2_glob;
  const PetscScalar *sensmu_loc;

  PetscFunctionBeginUser;
  ierr = TSGetStepNumber(ts,&curr_step);CHKERRQ(ierr);
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
    ierr = VecGetArrayRead(*mu,&sensmu_loc);CHKERRQ(ierr);
    ierr = VecGetArray(user->sens_mu1,&sensmu1_glob);CHKERRQ(ierr);
    ierr = VecGetArray(user->sens_mu2,&sensmu2_glob);CHKERRQ(ierr);
    sensmu1_glob[curr_step] = sensmu_loc[0];
    sensmu2_glob[curr_step] = sensmu_loc[1];
    ierr = VecRestoreArray(user->sens_mu1,&sensmu1_glob);CHKERRQ(ierr);
    ierr = VecRestoreArray(user->sens_mu2,&sensmu2_glob);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(*mu,&sensmu_loc);CHKERRQ(ierr);
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"SA analysis options.","");CHKERRQ(ierr);{
  ierr = PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-sa_method","Sensitivity analysis method (track or global)","",SAMethods,(PetscEnum)sa,(PetscEnum*)&sa,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  user.final_time = 0.1;
  user.max_steps  = 5;
  user.time_step  = user.final_time/user.max_steps;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create necessary matrix and vectors for forward solve.
     Create Jacp matrix for adjoint solve.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecCreateSeq(PETSC_COMM_WORLD,user.max_steps,&user.mu1);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_WORLD,user.max_steps,&user.mu2);CHKERRQ(ierr);
  ierr = VecSet(user.mu1,1.25);CHKERRQ(ierr);
  ierr = VecSet(user.mu2,1.0e2);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      For tracking method : create the global sensitivity array to
      accumulate sensitivity with respect to parameters at each step.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (sa == SA_TRACK) {
    ierr = VecCreateSeq(PETSC_COMM_WORLD,user.max_steps,&user.sens_mu1);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_WORLD,user.max_steps,&user.sens_mu2);CHKERRQ(ierr);
  }

  ierr = MatCreate(PETSC_COMM_WORLD,&user.A);CHKERRQ(ierr);
  ierr = MatSetSizes(user.A,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user.A);CHKERRQ(ierr);
  ierr = MatSetUp(user.A);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.A,&user.U,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Note that the dimensions of the Jacp matrix depend upon the
      sensitivity analysis method being used !
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&user.Jacp);CHKERRQ(ierr);
  if (sa == SA_TRACK) {
    ierr = MatSetSizes(user.Jacp,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRQ(ierr);
  }
  if (sa == SA_GLOBAL) {
    ierr = MatSetSizes(user.Jacp,PETSC_DECIDE,PETSC_DECIDE,2,user.max_steps*2);CHKERRQ(ierr);
  }
  ierr = MatSetFromOptions(user.Jacp);CHKERRQ(ierr);
  ierr = MatSetUp(user.Jacp);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetEquationType(ts,TS_EQ_ODE_EXPLICIT);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);

  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,&user);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,user.A,user.A,RHSJacobian,&user);CHKERRQ(ierr);
  if (sa == SA_TRACK) {
    ierr = TSSetRHSJacobianP(ts,user.Jacp,RHSJacobianP_track,&user);CHKERRQ(ierr);
  }
  if (sa == SA_GLOBAL) {
    ierr = TSSetRHSJacobianP(ts,user.Jacp,RHSJacobianP_global,&user);CHKERRQ(ierr);
  }

  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,user.final_time);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,user.final_time/user.max_steps);CHKERRQ(ierr);

  if (monitor) {
    ierr = TSMonitorSet(ts,Monitor,&user,NULL);CHKERRQ(ierr);
  }
  if (sa == SA_TRACK) {
    ierr = TSAdjointMonitorSet(ts,AdjointMonitor,&user,NULL);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecGetArray(user.U,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2.0;
  x_ptr[1] = -2.0/3.0;
  ierr = VecRestoreArray(user.U,&x_ptr);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Save trajectory of solution so that TSAdjointSolve() may be used
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Execute forward model and print solution.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,user.U);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n Solution of forward TS :\n");CHKERRQ(ierr);
  ierr = VecView(user.U,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n Forward TS solve successfull! Adjoint run begins!\n");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here! Create adjoint vectors.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreateVecs(user.A,&user.lambda,NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.Jacp,&user.mup,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions for the adjoint vector
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecGetArray(user.U,&u_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(user.lambda,&y_ptr);CHKERRQ(ierr);
  y_ptr[0] = 2*(u_ptr[0] - 1.5967);
  y_ptr[1] = 2*(u_ptr[1] - -(1.02969));
  ierr = VecRestoreArray(user.lambda,&y_ptr);CHKERRQ(ierr);
  ierr = VecRestoreArray(user.U,&y_ptr);CHKERRQ(ierr);
  ierr = VecSet(user.mup,0);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set number of cost functions.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetCostGradients(ts,1,&user.lambda,&user.mup);CHKERRQ(ierr);

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
      ierr = VecSet(user.mup,0);CHKERRQ(ierr);
      ierr = TSAdjointSetSteps(ts, 1);CHKERRQ(ierr);
      ierr = TSAdjointSolve(ts);CHKERRQ(ierr);
    }
  }
  if (sa == SA_GLOBAL) {
    ierr = TSAdjointSolve(ts);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Dispaly adjoint sensitivities wrt parameters and initial conditions
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (sa == SA_TRACK) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt  mu1: d[cost]/d[mu1]\n");CHKERRQ(ierr);
    ierr = VecView(user.sens_mu1,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt  mu2: d[cost]/d[mu2]\n");CHKERRQ(ierr);
    ierr = VecView(user.sens_mu2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  if (sa == SA_GLOBAL) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt  params: d[cost]/d[p], where p refers to \n\
                    the interlaced vector made by combining mu1,mu2\n");CHKERRQ(ierr);
    ierr = VecView(user.mup,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt initial conditions: d[cost]/d[u(t=0)]\n");CHKERRQ(ierr);
  ierr = VecView(user.lambda,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space!
     All PETSc objects should be destroyed when they are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&user.A);CHKERRQ(ierr);
  ierr = MatDestroy(&user.Jacp);CHKERRQ(ierr);
  ierr = VecDestroy(&user.U);CHKERRQ(ierr);
  ierr = VecDestroy(&user.lambda);CHKERRQ(ierr);
  ierr = VecDestroy(&user.mup);CHKERRQ(ierr);
  ierr = VecDestroy(&user.mu1);CHKERRQ(ierr);
  ierr = VecDestroy(&user.mu2);CHKERRQ(ierr);
  if (sa == SA_TRACK) {
    ierr = VecDestroy(&user.sens_mu1);CHKERRQ(ierr);
    ierr = VecDestroy(&user.sens_mu2);CHKERRQ(ierr);
  }
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
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

