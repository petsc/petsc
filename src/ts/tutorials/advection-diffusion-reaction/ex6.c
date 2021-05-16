
static char help[] ="Model Equations for Advection \n";

/*
    Modified from ex3.c
    Page 9, Section 1.2 Model Equations for Advection-Diffusion

          u_t + a u_x = 0, 0<= x <= 1.0

   The initial conditions used here different from the book.

   Example:
     ./ex6 -ts_monitor -ts_view_solution -ts_max_steps 100 -ts_monitor_solution draw -draw_pause .1
     ./ex6 -ts_monitor -ts_max_steps 100 -ts_monitor_lg_error -draw_pause .1
*/

#include <petscts.h>
#include <petscdm.h>
#include <petscdmda.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/
typedef struct {
  PetscReal a;   /* advection strength */
} AppCtx;

/* User-defined routines */
extern PetscErrorCode InitialConditions(TS,Vec,AppCtx*);
extern PetscErrorCode Solution(TS,PetscReal,Vec,AppCtx*);
extern PetscErrorCode IFunction_LaxFriedrichs(TS,PetscReal,Vec,Vec,Vec,void*);
extern PetscErrorCode IFunction_LaxWendroff(TS,PetscReal,Vec,Vec,Vec,void*);

int main(int argc,char **argv)
{
  AppCtx         appctx;                 /* user-defined application context */
  TS             ts;                     /* timestepping context */
  Vec            U;                      /* approximate solution vector */
  PetscErrorCode ierr;
  PetscReal      dt;
  DM             da;
  PetscInt       M;
  PetscMPIInt    rank;
  PetscBool      useLaxWendroff = PETSC_TRUE;

  /* Initialize program and set problem parameters */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  appctx.a  = -1.0;
  ierr      = PetscOptionsGetReal(NULL,NULL,"-a",&appctx.a,NULL);CHKERRQ(ierr);

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC, 60, 1, 1,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);

  /* Create vector data structures for approximate and exact solutions */
  ierr = DMCreateGlobalVector(da,&U);CHKERRQ(ierr);

  /* Create timestepping solver context */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);

  /* Function evaluation */
  ierr = PetscOptionsGetBool(NULL,NULL,"-useLaxWendroff",&useLaxWendroff,NULL);CHKERRQ(ierr);
  if (useLaxWendroff) {
    if (!rank) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"... Use Lax-Wendroff finite volume\n");CHKERRQ(ierr);
    }
    ierr = TSSetIFunction(ts,NULL,IFunction_LaxWendroff,&appctx);CHKERRQ(ierr);
  } else {
    if (!rank) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"... Use Lax-LaxFriedrichs finite difference\n");CHKERRQ(ierr);
    }
    ierr = TSSetIFunction(ts,NULL,IFunction_LaxFriedrichs,&appctx);CHKERRQ(ierr);
  }

  /* Customize timestepping solver */
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&M,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  dt = 1.0/(PetscAbsReal(appctx.a)*M);
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,100);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,100.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* Evaluate initial conditions */
  ierr = InitialConditions(ts,U,&appctx);CHKERRQ(ierr);

  /* For testing accuracy of TS with already known solution, e.g., '-ts_monitor_lg_error' */
  ierr = TSSetSolutionFunction(ts,(PetscErrorCode (*)(TS,PetscReal,Vec,void*))Solution,&appctx);CHKERRQ(ierr);

  /* Run the timestepping solver */
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  /* Free work space */
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
/* --------------------------------------------------------------------- */
/*
   InitialConditions - Computes the solution at the initial time.

   Input Parameter:
   u - uninitialized solution vector (global)
   appctx - user-defined application context

   Output Parameter:
   u - vector with solution at initial time (global)
*/
PetscErrorCode InitialConditions(TS ts,Vec U,AppCtx *appctx)
{
  PetscScalar    *u;
  PetscErrorCode ierr;
  PetscInt       i,mstart,mend,um,M;
  DM             da;
  PetscReal      h;

  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&mstart,0,0,&um,0,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&M,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  h    = 1.0/M;
  mend = mstart + um;
  /*
    Get a pointer to vector data.
    - For default PETSc vectors, VecGetArray() returns a pointer to
      the data array.  Otherwise, the routine is implementation dependent.
    - You MUST call VecRestoreArray() when you no longer need access to
      the array.
    - Note that the Fortran interface to VecGetArray() differs from the
      C version.  See the users manual for details.
  */
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);

  /*
     We initialize the solution array by simply writing the solution
     directly into the array locations.  Alternatively, we could use
     VecSetValues() or VecSetValuesLocal().
  */
  for (i=mstart; i<mend; i++) u[i] = PetscSinReal(PETSC_PI*i*6.*h) + 3.*PetscSinReal(PETSC_PI*i*2.*h);

  /* Restore vector */
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  return 0;
}
/* --------------------------------------------------------------------- */
/*
   Solution - Computes the exact solution at a given time

   Input Parameters:
   t - current time
   solution - vector in which exact solution will be computed
   appctx - user-defined application context

   Output Parameter:
   solution - vector with the newly computed exact solution
              u(x,t) = sin(6*PI*(x - a*t)) + 3 * sin(2*PI*(x - a*t))
*/
PetscErrorCode Solution(TS ts,PetscReal t,Vec U,AppCtx *appctx)
{
  PetscScalar    *u;
  PetscReal      a=appctx->a,h,PI6,PI2;
  PetscErrorCode ierr;
  PetscInt       i,mstart,mend,um,M;
  DM             da;

  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&mstart,0,0,&um,0,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&M,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  h    = 1.0/M;
  mend = mstart + um;

  /* Get a pointer to vector data. */
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);

  /* u[i] = sin(6*PI*(x[i] - a*t)) + 3 * sin(2*PI*(x[i] - a*t)) */
  PI6 = PETSC_PI*6.;
  PI2 = PETSC_PI*2.;
  for (i=mstart; i<mend; i++) {
    u[i] = PetscSinReal(PI6*(i*h - a*t)) + 3.*PetscSinReal(PI2*(i*h - a*t));
  }

  /* Restore vector */
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  return 0;
}

/* --------------------------------------------------------------------- */
/*
 Use Lax-Friedrichs method to evaluate F(u,t) = du/dt + a *  du/dx

 See https://en.wikipedia.org/wiki/Lax%E2%80%93Friedrichs_method
 */
PetscErrorCode IFunction_LaxFriedrichs(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,void* ctx)
{
  PetscErrorCode ierr;
  AppCtx         *appctx=(AppCtx*)ctx;
  PetscInt       mstart,mend,M,i,um;
  DM             da;
  Vec            Uold,localUold;
  PetscScalar    *uarray,*f,*uoldarray,h,uave,c;
  PetscReal      dt;

  PetscFunctionBegin;
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&Uold);CHKERRQ(ierr);

  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0,&M,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&mstart,0,0,&um,0,0);CHKERRQ(ierr);
  h    = 1.0/M;
  mend = mstart + um;
  /* printf(" mstart %d, um %d\n",mstart,um); */

  ierr = DMGetLocalVector(da,&localUold);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,Uold,INSERT_VALUES,localUold);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,Uold,INSERT_VALUES,localUold);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArrayRead(da,U,&uarray);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,localUold,&uoldarray);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  /* advection */
  c = appctx->a*dt/h; /* Courant-Friedrichs-Lewy number (CFL number) */

  for (i=mstart; i<mend; i++) {
    uave = 0.5*(uoldarray[i-1] + uoldarray[i+1]);
    f[i] = uarray[i] - uave + c*0.5*(uoldarray[i+1] - uoldarray[i-1]);
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArrayRead(da,U,&uarray);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,localUold,&uoldarray);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localUold);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 Use Lax-Wendroff method to evaluate F(u,t) = du/dt + a *  du/dx
*/
PetscErrorCode IFunction_LaxWendroff(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,void* ctx)
{
  PetscErrorCode ierr;
  AppCtx         *appctx=(AppCtx*)ctx;
  PetscInt       mstart,mend,M,i,um;
  DM             da;
  Vec            Uold,localUold;
  PetscScalar    *uarray,*f,*uoldarray,h,RFlux,LFlux,lambda;
  PetscReal      dt,a;

  PetscFunctionBegin;
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&Uold);CHKERRQ(ierr);

  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0,&M,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&mstart,0,0,&um,0,0);CHKERRQ(ierr);
  h    = 1.0/M;
  mend = mstart + um;
  /* printf(" mstart %d, um %d\n",mstart,um); */

  ierr = DMGetLocalVector(da,&localUold);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,Uold,INSERT_VALUES,localUold);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,Uold,INSERT_VALUES,localUold);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArrayRead(da,U,&uarray);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,localUold,&uoldarray);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  /* advection -- finite volume (appctx->a < 0 -- can be relaxed?) */
  lambda = dt/h;
  a = appctx->a;

  for (i=mstart; i<mend; i++) {
    RFlux = 0.5 * a * (uoldarray[i+1] + uoldarray[i]) - a*a*0.5*lambda * (uoldarray[i+1] - uoldarray[i]);
    LFlux = 0.5 * a * (uoldarray[i-1] + uoldarray[i]) - a*a*0.5*lambda * (uoldarray[i] - uoldarray[i-1]);
    f[i]  = uarray[i] - uoldarray[i] + lambda * (RFlux - LFlux);
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArrayRead(da,U,&uarray);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,localUold,&uoldarray);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localUold);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   test:
      args: -ts_max_steps 10 -ts_monitor

   test:
      suffix: 2
      nsize: 3
      args: -ts_max_steps 10 -ts_monitor
      output_file: output/ex6_1.out

   test:
      suffix: 3
      args: -ts_max_steps 10 -ts_monitor -useLaxWendroff false

   test:
      suffix: 4
      nsize: 3
      args: -ts_max_steps 10 -ts_monitor -useLaxWendroff false
      output_file: output/ex6_3.out

TEST*/
