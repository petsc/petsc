
static char help[] ="Solves a time-dependent linear PDE with discontinuous right hand side.\n";

/* ------------------------------------------------------------------------

   This program solves the one-dimensional quench front problem modeling a cooled
   liquid rising on a hot metal rod
       u_t = u_xx + g(u),
   with
       g(u) = -Au if u <= u_c,
            =   0 if u >  u_c
   on the domain 0 <= x <= 1, with the boundary conditions
       u(t,0) = 0, u_x(t,1) = 0,
   and the initial condition
       u(0,x) = 0              if 0 <= x <= 0.1,
              = (x - 0.1)/0.15 if 0.1 < x < 0.25
              = 1              if 0.25 <= x <= 1
   We discretize the right-hand side using finite differences with
   uniform grid spacing h:
       u_xx = (u_{i+1} - 2u_{i} + u_{i-1})/(h^2)

Reference: L. Shampine and S. Thompson, "Event Location for Ordinary Differential Equations",
           http://www.radford.edu/~thompson/webddes/eventsweb.pdf
  ------------------------------------------------------------------------- */

#include <petscdmda.h>
#include <petscts.h>
/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/
typedef struct {
  PetscReal A;
  PetscReal uc;
  PetscInt  *sw;
} AppCtx;

PetscErrorCode InitialConditions(Vec U,DM da,AppCtx *app)
{
  PetscErrorCode ierr;
  Vec            xcoord;
  PetscScalar    *x,*u;
  PetscInt       lsize,M,xs,xm,i;

  PetscFunctionBegin;
  CHKERRQ(DMGetCoordinates(da,&xcoord));
  CHKERRQ(DMDAVecGetArrayRead(da,xcoord,&x));

  CHKERRQ(VecGetLocalSize(U,&lsize));
  CHKERRQ(PetscMalloc1(lsize,&app->sw));

  CHKERRQ(DMDAVecGetArray(da,U,&u));

  CHKERRQ(DMDAGetInfo(da,0,&M,0,0,0,0,0,0,0,0,0,0,0));
  CHKERRQ(DMDAGetCorners(da,&xs,0,0,&xm,0,0));

  for (i=xs; i<xs+xm;i++) {
    if (x[i] <= 0.1) u[i] = 0.;
    else if (x[i] > 0.1 && x[i] < 0.25) u[i] = (x[i] - 0.1)/0.15;
    else u[i] = 1.0;

    app->sw[i-xs] = 1;
  }
  CHKERRQ(DMDAVecRestoreArray(da,U,&u));
  CHKERRQ(DMDAVecRestoreArrayRead(da,xcoord,&x));
  PetscFunctionReturn(0);
}

PetscErrorCode EventFunction(TS ts,PetscReal t,Vec U,PetscScalar *fvalue,void *ctx)
{
  AppCtx            *app=(AppCtx*)ctx;
  PetscErrorCode    ierr;
  const PetscScalar *u;
  PetscInt          i,lsize;

  PetscFunctionBegin;
  CHKERRQ(VecGetLocalSize(U,&lsize));
  CHKERRQ(VecGetArrayRead(U,&u));
  for (i=0; i < lsize;i++) fvalue[i] = u[i] - app->uc;
  CHKERRQ(VecRestoreArrayRead(U,&u));
  PetscFunctionReturn(0);
}

PetscErrorCode PostEventFunction(TS ts,PetscInt nevents_zero,PetscInt events_zero[],PetscReal t,Vec U,PetscBool forwardsolve,void* ctx)
{
  AppCtx         *app=(AppCtx*)ctx;
  PetscInt       i,idx;

  PetscFunctionBegin;
  for (i=0; i < nevents_zero; i ++) {
    idx = events_zero[i];
    app->sw[idx] = 0;
  }
  PetscFunctionReturn(0);
}

/*
     Defines the ODE passed to the ODE solver
*/
static PetscErrorCode IFunction(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,void *ctx)
{
  AppCtx            *app=(AppCtx*)ctx;
  PetscErrorCode    ierr;
  PetscScalar       *f;
  const PetscScalar *u,*udot;
  DM                da;
  PetscInt          M,xs,xm,i;
  PetscReal         h,h2;
  Vec               Ulocal;

  PetscFunctionBegin;
  CHKERRQ(TSGetDM(ts,&da));

  CHKERRQ(DMDAGetInfo(da,0,&M,0,0,0,0,0,0,0,0,0,0,0));
  CHKERRQ(DMDAGetCorners(da,&xs,0,0,&xm,0,0));

  CHKERRQ(DMGetLocalVector(da,&Ulocal));
  CHKERRQ(DMGlobalToLocalBegin(da,U,INSERT_VALUES,Ulocal));
  CHKERRQ(DMGlobalToLocalEnd(da,U,INSERT_VALUES,Ulocal));

  h = 1.0/(M-1); h2 = h*h;
  CHKERRQ(DMDAVecGetArrayRead(da,Udot,&udot));
  CHKERRQ(DMDAVecGetArrayRead(da,Ulocal,&u));
  CHKERRQ(DMDAVecGetArray(da,F,&f));

  for (i=xs; i<xs+xm;i++) {
    if (i == 0) {
      f[i] = u[i];
    } else if (i == M - 1) {
      f[i] = (u[i] - u[i-1])/h;
    } else {
      f[i] = (u[i+1] - 2*u[i] + u[i-1])/h2 + app->sw[i-xs]*(-app->A*u[i]) - udot[i];
    }
  }

  CHKERRQ(DMDAVecRestoreArrayRead(da,Udot,&udot));
  CHKERRQ(DMDAVecRestoreArrayRead(da,Ulocal,&u));
  CHKERRQ(DMDAVecRestoreArray(da,F,&f));
  CHKERRQ(DMRestoreLocalVector(da,&Ulocal));

  PetscFunctionReturn(0);
}

/*
     Defines the Jacobian of the ODE passed to the ODE solver. See TSSetIJacobian() for the meaning of a and the Jacobian.
*/
static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat B,void *ctx)
{
  AppCtx         *app=(AppCtx*)ctx;
  PetscErrorCode ierr;
  DM             da;
  MatStencil     row,col[3];
  PetscScalar    v[3];
  PetscInt       M,xs,xm,i;
  PetscReal      h,h2;

  PetscFunctionBegin;
  CHKERRQ(TSGetDM(ts,&da));

  CHKERRQ(DMDAGetInfo(da,0,&M,0,0,0,0,0,0,0,0,0,0,0));
  CHKERRQ(DMDAGetCorners(da,&xs,0,0,&xm,0,0));

  h = 1.0/(M-1); h2 = h*h;
  for (i=xs; i < xs + xm; i++) {
    row.i = i;
    if (i == 0) {
      v[0]     = 1.0;
      CHKERRQ(MatSetValuesStencil(A,1,&row,1,&row,v,INSERT_VALUES));
    } else if (i == M-1) {
      col[0].i = i;   v[0] = 1/h;
      col[1].i = i-1; v[1] = -1/h;
      CHKERRQ(MatSetValuesStencil(A,1,&row,2,col,v,INSERT_VALUES));
    } else {
      col[0].i = i+1; v[0] = 1/h2;
      col[1].i = i;   v[1] = -2/h2 + app->sw[i-xs]*(-app->A) - a;
      col[2].i = i-1; v[2] = 1/h2;
      CHKERRQ(MatSetValuesStencil(A,1,&row,3,col,v,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;            /* ODE integrator */
  Vec            U;             /* solution will be stored here */
  Mat            J;             /* Jacobian matrix */
  PetscErrorCode ierr;
  PetscInt       n = 16;
  AppCtx         app;
  DM             da;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex22 options","");CHKERRQ(ierr);
  {
    app.A = 200000;
    CHKERRQ(PetscOptionsReal("-A","","",app.A,&app.A,NULL));
    app.uc = 0.5;
    CHKERRQ(PetscOptionsReal("-uc","","",app.uc,&app.uc,NULL));
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,-n,1,1,0,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMDASetUniformCoordinates(da,0.0,1.0,0,0,0,0));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(DMCreateMatrix(da,&J));
  CHKERRQ(DMCreateGlobalVector(da,&U));

  CHKERRQ(InitialConditions(U,da,&app));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(TSSetType(ts,TSROSW));
  CHKERRQ(TSSetIFunction(ts,NULL,(TSIFunction) IFunction,(void*)&app));
  CHKERRQ(TSSetIJacobian(ts,J,J,(TSIJacobian)IJacobian,(void*)&app));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetSolution(ts,U));

  CHKERRQ(TSSetDM(ts,da));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetTimeStep(ts,0.1));
  CHKERRQ(TSSetMaxTime(ts,30.0));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  CHKERRQ(TSSetFromOptions(ts));

  PetscInt lsize;
  CHKERRQ(VecGetLocalSize(U,&lsize));
  PetscInt *direction;
  PetscBool *terminate;
  PetscInt  i;
  CHKERRQ(PetscMalloc1(lsize,&direction));
  CHKERRQ(PetscMalloc1(lsize,&terminate));
  for (i=0; i < lsize; i++) {
    direction[i] = -1;
    terminate[i] = PETSC_FALSE;
  }
  CHKERRQ(TSSetEventHandler(ts,lsize,direction,terminate,EventFunction,PostEventFunction,(void*)&app));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Run timestepping solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    CHKERRQ(TSSolve(ts,U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatDestroy(&J));
  CHKERRQ(VecDestroy(&U));
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(PetscFree(direction));
  CHKERRQ(PetscFree(terminate));

  CHKERRQ(PetscFree(app.sw));
  CHKERRQ(PetscFinalize());
  return 0;
}
