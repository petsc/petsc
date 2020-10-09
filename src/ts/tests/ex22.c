
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
  ierr = DMGetCoordinates(da,&xcoord);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,xcoord,&x);CHKERRQ(ierr);

  ierr = VecGetLocalSize(U,&lsize);CHKERRQ(ierr);
  ierr = PetscMalloc1(lsize,&app->sw);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);

  ierr = DMDAGetInfo(da,0,&M,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);

  for (i=xs; i<xs+xm;i++) {
    if (x[i] <= 0.1) u[i] = 0.;
    else if (x[i] > 0.1 && x[i] < 0.25) u[i] = (x[i] - 0.1)/0.15;
    else u[i] = 1.0;

    app->sw[i-xs] = 1;
  }
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,xcoord,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EventFunction(TS ts,PetscReal t,Vec U,PetscScalar *fvalue,void *ctx)
{
  AppCtx            *app=(AppCtx*)ctx;
  PetscErrorCode    ierr;
  const PetscScalar *u;
  PetscInt          i,lsize;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(U,&lsize);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  for (i=0; i < lsize;i++) fvalue[i] = u[i] - app->uc;
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
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
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);

  ierr = DMDAGetInfo(da,0,&M,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);

  ierr = DMGetLocalVector(da,&Ulocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,Ulocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,Ulocal);CHKERRQ(ierr);

  h = 1.0/(M-1); h2 = h*h;
  ierr = DMDAVecGetArrayRead(da,Udot,&udot);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,Ulocal,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  for (i=xs; i<xs+xm;i++) {
    if (i == 0) {
      f[i] = u[i];
    } else if (i == M - 1) {
      f[i] = (u[i] - u[i-1])/h;
    } else {
      f[i] = (u[i+1] - 2*u[i] + u[i-1])/h2 + app->sw[i-xs]*(-app->A*u[i]) - udot[i];
    }
  }

  ierr = DMDAVecRestoreArrayRead(da,Udot,&udot);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,Ulocal,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&Ulocal);CHKERRQ(ierr);

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
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);

  ierr = DMDAGetInfo(da,0,&M,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);

  h = 1.0/(M-1); h2 = h*h;
  for (i=xs; i < xs + xm; i++) {
    row.i = i;
    if (i == 0) {
      v[0]     = 1.0;
      ierr = MatSetValuesStencil(A,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
    } else if (i == M-1) {
      col[0].i = i;   v[0] = 1/h;
      col[1].i = i-1; v[1] = -1/h;
      ierr = MatSetValuesStencil(A,1,&row,2,col,v,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      col[0].i = i+1; v[0] = 1/h2;
      col[1].i = i;   v[1] = -2/h2 + app->sw[i-xs]*(-app->A) - a;
      col[2].i = i-1; v[2] = 1/h2;
      ierr = MatSetValuesStencil(A,1,&row,3,col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex22 options","");CHKERRQ(ierr);
  {
    app.A = 200000;
    ierr = PetscOptionsReal("-A","","",app.A,&app.A,NULL);CHKERRQ(ierr);
    app.uc = 0.5;
    ierr = PetscOptionsReal("-uc","","",app.uc,&app.uc,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,-n,1,1,0,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0,0,0,0);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateMatrix(da,&J);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&U);CHKERRQ(ierr);

  ierr = InitialConditions(U,da,&app);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSROSW);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,(TSIFunction) IFunction,(void*)&app);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,(TSIJacobian)IJacobian,(void*)&app);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);

  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetTimeStep(ts,0.1);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,30.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  PetscInt lsize;
  ierr = VecGetLocalSize(U,&lsize);CHKERRQ(ierr);
  PetscInt *direction;
  PetscBool *terminate;
  PetscInt  i;
  ierr = PetscMalloc1(lsize,&direction);CHKERRQ(ierr);
  ierr = PetscMalloc1(lsize,&terminate);CHKERRQ(ierr);
  for (i=0; i < lsize; i++) {
    direction[i] = -1;
    terminate[i] = PETSC_FALSE;
  }
  ierr = TSSetEventHandler(ts,lsize,direction,terminate,EventFunction,PostEventFunction,(void*)&app);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Run timestepping solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = TSSolve(ts,U);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFree(direction);CHKERRQ(ierr);
  ierr = PetscFree(terminate);CHKERRQ(ierr);

  ierr = PetscFree(app.sw);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
