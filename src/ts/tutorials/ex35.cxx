static const char help[] = "Time-dependent Brusselator reaction-diffusion PDE in 1d. Demonstrates IMEX methods and uses MOAB.\n";
/*
   u_t - alpha u_xx = A + u^2 v - (B+1) u
   v_t - alpha v_xx = B u - u^2 v
   0 < x < 1;
   A = 1, B = 3, alpha = 1/50

   Initial conditions:
   u(x,0) = 1 + sin(2 pi x)
   v(x,0) = 3

   Boundary conditions:
   u(0,t) = u(1,t) = 1
   v(0,t) = v(1,t) = 3
*/

// PETSc includes:
#include <petscts.h>
#include <petscdmmoab.h>

typedef struct {
  PetscScalar u,v;
} Field;

struct pUserCtx {
  PetscReal A,B;        /* Reaction coefficients */
  PetscReal alpha;      /* Diffusion coefficient */
  Field leftbc;         /* Dirichlet boundary conditions at left boundary */
  Field rightbc;        /* Dirichlet boundary conditions at right boundary */
  PetscInt  n,npts;       /* Number of mesh points */
  PetscInt  ntsteps;    /* Number of time steps */
  PetscInt nvars;       /* Number of variables in the equation system */
  PetscBool io;
};
typedef pUserCtx* UserCtx;

PetscErrorCode Initialize_AppContext(UserCtx *puser)
{
  UserCtx           user;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&user);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Advection-reaction options","ex35.cxx");CHKERRQ(ierr);
  {
    user->nvars  = 2;
    user->A      = 1;
    user->B      = 3;
    user->alpha  = 0.02;
    user->leftbc.u  = 1;
    user->rightbc.u = 1;
    user->leftbc.v  = 3;
    user->rightbc.v = 3;
    user->n      = 10;
    user->ntsteps = 10000;
    user->io = PETSC_FALSE;
    ierr = PetscOptionsReal("-A","Reaction rate","ex35.cxx",user->A,&user->A,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-B","Reaction rate","ex35.cxx",user->B,&user->B,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-alpha","Diffusion coefficient","ex35.cxx",user->alpha,&user->alpha,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsScalar("-uleft","Dirichlet boundary condition","ex35.cxx",user->leftbc.u,&user->leftbc.u,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsScalar("-uright","Dirichlet boundary condition","ex35.cxx",user->rightbc.u,&user->rightbc.u,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsScalar("-vleft","Dirichlet boundary condition","ex35.cxx",user->leftbc.v,&user->leftbc.v,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsScalar("-vright","Dirichlet boundary condition","ex35.cxx",user->rightbc.v,&user->rightbc.v,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-n","Number of 1-D elements","ex35.cxx",user->n,&user->n,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ndt","Number of time steps","ex35.cxx",user->ntsteps,&user->ntsteps,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-io","Write the mesh and solution output to a file.","ex35.cxx",user->io,&user->io,NULL);CHKERRQ(ierr);
    user->npts   = user->n+1;
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  *puser = user;
  PetscFunctionReturn(0);
}

PetscErrorCode Destroy_AppContext(UserCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(*user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormInitialSolution(TS,Vec,void*);
static PetscErrorCode FormRHSFunction(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormIFunction(TS,PetscReal,Vec,Vec,Vec,void*);
static PetscErrorCode FormIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);

/****************
 *              *
 *     MAIN     *
 *              *
 ****************/
int main(int argc,char **argv)
{
  TS                ts;         /* nonlinear solver */
  Vec               X;          /* solution, residual vectors */
  Mat               J;          /* Jacobian matrix */
  PetscInt          steps,mx;
  PetscErrorCode    ierr;
  PetscReal         hx,dt,ftime;
  UserCtx           user;       /* user-defined work context */
  TSConvergedReason reason;

  DM                dm;
  const char        *fields[2] = {"U","V"};

  ierr = PetscInitialize(&argc,&argv,(char *)0,help);if (ierr) return ierr;

  /* Initialize the user context struct */
  ierr = Initialize_AppContext(&user);CHKERRQ(ierr);

  /* Fill in the user defined work context: */
  ierr = DMMoabCreateBoxMesh(PETSC_COMM_WORLD, 1, PETSC_FALSE, NULL, user->n, 1, &dm);CHKERRQ(ierr);
  ierr = DMMoabSetFieldNames(dm, user->nvars, fields);CHKERRQ(ierr);
  ierr = DMMoabSetBlockSize(dm, user->nvars);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);

  /* SetUp the data structures for DMMOAB */
  ierr = DMSetUp(dm);CHKERRQ(ierr);

  /*  Create timestepping solver context */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSSetEquationType(ts,TS_EQ_DAE_IMPLICIT_INDEX1);CHKERRQ(ierr);
  ierr = DMSetMatType(dm,MATBAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm,&J);CHKERRQ(ierr);

  ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,user);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,FormIFunction,user);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,FormIJacobian,user);CHKERRQ(ierr);

  ftime = 10.0;
  ierr = TSSetMaxSteps(ts,user->ntsteps);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create the solution vector and set the initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(dm, &X);CHKERRQ(ierr);

  ierr = FormInitialSolution(ts,X,user);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = VecGetSize(X,&mx);CHKERRQ(ierr);
  hx = 1.0/(PetscReal)(mx/2-1);
  dt = 0.4 * PetscSqr(hx) / user->alpha; /* Diffusive stability limit */
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s at time %g after %D steps\n",TSConvergedReasons[reason],ftime,steps);CHKERRQ(ierr);

  if (user->io) {
    /* Print the numerical solution to screen and then dump to file */
    ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    /* Write out the solution along with the mesh */
    ierr = DMMoabSetGlobalFieldVector(dm, X);CHKERRQ(ierr);
#ifdef MOAB_HAVE_HDF5
    ierr = DMMoabOutput(dm, "ex35.h5m", "");CHKERRQ(ierr);
#else
    /* MOAB does not support true parallel writers that aren't HDF5 based
       And so if you are using VTK as the output format in parallel,
       the data could be jumbled due to the order in which the processors
       write out their parts of the mesh and solution tags
    */
    ierr = DMMoabOutput(dm, "ex35.vtk", "");CHKERRQ(ierr);
#endif
  }

  /* Free work space.
     Free all PETSc related resources: */
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);

  /* Free all MOAB related resources: */
  ierr = Destroy_AppContext(&user);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*
  IJacobian - Compute IJacobian = dF/dU + a dF/dUdot
*/
PetscErrorCode FormIJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat J,Mat Jpre,void *ptr)
{
  UserCtx             user = (UserCtx)ptr;
  PetscErrorCode      ierr;
  PetscInt            dof;
  PetscReal           hx;
  DM                  dm;
  const moab::Range   *vlocal;
  PetscBool           vonboundary;

  PetscFunctionBegin;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);

  /* get the essential MOAB mesh related quantities needed for FEM assembly */
  ierr = DMMoabGetLocalVertices(dm, &vlocal, NULL);CHKERRQ(ierr);

  /* compute local element sizes - structured grid */
  hx = 1.0/user->n;

  /* Compute function over the locally owned part of the grid
     Assemble the operator by looping over edges and computing
     contribution for each vertex dof                         */
  for (moab::Range::iterator iter = vlocal->begin(); iter != vlocal->end(); iter++) {
    const moab::EntityHandle vhandle = *iter;

    ierr = DMMoabGetDofsBlocked(dm, 1, &vhandle, &dof);CHKERRQ(ierr);

    /* check if vertex is on the boundary */
    ierr = DMMoabIsEntityOnBoundary(dm,vhandle,&vonboundary);CHKERRQ(ierr);

    if (vonboundary) {
      const PetscScalar bcvals[2][2] = {{hx,0},{0,hx}};
      ierr = MatSetValuesBlocked(Jpre,1,&dof,1,&dof,&bcvals[0][0],INSERT_VALUES);CHKERRQ(ierr);
    }
    else {
      const PetscInt    row           = dof,col[] = {dof-1,dof,dof+1};
      const PetscScalar dxxL          = -user->alpha/hx,dxx0 = 2.*user->alpha/hx,dxxR = -user->alpha/hx;
      const PetscScalar vals[2][3][2] = {{{dxxL,0},{a *hx+dxx0,0},{dxxR,0}},
                                         {{0,dxxL},{0,a*hx+dxx0},{0,dxxR}}};
      ierr = MatSetValuesBlocked(Jpre,1,&row,3,col,&vals[0][0][0],INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J != Jpre) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ptr)
{
  UserCtx           user = (UserCtx)ptr;
  DM                dm;
  PetscReal         hx;
  const Field       *x;
  Field             *f;
  PetscInt          dof;
  const moab::Range *ownedvtx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  hx = 1.0/user->n;
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = VecSet(F,0.0);CHKERRQ(ierr);

  ierr = DMMoabVecGetArrayRead(dm, X, &x);CHKERRQ(ierr);
  ierr = DMMoabVecGetArray(dm, F, &f);CHKERRQ(ierr);

  ierr = DMMoabGetLocalVertices(dm, &ownedvtx, NULL);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (moab::Range::iterator iter = ownedvtx->begin(); iter != ownedvtx->end(); iter++) {
    const moab::EntityHandle vhandle = *iter;
    ierr = DMMoabGetDofsBlockedLocal(dm, 1, &vhandle, &dof);CHKERRQ(ierr);

    PetscScalar u = x[dof].u,v = x[dof].v;
    f[dof].u = hx*(user->A + u*u*v - (user->B+1)*u);
    f[dof].v = hx*(user->B*u - u*u*v);
  }

  /* Restore vectors */
  ierr = DMMoabVecRestoreArrayRead(dm, X, &x);CHKERRQ(ierr);
  ierr = DMMoabVecRestoreArray(dm, F, &f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormIFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  UserCtx         user = (UserCtx)ctx;
  DM              dm;
  Field           *x,*xdot,*f;
  PetscReal       hx;
  Vec             Xloc;
  PetscErrorCode  ierr;
  PetscInt        i,bcindx;
  PetscBool       elem_on_boundary;
  const moab::Range   *vlocal;

  PetscFunctionBegin;
  hx = 1.0/user->n;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);

  /* get the essential MOAB mesh related quantities needed for FEM assembly */
  ierr = DMMoabGetLocalVertices(dm, &vlocal, NULL);CHKERRQ(ierr);

  /* reset the residual vector */
  ierr = VecSet(F,0.0);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);

  /* get the local representation of the arrays from Vectors */
  ierr = DMMoabVecGetArrayRead(dm, Xloc, &x);CHKERRQ(ierr);
  ierr = DMMoabVecGetArrayRead(dm, Xdot, &xdot);CHKERRQ(ierr);
  ierr = DMMoabVecGetArray(dm, F, &f);CHKERRQ(ierr);

  /* loop over local elements */
  for (moab::Range::iterator iter = vlocal->begin(); iter != vlocal->end(); iter++) {
    const moab::EntityHandle vhandle = *iter;

    ierr = DMMoabGetDofsBlockedLocal(dm,1,&vhandle,&i);CHKERRQ(ierr);

    /* check if vertex is on the boundary */
    ierr = DMMoabIsEntityOnBoundary(dm,vhandle,&elem_on_boundary);CHKERRQ(ierr);

    if (elem_on_boundary) {
      ierr = DMMoabGetDofsBlocked(dm, 1, &vhandle, &bcindx);CHKERRQ(ierr);
      if (bcindx == 0) {  /* Apply left BC */
        f[i].u = hx * (x[i].u - user->leftbc.u);
        f[i].v = hx * (x[i].v - user->leftbc.v);
      } else {       /* Apply right BC */
        f[i].u = hx * (x[i].u - user->rightbc.u);
        f[i].v = hx * (x[i].v - user->rightbc.v);
      }
    }
    else {
      f[i].u = hx * xdot[i].u - user->alpha * (x[i-1].u - 2.*x[i].u + x[i+1].u) / hx;
      f[i].v = hx * xdot[i].v - user->alpha * (x[i-1].v - 2.*x[i].v + x[i+1].v) / hx;
    }
  }

  /* Restore data */
  ierr = DMMoabVecRestoreArrayRead(dm, Xloc, &x);CHKERRQ(ierr);
  ierr = DMMoabVecRestoreArrayRead(dm, Xdot, &xdot);CHKERRQ(ierr);
  ierr = DMMoabVecRestoreArray(dm, F, &f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &Xloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormInitialSolution(TS ts,Vec X,void *ctx)
{
  UserCtx           user = (UserCtx)ctx;
  PetscReal         vpos[3];
  DM                dm;
  Field             *x;
  PetscErrorCode    ierr;
  const moab::Range *vowned;
  PetscInt          dof;
  moab::Range::iterator iter;

  PetscFunctionBegin;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);

  /* get the essential MOAB mesh related quantities needed for FEM assembly */
  ierr = DMMoabGetLocalVertices(dm, &vowned, NULL);CHKERRQ(ierr);

  ierr = VecSet(X, 0.0);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMMoabVecGetArray(dm, X, &x);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (moab::Range::iterator iter = vowned->begin(); iter != vowned->end(); iter++) {
    const moab::EntityHandle vhandle = *iter;
    ierr = DMMoabGetDofsBlockedLocal(dm, 1, &vhandle, &dof);CHKERRQ(ierr);

    /* compute the mid-point of the element and use a 1-point lumped quadrature */
    ierr = DMMoabGetVertexCoordinates(dm,1,&vhandle,vpos);CHKERRQ(ierr);

    PetscReal xi = vpos[0];
    x[dof].u = user->leftbc.u*(1.-xi) + user->rightbc.u*xi + PetscSinReal(2.*PETSC_PI*xi);
    x[dof].v = user->leftbc.v*(1.-xi) + user->rightbc.v*xi;
  }

  /* Restore vectors */
  ierr = DMMoabVecRestoreArray(dm, X, &x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

    build:
      requires: moab

    test:
      args: -n 20 -ts_type rosw -ts_rosw_type 2p -ts_dt 5e-2 -ts_adapt_type none

    test:
      suffix: 2
      nsize: 2
      args: -n 50 -ts_type glee -ts_adapt_type none -ts_dt 0.1 -io
      TODO:

TEST*/
