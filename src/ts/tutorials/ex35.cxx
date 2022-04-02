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

  PetscFunctionBegin;
  PetscCall(PetscNew(&user));
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Advection-reaction options","ex35.cxx");
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
    PetscCall(PetscOptionsReal("-A","Reaction rate","ex35.cxx",user->A,&user->A,NULL));
    PetscCall(PetscOptionsReal("-B","Reaction rate","ex35.cxx",user->B,&user->B,NULL));
    PetscCall(PetscOptionsReal("-alpha","Diffusion coefficient","ex35.cxx",user->alpha,&user->alpha,NULL));
    PetscCall(PetscOptionsScalar("-uleft","Dirichlet boundary condition","ex35.cxx",user->leftbc.u,&user->leftbc.u,NULL));
    PetscCall(PetscOptionsScalar("-uright","Dirichlet boundary condition","ex35.cxx",user->rightbc.u,&user->rightbc.u,NULL));
    PetscCall(PetscOptionsScalar("-vleft","Dirichlet boundary condition","ex35.cxx",user->leftbc.v,&user->leftbc.v,NULL));
    PetscCall(PetscOptionsScalar("-vright","Dirichlet boundary condition","ex35.cxx",user->rightbc.v,&user->rightbc.v,NULL));
    PetscCall(PetscOptionsInt("-n","Number of 1-D elements","ex35.cxx",user->n,&user->n,NULL));
    PetscCall(PetscOptionsInt("-ndt","Number of time steps","ex35.cxx",user->ntsteps,&user->ntsteps,NULL));
    PetscCall(PetscOptionsBool("-io","Write the mesh and solution output to a file.","ex35.cxx",user->io,&user->io,NULL));
    user->npts   = user->n+1;
  }
  PetscOptionsEnd();

  *puser = user;
  PetscFunctionReturn(0);
}

PetscErrorCode Destroy_AppContext(UserCtx *user)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(*user));
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
  PetscReal         hx,dt,ftime;
  UserCtx           user;       /* user-defined work context */
  TSConvergedReason reason;
  DM                dm;
  const char        *fields[2] = {"U","V"};

  PetscCall(PetscInitialize(&argc,&argv,(char *)0,help));

  /* Initialize the user context struct */
  PetscCall(Initialize_AppContext(&user));

  /* Fill in the user defined work context: */
  PetscCall(DMMoabCreateBoxMesh(PETSC_COMM_WORLD, 1, PETSC_FALSE, NULL, user->n, 1, &dm));
  PetscCall(DMMoabSetFieldNames(dm, user->nvars, fields));
  PetscCall(DMMoabSetBlockSize(dm, user->nvars));
  PetscCall(DMSetFromOptions(dm));

  /* SetUp the data structures for DMMOAB */
  PetscCall(DMSetUp(dm));

  /*  Create timestepping solver context */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetDM(ts, dm));
  PetscCall(TSSetType(ts,TSARKIMEX));
  PetscCall(TSSetEquationType(ts,TS_EQ_DAE_IMPLICIT_INDEX1));
  PetscCall(DMSetMatType(dm,MATBAIJ));
  PetscCall(DMCreateMatrix(dm,&J));

  PetscCall(TSSetRHSFunction(ts,NULL,FormRHSFunction,user));
  PetscCall(TSSetIFunction(ts,NULL,FormIFunction,user));
  PetscCall(TSSetIJacobian(ts,J,J,FormIJacobian,user));

  ftime = 10.0;
  PetscCall(TSSetMaxSteps(ts,user->ntsteps));
  PetscCall(TSSetMaxTime(ts,ftime));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create the solution vector and set the initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMCreateGlobalVector(dm, &X));

  PetscCall(FormInitialSolution(ts,X,user));
  PetscCall(TSSetSolution(ts,X));
  PetscCall(VecGetSize(X,&mx));
  hx = 1.0/(PetscReal)(mx/2-1);
  dt = 0.4 * PetscSqr(hx) / user->alpha; /* Diffusive stability limit */
  PetscCall(TSSetTimeStep(ts,dt));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts,X));
  PetscCall(TSGetSolveTime(ts,&ftime));
  PetscCall(TSGetStepNumber(ts,&steps));
  PetscCall(TSGetConvergedReason(ts,&reason));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%s at time %g after %D steps\n",TSConvergedReasons[reason],ftime,steps));

  if (user->io) {
    /* Print the numerical solution to screen and then dump to file */
    PetscCall(VecView(X,PETSC_VIEWER_STDOUT_WORLD));

    /* Write out the solution along with the mesh */
    PetscCall(DMMoabSetGlobalFieldVector(dm, X));
#ifdef MOAB_HAVE_HDF5
    PetscCall(DMMoabOutput(dm, "ex35.h5m", ""));
#else
    /* MOAB does not support true parallel writers that aren't HDF5 based
       And so if you are using VTK as the output format in parallel,
       the data could be jumbled due to the order in which the processors
       write out their parts of the mesh and solution tags
    */
    PetscCall(DMMoabOutput(dm, "ex35.vtk", ""));
#endif
  }

  /* Free work space.
     Free all PETSc related resources: */
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&X));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&dm));

  /* Free all MOAB related resources: */
  PetscCall(Destroy_AppContext(&user));
  PetscCall(PetscFinalize());
  return 0;
}

/*
  IJacobian - Compute IJacobian = dF/dU + a dF/dUdot
*/
PetscErrorCode FormIJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat J,Mat Jpre,void *ptr)
{
  UserCtx             user = (UserCtx)ptr;
  PetscInt            dof;
  PetscReal           hx;
  DM                  dm;
  const moab::Range   *vlocal;
  PetscBool           vonboundary;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &dm));

  /* get the essential MOAB mesh related quantities needed for FEM assembly */
  PetscCall(DMMoabGetLocalVertices(dm, &vlocal, NULL));

  /* compute local element sizes - structured grid */
  hx = 1.0/user->n;

  /* Compute function over the locally owned part of the grid
     Assemble the operator by looping over edges and computing
     contribution for each vertex dof                         */
  for (moab::Range::iterator iter = vlocal->begin(); iter != vlocal->end(); iter++) {
    const moab::EntityHandle vhandle = *iter;

    PetscCall(DMMoabGetDofsBlocked(dm, 1, &vhandle, &dof));

    /* check if vertex is on the boundary */
    PetscCall(DMMoabIsEntityOnBoundary(dm,vhandle,&vonboundary));

    if (vonboundary) {
      const PetscScalar bcvals[2][2] = {{hx,0},{0,hx}};
      PetscCall(MatSetValuesBlocked(Jpre,1,&dof,1,&dof,&bcvals[0][0],INSERT_VALUES));
    }
    else {
      const PetscInt    row           = dof,col[] = {dof-1,dof,dof+1};
      const PetscScalar dxxL          = -user->alpha/hx,dxx0 = 2.*user->alpha/hx,dxxR = -user->alpha/hx;
      const PetscScalar vals[2][3][2] = {{{dxxL,0},{a *hx+dxx0,0},{dxxR,0}},
                                         {{0,dxxL},{0,a*hx+dxx0},{0,dxxR}}};
      PetscCall(MatSetValuesBlocked(Jpre,1,&row,3,col,&vals[0][0][0],INSERT_VALUES));
    }
  }

  PetscCall(MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY));
  if (J != Jpre) {
    PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
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

  PetscFunctionBegin;
  hx = 1.0/user->n;
  PetscCall(TSGetDM(ts,&dm));

  /* Get pointers to vector data */
  PetscCall(VecSet(F,0.0));

  PetscCall(DMMoabVecGetArrayRead(dm, X, &x));
  PetscCall(DMMoabVecGetArray(dm, F, &f));

  PetscCall(DMMoabGetLocalVertices(dm, &ownedvtx, NULL));

  /* Compute function over the locally owned part of the grid */
  for (moab::Range::iterator iter = ownedvtx->begin(); iter != ownedvtx->end(); iter++) {
    const moab::EntityHandle vhandle = *iter;
    PetscCall(DMMoabGetDofsBlockedLocal(dm, 1, &vhandle, &dof));

    PetscScalar u = x[dof].u,v = x[dof].v;
    f[dof].u = hx*(user->A + u*u*v - (user->B+1)*u);
    f[dof].v = hx*(user->B*u - u*u*v);
  }

  /* Restore vectors */
  PetscCall(DMMoabVecRestoreArrayRead(dm, X, &x));
  PetscCall(DMMoabVecRestoreArray(dm, F, &f));
  PetscFunctionReturn(0);
}

static PetscErrorCode FormIFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  UserCtx         user = (UserCtx)ctx;
  DM              dm;
  Field           *x,*xdot,*f;
  PetscReal       hx;
  Vec             Xloc;
  PetscInt        i,bcindx;
  PetscBool       elem_on_boundary;
  const moab::Range   *vlocal;

  PetscFunctionBegin;
  hx = 1.0/user->n;
  PetscCall(TSGetDM(ts, &dm));

  /* get the essential MOAB mesh related quantities needed for FEM assembly */
  PetscCall(DMMoabGetLocalVertices(dm, &vlocal, NULL));

  /* reset the residual vector */
  PetscCall(VecSet(F,0.0));

  PetscCall(DMGetLocalVector(dm,&Xloc));
  PetscCall(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
  PetscCall(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));

  /* get the local representation of the arrays from Vectors */
  PetscCall(DMMoabVecGetArrayRead(dm, Xloc, &x));
  PetscCall(DMMoabVecGetArrayRead(dm, Xdot, &xdot));
  PetscCall(DMMoabVecGetArray(dm, F, &f));

  /* loop over local elements */
  for (moab::Range::iterator iter = vlocal->begin(); iter != vlocal->end(); iter++) {
    const moab::EntityHandle vhandle = *iter;

    PetscCall(DMMoabGetDofsBlockedLocal(dm,1,&vhandle,&i));

    /* check if vertex is on the boundary */
    PetscCall(DMMoabIsEntityOnBoundary(dm,vhandle,&elem_on_boundary));

    if (elem_on_boundary) {
      PetscCall(DMMoabGetDofsBlocked(dm, 1, &vhandle, &bcindx));
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
  PetscCall(DMMoabVecRestoreArrayRead(dm, Xloc, &x));
  PetscCall(DMMoabVecRestoreArrayRead(dm, Xdot, &xdot));
  PetscCall(DMMoabVecRestoreArray(dm, F, &f));
  PetscCall(DMRestoreLocalVector(dm, &Xloc));
  PetscFunctionReturn(0);
}

PetscErrorCode FormInitialSolution(TS ts,Vec X,void *ctx)
{
  UserCtx           user = (UserCtx)ctx;
  PetscReal         vpos[3];
  DM                dm;
  Field             *x;
  const moab::Range *vowned;
  PetscInt          dof;
  moab::Range::iterator iter;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &dm));

  /* get the essential MOAB mesh related quantities needed for FEM assembly */
  PetscCall(DMMoabGetLocalVertices(dm, &vowned, NULL));

  PetscCall(VecSet(X, 0.0));

  /* Get pointers to vector data */
  PetscCall(DMMoabVecGetArray(dm, X, &x));

  /* Compute function over the locally owned part of the grid */
  for (moab::Range::iterator iter = vowned->begin(); iter != vowned->end(); iter++) {
    const moab::EntityHandle vhandle = *iter;
    PetscCall(DMMoabGetDofsBlockedLocal(dm, 1, &vhandle, &dof));

    /* compute the mid-point of the element and use a 1-point lumped quadrature */
    PetscCall(DMMoabGetVertexCoordinates(dm,1,&vhandle,vpos));

    PetscReal xi = vpos[0];
    x[dof].u = user->leftbc.u*(1.-xi) + user->rightbc.u*xi + PetscSinReal(2.*PETSC_PI*xi);
    x[dof].v = user->leftbc.v*(1.-xi) + user->rightbc.v*xi;
  }

  /* Restore vectors */
  PetscCall(DMMoabVecRestoreArray(dm, X, &x));
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
