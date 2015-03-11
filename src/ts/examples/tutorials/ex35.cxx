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
  PetscInt ftype;       /* The type of function assembly routine to use in residual calculation
                           0 (default) = MOAB-Ops, 1 = Block-Ops, 2 = Ghosted-Ops  */
  PetscBool io;
};
typedef pUserCtx* UserCtx;

#undef __FUNCT__
#define __FUNCT__ "Initialize_AppContext"
PetscErrorCode Initialize_AppContext(UserCtx *puser)
{
  UserCtx           user;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&user);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Advection-reaction options","ex35.cxx");
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
    user->ftype = 0;
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
    ierr = PetscOptionsInt("-ftype","Type of function evaluation model for FEM assembly","ex35.cxx",user->ftype,&user->ftype,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-io","Write the mesh and solution output to a file.","ex35.cxx",user->io,&user->io,NULL);CHKERRQ(ierr);
    user->npts   = user->n+1;
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  *puser = user;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Destroy_AppContext"
PetscErrorCode Destroy_AppContext(UserCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(*user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormInitialSolution(TS,Vec,void*);
static PetscErrorCode FormRHSFunction(TS,PetscReal,Vec,Vec,void*);

static PetscErrorCode FormIFunctionGhosted(TS,PetscReal,Vec,Vec,Vec,void*);
static PetscErrorCode FormIFunctionGlobalBlocked(TS,PetscReal,Vec,Vec,Vec,void*);
static PetscErrorCode FormIFunctionMOAB(TS,PetscReal,Vec,Vec,Vec,void*);
static PetscErrorCode FormIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);

/****************
 *              *
 *     MAIN     *
 *              *
 ****************/
#undef __FUNCT__
#define __FUNCT__ "main"
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
  moab::ErrorCode   merr;
  moab::Interface*  mbImpl;
  moab::Tag         solndofs;
  const moab::Range *ownedvtx;
  const PetscReal   bounds[2] = {0.0,1.0};
  const char        *fields[2] = {"U","V"};
  PetscScalar       deflt[2]={0.0,0.0};

  ierr = PetscInitialize(&argc,&argv,(char *)0,help);CHKERRQ(ierr);

  /* Initialize the user context struct */
  ierr = Initialize_AppContext(&user);CHKERRQ(ierr);

  /* Fill in the user defined work context: */
  ierr = DMMoabCreateBoxMesh(PETSC_COMM_WORLD, 1, PETSC_FALSE, bounds, user->n, 1, &dm);CHKERRQ(ierr);
  ierr = DMMoabSetBlockSize(dm, user->nvars);CHKERRQ(ierr);
  ierr = DMMoabSetFieldNames(dm, user->nvars, fields);CHKERRQ(ierr);
  ierr = DMSetMatType(dm,MATBAIJ);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);

  /* SetUp the data structures for DMMOAB */
  ierr = DMSetUp(dm);CHKERRQ(ierr);

  ierr = DMMoabGetInterface(dm, &mbImpl);CHKERRQ(ierr);

  /*  Create timestepping solver context */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,user);CHKERRQ(ierr);
  if (user->ftype == 1) {
    ierr = TSSetIFunction(ts,NULL,FormIFunctionGlobalBlocked,user);CHKERRQ(ierr);
  } else if(user->ftype == 2) {
    ierr = TSSetIFunction(ts,NULL,FormIFunctionGhosted,user);CHKERRQ(ierr);
  } else {
    ierr = TSSetIFunction(ts,NULL,FormIFunctionMOAB,user);CHKERRQ(ierr);
  }
  ierr = DMCreateMatrix(dm,&J);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,FormIJacobian,user);CHKERRQ(ierr);

  ftime = 10.0;
  ierr = TSSetDuration(ts,user->ntsteps,ftime);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create the solution vector and set the initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* Use the call to DMMoabCreateVector for creating a named global MOAB Vec object.
     Alternately, use the following call to DM for creating an unnamed (anonymous) global 
     MOAB Vec object.

         ierr = DMCreateGlobalVector(dm, &X);CHKERRQ(ierr);
  */
  ierr = DMMoabGetLocalVertices(dm, &ownedvtx, NULL);CHKERRQ(ierr);
  /* create a tag to store the unknown fields in the problem */
  merr = mbImpl->tag_get_handle("UNKNOWNS",2,moab::MB_TYPE_DOUBLE,solndofs,
                                  moab::MB_TAG_DENSE|moab::MB_TAG_CREAT,deflt);MBERRNM(merr);

  ierr = DMMoabCreateVector(dm, solndofs, ownedvtx, PETSC_TRUE, PETSC_FALSE, &X);CHKERRQ(ierr);

  ierr = FormInitialSolution(ts,X,user);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = VecGetSize(X,&mx);CHKERRQ(ierr);
  hx = 1.0/(PetscReal)(mx/2-1);
  dt = 0.4 * PetscSqr(hx) / user->alpha; /* Diffusive stability limit */
  ierr = TSSetInitialTimeStep(ts,0.0,dt);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s at time %g after %D steps\n",TSConvergedReasons[reason],ftime,steps);CHKERRQ(ierr);

  if (user->io) {
    /* Print the numerical solution to screen and then dump to file */
    ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    /* Write out the solution along with the mesh */
    ierr = DMMoabSetGlobalFieldVector(dm, X);CHKERRQ(ierr);
#ifdef MOAB_HDF5_H
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
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormRHSFunction"
static PetscErrorCode FormRHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ptr)
{
  UserCtx           user = (UserCtx)ptr;
  DM                dm;
  PetscReal         hx;
  const Field       *x;
  Field             *f;
  PetscInt          dof_index;
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
  for(moab::Range::iterator iter = ownedvtx->begin(); iter != ownedvtx->end(); iter++) {
    const moab::EntityHandle vhandle = *iter;
    ierr = DMMoabGetDofsBlockedLocal(dm, 1, &vhandle, &dof_index);CHKERRQ(ierr);

    const Field& xx = x[dof_index];
    f[dof_index].u = hx*(user->A + xx.u*xx.u*xx.v - (user->B+1)*xx.u);
    f[dof_index].v = hx*(user->B*xx.u - xx.u*xx.u*xx.v);
  }

  /* Restore vectors */
  ierr = DMMoabVecRestoreArrayRead(dm, X, &x);CHKERRQ(ierr);
  ierr = DMMoabVecRestoreArray(dm, F, &f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*
  IJacobian - Compute IJacobian = dF/dU + a dF/dUdot
*/
#undef __FUNCT__
#define __FUNCT__ "FormIJacobian"
PetscErrorCode FormIJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat J,Mat Jpre,void *ptr)
{
  UserCtx             user = (UserCtx)ptr;
  PetscErrorCode      ierr;
  const moab::EntityHandle *connect;
  PetscInt            vpere=2;
  PetscReal           hx;
  DM                  dm;
  moab::Interface*    mbImpl;
  const moab::Range   *elocal;
  PetscInt            dof_indices[2];
  PetscBool           elem_on_boundary;

  PetscFunctionBegin;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);

  /* get the essential MOAB mesh related quantities needed for FEM assembly */
  ierr = DMMoabGetInterface(dm, &mbImpl);CHKERRQ(ierr);
  ierr = DMMoabGetLocalElements(dm, &elocal);CHKERRQ(ierr);

  /* zero out the discrete operator */
  ierr = MatZeroEntries(Jpre);CHKERRQ(ierr);

  /* compute local element sizes - structured grid */
  hx = 1.0/user->n;

  const int& idl = dof_indices[0];
  const int& idr = dof_indices[1];
  const PetscScalar dxxL = user->alpha/hx,dxxR = -user->alpha/hx;
  const PetscScalar bcvals[2][2] = {{hx,0},{0,hx}};
  const PetscScalar e_vals[2][2][2] = {{{a *hx/2+dxxL,0},{dxxR,0}},
                                      {{0,a*hx/2+dxxL},{0,dxxR}}};

  /* Compute function over the locally owned part of the grid 
     Assemble the operator by looping over edges and computing
     contribution for each vertex dof                         */
  for(moab::Range::iterator iter = elocal->begin(); iter != elocal->end(); iter++) {
    const moab::EntityHandle ehandle = *iter;

    // Get connectivity information in canonical order
    ierr = DMMoabGetElementConnectivity(dm, ehandle, &vpere, &connect);CHKERRQ(ierr);
    if (vpere != 2) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only EDGE2 element bases are supported in the current example. n(Connectivity)=%D.\n", vpere);

    ierr = DMMoabGetDofsBlocked(dm, vpere, connect, dof_indices);CHKERRQ(ierr);

    const PetscInt lcols[] = {idl,idr}, rcols[] = {idr, idl};

    /* check if element is on the boundary */
    ierr = DMMoabIsEntityOnBoundary(dm,ehandle,&elem_on_boundary);CHKERRQ(ierr);

    if (elem_on_boundary) {
      if (idl == 0) {
        // Left Boundary conditions...
        ierr = MatSetValuesBlocked(Jpre,1,&idl,1,&idl,&bcvals[0][0],ADD_VALUES);CHKERRQ(ierr);
        ierr = MatSetValuesBlocked(Jpre,1,&idr,2,rcols,&e_vals[0][0][0],ADD_VALUES);CHKERRQ(ierr);
      }
      else {
        // Right Boundary conditions...
        ierr = MatSetValuesBlocked(Jpre,1,&idr,1,&idr,&bcvals[0][0],ADD_VALUES);CHKERRQ(ierr);
        ierr = MatSetValuesBlocked(Jpre,1,&idl,2,lcols,&e_vals[0][0][0],ADD_VALUES);CHKERRQ(ierr);
      }
    }
    else {
      ierr = MatSetValuesBlocked(Jpre,1,&idr,2,rcols,&e_vals[0][0][0],ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValuesBlocked(Jpre,1,&idl,2,lcols,&e_vals[0][0][0],ADD_VALUES);CHKERRQ(ierr);
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


#undef __FUNCT__
#define __FUNCT__ "FormInitialSolution"
PetscErrorCode FormInitialSolution(TS ts,Vec X,void *ctx)
{
  UserCtx           user = (UserCtx)ctx;
  PetscReal         vpos[3];
  DM                dm;
  Field             *x;
  PetscErrorCode    ierr;
  moab::Interface*  mbImpl;
  const moab::Range *vowned;
  PetscInt          dof_index;
  moab::Range::iterator iter;

  PetscFunctionBegin;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  
  /* get the essential MOAB mesh related quantities needed for FEM assembly */
  ierr = DMMoabGetInterface(dm, &mbImpl);CHKERRQ(ierr);

  ierr = DMMoabGetLocalVertices(dm, &vowned, NULL);CHKERRQ(ierr);

  ierr = VecSet(X, 0.0);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMMoabVecGetArray(dm, X, &x);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for(moab::Range::iterator iter = vowned->begin(); iter != vowned->end(); iter++) {
    const moab::EntityHandle vhandle = *iter;
    ierr = DMMoabGetDofsBlockedLocal(dm, 1, &vhandle, &dof_index);CHKERRQ(ierr);

    /* compute the mid-point of the element and use a 1-point lumped quadrature */
    ierr = DMMoabGetVertexCoordinates(dm,1,&vhandle,vpos);CHKERRQ(ierr);

    PetscReal xi = vpos[0];
    x[dof_index].u = user->leftbc.u*(1.-xi) + user->rightbc.u*xi + sin(2.*PETSC_PI*xi);
    x[dof_index].v = user->leftbc.v*(1.-xi) + user->rightbc.v*xi;
  }

  /* Restore vectors */
  ierr = DMMoabVecRestoreArray(dm, X, &x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FormIFunctionMOAB"
static PetscErrorCode FormIFunctionMOAB(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  UserCtx         user = (UserCtx)ctx;
  DM              dm;
  Field           *x,*xdot,*f;
  PetscReal       hx,vpos[2*3];
  PetscErrorCode  ierr;
  PetscInt        dof_indices[2],bc_indices[2];
  const moab::EntityHandle *connect;
  PetscInt        vpere=2,nloc,ngh;
  PetscBool       elem_on_boundary;
  const int& idx_left = dof_indices[0];
  const int& idx_right = dof_indices[1];
  moab::Interface*  mbImpl;
  const moab::Range   *elocal;

  PetscFunctionBegin;
  hx = 1.0/user->n;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);

  /* get the essential MOAB mesh related quantities needed for FEM assembly */
  ierr = DMMoabGetInterface(dm, &mbImpl);CHKERRQ(ierr);
  ierr = DMMoabGetLocalElements(dm, &elocal);CHKERRQ(ierr);
  ierr = DMMoabGetLocalSize(dm, NULL, NULL, &nloc, &ngh);

  /* reset the residual vector */
  ierr = VecSet(F,0.0);CHKERRQ(ierr);

  /* get the local representation of the arrays from Vectors */
  ierr = DMMoabVecGetArrayRead(dm, X, &x);CHKERRQ(ierr);
  ierr = DMMoabVecGetArrayRead(dm, Xdot, &xdot);CHKERRQ(ierr);
  ierr = DMMoabVecGetArray(dm, F, &f);CHKERRQ(ierr);

  /* loop over local elements */
  for(moab::Range::iterator iter = elocal->begin(); iter != elocal->end(); iter++) {
    const moab::EntityHandle ehandle = *iter;

    // Get connectivity information in canonical order
    ierr = DMMoabGetElementConnectivity(dm, ehandle, &vpere, &connect);CHKERRQ(ierr);
    if (vpere != 2) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only EDGE2 element bases are supported in the current example. n(Connectivity)=%D.\n", vpere);

    /* compute the mid-point of the element and use a 1-point lumped quadrature */
    ierr = DMMoabGetVertexCoordinates(dm,vpere,connect,vpos);CHKERRQ(ierr);

    ierr = DMMoabGetDofsBlockedLocal(dm, vpere, connect, dof_indices);CHKERRQ(ierr);

    /* check if element is on the boundary */
    ierr = DMMoabIsEntityOnBoundary(dm,ehandle,&elem_on_boundary);CHKERRQ(ierr);

    if (elem_on_boundary) {
      ierr = DMMoabGetDofsBlocked(dm, vpere, connect, bc_indices);CHKERRQ(ierr);
      if (bc_indices[0] == 0) {      /* Apply left BC */
        f[idx_left].u = hx * (x[idx_left].u - user->leftbc.u);
        f[idx_left].v = hx * (x[idx_left].v - user->leftbc.v);
        f[idx_right].u += user->alpha*(x[idx_right].u-x[idx_left].u)/hx;
        f[idx_right].v += user->alpha*(x[idx_right].v-x[idx_left].v)/hx;
      }
      else {                        /* Apply right BC */
        f[idx_left].u += hx * xdot[idx_left].u + user->alpha*(x[idx_left].u - x[idx_right].u)/hx;
        f[idx_left].v += hx * xdot[idx_left].v + user->alpha*(x[idx_left].v - x[idx_right].v)/hx;
        f[idx_right].u = hx * (x[idx_right].u - user->rightbc.u);
        f[idx_right].v = hx * (x[idx_right].v - user->rightbc.v);
      }
    }
    else {
      f[idx_left].u += hx * xdot[idx_left].u + user->alpha*(x[idx_left].u - x[idx_right].u)/hx;
      f[idx_left].v += hx * xdot[idx_left].v + user->alpha*(x[idx_left].v - x[idx_right].v)/hx;
      f[idx_right].u += user->alpha*(x[idx_right].u-x[idx_left].u)/hx;
      f[idx_right].v += user->alpha*(x[idx_right].v-x[idx_left].v)/hx;
    }
  }

  /* Restore data */
  ierr = DMMoabVecRestoreArrayRead(dm, X, &x);CHKERRQ(ierr);
  ierr = DMMoabVecRestoreArrayRead(dm, Xdot, &xdot);CHKERRQ(ierr);
  ierr = DMMoabVecRestoreArray(dm, F, &f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FormIFunctionGlobalBlocked"
static PetscErrorCode FormIFunctionGlobalBlocked(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ptr)
{
  UserCtx             user = (UserCtx)ptr;
  Field               *x,*xdot;
  Vec                 xltmp, xdtmp, xlocal,xdotlocal,flocal;
  PetscReal           hx;
  PetscErrorCode      ierr;
  DM                  dm;
  PetscBool           elem_on_boundary;

  PetscInt            dof_indices[2],gdof[2];
  PetscInt            i,vpere=2;
  const moab::EntityHandle *connect;
  moab::Interface*  mbImpl;
  const moab::Range   *elocal;
  PetscInt           *dofs;
  const int& idl = dof_indices[0];
  const int& idr = dof_indices[1];

  PetscFunctionBegin;
  hx = 1.0/user->n;

  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);

  /* get the essential MOAB mesh related quantities needed for FEM assembly */
  ierr = DMMoabGetInterface(dm, &mbImpl);CHKERRQ(ierr);
  ierr = DMMoabGetLocalElements(dm, &elocal);CHKERRQ(ierr);

  // reset the residual vector before assembly
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);

  /* get the local representation of the arrays from Vectors */
  ierr = DMGetLocalVector(dm,&xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = VecGhostGetLocalForm(xlocal,&xltmp);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(xlocal,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(xlocal,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&xdotlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,Xdot,INSERT_VALUES,xdotlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,Xdot,INSERT_VALUES,xdotlocal);CHKERRQ(ierr);
  ierr = VecGhostGetLocalForm(xdotlocal,&xdtmp);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(xdotlocal,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(xdotlocal,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = VecGetArrayRead(xltmp, (const PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xdtmp, (const PetscScalar**)&xdot);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&flocal);CHKERRQ(ierr);
  ierr = VecSet(flocal, 0.0);CHKERRQ(ierr);

  ierr = DMMoabGetVertexDofsBlocked(dm, &dofs);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid 
     Assemble the operator by looping over edges and computing
     contribution for each vertex dof                         */

  /* loop over local elements */
  for(moab::Range::iterator iter = elocal->begin(); iter != elocal->end(); iter++) {
    const moab::EntityHandle ehandle = *iter;

    // Get connectivity information in canonical order
    ierr = DMMoabGetElementConnectivity(dm, ehandle, &vpere, &connect);CHKERRQ(ierr);
    if (vpere != 2) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only EDGE2 element bases are supported in the current example. Connectivity=%D.\n", vpere);
    
    ierr = DMMoabGetDofsBlockedLocal(dm, vpere, connect, dof_indices);CHKERRQ(ierr);
    for (i=0; i<2; i++)
      gdof[i] = dofs[connect[i]];

    /* check if element is on the boundary */
    ierr = DMMoabIsEntityOnBoundary(dm,ehandle,&elem_on_boundary);CHKERRQ(ierr);

    if (elem_on_boundary) {
      if (gdof[0] == 0) {
        const PetscScalar vals[4] = { hx * (x[idl].u - user->leftbc.u),
                                 hx * (x[idl].v - user->leftbc.v),
                                 hx/2 * xdot[idr].u + user->alpha * ( x[idr].u - x[idl].u ) / hx,
                                 hx/2 * xdot[idr].v + user->alpha * ( x[idr].v - x[idl].v ) / hx};

        ierr = VecSetValuesBlocked(flocal, vpere, gdof, vals, ADD_VALUES);CHKERRQ(ierr);
      }
      else {
        const PetscScalar vals[4] = { hx/2 * xdot[idl].u + user->alpha * ( x[idl].u - x[idr].u ) / hx,
                                 hx/2 * xdot[idl].v + user->alpha * ( x[idl].v - x[idr].v ) / hx,
                                 hx * (x[idr].u - user->rightbc.u),
                                 hx * (x[idr].v - user->rightbc.v) };

        ierr = VecSetValuesBlocked(flocal, vpere, gdof, vals, ADD_VALUES);CHKERRQ(ierr);
      }
    }
    else {
      const PetscScalar vals[4] = { hx/2 * xdot[idl].u + user->alpha * (x[idl].u - x[idr].u)/ hx,
                               hx/2 * xdot[idl].v + user->alpha * (x[idl].v - x[idr].v)/ hx,
                               hx/2 * xdot[idr].u + user->alpha * (x[idr].u - x[idl].u)/ hx,
                               hx/2 * xdot[idr].v + user->alpha * (x[idr].v - x[idl].v)/ hx };

      ierr = VecSetValuesBlocked(flocal, vpere, gdof, vals, ADD_VALUES);CHKERRQ(ierr);
    }
  }

  // Restore all the local vector data array references
  ierr = VecRestoreArrayRead(xltmp, (const PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xdtmp, (const PetscScalar**)&xdot);CHKERRQ(ierr);

  ierr = VecGhostRestoreLocalForm(xlocal, &xltmp);CHKERRQ(ierr);
  ierr = VecGhostRestoreLocalForm(xdotlocal, &xdtmp);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm, &xlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &xdotlocal);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(flocal);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(flocal);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(dm,flocal,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm,flocal,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &flocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FormIFunctionGhosted"
static PetscErrorCode FormIFunctionGhosted(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ptr)
{
  UserCtx             user = (UserCtx)ptr;
  Field               *x,*xdot;
  Field               *f;
  Vec                 xltmp, xdtmp, xlocal,xdotlocal, flocal,fltmp;
  PetscReal           hx;
  PetscErrorCode      ierr;
  DM                  dm;
  PetscInt           *dofs;

  PetscBool           elem_on_boundary;
  PetscInt            vpere=2;  
  const moab::EntityHandle  *connect;
  PetscInt            dof_indices[2];

  moab::Interface*  mbImpl;
  const moab::Range *elocal;

  PetscFunctionBegin;
  hx = 1.0/user->n;

  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);

  /* get the essential MOAB mesh related quantities needed for FEM assembly */
  ierr = DMMoabGetInterface(dm, &mbImpl);CHKERRQ(ierr);
  ierr = DMMoabGetLocalElements(dm, &elocal);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = VecGhostGetLocalForm(xlocal,&xltmp);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(xlocal,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(xlocal,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&xdotlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,Xdot,INSERT_VALUES,xdotlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,Xdot,INSERT_VALUES,xdotlocal);CHKERRQ(ierr);
  ierr = VecGhostGetLocalForm(xdotlocal,&xdtmp);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(xdotlocal,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(xdotlocal,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&flocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,F,INSERT_VALUES,flocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,F,INSERT_VALUES,flocal);CHKERRQ(ierr);
  ierr = VecGhostGetLocalForm(flocal,&fltmp);CHKERRQ(ierr);

  // reset the residual vector before assembly
  ierr = VecSet(fltmp, 0.0);CHKERRQ(ierr);

  ierr = DMMoabGetVertexDofsBlocked(dm, &dofs);CHKERRQ(ierr);

  ierr = VecGetArrayRead(xltmp, (const PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xdtmp, (const PetscScalar**)&xdot);CHKERRQ(ierr);
  ierr = VecGetArray(fltmp, (PetscScalar**)&f);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid 
     Assemble the operator by looping over edges and computing
     contribution for each vertex dof                         */

  const int& idl = dof_indices[0];
  const int& idr = dof_indices[1];

  /* loop over local elements */
  for(moab::Range::iterator iter = elocal->begin(); iter != elocal->end(); iter++) {
    const moab::EntityHandle ehandle = *iter;

    // Get connectivity information in canonical order
    ierr = DMMoabGetElementConnectivity(dm, ehandle, &vpere, &connect);CHKERRQ(ierr);
    if (vpere != 2) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only EDGE2 element bases are supported in the current example. Connectivity=%D.\n", vpere);
    
    ierr = DMMoabGetDofsBlockedLocal(dm, vpere, connect, dof_indices);CHKERRQ(ierr);

    /* check if element is on the boundary */
    ierr = DMMoabIsEntityOnBoundary(dm,ehandle,&elem_on_boundary);CHKERRQ(ierr);

    if (elem_on_boundary) {
      if (dofs[connect[0]] == 0) {
        f[idl].u = hx * (x[idl].u - user->leftbc.u);
        f[idl].v = hx * (x[idl].v - user->leftbc.v);
        f[idr].u += hx/2 * xdot[idr].u + user->alpha * ( x[idr].u - x[idl].u ) / hx;
        f[idr].v += hx/2 * xdot[idr].v + user->alpha * ( x[idr].v - x[idl].v ) / hx;
      }
      else {
        f[idr].u = hx * (x[idr].u - user->rightbc.u);
        f[idr].v = hx * (x[idr].v - user->rightbc.v);
        f[idl].u += hx/2 * xdot[idl].u + user->alpha * ( x[idl].u - x[idr].u ) / hx;
        f[idl].v += hx/2 * xdot[idl].v + user->alpha * ( x[idl].v - x[idr].v ) / hx;
      }
    }
    else {
      f[idl].u += hx/2 * xdot[idl].u + user->alpha * (x[idl].u - x[idr].u)/ hx;
      f[idl].v += hx/2 * xdot[idl].v + user->alpha * (x[idl].v - x[idr].v)/ hx;
      f[idr].u += hx/2 * xdot[idr].u + user->alpha * (x[idr].u - x[idl].u)/ hx;
      f[idr].v += hx/2 * xdot[idr].v + user->alpha * (x[idr].v - x[idl].v)/ hx;
    }
  }

  // Restore all the local vector data array references
  ierr = VecRestoreArrayRead(xltmp, (const PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xdtmp, (const PetscScalar**)&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(fltmp, (PetscScalar**)&f);CHKERRQ(ierr);

  ierr = VecGhostUpdateBegin(flocal,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(flocal,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  ierr = VecGhostRestoreLocalForm(xlocal, &xltmp);CHKERRQ(ierr);
  ierr = VecGhostRestoreLocalForm(xdotlocal, &xdtmp);CHKERRQ(ierr);
  ierr = VecGhostRestoreLocalForm(flocal,&fltmp);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(dm,flocal,INSERT_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm,flocal,INSERT_VALUES,F);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm, &xlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &xdotlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &flocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

