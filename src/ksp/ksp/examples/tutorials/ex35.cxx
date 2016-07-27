/*T
   Concepts: KSP^solving a system of linear equations using a MOAB based DM implementation.
   Concepts: KSP^Laplacian, 2d
   Processors: n
T*/

/*
Inhomogeneous Laplacian in 2D. Modeled by the partial differential equation

   -div \rho grad u = f,  0 < x,y < 1,

Problem 1: (Default)

  Use the forcing function

     f = e^{-((x-xr)^2+(y-yr)^2)/\nu}

  with Dirichlet boundary conditions

     u = f(x,y) for x = 0, x = 1, y = 0, y = 1

  or pure Neumman boundary conditions

Problem 2:

  Use the following exact solution with Dirichlet boundary condition

    u = sin(pi*x)*sin(pi*y)

  and generate an appropriate forcing function to measure convergence.

Usage:

  Run with different values of \rho and \nu (problem 1) to control diffusion and gaussian source spread. This uses the internal mesh generator implemented in DMMoab.

    mpiexec -n $NP ./ex35 -n 20 -nu 0.02 -rho 0.01
    mpiexec -n $NP ./ex35 -n 40 -nu 0.01 -rho 0.005 -io -ksp_monitor
    mpiexec -n $NP ./ex35 -n 80 -nu 0.01 -rho 0.005 -io -ksp_monitor -pc_type hypre
    mpiexec -n $NP ./ex35 -n 160 -bc_type neumann -nu 0.005 -rho 0.01 -io
    mpiexec -n $NP ./ex35 -n 320 -bc_type neumann -nu 0.001 -rho 1 -io

  Measure convergence rate with uniform refinement with the options: "-problem 2 -error".

    mpiexec -n $NP ./ex35 -problem 2 -error -n 16 
    mpiexec -n $NP ./ex35 -problem 2 -error -n 32
    mpiexec -n $NP ./ex35 -problem 2 -error -n 64
    mpiexec -n $NP ./ex35 -problem 2 -error -n 128
    mpiexec -n $NP ./ex35 -problem 2 -error -n 256
    mpiexec -n $NP ./ex35 -problem 2 -error -n 512

  Now, you could alternately load an external MOAB mesh that discretizes the unit square and use that to run the solver.

    mpiexec -n $NP ./ex35 -problem 1 -file ./external_mesh.h5m 
    mpiexec -n $NP ./ex35 -problem 2 -file ./external_mesh.h5m -error
*/

static char help[] = "\
                      Solves 2D inhomogeneous Laplacian equation with a Gaussian source.\n \
                      Usage: ./ex35 -bc_type dirichlet -nu .01 -n 10\n";


/* PETSc includes */
#include <petscksp.h>
#include <petscdmmoab.h>

#define LOCAL_ASSEMBLY

typedef enum {DIRICHLET, NEUMANN} BCType;

typedef struct {
  PetscInt  dim,n,problem,nlevels;
  PetscReal rho;
  PetscReal bounds[6];
  PetscReal xref,yref;
  PetscReal nu;
  PetscInt  VPERE;
  BCType    bcType;
  char filename[PETSC_MAX_PATH_LEN];
} UserContext;

extern PetscErrorCode ComputeMatrix(KSP,Mat,Mat,void*);
extern PetscErrorCode ComputeRHS(KSP,Vec,void*);
static PetscErrorCode ComputeDiscreteL2Error(KSP ksp,Vec err,UserContext *user);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  KSP             ksp;
  PC              pc;
  Mat             R;
  DM              dm,dmref,*dmhierarchy;
  
  UserContext     user;
  const char      *bcTypes[2] = {"dirichlet","neumann"};
  const char      *fields[1] = {"T-Variable"};
  PetscErrorCode  ierr;
  PetscInt        k,bc,np;
  Vec             b,x,errv;
  PetscBool       use_extfile,io,error,usesimplex,usemg;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&np);CHKERRQ(ierr);

  ierr        = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the inhomogeneous Poisson equation", "ex35.cxx");
  user.dim    = 2;
  user.problem= 1;
  ierr        = PetscOptionsInt("-problem", "The type of problem being solved (controls forcing function)", "ex35.cxx", user.problem, &user.problem, NULL);CHKERRQ(ierr);
  user.n      = 2;
  ierr        = PetscOptionsInt("-n", "The elements in each direction", "ex35.cxx", user.n, &user.n, NULL);CHKERRQ(ierr);
  user.nlevels= 2;
  ierr        = PetscOptionsInt("-levels", "Number of levels in the multigrid hierarchy", "ex36.cxx", user.nlevels, &user.nlevels, NULL);CHKERRQ(ierr);
  user.rho    = 0.1;
  ierr        = PetscOptionsReal("-rho", "The conductivity", "ex35.cxx", user.rho, &user.rho, NULL);CHKERRQ(ierr);
  user.bounds[0] = user.bounds[2] = user.bounds[4] = 0.0;
  user.bounds[1] = user.bounds[3] = user.bounds[5] = 1.0;
  ierr        = PetscOptionsReal("-x", "The domain size in x-direction", "ex35.cxx", user.bounds[1], &user.bounds[1], NULL);CHKERRQ(ierr);
  ierr        = PetscOptionsReal("-y", "The domain size in y-direction", "ex35.cxx", user.bounds[3], &user.bounds[3], NULL);CHKERRQ(ierr);
  user.xref   = user.bounds[1]/2;
  ierr        = PetscOptionsReal("-xref", "The x-coordinate of Gaussian center (for -problem 1)", "ex35.cxx", user.xref, &user.xref, NULL);CHKERRQ(ierr);
  user.yref   = user.bounds[3]/2;
  ierr        = PetscOptionsReal("-yref", "The y-coordinate of Gaussian center (for -problem 1)", "ex35.cxx", user.yref, &user.yref, NULL);CHKERRQ(ierr);
  user.nu     = 0.05;
  ierr        = PetscOptionsReal("-nu", "The width of the Gaussian source (for -problem 1)", "ex35.cxx", user.nu, &user.nu, NULL);CHKERRQ(ierr);
  usemg       = PETSC_FALSE;
  ierr        = PetscOptionsBool("-mg", "Use multigrid preconditioner", "ex36.cxx", usemg, &usemg, NULL);CHKERRQ(ierr);
  io          = PETSC_FALSE;
  ierr        = PetscOptionsBool("-io", "Write out the solution and mesh data", "ex35.cxx", io, &io, NULL);CHKERRQ(ierr);
  usesimplex  = PETSC_FALSE;
  ierr        = PetscOptionsBool("-tri", "Use triangles to discretize the domain", "ex35.cxx", usesimplex, &usesimplex, NULL);CHKERRQ(ierr);
  error       = PETSC_FALSE;
  ierr        = PetscOptionsBool("-error", "Compute the discrete L_2 and L_inf errors of the solution", "ex35.cxx", error, &error, NULL);CHKERRQ(ierr);
  bc          = (PetscInt)DIRICHLET;
  ierr        = PetscOptionsEList("-bc_type","Type of boundary condition","ex35.cxx",bcTypes,2,bcTypes[0],&bc,NULL);CHKERRQ(ierr);
  user.bcType = (BCType)bc;
  ierr        = PetscOptionsString("-file", "The mesh file for the problem", "ex35.cxx", "",user.filename,PETSC_MAX_PATH_LEN,&use_extfile);CHKERRQ(ierr);
  ierr        = PetscOptionsEnd();

  user.VPERE  = (usesimplex ? 3:4);

  /* Create the DM object from either a mesh file or from in-memory structured grid */
  if (use_extfile) {
    ierr = DMMoabLoadFromFile(PETSC_COMM_WORLD, user.dim, 1, user.filename, (np==1 ? "" : ""), &dm);CHKERRQ(ierr);
  }
  else {
    ierr = DMMoabCreateBoxMesh(PETSC_COMM_WORLD, user.dim, usesimplex, user.bounds, user.n, 1, &dm);CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMMoabSetFieldNames(dm, 1, fields);CHKERRQ(ierr);

  /* SetUp the data structures for DMMOAB */
  ierr = DMSetUp(dm);CHKERRQ(ierr);

  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetComputeRHS(ksp,ComputeRHS,&user);CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp,ComputeMatrix,&user);CHKERRQ(ierr);

  if (user.nlevels)
  {
    KSPGetPC(ksp,&pc);
    ierr = PetscMalloc(sizeof(DM)*(user.nlevels+1),&dmhierarchy);
    for (k=0; k<=user.nlevels; k++) dmhierarchy[k] = NULL;

    PetscPrintf(PETSC_COMM_WORLD, "Number of levels: %d\n", user.nlevels);
    ierr = DMMoabGenerateHierarchy(dm,user.nlevels,PETSC_NULL);CHKERRQ(ierr);

    // coarsest grid = 0
    // finest grid = nlevels
    dmhierarchy[0] = dm;
    PetscBool usehierarchy=PETSC_FALSE;
    if (usehierarchy) {
      ierr = DMRefineHierarchy(dm,user.nlevels,&dmhierarchy[1]);CHKERRQ(ierr);
    }
    else {
      PetscPrintf(PETSC_COMM_WORLD, "Level %D\n", 0);
      for (k=1; k<=user.nlevels; k++) {
        PetscPrintf(PETSC_COMM_WORLD, "Level %D\n", k);
        ierr = DMRefine(dmhierarchy[k-1],MPI_COMM_NULL,&dmhierarchy[k]);CHKERRQ(ierr);
      }
    }
    dmref = dmhierarchy[user.nlevels];
    PetscObjectReference((PetscObject)dmref);

    if (usemg) {
      ierr = PCSetType(pc,PCMG);CHKERRQ(ierr);
      ierr = PCMGSetType(pc,PC_MG_MULTIPLICATIVE);CHKERRQ(ierr);
      ierr = PCMGSetLevels(pc,user.nlevels+1,NULL);CHKERRQ(ierr);
      ierr = PCMGSetGalerkin(pc,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PCMGSetCycleType(pc, PC_MG_CYCLE_V);CHKERRQ(ierr);
      ierr = PCMGSetNumberSmoothUp(pc,2);CHKERRQ(ierr);
      ierr = PCMGSetNumberSmoothDown(pc,2);CHKERRQ(ierr);

      for (k=1; k<=user.nlevels; k++) {
        ierr = DMCreateInterpolation(dmhierarchy[k-1],dmhierarchy[k],&R,NULL);CHKERRQ(ierr);
        ierr = PCMGSetInterpolation(pc,k,R);CHKERRQ(ierr);
        //ierr = PCMGSetRestriction(pc,k,R);CHKERRQ(ierr);
        ierr = MatDestroy(&R);CHKERRQ(ierr);
      }
    }

    for (k=1; k<=user.nlevels; k++) {
      ierr = DMDestroy(&dmhierarchy[k]);CHKERRQ(ierr);
    }
    ierr = PetscFree(dmhierarchy);CHKERRQ(ierr);
  }
  else {
    dmref = dm;
    PetscObjectReference((PetscObject)dmref);
  }

  ierr = KSPSetDM(ksp,dmref);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* Perform the actual solve */
  ierr = KSPSolve(ksp,NULL,NULL);CHKERRQ(ierr);
  ierr = KSPGetSolution(ksp,&x);CHKERRQ(ierr);
  ierr = KSPGetRhs(ksp,&b);CHKERRQ(ierr);

  if (error) {
    ierr = DMGetGlobalVector(dmref, &errv);CHKERRQ(ierr);
    ierr = ComputeDiscreteL2Error(ksp, errv, &user);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dmref, &errv);CHKERRQ(ierr);
  }

  if (io) {
    /* Write out the solution along with the mesh */
    ierr = DMMoabSetGlobalFieldVector(dmref, x);CHKERRQ(ierr);
#ifdef MOAB_HAVE_HDF5
    ierr = DMMoabOutput(dmref, "ex35.h5m", NULL);CHKERRQ(ierr);
#else
    /* MOAB does not support true parallel writers that aren't HDF5 based
       And so if you are using VTK as the output format in parallel,
       the data could be jumbled due to the order in which the processors
       write out their parts of the mesh and solution tags */
    ierr = DMMoabOutput(dmref, "ex35.vtk", NULL);CHKERRQ(ierr);
#endif
  }

  /* Cleanup objects */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = DMDestroy(&dmref);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRho"
PetscScalar ComputeRho(PetscReal coords[3], UserContext* user)
{
  switch(user->problem) {
    case 2:
      return user->rho;
    case 1:
    default:
      if ((coords[0] > user->bounds[1]/3.0) && (coords[0] < 2.0*user->bounds[1]/3.0) && (coords[1] > user->bounds[3]/3.0) && (coords[1] < 2.0*user->bounds[3]/3.0)) {
        return user->rho;
      } else {
        return 1.0;
      }
  }
}

#undef __FUNCT__
#define __FUNCT__ "ExactSolution"
PetscScalar ExactSolution(PetscReal coords[3], UserContext* user)
{
  switch(user->problem) {
    case 2:
      return sin(PETSC_PI*coords[0]/user->bounds[1])*sin(PETSC_PI*coords[1]/user->bounds[3]);
    case 1:
    default:
      SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Exact solution for -problem = [%D] is not available.\n", user->problem);
  }
}

#undef __FUNCT__
#define __FUNCT__ "ComputeForcingFunction"
PetscScalar ComputeForcingFunction(PetscReal coords[3], UserContext* user)
{
  switch(user->problem) {
    case 2:
      return PETSC_PI*PETSC_PI*ComputeRho(coords, user)*(1.0/user->bounds[1]/user->bounds[1]+1.0/user->bounds[3]/user->bounds[3])*sin(PETSC_PI*coords[0]/user->bounds[1])*sin(PETSC_PI*coords[1]/user->bounds[3]);
    case 1:
    default:
      const PetscScalar xx=(coords[0]-user->xref)*(coords[0]-user->xref);
      const PetscScalar yy=(coords[1]-user->yref)*(coords[1]-user->yref);
      return PetscExpScalar(-(xx+yy)/user->nu);
  }
}


#undef __FUNCT__
#define __FUNCT__ "ComputeRHS"
PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ptr)
{
  UserContext*      user = (UserContext*)ptr;
  DM                dm;
  PetscInt          dof_indices[user->VPERE];
  PetscBool         dbdry[user->VPERE];
  PetscReal         vpos[user->VPERE*3];
  PetscScalar       ff;
  PetscInt          i,q,num_conn,npoints;
  const moab::EntityHandle *connect;
  const moab::Range *elocal;
  moab::Interface*  mbImpl;
  PetscScalar       localv[user->VPERE];
  PetscReal         *phi,*phypts,*jxw;
  PetscBool         elem_on_boundary;
  PetscQuadrature   quadratureObj;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = KSPGetDM(ksp,&dm);CHKERRQ(ierr);

  /* reset the RHS */
  ierr = VecSet(b, 0.0);CHKERRQ(ierr);

  ierr = DMMoabFEMCreateQuadratureDefault (2, user->VPERE, &quadratureObj);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quadratureObj, NULL, &npoints, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscMalloc3(user->VPERE*npoints,&phi, npoints*3,&phypts, npoints,&jxw);CHKERRQ(ierr);

  /* get the essential MOAB mesh related quantities needed for FEM assembly */
  ierr = DMMoabGetInterface(dm, &mbImpl);CHKERRQ(ierr);
  ierr = DMMoabGetLocalElements(dm, &elocal);CHKERRQ(ierr);

  /* loop over local elements */
  for(moab::Range::iterator iter = elocal->begin(); iter != elocal->end(); iter++) {
    const moab::EntityHandle ehandle = *iter;

    ierr = PetscMemzero(localv,sizeof(PetscScalar)*user->VPERE);CHKERRQ(ierr);

    // Get connectivity information:
    ierr = DMMoabGetElementConnectivity(dm, ehandle, &num_conn, &connect);CHKERRQ(ierr);
    if (num_conn != user->VPERE) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only TRI3/QUAD4 element bases are supported in the current example. n(Connectivity)=%D.\n", num_conn);

    /* compute the mid-point of the element and use a 1-point lumped quadrature */
    ierr = DMMoabGetVertexCoordinates(dm,num_conn,connect,vpos);CHKERRQ(ierr);

    /* get the global DOF number to appropriately set the element contribution in the RHS vector */
#ifdef LOCAL_ASSEMBLY
    ierr = DMMoabGetFieldDofsLocal(dm, num_conn, connect, 0, dof_indices);CHKERRQ(ierr);
#else
    ierr = DMMoabGetFieldDofs(dm, num_conn, connect, 0, dof_indices);CHKERRQ(ierr);    
#endif

    /* 1) compute the basis functions and the derivatives wrt x and y directions
       2) compute the quadrature points transformed to the physical space */
    ierr = DMMoabFEMComputeBasis(2, user->VPERE, vpos, quadratureObj, phypts, jxw, phi, 0);CHKERRQ(ierr);

    /* Compute function over the locally owned part of the grid */
    for (q=0; q<npoints; ++q) {
      ff = ComputeForcingFunction(&phypts[3*q], user);

      for (i=0; i < user->VPERE; ++i) {
        localv[i] += jxw[q] * phi[q*user->VPERE+i] * ff;
      }
    }

    /* check if element is on the boundary */
    ierr = DMMoabIsEntityOnBoundary(dm,ehandle,&elem_on_boundary);CHKERRQ(ierr);

    /* apply dirichlet boundary conditions */
    if (elem_on_boundary && user->bcType == DIRICHLET) {

      /* get the list of nodes on boundary so that we can enforce dirichlet conditions strongly */
      ierr = DMMoabCheckBoundaryVertices(dm,num_conn,connect,dbdry);CHKERRQ(ierr);

      for (i=0; i < user->VPERE; ++i) {
        if (dbdry[i]) {  /* dirichlet node */
          /* think about strongly imposing dirichlet */
          localv[i] = ComputeForcingFunction(&vpos[3*i], user);
        }
      }
    }

#ifdef LOCAL_ASSEMBLY
    /* set the values directly into appropriate locations. Can alternately use VecSetValues */
    ierr = VecSetValuesLocal(b, user->VPERE, dof_indices, localv, ADD_VALUES);CHKERRQ(ierr);
#else
    ierr = VecSetValues(b, user->VPERE, dof_indices, localv, ADD_VALUES);CHKERRQ(ierr);
#endif
  }

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;
    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nullspace,b);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  }

  /* Restore vectors */
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  ierr = PetscFree3(phi,phypts,jxw);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&quadratureObj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeMatrix"
PetscErrorCode ComputeMatrix(KSP ksp,Mat J,Mat jac,void *ctx)
{
  UserContext       *user = (UserContext*)ctx;
  DM                dm;
  PetscInt          i,j,q,num_conn,nglobale,nglobalv,npoints,hlevel;
  PetscInt          dof_indices[user->VPERE];
  PetscReal         vpos[user->VPERE*3],rho;
  PetscBool         dbdry[user->VPERE];
  const moab::EntityHandle *connect;
  const moab::Range *elocal;
  moab::Interface*  mbImpl;
  PetscBool         elem_on_boundary;
  PetscScalar       array[user->VPERE*user->VPERE];
  PetscReal         *phi, *dphi[2], *phypts, *jxw;
  PetscQuadrature   quadratureObj;
  PetscErrorCode    ierr;
 
  PetscFunctionBeginUser;
  ierr = KSPGetDM(ksp,&dm);CHKERRQ(ierr);

  /* get the essential MOAB mesh related quantities needed for FEM assembly */  
  ierr = DMMoabGetInterface(dm, &mbImpl);CHKERRQ(ierr);
  ierr = DMMoabGetLocalElements(dm, &elocal);CHKERRQ(ierr);
  ierr = DMMoabGetSize(dm, &nglobale, &nglobalv);CHKERRQ(ierr);
  ierr = DMMoabGetHierarchyLevel(dm, &hlevel);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "ComputeMatrix: Level = %d, N(elements) = %d, N(vertices) = %d \n", hlevel, nglobale, nglobalv);

  ierr = DMMoabFEMCreateQuadratureDefault ( 2, user->VPERE, &quadratureObj );CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quadratureObj, NULL, &npoints, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscMalloc5(user->VPERE*npoints,&phi, user->VPERE*npoints,&dphi[0], user->VPERE*npoints,&dphi[1], npoints*3,&phypts, npoints,&jxw);CHKERRQ(ierr);

  /* loop over local elements */
  for(moab::Range::iterator iter = elocal->begin(); iter != elocal->end(); iter++) {
    const moab::EntityHandle ehandle = *iter;

    // Get connectivity information:
    ierr = DMMoabGetElementConnectivity(dm, ehandle, &num_conn, &connect);CHKERRQ(ierr);
    if (num_conn != user->VPERE) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only QUAD4 or TRI3 element bases are supported in the current example. Connectivity=%D.\n", num_conn);

    /* compute the mid-point of the element and use a 1-point lumped quadrature */
    ierr = DMMoabGetVertexCoordinates(dm,num_conn,connect,vpos);CHKERRQ(ierr);

    /* get the global DOF number to appropriately set the element contribution in the RHS vector */
#ifdef LOCAL_ASSEMBLY
    ierr = DMMoabGetFieldDofsLocal(dm, num_conn, connect, 0, dof_indices);CHKERRQ(ierr);
#else
    ierr = DMMoabGetFieldDofs(dm, num_conn, connect, 0, dof_indices);CHKERRQ(ierr);
#endif

    /* 1) compute the basis functions and the derivatives wrt x and y directions
       2) compute the quadrature points transformed to the physical space */
    ierr = DMMoabFEMComputeBasis(2, user->VPERE, vpos, quadratureObj, phypts, jxw, phi, dphi);CHKERRQ(ierr);

    ierr = PetscMemzero(array, user->VPERE*user->VPERE*sizeof(PetscScalar));

    /* Compute function over the locally owned part of the grid */
    for (q=0; q<npoints; ++q) {
      /* compute the inhomogeneous diffusion coefficient at the first quadrature point 
         -- for large spatial variations, embed this property evaluation inside quadrature loop */
      rho = ComputeRho(&phypts[q*3], user);

      for (i=0; i < user->VPERE; ++i) {
        for (j=0; j < user->VPERE; ++j) {
          array[i*user->VPERE+j] += jxw[q] * rho * ( dphi[0][q*user->VPERE+i]*dphi[0][q*user->VPERE+j] + 
                                                     dphi[1][q*user->VPERE+i]*dphi[1][q*user->VPERE+j] );
        }
      }
    }

    /* check if element is on the boundary */
    ierr = DMMoabIsEntityOnBoundary(dm,ehandle,&elem_on_boundary);CHKERRQ(ierr);

    /* apply dirichlet boundary conditions */
    if (elem_on_boundary && user->bcType == DIRICHLET) {

      /* get the list of nodes on boundary so that we can enforce dirichlet conditions strongly */
      ierr = DMMoabCheckBoundaryVertices(dm,num_conn,connect,dbdry);CHKERRQ(ierr);

      for (i=0; i < user->VPERE; ++i) {
        if (dbdry[i]) {  /* dirichlet node */
          /* think about strongly imposing dirichlet */
          for (j=0; j < user->VPERE; ++j) {
            /* TODO: symmetrize the system - need the RHS */
            array[i*user->VPERE+j] = 0.0;
          }
          array[i*user->VPERE+i] = 1.0;
        }
      }
    }

    /* set the values directly into appropriate locations. */
#ifdef LOCAL_ASSEMBLY
    ierr = MatSetValuesLocal(jac, user->VPERE, dof_indices, user->VPERE, dof_indices, array, ADD_VALUES);CHKERRQ(ierr);
#else
    ierr = MatSetValues(jac, user->VPERE, dof_indices, user->VPERE, dof_indices, array, ADD_VALUES);CHKERRQ(ierr);
#endif
  }

  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
    ierr = MatSetNullSpace(J,nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  }
  ierr = PetscFree5(phi,dphi[0],dphi[1],phypts,jxw);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&quadratureObj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ComputeDiscreteL2Error"
PetscErrorCode ComputeDiscreteL2Error(KSP ksp,Vec err,UserContext *user)
{
  DM                dm;
  Vec               sol;
  PetscScalar       vpos[3];
  const PetscScalar *x;
  PetscScalar       *e;
  PetscReal         l2err=0.0,linferr=0.0,global_l2,global_linf;
  PetscInt          dof_index,N;
  const moab::Range *ownedvtx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = KSPGetDM(ksp,&dm);CHKERRQ(ierr);

  /* get the solution vector */
  ierr = KSPGetSolution(ksp, &sol);CHKERRQ(ierr);

  /* Get the internal reference to the vector arrays */
  ierr = DMMoabVecGetArrayRead(dm, sol, &x);CHKERRQ(ierr);
  ierr = VecGetSize(sol, &N);CHKERRQ(ierr);
  if (err) {
    /* reset the error vector */
    ierr = VecSet(err, 0.0);CHKERRQ(ierr);
    /* get array reference */
    ierr = DMMoabVecGetArray(dm, err, &e);CHKERRQ(ierr);
  }

  ierr = DMMoabGetLocalVertices(dm, &ownedvtx, NULL);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for(moab::Range::iterator iter = ownedvtx->begin(); iter != ownedvtx->end(); iter++) {
    const moab::EntityHandle vhandle = *iter;
    ierr = DMMoabGetDofsBlockedLocal(dm, 1, &vhandle, &dof_index);CHKERRQ(ierr);

    /* compute the mid-point of the element and use a 1-point lumped quadrature */
    ierr = DMMoabGetVertexCoordinates(dm,1,&vhandle,vpos);CHKERRQ(ierr);

    /* compute the discrete L2 error against the exact solution */
    const PetscScalar lerr = (ExactSolution(vpos, user) - x[dof_index]);
    l2err += lerr*lerr;
    if (linferr<fabs(lerr))
      linferr=fabs(lerr);

    if (err) { /* set the discrete L2 error against the exact solution */
      e[dof_index] = lerr;
    }
  }

  ierr = MPI_Allreduce(&l2err, &global_l2, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&linferr, &global_linf, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Computed Errors: L_2 = %f, L_inf = %f\n", sqrt(global_l2/N), global_linf);CHKERRQ(ierr);

  /* Restore vectors */
  ierr = DMMoabVecRestoreArrayRead(dm, sol, &x);CHKERRQ(ierr);
  if (err) {
    ierr = DMMoabVecRestoreArray(dm, err, &e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

