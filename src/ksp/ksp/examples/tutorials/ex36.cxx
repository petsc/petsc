/*T
   Concepts: KSP^solving a system of linear equations using a MOAB based DM implementation.
   Concepts: KSP^Laplacian, 3d
   Processors: n
T*/

/*
Inhomogeneous Laplacian in 3-D. Modeled by the partial differential equation

   -div \rho grad u = f,  0 < x,y < 1,

with forcing function

   f = e^{-((x-xr)^2+(y-yr)^2)/\nu}

with Dirichlet boundary conditions

   u = f(x,y) for x = 0, x = 1, y = 0, y = 1

or pure Neumman boundary conditions

Usage:
    ./ex36 -bc_type dirichlet -nu .01 -rho .01 -file input/quad_2p.h5m -dmmb_rw_dbg 0 -n 50

*/

static char help[] = "\
                      Solves a three dimensional inhomogeneous Laplacian equation with a Gaussian source.\n \
                      Usage: ./ex36 -bc_type dirichlet -nu .01 -n 10\n";

#define PROBLEM 2

/* PETSc includes */
#include <petscksp.h>
#include <petscdmmoab.h>

static PetscErrorCode ComputeMatrix_MOAB(KSP,Mat,Mat,void*);
static PetscErrorCode ComputeRHS_MOAB(KSP,Vec,void*);

static PetscErrorCode Compute_Basis ( int nverts, PetscReal *coords/*nverts*3*/, PetscInt npts, PetscReal *quad/*npts*3*/, PetscReal *pts/*npts*3*/, 
        PetscReal *jxw/*npts*/, PetscReal *phi/*npts*/, PetscReal *dphidx/*npts*/, PetscReal *dphidy/*npts*/, PetscReal *dphidz/*npts*/);

typedef enum {DIRICHLET, NEUMANN} BCType;

typedef struct {
  PetscInt  dim,n,nlevels;
  PetscReal rho;
  PetscReal xyzref[3];
  PetscReal nu;
  BCType    bcType;
  char      filename[PETSC_MAX_PATH_LEN];

  /* Discretization parameters */
  int NQPTS,VPERE;
} UserContext;

static PetscErrorCode ComputeDiscreteL2Error(KSP ksp,Vec err,UserContext *user);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  const char    *bcTypes[2] = {"dirichlet","neumann"};
  const char    *fields[1] = {"T-Variable"};
  DM             dm,dmref,*dmhierarchy;
  UserContext    user;
  PetscInt       k,bc,np;
  KSP            ksp;
  PC             pc;
  Vec            errv;
  Mat            R;
  Vec            b,x;
  PetscBool      use_extfile,io,error,usesimplex,usemg;
  PetscErrorCode ierr;

  PetscInitialize(&argc,&argv,(char*)0,help);

  MPI_Comm_size(PETSC_COMM_WORLD,&np);

  ierr        = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the inhomogeneous Poisson equation", "ex36.cxx");
  user.dim    = 3;
  //ierr        = PetscOptionsInt("-dim", "The dimension of the problem", "ex36.cxx", user.dim, &user.dim, NULL);CHKERRQ(ierr);
  user.n      = 2;
  ierr        = PetscOptionsInt("-n", "The elements in each direction", "ex36.cxx", user.n, &user.n, NULL);CHKERRQ(ierr);
  user.nlevels= 2;
  ierr        = PetscOptionsInt("-levels", "Number of levels in the multigrid hierarchy", "ex36.cxx", user.nlevels, &user.nlevels, NULL);CHKERRQ(ierr);
  user.rho    = 1.0;
  ierr        = PetscOptionsReal("-rho", "The conductivity", "ex36.cxx", user.rho, &user.rho, NULL);CHKERRQ(ierr);
  user.xyzref[0]   = user.xyzref[1]   = user.xyzref[2]   = 0.5;
  ierr        = PetscOptionsReal("-xyzref", "The x-coordinate of Gaussian center", "ex36.cxx", user.xyzref[0], &user.xyzref[0], NULL);CHKERRQ(ierr);
  user.xyzref[1]   = user.xyzref[2]   = user.xyzref[0];
  user.nu     = 0.1;
  ierr        = PetscOptionsReal("-nu", "The width of the Gaussian source", "ex36.cxx", user.nu, &user.nu, NULL);CHKERRQ(ierr);
  io          = PETSC_FALSE;
  usemg       = PETSC_FALSE;
  ierr        = PetscOptionsBool("-mg", "Use multigrid preconditioner", "ex36.cxx", usemg, &usemg, NULL);CHKERRQ(ierr);
  ierr        = PetscOptionsBool("-io", "Write out the solution and mesh data", "ex36.cxx", io, &io, NULL);CHKERRQ(ierr);
  error       = PETSC_FALSE;
  ierr        = PetscOptionsBool("-error", "Compute the discrete L_2 and L_inf errors of the solution", "ex35.cxx", error, &error, NULL);CHKERRQ(ierr);
  bc          = (PetscInt)DIRICHLET;
  ierr        = PetscOptionsEList("-bc_type","Type of boundary condition","ex36.cxx",bcTypes,2,bcTypes[0],&bc,NULL);CHKERRQ(ierr);
  user.bcType = (BCType)bc;
  ierr        = PetscOptionsString("-file", "The mesh file for the problem", "ex36.cxx", "",user.filename,PETSC_MAX_PATH_LEN,&use_extfile);CHKERRQ(ierr);

  usesimplex=PETSC_FALSE;
  ierr        = PetscOptionsBool("-tet", "Use tetrahedra to discretize the unit cube domain", "ex36.cxx", usesimplex, &usesimplex, NULL);CHKERRQ(ierr);
  ierr        = PetscOptionsEnd();

  user.NQPTS  = (usesimplex ? 4:8);
  user.VPERE  = (usesimplex ? 4:8);

  /* Create the DM object from either a mesh file or from in-memory structured grid */
  if (use_extfile) {
    ierr = DMMoabLoadFromFile(PETSC_COMM_WORLD, user.dim, 1, user.filename, (np==1 ? "" : ""), &dm);CHKERRQ(ierr);
  }
  else {
    ierr = DMMoabCreateBoxMesh(PETSC_COMM_WORLD, user.dim, usesimplex, NULL, user.n, 1, &dm);CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMMoabSetFieldNames(dm, 1, fields);CHKERRQ(ierr);

  /* SetUp the data structures for DMMOAB */
  ierr = DMSetUp(dm);CHKERRQ(ierr);

  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetComputeRHS(ksp,ComputeRHS_MOAB,&user);CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp,ComputeMatrix_MOAB,&user);CHKERRQ(ierr);

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

    ierr = KSPSetDM(ksp,dmref);CHKERRQ(ierr);
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
        //MatView(R,0);
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
    ierr = KSPSetDM(ksp,dmref);CHKERRQ(ierr);
  }

  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* Perform the actual solve */
  ierr = KSPSolve(ksp,NULL,NULL);CHKERRQ(ierr);
  ierr = KSPGetSolution(ksp,&x);CHKERRQ(ierr);
  ierr = KSPGetRhs(ksp,&b);CHKERRQ(ierr);

  //VecView(b,0);
  //VecView(x,0);

  if (error) {
    ierr = DMGetGlobalVector(dmref, &errv);CHKERRQ(ierr);
    ierr = ComputeDiscreteL2Error(ksp, errv, &user);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dmref, &errv);CHKERRQ(ierr);
  }

  if (io) {
    /* Write out the solution along with the mesh */
    ierr = DMMoabSetGlobalFieldVector(dmref, x);CHKERRQ(ierr);
#ifdef MOAB_HAVE_HDF5
    ierr = DMMoabOutput(dmref, "ex36.h5m", "");CHKERRQ(ierr);
#else
    /* MOAB does not support true parallel writers that aren't HDF5 based
       And so if you are using VTK as the output format in parallel,
       the data could be jumbled due to the order in which the processors
       write out their parts of the mesh and solution tags
    */
    ierr = DMMoabOutput(dmref, "ex36.vtk", "");CHKERRQ(ierr);
#endif
  }

  /* Cleanup objects */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = DMDestroy(&dmref);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}


#undef __FUNCT__
#define __FUNCT__ "ComputeDiffusionCoefficient"
PetscReal ComputeDiffusionCoefficient(PetscReal coords[3], UserContext* user)
{
#if (PROBLEM == 1)
  if ((coords[0] > 1.0/3.0) && (coords[0] < 2.0/3.0) && 
      (coords[1] > 1.0/3.0) && (coords[1] < 2.0/3.0) && 
      (coords[2] > 1.0/3.0) && (coords[2] < 2.0/3.0))
  {
    return user->rho;
  }
  else
    return 1.0;
#else
  return 1.0;
#endif
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRho_MOAB"
double ExactSolution(PetscReal coords[3], UserContext* user)
{
#if (PROBLEM == 1)
  const PetscScalar xx=(coords[0]-user->xyzref[0])*(coords[0]-user->xyzref[0]);
  const PetscScalar yy=(coords[1]-user->xyzref[1])*(coords[1]-user->xyzref[1]);
  const PetscScalar zz=(coords[2]-user->xyzref[2])*(coords[2]-user->xyzref[2]);
  return PetscExpScalar(-(xx+yy+zz)/user->nu);
#else
  return sin(PETSC_PI*coords[0])*sin(PETSC_PI*coords[1])*sin(PETSC_PI*coords[2]);
#endif
}

PetscReal exact_solution(PetscReal x, PetscReal y, PetscReal z)
{
  PetscReal coords[3] = {x,y,z};
  return ExactSolution(coords,0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRho_MOAB"
double ForcingFunction(PetscReal coords[3], UserContext* user)
{
  const PetscReal exact = ExactSolution(coords, user);
#if (PROBLEM == 1)
  return exact;
#else
  /*
  const PetscReal eps = 1.e-6;

  const PetscReal uxx = (exact_solution(coords[0]-eps,coords[1],coords[2]) +
                    exact_solution(coords[0]+eps,coords[1],coords[2]) +
                    -2.*exact)/eps/eps;

  const PetscReal uyy = (exact_solution(coords[0],coords[1]-eps,coords[2]) +
                    exact_solution(coords[0],coords[1]+eps,coords[2]) +
                    -2.*exact)/eps/eps;

  const PetscReal uzz = (exact_solution(coords[0],coords[1],coords[2]-eps) +
                    exact_solution(coords[0],coords[1],coords[2]+eps) +
                    -2.*exact)/eps/eps;
  return - (uxx + uyy + uzz);
  */
  //return 1.0;
  return 3.0*PETSC_PI*PETSC_PI*exact;
#endif
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRHS_MOAB"
PetscErrorCode ComputeRHS_MOAB(KSP ksp,Vec b,void *ptr)
{
  UserContext*      user = (UserContext*)ptr;
  DM                dm;
  PetscInt          dof_indices[user->VPERE];
  PetscBool         dbdry[user->VPERE];
  PetscReal         vpos[user->VPERE*3],quadrature[user->NQPTS*3],phypts[user->NQPTS*3];
  PetscReal         jxw[user->NQPTS],phi[user->VPERE*user->NQPTS];
  PetscInt          i,q,num_conn;
  const moab::EntityHandle *connect;
  const moab::Range *elocal;
  moab::Interface*  mbImpl;
  PetscScalar       localv[user->VPERE];
  PetscBool         elem_on_boundary;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = KSPGetDM(ksp,&dm);CHKERRQ(ierr);

  /* reset the RHS */
  ierr = VecSet(b, 0.0);CHKERRQ(ierr);

  /* get the essential MOAB mesh related quantities needed for FEM assembly */
  ierr = DMMoabGetInterface(dm, &mbImpl);CHKERRQ(ierr);
  ierr = DMMoabGetLocalElements(dm, &elocal);CHKERRQ(ierr);

  /* loop over local elements */
  for(moab::Range::iterator iter = elocal->begin(); iter != elocal->end(); iter++) {
    const moab::EntityHandle ehandle = *iter;

    ierr = PetscMemzero(localv,sizeof(PetscScalar)*user->VPERE);CHKERRQ(ierr);

    // Get connectivity information:
    ierr = DMMoabGetElementConnectivity(dm, ehandle, &num_conn, &connect);CHKERRQ(ierr);
    if (num_conn != user->VPERE) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only HEX8 element bases are supported in the current example. n(Connectivity)=%D.\n", num_conn);

    /* compute the mid-point of the element and use a 1-point lumped quadrature */
    ierr = DMMoabGetVertexCoordinates(dm,num_conn,connect,vpos);CHKERRQ(ierr);

    /* get the global DOF number to appropriately set the element contribution in the RHS vector */
    ierr = DMMoabGetFieldDofsLocal(dm, num_conn, connect, 0, dof_indices);CHKERRQ(ierr);

    /* compute the quadrature points transformed to the physical space and then
       compute the basis functions to compute local operators */
    ierr = Compute_Basis(user->VPERE, vpos, user->NQPTS, quadrature, phypts, jxw, phi, 0, 0, 0);CHKERRQ(ierr);

    /* Compute function over the locally owned part of the grid */
    for (q=0; q<user->NQPTS; ++q) {
      const double ff = ForcingFunction(&phypts[3*q], user);
      for (i=0; i < user->VPERE; ++i) {
        localv[i] += jxw[q] * phi[q*user->VPERE+i] * ff;
      }
    }

    /* check if element is on the boundary */
    ierr = DMMoabIsEntityOnBoundary(dm,ehandle,&elem_on_boundary);CHKERRQ(ierr);
    //ierr = DMMoabCheckBoundaryVertices(dm,num_conn,connect,dbdry);CHKERRQ(ierr);
    //elem_on_boundary=PETSC_FALSE;
    //for (i=0; i< user->VPERE; ++i)
    //  if (dbdry[i]) elem_on_boundary=PETSC_TRUE;

    /* apply dirichlet boundary conditions */
    if (elem_on_boundary && user->bcType == DIRICHLET) {

      /* get the list of nodes on boundary so that we can enforce dirichlet conditions strongly */
      ierr = DMMoabCheckBoundaryVertices(dm,num_conn,connect,dbdry);CHKERRQ(ierr);

      for (i=0; i < user->VPERE; ++i) {
        if (dbdry[i]) {  /* dirichlet node */
          /* think about strongly imposing dirichlet */
          localv[i] = ForcingFunction(&vpos[3*i], user);
        }
      }
    }
#if 0
    PetscPrintf(PETSC_COMM_WORLD,"Local vector %d\n\t", ehandle);
    for (i=0; i < user->VPERE; ++i) {
      PetscPrintf(PETSC_COMM_WORLD,"%g ", localv[i]);
    }
    PetscPrintf(PETSC_COMM_WORLD,"\n");
#endif
    /* set the values directly into appropriate locations. Can alternately use VecSetValues */
    ierr = VecSetValuesLocal(b, user->VPERE, dof_indices, localv, ADD_VALUES);CHKERRQ(ierr);
  }

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  if (user->bcType == NEUMANN && false) {
    MatNullSpace nullspace;

    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nullspace,b);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  }

  /* Restore vectors */
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  // VecView(b,0);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ComputeMatrix_MOAB"
PetscErrorCode ComputeMatrix_MOAB(KSP ksp,Mat J,Mat jac,void *ctx)
{
  UserContext       *user = (UserContext*)ctx;
  DM                dm;
  PetscInt          i,j,q,num_conn;
  PetscInt          dof_indices[user->VPERE];
  PetscReal         vpos[user->VPERE*3],quadrature[user->NQPTS*3],phypts[user->NQPTS*3];
  PetscBool         dbdry[user->VPERE];
  const moab::EntityHandle *connect;
  const moab::Range *elocal;
  moab::Interface*  mbImpl;
  PetscBool         elem_on_boundary;
  PetscScalar       array[user->VPERE*user->VPERE];
  PetscReal         jxw[user->NQPTS],phi[user->VPERE*user->NQPTS];
  PetscReal         dphidx[user->VPERE*user->NQPTS], dphidy[user->VPERE*user->NQPTS], dphidz[user->VPERE*user->NQPTS];
  PetscReal         rho;
  PetscErrorCode    ierr;
 
  PetscFunctionBeginUser;
  ierr      = KSPGetDM(ksp,&dm);CHKERRQ(ierr);

  /* get the essential MOAB mesh related quantities needed for FEM assembly */  
  ierr = DMMoabGetInterface(dm, &mbImpl);CHKERRQ(ierr);
  ierr = DMMoabGetLocalElements(dm, &elocal);CHKERRQ(ierr);

  /* loop over local elements */
  for(moab::Range::iterator iter = elocal->begin(); iter != elocal->end(); iter++) {
    const moab::EntityHandle ehandle = *iter;

    // Get connectivity information:
    ierr = DMMoabGetElementConnectivity(dm, ehandle, &num_conn, &connect);CHKERRQ(ierr);
    if (num_conn != user->VPERE) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only HEX8 element bases are supported in the current example. Connectivity=%D.\n", num_conn);

    /* compute the mid-point of the element and use a 1-point lumped quadrature */
    ierr = DMMoabGetVertexCoordinates(dm,user->VPERE,connect,vpos);CHKERRQ(ierr);

    /* get the global DOF number to appropriately set the element contribution in the RHS vector */
    ierr = DMMoabGetFieldDofsLocal(dm, user->VPERE, connect, 0, dof_indices);CHKERRQ(ierr);

    /* compute the quadrature points transformed to the physical space and
       compute the basis functions and the derivatives wrt x, y and z directions */
    ierr = Compute_Basis(user->VPERE, vpos, user->NQPTS, quadrature, phypts, jxw, phi, dphidx, dphidy, dphidz);CHKERRQ(ierr);

    ierr = PetscMemzero(array, user->VPERE*user->VPERE*sizeof(PetscScalar));

    /* Compute function over the locally owned part of the grid */
    for (q=0; q < user->NQPTS; ++q) {

      /* compute the inhomogeneous diffusion coefficient at the first quadrature point 
          -- for large spatial variations, embed this property evaluation inside quadrature loop */
      rho = ComputeDiffusionCoefficient(&phypts[q*3], user);

      for (i=0; i < user->VPERE; ++i) {
        for (j=0; j < user->VPERE; ++j) {
          array[i*user->VPERE+j] += jxw[q] * rho * ( dphidx[q*user->VPERE+i]*dphidx[q*user->VPERE+j] + 
                                                     dphidy[q*user->VPERE+i]*dphidy[q*user->VPERE+j] +
                                                     dphidz[q*user->VPERE+i]*dphidz[q*user->VPERE+j] );
        }
      }
    }

    /* check if element is on the boundary */
    //ierr = DMMoabIsEntityOnBoundary(dm,ehandle,&elem_on_boundary);CHKERRQ(ierr);
    ierr = DMMoabCheckBoundaryVertices(dm,user->VPERE,connect,dbdry);CHKERRQ(ierr);
    elem_on_boundary=PETSC_FALSE;
    for (i=0; i < user->VPERE; ++i)
      if (dbdry[i]) elem_on_boundary=PETSC_TRUE;

    /* apply dirichlet boundary conditions */
    if (elem_on_boundary && user->bcType == DIRICHLET) {

      /* get the list of nodes on boundary so that we can enforce dirichlet conditions strongly */
      //ierr = DMMoabCheckBoundaryVertices(dm,user->VPERE,connect,dbdry);CHKERRQ(ierr);

      for (i=0; i < user->VPERE; ++i) {
        if (dbdry[i]) {  /* dirichlet node */
          // PetscPrintf (PETSC_COMM_WORLD, "At the boundary:= dof_indices[%d] = %d\n", i, dof_indices[i]);
          /* think about strongly imposing dirichlet */
          for (j=0; j < user->VPERE; ++j) {
            /* TODO: symmetrize the system - need the RHS */
            array[i*user->VPERE+j] = 0.0;
          }
          array[i*user->VPERE+i] = 1.0;
        }
      }
    }

#if 0
    PetscPrintf(PETSC_COMM_WORLD,"Local matrix %d\n\t", ehandle);
    for (i=0; i < user->VPERE; ++i) {
      for (j=0; j < user->VPERE; ++j) {
        PetscPrintf(PETSC_COMM_WORLD,"%g ", array[i*user->VPERE+j]);
      }
      PetscPrintf(PETSC_COMM_WORLD,"\n\t");
    }
    PetscPrintf(PETSC_COMM_WORLD,"\n");
#endif
    /* set the values directly into appropriate locations. Can alternately use VecSetValues */
    ierr = MatSetValuesLocal(jac, user->VPERE, dof_indices, user->VPERE, dof_indices, array, ADD_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (user->bcType == NEUMANN && false) {
    MatNullSpace nullspace;

    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
    ierr = MatSetNullSpace(jac,nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  }
  //MatView(jac,0);
  PetscFunctionReturn(0);
}


double determinant_mat_3x3 ( PetscReal inmat[3*3] )
{
  return   inmat[0]*(inmat[8]*inmat[4]-inmat[7]*inmat[5])
         - inmat[3]*(inmat[8]*inmat[1]-inmat[7]*inmat[2])
         + inmat[6]*(inmat[5]*inmat[1]-inmat[4]*inmat[2]);
}

PetscErrorCode invert_mat_3x3 (PetscReal *inmat, PetscReal *outmat, PetscScalar *determinant)
{
  if (!inmat) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_POINTER,"Invalid input matrix specified for 3x3 inversion.");
  double det = determinant_mat_3x3(inmat);
  if (outmat) {
    outmat[0]= (inmat[8]*inmat[4]-inmat[7]*inmat[5])/det;
    outmat[1]=-(inmat[8]*inmat[1]-inmat[7]*inmat[2])/det;
    outmat[2]= (inmat[5]*inmat[1]-inmat[4]*inmat[2])/det;
    outmat[3]=-(inmat[8]*inmat[3]-inmat[6]*inmat[5])/det;
    outmat[4]= (inmat[8]*inmat[0]-inmat[6]*inmat[2])/det;
    outmat[5]=-(inmat[5]*inmat[0]-inmat[3]*inmat[2])/det;
    outmat[6]= (inmat[7]*inmat[3]-inmat[6]*inmat[4])/det;
    outmat[7]=-(inmat[7]*inmat[0]-inmat[6]*inmat[1])/det;
    outmat[8]= (inmat[4]*inmat[0]-inmat[3]*inmat[1])/det;
  }
  if (determinant) *determinant=det;
  PetscFunctionReturn(0);
}


double determinant_mat_4x4 ( PetscReal inmat[4*4] )
{
  return
      inmat[0+0*4] * (
          inmat[1+1*4] * ( inmat[2+2*4] * inmat[3+3*4] - inmat[2+3*4] * inmat[3+2*4] )
        - inmat[1+2*4] * ( inmat[2+1*4] * inmat[3+3*4] - inmat[2+3*4] * inmat[3+1*4] )
        + inmat[1+3*4] * ( inmat[2+1*4] * inmat[3+2*4] - inmat[2+2*4] * inmat[3+1*4] ) )
    - inmat[0+1*4] * (
          inmat[1+0*4] * ( inmat[2+2*4] * inmat[3+3*4] - inmat[2+3*4] * inmat[3+2*4] )
        - inmat[1+2*4] * ( inmat[2+0*4] * inmat[3+3*4] - inmat[2+3*4] * inmat[3+0*4] )
        + inmat[1+3*4] * ( inmat[2+0*4] * inmat[3+2*4] - inmat[2+2*4] * inmat[3+0*4] ) )
    + inmat[0+2*4] * (
          inmat[1+0*4] * ( inmat[2+1*4] * inmat[3+3*4] - inmat[2+3*4] * inmat[3+1*4] )
        - inmat[1+1*4] * ( inmat[2+0*4] * inmat[3+3*4] - inmat[2+3*4] * inmat[3+0*4] )
        + inmat[1+3*4] * ( inmat[2+0*4] * inmat[3+1*4] - inmat[2+1*4] * inmat[3+0*4] ) )
    - inmat[0+3*4] * (
          inmat[1+0*4] * ( inmat[2+1*4] * inmat[3+2*4] - inmat[2+2*4] * inmat[3+1*4] )
        - inmat[1+1*4] * ( inmat[2+0*4] * inmat[3+2*4] - inmat[2+2*4] * inmat[3+0*4] )
        + inmat[1+2*4] * ( inmat[2+0*4] * inmat[3+1*4] - inmat[2+1*4] * inmat[3+0*4] ) );
}

PetscErrorCode invert_mat_4x4 (PetscReal *inmat, PetscReal *outmat, PetscScalar *determinant)
{
  if (!inmat) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_POINTER,"Invalid input matrix specified for 4x4 inversion.");
  double det = determinant_mat_4x4(inmat);
  if (outmat) {
    outmat[0]=  (inmat[5]*inmat[10]*inmat[15]+inmat[6]*inmat[11]*inmat[13]+inmat[7]*inmat[9]*inmat[14]-inmat[5]*inmat[11]*inmat[14]-inmat[6]*inmat[9]*inmat[15]-inmat[7]*inmat[10]*inmat[13])/det;
    outmat[1]=  (inmat[1]*inmat[11]*inmat[14]+inmat[2]*inmat[9]*inmat[15]+inmat[3]*inmat[10]*inmat[13]-inmat[1]*inmat[10]*inmat[15]-inmat[2]*inmat[11]*inmat[13]-inmat[3]*inmat[9]*inmat[14])/det;
    outmat[2]=  (inmat[1]*inmat[6]*inmat[15]+inmat[2]*inmat[7]*inmat[13]+inmat[3]*inmat[5]*inmat[14]-inmat[1]*inmat[7]*inmat[14]-inmat[2]*inmat[5]*inmat[15]-inmat[3]*inmat[6]*inmat[13])/det;
    outmat[3]=  (inmat[1]*inmat[7]*inmat[10]+inmat[2]*inmat[5]*inmat[11]+inmat[3]*inmat[6]*inmat[9]-inmat[1]*inmat[6]*inmat[11]-inmat[2]*inmat[7]*inmat[9]-inmat[3]*inmat[5]*inmat[10])/det;
    outmat[4]=  (inmat[4]*inmat[11]*inmat[14]+inmat[6]*inmat[8]*inmat[15]+inmat[7]*inmat[10]*inmat[12]-inmat[4]*inmat[10]*inmat[15]-inmat[6]*inmat[11]*inmat[12]-inmat[7]*inmat[8]*inmat[14])/det;
    outmat[5]=  (inmat[0]*inmat[10]*inmat[15]+inmat[2]*inmat[11]*inmat[12]+inmat[3]*inmat[8]*inmat[14]-inmat[0]*inmat[11]*inmat[14]-inmat[2]*inmat[8]*inmat[15]-inmat[3]*inmat[10]*inmat[12])/det;
    outmat[6]=  (inmat[0]*inmat[7]*inmat[14]+inmat[2]*inmat[4]*inmat[15]+inmat[3]*inmat[6]*inmat[12]-inmat[0]*inmat[6]*inmat[15]-inmat[2]*inmat[7]*inmat[12]-inmat[3]*inmat[4]*inmat[14])/det;
    outmat[7]=  (inmat[0]*inmat[6]*inmat[11]+inmat[2]*inmat[7]*inmat[8]+inmat[3]*inmat[4]*inmat[10]-inmat[0]*inmat[7]*inmat[10]-inmat[2]*inmat[4]*inmat[11]-inmat[3]*inmat[6]*inmat[8])/det;
    outmat[8]=  (inmat[4]*inmat[9]*inmat[15]+inmat[5]*inmat[11]*inmat[12]+inmat[7]*inmat[8]*inmat[13]-inmat[4]*inmat[11]*inmat[13]-inmat[5]*inmat[8]*inmat[15]-inmat[7]*inmat[9]*inmat[12])/det;
    outmat[9]=  (inmat[0]*inmat[11]*inmat[13]+inmat[1]*inmat[8]*inmat[15]+inmat[3]*inmat[9]*inmat[12]-inmat[0]*inmat[9]*inmat[15]-inmat[1]*inmat[11]*inmat[12]-inmat[3]*inmat[8]*inmat[13])/det;
    outmat[10]= (inmat[0]*inmat[5]*inmat[15]+inmat[1]*inmat[7]*inmat[12]+inmat[3]*inmat[4]*inmat[13]-inmat[0]*inmat[7]*inmat[13]-inmat[1]*inmat[4]*inmat[15]-inmat[3]*inmat[5]*inmat[12])/det;
    outmat[11]= (inmat[0]*inmat[7]*inmat[9]+inmat[1]*inmat[4]*inmat[11]+inmat[3]*inmat[5]*inmat[8]-inmat[0]*inmat[5]*inmat[11]-inmat[1]*inmat[7]*inmat[8]-inmat[3]*inmat[4]*inmat[9])/det;
    outmat[12]= (inmat[4]*inmat[10]*inmat[13]+inmat[5]*inmat[8]*inmat[14]+inmat[6]*inmat[9]*inmat[12]-inmat[4]*inmat[9]*inmat[14]-inmat[5]*inmat[10]*inmat[12]-inmat[6]*inmat[8]*inmat[13])/det;
    outmat[13]= (inmat[0]*inmat[9]*inmat[14]+inmat[1]*inmat[10]*inmat[12]+inmat[2]*inmat[8]*inmat[13]-inmat[0]*inmat[10]*inmat[13]-inmat[1]*inmat[8]*inmat[14]-inmat[2]*inmat[9]*inmat[12])/det;
    outmat[14]= (inmat[0]*inmat[6]*inmat[13]+inmat[1]*inmat[4]*inmat[14]+inmat[2]*inmat[5]*inmat[12]-inmat[0]*inmat[5]*inmat[14]-inmat[1]*inmat[6]*inmat[12]-inmat[2]*inmat[4]*inmat[13])/det;
    outmat[15]= (inmat[0]*inmat[5]*inmat[10]+inmat[1]*inmat[6]*inmat[8]+inmat[2]*inmat[4]*inmat[9]-inmat[0]*inmat[6]*inmat[9]-inmat[1]*inmat[4]*inmat[10]-inmat[2]*inmat[5]*inmat[8])/det;
  }
  if (determinant) *determinant=det;
  PetscFunctionReturn(0);
}


/*
*  Purpose: Compute_Basis: all bases at N points for a HEX8 element.
*
*  Discussion:
*
*    The routine is given the coordinates of the vertices of a hexahedra.
*    It works directly with these coordinates, and does not refer to a 
*    reference element.
*
*    The sides of the element are presumed to lie along coordinate axes.
*
*    The routine evaluates the basis functions associated with each corner,
*    and their derivatives with respect to X and Y.
*
*  Physical Element HEX8:
*
*      8------7        t  s
*     /|     /|        | /
*    5------6 |        |/
*    | |    | |        0-------r
*    | 4----|-3        
*    |/     |/        
*    1------2        
*     
*
*  Parameters:
*
*    Input, PetscScalar Q[3*8], the coordinates of the vertices.
*    It is common to list these points in counter clockwise order.
*
*    Input, int N, the number of evaluation points.
*
*    Input, PetscScalar P[3*N], the evaluation points.
*
*    Output, PetscScalar PHI[8*N], the bases at the evaluation points.
*
*    Output, PetscScalar DPHIDX[8*N], DPHIDY[8*N], the derivatives of the
*    bases at the evaluation points.
*
*  Original Author: John Burkardt (http://people.sc.fsu.edu/~jburkardt/cpp_src/fem3d_pack/fem3d_pack.cpp)
*  Modified by Vijay Mahadevan
*/
#undef __FUNCT__
#define __FUNCT__ "Compute_Basis"
PetscErrorCode Compute_Basis ( PetscInt nverts, PetscReal *coords/*nverts*3*/, PetscInt npts, PetscReal *quad/*npts*3*/, PetscReal *phypts/*npts*3*/, 
        PetscReal *jxw/*npts*/, PetscReal *phi/*npts*/, PetscReal *dphidx/*npts*/, PetscReal *dphidy/*npts*/, PetscReal *dphidz/*npts*/)
{
  PetscReal volume;
  int i,j;
  PetscReal jacobian[9],ijacobian[9];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Reset arrays. */
  ierr = PetscMemzero(phi,npts*nverts*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemzero(quad,npts*3*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemzero(phypts,npts*3*sizeof(PetscReal));CHKERRQ(ierr);
  if (dphidx) {
    ierr = PetscMemzero(dphidx,npts*nverts*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemzero(dphidy,npts*nverts*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemzero(dphidz,npts*nverts*sizeof(PetscReal));CHKERRQ(ierr);
  }

  if (nverts == 8) { // Linear Hexahedra

    /* 3-D 2-point tensor product Gaussian quadrature */
    quad[0]=-0.5773502691896257; quad[1]=-0.5773502691896257;  quad[2]=-0.5773502691896257;
    quad[3]=-0.5773502691896257; quad[4]=0.5773502691896257;   quad[5]=-0.5773502691896257;
    quad[6]=0.5773502691896257;  quad[7]=-0.5773502691896257;  quad[8]=-0.5773502691896257;
    quad[9]=0.5773502691896257;  quad[10]=0.5773502691896257;  quad[11]=-0.5773502691896257;

    quad[12]=-0.5773502691896257; quad[13]=-0.5773502691896257;  quad[14]=0.5773502691896257;
    quad[15]=-0.5773502691896257; quad[16]=0.5773502691896257;   quad[17]=0.5773502691896257;
    quad[18]=0.5773502691896257;  quad[19]=-0.5773502691896257;  quad[20]=0.5773502691896257;
    quad[21]=0.5773502691896257;  quad[22]=0.5773502691896257;   quad[23]=0.5773502691896257;

    /* transform quadrature bounds: [-1, 1] => [0, 1] */
    //for (i=0; i<npts*3; ++i) quad[i]=0.5*quad[i]+0.5;

    for (j=0;j<npts;j++)
    {
      const int offset = j*nverts;
      const double& r = quad[j*3+0];
      const double& s = quad[j*3+1];
      const double& t = quad[j*3+2];

      phi[offset+0] = ( 1.0 - r ) * ( 1.0 - s ) * ( 1.0 - t ) / 8;
      phi[offset+1] = ( 1.0 + r ) * ( 1.0 - s ) * ( 1.0 - t ) / 8;
      phi[offset+2] = ( 1.0 + r ) * ( 1.0 + s ) * ( 1.0 - t ) / 8;
      phi[offset+3] = ( 1.0 - r ) * ( 1.0 + s ) * ( 1.0 - t ) / 8;
      phi[offset+4] = ( 1.0 - r ) * ( 1.0 - s ) * ( 1.0 + t ) / 8;
      phi[offset+5] = ( 1.0 + r ) * ( 1.0 - s ) * ( 1.0 + t ) / 8;
      phi[offset+6] = ( 1.0 + r ) * ( 1.0 + s ) * ( 1.0 + t ) / 8;
      phi[offset+7] = ( 1.0 - r ) * ( 1.0 + s ) * ( 1.0 + t ) / 8;

      const double dNi_dxi[8]  = { - ( 1.0 - s ) * ( 1.0 - t ),
                                     ( 1.0 - s ) * ( 1.0 - t ),
                                     ( 1.0 + s ) * ( 1.0 - t ),
                                   - ( 1.0 + s ) * ( 1.0 - t ),
                                   - ( 1.0 - s ) * ( 1.0 + t ),
                                     ( 1.0 - s ) * ( 1.0 + t ),
                                     ( 1.0 + s ) * ( 1.0 + t ),
                                   - ( 1.0 + s ) * ( 1.0 + t )  };

      const double dNi_deta[8]  = { - ( 1.0 - r ) * ( 1.0 - t ),
                                    - ( 1.0 + r ) * ( 1.0 - t ),
                                      ( 1.0 + r ) * ( 1.0 - t ),
                                      ( 1.0 - r ) * ( 1.0 - t ),
                                    - ( 1.0 - r ) * ( 1.0 + t ),
                                    - ( 1.0 + r ) * ( 1.0 + t ),
                                      ( 1.0 + r ) * ( 1.0 + t ),
                                      ( 1.0 - r ) * ( 1.0 + t ) };

      const double dNi_dzeta[8]  = { - ( 1.0 - r ) * ( 1.0 - s ),
                                     - ( 1.0 + r ) * ( 1.0 - s ),
                                     - ( 1.0 + r ) * ( 1.0 + s ),
                                     - ( 1.0 - r ) * ( 1.0 + s ),
                                       ( 1.0 - r ) * ( 1.0 - s ),
                                       ( 1.0 + r ) * ( 1.0 - s ),
                                       ( 1.0 + r ) * ( 1.0 + s ),
                                       ( 1.0 - r ) * ( 1.0 + s ) };

      ierr = PetscMemzero(jacobian,9*sizeof(PetscReal));CHKERRQ(ierr);
      ierr = PetscMemzero(ijacobian,9*sizeof(PetscReal));CHKERRQ(ierr);
      double factor = 1.0/8;
      for (i = 0; i < nverts; ++i) {
        const PetscScalar* vertex = coords+i*3;
        jacobian[0] += dNi_dxi[i]   * vertex[0];
        jacobian[3] += dNi_dxi[i]   * vertex[1];
        jacobian[6] += dNi_dxi[i]   * vertex[2];
        jacobian[1] += dNi_deta[i]  * vertex[0];
        jacobian[4] += dNi_deta[i]  * vertex[1];
        jacobian[7] += dNi_deta[i]  * vertex[2];
        jacobian[2] += dNi_dzeta[i] * vertex[0];
        jacobian[5] += dNi_dzeta[i] * vertex[1];
        jacobian[8] += dNi_dzeta[i] * vertex[2];
      }

      /* invert the jacobian */
      ierr = invert_mat_3x3(jacobian, ijacobian, &volume);CHKERRQ(ierr);

      jxw[j] = factor*volume/npts;

      /*  Divide by element jacobian. */
      for ( i = 0; i < nverts; ++i ) {
        const PetscScalar* vertex = coords+i*3;
        for (int k = 0; k < 3; ++k) {
          phypts[3*j+k] += phi[i+offset] * vertex[k];
          if (dphidx) dphidx[i+offset] += dNi_dxi[i]   * ijacobian[0*3+k];
          if (dphidy) dphidy[i+offset] += dNi_deta[i]  * ijacobian[1*3+k];
          if (dphidz) dphidz[i+offset] += dNi_dzeta[i] * ijacobian[2*3+k];
        }
      }
    }
  }
  else if (nverts == 4) { // Linear Tetrahedra
    // KEAST rule 2, order 4
    quad[0]=0.5854101966249685; quad[1]=0.1381966011250105; quad[2]=0.1381966011250105;
    quad[3]=0.1381966011250105; quad[4]=0.5854101966249685; quad[5]=0.1381966011250105;
    quad[6]=0.1381966011250105; quad[7]=0.1381966011250105; quad[8]=0.5854101966249685;
    quad[9]=0.1381966011250105; quad[10]=0.1381966011250105; quad[11]=0.1381966011250105;

    ierr = PetscMemzero(jacobian,9*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemzero(ijacobian,9*sizeof(PetscReal));CHKERRQ(ierr);

    jacobian[0] = coords[1*3+0]-coords[0*3+0];  jacobian[1] = coords[2*3+0]-coords[0*3+0]; jacobian[2] = coords[3*3+0]-coords[0*3+0];
    jacobian[3] = coords[1*3+1]-coords[0*3+1];  jacobian[4] = coords[2*3+1]-coords[0*3+1]; jacobian[5] = coords[3*3+1]-coords[0*3+1];
    jacobian[6] = coords[1*3+2]-coords[0*3+2];  jacobian[7] = coords[2*3+2]-coords[0*3+2]; jacobian[8] = coords[3*3+2]-coords[0*3+2];

    /* invert the jacobian */
    ierr = invert_mat_3x3(jacobian, ijacobian, &volume);CHKERRQ(ierr);

    if ( volume < 1e-8 ) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Tetrahedral element has zero volume: %g. Degenerate element or invalid connectivity\n", volume);

    for ( j = 0; j < npts; j++ )
    {
      const int offset = j*nverts;
      const double factor = 1.0/6;
      const double& r = quad[j*3+0];
      const double& s = quad[j*3+1];
      const double& t = quad[j*3+2];

      jxw[j] = factor*volume/npts;

      phi[offset+0] = 1.0 - r - s - t;
      phi[offset+1] = r;
      phi[offset+2] = s;
      phi[offset+3] = t;

      if (dphidx) {
        dphidx[0+offset] = ( coords[1+2*3] * ( coords[2+1*3] - coords[2+3*3] ) 
                           - coords[1+1*3] * ( coords[2+2*3] - coords[2+3*3] )
                           - coords[1+3*3] * ( coords[2+1*3] - coords[2+2*3] )
                           ) / volume;
        dphidx[1+offset] = -( coords[1+2*3] * ( coords[2+0*3] - coords[2+3*3] )
                           - coords[1+0*3] * ( coords[2+2*3] - coords[2+3*3] )
                           - coords[1+3*3] * ( coords[2+0*3] - coords[2+2*3] )
                           ) / volume;
        dphidx[2+offset] = ( coords[1+1*3] * ( coords[2+0*3] - coords[2+3*3] )
                           - coords[1+0*3] * ( coords[2+1*3] - coords[2+3*3] )
                           - coords[1+3*3] * ( coords[2+0*3] - coords[2+1*3] )
                           ) / volume;
        dphidx[3+offset] = -dphidx[0+offset] - dphidx[1+offset] - dphidx[2+offset];
      }

      if (dphidy) {
        dphidy[0+offset] = ( coords[0+1*3] * ( coords[2+2*3] - coords[2+3*3] )
                           - coords[0+2*3] * ( coords[2+1*3] - coords[2+3*3] )
                           + coords[0+3*3] * ( coords[2+1*3] - coords[2+2*3] )
                           ) / volume;
        dphidy[1+offset] = -( coords[0+0*3] * ( coords[2+2*3] - coords[2+3*3] )
                           - coords[0+2*3] * ( coords[2+0*3] - coords[2+3*3] )
                           + coords[0+3*3] * ( coords[2+0*3] - coords[2+2*3] )
                           ) / volume;
        dphidy[2+offset] = ( coords[0+0*3] * ( coords[2+1*3] - coords[2+3*3] )
                           - coords[0+1*3] * ( coords[2+0*3] - coords[2+3*3] )
                           + coords[0+3*3] * ( coords[2+0*3] - coords[2+1*3] )
                           ) / volume;
        dphidy[3+offset] = -dphidy[0+offset] - dphidy[1+offset] - dphidy[2+offset];
      }


      if (dphidz) {
        dphidz[0+offset] = ( coords[0+1*3] * (coords[1+3*3]-coords[1+2*3])
                           - coords[0+2*3] * (coords[1+3*3]-coords[1+1*3])
                           + coords[0+3*3] * (coords[1+2*3]-coords[1+1*3])
                           ) / volume;
        dphidz[1+offset] = -( coords[0+0*3] * (coords[1+3*3]-coords[1+2*3])
                           + coords[0+2*3] * (coords[1+0*3]-coords[1+3*3]) 
                           - coords[0+3*3] * (coords[1+0*3]-coords[1+2*3])
                           ) / volume;
        dphidz[2+offset] = ( coords[0+0*3] * (coords[1+3*3]-coords[1+1*3])
                           + coords[0+1*3] * (coords[1+0*3]-coords[1+3*3]) 
                           - coords[0+3*3] * (coords[1+0*3]-coords[1+1*3])
                           ) / volume;
        dphidz[3+offset] = -dphidz[0+offset] - dphidz[1+offset] - dphidz[2+offset];
      }

      for (i = 0; i < nverts; ++i) {
        const PetscScalar* vertices = coords+i*3;
        for (int k = 0; k < 3; ++k)
          phypts[3*j+k] += phi[i+offset] * vertices[k];
      }
    } // Tetrahedra -- ends
  }
  else
  {
    SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"The number of entity vertices are invalid. Currently only support HEX8: %D",nverts);
  }
#if 0
  /* verify if the computed basis functions are consistent */
  for ( j = 0; j < npts; j++ ) {
    PetscScalar phisum=0,dphixsum=0,dphiysum=0,dphizsum=0;
    const int offset = j*nverts;
    for ( i = 0; i < nverts; i++ ) {
      phisum += phi[i+offset];
      if (dphidx) dphixsum += dphidx[i+offset];
      if (dphidy) dphiysum += dphidy[i+offset];
      if (dphidz) dphizsum += dphidz[i+offset];
      if (dphidx) PetscPrintf(PETSC_COMM_WORLD, "\t Values [%d]: [JxW] [phi, dphidx, dphidy, dphidz] = %g, %g, %g, %g, %g\n", j, jxw[j], phi[i+offset], dphidx[i+offset], dphidy[i+offset], dphidz[i+offset]);
    }
    if (dphidx) PetscPrintf(PETSC_COMM_WORLD, "Sum of basis at quadrature point %D (%g, %g, %g) = %g, %g, %g, %g\n", j, quad[3*j+0], quad[3*j+1], quad[3*j+2], phisum, dphixsum, dphiysum, dphizsum);
    //PetscPrintf(PETSC_COMM_WORLD, "Sum of basis at quadrature point %D (%g, %g, %g) = %g, %g, %g, %g\n", j, pts[3*j+0], pts[3*j+1], pts[3*j+2], phisum, dphixsum, dphiysum, dphizsum);
  }
  //if (dphidx) exit(0);
#endif
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

    if (err) {
      /* set the discrete L2 error against the exact solution */
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

