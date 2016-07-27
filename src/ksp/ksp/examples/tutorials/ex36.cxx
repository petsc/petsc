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

Sovle the Laplace-Beltrami equations on the manifold surface of a volume ?
https://www.dealii.org/developer/doxygen/deal.II/step_38.html

OR

http://www.maths.manchester.ac.uk/~mheil/doc/poisson/eighth_sphere_poisson/html/index.html
*/

static char help[] = "\
                      Solves a three dimensional inhomogeneous Laplacian equation with a Gaussian source.\n \
                      Usage: ./ex36 -bc_type dirichlet -nu .01 -n 10\n";

#define PROBLEM 2

/* PETSc includes */
#include <petscksp.h>
#include <petscdmmoab.h>

typedef enum {DIRICHLET, NEUMANN} BCType;

typedef struct {
  PetscInt  dim,n,nlevels;
  PetscReal rho;
  PetscReal xyzref[3];
  PetscReal nu;
  BCType    bcType;
  char      filename[PETSC_MAX_PATH_LEN];

  /* Discretization parameters */
  int VPERE;
} UserContext;

static PetscErrorCode ComputeMatrix_MOAB(KSP,Mat,Mat,void*);
static PetscErrorCode ComputeRHS_MOAB(KSP,Vec,void*);
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
  user.dim    = 3; /* 3-Dimensional problem */
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
  return 3.0*PETSC_PI*PETSC_PI*exact;
#endif
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRHS_MOAB"
PetscErrorCode ComputeRHS_MOAB(KSP ksp,Vec b,void *ptr)
{
  UserContext*      user = (UserContext*)ptr;
  DM                dm;
  PetscInt          dof_indices[user->VPERE],npoints;
  PetscBool         dbdry[user->VPERE];
  PetscReal         vpos[user->VPERE*3];
  PetscInt          i,q,num_conn;
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

  ierr = DMMoabFEMCreateQuadratureDefault (3, user->VPERE, &quadratureObj);CHKERRQ(ierr);
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
    if (num_conn != user->VPERE) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only HEX8 element bases are supported in the current example. n(Connectivity)=%D.\n", num_conn);

    /* compute the mid-point of the element and use a 1-point lumped quadrature */
    ierr = DMMoabGetVertexCoordinates(dm,num_conn,connect,vpos);CHKERRQ(ierr);

    /* get the global DOF number to appropriately set the element contribution in the RHS vector */
    ierr = DMMoabGetFieldDofsLocal(dm, num_conn, connect, 0, dof_indices);CHKERRQ(ierr);

    /* compute the quadrature points transformed to the physical space and then
       compute the basis functions to compute local operators */
    // ierr = Compute_Basis(user->VPERE, vpos, user->NQPTS, quadrature, phypts, jxw, phi, 0, 0, 0);CHKERRQ(ierr);
    ierr = DMMoabFEMComputeBasis(3, user->VPERE, vpos, quadratureObj, phypts, jxw, phi, 0);CHKERRQ(ierr);

    /* Compute function over the locally owned part of the grid */
    for (q=0; q<npoints; ++q) {
      const double ff = ForcingFunction(&phypts[3*q], user);
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
          localv[i] = ForcingFunction(&vpos[3*i], user);
        }
      }
    }

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
  ierr = PetscFree3(phi,phypts,jxw);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&quadratureObj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ComputeMatrix_MOAB"
PetscErrorCode ComputeMatrix_MOAB(KSP ksp,Mat J,Mat jac,void *ctx)
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
  PetscReal         *phi, *dphi[3], *phypts, *jxw;
  PetscQuadrature   quadratureObj;
  PetscErrorCode    ierr;
 
  PetscFunctionBeginUser;
  ierr      = KSPGetDM(ksp,&dm);CHKERRQ(ierr);

  /* get the essential MOAB mesh related quantities needed for FEM assembly */  
  ierr = DMMoabGetInterface(dm, &mbImpl);CHKERRQ(ierr);
  ierr = DMMoabGetLocalElements(dm, &elocal);CHKERRQ(ierr);
  ierr = DMMoabGetSize(dm, &nglobale, &nglobalv);CHKERRQ(ierr);
  ierr = DMMoabGetHierarchyLevel(dm, &hlevel);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "ComputeMatrix: Level = %d, N(elements) = %d, N(vertices) = %d \n", hlevel, nglobale, nglobalv);

  ierr = DMMoabFEMCreateQuadratureDefault ( 3, user->VPERE, &quadratureObj );CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quadratureObj, NULL, &npoints, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscMalloc6(user->VPERE*npoints,&phi, user->VPERE*npoints,&dphi[0], user->VPERE*npoints,&dphi[1], user->VPERE*npoints,&dphi[2], npoints*3,&phypts, npoints,&jxw);CHKERRQ(ierr);

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
    // ierr = Compute_Basis(user->VPERE, vpos, user->NQPTS, quadrature, phypts, jxw, phi, dphidx, dphidy, dphidz);CHKERRQ(ierr);
    ierr = DMMoabFEMComputeBasis(3, user->VPERE, vpos, quadratureObj, phypts, jxw, phi, dphi);CHKERRQ(ierr);

    ierr = PetscMemzero(array, user->VPERE*user->VPERE*sizeof(PetscScalar));

    /* Compute function over the locally owned part of the grid */
    for (q=0; q < npoints; ++q) {

      /* compute the inhomogeneous diffusion coefficient at the first quadrature point 
          -- for large spatial variations, embed this property evaluation inside quadrature loop */
      rho = ComputeDiffusionCoefficient(&phypts[q*3], user);

      for (i=0; i < user->VPERE; ++i) {
        for (j=0; j < user->VPERE; ++j) {
          array[i*user->VPERE+j] += jxw[q] * rho * ( dphi[0][q*user->VPERE+i]*dphi[0][q*user->VPERE+j] + 
                                                     dphi[1][q*user->VPERE+i]*dphi[1][q*user->VPERE+j] +
                                                     dphi[2][q*user->VPERE+i]*dphi[2][q*user->VPERE+j] );
        }
      }
    }

    /* check if element is on the boundary */
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
  ierr = PetscFree6(phi,dphi[0],dphi[1],dphi[2],phypts,jxw);CHKERRQ(ierr);
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

