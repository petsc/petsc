/*T
   Concepts: KSP^solving a system of linear equations using a MOAB based DM implementation.
   Concepts: KSP^Laplacian, 2d
   Processors: n
T*/



/*
Inhomogeneous Laplacian in 2D. Modeled by the partial differential equation

   -div \rho grad u = f,  0 < x,y < 1,

Problem 1: (Default)

  Use the following exact solution with Dirichlet boundary condition

    u = sin(pi*x)*sin(pi*y)

  and generate an appropriate forcing function to measure convergence.

Usage:

  Measure convergence rate with uniform refinement with the options: "-problem 1 -error".

    mpiexec -n $NP ./ex35 -problem 1 -error -n 16 -levels 5 -pc_type gamg
    mpiexec -n $NP ./ex35 -problem 1 -error -n 32 -levels 4 -pc_type gamg
    mpiexec -n $NP ./ex35 -problem 1 -error -n 64 -levels 3 -pc_type mg
    mpiexec -n $NP ./ex35 -problem 1 -error -n 128 -levels 2 -pc_type mg
    mpiexec -n $NP ./ex35 -problem 1 -error -n 256 -levels 1 -mg
    mpiexec -n $NP ./ex35 -problem 1 -error -n 512 -levels 0 -mg

  Or with an external mesh file representing [0, 1]^2,

    mpiexec -n $NP ./ex35 -problem 1 -file ./external_mesh.h5m -levels 2 -error -pc_type mg

Problem 2:

  Use the forcing function

     f = e^{-((x-xr)^2+(y-yr)^2)/\nu}

  with Dirichlet boundary conditions

     u = f(x,y) for x = 0, x = 1, y = 0, y = 1

  or pure Neumman boundary conditions

Usage:

  Run with different values of \rho and \nu (problem 1) to control diffusion and gaussian source spread. This uses the internal mesh generator implemented in DMMoab.

    mpiexec -n $NP ./ex35 -problem 2 -n 20 -nu 0.02 -rho 0.01
    mpiexec -n $NP ./ex35 -problem 2 -n 40 -nu 0.01 -rho 0.005 -io -ksp_monitor
    mpiexec -n $NP ./ex35 -problem 2 -n 80 -nu 0.01 -rho 0.005 -io -ksp_monitor -pc_type hypre
    mpiexec -n $NP ./ex35 -problem 2 -n 160 -bc neumann -nu 0.005 -rho 0.01 -io
    mpiexec -n $NP ./ex35 -problem 2 -n 320 -bc neumann -nu 0.001 -rho 1 -io

  Or with an external mesh file representing [0, 1]^2,

    mpiexec -n $NP ./ex35 -problem 2 -file ./external_mesh.h5m -levels 1 -pc_type gamg

Problem 3:

  Use the forcing function

     f = \nu*sin(\pi*x/LX)*sin(\pi*y/LY)

  with Dirichlet boundary conditions

     u = 0.0, for x = 0, x = 1, y = 0, y = 1 (outer boundary) &&
     u = 1.0, for (x-0.5)^2 + (y-0.5)^2 = 0.01

Usage:

  Now, you could alternately load an external MOAB mesh that discretizes the unit square and use that to run the solver.

    mpiexec -n $NP ./ex35 -problem 3 -file input/square_with_hole.h5m -mg
    mpiexec -n $NP ./ex35 -problem 3 -file input/square_with_hole.h5m -mg -levels 2 -io -ksp_monitor
    mpiexec -n $NP ./ex35 -problem 3 -file input/square_with_hole.h5m -io -ksp_monitor -pc_type hypre

*/

static char help[] = "\
                      Solves the 2D inhomogeneous Laplacian equation.\n \
                      Usage: ./ex35 -problem 1 -error -n 4 -levels 3 -mg\n  \
                      Usage: ./ex35 -problem 2 -n 80 -nu 0.01 -rho 0.005 -io -ksp_monitor -pc_type gamg\n  \
                      Usage: ./ex35 -problem 3 -file input/square_with_hole.h5m -mg\n";


/* PETSc includes */
#include <petscksp.h>
#include <petscdmmoab.h>

#define LOCAL_ASSEMBLY

typedef enum {DIRICHLET, NEUMANN} BCType;

typedef struct {
  PetscInt  dim, n, problem, nlevels;
  PetscReal rho;
  PetscReal bounds[6];
  PetscReal xref, yref;
  PetscReal nu;
  PetscInt  VPERE;
  BCType    bcType;
  PetscBool use_extfile, io, error, usetri, usemg;
  char filename[PETSC_MAX_PATH_LEN];
} UserContext;

static PetscErrorCode ComputeMatrix(KSP, Mat, Mat, void*);
static PetscErrorCode ComputeRHS(KSP, Vec, void*);
static PetscErrorCode ComputeDiscreteL2Error(KSP, Vec, UserContext*);
static PetscErrorCode InitializeOptions(UserContext*);

int main(int argc, char **argv)
{
  KSP             ksp;
  PC              pc;
  Mat             R;
  DM              dm, dmref, *dmhierarchy;

  UserContext     user;
  const char      *fields[1] = {"T-Variable"};
  PetscErrorCode  ierr;
  PetscInt        k;
  Vec             b, x, errv;

  ierr = PetscInitialize(&argc, &argv, (char*)0, help);if (ierr) return ierr;

  ierr = InitializeOptions(&user);CHKERRQ(ierr);

  /* Create the DM object from either a mesh file or from in-memory structured grid */
  if (user.use_extfile) {
    ierr = DMMoabLoadFromFile(PETSC_COMM_WORLD, user.dim, 1, user.filename, "", &dm);CHKERRQ(ierr);
  }
  else {
    ierr = DMMoabCreateBoxMesh(PETSC_COMM_WORLD, user.dim, user.usetri, user.bounds, user.n, 1, &dm);CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMMoabSetFieldNames(dm, 1, fields);CHKERRQ(ierr);

  /* SetUp the data structures for DMMOAB */
  ierr = DMSetUp(dm);CHKERRQ(ierr);

  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
  ierr = KSPSetComputeRHS(ksp, ComputeRHS, &user);CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp, ComputeMatrix, &user);CHKERRQ(ierr);

  if (user.nlevels)
  {
    ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(DM) * (user.nlevels + 1), &dmhierarchy);
    for (k = 0; k <= user.nlevels; k++) dmhierarchy[k] = NULL;

    ierr = PetscPrintf(PETSC_COMM_WORLD, "Number of mesh hierarchy levels: %d\n", user.nlevels);CHKERRQ(ierr);
    ierr = DMMoabGenerateHierarchy(dm, user.nlevels, PETSC_NULL);CHKERRQ(ierr);

    /* coarsest grid = 0, finest grid = nlevels */
    dmhierarchy[0] = dm;
    PetscBool usehierarchy = PETSC_FALSE;
    if (usehierarchy) {
      ierr = DMRefineHierarchy(dm, user.nlevels, &dmhierarchy[1]);CHKERRQ(ierr);
    }
    else {
      for (k = 1; k <= user.nlevels; k++) {
        ierr = DMRefine(dmhierarchy[k - 1], MPI_COMM_NULL, &dmhierarchy[k]);CHKERRQ(ierr);
      }
    }
    dmref = dmhierarchy[user.nlevels];
    PetscObjectReference((PetscObject)dmref);

    if (user.usemg) {
      ierr = PCSetType(pc, PCMG);CHKERRQ(ierr);
      ierr = PCMGSetLevels(pc, user.nlevels + 1, NULL);CHKERRQ(ierr);
      ierr = PCMGSetType(pc, PC_MG_MULTIPLICATIVE);CHKERRQ(ierr);
      ierr = PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH);CHKERRQ(ierr);
      ierr = PCMGSetCycleType(pc, PC_MG_CYCLE_V);CHKERRQ(ierr);
      ierr = PCMGSetNumberSmooth(pc, 2);CHKERRQ(ierr);

      for (k = 1; k <= user.nlevels; k++) {
        ierr = DMCreateInterpolation(dmhierarchy[k - 1], dmhierarchy[k], &R, NULL);CHKERRQ(ierr);
        ierr = PCMGSetInterpolation(pc, k, R);CHKERRQ(ierr);
        ierr = MatDestroy(&R);CHKERRQ(ierr);
      }
    }

    for (k = 1; k <= user.nlevels; k++) {
      ierr = DMDestroy(&dmhierarchy[k]);CHKERRQ(ierr);
    }
    ierr = PetscFree(dmhierarchy);CHKERRQ(ierr);
  }
  else {
    dmref = dm;
    PetscObjectReference((PetscObject)dm);
  }

  ierr = KSPSetDM(ksp, dmref);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* Perform the actual solve */
  ierr = KSPSolve(ksp, NULL, NULL);CHKERRQ(ierr);
  ierr = KSPGetSolution(ksp, &x);CHKERRQ(ierr);
  ierr = KSPGetRhs(ksp, &b);CHKERRQ(ierr);

  if (user.error) {
    ierr = VecDuplicate(b, &errv);CHKERRQ(ierr);
    ierr = ComputeDiscreteL2Error(ksp, errv, &user);CHKERRQ(ierr);
    ierr = VecDestroy(&errv);CHKERRQ(ierr);
  }

  if (user.io) {
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

PetscScalar ComputeDiffusionCoefficient(PetscReal coords[3], UserContext* user)
{
  switch (user->problem) {
  case 2:
    if ((coords[0] > user->bounds[1] / 3.0) && (coords[0] < 2.0 * user->bounds[1] / 3.0) && (coords[1] > user->bounds[3] / 3.0) && (coords[1] < 2.0 * user->bounds[3] / 3.0)) {
      return user->rho;
    }
    else {
      return 1.0;
    }
  case 1:
  case 3:
  default:
    return user->rho;
  }
}

PetscScalar ExactSolution(PetscReal coords[3], UserContext* user)
{
  switch (user->problem) {
  case 1:
    return sin(PETSC_PI * coords[0] / user->bounds[1]) * sin(PETSC_PI * coords[1] / user->bounds[3]);
  case 3:
  case 2:
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Exact solution for -problem = [%D] is not available.\n", user->problem);
  }
}

PetscScalar ComputeForcingFunction(PetscReal coords[3], UserContext* user)
{
  switch (user->problem) {
  case 3:
    return user->nu * sin(PETSC_PI * coords[0] / user->bounds[1]) * sin(PETSC_PI * coords[1] / user->bounds[3]);
  case 2:
    return PetscExpScalar(- ( (coords[0] - user->xref) * (coords[0] - user->xref) + (coords[1] - user->yref) * (coords[1] - user->yref)) / user->nu);
  case 1:
  default:
    return PETSC_PI * PETSC_PI * ComputeDiffusionCoefficient(coords, user) *
            (1.0 / user->bounds[1] / user->bounds[1] + 1.0 / user->bounds[3] / user->bounds[3]) * sin(PETSC_PI * coords[0] / user->bounds[1]) * sin(PETSC_PI * coords[1] / user->bounds[3]);
  }
}


#define BCHECKEPS 1e-10
#define BCHECK(coordxyz,truetrace) ((coordxyz < truetrace+BCHECKEPS && coordxyz > truetrace-BCHECKEPS))

PetscScalar EvaluateStrongDirichletCondition(PetscReal coords[3], UserContext* user)
{
  switch (user->problem) {
  case 3:
    if (BCHECK(coords[0], user->bounds[0]) || BCHECK(coords[0], user->bounds[1]) || BCHECK(coords[1], user->bounds[2]) || BCHECK(coords[1], user->bounds[3]))
      return 0.0;
    else // ( coords[0]*coords[0] + coords[1]*coords[1] < 0.04 + BCHECKEPS)
      return 1.0;
  case 2:
    return ComputeForcingFunction(coords, user);
  case 1:
  default:
    return ExactSolution(coords, user);
  }
}

PetscErrorCode ComputeRHS(KSP ksp, Vec b, void *ptr)
{
  UserContext*      user = (UserContext*)ptr;
  DM                dm;
  PetscInt          dof_indices[4];
  PetscBool         dbdry[4];
  PetscReal         vpos[4 * 3];
  PetscScalar       ff;
  PetscInt          i, q, nconn, nc, npoints;
  const moab::EntityHandle *connect;
  const moab::Range *elocal;
  moab::Interface*  mbImpl;
  PetscScalar       localv[4];
  PetscReal         *phi, *phypts, *jxw;
  PetscBool         elem_on_boundary;
  PetscQuadrature   quadratureObj;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = KSPGetDM(ksp, &dm);CHKERRQ(ierr);

  /* reset the RHS */
  ierr = VecSet(b, 0.0);CHKERRQ(ierr);

  ierr = DMMoabFEMCreateQuadratureDefault (2, user->VPERE, &quadratureObj);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quadratureObj, NULL, &nc, &npoints, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscMalloc3(user->VPERE * npoints, &phi, npoints * 3, &phypts, npoints, &jxw);CHKERRQ(ierr);

  /* get the essential MOAB mesh related quantities needed for FEM assembly */
  ierr = DMMoabGetInterface(dm, &mbImpl);CHKERRQ(ierr);
  ierr = DMMoabGetLocalElements(dm, &elocal);CHKERRQ(ierr);

  /* loop over local elements */
  for (moab::Range::iterator iter = elocal->begin(); iter != elocal->end(); iter++) {
    const moab::EntityHandle ehandle = *iter;

    /* Get connectivity information: */
    ierr = DMMoabGetElementConnectivity(dm, ehandle, &nconn, &connect);CHKERRQ(ierr);
    if (nconn != 3 && nconn != 4) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only TRI3/QUAD4 element bases are supported in the current example. n(Connectivity)=%D.\n", nconn);

    ierr = PetscArrayzero(localv, nconn);CHKERRQ(ierr);

    /* get the coordinates of the element vertices */
    ierr = DMMoabGetVertexCoordinates(dm, nconn, connect, vpos);CHKERRQ(ierr);

    /* get the local DoF numbers to appropriately set the element contribution in the operator */
#ifdef LOCAL_ASSEMBLY
    ierr = DMMoabGetFieldDofsLocal(dm, nconn, connect, 0, dof_indices);CHKERRQ(ierr);
#else
    ierr = DMMoabGetFieldDofs(dm, nconn, connect, 0, dof_indices);CHKERRQ(ierr);
#endif

    /* 1) compute the basis functions and the derivatives wrt x and y directions
       2) compute the quadrature points transformed to the physical space */
    ierr = DMMoabFEMComputeBasis(2, nconn, vpos, quadratureObj, phypts, jxw, phi, NULL);CHKERRQ(ierr);

    /* Compute function over the locally owned part of the grid */
    for (q = 0; q < npoints; ++q) {
      ff = ComputeForcingFunction(&phypts[3 * q], user);

      for (i = 0; i < nconn; ++i) {
        localv[i] += jxw[q] * phi[q * nconn + i] * ff;
      }
    }

    /* check if element is on the boundary */
    ierr = DMMoabIsEntityOnBoundary(dm, ehandle, &elem_on_boundary);CHKERRQ(ierr);

    /* apply dirichlet boundary conditions */
    if (elem_on_boundary && user->bcType == DIRICHLET) {

      /* get the list of nodes on boundary so that we can enforce dirichlet conditions strongly */
      ierr = DMMoabCheckBoundaryVertices(dm, nconn, connect, dbdry);CHKERRQ(ierr);

      for (i = 0; i < nconn; ++i) {
        if (dbdry[i]) {  /* dirichlet node */
          /* think about strongly imposing dirichlet */
          localv[i] = EvaluateStrongDirichletCondition(&vpos[3 * i], user);
        }
      }
    }

#ifdef LOCAL_ASSEMBLY
    /* set the values directly into appropriate locations. Can alternately use VecSetValues */
    ierr = VecSetValuesLocal(b, nconn, dof_indices, localv, ADD_VALUES);CHKERRQ(ierr);
#else
    ierr = VecSetValues(b, nconn, dof_indices, localv, ADD_VALUES);CHKERRQ(ierr);
#endif
  }

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;
    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nullspace, b);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  }

  /* Restore vectors */
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  ierr = PetscFree3(phi, phypts, jxw);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&quadratureObj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMatrix(KSP ksp, Mat J, Mat jac, void *ctx)
{
  UserContext       *user = (UserContext*)ctx;
  DM                dm;
  PetscInt          i, j, q, nconn, nglobale, nglobalv, nc, npoints, hlevel;
  PetscInt          dof_indices[4];
  PetscReal         vpos[4 * 3], rho;
  PetscBool         dbdry[4];
  const moab::EntityHandle *connect;
  const moab::Range *elocal;
  moab::Interface*  mbImpl;
  PetscBool         elem_on_boundary;
  PetscScalar       array[4 * 4];
  PetscReal         *phi, *dphi[2], *phypts, *jxw;
  PetscQuadrature   quadratureObj;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = KSPGetDM(ksp, &dm);CHKERRQ(ierr);

  /* get the essential MOAB mesh related quantities needed for FEM assembly */
  ierr = DMMoabGetInterface(dm, &mbImpl);CHKERRQ(ierr);
  ierr = DMMoabGetLocalElements(dm, &elocal);CHKERRQ(ierr);
  ierr = DMMoabGetSize(dm, &nglobale, &nglobalv);CHKERRQ(ierr);
  ierr = DMMoabGetHierarchyLevel(dm, &hlevel);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "ComputeMatrix: Level = %d, N(elements) = %d, N(vertices) = %d \n", hlevel, nglobale, nglobalv);CHKERRQ(ierr);

  ierr = DMMoabFEMCreateQuadratureDefault ( 2, user->VPERE, &quadratureObj);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quadratureObj, NULL, &nc, &npoints, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscMalloc5(user->VPERE * npoints, &phi, user->VPERE * npoints, &dphi[0], user->VPERE * npoints, &dphi[1], npoints * 3, &phypts, npoints, &jxw);CHKERRQ(ierr);

  /* loop over local elements */
  for (moab::Range::iterator iter = elocal->begin(); iter != elocal->end(); iter++) {
    const moab::EntityHandle ehandle = *iter;

    // Get connectivity information:
    ierr = DMMoabGetElementConnectivity(dm, ehandle, &nconn, &connect);CHKERRQ(ierr);
    if (nconn != 3 && nconn != 4) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only QUAD4 or TRI3 element bases are supported in the current example. Connectivity=%D.\n", nconn);

    /* compute the mid-point of the element and use a 1-point lumped quadrature */
    ierr = DMMoabGetVertexCoordinates(dm, nconn, connect, vpos);CHKERRQ(ierr);

    /* get the global DOF number to appropriately set the element contribution in the RHS vector */
#ifdef LOCAL_ASSEMBLY
    ierr = DMMoabGetFieldDofsLocal(dm, nconn, connect, 0, dof_indices);CHKERRQ(ierr);
#else
    ierr = DMMoabGetFieldDofs(dm, nconn, connect, 0, dof_indices);CHKERRQ(ierr);
#endif

    /* 1) compute the basis functions and the derivatives wrt x and y directions
       2) compute the quadrature points transformed to the physical space */
    ierr = DMMoabFEMComputeBasis(2, nconn, vpos, quadratureObj, phypts, jxw, phi, dphi);CHKERRQ(ierr);

    ierr = PetscArrayzero(array, nconn * nconn);

    /* Compute function over the locally owned part of the grid */
    for (q = 0; q < npoints; ++q) {
      /* compute the inhomogeneous (piece-wise constant) diffusion coefficient at the quadrature point
        -- for large spatial variations (within an element), embed this property evaluation inside the quadrature loop
      */
      rho = ComputeDiffusionCoefficient(&phypts[q * 3], user);

      for (i = 0; i < nconn; ++i) {
        for (j = 0; j < nconn; ++j) {
          array[i * nconn + j] += jxw[q] * rho * ( dphi[0][q * nconn + i] * dphi[0][q * nconn + j] +
                                                   dphi[1][q * nconn + i] * dphi[1][q * nconn + j]);
        }
      }
    }

    /* check if element is on the boundary */
    ierr = DMMoabIsEntityOnBoundary(dm, ehandle, &elem_on_boundary);CHKERRQ(ierr);

    /* apply dirichlet boundary conditions */
    if (elem_on_boundary && user->bcType == DIRICHLET) {

      /* get the list of nodes on boundary so that we can enforce dirichlet conditions strongly */
      ierr = DMMoabCheckBoundaryVertices(dm, nconn, connect, dbdry);CHKERRQ(ierr);

      for (i = 0; i < nconn; ++i) {
        if (dbdry[i]) {  /* dirichlet node */
          /* think about strongly imposing dirichlet */
          for (j = 0; j < nconn; ++j) {
            /* TODO: symmetrize the system - need the RHS */
            array[i * nconn + j] = 0.0;
          }
          array[i * nconn + i] = 1.0;
        }
      }
    }

    /* set the values directly into appropriate locations. */
#ifdef LOCAL_ASSEMBLY
    ierr = MatSetValuesLocal(jac, nconn, dof_indices, nconn, dof_indices, array, ADD_VALUES);CHKERRQ(ierr);
#else
    ierr = MatSetValues(jac, nconn, dof_indices, nconn, dof_indices, array, ADD_VALUES);CHKERRQ(ierr);
#endif
  }

  ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace);CHKERRQ(ierr);
    ierr = MatSetNullSpace(J, nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  }
  ierr = PetscFree5(phi, dphi[0], dphi[1], phypts, jxw);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&quadratureObj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode ComputeDiscreteL2Error(KSP ksp, Vec err, UserContext *user)
{
  DM                dm;
  Vec               sol;
  PetscScalar       vpos[3];
  const PetscScalar *x;
  PetscScalar       *e;
  PetscReal         l2err = 0.0, linferr = 0.0, global_l2, global_linf;
  PetscInt          dof_index, N;
  const moab::Range *ownedvtx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = KSPGetDM(ksp, &dm);CHKERRQ(ierr);

  /* get the solution vector */
  ierr = KSPGetSolution(ksp, &sol);CHKERRQ(ierr);

  /* Get the internal reference to the vector arrays */
  ierr = VecGetArrayRead(sol, &x);CHKERRQ(ierr);
  ierr = VecGetSize(sol, &N);CHKERRQ(ierr);
  if (err) {
    /* reset the error vector */
    ierr = VecSet(err, 0.0);CHKERRQ(ierr);
    /* get array reference */
    ierr = VecGetArray(err, &e);CHKERRQ(ierr);
  }

  ierr = DMMoabGetLocalVertices(dm, &ownedvtx, NULL);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (moab::Range::iterator iter = ownedvtx->begin(); iter != ownedvtx->end(); iter++) {
    const moab::EntityHandle vhandle = *iter;

    /* get the local DoF numbers to appropriately set the element contribution in the operator */
#ifdef LOCAL_ASSEMBLY
    ierr = DMMoabGetFieldDofsLocal(dm, 1, &vhandle, 0, &dof_index);CHKERRQ(ierr);
#else
    ierr = DMMoabGetFieldDofs(dm, 1, &vhandle, 0, &dof_index);CHKERRQ(ierr);
#endif

    /* compute the mid-point of the element and use a 1-point lumped quadrature */
    ierr = DMMoabGetVertexCoordinates(dm, 1, &vhandle, vpos);CHKERRQ(ierr);

    /* compute the discrete L2 error against the exact solution */
    const PetscScalar lerr = (ExactSolution(vpos, user) - x[dof_index]);
    l2err += lerr * lerr;
    if (linferr < fabs(lerr))
      linferr = fabs(lerr);

    if (err) { /* set the discrete L2 error against the exact solution */
      e[dof_index] = lerr;
    }
  }

  ierr = MPI_Allreduce(&l2err, &global_l2, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&linferr, &global_linf, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Computed Errors: L_2 = %f, L_inf = %f\n", sqrt(global_l2 / N), global_linf);CHKERRQ(ierr);

  /* Restore vectors */
  ierr = VecRestoreArrayRead(sol, &x);CHKERRQ(ierr);
  if (err) {
    ierr = VecRestoreArray(err, &e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode InitializeOptions(UserContext* user)
{
  const char     *bcTypes[2] = {"dirichlet", "neumann"};
  PetscInt       bc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* set default parameters */
  user->dim     = 2;
  user->problem = 1;
  user->n       = 2;
  user->nlevels = 2;
  user->rho    = 0.1;
  user->bounds[0] = user->bounds[2] = user->bounds[4] = 0.0;
  user->bounds[1] = user->bounds[3] = user->bounds[5] = 1.0;
  user->xref   = user->bounds[1] / 2;
  user->yref   = user->bounds[3] / 2;
  user->nu     = 0.05;
  user->usemg  = PETSC_FALSE;
  user->io     = PETSC_FALSE;
  user->usetri = PETSC_FALSE;
  user->error  = PETSC_FALSE;
  bc           = (PetscInt)DIRICHLET;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the inhomogeneous Poisson equation", "ex35.cxx");
  ierr = PetscOptionsInt("-problem", "The type of problem being solved (controls forcing function)", "ex35.cxx", user->problem, &user->problem, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-n", "The elements in each direction", "ex35.cxx", user->n, &user->n, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-levels", "Number of levels in the multigrid hierarchy", "ex35.cxx", user->nlevels, &user->nlevels, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-rho", "The conductivity", "ex35.cxx", user->rho, &user->rho, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-x", "The domain size in x-direction", "ex35.cxx", user->bounds[1], &user->bounds[1], NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-y", "The domain size in y-direction", "ex35.cxx", user->bounds[3], &user->bounds[3], NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-xref", "The x-coordinate of Gaussian center (for -problem 1)", "ex35.cxx", user->xref, &user->xref, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-yref", "The y-coordinate of Gaussian center (for -problem 1)", "ex35.cxx", user->yref, &user->yref, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-nu", "The width of the Gaussian source (for -problem 1)", "ex35.cxx", user->nu, &user->nu, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mg", "Use multigrid preconditioner", "ex35.cxx", user->usemg, &user->usemg, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-io", "Write out the solution and mesh data", "ex35.cxx", user->io, &user->io, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tri", "Use triangles to discretize the domain", "ex35.cxx", user->usetri, &user->usetri, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-error", "Compute the discrete L_2 and L_inf errors of the solution", "ex35.cxx", user->error, &user->error, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-bc", "Type of boundary condition", "ex35.cxx", bcTypes, 2, bcTypes[0], &bc, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-file", "The mesh file for the problem", "ex35.cxx", "", user->filename, sizeof(user->filename), &user->use_extfile);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  if (user->problem < 1 || user->problem > 3) user->problem = 1;
  user->bcType = (BCType)bc;
  user->VPERE  = (user->usetri ? 3 : 4);
  if (user->problem == 3) {
    user->bounds[0] = user->bounds[2] = -0.5;
    user->bounds[1] = user->bounds[3] = 0.5;
    user->bounds[4] = user->bounds[5] = 0.5;
  }
  PetscFunctionReturn(0);
}


/*TEST

   build:
      requires: moab

   test:
      args: -levels 0 -nu .01 -n 10 -ksp_type cg -pc_type sor -ksp_converged_reason

   test:
      suffix: 2
      nsize: 2
      requires: hdf5
      args: -levels 3 -nu .01 -n 2 -mg -ksp_converged_reason

   test:
      suffix: 3
      nsize: 2
      requires: hdf5
      args: -problem 3 -file data/ex35_mesh.h5m -mg -levels 1 -ksp_converged_reason
      localrunfiles: data

TEST*/
