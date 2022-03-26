/*T
   Concepts: KSP^solving a system of linear equations using a MOAB based DM implementation.
   Concepts: KSP^Laplacian, 3d
   Processors: n
T*/

/*
Inhomogeneous Laplacian in 3-D. Modeled by the partial differential equation

   -div \rho grad u + \alpha u = f,  0 < x,y,z < 1,

Problem 1: (Default)

  Use the following exact solution with Dirichlet boundary condition

    u = sin(pi*x)*sin(pi*y)*sin(pi*z)

  with Dirichlet boundary conditions

     u = f(x,y,z) for x = 0, x = 1, y = 0, y = 1, z = 0, z = 1

  and \rho = 1.0, \alpha = 10.0 uniformly in the domain.

  Use an appropriate forcing function to measure convergence.

Usage:

  Measure convergence rate with uniform refinement with the options: "-problem 1 -error".

    mpiexec -n $NP ./ex36 -problem 1 -error -n 4 -levels 5 -pc_type gamg
    mpiexec -n $NP ./ex36 -problem 1 -error -n 8 -levels 4 -pc_type gamg
    mpiexec -n $NP ./ex36 -problem 1 -error -n 16 -levels 3 -pc_type mg
    mpiexec -n $NP ./ex36 -problem 1 -error -n 32 -levels 2 -pc_type mg
    mpiexec -n $NP ./ex36 -problem 1 -error -n 64 -levels 1 -mg
    mpiexec -n $NP ./ex36 -problem 1 -error -n 128 -levels 0 -mg

  Or with an external mesh file representing [0, 1]^3,

    mpiexec -n $NP ./ex36 -problem 1 -file ./external_mesh.h5m -levels 2 -error -pc_type mg

Problem 2:

  Use the forcing function

     f = e^{-((x-xr)^2+(y-yr)^2+(z-zr)^2)/\nu}

  with Dirichlet boundary conditions

     u = f(x,y,z) for x = 0, x = 1, y = 0, y = 1, z = 0, z = 1

  or pure Neumman boundary conditions

Usage:

  Run with different values of \rho and \nu (problem 1) to control diffusion and gaussian source spread. This uses the internal mesh generator implemented in DMMoab.

    mpiexec -n $NP ./ex36 -problem 2 -n 20 -nu 0.02 -rho 0.01
    mpiexec -n $NP ./ex36 -problem 2 -n 40 -nu 0.01 -rho 0.005 -io -ksp_monitor
    mpiexec -n $NP ./ex36 -problem 2 -n 80 -nu 0.01 -rho 0.005 -io -ksp_monitor -pc_type gamg
    mpiexec -n $NP ./ex36 -problem 2 -n 160 -bc neumann -nu 0.005 -rho 0.01 -io
    mpiexec -n $NP ./ex36 -problem 2 -n 320 -bc neumann -nu 0.001 -rho 1 -io

  Or with an external mesh file representing [0, 1]^3,

    mpiexec -n $NP ./ex36 -problem 2 -file ./external_mesh.h5m -levels 1 -pc_type gamg

*/

static char help[] = "\
                      Solves a three dimensional inhomogeneous Laplacian equation with a Gaussian source.\n \
                      Usage: ./ex36 -bc dirichlet -nu .01 -n 10\n";

/* PETSc includes */
#include <petscksp.h>
#include <petscdmmoab.h>

typedef enum {DIRICHLET, NEUMANN} BCType;

typedef struct {
  PetscInt  problem, dim, n, nlevels;
  PetscReal rho;
  PetscReal bounds[6];
  PetscReal xyzref[3];
  PetscReal nu;
  BCType    bcType;
  char      filename[PETSC_MAX_PATH_LEN];
  PetscBool use_extfile, io, error, usetet, usemg;

  /* Discretization parameters */
  int VPERE;
} UserContext;

static PetscErrorCode ComputeMatrix_MOAB(KSP, Mat, Mat, void*);
static PetscErrorCode ComputeRHS_MOAB(KSP, Vec, void*);
static PetscErrorCode ComputeDiscreteL2Error(KSP ksp, Vec err, UserContext *user);
static PetscErrorCode InitializeOptions(UserContext* user);

int main(int argc, char **argv)
{
  const char    *fields[1] = {"T-Variable"};
  DM             dm, dmref, *dmhierarchy;
  UserContext    user;
  PetscInt       k;
  KSP            ksp;
  PC             pc;
  Vec            errv;
  Mat            R;
  Vec            b, x;

  PetscCall(PetscInitialize(&argc, &argv, (char*)0, help));

  PetscCall(InitializeOptions(&user));

  /* Create the DM object from either a mesh file or from in-memory structured grid */
  if (user.use_extfile) {
    PetscCall(DMMoabLoadFromFile(PETSC_COMM_WORLD, user.dim, 1, user.filename, "", &dm));
  } else {
    PetscCall(DMMoabCreateBoxMesh(PETSC_COMM_WORLD, user.dim, user.usetet, NULL, user.n, 1, &dm));
  }
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMMoabSetFieldNames(dm, 1, fields));

  /* SetUp the data structures for DMMOAB */
  PetscCall(DMSetUp(dm));

  PetscCall(DMSetApplicationContext(dm, &user));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetComputeRHS(ksp, ComputeRHS_MOAB, &user));
  PetscCall(KSPSetComputeOperators(ksp, ComputeMatrix_MOAB, &user));

  if (user.nlevels) {
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PetscMalloc(sizeof(DM) * (user.nlevels + 1), &dmhierarchy));
    for (k = 0; k <= user.nlevels; k++) dmhierarchy[k] = NULL;

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of mesh hierarchy levels: %d\n", user.nlevels));
    PetscCall(DMMoabGenerateHierarchy(dm, user.nlevels, PETSC_NULL));

    // coarsest grid = 0
    // finest grid = nlevels
    dmhierarchy[0] = dm;
    PetscBool usehierarchy = PETSC_FALSE;
    if (usehierarchy) {
      PetscCall(DMRefineHierarchy(dm, user.nlevels, &dmhierarchy[1]));
    } else {
      for (k = 1; k <= user.nlevels; k++) {
        PetscCall(DMRefine(dmhierarchy[k - 1], MPI_COMM_NULL, &dmhierarchy[k]));
      }
    }
    dmref = dmhierarchy[user.nlevels];
    PetscObjectReference((PetscObject)dmref);

    if (user.usemg) {
      PetscCall(PCSetType(pc, PCMG));
      PetscCall(PCMGSetLevels(pc, user.nlevels + 1, NULL));
      PetscCall(PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
      PetscCall(PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH));
      PetscCall(PCMGSetCycleType(pc, PC_MG_CYCLE_V));
      PetscCall(PCMGSetNumberSmooth(pc, 2));

      for (k = 1; k <= user.nlevels; k++) {
        PetscCall(DMCreateInterpolation(dmhierarchy[k - 1], dmhierarchy[k], &R, NULL));
        PetscCall(PCMGSetInterpolation(pc, k, R));
        PetscCall(MatDestroy(&R));
      }
    }

    for (k = 1; k <= user.nlevels; k++) {
      PetscCall(DMDestroy(&dmhierarchy[k]));
    }
    PetscCall(PetscFree(dmhierarchy));
  } else {
    dmref = dm;
    PetscObjectReference((PetscObject)dm);
  }

  PetscCall(KSPSetDM(ksp, dmref));
  PetscCall(KSPSetFromOptions(ksp));

  /* Perform the actual solve */
  PetscCall(KSPSolve(ksp, NULL, NULL));
  PetscCall(KSPGetSolution(ksp, &x));
  PetscCall(KSPGetRhs(ksp, &b));

  if (user.error) {
    PetscCall(VecDuplicate(b, &errv));
    PetscCall(ComputeDiscreteL2Error(ksp, errv, &user));
    PetscCall(VecDestroy(&errv));
  }

  if (user.io) {
    /* Write out the solution along with the mesh */
    PetscCall(DMMoabSetGlobalFieldVector(dmref, x));
#ifdef MOAB_HAVE_HDF5
    PetscCall(DMMoabOutput(dmref, "ex36.h5m", ""));
#else
    /* MOAB does not support true parallel writers that aren't HDF5 based
       And so if you are using VTK as the output format in parallel,
       the data could be jumbled due to the order in which the processors
       write out their parts of the mesh and solution tags
    */
    PetscCall(DMMoabOutput(dmref, "ex36.vtk", ""));
#endif
  }

  /* Cleanup objects */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(DMDestroy(&dmref));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

PetscReal ComputeDiffusionCoefficient(PetscReal coords[3], UserContext* user)
{
  if (user->problem == 2) {
    if ((coords[0] > 1.0 / 3.0) && (coords[0] < 2.0 / 3.0) && (coords[1] > 1.0 / 3.0) && (coords[1] < 2.0 / 3.0) && (coords[2] > 1.0 / 3.0) && (coords[2] < 2.0 / 3.0)) return user->rho;
    else  return 1.0;
  } else return 1.0; /* problem = 1 */
}

PetscReal ComputeReactionCoefficient(PetscReal coords[3], UserContext* user)
{
  if (user->problem == 2) {
    if ((coords[0] > 1.0 / 3.0) && (coords[0] < 2.0 / 3.0) && (coords[1] > 1.0 / 3.0) && (coords[1] < 2.0 / 3.0) && (coords[2] > 1.0 / 3.0) && (coords[2] < 2.0 / 3.0)) return 10.0;
    else return 0.0;
  }
  else return 5.0; /* problem = 1 */
}

double ExactSolution(PetscReal coords[3], UserContext* user)
{
  if (user->problem == 2) {
    const PetscScalar xx = (coords[0] - user->xyzref[0]) * (coords[0] - user->xyzref[0]);
    const PetscScalar yy = (coords[1] - user->xyzref[1]) * (coords[1] - user->xyzref[1]);
    const PetscScalar zz = (coords[2] - user->xyzref[2]) * (coords[2] - user->xyzref[2]);
    return PetscExpScalar(-(xx + yy + zz) / user->nu);
  } else return sin(PETSC_PI * coords[0]) * sin(PETSC_PI * coords[1]) * sin(PETSC_PI * coords[2]);
}

PetscReal exact_solution(PetscReal x, PetscReal y, PetscReal z)
{
  PetscReal coords[3] = {x, y, z};
  return ExactSolution(coords, 0);
}

double ForcingFunction(PetscReal coords[3], UserContext* user)
{
  const PetscReal exact = ExactSolution(coords, user);
  if (user->problem == 2) {
    const PetscReal duxyz = ( (coords[0] - user->xyzref[0]) + (coords[1] - user->xyzref[1]) + (coords[2] - user->xyzref[2]));
    return (4.0 / user->nu * duxyz * duxyz - 6.0) * exact / user->nu;
  } else {
    const PetscReal reac = ComputeReactionCoefficient(coords, user);
    return (3.0 * PETSC_PI * PETSC_PI + reac) * exact;
  }
}

PetscErrorCode ComputeRHS_MOAB(KSP ksp, Vec b, void *ptr)
{
  UserContext*      user = (UserContext*)ptr;
  DM                dm;
  PetscInt          dof_indices[8], nc, npoints;
  PetscBool         dbdry[8];
  PetscReal         vpos[8 * 3];
  PetscInt          i, q, nconn;
  const moab::EntityHandle *connect;
  const moab::Range *elocal;
  moab::Interface*  mbImpl;
  PetscScalar       localv[8];
  PetscReal         *phi, *phypts, *jxw;
  PetscBool         elem_on_boundary;
  PetscQuadrature   quadratureObj;

  PetscFunctionBegin;
  PetscCall(KSPGetDM(ksp, &dm));

  /* reset the RHS */
  PetscCall(VecSet(b, 0.0));

  PetscCall(DMMoabFEMCreateQuadratureDefault (user->dim, user->VPERE, &quadratureObj));
  PetscCall(PetscQuadratureGetData(quadratureObj, NULL, &nc, &npoints, NULL, NULL));
  PetscCall(PetscMalloc3(user->VPERE * npoints, &phi, npoints * 3, &phypts, npoints, &jxw));

  /* get the essential MOAB mesh related quantities needed for FEM assembly */
  PetscCall(DMMoabGetInterface(dm, &mbImpl));
  PetscCall(DMMoabGetLocalElements(dm, &elocal));

  /* loop over local elements */
  for (moab::Range::iterator iter = elocal->begin(); iter != elocal->end(); iter++) {
    const moab::EntityHandle ehandle = *iter;

    /* Get connectivity information: */
    PetscCall(DMMoabGetElementConnectivity(dm, ehandle, &nconn, &connect));
    PetscCheckFalse(nconn != 4 && nconn != 8,PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only HEX8/TET4 element bases are supported in the current example. n(Connectivity)=%D.", nconn);

    /* get the coordinates of the element vertices */
    PetscCall(DMMoabGetVertexCoordinates(dm, nconn, connect, vpos));

    /* get the local DoF numbers to appropriately set the element contribution in the operator */
    PetscCall(DMMoabGetFieldDofsLocal(dm, nconn, connect, 0, dof_indices));

    /* compute the quadrature points transformed to the physical space and then
       compute the basis functions to compute local operators */
    PetscCall(DMMoabFEMComputeBasis(user->dim, nconn, vpos, quadratureObj, phypts, jxw, phi, NULL));

    PetscCall(PetscArrayzero(localv, nconn));
    /* Compute function over the locally owned part of the grid */
    for (q = 0; q < npoints; ++q) {
      const double ff = ForcingFunction(&phypts[3 * q], user);
      const PetscInt offset = q * nconn;

      for (i = 0; i < nconn; ++i) {
        localv[i] += jxw[q] * phi[offset + i] * ff;
      }
    }

    /* check if element is on the boundary */
    PetscCall(DMMoabIsEntityOnBoundary(dm, ehandle, &elem_on_boundary));

    /* apply dirichlet boundary conditions */
    if (elem_on_boundary && user->bcType == DIRICHLET) {

      /* get the list of nodes on boundary so that we can enforce dirichlet conditions strongly */
      PetscCall(DMMoabCheckBoundaryVertices(dm, nconn, connect, dbdry));

      for (i = 0; i < nconn; ++i) {
        if (dbdry[i]) {  /* dirichlet node */
          /* think about strongly imposing dirichlet */
          localv[i] = ForcingFunction(&vpos[3 * i], user);
        }
      }
    }

    /* set the values directly into appropriate locations. Can alternately use VecSetValues */
    PetscCall(VecSetValuesLocal(b, nconn, dof_indices, localv, ADD_VALUES));
  }

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  if (user->bcType == NEUMANN && false) {
    MatNullSpace nullspace;

    PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace));
    PetscCall(MatNullSpaceRemove(nullspace, b));
    PetscCall(MatNullSpaceDestroy(&nullspace));
  }

  /* Restore vectors */
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  PetscCall(PetscFree3(phi, phypts, jxw));
  PetscCall(PetscQuadratureDestroy(&quadratureObj));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMatrix_MOAB(KSP ksp, Mat J, Mat jac, void *ctx)
{
  UserContext       *user = (UserContext*)ctx;
  DM                dm;
  PetscInt          i, j, q, nconn, nglobale, nglobalv, nc, npoints, hlevel;
  PetscInt          dof_indices[8];
  PetscReal         vpos[8 * 3], rho, alpha;
  PetscBool         dbdry[8];
  const moab::EntityHandle *connect;
  const moab::Range *elocal;
  moab::Interface*  mbImpl;
  PetscBool         elem_on_boundary;
  PetscScalar       array[8 * 8];
  PetscReal         *phi, *dphi[3], *phypts, *jxw;
  PetscQuadrature   quadratureObj;

  PetscFunctionBeginUser;
  PetscCall(KSPGetDM(ksp, &dm));

  /* get the essential MOAB mesh related quantities needed for FEM assembly */
  PetscCall(DMMoabGetInterface(dm, &mbImpl));
  PetscCall(DMMoabGetLocalElements(dm, &elocal));
  PetscCall(DMMoabGetSize(dm, &nglobale, &nglobalv));
  PetscCall(DMMoabGetHierarchyLevel(dm, &hlevel));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "ComputeMatrix: Level = %d, N(elements) = %d, N(vertices) = %d \n", hlevel, nglobale, nglobalv));

  PetscCall(DMMoabFEMCreateQuadratureDefault ( user->dim, user->VPERE, &quadratureObj));
  PetscCall(PetscQuadratureGetData(quadratureObj, NULL, &nc, &npoints, NULL, NULL));
  PetscCall(PetscMalloc6(user->VPERE * npoints, &phi, user->VPERE * npoints, &dphi[0], user->VPERE * npoints, &dphi[1], user->VPERE * npoints, &dphi[2], npoints * 3, &phypts, npoints, &jxw));

  /* loop over local elements */
  for (moab::Range::iterator iter = elocal->begin(); iter != elocal->end(); iter++) {
    const moab::EntityHandle ehandle = *iter;

    /* Get connectivity information: */
    PetscCall(DMMoabGetElementConnectivity(dm, ehandle, &nconn, &connect));
    PetscCheckFalse(nconn != 4 && nconn != 8,PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only HEX8/TET4 element bases are supported in the current example. n(Connectivity)=%D.", nconn);

    /* get the coordinates of the element vertices */
    PetscCall(DMMoabGetVertexCoordinates(dm, nconn, connect, vpos));

    /* get the local DoF numbers to appropriately set the element contribution in the operator */
    PetscCall(DMMoabGetFieldDofsLocal(dm, nconn, connect, 0, dof_indices));

    /* compute the quadrature points transformed to the physical space and
       compute the basis functions and the derivatives wrt x, y and z directions */
    PetscCall(DMMoabFEMComputeBasis(user->dim, nconn, vpos, quadratureObj, phypts, jxw, phi, dphi));

    PetscCall(PetscArrayzero(array, nconn * nconn));

    /* Compute function over the locally owned part of the grid */
    for (q = 0; q < npoints; ++q) {

      /* compute the inhomogeneous diffusion coefficient at the quadrature point
          -- for large spatial variations, embed this property evaluation inside quadrature loop */
      rho   = ComputeDiffusionCoefficient(&phypts[q * 3], user);
      alpha = ComputeReactionCoefficient (&phypts[q * 3], user);

      const PetscInt offset = q * nconn;

      for (i = 0; i < nconn; ++i) {
        for (j = 0; j < nconn; ++j) {
          array[i * nconn + j] += jxw[q] * ( rho * ( dphi[0][offset + i] * dphi[0][offset + j] +
                                                     dphi[1][offset + i] * dphi[1][offset + j] +
                                                     dphi[2][offset + i] * dphi[2][offset + j])
                                             + alpha * ( phi[offset + i] * phi[offset + j]));
        }
      }
    }

    /* check if element is on the boundary */
    PetscCall(DMMoabIsEntityOnBoundary(dm, ehandle, &elem_on_boundary));

    /* apply dirichlet boundary conditions */
    if (elem_on_boundary && user->bcType == DIRICHLET) {

      /* get the list of nodes on boundary so that we can enforce dirichlet conditions strongly */
      PetscCall(DMMoabCheckBoundaryVertices(dm, nconn, connect, dbdry));

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

    /* set the values directly into appropriate locations. Can alternately use VecSetValues */
    PetscCall(MatSetValuesLocal(jac, nconn, dof_indices, nconn, dof_indices, array, ADD_VALUES));
  }

  PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));

  if (user->bcType == NEUMANN && false) {
    MatNullSpace nullspace;

    PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace));
    PetscCall(MatSetNullSpace(jac, nullspace));
    PetscCall(MatNullSpaceDestroy(&nullspace));
  }
  PetscCall(PetscFree6(phi, dphi[0], dphi[1], dphi[2], phypts, jxw));
  PetscCall(PetscQuadratureDestroy(&quadratureObj));
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

  PetscFunctionBegin;
  PetscCall(KSPGetDM(ksp, &dm));

  /* get the solution vector */
  PetscCall(KSPGetSolution(ksp, &sol));

  /* Get the internal reference to the vector arrays */
  PetscCall(VecGetArrayRead(sol, &x));
  PetscCall(VecGetSize(sol, &N));
  if (err) {
    /* reset the error vector */
    PetscCall(VecSet(err, 0.0));
    /* get array reference */
    PetscCall(VecGetArray(err, &e));
  }

  PetscCall(DMMoabGetLocalVertices(dm, &ownedvtx, NULL));

  /* Compute function over the locally owned part of the grid */
  for (moab::Range::iterator iter = ownedvtx->begin(); iter != ownedvtx->end(); iter++) {
    const moab::EntityHandle vhandle = *iter;
    PetscCall(DMMoabGetDofsBlockedLocal(dm, 1, &vhandle, &dof_index));

    /* compute the mid-point of the element and use a 1-point lumped quadrature */
    PetscCall(DMMoabGetVertexCoordinates(dm, 1, &vhandle, vpos));

    /* compute the discrete L2 error against the exact solution */
    const PetscScalar lerr = (ExactSolution(vpos, user) - x[dof_index]);
    l2err += lerr * lerr;
    if (linferr < fabs(lerr)) linferr = fabs(lerr);

    if (err) {
      /* set the discrete L2 error against the exact solution */
      e[dof_index] = lerr;
    }
  }

  PetscCallMPI(MPI_Allreduce(&l2err, &global_l2, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&linferr, &global_linf, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Computed Errors: L_2 = %f, L_inf = %f\n", sqrt(global_l2 / N), global_linf));

  /* Restore vectors */
  PetscCall(VecRestoreArrayRead(sol, &x));
  if (err) {
    PetscCall(VecRestoreArray(err, &e));
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
  user->dim     = 3; /* 3-Dimensional problem */
  user->problem = 1;
  user->n       = 2;
  user->nlevels = 2;
  user->rho     = 0.1;
  user->bounds[0] = user->bounds[2] = user->bounds[4] = 0.0;
  user->bounds[1] = user->bounds[3] = user->bounds[5] = 1.0;
  user->xyzref[0] = user->bounds[1] / 2;
  user->xyzref[1] = user->bounds[3] / 2;
  user->xyzref[2] = user->bounds[5] / 2;
  user->nu     = 0.05;
  user->usemg  = PETSC_FALSE;
  user->io     = PETSC_FALSE;
  user->usetet = PETSC_FALSE;
  user->error  = PETSC_FALSE;
  bc           = (PetscInt)DIRICHLET;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the inhomogeneous Poisson equation", "ex36.cxx");PetscCall(ierr);
  PetscCall(PetscOptionsInt("-problem", "The type of problem being solved (controls forcing function)", "ex36.cxx", user->problem, &user->problem, NULL));
  PetscCall(PetscOptionsInt("-n", "The elements in each direction", "ex36.cxx", user->n, &user->n, NULL));
  PetscCall(PetscOptionsInt("-levels", "Number of levels in the multigrid hierarchy", "ex36.cxx", user->nlevels, &user->nlevels, NULL));
  PetscCall(PetscOptionsReal("-rho", "The conductivity", "ex36.cxx", user->rho, &user->rho, NULL));
  PetscCall(PetscOptionsReal("-x", "The domain size in x-direction", "ex36.cxx", user->bounds[1], &user->bounds[1], NULL));
  PetscCall(PetscOptionsReal("-y", "The domain size in y-direction", "ex36.cxx", user->bounds[3], &user->bounds[3], NULL));
  PetscCall(PetscOptionsReal("-z", "The domain size in y-direction", "ex36.cxx", user->bounds[5], &user->bounds[5], NULL));
  PetscCall(PetscOptionsReal("-xref", "The x-coordinate of Gaussian center (for -problem 1)", "ex36.cxx", user->xyzref[0], &user->xyzref[0], NULL));
  PetscCall(PetscOptionsReal("-yref", "The y-coordinate of Gaussian center (for -problem 1)", "ex36.cxx", user->xyzref[1], &user->xyzref[1], NULL));
  PetscCall(PetscOptionsReal("-zref", "The y-coordinate of Gaussian center (for -problem 1)", "ex36.cxx", user->xyzref[2], &user->xyzref[2], NULL));
  PetscCall(PetscOptionsReal("-nu", "The width of the Gaussian source (for -problem 1)", "ex36.cxx", user->nu, &user->nu, NULL));
  PetscCall(PetscOptionsBool("-mg", "Use multigrid preconditioner", "ex36.cxx", user->usemg, &user->usemg, NULL));
  PetscCall(PetscOptionsBool("-io", "Write out the solution and mesh data", "ex36.cxx", user->io, &user->io, NULL));
  PetscCall(PetscOptionsBool("-tet", "Use tetrahedra to discretize the domain", "ex36.cxx", user->usetet, &user->usetet, NULL));
  PetscCall(PetscOptionsBool("-error", "Compute the discrete L_2 and L_inf errors of the solution", "ex36.cxx", user->error, &user->error, NULL));
  PetscCall(PetscOptionsEList("-bc", "Type of boundary condition", "ex36.cxx", bcTypes, 2, bcTypes[0], &bc, NULL));
  PetscCall(PetscOptionsString("-file", "The mesh file for the problem", "ex36.cxx", "", user->filename, sizeof(user->filename), &user->use_extfile));
  ierr = PetscOptionsEnd();PetscCall(ierr);

  if (user->problem < 1 || user->problem > 2) user->problem = 1;
  user->bcType = (BCType)bc;
  user->VPERE  = (user->usetet ? 4 : 8);
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: moab

   test:
      args: -levels 1 -nu .01 -n 4 -mg -ksp_converged_reason

   test:
      suffix: 2
      nsize: 2
      requires: hdf5
      args: -levels 2 -nu .01 -n 2 -mg -ksp_converged_reason

TEST*/
