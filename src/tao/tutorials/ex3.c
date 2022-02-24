static char help[] = "Reduced formulation of the mother problem of PDE-constrained optimisation.\n";

/*F
  We solve the mother problem

  min 1/2 || y - y_d ||^2_2 + \alpha/2 || u ||^2_2

  subject to

          - \laplace y = u          on \Omega
                     y = 0          on \Gamma

  where u is in L^2 and y is in H^1_0.

  We formulate the reduced problem solely in terms of the control
  by using the state equation to express y in terms of u, and then
  apply LMVM/BLMVM to the resulting reduced problem.

  Mesh independence is achieved by configuring the Riesz map for the control
  space.

  Example meshes where the Riesz map is crucial can be downloaded from the
  http://people.maths.ox.ac.uk/~farrellp/meshes.tar.gz

  Contributed by: Patrick Farrell <patrick.farrell@maths.ox.ac.uk>

  Run with e.g.:
  ./ex3 -laplace_ksp_type cg -laplace_pc_type hypre -mat_lmvm_ksp_type cg -mat_lmvm_pc_type gamg -laplace_ksp_monitor_true_residual -tao_monitor -petscspace_degree 1 -tao_converged_reason -tao_gatol 1.0e-9 -dm_view hdf5:solution.h5 -sol_view hdf5:solution.h5::append -use_riesz 1 -f $DATAFILESPATH/meshes/mesh-1.h5

  and visualise in paraview with ../../../../petsc_gen_xdmf.py solution.h5.

  Toggle the Riesz map (-use_riesz 0) to see the difference setting the Riesz maps makes.

  TODO: broken for parallel runs
F*/

#include <petsc.h>
#include <petscfe.h>
#include <petscviewerhdf5.h>

typedef struct {
  DM  dm;
  Mat mass;
  Vec data;
  Vec state;
  Vec tmp1;
  Vec tmp2;
  Vec adjoint;
  Mat laplace;
  KSP ksp_laplace;
  PetscInt  num_bc_dofs;
  PetscInt* bc_indices;
  PetscScalar* bc_values;
  PetscBool use_riesz;
} AppCtx;

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscBool      flg;
  char           filename[2048];
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  filename[0] = '\0';
  user->use_riesz = PETSC_TRUE;

  ierr = PetscOptionsBegin(comm, "", "Poisson mother problem options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBool("-use_riesz", "Use the Riesz map to achieve mesh independence", "ex3.c", user->use_riesz, &user->use_riesz, NULL));
  CHKERRQ(PetscOptionsString("-f", "filename to read", "ex3.c", filename, filename, sizeof(filename), &flg));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (!flg) {
    CHKERRQ(DMCreate(comm, dm));
    CHKERRQ(DMSetType(*dm, DMPLEX));
  } else {
    /* TODO Eliminate this in favor of DMLoad() in new code */
#if defined(PETSC_HAVE_HDF5)
    const PetscInt vertices_per_cell = 3;
    PetscViewer    viewer;
    Vec            coordinates;
    Vec            topology;
    PetscInt       dim = 2, numCells;
    PetscInt       numVertices;
    PetscScalar*   coords;
    PetscScalar*   topo_f;
    PetscInt*      cells;
    PetscInt       j;
    DMLabel        label;

    /* Read in FEniCS HDF5 output */
    CHKERRQ(PetscViewerHDF5Open(comm, filename, FILE_MODE_READ, &viewer));

    /* create Vecs to read in the data from H5 */
    CHKERRQ(VecCreate(comm, &coordinates));
    CHKERRQ(PetscObjectSetName((PetscObject)coordinates, "coordinates"));
    CHKERRQ(VecSetBlockSize(coordinates, dim));
    CHKERRQ(VecCreate(comm, &topology));
    CHKERRQ(PetscObjectSetName((PetscObject)topology, "topology"));
    CHKERRQ(VecSetBlockSize(topology, vertices_per_cell));

    /* navigate to the right group */
    CHKERRQ(PetscViewerHDF5PushGroup(viewer, "/Mesh/mesh"));

    /* Read the Vecs */
    CHKERRQ(VecLoad(coordinates, viewer));
    CHKERRQ(VecLoad(topology, viewer));

    /* do some ugly calculations */
    CHKERRQ(VecGetSize(topology, &numCells));
    numCells = numCells / vertices_per_cell;
    CHKERRQ(VecGetSize(coordinates, &numVertices));
    numVertices = numVertices / dim;

    CHKERRQ(VecGetArray(coordinates, &coords));
    CHKERRQ(VecGetArray(topology, &topo_f));
    /* and now we have to convert the double representation to integers to pass over, argh */
    CHKERRQ(PetscMalloc1(numCells*vertices_per_cell, &cells));
    for (j = 0; j < numCells*vertices_per_cell; j++) {
      cells[j] = (PetscInt) topo_f[j];
    }

    /* Now create the DM */
    CHKERRQ(DMPlexCreateFromCellListPetsc(comm, dim, numCells, numVertices, vertices_per_cell, PETSC_TRUE, cells, dim, coords, dm));
    /* Check for flipped first cell */
    {
      PetscReal v0[3], J[9], invJ[9], detJ;

      CHKERRQ(DMPlexComputeCellGeometryFEM(*dm, 0, NULL, v0, J, invJ, &detJ));
      if (detJ < 0) {
        CHKERRQ(DMPlexOrientPoint(*dm, 0, -1));
        CHKERRQ(DMPlexComputeCellGeometryFEM(*dm, 0, NULL, v0, J, invJ, &detJ));
        PetscCheck(detJ >= 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Something is wrong");
      }
    }
    CHKERRQ(DMPlexOrient(*dm));
    CHKERRQ(DMCreateLabel(*dm, "marker"));
    CHKERRQ(DMGetLabel(*dm, "marker", &label));
    CHKERRQ(DMPlexMarkBoundaryFaces(*dm, 1, label));
    CHKERRQ(DMPlexLabelComplete(*dm, label));

    CHKERRQ(PetscViewerDestroy(&viewer));
    CHKERRQ(VecRestoreArray(coordinates, &coords));
    CHKERRQ(VecRestoreArray(topology, &topo_f));
    CHKERRQ(PetscFree(cells));
    CHKERRQ(VecDestroy(&coordinates));
    CHKERRQ(VecDestroy(&topology));
#else
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Reconfigure PETSc with --download-hdf5");
#endif
  }
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

void mass_kernel(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}

void laplace_kernel(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = 1.0;
}

/* data we seek to match */
PetscErrorCode data_kernel(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *y, void *ctx)
{
  *y = 1.0/(2*PETSC_PI*PETSC_PI) * PetscSinReal(PETSC_PI*x[0]) * PetscSinReal(PETSC_PI*x[1]);
  /* the associated control is sin(pi*x[0])*sin(pi*x[1]) */
  return 0;
}
PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  *u = 0.0;
  return 0;
}

PetscErrorCode CreateCtx(DM dm, AppCtx* user)
{

  DM             dm_mass;
  DM             dm_laplace;
  PetscDS        prob_mass;
  PetscDS        prob_laplace;
  PetscFE        fe;
  DMLabel        label;
  PetscSection   section;
  PetscInt       n, k, p, d;
  PetscInt       dof, off;
  IS             is;
  const PetscInt* points;
  const PetscInt dim = 2;
  const PetscInt id  = 1;
  PetscErrorCode (**wtf)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);

  PetscFunctionBeginUser;

  /* make the data we seek to match */
  CHKERRQ(PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, PETSC_TRUE, NULL, 4, &fe));

  CHKERRQ(DMSetField(dm, 0, NULL, (PetscObject) fe));
  CHKERRQ(DMCreateDS(dm));
  CHKERRQ(DMCreateGlobalVector(dm, &user->data));

  /* ugh, this is hideous */
  /* y_d = interpolate(Expression("sin(x[0]) + .."), V) */
  CHKERRQ(PetscMalloc(1 * sizeof(void (*)(const PetscReal[], PetscScalar *, void *)), &wtf));
  wtf[0] = data_kernel;
  CHKERRQ(DMProjectFunction(dm, 0.0, wtf, NULL, INSERT_VALUES, user->data));
  CHKERRQ(PetscFree(wtf));

  /* assemble(inner(u, v)*dx), almost */
  CHKERRQ(DMClone(dm, &dm_mass));
  CHKERRQ(DMCopyDisc(dm, dm_mass));
  CHKERRQ(DMSetNumFields(dm_mass, 1));
  CHKERRQ(DMPlexCopyCoordinates(dm, dm_mass)); /* why do I have to do this separately? */
  CHKERRQ(DMGetDS(dm_mass, &prob_mass));
  CHKERRQ(PetscDSSetJacobian(prob_mass, 0, 0, mass_kernel, NULL, NULL, NULL));
  CHKERRQ(PetscDSSetDiscretization(prob_mass, 0, (PetscObject) fe));
  CHKERRQ(DMCreateMatrix(dm_mass, &user->mass));
  CHKERRQ(DMPlexSNESComputeJacobianFEM(dm_mass, user->data, user->mass, user->mass, NULL));
  CHKERRQ(MatSetOption(user->mass, MAT_SYMMETRIC, PETSC_TRUE));
  CHKERRQ(DMDestroy(&dm_mass));

  /* inner(grad(u), grad(v))*dx with homogeneous Dirichlet boundary conditions */
  CHKERRQ(DMClone(dm, &dm_laplace));
  CHKERRQ(DMCopyDisc(dm, dm_laplace));
  CHKERRQ(DMSetNumFields(dm_laplace, 1));
  CHKERRQ(DMPlexCopyCoordinates(dm, dm_laplace));
  CHKERRQ(DMGetDS(dm_laplace, &prob_laplace));
  CHKERRQ(PetscDSSetJacobian(prob_laplace, 0, 0, NULL, NULL, NULL, laplace_kernel));
  CHKERRQ(PetscDSSetDiscretization(prob_laplace, 0, (PetscObject) fe));
  CHKERRQ(DMCreateMatrix(dm_laplace, &user->laplace));
  CHKERRQ(DMPlexSNESComputeJacobianFEM(dm_laplace, user->data, user->laplace, user->laplace, NULL));

  /* Code from Matt to get the indices associated with the boundary dofs */
  CHKERRQ(DMGetLabel(dm_laplace, "marker", &label));
  CHKERRQ(DMAddBoundary(dm_laplace, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) zero, NULL, NULL, NULL));
  CHKERRQ(DMGetLocalSection(dm_laplace, &section));
  CHKERRQ(DMLabelGetStratumSize(label, 1, &n));
  CHKERRQ(DMLabelGetStratumIS(label, 1, &is));
  CHKERRQ(ISGetIndices(is, &points));
  user->num_bc_dofs = 0;
  for (p = 0; p < n; ++p) {
    CHKERRQ(PetscSectionGetDof(section, points[p], &dof));
    user->num_bc_dofs += dof;
  }
  CHKERRQ(PetscMalloc1(user->num_bc_dofs, &user->bc_indices));
  for (p = 0, k = 0; p < n; ++p) {
    CHKERRQ(PetscSectionGetDof(section, points[p], &dof));
    CHKERRQ(PetscSectionGetOffset(section, points[p], &off));
    for (d = 0; d < dof; ++d) user->bc_indices[k++] = off+d;
  }
  CHKERRQ(ISRestoreIndices(is, &points));
  CHKERRQ(ISDestroy(&is));
  CHKERRQ(DMDestroy(&dm_laplace));

  /* This is how I handle boundary conditions. I can't figure out how to get
     plex to play with the way I want to impose the BCs. This loses symmetry,
     but not in a disastrous way. If someone can improve it, please do! */
  CHKERRQ(MatZeroRows(user->laplace, user->num_bc_dofs, user->bc_indices, 1.0, NULL, NULL));
  CHKERRQ(PetscCalloc1(user->num_bc_dofs, &user->bc_values));

  /* also create the KSP for solving the Laplace system */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD, &user->ksp_laplace));
  CHKERRQ(KSPSetOperators(user->ksp_laplace, user->laplace, user->laplace));
  CHKERRQ(KSPSetOptionsPrefix(user->ksp_laplace, "laplace_"));
  CHKERRQ(KSPSetFromOptions(user->ksp_laplace));

  /* A bit of setting up the user context */
  user->dm = dm;
  CHKERRQ(VecDuplicate(user->data, &user->state));
  CHKERRQ(VecDuplicate(user->data, &user->adjoint));
  CHKERRQ(VecDuplicate(user->data, &user->tmp1));
  CHKERRQ(VecDuplicate(user->data, &user->tmp2));

  CHKERRQ(PetscFEDestroy(&fe));

  PetscFunctionReturn(0);
}

PetscErrorCode DestroyCtx(AppCtx* user)
{
  PetscFunctionBeginUser;

  CHKERRQ(MatDestroy(&user->mass));
  CHKERRQ(MatDestroy(&user->laplace));
  CHKERRQ(KSPDestroy(&user->ksp_laplace));
  CHKERRQ(VecDestroy(&user->data));
  CHKERRQ(VecDestroy(&user->state));
  CHKERRQ(VecDestroy(&user->adjoint));
  CHKERRQ(VecDestroy(&user->tmp1));
  CHKERRQ(VecDestroy(&user->tmp2));
  CHKERRQ(PetscFree(user->bc_indices));
  CHKERRQ(PetscFree(user->bc_values));

  PetscFunctionReturn(0);
}

PetscErrorCode ReducedFunctionGradient(Tao tao, Vec u, PetscReal* func, Vec g, void* userv)
{
  AppCtx* user = (AppCtx*) userv;
  const PetscReal alpha = 1.0e-6; /* regularisation parameter */
  PetscReal inner;

  PetscFunctionBeginUser;

  CHKERRQ(MatMult(user->mass, u, user->tmp1));
  CHKERRQ(VecDot(u, user->tmp1, &inner));               /* regularisation contribution to */
  *func = alpha * 0.5 * inner;                                      /* the functional                 */

  CHKERRQ(VecSet(g, 0.0));
  CHKERRQ(VecAXPY(g, alpha, user->tmp1));               /* regularisation contribution to the gradient */

  /* Now compute the forward state. */
  CHKERRQ(VecSetValues(user->tmp1, user->num_bc_dofs, user->bc_indices, user->bc_values, INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(user->tmp1));
  CHKERRQ(VecAssemblyEnd(user->tmp1));
  CHKERRQ(KSPSolve(user->ksp_laplace, user->tmp1, user->state)); /* forward solve */

  /* Now compute the adjoint state also. */
  CHKERRQ(VecCopy(user->state, user->tmp1));
  CHKERRQ(VecAXPY(user->tmp1, -1.0, user->data));
  CHKERRQ(MatMult(user->mass, user->tmp1, user->tmp2));
  CHKERRQ(VecDot(user->tmp1, user->tmp2, &inner));      /* misfit contribution to */
  *func += 0.5 * inner;                                             /* the functional         */

  CHKERRQ(VecSetValues(user->tmp2, user->num_bc_dofs, user->bc_indices, user->bc_values, INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(user->tmp2));
  CHKERRQ(VecAssemblyEnd(user->tmp2));
  CHKERRQ(KSPSolve(user->ksp_laplace, user->tmp2, user->adjoint)); /* adjoint solve */

  /* And bring it home with the gradient. */
  CHKERRQ(MatMult(user->mass, user->adjoint, user->tmp1));
  CHKERRQ(VecAXPY(g, 1.0, user->tmp1));                 /* adjoint contribution to the gradient */

  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  Tao            tao;
  Vec            u, lb, ub;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  CHKERRQ(CreateCtx(dm, &user));

  CHKERRQ(DMCreateGlobalVector(dm, &u));
  CHKERRQ(VecSet(u, 0.0));
  CHKERRQ(VecDuplicate(u, &lb));
  CHKERRQ(VecDuplicate(u, &ub));
  CHKERRQ(VecSet(lb, 0.0)); /* satisfied at the minimum anyway */
  CHKERRQ(VecSet(ub, 0.8)); /* a nontrivial upper bound */

  CHKERRQ(TaoCreate(PETSC_COMM_WORLD, &tao));
  CHKERRQ(TaoSetSolution(tao, u));
  CHKERRQ(TaoSetObjectiveAndGradient(tao,NULL, ReducedFunctionGradient, &user));
  CHKERRQ(TaoSetVariableBounds(tao, lb, ub));
  CHKERRQ(TaoSetType(tao, TAOBLMVM));
  CHKERRQ(TaoSetFromOptions(tao));

  if (user.use_riesz) {
    CHKERRQ(TaoLMVMSetH0(tao, user.mass));       /* crucial for mesh independence */
    CHKERRQ(TaoSetGradientNorm(tao, user.mass));
  }

  CHKERRQ(TaoSolve(tao));

  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));
  CHKERRQ(VecViewFromOptions(u, NULL, "-sol_view"));

  CHKERRQ(TaoDestroy(&tao));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&lb));
  CHKERRQ(VecDestroy(&ub));
  CHKERRQ(DestroyCtx(&user));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    build:
      requires: !complex !single

    test:
      requires: hdf5 double datafilespath !defined(PETSC_USE_64BIT_INDICES) hypre
      args: -laplace_ksp_type cg -laplace_pc_type hypre -laplace_ksp_monitor_true_residual -laplace_ksp_max_it 5 -mat_lmvm_ksp_type cg -mat_lmvm_ksp_rtol 1e-5 -mat_lmvm_pc_type gamg -tao_monitor -petscspace_degree 1 -tao_converged_reason -tao_gatol 1.0e-6 -dm_view hdf5:solution.h5 -sol_view hdf5:solution.h5::append -use_riesz 1 -f $DATAFILESPATH/meshes/mesh-1.h5
      filter: sed -e "s/-nan/nan/g"

    test:
      suffix: guess_pod
      requires: double triangle
      args: -laplace_ksp_type cg -laplace_pc_type gamg -laplace_ksp_monitor_true_residual -laplace_ksp_max_it 8 -laplace_pc_gamg_esteig_ksp_type cg -laplace_pc_gamg_esteig_ksp_max_it 5 -laplace_mg_levels_ksp_chebyshev_esteig 0,0.25,0,1.0 -laplace_ksp_converged_reason -mat_lmvm_ksp_type cg -mat_lmvm_ksp_rtol 1e-5 -mat_lmvm_pc_type gamg -mat_lmvm_pc_gamg_esteig_ksp_type cg -mat_lmvm_pc_gamg_esteig_ksp_max_it 3 -tao_monitor -petscspace_degree 1 -tao_converged_reason -dm_refine 0 -laplace_ksp_guess_type pod -tao_gatol 1e-6
      filter: sed -e "s/-nan/nan/g" -e "s/-NaN/nan/g" -e "s/NaN/nan/g" -e "s/CONVERGED_RTOL iterations 9/CONVERGED_RTOL iterations 8/g" -e "s/CONVERGED_RTOL iterations 4/CONVERGED_RTOL iterations 3/g"

TEST*/
