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

  ierr = PetscOptionsBegin(comm, "", "Poisson mother problem options", "DMPLEX");PetscCall(ierr);
  PetscCall(PetscOptionsBool("-use_riesz", "Use the Riesz map to achieve mesh independence", "ex3.c", user->use_riesz, &user->use_riesz, NULL));
  PetscCall(PetscOptionsString("-f", "filename to read", "ex3.c", filename, filename, sizeof(filename), &flg));
  ierr = PetscOptionsEnd();PetscCall(ierr);

  if (!flg) {
    PetscCall(DMCreate(comm, dm));
    PetscCall(DMSetType(*dm, DMPLEX));
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
    PetscCall(PetscViewerHDF5Open(comm, filename, FILE_MODE_READ, &viewer));

    /* create Vecs to read in the data from H5 */
    PetscCall(VecCreate(comm, &coordinates));
    PetscCall(PetscObjectSetName((PetscObject)coordinates, "coordinates"));
    PetscCall(VecSetBlockSize(coordinates, dim));
    PetscCall(VecCreate(comm, &topology));
    PetscCall(PetscObjectSetName((PetscObject)topology, "topology"));
    PetscCall(VecSetBlockSize(topology, vertices_per_cell));

    /* navigate to the right group */
    PetscCall(PetscViewerHDF5PushGroup(viewer, "/Mesh/mesh"));

    /* Read the Vecs */
    PetscCall(VecLoad(coordinates, viewer));
    PetscCall(VecLoad(topology, viewer));

    /* do some ugly calculations */
    PetscCall(VecGetSize(topology, &numCells));
    numCells = numCells / vertices_per_cell;
    PetscCall(VecGetSize(coordinates, &numVertices));
    numVertices = numVertices / dim;

    PetscCall(VecGetArray(coordinates, &coords));
    PetscCall(VecGetArray(topology, &topo_f));
    /* and now we have to convert the double representation to integers to pass over, argh */
    PetscCall(PetscMalloc1(numCells*vertices_per_cell, &cells));
    for (j = 0; j < numCells*vertices_per_cell; j++) {
      cells[j] = (PetscInt) topo_f[j];
    }

    /* Now create the DM */
    PetscCall(DMPlexCreateFromCellListPetsc(comm, dim, numCells, numVertices, vertices_per_cell, PETSC_TRUE, cells, dim, coords, dm));
    /* Check for flipped first cell */
    {
      PetscReal v0[3], J[9], invJ[9], detJ;

      PetscCall(DMPlexComputeCellGeometryFEM(*dm, 0, NULL, v0, J, invJ, &detJ));
      if (detJ < 0) {
        PetscCall(DMPlexOrientPoint(*dm, 0, -1));
        PetscCall(DMPlexComputeCellGeometryFEM(*dm, 0, NULL, v0, J, invJ, &detJ));
        PetscCheck(detJ >= 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Something is wrong");
      }
    }
    PetscCall(DMPlexOrient(*dm));
    PetscCall(DMCreateLabel(*dm, "marker"));
    PetscCall(DMGetLabel(*dm, "marker", &label));
    PetscCall(DMPlexMarkBoundaryFaces(*dm, 1, label));
    PetscCall(DMPlexLabelComplete(*dm, label));

    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(VecRestoreArray(coordinates, &coords));
    PetscCall(VecRestoreArray(topology, &topo_f));
    PetscCall(PetscFree(cells));
    PetscCall(VecDestroy(&coordinates));
    PetscCall(VecDestroy(&topology));
#else
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Reconfigure PETSc with --download-hdf5");
#endif
  }
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
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
  PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, PETSC_TRUE, NULL, 4, &fe));

  PetscCall(DMSetField(dm, 0, NULL, (PetscObject) fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(DMCreateGlobalVector(dm, &user->data));

  /* ugh, this is hideous */
  /* y_d = interpolate(Expression("sin(x[0]) + .."), V) */
  PetscCall(PetscMalloc(1 * sizeof(void (*)(const PetscReal[], PetscScalar *, void *)), &wtf));
  wtf[0] = data_kernel;
  PetscCall(DMProjectFunction(dm, 0.0, wtf, NULL, INSERT_VALUES, user->data));
  PetscCall(PetscFree(wtf));

  /* assemble(inner(u, v)*dx), almost */
  PetscCall(DMClone(dm, &dm_mass));
  PetscCall(DMCopyDisc(dm, dm_mass));
  PetscCall(DMSetNumFields(dm_mass, 1));
  PetscCall(DMPlexCopyCoordinates(dm, dm_mass)); /* why do I have to do this separately? */
  PetscCall(DMGetDS(dm_mass, &prob_mass));
  PetscCall(PetscDSSetJacobian(prob_mass, 0, 0, mass_kernel, NULL, NULL, NULL));
  PetscCall(PetscDSSetDiscretization(prob_mass, 0, (PetscObject) fe));
  PetscCall(DMCreateMatrix(dm_mass, &user->mass));
  PetscCall(DMPlexSNESComputeJacobianFEM(dm_mass, user->data, user->mass, user->mass, NULL));
  PetscCall(MatSetOption(user->mass, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(DMDestroy(&dm_mass));

  /* inner(grad(u), grad(v))*dx with homogeneous Dirichlet boundary conditions */
  PetscCall(DMClone(dm, &dm_laplace));
  PetscCall(DMCopyDisc(dm, dm_laplace));
  PetscCall(DMSetNumFields(dm_laplace, 1));
  PetscCall(DMPlexCopyCoordinates(dm, dm_laplace));
  PetscCall(DMGetDS(dm_laplace, &prob_laplace));
  PetscCall(PetscDSSetJacobian(prob_laplace, 0, 0, NULL, NULL, NULL, laplace_kernel));
  PetscCall(PetscDSSetDiscretization(prob_laplace, 0, (PetscObject) fe));
  PetscCall(DMCreateMatrix(dm_laplace, &user->laplace));
  PetscCall(DMPlexSNESComputeJacobianFEM(dm_laplace, user->data, user->laplace, user->laplace, NULL));

  /* Code from Matt to get the indices associated with the boundary dofs */
  PetscCall(DMGetLabel(dm_laplace, "marker", &label));
  PetscCall(DMAddBoundary(dm_laplace, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) zero, NULL, NULL, NULL));
  PetscCall(DMGetLocalSection(dm_laplace, &section));
  PetscCall(DMLabelGetStratumSize(label, 1, &n));
  PetscCall(DMLabelGetStratumIS(label, 1, &is));
  PetscCall(ISGetIndices(is, &points));
  user->num_bc_dofs = 0;
  for (p = 0; p < n; ++p) {
    PetscCall(PetscSectionGetDof(section, points[p], &dof));
    user->num_bc_dofs += dof;
  }
  PetscCall(PetscMalloc1(user->num_bc_dofs, &user->bc_indices));
  for (p = 0, k = 0; p < n; ++p) {
    PetscCall(PetscSectionGetDof(section, points[p], &dof));
    PetscCall(PetscSectionGetOffset(section, points[p], &off));
    for (d = 0; d < dof; ++d) user->bc_indices[k++] = off+d;
  }
  PetscCall(ISRestoreIndices(is, &points));
  PetscCall(ISDestroy(&is));
  PetscCall(DMDestroy(&dm_laplace));

  /* This is how I handle boundary conditions. I can't figure out how to get
     plex to play with the way I want to impose the BCs. This loses symmetry,
     but not in a disastrous way. If someone can improve it, please do! */
  PetscCall(MatZeroRows(user->laplace, user->num_bc_dofs, user->bc_indices, 1.0, NULL, NULL));
  PetscCall(PetscCalloc1(user->num_bc_dofs, &user->bc_values));

  /* also create the KSP for solving the Laplace system */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &user->ksp_laplace));
  PetscCall(KSPSetOperators(user->ksp_laplace, user->laplace, user->laplace));
  PetscCall(KSPSetOptionsPrefix(user->ksp_laplace, "laplace_"));
  PetscCall(KSPSetFromOptions(user->ksp_laplace));

  /* A bit of setting up the user context */
  user->dm = dm;
  PetscCall(VecDuplicate(user->data, &user->state));
  PetscCall(VecDuplicate(user->data, &user->adjoint));
  PetscCall(VecDuplicate(user->data, &user->tmp1));
  PetscCall(VecDuplicate(user->data, &user->tmp2));

  PetscCall(PetscFEDestroy(&fe));

  PetscFunctionReturn(0);
}

PetscErrorCode DestroyCtx(AppCtx* user)
{
  PetscFunctionBeginUser;

  PetscCall(MatDestroy(&user->mass));
  PetscCall(MatDestroy(&user->laplace));
  PetscCall(KSPDestroy(&user->ksp_laplace));
  PetscCall(VecDestroy(&user->data));
  PetscCall(VecDestroy(&user->state));
  PetscCall(VecDestroy(&user->adjoint));
  PetscCall(VecDestroy(&user->tmp1));
  PetscCall(VecDestroy(&user->tmp2));
  PetscCall(PetscFree(user->bc_indices));
  PetscCall(PetscFree(user->bc_values));

  PetscFunctionReturn(0);
}

PetscErrorCode ReducedFunctionGradient(Tao tao, Vec u, PetscReal* func, Vec g, void* userv)
{
  AppCtx* user = (AppCtx*) userv;
  const PetscReal alpha = 1.0e-6; /* regularisation parameter */
  PetscReal inner;

  PetscFunctionBeginUser;

  PetscCall(MatMult(user->mass, u, user->tmp1));
  PetscCall(VecDot(u, user->tmp1, &inner));               /* regularisation contribution to */
  *func = alpha * 0.5 * inner;                                      /* the functional                 */

  PetscCall(VecSet(g, 0.0));
  PetscCall(VecAXPY(g, alpha, user->tmp1));               /* regularisation contribution to the gradient */

  /* Now compute the forward state. */
  PetscCall(VecSetValues(user->tmp1, user->num_bc_dofs, user->bc_indices, user->bc_values, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(user->tmp1));
  PetscCall(VecAssemblyEnd(user->tmp1));
  PetscCall(KSPSolve(user->ksp_laplace, user->tmp1, user->state)); /* forward solve */

  /* Now compute the adjoint state also. */
  PetscCall(VecCopy(user->state, user->tmp1));
  PetscCall(VecAXPY(user->tmp1, -1.0, user->data));
  PetscCall(MatMult(user->mass, user->tmp1, user->tmp2));
  PetscCall(VecDot(user->tmp1, user->tmp2, &inner));      /* misfit contribution to */
  *func += 0.5 * inner;                                             /* the functional         */

  PetscCall(VecSetValues(user->tmp2, user->num_bc_dofs, user->bc_indices, user->bc_values, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(user->tmp2));
  PetscCall(VecAssemblyEnd(user->tmp2));
  PetscCall(KSPSolve(user->ksp_laplace, user->tmp2, user->adjoint)); /* adjoint solve */

  /* And bring it home with the gradient. */
  PetscCall(MatMult(user->mass, user->adjoint, user->tmp1));
  PetscCall(VecAXPY(g, 1.0, user->tmp1));                 /* adjoint contribution to the gradient */

  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  Tao            tao;
  Vec            u, lb, ub;
  AppCtx         user;

  PetscCall(PetscInitialize(&argc, &argv, NULL,help));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(CreateCtx(dm, &user));

  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(VecSet(u, 0.0));
  PetscCall(VecDuplicate(u, &lb));
  PetscCall(VecDuplicate(u, &ub));
  PetscCall(VecSet(lb, 0.0)); /* satisfied at the minimum anyway */
  PetscCall(VecSet(ub, 0.8)); /* a nontrivial upper bound */

  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetSolution(tao, u));
  PetscCall(TaoSetObjectiveAndGradient(tao,NULL, ReducedFunctionGradient, &user));
  PetscCall(TaoSetVariableBounds(tao, lb, ub));
  PetscCall(TaoSetType(tao, TAOBLMVM));
  PetscCall(TaoSetFromOptions(tao));

  if (user.use_riesz) {
    PetscCall(TaoLMVMSetH0(tao, user.mass));       /* crucial for mesh independence */
    PetscCall(TaoSetGradientNorm(tao, user.mass));
  }

  PetscCall(TaoSolve(tao));

  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(VecViewFromOptions(u, NULL, "-sol_view"));

  PetscCall(TaoDestroy(&tao));
  PetscCall(DMDestroy(&dm));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&lb));
  PetscCall(VecDestroy(&ub));
  PetscCall(DestroyCtx(&user));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    build:
      requires: !complex !single

    test:
      requires: hdf5 double datafilespath !defined(PETSC_USE_64BIT_INDICES) hypre !cuda
      args: -laplace_ksp_type cg -laplace_pc_type hypre -laplace_ksp_monitor_true_residual -laplace_ksp_max_it 5 -mat_lmvm_ksp_type cg -mat_lmvm_ksp_rtol 1e-5 -mat_lmvm_pc_type gamg -tao_monitor -petscspace_degree 1 -tao_converged_reason -tao_gatol 1.0e-6 -dm_view hdf5:solution.h5 -sol_view hdf5:solution.h5::append -use_riesz 1 -f $DATAFILESPATH/meshes/mesh-1.h5
      filter: sed -e "s/-nan/nan/g"

    test:
      suffix: guess_pod
      requires: double triangle
      args: -laplace_ksp_type cg -laplace_pc_type gamg -laplace_ksp_monitor_true_residual -laplace_ksp_max_it 8 -laplace_pc_gamg_esteig_ksp_type cg -laplace_pc_gamg_esteig_ksp_max_it 5 -laplace_mg_levels_ksp_chebyshev_esteig 0,0.25,0,1.0 -laplace_ksp_converged_reason -mat_lmvm_ksp_type cg -mat_lmvm_ksp_rtol 1e-5 -mat_lmvm_pc_type gamg -mat_lmvm_pc_gamg_esteig_ksp_type cg -mat_lmvm_pc_gamg_esteig_ksp_max_it 3 -tao_monitor -petscspace_degree 1 -tao_converged_reason -dm_refine 0 -laplace_ksp_guess_type pod -tao_gatol 1e-6
      filter: sed -e "s/-nan/nan/g" -e "s/-NaN/nan/g" -e "s/NaN/nan/g" -e "s/CONVERGED_RTOL iterations 9/CONVERGED_RTOL iterations 8/g" -e "s/CONVERGED_RTOL iterations 4/CONVERGED_RTOL iterations 3/g"

TEST*/
