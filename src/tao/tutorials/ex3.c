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
  ierr = PetscOptionsBool("-use_riesz", "Use the Riesz map to achieve mesh independence", "ex3.c", user->use_riesz, &user->use_riesz, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-f", "filename to read", "ex3.c", filename, filename, sizeof(filename), &flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (!flg) {
    ierr = DMCreate(comm, dm);CHKERRQ(ierr);
    ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
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
    ierr = PetscViewerHDF5Open(comm, filename, FILE_MODE_READ, &viewer);CHKERRQ(ierr);

    /* create Vecs to read in the data from H5 */
    ierr = VecCreate(comm, &coordinates);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)coordinates, "coordinates");CHKERRQ(ierr);
    ierr = VecSetBlockSize(coordinates, dim);CHKERRQ(ierr);
    ierr = VecCreate(comm, &topology);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)topology, "topology");CHKERRQ(ierr);
    ierr = VecSetBlockSize(topology, vertices_per_cell);CHKERRQ(ierr);

    /* navigate to the right group */
    ierr = PetscViewerHDF5PushGroup(viewer, "/Mesh/mesh");CHKERRQ(ierr);

    /* Read the Vecs */
    ierr = VecLoad(coordinates, viewer);CHKERRQ(ierr);
    ierr = VecLoad(topology, viewer);CHKERRQ(ierr);

    /* do some ugly calculations */
    ierr = VecGetSize(topology, &numCells);CHKERRQ(ierr);
    numCells = numCells / vertices_per_cell;
    ierr = VecGetSize(coordinates, &numVertices);CHKERRQ(ierr);
    numVertices = numVertices / dim;

    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = VecGetArray(topology, &topo_f);CHKERRQ(ierr);
    /* and now we have to convert the double representation to integers to pass over, argh */
    ierr = PetscMalloc1(numCells*vertices_per_cell, &cells);CHKERRQ(ierr);
    for (j = 0; j < numCells*vertices_per_cell; j++) {
      cells[j] = (PetscInt) topo_f[j];
    }

    /* Now create the DM */
    ierr = DMPlexCreateFromCellListPetsc(comm, dim, numCells, numVertices, vertices_per_cell, PETSC_TRUE, cells, dim, coords, dm);CHKERRQ(ierr);
    /* Check for flipped first cell */
    {
      PetscReal v0[3], J[9], invJ[9], detJ;

      ierr = DMPlexComputeCellGeometryFEM(*dm, 0, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
      if (detJ < 0) {
        ierr = DMPlexOrientPoint(*dm, 0, -1);CHKERRQ(ierr);
        ierr = DMPlexComputeCellGeometryFEM(*dm, 0, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
        PetscCheckFalse(detJ < 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Something is wrong");
      }
    }
    ierr = DMPlexOrient(*dm);CHKERRQ(ierr);
    ierr = DMCreateLabel(*dm, "marker");CHKERRQ(ierr);
    ierr = DMGetLabel(*dm, "marker", &label);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(*dm, 1, label);CHKERRQ(ierr);
    ierr = DMPlexLabelComplete(*dm, label);CHKERRQ(ierr);

    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = VecRestoreArray(topology, &topo_f);CHKERRQ(ierr);
    ierr = PetscFree(cells);CHKERRQ(ierr);
    ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
    ierr = VecDestroy(&topology);CHKERRQ(ierr);
#else
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Reconfigure PETSc with --download-hdf5");
#endif
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
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
  PetscErrorCode ierr;

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
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, PETSC_TRUE, NULL, 4, &fe);CHKERRQ(ierr);

  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &user->data);CHKERRQ(ierr);

  /* ugh, this is hideous */
  /* y_d = interpolate(Expression("sin(x[0]) + .."), V) */
  ierr = PetscMalloc(1 * sizeof(void (*)(const PetscReal[], PetscScalar *, void *)), &wtf);CHKERRQ(ierr);
  wtf[0] = data_kernel;
  ierr = DMProjectFunction(dm, 0.0, wtf, NULL, INSERT_VALUES, user->data);CHKERRQ(ierr);
  ierr = PetscFree(wtf);CHKERRQ(ierr);

  /* assemble(inner(u, v)*dx), almost */
  ierr = DMClone(dm, &dm_mass);CHKERRQ(ierr);
  ierr = DMCopyDisc(dm, dm_mass);CHKERRQ(ierr);
  ierr = DMSetNumFields(dm_mass, 1);CHKERRQ(ierr);
  ierr = DMPlexCopyCoordinates(dm, dm_mass);CHKERRQ(ierr); /* why do I have to do this separately? */
  ierr = DMGetDS(dm_mass, &prob_mass);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob_mass, 0, 0, mass_kernel, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob_mass, 0, (PetscObject) fe);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm_mass, &user->mass);CHKERRQ(ierr);
  ierr = DMPlexSNESComputeJacobianFEM(dm_mass, user->data, user->mass, user->mass, NULL);CHKERRQ(ierr);
  ierr = MatSetOption(user->mass, MAT_SYMMETRIC, PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMDestroy(&dm_mass);CHKERRQ(ierr);

  /* inner(grad(u), grad(v))*dx with homogeneous Dirichlet boundary conditions */
  ierr = DMClone(dm, &dm_laplace);CHKERRQ(ierr);
  ierr = DMCopyDisc(dm, dm_laplace);CHKERRQ(ierr);
  ierr = DMSetNumFields(dm_laplace, 1);CHKERRQ(ierr);
  ierr = DMPlexCopyCoordinates(dm, dm_laplace);CHKERRQ(ierr);
  ierr = DMGetDS(dm_laplace, &prob_laplace);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob_laplace, 0, 0, NULL, NULL, NULL, laplace_kernel);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob_laplace, 0, (PetscObject) fe);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm_laplace, &user->laplace);CHKERRQ(ierr);
  ierr = DMPlexSNESComputeJacobianFEM(dm_laplace, user->data, user->laplace, user->laplace, NULL);CHKERRQ(ierr);

  /* Code from Matt to get the indices associated with the boundary dofs */
  ierr = DMGetLabel(dm_laplace, "marker", &label);CHKERRQ(ierr);
  ierr = DMAddBoundary(dm_laplace, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) zero, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm_laplace, &section);CHKERRQ(ierr);
  ierr = DMLabelGetStratumSize(label, 1, &n);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(label, 1, &is);CHKERRQ(ierr);
  ierr = ISGetIndices(is, &points);CHKERRQ(ierr);
  user->num_bc_dofs = 0;
  for (p = 0; p < n; ++p) {
    ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
    user->num_bc_dofs += dof;
  }
  ierr = PetscMalloc1(user->num_bc_dofs, &user->bc_indices);CHKERRQ(ierr);
  for (p = 0, k = 0; p < n; ++p) {
    ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, points[p], &off);CHKERRQ(ierr);
    for (d = 0; d < dof; ++d) user->bc_indices[k++] = off+d;
  }
  ierr = ISRestoreIndices(is, &points);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = DMDestroy(&dm_laplace);CHKERRQ(ierr);

  /* This is how I handle boundary conditions. I can't figure out how to get
     plex to play with the way I want to impose the BCs. This loses symmetry,
     but not in a disastrous way. If someone can improve it, please do! */
  ierr = MatZeroRows(user->laplace, user->num_bc_dofs, user->bc_indices, 1.0, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscCalloc1(user->num_bc_dofs, &user->bc_values);CHKERRQ(ierr);

  /* also create the KSP for solving the Laplace system */
  ierr = KSPCreate(PETSC_COMM_WORLD, &user->ksp_laplace);CHKERRQ(ierr);
  ierr = KSPSetOperators(user->ksp_laplace, user->laplace, user->laplace);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(user->ksp_laplace, "laplace_");CHKERRQ(ierr);
  ierr = KSPSetFromOptions(user->ksp_laplace);CHKERRQ(ierr);

  /* A bit of setting up the user context */
  user->dm = dm;
  ierr = VecDuplicate(user->data, &user->state);CHKERRQ(ierr);
  ierr = VecDuplicate(user->data, &user->adjoint);CHKERRQ(ierr);
  ierr = VecDuplicate(user->data, &user->tmp1);CHKERRQ(ierr);
  ierr = VecDuplicate(user->data, &user->tmp2);CHKERRQ(ierr);

  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode DestroyCtx(AppCtx* user)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  ierr = MatDestroy(&user->mass);CHKERRQ(ierr);
  ierr = MatDestroy(&user->laplace);CHKERRQ(ierr);
  ierr = KSPDestroy(&user->ksp_laplace);CHKERRQ(ierr);
  ierr = VecDestroy(&user->data);CHKERRQ(ierr);
  ierr = VecDestroy(&user->state);CHKERRQ(ierr);
  ierr = VecDestroy(&user->adjoint);CHKERRQ(ierr);
  ierr = VecDestroy(&user->tmp1);CHKERRQ(ierr);
  ierr = VecDestroy(&user->tmp2);CHKERRQ(ierr);
  ierr = PetscFree(user->bc_indices);CHKERRQ(ierr);
  ierr = PetscFree(user->bc_values);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ReducedFunctionGradient(Tao tao, Vec u, PetscReal* func, Vec g, void* userv)
{
  PetscErrorCode ierr;
  AppCtx* user = (AppCtx*) userv;
  const PetscReal alpha = 1.0e-6; /* regularisation parameter */
  PetscReal inner;

  PetscFunctionBeginUser;

  ierr = MatMult(user->mass, u, user->tmp1);CHKERRQ(ierr);
  ierr = VecDot(u, user->tmp1, &inner);CHKERRQ(ierr);               /* regularisation contribution to */
  *func = alpha * 0.5 * inner;                                      /* the functional                 */

  ierr = VecSet(g, 0.0);CHKERRQ(ierr);
  ierr = VecAXPY(g, alpha, user->tmp1);CHKERRQ(ierr);               /* regularisation contribution to the gradient */

  /* Now compute the forward state. */
  ierr = VecSetValues(user->tmp1, user->num_bc_dofs, user->bc_indices, user->bc_values, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(user->tmp1);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(user->tmp1);CHKERRQ(ierr);
  ierr = KSPSolve(user->ksp_laplace, user->tmp1, user->state);CHKERRQ(ierr); /* forward solve */

  /* Now compute the adjoint state also. */
  ierr = VecCopy(user->state, user->tmp1);CHKERRQ(ierr);
  ierr = VecAXPY(user->tmp1, -1.0, user->data);CHKERRQ(ierr);
  ierr = MatMult(user->mass, user->tmp1, user->tmp2);CHKERRQ(ierr);
  ierr = VecDot(user->tmp1, user->tmp2, &inner);CHKERRQ(ierr);      /* misfit contribution to */
  *func += 0.5 * inner;                                             /* the functional         */

  ierr = VecSetValues(user->tmp2, user->num_bc_dofs, user->bc_indices, user->bc_values, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(user->tmp2);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(user->tmp2);CHKERRQ(ierr);
  ierr = KSPSolve(user->ksp_laplace, user->tmp2, user->adjoint);CHKERRQ(ierr); /* adjoint solve */

  /* And bring it home with the gradient. */
  ierr = MatMult(user->mass, user->adjoint, user->tmp1);CHKERRQ(ierr);
  ierr = VecAXPY(g, 1.0, user->tmp1);CHKERRQ(ierr);                 /* adjoint contribution to the gradient */

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
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = CreateCtx(dm, &user);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecSet(u, 0.0);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &lb);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &ub);CHKERRQ(ierr);
  ierr = VecSet(lb, 0.0);CHKERRQ(ierr); /* satisfied at the minimum anyway */
  ierr = VecSet(ub, 0.8);CHKERRQ(ierr); /* a nontrivial upper bound */

  ierr = TaoCreate(PETSC_COMM_WORLD, &tao);CHKERRQ(ierr);
  ierr = TaoSetSolution(tao, u);CHKERRQ(ierr);
  ierr = TaoSetObjectiveAndGradient(tao,NULL, ReducedFunctionGradient, &user);CHKERRQ(ierr);
  ierr = TaoSetVariableBounds(tao, lb, ub);CHKERRQ(ierr);
  ierr = TaoSetType(tao, TAOBLMVM);CHKERRQ(ierr);
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

  if (user.use_riesz) {
    ierr = TaoLMVMSetH0(tao, user.mass);CHKERRQ(ierr);       /* crucial for mesh independence */
    ierr = TaoSetGradientNorm(tao, user.mass);CHKERRQ(ierr);
  }

  ierr = TaoSolve(tao);CHKERRQ(ierr);

  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-sol_view");CHKERRQ(ierr);

  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&lb);CHKERRQ(ierr);
  ierr = VecDestroy(&ub);CHKERRQ(ierr);
  ierr = DestroyCtx(&user);CHKERRQ(ierr);
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
