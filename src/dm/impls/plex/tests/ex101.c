static char help[] = "Verify isoperiodic cone corrections";

#include <petscdmplex.h>

// Creates periodic solution on a [0,1] x D domain for D dimension
static PetscErrorCode project_function(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscReal x_tot = 0;

  PetscFunctionBeginUser;
  for (PetscInt d = 0; d < dim; d++) x_tot += x[d];
  for (PetscInt c = 0; c < Nc; c++) {
    PetscScalar value = PetscSinReal(2 * M_PI * x_tot);
    if (PetscAbsScalar(value) < 1e-7) value = 0.;
    u[c] = value;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateFEField(DM dm)
{
  PetscInt degree;

  PetscFunctionBeginUser;
  { // Get degree of the coords section
    PetscFE    fe_coords;
    PetscSpace coord_space;
    DM         cdm;

    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMGetField(cdm, 0, NULL, (PetscObject *)&fe_coords));
    PetscCall(PetscFEGetBasisSpace(fe_coords, &coord_space));
    PetscCall(PetscSpaceGetDegree(coord_space, &degree, NULL));
  }

  PetscCall(DMClearFields(dm));
  PetscCall(DMSetLocalSection(dm, NULL)); // See https://gitlab.com/petsc/petsc/-/issues/1669

  { // Setup fe to load in the initial condition data
    PetscFE  fe;
    PetscInt dim;

    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, degree, PETSC_DETERMINE, &fe));
    PetscCall(DMAddField(dm, NULL, (PetscObject)fe));
    PetscCall(DMCreateDS(dm));
    PetscCall(PetscFEDestroy(&fe));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm  comm;
  DM        dm = NULL;
  Vec       V, V_G2L, V_local;
  PetscReal norm;
  PetscErrorCode (*funcs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx) = {project_function};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscCall(DMCreate(comm, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(PetscObjectSetName((PetscObject)dm, "ex101_dm"));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(CreateFEField(dm));

  // Verify that projected function on global vector (then projected onto local vector) is equal to projected function onto a local vector
  PetscCall(DMGetLocalVector(dm, &V_G2L));
  PetscCall(DMGetGlobalVector(dm, &V));
  PetscCall(DMProjectFunction(dm, 0, &funcs, NULL, INSERT_VALUES, V));
  PetscCall(DMGlobalToLocal(dm, V, INSERT_VALUES, V_G2L));

  PetscCall(DMGetLocalVector(dm, &V_local));
  PetscCall(DMProjectFunctionLocal(dm, 0, &funcs, NULL, INSERT_VALUES, V_local));

  PetscCall(VecAXPY(V_local, -1, V_G2L));
  PetscCall(VecNorm(V_local, NORM_2, &norm));
  PetscReal tol = PetscDefined(USE_REAL___FLOAT128) ? 1e-12 : 1e4 * PETSC_MACHINE_EPSILON;
  PetscCheck(norm < tol, PETSC_COMM_SELF, PETSC_ERR_PLIB, "GlobalToLocal result does not match Local projection by norm %g", (double)norm);

  PetscCall(DMRestoreGlobalVector(dm, &V));
  PetscCall(DMRestoreLocalVector(dm, &V_G2L));
  PetscCall(DMRestoreLocalVector(dm, &V_local));
  PetscCall(DMDestroy(&dm));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    nsize: 2
    args: -dm_plex_shape zbox -dm_plex_simplex 0 -dm_plex_dim 3 -dm_plex_box_faces 1,2,3 -dm_coord_space -dm_coord_petscspace_degree 3
    args: -dm_plex_box_bd periodic,periodic,periodic -dm_view ::ascii_info_detail -petscpartitioner_type simple

  test:
    requires: cgns
    suffix: cgns
    nsize: 3
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/2x2x2_Q3_wave.cgns -dm_plex_cgns_parallel -dm_view ::ascii_info_detail -dm_plex_box_label true -dm_plex_box_label_bd periodic,periodic,periodic -petscpartitioner_type simple

TEST*/
