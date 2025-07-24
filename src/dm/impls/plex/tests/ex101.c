static char help[] = "Verify isoperiodic cone corrections";

#include <petscdmplex.h>
#include <petscsf.h>
#define EX "ex101.c"

// Creates periodic solution on a [0,1] x D domain for D dimension
static PetscErrorCode project_function(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscReal x_tot = 0;

  PetscFunctionBeginUser;
  for (PetscInt d = 0; d < dim; d++) x_tot += x[d];
  for (PetscInt c = 0; c < Nc; c++) {
    PetscScalar value = c % 2 ? PetscSinReal(2 * M_PI * x_tot) : PetscCosReal(2 * M_PI * x_tot);
    if (PetscAbsScalar(value) < 1e-7) value = 0.;
    u[c] = value;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateFEField(DM dm, PetscBool use_natural_fe, PetscInt num_comps)
{
  PetscInt degree;

  PetscFunctionBeginUser;
  { // Get degree of the coords section
    PetscFE    fe;
    PetscSpace basis_space;

    if (use_natural_fe) {
      PetscCall(DMGetField(dm, 0, NULL, (PetscObject *)&fe));
    } else {
      DM cdm;
      PetscCall(DMGetCoordinateDM(dm, &cdm));
      PetscCall(DMGetField(cdm, 0, NULL, (PetscObject *)&fe));
    }
    PetscCall(PetscFEGetBasisSpace(fe, &basis_space));
    PetscCall(PetscSpaceGetDegree(basis_space, &degree, NULL));
  }

  PetscCall(DMClearFields(dm));
  PetscCall(DMSetLocalSection(dm, NULL)); // See https://gitlab.com/petsc/petsc/-/issues/1669

  { // Setup fe to load in the initial condition data
    PetscFE  fe;
    PetscInt dim;

    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, num_comps, PETSC_FALSE, degree, PETSC_DETERMINE, &fe));
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
  PetscBool test_cgns_load = PETSC_FALSE;
  PetscInt  num_comps      = 1;

  PetscErrorCode (*funcs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx) = {project_function};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscOptionsBegin(comm, "", "ex101.c Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-test_cgns_load", "Test VecLoad using CGNS file", EX, test_cgns_load, &test_cgns_load, NULL));
  PetscCall(PetscOptionsInt("-num_comps", "Number of components in FE field", EX, num_comps, &num_comps, NULL));
  PetscOptionsEnd();

  PetscCall(DMCreate(comm, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(PetscObjectSetName((PetscObject)dm, "ex101_dm"));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(CreateFEField(dm, PETSC_FALSE, num_comps));

  // Verify that projected function on global vector (then projected onto local vector) is equal to projected function onto a local vector
  PetscCall(DMGetLocalVector(dm, &V_G2L));
  PetscCall(DMGetGlobalVector(dm, &V));
  PetscCall(DMProjectFunction(dm, 0, &funcs, NULL, INSERT_VALUES, V));
  PetscCall(DMGlobalToLocal(dm, V, INSERT_VALUES, V_G2L));

  PetscCall(DMGetLocalVector(dm, &V_local));
  PetscCall(DMProjectFunctionLocal(dm, 0, &funcs, NULL, INSERT_VALUES, V_local));
  PetscCall(VecViewFromOptions(V_local, NULL, "-local_view"));

  PetscCall(VecAXPY(V_G2L, -1, V_local));
  PetscCall(VecNorm(V_G2L, NORM_MAX, &norm));
  PetscReal tol = PetscDefined(USE_REAL___FLOAT128) ? 1e-12 : 1e4 * PETSC_MACHINE_EPSILON;
  if (norm > tol) PetscCall(PetscPrintf(comm, "Error! GlobalToLocal result does not match Local projection by norm %g\n", (double)norm));

  if (test_cgns_load) {
#ifndef PETSC_HAVE_CGNS
    SETERRQ(comm, PETSC_ERR_SUP, "PETSc not compiled with CGNS support");
#else
    PetscViewer viewer;
    DM          dm_read, dm_read_output;
    Vec         V_read, V_read_project2local, V_read_output2local;
    const char *filename = "test_file.cgns";

    // Write previous solution to filename
    PetscCall(PetscViewerCGNSOpen(comm, filename, FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(V_local, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    // Write DM from written CGNS file
    PetscCall(DMPlexCreateFromFile(comm, filename, "ex101_dm_read", PETSC_TRUE, &dm_read));
    PetscCall(DMSetFromOptions(dm_read));
    PetscCall(DMViewFromOptions(dm_read, NULL, "-dm_view"));

    { // Force isoperiodic point SF to be created to update sfNatural.
      // Needs to be done before removing the field corresponding to sfNatural
      PetscSection dummy_section;
      PetscCall(DMGetGlobalSection(dm_read, &dummy_section));
    }
    PetscCall(CreateFEField(dm_read, PETSC_TRUE, num_comps));

    // Use OutputDM as this doesn't use the isoperiodic pointSF and is compatible with sfNatural
    PetscCall(DMGetOutputDM(dm_read, &dm_read_output));
    PetscCall(DMGetLocalVector(dm_read, &V_read_project2local));
    PetscCall(DMGetLocalVector(dm_read, &V_read_output2local));
    PetscCall(PetscObjectSetName((PetscObject)V_read_output2local, "V_read_output2local"));
    PetscCall(PetscObjectSetName((PetscObject)V_read_project2local, "V_read_project2local"));

    PetscCall(DMProjectFunctionLocal(dm_read_output, 0, &funcs, NULL, INSERT_VALUES, V_read_project2local));
    PetscCall(VecViewFromOptions(V_read_project2local, NULL, "-project2local_view"));

    { // Read solution from file and communicate to local Vec
      PetscCall(DMGetGlobalVector(dm_read_output, &V_read));
      PetscCall(PetscObjectSetName((PetscObject)V_read, "V_read"));

      PetscCall(PetscViewerCGNSOpen(comm, filename, FILE_MODE_READ, &viewer));
      PetscCall(VecLoad(V_read, viewer));
      PetscCall(DMGlobalToLocal(dm_read_output, V_read, INSERT_VALUES, V_read_output2local));

      PetscCall(PetscViewerDestroy(&viewer));
      PetscCall(DMRestoreGlobalVector(dm_read_output, &V_read));
    }
    PetscCall(VecViewFromOptions(V_read_output2local, NULL, "-output2local_view"));

    PetscCall(VecAXPY(V_read_output2local, -1, V_read_project2local));
    PetscCall(VecNorm(V_read_output2local, NORM_MAX, &norm));
    if (norm > tol) PetscCall(PetscPrintf(comm, "Error! CGNS VecLoad result does not match Local projection by norm %g\n", (double)norm));

    PetscCall(DMRestoreLocalVector(dm_read, &V_read_project2local));
    PetscCall(DMRestoreLocalVector(dm_read, &V_read_output2local));
    PetscCall(DMDestroy(&dm_read));
#endif
  }

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
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/2x2x2_Q3_wave.cgns -dm_plex_cgns_parallel -dm_view ::ascii_info_detail -dm_plex_box_label true -dm_plex_box_label_bd periodic,periodic,periodic -petscpartitioner_type simple -test_cgns_load -num_comps 2

  test:
    requires: cgns parmetis
    suffix: cgns_parmetis
    nsize: 3
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/2x2x2_Q3_wave.cgns -dm_plex_cgns_parallel -dm_plex_box_label true -dm_plex_box_label_bd periodic,periodic,periodic -petscpartitioner_type parmetis -test_cgns_load -num_comps 2
    output_file: output/empty.out

TEST*/
