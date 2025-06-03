#include "petscsf.h"
static char help[] = "Test CGNS writing output with isoperiodic boundaries\n\n";
// Also tests DMSetCoordinateDisc() for isoperiodic boundaries and projection = true

#include <petscdmplex.h>
#include <petscviewerhdf5.h>
#define EX "ex100.c"

static PetscErrorCode project_function(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscReal x_tot = 0;

  PetscFunctionBeginUser;
  for (PetscInt d = 0; d < dim; d++) x_tot += x[d];
  for (PetscInt c = 0; c < Nc; c++) u[c] = sin(2 * M_PI * x_tot);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM       dm_create, dm_read;
  Vec      V;
  char     file[PETSC_MAX_PATH_LEN];
  MPI_Comm comm;
  PetscInt solution_degree = 2;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");
  PetscCall(PetscOptionsInt("-solution_degree", "The input CGNS file", EX, solution_degree, &solution_degree, NULL));
  PetscCall(PetscOptionsString("-file", "The input CGNS file", EX, file, file, sizeof(file), NULL));
  PetscOptionsEnd();

  { // Create DM
    PetscCall(DMCreate(comm, &dm_create));
    PetscCall(DMSetType(dm_create, DMPLEX));
    PetscCall(DMSetOptionsPrefix(dm_create, "create_"));
    PetscCall(DMSetFromOptions(dm_create));
    PetscCall(DMViewFromOptions(dm_create, NULL, "-dm_create_view"));

    { // Setup fe to load in the initial condition data
      PetscFE  fe;
      PetscInt dim;

      PetscCall(DMGetDimension(dm_create, &dim));
      PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, solution_degree, PETSC_DETERMINE, &fe));
      PetscCall(DMAddField(dm_create, NULL, (PetscObject)fe));
      PetscCall(DMCreateDS(dm_create));
      PetscCall(PetscFEDestroy(&fe));
    }

    PetscErrorCode (*funcs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx) = {project_function};
    PetscCall(DMGetGlobalVector(dm_create, &V));
    PetscCall(DMProjectFunction(dm_create, 0, &funcs, NULL, INSERT_VALUES, V));
    PetscViewer viewer;
    PetscCall(PetscViewerCGNSOpen(comm, file, FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(V, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(DMRestoreGlobalVector(dm_create, &V));

    PetscCall(DMDestroy(&dm_create));
  }

  {
    PetscSection coord_section;
    PetscInt     cStart, cEnd;
    Vec          coords;
    PetscInt     cdim;
    PetscReal    ref_cell_bounding_box_size[3];

    PetscCall(DMPlexCreateFromFile(comm, file, "ex100_plex", PETSC_TRUE, &dm_read));
    PetscCall(DMViewFromOptions(dm_read, NULL, "-dm_read_view"));
    PetscCall(DMGetCoordinateDim(dm_read, &cdim));
    PetscCall(DMGetCoordinateSection(dm_read, &coord_section));
    PetscCall(DMGetCoordinates(dm_read, &coords));
    PetscCall(DMPlexGetHeightStratum(dm_read, 0, &cStart, &cEnd));
    for (PetscInt cell = cStart; cell < cEnd; cell++) {
      PetscInt     num_closure = 0;
      PetscScalar *cell_coords = NULL;
      PetscReal    cell_bounding_box_size[3], difference;
      PetscReal    min[3] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL};
      PetscReal    max[3] = {PETSC_MIN_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL};

      PetscCall(DMPlexVecGetClosure(dm_read, coord_section, coords, cell, &num_closure, &cell_coords));
      for (PetscInt n = 0; n < num_closure / cdim; n++) {
        for (PetscInt d = 0; d < cdim; ++d) {
          min[d] = PetscMin(min[d], PetscRealPart(cell_coords[n * cdim + d]));
          max[d] = PetscMax(max[d], PetscRealPart(cell_coords[n * cdim + d]));
        }
      }

      for (PetscInt d = 0; d < cdim; d++) cell_bounding_box_size[d] = max[d] - min[d];
      if (cell == cStart) PetscCall(PetscArraycpy(ref_cell_bounding_box_size, cell_bounding_box_size, cdim));

      for (PetscInt d = 0; d < cdim; d++) {
        difference = PetscAbsReal((ref_cell_bounding_box_size[d] - cell_bounding_box_size[d]) / ref_cell_bounding_box_size[d]);
        if (difference > PETSC_MACHINE_EPSILON * 100) {
          PetscPrintf(comm, "Cell %" PetscInt_FMT " doesn't match bounding box size of Cell %" PetscInt_FMT " in dimension %" PetscInt_FMT ". Relative difference = %g\n", cell, cStart, d, (double)difference);
        }
      }
      PetscCall(DMPlexVecRestoreClosure(dm_read, coord_section, coords, cell, &num_closure, &cell_coords));
    }
    PetscCall(DMDestroy(&dm_read));
  }

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: cgns
  test:
    # Coordinates of dm_create are linear, but the CGNS writer projects them to quadratic to match the solution.
    # The verification checks that all the cells of the resulting file are the same size
    args: -create_dm_plex_shape zbox -create_dm_plex_box_faces 2,2 -create_dm_plex_box_bd periodic,periodic -file test.cgns -dm_plex_cgns_parallel
    output_file: output/empty.out
TEST*/
