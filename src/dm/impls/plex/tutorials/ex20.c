static char help[] = "Test DMPlexGetLETKFLocalizationMatrix.\n\n";

#include <petscdmplex.h>

int main(int argc, char **argv)
{
  DM        dm;
  Mat       H, Q;
  PetscInt  numobservations;
  PetscInt  dim = 1, n;
  PetscInt  faces[3];
  PetscReal lower[3] = {0.0, 0.0, 0.0};
  PetscReal upper[3] = {1.0, 1.0, 1.0};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  /* Get dimension and from options. We need the data here and Plex does not have access functions */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dm_plex_dim", &dim, NULL));
  n = 3;
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces", faces, &n, NULL));
  n = 3;
  PetscCall(PetscOptionsGetRealArray(NULL, NULL, "-dm_plex_box_lower", lower, &n, NULL));
  n = 3;
  PetscCall(PetscOptionsGetRealArray(NULL, NULL, "-dm_plex_box_upper", upper, &n, NULL));

  /* Create the mesh using DMPlexCreateBoxMesh (could pass parameters) */
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  /* Verify dimension matches */
  PetscInt dmDim;
  PetscCall(DMGetDimension(dm, &dmDim));
  PetscCheck(dmDim == dim, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "DM dimension %" PetscInt_FMT " does not match requested dimension %" PetscInt_FMT, dmDim, dim);

  /* Set number of local observations to use: 3^dim */
  numobservations = 1;
  for (PetscInt d = 0; d < dim && d < 2; d++) numobservations *= 3;

  /* Get number of vertices */
  PetscInt vStart, vEnd, numVertices;
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  numVertices = vEnd - vStart;

  /* Create a section for vertices (required for Global Point mapping) */
  PetscSection section;
  PetscInt     pStart, pEnd;
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(PetscSectionCreate(PETSC_COMM_WORLD, &section));
  PetscCall(PetscSectionSetNumFields(section, 1));
  PetscCall(PetscSectionSetChart(section, pStart, pEnd));
  for (PetscInt v = vStart; v < vEnd; ++v) PetscCall(PetscSectionSetDof(section, v, 1));
  PetscCall(PetscSectionSetUp(section));
  PetscCall(DMSetLocalSection(dm, section));
  PetscCall(PetscSectionDestroy(&section));

  /* Create global section */
  PetscSection globalSection;
  PetscCall(DMGetGlobalSection(dm, &globalSection));

  /* Count observations (every other vertex in each dimension) */
  PetscInt numlocalobs = 0;
  {
    Vec                coordinates;
    PetscSection       coordSection;
    const PetscScalar *coordArray;
    PetscInt           offset;

    PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
    PetscCall(DMGetCoordinateSection(dm, &coordSection));
    PetscCall(VecGetArrayRead(coordinates, &coordArray));

    for (PetscInt v = vStart; v < vEnd; v++) {
      PetscReal coords[3] = {0.0, 0.0, 0.0};
      PetscBool isObs     = PETSC_TRUE;

      PetscCall(PetscSectionGetOffset(coordSection, v, &offset));
      for (PetscInt d = 0; d < dim; d++) coords[d] = PetscRealPart(coordArray[offset + d]);

      /* Check if this vertex is at an observation location (every other grid point) */
      for (PetscInt d = 0; d < dim; d++) {
        PetscReal gridSpacing = (upper[d] - lower[d]) / faces[d];
        PetscInt  gridIdx     = (PetscInt)((coords[d] - lower[d]) / gridSpacing + 0.5);
        if (gridIdx % 2 != 0) {
          isObs = PETSC_FALSE;
          break;
        }
      }
      if (isObs) numlocalobs++;
    }
    PetscCall(VecRestoreArrayRead(coordinates, &coordArray));
  }

  /* Create H matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &H));
  PetscCall(MatSetSizes(H, numlocalobs, PETSC_DECIDE, PETSC_DECIDE, numVertices));
  PetscCall(MatSetType(H, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(H, 1, NULL));
  PetscCall(MatMPIAIJSetPreallocation(H, 1, NULL, 0, NULL));
  PetscCall(PetscObjectSetName((PetscObject)H, "H_observation_operator"));

  /* Fill H matrix */
  {
    Vec                coordinates;
    PetscSection       coordSection;
    const PetscScalar *coordArray;
    PetscInt           obsIdx = 0;
    PetscInt           offset;

    PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
    PetscCall(DMGetCoordinateSection(dm, &coordSection));
    PetscCall(VecGetArrayRead(coordinates, &coordArray));

    for (PetscInt v = vStart; v < vEnd; v++) {
      PetscReal coords[3] = {0.0, 0.0, 0.0};
      PetscBool isObs     = PETSC_TRUE;

      PetscCall(PetscSectionGetOffset(coordSection, v, &offset));
      for (PetscInt d = 0; d < dim; d++) coords[d] = PetscRealPart(coordArray[offset + d]);

      /* Check if this vertex is at an observation location (every other grid point) */
      for (PetscInt d = 0; d < dim; d++) {
        PetscReal gridSpacing = (upper[d] - lower[d]) / faces[d];
        PetscInt  gridIdx     = (PetscInt)((coords[d] - lower[d]) / gridSpacing + 0.5);
        if (gridIdx % 2 != 0) {
          isObs = PETSC_FALSE;
          break;
        }
      }

      if (isObs) {
        PetscCall(MatSetValue(H, obsIdx, v - vStart, 1.0, INSERT_VALUES));
        obsIdx++;
      }
    }
    PetscCall(VecRestoreArrayRead(coordinates, &coordArray));
  }
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));

  /* View H */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Observation Operator H:\n"));
  PetscCall(MatView(H, PETSC_VIEWER_STDOUT_WORLD));

  /* Perturb interior vertex coordinates */
  {
    Vec           coordinates;
    PetscSection  coordSection;
    PetscScalar  *coordArray;
    unsigned long seed = 123456789;

    PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
    PetscCall(DMGetCoordinateSection(dm, &coordSection));
    PetscCall(VecGetArray(coordinates, &coordArray));

    for (PetscInt v = vStart; v < vEnd; v++) {
      PetscInt offset;

      PetscCall(PetscSectionGetOffset(coordSection, v, &offset));

      for (PetscInt d = 0; d < dim; d++) {
        PetscReal noise, gridSpacing = (upper[d] - lower[d]) / faces[d];

        seed  = (1103515245 * seed + 12345) % 2147483648;
        noise = (PetscReal)seed / 2147483648.0;
        coordArray[offset + d] += (noise - 0.5) * 0.05 * gridSpacing;
      }
    }
    PetscCall(VecRestoreArray(coordinates, &coordArray));
  }

  /* Call the function */
  PetscCall(DMPlexGetLETKFLocalizationMatrix(dm, numobservations, numlocalobs, H, &Q));
  PetscCall(PetscObjectSetName((PetscObject)Q, "Q_localization"));

  /* View Q */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Localization Matrix Q:\n"));
  PetscCall(MatView(Q, PETSC_VIEWER_STDOUT_WORLD));

  /* Cleanup */
  PetscCall(MatDestroy(&H));
  PetscCall(MatDestroy(&Q));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    requires: kokkos
    suffix: 1
    diff_args: -j
    args: -dm_plex_dim 1 -dm_plex_box_faces 16 -dm_plex_simplex 0

  test:
    requires: kokkos
    suffix: 2
    diff_args: -j
    args: -dm_plex_dim 2 -dm_plex_box_faces 8,8 -dm_plex_simplex 0

  test:
    requires: kokkos
    suffix: 3
    diff_args: -j
    args: -dm_plex_dim 3 -dm_plex_box_faces 5,5,5 -dm_plex_simplex 0

TEST*/
