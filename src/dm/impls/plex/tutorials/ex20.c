static char help[] = "Test DMPlexGetLETKFLocalizationMatrix.\n\n";

#include <petscdmplex.h>
#include <petscdmda.h>

int main(int argc, char **argv)
{
  DM        dm;
  Mat       H, Q             = NULL;
  PetscInt  nvertexobs, ndof = 1, n_state_global;
  PetscInt  dim       = 1, n, vStart, vEnd;
  PetscInt  faces[3]  = {1, 1, 1};
  PetscReal lower[3]  = {0.0, 0.0, 0.0};
  PetscReal upper[3]  = {1.0, 1.0, 1.0};
  Vec       Vecxyz[3] = {NULL, NULL, NULL};
  PetscBool isda, isplex, print = PETSC_FALSE;
  char      type[256] = DMPLEX;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  /* Get dimension and from options. We need the data here and Plex does not have access functions */
  PetscOptionsBegin(PETSC_COMM_WORLD, "", "DMField Tutorial Options", "DM");
  PetscCall(PetscOptionsFList("-dm_type", "DM implementation on which to define field", "ex20.c", DMList, type, type, 256, NULL));
  PetscCall(PetscStrncmp(type, DMPLEX, 256, &isplex));
  PetscCall(PetscStrncmp(type, DMDA, 256, &isda));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-ex20_print", &print, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dm_plex_dim", &dim, NULL));
  PetscCheck(dim <= 3 && dim >= 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "dm_plex_dim = %" PetscInt_FMT, dim);
  n = 3;
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces", faces, &n, NULL));
  PetscCheck(n == 0 || n == dim, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "dm_plex_box_faces dimension %" PetscInt_FMT " does not match requested dimension %" PetscInt_FMT, n, dim);
  n = 3;
  PetscCall(PetscOptionsGetRealArray(NULL, NULL, "-dm_plex_box_lower", lower, &n, NULL));
  PetscCheck(n == 0 || n == dim, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "dm_plex_box_lower dimension %" PetscInt_FMT " does not match requested dimension %" PetscInt_FMT, n, dim);
  n = 3;
  PetscCall(PetscOptionsGetRealArray(NULL, NULL, "-dm_plex_box_upper", upper, &n, NULL));
  PetscCheck(n == 0 || n == dim, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "dm_plex_box_upper dimension %" PetscInt_FMT " does not match requested dimension %" PetscInt_FMT, n, dim);
  PetscOptionsEnd();

  if (isplex) {
    PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
    PetscCall(DMSetType(dm, DMPLEX));
    PetscCall(DMSetFromOptions(dm));
    {
      PetscSection section;
      PetscInt     pStart, pEnd;

      PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
      PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
      PetscCall(PetscSectionCreate(PETSC_COMM_WORLD, &section));
      PetscCall(PetscSectionSetNumFields(section, 1));
      PetscCall(PetscSectionSetChart(section, pStart, pEnd));
      for (PetscInt v = vStart; v < vEnd; ++v) PetscCall(PetscSectionSetDof(section, v, 1));
      PetscCall(PetscSectionSetUp(section));
      PetscCall(DMSetLocalSection(dm, section));
      PetscCall(PetscSectionDestroy(&section));

      for (PetscInt d = 0; d < dim; d++) {
        Vec                loc_vec;
        Vec                coordinates;
        PetscSection       coordSection, s;
        const PetscScalar *coordArray;
        PetscScalar       *xArray;

        PetscCall(DMCreateGlobalVector(dm, &Vecxyz[d]));
        PetscCall(PetscObjectSetName((PetscObject)Vecxyz[d], d == 0 ? "x_coordinate" : (d == 1 ? "y_coordinate" : "z_coordinate")));
        PetscCall(DMGetLocalVector(dm, &loc_vec));

        PetscCall(DMGetLocalSection(dm, &s));
        PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
        PetscCall(DMGetCoordinateSection(dm, &coordSection));
        PetscCall(VecGetArrayRead(coordinates, &coordArray));
        PetscCall(VecGetArray(loc_vec, &xArray));

        for (PetscInt v = vStart; v < vEnd; v++) {
          PetscInt cOff, sOff;

          PetscCall(PetscSectionGetOffset(coordSection, v, &cOff));
          PetscCall(PetscSectionGetOffset(s, v, &sOff));
          xArray[sOff] = coordArray[cOff + d];
        }
        PetscCall(VecRestoreArrayRead(coordinates, &coordArray));
        PetscCall(VecRestoreArray(loc_vec, &xArray));

        PetscCall(DMLocalToGlobal(dm, loc_vec, INSERT_VALUES, Vecxyz[d]));
        PetscCall(DMRestoreLocalVector(dm, &loc_vec));
        PetscCall(VecGetSize(Vecxyz[d], &n_state_global));
      }
    }

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Created DMPlex in %" PetscInt_FMT "D with faces (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT "), global vector size %" PetscInt_FMT "\n", dim, faces[0], faces[1], faces[2], n_state_global));
  } else if (isda) {
    switch (dim) {
    case 1:
      PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, faces[0], ndof, 1, NULL, &dm));
      break;
    case 2:
      PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, faces[0], faces[1] + 1, PETSC_DETERMINE, PETSC_DETERMINE, ndof, 1, NULL, NULL, &dm));
      break;
    default:
      PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, faces[0], faces[1] + 1, faces[2] + 1, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE, ndof, 1, NULL, NULL, NULL, &dm));
      break;
    }
    PetscCall(DMSetUp(dm));
    PetscCall(DMDASetUniformCoordinates(dm, lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]));
    {
      Vec coord;
      PetscCall(DMGetCoordinates(dm, &coord));
      for (PetscInt d = 0; d < dim; d++) {
        PetscCall(DMCreateGlobalVector(dm, &Vecxyz[d]));
        PetscCall(VecSetFromOptions(Vecxyz[d]));
        PetscCall(PetscObjectSetName((PetscObject)Vecxyz[d], d == 0 ? "x_coordinate" : (d == 1 ? "y_coordinate" : "z_coordinate")));
        PetscCall(VecStrideGather(coord, d, Vecxyz[d], INSERT_VALUES));
        PetscCall(VecGetSize(Vecxyz[d], &n_state_global));
      }
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Created DMDA of type %s in %" PetscInt_FMT "D with faces (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT "), global vector size %" PetscInt_FMT "\n", type, dim, faces[0], faces[1], faces[2], n_state_global));
  } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "This test does not run for DM type %s", type);
  PetscCall(DMViewFromOptions(dm, NULL, "-ex20_dm_view")); // PetscSleep(10);

  /* Set number of local observations to use: 3^dim */
  nvertexobs = 1;
  for (PetscInt d = 0; d < dim && d < 2; d++) nvertexobs *= 3;
  PetscCheck(nvertexobs > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "nvertexobs %" PetscInt_FMT " must be > 0 locally for now", nvertexobs);

  /* Count observations (every other vertex in each dimension) */
  PetscInt   nobs_local = 0;
  PetscBool *isObs;
  PetscInt   nloc;

  PetscCall(VecGetLocalSize(Vecxyz[0], &nloc));
  PetscCall(PetscMalloc1(nloc, &isObs));
  {
    const PetscScalar *coords[3];
    PetscReal          gridSpacing[3];
    for (PetscInt d = 0; d < dim; d++) PetscCall(VecGetArrayRead(Vecxyz[d], &coords[d]));
    for (PetscInt d = 0; d < dim; d++) gridSpacing[d] = (upper[d] - lower[d]) / faces[d];

    for (PetscInt v = 0; v < nloc; v++) {
      PetscReal c[3] = {0.0, 0.0, 0.0};

      isObs[v] = PETSC_TRUE;
      for (PetscInt d = 0; d < dim; d++) c[d] = PetscRealPart(coords[d][v]);
      /* Check if this vertex is at an observation location (every other grid point) */
      for (PetscInt d = 0; d < dim; d++) {
        PetscReal relCoord = c[d] - lower[d];
        PetscInt  gridIdx  = (PetscInt)PetscFloorReal(relCoord / gridSpacing[d] + 0.5);
        PetscCheck(PetscAbsReal(relCoord - gridIdx * gridSpacing[d]) < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "Error vertex v %" PetscInt_FMT " (dim %" PetscInt_FMT "): %g not on grid (h= %g, distance to grid %g)", v, d, (double)c[d], (double)gridSpacing[d], (double)PetscAbsReal(relCoord - gridIdx * gridSpacing[d]));
        if (gridIdx % 2 != 0) {
          isObs[v] = PETSC_FALSE;
          break;
        }
      }
      if (isObs[v]) nobs_local++;
    }
    for (PetscInt d = 0; d < dim; d++) PetscCall(VecRestoreArrayRead(Vecxyz[d], &coords[d]));
  }

  /* Create H matrix n_obs X n_state */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &H));
  PetscCall(MatSetSizes(H, nobs_local, PETSC_DECIDE, PETSC_DECIDE, n_state_global)); //
  PetscCall(MatSetBlockSizes(H, 1, ndof));
  PetscCall(MatSetType(H, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(H, 1, NULL));
  PetscCall(MatMPIAIJSetPreallocation(H, 1, NULL, 1, NULL)); // assumes boolean observation operator, could use interpolation
  PetscCall(PetscObjectSetName((PetscObject)H, "H_observation_operator"));
  PetscCall(MatSetFromOptions(H));

  /* Fill H matrix */
  PetscInt globalRowIdx, globalColIdx, obsIdx = 0;
  PetscCall(VecGetOwnershipRange(Vecxyz[0], &globalColIdx, NULL));
  PetscCall(MatGetOwnershipRange(H, &globalRowIdx, NULL));
  for (PetscInt v = 0; v < nloc; v++) {
    if (isObs[v]) {
      PetscInt grow = globalRowIdx + obsIdx++, gcol = globalColIdx + v;
      PetscCall(MatSetValue(H, grow, gcol, 1.0, INSERT_VALUES));
    }
  }
  PetscCall(PetscFree(isObs));
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));

  /* View H */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Observation Operator H:\n"));
  if (print) PetscCall(MatView(H, PETSC_VIEWER_STDOUT_WORLD));

  /* Perturb interior vertex coordinates */
  {
    PetscScalar  *coords[3];
    PetscInt      nloc;
    unsigned long seed = 123456789;

    PetscCall(VecGetLocalSize(Vecxyz[0], &nloc));
    for (PetscInt d = 0; d < dim; d++) PetscCall(VecGetArray(Vecxyz[d], &coords[d]));

    for (PetscInt v = 0; v < nloc; v++) {
      for (PetscInt d = 0; d < dim; d++) {
        PetscReal noise, gridSpacing = (upper[d] - lower[d]) / faces[d];

        seed  = (1103515245 * seed + 12345) % 2147483648;
        noise = (PetscReal)seed / 2147483648.0;
        coords[d][v] += (noise - 0.5) * 0.001 * gridSpacing;
      }
    }
    for (PetscInt d = 0; d < dim; d++) PetscCall(VecRestoreArray(Vecxyz[d], &coords[d]));
  }

  /* Call the function */
  PetscCall(DMPlexGetLETKFLocalizationMatrix(nvertexobs, nobs_local, ndof, Vecxyz, H, &Q));
  PetscCall(PetscObjectSetName((PetscObject)Q, "Q_localization"));

  // View Q
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Localization Matrix Q:\n"));
  if (print) PetscCall(MatView(Q, PETSC_VIEWER_STDOUT_WORLD));

  /* Cleanup */
  for (PetscInt d = 0; d < dim; d++) PetscCall(VecDestroy(&Vecxyz[d]));
  PetscCall(MatDestroy(&H));
  PetscCall(MatDestroy(&Q));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    requires: kokkos_kernels
    suffix: 1
    diff_args: -j
    args: -dm_plex_dim 1 -dm_plex_box_faces 16 -dm_plex_simplex 0 -dm_plex_box_bd periodic -dm_plex_box_upper 5 -ex20_print -ex20_dm_view -ex20_dm_view -mat_type aijkokkos -dm_vec_type kokkos

  test:
    requires: kokkos_kernels
    suffix: 2
    diff_args: -j
    args: -dm_plex_dim 2 -dm_plex_box_faces 7,7 -dm_plex_simplex 0 -dm_plex_box_bd periodic,none -dm_plex_box_upper 5,5 -ex20_print -ex20_dm_view -ex20_dm_view -mat_type aijkokkos -dm_vec_type kokkos

  test:
    requires: kokkos_kernels
    suffix: da2
    diff_args: -j
    args: -dm_type da -dm_plex_dim 2 -dm_plex_box_faces 7,7 -dm_plex_box_upper 5,5 -ex20_print -ex20_dm_view -ex20_dm_view -mat_type aijkokkos -vec_type kokkos

  test:
    requires: kokkos_kernels
    suffix: 3
    diff_args: -j
    args: -dm_plex_dim 3 -dm_plex_box_faces 5,5,5 -dm_plex_simplex 0 -dm_plex_box_bd periodic,none,none -dm_plex_box_upper 5,5,5 -ex20_print -ex20_dm_view -mat_type aijkokkos -dm_vec_type kokkos

TEST*/
