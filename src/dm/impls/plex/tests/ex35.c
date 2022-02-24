static char help[] = "Exhaustive memory tracking for DMPlex.\n\n\n";

#include <petscdmplex.h>

static PetscErrorCode EstimateMemory(DM dm, PetscLogDouble *est)
{
  DMLabel        marker;
  PetscInt       cdim, depth, d, pStart, pEnd, p, Nd[4] = {0, 0, 0, 0}, lsize = 0, rmem = 0, imem = 0;
  PetscInt       coneSecMem = 0, coneMem = 0, supportSecMem = 0, supportMem = 0, labelMem = 0;

  PetscFunctionBeginUser;
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "Memory Estimates\n"));
  CHKERRQ(DMGetCoordinateDim(dm, &cdim));
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  for (d = 0; d <= depth; ++d) {
    PetscInt start, end;

    CHKERRQ(DMPlexGetDepthStratum(dm, d, &start, &end));
    Nd[d] = end - start;
  }
  /* Coordinates: 3 Nv reals + 2*Nv + 2*Nv ints */
  rmem += cdim*Nd[0];
  imem += 2*Nd[0] + 2*Nd[0];
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "  Coordinate mem:  %D %D\n", cdim*Nd[0]*sizeof(PetscReal), 4*Nd[0]*sizeof(PetscInt)));
  /* Depth:       Nc+Nf+Ne+Nv ints */
  for (d = 0; d <= depth; ++d) labelMem += Nd[d];
  /* Cell Type:   Nc+Nf+Ne+Nv ints */
  for (d = 0; d <= depth; ++d) labelMem += Nd[d];
  /* Marker */
  CHKERRQ(DMGetLabel(dm, "marker", &marker));
  if (marker) CHKERRQ(DMLabelGetStratumSize(marker, 1, &lsize));
  labelMem += lsize;
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "  Label mem:       %D\n", labelMem*sizeof(PetscInt)));
  //imem += labelMem;
  /* Cones and Orientations:       4 Nc + 3 Nf + 2 Ne ints + (Nc+Nf+Ne) ints no separate orientation section */
  for (d = 0; d <= depth; ++d) coneSecMem += 2*Nd[d];
  for (p = pStart; p < pEnd; ++p) {
    PetscInt csize;

    CHKERRQ(DMPlexGetConeSize(dm, p, &csize));
    coneMem += csize;
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "  Cone mem:        %D %D (%D)\n", coneMem*sizeof(PetscInt), coneSecMem*sizeof(PetscInt), coneMem*sizeof(PetscInt)));
  imem += 2*coneMem + coneSecMem;
  /* Supports:       4 Nc + 3 Nf + 2 Ne ints + Nc+Nf+Ne ints */
  for (d = 0; d <= depth; ++d) supportSecMem += 2*Nd[d];
  for (p = pStart; p < pEnd; ++p) {
    PetscInt ssize;

    CHKERRQ(DMPlexGetSupportSize(dm, p, &ssize));
    supportMem += ssize;
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "  Support mem:     %D %D\n", supportMem*sizeof(PetscInt), supportSecMem*sizeof(PetscInt)));
  imem += supportMem + supportSecMem;
  *est = ((PetscLogDouble) imem)*sizeof(PetscInt) + ((PetscLogDouble) rmem)*sizeof(PetscReal);
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "  Estimated memory %D\n", (PetscInt) *est));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  PetscBool      trace = PETSC_FALSE, checkMemory = PETSC_TRUE, auxMemory = PETSC_FALSE;
  PetscLogDouble before, after, est = 0, clean, max;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetBool(NULL, NULL, "-trace", &trace, NULL));
  CHKERRQ(PetscOptionsGetBool(NULL, NULL, "-check_memory", &checkMemory, NULL));
  CHKERRQ(PetscOptionsGetBool(NULL, NULL, "-aux_memory", &auxMemory, NULL));
  CHKERRQ(PetscMemorySetGetMaximumUsage());
  CHKERRQ(PetscMallocGetCurrentUsage(&before));
  if (trace) CHKERRQ(PetscMallocTraceSet(NULL, PETSC_TRUE, 5000.));
  CHKERRQ(DMCreate(PETSC_COMM_WORLD, &dm));
  CHKERRQ(DMSetType(dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(dm));
  if (trace) CHKERRQ(PetscMallocTraceSet(NULL, PETSC_FALSE, 5000));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));
  CHKERRQ(PetscMallocGetCurrentUsage(&after));
  CHKERRQ(PetscMemoryGetMaximumUsage(&max));
  CHKERRQ(EstimateMemory(dm, &est));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscMallocGetCurrentUsage(&clean));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Measured Memory\n"));
  if (auxMemory) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "  Initial memory         %D\n  Extra memory for build %D\n  Memory after destroy   %D\n",
                       (PetscInt) before, (PetscInt) (max-after), (PetscInt) clean);CHKERRQ(ierr);
  }
  if (checkMemory) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "  Memory for mesh  %D\n", (PetscInt) (after-before)));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Discrepancy %D\n", (PetscInt) PetscAbsReal(after-before-est)));
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  build:
    requires: !defined(PETSC_USE_64BIT_INDICES) double !complex !defined(PETSCTEST_VALGRIND)

  # Memory checks cannot be included in tests because the allocated memory differs among environments
  testset:
    args: -malloc_requested_size -dm_plex_box_faces 5,5 -check_memory 0
    test:
      suffix: tri
      requires: triangle
      args: -dm_plex_simplex 1 -dm_plex_interpolate 0

    test:
      suffix: tri_interp
      requires: triangle
      args: -dm_plex_simplex 1 -dm_plex_interpolate 1

    test:
      suffix: quad
      args: -dm_plex_simplex 0 -dm_plex_interpolate 0

    test:
      suffix: quad_interp
      args: -dm_plex_simplex 0 -dm_plex_interpolate 1

  # Memory checks cannot be included in tests because the allocated memory differs among environments
  testset:
    args: -malloc_requested_size -dm_plex_dim 3 -dm_plex_box_faces 5,5,5 -check_memory 0
    test:
      suffix: tet
      requires: ctetgen
      args: -dm_plex_simplex 1 -dm_plex_interpolate 0

    test:
      suffix: tet_interp
      requires: ctetgen
      args: -dm_plex_simplex 1 -dm_plex_interpolate 1

    test:
      suffix: hex
      args: -dm_plex_simplex 0 -dm_plex_interpolate 0

    test:
      suffix: hex_interp
      args: -dm_plex_simplex 0 -dm_plex_interpolate 1
TEST*/
