static char help[] = "Exhaustive memory tracking for DMPlex.\n\n\n";

#include <petscdmplex.h>

static PetscErrorCode EstimateMemory(DM dm, PetscLogDouble *est)
{
  DMLabel        marker;
  PetscInt       cdim, depth, d, pStart, pEnd, p, Nd[4] = {0, 0, 0, 0}, lsize = 0, rmem = 0, imem = 0;
  PetscInt       coneSecMem = 0, coneMem = 0, supportSecMem = 0, supportMem = 0, labelMem = 0;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscPrintf(PETSC_COMM_SELF, "Memory Estimates\n");CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {
    PetscInt start, end;

    ierr = DMPlexGetDepthStratum(dm, d, &start, &end);CHKERRQ(ierr);
    Nd[d] = end - start;
  }
  /* Coordinates: 3 Nv reals + 2*Nv + 2*Nv ints */
  rmem += cdim*Nd[0];
  imem += 2*Nd[0] + 2*Nd[0];
  ierr = PetscPrintf(PETSC_COMM_SELF, "  Coordinate mem:  %D %D\n", cdim*Nd[0]*sizeof(PetscReal), 4*Nd[0]*sizeof(PetscInt));CHKERRQ(ierr);
  /* Depth:       Nc+Nf+Ne+Nv ints */
  for (d = 0; d <= depth; ++d) labelMem += Nd[d];
  /* Cell Type:   Nc+Nf+Ne+Nv ints */
  for (d = 0; d <= depth; ++d) labelMem += Nd[d];
  /* Marker */
  ierr = DMGetLabel(dm, "marker", &marker);CHKERRQ(ierr);
  if (marker) {ierr = DMLabelGetStratumSize(marker, 1, &lsize);CHKERRQ(ierr);}
  labelMem += lsize;
  ierr = PetscPrintf(PETSC_COMM_SELF, "  Label mem:       %D\n", labelMem*sizeof(PetscInt));CHKERRQ(ierr);
  //imem += labelMem;
  /* Cones and Orientations:       4 Nc + 3 Nf + 2 Ne ints + (Nc+Nf+Ne) ints no separate orientation section */
  for (d = 0; d <= depth; ++d) coneSecMem += 2*Nd[d];
  for (p = pStart; p < pEnd; ++p) {
    PetscInt csize;

    ierr = DMPlexGetConeSize(dm, p, &csize);CHKERRQ(ierr);
    coneMem += csize;
  }
  ierr = PetscPrintf(PETSC_COMM_SELF, "  Cone mem:        %D %D (%D)\n", coneMem*sizeof(PetscInt), coneSecMem*sizeof(PetscInt), coneMem*sizeof(PetscInt));CHKERRQ(ierr);
  imem += 2*coneMem + coneSecMem;
  /* Supports:       4 Nc + 3 Nf + 2 Ne ints + Nc+Nf+Ne ints */
  for (d = 0; d <= depth; ++d) supportSecMem += 2*Nd[d];
  for (p = pStart; p < pEnd; ++p) {
    PetscInt ssize;

    ierr = DMPlexGetSupportSize(dm, p, &ssize);CHKERRQ(ierr);
    supportMem += ssize;
  }
  ierr = PetscPrintf(PETSC_COMM_SELF, "  Support mem:     %D %D\n", supportMem*sizeof(PetscInt), supportSecMem*sizeof(PetscInt));CHKERRQ(ierr);
  imem += supportMem + supportSecMem;
  *est = ((PetscLogDouble) imem)*sizeof(PetscInt) + ((PetscLogDouble) rmem)*sizeof(PetscReal);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "  Estimated memory %D\n", (PetscInt) *est);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  PetscBool      trace = PETSC_FALSE, checkMemory = PETSC_TRUE, auxMemory = PETSC_FALSE;
  PetscLogDouble before, after, est = 0, clean, max;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = PetscOptionsGetBool(NULL, NULL, "-trace", &trace, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-check_memory", &checkMemory, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-aux_memory", &auxMemory, NULL);CHKERRQ(ierr);
  ierr = PetscMemorySetGetMaximumUsage();CHKERRQ(ierr);
  ierr = PetscMallocGetCurrentUsage(&before);CHKERRQ(ierr);
  if (trace) {ierr = PetscMallocTraceSet(NULL, PETSC_TRUE, 5000.);CHKERRQ(ierr);}
  ierr = DMCreate(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  if (trace) {ierr = PetscMallocTraceSet(NULL, PETSC_FALSE, 5000);CHKERRQ(ierr);}
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = PetscMallocGetCurrentUsage(&after);CHKERRQ(ierr);
  ierr = PetscMemoryGetMaximumUsage(&max);CHKERRQ(ierr);
  ierr = EstimateMemory(dm, &est);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscMallocGetCurrentUsage(&clean);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Measured Memory\n");CHKERRQ(ierr);
  if (auxMemory) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "  Initial memory         %D\n  Extra memory for build %D\n  Memory after destroy   %D\n",
                       (PetscInt) before, (PetscInt) (max-after), (PetscInt) clean);CHKERRQ(ierr);
  }
  if (checkMemory) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "  Memory for mesh  %D\n", (PetscInt) (after-before));CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Discrepancy %D\n", (PetscInt) PetscAbsReal(after-before-est));CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  build:
    requires: !define(PETSC_USE_64BIT_INDICES) double !complex !define(PETSCTEST_VALGRIND)

  # Mempry checks cannot be included in tests because the allocated memory differs among environments
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
