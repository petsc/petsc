static char help[] = "Exhaustive memory tracking for DMPlex.\n\n\n";

#include <petscdmplex.h>

static PetscErrorCode EstimateMemory(DM dm, PetscLogDouble *est)
{
  DMLabel        marker;
  PetscInt       cdim, depth, d, pStart, pEnd, p, Nd[4] = {0, 0, 0, 0}, lsize = 0, rmem = 0, imem = 0;
  PetscInt       coneSecMem = 0, coneMem = 0, supportSecMem = 0, supportMem = 0, labelMem = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Memory Estimates\n"));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  for (d = 0; d <= depth; ++d) {
    PetscInt start, end;

    PetscCall(DMPlexGetDepthStratum(dm, d, &start, &end));
    Nd[d] = end - start;
  }
  /* Coordinates: 3 Nv reals + 2*Nv + 2*Nv ints */
  rmem += cdim*Nd[0];
  imem += 2*Nd[0] + 2*Nd[0];
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Coordinate mem:  %" PetscInt_FMT " %" PetscInt_FMT "\n", (PetscInt)(cdim*Nd[0]*sizeof(PetscReal)), (PetscInt)(4*Nd[0]*sizeof(PetscInt))));
  /* Depth:       Nc+Nf+Ne+Nv ints */
  for (d = 0; d <= depth; ++d) labelMem += Nd[d];
  /* Cell Type:   Nc+Nf+Ne+Nv ints */
  for (d = 0; d <= depth; ++d) labelMem += Nd[d];
  /* Marker */
  PetscCall(DMGetLabel(dm, "marker", &marker));
  if (marker) PetscCall(DMLabelGetStratumSize(marker, 1, &lsize));
  labelMem += lsize;
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Label mem:       %" PetscInt_FMT "\n", (PetscInt)(labelMem*sizeof(PetscInt))));
  //imem += labelMem;
  /* Cones and Orientations:       4 Nc + 3 Nf + 2 Ne ints + (Nc+Nf+Ne) ints no separate orientation section */
  for (d = 0; d <= depth; ++d) coneSecMem += 2*Nd[d];
  for (p = pStart; p < pEnd; ++p) {
    PetscInt csize;

    PetscCall(DMPlexGetConeSize(dm, p, &csize));
    coneMem += csize;
  }
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Cone mem:        %" PetscInt_FMT " %" PetscInt_FMT " (%" PetscInt_FMT ")\n", (PetscInt)(coneMem*sizeof(PetscInt)), (PetscInt)(coneSecMem*sizeof(PetscInt)), (PetscInt)(coneMem*sizeof(PetscInt))));
  imem += 2*coneMem + coneSecMem;
  /* Supports:       4 Nc + 3 Nf + 2 Ne ints + Nc+Nf+Ne ints */
  for (d = 0; d <= depth; ++d) supportSecMem += 2*Nd[d];
  for (p = pStart; p < pEnd; ++p) {
    PetscInt ssize;

    PetscCall(DMPlexGetSupportSize(dm, p, &ssize));
    supportMem += ssize;
  }
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Support mem:     %" PetscInt_FMT " %" PetscInt_FMT "\n", (PetscInt)(supportMem*sizeof(PetscInt)), (PetscInt)(supportSecMem*sizeof(PetscInt))));
  imem += supportMem + supportSecMem;
  *est = ((PetscLogDouble) imem)*sizeof(PetscInt) + ((PetscLogDouble) rmem)*sizeof(PetscReal);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Estimated memory %" PetscInt_FMT "\n", (PetscInt) *est));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  PetscBool      trace = PETSC_FALSE, checkMemory = PETSC_TRUE, auxMemory = PETSC_FALSE;
  PetscLogDouble before, after, est = 0, clean, max;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-trace", &trace, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-check_memory", &checkMemory, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-aux_memory", &auxMemory, NULL));
  PetscCall(PetscMemorySetGetMaximumUsage());
  PetscCall(PetscMallocGetCurrentUsage(&before));
  if (trace) PetscCall(PetscMallocTraceSet(NULL, PETSC_TRUE, 5000.));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  if (trace) PetscCall(PetscMallocTraceSet(NULL, PETSC_FALSE, 5000));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(PetscMallocGetCurrentUsage(&after));
  PetscCall(PetscMemoryGetMaximumUsage(&max));
  PetscCall(EstimateMemory(dm, &est));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscMallocGetCurrentUsage(&clean));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Measured Memory\n"));
  if (auxMemory) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Initial memory         %" PetscInt_FMT "\n  Extra memory for build %" PetscInt_FMT "\n  Memory after destroy   %" PetscInt_FMT "\n",
                          (PetscInt) before, (PetscInt) (max-after), (PetscInt) clean));
  }
  if (checkMemory) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Memory for mesh  %" PetscInt_FMT "\n", (PetscInt) (after-before)));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Discrepancy %" PetscInt_FMT "\n", (PetscInt) PetscAbsReal(after-before-est)));
  }
  PetscCall(PetscFinalize());
  return 0;
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
