static const char help[] = "Test DM section-SF transfers with Kokkos vectors on a different stream.\n";

#include <petscdmshell.h>
#include <petscsection.h>
#include <petscsf.h>
#include <petscdevice.h>

int main(int argc, char **argv)
{
  DM                 dm;
  PetscSection       section;
  PetscSF            sf;
  PetscSFNode        remote[2];
  Vec                g, l;
  PetscDeviceContext saved, other, current;
  const PetscScalar *values;
  PetscInt           local[] = {0, 2};
  PetscMPIInt        rank, size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscDeviceContextGetCurrentContext(&saved));
  PetscCall(PetscDeviceContextDuplicate(saved, &other));
  PetscCall(PetscDeviceContextSetStreamType(other, PETSC_STREAM_NONBLOCKING));
  PetscCall(PetscDeviceContextSetUp(other));

  // Initialize Kokkos on a context that stays alive until PetscFinalize().
  PetscCall(VecCreate(PETSC_COMM_WORLD, &g));
  PetscCall(VecSetSizes(g, 4, PETSC_DECIDE));
  PetscCall(VecSetType(g, VECMPIKOKKOS));
  PetscCall(VecCreate(PETSC_COMM_SELF, &l));
  PetscCall(VecSetSizes(l, 4, 4));
  PetscCall(VecSetType(l, VECSEQKOKKOS));
  PetscCall(DMShellCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(PetscSectionCreate(PETSC_COMM_WORLD, &section));
  PetscCall(PetscSectionSetChart(section, 0, 4));
  for (PetscInt i = 0; i < 4; ++i) PetscCall(PetscSectionSetDof(section, i, 1));
  PetscCall(PetscSectionSetUp(section));
  PetscCall(DMSetLocalSection(dm, section));
  PetscCall(PetscSectionDestroy(&section));
  remote[0].rank  = rank;
  remote[0].index = 3;
  remote[1].rank  = (rank + 1) % size;
  remote[1].index = 1;
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD, &sf));
  PetscCall(PetscSFSetGraph(sf, 4, 2, local, PETSC_COPY_VALUES, remote, PETSC_COPY_VALUES));
  PetscCall(PetscSFSetFromOptions(sf));
  PetscCall(DMSetSectionSF(dm, sf));
  PetscCall(PetscSFDestroy(&sf));

  // Warm the SF links on Kokkos' stream before switching to a nonblocking stream.
  for (PetscInt pass = 0; pass < 2; ++pass) {
    current = pass ? other : saved;
    PetscCall(PetscDeviceContextSetCurrentContext(current));
    for (PetscInt reverse = 0; reverse < 2; ++reverse) {
      Vec input = reverse ? l : g, output = reverse ? g : l;

      PetscCall(VecSet(input, 1));
      PetscCall(VecSet(output, 0));
      PetscCall(PetscDeviceContextSynchronize(saved));
      if (pass) PetscCall(PetscDeviceContextDelay(saved, 0.05));
      PetscCall(VecScale(input, 2));
      if (reverse) PetscCall(DMLocalToGlobalBegin(dm, l, ADD_VALUES, g));
      else PetscCall(DMGlobalToLocalBegin(dm, g, INSERT_VALUES, l));
      if (pass) PetscCall(PetscDeviceContextDelay(current, 0.05));
      if (reverse) PetscCall(DMLocalToGlobalEnd(dm, l, ADD_VALUES, g));
      else PetscCall(DMGlobalToLocalEnd(dm, g, INSERT_VALUES, l));
      PetscCall(VecScale(output, 3));
      PetscCall(PetscDeviceContextSynchronize(saved));
      PetscCall(PetscDeviceContextSynchronize(current));
      PetscCall(VecGetArrayRead(output, &values));
      for (PetscInt i = 0; i < 4; ++i) {
        PetscScalar expected = (i == (reverse ? 3 : 0) || i == (reverse ? 1 : 2)) ? 6 : 0;

        PetscCheck(values[i] == expected, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Pass %" PetscInt_FMT ", reverse %" PetscInt_FMT ": incorrect DM transfer at local entry %" PetscInt_FMT, pass, reverse, i);
      }
      PetscCall(VecRestoreArrayRead(output, &values));
    }
  }
  PetscCall(DMDestroy(&dm));
  PetscCall(VecDestroy(&g));
  PetscCall(VecDestroy(&l));
  PetscCall(PetscDeviceContextSetCurrentContext(saved));
  PetscCall(PetscDeviceContextDestroy(&other));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: kokkos_kernels

  testset:
    nsize: 2
    output_file: output/empty.out
    test:
      suffix: host
      requires: !cuda !hip !sycl
    test:
      suffix: cuda
      requires: cuda
      args: -sf_backend cuda -use_gpu_aware_mpi 0
    test:
      suffix: hip
      requires: hip
      args: -sf_backend hip -use_gpu_aware_mpi 0

TEST*/
