static char help[] = "Testing MatCreateSeqAIJKokkosWithKokkosViews() and building Mat on the device.\n\n";

#include <petscvec_kokkos.hpp>
#include <petscdevice.h>
#include <petscmat.h>
#include <petscmat_kokkos.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

using HostMirrorMemorySpace = Kokkos::DualView<PetscScalar *>::host_mirror_space::memory_space;

int main(int argc, char **argv)
{
  Mat             A, B, BT;
  PetscInt        i, j, column, M, N, m, n, m_ab, n_ab;
  PetscInt       *di, *dj, *oi, *oj, nd;
  const PetscInt *garray;
  PetscInt       *garray_h;
  PetscScalar    *oa, *da;
  PetscScalar     value;
  PetscRandom     rctx;
  PetscBool       equal, done;
  Mat             AA, AB;
  PetscMPIInt     size, rank;

  // ~~~~~~~~~~~~~~~~~~~~~
  // This test shows the routines needed to build a kokkos matrix without preallocation
  // on the host
  // ~~~~~~~~~~~~~~~~~~~~~

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size > 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Must run with 2 or more processes");
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* Create a mpiaij matrix for checking */
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, 5, 5, PETSC_DECIDE, PETSC_DECIDE, 0, NULL, 0, NULL, &A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetUp(A));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(PetscRandomSetFromOptions(rctx));

  for (i = 5 * rank; i < 5 * rank + 5; i++) {
    for (j = 0; j < 5 * size; j++) {
      PetscCall(PetscRandomGetValue(rctx, &value));
      column = (PetscInt)(5 * size * PetscRealPart(value));

      // rank 0 has no off-process entries
      if (rank == 0 && (column < i || column >= i)) column = i;

      PetscCall(PetscRandomGetValue(rctx, &value));
      PetscCall(MatSetValues(A, 1, &i, 1, &column, &value, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatGetSize(A, &M, &N));
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatMPIAIJGetSeqAIJ(A, &AA, &AB, &garray));
  PetscCall(MatGetRowIJ(AA, 0, PETSC_FALSE, PETSC_FALSE, &nd, (const PetscInt **)&di, (const PetscInt **)&dj, &done));
  PetscCall(MatSeqAIJGetArray(AA, &da));
  PetscCall(MatGetRowIJ(AB, 0, PETSC_FALSE, PETSC_FALSE, &nd, (const PetscInt **)&oi, (const PetscInt **)&oj, &done));
  PetscCall(MatSeqAIJGetArray(AB, &oa));
  PetscCall(MatGetSize(AB, &m_ab, &n_ab));

  Mat output_mat_local, output_mat_nonlocal;
  // Be careful about scope given the kokkos memory reference counts
  {
    // Local
    Kokkos::View<PetscScalar *> a_local_d;
    Kokkos::View<PetscInt *>    i_local_d;
    Kokkos::View<PetscInt *>    j_local_d;

    // Nonlocal
    Kokkos::View<PetscScalar *> a_nonlocal_d;
    Kokkos::View<PetscInt *>    i_nonlocal_d;
    Kokkos::View<PetscInt *>    j_nonlocal_d;

    // Create device memory
    PetscCallCXX(a_local_d = Kokkos::View<PetscScalar *>("a_local_d", di[5]));
    PetscCallCXX(i_local_d = Kokkos::View<PetscInt *>("i_local_d", m + 1));
    PetscCallCXX(j_local_d = Kokkos::View<PetscInt *>("j_local_d", di[5]));

    // Create non-local device memory
    PetscCallCXX(a_nonlocal_d = Kokkos::View<PetscScalar *>("a_nonlocal_d", oi[5]));
    PetscCallCXX(i_nonlocal_d = Kokkos::View<PetscInt *>("i_nonlocal_d", m + 1));
    PetscCallCXX(j_nonlocal_d = Kokkos::View<PetscInt *>("j_nonlocal_d", oi[5]));

    // ~~~~~~~~~~~~~~~~~~~~~
    // Could fill the aij on the device - we're just going to test
    // by copying in the existing host values
    // ~~~~~~~~~~~~~~~~~~~~~
    Kokkos::View<PetscScalar *, HostMirrorMemorySpace> a_local_h;
    Kokkos::View<PetscInt *, HostMirrorMemorySpace>    i_local_h;
    Kokkos::View<PetscInt *, HostMirrorMemorySpace>    j_local_h;
    Kokkos::View<PetscScalar *, HostMirrorMemorySpace> a_nonlocal_h;
    Kokkos::View<PetscInt *, HostMirrorMemorySpace>    i_nonlocal_h;
    Kokkos::View<PetscInt *, HostMirrorMemorySpace>    j_nonlocal_h;

    PetscCallCXX(a_local_h = Kokkos::View<PetscScalar *, HostMirrorMemorySpace>(da, di[5]));
    PetscCallCXX(i_local_h = Kokkos::View<PetscInt *, HostMirrorMemorySpace>(di, m + 1));
    PetscCallCXX(j_local_h = Kokkos::View<PetscInt *, HostMirrorMemorySpace>(dj, di[5]));
    PetscCallCXX(a_nonlocal_h = Kokkos::View<PetscScalar *, HostMirrorMemorySpace>(oa, oi[5]));
    PetscCallCXX(i_nonlocal_h = Kokkos::View<PetscInt *, HostMirrorMemorySpace>(oi, m + 1));
    PetscCallCXX(j_nonlocal_h = Kokkos::View<PetscInt *, HostMirrorMemorySpace>(oj, oi[5]));

    // Haven't specified an exec space so these will all be synchronous
    // and finish without a need to call fence after
    PetscCallCXX(Kokkos::deep_copy(a_local_d, a_local_h));
    PetscCallCXX(Kokkos::deep_copy(i_local_d, i_local_h));
    PetscCallCXX(Kokkos::deep_copy(j_local_d, j_local_h));
    PetscCallCXX(Kokkos::deep_copy(a_nonlocal_d, a_nonlocal_h));
    PetscCallCXX(Kokkos::deep_copy(i_nonlocal_d, i_nonlocal_h));
    PetscCallCXX(Kokkos::deep_copy(j_nonlocal_d, j_nonlocal_h));

    // The garray passed in has to be on the host, but it can be created
    // on device and copied to the host
    // We're just going to copy the existing host values here
    PetscCall(PetscMalloc1(n_ab, &garray_h));
    for (int i = 0; i < n_ab; i++) garray_h[i] = garray[i];

    // ~~~~~~~~~~~~~~~~~~~~~

    // ~~~~~~~~~~~~~~~~~
    // Test MatCreateSeqAIJKokkosWithKokkosViews
    // ~~~~~~~~~~~~~~~~~

    // We can create our local diagonal block matrix directly on the device
    PetscCall(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, m, n, i_local_d, j_local_d, a_local_d, &output_mat_local));

    // We can create our nonlocal diagonal block matrix directly on the device
    PetscCall(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, m, n_ab, i_nonlocal_d, j_nonlocal_d, a_nonlocal_d, &output_mat_nonlocal));

    // Build our MPI matrix
    // If we provide garray and output_mat_nonlocal with local indices and the compactified size
    // almost nothing happens on the host
    PetscCall(MatCreateMPIAIJWithSeqAIJ(PETSC_COMM_WORLD, M, N, output_mat_local, output_mat_nonlocal, garray_h, &B));

    PetscCall(MatEqual(A, B, &equal));
    PetscCall(MatRestoreRowIJ(AA, 0, PETSC_FALSE, PETSC_FALSE, &nd, (const PetscInt **)&di, (const PetscInt **)&dj, &done));
    PetscCall(MatSeqAIJRestoreArray(AA, &da));
    PetscCall(MatRestoreRowIJ(AB, 0, PETSC_FALSE, PETSC_FALSE, &nd, (const PetscInt **)&oi, (const PetscInt **)&oj, &done));
    PetscCall(MatSeqAIJRestoreArray(AB, &oa));
    PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Likely a bug in MatCreateSeqAIJKokkosWithKokkosViews()");
  }

  PetscCall(MatTranspose(B, MAT_INITIAL_MATRIX, &BT));

  /* Free spaces */
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&BT));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: kokkos_kernels

  test:
    nsize: 2
    args: -mat_type aijkokkos
    requires: kokkos_kernels
    output_file: output/empty.out

TEST*/
