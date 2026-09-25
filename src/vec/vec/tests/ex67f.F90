!
! Description: Tests VecCUDAGetArray() and friends along with OpenMP target offload.
! It is the Fortran version of ex67.c.
!
#include <petsc/finclude/petscvec.h>
program main
  use petscvec
  implicit none

  PetscInt, parameter :: n = 16
  PetscScalar, parameter :: one = 1.0, two = 2.0, three = 3.0, hundred = 100.0
  PetscScalar, pointer, dimension(:) :: a
  PetscInt :: i
  PetscErrorCode :: ierr
  Vec :: x
  PetscDeviceContext :: dctx

  PetscCallA(PetscInitialize(ierr))
  PetscCallA(PetscDeviceContextGetCurrentContext(dctx, ierr))

  PetscCallA(VecCreate(PETSC_COMM_WORLD, x, ierr))
  PetscCallA(VecSetSizes(x, PETSC_DECIDE, n, ierr))
  PetscCallA(VecSetType(x, 'cuda', ierr))

!  A PETSc write queued on the current-context stream; the first OpenMP kernel overwrites the
!  same buffer, so the two race (write-after-write) unless PETSc's stream is synchronized first.

  PetscCallA(VecSet(x, hundred, ierr))

!  This test interleaves PETSc Vec operations with OpenMP target offload to exercise their
!  stream interaction. PETSc queues work on the current device context's stream, and the
!  VecCUDAGetArray*() calls return without synchronizing; the omp target regions run on a
!  separate, runtime-managed stream with no implicit ordering relative to PETSc's. So whenever
!  an OpenMP kernel touches a buffer a preceding PETSc operation is still writing on PETSc's
!  stream (e.g. VecSet() or VecScale() here), the current device context must be synchronized first,
!  or the two streams race. The omp target regions have no nowait, so they block the host on
!  return, which orders the subsequent PETSc calls after the kernel.

!  Write access: overwrite with x(i) = i.

  PetscCallA(VecCUDAGetArrayWrite(x, a, ierr))
  PetscCallA(PetscDeviceContextSynchronize(dctx, ierr))
  !$omp target teams distribute parallel do is_device_ptr(a)
  do i = 1, n
    a(i) = i
  end do
  !$omp end target teams distribute parallel do
  PetscCallA(VecCUDARestoreArrayWrite(x, a, ierr))

!  A PETSc operation the next OpenMP kernel depends on: double every entry, x(i) = 2*i.

  PetscCallA(VecScale(x, two, ierr))

!  Read-write access: add to what VecScale() produced, so sync PETSc's stream first, x(i) = 2*i + 3.

  PetscCallA(VecCUDAGetArray(x, a, ierr))
  PetscCallA(PetscDeviceContextSynchronize(dctx, ierr))
  !$omp target teams distribute parallel do is_device_ptr(a)
  do i = 1, n
    a(i) = a(i) + three
  end do
  !$omp end target teams distribute parallel do
  PetscCallA(VecCUDARestoreArray(x, a, ierr))

!  A PETSc operation consuming the OpenMP result; the synchronous target above ordered it, x(i) = 2*i + 4.

  PetscCallA(VecShift(x, one, ierr))

!  Read-only access round-trip.

  PetscCallA(VecCUDAGetArrayRead(x, a, ierr))
  PetscCallA(VecCUDARestoreArrayRead(x, a, ierr))

  PetscCallA(PetscObjectSetName(x, 'x', ierr))
  PetscCallA(VecView(x, PETSC_VIEWER_STDOUT_WORLD, ierr))

  PetscCallA(VecDestroy(x, ierr))
  PetscCallA(PetscFinalize(ierr))
end

!
!/*TEST
!
! test:
!    requires: cuda defined(PETSC_HAVE_OPENMP_TARGET_OFFLOAD_FC)
!    output_file: output/ex67_1.out
!
!TEST*/
