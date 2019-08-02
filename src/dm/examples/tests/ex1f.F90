!
! Test the workaround for a bug in OpenMPI-2.1.1 on Ubuntu 18.04.2
! See https://lists.mcs.anl.gov/pipermail/petsc-dev/2019-July/024803.html
!
! Contributed-by: 	Fabian Jakub  <Fabian.Jakub@physik.uni-muenchen.de>
program main
#include "petsc/finclude/petsc.h"

  use petsc
  implicit none

  PetscInt, parameter :: Ndof=1, stencil_size=1
  PetscInt, parameter :: Nx=3, Ny=3
  PetscErrorCode :: myid, commsize, ierr
  PetscScalar, pointer :: xv1d(:)

  type(tDM) :: da
  type(tVec) :: gVec!, naturalVec


  call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
  call mpi_comm_rank(PETSC_COMM_WORLD, myid, ierr)
  call mpi_comm_size(PETSC_COMM_WORLD, commsize, ierr)

  call DMDACreate2d(PETSC_COMM_WORLD, &
    DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, &
    DMDA_STENCIL_STAR, &
    Nx, Ny, PETSC_DECIDE, PETSC_DECIDE, Ndof, stencil_size, &
    PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, da, ierr)
  call DMSetup(da, ierr)
  call DMSetFromOptions(da, ierr)

  call DMCreateGlobalVector(da, gVec, ierr)
  call VecGetArrayF90(gVec, xv1d, ierr)
  xv1d(:) = real(myid, kind(xv1d))
  !print *,myid, 'xv1d', xv1d, ':', xv1d
  call VecRestoreArrayF90(gVec, xv1d, ierr)

  call PetscObjectViewFromOptions(gVec, PETSC_NULL_VEC, "-show_gVec", ierr)

  call VecDestroy(gVec, ierr)
  call DMDestroy(da, ierr)
  call PetscFinalize(ierr)
end program

!/*TEST
!
!   test:
!      nsize: 9
!      args: -show_gVec
!TEST*/

