! setting up 3-D DMPlex using DMPlexCreateFromDAG(), DMPlexInterpolate() and
! DMPlexComputeCellGeometryFVM()
! Contributed by Adrian Croucher <a.croucher@auckland.ac.nz>
      program main
      use petscsys
      use petscdmplex
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscdmplex.h>
      implicit none
      DM :: dm, dmi
      PetscFV :: fvm
      PetscInt, parameter :: dim = 3

      PetscInt :: depth = 1
      PetscErrorCode :: ierr
      PetscInt, dimension(2) :: numPoints
      PetscInt, dimension(14) :: coneSize
      PetscInt, dimension(16) :: cones
      PetscInt, dimension(16) :: coneOrientations
      PetscScalar, dimension(36) :: vertexCoords
      PetscReal ::  vol = 0.
      PetscReal, target, dimension(dim)  :: centroid
      PetscReal, target, dimension(dim)  :: normal
      PetscReal, pointer :: pcentroid(:)
      PetscReal, pointer :: pnormal(:)

      PetscReal, target, dimension(dim)  :: v0
      PetscReal, target, dimension(dim*dim)  :: J
      PetscReal, target, dimension(dim*dim)  :: invJ
      PetscReal, pointer :: pv0(:)
      PetscReal, pointer :: pJ(:)
      PetscReal, pointer :: pinvJ(:)
      PetscReal :: detJ

      PetscInt :: i

      numPoints = [12, 2]
      coneSize  = [8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      cones = [2,5,4,3,6,7,8,9,  3,4,11,10,7,12,13,8]
      coneOrientations = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0]
      vertexCoords = [                                                  &
     &  -0.5,0.0,0.0, 0.0,0.0,0.0, 0.0,1.0,0.0, -0.5,1.0,0.0,           &
     &  -0.5,0.0,1.0, 0.0,0.0,1.0, 0.0,1.0,1.0, -0.5,1.0,1.0,           &
     &   0.5,0.0,0.0, 0.5,1.0,0.0, 0.5,0.0,1.0,  0.5,1.0,1.0 ]
      pcentroid => centroid
      pnormal => normal

      pv0 => v0
      pJ => J
      pinvJ => invJ

      PetscCallA(PetscInitialize(ierr))
      PetscCallA(DMPlexCreate(PETSC_COMM_WORLD, dm, ierr))
      PetscCallA(PetscObjectSetName(dm, 'testplex', ierr))
      PetscCallA(DMSetDimension(dm, dim, ierr))

      PetscCallA(DMPlexCreateFromDAG(dm, depth, numPoints, coneSize, cones,coneOrientations, vertexCoords, ierr))

      PetscCallA(DMPlexInterpolate(dm, dmi, ierr))
      PetscCallA(DMPlexCopyCoordinates(dm, dmi, ierr))
      PetscCallA(DMDestroy(dm, ierr))
      dm = dmi

      PetscCallA(DMView(dm, PETSC_VIEWER_STDOUT_WORLD, ierr))

      do i = 0, 1
        PetscCallA(DMPlexComputeCellGeometryFVM(dm, i, vol, pcentroid, pnormal, ierr))
        write(*, '(a, i2, a, f8.4, a, 3(f8.4, 1x))') 'cell: ', i, ' volume: ', vol, ' centroid: ',pcentroid(1), pcentroid(2), pcentroid(3)
        PetscCallA(DMPlexComputeCellGeometryAffineFEM(dm, i, pv0, pJ, pinvJ,detJ, ierr))
      end do

      PetscCallA(PetscFVCreate(PETSC_COMM_WORLD, fvm, ierr))
      PetscCallA(PetscFVSetUp(fvm, ierr))
      PetscCallA(PetscFVDestroy(fvm, ierr))

      PetscCallA(DMDestroy(dm, ierr))
      PetscCallA(PetscFinalize(ierr))
      end program main

!/*TEST
!
!   test:
!      suffix: 0
!
!TEST*/
