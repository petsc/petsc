program  ex1f90
#include <petsc/finclude/petscdmlabel.h>
  use petscdm
  use petscdmlabel
  implicit NONE

  type(tDM)                         :: dm, dmDist
  character(len=PETSC_MAX_PATH_LEN) :: filename
  PetscBool                         :: interpolate = PETSC_FALSE
  PetscBool                         :: flg
  PetscErrorCode                    :: ierr
  PetscInt                          :: izero
  izero = 0

  PetscCallA(PetscInitialize(ierr))
  PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-i",filename,flg,ierr))
  PetscCallA(PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-interpolate",interpolate,flg,ierr))

  PetscCallA(DMPlexCreateFromFile(PETSC_COMM_WORLD,filename,"ex1f90_plex",interpolate,dm,ierr))
  PetscCallA(DMPlexDistribute(dm,izero,PETSC_NULL_SF,dmDist,ierr))
  if (dmDist /= PETSC_NULL_DM) then
    PetscCallA(DMDestroy(dm,ierr))
    dm = dmDist
  end if

  PetscCallA(ViewLabels(dm,PETSC_VIEWER_STDOUT_WORLD,ierr))
  PetscCallA(DMDestroy(dm,ierr))
  PetscCallA(PetscFinalize(ierr))

contains
  subroutine ViewLabels(dm,viewer,ierr)
    type(tDM)                        :: dm
    type(tPetscViewer)               :: viewer
    PetscErrorCode                   :: ierr

    DMLabel                          :: label
    type(tIS)                        :: labelIS
    character(len=PETSC_MAX_PATH_LEN):: labelName,IObuffer
    PetscInt                         :: numLabels,l

    PetscCall(DMGetNumLabels(dm, numLabels, ierr))
    write(IObuffer,*) 'Number of labels: ', numLabels, '\n'
    PetscCall(PetscViewerASCIIPrintf(viewer, IObuffer, ierr))
    do l = 0, numLabels-1
      PetscCall(DMGetLabelName(dm, l, labelName, ierr))
      write(IObuffer,*) 'label ',l,' name: ',trim(labelName),'\n'
      PetscCall(PetscViewerASCIIPrintf(viewer, IObuffer, ierr))

      PetscCall(PetscViewerASCIIPrintf(viewer, "IS of values\n", ierr))
      PetscCall(DMGetLabel(dm, labelName, label, ierr))
      PetscCall(DMLabelGetValueIS(label, labelIS, ierr))
!      PetscCall(PetscViewerASCIIPushTab(viewer,ierr))
      PetscCall(ISView(labelIS, viewer, ierr))
!      PetscCall(PetscViewerASCIIPopTab(viewer,ierr))
      PetscCall(ISDestroy(labelIS, ierr))
      PetscCall(PetscViewerASCIIPrintf(viewer, "\n", ierr))
    end do

    PetscCall(PetscViewerASCIIPrintf(viewer,"\n\nCell Set label IS\n",ierr))
    PetscCall(DMGetLabel(dm, "Cell Sets", label, ierr))
    PetscCall(DMLabelGetValueIS(label, labelIS, ierr))
    PetscCall(ISView(labelIS, viewer, ierr))
    PetscCall(ISDestroy(labelIS, ierr))
  end subroutine viewLabels
end program ex1F90

!/*TEST
!
!  test:
!    suffix: 0
!    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo -interpolate
!    requires: exodusii
!
!TEST*/
