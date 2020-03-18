program  ex1f90
#include <petsc/finclude/petscdmlabel.h>
  use petscdm
  use petscdmlabel
  implicit NONE

  type(tDM)           :: dm, dmDist
  character(len=2048) :: filename
  integer,parameter   :: len=2048
  PetscBool           :: interpolate = PETSC_FALSE
  PetscBool           :: flg
  PetscErrorCode      :: ierr
  PetscInt            :: izero
  izero = 0

  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
    if (ierr .ne. 0) then
    print*,'Unable to initialize PETSc'
    stop
  endif
  call PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-i",filename,flg,ierr);CHKERRA(ierr)
  call PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-interpolate",interpolate,flg,ierr);CHKERRA(ierr)

  call DMPlexCreateFromFile(PETSC_COMM_WORLD,filename,interpolate,dm,ierr);CHKERRA(ierr);
  call DMPlexDistribute(dm,izero,PETSC_NULL_SF,dmDist,ierr);CHKERRA(ierr)
  if (dmDist /= PETSC_NULL_DM) then
    call DMDestroy(dm,ierr);CHKERRA(ierr)
    dm = dmDist
  end if

  call ViewLabels(dm,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)
  call DMDestroy(dm,ierr);CHKERRA(ierr)
  call PetscFinalize(ierr)

contains
  subroutine ViewLabels(dm,viewer,ierr)
    type(tDM)                        :: dm
    type(tPetscViewer)               :: viewer
    PetscErrorCode                   :: ierr

    DMLabel                          :: label
    type(tIS)                        :: labelIS
    character(len=2048)              :: labelName,IObuffer
    PetscInt                         :: numLabels,l

    call DMGetNumLabels(dm, numLabels, ierr);CHKERRQ(ierr);
    write(IObuffer,*) 'Number of labels: ', numLabels, '\n'
    call PetscViewerASCIIPrintf(viewer, IObuffer, ierr);CHKERRQ(ierr)
    do l = 0, numLabels-1
      call DMGetLabelName(dm, l, labelName, ierr);CHKERRQ(ierr)
      write(IObuffer,*) 'label ',l,' name: ',trim(labelName),'\n'
      call PetscViewerASCIIPrintf(viewer, IObuffer, ierr);CHKERRQ(ierr)

      call PetscViewerASCIIPrintf(viewer, "IS of values\n", ierr);CHKERRQ(ierr)
      call DMGetLabel(dm, labelName, label, ierr);CHKERRQ(ierr)
      call DMLabelGetValueIS(label, labelIS, ierr);CHKERRQ(ierr)
!      call PetscViewerASCIIPushTab(viewer,ierr);CHKERRQ(ierr)
      call ISView(labelIS, viewer, ierr);CHKERRQ(ierr)
!      call PetscViewerASCIIPopTab(viewer,ierr);CHKERRQ(ierr)
      call ISDestroy(labelIS, ierr);CHKERRQ(ierr)
      call PetscViewerASCIIPrintf(viewer, "\n", ierr);CHKERRQ(ierr)
    end do

    call PetscViewerASCIIPrintf(viewer,"\n\nCell Set label IS\n",ierr);CHKERRQ(ierr)
    call DMGetLabel(dm, "Cell Sets", label, ierr);CHKERRQ(ierr)
    call DMLabelGetValueIS(label, labelIS, ierr);CHKERRQ(ierr)
    call ISView(labelIS, viewer, ierr);CHKERRQ(ierr)
    call ISDestroy(labelIS, ierr);CHKERRQ(ierr)
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
