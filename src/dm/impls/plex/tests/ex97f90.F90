program ex97f90
#include "petsc/finclude/petsc.h"
    use petsc
    implicit none

    ! Get the fortran kind associated with PetscInt and PetscReal so that we can use literal constants.
    PetscInt                           :: dummyPetscInt
    PetscReal                          :: dummyPetscreal
    integer,parameter                  :: kPI = kind(dummyPetscInt)
    integer,parameter                  :: kPR = kind(dummyPetscReal)

    type(tDM)                          :: dm
    type(tDMLabel)                     :: label
    character(len=PETSC_MAX_PATH_LEN)  :: ifilename,iobuffer
    DMPolytopeType                     :: cellType
    PetscInt                           :: pStart,pEnd,p
    PetscErrorCode                     :: ierr
    PetscBool                          :: flg

    PetscCallA(PetscInitialize(ierr))

    PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-i",ifilename,flg,ierr))
    if (.not. flg) then
        SETERRA(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"missing input file name -i <input file name>")
    end if

    PetscCallA(DMPlexCreateFromFile(PETSC_COMM_WORLD,ifilename,PETSC_NULL_CHARACTER,PETSC_TRUE,dm,ierr))
    PetscCallA(DMPlexDistributeSetDefault(dm,PETSC_FALSE,ierr))
    PetscCallA(PetscObjectSetName(dm,"ex97f90",ierr))
    PetscCallA(DMSetFromOptions(dm,ierr))
    PetscCallA(DMViewFromOptions(dm,PETSC_NULL_OPTIONS,"-dm_view",ierr))

    PetscCallA(DMGetLabel(dm,'celltype',label,ierr))
    PetscCallA(DMLabelView(label,PETSC_VIEWER_STDOUT_WORLD,ierr))
    PetscCallA(DMPlexGetHeightStratum(dm,0_kPI,pStart,pEnd,ierr))
    Do p = pStart,pEnd-1
        PetscCallA(DMPlexGetCellType(dm,p,cellType,ierr))
        Write(IOBuffer,'("cell: ",i3," type: ",i3,"\n")' ) p,cellType
        PetscCallA(PetscPrintf(PETSC_COMM_SELF,IOBuffer,ierr))
    End Do
    PetscCallA(DMDestroy(dm,ierr))

    PetscCallA(PetscFinalize(ierr))
end program ex97f90

! /*TEST
!   build:
!     requires: !complex
!   testset:
!     args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/doublet-tet.msh -dm_view
!     nsize: 1
!     test:
!       suffix: 0
!       args:
! TEST*/
