program ex98f90
#include "petsc/finclude/petsc.h"
    use petsc
    implicit none

    ! Get the fortran kind associated with PetscInt and PetscReal so that we can use literal constants.
    PetscInt                           :: dummyPetscInt
    PetscReal                          :: dummyPetscreal
    integer,parameter                  :: kPI = kind(dummyPetscInt)
    integer,parameter                  :: kPR = kind(dummyPetscReal)

    type(tDM)                          :: dm,pdm
    type(tPetscSection)                :: section
    character(len=PETSC_MAX_PATH_LEN)  :: ifilename,iobuffer
    PetscInt                           :: sdim,s,pStart,pEnd,p,numVS,numPoints
    PetscInt,dimension(:),pointer      :: constraints
    type(tIS)                          :: setIS,pointIS
    PetscInt,dimension(:),pointer      :: setID,pointID
    PetscErrorCode                     :: ierr
    PetscBool                          :: flg
    PetscMPIInt                        :: numProc
    MPI_Comm                           :: comm

    PetscCallA(PetscInitialize(ierr))
    PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD,numProc,ierr))

    PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-i",ifilename,flg,ierr))
    if (.not. flg) then
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"missing input file name -i <input file name>")
    end if

    PetscCallA(DMPlexCreateFromFile(PETSC_COMM_WORLD,ifilename,PETSC_NULL_CHARACTER,PETSC_TRUE,dm,ierr))
    PetscCallA(DMPlexDistributeSetDefault(dm,PETSC_FALSE,ierr))
    PetscCallA(DMSetFromOptions(dm,ierr))

    if (numproc > 1) then
        PetscCallA(DMPlexDistribute(dm,0_kPI,PETSC_NULL_SF,pdm,ierr))
        PetscCallA(DMDestroy(dm,ierr))
        dm = pdm
    end if
    PetscCallA(DMViewFromOptions(dm,PETSC_NULL_OPTIONS,"-dm_view",ierr))

    PetscCallA(DMGetDimension(dm,sdim,ierr))
    PetscCallA(PetscObjectGetComm(dm,comm,ierr))
    PetscCallA(PetscSectionCreate(comm,section,ierr))
    PetscCallA(PetscSectionSetNumFields(section,1_kPI,ierr))
    PetscCallA(PetscSectionSetFieldName(section,0_kPI,"U",ierr))
    PetscCallA(PetscSectionSetFieldComponents(section,0_kPI,sdim,ierr))
    PetscCallA(DMPlexGetChart(dm,pStart,pEnd,ierr))
    PetscCallA(PetscSectionSetChart(section,pStart,pEnd,ierr))

    ! initialize the section storage for a P1 field
    PetscCallA(DMPlexGetDepthStratum(dm,0_kPI,pStart,pEnd,ierr))
    do p = pStart,pEnd-1
        PetscCallA(PetscSectionSetDof(section,p,sdim,ierr))
        PetscCallA(PetscSectionSetFieldDof(section,p,0_kPI,sdim,ierr))
    end do

    ! add constraints at all vertices belonging to a vertex set:
    ! first pass is to reserve space
    PetscCallA(DMGetLabelSize(dm,"Vertex Sets",numVS,ierr))
    write(iobuffer,'("# Vertex set: ",i3,"\n")' ) numVS
    PetscCallA(PetscPrintf(PETSC_COMM_WORLD,iobuffer,ierr))
    PetscCallA(DMGetLabelIdIS(dm,"Vertex Sets",setIS,ierr))
    PetscCallA(ISGetIndicesF90(setIS,setID,ierr))
    do s = 1,numVS
        PetscCallA(DMGetStratumIS(dm,"Vertex Sets",setID(s),pointIS,ierr))
        PetscCallA(DMGetStratumSize(dm,"Vertex Sets",setID(s),numPoints,ierr))
        write(iobuffer,'("set ",i3," size ",i3,"\n")' ) s,numPoints
        PetscCallA(PetscPrintf(PETSC_COMM_WORLD,iobuffer,ierr))
        PetscCallA(ISGetIndicesF90(pointIS,pointID,ierr))
        do p = 1,numPoints
            write(iobuffer,'("   point ",i3,"\n")' ) pointID(p)
            PetscCallA(PetscPrintf(PETSC_COMM_WORLD,iobuffer,ierr))
            PetscCallA(PetscSectionSetConstraintDof(section,pointID(p),1_kPI,ierr))
            PetscCallA(PetscSectionSetFieldConstraintDof(section,pointID(p),0_kPI,1_kPI,ierr))
        end do
        PetscCallA(ISRestoreIndicesF90(pointIS,pointID,ierr))
        PetscCallA(ISDestroy(pointIS,ierr))
    end do

    PetscCallA(PetscSectionSetUp(section,ierr))

    ! add constraints at all vertices belonging to a vertex set:
    ! second pass is to assign constraints to a specific component / dof
    allocate(constraints(1))
    do s = 1,numVS
        PetscCallA(DMGetStratumIS(dm,"Vertex Sets",setID(s),pointIS,ierr))
        PetscCallA(DMGetStratumSize(dm,"Vertex Sets",setID(s),numPoints,ierr))
        PetscCallA(ISGetIndicesF90(pointIS,pointID,ierr))
        do p = 1,numPoints
            constraints(1) = mod(setID(s),sdim)
            PetscCallA(PetscSectionSetConstraintIndicesF90(section,pointID(p),constraints,ierr))
            PetscCallA(PetscSectionSetFieldConstraintIndicesF90(section,pointID(p),0_kPI,constraints,ierr))
        end do
        PetscCallA(ISRestoreIndicesF90(pointIS,pointID,ierr))
        PetscCallA(ISDestroy(pointIS,ierr))
    end do
    deallocate(constraints)
    PetscCallA(ISRestoreIndicesF90(setIS,setID,ierr))
    PetscCallA(ISDestroy(setIS,ierr))
    PetscCallA(PetscObjectViewFromOptions(section,PETSC_NULL_SECTION,"-dm_section_view",ierr))

    PetscCallA(PetscSectionDestroy(section,ierr))
    PetscCallA(DMDestroy(dm,ierr))

    PetscCallA(PetscFinalize(ierr))
end program ex98f90

! /*TEST
!   build:
!     requires: exodusii pnetcdf !complex
!   testset:
!     args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/SquareFaceSet.exo -dm_view -dm_section_view
!     nsize: 1
!     test:
!       suffix: 0
!       args:
! TEST*/
