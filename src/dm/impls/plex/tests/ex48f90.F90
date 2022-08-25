program ex47f90
#include "petsc/finclude/petsc.h"
#include "petsc/finclude/petscvec.h"
    use petsc
    use petscvec
    implicit none

    Type(tDM)                         :: dm
    Type(tPetscSection)               :: section
    Character(len=PETSC_MAX_PATH_LEN) :: IOBuffer
    PetscInt                          :: dof,p,pStart,pEnd,d
    Type(tVec)                        :: v
    PetscInt                          :: zero = 0
    PetscInt                          :: one = 1
    PetscInt                          :: two = 2
    PetscScalar,Dimension(:),Pointer  :: val
    PetscScalar, pointer              :: x(:)
    PetscErrorCode                    :: ierr

    PetscCallA(PetscInitialize(ierr))

    PetscCallA(DMCreate(PETSC_COMM_WORLD, dm, ierr))
    PetscCallA(DMSetType(dm, DMPLEX, ierr))
    PetscCallA(DMSetFromOptions(dm, ierr))
    PetscCallA(DMViewFromOptions(dm,PETSC_NULL_OPTIONS,"-d_view",ierr))

    PetscCallA(PetscSectionCreate(PETSC_COMM_WORLD,section,ierr))
    PetscCallA(DMPlexGetChart(dm,pStart,pEnd,ierr))
    PetscCallA(PetscSectionSetChart(section, pStart, pEnd,ierr))
    PetscCallA(DMPlexGetHeightStratum(dm,zero,pStart,pEnd,ierr))
    Do p = pStart,pEnd-1
        PetscCallA(PetscSectionSetDof(section,p,one,ierr))
    End Do
    PetscCallA(DMPlexGetDepthStratum(dm,zero,pStart,pEnd,ierr))
    Do p = pStart,pEnd-1
        PetscCallA(PetscSectionSetDof(section,p,two,ierr))
    End Do
    PetscCallA(PetscSectionSetUp(section,ierr))
    PetscCallA(DMSetLocalSection(dm, section,ierr))
    PetscCallA(PetscSectionViewFromOptions(section,PETSC_NULL_OPTIONS,"-s_view",ierr))

    PetscCallA(DMCreateGlobalVector(dm,v,ierr))

    PetscCallA(DMPlexGetChart(dm,pStart,pEnd,ierr))
    Do p = pStart,pEnd-1
        PetscCallA(PetscSectionGetDof(section,p,dof,ierr))
        Allocate(val(dof))
        Do d = 1,dof
            val(d) = 100*p + d-1;
        End Do
        PetscCallA(VecSetValuesSectionF90(v,section,p,val,INSERT_VALUES,ierr))
        DeAllocate(val)
    End Do
    PetscCallA(VecView(v,PETSC_VIEWER_STDOUT_WORLD,ierr))

    Do p = pStart,pEnd-1
        PetscCallA(PetscSectionGetDof(section,p,dof,ierr))
        PetscCallA(VecGetValuesSectionF90(v,section,p,x,ierr))
        Write(IOBuffer,*) "Point ",p," dof ",dof,"\n"
        PetscCallA(PetscPrintf(PETSC_COMM_SELF,IOBuffer,ierr))
        PetscCallA(VecRestoreValuesSectionF90(v,section,p,x,ierr))
    End Do

    PetscCallA(PetscSectionDestroy(section,ierr))
    PetscCallA(VecDestroy(v,ierr))
    PetscCallA(DMDestroy(dm,ierr))
    PetscCallA(PetscFinalize(ierr))
end program ex47f90

/*TEST

  test:
    suffix: 0
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/quads-q2.msh

TEST*/
