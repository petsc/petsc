program ex60F90

#include <petsc/finclude/petscsys.h>
    use petsc
    implicit none

    PetscBool                        :: flg
    Character(len=256)               :: outputString
    PetscScalar,dimension(:),pointer :: sopt
    PetscBool,dimension(:),pointer   :: bopt
    PetscInt                         :: nopt
    PetscErrorCode                   :: ierr

    PetscCallA(PetscInitialize(ierr))
    nopt = 3
    allocate(bopt(nopt))
    PetscCallA(PetscOptionsGetBoolArray(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-bopt",bopt,nopt,flg,ierr))
    Write(outputString,'("bopt: ",3(l7,"  ")," nopt: ",i3," flg ",l7,"\n")' ) bopt,nopt,flg
    PetscCallA(PetscPrintf(PETSC_COMM_WORLD,outputString,ierr))

    nopt = 3
    allocate(sopt(nopt))
    PetscCallA(PetscOptionsGetScalarArray(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-sopt",sopt,nopt,flg,ierr))
    Write(outputString,'("sopt: ",3(es12.5,"  ")," nopt: ",i3," flg ",l7,"\n")' ) sopt,nopt,flg
    PetscCallA(PetscPrintf(PETSC_COMM_WORLD,outputString,ierr))

    deallocate(bopt)
    deallocate(sopt)
    PetscCallA(PetscFinalize(ierr))
end program ex60F90

!/*TEST
!
!   test:
!      requires: !complex
!      suffix: 0
!      args: -bopt yes,true,0 -sopt -1,2,3,4
!
!TEST*/
