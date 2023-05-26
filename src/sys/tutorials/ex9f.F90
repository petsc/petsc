!
!   Example of using PetscOptionsBegin in Fortran
program ex9f
#include "petsc/finclude/petsc.h"
    use petsc
    implicit none

    PetscReal,Parameter                       :: PReal = 1.0
    Integer,Parameter                         :: Pr = Selected_Real_Kind(Precision(PReal))
    PetscInt,Parameter                        :: PInt = 1
    Integer,Parameter                         :: Pi = kind(PInt)

    PetscErrorCode                            :: ierr
    PetscBool                                 :: flg
    PetscInt                                  :: nopt = 3_Pi
    PetscBool                                 :: bvalue, bdefault = PETSC_TRUE
    PetscBool,dimension(:),pointer            :: barray
    PetscEnum                                 :: evalue, edefault = 2
    PetscInt                                  :: ivalue, idefault = 2_Pi
    PetscInt,dimension(:),pointer             :: iarray
    PetscReal                                 :: rvalue, rdefault = 1.23_Pr
    PetscReal,dimension(:),pointer            :: rarray
    PetscScalar                               :: svalue, sdefault = -4.56_Pr
    PetscScalar,dimension(:),pointer          :: sarray
    character(len=256)                        :: IOBuffer
    character(len=256)                        :: stvalue,stdefault
    character(len=256)                        :: list(6)

    PetscCallA(PetscInitialize(ierr))
    list(1)   = 'a123   '
    list(2)   = 'b456   '
    list(3)   = 'c789   '
    list(4)   = 'list   '
    list(5)   = 'prefix_'
    list(6)   = ''
    stdefault = 'oulala oulala'

    Allocate(iarray(nopt),source=-1_Pi)
    Allocate(rarray(nopt),source=-99.0_pr)
    Allocate(barray(nopt),source=PETSC_FALSE)
    Allocate(sarray(nopt))
    sarray = 123.456_Pr

    PetscCallA(PetscOptionsBegin(PETSC_COMM_WORLD,'prefix_','Setting options for my application','Section 1',ierr))
        PetscCallA(PetscOptionsBool('-bool','Get an application bool','Man page',bdefault,bvalue,flg,ierr))
        if (flg) then
            write(IOBuffer,'("The bool value was set to ",L1,"\n")') bvalue
            PetscCallA(PetscPrintf(PETSC_COMM_WORLD,IOBuffer,ierr))
        endif
        PetscCallA(PetscOptionsBoolArray('-boolarray','Get an application bool array','Man page',barray,nopt,flg,ierr))
        if (flg) then
            write(IOBuffer,'("The bool array was set to ",*(L1," "))') barray
            PetscCallA(PetscPrintf(PETSC_COMM_WORLD,trim(IOBuffer)//"\n",ierr))
        endif
        PetscCallA(PetscOptionsEnum('-enum','Get an application enum','Man page',list,edefault,evalue,flg,ierr))
        if (flg) then
            write(IOBuffer,'("The bool value was set to ",A,"\n")') trim(list(evalue+1))
            PetscCallA(PetscPrintf(PETSC_COMM_WORLD,IOBuffer,ierr))
        endif
        PetscCallA(PetscOptionsInt('-int','Get an application int','Man page',idefault,ivalue,flg,ierr))
        if (flg) then
            write(IOBuffer,'("The integer value was set to ",I8,"\n")') ivalue
            PetscCallA(PetscPrintf(PETSC_COMM_WORLD,IOBuffer,ierr))
        endif
        PetscCallA(PetscOptionsIntArray('-intarray','Get an application int array','Man page',iarray,nopt,flg,ierr))
        if (flg) then
            write(IOBuffer, '("The integer array was set to ",*(I8," "))') iarray
            PetscCallA(PetscPrintf(PETSC_COMM_WORLD,trim(IOBuffer)//"\n",ierr))
        endif
        PetscCallA(PetscOptionsReal('-real','Get an application real','Man page',rdefault,rvalue,flg,ierr))
        if (flg) then
            write(IOBuffer,'("The real value was set to ",ES12.5,"\n")') rvalue
            PetscCallA(PetscPrintf(PETSC_COMM_WORLD,IOBuffer,ierr))
        endif
        PetscCallA(PetscOptionsRealArray('-realarray','Get an application real array','Man page',rarray,nopt,flg,ierr))
        if (flg) then
            write(IOBuffer,'("The real array was set to ",*(ES12.5," "))') rarray
            PetscCallA(PetscPrintf(PETSC_COMM_WORLD,trim(IOBuffer)//"\n",ierr))
        endif
        PetscCallA(PetscOptionsScalar('-scalar','Get an application scalar','Man page',sdefault,svalue,flg,ierr))
        if (flg) then
            write(IOBuffer,'("The scalar value was set to ",ES12.5,"\n")') svalue
            PetscCallA(PetscPrintf(PETSC_COMM_WORLD,IOBuffer,ierr))
        endif
        PetscCallA(PetscOptionsScalarArray('-scalararray','Get an application scalar array','Man page',sarray,nopt,flg,ierr))
        if (flg) then
            write(IOBuffer,'("The scalar array was set to ",*(ES12.5," "))') sarray
            PetscCallA(PetscPrintf(PETSC_COMM_WORLD,trim(IOBuffer)//"\n",ierr))
        endif
        PetscCallA(PetscOptionsString('-string','Get an application string','Man page',stdefault,stvalue,flg,ierr))
        if (flg) then
            write(IOBuffer,'("The string value was set to ",A,"\n")') trim(stvalue)
            PetscCallA(PetscPrintf(PETSC_COMM_WORLD,IOBuffer,ierr))
        endif
    PetscCallA(PetscOptionsEnd(ierr))

    deallocate(iarray)
    deallocate(rarray)
    deallocate(barray)
    deallocate(sarray)
    PetscCallA(PetscFinalize(ierr))
end program ex9f

!
!/*TEST
!
!   build:
!      requires: defined(PETSC_USING_F2003) defined(PETSC_USING_F90FREEFORM) !complex
!
!   test:
!
!   test:
!      suffix: 2
!      args: -prefix_int 22 -prefix_intarray 2-5 -prefix_real 2.34 -prefix_realarray -3,-4,5.5 -prefix_scalar 7.89 -prefix_scalararray 1.,2.,3. -prefix_bool no -prefix_boolarray 1,no,true -prefix_string This_is_a_test_of_the_emergency_alert_system
!
!TEST*/
