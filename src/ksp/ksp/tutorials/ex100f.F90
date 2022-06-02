      program main
#include "petsc/finclude/petscksp.h"
      use petscksp

      PetscInt       N
      PetscBool      draw, flg
      PetscReal      rnorm,rtwo
      PetscScalar    one,mone
      Mat            A
      Vec            b,x,r
      KSP            ksp
      PC             pc
      PetscErrorCode ierr

      N    = 100
      draw = .FALSE.
      one  =  1.0
      mone = -1.0
      rtwo = 2.0

      PetscCallA(PetscInitialize(ierr))
      PetscCallA(PetscPythonInitialize(PETSC_NULL_CHARACTER,PETSC_NULL_CHARACTER,ierr))

      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-N', N,flg,ierr))
      PetscCallA(PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-draw',draw,flg,ierr))

      PetscCallA(MatCreate(PETSC_COMM_WORLD,A,ierr))
      PetscCallA(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N,ierr))
      PetscCallA(MatSetType(A,'python',ierr))
      PetscCallA(MatPythonSetType(A,'example100.py:Laplace1D',ierr))
      PetscCallA(MatSetUp(A,ierr))

      PetscCallA(MatCreateVecs(A,x,b,ierr))
      PetscCallA(VecSet(b,one,ierr))

      PetscCallA(KSPCreate(PETSC_COMM_WORLD,ksp,ierr))
      PetscCallA(KSPSetType(ksp,'python',ierr))
      PetscCallA(KSPPythonSetType(ksp,'example100.py:ConjGrad',ierr))

      PetscCallA(KSPGetPC(ksp,pc,ierr))
      PetscCallA(PCSetType(pc,'python',ierr))
      PetscCallA(PCPythonSetType(pc,'example100.py:Jacobi',ierr))

      PetscCallA(KSPSetOperators(ksp,A,A,ierr))
      PetscCallA(KSPSetFromOptions(ksp,ierr))
      PetscCallA(KSPSolve(ksp,b,x,ierr))

      PetscCallA(VecDuplicate(b,r,ierr))
      PetscCallA(MatMult(A,x,r,ierr))
      PetscCallA(VecAYPX(r,mone,b,ierr))
      PetscCallA(VecNorm(r,NORM_2,rnorm,ierr))
      print*,'error norm = ',rnorm

      if (draw) then
         PetscCallA(VecView(x,PETSC_VIEWER_DRAW_WORLD,ierr))
         PetscCallA(PetscSleep(rtwo,ierr))
      endif

      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(VecDestroy(b,ierr))
      PetscCallA(VecDestroy(r,ierr))
      PetscCallA(MatDestroy(A,ierr))
      PetscCallA(KSPDestroy(ksp,ierr))

      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!    test:
!      requires: petsc4py
!      localrunfiles: example100.py
!
!TEST*/
