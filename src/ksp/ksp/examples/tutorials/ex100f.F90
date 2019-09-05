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

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'PetscInitialize failed'
        stop
      endif
      call PetscPythonInitialize(PETSC_NULL_CHARACTER,PETSC_NULL_CHARACTER,ierr);CHKERRA(ierr)

      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-N', N,flg,ierr);CHKERRA(ierr)
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-draw',draw,flg,ierr);CHKERRA(ierr)

      call MatCreate(PETSC_COMM_WORLD,A,ierr);CHKERRA(ierr)
      call MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N,ierr);CHKERRA(ierr)
      call MatSetType(A,'python',ierr);CHKERRA(ierr)
      call MatPythonSetType(A,'example100.py:Laplace1D',ierr);CHKERRA(ierr)
      call MatSetUp(A,ierr);CHKERRA(ierr)

      call MatCreateVecs(A,x,b,ierr);CHKERRA(ierr)
      call VecSet(b,one,ierr);CHKERRA(ierr)

      call KSPCreate(PETSC_COMM_WORLD,ksp,ierr);CHKERRA(ierr)
      call KSPSetType(ksp,'python',ierr);CHKERRA(ierr)
      call KSPPythonSetType(ksp,'example100.py:ConjGrad',ierr);CHKERRA(ierr)

      call KSPGetPC(ksp,pc,ierr);CHKERRA(ierr)
      call PCSetType(pc,'python',ierr);CHKERRA(ierr)
      call PCPythonSetType(pc,'example100.py:Jacobi',ierr);CHKERRA(ierr)

      call KSPSetOperators(ksp,A,A,ierr);CHKERRA(ierr)
      call KSPSetFromOptions(ksp,ierr);CHKERRA(ierr)
      call KSPSolve(ksp,b,x,ierr);CHKERRA(ierr)

      call VecDuplicate(b,r,ierr);CHKERRA(ierr)
      call MatMult(A,x,r,ierr);CHKERRA(ierr)
      call VecAYPX(r,mone,b,ierr);CHKERRA(ierr)
      call VecNorm(r,NORM_2,rnorm,ierr);CHKERRA(ierr)
      print*,'error norm = ',rnorm

      if (draw) then
         call VecView(x,PETSC_VIEWER_DRAW_WORLD,ierr);CHKERRA(ierr)
         call PetscSleep(rtwo,ierr);CHKERRA(ierr)
      endif

      call VecDestroy(x,ierr);CHKERRA(ierr)
      call VecDestroy(b,ierr);CHKERRA(ierr)
      call VecDestroy(r,ierr);CHKERRA(ierr)
      call MatDestroy(A,ierr);CHKERRA(ierr)
      call KSPDestroy(ksp,ierr);CHKERRA(ierr)

      call PetscFinalize(ierr);CHKERRA(ierr)
      end

!/*TEST
!
!    test:
!      requires: petsc4py
!      localrunfiles: example100.py
!
!TEST*/
