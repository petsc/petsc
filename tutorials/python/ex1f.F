      program main
#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscmat.h"
#include "finclude/petscpc.h"
#include "finclude/petscksp.h"

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
      call PetscPythonInitialize(PETSC_NULL_CHARACTER,                  &
     &     PETSC_NULL_CHARACTER,ierr)

      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-N',                &
     &     N,flg,ierr)
      call PetscOptionsGetBool(PETSC_NULL_CHARACTER,'-draw',            &
     &     draw,flg,ierr)

      call MatCreate(PETSC_COMM_WORLD,A,ierr)
      call MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N,ierr)
      call MatSetType(A,'python',ierr)
      call MatPythonSetType(A,'example1.py:Laplace1D',ierr)
      call MatSetUp(A,ierr)

      call MatGetVecs(A,x,b,ierr)
      call VecSet(b,one,ierr)

      call KSPCreate(PETSC_COMM_WORLD,ksp,ierr)
      call KSPSetType(ksp,'python',ierr)
      call KSPPythonSetType(ksp,'example1.py:ConjGrad',ierr)

      call KSPGetPC(ksp,pc,ierr)
      call PCSetType(pc,'python',ierr)
      call PCPythonSetType(pc,'example1.py:Jacobi',ierr)

      call KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN,ierr)
      call KSPSetFromOptions(ksp,ierr)
      call KSPSolve(ksp,b,x,ierr)

      call VecDuplicate(b,r,ierr)
      call MatMult(A,x,r,ierr)
      call VecAYPX(r,mone,b,ierr)
      call VecNorm(r,NORM_2,rnorm,ierr)
      print*,'error norm = ',rnorm

      if (draw) then
         call VecView(x,PETSC_VIEWER_DRAW_WORLD,ierr)
         call PetscSleep(rtwo,ierr)
      endif

      call VecDestroy(x,ierr)
      call VecDestroy(b,ierr)
      call VecDestroy(r,ierr)
      call MatDestroy(A,ierr)
      call KSPDestroy(ksp,ierr)

      call PetscFinalize(ierr)
      end
