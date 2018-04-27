! Program usage: mpiexec -n 1 toyf[-help] [all TAO options]

!
!min f=(x1-x2)^2 + (x2-2)^2 -2*x1-2*x2
!s.t.     x1^2 + x2 = 2
!      0 <= x1^2 - x2 <= 1
!      -1 <= x1,x2 <= 2
!----------------------------------------------------------------------

      module toymodule
#include "petsc/finclude/petsctao.h"
      use petsctao

      Vec x0,xl,xu
      Vec ce,ci,bl,bu
      Mat Ae,Ai,Hess
      PetscInt n,ne,ni
      end module

      program toyf
      use toymodule
      implicit none

      PetscErrorCode       ierr
      Tao                  tao
      KSP                  ksp
      PC                   pc
!      PetscReal            zero

      external FormFunctionGradient,FormHessian
      external FormInequalityConstraints,FormEqualityConstraints
      external FormInequalityJacobian,FormEqualityJacobian


      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
         print*,'Unable to initialize PETSc'
         stop
      endif

      call PetscPrintf(PETSC_COMM_SELF,'Solution should be f(1,1)=-2\n',ierr);CHKERRA(ierr)

      call InitializeProblem(ierr);CHKERRA(ierr)

      call TaoCreate(PETSC_COMM_SELF,tao,ierr);CHKERRA(ierr)
      call TaoSetType(tao,TAOIPM,ierr);CHKERRA(ierr)
      call TaoSetInitialVector(tao,x0,ierr);CHKERRA(ierr)
      call TaoSetVariableBounds(tao,xl,xu,ierr);CHKERRA(ierr)
      call TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,0,ierr);CHKERRA(ierr)
      call TaoSetEqualityConstraintsRoutine(tao,ce,FormEqualityConstraints,0,ierr);CHKERRA(ierr)
      call TaoSetInequalityConstraintsRoutine(tao,ci,FormInequalityConstraints,0,ierr);CHKERRA(ierr)
      call TaoSetJacobianEqualityRoutine(tao,Ae,Ae,FormEqualityJacobian,0,ierr);CHKERRA(ierr)
      call TaoSetJacobianInequalityRoutine(tao,Ai,Ai,FormInequalityJacobian,0,ierr);CHKERRA(ierr)
      call TaoSetHessianRoutine(tao,Hess,Hess,FormHessian,0,ierr);CHKERRA(ierr)

      call TaoSetFromOptions(tao,ierr);CHKERRA(ierr)
      call TaoGetKSP(tao,ksp,ierr);CHKERRA(ierr)
      call KSPGetPC(ksp,pc,ierr);CHKERRA(ierr)
      call PCSetType(pc,PCLU,ierr);CHKERRA(ierr)
      call PCFactorSetMatSolverType(pc,MATSOLVERSUPERLU,ierr);CHKERRA(ierr)
      call KSPSetType(ksp,KSPPREONLY,ierr);CHKERRA(ierr)
      call KSPSetFromOptions(ksp,ierr);CHKERRA(ierr)

!      zero = 0;
!      call TaoSetTolerances(tao,zero,zero,zero,ierr);CHKERRA(ierr)

      ! Solve
      call TaoSolve(tao,ierr);CHKERRA(ierr)

      ! Finalize Memory
      call DestroyProblem(ierr);CHKERRA(ierr)

      call TaoDestroy(tao,ierr);CHKERRA(ierr)
      call PetscFinalize(ierr)

      stop
      end program toyf


      subroutine InitializeProblem(ierr)
      use toymodule
      implicit none
      PetscReal zero,minus1,two,one
      PetscInt done
      PetscErrorCode ierr
      n = 2
      zero = 0
      minus1 = -1
      two= 2
      one = 1
      done = 1

      call VecCreateSeq(PETSC_COMM_SELF,n,x0,ierr);CHKERRQ(ierr)
      call VecDuplicate(x0,xl,ierr);CHKERRQ(ierr)
      call VecDuplicate(x0,xu,ierr);CHKERRQ(ierr)
      call VecSet(x0,zero,ierr);CHKERRQ(ierr)
      call VecSet(xl,minus1,ierr);CHKERRQ(ierr)
      call VecSet(xu,two,ierr);CHKERRQ(ierr)

      ne = 1
      call VecCreateSeq(PETSC_COMM_SELF,ne,ce,ierr);CHKERRQ(ierr)

      ni = 2
      call VecCreateSeq(PETSC_COMM_SELF,ni,ci,ierr);CHKERRQ(ierr)

      call MatCreateSeqAIJ(PETSC_COMM_SELF,ne,n,n,PETSC_NULL_INTEGER,Ae,ierr);CHKERRQ(ierr)
      call MatCreateSeqAIJ(PETSC_COMM_SELF,ni,n,n,PETSC_NULL_INTEGER,Ai,ierr);CHKERRQ(ierr)
      call MatSetFromOptions(Ae,ierr);CHKERRQ(ierr)
      call MatSetFromOptions(Ai,ierr);CHKERRQ(ierr)


      call MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,done,PETSC_NULL_INTEGER,Hess,ierr);CHKERRQ(ierr)
      call MatSetFromOptions(Hess,ierr);CHKERRQ(ierr)
      ierr = 0
      end subroutine InitializeProblem


      subroutine DestroyProblem(ierr)
      use toymodule
      implicit none

      PetscErrorCode ierr

      call MatDestroy(Ae,ierr);CHKERRQ(ierr)
      call MatDestroy(Ai,ierr);CHKERRQ(ierr)
      call MatDestroy(Hess,ierr);CHKERRQ(ierr)

      call VecDestroy(x0,ierr);CHKERRQ(ierr)
      call VecDestroy(ce,ierr);CHKERRQ(ierr)
      call VecDestroy(ci,ierr);CHKERRQ(ierr)
      call VecDestroy(xl,ierr);CHKERRQ(ierr)
      call VecDestroy(xu,ierr);CHKERRQ(ierr)
      ierr = 0
      end subroutine DestroyProblem

      subroutine FormFunctionGradient(tao, X, f, G, dummy, ierr)
      use toymodule
      implicit none

      PetscErrorCode ierr
      PetscInt dummy
      Vec X,G
      Tao tao
      PetscScalar f
      PetscScalar x_v(0:1),g_v(0:1)
      PetscOffset x_i,g_i


      call VecGetArrayRead(X,x_v,x_i,ierr);CHKERRQ(ierr)
      call VecGetArray(G,g_v,g_i,ierr);CHKERRQ(ierr)
      f=(x_v(x_i)-2.0)*(x_v(x_i)-2.0)+(x_v(x_i+1)-2.0)*(x_v(x_i+1)-2.0)-2.0*(x_v(x_i)+x_v(x_i+1))
      g_v(g_i) = 2.0*(x_v(x_i)-2.0) - 2.0
      g_v(g_i+1) = 2.0*(x_v(x_i+1)-2.0) - 2.0
      call VecRestoreArrayRead(X,x_v,x_i,ierr);CHKERRQ(ierr)
      call VecRestoreArray(G,g_v,g_i,ierr);CHKERRQ(ierr)
      ierr = 0
      end subroutine FormFunctionGradient


      subroutine FormHessian(tao,X,H,Hpre,dummy,ierr)
      use toymodule
      implicit none

      Tao        tao
      Vec              X
      Mat              H, Hpre
      PetscErrorCode   ierr
      PetscInt         dummy

      PetscScalar      de_v(0:1),di_v(0:1)
      PetscOffset      de_i,di_i
      PetscInt         zero(1),ones
      PetscInt         one(1)
      PetscScalar      two(1)
      PetscScalar      val(1)
      Vec DE,DI
      zero(1) = 0
      one(1) = 1
      two(1) = 2.0d0
      ones = 1


      ! fix indices on matsetvalues
      call TaoGetDualVariables(tao,DE,DI,ierr);CHKERRQ(ierr)

      call VecGetArrayRead(DE,de_v,de_i,ierr);CHKERRQ(ierr)
      call VecGetArrayRead(DI,di_v,di_i,ierr);CHKERRQ(ierr)

      val(1)=2 * (1 + de_v(de_i) + di_v(di_i) - di_v(di_i+1))

      call VecRestoreArrayRead(DE,de_v,de_i,ierr);CHKERRQ(ierr)
      call VecRestoreArrayRead(DI,di_v,di_i,ierr);CHKERRQ(ierr)

      call MatSetValues(H,ones,zero,ones,zero,val,INSERT_VALUES,ierr);CHKERRQ(ierr)
      call MatSetValues(H,ones,one,ones,one,two,INSERT_VALUES,ierr);CHKERRQ(ierr)

      call MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY,ierr);CHKERRQ(ierr)
      call MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY,ierr);CHKERRQ(ierr)

      ierr = 0
      end subroutine FormHessian

      subroutine FormInequalityConstraints(tao,X,C,dummy,ierr)
      use toymodule
      implicit none

      Tao      tao
      Vec            X,C
      PetscInt       dummy
      PetscErrorCode ierr
      PetscScalar    x_v(0:1),c_v(0:1)
      PetscOffset    x_i,c_i

      call VecGetArrayRead(X,x_v,x_i,ierr);CHKERRQ(ierr)
      call VecGetArray(C,c_v,c_i,ierr);CHKERRQ(ierr)
      c_v(c_i) = x_v(x_i)*x_v(x_i) - x_v(x_i+1)
      c_v(c_i+1) = -x_v(x_i)*x_v(x_i) + x_v(x_i+1) + 1
      call VecRestoreArrayRead(X,x_v,x_i,ierr);CHKERRQ(ierr)
      call VecRestoreArray(C,c_v,c_i,ierr);CHKERRQ(ierr)

      ierr = 0
      end subroutine FormInequalityConstraints


      subroutine FormEqualityConstraints(tao,X,C,dummy,ierr)
      use toymodule
      implicit none

      Tao      tao
      Vec            X,C
      PetscInt       dummy
      PetscErrorCode ierr
      PetscScalar    x_v(0:1),c_v(0:1)
      PetscOffset    x_i,c_i
      call VecGetArrayRead(X,x_v,x_i,ierr);CHKERRQ(ierr)
      call VecGetArray(C,c_v,c_i,ierr);CHKERRQ(ierr)
      c_v(c_i) = x_v(x_i)*x_v(x_i) + x_v(x_i+1) - 2
      call VecRestoreArrayRead(X,x_v,x_i,ierr);CHKERRQ(ierr)
      call VecRestoreArray(C,c_v,c_i,ierr);CHKERRQ(ierr)
      ierr = 0
      end subroutine FormEqualityConstraints


      subroutine FormInequalityJacobian(tao,X,JI,JIpre,dummy,ierr)
      use toymodule
      implicit none

      Tao       tao
      Vec             X
      Mat             JI,JIpre
      PetscInt        dummy
      PetscErrorCode  ierr

      PetscInt        rows(2),two
      PetscInt        cols(2)
      PetscScalar     vals(4),x_v(0:1)
      PetscOffset     x_i

      two = 2
      call VecGetArrayRead(X,x_v,x_i,ierr)
      CHKERRQ(ierr)
      rows(1)=0
      rows(2) = 1
      cols(1) = 0
      cols(2) = 1
      vals(1) = 2*x_v(x_i)
      vals(2) = -1
      vals(3) = -2*x_v(x_i)
      vals(4) = 1

      call VecRestoreArrayRead(X,x_v,x_i,ierr);CHKERRQ(ierr)
      call MatSetValues(JI,two,rows,two,cols,vals,INSERT_VALUES,ierr);CHKERRQ(ierr)
      call MatAssemblyBegin(JI,MAT_FINAL_ASSEMBLY,ierr);CHKERRQ(ierr)
      call MatAssemblyEnd(JI,MAT_FINAL_ASSEMBLY,ierr);CHKERRQ(ierr)
      ierr = 0
      end subroutine FormInequalityJacobian

      subroutine FormEqualityJacobian(tao,X,JE,JEpre,dummy,ierr)
      use toymodule
      implicit none

      Tao       tao
      Vec             X
      Mat             JE,JEpre
      PetscInt        dummy
      PetscErrorCode  ierr

      PetscInt        rows(2),one,two
      PetscScalar     vals(4),x_v(0:1)
      PetscOffset     x_i
      one = 1
      two = 2
      
      call VecGetArrayRead(X,x_v,x_i,ierr);CHKERRQ(ierr)
      rows(1)=0
      rows(2) = 1
      vals(1) = 2*x_v(x_i)
      vals(2) = 1

      call VecRestoreArrayRead(X,x_v,x_i,ierr);CHKERRQ(ierr)
      call MatSetValues(JE,one,rows,two,rows,vals,INSERT_VALUES,ierr);CHKERRQ(ierr)
      call MatAssemblyBegin(JE,MAT_FINAL_ASSEMBLY,ierr);CHKERRQ(ierr)
      call MatAssemblyEnd(JE,MAT_FINAL_ASSEMBLY,ierr);CHKERRQ(ierr)
      ierr = 0
      end subroutine FormEqualityJacobian

!/*TEST
!
!   build:
!      requires: !complex !single
!
!   test:
!      requires: superlu
!      args: -tao_monitor -tao_view -tao_gatol 1.e-5
!      filter: Error: grep -v IEEE_DENORMAL
!
!TEST*/
