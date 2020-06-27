!
!
!     Solves the problem A x - x^3 + 1 = 0 via Picard iteration
!
      module ex21fmodule
        use petscsnes
#include <petsc/finclude/petscsnes.h>
        type userctx
          Mat A
        end type userctx
      end module ex21fmodule

      program main
#include <petsc/finclude/petscsnes.h>
      use ex21fmodule
      implicit none
      SNES snes
      PetscErrorCode ierr
      Vec res,x
      type(userctx) user
      PetscScalar val
      PetscInt one,zero,two
      external FormFunction,FormJacobian

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif

      one = 1
      zero = 0
      two = 2
      call MatCreateSeqAIJ(PETSC_COMM_SELF,two,two,two,PETSC_NULL_INTEGER,user%A,ierr)
      val = 2.0; call MatSetValues(user%A,one,zero,one,zero,val,INSERT_VALUES,ierr)
      val = -1.0; call MatSetValues(user%A,one,zero,one,one,val,INSERT_VALUES,ierr)
      val = -1.0; call MatSetValues(user%A,one,one,one,zero,val,INSERT_VALUES,ierr)
      val = 1.0; call MatSetValues(user%A,one,one,one,one,val,INSERT_VALUES,ierr)
      call MatAssemblyBegin(user%A,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(user%A,MAT_FINAL_ASSEMBLY,ierr)

      call MatCreateVecs(user%A,x,res,ierr)

      call SNESCreate(PETSC_COMM_SELF,snes, ierr)
      call SNESSetPicard(snes, res, FormFunction, user%A, user%A, FormJacobian, user, ierr)
      call SNESSetFromOptions(snes,ierr)
      call SNESSolve(snes, PETSC_NULL_VEC, x, ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(res,ierr)
      call MatDestroy(user%A,ierr)
      call SNESDestroy(snes,ierr)
      call PetscFinalize(ierr)
      end


      subroutine FormFunction(snes, x, f, user, ierr)
      use ex21fmodule
      SNES snes
      Vec  x, f
      type(userctx) user
      PetscErrorCode ierr
      PetscInt i,n
      PetscScalar, pointer :: xx(:),ff(:)

      call MatMult(user%A, x, f, ierr)
      call VecGetArrayF90(f,ff,ierr)
      call VecGetArrayReadF90(x,xx,ierr)
      call VecGetLocalSize(x,n,ierr)
      do 10, i=1,n
         ff(i) = ff(i) - xx(i)*xx(i)*xx(i)*xx(i) + 1.0
 10   continue
      call VecRestoreArrayF90(f,ff,ierr)
      call VecRestoreArrayReadF90(x,xx,ierr)
      end subroutine

!      The matrix is constant so no need to recompute it
      subroutine FormJacobian(snes, x, jac, jacb, user, ierr)
      use ex21fmodule
      SNES snes
      Vec  x
      type(userctx) user
      Mat  jac, jacb
      PetscErrorCode ierr
      end subroutine

!/*TEST
!
!   test:
!     nsize: 1
!     requires: !single
!     args: -snes_monitor -snes_converged_reason -snes_view -pc_type lu
!
!TEST*/
