
      program ex34f
#include "include/finclude/petscall.h90"

      Interface DMCompositeGetEntries
        Subroutine DMCompositeGetEntries1(dm, d1,ierr)
          DM  dm
          PetscErrorCode ierr
        End Subroutine

        Subroutine DMCompositeGetEntries2(dm, d1,d2,ierr)
          DM  dm
          DM d1,d2
          PetscErrorCode ierr
        End Subroutine

        Subroutine DMCompositeGetEntries3(dm, d1,d2,d3,ierr)
          DM  dm
          DM d1,d2,d3
          PetscErrorCode ierr
        End Subroutine

        Subroutine DMCompositeGetEntries4(dm, d1,d2,d3,d4,ierr)
          DM  dm
          DM d1,d2,d3,d4
          PetscErrorCode ierr
        End Subroutine
      End Interface

      DMMG             dmmg
      PetscErrorCode   ierr
      DA               da
      DM               dm
      external         FormInitialGuess,FormFunction


      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

!      Unknowns consist of a one-dimensional field, a zero dimensional pool
!          another one-dimensional field and a final zero dimensional pool

      call DMCompositeCreate(PETSC_COMM_WORLD,dm,ierr)
      call DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,5,1,1,              &
     &                PETSC_NULL_INTEGER,da,ierr)
      call DMCompositeAddDA(dm,da,ierr)
      call DMCompositeAddArray(dm,0,10,ierr)
      call DMCompositeAddDA(dm,da,ierr)
      call DMCompositeAddArray(dm,0,10,ierr)
      call DADestroy(da,ierr)

!       Create solver object and associate it with the unknowns

      call DMMGCreate(PETSC_COMM_WORLD,1,PETSC_NULL_OBJECT,dmmg,ierr)
      call DMMGSetDM(dmmg,dm,ierr)
      call DMCompositeDestroy(dm,ierr)
      call DMMGSetInitialGuess(dmmg,FormInitialGuess,ierr)
      call DMMGSetSNES(dmmg,FormFunction,PETSC_NULL_OBJECT,ierr)

      call DMMGSolve(dmmg,ierr)

      call DMMGDestroy(dmmg,ierr)
      call PetscFinalize(ierr)
      end

      subroutine FormInitialGuess(dmmg,v,ierr)
#include "include/finclude/petscall.h90"
      DMMG dmmg
      Vec  v
      PetscErrorCode ierr

      return
      end      

      subroutine FormFunction(dmmg,x,f,dummy,ierr)
#include "include/finclude/petscall.h90"
      DMMG dmmg
      Vec  x,f
      PetscErrorCode ierr
      PetscObject dummy

      return
      end      

