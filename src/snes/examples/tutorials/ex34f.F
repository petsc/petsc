
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
      Vec             v

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call DMCompositeGetEntries(da,v,v,v,ierr)
      call PetscFinalize()

      end

      Subroutine cDMCompositeGetEntries_3(dm, d1,d2,d3,ierr)
#include "include/finclude/petscall.h90"
      DM dm
      DM d1,d2,d3
      PetscErrorCode ierr

      return 
      end
