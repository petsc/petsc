
      program ex34f
#include "include/finclude/petscall.h90"

      Interface 
        Subroutine DMCompositeGetEntries(dm, d1,d2,d3,d4,d5)
          DM  dm
          DM, optional:: d1,d2,d3,d4,d5
        End Subroutine
      End Interface

      DMMG             dmmg
      PetscErrorCode   ierr
      DA               da
      Vec             v

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call DMCompositeGetEntries(da,v,v,v,v,v)
      call PetscFinalize()

      end

      Subroutine DMCompositeGetEntries(dm, d1,d2,d3,d4,d5)
#include "include/finclude/petscall.h90"
      DM dm
      DM, optional:: d1,d2,d3,d4,d5

      if (present(d5)) then
        call DMCompositeGetEntries5(dm,d1,d2,d3,d4,d5)
      else if (present(d4)) then
        print*, 4
      else if (present(d3)) then
      else if (present(d2)) then
      else if (present(d1)) then
      endif

      print*,"hello"
      return 
      end
