!
!     Utility routine used to set MPIUNI commonblock
!
      subroutine MPIUNISetCommonBlock()
      implicit none
#include "petsc/mpiuni/mpif.h"
      call MPIUNISetFortranBasePointers(MPI_IN_PLACE)
      return
      end
