!
!
      subroutine MPIUNISetModuleBlock()
      use mpiuni
      implicit none
      call MPIUNISetFortranBasePointers(MPI_IN_PLACE)
      end subroutine MPIUNISetModuleBlock
