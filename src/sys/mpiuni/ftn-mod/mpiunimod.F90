module mpiuni
  use, intrinsic :: ISO_C_binding
#include <petsc/mpiuni/mpif.h>
  integer :: MPI_IN_PLACE
#if defined(PETSC_DLL_EXPORTS)
!DEC$ ATTRIBUTES DLLEXPORT::MPI_IN_PLACE
#endif
end module mpiuni
