        module mpiuni
#include <petsc/mpiuni/mpif.h>
         integer MPI_IN_PLACE
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::MPI_IN_PLACE
#endif
        end module

