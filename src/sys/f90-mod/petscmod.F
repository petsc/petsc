#define PETSC_USE_FORTRAN_MODULES

        module petscdef
#include "petscconf.h"
#if defined(PETSC_HAVE_MPI_F90MODULE) || defined(PETSC_HAVE_MPIUNI)
        use mpi
#define PETSC_AVOID_MPIF_H
#endif
#include "finclude/petsc.h"
#include "finclude/petscviewer.h"
#include "finclude/ftn-custom/petscviewerdef.h90"
#include "finclude/petscdraw.h"
#include "finclude/petsclog.h"
        end module

        module petsc
        use petscdef
#include "finclude/petscviewer.h90"
        end module

        module petscsys
        use petsc
#include "finclude/petscsys.h"
        end module

