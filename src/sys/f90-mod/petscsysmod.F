#define PETSC_USE_FORTRAN_MODULES

        module petscsysdef
#include <petscconf.h>
#if defined(PETSC_HAVE_MPI_F90MODULE) || defined(PETSC_HAVE_MPIUNI)
        use mpi
#define PETSC_AVOID_MPIF_H
#endif
#include <finclude/petscsys.h>
#include <finclude/petscviewer.h>
#include <finclude/ftn-custom/petscviewerdef.h90>
#include <finclude/petscdraw.h>
#include <finclude/petsclog.h>
        end module

        module petscsys
        use petscsysdef
#include <finclude/petscviewer.h90>
        end module

