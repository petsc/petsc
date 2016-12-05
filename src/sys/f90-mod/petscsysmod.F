
        module petscsysdef
#include <petscconf.h>
#if defined(PETSC_HAVE_MPI_F90MODULE) || defined(PETSC_HAVE_MPIUNI)
        use mpi
#define PETSC_AVOID_MPIF_H
#endif
#include <../src/sys/f90-mod/petscsys.h>
#include <../src/sys/f90-mod/petscdraw.h>
#include <../src/sys/f90-mod/petscviewer.h>
        end module

        module petscsys
        use petscsysdef
#include <../src/sys/f90-mod/petscsys.h90>
        interface
#include <../src/sys/f90-mod/ftn-auto-interfaces/petscsys.h90>
        end interface
        end module

