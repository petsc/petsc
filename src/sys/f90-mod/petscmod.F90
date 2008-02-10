#define PETSC_USE_FORTRAN_MODULES

#define PETSC_AVOID_DECLARATIONS
#include "include/finclude/petscall.h"
#undef PETSC_AVOID_DECLARATIONS

        module petscdef
#if defined(PETSC_HAVE_MPI_F90MODULE)
        use mpi
#define PETSC_AVOID_MPIF_H
#endif
#include "include/finclude/petsc.h"
#include "include/finclude/petscviewer.h"
#include "include/finclude/petscdraw.h"
#include "include/finclude/petsclog.h"
        end module

        module petsc
        use petscdef
#include "include/finclude/petscviewer.h90"
        end module

        module petscsys
        use petsc
#include "include/finclude/petscsys.h"
        end module

        module petscisdef
        use petsc
#include "include/finclude/petscis.h"
        end module

        module petscis
        use petscisdef
#include "include/finclude/petscis.h90"
        end module

        module petscvecdef
        use petscis
#include "include/finclude/petscvec.h"
        end module

        module petscvec
        use petscvecdef
#include "include/finclude/petscvec.h90"
        end module

        module petscmatdef
        use petscvec
#include "include/finclude/petscmat.h"
        end module

        module petscmat
        use petscmatdef
#include "include/finclude/petscmat.h90"
        end module

        module petscao
        use petscmat
#include "include/finclude/petscao.h"
        end module

        module petscpc
        use petscmat
#include "include/finclude/petscpc.h"
        end module

        module petscksp
        use petscpc
#include "include/finclude/petscksp.h"
        end module

        module petscmg
        use petscksp
#include "include/finclude/petscmg.h"
        end module

        module petscdadef
        use petscksp
#include "include/finclude/petscda.h"
        end module

        module petscda
        use petscdadef
#include "include/finclude/petscda.h90"
        end module


        module petscsnes
        use petscksp
#include "include/finclude/petscsnes.h"
        end module

        module petscts
        use petscsnes
#include "include/finclude/petscts.h"
        end module
