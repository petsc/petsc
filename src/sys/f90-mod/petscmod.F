#define PETSC_USE_FORTRAN_MODULES

#include "finclude/petscalldef.h"

        module petscdef
#if defined(PETSC_HAVE_MPI_F90MODULE)
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

        module petscisdef
        use petsc
#include "finclude/petscis.h"
#include "finclude/ftn-custom/petscisdef.h90"
        end module

        module petscis
        use petscisdef
#include "finclude/petscis.h90"
        end module

        module petscvecdef
        use petscis
#include "finclude/petscvec.h"
#include "finclude/ftn-custom/petscvecdef.h90"
        end module

        module petscvec
        use petscvecdef
#include "finclude/petscvec.h90"
        end module

        module petscmatdef
        use petscvec
#include "finclude/petscmat.h"
#include "finclude/ftn-custom/petscmatdef.h90"
        end module

        module petscmat
        use petscmatdef
#include "finclude/petscmat.h90"
        end module

        module petscao
        use petscmat
#include "finclude/petscao.h"
        end module

        module petscpcdef
        use petscmat
#include "finclude/petscpc.h"
#include "finclude/ftn-custom/petscpcdef.h90"
        end module

        module petscpc
        use petscpcdef
#include "finclude/petscpc.h90"
        end module

        module petsckspdef
        use petscpc
#include "finclude/petscksp.h"
#include "finclude/ftn-custom/petsckspdef.h90"
        end module

        module petscksp
        use petsckspdef
#include "finclude/petscksp.h90"
        end module

        module petscmgdef
        use petscksp
#include "finclude/petscmg.h"
        end module

        module petscmg
        use petscmgdef
        end module

        module petscdadef
        use petscksp
#include "finclude/petscda.h"
#include "finclude/ftn-custom/petscdadef.h90"
        end module

        module petscda
        use petscdadef
#include "finclude/petscda.h90"
        end module


        module petscsnesdef
        use petscksp
#include "finclude/petscsnes.h"
#include "finclude/ftn-custom/petscsnesdef.h90"
        end module

        module petscsnes
        use petscsnesdef
#include "finclude/petscsnes.h90"
        end module

        module petsctsdef
        use petscsnes
#include "finclude/petscts.h"
#include "finclude/ftn-custom/petsctsdef.h90"
        end module

        module petscts
        use petsctsdef
#include "finclude/petscts.h90"
        end module

        module petscmeshdef
#include "finclude/petscmesh.h"
#include "finclude/ftn-custom/petscmeshdef.h90"
        end module

        module petscmesh
        use petscmeshdef
#include "finclude/petscmesh.h90"
        end module
