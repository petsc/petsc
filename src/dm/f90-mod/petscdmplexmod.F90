
        module petscdmplexdef
        use petscdmdef
#include <../src/dm/f90-mod/petscdmplex.h>
        end module petscdmplexdef

        module petscdmnetworkdef
        use petscdmplexdef
#include <../src/dm/f90-mod/petscdmnetwork.h>
        end module petscdmnetworkdef

        module petscdmplex
        use petscdmlabel
        use petscdmplexdef
#include <../src/dm/f90-mod/petscdmplex.h90>
        interface
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmplex.h90>
        end interface
        end module petscdmplex

        module petscdmnetwork
        use petscdmplex
#include <../src/dm/f90-mod/petscdmnetwork.h90>
        interface
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmnetwork.h90>
        end interface
        end module petscdmnetwork
