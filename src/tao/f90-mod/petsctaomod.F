
        module petsctaodef
        use petsckspdef
#include <../src/tao/f90-mod/petsctao.h>
        end module

        module petsctao
        use petsctaodef
        use petscksp
#include <../src/tao/f90-mod/petsctao.h90>
        interface
#include <../src/tao/f90-mod/ftn-auto-interfaces/petsctao.h90>
        end interface
        end module


! The all encompassing petsc module

        module petscdef
        use petscdmdadef
        use petscdmplexdef
        use petscdmnetworkdef
        use petscdmpatchdef
        use petscdmforestdef
        use petscdmlabeldef
        use petsctsdef
        use petsctaodef
        end module

        module petsc
        use petscdmda
        use petscdmplex
        use petscdmnetwork
        use petscdmpatch
        use petscdmforest
        use petscdmlabel
        use petscdt
        use petscts
        use petsctao
        end module
