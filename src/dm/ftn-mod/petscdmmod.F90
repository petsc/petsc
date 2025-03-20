        module petscdmdef
        use petscvecdef
        use petscmatdef
#include <../ftn/dm/petscall.h>
#include <../ftn/dm/petscspace.h>
#include <../ftn/dm/petscdualspace.h>

       type ttPetscTabulation
         sequence
         PetscInt                K
         PetscInt                Nr
         PetscInt                Np
         PetscInt                Nb
         PetscInt                Nc
         PetscInt                cdim
         PetscReal2d, pointer :: T(:)
       end type ttPetscTabulation

       type tPetscTabulation
         type(ttPetscTabulation), pointer :: ptr
       end type tPetscTabulation

       end module petscdmdef
!     ----------------------------------------------

!     Needed by Fortran stub petscdsgettabulation_()
      subroutine F90Array1dCreateTabulation(array,start,len,ptr)
      use petscdmdef
      implicit none
      PetscInt                    start,len
      PetscTabulation, target  :: array(start:start+len-1)
      PetscTabulation, pointer :: ptr(:)
      ptr => array
      print*,'create tab', array(1)%ptr%K,array(1)%ptr%cdim
      print*,ptr(1)%ptr%K,ptr(1)%ptr%cdim
      end subroutine
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: F90Array1dCreateTabulation
#endif

      subroutine F90Array1dDestroyTabulation(ptr)
      use petscdmdef
      implicit none
      PetscTabulation, pointer :: ptr(:)
      nullify(ptr)
      end subroutine
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: F90Array1dDestroyTabulation
#endif

        module petscdm
        use petscmat
        use petscdmdef
#include <../src/dm/ftn-mod/petscdm.h90>
#include <../src/dm/ftn-mod/petscdt.h90>
#include <../ftn/dm/petscall.h90>
#include <../ftn/dm/petscspace.h90>
#include <../ftn/dm/petscdualspace.h90>

        interface PetscDSGetTabulationSetSizes
        subroutine PetscDSGetTabulationSetSizes(ds,i, tab,ierr)
          import tPetscDS, ttPetscTabulation
          PetscErrorCode              ierr
          type(ttPetscTabulation)     tab
          PetscDS                     ds
          PetscInt                    i
        end subroutine
        end interface

        interface PetscDSGetTabulationSetPointers
        subroutine PetscDSGetTabulationSetPointers(ds,i, T,ierr)
          import tPetscDS, ttPetscTabulation,tPetscReal2d
          PetscErrorCode              ierr
          type(tPetscReal2d), pointer :: T(:)
          PetscDS                     ds
          PetscInt                    i
        end subroutine
        end interface

        interface PetscDSGetTabulation
          module procedure PetscDSGetTabulation
       end interface

        interface PetscDSRestoreTabulation
          module procedure PetscDSRestoreTabulation
       end interface

       contains

#include <../ftn/dm/petscall.hf90>
#include <../ftn/dm/petscspace.hf90>
#include <../ftn/dm/petscdualspace.hf90>

        Subroutine PetscDSGetTabulation(ds,tab,ierr)
          PetscErrorCode              ierr
          PetscTabulation, pointer :: tab(:)
          PetscDS                     ds

          PetscInt  Nf, i
          call PetscDSGetNumFields(ds, Nf, ierr)
          allocate(tab(Nf))
          do i=1,Nf
             allocate(tab(i)%ptr)
             CHKMEMQ
             call PetscDSGetTabulationSetSizes(ds, i, tab(i)%ptr, ierr)
             CHKMEMQ
             allocate(tab(i)%ptr%T(tab(i)%ptr%K+1))
             call PetscDSGetTabulationSetPointers(ds, i, tab(i)%ptr%T, ierr)
             CHKMEMQ
          enddo
        End Subroutine PetscDSGetTabulation

        Subroutine PetscDSRestoreTabulation(ds,tab,ierr)
          PetscErrorCode              ierr
          PetscTabulation, pointer :: tab(:)
          PetscDS                     ds

          PetscInt  Nf, i
          call PetscDSGetNumFields(ds, Nf, ierr)
          do i=1,Nf
             deallocate(tab(i)%ptr%T)
             deallocate(tab(i)%ptr)
          enddo
          deallocate(tab)
        End Subroutine PetscDSRestoreTabulation

        end module petscdm

!     ----------------------------------------------

        module petscdmdadef
        use petscdmdef
        use petscaodef
        use petscpfdef
#include <petsc/finclude/petscao.h>
#include <petsc/finclude/petscdmda.h>
#include <../ftn/dm/petscdmda.h>
        end module petscdmdadef

        module petscdmda
        use petscdm
        use petscdmdadef

#include <../src/dm/ftn-mod/petscdmda.h90>
#include <../ftn/dm/petscdmda.h90>

        contains

#include <../ftn/dm/petscdmda.hf90>
        end module petscdmda

!     ----------------------------------------------

        module petscdmplex
        use petscdm
        use petscdmdef
#include <petsc/finclude/petscfv.h>
#include <petsc/finclude/petscdmplex.h>
#include <petsc/finclude/petscdmplextransform.h>
#include <../src/dm/ftn-mod/petscdmplex.h90>
#include <../ftn/dm/petscfv.h>
#include <../ftn/dm/petscdmplex.h>
#include <../ftn/dm/petscdmplextransform.h>

#include <../ftn/dm/petscfv.h90>
#include <../ftn/dm/petscdmplex.h90>
#include <../ftn/dm/petscdmplextransform.h90>

        contains

#include <../ftn/dm/petscfv.hf90>
#include <../ftn/dm/petscdmplex.hf90>
#include <../ftn/dm/petscdmplextransform.hf90>
        end module petscdmplex

!     ----------------------------------------------

        module petscdmstag
        use petscdmdef
#include <petsc/finclude/petscdmstag.h>
#include <../ftn/dm/petscdmstag.h>

#include <../ftn/dm/petscdmstag.h90>

        contains

#include <../ftn/dm/petscdmstag.hf90>
        end module petscdmstag

!     ----------------------------------------------

        module petscdmswarm
        use petscdm
        use petscdmdef
#include <petsc/finclude/petscdmswarm.h>
#include <../ftn/dm/petscdmswarm.h>

#include <../src/dm/ftn-mod/petscdmswarm.h90>
#include <../ftn/dm/petscdmswarm.h90>

        contains

#include <../ftn/dm/petscdmswarm.hf90>
        end module petscdmswarm

!     ----------------------------------------------

        module petscdmcomposite
        use petscdm
#include <petsc/finclude/petscdmcomposite.h>

#include <../src/dm/ftn-mod/petscdmcomposite.h90>
#include <../ftn/dm/petscdmcomposite.h90>
        end module petscdmcomposite

!     ----------------------------------------------

        module petscdmforest
        use petscdm
#include <petsc/finclude/petscdmforest.h>
#include <../ftn/dm/petscdmforest.h>
#include <../ftn/dm/petscdmforest.h90>
        end module petscdmforest

!     ----------------------------------------------

        module petscdmnetwork
        use petscdm
#include <petsc/finclude/petscdmnetwork.h>
#include <../ftn/dm/petscdmnetwork.h>

#include <../ftn/dm/petscdmnetwork.h90>

        contains

#include <../ftn/dm/petscdmnetwork.hf90>
        end module petscdmnetwork

!     ----------------------------------------------

        module petscdmadaptor
        use petscdm
        use petscdmdef
!        use petscsnes
#include <petsc/finclude/petscdmadaptor.h>
#include <../ftn/dm/petscdmadaptor.h>

!#include <../ftn/dm/petscdmadaptor.h90>

        contains

!#include <../ftn/dm/petscdmadaptor.hf90>
        end module petscdmadaptor
