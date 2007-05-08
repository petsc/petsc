
      module MyModule
#include "include/finclude/petsc.h"
#include "include/finclude/petscbag.h"
#include "include/finclude/petscviewer.h"

!   Data structure used to contain information about the problem
!   You can add physical values etc here

      type appctx
        PetscInt :: nxc
        PetscScalar :: x
        PetscTruth :: t
        character*(80) :: c
        PetscReal :: y
      end type appctx
      end module MyModule

      module MyInterface
      Interface PetscBagGetData
        Subroutine PetscBagGetData(bag,ctx,ierr)
          use MyModule
          PetscBag bag
          type(AppCtx), pointer :: ctx
          PetscErrorCode ierr
        End Subroutine
      End Interface PetscBagGetData
      End module MyInterface

      program ex5f90
      use MyModule
      use MyInterface
      implicit none
#define PETSC_AVOID_DECLARATIONS
#include "include/finclude/petsc.h"
#include "include/finclude/petscbag.h"
#include "include/finclude/petscviewer.h"

      PetscBag bag
      PetscErrorCode ierr
      type(AppCtx), pointer :: ctx
      PetscViewer viewer
      
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call PetscBagCreate(PETSC_COMM_WORLD,sizeof(ctx),bag,ierr)
      call PetscBagGetData(bag,ctx,ierr)
      call PetscBagRegisterInt(bag,ctx%nxc ,56,'nxc','nxc_variable help message',ierr)
      call PetscBagRegisterScalar(bag,ctx%x ,103.2d0,'x','x variable help message',ierr)
      call PetscBagRegisterTruth(bag,ctx%t ,PETSC_TRUE,'t','t boolean help message',ierr)
      call PetscBagRegisterString(bag,ctx%c,'hello','c','string help message',ierr)
      call PetscBagRegisterReal(bag,ctx%y ,-11.0d0,'y','y variable help message',ierr)
      ctx%nxc = 23
      ctx%x   = 155.4
      ctx%c   = 'a whole new string'
      call PetscBagView(bag,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call PetscBagView(bag,PETSC_VIEWER_BINARY_WORLD,ierr)
      call PetscBagDestroy(bag,ierr)

      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'binaryoutput',FILE_MODE_READ,viewer,ierr)
      call PetscBagLoad(viewer,bag,ierr)
      call PetscViewerDestroy(viewer,ierr)
      call PetscBagView(bag,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call PetscBagDestroy(bag,ierr)

      call PetscFinalize(ierr)
      end
