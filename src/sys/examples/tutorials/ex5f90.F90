
      module MyModule
#include "finclude/petscsysdef.h"
#include "finclude/petscbagdef.h"
#include "finclude/petscviewerdef.h"
!   Data structure used to contain information about the problem
!   You can add physical values etc here

      type appctx
        PetscScalar :: x
        PetscReal :: y
        PetscInt :: nxc
        PetscTruth :: t
        character*(80) :: c

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
#include "finclude/petscsys.h"
#include "finclude/petscbag.h"
#include "finclude/petscviewer.h"

      PetscBag bag
      PetscErrorCode ierr
      type(AppCtx), pointer :: ctx
      PetscViewer viewer
      PetscSizeT sizeofctx,sizeofint
      PetscSizeT sizeofscalar,sizeoftruth
      PetscSizeT sizeofchar,sizeofreal
      
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

!      compute size of ctx
      call PetscDataTypeGetSize(PETSC_INT,sizeofint,ierr)
      call PetscDataTypeGetSize(PETSC_SCALAR,sizeofscalar,ierr)
      call PetscDataTypeGetSize(PETSC_TRUTH,sizeoftruth,ierr)
      call PetscDataTypeGetSize(PETSC_CHAR,sizeofchar,ierr)
      call PetscDataTypeGetSize(PETSC_REAL,sizeofreal,ierr)

!     really need a sizeof(ctx) operator here. There could be padding inside the
!     structure due to alignment issues - so, this computed value cold be wrong.
      sizeofctx = sizeofint + sizeofscalar+sizeoftruth+sizeofchar*80+sizeofreal

      call PetscBagCreate(PETSC_COMM_WORLD,sizeofctx,bag,ierr)
      call PetscBagGetData(bag,ctx,ierr)
      call PetscBagRegisterInt(bag,ctx%nxc ,56,'nxc','nxc_variable help message',ierr)
      call PetscBagRegisterScalar(bag,ctx%x ,103.2d0,'x','x variable help message',ierr)
      call PetscBagRegisterTruth(bag,ctx%t ,PETSC_TRUE,'t','t boolean help message',ierr)
      call PetscBagRegisterString(bag,ctx%c,'hello','c','string help message',ierr)
      call PetscBagRegisterReal(bag,ctx%y ,-11.0d0,'y','y variable help message',ierr)
      ctx%nxc = 23
      ctx%x   = 155.4
      ctx%c   = 'a whole new string'
      ctx%t   = PETSC_TRUE
      call PetscBagView(bag,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call PetscBagView(bag,PETSC_VIEWER_BINARY_WORLD,ierr)
      call PetscBagDestroy(bag,ierr)

      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'binaryoutput',FILE_MODE_READ,viewer,ierr)
      call PetscBagLoad(viewer,bag,ierr)
      call PetscViewerDestroy(viewer,ierr)
      call PetscBagView(bag,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call PetscBagGetData(bag,ctx,ierr)
      call PetscBagDestroy(bag,ierr)

      call PetscFinalize(ierr)
      end
