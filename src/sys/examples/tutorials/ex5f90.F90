#define PETSC_USE_FORTRAN_MODULES 1
#include <finclude/petscsysdef.h>
#include <finclude/petscbagdef.h>
#include <finclude/petscviewerdef.h>

      module Bag_data_module
!     Data structure used to contain information about the problem
!     You can add physical values etc here

      type tuple
         PetscReal:: x1,x2
      end type tuple

      type bag_data_type
         PetscScalar :: x
         PetscReal :: y
         PetscInt  :: nxc
         PetscReal :: rarray(3)
         PetscBool  :: t
         PetscEnum :: enum
         character*(80) :: c
         type(tuple) :: pos
      end type bag_data_type
      end module Bag_data_module

      module Bag_interface_module
      use Bag_data_module

      interface PetscBagGetData
         subroutine PetscBagGetData(bag,data,ierr)
           use Bag_data_module
           PetscBag bag
           type(bag_data_type),pointer :: data
           PetscErrorCode ierr
         end subroutine PetscBagGetData
      end interface
      end module Bag_interface_module

      program ex5f90
      use Bag_interface_module
      use petsc
      implicit none

      PetscBag bag
      PetscErrorCode ierr
      type(bag_data_type), pointer :: data
      type(bag_data_type)          :: dummydata
      character(len=1),pointer     :: dummychar(:)
      PetscViewer viewer
      PetscSizeT sizeofbag,sizeofint
      PetscSizeT sizeofscalar,sizeoftruth
      PetscSizeT sizeofchar,sizeofreal
      Character(len=99) list(6)
      
      Call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      list(1) = 'a123'
      list(2) = 'b456'
      list(3) = 'c789'
      list(4) = 'list'
      list(5) = 'prefix_'
      list(6) = ''

!   compute size of the data
!      call PetscDataTypeGetSize(PETSC_INT,sizeofint,ierr)
!      call PetscDataTypeGetSize(PETSC_SCALAR,sizeofscalar,ierr)
!      call PetscDataTypeGetSize(PETSC_BOOL,sizeoftruth,ierr)
       call PetscDataTypeGetSize(PETSC_CHAR,sizeofchar,ierr)
!      call PetscDataTypeGetSize(PETSC_REAL,sizeofreal,ierr)

!     really need a sizeof(data) operator here. There could be padding inside the
!     structure due to alignment issues - so, this computed value cold be wrong.
!      sizeofbag = sizeofint + sizeofscalar + sizeoftruth + sizeofchar*80 &
!     &       + 3*sizeofreal+3*sizeofreal
!     That is correct... unless the sequence keyword is used in the derived
!     types, this length will be wrong because of padding
!     this is a situation where the transfer function is very helpful...
      sizeofbag = size(transfer(dummydata,dummychar))*sizeofchar
      

! create the bag
      call PetscBagCreate(PETSC_COMM_WORLD,sizeofbag,bag,ierr)
      call PetscBagGetData(bag,data,ierr)
      call PetscBagSetName(bag,'demo parameters',                        &
     &      'super secret demo parameters in a bag',ierr)
      call PetscBagSetOptionsPrefix(bag, 'pbag_', ierr)

! register the data within the bag, grabbing values from the options database
      call PetscBagRegisterInt(bag,data%nxc ,56,'nxc',                   &
     &      'nxc_variable help message',ierr)
      call PetscBagRegisterRealArray(bag,data%rarray ,3,'rarray',        &
     &      'rarray help message',ierr)
      call PetscBagRegisterScalar(bag,data%x ,103.2d0,'x',               &
     &      'x variable help message',ierr)
      call PetscBagRegisterBool(bag,data%t ,PETSC_TRUE,'t',              &
     &      't boolean help message',ierr)
      call PetscBagRegisterString(bag,data%c,'hello','c',                &
     &      'string help message',ierr)
      call PetscBagRegisterReal(bag,data%y ,-11.0d0,'y',                 &
     &       'y variable help message',ierr)
      call PetscBagRegisterReal(bag,data%pos%x1 ,1.0d0,'pos_x1',         &
     &      'tuple value 1 help message',ierr)
      call PetscBagRegisterReal(bag,data%pos%x2 ,2.0d0,'pos_x2',         &
     &      'tuple value 2 help message',ierr)
      call PetscBagRegisterEnum(bag,data%enum ,list,1,'enum',            &
     &      'tuple value 2 help message',ierr)
      call PetscBagView(bag,PETSC_VIEWER_STDOUT_WORLD,ierr)

      data%nxc = 23
      data%rarray(1) = -1.0
      data%rarray(2) = -2.0
      data%rarray(3) = -3.0
      data%x   = 155.4
      data%c   = 'a whole new string'
      data%t   = PETSC_TRUE
      call PetscBagView(bag,PETSC_VIEWER_BINARY_WORLD,ierr)

      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'binaryoutput',        &
     &      FILE_MODE_READ,viewer,ierr)
      call PetscBagLoad(viewer,bag,ierr)
      call PetscViewerDestroy(viewer,ierr)
      call PetscBagView(bag,PETSC_VIEWER_STDOUT_WORLD,ierr)
      
      call PetscBagSetFromOptions(bag,ierr)
      call PetscBagView(bag,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call PetscBagDestroy(bag,ierr)

      call PetscFinalize(ierr)
      end program ex5f90
