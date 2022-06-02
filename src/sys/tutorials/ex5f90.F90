
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscbag.h>
#include <petsc/finclude/petscviewer.h>

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
         PetscBool  :: tarray(3)
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
      PetscSizeT sizeofbag
      Character(len=99) list(6)
      PetscInt three,int56
      PetscReal value
      PetscScalar svalue

      PetscCallA(PetscInitialize(ierr))
      list(1) = 'a123'
      list(2) = 'b456'
      list(3) = 'c789'
      list(4) = 'list'
      list(5) = 'prefix_'
      list(6) = ''
!     cannot just pass a 3 to PetscBagRegisterXXXArray() because it is expecting a PetscInt
      three   = 3

!   compute size of the data
!
      sizeofbag = size(transfer(dummydata,dummychar))

! create the bag
      PetscCallA(PetscBagCreate(PETSC_COMM_WORLD,sizeofbag,bag,ierr))
      PetscCallA(PetscBagGetData(bag,data,ierr))
      PetscCallA(PetscBagSetName(bag,'demo parameters','super secret demo parameters in a bag',ierr))
      PetscCallA(PetscBagSetOptionsPrefix(bag, 'pbag_', ierr))

! register the data within the bag, grabbing values from the options database
!     Need to put the value into a variable for 64 bit indices
      int56 = 56
      PetscCallA(PetscBagRegisterInt(bag,data%nxc ,int56,'nxc','nxc_variable help message',ierr))
      PetscCallA(PetscBagRegisterRealArray(bag,data%rarray,three,'rarray','rarray help message',ierr))
!     Need to put the value into a variable to pass correctly for 128 bit quad precision numbers
      svalue = 103.20
      PetscCallA(PetscBagRegisterScalar(bag,data%x ,svalue,'x','x variable help message',ierr))
      PetscCallA(PetscBagRegisterBool(bag,data%t ,PETSC_TRUE,'t','t boolean help message',ierr))
      PetscCallA(PetscBagRegisterBoolArray(bag,data%tarray,three,'tarray','tarray help message',ierr))
      PetscCallA(PetscBagRegisterString(bag,data%c,'hello','c','string help message',ierr))
      value = -11.00
      PetscCallA(PetscBagRegisterReal(bag,data%y ,value,'y','y variable help message',ierr))
      value = 1.00
      PetscCallA(PetscBagRegisterReal(bag,data%pos%x1 ,value,'pos_x1','tuple value 1 help message',ierr))
      value = 2.00
      PetscCallA(PetscBagRegisterReal(bag,data%pos%x2 ,value,'pos_x2','tuple value 2 help message',ierr))
      PetscCallA(PetscBagRegisterEnum(bag,data%enum ,list,1,'enum','tuple value 2 help message',ierr))
      PetscCallA(PetscBagView(bag,PETSC_VIEWER_STDOUT_WORLD,ierr))

      data%nxc = 23
      data%rarray(1) = -1.0
      data%rarray(2) = -2.0
      data%rarray(3) = -3.0
      data%x   = 155.4
      data%c   = 'a whole new string'
      data%t   = PETSC_TRUE
      data%tarray   = (/PETSC_TRUE,PETSC_FALSE,PETSC_TRUE/)
      PetscCallA(PetscBagView(bag,PETSC_VIEWER_BINARY_WORLD,ierr))

      PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_WORLD,'binaryoutput',FILE_MODE_READ,viewer,ierr))
      PetscCallA(PetscBagLoad(viewer,bag,ierr))
      PetscCallA(PetscViewerDestroy(viewer,ierr))
      PetscCallA(PetscBagView(bag,PETSC_VIEWER_STDOUT_WORLD,ierr))

      PetscCallA(PetscBagSetFromOptions(bag,ierr))
      PetscCallA(PetscBagView(bag,PETSC_VIEWER_STDOUT_WORLD,ierr))
      PetscCallA(PetscBagDestroy(bag,ierr))

      PetscCallA(PetscFinalize(ierr))
      end program ex5f90

!
!/*TEST
!
!   build:
!      requires: defined(PETSC_USING_F2003) defined(PETSC_USING_F90FREEFORM)
!
!   test:
!      args: -pbag_rarray 4,5,88
!
!TEST*/
