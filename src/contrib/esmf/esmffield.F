
!
!    "$Id: esmffield.F,v 1.2 2000/09/11 15:46:41 bsmith Exp bsmith $"
!
      module esmff
      use esmfg
#include "finclude/petscdef.h"
#include "finclude/petscvec.h"

!--------------------------------------------------------------

      type esmffield
        type (esmfbase)       base
        type (esmftimestamp)  timestamp
      end type esmffield

      type esmffieldstructuredrectangular
        type (esmffield) field

        type (esmfgridstructuredrectangular), pointer :: grid
        double precision, pointer :: localindexaccess(:,:,:)
      end type esmffieldstructuredrectangular

!--------------------------------------------------------------
!     Defines the polymorphic functions for fields
!--------------------------------------------------------------

      interface ESMFFieldView
        module procedure ESMFFieldView_SR
      end interface

!--------------------------------------------------------------

      interface ESMFFieldCreate
        module procedure ESMFFieldCreate_SR
      end interface

!--------------------------------------------------------------

      interface ESMFFieldDestroy
        module procedure ESMFFieldDestroy_SR
      end interface

!--------------------------------------------------------------

      interface ESMFFieldUpdate
        module procedure ESMFFieldUpdate_SR,ESMFFieldUpdate_SR1
      end interface

!--------------------------------------------------------------

      interface ESMFFieldSet
        module procedure ESMFFieldSet_SR1,ESMFFieldSet_SR
      end interface

!--------------------------------------------------------------

      interface ESMFFieldScale
        module procedure ESMFFieldScale_SR,ESMFFieldScale_SR1
      end interface

!--------------------------------------------------------------

      interface ESMFFieldDuplicate
        module procedure ESMFFieldDuplicate_SR
      end interface

!--------------------------------------------------------------

      interface ESMFFieldCopy
        module procedure ESMFFieldCopy_SR
      end interface

!--------------------------------------------------------------

      interface ESMFFieldRead
        module procedure ESMFFieldRead_SR
      end interface

!--------------------------------------------------------------

      interface ESMFFieldWrite
        module procedure ESMFFieldWrite_SR
      end interface

!--------------------------------------------------------------
!    Actual functions (methods) that act on the objects
!  These are not called directly by the application codes
!--------------------------------------------------------------
      contains

      subroutine ESMFFieldView_SR(field,ierr)
        implicit none
        type (esmffieldstructuredrectangular), pointer :: field
        integer         ierr,i,nc

        ierr = 0
        call ESMFGridView(field%grid,ierr)
!   currently not coordinating between various processors; hence
!   print results will be jumbled
        do nc=1,field%grid%nc
          do i=1,field%grid%ghostsizes(1)
            print "(1000f8.3)", field%localindexaccess(nc,i,:)
          enddo
        enddo
      end subroutine ESMFFieldView_SR

!--------------------------------------------------------------

      function ESMFFieldDuplicate_SR(field,ierr) result (nfield)
        implicit none
        type (esmffieldstructuredrectangular), pointer :: field
        type (esmffieldstructuredrectangular), pointer :: nfield
        integer         ierr

        ierr = 0
        nfield => ESMFFieldCreate(field%grid,ierr)
      end function ESMFFieldDuplicate_SR

!--------------------------------------------------------------

      subroutine ESMFFieldRead_SR(field,io,ierr)
        implicit none
        type (esmffieldstructuredrectangular), pointer :: field
        type (esmfio), pointer :: io
        integer         ierr

        ierr = 0
      end subroutine ESMFFieldRead_SR

!--------------------------------------------------------------

      subroutine ESMFFieldWrite_SR(field,io,ierr)
        implicit none
        type (esmffieldstructuredrectangular), pointer :: field
        type (esmfio), pointer :: io
        integer         ierr

        ierr = 0
      end subroutine ESMFFieldWrite_SR

!--------------------------------------------------------------

      subroutine ESMFFieldCopy_SR(field,nfield,ierr)
        implicit none
        type (esmffieldstructuredrectangular), pointer :: field
        type (esmffieldstructuredrectangular), pointer :: nfield
        integer         ierr

        ierr = 0
!       check if share common grid
!        if (field%grid /= nfield%grid) then
!          ierr = 1
!          return
!        endif
        nfield%localindexaccess = field%localindexaccess
      end subroutine ESMFFieldCopy_SR

!-------------------------------------------------------------

      subroutine ESMFFieldUpdate_SR(field1,field2,ierr)
        implicit none
        type (esmffieldstructuredrectangular), pointer :: field1
        type (esmffieldstructuredrectangular), pointer :: field2
        integer         ierr

        Vec v1,v2

        ierr = 0
!         Creating the vector each time is not the most 
!         efficient, but is good enough for tests
        call DACreateLocalVector(field1%grid%da,v1,ierr)
        call DACreateLocalVector(field2%grid%da,v2,ierr)
        call VecPlaceArray(v1,field1%localindexaccess,ierr)
        call VecPlaceArray(v2,field2%localindexaccess,ierr)
        call DALocalToLocalBegin(field1%grid%da,v1,INSERT_VALUES,v2)
        call DALocalToLocalEnd(field1%grid%da,v1,INSERT_VALUES,v2)
        call VecDestroy(v1,ierr)
        call VecDestroy(v2,ierr)
      end subroutine ESMFFieldUpdate_SR

!-------------------------------------------------------------

      subroutine ESMFFieldUpdate_SR1(field,ierr)
        implicit none
        type (esmffieldstructuredrectangular), pointer :: field
        integer         ierr
        call ESMFFieldUpdate(field,field,ierr)
      end subroutine ESMFFieldUpdate_SR1

!-------------------------------------------------------------

      subroutine ESMFFieldSet_SR(field,value,ierr)
        implicit none
        type (esmffieldstructuredrectangular), pointer :: field
        double precision, pointer :: value(:)
        integer         ierr,i
        ierr = 0
        do i=1,field%grid%nc
          field%localindexaccess(i,:,:) = value(i)
        enddo
      end subroutine ESMFFieldSet_SR

!-------------------------------------------------------------

      subroutine ESMFFieldSet_SR1(field,value,ierr)
        implicit none
        type (esmffieldstructuredrectangular), pointer :: field
        double precision value
        integer         ierr
        ierr = 0
        field%localindexaccess = value
      end subroutine ESMFFieldSet_SR1

!-------------------------------------------------------------

      subroutine ESMFFieldScale_SR(field,value,ierr)
        implicit none
        type (esmffieldstructuredrectangular), pointer :: field
        double precision, pointer :: value(:)
        integer         ierr,i
        ierr = 0
        do i=1,field%grid%nc
          field%localindexaccess(i,:,:) =                               &
     &                       field%localindexaccess(i,:,:)*value(i)
        enddo
      end subroutine ESMFFieldScale_SR

!-------------------------------------------------------------

      subroutine ESMFFieldScale_SR1(field,value,ierr)
        implicit none
        type (esmffieldstructuredrectangular), pointer :: field
        double precision value
        integer         ierr
        ierr = 0
        field%localindexaccess = field%localindexaccess*value
      end subroutine ESMFFieldScale_SR1

!--------------------------------------------------------------
!
      function ESMFFieldCreate_SR(grid,ierr) result (field)
        implicit none
        type (esmfgridstructuredrectangular), pointer :: grid
        type (esmffieldstructuredrectangular), pointer :: field
        integer ierr

        type (esmffieldstructuredrectangular), pointer :: f
 
        ierr = 0
        allocate(f)
        f%field%base%refcount = 0
        f%field%base%comm     = grid%grid%base%comm
        f%field%base%typename = 'esmffieldstructuredrectangular'

        f%grid => grid
        grid%grid%base%refcount = grid%grid%base%refcount + 1

        allocate(f%localindexaccess(grid%nc,grid%ghostsizes(1),         &
     &                              grid%ghostsizes(2)))

        field => f
      end function ESMFFieldCreate_SR

!--------------------------------------------------------------

      subroutine ESMFFieldDestroy_SR(field,ierr)
        implicit none
        integer  ierr
        type (esmffieldstructuredrectangular), pointer :: field

        ierr = 0
        if (field%field%base%refcount > 0) then
          field%field%base%refcount = field%field%base%refcount - 1
          return
        endif
        call ESMFGridDestroy(field%grid,ierr)
        deallocate(field)
      end subroutine ESMFFieldDestroy_SR

!--------------------------------------------------------------

      end module esmff





