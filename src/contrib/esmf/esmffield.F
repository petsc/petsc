
!
!    "$Id: esmffield.F,v 1.1 2000/09/07 15:22:58 bsmith Exp bsmith $"
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

       ESMFFieldView_SR(field) result (ierr)
        implicit none
        type (esmffieldstructuredrectangular), pointer :: field
        integer         ierr,i,nc

        ierr = 0
        ierr = ESMFGridView(field%grid)
!   currently not coordinating between various processors; hence
!   print results will be jumbled
        do nc=1,field%grid%nc
          do i=1,field%grid%ghostsizes(1)
            print "(1000f8.3)", field%localindexaccess(nc,i,:)
          enddo
        enddo
      end function ESMFFieldView_SR

!--------------------------------------------------------------

      function ESMFFieldDuplicate_SR(field) result (nfield)
        implicit none
        type (esmffieldstructuredrectangular), pointer :: field
        type (esmffieldstructuredrectangular), pointer :: nfield
        integer         ierr

        ierr = 0
        nfield => ESMFFieldCreate(field%grid)
      end function ESMFFieldDuplicate_SR

!--------------------------------------------------------------

      function ESMFFieldRead_SR(field,io) result (ierr)
        implicit none
        type (esmffieldstructuredrectangular), pointer :: field
        type (esmfio), pointer :: io
        integer         ierr

        ierr = 0
      end function ESMFFieldRead_SR

!--------------------------------------------------------------

      function ESMFFieldWrite_SR(field,io) result (ierr)
        implicit none
        type (esmffieldstructuredrectangular), pointer :: field
        type (esmfio), pointer :: io
        integer         ierr

        ierr = 0
      end function ESMFFieldWrite_SR

!--------------------------------------------------------------

      function ESMFFieldCopy_SR(field,nfield) result (ierr)
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
      end function ESMFFieldCopy_SR

!-------------------------------------------------------------

      function ESMFFieldUpdate_SR(field1,field2) result (ierr)
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
      end function ESMFFieldUpdate_SR

!-------------------------------------------------------------

      function ESMFFieldUpdate_SR1(field) result (ierr)
        implicit none
        type (esmffieldstructuredrectangular), pointer :: field
        integer         ierr
        ierr = ESMFFieldUpdate(field,field)
      end function ESMFFieldUpdate_SR1

!-------------------------------------------------------------

      function ESMFFieldSet_SR(field,value) result (ierr)
        implicit none
        type (esmffieldstructuredrectangular), pointer :: field
        double precision, pointer :: value(:)
        integer         ierr,i
        ierr = 0
        do i=1,field%grid%nc
          field%localindexaccess(i,:,:) = value(i)
        enddo
      end function ESMFFieldSet_SR

!-------------------------------------------------------------

      function ESMFFieldSet_SR1(field,value) result (ierr)
        implicit none
        type (esmffieldstructuredrectangular), pointer :: field
        double precision value
        integer         ierr
        ierr = 0
        field%localindexaccess = value
      end function ESMFFieldSet_SR1

!-------------------------------------------------------------

      function ESMFFieldScale_SR(field,value) result (ierr)
        implicit none
        type (esmffieldstructuredrectangular), pointer :: field
        double precision, pointer :: value(:)
        integer         ierr,i
        ierr = 0
        do i=1,field%grid%nc
          field%localindexaccess(i,:,:) =                               &
     &                       field%localindexaccess(i,:,:)*value(i)
        enddo
      end function ESMFFieldScale_SR

!-------------------------------------------------------------

      function ESMFFieldScale_SR1(field,value) result (ierr)
        implicit none
        type (esmffieldstructuredrectangular), pointer :: field
        double precision value
        integer         ierr
        ierr = 0
        field%localindexaccess = field%localindexaccess*value
      end function ESMFFieldScale_SR1

!--------------------------------------------------------------
!
      function ESMFFieldCreate_SR(grid) result (field)
        implicit none
        type (esmfgridstructuredrectangular), pointer :: grid
        type (esmffieldstructuredrectangular), pointer :: field


        integer                                          ierr
        type (esmffieldstructuredrectangular), pointer :: f

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

      function ESMFFieldDestroy_SR(field) result (ierr)
        implicit none
        integer  ierr
        type (esmffieldstructuredrectangular), pointer :: field

        ierr = 0
        if (field%field%base%refcount > 0) then
          field%field%base%refcount = field%field%base%refcount - 1
          return
        endif
        ierr = ESMFGridDestroy(field%grid)
        deallocate(field)
      end function ESMFFieldDestroy_SR

!--------------------------------------------------------------

      end module esmff





