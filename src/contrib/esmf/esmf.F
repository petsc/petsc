

!
!    "$Id: esmfgrid.F,v 1.3 2000/09/07 16:42:16 bsmith Exp bsmith $"
!
      module esmfg
#include "finclude/petsc.h"
#include "finclude/petscda.h"

!--------------------------------------------------------------

      type esmfgrid
        character*(64) typename 
        MPI_Comm       comm
        integer        refcount
        integer        dimension
      end type esmfgrid

!--------------------------------------------------------------

      type esmfgridstructuredrectangular
        type (esmfgrid) grid
        integer         nc
        integer         localranges(2,2)
        integer         ghostranges(2,2)
        integer         globalsizes(2)
        integer         localsizes(2)
        integer         ghostsizes(2)
        integer         da
      end type esmfgridstructuredrectangular

!--------------------------------------------------------------

      type esmfgridunstructuredquads
        type (esmfgrid) grid
        integer                      nlocalcells,nghostcells
        integer, pointer ::          cells(:,:)
        double precision, pointer :: vertices(:,:)
      end type esmfgridunstructuredquads

!--------------------------------------------------------------
!--------------------------------------------------------------

      interface ESMFGridView
        module procedure ESMFGridViewBase,ESMFGridViewSR,               &
     &                   ESMFGridViewUQ
      end interface

!--------------------------------------------------------------

      interface ESMFGridCreate
        module procedure ESMFGridCreateSR,ESMFGridCreateUQ
      end interface

!--------------------------------------------------------------

      interface ESMFGridDestroy
        module procedure ESMFGridDestroySR,ESMFGridDestroyUQ
      end interface

!--------------------------------------------------------------
!--------------------------------------------------------------
      contains
      
      function ESMFGridViewBase(grid) result (ierr)
        implicit none
        type (esmfgrid), pointer :: grid
        integer         ierr

        call CESMFGridView(grid)
        ierr = 0
      end function ESMFGridViewBase

!--------------------------------------------------------------

      function ESMFGridViewSR(grid) result (ierr)
        implicit none
        type (esmfgridstructuredrectangular), pointer :: grid
        integer         ierr

        call CESMFGridView(grid)
        call CESMFGridViewSR(grid)
        ierr = 0
      end function ESMFGridViewSR

!--------------------------------------------------------------

      function ESMFGridViewUQ(grid) result (ierr)
        implicit none
        type (esmfgridunstructuredquads), pointer :: grid
        integer         ierr

        call CESMFGridView(grid)
        ierr = 0
      end function ESMFGridViewUQ

!--------------------------------------------------------------
!
!        coors(1,1) = x at lower left corner
!             (1,2) = y at lower left corner
!             (2,1) = x at upper right corner
!             (2,2) = y at upper right corner
!        nc - number of components per grid point
!        sw - stencil width
!        nx,ny - number of grid points in each direction
!
      function ESMFGridCreateSR(comm,nx,ny,coors,nc,sw) result (grid)
        implicit none
        type (esmfgridstructuredrectangular), pointer :: grid
        MPI_Comm                              comm
        integer                               nx,ny,nc,sw
        double precision                      coors(2,2)

        integer                                          ierr
        type (esmfgridstructuredrectangular), pointer :: g

        allocate(g)
        g%grid%comm        = comm
        g%grid%dimension   = 2
        g%grid%refcount    = 0
        g%grid%typename    = 'esmfgridstructuredrectangular'
        g%globalsizes(1)   = nx
        g%globalsizes(2)   = ny
        g%nc               = nc

!
!   create private data structure that contains the information about
!   the distributed grid
!        
        call DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_STAR,nx,ny,        &
     &                  PETSC_DECIDE,PETSC_DECIDE,nc,sw,                  &
     &                  PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,            &
     &                  g%da,ierr)
        call DASetUniformCoordinates(g%da,coors(1,1),coors(2,1),          &
     &                               coors(1,2),coors(2,2),0,0,ierr)

        g%globalsizes(1) = nx
        g%globalsizes(2) = ny

        call DAGetGhostCorners(g%da,g%ghostranges(1,1),                   &
     &                         g%ghostranges(1,2),PETSC_NULL_INTEGER,     &
     &                         g%ghostsizes(1),g%ghostsizes(2),           &
     &                         PETSC_NULL_INTEGER,ierr)
         g%ghostranges(2,1) = g%ghostranges(1,1) + g%ghostsizes(1) - 1
         g%ghostranges(2,2) = g%ghostranges(1,2) + g%ghostsizes(2) - 1
        call DAGetCorners(g%da,g%localranges(1,1),                        &
     &                         g%localranges(1,2),PETSC_NULL_INTEGER,     &
     &                         g%localsizes(1),g%localsizes(2),           &
     &                         PETSC_NULL_INTEGER,ierr)
         g%localranges(2,1) = g%localranges(1,1) + g%localsizes(1) - 1
         g%localranges(2,2) = g%localranges(1,2) + g%localsizes(2) - 1


!        Some f90 compilers (SGI, Solaris) require this
        grid => g
      end function ESMFGridCreateSR

!--------------------------------------------------------------

      function ESMFGridCreateUQ(comm,file) result (grid)
        implicit none
        type (esmfgridunstructuredquads), pointer :: grid
        character*(*)                                file
        MPI_Comm                                     comm

        type (esmfgridunstructuredquads), pointer :: g
        integer                                      ierr

        allocate(g)
        g%grid%comm        = comm
        g%grid%dimension   = 2
        g%grid%refcount    = 0
        g%grid%typename    = 'esmfgridunstructuredquads'

!        Some f90 compilers (SGI, Solaris) require this
        grid => g
      end function ESMFGridCreateUQ

!--------------------------------------------------------------

      function ESMFGridDestroySR(grid) result (ierr)
        implicit none
        integer  ierr
        type (esmfgridstructuredrectangular), pointer :: grid

        ierr = 0
        if (grid%grid%refcount > 0) then
          grid%grid%refcount = grid%grid%refcount - 1
          return
        endif
        call DADestroy(grid%da,ierr)
        deallocate(grid)
      end function ESMFGridDestroySR

!--------------------------------------------------------------

      function ESMFGridDestroyUQ(grid) result (ierr)
        implicit none
        integer ierr
        type (esmfgridunstructuredquads), pointer :: grid

        ierr = 0
        if (grid%grid%refcount > 0) then
          grid%grid%refcount = grid%grid%refcount - 1
          return
        endif
        deallocate(grid)
      end function ESMFGridDestroyUQ

      end module esmfg





