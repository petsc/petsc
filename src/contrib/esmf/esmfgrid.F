!
!    "$Id: esmfgrid.F90,v 1.5 2000/09/11 16:30:19 bsmith Exp bsmith $"
!
      module esmfg
        use esmf
#include "finclude/petscdef.h"
#include "finclude/petscda.h"

!--------------------------------------------------------------
!
!   All grids share this commom basic data
!
      type esmfgrid
        type (esmfbase)  base
        integer          dimension
      end type esmfgrid

!--------------------------------------------------------------
!
!    Representation of a simple (nonstaggered) rectangular grid
!  in two dimensions
!   nc - number of degrees of freedom per grid point (number of components)
!   sw - stencil width (how many layers of ghost nodes)
!
      type esmfgridstructuredrectangular
        type (esmfgrid) grid
        integer         nc,sw
        integer         localranges(2,2)
        integer         ghostranges(2,2)
        integer         globalsizes(2)
        integer         localsizes(2)
        integer         ghostsizes(2)
        integer         da
      end type esmfgridstructuredrectangular

!--------------------------------------------------------------
!
!  Representation of a simple grid of quadralaterals
!
      type esmfgridunstructuredquads
        type (esmfgrid) grid
        integer                      nlocalcells,nghostcells
        integer, pointer ::          cells(:,:)
        double precision, pointer :: vertices(:,:)
      end type esmfgridunstructuredquads

!--------------------------------------------------------------
!     Defines the polymorphic functions for grids
!--------------------------------------------------------------
      interface ESMFGridView
        module procedure ESMFGridView_Base,ESMFGridView_SR,ESMFGridView_UQ
      end interface

!--------------------------------------------------------------

      interface ESMFGridCreate
        module procedure ESMFGridCreate_SR,ESMFGridCreate_UQ
      end interface

!--------------------------------------------------------------

      interface ESMFGridDestroy
        module procedure ESMFGridDestroy_SR,ESMFGridDestroy_UQ
      end interface

!--------------------------------------------------------------
!    Actual functions (methods) that act on the objects
!  These are not called directly by the application codes
!--------------------------------------------------------------
      contains

      subroutine ESMFGridView_Base(grid,ierr)
        implicit none
        type (esmfgrid), pointer :: grid
        integer         ierr

        call CESMFGridView(grid)
        ierr = 0
      end subroutine ESMFGridView_Base

!--------------------------------------------------------------

      subroutine ESMFGridView_SR(grid,ierr)
        implicit none
        type (esmfgridstructuredrectangular), pointer :: grid
        integer         ierr

        call CESMFGridView(grid)
        call CESMFGridViewSR(grid)
        ierr = 0
      end subroutine ESMFGridView_SR

!--------------------------------------------------------------

      subroutine ESMFGridView_UQ(grid,ierr)
        implicit none
        type (esmfgridunstructuredquads), pointer :: grid
        integer         ierr

        call CESMFGridView(grid)
        ierr = 0
      end subroutine ESMFGridView_UQ

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
      function ESMFGridCreate_SR(comm,nx,ny,coors,nc,sw,ierr) result (grid)
        implicit none
        type (esmfgridstructuredrectangular), pointer :: grid
        MPI_Comm                              comm
        integer                               nx,ny,nc,sw,ierr
        double precision                      coors(2,2)

        type (esmfgridstructuredrectangular), pointer :: g

        ierr = 0
        allocate(g)
        g%grid%base%comm        = comm
        g%grid%base%refcount    = 0
        g%grid%base%typename    = 'esmfgridstructuredrectangular'
        g%grid%dimension   = 2
        g%globalsizes(1)   = nx
        g%globalsizes(2)   = ny
        g%nc               = nc
        g%sw               = sw

!
!   create private data structure that contains the information about
!   the distributed grid
!        
        call DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_STAR,nx,ny,&
                        PETSC_DECIDE,PETSC_DECIDE,nc,sw,&
                        PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,&
                        g%da,ierr)
        call DASetUniformCoordinates(g%da,coors(1,1),coors(2,1),&        
                                     coors(1,2),coors(2,2),0,0,ierr)

        g%globalsizes(1) = nx
        g%globalsizes(2) = ny

        call DAGetGhostCorners(g%da,g%ghostranges(1,1),&               
                               g%ghostranges(1,2),PETSC_NULL_INTEGER,&
                               g%ghostsizes(1),g%ghostsizes(2),&
                               PETSC_NULL_INTEGER,ierr)
        g%ghostranges(1,1) = g%ghostranges(1,1) + 1
        g%ghostranges(1,2) = g%ghostranges(1,2) + 1

        g%ghostranges(2,1) = g%ghostranges(1,1) + g%ghostsizes(1) - 1
        g%ghostranges(2,2) = g%ghostranges(1,2) + g%ghostsizes(2) - 1
        call DAGetCorners(g%da,g%localranges(1,1),&
                              g%localranges(1,2),PETSC_NULL_INTEGER,&
                              g%localsizes(1),g%localsizes(2),&
                              PETSC_NULL_INTEGER,ierr)
 
        g%localranges(1,1) = g%localranges(1,1) + 1
        g%localranges(1,2) = g%localranges(1,2) + 1
        g%localranges(2,1) = g%localranges(1,1) + g%localsizes(1) - 1
        g%localranges(2,2) = g%localranges(1,2) + g%localsizes(2) - 1

        grid => g
      end function ESMFGridCreate_SR

!--------------------------------------------------------------

      function ESMFGridCreate_UQ(comm,file,ierr) result (grid)
        implicit none
        type (esmfgridunstructuredquads), pointer :: grid
        character*(*)                                file
        MPI_Comm                                     comm
        integer ierr

        type (esmfgridunstructuredquads), pointer :: g

        ierr = 0
        allocate(g)
        g%grid%base%comm        = comm
        g%grid%base%refcount    = 0
        g%grid%base%typename    = 'esmfgridunstructuredquads'
        g%grid%dimension   = 2

        grid => g
      end function ESMFGridCreate_UQ

!--------------------------------------------------------------

      subroutine ESMFGridDestroy_SR(grid,ierr)
        implicit none
        integer  ierr
        type (esmfgridstructuredrectangular), pointer :: grid

        ierr = 0
        if (grid%grid%base%refcount > 0) then
          grid%grid%base%refcount = grid%grid%base%refcount - 1
          return
        endif
        call DADestroy(grid%da,ierr)
        deallocate(grid)
      end subroutine ESMFGridDestroy_SR

!--------------------------------------------------------------

      subroutine ESMFGridDestroy_UQ(grid,ierr)
        implicit none
        integer ierr
        type (esmfgridunstructuredquads), pointer :: grid

        ierr = 0
        if (grid%grid%base%refcount > 0) then
          grid%grid%base%refcount = grid%grid%base%refcount - 1
          return
        endif
        deallocate(grid)
      end subroutine ESMFGridDestroy_UQ

      end module esmfg





