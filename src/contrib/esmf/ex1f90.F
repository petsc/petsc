!
!    "$Id: ex1f90.F,v 1.2 2000/09/07 15:04:08 bsmith Exp bsmith $"
!
      program main 
      use esmf
      use esmfg
      use esmff
      implicit none
      integer ierr,mrank

      type (esmfgridunstructuredquads),pointer :: grid
      type (esmfgridstructuredrectangular),pointer :: rgrid
      type (esmffieldstructuredrectangular),pointer :: rfield
      type (esmffieldstructuredrectangular),pointer :: nfield
      double precision coors(2,2)

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr);


      grid => ESMFGridCreate(PETSC_COMM_WORLD,'gridfile')


      ierr = ESMFGridView(grid)
      ierr = ESMFGridDestroy(grid)

      coors(1,1) = 0.0
      coors(1,2) = 0.0
      coors(2,1) = 1.0
      coors(2,2) = 1.0

      rgrid => ESMFGridCreate(PETSC_COMM_WORLD,4,4,coors,1,1)
      ierr = ESMFGridView(rgrid)

      rfield => ESMFFieldCreate(rgrid)

      call MPI_Comm_rank(PETSC_COMM_WORLD,mrank,ierr)
      ierr = ESMFFieldSet(rfield,mrank+1.d0)
      ierr = ESMFFieldUpdate(rfield)
      ierr = ESMFFieldView(rfield)
      ierr = ESMFFieldUpdate(rfield,rfield)

      nfield => ESMFFieldDuplicate(rfield)

      ierr = ESMFFieldDestroy(rfield)
      ierr = ESMFFieldDestroy(nfield)

      ierr = ESMFGridDestroy(rgrid)
      call PetscFinalize();
      end program

