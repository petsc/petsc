!
!    "$Id: ex1f90.F,v 1.1 2000/09/06 12:30:18 bsmith Exp bsmith $"
!
      program main 
      use esmf
      integer ierr

      type (esmfgridunstructuredquads),pointer :: grid(:)
      type (esmfgridstructuredrectangular),pointer :: rgrid(:)
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
      ierr = ESMFGridDestroy(rgrid)
      call PetscFinalize();
      end program

