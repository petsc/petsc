!
! "$Id: ex11f.F,v 1.2 1999/01/12 23:13:48 bsmith Exp balay $";
!
!
! Program usage:  mpirun ex1 [-help] [all PETSc options] 
!
!
!/*T
!   Concepts: Vectors^Norms of sub-vectors;
!   Routines: VecCreate(); VecSetFromOptions(); VecSet(); VecSetBlockSize(); VecStrideNorm(); VecNorm(); 
!   Processors: n
!T*/

      program ex11f
      implicit none

!
!  The following include statements are required for Fortran programs
!  that use PETSc vectors:
!     petsc.h  - base PETSc routines
!     vec.h    - vectors
!  Additional include statements may be needed if using additional
!  PETSc routines in a Fortran program, e.g.,
!     viewer.h - viewers
!     is.h     - index sets
!
#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"
!

      Vec      x
      double precision  norm
      integer           n, ierr, flg,rank
      Scalar            one

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)

      n   = 20
      one = 1.d0
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-n',n,flg,ierr)

!
!     Create a vector, specifying only its global dimension.
!     When using VecCreate() and VecSetFromOptions(), the vector format (currently parallel,
!     shared, or sequential) is determined at runtime.  Also, the parallel
!     partitioning of the vector is determined by PETSc at runtime.
!
!     Routines for creating particular vector types directly are:
!        VecCreateSeq() - uniprocessor vector
!        VecCreateMPI() - distributed vector, where the user can
!                         determine the parallel partitioning
!        VecCreateShared() - parallel vector that uses shared memory
!                            (available only on the SGI); otherwise,
!                            is the same as VecCreateMPI()
!
!     With VecCreate() and VecSetFromOptions() the option -vec_type mpi or -vec_type shared causes the 
!     particular type of vector to be formed.

      call VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,n,x,ierr)
      call VecSetFromOptions(x,ierr)

!
!     Set the vectors to entries to a constant value.
!
      call VecSet(one,x,ierr) 

      call VecNorm(x,NORM_2,norm,ierr) 
      if (rank .eq. 0) then
        print*,'Norm of entire vector',norm
      endif

      call VecSetBlockSize(x,2,ierr)
      call VecStrideNorm(x,0,NORM_2,norm,ierr) 
      if (rank .eq. 0) then
        print*,'Norm of subvector',norm
      endif


      call VecStrideNorm(x,1,NORM_2,norm,ierr) 
      if (rank .eq. 0) then
        print*,'Norm of subvector',norm
      endif

      call VecStrideNorm(x,1,NORM_1,norm,ierr) 
      if (rank .eq. 0) then
        print*,'Norm of subvector',norm
      endif

      call VecStrideNorm(x,1,NORM_INFINITY,norm,ierr) 
      if (rank .eq. 0) then
        print*,'Norm of subvector',norm
      endif

!
!     Free work space.  All PETSc objects should be destroyed when they
!     are no longer needed.

      call VecDestroy(x,ierr)  
      call PetscFinalize(ierr)
      end
 
