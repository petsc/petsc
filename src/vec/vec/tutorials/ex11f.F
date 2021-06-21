!
!
!
!/*T
!   Concepts: vectors^norms of sub-vectors;
!   Processors: n
!T*/

      program main
#include <petsc/finclude/petscvec.h>
      use petscvec
      implicit none

      Vec               x
      PetscReal         norm
      PetscBool  flg
      PetscMPIInt rank
      PetscInt n,bs,comp
      PetscErrorCode ierr
      PetscScalar       one

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)

      n   = 20
      one = 1.0
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,       &
     &                        '-n',n,flg,ierr)

!
!     Create a vector, specifying only its global dimension.
!     When using VecCreate(), VecSetSizes() and VecSetFromOptions(),
!     the vector format (currently parallel,
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
!     With VecCreate(), VecSetSizes() and VecSetFromOptions() the option
!     -vec_type mpi or -vec_type shared causes the
!     particular type of vector to be formed.

      call VecCreate(PETSC_COMM_WORLD,x,ierr)
      call VecSetSizes(x,PETSC_DECIDE,n,ierr)
      bs = 2
      call VecSetBlockSize(x,bs,ierr)
      call VecSetFromOptions(x,ierr)

!
!     Set the vectors to entries to a constant value.
!
      call VecSet(x,one,ierr)

      call VecNorm(x,NORM_2,norm,ierr)
      if (rank .eq. 0) then
         write (6,100) norm
 100     format ('L_2 Norm of entire vector ',1pe9.2)
      endif

      comp = 0
      call VecStrideNorm(x,comp,NORM_2,norm,ierr)
      if (rank .eq. 0) then
         write (6,200) norm
 200     format ('L_2 Norm of subvector 0',1pe9.2)
      endif

      comp = 1
      call VecStrideNorm(x,comp,NORM_2,norm,ierr)
      if (rank .eq. 0) then
         write (6,300) norm
 300     format ('L_2 Norm of subvector 1',1pe9.2)
      endif

      call VecStrideNorm(x,comp,NORM_1,norm,ierr)
      if (rank .eq. 0) then
         write (6,400) norm
 400     format ('L_1 Norm of subvector 0',1pe9.2)
      endif

      call VecStrideNorm(x,comp,NORM_INFINITY,norm,ierr)
      if (rank .eq. 0) then
         write (6,500) norm
 500     format ('L_1 Norm of subvector 1',1pe9.2)
      endif

!
!     Free work space.  All PETSc objects should be destroyed when they
!     are no longer needed.

      call VecDestroy(x,ierr)
      call PetscFinalize(ierr)
      end

!/*TEST
!
!     test:
!       nsize: 2
!
!TEST*/
