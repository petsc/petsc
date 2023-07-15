!
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

      PetscCallA(PetscInitialize(ierr))
      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))

      n   = 20
      one = 1.0
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',n,flg,ierr))

!
!     Create a vector, specifying only its global dimension.
!     When using VecCreate(), VecSetSizes() and VecSetFromOptions(),
!     the vector format (currently parallel,
!     shared, or sequential) is determined at runtime.  Also, the parallel
!     partitioning of the vector is determined by PETSc at runtime.

      PetscCallA(VecCreate(PETSC_COMM_WORLD,x,ierr))
      PetscCallA(VecSetSizes(x,PETSC_DECIDE,n,ierr))
      bs = 2
      PetscCallA(VecSetBlockSize(x,bs,ierr))
      PetscCallA(VecSetFromOptions(x,ierr))

!
!     Set the vectors to entries to a constant value.
!
      PetscCallA(VecSet(x,one,ierr))

      PetscCallA(VecNorm(x,NORM_2,norm,ierr))
      if (rank .eq. 0) then
         write (6,100) norm
 100     format ('L_2 Norm of entire vector ',1pe9.2)
      endif

      comp = 0
      PetscCallA(VecStrideNorm(x,comp,NORM_2,norm,ierr))
      if (rank .eq. 0) then
         write (6,200) norm
 200     format ('L_2 Norm of subvector 0',1pe9.2)
      endif

      comp = 1
      PetscCallA(VecStrideNorm(x,comp,NORM_2,norm,ierr))
      if (rank .eq. 0) then
         write (6,300) norm
 300     format ('L_2 Norm of subvector 1',1pe9.2)
      endif

      PetscCallA(VecStrideNorm(x,comp,NORM_1,norm,ierr))
      if (rank .eq. 0) then
         write (6,400) norm
 400     format ('L_1 Norm of subvector 0',1pe9.2)
      endif

      PetscCallA(VecStrideNorm(x,comp,NORM_INFINITY,norm,ierr))
      if (rank .eq. 0) then
         write (6,500) norm
 500     format ('L_1 Norm of subvector 1',1pe9.2)
      endif

!
!     Free work space.  All PETSc objects should be destroyed when they
!     are no longer needed.

      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!     test:
!       nsize: 2
!
!TEST*/
