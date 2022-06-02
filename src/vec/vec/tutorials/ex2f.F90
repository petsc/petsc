!
!
!  Description: Builds a parallel vector with 1 component on the first
!               processor, 2 on the second, etc.  Then each processor adds
!               one to all elements except the last rank.
!
! -----------------------------------------------------------------------

      program main
#include <petsc/finclude/petscvec.h>
      use petscvec
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      Vec     x
      PetscInt N,i,ione
      PetscErrorCode ierr
      PetscMPIInt rank
      PetscScalar  one

      PetscCallA(PetscInitialize(ierr))
      one   = 1.0
      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))

!  Create a parallel vector.
!   - In this case, we specify the size of the local portion on
!     each processor, and PETSc computes the global size.  Alternatively,
!     if we pass the global size and use PETSC_DECIDE for the
!     local size PETSc will choose a reasonable partition trying
!     to put nearly an equal number of elements on each processor.

      N = rank + 1
      PetscCallA(VecCreateMPI(PETSC_COMM_WORLD,N,PETSC_DECIDE,x,ierr))
      PetscCallA(VecGetSize(x,N,ierr))
      PetscCallA(VecSet(x,one,ierr))

!  Set the vector elements.
!   - Note that VecSetValues() uses 0-based row and column numbers
!     in Fortran as well as in C.
!   - Always specify global locations of vector entries.
!   - Each processor can contribute any vector entries,
!     regardless of which processor "owns" them; any nonlocal
!     contributions will be transferred to the appropriate processor
!     during the assembly process.
!   - In this example, the flag ADD_VALUES indicates that all
!     contributions will be added together.

      ione = 1
      do 100 i=0,N-rank-1
         PetscCallA(VecSetValues(x,ione,i,one,ADD_VALUES,ierr))
 100  continue

!  Assemble vector, using the 2-step process:
!    VecAssemblyBegin(), VecAssemblyEnd()
!  Computations can be done while messages are in transition
!  by placing code between these two statements.

      PetscCallA(VecAssemblyBegin(x,ierr))
      PetscCallA(VecAssemblyEnd(x,ierr))

!     Test VecGetValues() with scalar entries
      if (rank .eq. 0) then
        ione = 0
        PetscCallA(VecGetValues(x,ione,i,one,ierr))
      endif

!  View the vector; then destroy it.

      PetscCallA(VecView(x,PETSC_VIEWER_STDOUT_WORLD,ierr))
      PetscCallA(VecDestroy(x,ierr))

      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!     test:
!       nsize: 2
!       filter: grep -v " MPI process"
!
!TEST*/
