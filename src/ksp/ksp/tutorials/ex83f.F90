!
!     Creates a tridiagonal sparse matrix explicitly in Fortran and solves a linear system with it
!
!     The matrix is provided in two three ways
!       Compressed sparse row: ia(), ja(), and a()
!     Entry triples:  rows(), cols(), and a()
!     Entry triples in a way that supports new nonzero values with the same nonzero structure
!
      program main
#include <petsc/finclude/petscksp.h>
      use petscksp
      implicit none

      PetscInt i,n,nz
      PetscBool flg,equal
      PetscErrorCode ierr
      PetscInt,ALLOCATABLE :: ia(:)
      PetscInt,ALLOCATABLE :: ja(:)
      PetscScalar,ALLOCATABLE :: a(:)
      PetscScalar,ALLOCATABLE :: x(:)
      PetscScalar,ALLOCATABLE :: b(:)

      PetscInt,ALLOCATABLE :: rows(:)
      PetscInt,ALLOCATABLE :: cols(:)

      Mat J,Jt,Jr
      Vec rhs,solution
      KSP ksp
      PC pc

      PetscCallA(PetscInitialize(ierr))

      n = 3
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',n,flg,ierr))
      nz = 3*n - 4;

      ALLOCATE (b(n),x(n))

!     Fill the sparse matrix representation
      ALLOCATE (ia(n+1),ja(nz),a(nz))
      ALLOCATE (rows(nz),cols(nz))

      do i=1,n
        b(i) = 1.0
      enddo

!     PETSc ia() and ja() values begin at 0, not 1, you may need to shift the indices used in your code
      ia(1) = 0
      ia(2) = 1
      do i=3,n
         ia(i) = ia(i-1) + 3
      enddo
      ia(n+1) = ia(n) + 1

      ja(1) = 0
      rows(1) = 0; cols(1) = 0
      a(1)  = 1.0
      do i=2,n-1
         ja(2+3*(i-2))   = i-2
         rows(2+3*(i-2)) = i-1; cols(2+3*(i-2)) = i-2
         a(2+3*(i-2))    = -1.0;
         ja(2+3*(i-2)+1) = i-1
         rows(2+3*(i-2)+1) = i-1; cols(2+3*(i-2)+1) = i-1
         a(2+3*(i-2)+1)  = 2.0;
         ja(2+3*(i-2)+2) = i
         rows(2+3*(i-2)+2) = i-1; cols(2+3*(i-2)+2) = i
         a(2+3*(i-2)+2)  = -1.0;
      enddo
      ja(nz) = n-1
      rows(nz) = n-1; cols(nz) = n-1
      a(nz) = 1.0

      PetscCallA(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,n,n,ia,ja,a,J,ierr))
      PetscCallA(MatCreateSeqAIJFromTriple(PETSC_COMM_SELF,n,n,rows,cols,a,Jt,nz,PETSC_FALSE,ierr))
      PetscCallA(MatEqual(J,Jt,equal,ierr))
      if (equal .neqv. PETSC_TRUE) then
         SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Matrices J and Jt should be equal')
      endif
      PetscCallA(MatDestroy(Jt,ierr))
      PetscCallA(MatCreate(PETSC_COMM_SELF,Jr,ierr))
      PetscCallA(MatSetSizes(Jr,n,n,n,n,ierr))
      PetscCallA(MatSetType(Jr,MATSEQAIJ,ierr))
      PetscCallA(MatSetPreallocationCOO(Jr,nz,rows,cols,ierr))
      PetscCallA(MatSetValuesCOO(Jr,a,INSERT_VALUES,ierr))
      PetscCallA(MatEqual(J,Jr,equal,ierr))
      if (equal .neqv. PETSC_TRUE) then
         SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Matrices J and Jr should be equal')
      endif

      PetscCallA(VecCreateSeqWithArray(PETSC_COMM_SELF,1,n,b,rhs,ierr))
      PetscCallA(VecCreateSeqWithArray(PETSC_COMM_SELF,1,n,x,solution,ierr))

      PetscCallA(KSPCreate(PETSC_COMM_SELF,ksp,ierr))
      PetscCallA(KSPSetErrorIfNotConverged(ksp,PETSC_TRUE,ierr))
!     Default to a direct sparse LU solver for robustness
      PetscCallA(KSPGetPC(ksp,pc,ierr))
      PetscCallA(PCSetType(pc,PCLU,ierr))
      PetscCallA(KSPSetFromOptions(ksp,ierr))
      PetscCallA(KSPSetOperators(ksp,J,J,ierr))

      PetscCallA(KSPSolve(ksp,rhs,solution,ierr))

!     Keep the same size and nonzero structure of the matrix but change its numerical entries
      do i=2,n-1
         a(2+3*(i-2)+1)  = 4.0;
      enddo
      PetscCallA(PetscObjectStateIncrease(J,ierr))
      PetscCallA(MatSetValuesCOO(Jr,a,INSERT_VALUES,ierr))
      PetscCallA(MatEqual(J,Jr,equal,ierr))
      if (equal .neqv. PETSC_TRUE) then
         SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Matrices J and Jr should be equal')
      endif
      PetscCallA(MatDestroy(Jr,ierr))

      PetscCallA(KSPSolve(ksp,rhs,solution,ierr))

      PetscCallA(KSPDestroy(ksp,ierr))
      PetscCallA(VecDestroy(rhs,ierr))
      PetscCallA(VecDestroy(solution,ierr))
      PetscCallA(MatDestroy(J,ierr))

      DEALLOCATE (b,x)
      DEALLOCATE (ia,ja,a)
      DEALLOCATE (rows,cols)

      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!     test:
!       args: -ksp_monitor -ksp_view
!
!TEST*/

