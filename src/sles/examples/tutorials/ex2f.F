
C    "$Id: ex7.F,v 1.9 1995/11/30 22:34:58 bsmith Exp bsmith $";

#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"
#include "include/finclude/mat.h"
#include "include/finclude/pc.h"
#include "include/finclude/sles.h"
C
C  This example demonstrates basic use of the SLES Fortran interface.
C
      integer          i, j, II, JJ, ierr, m, n
      integer          rank, size, its, Istart, Iend
      double precision v, zero, one, none
      Vec              x, u, b
      Mat              A 
      SLES             sles
      double precision norm

      zero = 0.0
      one  = 1.0
      none = -1.0
      call PetscInitialize(ierr)
      m = 3
      n = 3
      call OptionsGetInt(PETSC_NULL,'-m',m,ierr)
      call OptionsGetInt(PETSC_NULL,'-n',n,ierr)
      call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
      call MPI_Comm_size(MPI_COMM_WORLD,size,ierr)

C Create and assemble matrix
      call MatCreateMPIAIJ(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
     *                      m*n,m*n,0,PETSC_NULL,0,PETSC_NULL,A,ierr)
      call MatGetOwnershipRange(A,Istart,Iend,ierr)
      do 10, II=Istart,Iend-1
        v = -1.0
        i = II/n
        j = II - i*n  
        if ( i.gt.0 ) then
          JJ = II - n
          call MatSetValues(A,1,II,1,JJ,v,ADD_VALUES,ierr)
        endif
        if ( i.lt.m-1 ) then
          JJ = II + n
          call MatSetValues(A,1,II,1,JJ,v,ADD_VALUES,ierr)
        endif
        if ( j.gt.0 ) then
          JJ = II - 1
          call MatSetValues(A,1,II,1,JJ,v,ADD_VALUES,ierr)
        endif
        if ( j.lt.n-1 ) then
          JJ = II + 1
          call MatSetValues(A,1,II,1,JJ,v,ADD_VALUES,ierr)
        endif
        v = 4.0
        call  MatSetValues(A,1,II,1,II,v,ADD_VALUES,ierr)
 10   continue
      call MatAssemblyBegin(A,FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(A,FINAL_ASSEMBLY,ierr)

C Create vectors for exact solution, approx solution, and RHS
      call VecCreateMPI(MPI_COMM_WORLD,PETSC_DECIDE,m*n,u,ierr)
      call VecDuplicate(u,b,ierr)
      call VecDuplicate(b,x,ierr)
      call VecSet(one,u,ierr)
      call VecSet(zero,x,ierr)
      call MatMult(A,u,b,ierr)

C Create SLES context and set operators
      call SLESCreate(MPI_COMM_WORLD,sles,ierr)
      call SLESSetOperators(sles,A,A,ALLMAT_DIFFERENT_NONZERO_PATTERN,
     *                      ierr)
C Set options
      call SLESSetFromOptions(sles,ierr)

C Solve linear system
      call SLESSolve(sles,b,x,its,ierr)

C Check the error
      call VecAXPY(none,u,x,ierr)
      call VecNorm(x,NORM_2,norm,ierr)
      if (rank .eq. 0) then
        if (norm .gt. 1.e-12) then
           write(6,100) norm, its
        else
           write(6,110) its
        endif
      endif
  100 format('Norm of error ',e10.4,' iterations ',i5)
  110 format('Norm of error < 1.e-12, iterations ',i5)

C Destroy work space
      call SLESDestroy(sles,ierr)
      call VecDestroy(u,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(b,ierr)
      call MatDestroy(A,ierr)
      call PetscFinalize(ierr)
      stop
      end
