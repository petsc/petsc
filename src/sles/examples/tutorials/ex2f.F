
C    "$Id: ex7.F,v 1.1 1995/10/03 13:20:36 curfman Exp curfman $";

#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"
#include "include/finclude/mat.h"
#include "include/finclude/pc.h"
#include "include/finclude/sles.h"

      integer          i, j, II, JJ, ierr, m, n
      integer          mytid, numtids, its, Istart, Iend
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
      call OptionsGetInt(0,'-m',m,ierr)
      call OptionsGetInt(0,'-n',n,ierr)
      call MPI_Comm_rank(MPI_COMM_WORLD,mytid,ierr)
      call MPI_Comm_size(MPI_COMM_WORLD,numtids,ierr)


C      /* Create and assemble matrix */
      call MatCreateMPIAIJ(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
     *                      m*n,m*n,0,0,0,0,A,ierr)
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

C   /* Create vectors for exact solution, approx solution, and RHS */
      call VecCreateMPI(MPI_COMM_WORLD,PETSC_DECIDE,m*n,u,ierr)
      call VecDuplicate(u,b,ierr)
      call VecDuplicate(b,x,ierr)
      call VecSet(one,u,ierr)
      call VecSet(zero,x,ierr)
      call MatMult(A,u,b,ierr)

C  /* Create SLES context and set operators */
      call SLESCreate(MPI_COMM_WORLD,sles,ierr)
      call SLESSetOperators(sles,A,A,ALLMAT_DIFFERENT_NONZERO_PATTERN,
     *                      ierr)
C  /* Set options */
      call SLESSetFromOptions(sles,ierr)

      call SLESSolve(sles,b,x,its,ierr)

C  /* Check the error */
      call VecAXPY(none,u,x,ierr)
      call VecNorm(x,norm,ierr)
      if (norm .gt. 1.e-12) then
         print*,'Norm of error ',norm,' iterations ',its
      else 
         print*,'Norm of error < 1.e-12, iterations ',its
      endif

C  /* Destroy work space */
      call SLESDestroy(sles,ierr)
      call VecDestroy(u,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(b,ierr)
      call MatDestroy(A,ierr)
      call PetscFinalize(ierr)
      stop
      end
