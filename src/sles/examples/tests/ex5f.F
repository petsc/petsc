!
!      "$Id: ex5f.F,v 1.6 1998/04/15 18:02:14 balay Exp balay $";
!
       implicit none

#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"
#include "include/finclude/mat.h"
#include "include/finclude/pc.h"
#include "include/finclude/sles.h"
#include "include/finclude/viewer.h"
!
!      Solves a linear system matrix free
!

      Mat        A
      Vec        x,y
      integer    m,ierr,its 
      SLES       sles
      external   mymatmult

      m = 10
     
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      call SLESCreate(PETSC_COMM_SELF,sles,ierr)

      call MatCreateShell(PETSC_COMM_SELF,m,m,m,m,PETSC_NULL_OBJECT,    &
     &     A,ierr)
      call MatShellSetOperation(A,MATOP_MULT,mymatmult,ierr)

      call VecCreateSeq(PETSC_COMM_SELF,m,x,ierr)
      call VecDuplicate(x,y,ierr)
      call VecSet(1.d0,x,ierr)

      call SLESSetOperators(sles,A,A,SAME_NONZERO_PATTERN,ierr)
      call SLESSetFromOptions(sles,ierr)

      call SLESSolve(sles,x,y,its,ierr)

      call MatDestroy(A,ierr)
      call SLESDestroy(sles,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(y,ierr)

      call PetscFinalize(ierr)
      end


!  This is a bogus multiply that copies the vector. This corresponds to 
!  an identity matrix A
 
      subroutine mymatmult(A,x,y,ierr)
     
      Mat A
      Vec x,y
      integer m,ierr
 
      m = 10

      call VecCopy(y,x,ierr)

      return 
      end
      
