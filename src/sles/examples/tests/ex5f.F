
C      "$Id: ex5f.F,v 1.3 1997/07/10 20:21:42 balay Exp bsmith $";

       implicit none

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/sles.h"
#include "include/FINCLUDE/viewer.h"
C
C      Solves a linear system matrix free
C

      Mat        A
      Vec        x,y
      integer    m,ierr,its 
      SLES       sles
      external   mymatmult

      m = 10
     
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      call SLESCreate(PETSC_COMM_SELF,sles,ierr)

      call MatCreateShell(PETSC_COMM_SELF,m,m,m,m,PETSC_NULL,A,ierr)
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


C  This is a bogus multiply that copies the vector. This corresponds to 
C  an identity matrix A
 
      subroutine mymatmult(A,x,y,ierr)
     
      Mat A
      Vec x,y
      integer m,ierr
 
      m = 10

      call VecCopy(y,x,ierr)

      return 
      end
      
