!
!
      program main
#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
!
!  This example demonstrates basic use of the PETSc Fortran interface
!  to vectors.
!
       PetscInt  n
       PetscErrorCode ierr
       PetscBool  flg
       PetscScalar      one,two,three,dot
       PetscReal        norm,rdot
       Vec              x,y,w

       n     = 20
       one   = 1.0
       two   = 2.0
       three = 3.0

       call PetscInitialize(PETSC_NULL_CHARACTER,ierr)                  
       call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-n',n,flg,ierr)

! Create a vector, then duplicate it
       call VecCreate(PETSC_COMM_WORLD,x,ierr)
       call VecSetSizes(x,PETSC_DECIDE,n,ierr)
       call VecSetFromOptions(x,ierr)
       call VecDuplicate(x,y,ierr)
       call VecDuplicate(x,w,ierr)

       call VecSet(x,one,ierr)
       call VecSet(y,two,ierr)

       call VecDot(x,y,dot,ierr)
       rdot = PetscRealPart(dot)
       write(6,100) rdot
  100  format('Result of inner product ',f10.4)

       call VecScale(x,two,ierr)
       call VecNorm(x,NORM_2,norm,ierr)
       write(6,110) norm
  110  format('Result of scaling ',f10.4)

       call VecCopy(x,w,ierr)
       call VecNorm(w,NORM_2,norm,ierr)
       write(6,120) norm
  120  format('Result of copy ',f10.4)

       call VecAXPY(y,three,x,ierr)
       call VecNorm(y,NORM_2,norm,ierr)
       write(6,130) norm
  130  format('Result of axpy ',f10.4)

       call VecDestroy(x,ierr)
       call VecDestroy(y,ierr)
       call VecDestroy(w,ierr)
       call PetscFinalize(ierr)
       end

 
