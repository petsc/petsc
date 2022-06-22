!
!
!
! -----------------------------------------------------------------------

      program main
#include <petsc/finclude/petscvec.h>
      use petscvec
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Variables:
!     x, y, w - vectors
!     z       - array of vectors
!
      Vec              x,y,w
      Vec, pointer :: z(:)
      PetscReal norm,v,v1,v2
      PetscInt         n,ithree
      PetscErrorCode   ierr
      PetscMPIInt      rank
      PetscBool        flg
      PetscScalar      one,two,three
      PetscScalar      dots(3),dot
      PetscReal        nfloat

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      PetscCallA(PetscInitialize(ierr))
      one   = 1.0
      two   = 2.0
      three = 3.0
      n     = 20
      ithree = 3

      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',n,flg,ierr))
      nfloat = n
      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))

!  Create a vector, specifying only its global dimension.
!  When using VecCreate(), VecSetSizes() and VecSetFromOptions(),
!  the vector format (currently parallel
!  or sequential) is determined at runtime.  Also, the parallel
!  partitioning of the vector is determined by PETSc at runtime.
!
!  Routines for creating particular vector types directly are:
!     VecCreateSeq() - uniprocessor vector
!     VecCreateMPI() - distributed vector, where the user can
!                      determine the parallel partitioning

      PetscCallA(VecCreate(PETSC_COMM_WORLD,x,ierr))
      PetscCallA(VecSetSizes(x,PETSC_DECIDE,n,ierr))
      PetscCallA(VecSetFromOptions(x,ierr))

!  Duplicate some work vectors (of the same format and
!  partitioning as the initial vector).

      PetscCallA(VecDuplicate(x,y,ierr))
      PetscCallA(VecDuplicate(x,w,ierr))

!  Duplicate more work vectors (of the same format and
!  partitioning as the initial vector).  Here we duplicate
!  an array of vectors, which is often more convenient than
!  duplicating individual ones.

      PetscCallA(VecDuplicateVecsF90(x,ithree,z,ierr))

!  Set the vectors to entries to a constant value.

      PetscCallA(VecSet(x,one,ierr))
      PetscCallA(VecSet(y,two,ierr))
      PetscCallA(VecSet(z(1),one,ierr))
      PetscCallA(VecSet(z(2),two,ierr))
      PetscCallA(VecSet(z(3),three,ierr))

!  Demonstrate various basic vector routines.

      PetscCallA(VecDot(x,x,dot,ierr))
      PetscCallA(VecMDot(x,ithree,z,dots,ierr))

!  Note: If using a complex numbers version of PETSc, then
!  PETSC_USE_COMPLEX is defined in the makefiles; otherwise,
!  (when using real numbers) it is undefined.

      if (rank .eq. 0) then
#if defined(PETSC_USE_COMPLEX)
         write(6,100) int(PetscRealPart(dot))
         write(6,110) int(PetscRealPart(dots(1))),int(PetscRealPart(dots(2))),int(PetscRealPart(dots(3)))
#else
         write(6,100) int(dot)
         write(6,110) int(dots(1)),int(dots(2)),int(dots(3))
#endif
         write(6,120)
      endif
 100  format ('Vector length ',i6)
 110  format ('Vector length ',3(i6))
 120  format ('All other values should be near zero')

      PetscCallA(VecScale(x,two,ierr))
      PetscCallA(VecNorm(x,NORM_2,norm,ierr))
      v = abs(norm-2.0*sqrt(nfloat))
      if (v .gt. -1.d-10 .and. v .lt. 1.d-10) v = 0.0
      if (rank .eq. 0) write(6,130) v
 130  format ('VecScale ',1pe9.2)

      PetscCallA(VecCopy(x,w,ierr))
      PetscCallA(VecNorm(w,NORM_2,norm,ierr))
      v = abs(norm-2.0*sqrt(nfloat))
      if (v .gt. -1.d-10 .and. v .lt. 1.d-10) v = 0.0
      if (rank .eq. 0) write(6,140) v
 140  format ('VecCopy ',1pe9.2)

      PetscCallA(VecAXPY(y,three,x,ierr))
      PetscCallA(VecNorm(y,NORM_2,norm,ierr))
      v = abs(norm-8.0*sqrt(nfloat))
      if (v .gt. -1.d-10 .and. v .lt. 1.d-10) v = 0.0
      if (rank .eq. 0) write(6,150) v
 150  format ('VecAXPY ',1pe9.2)

      PetscCallA(VecAYPX(y,two,x,ierr))
      PetscCallA(VecNorm(y,NORM_2,norm,ierr))
      v = abs(norm-18.0*sqrt(nfloat))
      if (v .gt. -1.d-10 .and. v .lt. 1.d-10) v = 0.0
      if (rank .eq. 0) write(6,160) v
 160  format ('VecAYXP ',1pe9.2)

      PetscCallA(VecSwap(x,y,ierr))
      PetscCallA(VecNorm(y,NORM_2,norm,ierr))
      v = abs(norm-2.0*sqrt(nfloat))
      if (v .gt. -1.d-10 .and. v .lt. 1.d-10) v = 0.0
      if (rank .eq. 0) write(6,170) v
 170  format ('VecSwap ',1pe9.2)

      PetscCallA(VecNorm(x,NORM_2,norm,ierr))
      v = abs(norm-18.0*sqrt(nfloat))
      if (v .gt. -1.d-10 .and. v .lt. 1.d-10) v = 0.0
      if (rank .eq. 0) write(6,180) v
 180  format ('VecSwap ',1pe9.2)

      PetscCallA(VecWAXPY(w,two,x,y,ierr))
      PetscCallA(VecNorm(w,NORM_2,norm,ierr))
      v = abs(norm-38.0*sqrt(nfloat))
      if (v .gt. -1.d-10 .and. v .lt. 1.d-10) v = 0.0
      if (rank .eq. 0) write(6,190) v
 190  format ('VecWAXPY ',1pe9.2)

      PetscCallA(VecPointwiseMult(w,y,x,ierr))
      PetscCallA(VecNorm(w,NORM_2,norm,ierr))
      v = abs(norm-36.0*sqrt(nfloat))
      if (v .gt. -1.d-10 .and. v .lt. 1.d-10) v = 0.0
      if (rank .eq. 0) write(6,200) v
 200  format ('VecPointwiseMult ',1pe9.2)

      PetscCallA(VecPointwiseDivide(w,x,y,ierr))
      PetscCallA(VecNorm(w,NORM_2,norm,ierr))
      v = abs(norm-9.0*sqrt(nfloat))
      if (v .gt. -1.d-10 .and. v .lt. 1.d-10) v = 0.0
      if (rank .eq. 0) write(6,210) v
 210  format ('VecPointwiseDivide ',1pe9.2)

      dots(1) = one
      dots(2) = three
      dots(3) = two
      PetscCallA(VecSet(x,one,ierr))
      PetscCallA(VecMAXPY(x,ithree,dots,z,ierr))
      PetscCallA(VecNorm(z(1),NORM_2,norm,ierr))
      v = abs(norm-sqrt(nfloat))
      if (v .gt. -1.d-10 .and. v .lt. 1.d-10) v = 0.0
      PetscCallA(VecNorm(z(2),NORM_2,norm,ierr))
      v1 = abs(norm-2.0*sqrt(nfloat))
      if (v1 .gt. -1.d-10 .and. v1 .lt. 1.d-10) v1 = 0.0
      PetscCallA(VecNorm(z(3),NORM_2,norm,ierr))
      v2 = abs(norm-3.0*sqrt(nfloat))
      if (v2 .gt. -1.d-10 .and. v2 .lt. 1.d-10) v2 = 0.0
      if (rank .eq. 0) write(6,220) v,v1,v2
 220  format ('VecMAXPY ',3(1pe9.2))

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.

      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(VecDestroy(y,ierr))
      PetscCallA(VecDestroy(w,ierr))
      PetscCallA(VecDestroyVecsF90(ithree,z,ierr))
      PetscCallA(PetscFinalize(ierr))

      end

!
!/*TEST
!
!     test:
!       nsize: 2
!
!TEST*/
