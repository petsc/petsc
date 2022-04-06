!
!
!
! -----------------------------------------------------------------------

      program main

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!
!     This examples uses Fortran 90 MODULES instead of include files
!
#include <petsc/finclude/petscvec.h>
      use petscvec
      implicit none

!
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Variables:
!     x, y, w - vectors
!     z       - array of vectors
!
      type(tVec)       x,y,w
      type(tVec), pointer :: z(:)

      PetscReal norm,v,v1,v2,tol
      PetscInt  n,ithree
      PetscErrorCode ierr
      PetscMPIInt  rank
      PetscBool       flg
      PetscScalar      one,two,three
      PetscScalar      dots(3),dot
      PetscReal        nfloat

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
         print*,'Unable to initialize PETSc'
         stop
      endif

      tol   = 1.e-10_PETSC_REAL_KIND
      one   = 1.0
      two   = 2.0
      three = 3.0
      n     = 20
      ithree = 3

      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',n,flg,ierr);CHKERRA(ierr)
      nfloat = n
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr);CHKERRA(ierr)

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

      call VecCreate(PETSC_COMM_WORLD,x,ierr);CHKERRA(ierr)
      call VecSetSizes(x,PETSC_DECIDE,n,ierr);CHKERRA(ierr)
      call VecSetFromOptions(x,ierr);CHKERRA(ierr)

!  Duplicate some work vectors (of the same format and
!  partitioning as the initial vector).

      call VecDuplicate(x,y,ierr);CHKERRA(ierr)
      call VecDuplicate(x,w,ierr);CHKERRA(ierr)

!  Duplicate more work vectors (of the same format and
!  partitioning as the initial vector).  Here we duplicate
!  an array of vectors, which is often more convenient than
!  duplicating individual ones.

      call VecDuplicateVecsF90(x,ithree,z,ierr);CHKERRA(ierr)

!  Set the vectors to entries to a constant value.

      call VecSet(x,one,ierr);CHKERRA(ierr)
      call VecSet(y,two,ierr);CHKERRA(ierr)
      call VecSet(z(1),one,ierr);CHKERRA(ierr)
      call VecSet(z(2),two,ierr);CHKERRA(ierr)
      call VecSet(z(3),three,ierr);CHKERRA(ierr)

!  Demonstrate various basic vector routines.

      call VecDot(x,x,dot,ierr);CHKERRA(ierr)
      call VecMDot(x,ithree,z,dots,ierr);CHKERRA(ierr)

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

      call VecScale(x,two,ierr);CHKERRA(ierr)
      call VecNorm(x,NORM_2,norm,ierr);CHKERRA(ierr)
      v = abs(norm-2.0*sqrt(nfloat))
      if (v .gt. -tol .and. v .lt. tol) v = 0.0
      if (rank .eq. 0) write(6,130) v
 130  format ('VecScale ',1pe9.2)

      call VecCopy(x,w,ierr);CHKERRA(ierr)
      call VecNorm(w,NORM_2,norm,ierr);CHKERRA(ierr)
      v = abs(norm-2.0*sqrt(nfloat))
      if (v .gt. -tol .and. v .lt. tol) v = 0.0
      if (rank .eq. 0) write(6,140) v
 140  format ('VecCopy ',1pe9.2)

      call VecAXPY(y,three,x,ierr);CHKERRA(ierr)
      call VecNorm(y,NORM_2,norm,ierr);CHKERRA(ierr)
      v = abs(norm-8.0*sqrt(nfloat))
      if (v .gt. -tol .and. v .lt. tol) v = 0.0
      if (rank .eq. 0) write(6,150) v
 150  format ('VecAXPY ',1pe9.2)

      call VecAYPX(y,two,x,ierr);CHKERRA(ierr)
      call VecNorm(y,NORM_2,norm,ierr);CHKERRA(ierr)
      v = abs(norm-18.0*sqrt(nfloat))
      if (v .gt. -tol .and. v .lt. tol) v = 0.0
      if (rank .eq. 0) write(6,160) v
 160  format ('VecAYXP ',1pe9.2)

      call VecSwap(x,y,ierr);CHKERRA(ierr)
      call VecNorm(y,NORM_2,norm,ierr);CHKERRA(ierr)
      v = abs(norm-2.0*sqrt(nfloat))
      if (v .gt. -tol .and. v .lt. tol) v = 0.0
      if (rank .eq. 0) write(6,170) v
 170  format ('VecSwap ',1pe9.2)

      call VecNorm(x,NORM_2,norm,ierr);CHKERRA(ierr)
      v = abs(norm-18.0*sqrt(nfloat))
      if (v .gt. -tol .and. v .lt. tol) v = 0.0
      if (rank .eq. 0) write(6,180) v
 180  format ('VecSwap ',1pe9.2)

      call VecWAXPY(w,two,x,y,ierr);CHKERRA(ierr)
      call VecNorm(w,NORM_2,norm,ierr);CHKERRA(ierr)
      v = abs(norm-38.0*sqrt(nfloat))
      if (v .gt. -tol .and. v .lt. tol) v = 0.0
      if (rank .eq. 0) write(6,190) v
 190  format ('VecWAXPY ',1pe9.2)

      call VecPointwiseMult(w,y,x,ierr);CHKERRA(ierr)
      call VecNorm(w,NORM_2,norm,ierr);CHKERRA(ierr)
      v = abs(norm-36.0*sqrt(nfloat))
      if (v .gt. -tol .and. v .lt. tol) v = 0.0
      if (rank .eq. 0) write(6,200) v
 200  format ('VecPointwiseMult ',1pe9.2)

      call VecPointwiseDivide(w,x,y,ierr);CHKERRA(ierr)
      call VecNorm(w,NORM_2,norm,ierr);CHKERRA(ierr)
      v = abs(norm-9.0*sqrt(nfloat))
      if (v .gt. -tol .and. v .lt. tol) v = 0.0
      if (rank .eq. 0) write(6,210) v
 210  format ('VecPointwiseDivide ',1pe9.2)

      dots(1) = one
      dots(2) = three
      dots(3) = two
      call VecSet(x,one,ierr);CHKERRA(ierr)
      call VecMAXPY(x,ithree,dots,z,ierr);CHKERRA(ierr)
      call VecNorm(z(1),NORM_2,norm,ierr);CHKERRA(ierr)
      v = abs(norm-sqrt(nfloat))
      if (v .gt. -tol .and. v .lt. tol) v = 0.0
      call VecNorm(z(2),NORM_2,norm,ierr);CHKERRA(ierr)
      v1 = abs(norm-2.0*sqrt(nfloat))
      if (v1 .gt. -tol .and. v1 .lt. tol) v1 = 0.0
      call VecNorm(z(3),NORM_2,norm,ierr);CHKERRA(ierr)
      v2 = abs(norm-3.0*sqrt(nfloat))
      if (v2 .gt. -tol .and. v2 .lt. tol) v2 = 0.0
      if (rank .eq. 0) write(6,220) v,v1,v2
 220  format ('VecMAXPY ',3(1pe9.2))

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.

      call VecDestroy(x,ierr);CHKERRA(ierr)
      call VecDestroy(y,ierr);CHKERRA(ierr)
      call VecDestroy(w,ierr);CHKERRA(ierr)
      call VecDestroyVecsF90(ithree,z,ierr);CHKERRA(ierr)
      call PetscFinalize(ierr)

      end

!/*TEST
!
!     test:
!
!TEST*/
