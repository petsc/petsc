
C    "$Id: ex1.c,v 1.32 1996/10/23 22:13:25 bsmith Exp curfman $"
C
C/*T
C   Concepts: Vectors^Using basic vector routines;
C   Routines: VecCreate(); VecDuplicate(); VecSet(); VecValid(); VecDot();
C   Routines: VecMDot(); VecScale(); VecNorm(); VecCopy(); VecAXPY(); VecAYPX();
C   Routines: VecWAXPY(); VecPointwiseMult(); VecPointwiseDivide(); VecSwap();
C   Routines: VecDestroy(); VecDestroyVecs();
C   Processors: n
CT*/
C
C -----------------------------------------------------------------------

      program ex1f
      implicit none

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                    Include files
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C
C  The following include statements are required for Fortran programs
C  that use PETSc vectors:
C     petsc.h  - base PETSc routines
C     vec.h    - vectors
C  Additional include statements may be needed if using additional
C  PETSc routines in a Fortran program, e.g.,
C     viewer.h - viewers
C     is.h     - index sets
C
#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
C
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                   Variable declarations
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C
C  Variables:
C     x, y, w - vectors
C     z       - array of vectors
C
C  Note:  "Double" -> "double precision" except for machines such
C          as the Cray T3d, where "Double" -> "real"

      Vec     x, y, w, z(3)
      Double  norm, v
      integer n, ierr, flg, rank
      Scalar  one, two, three, dots(3), dot

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                 Beginning of program
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      one   = 1.0
      two   = 2.0
      three = 3.0
      n     = 20
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-n',n,flg,ierr)
      call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)

C  Create a vector, specifying only its global dimension.
C  When using VecCreate(), the vector format (currently parallel
C  or sequential) is determined at runtime.  Also, the parallel
C  partitioning of the vector is determined by PETSc at runtime.
C
C  Routines for creating particular vector types directly are:
C     VecCreateSeq() - uniprocessor vector
C     VecCreateMPI() - distributed vector, where the user can
C                      determine the parallel partitioning

      call VecCreate(PETSC_COMM_WORLD,n,x,ierr)

C  Duplicate some work vectors (of the same format and
C  partitioning as the initial vector).

      call VecDuplicate(x,y,ierr)
      call VecDuplicate(x,w,ierr)

C  Duplicate more work vectors (of the same format and
C  partitioning as the initial vector).  Here we duplicate
C  an array of vectors, which is often more convenient than
C  duplicating individual ones.

      call VecDuplicateVecs(x,3,z,ierr)

C  Set the vectors to entries to a constant value.

      call VecSet(one,x,ierr)
      call VecSet(two,y,ierr)
      call VecSet(one,z(1),ierr)
      call VecSet(two,z(2),ierr)
      call VecSet(three,z(3),ierr)

C  Demonstrate various basic vector routines.

      call VecDot(x,x,dot,ierr)
      call VecMDot(3,x,z,dots,ierr)

C  Note: If using a complex numbers version of PETSc, then
C  PETSC_COMPLEX is defined in the makefiles; otherwise,
C  (when using real numbers) it is undefined.

      if (rank .eq. 0) then
#if defined(PETSC_COMPLEX)
         write(6,100) int(real(dot))
         write(6,110) int(real(dots(1))),int(real(dots(2))),
     &                int(real(dots(3)))
#else
         write(6,100) int(dot)
         write(6,110) int(dots(1)),int(dots(2)),int(dots(3))
#endif
         write(6,120)
      endif
 100  format ("Vector length ",i6)
 110  format ("Vector length ",3(i6))
 120  format ("All other values should be near zero")

      call VecScale(two,x,ierr)
      call VecNorm(x,NORM_2,norm,ierr)
      v = norm-2.0*sqrt(dble(n))
      if (v .gt. -1.d-10 .and. v .lt. 1.d-10) v = 0.0
      write(6,130) v
 130  format ("VecScale ",e8.2)

      call VecCopy(x,w,ierr)
      call VecNorm(w,NORM_2,norm,ierr)
      v = norm-2.0*sqrt(dble(n))
      if (v .gt. -1.d-10 .and. v .lt. 1.d-10) v = 0.0
      write(6,140) v
 140  format ("VecCopy ",e8.2)

      call VecAXPY(three,x,y,ierr)
      call VecNorm(y,NORM_2,norm,ierr)
      v = norm-8.0*sqrt(dble(n))
      if (v .gt. -1.d-10 .and. v .lt. 1.d-10) v = 0.0
      write(6,150) v
 150  format ("VecAXPY ",e8.2)

      call VecAYPX(two,x,y,ierr)
      call VecNorm(y,NORM_2,norm,ierr)
      v = norm-18.0*sqrt(dble(n))
      if (v .gt. -1.d-10 .and. v .lt. 1.d-10) v = 0.0
      write(6,160) v
 160  format ("VecAYXP ",e8.2)

      call VecSwap(x,y,ierr)
      call VecNorm(y,NORM_2,norm,ierr)
      v = norm-2.0*sqrt(dble(n))
      if (v .gt. -1.d-10 .and. v .lt. 1.d-10) v = 0.0
      write(6,170) v
 170  format ("VecSwap ",e8.2)

      call VecNorm(x,NORM_2,norm,ierr)
      v = norm-18.0*sqrt(dble(n))
      if (v .gt. -1.d-10 .and. v .lt. 1.d-10) v = 0.0
      write(6,180) v
 180  format ("VecSwap ",e8.2)

      call VecWAXPY(two,x,y,w,ierr)
      call VecNorm(w,NORM_2,norm,ierr)
      v = norm-38.0*sqrt(dble(n))
      if (v .gt. -1.d-10 .and. v .lt. 1.d-10) v = 0.0
      write(6,190) v
 190  format ("VecWAXPY ",e8.2)

      call VecPointwiseMult(y,x,w,ierr)
      call VecNorm(w,NORM_2,norm,ierr)
      v = norm-36.0*sqrt(dble(n))
      if (v .gt. -1.d-10 .and. v .lt. 1.d-10) v = 0.0
      write(6,200) v
 200  format ("VecPointwiseMult ",e8.2)

      call VecPointwiseDivide(x,y,w,ierr)
      call VecNorm(w,NORM_2,norm,ierr)
      v = norm-9.0*sqrt(dble(n))
      if (v .gt. -1.d-10 .and. v .lt. 1.d-10) v = 0.0
      write(6,210) v
 210  format ("VecPointwiseDivide ",e8.2)


C  Test whether vector has been corrupted (just to demonstrate this
C  routine) not needed in most application codes.

      call VecValid(x,flg,ierr)
      if (flg .ne. PETSC_TRUE) then
         if (rank .eq. 0)
     &      write(6,*) 'Corrupted vector!'
         SETERRA(1,0)
      endif

C  Free work space.  All PETSc objects should be destroyed when they
C  are no longer needed.

      call VecDestroy(x,ierr)
      call VecDestroy(y,ierr)
      call VecDestroy(w,ierr)
      call VecDestroyVecs(z,3,ierr)
      call PetscFinalize(ierr)

      end
 
