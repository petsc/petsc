        module petscmpi
#include <petscconf.h>
#include "petsc/finclude/petscsys.h"
#if defined(PETSC_HAVE_MPIUNI)
        use mpiuni
#else
#if defined(PETSC_HAVE_MPI_F90MODULE)
        use mpi
#else
#include "mpif.h"
#endif
#endif

        public:: MPIU_REAL, MPIU_SUM, MPIU_SCALAR, MPIU_INTEGER
        public:: PETSC_COMM_WORLD, PETSC_COMM_SELF

! ----------------------------------------------------------------------------
!    BEGIN PETSc aliases for MPI_ constants
!
!   These values for __float128 are handled in the common block (below)
!     and transmitted from the C code
!
      integer4 :: MPIU_REAL
      integer4 :: MPIU_SUM
      integer4 :: MPIU_SCALAR
      integer4 :: MPIU_INTEGER

      MPI_Comm::PETSC_COMM_WORLD=0
      MPI_Comm::PETSC_COMM_SELF=0

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::MPIU_REAL
!DEC$ ATTRIBUTES DLLEXPORT::MPIU_SUM
!DEC$ ATTRIBUTES DLLEXPORT::MPIU_SCALAR
!DEC$ ATTRIBUTES DLLEXPORT::MPIU_INTEGER
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_COMM_SELF
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_COMM_WORLD
#endif
      end module

        module petscsysdefdummy
#if defined(PETSC_HAVE_MPI_F90MODULE_VISIBILITY)
        use petscmpi
#else
        use petscmpi, only: MPIU_REAL,MPIU_SUM,MPIU_SCALAR,MPIU_INTEGER,PETSC_COMM_WORLD,PETSC_COMM_SELF
#endif
#include <../src/sys/f90-mod/petscsys.h>
#include <../src/sys/f90-mod/petscdraw.h>
#include <../src/sys/f90-mod/petscviewer.h>
#include <../src/sys/f90-mod/petscbag.h>
#include <../src/sys/f90-mod/petscerror.h>
#include <../src/sys/f90-mod/petsclog.h>
        end module petscsysdefdummy

        module petscsysdef
        use petscsysdefdummy
        interface operator(.ne.)
          function petscviewernotequal(A,B)
            import tPetscViewer
            logical petscviewernotequal
            type(tPetscViewer), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator(.eq.)
          function petscviewerequals(A,B)
            import tPetscViewer
            logical petscviewerequals
            type(tPetscViewer), intent(in) :: A,B
          end function
        end interface operator (.eq.)

        interface operator(.ne.)
        function petscrandomnotequal(A,B)
          import tPetscRandom
          logical petscrandomnotequal
          type(tPetscRandom), intent(in) :: A,B
        end function
        end interface operator (.ne.)
        interface operator(.eq.)
        function petscrandomequals(A,B)
          import tPetscRandom
          logical petscrandomequals
          type(tPetscRandom), intent(in) :: A,B
        end function
        end interface operator (.eq.)

        Interface petscbinaryread
        subroutine petscbinaryreadcomplex(fd,data,num,count,type,z)
          integer fd
          PetscComplex data(*)
          PetscInt num
          PetscInt count
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinaryreadreal(fd,data,num,count,type,z)
          integer fd
          PetscReal data(*)
          PetscInt num
          PetscInt count
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinaryreadint(fd,data,num,count,type,z)
          integer fd
          PetscInt data(*)
          PetscInt num
          PetscInt count
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinaryreadcomplex1(fd,data,num,count,type,z)
          integer fd
          PetscComplex data
          PetscInt num
          PetscInt count
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinaryreadreal1(fd,data,num,count,type,z)
          integer fd
          PetscReal data
          PetscInt num
          PetscInt count
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinaryreadint1(fd,data,num,count,type,z)
          integer fd
          PetscInt data
          PetscInt num
          PetscInt count
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinaryreadcomplexcnt(fd,data,num,count,type,z)
          integer fd
          PetscComplex data(*)
          PetscInt num
          PetscInt count(1)
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinaryreadrealcnt(fd,data,num,count,type,z)
          integer fd
          PetscReal data(*)
          PetscInt num
          PetscInt count(1)
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinaryreadintcnt(fd,data,num,count,type,z)
          integer fd
          PetscInt data(*)
          PetscInt num
          PetscInt count(1)
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinaryreadcomplex1cnt(fd,data,num,count,type,z)
          integer fd
          PetscComplex data
          PetscInt num
          PetscInt count(1)
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinaryreadreal1cnt(fd,data,num,count,type,z)
          integer fd
          PetscReal data
          PetscInt num
          PetscInt count(1)
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinaryreadint1cnt(fd,data,num,count,type,z)
          integer fd
          PetscInt data
          PetscInt num
          PetscInt count(1)
          PetscDataType type
          PetscErrorCode z
        end subroutine
        end Interface

        Interface petscbinarywrite
        subroutine petscbinarywritecomplex(fd,data,num,type,z)
          integer fd
          PetscComplex data(*)
          PetscInt num
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinarywritereal(fd,data,num,type,z)
          integer fd
          PetscReal data(*)
          PetscInt num
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinarywriteint(fd,data,num,type,z)
          integer fd
          PetscInt data(*)
          PetscInt num
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinarywritecomplex1(fd,data,num,type,z)
          integer fd
          PetscComplex data
          PetscInt num
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinarywritereal1(fd,data,num,type,z)
          integer fd
          PetscReal data
          PetscInt num
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinarywriteint1(fd,data,num,type,z)
          integer fd
          PetscInt data
          PetscInt num
          PetscDataType type
          PetscErrorCode z
          end subroutine
        end Interface

        Interface petscintview
        subroutine petscintview(N,idx,viewer,ierr)
          use petscsysdefdummy, only: tPetscViewer
          PetscInt N
          PetscInt idx(*)
          PetscViewer viewer
          PetscErrorCode ierr
        end subroutine
        end Interface

        Interface petscscalarview
        subroutine petscscalarview(N,s,viewer,ierr)
          use petscsysdefdummy, only: tPetscViewer
          PetscInt N
          PetscScalar s(*)
          PetscViewer viewer
          PetscErrorCode ierr
        end subroutine
        end Interface

        Interface petscrealview
        subroutine petscrealview(N,s,viewer,ierr)
          use petscsysdefdummy, only: tPetscViewer
          PetscInt N
          PetscReal s(*)
          PetscViewer viewer
          PetscErrorCode ierr
        end subroutine
        end Interface

        end module

        function petscviewernotequal(A,B)
          use petscsysdefdummy, only: tPetscViewer
          logical petscviewernotequal
          type(tPetscViewer), intent(in) :: A,B
          petscviewernotequal = (A%v .ne. B%v)
        end function
        function petscviewerequals(A,B)
          use petscsysdefdummy, only: tPetscViewer
          logical petscviewerequals
          type(tPetscViewer), intent(in) :: A,B
          petscviewerequals = (A%v .eq. B%v)
        end function

        function petscrandomnotequal(A,B)
          use petscsysdefdummy, only: tPetscRandom
          logical petscrandomnotequal
          type(tPetscRandom), intent(in) :: A,B
          petscrandomnotequal = (A%v .ne. B%v)
        end function
        function petscrandomequals(A,B)
          use petscsysdefdummy, only: tPetscRandom
          logical petscrandomequals
          type(tPetscRandom), intent(in) :: A,B
          petscrandomequals = (A%v .eq. B%v)
        end function
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::petscviewernotequal
!DEC$ ATTRIBUTES DLLEXPORT::petscviewerequals
!DEC$ ATTRIBUTES DLLEXPORT::petscrandomnotequal
!DEC$ ATTRIBUTES DLLEXPORT::petscrandomequals
#endif
        module petscsys
        use,intrinsic :: iso_c_binding
        use petscsysdef
        PetscChar(80) PETSC_NULL_CHARACTER = ''
        PetscInt PETSC_NULL_INTEGER(1)
        PetscFortranDouble PETSC_NULL_DOUBLE(1)
        PetscScalar PETSC_NULL_SCALAR(1)
        PetscReal PETSC_NULL_REAL(1)
        PetscBool PETSC_NULL_BOOL
        MPI_Comm  PETSC_NULL_MPI_COMM(1)
!
!
!
!
!     Basic math constants
!
        PetscReal PETSC_PI
        PetscReal PETSC_MAX_REAL
        PetscReal PETSC_MIN_REAL
        PetscReal PETSC_MACHINE_EPSILON
        PetscReal PETSC_SQRT_MACHINE_EPSILON
        PetscReal PETSC_SMALL
        PetscReal PETSC_INFINITY
        PetscReal PETSC_NINFINITY

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_CHARACTER
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_INTEGER
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_DOUBLE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_SCALAR
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_REAL
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_BOOL
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_MPI_COMM
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_PI
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_MAX_REAL
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_MIN_REAL
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_MACHINE_EPSILON
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_SQRT_MACHINE_EPSILON
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_SMALL
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_INFINITY
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NINFINITY
#endif

#include <../src/sys/f90-mod/petscsys.h90>
        interface
#include <../src/sys/f90-mod/ftn-auto-interfaces/petscsys.h90>
        end interface
        interface PetscInitialize
          module procedure PetscInitializeWithHelp, PetscInitializeNoHelp, PetscInitializeNoArguments
        end interface

      contains
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PetscInitializeWithHelp
#endif
      subroutine PetscInitializeWithHelp(filename,help,ierr)
          character(len=*)           :: filename
          character(len=*)           :: help
          PetscErrorCode             :: ierr

          if (filename .ne. PETSC_NULL_CHARACTER) then
             PetscCall(PetscInitializeF(trim(filename),help,PETSC_TRUE,ierr))
          else
             PetscCall(PetscInitializeF(filename,help,PETSC_TRUE,ierr))
          endif
        end subroutine PetscInitializeWithHelp

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PetscInitializeNoHelp
#endif
        subroutine PetscInitializeNoHelp(filename,ierr)
          character(len=*)           :: filename
          PetscErrorCode             :: ierr

          if (filename .ne. PETSC_NULL_CHARACTER) then
             PetscCall(PetscInitializeF(trim(filename),PETSC_NULL_CHARACTER,PETSC_TRUE,ierr))
          else
             PetscCall(PetscInitializeF(filename,PETSC_NULL_CHARACTER,PETSC_TRUE,ierr))
          endif
        end subroutine PetscInitializeNoHelp

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PetscInitializeNoArguments
#endif
        subroutine PetscInitializeNoArguments(ierr)
          PetscErrorCode             :: ierr

          PetscCall(PetscInitializeF(PETSC_NULL_CHARACTER,PETSC_NULL_CHARACTER,PETSC_TRUE,ierr))
        end subroutine PetscInitializeNoArguments
        end module

        subroutine PetscSetCOMM(c1,c2)
        use petscmpi, only: PETSC_COMM_WORLD,PETSC_COMM_SELF

        implicit none
        MPI_Comm c1,c2

        PETSC_COMM_WORLD    = c1
        PETSC_COMM_SELF     = c2
        return
        end

        subroutine PetscGetCOMM(c1)
        use petscmpi, only: PETSC_COMM_WORLD
        implicit none
        MPI_Comm c1

        c1 = PETSC_COMM_WORLD
        return
        end

        subroutine PetscSetModuleBlock()
        use petscsys, only: PETSC_NULL_CHARACTER,PETSC_NULL_INTEGER,&
             PETSC_NULL_SCALAR,PETSC_NULL_DOUBLE,PETSC_NULL_REAL,&
             PETSC_NULL_BOOL,PETSC_NULL_FUNCTION,PETSC_NULL_MPI_COMM
        implicit none

        call PetscSetFortranBasePointers(PETSC_NULL_CHARACTER,            &
     &     PETSC_NULL_INTEGER,PETSC_NULL_SCALAR,                        &
     &     PETSC_NULL_DOUBLE,PETSC_NULL_REAL,                           &
     &     PETSC_NULL_BOOL,PETSC_NULL_FUNCTION,PETSC_NULL_MPI_COMM)

        return
        end

        subroutine PetscSetModuleBlockMPI(freal,fscalar,fsum,finteger)
        use petscmpi, only: MPIU_REAL,MPIU_SUM,MPIU_SCALAR,MPIU_INTEGER
        implicit none

        integer4 freal,fscalar,fsum,finteger

        MPIU_REAL    = freal
        MPIU_SCALAR  = fscalar
        MPIU_SUM     = fsum
        MPIU_INTEGER = finteger

        return
        end

        subroutine PetscSetModuleBlockNumeric(pi,maxreal,minreal,eps,       &
     &     seps,small,pinf,pninf)
        use petscsys, only: PETSC_PI,PETSC_MAX_REAL,PETSC_MIN_REAL,&
             PETSC_MACHINE_EPSILON,PETSC_SQRT_MACHINE_EPSILON,&
             PETSC_SMALL,PETSC_INFINITY,PETSC_NINFINITY
        implicit none

        PetscReal pi,maxreal,minreal,eps,seps
        PetscReal small,pinf,pninf

        PETSC_PI = pi
        PETSC_MAX_REAL = maxreal
        PETSC_MIN_REAL = minreal
        PETSC_MACHINE_EPSILON = eps
        PETSC_SQRT_MACHINE_EPSILON = seps
        PETSC_SMALL = small
        PETSC_INFINITY = pinf
        PETSC_NINFINITY = pninf

        return
        end

