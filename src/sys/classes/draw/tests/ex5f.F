!
!
      program main
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscdraw.h>
      use petscsys
      implicit none
!
!  This example demonstrates basic use of the Fortran interface for
!  PetscDraw routines.
!
      PetscDraw         draw
      PetscDrawLG       lg
      PetscDrawAxis     axis
      PetscErrorCode    ierr
      PetscBool         flg
      integer           x,y,width,height
      PetscScalar       xd,yd
      PetscReal         ten
      PetscInt          i,n,w,h
      PetscInt          one

      n      = 15
      x      = 0
      y      = 0
      w      = 400
      h      = 300
      ten    = 10.0
      one    = 1

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
         print*,'Unable to initialize PETSc'
         stop
      endif

!  GetInt requires a PetscInt so have to do this ugly setting
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,         &
     &                        '-width',w, flg,ierr)
      width = int(w)
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,         &
     &                        '-height',h,flg,ierr)
      height = int(h)
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,         &
     &                        '-n',n,flg,ierr)

      call PetscDrawCreate(PETSC_COMM_WORLD,PETSC_NULL_CHARACTER,        &
     &               PETSC_NULL_CHARACTER,x,y,width,height,draw,ierr)
      call PetscDrawSetFromOptions(draw,ierr)

      call PetscDrawLGCreate(draw,one,lg,ierr)
      call PetscDrawLGGetAxis(lg,axis,ierr)
      call PetscDrawAxisSetColors(axis,PETSC_DRAW_BLACK,PETSC_DRAW_RED, &
     &     PETSC_DRAW_BLUE,ierr)
      call PetscDrawAxisSetLabels(axis,'toplabel','xlabel','ylabel',    &
     &     ierr)

      do 10, i=0,n-1
        xd = real(i) - 5.0
        yd = xd*xd
        call PetscDrawLGAddPoint(lg,xd,yd,ierr)
 10   continue

      call PetscDrawLGSetUseMarkers(lg,PETSC_TRUE,ierr)
      call PetscDrawLGDraw(lg,ierr)

      call PetscSleep(ten,ierr)

      call PetscDrawLGDestroy(lg,ierr)
      call PetscDrawDestroy(draw,ierr)
      call PetscFinalize(ierr)
      end

!/*TEST
!
!   build:
!     requires: x
!
!   test:
!     output_file: output/ex1_1.out
!
!TEST*/
