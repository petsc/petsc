!
!
!  Tests PescOffsetFortran()
!  duplicated
      program main
       implicit none

#include <finclude/petscsys.h>
#include <finclude/petscis.h>
#include <finclude/petscvec.h>

      PetscErrorCode ierr
      PetscInt  n
      PetscMPIInt size

      PetscScalar  v_v1(1),v_v2(1)
      Vec     v
      PetscInt i
      PetscOffset i_v1,i_v2

      n=8
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)
      if (size .gt. 1) then
        print*,'Example for one processor only'
        call MPI_Abort(MPI_COMM_WORLD,0,ierr)
      endif

      call VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,n,v,ierr)
      call VecGetArray(v,v_v1,i_v1,ierr)

      do 10, i=1,n
        v_v1(i_v1 + i) = i
 10   continue
      call VecRestoreArray(v,v_v1,i_v1,ierr)

      call VecView(v,PETSC_VIEWER_STDOUT_WORLD,ierr)

      call VecGetArray(v,v_v1,i_v1,ierr)
      call PetscOffsetFortran(v_v2,v_v1,i_v2,ierr)
      i_v2 = i_v1 + i_v2
      do 20, i=1,n
        print*,i,v_v2(i_v2 + i)
 20   continue
      call VecRestoreArray(v,v_v1,i_v1,ierr)

      call VecDestroy(v,ierr)
      call PetscFinalize(ierr)

      end



