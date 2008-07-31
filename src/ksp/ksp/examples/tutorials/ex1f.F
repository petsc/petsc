
!#include "include/finclude/petscdef.h"

      program  flamD
     

      implicit none

#include "include/finclude/petsc.h"
#include "include/finclude/petscdraw.h"
#include "include/finclude/petscviewer.h"
#include "include/finclude/petscis.h"
#include "include/finclude/petscvec.h"
#include "include/finclude/petscmat.h"
#include "include/finclude/petscpc.h"
#include "include/finclude/petscksp.h"
#include "include/finclude/petscda.h"
#include "include/finclude/petscviewer.h90"
#include "include/finclude/petscis.h90"
#include "include/finclude/petscvec.h90"
#include "include/finclude/petscmat.h90"
#include "include/finclude/petscda.h90"
      save



      integer :: i,j,k

      integer :: ierr

      integer :: status

      integer :: nx
      integer :: ny
      integer :: nz
      integer :: dof1 
 
      integer :: domainsize
      integer :: rank

      DA          daa

      Vec         RHS
      
      integer :: xs,xm
      integer :: gxs,gxm

      integer :: ys,ym
      integer :: gys,gym

      integer :: zs,zm
      integer :: gzs,gzm

      integer :: overlap


      PetscScalar::zeroval=0. 

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
           CHKERRQ(ierr)


!-----------------------------------------------------------------------
!     initialization
!-----------------------------------------------------------------------    

      call MPI_Comm_size(PETSC_COMM_WORLD,domainsize,ierr)
           CHKERRQ(ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
           CHKERRQ(ierr)

!-----------------------------------------------------------------------

      nx=51
      ny=51
      nz=51
      dof1=1

      overlap=4

!-----------------------------------------------------------------------
!    Domain Decomposition
!-----------------------------------------------------------------------


      call DACreate3d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,
     &                nx,ny,nz,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,
     &                dof1,overlap, 
     &                PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,            
     &                PETSC_NULL_INTEGER,daa,ierr)  
           CHKERRQ(ierr) 

      call DAGetCorners(daa,xs,ys,zs,xm,ym,zm,ierr)
           CHKERRQ(ierr)

      gym=ny 
      gzm=nz 
      
      call DAGetGhostCorners(daa,gxs,gys,gzs,gxm,gym,gzm,ierr)
           CHKERRQ(ierr)

!------------------------------------------------------------------

      write(*,*)'rank=',rank,'gym=',gym,'gzm=',gzm

      do k=1,gzm
         do j=1,gym


            call VecCreate(PETSC_COMM_WORLD,RHS,ierr)
                 CHKERRQ(ierr)

            call VecSetSizes(RHS,PETSC_DECIDE,nx,ierr)
                 CHKERRQ(ierr)

            call VecSetFromOptions(RHS,ierr)
                 CHKERRQ(ierr)

            call VecDestroy(RHS,ierr)
                 CHKERRQ(ierr)
       print*,k,j

      enddo
      enddo


!------------------------------------------------------------------

      call DADestroy(daa,ierr)



      call PetscFinalize(ierr)
       
      end

!*********************************************************************


