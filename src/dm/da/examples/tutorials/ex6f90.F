!-----------------------------------------------------------------------
!
!     manages tokamak edge region.
!
!     This is a translation of ex6.c into F90 by Alan Glasser, LANL
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
!     code organization.
!-----------------------------------------------------------------------
!     0. barry_mod.
!     1. barry_get_global_corners.
!     2. barry_get_global_vector.
!     3. barry_get_local_vector.
!     4. barry_global_to_local.
!     5. barry_destroy_fa.
!     6. barry_create_fa.
!     7. barry_draw_patch.
!     8. barry_draw_fa.
!     9. barry_map_region3.
!     10. barry_map_region2.
!     11. barry_map_region1.
!     12. barry_main.
!-----------------------------------------------------------------------
!     subprogram 0. barry_mod.
!     module declarations.
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
!     declarations.
!-----------------------------------------------------------------------
      MODULE barry_mod
      IMPLICIT NONE

#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscda.h"
#include "finclude/petscmat.h"
#include "finclude/petscis.h"
#include "finclude/petscao.h"
#include "finclude/petscviewer.h"
#include "finclude/petscdraw.h"

      TYPE :: fa_type
      MPI_Comm, DIMENSION(0:2) :: comm
      INTEGER, DIMENSION(0:2) :: xl,yl,ml,nl,xg,yg,mg,ng,offl,offg
      Vec :: g,l
      VecScatter :: vscat
      INTEGER :: p1,p2,r1,r2,r1g,r2g,sw
      END TYPE fa_type

      TYPE :: patch_type
      INTEGER :: mx,my
      PetscScalar, DIMENSION(:,:,:), POINTER :: xy
      END TYPE patch_type

      LOGICAL, PARAMETER :: diagnose=.FALSE.
      INTEGER :: ierr
      REAL(8), PARAMETER :: pi=3.1415926535897932385_8,twopi=2*pi

      CONTAINS
!-----------------------------------------------------------------------
!     subprogram 1. barry_get_global_corners.
!     returns global corner data.
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
!     declarations.
!-----------------------------------------------------------------------
      SUBROUTINE barry_get_global_corners(fa,j,x,y,m,n)

      TYPE(fa_type), INTENT(IN) :: fa
      INTEGER, INTENT(IN) :: j
      INTEGER, INTENT(OUT) :: x,y,m,n
!-----------------------------------------------------------------------
!     computations.
!-----------------------------------------------------------------------
      IF(fa%comm(j) /= 0)THEN
         x=fa%xg(j)
         y=fa%yg(j)
         m=fa%mg(j)
         n=fa%ng(j)
      ELSE
         x=0
         y=0
         m=0
         n=0
      ENDIF
!-----------------------------------------------------------------------
!     terminate.
!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE barry_get_global_corners
!-----------------------------------------------------------------------
!     subprogram 2. barry_get_global_vector.
!     returns local vector.
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
!     declarations.
!-----------------------------------------------------------------------
      SUBROUTINE barry_get_global_vector(fa,v)

      TYPE(fa_type), INTENT(IN) :: fa
      Vec, INTENT(OUT) :: v
!-----------------------------------------------------------------------
!     computations.
!-----------------------------------------------------------------------
      CALL VecDuplicate(fa%g,v,ierr)
!-----------------------------------------------------------------------
!     terminate.
!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE barry_get_global_vector
!-----------------------------------------------------------------------
!     subprogram 3. barry_get_local_vector.
!     returns local vector.
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
!     declarations.
!-----------------------------------------------------------------------
      SUBROUTINE barry_get_local_vector(fa,v)

      TYPE(fa_type), INTENT(IN) :: fa
      Vec, INTENT(OUT) :: v
!-----------------------------------------------------------------------
!     computations.
!-----------------------------------------------------------------------
      CALL VecDuplicate(fa%l,v,ierr)
!-----------------------------------------------------------------------
!     terminate.
!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE barry_get_local_vector
!-----------------------------------------------------------------------
!     subprogram 4. barry_global_to_local.
!     performs VecScatter.
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
!     declarations.
!-----------------------------------------------------------------------
      SUBROUTINE barry_global_to_local(fa,g,l)

      TYPE(fa_type), INTENT(IN) :: fa
      Vec, INTENT(IN) :: g
      Vec, INTENT(OUT) :: l
!-----------------------------------------------------------------------
!     computations.
!-----------------------------------------------------------------------
      CALL VecScatterBegin(fa%vscat,g,l,INSERT_VALUES,SCATTER_FORWARD,     &
     &     ierr)
      CALL VecScatterEnd(fa%vscat,g,l,INSERT_VALUES,SCATTER_FORWARD,       &
     &     ierr)
!-----------------------------------------------------------------------
!     terminate.
!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE barry_global_to_local
!-----------------------------------------------------------------------
!     subprogram 5. barry_destroy_fa.
!     destroys fa.
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
!     declarations.
!-----------------------------------------------------------------------
      SUBROUTINE barry_destroy_fa(fa)

      TYPE(fa_type), INTENT(OUT) :: fa
!-----------------------------------------------------------------------
!     computations.
!-----------------------------------------------------------------------
      CALL VecDestroy(fa%g,ierr)
      CALL VecDestroy(fa%l,ierr)
      CALL VecScatterDestroy(fa%vscat,ierr)
!-----------------------------------------------------------------------
!     terminate.
!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE barry_destroy_fa
!-----------------------------------------------------------------------
!     subprogram 6. barry_create_fa.
!     creates fa.
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
!     declarations.
!-----------------------------------------------------------------------
      SUBROUTINE barry_create_fa(fa)

      TYPE(fa_type), INTENT(OUT) :: fa

      INTEGER :: rank,tonglobal,globalrstart,x,nx,y,ny,i,j,k,ig,              &
     &     fromnglobal,nscat,nlocal,cntl1,cntl2,cntl3,il,it
      INTEGER, DIMENSION(0:2) :: offt
      INTEGER, DIMENSION(:), POINTER :: tonatural,fromnatural,                &
     &     to,from,indices
      PetscScalar, DIMENSION(1) :: globalarray
      PetscScalar, DIMENSION(1) :: localarray
      PetscScalar, DIMENSION(1) :: toarray

      DA :: da1=0,da2=0,da3=0
      Vec :: vl1=0,vl2=0,vl3=0
      Vec :: vg1=0,vg2=0,vg3=0
      AO :: toao,globalao
      IS :: tois,globalis,is
      Vec :: tovec,globalvec,localvec
      VecScatter :: vscat
!-----------------------------------------------------------------------
!     set integer members of fa.
!-----------------------------------------------------------------------
      fa%p1=24
      fa%p2=15
      fa%r1=6 
      fa%r2=6
      fa%sw=1
      fa%r1g=fa%r1+fa%sw
      fa%r2g=fa%r2+fa%sw
!-----------------------------------------------------------------------
!     set up communicators.
!-----------------------------------------------------------------------
      CALL MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      fa%comm=PETSC_COMM_WORLD
!-----------------------------------------------------------------------
!     set up distributed arrays.
!-----------------------------------------------------------------------
      IF(fa%comm(1) /= 0)THEN
         CALL DACreate2d(fa%comm(1),DA_XPERIODIC,DA_STENCIL_BOX,             &
     &        fa%p2,fa%r2g,PETSC_DECIDE,PETSC_DECIDE,1,fa%sw,                &
     &        PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,da2,ierr)
         CALL DAGetLocalVector(da2,vl2,ierr)
         CALL DAGetGlobalVector(da2,vg2,ierr)
      ENDIF
      IF(fa%comm(2) /= 0)THEN
         CALL DACreate2d(fa%comm(2),DA_NONPERIODIC,DA_STENCIL_BOX,           &
     &        fa%p1-fa%p2,fa%r2g,PETSC_DECIDE,PETSC_DECIDE,1,fa%sw,          &
     &        PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,da3,ierr)
         CALL DAGetLocalVector(da3,vl3,ierr)
         CALL DAGetGlobalVector(da3,vg3,ierr)
      ENDIF
      IF(fa%comm(0) /= 0)THEN
         CALL DACreate2d(fa%comm(0),DA_NONPERIODIC,DA_STENCIL_BOX,           &
     &        fa%p1,fa%r1g,PETSC_DECIDE,PETSC_DECIDE,1,fa%sw,                &
     &        PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,da1,ierr)
         CALL DAGetLocalVector(da1,vl1,ierr)
         CALL DAGetGlobalVector(da1,vg1,ierr)
      ENDIF
!-----------------------------------------------------------------------
!     count the number of unknowns owned on each processor and determine
!     the starting point of each processors ownership 
!     for global vector with redundancy.
!-----------------------------------------------------------------------
      tonglobal = 0
      IF(fa%comm(1) /= 0)THEN
         CALL DAGetCorners(da2,x,y,0,nx,ny,0,ierr)
         tonglobal=tonglobal+nx*ny
      ENDIF
      IF(fa%comm(2) /= 0)THEN
         CALL DAGetCorners(da3,x,y,0,nx,ny,0,ierr)
         tonglobal=tonglobal+nx*ny
      ENDIF
      IF(fa%comm(0) /= 0)THEN
         CALL DAGetCorners(da1,x,y,0,nx,ny,0,ierr)
         tonglobal=tonglobal+nx*ny
      ENDIF
      WRITE(*,'("[",i1,"]",a,i3)')                                               &
     &     rank," Number of unknowns owned ",tonglobal
!-----------------------------------------------------------------------
!     get tonatural number for each node
!-----------------------------------------------------------------------
      ALLOCATE(tonatural(0:tonglobal))
      tonglobal=0
      IF(fa%comm(1) /= 0)THEN
         CALL DAGetCorners(da2,x,y,0,nx,ny,0,ierr)
         DO j=0,ny-1
            DO i=0,nx-1
               tonatural(tonglobal)=(fa%p1-fa%p2)/2+x+i+fa%p1*(y+j)
               tonglobal=tonglobal+1
            ENDDO
         ENDDO
      ENDIF
      IF(fa%comm(2) /= 0)THEN
         CALL DAGetCorners(da3,x,y,0,nx,ny,0,ierr)
         DO j=0,ny-1
            DO i=0,nx-1
               IF(x+i < (fa%p1-fa%p2)/2)THEN
                  tonatural(tonglobal)=x+i+fa%p1*(y+j)
               ELSE
                  tonatural(tonglobal)=fa%p2+x+i+fa%p1*(y+j)
               ENDIF
               tonglobal=tonglobal+1
            ENDDO
         ENDDO
      ENDIF
      IF(fa%comm(0) /= 0)THEN
         CALL DAGetCorners(da1,x,y,0,nx,ny,0,ierr)
         DO j=0,ny-1
            DO i=0,nx-1
               tonatural(tonglobal)=fa%p1*fa%r2g+x+i+fa%p1*(y+j)
               tonglobal=tonglobal+1
            ENDDO
         ENDDO
      ENDIF
!-----------------------------------------------------------------------
!     diagnose tonatural.
!-----------------------------------------------------------------------
      IF(diagnose)THEN
         WRITE(*,'(a,i3,a)')"tonglobal = ",tonglobal,", tonatural = "
         WRITE(*,'(2i4)')(i,tonatural(i),i=0,tonglobal-1)
      ENDIF
!-----------------------------------------------------------------------
!     create application ordering and deallocate tonatural.
!-----------------------------------------------------------------------
      CALL AOCreateBasic(PETSC_COMM_WORLD,tonglobal,tonatural,                   &
     &     PETSC_NULL_INTEGER,toao,ierr)
      DEALLOCATE(tonatural)
!-----------------------------------------------------------------------
!     count the number of unknowns owned on each processor and determine
!     the starting point of each processors ownership for global vector
!     without redundancy.
!-----------------------------------------------------------------------
      fromnglobal=0
      fa%offg(1)=0
      offt(1)=0
      IF(fa%comm(1) /= 0)THEN
         CALL DAGetCorners(da2,x,y,0,nx,ny,0,ierr)
         offt(2)=nx*ny
         IF(y+ny == fa%r2g)ny=ny-1
         fromnglobal=fromnglobal+nx*ny
         fa%offg(2)=fromnglobal
      ELSE
         offt(2)=0
         fa%offg(2)=0
      ENDIF
      IF(fa%comm(2) /= 0)THEN
         CALL DAGetCorners(da3,x,y,0,nx,ny,0,ierr)
         offt(0)=offt(2)+nx*ny
         IF(y+ny == fa%r2g)ny=ny-1
         fromnglobal=fromnglobal+nx*ny
         fa%offg(0)=fromnglobal
      ELSE
         offt(0)=offt(2)
         fa%offg(0)=fromnglobal
      ENDIF
      IF(fa%comm(0) /= 0)THEN
         CALL DAGetCorners(da1,x,y,0,nx,ny,0,ierr)
         IF(y == 0)ny=ny-1
         fromnglobal=fromnglobal+nx*ny
      ENDIF
      CALL MPI_Scan(fromnglobal,globalrstart,1,MPI_INTEGER,                      &
     &     MPI_SUM,PETSC_COMM_WORLD,ierr)
      globalrstart=globalrstart-fromnglobal
      WRITE(*,'("[",i1,"]",a,i3)')                                               &
     &     rank," Number of unknowns owned ",fromnglobal
!-----------------------------------------------------------------------
!     get fromnatural number for each node.
!-----------------------------------------------------------------------
      ALLOCATE(fromnatural(0:fromnglobal))
      fromnglobal=0
      IF(fa%comm(1) /= 0)THEN
         CALL DAGetCorners(da2,x,y,0,nx,ny,0,ierr)
         IF(y+ny == fa%r2g)ny=ny-1
         fa%xg(1)=x
         fa%yg(1)=y
         fa%mg(1)=nx
         fa%ng(1)=ny
         CALL DAGetGhostCorners(da2,fa%xl(1),fa%yl(1),0,fa%ml(1),                &
     &        fa%nl(1),0,ierr)
         DO j=0,ny-1
            DO i=0,nx-1
               fromnatural(fromnglobal)                                          &
     &              =(fa%p1-fa%p2)/2+x+i+fa%p1*(y+j)
               fromnglobal=fromnglobal+1
            ENDDO
         ENDDO
      ENDIF
      IF(fa%comm(2) /= 0)THEN
         CALL DAGetCorners(da3,x,y,0,nx,ny,0,ierr)
         IF(y+ny == fa%r2g)ny=ny-1
         fa%xg(2)=x
         fa%yg(2)=y
         fa%mg(2)=nx
         fa%ng(2)=ny
         CALL DAGetGhostCorners(da3,fa%xl(2),fa%yl(2),0,fa%ml(2),                &
     &        fa%nl(2),0,ierr)
         DO j=0,ny-1
            DO i=0,nx-1
               IF(x+i < (fa%p1-fa%p2)/2)THEN
                  fromnatural(fromnglobal)=x+i+fa%p1*(y+j)
               ELSE
                  fromnatural(fromnglobal)=fa%p2+x+i+fa%p1*(y+j)
               ENDIF
               fromnglobal=fromnglobal+1
            ENDDO
         ENDDO
      ENDIF
      IF(fa%comm(0) /= 0)THEN
         CALL DAGetCorners(da1,x,y,0,nx,ny,0,ierr)
         IF(y == 0)THEN
            ny=ny-1
         ELSE 
            y=y-1
         ENDIF
         fa%xg(0)=x
         fa%yg(0)=y
         fa%mg(0)=nx
         fa%ng(0)=ny
         CALL DAGetGhostCorners(da1,fa%xl(0),fa%yl(0),0,fa%ml(0),               &
     &        fa%nl(0),0,ierr)
         DO j=0,ny-1
            DO i=0,nx-1
               fromnatural(fromnglobal)=fa%p1*fa%r2+x+i+fa%p1*(y+j)
               fromnglobal=fromnglobal+1
            ENDDO
         ENDDO
      ENDIF
!-----------------------------------------------------------------------
!     diagnose fromnatural.
!-----------------------------------------------------------------------
      IF(diagnose)THEN
         WRITE(*,'(a,i3,a)')"fromnglobal = ",fromnglobal                        &
     &        ,", fromnatural = "
         WRITE(*,'(2i4)')(i,fromnatural(i),i=0,fromnglobal-1)
      ENDIF
!-----------------------------------------------------------------------
!     create application ordering and deallocate fromnatural.
!-----------------------------------------------------------------------
      CALL AOCreateBasic(PETSC_COMM_WORLD,fromnglobal,fromnatural,              &
     &     PETSC_NULL_INTEGER,globalao,ierr)
      DEALLOCATE(fromnatural)
!-----------------------------------------------------------------------
!     set up scatter that updates 1 from 2 and 3 and 3 and 2 from 1
!-----------------------------------------------------------------------
      ALLOCATE(to(0:tonglobal),from(0:tonglobal))
      nscat=0
      IF(fa%comm(1) /= 0)THEN
         CALL DAGetCorners(da2,x,y,0,nx,ny,0,ierr)
         DO j=0,ny-1
            DO i=0,nx-1
               to(nscat)=(fa%p1-fa%p2)/2+x+i+fa%p1*(y+j)
               from(nscat)=to(nscat)
               nscat=nscat+1
            ENDDO
         ENDDO
      ENDIF
      IF(fa%comm(2) /= 0)THEN
         CALL DAGetCorners(da3,x,y,0,nx,ny,0,ierr)
         DO j=0,ny-1
            DO i=0,nx-1
               IF(x+i < (fa%p1-fa%p2)/2)THEN
                  to(nscat)=x+i+fa%p1*(y+j)
               ELSE
                  to(nscat)=fa%p2+x+i+fa%p1*(y+j)
               ENDIF
               from(nscat)=to(nscat)
               nscat=nscat+1
            ENDDO
         ENDDO
      ENDIF
      IF(fa%comm(0) /= 0)THEN
         CALL DAGetCorners(da1,x,y,0,nx,ny,0,ierr)
         DO j=0,ny-1
            DO i=0,nx-1
               to(nscat)=fa%p1*fa%r2g+x+i+fa%p1*(y+j)
               from(nscat)=fa%p1*(fa%r2-1)+x+i+fa%p1*(y+j)
               nscat=nscat+1
            ENDDO
         ENDDO
      ENDIF
!-----------------------------------------------------------------------
!     diagnose to and from.
!-----------------------------------------------------------------------
      IF(diagnose)THEN
         WRITE(*,'(a,i3,a)')"nscat = ",nscat,", to, from = "
         WRITE(*,'(3i4)')(i,to(i),from(i),i=0,nscat-1)
      ENDIF
!-----------------------------------------------------------------------
!     create vecscatter.
!-----------------------------------------------------------------------
      CALL AOApplicationToPetsc(toao,nscat,to,ierr)
      CALL AOApplicationToPetsc(globalao,nscat,from,ierr)
      CALL ISCreateGeneral(PETSC_COMM_WORLD,nscat,to,tois,ierr)
      CALL ISCreateGeneral(PETSC_COMM_WORLD,nscat,from,globalis,ierr)
      CALL VecCreateMPI(PETSC_COMM_WORLD,tonglobal,PETSC_DETERMINE,                   &
     &     tovec,ierr)
      CALL VecCreateMPI(PETSC_COMM_WORLD,fromnglobal,PETSC_DETERMINE,                 &
     &     globalvec,ierr)
      CALL VecScatterCreate(globalvec,globalis,tovec,tois,vscat,ierr)
!-----------------------------------------------------------------------
!     clean up.
!-----------------------------------------------------------------------
      CALL ISDestroy(tois,ierr)
      CALL ISDestroy(globalis,ierr)
      CALL AODestroy(globalao,ierr)
      CALL AODestroy(toao,ierr)
      DEALLOCATE(to,from)
!-----------------------------------------------------------------------
!     fill up global vector without redundant values with PETSc global
!     numbering
!-----------------------------------------------------------------------
      CALL VecGetArray(globalvec,globalarray,ig,ierr)
      k=ig
      IF(diagnose)WRITE(*,'(a,i3,a)')                                                &
     &     "fromnglobal = ",fromnglobal,", globalarray = "
      DO i=0,fromnglobal-1
         k=k+1
         globalarray(k)=globalrstart+i
         IF(diagnose)WRITE(*,'(i4,f11.3)')i,globalarray(k)
      ENDDO
      CALL VecRestoreArray(globalvec,globalarray,ig,ierr)
!-----------------------------------------------------------------------
!     scatter PETSc global indices to redundant valued array.
!-----------------------------------------------------------------------
      CALL VecScatterBegin(vscat,globalvec,tovec,INSERT_VALUES,                      &
     &     SCATTER_FORWARD,ierr)
      CALL VecScatterEnd(vscat,globalvec,tovec,INSERT_VALUES,                        &
     &     SCATTER_FORWARD,ierr)
!-----------------------------------------------------------------------
!     create local vector as concatenation of local vectors
!-----------------------------------------------------------------------
      nlocal=0
      cntl1=0
      cntl2=0
      cntl3=0
      IF(fa%comm(1) /= 0)THEN
         CALL VecGetSize(vl2,cntl2,ierr)
         nlocal=nlocal+cntl2
      ENDIF
      IF(fa%comm(2) /= 0)THEN
         CALL VecGetSize(vl3,cntl3,ierr)
         nlocal=nlocal+cntl3
      ENDIF
      IF(fa%comm(0) /= 0)THEN
         CALL VecGetSize(vl1,cntl1,ierr)
         nlocal=nlocal+cntl1
      ENDIF
      fa%offl(0)=cntl2+cntl3
      fa%offl(1)=0
      fa%offl(2)=cntl2
      CALL VecCreateSeq(PETSC_COMM_SELF,nlocal,localvec,ierr)
!-----------------------------------------------------------------------
!     cheat so that vl1, vl2, vl3 shared array memory with localvec.
!-----------------------------------------------------------------------
      CALL VecGetArray(localvec,localarray,il,ierr)
      CALL VecGetArray(tovec,toarray,it,ierr)
      IF(fa%comm(1) /= 0)THEN
         CALL VecPlaceArray(vl2,localarray(il+1+fa%offl(1)),ierr)
         CALL VecPlaceArray(vg2,toarray(it+1+offt(1)),ierr)
         CALL DAGlobalToLocalBegin(da2,vg2,INSERT_VALUES,vl2,ierr)
         CALL DAGlobalToLocalEnd(da2,vg2,INSERT_VALUES,vl2,ierr)
         CALL DARestoreGlobalVector(da2,vg2,ierr)
      ENDIF
      IF(fa%comm(2) /= 0)THEN
         CALL VecPlaceArray(vl3,localarray(il+1+fa%offl(2)),ierr)
         CALL VecPlaceArray(vg3,toarray(it+1+offt(2)),ierr)
         CALL DAGlobalToLocalBegin(da3,vg3,INSERT_VALUES,vl3,ierr)
         CALL DAGlobalToLocalEnd(da3,vg3,INSERT_VALUES,vl3,ierr)
         CALL DARestoreGlobalVector(da3,vg3,ierr)
      ENDIF
      IF(fa%comm(0) /= 0)THEN
         CALL VecPlaceArray(vl1,localarray(il+1+fa%offl(0)),ierr)
         CALL VecPlaceArray(vg1,toarray(it+1+offt(0)),ierr)
         CALL DAGlobalToLocalBegin(da1,vg1,INSERT_VALUES,vl1,ierr)
         CALL DAGlobalToLocalEnd(da1,vg1,INSERT_VALUES,vl1,ierr)
         CALL DARestoreGlobalVector(da1,vg1,ierr)
      ENDIF
      CALL VecRestoreArray(localvec,localarray,il,ierr)
      CALL VecRestoreArray(tovec,toarray,it,ierr)
!-----------------------------------------------------------------------
!     clean up.
!-----------------------------------------------------------------------
      CALL VecScatterDestroy(vscat,ierr)
      CALL VecDestroy(tovec,ierr)
!-----------------------------------------------------------------------
!     compute index set.
!-----------------------------------------------------------------------
      ALLOCATE(indices(0:nlocal-1))
      CALL VecGetArray(localvec,localarray,il,ierr)
      k=il
      IF(diagnose)WRITE(*,'(a,i3,a3,i4,a)')                                     &
     &     "nlocal = ", nlocal,", offl = ",fa%offl,", indices = "
      DO i=0,nlocal-1
         k=k+1
         indices(i)= int(2*localarray(k))
         IF(diagnose)WRITE(*,'(2i4)')i,indices(i)
      ENDDO
      CALL VecRestoreArray(localvec,localarray,il,ierr)
      CALL ISCreateBlock(PETSC_COMM_WORLD,2,nlocal,indices,is,ierr)
      DEALLOCATE(indices)
!-----------------------------------------------------------------------
!     create local and global vectors.
!-----------------------------------------------------------------------
      CALL VecCreateSeq(PETSC_COMM_SELF,2*nlocal,fa%l,ierr)
      CALL VecCreateMPI(PETSC_COMM_WORLD,2*fromnglobal,PETSC_DETERMINE,         &
     &     fa%g,ierr)
!-----------------------------------------------------------------------
!     create final scatter that goes directly from globalvec to localvec
!     this is the one to be used in the application code.
!-----------------------------------------------------------------------
      CALL VecScatterCreate(fa%g,is,fa%l,PETSC_NULL_OBJECT,                     &
     &     fa%vscat,ierr)
!-----------------------------------------------------------------------
!     clean up.
!-----------------------------------------------------------------------
      CALL ISDestroy(is,ierr)
      CALL VecDestroy(globalvec,ierr)
      CALL VecDestroy(localvec,ierr)
      IF(fa%comm(0) /= 0)THEN
         CALL DARestoreLocalVector(da1,vl1,ierr)
         CALL DADestroy(da1,ierr)
      ENDIF
      IF(fa%comm(1) /= 0)THEN
         CALL DARestoreLocalVector(da2,vl2,ierr)
         CALL DADestroy(da2,ierr)
      ENDIF
      IF(fa%comm(2) /= 0)THEN
         CALL DARestoreLocalVector(da3,vl3,ierr)
         CALL DADestroy(da3,ierr)
      ENDIF
!-----------------------------------------------------------------------
!     terminate.
!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE barry_create_fa
!-----------------------------------------------------------------------
!     subprogram 7. barry_draw_patch.
!     crude graphics to test that the ghost points are properly updated.
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
!     declarations.
!-----------------------------------------------------------------------
      SUBROUTINE barry_draw_patch(draw,patch,ierr)

      PetscDraw, INTENT(IN) :: draw
      TYPE(patch_type), DIMENSION(0:2), INTENT(IN) :: patch
      INTEGER, INTENT(OUT) :: ierr

      INTEGER :: ix,iy,j
      PetscReal, DIMENSION(4) :: x,y
!-----------------------------------------------------------------------
!     draw it.
!-----------------------------------------------------------------------
      DO j=0,2
         DO iy=1,patch(j)%my
            DO ix=1,patch(j)%mx
               x(1)=patch(j)%xy(1,ix-1,iy-1)
               y(1)=patch(j)%xy(2,ix-1,iy-1)
               x(2)=patch(j)%xy(1,ix,iy-1)
               y(2)=patch(j)%xy(2,ix,iy-1)
               x(3)=patch(j)%xy(1,ix,iy)
               y(3)=patch(j)%xy(2,ix,iy)
               x(4)=patch(j)%xy(1,ix-1,iy)
               y(4)=patch(j)%xy(2,ix-1,iy)
               CALL PetscDrawLine(draw,x(1),y(1),x(2),y(2),                        &
     &              PETSC_DRAW_BLACK,ierr)
               CALL PetscDrawLine(draw,x(2),y(2),x(3),y(3),                        &
     &              PETSC_DRAW_BLACK,ierr)
               CALL PetscDrawLine(draw,x(3),y(3),x(4),y(4),                        &
     &              PETSC_DRAW_BLACK,ierr)
               CALL PetscDrawLine(draw,x(4),y(4),x(1),y(1),                        &
     &              PETSC_DRAW_BLACK,ierr)
            ENDDO
         ENDDO
      ENDDO
!-----------------------------------------------------------------------
!     terminate.
!-----------------------------------------------------------------------
      ierr=0
      RETURN
      END SUBROUTINE barry_draw_patch
!-----------------------------------------------------------------------
!     subprogram 8. barry_draw_fa.
!     deallocates local array.
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
!     declarations.
!-----------------------------------------------------------------------
      SUBROUTINE barry_draw_fa(fa,v)

      TYPE(fa_type), INTENT(IN) :: fa
      Vec, INTENT(IN) :: v

      INTEGER :: iv,vn,ln,j,offset
      REAL(8), DIMENSION(1) :: va
      TYPE(patch_type), DIMENSION(0:2) :: patch
      PetscDraw :: draw
      PetscReal :: xmin,xmax,ymin,ymax
      PetscReal :: ymint,xmint,ymaxt,xmaxt
!-----------------------------------------------------------------------
!     set extrema.
!-----------------------------------------------------------------------
      xmin=HUGE(xmin)
      xmax=-HUGE(xmax)
      ymin=HUGE(ymin)
      ymax=-HUGE(ymax)
      xmint=HUGE(xmint)
      xmaxt=-HUGE(xmaxt)
      ymint=HUGE(ymint)
      ymaxt=-HUGE(ymaxt)
!-----------------------------------------------------------------------
!     get arrays and sizes.
!-----------------------------------------------------------------------
      CALL VecGetArray(v,va,iv,ierr)
      CALL VecGetSize(v,vn,ierr)
      CALL VecGetSize(fa%l,ln,ierr)
!-----------------------------------------------------------------------
!     fill arrays.
!-----------------------------------------------------------------------
      DO j=0,2
         IF(vn == ln)THEN
            offset=iv+2*fa%offl(j)
            patch(j)%mx=fa%ml(j)-1
            patch(j)%my=fa%nl(j)-1
         ELSE
            offset=iv+2*fa%offg(j)
            patch(j)%mx=fa%mg(j)-1
            patch(j)%my=fa%ng(j)-1
         ENDIF
         ALLOCATE(patch(j)%xy(2,0:patch(j)%mx,0:patch(j)%my))
         patch(j)%xy=RESHAPE(va(offset+1:offset+SIZE(patch(j)%xy)),                &
     &        (/2,patch(j)%mx+1,patch(j)%my+1/))
      ENDDO
!-----------------------------------------------------------------------
!     compute extrema over processor.
!-----------------------------------------------------------------------
      DO j=0,2
         xmin=MIN(xmin,MINVAL(patch(j)%xy(1,:,:)))
         xmax=MAX(xmax,MAXVAL(patch(j)%xy(1,:,:)))
         ymin=MIN(ymin,MINVAL(patch(j)%xy(2,:,:)))
         ymax=MAX(ymax,MAXVAL(patch(j)%xy(2,:,:)))
      ENDDO
!-----------------------------------------------------------------------
!     compute global extrema.
!-----------------------------------------------------------------------
      CALL MPI_Allreduce(xmin,xmint,1,MPI_DOUBLE_PRECISION,MPI_MIN,                 &
     &     PETSC_COMM_WORLD,ierr)
      CALL MPI_Allreduce(xmax,xmaxt,1,MPI_DOUBLE_PRECISION,MPI_MAX,                 &
     &     PETSC_COMM_WORLD,ierr)
      CALL MPI_Allreduce(ymin,ymint,1,MPI_DOUBLE_PRECISION,MPI_MIN,                 &
     &     PETSC_COMM_WORLD,ierr)
      CALL MPI_Allreduce(ymax,ymaxt,1,MPI_DOUBLE_PRECISION,MPI_MAX,                 &
     &     PETSC_COMM_WORLD,ierr)
!-----------------------------------------------------------------------
!     set margins.
!-----------------------------------------------------------------------
      xmin=xmint-.2*(xmaxt-xmint)
      xmax=xmaxt+.2*(xmaxt-xmint)
      ymin=ymint-.2*(ymaxt-ymint)
      ymax=ymaxt+.2*(ymaxt-ymint)
!-----------------------------------------------------------------------
!     draw it.
!-----------------------------------------------------------------------
#if defined(PETSC_HAVE_X11)
      CALL PetscDrawOpenX(PETSC_COMM_WORLD,0,"meshes",PETSC_DECIDE,                  &
     &     PETSC_DECIDE,700,700,draw,ierr)
      CALL PetscDrawSetCoordinates(draw,xmin,ymin,xmax,ymax,ierr)
      CALL PetscDrawZoom(draw,barry_draw_patch,patch,ierr)
#endif
!-----------------------------------------------------------------------
!     clean up.
!-----------------------------------------------------------------------
      CALL VecRestoreArray(v,va,iv,ierr)
#if defined(PETSC_HAVE_X11)
      CALL PetscDrawDestroy(draw,ierr)
#endif
!-----------------------------------------------------------------------
!     terminate.
!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE barry_draw_fa
!-----------------------------------------------------------------------
!     subprogram 9. barry_map_region3.
!     fills region 3 coordinates.
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
!     declarations.
!-----------------------------------------------------------------------
      SUBROUTINE barry_map_region3(fa,g)

      TYPE(fa_type), INTENT(INOUT) :: fa
      Vec, INTENT(INOUT) :: g

      INTEGER :: i,j,k,x,y,m,n,ig
      REAL(8) :: r0=1,theta0=pi/6,r,theta,dr,dt
      REAL(8), DIMENSION(1) :: ga
!-----------------------------------------------------------------------
!     format statements.
!-----------------------------------------------------------------------
 10   FORMAT(/5x,"j",5x,"i",6x,"r",8x,"theta",8x,"x",10x,"y"/)
 20   FORMAT(2i6,4f11.3)
!-----------------------------------------------------------------------
!     preliminary computations.
!-----------------------------------------------------------------------
      dr=r0/(fa%r2-1)
      dt=twopi/(3*(fa%p1-fa%p2-1))
      CALL barry_get_global_corners(fa,2,x,y,m,n)
!-----------------------------------------------------------------------
!     fill array.
!-----------------------------------------------------------------------
      CALL VecGetArray(g,ga,ig,ierr)
      k=ig+2*fa%offg(2)
      IF(diagnose)THEN
         WRITE(*,'(a)')"region 3"
         WRITE(*,10)
      ENDIF
      DO j=y,y+n-1
         r=r0+j*dr
         DO i=x,x+m-1
            theta=theta0+i*dt
            ga(k+1)=r*COS(theta)
            ga(k+2)=r*SIN(theta)-4*r0
            IF(diagnose)WRITE(*,20)j,i,r,theta,ga(k+1),ga(k+2)
            k=k+2
         ENDDO
         IF(diagnose)WRITE(*,10)
      ENDDO
      CALL VecRestoreArray(g,ga,ig,ierr)
!-----------------------------------------------------------------------
!     terminate.
!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE barry_map_region3
!-----------------------------------------------------------------------
!     subprogram 10. barry_map_region2.
!     fills region 2 coordinates.
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
!     declarations.
!-----------------------------------------------------------------------
      SUBROUTINE barry_map_region2(fa,g)

      TYPE(fa_type), INTENT(INOUT) :: fa
      Vec, INTENT(INOUT) :: g

      INTEGER :: i,j,k,x,y,m,n,ig
      REAL(8) :: r0=1,theta0=-pi/2,r,theta,dr,dt
      REAL(8), DIMENSION(1) :: ga
!-----------------------------------------------------------------------
!     format statements.
!-----------------------------------------------------------------------
 10   FORMAT(/5x,"j",5x,"i",6x,"r",8x,"theta",8x,"x",10x,"y"/)
 20   FORMAT(2i6,4f11.3)
!-----------------------------------------------------------------------
!     preliminary computations.
!-----------------------------------------------------------------------
      dr=r0/(fa%r2-1)
      dt=twopi/fa%p2
      CALL barry_get_global_corners(fa,1,x,y,m,n)
!-----------------------------------------------------------------------
!     fill array.
!-----------------------------------------------------------------------
      CALL VecGetArray(g,ga,ig,ierr)
      k=ig+2*fa%offg(1)
      IF(diagnose)THEN
         WRITE(*,'(a)')"region 2"
         WRITE(*,10)
      ENDIF
      DO j=y,y+n-1
         r=r0+j*dr
         DO i=x,x+m-1
            theta=theta0+i*dt
            ga(k+1)=r*COS(theta)
            ga(k+2)=r*SIN(theta)
            IF(diagnose)WRITE(*,20)j,i,r,theta,ga(k+1),ga(k+2)
            k=k+2
         ENDDO
         IF(diagnose)WRITE(*,10)
      ENDDO
      CALL VecRestoreArray(g,ga,ig,ierr)
!-----------------------------------------------------------------------
!     terminate.
!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE barry_map_region2
!-----------------------------------------------------------------------
!     subprogram 11. barry_map_region1.
!     fills region 1 coordinates.
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
!     declarations.
!-----------------------------------------------------------------------
      SUBROUTINE barry_map_region1(fa,g)

      TYPE(fa_type), INTENT(INOUT) :: fa
      Vec, INTENT(INOUT) :: g

      INTEGER :: i,j,k,x,y,m,n,ig
      REAL(8) :: r0=1,r,theta,dr,dt1,dt3
      REAL(8), DIMENSION(1) :: ga
!-----------------------------------------------------------------------
!     format statements.
!-----------------------------------------------------------------------
 10   FORMAT(/5x,"j",5x,"i",6x,"r",8x,"theta",8x,"x",10x,"y"/)
 20   FORMAT(2i6,4f11.3)
!-----------------------------------------------------------------------
!     preliminary computations.
!-----------------------------------------------------------------------
      dr=r0/(fa%r1-1)
      dt1=twopi/fa%p2
      dt3=twopi/(3*(fa%p1-fa%p2-1))
      CALL barry_get_global_corners(fa,0,x,y,m,n)
!-----------------------------------------------------------------------
!     fill array.
!-----------------------------------------------------------------------
      CALL VecGetArray(g,ga,ig,ierr)
      k=ig+2*fa%offg(0)
      IF(diagnose)THEN
         WRITE(*,'(a)')"region 1"
         WRITE(*,10)
      ENDIF
      DO j=y,y+n-1
         r=2*r0+j*dr
         DO i=x,x+m-1
            IF(i < (fa%p1-fa%p2)/2)THEN
               theta=i*dt3
               ga(k+1)=r*COS(theta)
               ga(k+2)=r*SIN(theta)-4*r0
            ELSEIF(i > fa%p2 + (fa%p1-fa%p2)/2)then
               theta=pi+i*dt3
               ga(k+1)=r*COS(PETSC_PI+i*dt3)
               ga(k+2)=r*SIN(PETSC_PI+i*dt3)-4*r0
            ELSE
               theta=(i-(fa%p1-fa%p2)/2)*dt1-pi/2
               ga(k+1)=r*COS(theta)
               ga(k+2)=r*SIN(theta)
            ENDIF
            IF(diagnose)WRITE(*,20)j,i,r,theta,ga(k+1),ga(k+2)
            k=k+2
         ENDDO
         IF(diagnose)WRITE(*,10)
      ENDDO
      CALL VecRestoreArray(g,ga,ig,ierr)
!-----------------------------------------------------------------------
!     terminate.
!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE barry_map_region1
      END MODULE barry_mod
!-----------------------------------------------------------------------
!     subprogram 12. barry_main.
!     main program.
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
!     declarations.
!-----------------------------------------------------------------------
      PROGRAM barry_main
      USE barry_mod
      IMPLICIT NONE

      TYPE(fa_type) :: fa
      Vec :: g,l
!-----------------------------------------------------------------------
!     initialize and create structure, and get vectors.
!-----------------------------------------------------------------------
      CALL PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      CALL barry_create_fa(fa)
      CALL barry_get_global_vector(fa,g)
      CALL barry_get_local_vector(fa,l)
!-----------------------------------------------------------------------
!     map regions.
!-----------------------------------------------------------------------
      CALL barry_map_region1(fa,g)
      CALL barry_map_region2(fa,g)
      CALL barry_map_region3(fa,g)
!-----------------------------------------------------------------------
!     graphic output.
!-----------------------------------------------------------------------
      CALL barry_global_to_local(fa,g,l)
      CALL barry_draw_fa(fa,g)
      CALL barry_draw_fa(fa,l)
!-----------------------------------------------------------------------
!     clean up and finalize.
!-----------------------------------------------------------------------
      CALL VecDestroy(g,ierr)
      CALL VecDestroy(l,ierr)
      CALL barry_destroy_fa(fa)
      CALL PetscFinalize(ierr)
!-----------------------------------------------------------------------
!     terminate.
!-----------------------------------------------------------------------
      END PROGRAM barry_main
