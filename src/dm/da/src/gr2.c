#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: gr2.c,v 1.2 1998/12/23 22:53:37 bsmith Exp balay $";
#endif

/* 
   Plots vectors obtained with DACreate2d()
*/

#include "da.h"      /*I  "da.h"   I*/

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "VecView_MPI_Draw_DA2d"
int VecView_MPI_Draw_DA2d(Vec xin,Viewer v)
{
  DA             da;
  int            i,rank,size,ierr,n,tag1,tag2,igstart,igsize,N,step;
  int            istart,isize,j,M,jgsize,jgstart;
  double         coors[4],ymin,ymax,min,max,xmin,xmax;
  Scalar         *array,*xy;
  Draw           draw;
  PetscTruth     isnull;
  MPI_Comm       comm;
  DrawAxis       axis;
  Vec            xcoor;
  DAPeriodicType periodic;

  PetscFunctionBegin;
  ierr = ViewerDrawGetDraw(v,0,&draw);CHKERRQ(ierr);
  ierr = DrawIsNull(draw,&isnull); CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);

  ierr = PetscObjectQuery((PetscObject)xin,"DA",(PetscObject*) &da);CHKERRQ(ierr);
  if (!da) SETERRQ(1,1,"Vector not generated from a DA");

  ierr = DAGetInfo(da,0,&M,&N,0,0,0,0,&step,0,&periodic);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&igstart,&jgstart,0,&igsize,&jgsize,0);CHKERRQ(ierr);
  ierr = DAGetCorners(da,&istart,0,0,&isize,0,0);CHKERRQ(ierr);
  ierr = VecGetArray(xin,&array); CHKERRQ(ierr);
  ierr = VecGetLocalSize(xin,&n); CHKERRQ(ierr);
  n    = n/step;

  /* get coordinates of nodes */
  ierr = DAGetCoordinates(da,&xcoor);CHKERRQ(ierr);
  if (!xcoor) {
    ierr = DACreateUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
    ierr = DAGetCoordinates(da,&xcoor);CHKERRQ(ierr);
  }
  ierr = VecGetArray(xcoor,&xy);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  MPI_Comm_size(comm,&size); 
  MPI_Comm_rank(comm,&rank);

  /*
      Determine the min and max x coordinate in plot 
  */
  min = 1.e20; max = -1.e20;
  for ( i=0; i<igsize*jgsize; i++ ) {
    min = PetscMin(min,PetscReal(xy[2*i]));
    max = PetscMax(max,PetscReal(xy[2*i]));
  }
  ierr = MPI_Allreduce(&min,&xmin,1,MPI_DOUBLE,MPI_MIN,comm);CHKERRQ(ierr);


  for ( j=0; j<step; j++ ) {
    ierr = ViewerDrawGetDraw(v,j,&draw);CHKERRQ(ierr);
    ierr = DrawCheckResizedWindow(draw);CHKERRQ(ierr);

    /*
        Determine the min and max y coordinate in plot 
    */
    min = 1.e20; max = -1.e20;
    for ( i=0; i<n; i++ ) {
#if defined(USE_PETSC_COMPLEX)
      if (PetscReal(array[j+i*step]) < min) min = PetscReal(array[j+i*step]);
      if (PetscReal(array[j+i*step]) > max) max = PetscReal(array[j+i*step]);
#else
      if (array[j+i*step] < min) min = array[j+i*step];
      if (array[j+i*step] > max) max = array[j+i*step];
#endif
    }
    if (min + 1.e-10 > max) {
      min -= 1.e-5;
      max += 1.e-5;
    }
    ierr = MPI_Reduce(&min,&ymin,1,MPI_DOUBLE,MPI_MIN,0,comm);CHKERRQ(ierr);
    ierr = MPI_Reduce(&max,&ymax,1,MPI_DOUBLE,MPI_MAX,0,comm);CHKERRQ(ierr);

    ierr = DrawSynchronizedClear(draw); CHKERRQ(ierr);
    ierr = ViewerDrawGetDrawAxis(v,j,&axis); CHKERRQ(ierr);
    PLogObjectParent(draw,axis);
    if (!rank) {
      char *title;

      ierr = DrawAxisSetLimits(axis,xmin,xmax,ymin,ymax); CHKERRQ(ierr);
      ierr = DrawAxisDraw(axis); CHKERRQ(ierr);
      ierr = DrawGetCoordinates(draw,coors,coors+1,coors+2,coors+3);CHKERRQ(ierr);
      ierr = DAGetFieldName(da,j,&title);CHKERRQ(ierr);
      ierr = DrawSetTitle(draw,title);CHKERRQ(ierr);
    }
    ierr = MPI_Bcast(coors,4,MPI_DOUBLE,0,comm);CHKERRQ(ierr);
    if (rank) {
      ierr = DrawSetCoordinates(draw,coors[0],coors[1],coors[2],coors[3]);CHKERRQ(ierr);
    }

    /* draw local part of vector */
    PetscObjectGetNewTag((PetscObject)xin,&tag1);CHKERRQ(ierr);
    PetscObjectGetNewTag((PetscObject)xin,&tag2);CHKERRQ(ierr);

    ierr = DrawSynchronizedFlush(draw); CHKERRQ(ierr);
    ierr = DrawPause(draw); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xcoor,&xy);CHKERRQ(ierr);
  ierr = VecRestoreArray(xin,&array); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
