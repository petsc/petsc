#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: gr2.c,v 1.3 1999/01/08 14:52:28 balay Exp bsmith $";
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
  int            i,rank,size,ierr,n,igstart,igsize,N,step;
  int            istart,isize,j,M,jgsize,jgstart;
  double         coors[4],ymin,ymax,min,max,xmin,xmax,xminw,xmaxw,yminw,ymaxw;
  Scalar         *array,*xy;
  Draw           draw;
  PetscTruth     isnull;
  MPI_Comm       comm;
  Vec            xlocal,xcoor;
  DAPeriodicType periodic;

  PetscFunctionBegin;
  ierr = ViewerDrawGetDraw(v,0,&draw);CHKERRQ(ierr);
  ierr = DrawIsNull(draw,&isnull); CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);

  ierr = PetscObjectQuery((PetscObject)xin,"DA",(PetscObject*) &da);CHKERRQ(ierr);
  if (!da) SETERRQ(1,1,"Vector not generated from a DA");

  /*
      Get local (ghosted) values of vector
  */
  ierr = DACreateLocalVector(da,&xlocal);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,xin,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,xin,INSERT_VALUES,xlocal);CHKERRQ(ierr);

  ierr = DAGetInfo(da,0,&M,&N,0,0,0,0,&step,0,&periodic);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&igstart,&jgstart,0,&igsize,&jgsize,0);CHKERRQ(ierr);
  ierr = DAGetCorners(da,&istart,0,0,&isize,0,0);CHKERRQ(ierr);
  ierr = VecGetArray(xlocal,&array); CHKERRQ(ierr);

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
  n   = igsize*jgsize;
  xminw = 1.e20; xmaxw = -1.e20;
  yminw = 1.e20; ymaxw = -1.e20;
  for ( i=0; i<n; i++ ) {
    xminw = PetscMin(xminw,PetscReal(xy[2*i]));
    xmaxw = PetscMax(xmaxw,PetscReal(xy[2*i]));
    yminw = PetscMin(yminw,PetscReal(xy[2*i+1]));
    ymaxw = PetscMax(ymaxw,PetscReal(xy[2*i+1]));
  }
  ierr = MPI_Allreduce(&xminw,&xmin,1,MPI_DOUBLE,MPI_MIN,comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&xmaxw,&xmax,1,MPI_DOUBLE,MPI_MAX,comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&yminw,&ymin,1,MPI_DOUBLE,MPI_MIN,comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&ymaxw,&ymax,1,MPI_DOUBLE,MPI_MAX,comm);CHKERRQ(ierr);
  coors[0] = xmin - .05*(xmax- xmin); coors[2] = xmax + .05*(xmax - xmin);
  coors[1] = ymin - .05*(ymax- ymin); coors[3] = ymax + .05*(ymax - ymin);
  ierr = MPI_Bcast(coors,4,MPI_DOUBLE,0,comm);CHKERRQ(ierr);

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

    if (!rank) {
      char *title;

      ierr = DrawGetCoordinates(draw,coors,coors+1,coors+2,coors+3);CHKERRQ(ierr);
      ierr = DAGetFieldName(da,j,&title);CHKERRQ(ierr);
      ierr = DrawSetTitle(draw,title);CHKERRQ(ierr);
    }
    ierr = DrawSetCoordinates(draw,coors[0],coors[1],coors[2],coors[3]);CHKERRQ(ierr);

    ierr = DrawSynchronizedFlush(draw); CHKERRQ(ierr);
    ierr = DrawPause(draw); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xcoor,&xy);CHKERRQ(ierr);
  ierr = VecRestoreArray(xin,&array); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
