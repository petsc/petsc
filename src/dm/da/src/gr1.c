#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: gr1.c,v 1.1 1998/11/25 17:08:34 bsmith Exp bsmith $";
#endif

/* 
   Plots vectors obtained with DACreate1d()

   How should we handle multi-component problems?
*/

#include "da.h"      /*I  "da.h"   I*/

int DACreateDefaultCoordinates(DA da)
{
  int            i,ierr,N,igstart,igsize,dim;
  double         h;
  Vec            xcoor;
  DAPeriodicType periodic;
  Scalar         *xg;

  PetscFunctionBegin;
  ierr = DAGetInfo(da,&dim,&N,0,0,0,0,0,0,0,&periodic);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&igstart,0,0,&igsize,0,0);CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,igsize,&xcoor);CHKERRQ(ierr);
  if (periodic == DA_NONPERIODIC) h = 1.0/(N-1);
  else                            h = 1.0/N;
  ierr = VecGetArray(xcoor,&xg);CHKERRQ(ierr);
  for ( i=0; i<igsize; i++ ) {
    xg[i] = h*(i+igstart);
  }
  ierr = VecRestoreArray(xcoor,&xg);CHKERRQ(ierr);
  ierr = DASetCoordinates(da,xcoor);CHKERRQ(ierr);
  PLogObjectParent(da,xcoor);

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "VecView_MPI_Draw_DA1d"
int VecView_MPI_Draw_DA1d(Vec xin,Viewer v)
{
  DA             da;
  int            i,rank,size,ierr,start,n,tag1,tag2,igstart,igsize,N,step;
  int            istart,isize,j;
  MPI_Status     status;
  double         coors[4],ymin,ymax,min,max,xmin,xmax,tmp;
  Scalar         *array,*xg;
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

  ierr = DAGetInfo(da,0,&N,0,0,0,0,0,&step,0,&periodic);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&igstart,0,0,&igsize,0,0);CHKERRQ(ierr);
  ierr = DAGetCorners(da,&istart,0,0,&isize,0,0);CHKERRQ(ierr);
  ierr = VecGetArray(xin,&array); CHKERRQ(ierr);
  ierr = VecGetLocalSize(xin,&n); CHKERRQ(ierr);
  n    = n/step;

  /* get coordinates of nodes */
  ierr = DAGetCoordinates(da,&xcoor);CHKERRQ(ierr);
  if (!xcoor) {
    ierr = DACreateDefaultCoordinates(da);CHKERRQ(ierr);
    ierr = DAGetCoordinates(da,&xcoor);CHKERRQ(ierr);
  }
  ierr = VecGetArray(xcoor,&xg);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  MPI_Comm_size(comm,&size); 
  MPI_Comm_rank(comm,&rank);

  /*
      Determine the min and max x coordinate in plot 
  */
  if (rank == 0) {
    xmin = PetscReal(xg[istart-igstart]);
  } 
  if (rank == size-1) {
    xmax = PetscReal(xg[igsize-1]);
  }
  MPI_Bcast(&xmin,1,MPI_DOUBLE,0,comm);
  MPI_Bcast(&xmax,1,MPI_DOUBLE,size-1,comm);

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
      ierr = DrawAxisSetLimits(axis,xmin,xmax,ymin,ymax); CHKERRQ(ierr);
      ierr = DrawAxisDraw(axis); CHKERRQ(ierr);
      ierr = DrawGetCoordinates(draw,coors,coors+1,coors+2,coors+3);CHKERRQ(ierr);
    }
    ierr = MPI_Bcast(coors,4,MPI_DOUBLE,0,comm);CHKERRQ(ierr);
    if (rank) {
      ierr = DrawSetCoordinates(draw,coors[0],coors[1],coors[2],coors[3]);CHKERRQ(ierr);
    }

    /* draw local part of vector */
    PetscObjectGetNewTag((PetscObject)xin,&tag1);CHKERRQ(ierr);
    PetscObjectGetNewTag((PetscObject)xin,&tag2);CHKERRQ(ierr);
    if (rank < size-1) { /*send value to right */
      ierr = MPI_Send(&array[j+(n-1)*step],1,MPI_DOUBLE,rank+1,tag1,comm);CHKERRQ(ierr);
    }
    if (rank == 0 && periodic) { /* first processor sends first value to last */
      ierr = MPI_Send(&array[j],1,MPI_DOUBLE,size-1,tag2,comm);CHKERRQ(ierr);
    }

    start = istart - igstart;
    for ( i=1; i<n; i++ ) {
#if !defined(USE_PETSC_COMPLEX)
      ierr = DrawLine(draw,xg[i-1+start],array[j+step*(i-1)],xg[i+start],array[j+step*i],
                      DRAW_RED);CHKERRQ(ierr);
#else
      ierr = DrawLine(draw,PetscReal(xg[i-1+start]),PetscReal(array[j+step*(i-1)]),
                      PetscReal(xg[i+start]),PetscReal(array[j+step*i]),DRAW_RED);CHKERRQ(ierr);
#endif
    }
    if (rank) { /* receive value from left */
      ierr = MPI_Recv(&tmp,1,MPI_DOUBLE,rank-1,tag1,comm,&status);CHKERRQ(ierr);
#if !defined(USE_PETSC_COMPLEX)
      ierr = DrawLine(draw,xg[start-1],tmp,xg[start],array[j],DRAW_RED);CHKERRQ(ierr);
#else
      ierr = DrawLine(draw,PetscReal(xg[start-1]),tmp,PetscReal(xg[start]),PetscReal(array[j]),
                      DRAW_RED);CHKERRQ(ierr);
#endif
    }
    if (rank == size-1 && periodic) {
      ierr = MPI_Recv(&tmp,1,MPI_DOUBLE,0,tag2,comm,&status);CHKERRQ(ierr);
#if !defined(USE_PETSC_COMPLEX)
      ierr = DrawLine(draw,xg[n+start-1],array[j+step*(n-1)],xg[n+start],tmp,DRAW_RED);CHKERRQ(ierr);
#else
      ierr = DrawLine(draw,PetscReal(xg[n+start-1]),PetscReal(array[j+step*(n-1)]),
                      PetscReal(xg[n+start]),tmp,DRAW_RED);CHKERRQ(ierr);
#endif
    }
    PetscObjectRestoreNewTag((PetscObject)xin,&tag2);CHKERRQ(ierr);
    PetscObjectRestoreNewTag((PetscObject)xin,&tag1);CHKERRQ(ierr);
    ierr = DrawSynchronizedFlush(draw); CHKERRQ(ierr);
    ierr = DrawPause(draw); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xcoor,&xg);CHKERRQ(ierr);
  ierr = VecRestoreArray(xin,&array); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
