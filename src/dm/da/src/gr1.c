/*$Id: gr1.c,v 1.23 2000/07/10 03:12:46 bsmith Exp bsmith $*/

/* 
   Plots vectors obtained with DACreate1d()
*/

#include "petscda.h"      /*I  "petscda.h"   I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DASetUniformCoordinates"
/*@
    DASetUniformCoordinates - Sets a DA coordinates to be a uniform grid

  Collective on DA

  Input Parameters:
+  da - the distributed array object
.  xmin,xmax - extremes in the x direction
.  xmin,xmax - extremes in the y direction
-  xmin,xmax - extremes in the z direction

  Level: beginner

.seealso: DASetCoordinates(), DAGetCoordinates(), DACreate1d(), DACreate2d(), DACreate3d()

@*/
int DASetUniformCoordinates(DA da,double xmin,double xmax,double ymin,double ymax,double zmin,double zmax)
{
  int            i,j,k,ierr,M,N,P,istart,isize,jstart,jsize,kstart,ksize,dim,cnt;
  double         hx,hy,hz_;
  Vec            xcoor;
  DAPeriodicType periodic;
  Scalar         *coors;

  PetscFunctionBegin;
  if (xmax <= xmin) SETERRQ2(1,"Xmax must be larger than xmin %g %g",xmin,xmax);

  ierr = DAGetInfo(da,&dim,&M,&N,&P,0,0,0,0,0,&periodic,0);CHKERRQ(ierr);
  ierr = DAGetCorners(da,&istart,&jstart,&kstart,&isize,&jsize,&ksize);CHKERRQ(ierr);

  if (dim == 1) {
    ierr = VecCreateMPI(PETSC_COMM_WORLD,isize,PETSC_DETERMINE,&xcoor);CHKERRQ(ierr);
    if (periodic == DA_NONPERIODIC) hx = (xmax-xmin)/(M-1);
    else                            hx = (xmax-xmin)/M;
    ierr = VecGetArray(xcoor,&coors);CHKERRQ(ierr);
    for (i=0; i<isize; i++) {
      coors[i] = xmin + hx*(i+istart);
    }
    ierr = VecRestoreArray(xcoor,&coors);CHKERRQ(ierr);
  } else if (dim == 2) {
    if (ymax <= ymin) SETERRQ2(1,"Ymax must be larger than ymin %g %g",ymin,ymax);
    ierr = VecCreateMPI(PETSC_COMM_WORLD,2*isize*jsize,PETSC_DETERMINE,&xcoor);CHKERRQ(ierr);
    ierr = VecSetBlockSize(xcoor,2);CHKERRQ(ierr);
    if (periodic == DA_XPERIODIC || periodic == DA_XYPERIODIC) hx = (xmax-xmin)/(M);
    else                                                       hx = (xmax-xmin)/(M-1);
    if (periodic == DA_YPERIODIC || periodic == DA_XYPERIODIC) hy = (ymax-ymin)/(N);
    else                                                       hy = (ymax-ymin)/(N-1);
    ierr = VecGetArray(xcoor,&coors);CHKERRQ(ierr);
    cnt  = 0;
    for (j=0; j<jsize; j++) {
      for (i=0; i<isize; i++) {
        coors[cnt++] = xmin + hx*(i+istart);
        coors[cnt++] = ymin + hy*(j+jstart);
      }
    }
    ierr = VecRestoreArray(xcoor,&coors);CHKERRQ(ierr);
  } else if (dim == 3) {
    if (ymax <= ymin) SETERRQ2(1,"Ymax must be larger than ymin %g %g",ymin,ymax);
    if (zmax <= zmin) SETERRQ2(1,"Zmax must be larger than zmin %g %g",zmin,zmax);
    ierr = VecCreateMPI(PETSC_COMM_WORLD,3*isize*jsize*ksize,PETSC_DETERMINE,&xcoor);CHKERRQ(ierr);
    ierr = VecSetBlockSize(xcoor,3);CHKERRQ(ierr);
    if (periodic == DA_XPERIODIC || periodic == DA_XYPERIODIC || periodic == DA_XZPERIODIC ||
        periodic == DA_XYZPERIODIC)                            hx = (xmax-xmin)/(M);
    else                                                       hx = (xmax-xmin)/(M-1);
    if (periodic == DA_YPERIODIC || periodic == DA_XYPERIODIC || periodic == DA_YZPERIODIC ||
        periodic == DA_XYZPERIODIC)                            hy = (ymax-ymin)/(N);
    else                                                       hy = (ymax-ymin)/(N-1);
    if (periodic == DA_ZPERIODIC || periodic == DA_XZPERIODIC || periodic == DA_YZPERIODIC || 
        periodic == DA_XYZPERIODIC)                            hz_ = (zmax-zmin)/(P);
    else                                                       hz_ = (zmax-zmin)/(P-1);
    ierr = VecGetArray(xcoor,&coors);CHKERRQ(ierr);
    cnt  = 0;
    for (k=0; k<ksize; k++) {
      for (j=0; j<jsize; j++) {
        for (i=0; i<isize; i++) {
          coors[cnt++] = xmin + hx*(i+istart);
          coors[cnt++] = ymin + hy*(j+jstart);
          coors[cnt++] = zmin + hz_*(k+kstart);
        }
      }
    }
    ierr = VecRestoreArray(xcoor,&coors);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"Cannot create uniform coordinates for this dimension %d\n",dim);
  }
  ierr = DASetCoordinates(da,xcoor);CHKERRQ(ierr);
  PLogObjectParent(da,xcoor);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"VecView_MPI_Draw_DA1d"
int VecView_MPI_Draw_DA1d(Vec xin,Viewer v)
{
  DA             da;
  int            i,rank,size,ierr,n,tag1,tag2,N,step;
  int            istart,isize,j;
  MPI_Status     status;
  double         coors[4],ymin,ymax,min,max,xmin,xmax,tmp,xgtmp;
  Scalar         *array,*xg;
  Draw           draw;
  PetscTruth     isnull;
  MPI_Comm       comm;
  DrawAxis       axis;
  Vec            xcoor;
  DAPeriodicType periodic;

  PetscFunctionBegin;
  ierr = ViewerDrawGetDraw(v,0,&draw);CHKERRQ(ierr);
  ierr = DrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);

  ierr = PetscObjectQuery((PetscObject)xin,"DA",(PetscObject*)&da);CHKERRQ(ierr);
  if (!da) SETERRQ(1,"Vector not generated from a DA");

  ierr = DAGetInfo(da,0,&N,0,0,0,0,0,&step,0,&periodic,0);CHKERRQ(ierr);
  ierr = DAGetCorners(da,&istart,0,0,&isize,0,0);CHKERRQ(ierr);
  ierr = VecGetArray(xin,&array);CHKERRQ(ierr);
  ierr = VecGetLocalSize(xin,&n);CHKERRQ(ierr);
  n    = n/step;

  /* get coordinates of nodes */
  ierr = DAGetCoordinates(da,&xcoor);CHKERRQ(ierr);
  if (!xcoor) {
    ierr = DASetUniformCoordinates(da,0.0,1.0,0.0,0.0,0.0,0.0);CHKERRQ(ierr);
    ierr = DAGetCoordinates(da,&xcoor);CHKERRQ(ierr);
  }
  ierr = VecGetArray(xcoor,&xg);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr); 
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  /*
      Determine the min and max x coordinate in plot 
  */
  if (!rank) {
    xmin = PetscRealPart(xg[0]);
  } 
  if (rank == size-1) {
    xmax = PetscRealPart(xg[n-1]);
  }
  ierr = MPI_Bcast(&xmin,1,MPI_DOUBLE,0,comm);CHKERRQ(ierr);
  ierr = MPI_Bcast(&xmax,1,MPI_DOUBLE,size-1,comm);CHKERRQ(ierr);

  for (j=0; j<step; j++) {
    ierr = ViewerDrawGetDraw(v,j,&draw);CHKERRQ(ierr);
    ierr = DrawCheckResizedWindow(draw);CHKERRQ(ierr);

    /*
        Determine the min and max y coordinate in plot 
    */
    min = 1.e20; max = -1.e20;
    for (i=0; i<n; i++) {
#if defined(PETSC_USE_COMPLEX)
      if (PetscRealPart(array[j+i*step]) < min) min = PetscRealPart(array[j+i*step]);
      if (PetscRealPart(array[j+i*step]) > max) max = PetscRealPart(array[j+i*step]);
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

    ierr = DrawSynchronizedClear(draw);CHKERRQ(ierr);
    ierr = ViewerDrawGetDrawAxis(v,j,&axis);CHKERRQ(ierr);
    PLogObjectParent(draw,axis);
    if (!rank) {
      char *title;

      ierr = DrawAxisSetLimits(axis,xmin,xmax,ymin,ymax);CHKERRQ(ierr);
      ierr = DrawAxisDraw(axis);CHKERRQ(ierr);
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
    if (rank < size-1) { /*send value to right */
      ierr = MPI_Send(&array[j+(n-1)*step],1,MPI_DOUBLE,rank+1,tag1,comm);CHKERRQ(ierr);
      ierr = MPI_Send(&xg[n-1],1,MPI_DOUBLE,rank+1,tag1,comm);CHKERRQ(ierr);
    }
    if (!rank && periodic) { /* first processor sends first value to last */
      ierr = MPI_Send(&array[j],1,MPI_DOUBLE,size-1,tag2,comm);CHKERRQ(ierr);
    }

    for (i=1; i<n; i++) {
#if !defined(PETSC_USE_COMPLEX)
      ierr = DrawLine(draw,xg[i-1],array[j+step*(i-1)],xg[i],array[j+step*i],
                      DRAW_RED);CHKERRQ(ierr);
#else
      ierr = DrawLine(draw,PetscRealPart(xg[i-1]),PetscRealPart(array[j+step*(i-1)]),
                      PetscRealPart(xg[i]),PetscRealPart(array[j+step*i]),DRAW_RED);CHKERRQ(ierr);
#endif
    }
    if (rank) { /* receive value from left */
      ierr = MPI_Recv(&tmp,1,MPI_DOUBLE,rank-1,tag1,comm,&status);CHKERRQ(ierr);
      ierr = MPI_Recv(&xgtmp,1,MPI_DOUBLE,rank-1,tag1,comm,&status);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
      ierr = DrawLine(draw,xgtmp,tmp,xg[0],array[j],DRAW_RED);CHKERRQ(ierr);
#else
      ierr = DrawLine(draw,xgtmp,tmp,PetscRealPart(xg[0]),PetscRealPart(array[j]),
                      DRAW_RED);CHKERRQ(ierr);
#endif
    }
    if (rank == size-1 && periodic) {
      ierr = MPI_Recv(&tmp,1,MPI_DOUBLE,0,tag2,comm,&status);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
      ierr = DrawLine(draw,xg[n-2],array[j+step*(n-1)],xg[n-1],tmp,DRAW_RED);CHKERRQ(ierr);
#else
      ierr = DrawLine(draw,PetscRealPart(xg[n-2]),PetscRealPart(array[j+step*(n-1)]),
                      PetscRealPart(xg[n-1]),tmp,DRAW_RED);CHKERRQ(ierr);
#endif
    }
    ierr = DrawSynchronizedFlush(draw);CHKERRQ(ierr);
    ierr = DrawPause(draw);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xcoor,&xg);CHKERRQ(ierr);
  ierr = VecRestoreArray(xin,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

