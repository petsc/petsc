#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: gr2.c,v 1.8 1999/02/25 22:56:46 bsmith Exp bsmith $";
#endif

/* 
   Plots vectors obtained with DACreate2d()
*/

#include "src/dm/da/daimpl.h"      /*I  "da.h"   I*/


EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "VecView_MPI_Draw_DA2d"
int VecView_MPI_Draw_DA2d(Vec xin,Viewer viewer)
{
  DA             da,dac,dag;
  int            i,rank,size,ierr,igstart,N,step,s,M;
  int            istart,isize,j,jgstart;
  int            c1, c2, c3, c4, k,id,n,m,*lx,*ly;
  double         coors[4],ymin,ymax,min,max,xmin,xmax;
  double         x1, x2, x3, x4, y_1, y2, y3, y4,scale;
  double         minw,maxw;
  Scalar         *v,*xy;
  Draw           draw,popup;
  PetscTruth     isnull;
  MPI_Comm       comm;
  Vec            xlocal,xcoor,xcoorl;
  DAPeriodicType periodic;
  DAStencilType  st;

  PetscFunctionBegin;
  ierr = ViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = DrawIsNull(draw,&isnull); CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);

  ierr = PetscObjectQuery((PetscObject)xin,"DA",(PetscObject*) &da);CHKERRQ(ierr);
  if (!da) SETERRQ(1,1,"Vector not generated from a DA");

  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);

  ierr = DAGetInfo(da,0,&M,&N,0,&m,&n,0,0,&s,&periodic,&st);CHKERRQ(ierr);
  ierr = DAGetOwnershipRange(da,&lx,&ly,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscObjectQuery((PetscObject)da,"GraphicsGhosted",(PetscObject*) &xlocal);CHKERRQ(ierr);
  if (!xlocal) {
    if (periodic != DA_NONPERIODIC || s != 1 || st != DA_STENCIL_BOX) {
      /* 
         if original da is not of stencil width one, or periodic or not a box stencil then
         create a special DA to handle one level of ghost points for graphics
      */
      ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_BOX,M,N,m,n,1,s,lx,ly,&dac);CHKERRQ(ierr); 
      PLogInfo(da,"VecView_MPI_Draw_DA2d:Creating auxilary DA for managing graphics ghost points\n");
    } else {
      /* otherwise we can use the da we already have */
      dac = da;
    }
    /* create local vector for holding ghosted values used in graphics */
    ierr = DACreateLocalVector(dac,&xlocal);CHKERRQ(ierr);
    if (dac != da) {ierr = PetscObjectDereference((PetscObject)dac);CHKERRQ(ierr);}
    ierr = PetscObjectCompose((PetscObject)da,"GraphicsGhosted",(PetscObject)xlocal);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)xlocal);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectQuery((PetscObject)xlocal,"DA",(PetscObject*) &dac);CHKERRQ(ierr);
  }

  /*
      Get local (ghosted) values of vector
  */
  ierr = DAGlobalToLocalBegin(dac,xin,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(dac,xin,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = VecGetArray(xlocal,&v); CHKERRQ(ierr);


  /* get coordinates of nodes */
  ierr = DAGetCoordinates(da,&xcoor);CHKERRQ(ierr);
  if (!xcoor) {
    ierr = DACreateUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
    ierr = DAGetCoordinates(da,&xcoor);CHKERRQ(ierr);
  }

  /*
      Determine the min and max  coordinates in plot 
  */
  ierr = VecStrideMin(xcoor,0,PETSC_NULL,&xmin);CHKERRQ(ierr);
  ierr = VecStrideMax(xcoor,0,PETSC_NULL,&xmax);CHKERRQ(ierr);
  ierr = VecStrideMin(xcoor,1,PETSC_NULL,&ymin);CHKERRQ(ierr);
  ierr = VecStrideMax(xcoor,1,PETSC_NULL,&ymax);CHKERRQ(ierr);
  coors[0] = xmin - .05*(xmax- xmin); coors[2] = xmax + .05*(xmax - xmin);
  coors[1] = ymin - .05*(ymax- ymin); coors[3] = ymax + .05*(ymax - ymin);
  PLogInfo(da,"VecView_MPI_Draw_DA2d:Preparing DA 2d contour plot coordinates %g %g %g %g\n",
           coors[0],coors[1],coors[2],coors[3]);

  /*
       get local ghosted version of coordinates 
  */
  ierr = PetscObjectQuery((PetscObject)da,"GraphicsCoordinateGhosted",(PetscObject*) &xcoorl);CHKERRQ(ierr);
  if (!xcoorl) {
    /* create DA to get local version of graphics */
    ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_BOX,M,N,m,n,2,1,lx,ly,&dag);CHKERRQ(ierr); 
    PLogInfo(dag,"VecView_MPI_Draw_DA2d:Creating auxilary DA for managing graphics coordinates ghost points\n");
    ierr = DACreateLocalVector(dag,&xcoorl);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)da,"GraphicsCoordinateGhosted",(PetscObject)xcoorl);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)dag);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)xcoorl);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectQuery((PetscObject)xcoorl,"DA",(PetscObject*) &dag);CHKERRQ(ierr);
  }
  ierr = DAGlobalToLocalBegin(dag,xcoor,INSERT_VALUES,xcoorl);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(dag,xcoor,INSERT_VALUES,xcoorl);CHKERRQ(ierr);
  ierr = VecGetArray(xcoorl,&xy);CHKERRQ(ierr);
  
  MPI_Comm_rank(comm,&rank);
  ierr = DAGetInfo(dac,0,&M,&N,0,0,0,0,&step,0,&periodic,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(dac,&igstart,&jgstart,0,&m,&n,0);CHKERRQ(ierr);
  ierr = DAGetCorners(dac,&istart,0,0,&isize,0,0);CHKERRQ(ierr);

  for ( k=0; k<step; k++ ) {
    ierr = ViewerDrawGetDraw(viewer,k,&draw);CHKERRQ(ierr);
    ierr = DrawCheckResizedWindow(draw);CHKERRQ(ierr);
    ierr = DrawSynchronizedClear(draw); CHKERRQ(ierr);

    /*
        Determine the min and max coordinate in plot 
    */
    ierr = VecStrideMin(xin,k,PETSC_NULL,&min);CHKERRQ(ierr);
    ierr = VecStrideMax(xin,k,PETSC_NULL,&max);CHKERRQ(ierr);
    if (min + 1.e-10 > max) {
      min -= 1.e-5;
      max += 1.e-5;
    }

    if (!rank) {
      char *title;

      ierr = DAGetFieldName(da,k,&title);CHKERRQ(ierr);
      ierr = DrawSetTitle(draw,title);CHKERRQ(ierr);
    }
    ierr = DrawSetCoordinates(draw,coors[0],coors[1],coors[2],coors[3]);CHKERRQ(ierr);
    PLogInfo(da,"VecView_MPI_Draw_DA2d:DA 2d contour plot min %g max %g\n",min,max);

    scale = (245.0 - DRAW_BASIC_COLORS)/(max - min);

    /* Draw the contour plot patch */
    for ( j=0; j<n-1; j++ ) {
      for ( i=0; i<m-1; i++ ) {
#if !defined(USE_PETSC_COMPLEX)
        id = i+j*m;    x1 = xy[2*id];y_1 = xy[2*id+1];c1 = (int)(DRAW_BASIC_COLORS+scale*(v[k+step*id]-min));
        id = i+j*m+1;  x2 = xy[2*id];y2  = y_1;       c2 = (int)(DRAW_BASIC_COLORS+scale*(v[k+step*id]-min));
        id = i+j*m+1+m;x3 = x2;      y3  = xy[2*id+1];c3 = (int)(DRAW_BASIC_COLORS+scale*(v[k+step*id]-min));
        id = i+j*m+m;  x4 = x1;      y4  = y3;        c4 = (int)(DRAW_BASIC_COLORS+scale*(v[k+step*id]-min));
#else
        id = i+j*m;    x1 = PetscReal(xy[2*id]);y_1 = PetscReal(xy[2*id+1]);c1 = (int)(DRAW_BASIC_COLORS+scale*(PetscReal(v[k+step*id])-min));
        id = i+j*m+1;  x2 = PetscReal(xy[2*id]);y2  = y_1;       c2 = (int)(DRAW_BASIC_COLORS+scale*(PetscReal(v[k+step*id])-min));
        id = i+j*m+1+m;x3 = x2;      y3  = PetscReal(xy[2*id+1]);c3 = (int)(DRAW_BASIC_COLORS+scale*(PetscReal(v[k+step*id])-min));
        id = i+j*m+m;  x4 = x1;      y4  = y3;        c4 = (int)(DRAW_BASIC_COLORS+scale*(PetscReal(v[k+step*id])-min));
#endif
        ierr = DrawTriangle(draw,x1,y_1,x2,y2,x3,y3,c1,c2,c3); CHKERRQ(ierr);
        ierr = DrawTriangle(draw,x1,y_1,x3,y3,x4,y4,c1,c3,c4); CHKERRQ(ierr);
      }
    }

    ierr = DrawGetPopup(draw,&popup); CHKERRQ(ierr);
    ierr = DrawScalePopup(popup,min,max); CHKERRQ(ierr);


    ierr = DrawSynchronizedFlush(draw); CHKERRQ(ierr);
    ierr = DrawPause(draw); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xcoorl,&xy);CHKERRQ(ierr);
  ierr = VecRestoreArray(xlocal,&v); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END




