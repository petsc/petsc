#ifndef lint
static char vcid[] = "$Id: cmesh.c,v 1.38 1997/03/26 01:34:23 bsmith Exp bsmith $";
#endif

#include "src/draw/drawimpl.h"   /*I "draw.h" I*/
#include "vec.h"        /*I "vec.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawScalePopup" /* ADIC Ignore */
int DrawScalePopup(Draw popup,double min,double max)
{
  double   xl = 0.0, yl = 0.0, xr = 1.0, yr = 1.0,value;
  int      i,c = 32,rank;
  char     string[32];
  MPI_Comm comm;

  PetscObjectGetComm((PetscObject) popup,&comm);
  MPI_Comm_rank(comm,&rank);
  if (rank) return 0;

  for ( i=0; i<10; i++ ) {
    DrawRectangle(popup,xl,yl,xr,yr,c,c,c,c);
    yl += .1; yr += .1; c = (int) ((double) c + (200.-32.)/9.);
  }
  for ( i=0; i<10; i++ ) {
    value = -min + i*(max-min)/9.0;
    /* look for a value that should be zero, but is not due to round-off */
    if (PetscAbsDouble(value) < 1.e-10 && max-min > 1.e-6) value = 0.0;
    sprintf(string,"%g",value);
    DrawText(popup,.2,.02 + i/10.0,DRAW_BLACK,string);
  }
  DrawSetTitle(popup,"Contour Scale");
  DrawFlush(popup);
  return 0;
}


#undef __FUNC__  
#define __FUNC__ "DrawTensorContour" /* ADIC Ignore */
/*@
   DrawTensorContour - Draws a contour plot for a two-dimensional array
   that is stored as a PETSc vector.

   Input Parameters:
.   win - the window to draw in
.   m,n - the global number of mesh points in the x and y directions
.   x,y - the locations of the global mesh points (optional, use PETSC_NULL
          to indicate uniform spacing on [0,1])
.   V - the vector

   Options Database Keyes:
$  -draw_x_private_colormap
$  -draw_contour_grid


    Note: 
    This may be a basic enough function to be a graphics primative
    but at this time it uses DrawTriangle().

.keywords: Draw, tensor, contour, vector
@*/
int DrawTensorContour(Draw win,int m,int n,double *x,double *y,Vec V)
{
  int           xin = 1, yin = 1, i, N, rank, ierr;
  int           pause,showgrid;
  double        h, *v, min, max, scale = 1.0;
  Vec           W;
  IS            from, to;
  VecScatter    ctx;
  PetscObject   vobj = (PetscObject) win;
  Draw          popup;

  if (vobj->cookie == DRAW_COOKIE && vobj->type == DRAW_NULLWINDOW) return 0;
  MPI_Comm_rank(win->comm,&rank);

  /* move entire vector to first processor */
  if (rank == 0) {
    ierr = VecGetSize(V,&N); CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,N,&W); CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&from); CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&to); CHKERRQ(ierr);
  }
  else {
    ierr = VecCreateSeq(PETSC_COMM_SELF,0,&W); CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&from); CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&to); CHKERRQ(ierr);
  }
  PLogObjectParent(win,W); PLogObjectParent(win,from); PLogObjectParent(win,to);
  ierr = VecScatterCreate(V,from,W,to,&ctx); CHKERRQ(ierr);
  PLogObjectParent(win,ctx);
  ierr = VecScatterBegin(V,W,INSERT_VALUES,SCATTER_FORWARD,ctx); CHKERRQ(ierr);
  ierr = VecScatterEnd(V,W,INSERT_VALUES,SCATTER_FORWARD,ctx); CHKERRQ(ierr);
  ISDestroy(from); ISDestroy(to); VecScatterDestroy(ctx);

  /* create scale window */
  ierr = DrawCreatePopUp(win,&popup); CHKERRQ(ierr);

   
  if (rank == 0) {
#if !defined(PETSC_COMPLEX)
    double  xl = 0.0, yl = 0.0, xr = 1.0, yr = .1;

    ierr = VecGetArray(W,&v); CHKERRQ(ierr);

    /* Scale the color values between 32 and 256 */
    ierr = VecMax(W,0,&max); CHKERRQ(ierr); ierr = VecMin(W,0,&min); CHKERRQ(ierr);
    if (max - min < 1.e-7) {min -= 5.e-8; max += 5.e-8;}

    /* Draw the scale window */
    ierr = DrawScalePopup(popup,min,max); CHKERRQ(ierr);

    ierr = OptionsHasName(PETSC_NULL,"-draw_contour_grid",&showgrid);CHKERRQ(ierr);

    /* fill up x and y coordinates */
    if (!x) {
      xin = 0; 
      x = (double *) PetscMalloc( m*sizeof(double) ); CHKPTRQ(x);
      h = 1.0/(m-1);
      x[0] = 0.0;
      for ( i=1; i<m; i++ ) x[i] = x[i-1] + h;
    }
    if (!y) {
      yin = 0; 
      y = (double *) PetscMalloc( n*sizeof(double) ); CHKPTRQ(y);
      h = 1.0/(n-1);
      y[0] = 0.0;
      for ( i=1; i<n; i++ ) y[i] = y[i-1] + h;
    }

    /* Draw the contour plot */
    ierr = DrawGetPause(win,&pause); CHKERRA(ierr);
    while (1) {
      ierr = DrawTensorContourPatch(win,m,n,x,y,max,min,v);CHKERRQ(ierr);

      if (showgrid) {
        for ( i=0; i<m; i++ ) {
          DrawLine(win,x[i],y[0],x[i],y[n-1],DRAW_BLACK);
        }
        for ( i=0; i<n; i++ ) {
          DrawLine(win,x[0],y[i],x[m-1],y[i],DRAW_BLACK);
        }
      }

      if (pause == -1) {
        DrawButton  button;
        double      xc,yc;
    
        ierr = DrawGetCoordinates(win,&xl,&yl,&xr,&yr); CHKERRQ(ierr);

        ierr = DrawGetMouseButton(win,&button,&xc,&yc,0,0); CHKERRQ(ierr);
        if  (button == BUTTON_RIGHT) break;
        if (button == BUTTON_LEFT) scale = .5;
        else if (button == BUTTON_CENTER) scale = 2.;
         xl = scale*(xl  - xc) + xc;
         xr = scale*(xr  - xc) + xc;
         yl = scale*(yl  - yc) + yc;
         yr = scale*(yr  - yc) + yc;
         ierr = DrawSetCoordinates(win,xl,yl,xr,yr); CHKERRQ(ierr);
      } else {break;}
     
      ierr = DrawCheckResizedWindow(win); CHKERRQ(ierr);
      ierr = DrawClear(win); CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(W,&v); CHKERRQ(ierr);
    
    if (!xin) PetscFree(x); 
    if (!yin) PetscFree(y);
#endif
  }


  ierr = VecDestroy(W); CHKERRQ(ierr);

  return 0;
}
