#ifndef lint
static char vcid[] = "$Id: cmesh.c,v 1.29 1996/08/15 12:45:07 bsmith Exp bsmith $";
#endif

#include "src/draw/drawimpl.h"   /*I "draw.h" I*/
#include "vec.h"        /*I "vec.h" I*/

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
  int           xin = 1, yin = 1, c1, c2, c3, c4, i, N, rank, ierr;
  int           pause,showgrid;
  double        h, x1, x2, x3, x4, y1, y2, y3, y4, *v, min, max;
  Scalar        scale;
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
    ierr = VecCreateSeq(MPI_COMM_SELF,N,&W); CHKERRQ(ierr);
    ierr = ISCreateStride(MPI_COMM_SELF,N,0,1,&from); CHKERRQ(ierr);
    ierr = ISCreateStride(MPI_COMM_SELF,N,0,1,&to); CHKERRQ(ierr);
  }
  else {
    ierr = VecCreateSeq(MPI_COMM_SELF,0,&W); CHKERRQ(ierr);
    ierr = ISCreateStride(MPI_COMM_SELF,0,0,1,&from); CHKERRQ(ierr);
    ierr = ISCreateStride(MPI_COMM_SELF,0,0,1,&to); CHKERRQ(ierr);
  }
  PLogObjectParent(win,W); PLogObjectParent(win,from); PLogObjectParent(win,to);
  ierr = VecScatterCreate(V,from,W,to,&ctx); CHKERRQ(ierr);
  PLogObjectParent(win,ctx);
  ierr = VecScatterBegin(V,W,INSERT_VALUES,SCATTER_ALL,ctx); CHKERRQ(ierr);
  ierr = VecScatterEnd(V,W,INSERT_VALUES,SCATTER_ALL,ctx); CHKERRQ(ierr);
  ISDestroy(from); ISDestroy(to); VecScatterDestroy(ctx);

  /* create scale window */
  ierr = DrawCreatePopUp(win,&popup); CHKERRQ(ierr);

   
  if (rank == 0) {
#if !defined(PETSC_COMPLEX)
    double  xl = 0.0, yl = 0.0, xr = 1.0, yr = .1;
    int     c = 32;
    char    string[16];

    ierr = VecGetArray(W,&v); CHKERRQ(ierr);

    /* Scale the color values between 32 and 256 */
    ierr = VecMax(W,0,&max); CHKERRQ(ierr); ierr = VecMin(W,0,&min); CHKERRQ(ierr);
    scale = (200.0 - 32.0)/(max - min);
    min = -min; ierr = VecShift(&min,W); CHKERRQ(ierr);
    ierr = VecScale(&scale,W); CHKERRQ(ierr);

    /* Draw the scale window */
    for ( i=0; i<10; i++ ) {
      DrawRectangle(popup,xl,yl,xr,yr,c,c,c,c);
      yl += .1; yr += .1; c = (int) ((double) c + (200.-32.)/9.);
    }
    for ( i=0; i<10; i++ ) {
      sprintf(string,"%g",-min + i*(max-min)/9.0);
      DrawText(popup,.2,.02 + i/10.0,DRAW_BLACK,string);
    }
    DrawSetTitle(popup,"Contour Scale");
    DrawFlush(popup);

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
      for ( i=0; i<N; i++ ) {
        if (!((i+1) % m) ) continue;  /* last column on right is skipped */
        if (i+m+1 >= N) continue;

        x1 = x[i % m];     y1 = y[i/m];        c1 = (int) (32. + v[i]);
        x2 = x[(i+1) % m]; y2 = y1;            c2 = (int) (32. + v[i+1]);
        x3 = x2;           y3 = y[(i/m) + 1];  c3 = (int) (32. + v[i+m+1]);
        x4 = x1;           y4 = y3;            c4 = (int) (32. + v[i+m]);
 
        ierr = DrawTriangle(win,x1,y1,x2,y2,x3,y3,c1,c2,c3); CHKERRQ(ierr);
        ierr = DrawTriangle(win,x1,y1,x3,y3,x4,y4,c1,c3,c4); CHKERRQ(ierr);
      }

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
