
#include "drawimpl.h"
#include "vec.h"

/*@
   DrawTensorContour - Draws a coutour plot for a two-dimensional array
   that is stored as a PETSc vector.

   Input Parameters:
.   win - the window to draw in
.   m,n - the number of mesh points in the x and y directions
.   x,y - the locations of the mesh points, if null uses uniform [0,1]
.   v - the array of values

    Note: 
    This may be a basic enough function to be a graphics primative
    but at this time it uses DrawTriangle().

.keywords: Draw, tensor, contour, vector
@*/
int DrawTensorContour(DrawCtx win,int m,int n,double *x,double *y,Vec V)
{
  int           xin = 1, yin = 1, c1, c2, c3, c4, i, N, mytid;
  int           ierr;
  double        h,x1,x2,x3,x4,y1,y2,y3,y4,*v,min,max;
  Scalar        scale;
  Vec           W;
  IS            from,to;
  VecScatterCtx ctx;
  PetscObject   vobj = (PetscObject) win;

  if (vobj->cookie == DRAW_COOKIE && vobj->type == NULLWINDOW) return 0;

  MPI_Comm_rank(win->comm,&mytid);

  /* move entire vector to first processor */
  if (mytid == 0) {
    VecGetSize(V,&N);
    ierr = VecCreateSequential(MPI_COMM_SELF,N,&W); CHKERRQ(ierr);
    ierr = ISCreateStrideSequential(MPI_COMM_SELF,N,0,1,&from); CHKERRQ(ierr);
    ierr = ISCreateStrideSequential(MPI_COMM_SELF,N,0,1,&to); CHKERRQ(ierr);
  }
  else {
    ierr = VecCreateSequential(MPI_COMM_SELF,0,&W); CHKERRQ(ierr);
    ierr = ISCreateStrideSequential(MPI_COMM_SELF,0,0,1,&from); CHKERRQ(ierr);
    ierr = ISCreateStrideSequential(MPI_COMM_SELF,0,0,1,&to); CHKERRQ(ierr);
  }
  PLogObjectParent(win,W);PLogObjectParent(win,from);PLogObjectParent(win,to);
  ierr = VecScatterCtxCreate(V,from,W,to,&ctx); CHKERRQ(ierr);
  PLogObjectParent(win,ctx);
  ierr = VecScatterBegin(V,W,INSERTVALUES,SCATTERALL,ctx); 
  CHKERRQ(ierr);
  ierr = VecScatterEnd(V,W,INSERTVALUES,SCATTERALL,ctx); CHKERRQ(ierr);
  ISDestroy(from); ISDestroy(to); VecScatterCtxDestroy(ctx);

  if (mytid == 0) {
    VecGetArray(W,&v);

    /* scale the color values between 32 and 256 */
    VecMax(W,0,&max); VecMin(W,0,&min);
    scale = (200.0 - 32.0)/(max - min);
    VecScale(&scale,W);

    if (!x) {
      xin = 0; 
      x = (double *) PETSCMALLOC( m*sizeof(double) ); CHKPTRQ(x);
      h = 1.0/(m-1);
      x[0] = 0.0;
      for ( i=1; i<m; i++ ) x[i] = x[i-1] + h;
    }
    if (!y) {
      yin = 0; 
      y = (double *) PETSCMALLOC( n*sizeof(double) ); CHKPTRQ(y);
      h = 1.0/(n-1);
      y[0] = 0.0;
      for ( i=1; i<n; i++ ) y[i] = y[i-1] + h;
    }

    for ( i=0; i<N; i++ ) {
      if (!((i+1) % m) ) continue;  /* last column on right is skipped */
      if (i+m+1 >= N) continue;

      x1 = x[i % m];     y1 = y[i/m];        c1 = (int) (32. + v[i]);
      x2 = x[(i+1) % m]; y2 = y1;            c2 = (int) (32. + v[i+1]);
      x3 = x2;           y3 = y[(i/m) + 1];  c3 = (int) (32. + v[i+m+1]);
      x4 = x1;           y4 = y3;            c4 = (int) (32. + v[i+m]);

      DrawTriangle(win,x1,y1,x2,y2,x3,y3,c1,c2,c3);
      DrawTriangle(win,x1,y1,x3,y3,x4,y4,c1,c3,c4);
    }
    VecRestoreArray(W,&v);
    if (!xin) PETSCFREE(x); 
    if (!yin) PETSCFREE(y);
  }
  VecDestroy(W);
  return 0;
}
