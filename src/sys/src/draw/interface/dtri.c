/*$Id: dtri.c,v 1.43 2000/05/05 22:13:25 balay Exp bsmith $*/
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawTriangle"></a>*/"DrawTriangle" 
/*@
   DrawTriangle - Draws a triangle  onto a drawable.

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  x1,y1,x2,y2,x3,y3 - the coordinates of the vertices
-  c1,c2,c3 - the colors of the three vertices in the same order as the xi,yi

   Level: beginner

.keywords: draw, triangle
@*/
int DrawTriangle(Draw draw,PetscReal x1,PetscReal y_1,PetscReal x2,PetscReal y2,PetscReal x3,PetscReal y3,
                 int c1,int c2,int c3)
{
  int        ierr;
  PetscTruth isnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  ierr = PetscTypeCompare((PetscObject)draw,DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  ierr = (*draw->ops->triangle)(draw,x1,y_1,x2,y2,x3,y3,c1,c2,c3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ /*<a name="DrawScalePopup"></a>*/"DrawScalePopup" 
/*@
       DrawScalePopup - Draws a contour scale window. 

     Collective on Draw

  Input Parameters:
+    popup - the window (often a window obtained via DrawGetPopup()
.    min - minimum value being plotted
-    max - maximum value being plotted

  Level: intermediate

  Notes:
     All processors that share the draw MUST call this routine

@*/
int DrawScalePopup(Draw popup,PetscReal min,PetscReal max)
{
  PetscReal xl = 0.0,yl = 0.0,xr = 1.0,yr = 1.0,value;
  int       i,c = DRAW_BASIC_COLORS,rank,ierr;
  char      string[32];
  MPI_Comm  comm;

  PetscFunctionBegin;
  ierr = DrawCheckResizedWindow(popup);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)popup,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (rank) PetscFunctionReturn(0);

  for (i=0; i<10; i++) {
    ierr = DrawRectangle(popup,xl,yl,xr,yr,c,c,c,c);CHKERRQ(ierr);
    yl += .1; yr += .1; c = (int)((double)c + (245.-DRAW_BASIC_COLORS)/9.);
  }
  for (i=0; i<10; i++) {
    value = min + i*(max-min)/9.0;
    /* look for a value that should be zero, but is not due to round-off */
    if (PetscAbsDouble(value) < 1.e-10 && max-min > 1.e-6) value = 0.0;
    sprintf(string,"%g",value);
    ierr = DrawString(popup,.2,.02 + i/10.0,DRAW_BLACK,string);CHKERRQ(ierr);
  }
  ierr = DrawSetTitle(popup,"Contour Scale");CHKERRQ(ierr);
  ierr = DrawFlush(popup);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  int        m,n;
  PetscReal  *x,*y,min,max;
  Scalar     *v;
  PetscTruth showgrid;
} ZoomCtx;

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawTensorContour_Zoom"></a>*/"DrawTensorContour_Zoom" 
static int DrawTensorContour_Zoom(Draw win,void *dctx)
{
  int     i,ierr;
  ZoomCtx *ctx = (ZoomCtx*)dctx;

  PetscFunctionBegin;
  ierr = DrawTensorContourPatch(win,ctx->m,ctx->n,ctx->x,ctx->y,ctx->max,ctx->min,ctx->v);CHKERRQ(ierr);
  if (ctx->showgrid) {
    for (i=0; i<ctx->m; i++) {
      ierr = DrawLine(win,ctx->x[i],ctx->y[0],ctx->x[i],ctx->y[ctx->n-1],DRAW_BLACK);CHKERRQ(ierr);
    }
    for (i=0; i<ctx->n; i++) {
      ierr = DrawLine(win,ctx->x[0],ctx->y[i],ctx->x[ctx->m-1],ctx->y[i],DRAW_BLACK);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawTensorContour"></a>*/"DrawTensorContour" 
/*@C
   DrawTensorContour - Draws a contour plot for a two-dimensional array
   that is stored as a PETSc vector.

   Collective on Draw, but Draw must be sequential

   Input Parameters:
+   win   - the window to draw in
.   m,n   - the global number of mesh points in the x and y directions
.   xi,yi - the locations of the global mesh points (optional, use PETSC_NULL
            to indicate uniform spacing on [0,1])
-   V     - the values

   Options Database Keys:
+  -draw_x_shared_colormap - Indicates use of private colormap
-  -draw_contour_grid - Draws grid contour

   Level: intermediate

.keywords: Draw, tensor, contour, vector

.seealso: DrawTensorContourPatch()

@*/
int DrawTensorContour(Draw win,int m,int n,const PetscReal xi[],const PetscReal yi[],Scalar *v)
{
  int           N = m*n,ierr;
  PetscTruth    isnull;
  Draw          popup;
  MPI_Comm      comm;
  int           xin=1,yin=1,i,size;
  PetscReal     h;
  ZoomCtx       ctx;

  PetscFunctionBegin;
  ierr = DrawIsNull(win,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);
  ierr = PetscObjectGetComm((PetscObject)win,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(1,1,"May only be used with single processor Draw");

  if (N <= 0) {
    SETERRQ2(1,1,"n %d and m %d must be positive",m,n);
  }

  /* create scale window */
  ierr = DrawGetPopup(win,&popup);CHKERRQ(ierr);
  ierr = DrawCheckResizedWindow(win);CHKERRQ(ierr);

  ctx.v   = v;
  ctx.m   = m;
  ctx.n   = n;
  ctx.max = ctx.min = PetscRealPart(v[0]);
  for (i=0; i<N; i++) {
    if (ctx.max < PetscRealPart(ctx.v[i])) ctx.max = PetscRealPart(ctx.v[i]);
    if (ctx.min > PetscRealPart(ctx.v[i])) ctx.min = PetscRealPart(ctx.v[i]);
  }
  if (ctx.max - ctx.min < 1.e-7) {ctx.min -= 5.e-8; ctx.max += 5.e-8;}

  /* Draw the scale window */
  if (popup) {ierr = DrawScalePopup(popup,ctx.min,ctx.max);CHKERRQ(ierr);}

  ierr = OptionsHasName(PETSC_NULL,"-draw_contour_grid",&ctx.showgrid);CHKERRQ(ierr);

  /* fill up x and y coordinates */
  if (!xi) {
    xin      = 0; 
    ctx.x    = (PetscReal*)PetscMalloc(ctx.m*sizeof(PetscReal));CHKPTRQ(ctx.x);
    h        = 1.0/(ctx.m-1);
    ctx.x[0] = 0.0;
    for (i=1; i<ctx.m; i++) ctx.x[i] = ctx.x[i-1] + h;
  } else {
    ctx.x = (PetscReal*)xi;
  }
  if (!yi) {
    yin      = 0; 
    ctx.y    = (PetscReal*)PetscMalloc(ctx.n*sizeof(PetscReal));CHKPTRQ(ctx.y);
    h        = 1.0/(ctx.n-1);
    ctx.y[0] = 0.0;
    for (i=1; i<ctx.n; i++) ctx.y[i] = ctx.y[i-1] + h;
  } else {
    ctx.y = (PetscReal *)yi;
  }

  ierr = DrawZoom(win,(int (*)(Draw,void *))DrawTensorContour_Zoom,&ctx);CHKERRQ(ierr);
    
  if (!xin) {ierr = PetscFree(ctx.x);CHKERRQ(ierr);}
  if (!yin) {ierr = PetscFree(ctx.y);CHKERRQ(ierr);}

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawTensorContourPatch"></a>*/"DrawTensorContourPatch" 
/*@
   DrawTensorContourPatch - Draws a rectangular patch of a contour plot 
   for a two-dimensional array.

   Not Collective

   Input Parameters:
+  win - the window to draw in
.  m,n - the number of local mesh points in the x and y direction
.  x,y - the locations of the local mesh points
.  max,min - the maximum and minimum value in the entire contour
-  v - the data

   Options Database Keys:
.  -draw_x_shared_colormap - Activates private colormap

   Level: advanced

   Note: 
   This is a lower level support routine, usually the user will call
   DrawTensorContour(). 

.keywords: Draw, tensor, contour, vector

.seealso: DrawTensorContour()

@*/
int DrawTensorContourPatch(Draw draw,int m,int n,PetscReal *x,PetscReal *y,PetscReal max,PetscReal min,Scalar *v)
{
  int           c1,c2,c3,c4,i,j,ierr;
  PetscReal     x1,x2,x3,x4,y_1,y2,y3,y4,scale;

  PetscFunctionBegin;
  scale = (245.0 - DRAW_BASIC_COLORS)/(max - min);

  /* Draw the contour plot patch */
  for (j=0; j<n-1; j++) {
    for (i=0; i<m-1; i++) {
#if !defined(PETSC_USE_COMPLEX)
      x1 = x[i];  y_1 = y[j];  c1 = (int)(DRAW_BASIC_COLORS + scale*(v[i+j*m] - min));
      x2 = x[i+1];y2 = y_1;    c2 = (int)(DRAW_BASIC_COLORS + scale*(v[i+j*m+1]-min));
      x3 = x2;    y3 = y[j+1];c3 = (int)(DRAW_BASIC_COLORS + scale*(v[i+j*m+1+m]-min));
      x4 = x1;    y4 = y3;    c4 = (int)(DRAW_BASIC_COLORS + scale*(v[i+j*m+m]-min));
#else
      x1 = x[i];  y_1 = y[j];  c1 = (int)(DRAW_BASIC_COLORS + scale*PetscRealPart(v[i+j*m]-min));
      x2 = x[i+1];y2 = y_1;    c2 = (int)(DRAW_BASIC_COLORS + scale*PetscRealPart(v[i+j*m+1]-min));
      x3 = x2;    y3 = y[j+1];c3 = (int)(DRAW_BASIC_COLORS + scale*PetscRealPart(v[i+j*m+1+m]-min));
      x4 = x1;    y4 = y3;    c4 = (int)(DRAW_BASIC_COLORS + scale*PetscRealPart(v[i+j*m+m]-min));
#endif
      ierr = DrawTriangle(draw,x1,y_1,x2,y2,x3,y3,c1,c2,c3);CHKERRQ(ierr);
      ierr = DrawTriangle(draw,x1,y_1,x3,y3,x4,y4,c1,c3,c4);CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}
