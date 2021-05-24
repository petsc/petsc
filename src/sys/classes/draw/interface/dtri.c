
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <petsc/private/drawimpl.h>  /*I "petscdraw.h" I*/

/*@
   PetscDrawTriangle - PetscDraws a triangle  onto a drawable.

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  x1,y1,x2,y2,x3,y3 - the coordinates of the vertices
-  c1,c2,c3 - the colors of the three vertices in the same order as the xi,yi

   Level: beginner

.seealso: PetscDrawLine(), PetscDrawRectangle(), PetscDrawEllipse(), PetscDrawMarker(), PetscDrawPoint(), PetscDrawArrow()
@*/
PetscErrorCode  PetscDrawTriangle(PetscDraw draw,PetscReal x1,PetscReal y_1,PetscReal x2,PetscReal y2,PetscReal x3,PetscReal y3,int c1,int c2,int c3)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (!draw->ops->triangle) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"This draw type %s does not support drawing triangles",((PetscObject)draw)->type_name);
  ierr = (*draw->ops->triangle)(draw,x1,y_1,x2,y2,x3,y3,c1,c2,c3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscDrawScalePopup - draws a contour scale window.

   Collective on PetscDraw

   Input Parameters:
+  popup - the window (often a window obtained via PetscDrawGetPopup()
.  min - minimum value being plotted
-  max - maximum value being plotted

   Level: intermediate

   Notes:
    All processors that share the draw MUST call this routine

.seealso: PetscDrawGetPopup(), PetscDrawTensorContour()

@*/
PetscErrorCode  PetscDrawScalePopup(PetscDraw popup,PetscReal min,PetscReal max)
{
  PetscBool      isnull;
  PetscReal      xl = 0.0,yl = 0.0,xr = 1.0,yr = 1.0;
  PetscMPIInt    rank;
  PetscErrorCode ierr;
  int            i;
  char           string[32];

  PetscFunctionBegin;
  if (!popup) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(popup,PETSC_DRAW_CLASSID,1);
  ierr = PetscDrawIsNull(popup,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)popup),&rank);CHKERRMPI(ierr);

  ierr = PetscDrawCheckResizedWindow(popup);CHKERRQ(ierr);
  ierr = PetscDrawClear(popup);CHKERRQ(ierr);
  ierr = PetscDrawSetTitle(popup,"Contour Scale");CHKERRQ(ierr);
  ierr = PetscDrawSetCoordinates(popup,xl,yl,xr,yr);CHKERRQ(ierr);
  ierr = PetscDrawCollectiveBegin(popup);CHKERRQ(ierr);
  if (!rank) {
    for (i=0; i<10; i++) {
      int c = PetscDrawRealToColor((PetscReal)i/9,0,1);
      ierr = PetscDrawRectangle(popup,xl,yl,xr,yr,c,c,c,c);CHKERRQ(ierr);
      yl += 0.1;
    }
    for (i=0; i<10; i++) {
      PetscReal value = min + i*(max-min)/9;
      /* look for a value that should be zero, but is not due to round-off */
      if (PetscAbsReal(value) < 1.e-10 && max-min > 1.e-6) value = 0.0;
      ierr = PetscSNPrintf(string,sizeof(string),"%18.16e",(double)value);CHKERRQ(ierr);
      ierr = PetscDrawString(popup,0.2,0.02+i/10.0,PETSC_DRAW_BLACK,string);CHKERRQ(ierr);
    }
  }
  ierr = PetscDrawCollectiveEnd(popup);CHKERRQ(ierr);
  ierr = PetscDrawFlush(popup);CHKERRQ(ierr);
  ierr = PetscDrawSave(popup);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  int       m,n;
  PetscReal *x,*y,min,max,*v;
  PetscBool showgrid;
} ZoomCtx;

static PetscErrorCode PetscDrawTensorContour_Zoom(PetscDraw win,void *dctx)
{
  PetscErrorCode ierr;
  int            i;
  ZoomCtx        *ctx = (ZoomCtx*)dctx;

  PetscFunctionBegin;
  ierr = PetscDrawTensorContourPatch(win,ctx->m,ctx->n,ctx->x,ctx->y,ctx->min,ctx->max,ctx->v);CHKERRQ(ierr);
  if (ctx->showgrid) {
    for (i=0; i<ctx->m; i++) {
      ierr = PetscDrawLine(win,ctx->x[i],ctx->y[0],ctx->x[i],ctx->y[ctx->n-1],PETSC_DRAW_BLACK);CHKERRQ(ierr);
    }
    for (i=0; i<ctx->n; i++) {
      ierr = PetscDrawLine(win,ctx->x[0],ctx->y[i],ctx->x[ctx->m-1],ctx->y[i],PETSC_DRAW_BLACK);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawTensorContour - PetscDraws a contour plot for a two-dimensional array
   that is stored as a PETSc vector.

   Collective on PetscDraw, but PetscDraw must be sequential

   Input Parameters:
+   draw  - the draw context
.   m,n   - the global number of mesh points in the x and y directions
.   xi,yi - the locations of the global mesh points (optional, use NULL
            to indicate uniform spacing on [0,1])
-   V     - the values

   Options Database Keys:
+  -draw_x_shared_colormap - Indicates use of private colormap
-  -draw_contour_grid - PetscDraws grid contour

   Level: intermediate

.seealso: PetscDrawTensorContourPatch(), PetscDrawScalePopup()

@*/
PetscErrorCode  PetscDrawTensorContour(PetscDraw draw,int m,int n,const PetscReal xi[],const PetscReal yi[],PetscReal *v)
{
  PetscErrorCode ierr;
  int            N = m*n;
  PetscBool      isnull;
  PetscDraw      popup;
  int            xin=1,yin=1,i;
  PetscMPIInt    size;
  PetscReal      h;
  ZoomCtx        ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)draw),&size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"May only be used with single processor PetscDraw");
  if (N <= 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"n %d and m %d must be positive",m,n);

  ctx.v   = v;
  ctx.m   = m;
  ctx.n   = n;
  ctx.max = ctx.min = v[0];
  for (i=0; i<N; i++) {
    if (ctx.max < ctx.v[i]) ctx.max = ctx.v[i];
    if (ctx.min > ctx.v[i]) ctx.min = ctx.v[i];
  }
  if (ctx.max - ctx.min < 1.e-7) {ctx.min -= 5.e-8; ctx.max += 5.e-8;}

  /* PetscDraw the scale window */
  ierr = PetscDrawGetPopup(draw,&popup);CHKERRQ(ierr);
  ierr = PetscDrawScalePopup(popup,ctx.min,ctx.max);CHKERRQ(ierr);

  ctx.showgrid = PETSC_FALSE;
  ierr = PetscOptionsGetBool(((PetscObject)draw)->options,NULL,"-draw_contour_grid",&ctx.showgrid,NULL);CHKERRQ(ierr);

  /* fill up x and y coordinates */
  if (!xi) {
    xin      = 0;
    ierr     = PetscMalloc1(ctx.m,&ctx.x);CHKERRQ(ierr);
    h        = 1.0/(ctx.m-1);
    ctx.x[0] = 0.0;
    for (i=1; i<ctx.m; i++) ctx.x[i] = ctx.x[i-1] + h;
  } else ctx.x = (PetscReal*)xi;

  if (!yi) {
    yin      = 0;
    ierr     = PetscMalloc1(ctx.n,&ctx.y);CHKERRQ(ierr);
    h        = 1.0/(ctx.n-1);
    ctx.y[0] = 0.0;
    for (i=1; i<ctx.n; i++) ctx.y[i] = ctx.y[i-1] + h;
  } else ctx.y = (PetscReal*)yi;

  ierr = PetscDrawZoom(draw,PetscDrawTensorContour_Zoom,&ctx);CHKERRQ(ierr);

  if (!xin) {ierr = PetscFree(ctx.x);CHKERRQ(ierr);}
  if (!yin) {ierr = PetscFree(ctx.y);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
   PetscDrawTensorContourPatch - PetscDraws a rectangular patch of a contour plot
   for a two-dimensional array.

   Not Collective

   Input Parameters:
+  draw - the draw context
.  m,n - the number of local mesh points in the x and y direction
.  x,y - the locations of the local mesh points
.  min,max - the minimum and maximum value in the entire contour
-  v - the data

   Options Database Keys:
.  -draw_x_shared_colormap - Activates private colormap

   Level: advanced

   Note:
   This is a lower level support routine, usually the user will call
   PetscDrawTensorContour().

.seealso: PetscDrawTensorContour()

@*/
PetscErrorCode  PetscDrawTensorContourPatch(PetscDraw draw,int m,int n,PetscReal *x,PetscReal *y,PetscReal min,PetscReal max,PetscReal *v)
{
  PetscErrorCode ierr;
  int            c1,c2,c3,c4,i,j;
  PetscReal      x1,x2,x3,x4,y_1,y2,y3,y4;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  /* PetscDraw the contour plot patch */
  for (j=0; j<n-1; j++) {
    for (i=0; i<m-1; i++) {
      x1 = x[i];   y_1 = y[j];  c1 = PetscDrawRealToColor(v[i+j*m],min,max);
      x2 = x[i+1]; y2 = y_1;    c2 = PetscDrawRealToColor(v[i+j*m+1],min,max);
      x3 = x2;     y3 = y[j+1]; c3 = PetscDrawRealToColor(v[i+j*m+1+m],min,max);
      x4 = x1;     y4 = y3;     c4 = PetscDrawRealToColor(v[i+j*m+m],min,max);

      ierr = PetscDrawTriangle(draw,x1,y_1,x2,y2,x3,y3,c1,c2,c3);CHKERRQ(ierr);
      ierr = PetscDrawTriangle(draw,x1,y_1,x3,y3,x4,y4,c1,c3,c4);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
