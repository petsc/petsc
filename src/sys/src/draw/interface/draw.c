
#include "drawimpl.h"  /*I "draw.h" I*/
  
/*@
     DrawLine - Draws a line onto a drawable.

  Input Parameters:
.   ctx - the drawing context
.   xl,yl,xr,yr - the coordinates of the line endpoints
.   cl,cr - the colors of the two endpoints
@*/
int DrawLine(DrawCtx ctx,double xl,double yl,double xr,double yr,
               int cl, int cr)
{
  VALIDHEADER(ctx,DRAW_COOKIE);
  return (*ctx->ops->drawline)(ctx,xl,yl,xr,yr,cl,cr);
}

/*@

   DrawSetViewPort - Sets the portion of the window(page) that 
      draw routines will write to. 

  Input Parameters:
.  xl,yl,xr,yr - upper right and lower left corner of subwindow
                 These numbers must always be between 0.0 and 1.0.
                 Lower left corner is (0,0).
.  ctx - the drawing context
@*/
int DrawSetViewPort(DrawCtx ctx,double xl,double yl,double xr,double yr)
{
  VALIDHEADER(ctx,DRAW_COOKIE);
  ctx->port_xl = xl; ctx->port_yl = yl;
  ctx->port_xr = xr; ctx->port_yr = yr;
  return 0;
}

/*@
    DrawSetCoordinates - Sets the application coordinates of the 
          corners of the window (or page).

  Input Paramters:
.  ctx - the drawing object
.  xl,yl,xr,yr - the coordinates of the lower left corner and upper
                 right corner of the drawing region.
@*/
int DrawSetCoordinates(DrawCtx ctx,double xl,double yl,double xr, double yr)
{
  VALIDHEADER(ctx,DRAW_COOKIE);
  ctx->coor_xl = xl; ctx->coor_yl = yl;
  ctx->coor_xr = xr; ctx->coor_yr = yr;
  return 0;
}
/*@
   DrawSetDoubleBuffer - Sets a window to be double buffered. 

  Input Parameters:
.  ctx - the drawing context
@*/
int DrawSetDoubleBuffer(DrawCtx ctx)
{
  VALIDHEADER(ctx,DRAW_COOKIE);
  if (ctx->ops->doublebuff) return (*ctx->ops->doublebuff)(ctx);
  return 0;
}
/*@
   DrawFlush - Flushs graphical output.

  Input Parameters:
.  ctx - the drawing context
@*/
int DrawFlush(DrawCtx ctx)
{
  VALIDHEADER(ctx,DRAW_COOKIE);
  if (ctx->ops->flush) return (*ctx->ops->flush)(ctx);
  return 0;
}
