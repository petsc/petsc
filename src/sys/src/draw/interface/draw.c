#ifndef lint
static char vcid[] = "$Id: draw.c,v 1.16 1995/09/30 19:30:26 bsmith Exp bsmith $";
#endif
#include "drawimpl.h"  /*I "draw.h" I*/
  
/*@
     DrawLine - Draws a line onto a drawable.

  Input Parameters:
.   ctx - the drawing context
.   xl,yl,xr,yr - the coordinates of the line endpoints
.   cl - the colors of the endpoints
@*/
int DrawLine(DrawCtx ctx,double xl,double yl,double xr,double yr,int cl)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  return (*ctx->ops.drawline)(ctx,xl,yl,xr,yr,cl);
}

/*@
     DrawLineSetWidth - Sets the line width for future draws.
                     The width is relative to the user coordinates of 
                     the window. 0.0 denotes the natural width,
                     1.0 denotes the interior viewport. 

  Input Parameters:
.   ctx - the drawing context
.   width - the width in user coordinates
@*/
int DrawLineSetWidth(DrawCtx ctx,double width)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  return (*ctx->ops.drawlinewidth)(ctx,width);
}

/*@C
     DrawText - Draws text onto a drawable.

  Input Parameters:
.   ctx - the drawing context
.   xl,yl - the coordinates of lower left corner of text
.   cl - the color of the text
.   text - the text to draw
@*/
int DrawText(DrawCtx ctx,double xl,double yl,int cl,char *text)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  return (*ctx->ops.drawtext)(ctx,xl,yl,cl,text);
}

/*@C
     DrawTextVertical - Draws text onto a drawable.

  Input Parameters:
.   ctx - the drawing context
.   xl,yl - the coordinates of upper left corner of text
.   cl - the color of the text
.   text - the text to draw
@*/
int DrawTextVertical(DrawCtx ctx,double xl,double yl,int cl,char *text)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  return (*ctx->ops.drawtextvert)(ctx,xl,yl,cl,text);
}

/*@
     DrawTextSetSize - Sets the size for charactor text.
                     The width is relative to the user coordinates of 
                     the window. 0.0 denotes the natural width,
                     1.0 denotes the interior viewport. 

  Input Parameters:
.   ctx - the drawing context
.   width - the width in user coordinates
.   height - the charactor height

  Note: only a limited range of sizes are available.
 @*/
int DrawTextSetSize(DrawCtx ctx,double width,double height)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  return (*ctx->ops.drawtextsize)(ctx,width,height);
}
/*@
     DrawTextGetSize - Gets the size for charactor text.
                     The width is relative to the user coordinates of 
                     the window. 0.0 denotes the natural width,
                     1.0 denotes the interior viewport. 

  Input Parameters:
.   ctx - the drawing context
.   width - the width in user coordinates
.   height - the charactor height
 @*/
int DrawTextGetSize(DrawCtx ctx,double *width,double *height)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  return (*ctx->ops.drawtextgetsize)(ctx,width,height);
}

/*@
     DrawPoint - Draws a point onto a drawable.

  Input Parameters:
.   ctx - the drawing context
.   xl,yl - the coordinates of the point
.   cl - the color of the point
 @*/
int DrawPoint(DrawCtx ctx,double xl,double yl,int cl)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  return (*ctx->ops.drawpoint)(ctx,xl,yl,cl);
}

/*@
     DrawPointSetSize - Sets the point size for future draws.
                     The size is relative to the user coordinates of 
                     the window. 0.0 denotes the natural width,
                     1.0 denotes the interior viewport. 

  Input Parameters:
.   ctx - the drawing context
.   width - the width in user coordinates

  Note: Even a size of zero insures that a single pixel is colored.
 @*/
int DrawPointSetSize(DrawCtx ctx,double width)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  if (width < 0.0 || width > 1.0) SETERRQ(1,"DrawPointSetSize: Bad size");
  return (*ctx->ops.drawpointsize)(ctx,width);
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
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  if (xl < 0.0 || xr > 1.0 || yl < 0.0 || yr > 1.0 || xr <= xl || yr <= yl)
    SETERRQ(1,"DrawSetViewPort: Bad values"); 
  ctx->port_xl = xl; ctx->port_yl = yl;
  ctx->port_xr = xr; ctx->port_yr = yr;
  if (ctx->ops.viewport) return (*ctx->ops.viewport)(ctx,xl,yl,xr,yr);
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
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  ctx->coor_xl = xl; ctx->coor_yl = yl;
  ctx->coor_xr = xr; ctx->coor_yr = yr;
  return 0;
}

/*@
    DrawSetPause - Sets the amount of time that program pauses after 
         a DrawSyncFlush() is called. Defaults to zero unless the 
         -pause option is given.

  Input Paramters:
.  ctx - the drawing object
.  pause - number of seconds to pause, -1 implies until user input

@*/
int DrawSetPause(DrawCtx ctx,int pause)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  ctx->pause = pause;
  return 0;
}

/*@
    DrawGetCoordinates - Gets the application coordinates of the 
          corners of the window (or page).

  Input Paramters:
.  ctx - the drawing object

  Ouput Parameters:
.  xl,yl,xr,yr - the coordinates of the lower left corner and upper
                 right corner of the drawing region.
@*/
int DrawGetCoordinates(DrawCtx ctx,double *xl,double *yl,double *xr,double *yr)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (!xl || !xr || !yl || !yr) SETERRQ(1,"DrawGetCoordinates:Bad pointer");
  if (ctx->type == NULLWINDOW) return 0;
  *xl = ctx->coor_xl; *yl = ctx->coor_yl;
  *xr = ctx->coor_xr; *yr = ctx->coor_yr;
  return 0;
}

/*@
   DrawSetDoubleBuffer - Sets a window to be double buffered. 

  Input Parameters:
.  ctx - the drawing context
@*/
int DrawSetDoubleBuffer(DrawCtx ctx)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  if (ctx->ops.doublebuff) return (*ctx->ops.doublebuff)(ctx);
  return 0;
}

/*@
   DrawFlush - Flushs graphical output.

  Input Parameters:
.  ctx - the drawing context
@*/
int DrawFlush(DrawCtx ctx)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  if (ctx->ops.flush) return (*ctx->ops.flush)(ctx);
  return 0;
}

/*@
   DrawSyncFlush - Flushs graphical output. This waits until all 
      processors have arrived and flushed, then does a global flush.
      This is usually done to change the frame for double buffered 
      graphics.

  Input Parameters:
.  ctx - the drawing context
@*/
int DrawSyncFlush(DrawCtx ctx)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  if (ctx->ops.flush) return (*ctx->ops.sflush)(ctx);
  return 0;
}

/*@
   DrawClear - Clears graphical output.

  Input Parameters:
.  ctx - the drawing context
@*/
int DrawClear(DrawCtx ctx)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  if (ctx->ops.clear) return (*ctx->ops.clear)(ctx);
  return 0;
}
/*@C
   DrawDestroy - Deletes a draw context.

  Input Parameters:
.  ctx - the drawing context
@*/
int DrawDestroy(DrawCtx ctx)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->destroy) return (*ctx->destroy)((PetscObject)ctx);
  return 0;
}
/*@
     DrawRectangle -Draws a rectangle  onto a drawable.

  Input Parameters:
.   ctx - the drawing context
.   xl,yl,xr,yr - the coordinates of the lower left, upper right corners
.   c1,c2,c3,c4 - the colors of the four corners in counter clockwise order
 @*/
int DrawRectangle(DrawCtx ctx,double xl,double yl,double xr,double yr,
                              int c1, int c2,int c3,int c4)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  return (*ctx->ops.rectangle)(ctx,xl,yl,xr,yr,c1,c2,c3,c4);
}

/*@
     DrawTriangle -Draws a triangle  onto a drawable.

  Input Parameters:
.   ctx - the drawing context
.   x1,y1,x2,y2,x3,y3 - the coordinates of the vertices
.   c1,c2,c3 - the colors of the corners in counter clockwise order
 @*/
int DrawTriangle(DrawCtx ctx,double x1,double y1,double x2,double y2,
                 double x3,double y3,int c1, int c2,int c3)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  return (*ctx->ops.triangle)(ctx,x1,y1,x2,y2,x3,y3,c1,c2,c3);
}

int DrawDestroy_Null(PetscObject obj)
{
  PLogObjectDestroy(obj);
  PETSCHEADERDESTROY(obj); 
  return 0;
}

/*
     DrawOpenNull - Opens a null drawing context. All draw commands to 
  it are ignored.

 Output Parameter:
.  win - the drawing context

*/
int DrawOpenNull(MPI_Comm comm,DrawCtx *win)
{
  DrawCtx ctx;
  *win = 0;
  PETSCHEADERCREATE(ctx,_DrawCtx,DRAW_COOKIE,NULLWINDOW,comm);
  PLogObjectCreate(ctx);
  PetscZero(&ctx->ops,sizeof(struct _DrawOps));
  ctx->destroy = DrawDestroy_Null;
  ctx->view    = 0;
  ctx->pause   = 0;
  ctx->coor_xl = 0.0;  ctx->coor_xr = 1.0;
  ctx->coor_yl = 0.0;  ctx->coor_yr = 1.0;
  ctx->port_xl = 0.0;  ctx->port_xr = 1.0;
  ctx->port_yl = 0.0;  ctx->port_yr = 1.0;
  *win = ctx;
  return 0;
}






