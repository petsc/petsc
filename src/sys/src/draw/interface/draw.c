#ifndef lint
static char vcid[] = "$Id: draw.c,v 1.8 1995/03/06 04:28:49 bsmith Exp bsmith $";
#endif
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
  VALIDHEADER(ctx,DRAW_COOKIE);
  return (*ctx->ops->drawlinewidth)(ctx,width);
}

/*@
     DrawText - Draws text onto a drawable.

  Input Parameters:
.   ctx - the drawing context
.   xl,yl - the coordinates of lower left corner of text
.   cl - the color of the text
.   text - the text to draw
@*/
int DrawText(DrawCtx ctx,double xl,double yl,int cl,char *text)
{
  VALIDHEADER(ctx,DRAW_COOKIE);
  return (*ctx->ops->drawtext)(ctx,xl,yl,cl,text);
}

/*@
     DrawTextVertical - Draws text onto a drawable.

  Input Parameters:
.   ctx - the drawing context
.   xl,yl - the coordinates of upper left corner of text
.   cl - the color of the text
.   text - the text to draw
@*/
int DrawTextVertical(DrawCtx ctx,double xl,double yl,int cl,char *text)
{
  VALIDHEADER(ctx,DRAW_COOKIE);
  return (*ctx->ops->drawtextvert)(ctx,xl,yl,cl,text);
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
@*/
int DrawTextSetSize(DrawCtx ctx,double width,double height)
{
  VALIDHEADER(ctx,DRAW_COOKIE);
  return (*ctx->ops->drawtextsize)(ctx,width,height);
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
  VALIDHEADER(ctx,DRAW_COOKIE);
  return (*ctx->ops->drawtextgetsize)(ctx,width,height);
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
  VALIDHEADER(ctx,DRAW_COOKIE);
  return (*ctx->ops->drawpoint)(ctx,xl,yl,cl);
}

/*@
     DrawPointSetSize - Sets the point size for future draws.
                     The size is relative to the user coordinates of 
                     the window. 0.0 denotes the natural width,
                     1.0 denotes the interior viewport. 

  Input Parameters:
.   ctx - the drawing context
.   width - the width in user coordinates
@*/
int DrawPointSetSize(DrawCtx ctx,double width)
{
  VALIDHEADER(ctx,DRAW_COOKIE);
  return (*ctx->ops->drawpointsize)(ctx,width);
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
  if (ctx->ops->viewport) return (*ctx->ops->viewport)(ctx,xl,yl,xr,yr);
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
    DrawSetPause - Sets the amount of time that program pauses after 
         a DrawSyncFlush() is called. Defaults to zero unless the 
         -pause option is given.

  Input Paramters:
.  ctx - the drawing object
.  pause - number of seconds to pause, -1 implies until user input

@*/
int DrawSetPause(DrawCtx ctx,int pause)
{
  VALIDHEADER(ctx,DRAW_COOKIE);
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
  VALIDHEADER(ctx,DRAW_COOKIE);
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
  VALIDHEADER(ctx,DRAW_COOKIE);
  if (ctx->ops->flush) return (*ctx->ops->sflush)(ctx);
  return 0;
}

/*@
   DrawClear - Clears graphical output.

  Input Parameters:
.  ctx - the drawing context
@*/
int DrawClear(DrawCtx ctx)
{
  VALIDHEADER(ctx,DRAW_COOKIE);
  if (ctx->ops->clear) return (*ctx->ops->clear)(ctx);
  return 0;
}
/*@
   DrawDestroy - Deletes a draw context.

  Input Parameters:
.  ctx - the drawing context
@*/
int DrawDestroy(DrawCtx ctx)
{
  VALIDHEADER(ctx,DRAW_COOKIE);
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
  VALIDHEADER(ctx,DRAW_COOKIE);
  return (*ctx->ops->rectangle)(ctx,xl,yl,xr,yr,c1,c2,c3,c4);
}
/*@
     DrawTriangle -Draws a triangle  onto a drawable.

  Input Parameters:
.   ctx - the drawing context
.   x1,y1,x2,y2,x3,y3 - the coordinates of the vertices
.   c1,c2,c3 - the colors of the corners in counter clockwise order
@*/
int DrawTriangle(DrawCtx ctx,double x1,double y1,double x2,double y2,double x3,double y3,
               int c1, int c2,int c3)
{
  VALIDHEADER(ctx,DRAW_COOKIE);
  return (*ctx->ops->triangle)(ctx,x1,y1,x2,y2,x3,y3,c1,c2,c3);
}
