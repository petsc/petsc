
#include <stdio.h>
#include "ximpl.h"

#define XTRANS(win,xwin,x) \
   ((xwin)->w)*((win)->port_xl + (((x - (win)->coor_xl)*\
                                   ((win)->port_xr - (win)->port_xl))/\
                                   ((win)->coor_xr - (win)->coor_xl)));
#define YTRANS(win,xwin,y) \
   ((xwin)->h)*(1.0-(win)->port_yl - (((y - (win)->coor_yl)*\
                                       ((win)->port_yr - (win)->port_yl))/\
                                       ((win)->coor_yr - (win)->coor_yl)));

/*
    Defines the operations for the X Draw implementation.
*/

int XiDrawLine(DrawCtx Win, double xl, double yl, double xr, double yr,
                int cl, int cr)
{
  XiWindow* XiWin = (XiWindow*) Win->data;
  int       x1,y1,x2,y2, c = (cl + cr)/2;
  XiSetColor( XiWin, c );
  x1 = XTRANS(Win,XiWin,xl);   x2  = XTRANS(Win,XiWin,xr); 
  y1 = YTRANS(Win,XiWin,yl);   y2  = YTRANS(Win,XiWin,yr); 
  printf("line %g %g %g %g %d %d %d %d\n",xl,yl,xr,yr,x1,y1,x2,y2);
  XDrawLine( XiWin->disp, XiDrawable(XiWin), XiWin->gc.set, x1, y1, x2, y2);
  return 0;
}

int XiFlush(DrawCtx Win )
{
  XiWindow* XiWin = (XiWindow*) Win->data;
  if (XiWin->drw) {
    XCopyArea( XiWin->disp, XiWin->drw, XiWin->win, XiWin->gc.set, 0, 0, 
	       XiWin->w, XiWin->h, XiWin->x, XiWin->y );
  }
  XFlush( XiWin->disp );
  return 0;
}

int Xiviewport(DrawCtx Win,double xl,double yl,double xr,double yr)
{
  XiWindow*  XiWin = (XiWindow*) Win->data;
  XRectangle box;
  box.x = xl*XiWin->w;   box.y = (1.0-yr)*XiWin->h;
  box.width = (xr-xl)*XiWin->w;   box.height = (yr-yl)*XiWin->h;
  XSetClipRectangles(XiWin->disp,XiWin->gc.set,0,0,&box,1,Unsorted);
  return 0;
}

int XiClearWindow(DrawCtx Win)
{
  XiWindow*  XiWin = (XiWindow*) Win->data;
  int        x,  y,  w,  h;
  x = Win->port_xl*XiWin->w;
  w = (Win->port_xr - Win->port_xl)*XiWin->w;
  y = Win->port_yr*XiWin->h;
  h = (Win->port_yr - Win->port_yl)*XiWin->h;
  XiSetPixVal(XiWin, XiWin->background );
  XFillRectangle(XiWin->disp,XiDrawable(XiWin),XiWin->gc.set, x, y, w, h);
  return 0;
}

static struct _DrawOps DvOps = { 0,XiFlush,XiDrawLine,0,0,0,0,0,
                                 Xiviewport,XiClearWindow};

/*@
    DrawOpenX - Opens an X window for use with the Draw routines.

  Input Parameters:
.   display - the X display to open on, or null for the local machine
.   title - the title to put in the title bar
.   x,y - the screen coordinates of the upper left corner of window
.   width, height - the screen width and height in pixels

  Output Parameters:
.   ctx - the drawing context.
@*/
int DrawOpenX(char* display,char *title,int x,int y,int w,int h,
              DrawCtx* inctx)
{
  DrawCtx  ctx;
  XiWindow *Xwin;
  int      ierr;

  *inctx = 0;
  CREATEHEADER(ctx,_DrawCtx);
  ctx->cookie  = DRAW_COOKIE;
  ctx->type    = XWINDOW;
  ctx->ops     = &DvOps;
  ctx->destroy = 0;
  ctx->view    = 0;
  ctx->coor_xl = 0.0;  ctx->coor_xr = 1.0;
  ctx->coor_yl = 0.0;  ctx->coor_yr = 1.0;
  ctx->port_xl = 0.0;  ctx->port_xr = 1.0;
  ctx->port_yl = 0.0;  ctx->port_yr = 1.0;

  /* actually create and open the window */
  Xwin         = (XiWindow *) MALLOC( sizeof(XiWindow) ); CHKPTR(Xwin);
  ierr         = XiQuickWindow(Xwin,display,title,x,y,w,h,256); CHKERR(ierr);

  ctx->data    = (void *) Xwin;
  *inctx       = ctx;
  return 0;
}
