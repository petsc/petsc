#ifndef lint
static char vcid[] = "$Id: xops.c,v 1.18 1995/05/25 22:48:17 bsmith Exp bsmith $";
#endif
#include <stdio.h>
#include "ximpl.h"

#define XTRANS(win,xwin,x) \
   (int)(((xwin)->w)*((win)->port_xl + (((x - (win)->coor_xl)*\
                                   ((win)->port_xr - (win)->port_xl))/\
                                   ((win)->coor_xr - (win)->coor_xl))))
#define YTRANS(win,xwin,y) \
   (int)(((xwin)->h)*(1.0-(win)->port_yl - (((y - (win)->coor_yl)*\
                                       ((win)->port_yr - (win)->port_yl))/\
                                       ((win)->coor_yr - (win)->coor_yl))))

/*
    Defines the operations for the X Draw implementation.
*/

int DrawLine_X(DrawCtx Win, double xl, double yl, double xr, double yr,
                int cl, int cr)
{
  DrawCtx_X* XiWin = (DrawCtx_X*) Win->data;
  int       x1,y1,x2,y2, c = (cl + cr)/2;
  XiSetColor( XiWin, c );
  x1 = XTRANS(Win,XiWin,xl);   x2  = XTRANS(Win,XiWin,xr); 
  y1 = YTRANS(Win,XiWin,yl);   y2  = YTRANS(Win,XiWin,yr); 
  XDrawLine( XiWin->disp, XiDrawable(XiWin), XiWin->gc.set, x1, y1, x2, y2);
  return 0;
}

static int DrawPoint_X(DrawCtx Win,double x,double  y,int c)
{
  int    xx,yy;
  DrawCtx_X* XiWin = (DrawCtx_X*) Win->data;
  xx = XTRANS(Win,XiWin,x);  yy = YTRANS(Win,XiWin,y);
  XiSetColor( XiWin, c );
  XDrawPoint( XiWin->disp, XiDrawable(XiWin), XiWin->gc.set,xx, yy);
  return 0;
}

static int DrawRectangle_X(DrawCtx Win, double xl, double yl, double xr, double yr,
                int c1, int c2,int c3,int c4)
{
  DrawCtx_X* XiWin = (DrawCtx_X*) Win->data;
  int       x1,y1,w,h, c = (c1 + c2 + c3 + c4)/4;
  XiSetColor( XiWin, c );
  x1 = XTRANS(Win,XiWin,xl);   w  = XTRANS(Win,XiWin,xr) - x1; 
  y1 = YTRANS(Win,XiWin,yr);   h  = YTRANS(Win,XiWin,yl) - y1;
  XFillRectangle( XiWin->disp, XiDrawable(XiWin), XiWin->gc.set, x1, y1, w, h);
  return 0;
}

extern int XiDrawInterpolatedTriangle(DrawCtx_X*, int, int, int, 
                                int,int,int,int,int,int);

static int DrawTriangle_X(DrawCtx Win, double X1, double Y1, double X2, 
                          double Y2,double X3,double Y3, int c1, int c2,int c3)
{
  DrawCtx_X* XiWin = (DrawCtx_X*) Win->data;
  if (c1 == c2 && c2 == c3) {
    XPoint pt[3];
    XiSetColor( XiWin, c1 );
    pt[0].x = XTRANS(Win,XiWin,X1);
    pt[0].y = YTRANS(Win,XiWin,Y1); 
    pt[1].x = XTRANS(Win,XiWin,X2);
    pt[1].y = YTRANS(Win,XiWin,Y2); 
    pt[2].x = XTRANS(Win,XiWin,X3);
    pt[2].y = YTRANS(Win,XiWin,Y3); 
    XFillPolygon(XiWin->disp,XiDrawable(XiWin),XiWin->gc.set,pt,3,Convex,
                 CoordModeOrigin);
  }
  else {
    int x1,y1,x2,y2,x3,y3;
    x1 = XTRANS(Win,XiWin,X1);
    y1 = YTRANS(Win,XiWin,Y1); 
    x2 = XTRANS(Win,XiWin,X2);
    y2 = YTRANS(Win,XiWin,Y2); 
    x3 = XTRANS(Win,XiWin,X3);
    y3 = YTRANS(Win,XiWin,Y3); 
    XiDrawInterpolatedTriangle(XiWin,x1,y1,c1,x2,y2,c2,x3,y3,c3);
  }
  return 0;
}

static int DrawText_X(DrawCtx Win,double x,double  y,int c,char *chrs )
{
  int    xx,yy;
  DrawCtx_X* XiWin = (DrawCtx_X*) Win->data;
  xx = XTRANS(Win,XiWin,x);  yy = YTRANS(Win,XiWin,y);
  XiSetColor( XiWin, c );
  XDrawString( XiWin->disp, XiDrawable(XiWin), XiWin->gc.set,
               xx, yy - XiWin->font->font_descent, chrs, strlen(chrs) );
  return 0;
}

int XiFontFixed( DrawCtx_X*,int, int,XiFont **);
static int DrawTextSetSize_X(DrawCtx Win,double x,double  y)
{
  DrawCtx_X* XiWin = (DrawCtx_X*) Win->data;
  int       w,h;
  w = (int)((XiWin->w)*x*(Win->port_xr - Win->port_xl)/
                                   (Win->coor_xr - Win->coor_xl));
  h = (int)((XiWin->h)*y*(Win->port_yr - Win->port_yl)/
                                   (Win->coor_yr - Win->coor_yl));
  return XiFontFixed( XiWin,w, h, &XiWin->font);
}

int DrawTextGetSize_X(DrawCtx Win,double *x,double  *y)
{
  DrawCtx_X* XiWin = (DrawCtx_X*) Win->data;
  double    w,h;
  w = XiWin->font->font_w; h = XiWin->font->font_h;
  *x = w*(Win->coor_xr - Win->coor_xl)/
         (XiWin->w)*(Win->port_xr - Win->port_xl);
  *y = h*(Win->coor_yr - Win->coor_yl)/
         (XiWin->h)*(Win->port_yr - Win->port_yl);
  return 0;
}

int DrawTextVertical_X(DrawCtx Win,double x,double  y,int c,char *chrs )
{
  int       xx,yy,n = strlen(chrs),i;
  DrawCtx_X* XiWin = (DrawCtx_X*) Win->data;
  char      tmp[2];
  double    tw,th;
  
  tmp[1] = 0;
  XiSetColor( XiWin, c );
  DrawTextGetSize_X(Win,&tw,&th);
  xx = XTRANS(Win,XiWin,x);
  for ( i=0; i<n; i++ ) {
    tmp[0] = chrs[i];
    yy = YTRANS(Win,XiWin,y-th*i);
    XDrawString( XiWin->disp, XiDrawable(XiWin), XiWin->gc.set,
               xx, yy - XiWin->font->font_descent, tmp, 1 );
  }
  return 0;
}

static int DrawFlush_X(DrawCtx Win )
{
  DrawCtx_X* XiWin = (DrawCtx_X*) Win->data;
  XFlush( XiWin->disp );
  return 0;
}

#if defined(__cplusplus)
extern "C" {
#endif
extern void sleep(int);
#if defined(__cplusplus)
};
#endif

static int DrawSyncFlush_X(DrawCtx Win )
{
  int       rank;
  DrawCtx_X* XiWin = (DrawCtx_X*) Win->data;
  XFlush( XiWin->disp );
  if (XiWin->drw) {
    MPI_Comm_rank(Win->comm,&rank);
    /* make sure data has actually arrived at server */
    XSync(XiWin->disp,False);
    MPI_Barrier(Win->comm);
    if (!rank) {
      XCopyArea( XiWin->disp, XiWin->drw, XiWin->win, XiWin->gc.set, 0, 0, 
	       XiWin->w, XiWin->h, XiWin->x, XiWin->y );
      XFlush( XiWin->disp );
    }
  }
  if (Win->pause > 0) sleep(Win->pause);
  if (Win->pause < 0) getc(stdin);
  return 0;
}

static int DrawSetViewport_X(DrawCtx Win,double xl,double yl,double xr,double yr)
{
  DrawCtx_X*  XiWin = (DrawCtx_X*) Win->data;
  XRectangle box;
  box.x = (int) (xl*XiWin->w);   box.y = (int) ((1.0-yr)*XiWin->h);
  box.width = (int) ((xr-xl)*XiWin->w);box.height = (int) ((yr-yl)*XiWin->h);
  XSetClipRectangles(XiWin->disp,XiWin->gc.set,0,0,&box,1,Unsorted);
  return 0;
}

static int DrawClear_X(DrawCtx Win)
{
  DrawCtx_X*  XiWin = (DrawCtx_X*) Win->data;
  int        x,  y,  w,  h;
  x = (int) (Win->port_xl*XiWin->w);
  w = (int) ((Win->port_xr - Win->port_xl)*XiWin->w);
  y = (int) ((1.0-Win->port_yr)*XiWin->h);
  h = (int) ((Win->port_yr - Win->port_yl)*XiWin->h);
  XiSetPixVal(XiWin, XiWin->background );
  XFillRectangle(XiWin->disp,XiDrawable(XiWin),XiWin->gc.set, x, y, w, h);
  return 0;
}

static int DrawSetDoubleBuffer_X(DrawCtx Win)
{
  DrawCtx_X*  win = (DrawCtx_X*) Win->data;
  int        mytid;
  if (win->drw) return 0;

  MPI_Comm_rank(Win->comm,&mytid);
  if (!mytid) {
    win->drw = XCreatePixmap(win->disp,win->win,win->w,win->h,win->depth);
  }
  /* try to make sure it is actually done before passing info to all */
  XSync(win->disp,False);
  MPI_Bcast(&win->drw,1,MPI_UNSIGNED_LONG,0,Win->comm);
  return 0;
}
extern int XiQuickWindow(DrawCtx_X*,char*,char*,int,int,int,int,int);

static struct _DrawOps DvOps = { DrawSetDoubleBuffer_X,
                                 DrawFlush_X,DrawLine_X,0,DrawPoint_X,0,
                                 DrawText_X,DrawTextVertical_X,
                                 DrawTextSetSize_X,DrawTextGetSize_X,
                                 DrawSetViewport_X,DrawClear_X,
                                 DrawSyncFlush_X,
                                 DrawRectangle_X,
                                 DrawTriangle_X};

int DrawDestroy_X(PetscObject obj)
{
  DrawCtx  ctx = (DrawCtx) obj;
  DrawCtx_X *win = (DrawCtx_X *) ctx->data;
  FREE(win);
  PLogObjectDestroy(ctx);
  PETSCHEADERDESTROY(ctx);
  return 0;
}

extern int XiQuickWindowFromWindow(DrawCtx_X*,char*,Window,int);

/*@
    DrawOpenX - Opens an X window for use with the Draw routines.

  Input Parameters:
.   comm - communicator that will share window
.   display - the X display to open on, or null for the local machine
.   title - the title to put in the title bar
.   x,y - the screen coordinates of the upper left corner of window
.   width, height - the screen width and height in pixels

  Output Parameters:
.   ctx - the drawing context.
@*/
int DrawOpenX(MPI_Comm comm,char* display,char *title,int x,int y,int w,int h,
              DrawCtx* inctx)
{
  DrawCtx  ctx;
  DrawCtx_X *Xwin;
  int      ierr,numtid,mytid;
  char     string[128];

  if (OptionsHasName(0,"-nox")) {
    return DrawOpenNull(comm,inctx);
  }

  *inctx = 0;
  PETSCHEADERCREATE(ctx,_DrawCtx,DRAW_COOKIE,XWINDOW,comm);
  PLogObjectCreate(ctx);
  ctx->ops     = &DvOps;
  ctx->destroy = DrawDestroy_X;
  ctx->view    = 0;
  ctx->pause   = 0;
  ctx->coor_xl = 0.0;  ctx->coor_xr = 1.0;
  ctx->coor_yl = 0.0;  ctx->coor_yr = 1.0;
  ctx->port_xl = 0.0;  ctx->port_xr = 1.0;
  ctx->port_yl = 0.0;  ctx->port_yr = 1.0;

  OptionsGetInt(0,"-pause",&ctx->pause);

  /* actually create and open the window */
  Xwin         = (DrawCtx_X *) MALLOC( sizeof(DrawCtx_X) ); CHKPTR(Xwin);
  MEMSET(Xwin,0,sizeof(DrawCtx_X));
  MPI_Comm_size(comm,&numtid);
  MPI_Comm_rank(comm,&mytid);
  if (mytid == 0) {
    if (!display && OptionsGetString(0,"-display",string,128)) {
      display = string;
    }
    if (!display) {
      display = (char *) MALLOC( 128*sizeof(char) ); CHKPTR(display);
      MPIU_Set_display(comm,display,128);
    }
    ierr = XiQuickWindow(Xwin,display,title,x,y,w,h,256); CHKERR(ierr);
    if (display != string) FREE(display);
    MPI_Bcast(&Xwin->win,1,MPI_UNSIGNED_LONG,0,comm);
  }
  else {
    unsigned long win;
    if (!display && OptionsGetString(0,"-display",string,128)) {
      display = string;
    }
    if (!display) {
      display = (char *) MALLOC( 128*sizeof(char) ); CHKPTR(display);
      MPIU_Set_display(comm,display,128);
    }
    MPI_Bcast(&win,1,MPI_UNSIGNED_LONG,0,comm);
    ierr = XiQuickWindowFromWindow( Xwin,display, win,256 ); CHKERR(ierr);
    if (display != string) FREE(display);
  }
 
  ctx->data    = (void *) Xwin;
  *inctx       = ctx;
  return 0;
}


