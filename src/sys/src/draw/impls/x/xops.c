#ifndef lint
static char vcid[] = "$Id: xops.c,v 1.52 1996/08/04 23:13:24 bsmith Exp bsmith $";
#endif
/*
    Defines the operations for the X Draw implementation.
*/

#include "src/draw/impls/x/ximpl.h"

#if defined(HAVE_X11)

/*
     These macros transform from the users coordinates to the 
   X-window pixel coordinates.
*/
#define XTRANS(win,xwin,x) \
   (int)(((xwin)->w)*((win)->port_xl + (((x - (win)->coor_xl)*\
                                   ((win)->port_xr - (win)->port_xl))/\
                                   ((win)->coor_xr - (win)->coor_xl))))
#define YTRANS(win,xwin,y) \
   (int)(((xwin)->h)*(1.0-(win)->port_yl - (((y - (win)->coor_yl)*\
                                       ((win)->port_yr - (win)->port_yl))/\
                                       ((win)->coor_yr - (win)->coor_yl))))

int DrawLine_X(Draw Win, double xl, double yl, double xr, double yr,int cl)
{
  Draw_X* XiWin = (Draw_X*) Win->data;
  int     x1,y1,x2,y2;

  XiSetColor( XiWin, cl );
  x1 = XTRANS(Win,XiWin,xl);   x2  = XTRANS(Win,XiWin,xr); 
  y1 = YTRANS(Win,XiWin,yl);   y2  = YTRANS(Win,XiWin,yr); 
  XDrawLine( XiWin->disp, XiDrawable(XiWin), XiWin->gc.set, x1, y1, x2, y2);
  return 0;
}

static int DrawPoint_X(Draw Win,double x,double  y,int c)
{
  int     xx,yy;
  Draw_X* XiWin = (Draw_X*) Win->data;

  xx = XTRANS(Win,XiWin,x);  yy = YTRANS(Win,XiWin,y);
  XiSetColor( XiWin, c );
  XDrawPoint( XiWin->disp, XiDrawable(XiWin), XiWin->gc.set,xx, yy);
  return 0;
}

static int DrawRectangle_X(Draw Win, double xl, double yl, double xr, double yr,
                           int c1, int c2,int c3,int c4)
{
  Draw_X* XiWin = (Draw_X*) Win->data;
  int     x1,y1,w,h, c = (c1 + c2 + c3 + c4)/4;

  XiSetColor( XiWin, c );
  x1 = XTRANS(Win,XiWin,xl);   w  = XTRANS(Win,XiWin,xr) - x1; 
  y1 = YTRANS(Win,XiWin,yr);   h  = YTRANS(Win,XiWin,yl) - y1;
  if (w <= 0) w = 1; if (h <= 0) h = 1;
  XFillRectangle( XiWin->disp, XiDrawable(XiWin), XiWin->gc.set, x1, y1, w, h);
  return 0;
}

extern int XiDrawInterpolatedTriangle(Draw_X*,int,int,int,int,int,int,int,int,int);

static int DrawTriangle_X(Draw Win, double X1, double Y1, double X2, 
                          double Y2,double X3,double Y3, int c1, int c2,int c3)
{
  Draw_X* XiWin = (Draw_X*) Win->data;

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

static int DrawText_X(Draw Win,double x,double  y,int c,char *chrs )
{
  int     xx,yy;
  Draw_X* XiWin = (Draw_X*) Win->data;

  xx = XTRANS(Win,XiWin,x);  yy = YTRANS(Win,XiWin,y);
  XiSetColor( XiWin, c );
  XDrawString( XiWin->disp, XiDrawable(XiWin), XiWin->gc.set,
               xx, yy - XiWin->font->font_descent, chrs, PetscStrlen(chrs) );
  return 0;
}

int XiFontFixed( Draw_X*,int, int,XiFont **);

static int DrawTextSetSize_X(Draw Win,double x,double  y)
{
  Draw_X* XiWin = (Draw_X*) Win->data;
  int     w,h;

  w = (int)((XiWin->w)*x*(Win->port_xr - Win->port_xl)/(Win->coor_xr - Win->coor_xl));
  h = (int)((XiWin->h)*y*(Win->port_yr - Win->port_yl)/(Win->coor_yr - Win->coor_yl));
  PetscFree(XiWin->font);
  return XiFontFixed( XiWin,w, h, &XiWin->font);
}

int DrawTextGetSize_X(Draw Win,double *x,double  *y)
{
  Draw_X* XiWin = (Draw_X*) Win->data;
  double  w,h;

  w = XiWin->font->font_w; h = XiWin->font->font_h;
  *x = w*(Win->coor_xr - Win->coor_xl)/(XiWin->w)*(Win->port_xr - Win->port_xl);
  *y = h*(Win->coor_yr - Win->coor_yl)/(XiWin->h)*(Win->port_yr - Win->port_yl);
  return 0;
}

int DrawTextVertical_X(Draw Win,double x,double  y,int c,char *chrs )
{
  int     xx,yy,n = PetscStrlen(chrs),i;
  Draw_X* XiWin = (Draw_X*) Win->data;
  char    tmp[2];
  double  tw,th;
  
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

static int DrawFlush_X(Draw Win )
{
  Draw_X* XiWin = (Draw_X*) Win->data;
  XFlush( XiWin->disp ); XSync(XiWin->disp,False);
  return 0;
}

static int DrawSyncFlush_X(Draw Win )
{
  int     rank;
  Draw_X* XiWin = (Draw_X*) Win->data;

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
  return 0;
}

static int DrawSetViewport_X(Draw Win,double xl,double yl,double xr,double yr)
{
  Draw_X*    XiWin = (Draw_X*) Win->data;
  XRectangle box;

  box.x = (int) (xl*XiWin->w);   box.y = (int) ((1.0-yr)*XiWin->h);
  box.width = (int) ((xr-xl)*XiWin->w);box.height = (int) ((yr-yl)*XiWin->h);
  XSetClipRectangles(XiWin->disp,XiWin->gc.set,0,0,&box,1,Unsorted);
  return 0;
}

static int DrawClear_X(Draw Win)
{
  Draw_X*  XiWin = (Draw_X*) Win->data;
  int      x,  y,  w,  h;

  x = (int) (Win->port_xl*XiWin->w);
  w = (int) ((Win->port_xr - Win->port_xl)*XiWin->w);
  y = (int) ((1.0-Win->port_yr)*XiWin->h);
  h = (int) ((Win->port_yr - Win->port_yl)*XiWin->h);
  XiSetPixVal(XiWin, XiWin->background );
  XFillRectangle(XiWin->disp,XiDrawable(XiWin),XiWin->gc.set, x, y, w, h);
  return 0;
}

static int DrawSyncClear_X(Draw Win)
{
  int     rank;
  Draw_X* XiWin = (Draw_X*) Win->data;

  MPI_Barrier(Win->comm);
  MPI_Comm_rank(Win->comm,&rank);
  if (!rank) {
    DrawClear_X(Win);
    XFlush( XiWin->disp );
  }
  MPI_Barrier(Win->comm);
  return 0;
}

static int DrawSetDoubleBuffer_X(Draw Win)
{
  Draw_X*  win = (Draw_X*) Win->data;
  int      rank;
  if (win->drw) return 0;

  MPI_Comm_rank(Win->comm,&rank);
  if (!rank) {
    win->drw = XCreatePixmap(win->disp,win->win,win->w,win->h,win->depth);
  }
  /* try to make sure it is actually done before passing info to all */
  XSync(win->disp,False);
  MPI_Bcast(&win->drw,1,MPI_UNSIGNED_LONG,0,Win->comm);
  return 0;
}

static int DrawGetMouseButton_X(Draw draw,DrawButton *button,double* x_user,
                                double *y_user,double *x_phys,double *y_phys)
{
  XEvent       report;
  Draw_X*      win = (Draw_X*) draw->data;
  Window       root, child;
  int          root_x, root_y,px,py;
  unsigned int keys_button;

  XSelectInput( win->disp, win->win, ButtonPressMask | ButtonReleaseMask );

  while (XCheckTypedEvent( win->disp, ButtonPress, &report ));
  XMaskEvent( win->disp, ButtonReleaseMask, &report );
  switch (report.xbutton.button) {
    case Button1: *button = BUTTON_LEFT; break;
    case Button2: *button = BUTTON_CENTER; break;
    case Button3: *button = BUTTON_RIGHT; break;
  }
  XQueryPointer(win->disp, report.xmotion.window,&root,&child,&root_x,&root_y,
                &px,&py,&keys_button);

  if (x_phys) *x_phys = ((double) px)/((double) win->w);
  if (y_phys) *y_phys = 1.0 - ((double) py)/((double) win->h);

  if (x_user) *x_user = draw->coor_xl + ((((double) px)/((double) win->w)-draw->port_xl))*
                        (draw->coor_xr - draw->coor_xl)/(draw->port_xr - draw->port_xl);
  if (y_user) *y_user = draw->coor_yl + 
                        ((1.0 - ((double) py)/((double) win->h)-draw->port_yl))*
                        (draw->coor_yr - draw->coor_yl)/(draw->port_yr - draw->port_yl);
  return 0;
}

static int DrawPause_X(Draw draw)
{
  int ierr;

  if (draw->pause > 0) PetscSleep(draw->pause);
  else if (draw->pause < 0) {
    DrawButton button;
    int        rank;
    MPI_Comm_rank(draw->comm,&rank);
    if (rank) return 0;
    ierr = DrawGetMouseButton(draw,&button,0,0,0,0); CHKERRQ(ierr);
    if (button == BUTTON_RIGHT) SETERRQ(1,"DrawPause_X:User request exit");
    if (button == BUTTON_CENTER) draw->pause = 0;
  }
  return 0;
}

static struct _DrawOps DvOps = { DrawSetDoubleBuffer_X,
                                 DrawFlush_X,DrawLine_X,0,0,DrawPoint_X,0,
                                 DrawText_X,DrawTextVertical_X,
                                 DrawTextSetSize_X,DrawTextGetSize_X,
                                 DrawSetViewport_X,DrawClear_X,
                                 DrawSyncFlush_X,
                                 DrawRectangle_X,
                                 DrawTriangle_X,
                                 DrawGetMouseButton_X,
                                 DrawPause_X,
                                 DrawSyncClear_X, 
				 0, 0 };

int DrawDestroy_X(PetscObject obj)
{
  Draw   ctx = (Draw) obj;
  Draw_X *win = (Draw_X *) ctx->data;

  PetscFree(win->font);
  PetscFree(win);
  PLogObjectDestroy(ctx);
  PetscHeaderDestroy(ctx);
  return 0;
}

extern int XiQuickWindow(Draw_X*,char*,char*,int,int,int,int,int);
extern int XiQuickWindowFromWindow(Draw_X*,char*,Window,int);

/*@C
   DrawOpenX - Opens an X-window for use with the Draw routines.

   Input Parameters:
.  comm - the communicator that will share X-window
.  display - the X display on which to open, or null for the local machine
.  title - the title to put in the title bar
.  x, y - the screen coordinates of the upper left corner of window
.  w, h - the screen width and height in pixels

   Output Parameters:
.  ctx - the drawing context.

   Options Database Keys:
$  -nox : disables all x-windows output
$  -display <name> : name of machine for the X display
$  -draw_pause <pause> : sets time (in seconds) that the program pauses
    after DrawPause() is called (0 is default, -1 implies until user input).

.keywords: draw, open, x

.seealso: DrawSyncFlush()
@*/
int DrawOpenX(MPI_Comm comm,char* display,char *title,int x,int y,int w,int h,
              Draw* inctx)
{
  Draw   ctx;
  Draw_X *Xwin;
  int    ierr,size,rank,flg;
  char   string[128];

  ierr = OptionsHasName(PETSC_NULL,"-nox",&flg); CHKERRQ(ierr);
  if (flg) {
    return DrawOpenNull(comm,inctx);
  }

  *inctx = 0;
  PetscHeaderCreate(ctx,_Draw,DRAW_COOKIE,DRAW_XWINDOW,comm);
  PLogObjectCreate(ctx);
  PetscMemcpy(&ctx->ops,&DvOps,sizeof(DvOps));
  ctx->destroy = DrawDestroy_X;
  ctx->view    = 0;
  ctx->pause   = 0;
  ctx->coor_xl = 0.0;  ctx->coor_xr = 1.0;
  ctx->coor_yl = 0.0;  ctx->coor_yr = 1.0;
  ctx->port_xl = 0.0;  ctx->port_xr = 1.0;
  ctx->port_yl = 0.0;  ctx->port_yr = 1.0;

  ierr = OptionsGetInt(PETSC_NULL,"-draw_pause",&ctx->pause,&flg);CHKERRQ(ierr);

  /* actually create and open the window */
  Xwin         = (Draw_X *) PetscMalloc( sizeof(Draw_X) ); CHKPTRQ(Xwin);
  PLogObjectMemory(ctx,sizeof(Draw_X)+sizeof(struct _Draw));
  PetscMemzero(Xwin,sizeof(Draw_X));
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);

  if (!display) {
    PetscGetDisplay(string,128);
    display = string;
  }
  if (rank == 0) {
    if (x < 0 || y < 0) SETERRQ(1,"DrawOpenX:Negative corner of window");
    if (w <= 0 || h <= 0) SETERRQ(1,"DrawOpenX:Negative window width or height");
    ierr = XiQuickWindow(Xwin,display,title,x,y,w,h,256); CHKERRQ(ierr);
    MPI_Bcast(&Xwin->win,1,MPI_UNSIGNED_LONG,0,comm);
  }
  else {
    unsigned long win;
    MPI_Bcast(&win,1,MPI_UNSIGNED_LONG,0,comm);
    ierr = XiQuickWindowFromWindow( Xwin,display, win,256 ); CHKERRQ(ierr);
  }
 
  ctx->data    = (void *) Xwin;
  *inctx       = ctx;
  return 0;
}

#else

#include "draw.h"
int DrawOpenX(MPI_Comm comm,char* disp,char *ttl,int x,int y,int w,int h,Draw* ctx)
{
  return DrawOpenNull(comm,ctx);
}

#endif


/*@C
   ViewerDrawOpenX - Opens an X window for use as a viewer. If you want to 
        do graphics in this window, you must call ViewerDrawGetDraw() and
        perform the graphics on the Draw object.

   Input Parameters:
.  comm - communicator that will share window
.  display - the X display on which to open, or null for the local machine
.  title - the title to put in the title bar
.  x, y - the screen coordinates of the upper left corner of window
.  w, h - the screen width and height in pixels

   Output Parameters:
.  viewer - the viewer

   Format Options:
.   VIEWER_FORMAT_DRAW_BASIC
.   VIEWER_FORMAT_DRAW_LG     - displays using a line graph

   Options Database Keys:
$  -nox : disable all x-windows output
$  -display <name> : name of machine for the X display

.keywords: draw, open, x, viewer

.seealso: DrawOpenX()
@*/
int ViewerDrawOpenX(MPI_Comm comm,char* display,char *title,int x,int y,
                    int w,int h,Viewer *viewer)
{
  int    ierr;
  Viewer ctx;

  *viewer = 0;
  PetscHeaderCreate(ctx,_Viewer,VIEWER_COOKIE,DRAW_VIEWER,comm);
  PLogObjectCreate(ctx);
  ierr = DrawOpenX(comm,display,title,x,y,w,h,&ctx->draw);CHKERRQ(ierr);
  PLogObjectParent(ctx,ctx->draw);

  ctx->flush   = ViewerFlush_Draw;
  ctx->destroy = ViewerDestroy_Draw;
  ctx->format  = 0;

  /* 
     Create a DrawLG context for use with viewer format:
     VIEWER_FORMAT_DRAW_LG
  */
  ierr = DrawLGCreate(ctx->draw,1,&ctx->drawlg);CHKERRQ(ierr);
  *viewer      = ctx;
  return 0;
}

/* -------------------------------------------------------------------*/
/* 
     Default X window viewers, may be used at any time.
*/

Viewer VIEWER_DRAWX_SELF_PRIVATE = 0, VIEWER_DRAWX_WORLD_PRIVATE = 0;

int ViewerInitializeDrawXSelf_Private()
{
  int ierr,xywh[4],size = 4,flg;

  if (VIEWER_DRAWX_SELF_PRIVATE) return 0;
  xywh[0] = 300; xywh[1] = 0; xywh[2] = 300; xywh[3] = 300;
  ierr = OptionsGetIntArray(PETSC_NULL,"-draw_self_geometry",xywh,&size,&flg);
         CHKERRQ(ierr);
  ierr = ViewerDrawOpenX(MPI_COMM_WORLD,0,0,xywh[0],xywh[1],xywh[2],xywh[3],
                         &VIEWER_DRAWX_SELF_PRIVATE); CHKERRQ(ierr);
  return 0;
}

int ViewerInitializeDrawXWorld_Private()
{
  int ierr,xywh[4],size = 4,flg;

  if (VIEWER_DRAWX_WORLD_PRIVATE) return 0;
  xywh[0] = 0; xywh[1] = 0; xywh[2] = 300; xywh[3] = 300;
  ierr = OptionsGetIntArray(PETSC_NULL,"-draw_world_geometry",xywh,&size,&flg);
         CHKERRQ(ierr);
  ierr = ViewerDrawOpenX(MPI_COMM_WORLD,0,0,xywh[0],xywh[1],xywh[2],xywh[3],
                         &VIEWER_DRAWX_WORLD_PRIVATE); CHKERRQ(ierr);
  return 0;
}

int ViewerDestroyDrawX_Private()
{
  int ierr;

  if (VIEWER_DRAWX_WORLD_PRIVATE) {
    ierr = ViewerDestroy(VIEWER_DRAWX_WORLD_PRIVATE); CHKERRQ(ierr);
  }
  if (VIEWER_DRAWX_SELF_PRIVATE) {
    ierr = ViewerDestroy(VIEWER_DRAWX_SELF_PRIVATE); CHKERRQ(ierr);
  }
  return 0;
}
