/*
    Defines the operations for the X PetscDraw implementation.
*/

#include <../src/sys/classes/draw/impls/x/ximpl.h>         /*I  "petscsys.h" I*/

/*
     These macros transform from the users coordinates to the  X-window pixel coordinates.
*/
#define XTRANS(draw,xwin,x)  ((int)(((xwin)->w-1)*((draw)->port_xl + (((x - (draw)->coor_xl)*((draw)->port_xr - (draw)->port_xl))/((draw)->coor_xr - (draw)->coor_xl)))))
#define YTRANS(draw,xwin,y)  (((xwin)->h-1) - (int)(((xwin)->h-1)*((draw)->port_yl + (((y - (draw)->coor_yl)*((draw)->port_yr - (draw)->port_yl))/((draw)->coor_yr - (draw)->coor_yl)))))

#define ITRANS(draw,xwin,i)  ((draw)->coor_xl + (((PetscReal)(i))*((draw)->coor_xr - (draw)->coor_xl)/((xwin)->w-1) - (draw)->port_xl)/((draw)->port_xr - (draw)->port_xl))
#define JTRANS(draw,xwin,j)  ((draw)->coor_yl + (((PetscReal)(j))/((xwin)->h-1) + (draw)->port_yl - 1)*((draw)->coor_yr - (draw)->coor_yl)/((draw)->port_yl - (draw)->port_yr))

static PetscErrorCode PetscDrawSetViewport_X(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr)
{
  PetscDraw_X    *XiWin = (PetscDraw_X*)draw->data;
  int            xa,ya,xb,yb,xmax = XiWin->w-1,ymax = XiWin->h-1;
  XRectangle     box;

  PetscFunctionBegin;
  xa = (int)(xl*xmax); ya = ymax - (int)(yr*ymax);
  xb = (int)(xr*xmax); yb = ymax - (int)(yl*ymax);
  PetscDrawCollectiveBegin(draw);
  box.x = (short)xa; box.width  = (unsigned short)(xb + 1 - xa);
  box.y = (short)ya; box.height = (unsigned short)(yb + 1 - ya);
  XSetClipRectangles(XiWin->disp,XiWin->gc.set,0,0,&box,1,Unsorted);
  PetscDrawCollectiveEnd(draw);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawCoordinateToPixel_X(PetscDraw draw,PetscReal x,PetscReal y,int *i,int *j)
{
  PetscDraw_X *XiWin = (PetscDraw_X*)draw->data;

  PetscFunctionBegin;
  *i = XTRANS(draw,XiWin,x);
  *j = YTRANS(draw,XiWin,y);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawPixelToCoordinate_X(PetscDraw draw,int i,int j,PetscReal *x,PetscReal *y)
{
  PetscDraw_X *XiWin = (PetscDraw_X*)draw->data;

  PetscFunctionBegin;
  *x = ITRANS(draw,XiWin,i);
  *y = JTRANS(draw,XiWin,j);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawPoint_X(PetscDraw draw,PetscReal x,PetscReal y,int c)
{
  PetscDraw_X *XiWin = (PetscDraw_X*)draw->data;
  int         xx,yy,i,j;

  PetscFunctionBegin;
  xx = XTRANS(draw,XiWin,x);
  yy = YTRANS(draw,XiWin,y);
  PetscDrawXiSetColor(XiWin,c);
  for (i=-1; i<2; i++) {
    for (j=-1; j<2; j++) {
      XDrawPoint(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,xx+i,yy+j);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawPointPixel_X(PetscDraw draw,int x,int y,int c)
{
  PetscDraw_X *XiWin = (PetscDraw_X*)draw->data;

  PetscFunctionBegin;
  PetscDrawXiSetColor(XiWin,c);
  XDrawPoint(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,x,y);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawLine_X(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int cl)
{
  PetscDraw_X *XiWin = (PetscDraw_X*)draw->data;
  int         x_1,y_1,x_2,y_2;

  PetscFunctionBegin;
  PetscDrawXiSetColor(XiWin,cl);
  x_1 = XTRANS(draw,XiWin,xl); x_2  = XTRANS(draw,XiWin,xr);
  y_1 = YTRANS(draw,XiWin,yl); y_2  = YTRANS(draw,XiWin,yr);
  XDrawLine(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,x_1,y_1,x_2,y_2);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawArrow_X(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int cl)
{
  PetscDraw_X *XiWin = (PetscDraw_X*)draw->data;
  int         x_1,y_1,x_2,y_2;

  PetscFunctionBegin;
  PetscDrawXiSetColor(XiWin,cl);
  x_1 = XTRANS(draw,XiWin,xl); x_2 = XTRANS(draw,XiWin,xr);
  y_1 = YTRANS(draw,XiWin,yl); y_2 = YTRANS(draw,XiWin,yr);
  XDrawLine(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,x_1,y_1,x_2,y_2);
  if (x_1 == x_2 && y_1 == y_2) PetscFunctionReturn(0);
  if (x_1 == x_2 && PetscAbs(y_1 - y_2) > 7) {
    if (y_2 > y_1) {
      XDrawLine(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,x_2,y_2,x_2-3,y_2-3);
      XDrawLine(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,x_2,y_2,x_2+3,y_2-3);
    } else {
      XDrawLine(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,x_2,y_2,x_2-3,y_2+3);
      XDrawLine(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,x_2,y_2,x_2+3,y_2+3);
    }
  }
  if (y_1 == y_2 && PetscAbs(x_1 - x_2) > 7) {
    if (x_2 > x_1) {
      XDrawLine(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,x_2-3,y_2-3,x_2,y_2);
      XDrawLine(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,x_2-3,y_2+3,x_2,y_2);
    } else {
      XDrawLine(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,x_2,y_2,x_2+3,y_2-3);
      XDrawLine(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,x_2,y_2,x_2+3,y_2+3);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawRectangle_X(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int c1,int c2,int c3,int c4)
{
  PetscDraw_X *XiWin = (PetscDraw_X*)draw->data;
  int         x,y,w,h,c = (c1 + c2 + c3 + c4)/4;

  PetscFunctionBegin;
  PetscDrawXiSetColor(XiWin,c);
  x = XTRANS(draw,XiWin,xl); w = XTRANS(draw,XiWin,xr) + 1 - x; if (w <= 0) w = 1;
  y = YTRANS(draw,XiWin,yr); h = YTRANS(draw,XiWin,yl) + 1 - y; if (h <= 0) h = 1;
  XFillRectangle(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,x,y,w,h);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawEllipse_X(PetscDraw draw,PetscReal x,PetscReal y,PetscReal a,PetscReal b,int c)
{
  PetscDraw_X *XiWin = (PetscDraw_X*)draw->data;
  int         xA,yA,w,h;

  PetscFunctionBegin;
  PetscDrawXiSetColor(XiWin, c);
  xA = XTRANS(draw,XiWin, x - a/2); w = XTRANS(draw,XiWin, x + a/2) + 1 - xA; w = PetscAbs(w);
  yA = YTRANS(draw,XiWin, y + b/2); h = YTRANS(draw,XiWin, y - b/2) + 1 - yA; h = PetscAbs(h);
  XFillArc(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,xA,yA,w,h,0,360*64);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscDrawInterpolatedTriangle_X(PetscDraw_X*,int,int,int,int,int,int,int,int,int);

static PetscErrorCode PetscDrawTriangle_X(PetscDraw draw,PetscReal X1,PetscReal Y_1,PetscReal X2,PetscReal Y2,PetscReal X3,PetscReal Y3,int c1,int c2,int c3)
{
  PetscDraw_X *XiWin = (PetscDraw_X*)draw->data;

  PetscFunctionBegin;
  if (c1 == c2 && c2 == c3) {
    XPoint pt[3];
    PetscDrawXiSetColor(XiWin,c1);
    pt[0].x = XTRANS(draw,XiWin,X1);
    pt[0].y = YTRANS(draw,XiWin,Y_1);
    pt[1].x = XTRANS(draw,XiWin,X2);
    pt[1].y = YTRANS(draw,XiWin,Y2);
    pt[2].x = XTRANS(draw,XiWin,X3);
    pt[2].y = YTRANS(draw,XiWin,Y3);
    XFillPolygon(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,pt,3,Convex,CoordModeOrigin);
  } else {
    int x1,y_1,x2,y2,x3,y3;
    x1   = XTRANS(draw,XiWin,X1);
    y_1  = YTRANS(draw,XiWin,Y_1);
    x2   = XTRANS(draw,XiWin,X2);
    y2   = YTRANS(draw,XiWin,Y2);
    x3   = XTRANS(draw,XiWin,X3);
    y3   = YTRANS(draw,XiWin,Y3);
    PetscCall(PetscDrawInterpolatedTriangle_X(XiWin,x1,y_1,c1,x2,y2,c2,x3,y3,c3));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawStringSetSize_X(PetscDraw draw,PetscReal x,PetscReal y)
{
  PetscDraw_X *XiWin = (PetscDraw_X*)draw->data;
  int          w,h;

  PetscFunctionBegin;
  w    = (int)((XiWin->w)*x*(draw->port_xr - draw->port_xl)/(draw->coor_xr - draw->coor_xl));
  h    = (int)((XiWin->h)*y*(draw->port_yr - draw->port_yl)/(draw->coor_yr - draw->coor_yl));
  PetscCall(PetscFree(XiWin->font));
  PetscCall(PetscDrawXiFontFixed(XiWin,w,h,&XiWin->font));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawStringGetSize_X(PetscDraw draw,PetscReal *x,PetscReal  *y)
{
  PetscDraw_X *XiWin = (PetscDraw_X*)draw->data;
  PetscReal   w,h;

  PetscFunctionBegin;
  w = XiWin->font->font_w; h = XiWin->font->font_h;
  if (x) *x = w*(draw->coor_xr - draw->coor_xl)/((XiWin->w)*(draw->port_xr - draw->port_xl));
  if (y) *y = h*(draw->coor_yr - draw->coor_yl)/((XiWin->h)*(draw->port_yr - draw->port_yl));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawString_X(PetscDraw draw,PetscReal x,PetscReal y,int c,const char chrs[])
{
  PetscDraw_X *XiWin         = (PetscDraw_X*)draw->data;
  int          xx,yy,descent = XiWin->font->font_descent;
  size_t       len;
  char        *substr;
  PetscToken   token;

  PetscFunctionBegin;
  xx = XTRANS(draw,XiWin,x);
  yy = YTRANS(draw,XiWin,y);
  PetscDrawXiSetColor(XiWin,c);

  PetscCall(PetscTokenCreate(chrs,'\n',&token));
  PetscCall(PetscTokenFind(token,&substr));
  while (substr) {
    PetscCall(PetscStrlen(substr,&len));
    XDrawString(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,xx,yy-descent,substr,len);
    yy  += XiWin->font->font_h;
    PetscCall(PetscTokenFind(token,&substr));
  }
  PetscCall(PetscTokenDestroy(&token));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawStringVertical_X(PetscDraw draw,PetscReal x,PetscReal y,int c,const char text[])
{
  PetscDraw_X    *XiWin = (PetscDraw_X*)draw->data;
  int            xx,yy,offset = XiWin->font->font_h - XiWin->font->font_descent;
  char           chr[2] = {0, 0};

  PetscFunctionBegin;
  xx = XTRANS(draw,XiWin,x);
  yy = YTRANS(draw,XiWin,y);
  PetscDrawXiSetColor(XiWin,c);
  while ((chr[0] = *text++)) {
    XDrawString(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,xx,yy+offset,chr,1);
    yy += XiWin->font->font_h;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawFlush_X(PetscDraw draw)
{
  PetscDraw_X    *XiWin = (PetscDraw_X*)draw->data;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  /* make sure the X server processed requests from all processes */
  PetscDrawCollectiveBegin(draw);
  XSync(XiWin->disp,False);
  PetscDrawCollectiveEnd(draw);
  PetscCallMPI(MPI_Barrier(PetscObjectComm((PetscObject)draw)));

  /* transfer pixmap contents to window (only the first process does this) */
  if (XiWin->drw && XiWin->win) {
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)draw),&rank));
    PetscDrawCollectiveBegin(draw);
    if (rank == 0) XCopyArea(XiWin->disp,XiWin->drw,XiWin->win,XiWin->gc.set,0,0,XiWin->w,XiWin->h,0,0);
    if (rank == 0) XSync(XiWin->disp,False);
    PetscDrawCollectiveEnd(draw);
    PetscCallMPI(MPI_Barrier(PetscObjectComm((PetscObject)draw)));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawClear_X(PetscDraw draw)
{
  PetscDraw_X    *XiWin = (PetscDraw_X*)draw->data;
  int            xmax = XiWin->w-1,  ymax = XiWin->h-1;
  PetscReal      xl = draw->port_xl, yl = draw->port_yl;
  PetscReal      xr = draw->port_xr, yr = draw->port_yr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  /* make sure the X server processed requests from all processes */
  PetscDrawCollectiveBegin(draw);
  XSync(XiWin->disp,False);
  PetscDrawCollectiveEnd(draw);
  PetscCallMPI(MPI_Barrier(PetscObjectComm((PetscObject)draw)));

  /* only the first process handles the clearing business */
  PetscDrawCollectiveBegin(draw);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)draw),&rank));
  if (rank == 0) {
    int xa = (int)(xl*xmax), ya = ymax - (int)(yr*ymax);
    int xb = (int)(xr*xmax), yb = ymax - (int)(yl*ymax);
    unsigned int w = (unsigned int)(xb + 1 - xa);
    unsigned int h = (unsigned int)(yb + 1 - ya);
    PetscDrawXiSetPixVal(XiWin,XiWin->background);
    XFillRectangle(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,xa,ya,w,h);
    XSync(XiWin->disp,False);
  }
  PetscDrawCollectiveEnd(draw);
  PetscCallMPI(MPI_Barrier(PetscObjectComm((PetscObject)draw)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawSetDoubleBuffer_X(PetscDraw draw)
{
  PetscDraw_X    *win = (PetscDraw_X*)draw->data;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  if (win->drw) PetscFunctionReturn(0);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)draw),&rank));

  PetscDrawCollectiveBegin(draw);
  if (rank == 0) PetscCall(PetscDrawXiQuickPixmap(win));
  PetscDrawCollectiveEnd(draw);
  PetscCallMPI(MPI_Bcast(&win->drw,1,MPI_UNSIGNED_LONG,0,PetscObjectComm((PetscObject)draw)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawGetPopup_X(PetscDraw draw,PetscDraw *popup)
{
  PetscDraw_X *win = (PetscDraw_X*)draw->data;
  PetscBool    flg = PETSC_TRUE;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetBool(((PetscObject)draw)->options,((PetscObject)draw)->prefix,"-draw_popup",&flg,NULL));
  if (!flg || !win->win) {*popup = NULL; PetscFunctionReturn(0);}

  PetscCall(PetscDrawCreate(PetscObjectComm((PetscObject)draw),draw->display,NULL,win->x,win->y+win->h+10,220,220,popup));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*popup,"popup_"));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)*popup,((PetscObject)draw)->prefix));
  PetscCall(PetscDrawSetType(*popup,PETSC_DRAW_X));
  draw->popup = *popup;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawSetTitle_X(PetscDraw draw,const char title[])
{
  PetscDraw_X    *win = (PetscDraw_X*)draw->data;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  if (!win->win) PetscFunctionReturn(0);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)draw),&rank));
  PetscDrawCollectiveBegin(draw);
  if (rank == 0) {
    size_t        len;
    XTextProperty prop;
    PetscCall(PetscStrlen(title,&len));
    XGetWMName(win->disp,win->win,&prop);
    XFree((void*)prop.value);
    prop.value  = (unsigned char*)title;
    prop.nitems = (long)len;
    XSetWMName(win->disp,win->win,&prop);
  }
  PetscDrawCollectiveEnd(draw);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawCheckResizedWindow_X(PetscDraw draw)
{
  PetscDraw_X    *win = (PetscDraw_X*)draw->data;
  int            xywh[4];
  PetscMPIInt    rank;

  PetscFunctionBegin;
  if (!win->win) PetscFunctionReturn(0);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)draw),&rank));

  PetscDrawCollectiveBegin(draw);
  if (rank == 0) PetscCall(PetscDrawXiGetGeometry(win,xywh,xywh+1,xywh+2,xywh+3));
  PetscDrawCollectiveEnd(draw);
  PetscCallMPI(MPI_Bcast(xywh,4,MPI_INT,0,PetscObjectComm((PetscObject)draw)));

  /* record new window position */
  draw->x = win->x = xywh[0];
  draw->y = win->y = xywh[1];
  if (xywh[2] == win->w && xywh[3] == win->h) PetscFunctionReturn(0);
  /* record new window sizes */
  draw->w = win->w = xywh[2];
  draw->h = win->h = xywh[3];

  /* recreate pixmap (only first processor does this) */
  PetscDrawCollectiveBegin(draw);
  if (rank == 0 && win->drw) PetscCall(PetscDrawXiQuickPixmap(win));
  PetscDrawCollectiveEnd(draw);
  PetscCallMPI(MPI_Bcast(&win->drw,1,MPI_UNSIGNED_LONG,0,PetscObjectComm((PetscObject)draw)));
  /* reset the clipping */
  PetscCall(PetscDrawSetViewport_X(draw,draw->port_xl,draw->port_yl,draw->port_xr,draw->port_yr));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawResizeWindow_X(PetscDraw draw,int w,int h)
{
  PetscDraw_X    *win = (PetscDraw_X*)draw->data;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  if (w == win->w && h == win->h) PetscFunctionReturn(0);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)draw),&rank));

  if (win->win) {
    PetscDrawCollectiveBegin(draw);
    if (rank == 0) PetscCall(PetscDrawXiResizeWindow(win,w,h));
    PetscDrawCollectiveEnd(draw);
    PetscCall(PetscDrawCheckResizedWindow_X(draw));
  } else if (win->drw) {
    draw->w = win->w = w; draw->h = win->h = h;
    /* recreate pixmap (only first processor does this) */
    PetscDrawCollectiveBegin(draw);
    if (rank == 0) PetscCall(PetscDrawXiQuickPixmap(win));
    PetscCallMPI(MPI_Bcast(&win->drw,1,MPI_UNSIGNED_LONG,0,PetscObjectComm((PetscObject)draw)));
    /* reset the clipping */
    PetscDrawCollectiveEnd(draw);
    PetscCall(PetscDrawSetViewport_X(draw,draw->port_xl,draw->port_yl,draw->port_xr,draw->port_yr));
  }
  PetscFunctionReturn(0);
}

#include <X11/cursorfont.h>

static PetscErrorCode PetscDrawGetMouseButton_X(PetscDraw draw,PetscDrawButton *button,PetscReal *x_user,PetscReal *y_user,PetscReal *x_phys,PetscReal *y_phys)
{
  PetscDraw_X    *win = (PetscDraw_X*)draw->data;
  Cursor         cursor;
  XEvent         report;
  Window         root,child;
  int            root_x,root_y,px=0,py=0;
  unsigned int   w,h,border,depth;
  unsigned int   keys_button;
  PetscMPIInt    rank;
  PetscReal      xx,yy;

  PetscFunctionBegin;
  *button = PETSC_BUTTON_NONE;
  if (!win->win) PetscFunctionReturn(0);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)draw),&rank));

  PetscDrawCollectiveBegin(draw);
  if (rank) goto finally;

  /* change cursor to indicate input */
  cursor = XCreateFontCursor(win->disp,XC_hand2); PetscCheck(cursor,PETSC_COMM_SELF,PETSC_ERR_LIB,"Unable to create X cursor");
  XDefineCursor(win->disp,win->win,cursor);
  /* wait for mouse button events */
  XSelectInput(win->disp,win->win,ButtonPressMask|ButtonReleaseMask);
  while (XCheckTypedEvent(win->disp,ButtonPress,&report));
  XMaskEvent(win->disp,ButtonReleaseMask,&report);
  /* get mouse pointer coordinates */
  XQueryPointer(win->disp,report.xmotion.window,&root,&child,&root_x,&root_y,&px,&py,&keys_button);
  /* the user may resize the window before pressing the mouse button */
  XGetGeometry(win->disp,win->win,&root,&root_x,&root_y,&w,&h,&border,&depth);
  /* cleanup input event handler and cursor  */
  XSelectInput(win->disp,win->win,NoEventMask);
  XUndefineCursor(win->disp,win->win);
  XFreeCursor(win->disp, cursor);
  XSync(win->disp,False);

  switch (report.xbutton.button) {
  case Button1: *button = PETSC_BUTTON_LEFT; break;
  case Button2: *button = PETSC_BUTTON_CENTER; break;
  case Button3: *button = PETSC_BUTTON_RIGHT; break;
  case Button4: *button = PETSC_BUTTON_WHEEL_UP; break;
  case Button5: *button = PETSC_BUTTON_WHEEL_DOWN; break;
  }
  if (report.xbutton.state & ShiftMask) {
    switch (report.xbutton.button) {
    case Button1: *button = PETSC_BUTTON_LEFT_SHIFT; break;
    case Button2: *button = PETSC_BUTTON_CENTER_SHIFT; break;
    case Button3: *button = PETSC_BUTTON_RIGHT_SHIFT; break;
    }
  }
  xx = ((PetscReal)px)/w;
  yy = 1 - ((PetscReal)py)/h;
  if (x_user) *x_user = draw->coor_xl + (xx - draw->port_xl)*(draw->coor_xr - draw->coor_xl)/(draw->port_xr - draw->port_xl);
  if (y_user) *y_user = draw->coor_yl + (yy - draw->port_yl)*(draw->coor_yr - draw->coor_yl)/(draw->port_yr - draw->port_yl);
  if (x_phys) *x_phys = xx;
  if (y_phys) *y_phys = yy;

finally:
  PetscDrawCollectiveEnd(draw);
  PetscCall(PetscDrawCheckResizedWindow_X(draw));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawPause_X(PetscDraw draw)
{
  PetscDraw_X *win = (PetscDraw_X*)draw->data;

  PetscFunctionBegin;
  if (!win->win) PetscFunctionReturn(0);
  if (draw->pause > 0) PetscSleep(draw->pause);
  else if (draw->pause == -1) {
    PetscDrawButton button = PETSC_BUTTON_NONE;
    PetscCall(PetscDrawGetMouseButton(draw,&button,NULL,NULL,NULL,NULL));
    if (button == PETSC_BUTTON_CENTER) draw->pause = 0;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawDestroy_X(PetscDraw draw)
{
  PetscDraw_X *win = (PetscDraw_X*)draw->data;

  PetscFunctionBegin;
  PetscCall(PetscDrawDestroy(&draw->popup));
  PetscCall(PetscDrawXiClose(win));
  PetscCall(PetscFree(draw->data));
  PetscFunctionReturn(0);
}

static       PetscErrorCode PetscDrawGetSingleton_X(PetscDraw,PetscDraw*);
static       PetscErrorCode PetscDrawRestoreSingleton_X(PetscDraw,PetscDraw*);
PETSC_INTERN PetscErrorCode PetscDrawGetImage_X(PetscDraw,unsigned char[][3],unsigned int*,unsigned int*,unsigned char*[]);

static struct _PetscDrawOps DvOps = { PetscDrawSetDoubleBuffer_X,
                                      PetscDrawFlush_X,
                                      PetscDrawLine_X,
                                      NULL,
                                      NULL,
                                      PetscDrawPoint_X,
                                      NULL,
                                      PetscDrawString_X,
                                      PetscDrawStringVertical_X,
                                      PetscDrawStringSetSize_X,
                                      PetscDrawStringGetSize_X,
                                      PetscDrawSetViewport_X,
                                      PetscDrawClear_X,
                                      PetscDrawRectangle_X,
                                      PetscDrawTriangle_X,
                                      PetscDrawEllipse_X,
                                      PetscDrawGetMouseButton_X,
                                      PetscDrawPause_X,
                                      NULL,
                                      NULL,
                                      PetscDrawGetPopup_X,
                                      PetscDrawSetTitle_X,
                                      PetscDrawCheckResizedWindow_X,
                                      PetscDrawResizeWindow_X,
                                      PetscDrawDestroy_X,
                                      NULL,
                                      PetscDrawGetSingleton_X,
                                      PetscDrawRestoreSingleton_X,
                                      NULL,
                                      PetscDrawGetImage_X,
                                      NULL,
                                      PetscDrawArrow_X,
                                      PetscDrawCoordinateToPixel_X,
                                      PetscDrawPixelToCoordinate_X,
                                      PetscDrawPointPixel_X,
                                      NULL};

static PetscErrorCode PetscDrawGetSingleton_X(PetscDraw draw,PetscDraw *sdraw)
{
  PetscDraw_X *Xwin = (PetscDraw_X*)draw->data,*sXwin;

  PetscFunctionBegin;
  PetscCall(PetscDrawCreate(PETSC_COMM_SELF,draw->display,draw->title,draw->x,draw->y,draw->w,draw->h,sdraw));
  PetscCall(PetscObjectChangeTypeName((PetscObject)*sdraw,PETSC_DRAW_X));
  PetscCall(PetscMemcpy((*sdraw)->ops,&DvOps,sizeof(DvOps)));

  if (draw->popup) PetscCall(PetscDrawGetSingleton(draw->popup,&(*sdraw)->popup));
  (*sdraw)->pause   = draw->pause;
  (*sdraw)->coor_xl = draw->coor_xl;
  (*sdraw)->coor_xr = draw->coor_xr;
  (*sdraw)->coor_yl = draw->coor_yl;
  (*sdraw)->coor_yr = draw->coor_yr;
  (*sdraw)->port_xl = draw->port_xl;
  (*sdraw)->port_xr = draw->port_xr;
  (*sdraw)->port_yl = draw->port_yl;
  (*sdraw)->port_yr = draw->port_yr;

  /* share drawables (windows and/or pixmap) from the parent draw */
  PetscCall(PetscNewLog(*sdraw,&sXwin));
  (*sdraw)->data = (void*)sXwin;
  PetscCall(PetscDrawXiInit(sXwin,draw->display));
  if (Xwin->win) {
    PetscCall(PetscDrawXiQuickWindowFromWindow(sXwin,Xwin->win));
    sXwin->drw = Xwin->drw; /* XXX If the window is ever resized, this is wrong! */
  } else if (Xwin->drw) {
    PetscCall(PetscDrawXiColormap(sXwin));
    sXwin->drw = Xwin->drw;
  }
  PetscCall(PetscDrawXiGetGeometry(sXwin,&sXwin->x,&sXwin->y,&sXwin->w,&sXwin->h));
  (*sdraw)->x = sXwin->x; (*sdraw)->y = sXwin->y;
  (*sdraw)->w = sXwin->w; (*sdraw)->h = sXwin->h;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawRestoreSingleton_X(PetscDraw draw,PetscDraw *sdraw)
{
  PetscFunctionBegin;
  if (draw->popup && (*sdraw)->popup) {
    PetscBool isdrawx;
    PetscDraw_X *pXwin = (PetscDraw_X*)draw->popup->data;
    PetscDraw_X *sXwin = (PetscDraw_X*)(*sdraw)->popup->data;
    PetscCall(PetscObjectTypeCompare((PetscObject)draw->popup,PETSC_DRAW_X,&isdrawx));
    if (!isdrawx) goto finally;
    PetscCall(PetscObjectTypeCompare((PetscObject)(*sdraw)->popup,PETSC_DRAW_X,&isdrawx));
    if (!isdrawx) goto finally;
    if (sXwin->win == pXwin->win) {
      PetscCall(PetscDrawRestoreSingleton(draw->popup,&(*sdraw)->popup));
    }
  }
finally:
  PetscCall(PetscDrawDestroy(sdraw));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawXGetDisplaySize_Private(const char name[],int *width,int *height,PetscBool *has_display)
{
  Display *display;

  PetscFunctionBegin;
  display = XOpenDisplay(name);
  if (!display) {
    *width  = *height = 0;
    (*PetscErrorPrintf)("Unable to open display on %s\n\
    Make sure your COMPUTE NODES are authorized to connect\n\
    to this X server and either your DISPLAY variable\n\
    is set or you use the -display name option\n",name);
    *has_display = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  *has_display = PETSC_TRUE;
  *width  = (int)DisplayWidth(display,DefaultScreen(display));
  *height = (int)DisplayHeight(display,DefaultScreen(display));
  XCloseDisplay(display);
  PetscFunctionReturn(0);
}

/*MC
     PETSC_DRAW_X  - PETSc graphics device that uses either X windows or its virtual version Xvfb

   Options Database Keys:
+  -display <display> - sets the display to use
.  -x_virtual - forces use of a X virtual display Xvfb that will not display anything but -draw_save will still work.
                Xvfb is automatically started up in PetscSetDisplay() with this option
.  -draw_size w,h - percentage of screen (either 1, .5, .3, .25), or size in pixels
.  -geometry x,y,w,h - set location and size in pixels
.  -draw_virtual - do not open a window (draw on a pixmap), -draw_save will still work
-  -draw_double_buffer - avoid window flickering (draw on pixmap and flush to window)

   Level: beginner

.seealso: `PetscDrawOpenX()`, `PetscDrawSetDisplay()`, `PetscDrawSetFromOptions()`

M*/

PETSC_EXTERN PetscErrorCode PetscDrawCreate_X(PetscDraw draw)
{
  PetscDraw_X    *Xwin;
  PetscMPIInt    rank;
  int            x = draw->x,y = draw->y,w = draw->w,h = draw->h;
  static int     xavailable = 0,yavailable = 0,ybottom = 0,xmax = 0,ymax = 0;
  PetscBool      set,dvirtual = PETSC_FALSE,doublebuffer = PETSC_TRUE,has_display;
  PetscInt       xywh[4],osize = 4,nsizes=2;
  PetscReal      sizes[2] = {.3,.3};
  static size_t  DISPLAY_LENGTH = 265;

  PetscFunctionBegin;
  /* get the display variable */
  if (!draw->display) {
    PetscCall(PetscMalloc1(DISPLAY_LENGTH,&draw->display));
    PetscCall(PetscGetDisplay(draw->display,DISPLAY_LENGTH));
  }

  /* initialize the display size */
  if (!xmax) {
    PetscCall(PetscDrawXGetDisplaySize_Private(draw->display,&xmax,&ymax,&has_display));
    /* if some processors fail on this and others succed then this is a problem ! */
    if (!has_display) {
      (*PetscErrorPrintf)("PETSc unable to use X windows\nproceeding without graphics\n");
      PetscCall(PetscDrawSetType(draw,PETSC_DRAW_NULL));
      PetscFunctionReturn(0);
    }
  }

  /* allow user to set size of drawable */
  PetscCall(PetscOptionsGetRealArray(((PetscObject)draw)->options,((PetscObject)draw)->prefix,"-draw_size",sizes,&nsizes,&set));
  if (set && nsizes == 1 && sizes[0] > 1.0) sizes[1] = sizes[0];
  if (set) {
    if (sizes[0] > 1.0)       w = (int)sizes[0];
    else if (sizes[0] == 1.0) w = PETSC_DRAW_FULL_SIZE;
    else if (sizes[0] == .5)  w = PETSC_DRAW_HALF_SIZE;
    else if (sizes[0] == .3)  w = PETSC_DRAW_THIRD_SIZE;
    else if (sizes[0] == .25) w = PETSC_DRAW_QUARTER_SIZE;
    if (sizes[1] > 1.0)       h = (int)sizes[1];
    else if (sizes[1] == 1.0) h = PETSC_DRAW_FULL_SIZE;
    else if (sizes[1] == .5)  h = PETSC_DRAW_HALF_SIZE;
    else if (sizes[1] == .3)  h = PETSC_DRAW_THIRD_SIZE;
    else if (sizes[1] == .25) h = PETSC_DRAW_QUARTER_SIZE;
  }
  if (w == PETSC_DECIDE || w == PETSC_DEFAULT) w = draw->w = 300;
  if (h == PETSC_DECIDE || h == PETSC_DEFAULT) h = draw->h = 300;
  switch (w) {
  case PETSC_DRAW_FULL_SIZE:    w = draw->w = (xmax - 10);   break;
  case PETSC_DRAW_HALF_SIZE:    w = draw->w = (xmax - 20)/2; break;
  case PETSC_DRAW_THIRD_SIZE:   w = draw->w = (xmax - 30)/3; break;
  case PETSC_DRAW_QUARTER_SIZE: w = draw->w = (xmax - 40)/4; break;
  }
  switch (h) {
  case PETSC_DRAW_FULL_SIZE:    h = draw->h = (ymax - 10);   break;
  case PETSC_DRAW_HALF_SIZE:    h = draw->h = (ymax - 20)/2; break;
  case PETSC_DRAW_THIRD_SIZE:   h = draw->h = (ymax - 30)/3; break;
  case PETSC_DRAW_QUARTER_SIZE: h = draw->h = (ymax - 40)/4; break;
  }

  PetscCall(PetscOptionsGetBool(((PetscObject)draw)->options,((PetscObject)draw)->prefix,"-draw_virtual",&dvirtual,NULL));

  if (!dvirtual) {

    /* allow user to set location and size of window */
    xywh[0] = x; xywh[1] = y; xywh[2] = w; xywh[3] = h;
    PetscCall(PetscOptionsGetIntArray(((PetscObject)draw)->options,((PetscObject)draw)->prefix,"-geometry",xywh,&osize,NULL));
    x = (int)xywh[0]; y = (int)xywh[1]; w = (int)xywh[2]; h = (int)xywh[3];
    if (w == PETSC_DECIDE || w == PETSC_DEFAULT) w = 300;
    if (h == PETSC_DECIDE || h == PETSC_DEFAULT) h = 300;
    draw->x = x; draw->y = y; draw->w = w; draw->h = h;

    if (draw->x == PETSC_DECIDE || draw->y == PETSC_DECIDE) {
      /*
       PETSc tries to place windows starting in the upper left corner
        and moving across to the right.

       +0,0-------------------------------------------+
       |  Region used so far  +xavailable,yavailable  |
       |                      |                       |
       |                      |                       |
       +--------------------- +ybottom                |
       |                                              |
       |                                              |
       +----------------------------------------------+xmax,ymax

      */
      /*  First: can we add it to the right? */
      if (xavailable + w + 10 <= xmax) {
        x       = xavailable;
        y       = yavailable;
        ybottom = PetscMax(ybottom,y + h + 30);
      } else {
        /* No, so add it below on the left */
        xavailable = x = 0;
        yavailable = y = ybottom;
        ybottom    = ybottom + h + 30;
      }
    }
    /* update available region */
    xavailable = PetscMax(xavailable,x + w + 10);
    if (xavailable >= xmax) {
      xavailable = 0;
      yavailable = yavailable + h + 30;
      ybottom    = yavailable;
    }
    if (yavailable >= ymax) {
      y          = 0;
      yavailable = 0;
      ybottom    = 0;
    }

  } /* endif (!dvirtual) */

  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)draw),&rank));
  PetscCheck(rank != 0 || (w > 0 && h > 0),PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative window width or height");

  PetscCall(PetscNewLog(draw,&Xwin));
  PetscCall(PetscMemcpy(draw->ops,&DvOps,sizeof(DvOps)));
  draw->data = (void*)Xwin;

  PetscCall(PetscDrawXiInit(Xwin,draw->display));
  if (!dvirtual) {
    Xwin->x = x; Xwin->y = y;
    Xwin->w = w; Xwin->h = h;
    if (rank == 0) PetscCall(PetscDrawXiQuickWindow(Xwin,draw->title,x,y,w,h));
    PetscCallMPI(MPI_Bcast(&Xwin->win,1,MPI_UNSIGNED_LONG,0,PetscObjectComm((PetscObject)draw)));
    if (rank) PetscCall(PetscDrawXiQuickWindowFromWindow(Xwin,Xwin->win));
  } else {
    Xwin->x = 0; Xwin->y = 0;
    Xwin->w = w; Xwin->h = h;
    PetscCall(PetscDrawXiColormap(Xwin));
    if (rank == 0) PetscCall(PetscDrawXiQuickPixmap(Xwin));
    PetscCallMPI(MPI_Bcast(&Xwin->drw,1,MPI_UNSIGNED_LONG,0,PetscObjectComm((PetscObject)draw)));
  }
  PetscCall(PetscDrawXiGetGeometry(Xwin,&Xwin->x,&Xwin->y,&Xwin->w,&Xwin->h));
  draw->x = Xwin->x; draw->y = Xwin->y;
  draw->w = Xwin->w; draw->h = Xwin->h;

  PetscCall(PetscOptionsGetBool(((PetscObject)draw)->options,((PetscObject)draw)->prefix,"-draw_double_buffer",&doublebuffer,NULL));
  if (doublebuffer) PetscCall(PetscDrawSetDoubleBuffer(draw));
  PetscFunctionReturn(0);
}
