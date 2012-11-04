
/*
    Defines the operations for the X PetscDraw implementation.
*/

#include <../src/sys/draw/impls/x/ximpl.h>         /*I  "petscsys.h" I*/

/*
     These macros transform from the users coordinates to the  X-window pixel coordinates.
*/
#define XTRANS(draw,xwin,x)  (int)(((xwin)->w)*((draw)->port_xl + (((x - (draw)->coor_xl)*((draw)->port_xr - (draw)->port_xl))/((draw)->coor_xr - (draw)->coor_xl))))
#define YTRANS(draw,xwin,y)  (int)(((xwin)->h)*(1.0-(draw)->port_yl - (((y - (draw)->coor_yl)*((draw)->port_yr - (draw)->port_yl))/((draw)->coor_yr - (draw)->coor_yl))))

#define ITRANS(draw,xwin,i)  (draw)->coor_xl + (i*((draw)->coor_xr - (draw)->coor_xl)/((xwin)->w) - (draw)->port_xl)/((draw)->port_xr - (draw)->port_xl)
#define JTRANS(draw,xwin,j)  draw->coor_yl + (((double)j)/xwin->h + draw->port_yl - 1.0)*(draw->coor_yr - draw->coor_yl)/(draw->port_yl - draw->port_yr)

#undef __FUNCT__
#define __FUNCT__ "PetscDrawCoordinateToPixel_X"
PetscErrorCode PetscDrawCoordinateToPixel_X(PetscDraw draw,PetscReal x,PetscReal y,PetscInt *i,PetscInt *j)
{
  PetscDraw_X*   XiWin = (PetscDraw_X*)draw->data;

  PetscFunctionBegin;
  *i = XTRANS(draw,XiWin,x);
  *j = YTRANS(draw,XiWin,y);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawPixelToCoordinate_X"
PetscErrorCode PetscDrawPixelToCoordinate_X(PetscDraw draw,PetscInt i,PetscInt j,PetscReal *x,PetscReal *y)
{
  PetscDraw_X*   XiWin = (PetscDraw_X*)draw->data;

  PetscFunctionBegin;
  *x = ITRANS(draw,XiWin,i);
  *y = JTRANS(draw,XiWin,j);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawLine_X"
PetscErrorCode PetscDrawLine_X(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int cl)
{
  PetscDraw_X* XiWin = (PetscDraw_X*)draw->data;
  int          x1,y_1,x2,y2;

  PetscFunctionBegin;
  PetscDrawXiSetColor(XiWin,cl);
  x1 = XTRANS(draw,XiWin,xl);   x2  = XTRANS(draw,XiWin,xr);
  y_1 = YTRANS(draw,XiWin,yl);   y2  = YTRANS(draw,XiWin,yr);
  if (x1 == x2 && y_1 == y2) PetscFunctionReturn(0);
  XDrawLine(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,x1,y_1,x2,y2);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawArrow_X"
PetscErrorCode PetscDrawArrow_X(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int cl)
{
  PetscDraw_X* XiWin = (PetscDraw_X*)draw->data;
  int          x1,y_1,x2,y2;

  PetscFunctionBegin;
  PetscDrawXiSetColor(XiWin,cl);
  x1 = XTRANS(draw,XiWin,xl);   x2  = XTRANS(draw,XiWin,xr);
  y_1 = YTRANS(draw,XiWin,yl);   y2  = YTRANS(draw,XiWin,yr);
  if (x1 == x2 && y_1 == y2) PetscFunctionReturn(0);
  XDrawLine(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,x1,y_1,x2,y2);
  if (x1 == x2 && PetscAbs(y_1 - y2) > 7) {
    if (y2 > y_1) {
       XDrawLine(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,x2,y2,x2-3,y2-3);
       XDrawLine(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,x2,y2,x2+3,y2-3);
    } else {
       XDrawLine(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,x2,y2,x2-3,y2+3);
       XDrawLine(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,x2,y2,x2+3,y2+3);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawPoint_X"
static PetscErrorCode PetscDrawPoint_X(PetscDraw draw,PetscReal x,PetscReal  y,int c)
{
  int          xx,yy,i,j;
  PetscDraw_X* XiWin = (PetscDraw_X*)draw->data;

  PetscFunctionBegin;
  xx = XTRANS(draw,XiWin,x);  yy = YTRANS(draw,XiWin,y);
  PetscDrawXiSetColor(XiWin,c);
  for (i=-1; i<2; i++) {
    for (j=-1; j<2; j++) {
      XDrawPoint(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,xx+i,yy+j);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawPointPixel_X"
static PetscErrorCode PetscDrawPointPixel_X(PetscDraw draw,PetscInt x,PetscInt  y,int c)
{
  PetscDraw_X* XiWin = (PetscDraw_X*)draw->data;

  PetscFunctionBegin;
  PetscDrawXiSetColor(XiWin,c);
  XDrawPoint(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,x,y);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawRectangle_X"
static PetscErrorCode PetscDrawRectangle_X(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int c1,int c2,int c3,int c4)
{
  PetscDraw_X* XiWin = (PetscDraw_X*)draw->data;
  int          x1,y_1,w,h,c = (c1 + c2 + c3 + c4)/4;

  PetscFunctionBegin;
  PetscDrawXiSetColor(XiWin,c);
  x1 = XTRANS(draw,XiWin,xl);   w  = XTRANS(draw,XiWin,xr) - x1;
  y_1 = YTRANS(draw,XiWin,yr);   h  = YTRANS(draw,XiWin,yl) - y_1;
  if (w <= 0) w = 1; if (h <= 0) h = 1;
  XFillRectangle(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,x1,y_1,w,h);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawEllipse_X"
static PetscErrorCode PetscDrawEllipse_X(PetscDraw Win, PetscReal x, PetscReal y, PetscReal a, PetscReal b, int c)
{
  PetscDraw_X* XiWin = (PetscDraw_X*) Win->data;
  int          xA, yA, w, h;

  PetscFunctionBegin;
  PetscDrawXiSetColor(XiWin, c);
  xA = XTRANS(Win, XiWin, x - a/2.0); w = XTRANS(Win, XiWin, x + a/2.0) - xA;
  yA = YTRANS(Win, XiWin, y + b/2.0); h = PetscAbs(YTRANS(Win, XiWin, y - b/2.0) - yA);
  XFillArc(XiWin->disp, PetscDrawXiDrawable(XiWin), XiWin->gc.set, xA, yA, w, h, 0, 23040);
  PetscFunctionReturn(0);
}

extern PetscErrorCode PetscDrawInterpolatedTriangle_X(PetscDraw_X*,int,int,int,int,int,int,int,int,int);

#undef __FUNCT__
#define __FUNCT__ "PetscDrawTriangle_X"
static PetscErrorCode PetscDrawTriangle_X(PetscDraw draw,PetscReal X1,PetscReal Y_1,PetscReal X2,PetscReal Y2,PetscReal X3,PetscReal Y3,int c1,int c2,int c3)
{
  PetscDraw_X*   XiWin = (PetscDraw_X*)draw->data;
  PetscErrorCode ierr;

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
    ierr = PetscDrawInterpolatedTriangle_X(XiWin,x1,y_1,c1,x2,y2,c2,x3,y3,c3);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawString_X"
static PetscErrorCode PetscDrawString_X(PetscDraw draw,PetscReal x,PetscReal  y,int c,const char chrs[])
{
  PetscErrorCode ierr;
  int            xx,yy;
  size_t         len;
  PetscDraw_X    *XiWin = (PetscDraw_X*)draw->data;
  char           *substr;
  PetscToken     token;

  PetscFunctionBegin;
  xx = XTRANS(draw,XiWin,x);  yy = YTRANS(draw,XiWin,y);
  PetscDrawXiSetColor(XiWin,c);

  ierr = PetscTokenCreate(chrs,'\n',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&substr);CHKERRQ(ierr);
  ierr = PetscStrlen(substr,&len);CHKERRQ(ierr);
  XDrawString(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,xx,yy - XiWin->font->font_descent,substr,len);
  ierr = PetscTokenFind(token,&substr);CHKERRQ(ierr);
  while (substr) {
    yy  += 4*XiWin->font->font_descent;
    ierr = PetscStrlen(substr,&len);CHKERRQ(ierr);
    XDrawString(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,xx,yy - XiWin->font->font_descent,substr,len);
    ierr = PetscTokenFind(token,&substr);CHKERRQ(ierr);
  }
  ierr = PetscTokenDestroy(&token);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

extern PetscErrorCode PetscDrawXiFontFixed(PetscDraw_X*,int,int,PetscDrawXiFont **);

#undef __FUNCT__
#define __FUNCT__ "PetscDrawStringSetSize_X"
static PetscErrorCode PetscDrawStringSetSize_X(PetscDraw draw,PetscReal x,PetscReal  y)
{
  PetscDraw_X*   XiWin = (PetscDraw_X*)draw->data;
  PetscErrorCode ierr;
  int            w,h;

  PetscFunctionBegin;
  w = (int)((XiWin->w)*x*(draw->port_xr - draw->port_xl)/(draw->coor_xr - draw->coor_xl));
  h = (int)((XiWin->h)*y*(draw->port_yr - draw->port_yl)/(draw->coor_yr - draw->coor_yl));
  ierr = PetscFree(XiWin->font);CHKERRQ(ierr);
  ierr = PetscDrawXiFontFixed(XiWin,w,h,&XiWin->font);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawStringGetSize_X"
PetscErrorCode PetscDrawStringGetSize_X(PetscDraw draw,PetscReal *x,PetscReal  *y)
{
  PetscDraw_X* XiWin = (PetscDraw_X*)draw->data;
  PetscReal    w,h;

  PetscFunctionBegin;
  w = XiWin->font->font_w; h = XiWin->font->font_h;
  *x = w*(draw->coor_xr - draw->coor_xl)/((XiWin->w)*(draw->port_xr - draw->port_xl));
  *y = h*(draw->coor_yr - draw->coor_yl)/((XiWin->h)*(draw->port_yr - draw->port_yl));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawStringVertical_X"
PetscErrorCode PetscDrawStringVertical_X(PetscDraw draw,PetscReal x,PetscReal  y,int c,const char chrs[])
{
  PetscErrorCode ierr;
  int            xx,yy;
  PetscDraw_X    *XiWin = (PetscDraw_X*)draw->data;
  char           tmp[2];
  PetscReal      tw,th;
  size_t         i,n;

  PetscFunctionBegin;
  ierr   = PetscStrlen(chrs,&n);CHKERRQ(ierr);
  tmp[1] = 0;
  PetscDrawXiSetColor(XiWin,c);
  ierr = PetscDrawStringGetSize_X(draw,&tw,&th);CHKERRQ(ierr);
  xx = XTRANS(draw,XiWin,x);
  for (i=0; i<n; i++) {
    tmp[0] = chrs[i];
    yy = YTRANS(draw,XiWin,y-th*i);
    XDrawString(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set, xx,yy - XiWin->font->font_descent,tmp,1);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawFlush_X"
static PetscErrorCode PetscDrawFlush_X(PetscDraw draw)
{
  PetscDraw_X* XiWin = (PetscDraw_X*)draw->data;

  PetscFunctionBegin;
  if (XiWin->drw && XiWin->win) {
    XCopyArea(XiWin->disp,XiWin->drw,XiWin->win,XiWin->gc.set,0,0,XiWin->w,XiWin->h,0,0);
  }
  XFlush(XiWin->disp);
  XSync(XiWin->disp,False);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawSynchronizedFlush_X"
static PetscErrorCode PetscDrawSynchronizedFlush_X(PetscDraw draw)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscDraw_X*   XiWin = (PetscDraw_X*)draw->data;

  PetscFunctionBegin;
  if (XiWin->drw && XiWin->win) {
    ierr = MPI_Comm_rank(((PetscObject)draw)->comm,&rank);CHKERRQ(ierr);
    /* make sure data has actually arrived at server */
    XSync(XiWin->disp,False);
    ierr = MPI_Barrier(((PetscObject)draw)->comm);CHKERRQ(ierr);
    if (!rank) {
      XCopyArea(XiWin->disp,XiWin->drw,XiWin->win,XiWin->gc.set,0,0,XiWin->w,XiWin->h,0,0);
      XFlush(XiWin->disp);
      XSync(XiWin->disp,False);
    }
    ierr = MPI_Barrier(((PetscObject)draw)->comm);CHKERRQ(ierr);
  } else {
    ierr = MPI_Barrier(((PetscObject)draw)->comm);CHKERRQ(ierr);
    XSync(XiWin->disp,False);
    ierr = MPI_Barrier(((PetscObject)draw)->comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawSetViewport_X"
static PetscErrorCode PetscDrawSetViewport_X(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr)
{
  PetscDraw_X* XiWin = (PetscDraw_X*)draw->data;
  XRectangle   box;

  PetscFunctionBegin;
  box.x = (int)(xl*XiWin->w);   box.y = (int)((1.0-yr)*XiWin->h);
  box.width = (int)((xr-xl)*XiWin->w);box.height = (int)((yr-yl)*XiWin->h);
  XSetClipRectangles(XiWin->disp,XiWin->gc.set,0,0,&box,1,Unsorted);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawClear_X"
static PetscErrorCode PetscDrawClear_X(PetscDraw draw)
{
  PetscDraw_X*   XiWin = (PetscDraw_X*)draw->data;
  int            x, y, w, h;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDrawSave(draw);CHKERRQ(ierr);
  x = (int)(draw->port_xl*XiWin->w);
  w = (int)((draw->port_xr - draw->port_xl)*XiWin->w);
  y = (int)((1.0-draw->port_yr)*XiWin->h);
  h = (int)((draw->port_yr - draw->port_yl)*XiWin->h);
  PetscDrawXiSetPixVal(XiWin,XiWin->background);
  XFillRectangle(XiWin->disp,PetscDrawXiDrawable(XiWin),XiWin->gc.set,x,y,w,h);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawSynchronizedClear_X"
static PetscErrorCode PetscDrawSynchronizedClear_X(PetscDraw draw)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscDraw_X*   XiWin = (PetscDraw_X*)draw->data;

  PetscFunctionBegin;
  ierr = MPI_Barrier(((PetscObject)draw)->comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)draw)->comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscDrawClear_X(draw);CHKERRQ(ierr);
  }
  XFlush(XiWin->disp);
  ierr = MPI_Barrier(((PetscObject)draw)->comm);CHKERRQ(ierr);
  XSync(XiWin->disp,False);
  ierr = MPI_Barrier(((PetscObject)draw)->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawSetDoubleBuffer_X"
static PetscErrorCode PetscDrawSetDoubleBuffer_X(PetscDraw draw)
{
  PetscDraw_X*   win = (PetscDraw_X*)draw->data;
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  if (win->drw) PetscFunctionReturn(0);

  ierr = MPI_Comm_rank(((PetscObject)draw)->comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    win->drw = XCreatePixmap(win->disp,win->win,win->w,win->h,win->depth);
  }
  /* try to make sure it is actually done before passing info to all */
  XSync(win->disp,False);
  ierr = MPI_Bcast(&win->drw,1,MPI_UNSIGNED_LONG,0,((PetscObject)draw)->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <X11/cursorfont.h>

#undef __FUNCT__
#define __FUNCT__ "PetscDrawGetMouseButton_X"
static PetscErrorCode PetscDrawGetMouseButton_X(PetscDraw draw,PetscDrawButton *button,PetscReal* x_user,PetscReal *y_user,PetscReal *x_phys,PetscReal *y_phys)
{
  XEvent       report;
  PetscDraw_X* win = (PetscDraw_X*)draw->data;
  Window       root,child;
  int          root_x,root_y,px,py;
  unsigned int keys_button;
  Cursor       cursor = 0;

  PetscFunctionBegin;
  /* change cursor to indicate input */
  if (!cursor) {
    cursor = XCreateFontCursor(win->disp,XC_hand2);
    if (!cursor) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Unable to create X cursor");
  }
  XDefineCursor(win->disp,win->win,cursor);
  XSelectInput(win->disp,win->win,ButtonPressMask | ButtonReleaseMask);

  while (XCheckTypedEvent(win->disp,ButtonPress,&report));
  XMaskEvent(win->disp,ButtonReleaseMask,&report);
  switch (report.xbutton.button) {
    case Button1:
      if (report.xbutton.state & ShiftMask)
        *button = PETSC_BUTTON_LEFT_SHIFT;
      else
        *button = PETSC_BUTTON_LEFT;
      break;
    case Button2:
      if (report.xbutton.state & ShiftMask)
        *button = PETSC_BUTTON_CENTER_SHIFT;
      else
        *button = PETSC_BUTTON_CENTER;
      break;
    case Button3:
      if (report.xbutton.state & ShiftMask)
        *button = PETSC_BUTTON_RIGHT_SHIFT;
      else
        *button = PETSC_BUTTON_RIGHT;
      break;
  }
  XQueryPointer(win->disp,report.xmotion.window,&root,&child,&root_x,&root_y,&px,&py,&keys_button);

  if (x_phys) *x_phys = ((double)px)/((double)win->w);
  if (y_phys) *y_phys = 1.0 - ((double)py)/((double)win->h);

  if (x_user) *x_user = draw->coor_xl + ((((double)px)/((double)win->w)-draw->port_xl))*(draw->coor_xr - draw->coor_xl)/(draw->port_xr - draw->port_xl);
  if (y_user) *y_user = draw->coor_yl + ((1.0 - ((double)py)/((double)win->h)-draw->port_yl))*(draw->coor_yr - draw->coor_yl)/(draw->port_yr - draw->port_yl);

  XUndefineCursor(win->disp,win->win);
  XFlush(win->disp); 
  XSync(win->disp,False);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawPause_X"
static PetscErrorCode PetscDrawPause_X(PetscDraw draw)
{
  PetscErrorCode ierr;
  PetscDraw_X*   win = (PetscDraw_X*)draw->data;

  PetscFunctionBegin;
  if (!win->win) PetscFunctionReturn(0);
  if (draw->pause > 0) PetscSleep(draw->pause);
  else if (draw->pause == -1) {
    PetscDrawButton button;
    PetscMPIInt     rank;
    ierr = MPI_Comm_rank(((PetscObject)draw)->comm,&rank);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscDrawGetMouseButton(draw,&button,0,0,0,0);CHKERRQ(ierr);
      if (button == PETSC_BUTTON_CENTER) draw->pause = 0;
    }
    ierr = MPI_Bcast(&draw->pause,1,MPI_INT,0,((PetscObject)draw)->comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawGetPopup_X"
static PetscErrorCode PetscDrawGetPopup_X(PetscDraw draw,PetscDraw *popup)
{
  PetscErrorCode ierr;
  PetscDraw_X*   win = (PetscDraw_X*)draw->data;

  PetscFunctionBegin;
  ierr = PetscDrawOpenX(((PetscObject)draw)->comm,PETSC_NULL,PETSC_NULL,win->x,win->y+win->h+36,220,220,popup);CHKERRQ(ierr);
  draw->popup = *popup;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawSetTitle_X"
static PetscErrorCode PetscDrawSetTitle_X(PetscDraw draw,const char title[])
{
  PetscDraw_X    *win = (PetscDraw_X*)draw->data;
  XTextProperty  prop;
  PetscErrorCode ierr;
  size_t         len;

  PetscFunctionBegin;
  if (win->win) {
    XGetWMName(win->disp,win->win,&prop);
    XFree((void*)prop.value);
    prop.value  = (unsigned char *)title;
    ierr        = PetscStrlen(title,&len);CHKERRQ(ierr);
    prop.nitems = (long) len;
    XSetWMName(win->disp,win->win,&prop);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawResizeWindow_X"
static PetscErrorCode PetscDrawResizeWindow_X(PetscDraw draw,int w,int h)
{
  PetscDraw_X    *win = (PetscDraw_X*)draw->data;
  unsigned int   ww,hh,border,depth;
  int            x,y;
  PetscErrorCode ierr;
  Window         root;

  PetscFunctionBegin;
  if (win->win) {
    XResizeWindow(win->disp,win->win,w,h);
    XGetGeometry(win->disp,win->win,&root,&x,&y,&ww,&hh,&border,&depth);
    ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawCheckResizedWindow_X"
static PetscErrorCode PetscDrawCheckResizedWindow_X(PetscDraw draw)
{
  PetscDraw_X    *win = (PetscDraw_X*)draw->data;
  PetscErrorCode ierr;
  int            x,y;
  PetscMPIInt    rank;
  Window         root;
  unsigned int   w,h,border,depth,geo[2];
  PetscReal      xl,xr,yl,yr;
  XRectangle     box;

  PetscFunctionBegin;
  if (!win->win) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(((PetscObject)draw)->comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    XFlush(win->disp);
    XSync(win->disp,False);
    XSync(win->disp,False);
    XGetGeometry(win->disp,win->win,&root,&x,&y,geo,geo+1,&border,&depth);
  }
  ierr = MPI_Bcast(geo,2,MPI_INT,0,((PetscObject)draw)->comm);CHKERRQ(ierr);
  w = geo[0];
  h = geo[1];
  if (w == (unsigned int) win->w && h == (unsigned int) win->h) PetscFunctionReturn(0);

  /* record new window sizes */

  win->h = h; win->w = w;

  /* Free buffer space and create new version (only first processor does this) */
  if (win->drw) {
    win->drw = XCreatePixmap(win->disp,win->win,win->w,win->h,win->depth);
  }
  /* reset the clipping */
  xl = draw->port_xl; yl = draw->port_yl;
  xr = draw->port_xr; yr = draw->port_yr;
  box.x     = (int)(xl*win->w);     box.y      = (int)((1.0-yr)*win->h);
  box.width = (int)((xr-xl)*win->w);box.height = (int)((yr-yl)*win->h);
  XSetClipRectangles(win->disp,win->gc.set,0,0,&box,1,Unsorted);

  /* try to make sure it is actually done before passing info to all */
  XSync(win->disp,False);
  ierr = MPI_Bcast(&win->drw,1,MPI_UNSIGNED_LONG,0,((PetscObject)draw)->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawGetSingleton_X(PetscDraw,PetscDraw*);
static PetscErrorCode PetscDrawRestoreSingleton_X(PetscDraw,PetscDraw*);

#undef __FUNCT__
#define __FUNCT__ "PetscDrawDestroy_X"
PetscErrorCode PetscDrawDestroy_X(PetscDraw draw)
{
  PetscDraw_X    *win = (PetscDraw_X*)draw->data;
  PetscErrorCode ierr;
#if defined(PETSC_HAVE_POPEN)
  char           command[PETSC_MAX_PATH_LEN];
  PetscMPIInt    rank;
  FILE           *fd;
#endif

  PetscFunctionBegin;
  ierr = PetscDrawSynchronizedClear(draw);CHKERRQ(ierr);

#if defined(PETSC_HAVE_POPEN)
  ierr = MPI_Comm_rank(((PetscObject)draw)->comm,&rank);CHKERRQ(ierr);
  if (draw->savefilename && !rank && draw->savefilemovie) {
    ierr = PetscSNPrintf(command,PETSC_MAX_PATH_LEN,"ffmpeg  -i %s_%%d.Gif %s.m4v",draw->savefilename,draw->savefilename);CHKERRQ(ierr);
    ierr = PetscPOpen(((PetscObject)draw)->comm,PETSC_NULL,command,"r",&fd);CHKERRQ(ierr);
    ierr = PetscPClose(((PetscObject)draw)->comm,fd,PETSC_NULL);CHKERRQ(ierr);
  }
#endif

  XFreeGC(win->disp,win->gc.set);
  XCloseDisplay(win->disp);
  ierr = PetscDrawDestroy(&draw->popup);CHKERRQ(ierr);
  ierr = PetscFree(win->font);CHKERRQ(ierr);
  ierr = PetscFree(win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDrawSave_X(PetscDraw);
PetscErrorCode PetscDrawSetSave_X(PetscDraw,const char*);

static struct _PetscDrawOps DvOps = { PetscDrawSetDoubleBuffer_X,
                                 PetscDrawFlush_X,
                                 PetscDrawLine_X,
                                 0,
                                 0,
                                 PetscDrawPoint_X,
                                 0,
                                 PetscDrawString_X,
                                 PetscDrawStringVertical_X,
                                 PetscDrawStringSetSize_X,
                                 PetscDrawStringGetSize_X,
                                 PetscDrawSetViewport_X,
                                 PetscDrawClear_X,
                                 PetscDrawSynchronizedFlush_X,
                                 PetscDrawRectangle_X,
                                 PetscDrawTriangle_X,
                                 PetscDrawEllipse_X,
                                 PetscDrawGetMouseButton_X,
                                 PetscDrawPause_X,
                                 PetscDrawSynchronizedClear_X,
				 0,
                                 0,
                                 PetscDrawGetPopup_X,
                                 PetscDrawSetTitle_X,
                                 PetscDrawCheckResizedWindow_X,
                                 PetscDrawResizeWindow_X,
                                 PetscDrawDestroy_X,
                                 0,
                                 PetscDrawGetSingleton_X,
                                 PetscDrawRestoreSingleton_X,
#if defined(PETSC_HAVE_AFTERIMAGE)
                                 PetscDrawSave_X,
#else
                                 0,
#endif
                                 PetscDrawSetSave_X,
                                 0,
                                 PetscDrawArrow_X,
                                 PetscDrawCoordinateToPixel_X,
                                 PetscDrawPixelToCoordinate_X,
                                 PetscDrawPointPixel_X};


extern PetscErrorCode PetscDrawXiQuickWindow(PetscDraw_X*,char*,char*,int,int,int,int);
extern PetscErrorCode PetscDrawXiQuickWindowFromWindow(PetscDraw_X*,char*,Window);

#undef __FUNCT__
#define __FUNCT__ "PetscDrawGetSingleton_X"
static PetscErrorCode PetscDrawGetSingleton_X(PetscDraw draw,PetscDraw *sdraw)
{
  PetscErrorCode ierr;
  PetscDraw_X *Xwin = (PetscDraw_X*)draw->data,*sXwin;

  PetscFunctionBegin;

  ierr = PetscDrawCreate(PETSC_COMM_SELF,draw->display,draw->title,draw->x,draw->y,draw->w,draw->h,sdraw);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)*sdraw,PETSC_DRAW_X);CHKERRQ(ierr);
  ierr = PetscMemcpy((*sdraw)->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);
  (*sdraw)->ops->destroy = 0;

  (*sdraw)->pause   = draw->pause;
  (*sdraw)->coor_xl = draw->coor_xl;
  (*sdraw)->coor_xr = draw->coor_xr;
  (*sdraw)->coor_yl = draw->coor_yl;
  (*sdraw)->coor_yr = draw->coor_yr;
  (*sdraw)->port_xl = draw->port_xl;
  (*sdraw)->port_xr = draw->port_xr;
  (*sdraw)->port_yl = draw->port_yl;
  (*sdraw)->port_yr = draw->port_yr;
  (*sdraw)->popup   = draw->popup;

  /* actually create and open the window */
  ierr = PetscNew(PetscDraw_X,&sXwin);CHKERRQ(ierr);
  ierr = PetscDrawXiQuickWindowFromWindow(sXwin,draw->display,Xwin->win);CHKERRQ(ierr);

  sXwin->x       = Xwin->x;
  sXwin->y       = Xwin->y;
  sXwin->w       = Xwin->w;
  sXwin->h       = Xwin->h;
  (*sdraw)->data = (void*)sXwin;
 PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawRestoreSingleton_X"
static PetscErrorCode PetscDrawRestoreSingleton_X(PetscDraw draw,PetscDraw *sdraw)
{
  PetscErrorCode ierr;
  PetscDraw_X    *sXwin = (PetscDraw_X*)(*sdraw)->data;

  PetscFunctionBegin;
  XFreeGC(sXwin->disp,sXwin->gc.set);
  XCloseDisplay(sXwin->disp);
  ierr = PetscDrawDestroy(&(*sdraw)->popup);CHKERRQ(ierr);
  ierr = PetscFree((*sdraw)->title);CHKERRQ(ierr);
  ierr = PetscFree((*sdraw)->display);CHKERRQ(ierr);
  ierr = PetscFree(sXwin->font);CHKERRQ(ierr);
  ierr = PetscFree(sXwin);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(sdraw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawXGetDisplaySize_Private"
PetscErrorCode PetscDrawXGetDisplaySize_Private(const char name[],int *width,int *height)
{
  Display *display;

  PetscFunctionBegin;
  display = XOpenDisplay(name);
  if (!display) {
    *width  = 0;
    *height = 0;
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Unable to open display on %s\n.  Make sure your COMPUTE NODES are authorized to connect \n\
    to this X server and either your DISPLAY variable\n\
    is set or you use the -display name option\n",name);
  }
  *width  = DisplayWidth(display,0);
  *height = DisplayHeight(display,0);
  XCloseDisplay(display);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscDrawCreate_X"
PetscErrorCode  PetscDrawCreate_X(PetscDraw draw)
{
  PetscDraw_X    *Xwin;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       xywh[4],osize = 4;
  int            x = draw->x,y = draw->y,w = draw->w,h = draw->h;
  static int     xavailable = 0,yavailable = 0,xmax = 0,ymax = 0,ybottom = 0;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  if (!draw->display) {
    ierr = PetscMalloc(256*sizeof(char),&draw->display);CHKERRQ(ierr);
    ierr = PetscGetDisplay(draw->display,256);CHKERRQ(ierr);
  }

  /*
      Initialize the display size
  */
  if (!xmax) {
    ierr = PetscDrawXGetDisplaySize_Private(draw->display,&xmax,&ymax);
    /* if some processors fail on this and others succed then this is a problem ! */
    if (ierr) {
      (*PetscErrorPrintf)("PETSc unable to use X windows\nproceeding without graphics\n");
      ierr = PetscDrawSetType(draw,PETSC_DRAW_NULL);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }

  if (w == PETSC_DECIDE) w = draw->w = 300;
  if (h == PETSC_DECIDE) h = draw->h = 300;
  switch (w) {
    case PETSC_DRAW_FULL_SIZE: w = draw->w = xmax - 10;
                         break;
    case PETSC_DRAW_HALF_SIZE: w = draw->w = (xmax - 20)/2;
                         break;
    case PETSC_DRAW_THIRD_SIZE: w = draw->w = (xmax - 30)/3;
                         break;
    case PETSC_DRAW_QUARTER_SIZE: w = draw->w = (xmax - 40)/4;
                         break;
  }
  switch (h) {
    case PETSC_DRAW_FULL_SIZE: h = draw->h = ymax - 10;
                         break;
    case PETSC_DRAW_HALF_SIZE: h = draw->h = (ymax - 20)/2;
                         break;
    case PETSC_DRAW_THIRD_SIZE: h = draw->h = (ymax - 30)/3;
                         break;
    case PETSC_DRAW_QUARTER_SIZE: h = draw->h = (ymax - 40)/4;
                         break;
  }

  /* allow user to set location and size of window */
  xywh[0] = x; xywh[1] = y; xywh[2] = w; xywh[3] = h;
  ierr = PetscOptionsGetIntArray(PETSC_NULL,"-geometry",xywh,&osize,PETSC_NULL);CHKERRQ(ierr);
  x = (int) xywh[0]; y = (int) xywh[1]; w = (int) xywh[2]; h = (int) xywh[3];


  if (draw->x == PETSC_DECIDE || draw->y == PETSC_DECIDE) {
    /*
       PETSc tries to place windows starting in the upper left corner and
       moving across to the right.

              --------------------------------------------
              |  Region used so far +xavailable,yavailable |
              |                     +                      |
              |                     +                      |
              |++++++++++++++++++++++ybottom               |
              |                                            |
              |                                            |
              |--------------------------------------------|
    */
    /*  First: can we add it to the right? */
    if (xavailable+w+10 <= xmax) {
      x       = xavailable;
      y       = yavailable;
      ybottom = PetscMax(ybottom,y + h + 30);
    } else {
      /* No, so add it below on the left */
      xavailable = 0;
      x          = 0;
      yavailable = ybottom;
      y          = ybottom;
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

  ierr = PetscMemcpy(draw->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);

  /* actually create and open the window */
  ierr = PetscNew(PetscDraw_X,&Xwin);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(draw,sizeof(PetscDraw_X));CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)draw)->comm,&rank);CHKERRQ(ierr);

  if (!rank) {
    if (x < 0 || y < 0)   SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative corner of window");
    if (w <= 0 || h <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative window width or height");
    ierr = PetscDrawXiQuickWindow(Xwin,draw->display,draw->title,x,y,w,h);CHKERRQ(ierr);
    ierr = MPI_Bcast(&Xwin->win,1,MPI_UNSIGNED_LONG,0,((PetscObject)draw)->comm);CHKERRQ(ierr);
  } else {
    unsigned long win = 0;
    ierr = MPI_Bcast(&win,1,MPI_UNSIGNED_LONG,0,((PetscObject)draw)->comm);CHKERRQ(ierr);
    ierr = PetscDrawXiQuickWindowFromWindow(Xwin,draw->display,win);CHKERRQ(ierr);
  }

  Xwin->x      = x;
  Xwin->y      = y;
  Xwin->w      = w;
  Xwin->h      = h;
  draw->data    = (void*)Xwin;

  /*
    Need barrier here so processor 0 does not destroy the window before other
    processors have completed PetscDrawXiQuickWindow()
  */
  ierr = PetscDrawSynchronizedFlush(draw);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(PETSC_NULL,"-draw_double_buffer",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
     ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
EXTERN_C_END









