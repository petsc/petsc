
/*
   This file contains routines to open an X window display and window
   This consists of a number of routines that set the various
   fields in the Window structure, which is passed to
   all of these routines.

   Note that if you use the default visual and colormap, then you
   can use these routines with any X toolkit that will give you the
   Window id of the window that it is managing.  Use that instead of the
   call to PetscDrawXiCreateWindow .  Similarly for the Display.
*/

#include <../src/sys/classes/draw/impls/x/ximpl.h>

PETSC_INTERN PetscErrorCode PetscDrawSetColormap_X(PetscDraw_X*,Colormap);

/*
  PetscDrawXiOpenDisplay - Open and setup a display
*/
static PetscErrorCode PetscDrawXiOpenDisplay(PetscDraw_X *XiWin,const char display[])
{
  PetscFunctionBegin;
  XiWin->disp = XOpenDisplay(display);
  if (!XiWin->disp) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Unable to open display on %s\n\
    Make sure your COMPUTE NODES are authorized to connect \n\
    to this X server and either your DISPLAY variable\n\
    is set or you use the -display name option\n",display);
  }
  XiWin->screen     = DefaultScreen(XiWin->disp);
  XiWin->vis        = DefaultVisual(XiWin->disp,XiWin->screen);
  XiWin->depth      = DefaultDepth(XiWin->disp,XiWin->screen);
  XiWin->cmap       = DefaultColormap(XiWin->disp,XiWin->screen);
  XiWin->background = WhitePixel(XiWin->disp,XiWin->screen);
  XiWin->foreground = BlackPixel(XiWin->disp,XiWin->screen);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDrawXiClose(PetscDraw_X *XiWin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!XiWin) PetscFunctionReturn(0);
  ierr = PetscFree(XiWin->font);CHKERRQ(ierr);
  if (XiWin->disp) {
#if defined(PETSC_HAVE_SETJMP_H)
    jmp_buf              jmpbuf;
    PetscXIOErrorHandler xioerrhdl;
    ierr = PetscMemcpy(&jmpbuf,&PetscXIOErrorHandlerJumpBuf,sizeof(jmpbuf));CHKERRQ(ierr);
    xioerrhdl = PetscSetXIOErrorHandler(PetscXIOErrorHandlerJump);
    if (!setjmp(PetscXIOErrorHandlerJumpBuf))
#endif
    {
      XFreeGC(XiWin->disp,XiWin->gc.set);
      XCloseDisplay(XiWin->disp);
    }
    XiWin->disp = NULL;
#if defined(PETSC_HAVE_SETJMP_H)
    (void)PetscSetXIOErrorHandler(xioerrhdl);
    ierr = PetscMemcpy(&PetscXIOErrorHandlerJumpBuf,&jmpbuf,sizeof(jmpbuf));CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

/*
   PetscDrawXiCreateGC - setup the GC structure
*/
static PetscErrorCode PetscDrawXiCreateGC(PetscDraw_X *XiWin,PetscDrawXiPixVal fg)
{
  XGCValues gcvalues;             /* window graphics context values */

  PetscFunctionBegin;
  /* Set the graphics contexts */
  /* create a gc for the ROP_SET operation (writing the fg value to a pixel) */
  /* (do this with function GXcopy; GXset will automatically write 1) */
  gcvalues.function   = GXcopy;
  gcvalues.foreground = fg;
  XiWin->gc.cur_pix   = fg;
  XiWin->gc.set       = XCreateGC(XiWin->disp,RootWindow(XiWin->disp,XiWin->screen),GCFunction|GCForeground,&gcvalues);
  if (!XiWin->gc.set) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Unable to create X graphics context");
  PetscFunctionReturn(0);
}

/*
   PetscDrawXiInit - basic setup the draw (display, graphics context, font)
*/
PetscErrorCode PetscDrawXiInit(PetscDraw_X *XiWin,const char display[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscDrawXiOpenDisplay(XiWin,display);CHKERRQ(ierr);
  ierr = PetscDrawXiCreateGC(XiWin,XiWin->foreground);CHKERRQ(ierr);
  ierr = PetscDrawXiFontFixed(XiWin,6,10,&XiWin->font);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    This routine waits until the window is actually created or destroyed
    Returns 0 if window is mapped; 1 if window is destroyed.
 */
static PetscErrorCode PetscDrawXiWaitMap(PetscDraw_X *XiWin)
{
  XEvent event;

  PetscFunctionBegin;
  while (1) {
    XMaskEvent(XiWin->disp,ExposureMask|StructureNotifyMask,&event);
    if (event.xany.window != XiWin->win) break;
    else {
      switch (event.type) {
      case ConfigureNotify:
        /* window has been moved or resized */
        XiWin->w = event.xconfigure.width  - 2 * event.xconfigure.border_width;
        XiWin->h = event.xconfigure.height - 2 * event.xconfigure.border_width;
        break;
      case DestroyNotify:
        PetscFunctionReturn(1);
      case Expose:
        PetscFunctionReturn(0);
        /* else ignore event */
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
    Actually display a window at [x,y] with sizes (w,h)
*/
static PetscErrorCode PetscDrawXiDisplayWindow(PetscDraw_X *XiWin,char *label,int x,int y,int w,int h)
{
  unsigned int         wavail,havail;
  XSizeHints           size_hints;
  XWindowAttributes    in_window_attributes;
  XSetWindowAttributes window_attributes;
  unsigned int         border_width = 0;
  unsigned long        backgnd_pixel = WhitePixel(XiWin->disp,XiWin->screen);
  unsigned long        wmask;

  PetscFunctionBegin;
  /* get the available widths */
  wavail = DisplayWidth(XiWin->disp,XiWin->screen);
  havail = DisplayHeight(XiWin->disp,XiWin->screen);
  if (w <= 0 || h <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"X Window display has invalid height or width");
  if ((unsigned int)w > wavail) w = wavail;
  if ((unsigned int)h > havail) h = havail;

  if (x < 0) x = (int)(wavail - (unsigned int)w + (unsigned int)x);
  if (y < 0) y = (int)(havail - (unsigned int)h + (unsigned int)y);
  x = ((unsigned int)x + w > wavail) ? (int)(wavail - (unsigned int)w) : x;
  y = ((unsigned int)y + h > havail) ? (int)(havail - (unsigned int)h) : y;

  /* We need XCreateWindow since we may need an visual other than the default one */
  XGetWindowAttributes(XiWin->disp,RootWindow(XiWin->disp,XiWin->screen),&in_window_attributes);
  window_attributes.background_pixmap = None;
  window_attributes.background_pixel  = backgnd_pixel;
  /* No border for now */
  window_attributes.border_pixmap     = None;
  /*
  window_attributes.border_pixel      = border_pixel;
  */
  window_attributes.bit_gravity       = in_window_attributes.bit_gravity;
  window_attributes.win_gravity       = in_window_attributes.win_gravity;
  /* Backing store is too slow in color systems */
  window_attributes.backing_store     = NotUseful;
  window_attributes.backing_pixel     = backgnd_pixel;
  window_attributes.save_under        = 1;
  window_attributes.event_mask        = 0;
  window_attributes.do_not_propagate_mask = 0;
  window_attributes.override_redirect = 0;
  window_attributes.colormap          = XiWin->cmap;
  /* None for cursor does NOT mean none, it means cursor of Parent */
  window_attributes.cursor            = None;

  wmask = CWBackPixmap | CWBackPixel    | CWBorderPixmap  | CWBitGravity |
          CWWinGravity | CWBackingStore | CWBackingPixel  | CWOverrideRedirect |
          CWSaveUnder  | CWEventMask    | CWDontPropagate |
          CWCursor     | CWColormap;

  XiWin->win = XCreateWindow(XiWin->disp,RootWindow(XiWin->disp,XiWin->screen),x,y,w,h,border_width,XiWin->depth,InputOutput,XiWin->vis,wmask,&window_attributes);
  if (!XiWin->win) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Unable to open X window");

  /* set window manager hints */
  {
    XWMHints      wm_hints;
    XClassHint    class_hints;
    XTextProperty windowname,iconname;

    if (label) XStringListToTextProperty(&label,1,&windowname);
    else       XStringListToTextProperty(&label,0,&windowname);
    if (label) XStringListToTextProperty(&label,1,&iconname);
    else       XStringListToTextProperty(&label,0,&iconname);

    wm_hints.initial_state = NormalState;
    wm_hints.input         = True;
    wm_hints.flags         = StateHint|InputHint;

    /* These properties can be used by window managers to decide how to display a window */
    class_hints.res_name  = (char*)"petsc";
    class_hints.res_class = (char*)"PETSc";

    size_hints.x          = x;
    size_hints.y          = y;
    size_hints.min_width  = 4*border_width;
    size_hints.min_height = 4*border_width;
    size_hints.width      = w;
    size_hints.height     = h;
    size_hints.flags      = USPosition | USSize | PMinSize;

    XSetWMProperties(XiWin->disp,XiWin->win,&windowname,&iconname,NULL,0,&size_hints,&wm_hints,&class_hints);
    XFree((void*)windowname.value);
    XFree((void*)iconname.value);
  }

  /* make the window visible */
  XSelectInput(XiWin->disp,XiWin->win,ExposureMask|StructureNotifyMask);
  XMapWindow(XiWin->disp,XiWin->win);
  /* some window systems are cruel and interfere with the placement of
     windows.  We wait here for the window to be created or to die */
  if (PetscDrawXiWaitMap(XiWin)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Wait for X window failed");
  XSelectInput(XiWin->disp,XiWin->win,NoEventMask);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDrawXiQuickWindow(PetscDraw_X *XiWin,char *name,int x,int y,int nx,int ny)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDrawSetColormap_X(XiWin,(Colormap)0);CHKERRQ(ierr);
  ierr = PetscDrawXiDisplayWindow(XiWin,name,x,y,nx,ny);CHKERRQ(ierr);
  XSetWindowBackground(XiWin->disp,XiWin->win,XiWin->background);
  XClearWindow(XiWin->disp,XiWin->win);
  PetscFunctionReturn(0);
}

/*
   A version from an already defined window
*/
PetscErrorCode PetscDrawXiQuickWindowFromWindow(PetscDraw_X *XiWin,Window win)
{
  XWindowAttributes attributes;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  XiWin->win = win;
  XGetWindowAttributes(XiWin->disp,XiWin->win,&attributes);
  ierr = PetscDrawSetColormap_X(XiWin,attributes.colormap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDrawXiQuickPixmap(PetscDraw_X* XiWin)
{
  PetscFunctionBegin;
  if (XiWin->drw) XFreePixmap(XiWin->disp,XiWin->drw);
  XiWin->drw = XCreatePixmap(XiWin->disp,RootWindow(XiWin->disp,XiWin->screen),XiWin->w,XiWin->h,XiWin->depth);
  PetscDrawXiSetPixVal(XiWin,XiWin->background);
  XFillRectangle(XiWin->disp,XiWin->drw,XiWin->gc.set,0,0,XiWin->w,XiWin->h);
  XSync(XiWin->disp,False);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDrawXiResizeWindow(PetscDraw_X* XiWin,int w,int h)
{
  XEvent event;
  PetscFunctionBegin;
  XSelectInput(XiWin->disp,XiWin->win,StructureNotifyMask);
  XResizeWindow(XiWin->disp,XiWin->win,(unsigned int)w,(unsigned int)h);
  XWindowEvent(XiWin->disp,XiWin->win,StructureNotifyMask,&event);
  XSelectInput(XiWin->disp,XiWin->win,NoEventMask);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDrawXiGetGeometry(PetscDraw_X *XiWin,int *x,int *y,int *w,int *h)
{
  XWindowAttributes attributes;
  Window            root,parent,child;
  int               xx=0,yy=0;
  unsigned int      ww=0,hh=0,dummy;
  PetscFunctionBegin;
  if (XiWin->win) {
    XGetGeometry(XiWin->disp,XiWin->win,&parent,&xx,&yy,&ww,&hh,&dummy,&dummy);
    root = RootWindow(XiWin->disp,XiWin->screen);
    if (!XTranslateCoordinates(XiWin->disp,XiWin->win,root,0,0,&xx,&yy,&child)) {
      XGetWindowAttributes(XiWin->disp,XiWin->win,&attributes);
      root = attributes.screen->root;
      (void)XTranslateCoordinates(XiWin->disp,XiWin->win,root,0,0,&xx,&yy,&child);
    }
  } else if (XiWin->drw) {
    XGetGeometry(XiWin->disp,XiWin->drw,&root,&xx,&yy,&ww,&hh,&dummy,&dummy);
  }
  if (x) *x = xx;
  if (y) *y = yy;
  if (w) *w = (int)ww;
  if (h) *h = (int)hh;
  PetscFunctionReturn(0);
}
