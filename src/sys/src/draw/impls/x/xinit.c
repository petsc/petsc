#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: xinit.c,v 1.50 1999/01/31 16:05:02 bsmith Exp bsmith $";
#endif

/* 
   This file contains routines to open an X window display and window
   This consists of a number of routines that set the various
   fields in the Window structure, which is passed to 
   all of these routines.

   Note that if you use the default visual and colormap, then you
   can use these routines with any X toolkit that will give you the
   Window id of the window that it is managing.  Use that instead of the
   call to XiCreateWindow .  Similarly for the Display.
*/

#include "petsc.h"

#if defined(HAVE_X11)
#include "src/sys/src/draw/impls/x/ximpl.h"

extern int XiUniformHues(Draw_X *,int);
extern int Xi_wait_map( Draw_X*);
extern int XiInitColors(Draw_X*,Colormap);
extern int XiFontFixed(Draw_X*,int,int,XiFont** );

/*
  XiOpenDisplay - Open a display
*/
#undef __FUNC__  
#define __FUNC__ "XiOpenDisplay" 
int XiOpenDisplay(Draw_X* XiWin,char *display_name )
{
  PetscFunctionBegin;
  XiWin->disp = XOpenDisplay( display_name );
  if (!XiWin->disp) {
    SETERRQ1(1,1,"Unable to open display on %s\n.  Make sure your DISPLAY variable\n\
    is set, or you use the -display name option and xhost + has been\n\
    run on your displaying machine.\n",display_name);
  }
  XiWin->screen = DefaultScreen( XiWin->disp );
  PetscFunctionReturn(0);
}

/*  
    XiSetVisual - set the visual class for a window and colormap
*/
#undef __FUNC__  
#define __FUNC__ "XiSetVisual" 
int XiSetVisual(Draw_X* XiWin,int usedefaultcolormap,Colormap cmap)
{
  int ierr;

  PetscFunctionBegin;

  /*
       As a test due to problems on SGI reported in petsc-maint 1534 change
    to always use default visual
  */
  XiWin->vis    = DefaultVisual( XiWin->disp, XiWin->screen );
  XiWin->depth  = DefaultDepth(XiWin->disp,XiWin->screen);
  if (cmap)                    XiWin->cmap  = cmap;
  else if (usedefaultcolormap) XiWin->cmap  = DefaultColormap( XiWin->disp, XiWin->screen );
  else                         XiWin->cmap = 0;

  if (!XiWin->cmap) {
    XiWin->cmap = XCreateColormap(XiWin->disp,RootWindow(XiWin->disp,XiWin->screen),
                                  XiWin->vis,AllocAll);CHKPTRQ(XiWin->cmap);
  }

  if (XiWin->depth < 8) {
    SETERRQ(1,1,"PETSc Graphics require monitors with at least 8 bit color (256 colors)");
  }
  PLogInfo(0,"XiSetVisual:Always opening default visual X window\n");

  /* reset the number of colors from info on the display, the colormap */
  ierr = XiInitColors( XiWin, cmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
   XiSetGC - set the GC structure in the base window
*/
#undef __FUNC__  
#define __FUNC__ "XiSetGC" 
int XiSetGC(Draw_X* XiWin,PixVal fg )
{
  XGCValues       gcvalues;       /* window graphics context values */

  PetscFunctionBegin;
  /* Set the graphics contexts */
  /* create a gc for the ROP_SET operation (writing the fg value to a pixel) */
  /* (do this with function GXcopy; GXset will automatically write 1) */
  gcvalues.function   = GXcopy;
  gcvalues.foreground = fg;
  XiWin->gc.cur_pix   = fg;
  XiWin->gc.set = XCreateGC(XiWin->disp, RootWindow(XiWin->disp,XiWin->screen),
                              GCFunction | GCForeground, &gcvalues );
  PetscFunctionReturn(0);
}

/*
    Actually display a window at [x,y] with sizes (w,h)
    If w and/or h are 0, use the sizes in the fields of XiWin
    (which may have been set by, for example, XiSetWindowSize)
*/
int XiDisplayWindow( Draw_X* XiWin, char *label, int x, int y,
                     int w,int h,PixVal backgnd_pixel )
{
  unsigned int            wavail, havail;
  XSizeHints              size_hints;
  XWindowAttributes       in_window_attributes;
  XSetWindowAttributes    window_attributes;
  int                     depth, border_width;
  unsigned long           wmask;

  PetscFunctionBegin;
  /* get the available widths */
  wavail              = DisplayWidth(  XiWin->disp, XiWin->screen );
  havail              = DisplayHeight( XiWin->disp, XiWin->screen );
  if (w <= 0 || h <= 0) PetscFunctionReturn(2);
  if (w > wavail) w    = wavail;
  if (h > havail)  h   = havail;

  /* changed the next line from xtools version */
  border_width   = 0;
  if (x < 0) x   = 0;
  if (y < 0) y   = 0;
  x   = (x + w > wavail) ? wavail - w : x;
  y   = (y + h > havail) ? havail - h : y;

  /* We need XCreateWindow since we may need an visual other than
   the default one */
  XGetWindowAttributes( XiWin->disp, RootWindow(XiWin->disp,XiWin->screen),
                        &in_window_attributes );
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
  window_attributes.backing_store     = 0;
  window_attributes.backing_pixel     = backgnd_pixel;
  window_attributes.save_under        = 1;
  window_attributes.event_mask        = 0;
  window_attributes.do_not_propagate_mask = 0;
  window_attributes.override_redirect = 0;
  window_attributes.colormap          = XiWin->cmap;
  /* None for cursor does NOT mean none, it means Parent's cursor */
  window_attributes.cursor            = None; 
  wmask   = CWBackPixmap | CWBackPixel | CWBorderPixmap | CWBitGravity |
            CWWinGravity | CWBackingStore |CWBackingPixel|CWOverrideRedirect |
            CWSaveUnder  | CWEventMask    | CWDontPropagate |
            CWCursor     | CWColormap ;
  depth       = XiWin->depth;
  /* DefaultDepth( XiWin->disp, XiWin->screen ); */
  XiWin->win  = XCreateWindow( XiWin->disp, 
			     RootWindow(XiWin->disp,XiWin->screen),
                             x, y, w, h, border_width,
                             depth, InputOutput, XiWin->vis,
                             wmask, &window_attributes );

  if (!XiWin->win)  PetscFunctionReturn(2);

  /* set window manager hints */
  {
    XWMHints      wm_hints;
    XClassHint    class_hints;
    XTextProperty windowname,iconname;
    
    if (label) { XStringListToTextProperty(&label,1,&windowname);}
    else       { XStringListToTextProperty(&label,0,&windowname);}
    if (label) { XStringListToTextProperty(&label,1,&iconname);}
    else       { XStringListToTextProperty(&label,0,&iconname);}
    
    wm_hints.initial_state  = NormalState;
    wm_hints.input          = True;
    wm_hints.flags          = StateHint|InputHint;
 
    class_hints.res_name    = 0;
    class_hints.res_class   = "BaseClass"; /* this is nonsense */

    size_hints.x            = x;
    size_hints.y            = y;
    size_hints.min_width    = 4*border_width;
    size_hints.min_height   = 4*border_width;
    size_hints.width        = w;
    size_hints.height       = h;
    size_hints.flags        = USPosition | USSize | PMinSize;
 
    XSetWMProperties(XiWin->disp,XiWin->win,&windowname,&iconname,
                     0,0,&size_hints,&wm_hints,&class_hints);
  }
  /* make the window visible */
  XSelectInput( XiWin->disp, XiWin->win, ExposureMask | StructureNotifyMask );
  XMapWindow( XiWin->disp, XiWin->win );

  /* some window systems are cruel and interfere with the placement of
     windows.  We wait here for the window to be created or to die */
  if (Xi_wait_map( XiWin)){
    XiWin->win    = (Window)0;
    PetscFunctionReturn(1);
  }
  /* Initial values for the upper left corner */
  XiWin->x = 0;
  XiWin->y = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "XiQuickWindow" 
int XiQuickWindow(Draw_X* w,char* host,char* name,int x,int y,int nx,int ny)
{
  int         ierr,flag = 0;
  XVisualInfo vinfo;

  PetscFunctionBegin;
  ierr = XiOpenDisplay( w, host ); CHKERRQ(ierr);

  /*
      The reason these are slow is they call XAllocColor() for each 256 colors
    each one, I think, requires communication with the X server.
  */

  /* this is slow */
  ierr = OptionsHasName(PETSC_NULL,"-draw_x_shared_colormap",&flag); CHKERRQ(ierr);
  /*
        Need to determine if window supports allocating a private colormap,
    if not, set flag to 1
  */
  if (XMatchVisualInfo( w->disp, w->screen, 24, StaticColor, &vinfo) ||
      XMatchVisualInfo( w->disp, w->screen, 24, TrueColor, &vinfo) ||
      XMatchVisualInfo( w->disp, w->screen, 16, StaticColor, &vinfo) ||
      XMatchVisualInfo( w->disp, w->screen, 16, TrueColor, &vinfo)) {
    flag = 1;
  }

  ierr = XiSetVisual( w, flag, (Colormap)0); CHKERRQ(ierr);

  ierr = XiDisplayWindow( w, name, x, y, nx, ny, (PixVal)0 ); CHKERRQ(ierr);
  XiSetGC( w, w->cmapping[1] );
  XiSetPixVal(w, w->background );

  /* this is very slow */
  ierr = XiUniformHues(w,256-DRAW_BASIC_COLORS); CHKERRQ(ierr); 

  ierr = XiFontFixed( w,6, 10,&w->font ); CHKERRQ(ierr);
  XFillRectangle(w->disp,w->win,w->gc.set,0,0,nx,ny);
  PetscFunctionReturn(0);
}

/* 
   A version from an already defined window 
*/
#undef __FUNC__  
#define __FUNC__ "XiQuickWindowFromWindow" 
int XiQuickWindowFromWindow(Draw_X* w,char *host,Window win)
{
  Window            root;
  int               d,ierr;
  unsigned int      ud;
  XWindowAttributes attributes;

  PetscFunctionBegin;
  if (XiOpenDisplay( w, host )) {
    SETERRQ(PETSC_ERR_LIB,0,"Could not open display: make sure your DISPLAY variable\n\
    is set, or you use the [-display name] option and xhost + has been\n\
    run on your displaying machine.\n" );
  }

  w->win = win;
  XGetWindowAttributes(w->disp, w->win, &attributes);

  ierr = XiSetVisual( w, 1, attributes.colormap); CHKERRQ(ierr);

  XGetGeometry( w->disp, w->win, &root, &d, &d, 
	      (unsigned int *)&w->w, (unsigned int *)&w->h,&ud, &ud );
  w->x = w->y = 0;



  XiSetGC( w, w->cmapping[1] );
  XiSetPixVal(w, w->background );
  ierr = XiUniformHues(w,256-DRAW_BASIC_COLORS); CHKERRQ(ierr);
  ierr = XiFontFixed( w,6, 10,&w->font ); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
      XiSetWindowLabel - Sets new label in open window.
*/
#undef __FUNC__  
#define __FUNC__ "XiSetWindowLabel" 
int XiSetWindowLabel(Draw_X* Xiwin, char *label )
{
  XTextProperty prop;

  PetscFunctionBegin;
  XGetWMName(Xiwin->disp,Xiwin->win,&prop);
  prop.value = (unsigned char *)label; prop.nitems = (long) PetscStrlen(label);
  XSetWMName(Xiwin->disp,Xiwin->win,&prop);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "XiSetToBackground" 
int XiSetToBackground(Draw_X* XiWin )
{
  PetscFunctionBegin;
  if (XiWin->gc.cur_pix != XiWin->background) { 
    XSetForeground( XiWin->disp, XiWin->gc.set, XiWin->background ); 
    XiWin->gc.cur_pix   = XiWin->background;
  }
  PetscFunctionReturn(0);
}

#endif







