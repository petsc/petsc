
#include <stdio.h>
#include "ximpl.h"

extern int  XiUniformHues(XiWindow *,int);
extern int  Xi_wait_map( XiWindow*);
extern int XiInitColors(XiWindow*,Colormap,int);
extern int XiFontFixed(XiWindow*,int,int,XiFont** );
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

/*
  XiOpenDisplay - Open a display
  
  Input Parameters:
  XiWin        - pointer to base window structure
  display_name - either null ("") or of the form "host:0"
*/
int XiOpenDisplay(XiWindow* XiWin,char *display_name )
{
  XiWin->disp = XOpenDisplay( display_name );
  if (!XiWin->disp)  return 1;
  XiWin->screen = DefaultScreen( XiWin->disp );
  return 0;
}

/*  
    XiSetVisual - set the visual class for a window and colormap

    Input Parameters:
.   nc - number of colors.  Use the maximum of the visual if
    nc == 0.  Use nc = 2 for black and white displays.
  */
int XiSetVisual(XiWindow* XiWin,int q_default_visual,Colormap cmap,int nc )
{
  if (q_default_visual) {
    XiWin->vis    = DefaultVisual( XiWin->disp, XiWin->screen );
    XiWin->depth  = DefaultDepth(XiWin->disp,XiWin->screen);
    if (!cmap) XiWin->cmap  = DefaultColormap( XiWin->disp, XiWin->screen );
    else       XiWin->cmap  = cmap;
  }
  else {
    /* Try to match to some popular types */
    XVisualInfo vinfo;
    if (XMatchVisualInfo(XiWin->disp,XiWin->screen,24,DirectColor,&vinfo)) {
	XiWin->vis    = vinfo.visual;
	XiWin->depth  = 24;
    }
    else if (XMatchVisualInfo(XiWin->disp,XiWin->screen,8,PseudoColor,&vinfo)){
	XiWin->vis    = vinfo.visual;
	XiWin->depth  = 8;
    }
    else if (XMatchVisualInfo( XiWin->disp, XiWin->screen,
			 DefaultDepth(XiWin->disp,XiWin->screen), PseudoColor,
			 &vinfo)) {
	XiWin->vis    = vinfo.visual;
	XiWin->depth  = DefaultDepth(XiWin->disp,XiWin->screen);
    }
    else {
	XiWin->vis    = DefaultVisual( XiWin->disp, XiWin->screen );
	XiWin->depth  = DefaultDepth(XiWin->disp,XiWin->screen);
    }
    /* There are other types; perhaps this routine should accept a 
       "hint" on which types to look for. */
    XiWin->cmap = (Colormap) 0;
  }

  /* reset the number of colors from info on the display, the colormap */
  XiInitColors( XiWin, cmap, nc );
  return 0;
}

/* 
   XiSetGC - set the GC structure in the base window
  */
int XiSetGC(XiWindow* XiWin,PixVal fg )
{
  XGCValues       gcvalues;       /* window graphics context values */
  /* Set the graphics contexts */
  /* create a gc for the ROP_SET operation (writing the fg value to a pixel) */
  /* (do this with function GXcopy; GXset will automatically write 1) */
  gcvalues.function   = GXcopy;
  gcvalues.foreground = fg;
  XiWin->gc.cur_pix   = fg;
  XiWin->gc.set = XCreateGC(XiWin->disp, RootWindow(XiWin->disp,XiWin->screen),
                              GCFunction | GCForeground, &gcvalues );
  return 0;
}

/*
    Actually display a window at [x,y] with sizes (w,h)
    If w and/or h are 0, use the sizes in the fields of XiWin
    (which may have been set by, for example, XiSetWindowSize)
 */
int XiDisplayWindow( XiWindow* XiWin, char *label, int x, int y,
                     int w,int h,PixVal backgnd_pixel )
{
  unsigned int            wavail, havail;
  XSizeHints              size_hints;
  int                     q_user_pos;
  XWindowAttributes       in_window_attributes;
  XSetWindowAttributes    window_attributes;
  int                     depth, border_width;
  unsigned long           wmask;

  /* get the available widths */
  wavail              = DisplayWidth(  XiWin->disp, XiWin->screen );
  havail              = DisplayHeight( XiWin->disp, XiWin->screen );
  if (w <= 0 || h <= 0) return 2;
  if (w > wavail) w   = wavail;
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
  /* depth should really be the depth of the visual in use */
  depth       = DefaultDepth( XiWin->disp, XiWin->screen );
  XiWin->win  = XCreateWindow( XiWin->disp, 
			     RootWindow(XiWin->disp,XiWin->screen),
                             x, y, w, h, border_width,
                             depth, InputOutput, XiWin->vis,
                             wmask, &window_attributes );

  if (!XiWin->win)  return 2;

  /* set window manager hints */
  {
    XWMHints      wm_hints;
    XClassHint    class_hints;
    XTextProperty windowname,iconname;
    
    XStringListToTextProperty(&label,1,&windowname);
    XStringListToTextProperty(&label,1,&iconname);
    
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
    return 1;
  }
  /* Initial values for the upper left corner */
  XiWin->x = 0;
  XiWin->y = 0;
  return 0;
}

int XiQuickWindow(XiWindow* mywindow,char* host,char* name,int x,int y,
                   int nx,int ny,int nc )
{
  int ierr;
  if (XiOpenDisplay( mywindow, host )) {
    fprintf( stderr, "Could not open display\n" );
    return 1;
  }
  if (XiSetVisual( mywindow, 1, (Colormap)0, nc )) {
    fprintf( stderr, "Could not set visual to default\n" );
    return 1;
  }
  if (XiDisplayWindow( mywindow, name, x, y, nx, ny, (PixVal)0 )) {
    fprintf( stderr, "Could not display window\n" );
    return 1;
  }
  XiSetGC( mywindow, mywindow->cmapping[1] );
  XiSetPixVal(mywindow, mywindow->background );
  XFillRectangle(mywindow->disp,mywindow->win,mywindow->gc.set, 0, 0, 
                                             mywindow->w,mywindow->h);
  ierr = XiUniformHues(mywindow,nc-36); CHKERR(ierr);
  ierr = XiFontFixed( mywindow,6, 10,&mywindow->font ); CHKERR(ierr);
  return 0;
}

/* 
   And a version from an already defined window
 */
int XiQuickWindowFromWindow(XiWindow* mywindow,char *host,Window win,int nc)
{
  Window       root;
  int          d,ierr;
  unsigned int ud;
  if (XiOpenDisplay( mywindow, host )) {SETERR(1,"Cannot open display");}
  if (XiSetVisual( mywindow, 1, (Colormap)0, 0 )) {
    SETERR(1,"Cannot set visual in display");
  }

  mywindow->win = win;
  XGetGeometry( mywindow->disp, mywindow->win, &root, 
              &d, &d, 
	      (unsigned int *)&mywindow->w, (unsigned int *)&mywindow->h,
              &ud, &ud );
  mywindow->x = mywindow->y = 0;

  XiSetGC( mywindow, mywindow->cmapping[1] );
  XiSetPixVal(mywindow, mywindow->background );
  ierr = XiUniformHues(mywindow,nc-36); CHKERR(ierr);
  ierr = XiFontFixed( mywindow,6, 10,&mywindow->font ); CHKERR(ierr);
  return 0;
}


/*
      XiSetWindowLabel - Sets new label in open window.

  Input Parameters:
.  window - Window to set label for
.  label  - Label to give window
*/
int XiSetWindowLabel(XiWindow* Xiwin, char *label )
{
  XTextProperty prop;
  XGetWMName(Xiwin->disp,Xiwin->win,&prop);
  prop.value = (unsigned char *)label; prop.nitems = (long) strlen(label);
  XSetWMName(Xiwin->disp,Xiwin->win,&prop);
  return 0;
}

int XiSetToBackground(XiWindow* XiWin )
{
  if (XiWin->gc.cur_pix != XiWin->background) { 
    XSetForeground( XiWin->disp, XiWin->gc.set, XiWin->background ); 
    XiWin->gc.cur_pix   = XiWin->background;
  }
  return 0;
}








