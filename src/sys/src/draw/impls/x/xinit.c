
#include <stdio.h>
#include "ximpl.h"

/* 
   This file contains routines to open an X window display and window
   This consists of a number of routines that set the various
   fields in the XBaseWindow structure, which is passed to 
   all of these routines.

   Note that if you use the default visual and colormap, then you
   can use these routines with any X toolkit that will give you the
   Window id of the window that it is managing.  Use that instead of the
   call to XBCreateWindow .  Similarly for the Display.
 */


/*
  XBOpenDisplay - Open a display
  
  Input Parameters:
  XBWin        - pointer to base window structure
  display_name - either null ("") or of the form "host:0"
*/
int XBOpenDisplay(XBWindow XBWin,char *display_name )
{
  if (display_name && display_name[0] == 0)  display_name = 0;
  XBWin->disp = XOpenDisplay( display_name );

  if (!XBWin->disp)  return 1;

  /* Set the default screen */
  XBWin->screen = DefaultScreen( XBWin->disp );

  return 0;
}

/*  
    XBSetVisual - set the visual class for a window and colormap

    Input Parameters:
.   nc - number of colors.  Use the maximum of the visual if
    nc == 0.  Use nc = 2 for black and white displays.
  */
int XBSetVisual(XBWindow XBWin,int q_default_visual,Colormap cmap,int nc )
{
if (q_default_visual) {
    XBWin->vis    = DefaultVisual( XBWin->disp, XBWin->screen );
    XBWin->depth  = DefaultDepth(XBWin->disp,XBWin->screen);
    if (!cmap)
        XBWin->cmap  = DefaultColormap( XBWin->disp, XBWin->screen );
    else
        XBWin->cmap  = cmap;
    }
else {
    /* Try to match to some popular types */
    XVisualInfo vinfo;
    if (XMatchVisualInfo( XBWin->disp, XBWin->screen, 24, DirectColor,
			 &vinfo)) {
	XBWin->vis    = vinfo.visual;
	XBWin->depth  = 24;
	}
    else if (XMatchVisualInfo( XBWin->disp, XBWin->screen, 8, PseudoColor,
			 &vinfo)) {
	XBWin->vis    = vinfo.visual;
	XBWin->depth  = 8;
	}
    else if (XMatchVisualInfo( XBWin->disp, XBWin->screen,
			 DefaultDepth(XBWin->disp,XBWin->screen), PseudoColor,
			 &vinfo)) {
	XBWin->vis    = vinfo.visual;
	XBWin->depth  = DefaultDepth(XBWin->disp,XBWin->screen);
	}
    else {
	XBWin->vis    = DefaultVisual( XBWin->disp, XBWin->screen );
	XBWin->depth  = DefaultDepth(XBWin->disp,XBWin->screen);
	}
    /* There are other types; perhaps this routine should accept a 
       "hint" on which types to look for. */
    XBWin->cmap = (Colormap) 0;
    }

/* reset the number of colors from info on the display, the colormap */
XBInitColors( XBWin, cmap, nc );
return 0;
}

/* 
   XBSetGC - set the GC structure in the base window
  */
XBSetGC(XBWindow XBWin,PixVal fg )
{
XGCValues       gcvalues;       /* window graphics context values */

/* Set the graphics contexts */
/* create a gc for the ROP_SET operation (writing the fg value to a pixel) */
/* (do this with function GXcopy; GXset will automatically write 1) */
gcvalues.function   = GXcopy;
gcvalues.foreground = fg;
XBWin->gc.cur_pix   = fg;
XBWin->gc.set = XCreateGC( XBWin->disp, RootWindow(XBWin->disp,XBWin->screen),
                              GCFunction | GCForeground, &gcvalues );
return 0;
}


/*
    Actually display a window at [x,y] with sizes (w,h)
    If w and/or h are 0, use the sizes in the fields of XBWin
    (which may have been set by, for example, XBSetWindowSize)
 */
int XBDisplayWindow( XBWindow XBWin, char *label, int x, int y,
                     int w,int h,PixVal backgnd_pixel )
{
unsigned int    wavail, havail;
XSizeHints      size_hints;
int             q_user_pos;
XWindowAttributes       in_window_attributes;
XSetWindowAttributes    window_attributes;
int                     depth, border_width;
unsigned long           wmask;

/* get the available widths */
wavail              = DisplayWidth(  XBWin->disp, XBWin->screen );
havail              = DisplayHeight( XBWin->disp, XBWin->screen );

if (w <= 0 || h <= 0) return 2;

if (w > wavail)
    w   = wavail;
if (h > havail)
    h   = havail;

/* Find out if the user specified the position */
q_user_pos  = (x >= 0) && (y >= 0);

border_width   = 0;
if (x < 0) x   = 0;
if (y < 0) y   = 0;
x   = (x + w > wavail) ? wavail - w : x;
y   = (y + h > havail) ? havail - h : y;

/* We need XCreateWindow since we may need an visual other than
   the default one */
XGetWindowAttributes( XBWin->disp, RootWindow(XBWin->disp,XBWin->screen),
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
window_attributes.colormap          = XBWin->cmap;
/* None for cursor does NOT mean none, it means Parent's cursor */
window_attributes.cursor            = None; 
wmask   = CWBackPixmap | CWBackPixel | CWBorderPixmap | CWBitGravity |
          CWWinGravity | CWBackingStore | CWBackingPixel | CWOverrideRedirect |
          CWSaveUnder  | CWEventMask    | CWDontPropagate |
          CWCursor     | CWColormap ;
/* depth should really be the depth of the visual in use */
depth       = DefaultDepth( XBWin->disp, XBWin->screen );
XBWin->win  = XCreateWindow( XBWin->disp, 
			     RootWindow(XBWin->disp,XBWin->screen),
                             x, y, w, h, border_width,
                             depth, InputOutput, XBWin->vis,
                             wmask, &window_attributes );

if (!XBWin->win) 
    return 2;

/* set resize hints (prohibit?) */
size_hints.x            = x;
size_hints.y            = y;
size_hints.min_width    = 4*border_width;
size_hints.min_height   = 4*border_width;
size_hints.width        = w;
size_hints.height       = h;
if (q_user_pos)
    size_hints.flags        = USPosition | USSize | PMinSize; /*  | PAspect; */
else
    size_hints.flags        = PPosition  | PSize  | PMinSize; /*  | PAspect; */

/* Set the standard properties */
XSetStandardProperties( XBWin->disp, XBWin->win, label, label, 0,
                        (char **)0, 0, &size_hints );

/* make the window visible */
XSelectInput( XBWin->disp, XBWin->win, ExposureMask | StructureNotifyMask );
XMapWindow( XBWin->disp, XBWin->win );

/* some window systems are cruel and interfere with the placement of
   windows.  We wait here for the window to be created or to die */
if (XB_wait_map( XBWin, (void (*)())0 )) {
    XBWin->win    = (Window)0;
    return 0;
    }
/* Initial values for the upper left corner */
XBWin->x = 0;
XBWin->y = 0;
return 0;
}



/* 
   There should also be a routine that gets this data from a widget, so
   that this structure can be used with a widget set 
 */


int XBiQuickWindow(XBWindow mywindow,char* host,char* name,int x,int y,
                   int nx,int ny,int nc )
{
if (XBOpenDisplay( mywindow, host )) {
    fprintf( stderr, "Could not open display\n" );
    return 1;
    }
if (XBSetVisual( mywindow, 1, (Colormap)0, nc )) {
    fprintf( stderr, "Could not set visual to default\n" );
    return 1;
    }
if (XBOpenWindow( mywindow )) {
    fprintf( stderr, "Could not open the window\n" );
    return 1;
    }
if (XBDisplayWindow( mywindow, name, x, y, nx, ny, (PixVal)0 )) {
    fprintf( stderr, "Could not display window\n" );
    return 1;
    }
XBSetGC( mywindow, mywindow->cmapping[1] );
XBClearWindow(mywindow,0,0,mywindow->w,mywindow->h);
return 0;
}

/*
   XBQuickWindow - Create an X window

   Input parameters:
.  mywindow - A pointer to an XBWindow structure that will be used to hold
              information on the window.  This should be acquired with
	      XBWinCreate.
.  host     - name of the display
.  name     - title (name) of the window
.  x,y      - coordinates of the upper left corner of the window.  If <0,
              use user-positioning (normally, the window manager will
	      ask the user where to put the window)
.  nx,ny    - width and height of the window

   Note:
   This creates a window with various defaults (visual, colormap, etc)

   A small modification to this routine would allow Black and White windows
   to be used on color displays; this would be useful for testing codes.

   Use XBWinCreate to create a valid XBWindow for this routine.
*/

int DrawOpenX(char* display,char *name,int x,int y,int w,int h,DrawCtx* ctx)
{
  XBWindow mywindow;
  return XBiQuickWindow( mywindow, display, name, x, y, w, h, 0 );
}

/* 
   And a quick version (from an already defined window) 
 */
int XBQuickWindowFromWindow( XBWindow mywindow,char *host, Window win )
{
Window       root;
int          d;
unsigned int ud;

if (XBOpenDisplay( mywindow, host )) {
    fprintf( stderr, "Could not open display\n" );
    return 1;
    }
if (XBSetVisual( mywindow, 1, (Colormap)0, 0 )) {
    fprintf( stderr, "Could not set visual to default\n" );
    return 1;
    }

mywindow->win = win;
XGetGeometry( mywindow->disp, mywindow->win, &root, 
              &d, &d, 
	      (unsigned int *)&mywindow->w, (unsigned int *)&mywindow->h,
              &ud, &ud );
mywindow->x = mywindow->y = 0;

XBSetGC( mywindow, mywindow->cmapping[1] );
return 0;
}

/*
    XBFlush - Flush all X 11 requests.

    Input parameter:
.   XBWin - window

    Note:  Using an X drawing routine does not necessarily cause the
    the server to receive and draw the requests.  Use this routine
    if necessary to force the server to draw (doing so may slow down
    the program, so don't insert unnecessary XBFlush calls).

    If double-buffering is enabled, this routine copies from the buffer
    to the window before flushing the requests.  This is the appropriate
    action for animation.
*/
void XBFlush(XBWindow XBWin )
{
if (XBWin->drw) {
    XCopyArea( XBWin->disp, XBWin->drw, XBWin->win, XBWin->gc.set, 0, 0, 
	       XBWin->w, XBWin->h, XBWin->x, XBWin->y );
    }
XFlush( XBWin->disp );
}

/*
      XBSetWindowLabel - Sets new label in open window.

  Input Parameters:
.  window - Window to set label for
.  label  - Label to give window
*/
int XBSetWindowLabel(XBWindow XBwin, char *label )
{
  XTextProperty prop;
  XGetWMName(XBwin->disp,XBwin->win,&prop);
  prop.value = (unsigned char *)label; prop.nitems = (long) strlen(label);
  XSetWMName(XBwin->disp,XBwin->win,&prop);
  return 0;
}


PixVal XBWinForeground(XBWindow XBWin )
{
return XBWin->foreground;
}

void XBSetToForeground(XBWindow XBWin )
{
if (XBWin->gc.cur_pix != XBWin->foreground) { 
    XSetForeground( XBWin->disp, XBWin->gc.set, XBWin->foreground ); 
    XBWin->gc.cur_pix   = XBWin->foreground;
    }
}

void XBSetToBackground(XBWindow XBWin )
{
if (XBWin->gc.cur_pix != XBWin->background) { 
    XSetForeground( XBWin->disp, XBWin->gc.set, XBWin->background ); 
    XBWin->gc.cur_pix   = XBWin->background;
    }
}

/*
   XBSetForegroundByIndex - Set the foreground color to the given value

   Input Parameters:
.  XBWin - XBWindow structure
.  icolor - Index into the pre-establised window color map.
*/
void  XBSetForegroundByIndex( XBWindow XBWin,int icolor )
{
PixVal pixval = XBWin->cmapping[icolor];
if (XBWin->gc.cur_pix != pixval) { 
    XSetForeground( XBWin->disp, XBWin->gc.set, pixval ); 
    XBWin->gc.cur_pix   = pixval;
    }
}

PixVal XBGetPixvalByIndex(XBWindow XBWin,int icolor )
{
return XBWin->cmapping[icolor];
}

/*
   XBSetForeground - Set the foreground color to the given pixel value

   Input Parameters:
.  XBWin - XBWindow structure
.  pixval - Pixel value 
*/
void  XBSetForeground(XBWindow XBWin,PixVal pixval )
{
if (XBWin->gc.cur_pix != pixval) { 
    XSetForeground( XBWin->disp, XBWin->gc.set, pixval ); 
    XBWin->gc.cur_pix   = pixval;
    }
}

/*
  XBFillRectangle - Fills a rectangle

*/
void XBFillRectangle(XBWindow XBWin,int x,int y,int w,int h )
{
XFillRectangle( XBWin->disp, XBDrawable(XBWin), XBWin->gc.set, x, y, w, h );
}

/*
  XBDrawLine - Draws a line
*/
void XBDrawLine(XBWindow XBWin, int x1,int y1,int x2,int y2 )
{
XDrawLine( XBWin->disp, XBWin->win, XBWin->gc.set, x1, y1, x2, y2 );
}



