#ifndef lint
static char vcid[] = "$Id: wmap.c,v 1.2 1994/07/21 21:14:57 gropp Exp $";
#endif


#include "tools.h"
#include "xtools/ximpl.h"

/*
    This routine waits until the window is actually created or destroyed
    Returns 0 if window is mapped; 1 if window is destroyed.
 */
int XB_wait_map( XBWin, ExposeRoutine )
XBWindow XBWin;
void     (*ExposeRoutine)();
{
XEvent  event;
int     w, h;

/*
   This is a bug.  XSelectInput should be set BEFORE the window is mapped
 */
/*
XSelectInput( XBWin->disp, XBWin->win,
	      ExposureMask | StructureNotifyMask );
 */
while (1) {
    XMaskEvent( XBWin->disp, ExposureMask | StructureNotifyMask, &event );
    if (event.xany.window != XBWin->win) {
	/* Bug for now */
	}
    else {
	switch (event.type) {
	    case ConfigureNotify:
	    /* window has been moved or resized */
	    w       = event.xconfigure.width  - 2 * event.xconfigure.border_width;
	    h       = event.xconfigure.height - 2 * event.xconfigure.border_width;
	    XBWin->w  = w;
	    XBWin->h  = h;
	    break;
	case DestroyNotify:
	    /* printf( "in destroy notify\n" ); */
	    return 1;
	case Expose:
	    if (ExposeRoutine) 
		(*ExposeRoutine)( XBWin, event.xexpose.x, event.xexpose.y,
				  event.xexpose.width, event.xexpose.height );
	    return 0;
	/* else ignore event */
	    }
	}
    }
/* return 1; */
}

/*
    This routine reads a pixel from a window, thus insuring that everything
    has actually been drawn.  If the window is null, do ALL windows.
 */
void XBSync( XBWin )
XBWindow XBWin;
{
XImage   *xi;

if (XBWin->win) {
    xi = XGetImage( XBWin->disp, XBWin->win, 0, 0, 1, 1, AllPlanes,
                    XYPixmap );
    }
}
