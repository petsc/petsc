#ifndef lint
static char vcid[] = "$Id: pbvec.c,v 1.7 1995/03/06 03:56:21 bsmith Exp bsmith $";
#endif
#include "ximpl.h"

/*
    This routine waits until the window is actually created or destroyed
    Returns 0 if window is mapped; 1 if window is destroyed.
 */
int Xi_wait_map( XiWindow *XiWin)
{
  XEvent  event;
  int     w, h;

  /*
   This is a bug.  XSelectInput should be set BEFORE the window is mapped
  */
  /*
  XSelectInput( XiWin->disp, XiWin->win,
	        ExposureMask | StructureNotifyMask );
  */
  while (1) {
    XMaskEvent( XiWin->disp, ExposureMask | StructureNotifyMask, &event );
    if (event.xany.window != XiWin->win) {
	/* Bug for now */
    }
    else {
      switch (event.type) {
        case ConfigureNotify:
        /* window has been moved or resized */
        w       = event.xconfigure.width  - 2 * event.xconfigure.border_width;
        h       = event.xconfigure.height - 2 * event.xconfigure.border_width;
        XiWin->w  = w;
        XiWin->h  = h;
        break;
      case DestroyNotify:
        /* printf( "in destroy notify\n" ); */
        return 1;
      case Expose:
        return 0;
      /* else ignore event */
      }
    }
  }
  return 0;
}

