#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: wmap.c,v 1.21 1999/01/31 16:05:02 bsmith Exp bsmith $";
#endif

/* Include petsc in case it is including petscconf.h */
#include "petsc.h"

#if defined(PETSC_HAVE_X11)
#include "src/sys/src/draw/impls/x/ximpl.h"

/*
    This routine waits until the window is actually created or destroyed
    Returns 0 if window is mapped; 1 if window is destroyed.
 */
#undef __FUNC__  
#define __FUNC__ "Xi_wait_map" 
int Xi_wait_map( Draw_X *XiWin)
{
  XEvent  event;
  int     w, h;

  PetscFunctionBegin;
  /*
   This is a bug.  XSelectInput should be set BEFORE the window is mapped
  */
  /*
  XSelectInput(XiWin->disp, XiWin->win,ExposureMask | StructureNotifyMask);
  */
  while (1) {
    XMaskEvent( XiWin->disp, ExposureMask | StructureNotifyMask, &event );
    if (event.xany.window != XiWin->win) {
      break;
      /* Bug for now */
    }
    else {
      switch (event.type) {
        case ConfigureNotify:
        /* window has been moved or resized */
        w         = event.xconfigure.width  - 2 * event.xconfigure.border_width;
        h         = event.xconfigure.height - 2 * event.xconfigure.border_width;
        XiWin->w  = w;
        XiWin->h  = h;
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
#else
int dummy_wmap(void)
{
  PetscFunctionReturn(0);
}

#endif
