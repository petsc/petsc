
#include <../src/sys/classes/draw/impls/x/ximpl.h>

/*
    This routine waits until the window is actually created or destroyed
    Returns 0 if window is mapped; 1 if window is destroyed.
 */
#undef __FUNCT__
#define __FUNCT__ "PetscDrawXi_wait_map"
PetscErrorCode PetscDrawXi_wait_map(PetscDraw_X *XiWin)
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
