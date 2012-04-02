
#include <../src/sys/draw/impls/x/ximpl.h>

/*
    This routine waits until the window is actually created or destroyed
    Returns 0 if window is mapped; 1 if window is destroyed.
 */
#undef __FUNCT__  
#define __FUNCT__ "PetscDrawXi_wait_map" 
PetscErrorCode PetscDrawXi_wait_map(PetscDraw_X *PetscDrawXiWin)
{
  XEvent  event;
  int     w,h;

  PetscFunctionBegin;
  /*
   This is a bug.  XSelectInput should be set BEFORE the window is mapped
  */
  /*
  XSelectInput(PetscDrawXiWin->disp,PetscDrawXiWin->win,ExposureMask | StructureNotifyMask);
  */
  while (1) {
    XMaskEvent(PetscDrawXiWin->disp,ExposureMask | StructureNotifyMask,&event);
    if (event.xany.window != PetscDrawXiWin->win) {
      break;
      /* Bug for now */
    } else {
      switch (event.type) {
        case ConfigureNotify:
        /* window has been moved or resized */
        w         = event.xconfigure.width  - 2 * event.xconfigure.border_width;
        h         = event.xconfigure.height - 2 * event.xconfigure.border_width;
        PetscDrawXiWin->w  = w;
        PetscDrawXiWin->h  = h;
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
