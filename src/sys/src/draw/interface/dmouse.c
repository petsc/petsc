#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dmouse.c,v 1.8 1997/02/22 02:27:05 bsmith Exp balay $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawGetMouseButton" /* ADIC Ignore */
/*@
       DrawGetMouseButton - Returns location of mouse and which button was
            pressed. Waits for button to be pressed.

  Input Parameter:
.   draw - the window to be used

  Output Parameters:
.   button - one of BUTTON_LEFT, BUTTON_CENTER, BUTTON_RIGHT
.   x_user, y_user - user coordinates of location (user may pass in 0).
.   x_phys, y_phys - window coordinates (user may pass in 0).

@*/
int DrawGetMouseButton(Draw draw,DrawButton *button,double* x_user,double *y_user,
                       double *x_phys,double *y_phys)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  *button = BUTTON_NONE;
  if (draw->type == DRAW_NULLWINDOW) return 0;
  if (!draw->ops.getmousebutton) return 0;
  return (*draw->ops.getmousebutton)(draw,button,x_user,y_user,x_phys,y_phys);
}






