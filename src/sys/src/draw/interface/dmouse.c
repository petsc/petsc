#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dmouse.c,v 1.24 1999/10/13 20:36:30 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawGetMouseButton" 
/*@
    DrawGetMouseButton - Returns location of mouse and which button was
    pressed. Waits for button to be pressed.

    Not collective (Use DrawSynchronizedGetMouseButton() for collective)

    Input Parameter:
.   draw - the window to be used

    Output Parameters:
+   button - one of BUTTON_LEFT, BUTTON_CENTER, BUTTON_RIGHT
.   x_user, y_user - user coordinates of location (user may pass in 0).
-   x_phys, y_phys - window coordinates (user may pass in 0).

    Level: intermediate

    Notes:
    Only processor 0 of the communicator used to create the Draw may call this routine.

.seealso: DrawSynchronizedGetMouseButton()
@*/
int DrawGetMouseButton(Draw draw,DrawButton *button,double* x_user,double *y_user,
                       double *x_phys,double *y_phys)
{
  int        ierr;
  PetscTruth isnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  *button = BUTTON_NONE;
  ierr = PetscTypeCompare((PetscObject)draw,DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  if (!draw->ops->getmousebutton) PetscFunctionReturn(0);
  ierr = (*draw->ops->getmousebutton)(draw,button,x_user,y_user,x_phys,y_phys);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawSynchronizedGetMouseButton"
/*@
    DrawSynchronizedGetMouseButton - Returns location of mouse and which button was
    pressed. Waits for button to be pressed.

    Collective over Draw

    Input Parameter:
.   draw - the window to be used

    Output Parameters:
+   button - one of BUTTON_LEFT, BUTTON_CENTER, BUTTON_RIGHT
.   x_user, y_user - user coordinates of location (user may pass in 0).
-   x_phys, y_phys - window coordinates (user may pass in 0).

    Level: intermediate

.seealso: DrawGetMouseButton()
@*/
int DrawSynchronizedGetMouseButton(Draw draw,DrawButton *button,double* x_user,double *y_user,
                       double *x_phys,double *y_phys)
{
  double bcast[4];
  int    ierr,rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  ierr = MPI_Comm_rank(draw->comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = DrawGetMouseButton(draw,button,x_user,y_user,x_phys,y_phys);CHKERRQ(ierr);
  }
  if (button) {
     ierr = MPI_Bcast(button,1,MPI_INT,0,draw->comm);CHKERRQ(ierr);
  }
  if (x_user) bcast[0] = *x_user;
  if (y_user) bcast[1] = *y_user;
  if (x_phys) bcast[2] = *x_phys;
  if (y_phys) bcast[3] = *y_phys;
  ierr = MPI_Bcast(bcast,4,MPI_DOUBLE,0,draw->comm);CHKERRQ(ierr);  
  if (x_user) *x_user = bcast[0];
  if (y_user) *y_user = bcast[1];
  if (x_phys) *x_phys = bcast[2];
  if (y_phys) *y_phys = bcast[3];
  PetscFunctionReturn(0);
}






