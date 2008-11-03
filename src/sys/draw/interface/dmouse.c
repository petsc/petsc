#define PETSC_DLL
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "../src/sys/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawGetMouseButton" 
/*@
    PetscDrawGetMouseButton - Returns location of mouse and which button was
    pressed. Waits for button to be pressed.

    Not collective (Use PetscDrawSynchronizedGetMouseButton() for collective)

    Input Parameter:
.   draw - the window to be used

    Output Parameters:
+   button - one of BUTTON_LEFT, BUTTON_CENTER, BUTTON_RIGHT
.   x_user, y_user - user coordinates of location (user may pass in 0).
-   x_phys, y_phys - window coordinates (user may pass in 0).

    Level: intermediate

    Notes:
    Only processor 0 of the communicator used to create the PetscDraw may call this routine.

.seealso: PetscDrawSynchronizedGetMouseButton()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscDrawGetMouseButton(PetscDraw draw,PetscDrawButton *button,PetscReal* x_user,PetscReal *y_user,PetscReal *x_phys,PetscReal *y_phys)
{
  PetscErrorCode ierr;
  PetscTruth isnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);
  *button = BUTTON_NONE;
  ierr = PetscTypeCompare((PetscObject)draw,PETSC_DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  if (!draw->ops->getmousebutton) PetscFunctionReturn(0);
  ierr = (*draw->ops->getmousebutton)(draw,button,x_user,y_user,x_phys,y_phys);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSynchronizedGetMouseButton" 
/*@
    PetscDrawSynchronizedGetMouseButton - Returns location of mouse and which button was
    pressed. Waits for button to be pressed.

    Collective over PetscDraw

    Input Parameter:
.   draw - the window to be used

    Output Parameters:
+   button - one of BUTTON_LEFT, BUTTON_CENTER, BUTTON_RIGHT
.   x_user, y_user - user coordinates of location (user may pass in 0).
-   x_phys, y_phys - window coordinates (user may pass in 0).

    Level: intermediate

.seealso: PetscDrawGetMouseButton()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscDrawSynchronizedGetMouseButton(PetscDraw draw,PetscDrawButton *button,PetscReal* x_user,PetscReal *y_user,PetscReal *x_phys,PetscReal *y_phys)
{
  PetscReal      bcast[4];
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);
  ierr = MPI_Comm_rank(((PetscObject)draw)->comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscDrawGetMouseButton(draw,button,x_user,y_user,x_phys,y_phys);CHKERRQ(ierr);
  }
  if (button) {
     ierr = MPI_Bcast(button,1,MPI_INT,0,((PetscObject)draw)->comm);CHKERRQ(ierr);
  }
  if (x_user) bcast[0] = *x_user;
  if (y_user) bcast[1] = *y_user;
  if (x_phys) bcast[2] = *x_phys;
  if (y_phys) bcast[3] = *y_phys;
  ierr = MPI_Bcast(bcast,4,MPIU_REAL,0,((PetscObject)draw)->comm);CHKERRQ(ierr);  
  if (x_user) *x_user = bcast[0];
  if (y_user) *y_user = bcast[1];
  if (x_phys) *x_phys = bcast[2];
  if (y_phys) *y_phys = bcast[3];
  PetscFunctionReturn(0);
}






