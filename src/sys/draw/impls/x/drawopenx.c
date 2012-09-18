
/*
    Defines the operations for the X PetscDraw implementation.
*/

#include <../src/sys/draw/impls/x/ximpl.h>         /*I  "petscsys.h" I*/

#undef __FUNCT__
#define __FUNCT__ "PetscDrawOpenX"
/*@C
   PetscDrawOpenX - Opens an X-window for use with the PetscDraw routines.

   Collective on MPI_Comm

   Input Parameters:
+  comm - the communicator that will share X-window
.  display - the X display on which to open,or null for the local machine
.  title - the title to put in the title bar,or null for no title
.  x,y - the screen coordinates of the upper left corner of window
          may use PETSC_DECIDE for these two arguments, then PETSc places the
          window
-  w, h - the screen width and height in pixels,  or PETSC_DRAW_HALF_SIZE, PETSC_DRAW_FULL_SIZE,
          or PETSC_DRAW_THIRD_SIZE or PETSC_DRAW_QUARTER_SIZE

   Output Parameters:
.  draw - the drawing context.

   Options Database Keys:
+  -nox - Disables all x-windows output
.  -display <name> - Sets name of machine for the X display
.  -draw_pause <pause> - Sets time (in seconds) that the
       program pauses after PetscDrawPause() has been called
       (0 is default, -1 implies until user input).
.  -draw_x_shared_colormap - Causes PETSc to use a shared
       colormap. By default PETSc creates a separate color
       for its windows, you must put the mouse into the graphics
       window to see  the correct colors. This options forces
       PETSc to use the default colormap which will usually result
       in bad contour plots.
.  -draw_fast - does not create colormap for countour plots
.  -draw_double_buffer - Uses double buffering for smooth animation.
-  -geometry - Indicates location and size of window

   Level: beginner

   Note:
   When finished with the drawing context, it should be destroyed
   with PetscDrawDestroy().

   Note for Fortran Programmers:
   Whenever indicating null character data in a Fortran code,
   PETSC_NULL_CHARACTER must be employed; using PETSC_NULL is not
   correct for character data!  Thus, PETSC_NULL_CHARACTER can be
   used for the display and title input parameters.

   Concepts: X windows^drawing to

.seealso: PetscDrawSynchronizedFlush(), PetscDrawDestroy(), PetscDrawCreate(), PetscDrawOpnOpenGL()
@*/
PetscErrorCode  PetscDrawOpenX(MPI_Comm comm,const char display[],const char title[],int x,int y,int w,int h,PetscDraw* draw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDrawCreate(comm,display,title,x,y,w,h,draw);CHKERRQ(ierr);
  ierr = PetscDrawSetType(*draw,PETSC_DRAW_X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}









