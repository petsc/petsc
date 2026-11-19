// PetscDrawOpenSDL3() - open a window using SDL3

#include <../src/sys/classes/draw/impls/sdl3/sdl3impl.h> /*I  "petscdraw.h" I*/

/*@
  PetscDrawOpenSDL3 - Opens an SDL3 window for use with the `PetscDraw` routines.

  Collective

  Input Parameters:
+ comm  - the communicator that will share the window
. title - the title to put in the title bar, or `NULL` for no title
. x     - the x screen coordinates of the upper left corner of window (or `PETSC_DECIDE`)
. y     - the y screen coordinates of the upper left corner of window (or `PETSC_DECIDE`)
. w     - the screen width in pixels of (or `PETSC_DRAW_HALF_SIZE`, `PETSC_DRAW_FULL_SIZE`, or `PETSC_DRAW_THIRD_SIZE` or `PETSC_DRAW_QUARTER_SIZE`)
- h     - the screen height in pixels of (or `PETSC_DRAW_HALF_SIZE`, `PETSC_DRAW_FULL_SIZE`, or `PETSC_DRAW_THIRD_SIZE` or `PETSC_DRAW_QUARTER_SIZE`)

  Output Parameter:
. draw - the drawing context.

  Options Database Keys:
+ -nox                            - Disables all graphics output
. -draw_pause pause               - Sets time (in seconds) that the program pauses after `PetscDrawPause()` has been called
                                    (0 is default, -1 implies until user input)
. -draw_cmap name                 - Sets the colormap to use
. -draw_cmap_reverse (true|false) - Reverses the colormap
. -draw_cmap_brighten beta        - Brighten (0 < `beta` < 1) or darken (-1 < `beta` < 0) the colormap
. -draw_sdl3_font filename        - Font file for SDL3_ttf text rendering
- -geometry x,y,w,h               - Indicates location and size of window

  Level: beginner

  Notes:
  If `x` and `y` are both `PETSC_DECIDE` then PETSc places the window automatically.

  Every process of the communicator draws; the first process displays the combined drawing of all
  processes in the window. See `PETSC_DRAW_SDL3`.

  When finished with the drawing context, it should be destroyed
  with `PetscDrawDestroy()`.

.seealso: `PetscDrawFlush()`, `PetscDrawDestroy()`, `PetscDrawCreate()`, `PETSC_DRAW_SDL3`, `PetscDrawOpenX()`
@*/
PetscErrorCode PetscDrawOpenSDL3(MPI_Comm comm, const char title[], int x, int y, int w, int h, PetscDraw *draw)
{
  PetscFunctionBegin;
  PetscCall(PetscDrawCreate(comm, NULL, title, x, y, w, h, draw));
  PetscCall(PetscDrawSetType(*draw, PETSC_DRAW_SDL3));
  PetscFunctionReturn(PETSC_SUCCESS);
}
