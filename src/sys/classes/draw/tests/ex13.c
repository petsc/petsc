static char help[] = "Tests the SDL3 PetscDraw implementation using an offscreen surface\n";

#include <petscsys.h>
#include <petscdraw.h>
#include <petsc/private/drawimpl.h> // for draw->ops->getimage

// check that some pixel in a 3x3 neighborhood of the user coordinate has the given color
static PetscBool ProbeImage(PetscDraw draw, unsigned int w, unsigned int h, const unsigned char pixels[], PetscReal x, PetscReal y, int c)
{
  int i, j;

  PetscCallAbort(PETSC_COMM_SELF, PetscDrawCoordinateToPixel(draw, x, y, &i, &j));
  for (int dj = -1; dj <= 1; dj++) {
    for (int di = -1; di <= 1; di++) {
      int ii = i + di, jj = j + dj;

      if (ii >= 0 && ii < (int)w && jj >= 0 && jj < (int)h && pixels[jj * w + ii] == (unsigned char)c) return PETSC_TRUE;
    }
  }
  return PETSC_FALSE;
}

int main(int argc, char **argv)
{
  PetscDraw      draw, sdraw;
  unsigned char  palette[PETSC_DRAW_MAXCOLOR][3];
  unsigned char *pixels = NULL;
  unsigned int   w = 0, h = 0;
  PetscBool      match = PETSC_TRUE;
  PetscReal      tw, th;
  PetscMPIInt    rank, size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  // CI passes -nox, which would replace the offscreen SDL3 draw with PETSC_DRAW_NULL
  PetscCall(PetscOptionsClearValue(NULL, "-nox"));

  PetscCall(PetscDrawOpenSDL3(PETSC_COMM_WORLD, "Test SDL3", 0, 0, 240, 180, &draw));
  PetscCall(PetscDrawSetFromOptions(draw));
  // exercise viewports and clearing
  PetscCall(PetscDrawSetViewPort(draw, .25, .25, .75, .75));
  PetscCall(PetscDrawClear(draw));
  PetscCall(PetscDrawSetViewPort(draw, 0, 0, 1, 1));

  PetscCall(PetscDrawLine(draw, .1, .1, .9, .9, PETSC_DRAW_RED));
  PetscCall(PetscDrawRectangle(draw, .1, .55, .3, .75, PETSC_DRAW_BLUE, PETSC_DRAW_BLUE, PETSC_DRAW_BLUE, PETSC_DRAW_BLUE));
  PetscCall(PetscDrawTriangle(draw, .6, .1, .85, .1, .725, .4, PETSC_DRAW_GREEN, PETSC_DRAW_GREEN, PETSC_DRAW_GREEN));
  // exercise the color interpolated triangle
  PetscCall(PetscDrawTriangle(draw, .55, .55, .95, .55, .75, .95, PETSC_DRAW_RED, PETSC_DRAW_GREEN, PETSC_DRAW_BLUE));
  PetscCall(PetscDrawEllipse(draw, .2, .2, .15, .1, PETSC_DRAW_MAGENTA));
  PetscCall(PetscDrawArrow(draw, .45, .5, .55, .5, PETSC_DRAW_ORANGE));
  PetscCall(PetscDrawPoint(draw, .9, .6, PETSC_DRAW_CYAN));
  // only the last process draws this. The combined image must contain it.
  if (rank == size - 1) PetscCall(PetscDrawRectangle(draw, .88, .25, .95, .32, PETSC_DRAW_YELLOW, PETSC_DRAW_YELLOW, PETSC_DRAW_YELLOW, PETSC_DRAW_YELLOW));
  // exercise the SDL3_ttf text rendering
  PetscCall(PetscDrawStringSetSize(draw, .03, .05));
  PetscCall(PetscDrawString(draw, .35, .85, PETSC_DRAW_BLACK, "SDL3 text"));
  PetscCall(PetscDrawStringVertical(draw, .05, .3, PETSC_DRAW_BLACK, "Vert"));
  // each process opens a font
  PetscCall(PetscDrawStringGetSize(draw, &tw, &th));
  match = (PetscBool)(tw > 0 && th > 0);
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &match, 1, MPI_C_BOOL, MPI_LAND, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "text size positive: %s\n", match == PETSC_TRUE ? "yes" : "no"));
  // exercise text on a singleton, which borrows the font of draw, and text on draw after the singleton is destroyed
  PetscCall(PetscDrawGetSingleton(draw, &sdraw));
  PetscCall(PetscDrawString(sdraw, .35, .78, PETSC_DRAW_BLACK, "single"));
  PetscCall(PetscDrawRestoreSingleton(draw, &sdraw));
  PetscCall(PetscDrawString(draw, .6, .45, PETSC_DRAW_BLACK, "after"));
  PetscCall(PetscDrawFlush(draw));

  PetscUseTypeMethod(draw, getimage, palette, &w, &h, &pixels);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Image size: %u x %u\n", w, h));
  match = PETSC_TRUE;
  if (rank == 0) match = ProbeImage(draw, w, h, pixels, .3, .3, PETSC_DRAW_RED);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "line probe: %s\n", match == PETSC_TRUE ? "yes" : "no"));
  if (rank == 0) match = ProbeImage(draw, w, h, pixels, .2, .65, PETSC_DRAW_BLUE);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "rectangle probe: %s\n", match == PETSC_TRUE ? "yes" : "no"));
  if (rank == 0) match = ProbeImage(draw, w, h, pixels, .725, .15, PETSC_DRAW_GREEN);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "triangle probe: %s\n", match == PETSC_TRUE ? "yes" : "no"));
  if (rank == 0) match = ProbeImage(draw, w, h, pixels, .2, .2, PETSC_DRAW_MAGENTA);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "ellipse probe: %s\n", match == PETSC_TRUE ? "yes" : "no"));
  if (rank == 0) match = ProbeImage(draw, w, h, pixels, .5, .5, PETSC_DRAW_ORANGE);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "arrow probe: %s\n", match == PETSC_TRUE ? "yes" : "no"));
  if (rank == 0) match = ProbeImage(draw, w, h, pixels, .9, .6, PETSC_DRAW_CYAN);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "point probe: %s\n", match == PETSC_TRUE ? "yes" : "no"));
  if (rank == 0) match = ProbeImage(draw, w, h, pixels, .915, .285, PETSC_DRAW_YELLOW);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "last process probe: %s\n", match == PETSC_TRUE ? "yes" : "no"));
  if (rank == 0) match = ProbeImage(draw, w, h, pixels, .5, .97, PETSC_DRAW_WHITE);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "background probe: %s\n", match == PETSC_TRUE ? "yes" : "no"));
  if (rank == 0) PetscCall(PetscFree(pixels));

  // exercise resizing the offscreen surface
  PetscCall(PetscDrawResizeWindow(draw, 300, 200));
  PetscUseTypeMethod(draw, getimage, palette, &w, &h, &pixels);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Resized image: %u x %u\n", w, h));
  match = PETSC_TRUE;
  if (rank == 0) match = ProbeImage(draw, w, h, pixels, .5, .5, PETSC_DRAW_WHITE);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "clean after resize: %s\n", match == PETSC_TRUE ? "yes" : "no"));
  if (rank == 0) PetscCall(PetscFree(pixels));

  // only the first process draws, so the image needs no reduction
  if (rank == 0) PetscCall(PetscDrawRectangle(draw, .1, .1, .3, .3, PETSC_DRAW_BLUE, PETSC_DRAW_BLUE, PETSC_DRAW_BLUE, PETSC_DRAW_BLUE));
  PetscUseTypeMethod(draw, getimage, palette, &w, &h, &pixels);
  match = PETSC_TRUE;
  if (rank == 0) match = ProbeImage(draw, w, h, pixels, .2, .2, PETSC_DRAW_BLUE);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "first process probe: %s\n", match == PETSC_TRUE ? "yes" : "no"));
  if (rank == 0) PetscCall(PetscFree(pixels));

  // the drawing of the last process is cleared, so again the image needs no reduction
  if (rank == size - 1) PetscCall(PetscDrawRectangle(draw, .6, .6, .8, .8, PETSC_DRAW_RED, PETSC_DRAW_RED, PETSC_DRAW_RED, PETSC_DRAW_RED));
  PetscCall(PetscDrawClear(draw));
  if (rank == 0) PetscCall(PetscDrawRectangle(draw, .1, .1, .3, .3, PETSC_DRAW_BLUE, PETSC_DRAW_BLUE, PETSC_DRAW_BLUE, PETSC_DRAW_BLUE));
  PetscUseTypeMethod(draw, getimage, palette, &w, &h, &pixels);
  match = PETSC_TRUE;
  if (rank == 0)
    match = (PetscBool)(ProbeImage(draw, w, h, pixels, .2, .2, PETSC_DRAW_BLUE) == PETSC_TRUE && ProbeImage(draw, w, h, pixels, .7, .7, PETSC_DRAW_WHITE) == PETSC_TRUE && ProbeImage(draw, w, h, pixels, .7, .7, PETSC_DRAW_RED) == PETSC_FALSE);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cleared probe: %s\n", match == PETSC_TRUE ? "yes" : "no"));
  if (rank == 0) PetscCall(PetscFree(pixels));

  PetscCall(PetscDrawDestroy(&draw));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: sdl3

   test:
     requires: sdl3
     args: -draw_virtual
     output_file: output/ex13.out

   test:
     suffix: 2
     nsize: 2
     requires: sdl3
     args: -draw_virtual
     output_file: output/ex13.out

   test:
     suffix: 3
     requires: sdl3
     args: -draw_virtual -draw_size .5

TEST*/
