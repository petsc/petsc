/*
    Code for managing color for the SDL3 implementation of the PetscDraw routines.

    The palette uses entries [0..PETSC_DRAW_BASIC_COLORS-1] for the basic colors and
    entries [PETSC_DRAW_BASIC_COLORS..PETSC_DRAW_MAXCOLOR-1] for the contour colormap.
*/
#include <../src/sys/classes/draw/impls/sdl3/sdl3impl.h>

PetscErrorCode PetscDrawSDL3SetUpColormap(PetscDraw_SDL3 *sdl)
{
  int           ncolors = PETSC_DRAW_MAXCOLOR - PETSC_DRAW_BASIC_COLORS;
  unsigned char R[PETSC_DRAW_MAXCOLOR - PETSC_DRAW_BASIC_COLORS];
  unsigned char G[PETSC_DRAW_MAXCOLOR - PETSC_DRAW_BASIC_COLORS];
  unsigned char B[PETSC_DRAW_MAXCOLOR - PETSC_DRAW_BASIC_COLORS];
  PetscBool     fast = PETSC_FALSE;

  PetscFunctionBegin;
  // set the basic colors
  for (int i = 0; i < PETSC_DRAW_BASIC_COLORS; i++) {
    sdl->cpalette[i][0] = PetscDrawBasicColors[i][0];
    sdl->cpalette[i][1] = PetscDrawBasicColors[i][1];
    sdl->cpalette[i][2] = PetscDrawBasicColors[i][2];
  }

  // set the contour colors, honoring -draw_cmap and related options
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-draw_fast", &fast, NULL));
  if (fast == PETSC_TRUE) {
    // with -draw_fast no colormap is created for contour plots, as with the X implementation
    PetscCall(PetscMemzero(R, sizeof(R)));
    PetscCall(PetscMemzero(G, sizeof(G)));
    PetscCall(PetscMemzero(B, sizeof(B)));
  } else PetscCall(PetscDrawUtilitySetCmap(NULL, ncolors, R, G, B));
  for (int i = 0, k = PETSC_DRAW_BASIC_COLORS; i < ncolors; i++, k++) {
    sdl->cpalette[k][0] = R[i];
    sdl->cpalette[k][1] = G[i];
    sdl->cpalette[k][2] = B[i];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
