/*
   Defines the internal data structures for the SDL3
   implementation of the graphics functionality in PETSc.
*/

#pragma once
#include <petsc/private/drawimpl.h>

#include <SDL3/SDL.h>
#include <SDL3_ttf/SDL_ttf.h>

/*
  Each MPI process draws on its own offscreen surface. Cleared pixels are white with zero alpha. Drawn pixels have
  nonzero alpha, so an MPI_MAX reduction of the ARGB8888 pixel values combines the images of all processes.
  The first process displays the combined image in its window.
*/
typedef struct {
  SDL_Window     *win;       // window, only on the first process of a draw that is not virtual
  SDL_Renderer   *wrenderer; // renderer of the window
  SDL_Texture    *display;   // texture holding the combined image displayed in the window
  unsigned int   *image;     // combined image of all processes, only on the first process
  SDL_Surface    *surface;   // offscreen surface this process draws on
  unsigned int   *pixels;    // ARGB8888 pixel storage of surface, w*h contiguous values
  SDL_Renderer   *renderer;  // software renderer targeting surface
  SDL_Texture    *palette;   // PETSC_DRAW_MAXCOLOR x 1 colormap texture, for interpolating colors in colormap index space
  TTF_TextEngine *engine;    // text engine associated with renderer
  TTF_Font       *font;      // current font
  int             font_w, font_h;
  int             font_descent;
  unsigned char   cpalette[PETSC_DRAW_MAXCOLOR][3]; // map color -> RGB value
  int             x, y, w, h;                       // location and size of window
  PetscBool       isvirtual;                        // no window, the combined image is only available with PetscDrawGetImage_SDL3()
  PetscBool       shared;                           // surface, renderer, and window are shared with the parent of a singleton
  PetscBool       ownfont;                          // font was opened by this draw, which must close it
  PetscBool       drawn;                            // this process drew on surface since it was last cleared completely
  PetscBool      *pdrawn;                           // where drawing is recorded, drawn of the parent for a singleton
} PetscDraw_SDL3;

#if PetscDefined(USE_DEBUG)
  #define PetscDrawSDL3ValidColor(sdl, color) PetscCheck((color) >= 0 && (color) < PETSC_DRAW_MAXCOLOR, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Color value %d out of range [0..%d]", color, PETSC_DRAW_MAXCOLOR - 1)
#else
  #define PetscDrawSDL3ValidColor(sdl, color) \
    do { \
    } while (0)
#endif

// record that this process drew, so PetscDrawSDL3Combine() must include its surface
#define PetscDrawSDL3MarkDrawn(sdl) (*(sdl)->pdrawn = PETSC_TRUE)

#define PetscDrawSDL3SetColor(sdl, color) \
  do { \
    PetscDrawSDL3ValidColor(sdl, color); \
    (void)SDL_SetRenderDrawColor((sdl)->renderer, (sdl)->cpalette[(color)][0], (sdl)->cpalette[(color)][1], (sdl)->cpalette[(color)][2], SDL_ALPHA_OPAQUE); \
  } while (0)

PETSC_INTERN PetscErrorCode PetscDrawSDL3SetUpColormap(PetscDraw_SDL3 *);
PETSC_INTERN PetscErrorCode PetscDrawSDL3OpenWindow(PetscDraw_SDL3 *, const char[], int, int, int, int);
PETSC_INTERN PetscErrorCode PetscDrawSDL3CreateDisplay(PetscDraw_SDL3 *);
PETSC_INTERN PetscErrorCode PetscDrawSDL3OpenSurface(PetscDraw_SDL3 *, int, int);
PETSC_INTERN PetscErrorCode PetscDrawSDL3CloseSurface(PetscDraw_SDL3 *);
PETSC_INTERN PetscErrorCode PetscDrawSDL3ClearRect(PetscDraw_SDL3 *, const SDL_FRect *);
PETSC_INTERN PetscErrorCode PetscDrawSDL3Close(PetscDraw_SDL3 *);
PETSC_INTERN PetscErrorCode PetscDrawSDL3VideoAcquire(PetscBool *);
PETSC_INTERN PetscErrorCode PetscDrawSDL3VideoRelease(void);
PETSC_INTERN PetscErrorCode PetscDrawSDL3GetDisplaySize(int *, int *);
PETSC_INTERN PetscErrorCode PetscDrawSDL3SetFont(PetscDraw, int, int);
PETSC_INTERN PetscErrorCode PetscDrawSDL3Combine(PetscDraw, unsigned int **);
PETSC_INTERN PetscErrorCode PetscDrawGetImage_SDL3(PetscDraw, unsigned char[PETSC_DRAW_MAXCOLOR][3], unsigned int *, unsigned int *, unsigned char *[]);
