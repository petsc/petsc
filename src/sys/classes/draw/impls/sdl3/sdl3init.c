/*
   This file contains routines to open and close SDL3 windows and offscreen surfaces
   for the SDL3 implementation of the PetscDraw functionality.
*/

#include <../src/sys/classes/draw/impls/sdl3/sdl3impl.h>

static int PetscDrawSDL3VideoCount = 0; // number of windows sharing the video subsystem

/*
  PetscDrawSDL3VideoAcquire - initialize the SDL3 video subsystem on first use

  Output Parameter:
. ok - `PETSC_TRUE` if the video subsystem is available

  Note:
  Failure to initialize video (for example, on a machine without a display) is not an error.
  The caller then uses a `PETSC_DRAW_NULL` or offscreen draw.
  Each successful acquire must be matched with a `PetscDrawSDL3VideoRelease()`.
*/
PetscErrorCode PetscDrawSDL3VideoAcquire(PetscBool *ok)
{
  PetscFunctionBegin;
  *ok = PETSC_TRUE;
  // otherwise SDL3 converts SIGINT and SIGTERM to SDL_EVENT_QUIT events, which PETSc ignores
  (void)SDL_SetHint(SDL_HINT_NO_SIGNAL_HANDLERS, "1");
  if (PetscDrawSDL3VideoCount == 0 && !SDL_Init(SDL_INIT_VIDEO)) {
    PetscCall(PetscInfo(NULL, "Unable to initialize SDL3 video: %s\n", SDL_GetError()));
    *ok = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscDrawSDL3VideoCount++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// PetscDrawSDL3VideoRelease - release a reference obtained with PetscDrawSDL3VideoAcquire()
PetscErrorCode PetscDrawSDL3VideoRelease(void)
{
  PetscFunctionBegin;
  if (--PetscDrawSDL3VideoCount <= 0) {
    PetscDrawSDL3VideoCount = 0;
    SDL_QuitSubSystem(SDL_INIT_VIDEO);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDrawSDL3GetDisplaySize - get the size in pixels of the primary display

  Output Parameters:
+ w - width
- h - height
*/
PetscErrorCode PetscDrawSDL3GetDisplaySize(int *w, int *h)
{
  SDL_Rect bounds;

  PetscFunctionBegin;
  if (SDL_GetDisplayBounds(SDL_GetPrimaryDisplay(), &bounds)) {
    *w = bounds.w;
    *h = bounds.h;
  } else {
    PetscCall(PetscInfo(NULL, "Unable to query SDL3 display bounds: %s, using 1024x768\n", SDL_GetError()));
    *w = 1024;
    *h = 768;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// create the colormap texture of the renderer from the palette of the draw
static PetscErrorCode PetscDrawSDL3CreatePalette(PetscDraw_SDL3 *sdl)
{
  unsigned char pixels[PETSC_DRAW_MAXCOLOR][4];

  PetscFunctionBegin;
  for (int i = 0; i < PETSC_DRAW_MAXCOLOR; i++) {
    pixels[i][0] = sdl->cpalette[i][0];
    pixels[i][1] = sdl->cpalette[i][1];
    pixels[i][2] = sdl->cpalette[i][2];
    pixels[i][3] = SDL_ALPHA_OPAQUE;
  }
  sdl->palette = SDL_CreateTexture(sdl->renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STATIC, PETSC_DRAW_MAXCOLOR, 1);
  PetscCheck(sdl->palette != NULL, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unable to create SDL3 palette texture: %s", SDL_GetError());
  PetscCheck(SDL_UpdateTexture(sdl->palette, NULL, pixels, (int)sizeof(pixels)), PETSC_COMM_SELF, PETSC_ERR_LIB, "Unable to set SDL3 palette texture: %s", SDL_GetError());
  (void)SDL_SetTextureScaleMode(sdl->palette, SDL_SCALEMODE_NEAREST);
  (void)SDL_SetTextureBlendMode(sdl->palette, SDL_BLENDMODE_NONE);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Open a window at [x,y] with size (w,h) that displays the combined image of all processes
PetscErrorCode PetscDrawSDL3OpenWindow(PetscDraw_SDL3 *sdl, const char title[], int x, int y, int w, int h)
{
  PetscBool ok;

  PetscFunctionBegin;
  // each window holds its own reference to the video subsystem
  PetscCall(PetscDrawSDL3VideoAcquire(&ok));
  PetscCheck(ok == PETSC_TRUE, PETSC_COMM_SELF, PETSC_ERR_LIB, "SDL3 video subsystem not available: %s", SDL_GetError());
  sdl->win = SDL_CreateWindow(title != NULL ? title : "", w, h, SDL_WINDOW_RESIZABLE);
  PetscCheck(sdl->win != NULL, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unable to create SDL3 window: %s", SDL_GetError());
  (void)SDL_SetWindowPosition(sdl->win, x, y);
  sdl->wrenderer = SDL_CreateRenderer(sdl->win, NULL);
  PetscCheck(sdl->wrenderer != NULL, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unable to create SDL3 renderer: %s", SDL_GetError());
  sdl->x = x;
  sdl->y = y;
  PetscCall(PetscDrawSDL3CreateDisplay(sdl));
  (void)SDL_SetRenderDrawColor(sdl->wrenderer, sdl->cpalette[PETSC_DRAW_WHITE][0], sdl->cpalette[PETSC_DRAW_WHITE][1], sdl->cpalette[PETSC_DRAW_WHITE][2], SDL_ALPHA_OPAQUE);
  (void)SDL_RenderClear(sdl->wrenderer);
  (void)SDL_RenderPresent(sdl->wrenderer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// (re)create the window texture that receives the combined image. It has the current size of the draw.
PetscErrorCode PetscDrawSDL3CreateDisplay(PetscDraw_SDL3 *sdl)
{
  PetscFunctionBegin;
  if (sdl->display != NULL) SDL_DestroyTexture(sdl->display);
  // the alpha value of the combined image only marks drawn pixels, so it is ignored for display
  sdl->display = SDL_CreateTexture(sdl->wrenderer, SDL_PIXELFORMAT_XRGB8888, SDL_TEXTUREACCESS_STREAMING, sdl->w, sdl->h);
  PetscCheck(sdl->display != NULL, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unable to create SDL3 display texture: %s", SDL_GetError());
  (void)SDL_SetTextureScaleMode(sdl->display, SDL_SCALEMODE_NEAREST);
  (void)SDL_SetTextureBlendMode(sdl->display, SDL_BLENDMODE_NONE);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// fill a rectangle of the surface (all of it for NULL) with the background, white with zero alpha
PetscErrorCode PetscDrawSDL3ClearRect(PetscDraw_SDL3 *sdl, const SDL_FRect *rect)
{
  PetscFunctionBegin;
  (void)SDL_SetRenderDrawColor(sdl->renderer, sdl->cpalette[PETSC_DRAW_WHITE][0], sdl->cpalette[PETSC_DRAW_WHITE][1], sdl->cpalette[PETSC_DRAW_WHITE][2], SDL_ALPHA_TRANSPARENT);
  if (rect != NULL) (void)SDL_RenderFillRect(sdl->renderer, rect);
  else (void)SDL_RenderClear(sdl->renderer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Create the offscreen surface of size (w,h) this process draws on. It requires no window or video subsystem.
PetscErrorCode PetscDrawSDL3OpenSurface(PetscDraw_SDL3 *sdl, int w, int h)
{
  PetscFunctionBegin;
  // the pixel storage is contiguous so it can be passed directly to MPI_Reduce()
  PetscCall(PetscMalloc1((PetscCount)w * h, &sdl->pixels));
  sdl->surface = SDL_CreateSurfaceFrom(w, h, SDL_PIXELFORMAT_ARGB8888, sdl->pixels, w * (int)sizeof(*sdl->pixels));
  PetscCheck(sdl->surface != NULL, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unable to create SDL3 surface: %s", SDL_GetError());
  sdl->renderer = SDL_CreateSoftwareRenderer(sdl->surface);
  PetscCheck(sdl->renderer != NULL, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unable to create SDL3 software renderer: %s", SDL_GetError());
  sdl->engine = TTF_CreateRendererTextEngine(sdl->renderer);
  PetscCheck(sdl->engine != NULL, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unable to create SDL3_ttf text engine: %s", SDL_GetError());
  PetscCall(PetscDrawSDL3CreatePalette(sdl));
  sdl->w     = w;
  sdl->h     = h;
  sdl->drawn = PETSC_FALSE;
  PetscCall(PetscDrawSDL3ClearRect(sdl, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// destroy the offscreen surface and everything that depends on its renderer
PetscErrorCode PetscDrawSDL3CloseSurface(PetscDraw_SDL3 *sdl)
{
  PetscFunctionBegin;
  if (sdl->engine != NULL) TTF_DestroyRendererTextEngine(sdl->engine);
  if (sdl->palette != NULL) SDL_DestroyTexture(sdl->palette);
  if (sdl->renderer != NULL) SDL_DestroyRenderer(sdl->renderer);
  if (sdl->surface != NULL) SDL_DestroySurface(sdl->surface);
  PetscCall(PetscFree(sdl->pixels));
  sdl->engine   = NULL;
  sdl->palette  = NULL;
  sdl->renderer = NULL;
  sdl->surface  = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDrawSDL3Close(PetscDraw_SDL3 *sdl)
{
  PetscFunctionBegin;
  if (sdl == NULL) PetscFunctionReturn(PETSC_SUCCESS);
  if (sdl->font != NULL && sdl->ownfont == PETSC_TRUE) TTF_CloseFont(sdl->font);
  if (sdl->shared == PETSC_FALSE) {
    // PetscDrawCreate_SDL3() called TTF_Init() for each draw that has a surface, but not for singletons
    PetscBool ttf = (PetscBool)(sdl->renderer != NULL);

    PetscCall(PetscDrawSDL3CloseSurface(sdl));
    if (sdl->display != NULL) SDL_DestroyTexture(sdl->display);
    PetscCall(PetscFree(sdl->image));
    if (sdl->wrenderer != NULL) SDL_DestroyRenderer(sdl->wrenderer);
    if (sdl->win != NULL) {
      SDL_DestroyWindow(sdl->win);
      PetscCall(PetscDrawSDL3VideoRelease());
    }
    if (ttf == PETSC_TRUE) TTF_Quit();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
