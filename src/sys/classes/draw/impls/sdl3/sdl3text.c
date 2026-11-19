// This file contains code to manage access to fonts via SDL3_ttf

#include <../src/sys/classes/draw/impls/sdl3/sdl3impl.h>

// candidate fixed-width font files. PetscDrawSDL3SetFont() uses the first one that it can open.
// clang-format off
static const char *const PetscDrawSDL3FontFiles[] = {
#if defined(__APPLE__)
  "/System/Library/Fonts/Monaco.ttf",
  "/System/Library/Fonts/Menlo.ttc",
  "/System/Library/Fonts/Courier.ttc",
  "/System/Library/Fonts/Helvetica.ttc",
#elif defined(_WIN32)
  "C:\\Windows\\Fonts\\cour.ttf",
  "C:\\Windows\\Fonts\\arial.ttf",
#else
  "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
  "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
  "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
  "/usr/share/fonts/liberation-mono/LiberationMono-Regular.ttf",
  "/usr/share/fonts/liberation-mono-fonts/LiberationMono-Regular.ttf",
  "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
#endif
};
// clang-format on

/*
    PetscDrawSDL3SetFont - open a font with approximately the requested pixel size

    Not Collective, each MPI process opens its own font

    Input Parameters:
+   draw - the draw context, provides the options database
.   w    - desired character width in pixels
-   h    - desired character height in pixels

    Options Database Key:
.   -draw_sdl3_font filename - font file to use instead of the searched ones

*/
PetscErrorCode PetscDrawSDL3SetFont(PetscDraw draw, int w, int h)
{
  PetscDraw_SDL3 *sdl    = (PetscDraw_SDL3 *)draw->data;
  float           ptsize = (float)PetscMax(h, 4);
  TTF_Font       *font   = NULL;
  char            fontfile[PETSC_MAX_PATH_LEN];
  PetscBool       set;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetString(((PetscObject)draw)->options, NULL, "-draw_sdl3_font", fontfile, sizeof(fontfile), &set));
  if (set == PETSC_TRUE) {
    font = TTF_OpenFont(fontfile, ptsize);
    PetscCheck(font != NULL, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unable to open font file %s with SDL3_ttf: %s", fontfile, SDL_GetError());
  } else {
    for (size_t i = 0; i < PETSC_STATIC_ARRAY_LENGTH(PetscDrawSDL3FontFiles); i++) {
      font = TTF_OpenFont(PetscDrawSDL3FontFiles[i], ptsize);
      if (font != NULL) break;
    }
    PetscCheck(font != NULL, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unable to open any font with SDL3_ttf; use -draw_sdl3_font filename to specify a font file");
  }

  // a singleton draw borrows the font of its parent and must not close it
  if (sdl->font != NULL && sdl->ownfont == PETSC_TRUE) TTF_CloseFont(sdl->font);
  sdl->font    = font;
  sdl->ownfont = PETSC_TRUE;

  // measure the font: use the width of a digit, as in a fixed-width font
  (void)TTF_GetStringSize(sdl->font, "0", 1, &sdl->font_w, NULL);
  sdl->font_h       = TTF_GetFontHeight(sdl->font);
  sdl->font_descent = TTF_GetFontDescent(sdl->font);
  PetscFunctionReturn(PETSC_SUCCESS);
}
