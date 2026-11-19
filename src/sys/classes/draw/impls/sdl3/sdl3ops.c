// Defines the operations for the SDL3 PetscDraw implementation.

#include <../src/sys/classes/draw/impls/sdl3/sdl3impl.h> /*I  "petscdraw.h" I*/

// These macros transform from the users coordinates to the window pixel coordinates.
#define XTRANS(draw, sdl, x) ((int)(((sdl)->w - 1) * ((draw)->port_xl + ((((x) - (draw)->coor_xl) * ((draw)->port_xr - (draw)->port_xl)) / ((draw)->coor_xr - (draw)->coor_xl)))))
#define YTRANS(draw, sdl, y) (((sdl)->h - 1) - (int)(((sdl)->h - 1) * ((draw)->port_yl + ((((y) - (draw)->coor_yl) * ((draw)->port_yr - (draw)->port_yl)) / ((draw)->coor_yr - (draw)->coor_yl)))))

#define ITRANS(draw, sdl, i) ((draw)->coor_xl + (((PetscReal)(i)) * ((draw)->coor_xr - (draw)->coor_xl) / ((sdl)->w - 1) - (draw)->port_xl) / ((draw)->port_xr - (draw)->port_xl))
#define JTRANS(draw, sdl, j) ((draw)->coor_yl + (((PetscReal)(j)) / ((sdl)->h - 1) + (draw)->port_yl - 1) * ((draw)->coor_yr - (draw)->coor_yl) / ((draw)->port_yl - (draw)->port_yr))

static PetscErrorCode PetscDrawSetViewport_SDL3(PetscDraw draw, PetscReal xl, PetscReal yl, PetscReal xr, PetscReal yr)
{
  PetscDraw_SDL3 *sdl  = (PetscDraw_SDL3 *)draw->data;
  int             xmax = sdl->w - 1, ymax = sdl->h - 1;
  int             xa = (int)(xl * (PetscReal)xmax), ya = ymax - (int)(yr * (PetscReal)ymax);
  int             xb = (int)(xr * (PetscReal)xmax), yb = ymax - (int)(yl * (PetscReal)ymax);
  SDL_Rect        box = {xa, ya, xb + 1 - xa, yb + 1 - ya};

  PetscFunctionBegin;
  (void)SDL_SetRenderClipRect(sdl->renderer, &box);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawCoordinateToPixel_SDL3(PetscDraw draw, PetscReal x, PetscReal y, int *i, int *j)
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;

  PetscFunctionBegin;
  *i = XTRANS(draw, sdl, x);
  *j = YTRANS(draw, sdl, y);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawPixelToCoordinate_SDL3(PetscDraw draw, int i, int j, PetscReal *x, PetscReal *y)
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;

  PetscFunctionBegin;
  *x = ITRANS(draw, sdl, i);
  *y = JTRANS(draw, sdl, j);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawPoint_SDL3(PetscDraw draw, PetscReal x, PetscReal y, int c)
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;
  int             xx = XTRANS(draw, sdl, x), yy = YTRANS(draw, sdl, y);
  SDL_FRect       pix = {(float)(xx - 1), (float)(yy - 1), 3, 3};

  PetscFunctionBegin;
  PetscDrawSDL3MarkDrawn(sdl);
  PetscDrawSDL3SetColor(sdl, c);
  (void)SDL_RenderFillRect(sdl->renderer, &pix);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawPointPixel_SDL3(PetscDraw draw, int x, int y, int c)
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;

  PetscFunctionBegin;
  PetscDrawSDL3MarkDrawn(sdl);
  PetscDrawSDL3SetColor(sdl, c);
  (void)SDL_RenderPoint(sdl->renderer, (float)x, (float)y);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawLine_SDL3(PetscDraw draw, PetscReal xl, PetscReal yl, PetscReal xr, PetscReal yr, int cl)
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;
  int             x_1 = XTRANS(draw, sdl, xl), y_1 = YTRANS(draw, sdl, yl), x_2 = XTRANS(draw, sdl, xr), y_2 = YTRANS(draw, sdl, yr);

  PetscFunctionBegin;
  PetscDrawSDL3MarkDrawn(sdl);
  PetscDrawSDL3SetColor(sdl, cl);
  (void)SDL_RenderLine(sdl->renderer, (float)x_1, (float)y_1, (float)x_2, (float)y_2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawArrow_SDL3(PetscDraw draw, PetscReal xl, PetscReal yl, PetscReal xr, PetscReal yr, int cl)
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;
  int             x_1 = XTRANS(draw, sdl, xl), y_1 = YTRANS(draw, sdl, yl), x_2 = XTRANS(draw, sdl, xr), y_2 = YTRANS(draw, sdl, yr);

  PetscFunctionBegin;
  PetscDrawSDL3MarkDrawn(sdl);
  PetscDrawSDL3SetColor(sdl, cl);
  (void)SDL_RenderLine(sdl->renderer, (float)x_1, (float)y_1, (float)x_2, (float)y_2);
  if (x_1 == x_2 && y_1 == y_2) PetscFunctionReturn(PETSC_SUCCESS);
  if (x_1 == x_2 && PetscAbs(y_1 - y_2) > 7) {
    if (y_2 > y_1) {
      (void)SDL_RenderLine(sdl->renderer, (float)x_2, (float)y_2, (float)(x_2 - 3), (float)(y_2 - 3));
      (void)SDL_RenderLine(sdl->renderer, (float)x_2, (float)y_2, (float)(x_2 + 3), (float)(y_2 - 3));
    } else {
      (void)SDL_RenderLine(sdl->renderer, (float)x_2, (float)y_2, (float)(x_2 - 3), (float)(y_2 + 3));
      (void)SDL_RenderLine(sdl->renderer, (float)x_2, (float)y_2, (float)(x_2 + 3), (float)(y_2 + 3));
    }
  }
  if (y_1 == y_2 && PetscAbs(x_1 - x_2) > 7) {
    if (x_2 > x_1) {
      (void)SDL_RenderLine(sdl->renderer, (float)(x_2 - 3), (float)(y_2 - 3), (float)x_2, (float)y_2);
      (void)SDL_RenderLine(sdl->renderer, (float)(x_2 - 3), (float)(y_2 + 3), (float)x_2, (float)y_2);
    } else {
      (void)SDL_RenderLine(sdl->renderer, (float)x_2, (float)y_2, (float)(x_2 + 3), (float)(y_2 - 3));
      (void)SDL_RenderLine(sdl->renderer, (float)x_2, (float)y_2, (float)(x_2 + 3), (float)(y_2 + 3));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawRectangle_SDL3(PetscDraw draw, PetscReal xl, PetscReal yl, PetscReal xr, PetscReal yr, int c1, int c2, int c3, int c4)
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;
  int             x = XTRANS(draw, sdl, xl), y = YTRANS(draw, sdl, yr), c = (c1 + c2 + c3 + c4) / 4;
  int             w = XTRANS(draw, sdl, xr) + 1 - x, h = YTRANS(draw, sdl, yl) + 1 - y;
  SDL_FRect       rect;

  PetscFunctionBegin;
  PetscDrawSDL3MarkDrawn(sdl);
  PetscDrawSDL3SetColor(sdl, c);
  if (w <= 0) w = 1;
  if (h <= 0) h = 1;
  rect.x = (float)x;
  rect.y = (float)y;
  rect.w = (float)w;
  rect.h = (float)h;
  (void)SDL_RenderFillRect(sdl->renderer, &rect);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawEllipse_SDL3(PetscDraw draw, PetscReal x, PetscReal y, PetscReal a, PetscReal b, int c)
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;
  int             xc = XTRANS(draw, sdl, x), rx = PetscAbs(XTRANS(draw, sdl, x + a / 2) - XTRANS(draw, sdl, x - a / 2)) / 2;
  int             yc = YTRANS(draw, sdl, y), ry = PetscAbs(YTRANS(draw, sdl, y - b / 2) - YTRANS(draw, sdl, y + b / 2)) / 2;
  int             idx[3 * 64];
  SDL_Vertex      v[66];

  PetscFunctionBegin;
  PetscDrawSDL3MarkDrawn(sdl);
  PetscDrawSDL3ValidColor(sdl, c);
  // draw a triangle fan approximating the filled ellipse
  v[0].position.x = (float)xc;
  v[0].position.y = (float)yc;
  v[0].color.r    = sdl->cpalette[c][0] / 255.0f;
  v[0].color.g    = sdl->cpalette[c][1] / 255.0f;
  v[0].color.b    = sdl->cpalette[c][2] / 255.0f;
  v[0].color.a    = 1.0f;
  for (int i = 0; i <= 64; i++) {
    PetscReal theta = 2 * PETSC_PI * (PetscReal)i / 64;

    v[i + 1].position.x = (float)(xc + rx * PetscCosReal(theta));
    v[i + 1].position.y = (float)(yc + ry * PetscSinReal(theta));
    v[i + 1].color      = v[0].color;
  }
  for (int i = 0; i < 64; i++) {
    idx[3 * i]     = 0;
    idx[3 * i + 1] = i + 1;
    idx[3 * i + 2] = i + 2;
  }
  (void)SDL_RenderGeometry(sdl->renderer, NULL, v, 66, idx, 3 * 64);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawTriangle_SDL3(PetscDraw draw, PetscReal X1, PetscReal Y_1, PetscReal X2, PetscReal Y2, PetscReal X3, PetscReal Y3, int c1, int c2, int c3)
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;
  SDL_Vertex      v[3];
  int             xv[3] = {XTRANS(draw, sdl, X1), XTRANS(draw, sdl, X2), XTRANS(draw, sdl, X3)};
  int             yv[3] = {YTRANS(draw, sdl, Y_1), YTRANS(draw, sdl, Y2), YTRANS(draw, sdl, Y3)};
  int             cv[3] = {c1, c2, c3};

  PetscFunctionBegin;
  PetscDrawSDL3MarkDrawn(sdl);
  PetscDrawSDL3ValidColor(sdl, c1);
  PetscDrawSDL3ValidColor(sdl, c2);
  PetscDrawSDL3ValidColor(sdl, c3);
  // interpolate the colormap index over the triangle, as PETSC_DRAW_X does, by sampling the palette texture
  for (int i = 0; i < 3; i++) {
    v[i].position.x  = (float)xv[i];
    v[i].position.y  = (float)yv[i];
    v[i].color.r     = 1.0f;
    v[i].color.g     = 1.0f;
    v[i].color.b     = 1.0f;
    v[i].color.a     = 1.0f;
    v[i].tex_coord.x = (cv[i] + 0.5f) / PETSC_DRAW_MAXCOLOR;
    v[i].tex_coord.y = 0.5f;
  }
  (void)SDL_RenderGeometry(sdl->renderer, sdl->palette, v, 3, NULL, 0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawStringSetSize_SDL3(PetscDraw draw, PetscReal x, PetscReal y)
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;
  int             w   = (int)(sdl->w * x * (draw->port_xr - draw->port_xl) / (draw->coor_xr - draw->coor_xl));
  int             h   = (int)(sdl->h * y * (draw->port_yr - draw->port_yl) / (draw->coor_yr - draw->coor_yl));

  PetscFunctionBegin;
  PetscCall(PetscDrawSDL3SetFont(draw, w, h));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawStringGetSize_SDL3(PetscDraw draw, PetscReal *x, PetscReal *y)
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;
  PetscReal       w = sdl->font_w, h = sdl->font_h;

  PetscFunctionBegin;
  if (x != NULL) *x = w * (draw->coor_xr - draw->coor_xl) / (sdl->w * (draw->port_xr - draw->port_xl));
  if (y != NULL) *y = h * (draw->coor_yr - draw->coor_yl) / (sdl->h * (draw->port_yr - draw->port_yl));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawString_SDL3(PetscDraw draw, PetscReal x, PetscReal y, int c, const char chrs[])
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;
  int             xx = XTRANS(draw, sdl, x), yy = YTRANS(draw, sdl, y);
  size_t          len;
  const char     *substr;
  PetscToken      token;
  TTF_Text       *text;

  PetscFunctionBegin;
  PetscDrawSDL3MarkDrawn(sdl);
  PetscCheck(sdl->font != NULL, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unable to open any font with SDL3_ttf; use -draw_sdl3_font filename to specify a font file");
  PetscCall(PetscTokenCreate(chrs, '\n', &token));
  PetscCall(PetscTokenFind(token, &substr));
  while (substr != NULL) {
    PetscCall(PetscStrlen(substr, &len));
    if (len) {
      text = TTF_CreateText(sdl->engine, sdl->font, substr, len);
      PetscCheck(text != NULL, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unable to create SDL3_ttf text: %s", SDL_GetError());
      PetscDrawSDL3ValidColor(sdl, c);
      (void)TTF_SetTextColor(text, sdl->cpalette[c][0], sdl->cpalette[c][1], sdl->cpalette[c][2], SDL_ALPHA_OPAQUE);
      (void)TTF_DrawRendererText(text, (float)xx, (float)(yy - sdl->font_h));
      TTF_DestroyText(text);
    }
    yy += sdl->font_h;
    PetscCall(PetscTokenFind(token, &substr));
  }
  PetscCall(PetscTokenDestroy(&token));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawStringVertical_SDL3(PetscDraw draw, PetscReal x, PetscReal y, int c, const char chrs[])
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;
  int             xx = XTRANS(draw, sdl, x), yy = YTRANS(draw, sdl, y);
  char            chr[2] = {0, 0};
  TTF_Text       *text;

  PetscFunctionBegin;
  PetscDrawSDL3MarkDrawn(sdl);
  PetscCheck(sdl->font != NULL, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unable to open any font with SDL3_ttf; use -draw_sdl3_font filename to specify a font file");
  PetscDrawSDL3ValidColor(sdl, c);
  while ((chr[0] = *chrs++)) {
    text = TTF_CreateText(sdl->engine, sdl->font, chr, 1);
    PetscCheck(text != NULL, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unable to create SDL3_ttf text: %s", SDL_GetError());
    (void)TTF_SetTextColor(text, sdl->cpalette[c][0], sdl->cpalette[c][1], sdl->cpalette[c][2], SDL_ALPHA_OPAQUE);
    (void)TTF_DrawRendererText(text, (float)xx, (float)yy);
    TTF_DestroyText(text);
    yy += sdl->font_h;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// recreate the surface of each process, and the display texture of the window, with a new size
static PetscErrorCode PetscDrawSDL3SetSize(PetscDraw draw, int w, int h)
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;

  PetscFunctionBegin;
  draw->w = w;
  draw->h = h;
  PetscCall(PetscDrawSDL3CloseSurface(sdl));
  PetscCall(PetscDrawSDL3OpenSurface(sdl, w, h));
  // PetscDrawSDL3Combine() allocates the combined image of the new size
  PetscCall(PetscFree(sdl->image));
  if (sdl->win != NULL) PetscCall(PetscDrawSDL3CreateDisplay(sdl));
  // reset the clipping
  PetscCall(PetscDrawSetViewport_SDL3(draw, draw->port_xl, draw->port_yl, draw->port_xr, draw->port_yr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// process pending window events on the first process, and give all processes the new window location and size
static PetscErrorCode PetscDrawCheckResizedWindow_SDL3(PetscDraw draw)
{
  PetscDraw_SDL3 *sdl     = (PetscDraw_SDL3 *)draw->data;
  int             xywh[4] = {0, 0, 0, 0};
  PetscMPIInt     rank;

  PetscFunctionBegin;
  if (sdl->isvirtual == PETSC_TRUE || sdl->shared == PETSC_TRUE) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)draw), &rank));

  if (rank == 0) {
    SDL_Event event;

    while (SDL_PollEvent(&event)) {
      // ignore all events except resizing of this window, queried below
    }
    (void)SDL_GetWindowPosition(sdl->win, &xywh[0], &xywh[1]);
    (void)SDL_GetWindowSize(sdl->win, &xywh[2], &xywh[3]);
  }
  PetscCallMPI(MPI_Bcast(xywh, 4, MPI_INT, 0, PetscObjectComm((PetscObject)draw)));

  // record new window position
  draw->x = sdl->x = xywh[0];
  draw->y = sdl->y = xywh[1];
  // a minimized window may report no size
  if ((xywh[2] == sdl->w && xywh[3] == sdl->h) || xywh[2] <= 0 || xywh[3] <= 0) PetscFunctionReturn(PETSC_SUCCESS);
  // the resize loses the drawing, as with PETSC_DRAW_X. The application redraws it.
  PetscCall(PetscDrawSDL3SetSize(draw, xywh[2], xywh[3]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawFlush_SDL3(PetscDraw draw)
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;
  unsigned int   *image;

  PetscFunctionBegin;
  // keep the window responsive. PetscDrawCheckResizedWindow() handles a resize before the next frame.
  if (sdl->win != NULL) {
    SDL_Event event;

    while (SDL_PollEvent(&event)) { }
  }
  // PetscDrawGetImage_SDL3() combines the surfaces of a draw without a window
  if (sdl->isvirtual == PETSC_TRUE) {
    (void)SDL_FlushRenderer(sdl->renderer);
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (sdl->shared == PETSC_TRUE) {
    // a singleton displays only the surface of its own process, in the window of its parent if this process owns it
    (void)SDL_FlushRenderer(sdl->renderer);
    image = sdl->pixels;
  } else PetscCall(PetscDrawSDL3Combine(draw, &image));
  if (sdl->win != NULL) {
    (void)SDL_UpdateTexture(sdl->display, NULL, image, sdl->w * (int)sizeof(*image));
    (void)SDL_RenderTexture(sdl->wrenderer, sdl->display, NULL, NULL);
    (void)SDL_RenderPresent(sdl->wrenderer);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawClear_SDL3(PetscDraw draw)
{
  PetscDraw_SDL3 *sdl  = (PetscDraw_SDL3 *)draw->data;
  int             xmax = sdl->w - 1, ymax = sdl->h - 1;
  int             xa = (int)(draw->port_xl * xmax), ya = ymax - (int)(draw->port_yr * ymax);
  int             xb = (int)(draw->port_xr * xmax), yb = ymax - (int)(draw->port_yl * ymax);
  SDL_FRect       rect = {(float)xa, (float)ya, (float)(xb + 1 - xa), (float)(yb + 1 - ya)};

  PetscFunctionBegin;
  // each process clears its own surface
  PetscCall(PetscDrawSDL3ClearRect(sdl, &rect));
  // the surface holds no drawing once all of it is cleared
  if (xa <= 0 && ya <= 0 && xb >= xmax && yb >= ymax) *sdl->pdrawn = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawGetPopup_SDL3(PetscDraw draw, PetscDraw *popup)
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;
  PetscBool       flg = PETSC_TRUE;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetBool(((PetscObject)draw)->options, ((PetscObject)draw)->prefix, "-draw_popup", &flg, NULL));
  if (flg == PETSC_FALSE || sdl->isvirtual == PETSC_TRUE) {
    *popup = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscDrawCreate(PetscObjectComm((PetscObject)draw), NULL, NULL, sdl->x, sdl->y + sdl->h + 10, 220, 220, popup));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*popup, "popup_"));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)*popup, ((PetscObject)draw)->prefix));
  PetscCall(PetscDrawSetType(*popup, PETSC_DRAW_SDL3));
  draw->popup = *popup;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawSetTitle_SDL3(PetscDraw draw, const char title[])
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;

  PetscFunctionBegin;
  if (sdl->win != NULL) (void)SDL_SetWindowTitle(sdl->win, title);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawResizeWindow_SDL3(PetscDraw draw, int w, int h)
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;
  PetscMPIInt     rank;

  PetscFunctionBegin;
  if (w == sdl->w && h == sdl->h) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)draw), &rank));

  if (sdl->isvirtual == PETSC_FALSE) {
    if (rank == 0) {
      (void)SDL_SetWindowSize(sdl->win, w, h);
      // the resize can be asynchronous. Wait until the new size takes effect.
      (void)SDL_SyncWindow(sdl->win);
    }
    // the first process gives the size the window actually has to all processes
    PetscCall(PetscDrawCheckResizedWindow_SDL3(draw));
  } else PetscCall(PetscDrawSDL3SetSize(draw, w, h));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawGetMouseButton_SDL3(PetscDraw draw, PetscDrawButton *button, PetscReal *x_user, PetscReal *y_user, PetscReal *x_phys, PetscReal *y_phys)
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;
  PetscMPIInt     rank;
  PetscReal       xx, yy;
  float           px = 0, py = 0;

  PetscFunctionBegin;
  *button = PETSC_BUTTON_NONE;
  // only a singleton on the process that owns the window of its parent receives mouse input
  if (sdl->isvirtual == PETSC_TRUE || (sdl->shared == PETSC_TRUE && sdl->win == NULL)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)draw), &rank));

  if (rank == 0) {
    SDL_Event event;
    PetscBool got = PETSC_FALSE;
    int       ww = sdl->w, wh = sdl->h;

    while (got == PETSC_FALSE) {
      (void)SDL_WaitEvent(&event);
      switch (event.type) {
      case SDL_EVENT_MOUSE_BUTTON_UP:
        if (event.button.windowID != SDL_GetWindowID(sdl->win)) continue;
        px = event.button.x;
        py = event.button.y;
        switch (event.button.button) {
        case SDL_BUTTON_LEFT:
          *button = PETSC_BUTTON_LEFT;
          break;
        case SDL_BUTTON_MIDDLE:
          *button = PETSC_BUTTON_CENTER;
          break;
        case SDL_BUTTON_RIGHT:
          *button = PETSC_BUTTON_RIGHT;
          break;
        default:
          continue;
        }
        got = PETSC_TRUE;
        break;
      case SDL_EVENT_MOUSE_WHEEL:
        if (event.wheel.windowID != SDL_GetWindowID(sdl->win) || event.wheel.y == 0) continue;
        *button = (event.wheel.y > 0) ? PETSC_BUTTON_WHEEL_UP : PETSC_BUTTON_WHEEL_DOWN;
        px      = event.wheel.mouse_x;
        py      = event.wheel.mouse_y;
        got     = PETSC_TRUE;
        break;
      case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
        // the user asked to close the window. Stop waiting for a mouse button.
        if (event.window.windowID != SDL_GetWindowID(sdl->win)) continue;
        got = PETSC_TRUE;
        break;
      case SDL_EVENT_QUIT:
        got = PETSC_TRUE;
        break;
      default:
        continue;
      }
    }
    if (SDL_GetModState() & SDL_KMOD_SHIFT) {
      switch (*button) {
      case PETSC_BUTTON_LEFT:
        *button = PETSC_BUTTON_LEFT_SHIFT;
        break;
      case PETSC_BUTTON_CENTER:
        *button = PETSC_BUTTON_CENTER_SHIFT;
        break;
      case PETSC_BUTTON_RIGHT:
        *button = PETSC_BUTTON_RIGHT_SHIFT;
        break;
      default:
        break;
      }
    }
    // the user may resize the window before pressing the mouse button
    (void)SDL_GetWindowSize(sdl->win, &ww, &wh);
    xx = ((PetscReal)px) / ww;
    yy = 1 - ((PetscReal)py) / wh;
    if (x_user != NULL) *x_user = draw->coor_xl + (xx - draw->port_xl) * (draw->coor_xr - draw->coor_xl) / (draw->port_xr - draw->port_xl);
    if (y_user != NULL) *y_user = draw->coor_yl + (yy - draw->port_yl) * (draw->coor_yr - draw->coor_yl) / (draw->port_yr - draw->port_yl);
    if (x_phys != NULL) *x_phys = xx;
    if (y_phys != NULL) *y_phys = yy;
  }
  PetscCall(PetscDrawCheckResizedWindow_SDL3(draw));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawSetVisible_SDL3(PetscDraw draw, PetscBool visible)
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;

  PetscFunctionBegin;
  if (sdl->win != NULL) {
    if (visible == PETSC_TRUE) (void)SDL_ShowWindow(sdl->win);
    else (void)SDL_HideWindow(sdl->win);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawPause_SDL3(PetscDraw draw)
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;

  PetscFunctionBegin;
  if (sdl->isvirtual == PETSC_TRUE || (sdl->shared == PETSC_TRUE && sdl->win == NULL)) PetscFunctionReturn(PETSC_SUCCESS);
  if (draw->pause > 0) PetscCall(PetscSleep(draw->pause));
  else if (draw->pause == -1) {
    PetscDrawButton button = PETSC_BUTTON_NONE;

    PetscCall(PetscDrawGetMouseButton(draw, &button, NULL, NULL, NULL, NULL));
    if (button == PETSC_BUTTON_CENTER || button == PETSC_BUTTON_NONE) draw->pause = 0; // NONE means a close request
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawDestroy_SDL3(PetscDraw draw)
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;

  PetscFunctionBegin;
  PetscCall(PetscDrawDestroy(&draw->popup));
  PetscCall(PetscDrawSDL3Close(sdl));
  PetscCall(PetscFree(draw->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawGetSingleton_SDL3(PetscDraw, PetscDraw *);
static PetscErrorCode PetscDrawRestoreSingleton_SDL3(PetscDraw, PetscDraw *);

static struct _PetscDrawOps DvOps = {NULL, PetscDrawFlush_SDL3, PetscDrawLine_SDL3, NULL, NULL, PetscDrawPoint_SDL3, NULL, PetscDrawString_SDL3, PetscDrawStringVertical_SDL3, PetscDrawStringSetSize_SDL3, PetscDrawStringGetSize_SDL3, PetscDrawSetViewport_SDL3, PetscDrawClear_SDL3, PetscDrawRectangle_SDL3, PetscDrawTriangle_SDL3, PetscDrawEllipse_SDL3, PetscDrawGetMouseButton_SDL3, PetscDrawPause_SDL3, NULL, NULL, PetscDrawGetPopup_SDL3, PetscDrawSetTitle_SDL3, PetscDrawCheckResizedWindow_SDL3, PetscDrawResizeWindow_SDL3, PetscDrawDestroy_SDL3, NULL, PetscDrawGetSingleton_SDL3, PetscDrawRestoreSingleton_SDL3, NULL, PetscDrawGetImage_SDL3, NULL, PetscDrawArrow_SDL3, PetscDrawCoordinateToPixel_SDL3, PetscDrawPixelToCoordinate_SDL3, PetscDrawPointPixel_SDL3, NULL, PetscDrawSetVisible_SDL3};

static PetscErrorCode PetscDrawGetSingleton_SDL3(PetscDraw draw, PetscDraw *sdraw)
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data, *ssdl;

  PetscFunctionBegin;
  PetscCall(PetscDrawCreate(PETSC_COMM_SELF, NULL, draw->title, draw->x, draw->y, draw->w, draw->h, sdraw));
  PetscCall(PetscObjectChangeTypeName((PetscObject)*sdraw, PETSC_DRAW_SDL3));
  (*sdraw)->ops[0] = DvOps;
  // the size of the surface is owned by the parent
  (*sdraw)->ops->resizewindow = NULL;

  if (draw->popup != NULL) PetscCall(PetscDrawGetSingleton(draw->popup, &(*sdraw)->popup));
  (*sdraw)->pause   = draw->pause;
  (*sdraw)->coor_xl = draw->coor_xl;
  (*sdraw)->coor_xr = draw->coor_xr;
  (*sdraw)->coor_yl = draw->coor_yl;
  (*sdraw)->coor_yr = draw->coor_yr;
  (*sdraw)->port_xl = draw->port_xl;
  (*sdraw)->port_xr = draw->port_xr;
  (*sdraw)->port_yl = draw->port_yl;
  (*sdraw)->port_yr = draw->port_yr;

  // draw on the surface of the parent draw on this process, with its renderer, font, and window if this process owns it
  PetscCall(PetscNew(&ssdl));
  (*sdraw)->data = (void *)ssdl;
  PetscCall(PetscMemcpy(ssdl->cpalette, sdl->cpalette, sizeof(ssdl->cpalette)));
  ssdl->shared       = PETSC_TRUE;
  ssdl->isvirtual    = sdl->isvirtual;
  ssdl->win          = sdl->win;
  ssdl->wrenderer    = sdl->wrenderer;
  ssdl->display      = sdl->display;
  ssdl->renderer     = sdl->renderer;
  ssdl->palette      = sdl->palette;
  ssdl->surface      = sdl->surface;
  ssdl->pixels       = sdl->pixels;
  ssdl->pdrawn       = sdl->pdrawn;
  ssdl->engine       = sdl->engine;
  ssdl->font         = sdl->font;
  ssdl->ownfont      = PETSC_FALSE;
  ssdl->font_w       = sdl->font_w;
  ssdl->font_h       = sdl->font_h;
  ssdl->font_descent = sdl->font_descent;
  ssdl->x            = sdl->x;
  ssdl->y            = sdl->y;
  ssdl->w            = sdl->w;
  ssdl->h            = sdl->h;
  (*sdraw)->x        = ssdl->x;
  (*sdraw)->y        = ssdl->y;
  (*sdraw)->w        = ssdl->w;
  (*sdraw)->h        = ssdl->h;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawRestoreSingleton_SDL3(PetscDraw draw, PetscDraw *sdraw)
{
  PetscFunctionBegin;
  if (draw->popup != NULL && (*sdraw)->popup != NULL) {
    PetscBool issdl3;

    PetscCall(PetscObjectTypeCompare((PetscObject)draw->popup, PETSC_DRAW_SDL3, &issdl3));
    if (issdl3 == PETSC_FALSE) goto finally;
    PetscCall(PetscObjectTypeCompare((PetscObject)(*sdraw)->popup, PETSC_DRAW_SDL3, &issdl3));
    if (issdl3 == PETSC_FALSE) goto finally;
    PetscCall(PetscDrawRestoreSingleton(draw->popup, &(*sdraw)->popup));
  }
finally:
  PetscCall(PetscDrawDestroy(sdraw));
  // restore the clip rectangle of the renderer that the singleton shared
  PetscCall(PetscDrawSetViewport_SDL3(draw, draw->port_xl, draw->port_yl, draw->port_xr, draw->port_yr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSC_DRAW_SDL3 - PETSc graphics device that uses SDL3 windows or an offscreen SDL3 surface

  Options Database Keys:
+ -draw_size w,h             - percentage of screen (either 1, .5, .3, .25), or size in pixels
. -geometry x,y,w,h          - set location and size in pixels
. -draw_virtual (true|false) - do not open a window (draw on an offscreen surface), `-draw_save` will still work
- -draw_sdl3_font filename   - font file to use for text, searched for on standard font paths if not provided

  Level: beginner

  Notes:
  `SDL3_ttf` must have access to some font otherwise a runtime error is generated. One can use `-draw_sdl3_font filename` in that
  case to point to some available font.

  Each process draws on its own offscreen surface. `PetscDrawFlush()` combines the surfaces of all processes with
  `MPI_Reduce()`, and the first process displays the combined image in its window. Where several processes draw the
  same pixel, the combined image contains one of their colors. `PetscDrawSave()`, and thus `-draw_save`, also
  uses the combined image.

  A draw from `PetscDrawGetSingleton()` draws on the surface of its process. On the first process, `PetscDrawFlush()`
  of that draw displays only this surface in the window of the parent draw. The next `PetscDrawFlush()` of the parent
  draw again displays the combined image of all processes.

  Double buffering is always enabled for `PETSC_DRAW_SDL3` so `-draw_double_buffer 0` is ignored.

.seealso: `PetscDraw`, `PetscDrawOpenSDL3()`, `PetscDrawSetFromOptions()`
M*/

PETSC_EXTERN PetscErrorCode PetscDrawCreate_SDL3(PetscDraw draw)
{
  PetscDraw_SDL3 *sdl;
  PetscMPIInt     rank;
  int             x, y, w, h, sw, sh;
  static int      xmax = 0, ymax = 0;
  PetscBool       dvirtual = PETSC_FALSE, has_display = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetBool(((PetscObject)draw)->options, ((PetscObject)draw)->prefix, "-draw_virtual", &dvirtual, NULL));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)draw), &rank));

  // initialize the display size
  if (dvirtual == PETSC_FALSE) {
    if (rank == 0 && xmax) has_display = PETSC_TRUE;
    else if (rank == 0) {
      PetscCall(PetscDrawSDL3VideoAcquire(&has_display));
      if (has_display == PETSC_TRUE) {
        PetscCall(PetscDrawSDL3GetDisplaySize(&xmax, &ymax));
        // the window takes its own video reference in PetscDrawSDL3OpenWindow()
        PetscCall(PetscDrawSDL3VideoRelease());
      }
    }
    PetscCallMPI(MPI_Bcast(&has_display, 1, MPI_C_BOOL, 0, PetscObjectComm((PetscObject)draw)));
    if (has_display == PETSC_FALSE) {
      PetscCall((*PetscErrorPrintf)("PETSc unable to use SDL3 windows\nproceeding without graphics\n"));
      PetscCall(PetscDrawSetType(draw, PETSC_DRAW_NULL));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  // all processes use the display size of the first process, because PetscDrawSDL3Combine() needs equal surface sizes
  sw = xmax;
  sh = ymax;
  PetscCallMPI(MPI_Bcast(&sw, 1, MPI_INT, 0, PetscObjectComm((PetscObject)draw)));
  PetscCallMPI(MPI_Bcast(&sh, 1, MPI_INT, 0, PetscObjectComm((PetscObject)draw)));
  // an offscreen draw may have no display, so size fractions use the fallback of PetscDrawSDL3GetDisplaySize()
  if (sw == 0) sw = 1024;
  if (sh == 0) sh = 768;

  PetscCall(PetscDrawSetUpWindowGeometry_Private(draw, sw, sh, dvirtual, &x, &y, &w, &h));

  // every process draws on a surface of the window size
  PetscCheck(w > 0 && h > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Negative window width or height");
  // PetscDrawSDL3Combine() reduces the ARGB8888 pixel values as MPI_UNSIGNED
  PetscCheck(sizeof(unsigned int) == 4, PETSC_COMM_SELF, PETSC_ERR_SUP, "PETSC_DRAW_SDL3 requires a 4 byte unsigned int");

  PetscCall(PetscNew(&sdl));
  draw->ops[0]   = DvOps;
  draw->data     = (void *)sdl;
  draw->x        = x;
  draw->y        = y;
  draw->w        = w;
  draw->h        = h;
  sdl->x         = x;
  sdl->y         = y;
  sdl->w         = w;
  sdl->h         = h;
  sdl->isvirtual = dvirtual;
  sdl->pdrawn    = &sdl->drawn;

  PetscCall(PetscDrawSDL3SetUpColormap(sdl));
  // balanced by TTF_Quit() in PetscDrawSDL3Close()
  PetscCheck(TTF_Init(), PETSC_COMM_SELF, PETSC_ERR_LIB, "Unable to initialize SDL3_ttf: %s", SDL_GetError());
  PetscCall(PetscDrawSDL3OpenSurface(sdl, w, h));
  PetscCall(PetscDrawSDL3SetFont(draw, 6, 10));
  if (rank == 0 && dvirtual == PETSC_FALSE) PetscCall(PetscDrawSDL3OpenWindow(sdl, draw->title, x, y, w, h));
  PetscFunctionReturn(PETSC_SUCCESS);
}
