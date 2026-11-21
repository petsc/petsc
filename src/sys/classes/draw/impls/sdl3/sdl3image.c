// Code for combining the images of all processes and getting raster images out of them

#include <../src/sys/classes/draw/impls/sdl3/sdl3impl.h>

// Map an RGB pixel value to the nearest PETSc color value (index in the colormap)
static int PetscDrawSDL3MapColor(PetscDraw_SDL3 *sdl, unsigned char r, unsigned char g, unsigned char b)
{
  int best = 0, bestdist = 3 * 255 * 255 + 1;

  for (int i = 0; i < PETSC_DRAW_MAXCOLOR; i++) {
    int dr   = (int)r - sdl->cpalette[i][0];
    int dg   = (int)g - sdl->cpalette[i][1];
    int db   = (int)b - sdl->cpalette[i][2];
    int dist = dr * dr + dg * dg + db * db;

    if (dist < bestdist) {
      best     = i;
      bestdist = dist;
      if (!dist) break;
    }
  }
  return best;
}

/*
  PetscDrawSDL3Combine - combine the surfaces of all processes into one image on the first process

  Collective

  Output Parameter:
. image - on the first process, the combined w*h ARGB8888 pixel values, owned by the draw and valid until
          the next change of the draw size. `NULL` on the other processes.

  Notes:
  Cleared pixels have zero alpha and drawn pixels nonzero alpha, so the MPI_MAX reduction keeps, for each pixel,
  a value drawn by some process. Their RGB values are already blended against the white background.

  When only the first process drew since the other surfaces were last cleared completely, its surface is the
  combined image and the reduction is skipped.
*/
PetscErrorCode PetscDrawSDL3Combine(PetscDraw draw, unsigned int **image)
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;
  MPI_Comm        comm;
  PetscMPIInt     rank, size, n;
  PetscBool       others;

  PetscFunctionBegin;
  *image = NULL;
  PetscCall(PetscObjectGetComm((PetscObject)draw, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  // complete the pending drawing commands before reading the surface
  (void)SDL_FlushRenderer(sdl->renderer);
  if (size == 1) {
    *image = sdl->pixels;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  // determine if a process other than the first one has drawing to contribute
  others = (PetscBool)(rank != 0 && *sdl->pdrawn == PETSC_TRUE);
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &others, 1, MPI_C_BOOL, MPI_LOR, comm));
  if (others == PETSC_FALSE) {
    if (rank == 0) *image = sdl->pixels;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscMPIIntCast((PetscCount)sdl->w * sdl->h, &n));
  if (rank == 0) {
    if (sdl->image == NULL) PetscCall(PetscMalloc1(n, &sdl->image));
    *image = sdl->image;
  }
  PetscCallMPI(MPI_Reduce(sdl->pixels, *image, n, MPI_UNSIGNED, MPI_MAX, 0, comm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDrawGetImage_SDL3(PetscDraw draw, unsigned char palette[PETSC_DRAW_MAXCOLOR][3], unsigned int *out_w, unsigned int *out_h, unsigned char *out_pixels[])
{
  PetscDraw_SDL3 *sdl = (PetscDraw_SDL3 *)draw->data;
  unsigned int   *image;

  PetscFunctionBegin;
  if (out_w != NULL) *out_w = 0;
  if (out_h != NULL) *out_h = 0;
  if (out_pixels != NULL) *out_pixels = NULL;
  PetscCall(PetscDrawSDL3Combine(draw, &image));

  // only the first process returns image data
  if (image != NULL) {
    PetscCount     n    = (PetscCount)sdl->w * sdl->h;
    unsigned int   last = 0xFFFFFFFFu; // not a masked RGB value
    unsigned char *pixels;
    unsigned char  color = 0;

    // copy colormap palette to the caller
    PetscCall(PetscMemcpy(palette, sdl->cpalette, sizeof(sdl->cpalette)));
    // map the RGB values of the pixels to color indices. Neighboring pixels usually have the same value.
    PetscCall(PetscMalloc1(n, &pixels));
    for (PetscCount p = 0; p < n; p++) {
      unsigned int rgb = image[p] & 0xFFFFFFu;

      if (rgb != last) {
        last  = rgb;
        color = (unsigned char)PetscDrawSDL3MapColor(sdl, (unsigned char)(rgb >> 16), (unsigned char)(rgb >> 8), (unsigned char)rgb);
      }
      pixels[p] = color;
    }
    if (out_w != NULL) *out_w = (unsigned int)sdl->w;
    if (out_h != NULL) *out_h = (unsigned int)sdl->h;
    if (out_pixels != NULL) *out_pixels = pixels;
    else PetscCall(PetscFree(pixels));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
