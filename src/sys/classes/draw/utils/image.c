#include <petsc/private/petscimpl.h> /*I "petscsys.h" I*/

PETSC_EXTERN PetscErrorCode PetscDrawImageSave(const char[], const char[], unsigned char[][3], unsigned int, unsigned int, const unsigned char[]);
PETSC_EXTERN PetscErrorCode PetscDrawMovieSave(const char[], PetscInt, const char[], PetscInt, const char[]);
PETSC_EXTERN PetscErrorCode PetscDrawImageCheckFormat(const char *[]);
PETSC_EXTERN PetscErrorCode PetscDrawMovieCheckFormat(const char *[]);

/*
   Code to write images in PPM format
*/
PETSC_EXTERN PetscErrorCode PetscDrawImageSavePPM(const char filename[], unsigned char palette[][3], unsigned int w, unsigned int h, const unsigned char pixels[])
{
  int            fd;
  char           header[32];
  size_t         hdrlen;
  unsigned char *rgb;

  PetscFunctionBegin;
  PetscValidCharPointer(filename, 1);
  if (palette) PetscValidPointer(palette, 2);
  PetscValidCharPointer(pixels, 5);
  /* map pixels to RGB colors */
  if (palette) {
    int                  k, p, n = (int)(w * h);
    const unsigned char *colordef;
    PetscCall(PetscMalloc1(3 * w * h, &rgb));
    for (k = p = 0; k < n; k++) {
      colordef = palette[pixels[k]];
      rgb[p++] = colordef[0];
      rgb[p++] = colordef[1];
      rgb[p++] = colordef[2];
    }
  } else { /* assume pixels are RGB colors */
    rgb = (unsigned char *)pixels;
  }
  /* open file and write PPM header */
  PetscCall(PetscBinaryOpen(filename, FILE_MODE_WRITE, &fd));
  PetscCall(PetscSNPrintf(header, sizeof(header), "P6\n%d %d\n255\n%c", (int)w, (int)h, '\0'));
  PetscCall(PetscStrlen(header, &hdrlen));
  PetscCall(PetscBinaryWrite(fd, header, hdrlen, PETSC_CHAR));
  /* write image data and close file */
  PetscCall(PetscBinaryWrite(fd, rgb, 3 * w * h, PETSC_CHAR));
  PetscCall(PetscBinaryClose(fd));
  if (palette) PetscCall(PetscFree(rgb));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawImageSave_PPM(const char filename[], unsigned char palette[][3], unsigned int w, unsigned int h, const unsigned char pixels[])
{
  return PetscDrawImageSavePPM(filename, palette, w, h, pixels);
}

/*
   Code to write images in PNG format
*/
#if defined(PETSC_HAVE_LIBPNG)

  #include <png.h>

  #if defined(PNG_SETJMP_SUPPORTED)
    #ifndef png_jmpbuf
      #define png_jmpbuf(png_ptr) ((png_ptr)->jmpbuf)
    #endif
  #endif

PETSC_EXTERN PetscErrorCode PetscDrawImageSavePNG(const char filename[], unsigned char palette[][3], unsigned int w, unsigned int h, const unsigned char pixels[])
{
  FILE        *fp;
  png_struct  *png_ptr;
  png_info    *info_ptr;
  unsigned int row, stride = palette ? w : 3 * w;

  PetscFunctionBegin;
  PetscValidCharPointer(filename, 1);
  if (palette) PetscValidCharPointer(palette, 2);
  PetscValidCharPointer(pixels, 5);

  /* open file and create libpng structures */
  PetscCall(PetscFOpen(PETSC_COMM_SELF, filename, "wb", &fp));
  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  PetscCheck(png_ptr, PETSC_COMM_SELF, PETSC_ERR_LIB, "Cannot create PNG context");
  info_ptr = png_create_info_struct(png_ptr);
  PetscCheck(info_ptr, PETSC_COMM_SELF, PETSC_ERR_LIB, "Cannot create PNG context");

  /* setup libpng error handling */
  #if defined(PNG_SETJMP_SUPPORTED)
  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_write_struct(&png_ptr, &info_ptr);
    (void)PetscFClose(PETSC_COMM_SELF, fp);
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error writing PNG file %s", filename);
  }
  #endif

  /* setup PNG image metadata */
  png_init_io(png_ptr, fp);
  png_set_IHDR(png_ptr, info_ptr, w, h, /*depth*/ 8, palette ? PNG_COLOR_TYPE_PALETTE : PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
  if (palette) png_set_PLTE(png_ptr, info_ptr, (png_color *)palette, 256);

  /* write PNG image header and data */
  png_write_info(png_ptr, info_ptr);
  for (row = 0; row < h; row++) png_write_row(png_ptr, pixels + row * stride);
  png_write_end(png_ptr, NULL);

  /* destroy libpng structures and close file */
  png_destroy_write_struct(&png_ptr, &info_ptr);
  PetscCall(PetscFClose(PETSC_COMM_SELF, fp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawImageSave_PNG(const char filename[], unsigned char palette[][3], unsigned int w, unsigned int h, const unsigned char pixels[])
{
  return PetscDrawImageSavePNG(filename, palette, w, h, pixels);
}

#endif /*!PETSC_HAVE_LIBPNG*/

/*
   Code to write images in GIF format
*/
#if defined(PETSC_HAVE_GIFLIB)

  #include <gif_lib.h>

  #if !defined(GIFLIB_MAJOR) || GIFLIB_MAJOR < 5
    #define GifMakeMapObject            MakeMapObject
    #define GifFreeMapObject            FreeMapObject
    #define EGifOpenFileName(n, b, err) EGifOpenFileName(n, b)
    #define EGifOpenFileHandle(h, err)  EGifOpenFileName(h)
    #define EGifCloseFile(f, err)       EGifCloseFile(f)
    #define DGifOpenFileName(n, err)    DGifOpenFileName(n)
    #define DGifOpenFileHandle(h, err)  DGifOpenFileName(h)
    #define DGifCloseFile(f, err)       DGifCloseFile(f)
  #endif

PETSC_EXTERN PetscErrorCode PetscDrawImageSaveGIF(const char filename[], unsigned char palette[][3], unsigned int w, unsigned int h, const unsigned char pixels[])
{
  int             Row;
  int             Width      = (int)w;
  int             Height     = (int)h;
  int             ColorRes   = 8;
  int             ColorCount = 256;
  ColorMapObject *GifCMap    = NULL;
  GifFileType    *GifFile    = NULL;
  #define SETERRGIF(msg) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, msg ", GIF file: %s", filename)
  #define PetscCallGIF(msg, ...) \
    do { \
      int Error = __VA_ARGS__; \
      if (PetscUnlikely(Error != GIF_OK)) SETERRGIF(msg); \
    } while (0)

  PetscFunctionBegin;
  PetscValidCharPointer(filename, 1);
  PetscValidCharPointer(palette, 2);
  PetscValidCharPointer(pixels, 5);

  GifCMap = GifMakeMapObject(ColorCount, (GifColorType *)palette);
  if (!GifCMap) SETERRGIF("Allocating colormap");
  GifFile = EGifOpenFileName(filename, 0, NULL);
  if (!GifFile) SETERRGIF("Opening");
  PetscCallGIF("Writing screen descriptor", EGifPutScreenDesc(GifFile, Width, Height, ColorRes, 0, GifCMap));
  PetscCallGIF("Writing image descriptor", EGifPutImageDesc(GifFile, 0, 0, Width, Height, 0, NULL));
  for (Row = 0; Row < Height; Row++) PetscCallGIF("Writing image pixels", EGifPutLine(GifFile, (GifPixelType *)pixels + Row * Width, Width));
  PetscCallGIF("Closing", EGifCloseFile(GifFile, NULL));
  GifFreeMapObject(GifCMap);
  GifCMap = NULL;

  #undef SETERRGIF
  #undef CHKERRGIF
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawImageSave_GIF(const char filename[], unsigned char palette[][3], unsigned int w, unsigned int h, const unsigned char pixels[])
{
  return PetscDrawImageSaveGIF(filename, palette, w, h, pixels);
}

PETSC_EXTERN PetscErrorCode PetscDrawMovieSaveGIF(const char pattern[], PetscInt count, const char movie[])
{
  int          i, j, Row;
  char         image[PETSC_MAX_PATH_LEN];
  GifFileType *GifMovie = NULL;
  GifFileType *GifImage = NULL;
  #define SETERRGIF(msg, fn) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, msg " GIF file %s", fn)

  PetscFunctionBegin;
  PetscValidCharPointer(pattern, 1);
  PetscValidCharPointer(movie, 3);
  if (count < 1) PetscFunctionReturn(PETSC_SUCCESS);

  for (i = 0; i < count; i++) {
    PetscCall(PetscSNPrintf(image, sizeof(image), pattern, (int)i));
    /* open and read image file */
    if ((GifImage = DGifOpenFileName(image, NULL)) == NULL) SETERRGIF("Opening input", image);
    if (DGifSlurp(GifImage) != GIF_OK) SETERRGIF("Reading input", image);
    /* open movie file and write header */
    if (i == 0) {
      if ((GifMovie = EGifOpenFileName(movie, 0, NULL)) == NULL) SETERRGIF("Opening output", movie);
      if (EGifPutScreenDesc(GifMovie, GifImage->SWidth, GifImage->SHeight, GifImage->SColorResolution, GifImage->SBackGroundColor, GifImage->SColorMap) != GIF_OK) SETERRGIF("Writing screen descriptor,", movie);
    }
    /* loop over all frames in image */
    for (j = 0; j < GifImage->ImageCount; j++) {
      SavedImage     *sp            = &GifImage->SavedImages[j];
      GifImageDesc   *GifFrame      = &sp->ImageDesc;
      ColorMapObject *FrameColorMap = GifFrame->ColorMap ? GifFrame->ColorMap : GifImage->SColorMap;
      if (GifMovie->SColorMap && GifMovie->SColorMap->ColorCount == FrameColorMap->ColorCount && !memcmp(GifMovie->SColorMap->Colors, FrameColorMap->Colors, (size_t)FrameColorMap->ColorCount * sizeof(GifColorType))) FrameColorMap = NULL;
      /* add frame to movie */
      if (EGifPutImageDesc(GifMovie, GifFrame->Left, GifFrame->Top, GifFrame->Width, GifFrame->Height, GifFrame->Interlace, FrameColorMap) != GIF_OK) SETERRGIF("Writing image descriptor,", movie);
      for (Row = 0; Row < GifFrame->Height; Row++) {
        if (EGifPutLine(GifMovie, sp->RasterBits + Row * GifFrame->Width, GifFrame->Width) != GIF_OK) SETERRGIF("Writing image pixels,", movie);
      }
    }
    if (DGifCloseFile(GifImage, NULL) != GIF_OK) SETERRGIF("Closing input", image);
  }
  if (EGifCloseFile(GifMovie, NULL) != GIF_OK) SETERRGIF("Closing output", movie);

  #undef SETERRGIF
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif /*!PETSC_HAVE_GIFLIB*/

/*
   Code to write images in JPEG format
*/
#if defined(PETSC_HAVE_LIBJPEG)

  #include <jpeglib.h>

  #if defined(PETSC_HAVE_SETJMP_H)
    #include <setjmp.h>
static jmp_buf petsc_jpeg_jumpbuf;
static void    petsc_jpeg_error_longjmp(j_common_ptr cinfo)
{
  (void)cinfo;
  longjmp(petsc_jpeg_jumpbuf, 1);
}
  #endif

PETSC_EXTERN PetscErrorCode PetscDrawImageSaveJPG(const char filename[], unsigned char palette[][3], unsigned int w, unsigned int h, const unsigned char pixels[])
{
  unsigned char              *rgbpixels;
  FILE                       *fp;
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr       jerr;

  PetscFunctionBegin;
  PetscValidCharPointer(filename, 1);
  if (palette) PetscValidCharPointer(palette, 2);
  PetscValidCharPointer(pixels, 5);
  /* map pixels to RGB colors */
  if (palette) {
    int                  k, p, n = (int)(w * h);
    const unsigned char *colordef;
    PetscCall(PetscMalloc1(3 * w * h, &rgbpixels));
    for (k = p = 0; k < n; k++) {
      colordef       = palette[pixels[k]];
      rgbpixels[p++] = colordef[0];
      rgbpixels[p++] = colordef[1];
      rgbpixels[p++] = colordef[2];
    }
  } else { /* assume pixels are RGB colors */
    rgbpixels = (unsigned char *)pixels;
  }
  PetscCall(PetscFOpen(PETSC_COMM_SELF, filename, "wb", &fp));

  cinfo.err = jpeg_std_error(&jerr);
  #if defined(PETSC_HAVE_SETJMP_H)
  jerr.error_exit = petsc_jpeg_error_longjmp;
  if (setjmp(petsc_jpeg_jumpbuf)) {
    char message[JMSG_LENGTH_MAX];
    jerr.format_message((j_common_ptr)&cinfo, message);
    jpeg_destroy_compress(&cinfo);
    (void)PetscFClose(PETSC_COMM_SELF, fp);
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error writing JPEG file %s\n%s", filename, message);
  }
  #endif
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, fp);
  cinfo.image_width      = w;
  cinfo.image_height     = h;
  cinfo.input_components = 3;
  cinfo.in_color_space   = JCS_RGB;
  jpeg_set_defaults(&cinfo);
  jpeg_start_compress(&cinfo, TRUE);
  while (cinfo.next_scanline < cinfo.image_height) {
    unsigned char *rowptr = rgbpixels + cinfo.next_scanline * 3 * w;
    (void)jpeg_write_scanlines(&cinfo, &rowptr, 1);
  }
  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);

  PetscCall(PetscFClose(PETSC_COMM_SELF, fp));
  if (palette) PetscCall(PetscFree(rgbpixels));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawImageSave_JPG(const char filename[], unsigned char palette[][3], unsigned int w, unsigned int h, const unsigned char pixels[])
{
  return PetscDrawImageSaveJPG(filename, palette, w, h, pixels);
}

#endif /*!PETSC_HAVE_LIBJPEG*/

static struct {
  const char *extension;
  PetscErrorCode (*SaveImage)(const char[], unsigned char[][3], unsigned int, unsigned int, const unsigned char[]);
} PetscDrawImageSaveTable[] = {
#if defined(PETSC_HAVE_LIBPNG)
  {".png", PetscDrawImageSave_PNG},
#endif
#if defined(PETSC_HAVE_GIFLIB)
  {".gif", PetscDrawImageSave_GIF},
#endif
#if defined(PETSC_HAVE_LIBJPEG)
  {".jpg", PetscDrawImageSave_JPG},
#endif
  {".ppm", PetscDrawImageSave_PPM}
};

PetscErrorCode PetscDrawImageCheckFormat(const char *ext[])
{
  size_t    k;
  PetscBool match = PETSC_FALSE;

  PetscFunctionBegin;
  /* if extension is empty, return default format to caller */
  PetscValidPointer(ext, 1);
  if (!*ext || !**ext) {
    *ext = PetscDrawImageSaveTable[0].extension;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* check the extension matches a supported format */
  PetscValidCharPointer(*ext, 1);
  for (k = 0; k < PETSC_STATIC_ARRAY_LENGTH(PetscDrawImageSaveTable); k++) {
    PetscCall(PetscStrcasecmp(*ext, PetscDrawImageSaveTable[k].extension, &match));
    if (match && PetscDrawImageSaveTable[k].SaveImage) PetscFunctionReturn(PETSC_SUCCESS);
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Image extension %s not supported, use .ppm or see PetscDrawSetSave() for what ./configure option you may need", *ext);
}

PetscErrorCode PetscDrawImageSave(const char basename[], const char ext[], unsigned char palette[][3], unsigned int w, unsigned int h, const unsigned char pixels[])
{
  size_t    k;
  PetscBool match = PETSC_FALSE;
  char      filename[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscValidCharPointer(basename, 1);
  if (ext) PetscValidCharPointer(ext, 2);
  if (palette) PetscValidPointer(palette, 3);
  PetscValidCharPointer(pixels, 6);

  PetscCall(PetscDrawImageCheckFormat(&ext));
  PetscCall(PetscSNPrintf(filename, sizeof(filename), "%s%s", basename, ext));
  for (k = 0; k < PETSC_STATIC_ARRAY_LENGTH(PetscDrawImageSaveTable); k++) {
    PetscCall(PetscStrcasecmp(ext, PetscDrawImageSaveTable[k].extension, &match));
    if (match && PetscDrawImageSaveTable[k].SaveImage) {
      PetscCall(PetscDrawImageSaveTable[k].SaveImage(filename, palette, w, h, pixels));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Image extension %s not supported, use .ppm", ext);
}

PetscErrorCode PetscDrawMovieCheckFormat(const char *ext[])
{
  PetscFunctionBegin;
  PetscValidPointer(ext, 1);
  if (!*ext || !**ext) *ext = ".m4v";
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDrawMovieSave(const char basename[], PetscInt count, const char imext[], PetscInt fps, const char mvext[])
{
  char      input[PETSC_MAX_PATH_LEN];
  char      output[PETSC_MAX_PATH_LEN];
  PetscBool gifinput;

  PetscFunctionBegin;
  PetscValidCharPointer(basename, 1);
  PetscValidCharPointer(imext, 3);
  if (mvext) PetscValidCharPointer(mvext, 5);
  if (count < 1) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscStrcasecmp(imext, ".gif", &gifinput));
  PetscCall(PetscDrawMovieCheckFormat(&mvext));
  PetscCall(PetscSNPrintf(input, sizeof(input), "%s/%s_%%d%s", basename, basename, imext));
  PetscCall(PetscSNPrintf(output, sizeof(output), "%s%s", basename, mvext));

  /* use GIFLIB to generate an intermediate GIF animation */
#if defined(PETSC_HAVE_GIFLIB)
  if (gifinput) {
    char gifmovie[PETSC_MAX_PATH_LEN];
    PetscCall(PetscSNPrintf(gifmovie, sizeof(gifmovie), "%s/%s_movie.gif", basename, basename));
    PetscCall(PetscDrawMovieSaveGIF(input, count, gifmovie));
    PetscCall(PetscStrncpy(input, gifmovie, sizeof(input)));
  }
#endif

  /* use FFmpeg to generate a movie */
#if defined(PETSC_HAVE_POPEN)
  {
    FILE *fd;
    char  options[64] = "-loglevel error -y", extraopts[32] = "", framerate[24] = "";
    char  command[sizeof(options) + sizeof(extraopts) + sizeof(framerate) + PETSC_MAX_PATH_LEN * 2];
    if (fps > 0) PetscCall(PetscSNPrintf(framerate, sizeof(framerate), "-r %d", (int)fps));
    if (gifinput) {
      PetscCall(PetscStrlcat(options, " -f gif", sizeof(options)));
      PetscCall(PetscSNPrintf(extraopts, sizeof(extraopts), " -default_delay %d", (fps > 0) ? 100 / (int)fps : 4));
    } else {
      PetscCall(PetscStrlcat(options, " -f image2", sizeof(options)));
      if (fps > 0) PetscCall(PetscSNPrintf(extraopts, sizeof(extraopts), " -framerate %d", (int)fps));
    }
    if (extraopts[0]) PetscCall(PetscStrlcat(options, extraopts, sizeof(options)));
    PetscCall(PetscSNPrintf(command, sizeof(command), "ffmpeg %s -i \"%s\" %s \"%s\"", options, input, framerate, output));
    PetscCall(PetscPOpen(PETSC_COMM_SELF, NULL, command, "r", &fd));
    PetscCall(PetscPClose(PETSC_COMM_SELF, fd));
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
