#include <petsc/private/petscimpl.h>         /*I "petscsys.h" I*/

PETSC_EXTERN PetscErrorCode PetscDrawImageSave(const char[],const char[],unsigned char[][3],unsigned int,unsigned int,const unsigned char[]);
PETSC_EXTERN PetscErrorCode PetscDrawMovieSave(const char[],PetscInt,const char[],PetscInt,const char[]);
PETSC_EXTERN PetscErrorCode PetscDrawImageCheckFormat(const char *[]);
PETSC_EXTERN PetscErrorCode PetscDrawMovieCheckFormat(const char *[]);

/*
   Code to write images in PPM format
*/
#undef __FUNCT__
#define __FUNCT__ "PetscDrawImageSavePPM"
PETSC_EXTERN PetscErrorCode PetscDrawImageSavePPM(const char filename[],unsigned char palette[][3],unsigned int w,unsigned int h,const unsigned char pixels[])
{
  int            fd;
  char           header[32];
  size_t         hdrlen;
  unsigned char  *rgb;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(filename,1);
  if (palette) PetscValidCharPointer(palette,2);
  PetscValidCharPointer(pixels,5);
  /* map pixels to RGB colors */
  if (palette) {
    int k,p,n = (int)(w*h);
    const unsigned char *colordef;
    ierr = PetscMalloc1(3*w*h,&rgb);CHKERRQ(ierr);
    for (k=p=0; k<n; k++) {
      colordef = palette[pixels[k]];
      rgb[p++] = colordef[0];
      rgb[p++] = colordef[1];
      rgb[p++] = colordef[2];
    }
  } else { /* assume pixels are RGB colors */
    rgb = (unsigned char*)pixels;
  }
  /* open file and write PPM header */
  ierr = PetscBinaryOpen(filename,FILE_MODE_WRITE,&fd);CHKERRQ(ierr);
  ierr = PetscSNPrintf(header,sizeof(header),"P6\n%d %d\n255\n\0",(int)w,(int)h);CHKERRQ(ierr);
  ierr = PetscStrlen(header,&hdrlen);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(fd,header,hdrlen,PETSC_CHAR,PETSC_FALSE);CHKERRQ(ierr);
  /* write image data and close file */
  ierr = PetscBinaryWrite(fd,rgb,3*w*h,PETSC_CHAR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscBinaryClose(fd);CHKERRQ(ierr);
  if (palette) {ierr = PetscFree(rgb);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawImageSave_PPM(const char filename[],unsigned char palette[][3],unsigned int w,unsigned int h,const unsigned char pixels[])
{ return PetscDrawImageSavePPM(filename,palette,w,h,pixels); }


/*
   Code to write images in PNG format
*/
#if defined(PETSC_HAVE_LIBPNG)

#include <png.h>

#if defined(PNG_SETJMP_SUPPORTED)
# ifndef png_jmpbuf
#   define png_jmpbuf(png_ptr) ((png_ptr)->jmpbuf)
# endif
#endif

#undef __FUNCT__
#define __FUNCT__ "PetscDrawImageSavePNG"
PETSC_EXTERN PetscErrorCode PetscDrawImageSavePNG(const char filename[],unsigned char palette[][3],unsigned int w,unsigned int h,const unsigned char pixels[])
{
  FILE           *fp;
  png_struct     *png_ptr;
  png_info       *info_ptr;
  unsigned int   row, stride = palette ? w : 3*w;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(filename,1);
  if (palette) PetscValidCharPointer(palette,2);
  PetscValidCharPointer(pixels,5);

  /* open file and create libpng structures */
  ierr = PetscFOpen(PETSC_COMM_SELF,filename,"wb",&fp);CHKERRQ(ierr);
  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING,NULL,NULL,NULL);
  if (!png_ptr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Cannot create PNG context");
  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Cannot create PNG context");

  /* setup libpng error handling */
#if defined(PNG_SETJMP_SUPPORTED)
  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_write_struct(&png_ptr,&info_ptr);
    (void)PetscFClose(PETSC_COMM_SELF,fp);
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error writing PNG file %s",filename);
  }
#endif

  /* setup PNG image metadata */
  png_init_io(png_ptr, fp);
  png_set_IHDR(png_ptr, info_ptr, w, h, /*depth*/8,
               palette ? PNG_COLOR_TYPE_PALETTE : PNG_COLOR_TYPE_RGB,
               PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  if (palette)
    png_set_PLTE(png_ptr, info_ptr, (png_color*)palette, 256);

  /* write PNG image header and data */
  png_write_info(png_ptr, info_ptr);
  for (row = 0; row < h; row++)
    png_write_row(png_ptr, pixels + row*stride);
  png_write_end(png_ptr, NULL);

  /* destroy libpng structures and close file */
  png_destroy_write_struct(&png_ptr, &info_ptr);
  ierr = PetscFClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawImageSave_PNG(const char filename[],unsigned char palette[][3],unsigned int w,unsigned int h,const unsigned char pixels[])
{ return PetscDrawImageSavePNG(filename,palette,w,h,pixels); }

#endif/*!PETSC_HAVE_LIBPNG*/


/*
   Code to write images in GIF format
*/
#if defined(PETSC_HAVE_GIFLIB)

#include <gif_lib.h>

#if !defined(GIFLIB_MAJOR) || GIFLIB_MAJOR < 5
#define GifMakeMapObject          MakeMapObject
#define GifFreeMapObject          FreeMapObject
#define EGifOpenFileName(n,b,err) EGifOpenFileName(n,b)
#define EGifOpenFileHandle(h,err) EGifOpenFileName(h)
#define EGifCloseFile(f,err)      EGifCloseFile(f)
#endif

#undef __FUNCT__
#define __FUNCT__ "PetscDrawImageSaveGIF"
PETSC_EXTERN PetscErrorCode PetscDrawImageSaveGIF(const char filename[],unsigned char palette[][3],unsigned int w,unsigned int h,const unsigned char pixels[])
{
  int            Row, Error;
  int            Width  = (int)w;
  int            Height = (int)h;
  int            ColorRes   = 8;
  int            ColorCount = 256;
  ColorMapObject *GifCMap = NULL;
  GifFileType    *GifFile = NULL;
# define         SETERRGIF(msg) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,msg", GIF file: %s",filename)
# define         CHKERRGIF(msg) do {if (Error != GIF_OK) SETERRGIF(msg);} while(0)

  PetscFunctionBegin;
  PetscValidCharPointer(filename,1);
  PetscValidCharPointer(palette,2);
  PetscValidCharPointer(pixels,5);

  GifCMap = GifMakeMapObject(ColorCount, (GifColorType*)palette); if (!GifCMap) SETERRGIF("Allocating colormap");
  GifFile = EGifOpenFileName(filename, 0, NULL); if (!GifFile) SETERRGIF("Opening");
  Error = EGifPutScreenDesc(GifFile, Width, Height, ColorRes, 0, GifCMap); CHKERRGIF("Writing screen descriptor");
  Error = EGifPutImageDesc(GifFile, 0, 0, Width, Height, 0, NULL); CHKERRGIF("Writing image descriptor");
  for (Row = 0; Row < Height; Row++) {
    Error = EGifPutLine(GifFile, (GifPixelType*)pixels + Row*Width, Width); CHKERRGIF("Writing image pixels");
  }
  Error = EGifCloseFile(GifFile, NULL); CHKERRGIF("Closing");
  GifFreeMapObject(GifCMap); GifCMap = NULL;

# undef SETERRGIF
# undef CHKERRGIF
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawImageSave_GIF(const char filename[],unsigned char palette[][3],unsigned int w,unsigned int h,const unsigned char pixels[])
{ return PetscDrawImageSaveGIF(filename,palette,w,h,pixels); }

#endif/*!PETSC_HAVE_GIFLIB*/

/*
   Code to write images in JPEG format
*/
#if defined(PETSC_HAVE_LIBJPEG)

#include <jpeglib.h>

#if defined(PETSC_HAVE_SETJMP_H)
#include <setjmp.h>
static jmp_buf petsc_jpeg_jumpbuf;
static void petsc_jpeg_error_longjmp (j_common_ptr cinfo) { (void)cinfo; longjmp(petsc_jpeg_jumpbuf,1); }
#endif

#undef __FUNCT__
#define __FUNCT__ "PetscDrawImageSaveJPG"
PETSC_EXTERN PetscErrorCode PetscDrawImageSaveJPG(const char filename[],unsigned char palette[][3],unsigned int w,unsigned int h,const unsigned char pixels[])
{
  unsigned char               *rgbpixels;
  FILE                        *fp;
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr       jerr;
  PetscErrorCode              ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(filename,1);
  if (palette) PetscValidCharPointer(palette,2);
  PetscValidCharPointer(pixels,5);
  /* map pixels to RGB colors */
  if (palette) {
    int k,p,n = (int)(w*h);
    const unsigned char *colordef;
    ierr = PetscMalloc1(3*w*h,&rgbpixels);CHKERRQ(ierr);
    for (k=p=0; k<n; k++) {
      colordef = palette[pixels[k]];
      rgbpixels[p++] = colordef[0];
      rgbpixels[p++] = colordef[1];
      rgbpixels[p++] = colordef[2];
    }
  } else { /* assume pixels are RGB colors */
    rgbpixels = (unsigned char*)pixels;
  }
  ierr = PetscFOpen(PETSC_COMM_SELF,filename,"wb",&fp);CHKERRQ(ierr);

  cinfo.err = jpeg_std_error(&jerr);
#if defined(PETSC_HAVE_SETJMP_H)
  jerr.error_exit = petsc_jpeg_error_longjmp;
  if (setjmp(petsc_jpeg_jumpbuf)) {
    char message[JMSG_LENGTH_MAX];
    jerr.format_message((j_common_ptr)&cinfo,message);
    jpeg_destroy_compress(&cinfo);
    (void)PetscFClose(PETSC_COMM_SELF,fp);
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error writing JPEG file %s\n%s",filename,message);
  }
#endif
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo,fp);
  cinfo.image_width      = w;
  cinfo.image_height     = h;
  cinfo.input_components = 3;
  cinfo.in_color_space   = JCS_RGB;
  jpeg_set_defaults(&cinfo);
  jpeg_start_compress(&cinfo,TRUE);
  while (cinfo.next_scanline < cinfo.image_height) {
    unsigned char *rowptr = rgbpixels + cinfo.next_scanline * 3*w;
    (void)jpeg_write_scanlines(&cinfo,&rowptr,1);
  }
  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);

  ierr = PetscFClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr);
  if (palette) {ierr = PetscFree(rgbpixels);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawImageSave_JPG(const char filename[],unsigned char palette[][3],unsigned int w,unsigned int h,const unsigned char pixels[])
{ return PetscDrawImageSaveJPG(filename,palette,w,h,pixels); }

#endif/*!PETSC_HAVE_LIBJPEG*/

static struct {
  const char      *extension;
  PetscErrorCode (*SaveImage)(const char[],unsigned char[][3],unsigned int,unsigned int,const unsigned char[]);
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

#undef __FUNCT__
#define __FUNCT__ "PetscDrawImageCheckFormat"
PetscErrorCode PetscDrawImageCheckFormat(const char *ext[])
{
  size_t         k;
  PetscBool      match = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* if extension is empty, return default format to caller */
  PetscValidPointer(ext,1);
  if (!*ext || !**ext) {
    *ext = PetscDrawImageSaveTable[0].extension;
    PetscFunctionReturn(0);
  }
  /* check the extension mathes a supported format otherwise */
  PetscValidCharPointer(*ext,1);
  for (k=0; k<sizeof(PetscDrawImageSaveTable)/sizeof(PetscDrawImageSaveTable[0]); k++) {
    ierr = PetscStrcasecmp(*ext,PetscDrawImageSaveTable[k].extension,&match);CHKERRQ(ierr);
    if (match && PetscDrawImageSaveTable[k].SaveImage) PetscFunctionReturn(0);
  }
  SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Image extension %s not supported, use .ppm",*ext);
  PetscFunctionReturn(PETSC_ERR_SUP);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawImageSave"
PetscErrorCode PetscDrawImageSave(const char basename[],const char ext[],unsigned char palette[][3],unsigned int w,unsigned int h,const unsigned char pixels[])
{
  size_t         k;
  PetscBool      match = PETSC_FALSE;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(basename,1);
  if (ext) PetscValidCharPointer(ext,2);
  if (palette) PetscValidCharPointer(palette,3);
  PetscValidCharPointer(pixels,6);

  ierr = PetscDrawImageCheckFormat(&ext);CHKERRQ(ierr);
  ierr = PetscSNPrintf(filename,sizeof(filename),"%s%s",basename,ext);CHKERRQ(ierr);
  for (k=0; k<sizeof(PetscDrawImageSaveTable)/sizeof(PetscDrawImageSaveTable[0]); k++) {
    ierr = PetscStrcasecmp(ext,PetscDrawImageSaveTable[k].extension,&match);CHKERRQ(ierr);
    if (match && PetscDrawImageSaveTable[k].SaveImage) {
      ierr = PetscDrawImageSaveTable[k].SaveImage(filename,palette,w,h,pixels);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }
  SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Image extension %s not supported, use .ppm",ext);
  PetscFunctionReturn(PETSC_ERR_SUP);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawMovieCheckFormat"
PetscErrorCode PetscDrawMovieCheckFormat(const char *ext[])
{
  PetscFunctionBegin;
  PetscValidPointer(ext,1);
  if (!*ext || !**ext) *ext = ".m4v";
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawMovieSave"
PetscErrorCode PetscDrawMovieSave(const char basename[],PetscInt count,const char imext[],PetscInt fps,const char mvext[])
{
  PetscBool      imgif;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(basename,1);
  PetscValidCharPointer(imext,3);
  if (mvext) PetscValidCharPointer(mvext,4);
  if (count < 1) PetscFunctionReturn(0);

  ierr = PetscStrcasecmp(imext,".gif",&imgif);CHKERRQ(ierr);
  ierr = PetscDrawMovieCheckFormat(&mvext);CHKERRQ(ierr);

#if defined(PETSC_HAVE_POPEN)
  /* use ffmpeg to generate a movie */
  {
    PetscInt i;
    FILE     *fd;
    char     ffmpeg[64] = "ffmpeg -loglevel error -y", framerate[24] = "";
    char     input[PETSC_MAX_PATH_LEN], output[PETSC_MAX_PATH_LEN];
    char     command[sizeof(ffmpeg)+sizeof(framerate)+PETSC_MAX_PATH_LEN*2];
    if (fps > 0 && !imgif) { /* ffmpeg seems to have trouble with non-animated GIF input */
      ierr = PetscSNPrintf(framerate,sizeof(framerate)," -framerate %d",(int)fps);CHKERRQ(ierr);
      ierr = PetscStrcat(ffmpeg,framerate);CHKERRQ(ierr);
    }
    ierr = PetscSNPrintf(input,sizeof(input),"%s/%s_%%d%s",basename,basename,imext);CHKERRQ(ierr);
    ierr = PetscSNPrintf(output,sizeof(output),"%s%s",basename,mvext);CHKERRQ(ierr);
    if (imgif) {
      ierr = PetscStrcat(ffmpeg," -f concat");CHKERRQ(ierr);
      ierr = PetscSNPrintf(input,sizeof(input),"%s/%s.filelist",basename,basename);CHKERRQ(ierr);
      ierr = PetscFOpen(PETSC_COMM_SELF,input,"w",&fd);CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_SELF,fd,"# ffmpeg%s -f concat -i \"%s.filelist\" \"%s\"\n",framerate,basename,output);CHKERRQ(ierr);
      for (i=0; i<count; i++) {ierr = PetscFPrintf(PETSC_COMM_SELF,fd,"file '%s_%d%s'\n",basename,i,imext);CHKERRQ(ierr);}
      ierr = PetscFClose(PETSC_COMM_SELF,fd);CHKERRQ(ierr);
    }
    ierr = PetscSNPrintf(command,sizeof(command),"%s -i \"%s\" \"%s\"",ffmpeg,input,output);CHKERRQ(ierr);
    ierr = PetscPOpen(PETSC_COMM_SELF,NULL,command,"r",&fd);CHKERRQ(ierr);
    ierr = PetscPClose(PETSC_COMM_SELF,fd,NULL);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}
