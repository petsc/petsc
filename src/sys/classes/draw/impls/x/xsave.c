/*
    Code for saving images out of a X image or pixmap
*/

#include <../src/sys/classes/draw/impls/x/ximpl.h>

PETSC_INTERN PetscErrorCode PetscDrawSetSave_X(PetscDraw,const char[]);
PETSC_INTERN PetscErrorCode PetscDrawSave_X(PetscDraw);


#undef __FUNCT__
#define __FUNCT__ "PetscDrawSetSave_X"
PetscErrorCode PetscDrawSetSave_X(PetscDraw draw,const char filename[])
{
#if defined(PETSC_HAVE_POPEN) && defined(PETSC_HAVE_AFTERIMAGE)
  PetscMPIInt    rank;
  PetscErrorCode ierr;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_POPEN) && defined(PETSC_HAVE_AFTERIMAGE)
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)draw),&rank);CHKERRQ(ierr);
  if (!rank) {
    char  command[PETSC_MAX_PATH_LEN];
    FILE  *fd;
    int   err;
    ierr = PetscMemzero(command,sizeof(command));CHKERRQ(ierr);
    ierr = PetscSNPrintf(command,PETSC_MAX_PATH_LEN,"rm -fr %s %s.m4v",draw->savefilename,draw->savefilename);CHKERRQ(ierr);
    ierr = PetscPOpen(PETSC_COMM_SELF,NULL,command,"r",&fd);CHKERRQ(ierr);
    ierr = PetscPClose(PETSC_COMM_SELF,fd,&err);CHKERRQ(ierr);
    ierr = PetscSNPrintf(command,PETSC_MAX_PATH_LEN,"mkdir %s",draw->savefilename);CHKERRQ(ierr);
    ierr = PetscPOpen(PETSC_COMM_SELF,NULL,command,"r",&fd);CHKERRQ(ierr);
    ierr = PetscPClose(PETSC_COMM_SELF,fd,&err);CHKERRQ(ierr);
  }
#else
  (void)draw; (void)filename; /* unused */
#endif
  PetscFunctionReturn(0);
}

#if !defined(PETSC_HAVE_AFTERIMAGE)

/*
   Write an image in binary PPM format
*/
#undef __FUNCT__
#define __FUNCT__ "PetscDrawSaveImagePPM"
static PetscErrorCode PetscDrawSaveImagePPM(const char filename[],unsigned int w,unsigned int h,unsigned char rgb[])
{
  int            fd;
  char           header[32];
  size_t         hdrlen;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(filename,1);
  PetscValidCharPointer(rgb,4);
  /* open file and write PPM header */
  ierr = PetscBinaryOpen(filename,FILE_MODE_WRITE,&fd);CHKERRQ(ierr);
  ierr = PetscSNPrintf(header,sizeof(header),"P6\n%d %d\n255\n\0",(int)w,(int)h);CHKERRQ(ierr);
  ierr = PetscStrlen(header,&hdrlen);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(fd,header,hdrlen,PETSC_CHAR,PETSC_FALSE);CHKERRQ(ierr);
  /* write image data and close file */
  ierr = PetscBinaryWrite(fd,rgb,3*w*h,PETSC_CHAR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscBinaryClose(fd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Get RGB color entries out of the X colormap
*/
#undef __FUNCT__
#define __FUNCT__ "PetscDrawXiGetColorsRGB"
static PetscErrorCode PetscDrawXiGetColorsRGB(PetscDraw_X *Xwin,unsigned char rgb[256][3])
{
  int    k;
  XColor colordef[256];

  PetscFunctionBegin;
  for (k=0; k<256; k++) {
    colordef[k].pixel = Xwin->cmapping[k];
    colordef[k].flags = DoRed|DoGreen|DoBlue;
  }
  XQueryColors(Xwin->disp,Xwin->cmap,colordef,256);
  for (k=0; k<256; k++) {
    rgb[k][0] = (unsigned char)(colordef[k].red   >> 8);
    rgb[k][1] = (unsigned char)(colordef[k].green >> 8);
    rgb[k][2] = (unsigned char)(colordef[k].blue  >> 8);
  }
  PetscFunctionReturn(0);
}

/*
   Map a pixel value to PETSc color value (index in the colormap)
*/
#undef __FUNCT__
#define __FUNCT__ "PetscDrawXiPixelToColor"
PETSC_STATIC_INLINE int PetscDrawXiPixelToColor(PetscDraw_X *Xwin,PetscDrawXiPixVal pixval)
{
  int               color;
  PetscDrawXiPixVal *cmap = Xwin->cmapping;

  PetscFunctionBegin;
  for (color=0; color<256; color++)   /* slow linear search */
    if (cmap[color] == pixval) break; /* found color */
  if (PetscUnlikely(color == 256))    /* should not happen */
    color = PETSC_DRAW_BLACK;
  PetscFunctionReturn(color);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawSave_X"
PetscErrorCode PetscDrawSave_X(PetscDraw draw)
{
  PetscDraw_X      *Xwin = (PetscDraw_X*)draw->data;
  const char       *name = draw->savefilename;
  const char       *ext   = draw->savefilenameext;
  PetscInt         savecount;
  char             filename[PETSC_MAX_PATH_LEN];
  PetscMPIInt      rank;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (!draw->savefilename) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)draw),&rank);CHKERRQ(ierr);

  savecount = draw->savefilecount++;
  if (!rank && !savecount) {
    char path[PETSC_MAX_PATH_LEN];
    if (draw->savesinglefile) {
      ierr = PetscMemzero(path,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
      ierr = PetscSNPrintf(path,PETSC_MAX_PATH_LEN,"%s%s",name,ext);CHKERRQ(ierr);
      (void)remove(path);
    } else {
      ierr = PetscMemzero(path,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
      ierr = PetscSNPrintf(path,PETSC_MAX_PATH_LEN,"%s",name);CHKERRQ(ierr);
      ierr = PetscRMTree(path);CHKERRQ(ierr);
      ierr = PetscMkdir(path);CHKERRQ(ierr);
    }
    if (draw->savefilemovie) {
      ierr = PetscMemzero(path,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
      ierr = PetscSNPrintf(path,PETSC_MAX_PATH_LEN,"%s.m4v",name);CHKERRQ(ierr);
      (void)remove(path);
    }
  }
  ierr = PetscMemzero(filename,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  if (draw->savesinglefile) {
    ierr = PetscSNPrintf(filename,PETSC_MAX_PATH_LEN,"%s%s",name,ext);CHKERRQ(ierr);
  } else {
    ierr = PetscSNPrintf(filename,PETSC_MAX_PATH_LEN,"%s/%s_%d%s",name,name,(int)savecount,ext);CHKERRQ(ierr);
  }

  /* make sure the X server processed requests from all processes */
  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  XSync(Xwin->disp,True);
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  ierr = MPI_Barrier(PetscObjectComm((PetscObject)draw));CHKERRQ(ierr);

  /* only the first process handles the saving business */
  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  if (!rank) {
    Window        root;
    XImage        *image;
    unsigned char colors[256][3],*rgb;
    unsigned int  w,h,dummy;
    int           x,y,pos = 0;
    /* get image out of the drawable  */
    XGetGeometry(Xwin->disp,PetscDrawXiDrawable(Xwin),&root,&x,&y,&w,&h,&dummy,&dummy);
    image = XGetImage(Xwin->disp,PetscDrawXiDrawable(Xwin),0,0,w,h,AllPlanes,ZPixmap);
    if (!image) SETERRQ(PetscObjectComm((PetscObject)draw),PETSC_ERR_PLIB,"Cannot XGetImage()");
    /* get RGB colors out of the colormap */
    ierr = PetscDrawXiGetColorsRGB(Xwin,colors);CHKERRQ(ierr);
    /* loop over pixels and get RGB values */
    ierr = PetscMalloc1(3*w*h,&rgb);CHKERRQ(ierr);
    for (y=0; y<(int)h; y++) {
      for (x=0; x<(int)w; x++) {
        PetscDrawXiPixVal pixel = XGetPixel(image,x,y);
        int color = PetscDrawXiPixelToColor(Xwin,pixel);
        rgb[pos++] = colors[color][0];
        rgb[pos++] = colors[color][1];
        rgb[pos++] = colors[color][2];
      }
    }
    XDestroyImage(image);
    ierr = PetscDrawSaveImagePPM(filename,w,h,rgb);CHKERRQ(ierr);
    ierr = PetscFree(rgb);CHKERRQ(ierr);
  }
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  ierr = MPI_Barrier(PetscObjectComm((PetscObject)draw));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif

#if defined(PETSC_HAVE_AFTERIMAGE)
#include <afterimage.h>

/* String names of possible Afterimage formats */
const char *PetscAfterImageFormats[] = {
        ".Xpm",
        ".Xpm.Z",
        ".Xpm.gz",
        ".Png",
        ".Jpeg",
        ".Xcf", /* Gimp format */
        ".Ppm",
        ".Pnm",
        "MS Windows Bitmap",
        "MS Windows Icon",
        "MS Windows Cursor",
        ".Gif",
        ".Tiff",
        "Afterstep XMLScript",
        "Scalable Vector Graphics (SVG)",
        ".Xbm",
        "Targa",
        ".Pcx",
        ".HTML",
        "XML",
        "Unknown"
};

#undef __FUNCT__
#define __FUNCT__ "PetscAfterimageStringToFormat"
static PetscErrorCode PetscAfterimageStringToFormat(const char *ext,ASImageFileTypes *format)
{
  PetscInt       i;
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscStrcasecmp(".Jpg",ext,&flg);CHKERRQ(ierr);
  if (flg) ext = ".Jpeg";
  for (i=0; i<sizeof(PetscAfterImageFormats)/sizeof(char**); i++) {
    ierr = PetscStrcasecmp(PetscAfterImageFormats[i],ext,&flg);CHKERRQ(ierr);
    if (flg) {*format = (ASImageFileTypes)i; PetscFunctionReturn(0);}
  }
  *format = ASIT_Unknown;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_SAWS)
#include <petscviewersaws.h>
/*
  The PetscAfterimage object and functions are used to maintain a list of file images created by Afterimage that can
  be displayed by the SAWs webserver.
*/
typedef struct _P_PetscAfterimage *PetscAfterimage;
struct _P_PetscAfterimage {
  PetscAfterimage next;
  char            *filename;
  char            *ext;
  PetscInt        cnt;
} ;

static PetscAfterimage afterimages = 0;

#undef __FUNCT__
#define __FUNCT__ "PetscAfterimageDestroy"
static PetscErrorCode PetscAfterimageDestroy(void)
{
  PetscErrorCode ierr;
  PetscAfterimage       afterimage,oafterimage = afterimages;

  PetscFunctionBegin;
  while (oafterimage) {
    afterimage = oafterimage->next;
    ierr = PetscFree(oafterimage->filename);CHKERRQ(ierr);
    ierr = PetscFree(oafterimage->ext);CHKERRQ(ierr);
    ierr = PetscFree(oafterimage);CHKERRQ(ierr);
    oafterimage = afterimage;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscAfterimageAdd"
static PetscErrorCode PetscAfterimageAdd(const char *filename,const char *ext,PetscInt cnt)
{
  PetscErrorCode   ierr;
  PetscAfterimage  afterimage,oafterimage = afterimages;
  PetscBool        flg;

  PetscFunctionBegin;
  if (oafterimage){
    ierr = PetscStrcmp(filename,oafterimage->filename,&flg);CHKERRQ(ierr);
    if (flg) {
      oafterimage->cnt = cnt;
      PetscFunctionReturn(0);
    }
    while (oafterimage->next) {
      oafterimage = oafterimage->next;
      ierr = PetscStrcmp(filename,oafterimage->filename,&flg);CHKERRQ(ierr);
      if (flg) {
        oafterimage->cnt = cnt;
        PetscFunctionReturn(0);
      }
    }
    ierr = PetscNew(&afterimage);CHKERRQ(ierr);
    oafterimage->next = afterimage;
  } else {
    ierr = PetscNew(&afterimage);CHKERRQ(ierr);
    afterimages = afterimage;
  }
  ierr = PetscStrallocpy(filename,&afterimage->filename);CHKERRQ(ierr);
  ierr = PetscStrallocpy(ext,&afterimage->ext);CHKERRQ(ierr);
  afterimage->cnt = cnt;
  ierr = PetscRegisterFinalize(PetscAfterimageDestroy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif

#undef __FUNCT__
#define __FUNCT__ "PetscDrawSave_X"
PetscErrorCode PetscDrawSave_X(PetscDraw draw)
{
  PetscDraw_X      *drawx = (PetscDraw_X*)draw->data;
  XImage           *image;
  ASImage          *asimage;
  struct  ASVisual *asv;
  char             filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode   ierr;
  PetscMPIInt      rank;
  int              depth;
  ASImageFileTypes format = ASIT_Unknown;

  PetscFunctionBegin;
  if (!draw->savefilename) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)draw),&rank);CHKERRQ(ierr);

  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  XSync(drawx->disp,True);
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  ierr = MPI_Barrier(PetscObjectComm((PetscObject)draw));CHKERRQ(ierr);

  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  if (rank) goto finally; /* only process 0 handles the saving business */
  depth = DefaultDepth(drawx->disp,drawx->screen);
  asv   = create_asvisual(drawx->disp,drawx->screen,depth,NULL);if (!asv) SETERRQ(PetscObjectComm((PetscObject)draw),PETSC_ERR_PLIB,"Cannot create AfterImage ASVisual");
  image = XGetImage(drawx->disp,PetscDrawXiDrawable(drawx),0,0,drawx->w,drawx->h,AllPlanes,ZPixmap);
  if (!image) SETERRQ(PetscObjectComm((PetscObject)draw),PETSC_ERR_PLIB,"Cannot XGetImage()");
  asimage = picture_ximage2asimage(asv,image,0,0);if (!asimage) SETERRQ(PetscObjectComm((PetscObject)draw),PETSC_ERR_PLIB,"Cannot create AfterImage ASImage");
  if (draw->savesinglefile) {
    ierr = PetscSNPrintf(filename,PETSC_MAX_PATH_LEN,"%s/%s%s",draw->savefilename,draw->savefilename,draw->savefilenameext);CHKERRQ(ierr);
  } else {
    ierr = PetscSNPrintf(filename,PETSC_MAX_PATH_LEN,"%s/%s_%d%s",draw->savefilename,draw->savefilename,draw->savefilecount++,draw->savefilenameext);CHKERRQ(ierr);
  }
  ierr = PetscAfterimageStringToFormat(draw->savefilenameext,&format);CHKERRQ(ierr);
  ASImage2file(asimage,0,filename,format,0);
#if defined(PETSC_HAVE_SAWS)
  {
    char     body[4096];
    PetscAfterimage afterimage;
    size_t   len = 0;

    ierr = PetscAfterimageAdd(draw->savefilename,draw->savefilenameext,draw->savefilecount-1);CHKERRQ(ierr);
    afterimage  = afterimages;
    while (afterimage) {
      if (draw->savesinglefile) {
        ierr = PetscSNPrintf(body+len,4086-len,"<img src=\"%s/%s%s\" alt=\"None\">",afterimage->filename,afterimage->filename,afterimage->ext);CHKERRQ(ierr);
      } else {
        ierr = PetscSNPrintf(body+len,4086-len,"<img src=\"%s/%s_%d%s\" alt=\"None\">",afterimage->filename,afterimage->filename,afterimage->cnt,afterimage->ext);CHKERRQ(ierr);
      }
      ierr = PetscStrlen(body,&len);CHKERRQ(ierr);
      afterimage  = afterimage->next;
    }
    ierr = PetscStrcat(body,"<br>\n");CHKERRQ(ierr);
    if (draw->savefilecount > 0) PetscStackCallSAWs(SAWs_Pop_Body,("index.html",1));
    PetscStackCallSAWs(SAWs_Push_Body,("index.html",1,body));
  }
#endif
  destroy_asvisual(asv,0);
  XDestroyImage(image);
finally:
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  ierr = MPI_Barrier(PetscObjectComm((PetscObject)draw));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   There are routines wanted by AfterImage for PNG files
 */
void crc32(void) {;}
void inflateReset(void) {;}
void deflateReset(void) {;}
void deflateInit2(void) {;}
void deflateInit2_(void) {;}
void deflate(void) {;}
void deflateEnd(void) {;}

#endif
