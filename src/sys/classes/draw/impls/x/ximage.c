/*
    Code for getting raster images out of a X image or pixmap
*/

#include <../src/sys/classes/draw/impls/x/ximpl.h>

PETSC_INTERN PetscErrorCode PetscDrawGetImage_X(PetscDraw,unsigned char[][3],unsigned int*,unsigned int*,unsigned char*[]);

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
#define __FUNCT__ "PetscDrawGetImage_X"
PetscErrorCode PetscDrawGetImage_X(PetscDraw draw,unsigned char palette[256][3],unsigned int *out_w,unsigned int *out_h,unsigned char *out_pixels[])
{
  PetscDraw_X      *Xwin = (PetscDraw_X*)draw->data;
  PetscMPIInt      rank;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (out_w)      *out_w      = 0;
  if (out_h)      *out_h      = 0;
  if (out_pixels) *out_pixels = NULL;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)draw),&rank);CHKERRQ(ierr);

  /* make sure the X server processed requests from all processes */
  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  XSync(Xwin->disp,True);
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  ierr = MPI_Barrier(PetscObjectComm((PetscObject)draw));CHKERRQ(ierr);

  /* only the first process handles the saving business */
  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  if (!rank) {
    Window        root;
    XImage        *ximage;
    unsigned char *pixels = NULL;
    unsigned int  w,h,dummy;
    int           x,y,p;
    /* get RGB colors out of the colormap */
    ierr = PetscDrawXiGetColorsRGB(Xwin,palette);CHKERRQ(ierr);
    /* get image out of the drawable */
    XGetGeometry(Xwin->disp,PetscDrawXiDrawable(Xwin),&root,&x,&y,&w,&h,&dummy,&dummy);
    ximage = XGetImage(Xwin->disp,PetscDrawXiDrawable(Xwin),0,0,w,h,AllPlanes,ZPixmap);
    if (!ximage) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cannot XGetImage()");
    /* extract pixel values out of the image  */
    ierr = PetscMalloc1(w*h,&pixels);CHKERRQ(ierr);
    for (p=0,y=0; y<(int)h; y++)
      for (x=0; x<(int)w; x++)
        pixels[p++] = PetscDrawXiPixelToColor(Xwin,XGetPixel(ximage,x,y));
    XDestroyImage(ximage);
    *out_w      = w;
    *out_h      = h;
    *out_pixels = pixels;
  }
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
