#include <../src/sys/classes/draw/impls/image/drawimage.h>   /*I  "petscdraw.h" I*/
#include <petsc/private/drawimpl.h>                          /*I  "petscdraw.h" I*/

#if defined(PETSC_USE_DEBUG)
#define PetscDrawValidColor(color) PetscCheck((color)>=0&&(color)<256,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Color value %" PetscInt_FMT " out of range [0..255]",(PetscInt)(color))
#else
#define PetscDrawValidColor(color) do {} while (0)
#endif

#define XTRANS(draw,img,x)  ((int)(((img)->w-1)*((draw)->port_xl + ((((x) - (draw)->coor_xl)*((draw)->port_xr - (draw)->port_xl))/((draw)->coor_xr - (draw)->coor_xl)))))
#define YTRANS(draw,img,y)  (((img)->h-1) - (int)(((img)->h-1)*((draw)->port_yl + ((((y) - (draw)->coor_yl)*((draw)->port_yr - (draw)->port_yl))/((draw)->coor_yr - (draw)->coor_yl)))))

#define ITRANS(draw,img,i)  ((draw)->coor_xl + (((PetscReal)(i))*((draw)->coor_xr - (draw)->coor_xl)/((img)->w-1) - (draw)->port_xl)/((draw)->port_xr - (draw)->port_xl))
#define JTRANS(draw,img,j)  ((draw)->coor_yl + (((PetscReal)(j))/((img)->h-1) + (draw)->port_yl - 1)*((draw)->coor_yr - (draw)->coor_yl)/((draw)->port_yl - (draw)->port_yr))

static PetscErrorCode PetscDrawSetViewport_Image(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr)
{
  PetscImage img = (PetscImage)draw->data;
  PetscFunctionBegin;
  {
    int xmax = img->w - 1,   ymax = img->h - 1;
    int xa = (int)(xl*xmax), ya = ymax - (int)(yr*ymax);
    int xb = (int)(xr*xmax), yb = ymax - (int)(yl*ymax);
    PetscImageSetClip(img,xa,ya,xb+1-xa,yb+1-ya);
  }
  PetscFunctionReturn(0);
}

/*
static PetscErrorCode PetscDrawSetCoordinates_Image(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}*/
#define PetscDrawSetCoordinates_Image NULL

static PetscErrorCode PetscDrawCoordinateToPixel_Image(PetscDraw draw,PetscReal x,PetscReal y,int *i,int *j)
{
  PetscImage img = (PetscImage)draw->data;
  PetscFunctionBegin;
  if (i) *i = XTRANS(draw,img,x);
  if (j) *j = YTRANS(draw,img,y);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawPixelToCoordinate_Image(PetscDraw draw,int i,int j,PetscReal *x,PetscReal *y)
{
  PetscImage img = (PetscImage)draw->data;
  PetscFunctionBegin;
  if (x) *x = ITRANS(draw,img,i);
  if (y) *y = JTRANS(draw,img,j);
  PetscFunctionReturn(0);
}

/*
static PetscErrorCode PetscDrawPointSetSize_Image(PetscDraw draw,PetscReal width)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}*/
#define PetscDrawPointSetSize_Image NULL

static PetscErrorCode PetscDrawPoint_Image(PetscDraw draw,PetscReal x,PetscReal y,int c)
{
  PetscImage img = (PetscImage)draw->data;
  PetscFunctionBegin;
  PetscDrawValidColor(c);
  {
    int j, xx = XTRANS(draw,img,x);
    int i, yy = YTRANS(draw,img,y);
    for (i=-1; i<=1; i++)
      for (j=-1; j<=1; j++)
        PetscImageDrawPixel(img,xx+j,yy+i,c);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawPointPixel_Image(PetscDraw draw,int x,int y,int c)
{
  PetscImage img = (PetscImage)draw->data;
  PetscFunctionBegin;
  PetscDrawValidColor(c);
  {
    PetscImageDrawPixel(img,x,y,c);
  }
  PetscFunctionReturn(0);
}

/*
static PetscErrorCode PetscDrawLineSetWidth_Image(PetscDraw draw,PetscReal width)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}*/
#define PetscDrawLineSetWidth_Image NULL

static PetscErrorCode PetscDrawLineGetWidth_Image(PetscDraw draw,PetscReal *width)
{
  PetscImage img = (PetscImage)draw->data;
  PetscFunctionBegin;
  {
    int lw = 1;
    *width = lw*(draw->coor_xr - draw->coor_xl)/(img->w*(draw->port_xr - draw->port_xl));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawLine_Image(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int c)
{
  PetscImage img = (PetscImage)draw->data;
  PetscFunctionBegin;
  {
    int x_1 = XTRANS(draw,img,xl), x_2 = XTRANS(draw,img,xr);
    int y_1 = YTRANS(draw,img,yl), y_2 = YTRANS(draw,img,yr);
    PetscImageDrawLine(img,x_1,y_1,x_2,y_2,c);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawArrow_Image(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int c)
{
  PetscImage img = (PetscImage)draw->data;
  PetscFunctionBegin;
  PetscDrawValidColor(c);
  {
    int x_1 = XTRANS(draw,img,xl), x_2 = XTRANS(draw,img,xr);
    int y_1 = YTRANS(draw,img,yl), y_2 = YTRANS(draw,img,yr);
    if (x_1 == x_2 && y_1 == y_2) PetscFunctionReturn(0);
    PetscImageDrawLine(img,x_1,y_1,x_2,y_2,c);
    if (x_1 == x_2 && PetscAbs(y_1 - y_2) > 7) {
      if (y_2 > y_1) {
        PetscImageDrawLine(img,x_2,y_2,x_2-3,y_2-3,c);
        PetscImageDrawLine(img,x_2,y_2,x_2+3,y_2-3,c);
      } else {
        PetscImageDrawLine(img,x_2,y_2,x_2-3,y_2+3,c);
        PetscImageDrawLine(img,x_2,y_2,x_2+3,y_2+3,c);
      }
    }
    if (y_1 == y_2 && PetscAbs(x_1 - x_2) > 7) {
      if (x_2 > x_1) {
        PetscImageDrawLine(img,x_2-3,y_2-3,x_2,y_2,c);
        PetscImageDrawLine(img,x_2-3,y_2+3,x_2,y_2,c);
      } else {
        PetscImageDrawLine(img,x_2,y_2,x_2+3,y_2-3,c);
        PetscImageDrawLine(img,x_2,y_2,x_2+3,y_2+3,c);
      }
    }
   }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawRectangle_Image(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int c1,int c2,int c3,int c4)
{
  PetscImage img = (PetscImage)draw->data;
  PetscFunctionBegin;
  PetscDrawValidColor(c1);
  PetscDrawValidColor(c2);
  PetscDrawValidColor(c3);
  PetscDrawValidColor(c4);
  {
    int x = XTRANS(draw,img,xl), w = XTRANS(draw,img,xr) + 1 - x;
    int y = YTRANS(draw,img,yr), h = YTRANS(draw,img,yl) + 1 - y;
    int c  = (c1 + c2 + c3 + c4)/4;
    PetscImageDrawRectangle(img,x,y,w,h,c);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawEllipse_Image(PetscDraw draw,PetscReal x,PetscReal y,PetscReal a,PetscReal b,int c)
{
  PetscImage img = (PetscImage)draw->data;
  PetscFunctionBegin;
  PetscDrawValidColor(c);
  a = PetscAbsReal(a);
  b = PetscAbsReal(b);
  {
    int xc = XTRANS(draw,img,x), w = XTRANS(draw,img,x + a/2) + 0 - xc;
    int yc = YTRANS(draw,img,y), h = YTRANS(draw,img,y - b/2) + 0 - yc;
    if (PetscAbsReal(a-b) <= 0)  w = h = PetscMin(w,h); /* workaround truncation errors */
    PetscImageDrawEllipse(img,xc,yc,w,h,c);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawTriangle_Image(PetscDraw draw,PetscReal X_1,PetscReal Y_1,PetscReal X_2,PetscReal Y_2,PetscReal X_3,PetscReal Y_3,int c1,int c2,int c3)
{
  PetscImage img = (PetscImage)draw->data;
  PetscFunctionBegin;
  PetscDrawValidColor(c1);
  PetscDrawValidColor(c2);
  PetscDrawValidColor(c3);
  {
    int x_1 = XTRANS(draw,img,X_1), x_2 = XTRANS(draw,img,X_2), x_3 = XTRANS(draw,img,X_3);
    int y_1 = YTRANS(draw,img,Y_1), y_2 = YTRANS(draw,img,Y_2), y_3 = YTRANS(draw,img,Y_3);
    PetscImageDrawTriangle(img,x_1,y_1,c1,x_2,y_2,c2,x_3,y_3,c3);
  }
  PetscFunctionReturn(0);
}

/*
static PetscErrorCode PetscDrawStringSetSize_Image(PetscDraw draw,PetscReal w,PetscReal h)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}*/
#define PetscDrawStringSetSize_Image NULL

static PetscErrorCode PetscDrawStringGetSize_Image(PetscDraw draw,PetscReal *w,PetscReal  *h)
{
  PetscImage img = (PetscImage)draw->data;
  PetscFunctionBegin;
  {
    int tw = PetscImageFontWidth;
    int th = PetscImageFontHeight;
    if (w) *w = tw*(draw->coor_xr - draw->coor_xl)/(img->w*(draw->port_xr - draw->port_xl));
    if (h) *h = th*(draw->coor_yr - draw->coor_yl)/(img->h*(draw->port_yr - draw->port_yl));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawString_Image(PetscDraw draw,PetscReal x,PetscReal y,int c,const char text[])
{
  PetscImage     img = (PetscImage)draw->data;
  PetscToken     token;
  char           *subtext;
  PetscFunctionBegin;
  PetscDrawValidColor(c);
  {
    int xx = XTRANS(draw,img,x);
    int yy = YTRANS(draw,img,y);
    PetscCall(PetscTokenCreate(text,'\n',&token));
    PetscCall(PetscTokenFind(token,&subtext));
    while (subtext) {
      PetscImageDrawText(img,xx,yy,c,subtext);
      yy += PetscImageFontHeight;
      PetscCall(PetscTokenFind(token,&subtext));
    }
    PetscCall(PetscTokenDestroy(&token));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawStringVertical_Image(PetscDraw draw,PetscReal x,PetscReal y,int c,const char text[])
{
  PetscImage img = (PetscImage)draw->data;
  PetscFunctionBegin;
  PetscDrawValidColor(c);
  {
    char chr[2] = {0, 0};
    int  xx = XTRANS(draw,img,x);
    int  yy = YTRANS(draw,img,y);
    int  offset = PetscImageFontHeight;
    while ((chr[0] = *text++)) {
      PetscImageDrawText(img,xx,yy+offset,c,chr);
      yy += PetscImageFontHeight;
    }
  }
  PetscFunctionReturn(0);
}

/*
static PetscErrorCode PetscDrawStringBoxed_Image(PetscDraw draw,PetscReal sxl,PetscReal syl,int sc,int bc,const char text[],PetscReal *w,PetscReal *h)
{
  PetscFunctionBegin;
  if (w) *w = 0;
  if (h) *h = 0;
  PetscFunctionReturn(0);
*/
#define PetscDrawStringBoxed_Image NULL

/*
static PetscErrorCode PetscDrawFlush_Image(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}*/
#define PetscDrawFlush_Image NULL

static PetscErrorCode PetscDrawClear_Image(PetscDraw draw)
{
  PetscImage     img = (PetscImage)draw->data;
  PetscFunctionBegin;
  {
    PetscImageClear(img);
  }
  PetscFunctionReturn(0);
}

/*
static PetscErrorCode PetscDrawSetDoubleBuffer_Image(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}*/
#define PetscDrawSetDoubleBuffer_Image NULL

static PetscErrorCode PetscDrawGetPopup_Image(PetscDraw draw,PetscDraw *popup)
{
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetBool(((PetscObject)draw)->options,((PetscObject)draw)->prefix,"-draw_popup",&flg,NULL));
  if (!flg) {*popup = NULL; PetscFunctionReturn(0);}
  PetscCall(PetscDrawCreate(PetscObjectComm((PetscObject)draw),NULL,NULL,0,0,220,220,popup));
  PetscCall(PetscDrawSetType(*popup,PETSC_DRAW_IMAGE));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*popup,"popup_"));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)*popup,((PetscObject)draw)->prefix));
  draw->popup = *popup;
  PetscFunctionReturn(0);
}

/*
static PetscErrorCode PetscDrawSetTitle_Image(PetscDraw draw,const char title[])
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}*/
#define PetscDrawSetTitle_Image NULL

/*
static PetscErrorCode PetscDrawCheckResizedWindow_Image(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}*/
#define PetscDrawCheckResizedWindow_Image NULL

static PetscErrorCode PetscDrawResizeWindow_Image(PetscDraw draw,int w,int h)
{
  PetscImage     img = (PetscImage)draw->data;

  PetscFunctionBegin;
  if (w == img->w && h == img->h) PetscFunctionReturn(0);
  PetscCall(PetscFree(img->buffer));

  img->w = w; img->h = h;
  PetscCall(PetscCalloc1((size_t)(img->w*img->h),&img->buffer));
  PetscCall(PetscDrawSetViewport_Image(draw,draw->port_xl,draw->port_yl,draw->port_xr,draw->port_yr));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawDestroy_Image(PetscDraw draw)
{
  PetscImage     img = (PetscImage)draw->data;

  PetscFunctionBegin;
  PetscCall(PetscDrawDestroy(&draw->popup));
  PetscCall(PetscFree(img->buffer));
  PetscCall(PetscFree(draw->data));
  PetscFunctionReturn(0);
}

/*
static PetscErrorCode PetscDrawView_Image(PetscDraw draw,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}*/
#define PetscDrawView_Image NULL

/*
static PetscErrorCode PetscDrawGetMouseButton_Image(PetscDraw draw,PetscDrawButton *button,PetscReal *x_user,PetscReal *y_user,PetscReal *x_phys,PetscReal *y_phys)
{
  PetscFunctionBegin;
  *button = PETSC_BUTTON_NONE;
  if (x_user) *x_user = 0;
  if (y_user) *y_user = 0;
  if (x_phys) *x_phys = 0;
  if (y_phys) *y_phys = 0;
  PetscFunctionReturn(0);
}*/
#define PetscDrawGetMouseButton_Image NULL

/*
static PetscErrorCode PetscDrawPause_Image(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}*/
#define PetscDrawPause_Image NULL

/*
static PetscErrorCode PetscDrawBeginPage_Image(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}*/
#define PetscDrawBeginPage_Image NULL

/*
static PetscErrorCode PetscDrawEndPage_Image(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}*/
#define PetscDrawEndPage_Image NULL

static PetscErrorCode PetscDrawGetSingleton_Image(PetscDraw draw,PetscDraw *sdraw)
{
  PetscImage     pimg = (PetscImage)draw->data;
  PetscImage     simg;

  PetscFunctionBegin;
  PetscCall(PetscDrawCreate(PETSC_COMM_SELF,NULL,NULL,0,0,draw->w,draw->h,sdraw));
  PetscCall(PetscDrawSetType(*sdraw,PETSC_DRAW_IMAGE));
  (*sdraw)->ops->resizewindow = NULL;
  simg = (PetscImage)(*sdraw)->data;
  PetscCall(PetscArraycpy(simg->buffer,pimg->buffer,pimg->w*pimg->h));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawRestoreSingleton_Image(PetscDraw draw,PetscDraw *sdraw)
{
  PetscImage     pimg = (PetscImage)draw->data;
  PetscImage     simg = (PetscImage)(*sdraw)->data;

  PetscFunctionBegin;
  PetscCall(PetscArraycpy(pimg->buffer,simg->buffer,pimg->w*pimg->h));
  PetscCall(PetscDrawDestroy(sdraw));
  PetscFunctionReturn(0);
}

/*
static PetscErrorCode PetscDrawSave_Image(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}*/
#define PetscDrawSave_Image NULL

static PetscErrorCode PetscDrawGetImage_Image(PetscDraw draw,unsigned char palette[256][3],unsigned int *w,unsigned int *h,unsigned char *pixels[])
{
  PetscImage     img = (PetscImage)draw->data;
  unsigned char  *buffer = NULL;
  PetscMPIInt    rank,size;

  PetscFunctionBegin;
  if (w) *w = (unsigned int)img->w;
  if (h) *h = (unsigned int)img->h;
  if (pixels) *pixels = NULL;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)draw),&rank));
  if (rank == 0) {
    PetscCall(PetscMemcpy(palette,img->palette,sizeof(img->palette)));
    PetscCall(PetscMalloc1((size_t)(img->w*img->h),&buffer));
    if (pixels) *pixels = buffer;
  }
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)draw),&size));
  if (size == 1) {
    PetscCall(PetscArraycpy(buffer,img->buffer,img->w*img->h));
  } else {
    PetscCallMPI(MPI_Reduce(img->buffer,buffer,img->w*img->h,MPI_UNSIGNED_CHAR,MPI_MAX,0,PetscObjectComm((PetscObject)draw)));
  }
  PetscFunctionReturn(0);
}

static struct _PetscDrawOps DvOps = {
  PetscDrawSetDoubleBuffer_Image,
  PetscDrawFlush_Image,
  PetscDrawLine_Image,
  PetscDrawLineSetWidth_Image,
  PetscDrawLineGetWidth_Image,
  PetscDrawPoint_Image,
  PetscDrawPointSetSize_Image,
  PetscDrawString_Image,
  PetscDrawStringVertical_Image,
  PetscDrawStringSetSize_Image,
  PetscDrawStringGetSize_Image,
  PetscDrawSetViewport_Image,
  PetscDrawClear_Image,
  PetscDrawRectangle_Image,
  PetscDrawTriangle_Image,
  PetscDrawEllipse_Image,
  PetscDrawGetMouseButton_Image,
  PetscDrawPause_Image,
  PetscDrawBeginPage_Image,
  PetscDrawEndPage_Image,
  PetscDrawGetPopup_Image,
  PetscDrawSetTitle_Image,
  PetscDrawCheckResizedWindow_Image,
  PetscDrawResizeWindow_Image,
  PetscDrawDestroy_Image,
  PetscDrawView_Image,
  PetscDrawGetSingleton_Image,
  PetscDrawRestoreSingleton_Image,
  PetscDrawSave_Image,
  PetscDrawGetImage_Image,
  PetscDrawSetCoordinates_Image,
  PetscDrawArrow_Image,
  PetscDrawCoordinateToPixel_Image,
  PetscDrawPixelToCoordinate_Image,
  PetscDrawPointPixel_Image,
  PetscDrawStringBoxed_Image
};

static const unsigned char BasicColors[PETSC_DRAW_BASIC_COLORS][3] = {
  { 255, 255, 255 }, /* white */
  {   0,   0,   0 }, /* black */
  { 255,   0,   0 }, /* red */
  {   0, 255,   0 }, /* green */
  {   0, 255, 255 }, /* cyan */
  {   0,   0, 255 }, /* blue */
  { 255,   0, 255 }, /* magenta */
  { 127, 255, 212 }, /* aquamarine */
  {  34, 139,  34 }, /* forestgreen */
  { 255, 165,   0 }, /* orange */
  { 238, 130, 238 }, /* violet */
  { 165,  42,  42 }, /* brown */
  { 255, 192, 203 }, /* pink */
  { 255, 127,  80 }, /* coral */
  { 190, 190, 190 }, /* gray */
  { 255, 255,   0 }, /* yellow */
  { 255, 215,   0 }, /* gold */
  { 255, 182, 193 }, /* lightpink */
  {  72, 209, 204 }, /* mediumturquoise */
  { 240, 230, 140 }, /* khaki */
  { 105, 105, 105 }, /* dimgray */
  {  54, 205,  50 }, /* yellowgreen */
  { 135, 206, 235 }, /* skyblue */
  {   0, 100,   0 }, /* darkgreen */
  {   0,   0, 128 }, /* navyblue */
  { 244, 164,  96 }, /* sandybrown */
  {  95, 158, 160 }, /* cadetblue */
  { 176, 224, 230 }, /* powderblue */
  { 255,  20, 147 }, /* deeppink */
  { 216, 191, 216 }, /* thistle */
  {  50, 205,  50 }, /* limegreen */
  { 255, 240, 245 }, /* lavenderblush */
  { 221, 160, 221 }, /* plum */
};

/*MC
   PETSC_DRAW_IMAGE - PETSc graphics device that uses a raster buffer

   Options Database Keys:
.  -draw_size w,h - size of image in pixels

   Level: beginner

.seealso:  PetscDrawOpenImage(), PetscDrawSetFromOptions()
M*/
PETSC_EXTERN PetscErrorCode PetscDrawCreate_Image(PetscDraw);

PETSC_EXTERN PetscErrorCode PetscDrawCreate_Image(PetscDraw draw)
{
  PetscImage     img;
  int            w = draw->w, h = draw->h;
  PetscInt       size[2], nsize = 2;
  PetscBool      set;

  PetscFunctionBegin;
  draw->pause   = 0;
  draw->coor_xl = 0; draw->coor_xr = 1;
  draw->coor_yl = 0; draw->coor_yr = 1;
  draw->port_xl = 0; draw->port_xr = 1;
  draw->port_yl = 0; draw->port_yr = 1;

  size[0] = w; if (size[0] < 1) size[0] = 300;
  size[1] = h; if (size[1] < 1) size[1] = size[0];
  PetscCall(PetscOptionsGetIntArray(((PetscObject)draw)->options,((PetscObject)draw)->prefix,"-draw_size",size,&nsize,&set));
  if (set && nsize == 1) size[1] = size[0];
  if (size[0] < 1) size[0] = 300;
  if (size[1] < 1) size[1] = size[0];
  draw->w = w = size[0]; draw->x = 0;
  draw->h = h = size[1]; draw->x = 0;

  PetscCall(PetscNewLog(draw,&img));
  PetscCall(PetscMemcpy(draw->ops,&DvOps,sizeof(DvOps)));
  draw->data = (void*)img;

  img->w = w; img->h = h;
  PetscCall(PetscCalloc1((size_t)(img->w*img->h),&img->buffer));
  PetscImageSetClip(img,0,0,img->w,img->h);
  {
    int i,k,ncolors = 256-PETSC_DRAW_BASIC_COLORS;
    unsigned char R[256-PETSC_DRAW_BASIC_COLORS];
    unsigned char G[256-PETSC_DRAW_BASIC_COLORS];
    unsigned char B[256-PETSC_DRAW_BASIC_COLORS];
    PetscCall(PetscDrawUtilitySetCmap(NULL,ncolors,R,G,B));
    for (k=0; k<PETSC_DRAW_BASIC_COLORS; k++) {
      img->palette[k][0] = BasicColors[k][0];
      img->palette[k][1] = BasicColors[k][1];
      img->palette[k][2] = BasicColors[k][2];
    }
    for (i=0; i<ncolors; i++, k++) {
      img->palette[k][0] = R[i];
      img->palette[k][1] = G[i];
      img->palette[k][2] = B[i];
    }
  }

  if (!draw->savefilename) PetscCall(PetscDrawSetSave(draw,draw->title));
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawOpenImage - Opens an image for use with the PetscDraw routines.

   Collective

   Input Parameters:
+  comm - the communicator that will share image
-  filename - optional name of the file
-  w, h - the image width and height in pixels

   Output Parameters:
.  draw - the drawing context.

   Level: beginner

.seealso: PetscDrawSetSave(), PetscDrawSetFromOptions(), PetscDrawCreate(), PetscDrawDestroy()
@*/
PetscErrorCode PetscDrawOpenImage(MPI_Comm comm,const char filename[],int w,int h,PetscDraw *draw)
{
  PetscFunctionBegin;
  PetscCall(PetscDrawCreate(comm,NULL,NULL,0,0,w,h,draw));
  PetscCall(PetscDrawSetType(*draw,PETSC_DRAW_IMAGE));
  PetscCall(PetscDrawSetSave(*draw,filename));
  PetscFunctionReturn(0);
}
