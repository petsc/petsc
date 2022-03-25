/*
    Code for getting raster images out of a X image or pixmap
*/

#include <../src/sys/classes/draw/impls/x/ximpl.h>

PETSC_INTERN PetscErrorCode PetscDrawGetImage_X(PetscDraw,unsigned char[PETSC_DRAW_MAXCOLOR][3],unsigned int*,unsigned int*,unsigned char*[]);

static inline PetscErrorCode PetscArgSortPixVal(const PetscDrawXiPixVal v[PETSC_DRAW_MAXCOLOR],int idx[],int right)
{
  PetscDrawXiPixVal vl;
  int               i,last,tmp;
# define            SWAP(a,b) {tmp=a;a=b;b=tmp;}
  PetscFunctionBegin;
  if (right <= 1) {
    if (right == 1) {
      if (v[idx[0]] > v[idx[1]]) SWAP(idx[0],idx[1]);
    }
    PetscFunctionReturn(0);
  }
  SWAP(idx[0],idx[right/2]);
  vl = v[idx[0]]; last = 0;
  for (i=1; i<=right; i++)
    if (v[idx[i]] < vl) {last++; SWAP(idx[last],idx[i]);}
  SWAP(idx[0],idx[last]);
  PetscCall(PetscArgSortPixVal(v,idx,last-1));
  PetscCall(PetscArgSortPixVal(v,idx+last+1,right-(last+1)));
# undef SWAP
  PetscFunctionReturn(0);
}

/*
   Map a pixel value to PETSc color value (index in the colormap)
*/
static inline int PetscDrawXiPixelToColor(PetscDraw_X *Xwin,const int arg[PETSC_DRAW_MAXCOLOR],PetscDrawXiPixVal pix)
{
  const PetscDrawXiPixVal *cmap = Xwin->cmapping;
  int                     lo, mid, hi = PETSC_DRAW_MAXCOLOR;
  /* linear search the first few entries */
  for (lo=0; lo<8; lo++)
    if (pix == cmap[lo])
      return lo;
  /* binary search the remaining entries */
  while (hi - lo > 1) {
    mid = lo + (hi - lo)/2;
    if (pix < cmap[arg[mid]]) hi = mid;
    else                      lo = mid;
  }
  return arg[lo];
}

PetscErrorCode PetscDrawGetImage_X(PetscDraw draw,unsigned char palette[PETSC_DRAW_MAXCOLOR][3],unsigned int *out_w,unsigned int *out_h,unsigned char *out_pixels[])
{
  PetscDraw_X      *Xwin = (PetscDraw_X*)draw->data;
  PetscMPIInt      rank;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (out_w)      *out_w      = 0;
  if (out_h)      *out_h      = 0;
  if (out_pixels) *out_pixels = NULL;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)draw),&rank));

  /* make sure the X server processed requests from all processes */
  ierr = PetscDrawCollectiveBegin(draw);PetscCall(ierr);
  XSync(Xwin->disp,True);
  ierr = PetscDrawCollectiveEnd(draw);PetscCall(ierr);
  PetscCallMPI(MPI_Barrier(PetscObjectComm((PetscObject)draw)));

  /* only the first process return image data */
  ierr = PetscDrawCollectiveBegin(draw);PetscCall(ierr);
  if (rank == 0) {
    Window        root;
    XImage        *ximage;
    int           pmap[PETSC_DRAW_MAXCOLOR];
    unsigned char *pixels = NULL;
    unsigned int  w,h,dummy;
    int           x,y,p;
    /* copy colormap palette to the caller */
    PetscCall(PetscMemcpy(palette,Xwin->cpalette,sizeof(Xwin->cpalette)));
    /* get image out of the drawable */
    XGetGeometry(Xwin->disp,PetscDrawXiDrawable(Xwin),&root,&x,&y,&w,&h,&dummy,&dummy);
    ximage = XGetImage(Xwin->disp,PetscDrawXiDrawable(Xwin),0,0,w,h,AllPlanes,ZPixmap);
    PetscCheck(ximage,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cannot XGetImage()");
    /* build indirect sort permutation (a.k.a argsort) of the color -> pixel mapping */
    for (p=0; p<PETSC_DRAW_MAXCOLOR; p++) pmap[p] = p; /* identity permutation */
    PetscCall(PetscArgSortPixVal(Xwin->cmapping,pmap,255));
    /* extract pixel values out of the image and map them to color indices */
    PetscCall(PetscMalloc1(w*h,&pixels));
    for (p=0,y=0; y<(int)h; y++)
      for (x=0; x<(int)w; x++) {
        PetscDrawXiPixVal pix = XGetPixel(ximage,x,y);
        pixels[p++] = (unsigned char)PetscDrawXiPixelToColor(Xwin,pmap,pix);
      }
    XDestroyImage(ximage);
    *out_w      = w;
    *out_h      = h;
    *out_pixels = pixels;
  }
  ierr = PetscDrawCollectiveEnd(draw);PetscCall(ierr);
  PetscFunctionReturn(0);
}
