/*
   Defines the internal data structures for the X-windows
   implementation of the graphics functionality in PETSc.
*/

#if !defined(_XIMPL_H)
#define _XIMPL_H
#include <petsc/private/drawimpl.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>

typedef unsigned long PetscDrawXiPixVal;

typedef struct {
  GC                set;
  PetscDrawXiPixVal cur_pix;
} PetscDrawXiGC;

typedef struct {
  Font              fnt;
  int               font_w,font_h;
  int               font_descent;
  PetscDrawXiPixVal font_pix;
} PetscDrawXiFont;

typedef struct {
  Display           *disp;            /* Display */
  int               screen;           /* Screen of display */
  Visual            *vis;             /* Graphics visual */
  int               depth;            /* Depth of visual */
  PetscDrawXiGC     gc;               /* Graphics context */
  PetscDrawXiFont   *font;            /* Current font */
  Window            win;              /* Window */
  Drawable          drw;              /* Pixmap */
  Colormap          cmap;             /* Colormap */
  int               cmapsize;         /* Number of allocated colors */
  PetscDrawXiPixVal foreground;       /* Foreground pixel */
  PetscDrawXiPixVal background;       /* Background pixel */
  PetscDrawXiPixVal cmapping[PETSC_DRAW_MAXCOLOR];    /* Map color -> pixel value */
  unsigned char     cpalette[PETSC_DRAW_MAXCOLOR][3]; /* Map color -> RGB value*/
  int               x,y,w,h;          /* Location and size window */
} PetscDraw_X;

#define PetscDrawXiDrawable(w) ((w)->drw ? (w)->drw : (w)->win)

PETSC_STATIC_INLINE void PetscDrawXiSetPixVal(PetscDraw_X *W,PetscDrawXiPixVal pix)
{ if (W->gc.cur_pix != pix) { XSetForeground(W->disp,W->gc.set,pix); W->gc.cur_pix = pix; } }

#if defined(PETSC_USE_DEBUG)
#define PetscDrawXiValidColor(W,color) \
  do { if (PetscUnlikely((color)<0||(color)>=PETSC_DRAW_MAXCOLOR)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Color value %" PetscInt_FMT " out of range [0..%d]",(PetscInt)(color),PETSC_DRAW_MAXCOLOR-1); } while (0)
#else
#define PetscDrawXiValidColor(W,color) do {} while (0)
#endif

#define PetscDrawXiSetColor(W,color) do { PetscDrawXiValidColor(W,color); PetscDrawXiSetPixVal(W,(W)->cmapping[(color)]); } while (0)

PETSC_INTERN PetscErrorCode PetscDrawXiInit(PetscDraw_X*,const char[]);
PETSC_INTERN PetscErrorCode PetscDrawXiClose(PetscDraw_X*);
PETSC_INTERN PetscErrorCode PetscDrawXiFontFixed(PetscDraw_X*,int,int,PetscDrawXiFont**);
PETSC_INTERN PetscErrorCode PetscDrawXiColormap(PetscDraw_X*);
PETSC_INTERN PetscErrorCode PetscDrawXiQuickWindow(PetscDraw_X*,char*,int,int,int,int);
PETSC_INTERN PetscErrorCode PetscDrawXiQuickWindowFromWindow(PetscDraw_X*,Window);
PETSC_INTERN PetscErrorCode PetscDrawXiQuickPixmap(PetscDraw_X*);
PETSC_INTERN PetscErrorCode PetscDrawXiResizeWindow(PetscDraw_X*,int,int);
PETSC_INTERN PetscErrorCode PetscDrawXiGetGeometry(PetscDraw_X*,int*,int*,int*,int*);

#endif
