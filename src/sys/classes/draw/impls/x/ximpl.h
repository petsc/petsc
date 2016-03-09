
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
  Display           *disp;
  int               screen;
  Window            win;
  Visual            *vis;            /* Graphics visual */
  PetscDrawXiGC     gc;
  PetscDrawXiFont   *font;
  int               depth;           /* Depth of visual */
  int               numcolors;       /* Number of available colors */
  int               maxcolors;       /* Current number in use */
  Colormap          cmap;
  PetscDrawXiPixVal foreground,background;
  PetscDrawXiPixVal cmapping[256];
  int               x,y,w,h;      /* Size and location of window */
  Drawable          drw;
} PetscDraw_X;

#define PetscDrawXiDrawable(w) ((w)->drw ? (w)->drw : (w)->win)

#define PetscDrawXiSetPixVal(W,pix) do {         \
    if ((W)->gc.cur_pix != (pix)) {              \
      XSetForeground((W)->disp,(W)->gc.set,pix); \
      (W)->gc.cur_pix = pix;                     \
    }} while (0)

#define PetscDrawXiSetColor(W,color) do {         \
    if ((color) >= 256 || (color) < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Color value %D out of range [0..255]",(PetscInt)(color)); \
    PetscDrawXiSetPixVal(W,(W)->cmapping[color]); \
    } while (0)

PETSC_INTERN PetscErrorCode PetscDrawXiInit(PetscDraw_X*,const char[]);
PETSC_INTERN PetscErrorCode PetscDrawXiClose(PetscDraw_X*);
PETSC_INTERN PetscErrorCode PetscDrawXiFontFixed(PetscDraw_X*,int,int,PetscDrawXiFont**);
PETSC_INTERN PetscErrorCode PetscDrawXiColormap(PetscDraw_X*);
PETSC_INTERN PetscErrorCode PetscDrawXiQuickWindow(PetscDraw_X*,char*,int,int,int,int);
PETSC_INTERN PetscErrorCode PetscDrawXiQuickWindowFromWindow(PetscDraw_X*,Window);
PETSC_INTERN PetscErrorCode PetscDrawXiQuickPixmap(PetscDraw_X*);

#endif
