
/*
      Defines the internal data structures for the X-windows
   implementation of the graphics functionality in PETSc.
*/

#include <../src/sys/draw/drawimpl.h>

#if !defined(_XIMPL_H)
#define _XIMPL_H

#include <sys/types.h>
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
    int               numcolors,      /* Number of available colors */
                      maxcolors;       /* Current number in use */
    Colormap          cmap;
    PetscDrawXiPixVal foreground,background;
    PetscDrawXiPixVal cmapping[256];
    int               x,y,w,h;      /* Size and location of window */
    Drawable          drw;
} PetscDraw_X;

#define PetscDrawXiDrawable(w) ((w)->drw ? (w)->drw : (w)->win)

#define PetscDrawXiSetColor(Win,icolor)\
  {if (icolor >= 256 || icolor < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Color value out of range");\
   if ((Win)->gc.cur_pix != (Win)->cmapping[icolor]) { \
     XSetForeground((Win)->disp,(Win)->gc.set,(Win)->cmapping[icolor]); \
     (Win)->gc.cur_pix   = (Win)->cmapping[icolor];\
  }}

#define PetscDrawXiSetPixVal(Win,pix)\
  {if ((PetscDrawXiPixVal) (Win)->gc.cur_pix != pix) { \
     XSetForeground((Win)->disp,(Win)->gc.set,pix); \
     (Win)->gc.cur_pix   = pix;\
  }}

#endif
