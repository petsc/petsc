#define PETSC_DLL

/*
      Defines the internal data structures for the X-windows 
   implementation of the graphics functionality in PETSc.
*/

#include "../src/sys/draw/drawimpl.h"

#if !defined(_XIMPL_H)
#define _XIMPL_H

#include <sys/types.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

typedef unsigned long PixVal;

typedef struct {
    GC       set;
    PixVal   cur_pix;
} XiGC;

typedef struct {
  Font     fnt;
  int      font_w,font_h;
  int      font_descent;
  PixVal   font_pix;
} XiFont;

typedef struct {
    Display  *disp;
    int      screen;
    Window   win;
    Visual   *vis;            /* Graphics visual */
    XiGC     gc;
    XiFont   *font;
    int      depth;           /* Depth of visual */
    int      numcolors,      /* Number of available colors */
             maxcolors;       /* Current number in use */
    Colormap cmap;
    PixVal   foreground,background;
    PixVal   cmapping[256];
    int      x,y,w,h;      /* Size and location of window */
    Drawable drw;
} PetscDraw_X;

#define XiDrawable(w) ((w)->drw ? (w)->drw : (w)->win)

#define XiSetColor(Win,icolor)\
  {if (icolor >= 256 || icolor < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Color value out of range");\
   if ((Win)->gc.cur_pix != (Win)->cmapping[icolor]) { \
     XSetForeground((Win)->disp,(Win)->gc.set,(Win)->cmapping[icolor]); \
     (Win)->gc.cur_pix   = (Win)->cmapping[icolor];\
  }}

#define XiSetPixVal(Win,pix)\
  {if ((PixVal) (Win)->gc.cur_pix != pix) { \
     XSetForeground((Win)->disp,(Win)->gc.set,pix); \
     (Win)->gc.cur_pix   = pix;\
  }}

typedef struct {
  int      x,y,xh,yh,w,h;
} XiRegion;

typedef struct {
  XiRegion Box;
  int      width,HasColor,is_in;
  PixVal   Hi,Lo;
} XiDecoration;

#endif
