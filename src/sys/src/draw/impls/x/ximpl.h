
#if !defined(_XIMPL) && defined(HAVE_X11)
#define _XIMPL
#include "drawimpl.h"

#include <sys/types.h>  /* rs6000 likes */
#include <X11/Xlib.h>
#include <X11/Xutil.h>

typedef unsigned long PixVal;


typedef struct {
    GC       set;
    PixVal   cur_pix;
} XiGC;

typedef struct {
  Font     fnt;
  int      font_w, font_h;
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
    int      numcolors,       /* Number of available colors */
             maxcolors;       /* Current number in use */
    Colormap cmap;
    PixVal   foreground, background;
    PixVal   cmapping[256];
    int      x, y, w, h;      /* Size and location of window */
    Drawable drw;
} DrawCtx_X;

#define XiDrawable(w) ((w)->drw ? (w)->drw : (w)->win)

#define XiSetColor( Win,icolor )\
  {if ((Win)->gc.cur_pix != (Win)->cmapping[icolor]) { \
     XSetForeground( (Win)->disp, (Win)->gc.set, (Win)->cmapping[icolor] ); \
     (Win)->gc.cur_pix   = (Win)->cmapping[icolor];\
  }}

#define XiSetPixVal( Win,pix )\
  {if ((Win)->gc.cur_pix != pix) { \
     XSetForeground( (Win)->disp, (Win)->gc.set, pix ); \
     (Win)->gc.cur_pix   = pix;\
  }}

typedef struct {
  int      x, y, xh, yh, w, h;
} XiRegion;

typedef struct {
  XiRegion Box;
  int      width, HasColor, is_in;
  PixVal   Hi, Lo;
} XiDecoration;

#endif
