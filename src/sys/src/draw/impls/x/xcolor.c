#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: xcolor.c,v 1.42 1999/01/31 16:05:02 bsmith Exp bsmith $";
#endif


/*
    Code for managing color the X implementation of the Draw routines.

    Currently we default to using cmapping[0 to DRAW_BASIC_COLORS-1] for the basic colors and 
  cmapping[DRAW_BASIC_COLORS to 255] for a uniform hue of all the colors. But in the contour
  plot we only use from DRAW_BASIC_COLORS to 240 since the ones beyond that are too dark.

*/
#include "petsc.h"
#if defined(HAVE_X11)
#include "src/sys/src/draw/impls/x/ximpl.h"
#include <X11/Xatom.h>

static char *(colornames[DRAW_BASIC_COLORS]) = { "white", 
                                                 "black", 
                                                 "red", 
                                                 "green", 
                                                 "cyan",
                                                 "blue",
                                                 "magenta", 
                                                 "aquamarine",
                                                 "forestgreen", 
                                                 "orange", 
                                                 "violet",
                                                 "brown",
                                                 "pink", 
                                                 "coral",
                                                 "gray", 
                                                 "yellow",
                                                 "gold",
                                                 "lightpink",
                                                 "mediumturquoise",
                                                 "khaki",
                                                 "dimgray",
                                                 "yellowgreen",
                                                 "skyblue",
                                                 "darkgreen",
                                                 "navyblue",
                                                 "sandybrown",
                                                 "cadetblue",
                                                 "powderblue",
                                                 "deeppink",
                                                 "thistle",
                                                 "limegreen",
                                                 "lavenderblush" };

extern int      XiInitCmap( Draw_X* );
extern int      XiGetVisualClass( Draw_X * );
extern int      XiHlsToRgb(int,int,int,unsigned char*,unsigned char*,unsigned char*);
extern int      XiUniformHues( Draw_X *, int);

#undef __FUNC__  
#define __FUNC__ "XSetUpColorMap_Private"
/*
   Sets up a color map for a display. This is shared by all the windows
  opened on that display; this is to save time when windows are open so 
  each one does not have to create a color map which can take 15 to 20 seconds

     This is new code written 2/26/1999 Barry Smith, I hope it can replace
  some older, rather confusing code.
*/
static int       gNumcolors = 0;
static Colormap  gColormap  = 0;
int XSetUpColorMap_Shared(Display *display,int screen,Visual *visual)
{
  PetscFunctionBegin;
  gColormap   = DefaultColormap(display,screen);CHKPTRQ(gColormap);

  /* set the basic colors into the color map */

/*  ierr = XiInitCmap( XiWin );CHKERRQ(ierr); */

  /* set the uniform hue colors into the color map */

  PetscFunctionReturn(0);
}

int XSetUpColorMap_Private(Display *display,int screen,Visual *visual)
{
  PetscFunctionBegin;
  gColormap   = XCreateColormap(display,RootWindow(display,screen),
                                  visual,AllocAll);CHKPTRQ(gColormap);
  PetscFunctionReturn(0);
}

int XSetUpColorMap_X(Display *display,int screen,Visual *visual)
{
  int         ierr, sharedcolormap = 0;
  XVisualInfo vinfo;

  PetscFunctionBegin;
  /* 
     This is wrong; it needs to take the value from the visual 
  */
  gNumcolors = 1 << DefaultDepth( display, screen);

  ierr = OptionsHasName(PETSC_NULL,"-draw_x_shared_colormap",&sharedcolormap);CHKERRQ(ierr);
  /*
        Need to determine if window supports allocating a private colormap,
    if not, set flag to 1
  */
  if (XMatchVisualInfo(display, screen, 24, StaticColor, &vinfo) ||
      XMatchVisualInfo(display, screen, 24, TrueColor, &vinfo) ||
      XMatchVisualInfo(display, screen, 16, StaticColor, &vinfo) ||
      XMatchVisualInfo(display, screen, 16, TrueColor, &vinfo)) {
    sharedcolormap = 1;
  }

  /* generate the X color map object */
  if (sharedcolormap) {
    ierr = XSetUpColorMap_Shared(display,screen,visual);CHKERRQ(ierr);
  } else {
    ierr = XSetUpColorMap_Private(display,screen,visual);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "XiInitColors" 
int XiInitColors(Draw_X* XiWin,Colormap cmap)
{
  int ierr;

  PetscFunctionBegin;
  /* 
     Reset the number of colors from info on the display 
  
     This is wrong; it needs to take the value from the visual 
  */
  XiWin->numcolors = 1 << DefaultDepth( XiWin->disp, XiWin->screen );

  if (!XiWin->cmap) {
    XiWin->cmap = XCreateColormap(XiWin->disp,RootWindow(XiWin->disp,XiWin->screen),
                                  XiWin->vis,AllocAll);CHKPTRQ(XiWin->cmap);
  }

  /* get the initial colormap */
  ierr = XiInitCmap( XiWin );CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*
    Keep a record of which pixel numbers in the cmap have been 
  used so far; this is to allow us to try to reuse as much of the current
  colormap as possible.
*/
static long int cmap_pixvalues_used[256];
static int      cmap_base = 0;

/*
    Set the initial color map
*/
#undef __FUNC__  
#define __FUNC__ "XiInitCmap"
int XiInitCmap(Draw_X* XiWin )
{
  XColor   colordef,ecolordef;
  int      i,found;
  Colormap defaultmap = DefaultColormap( XiWin->disp, XiWin->screen );

  PetscFunctionBegin;
  cmap_base = 0;
  PetscMemzero(cmap_pixvalues_used,256*sizeof(long int));

  /* 
      Allocate black and white first, in the same order that
      their "pixel" values are, in case the pixel values assigned
      start from 0 
  */
  /*
     Look up the colors so that they can use server standards
     (and be corrected for the monitor) 

     This seems to be very slow
  */
  for (i=0; i<DRAW_BASIC_COLORS; i++) {
    if (defaultmap == XiWin->cmap) { 
      XAllocNamedColor( XiWin->disp, XiWin->cmap, colornames[i], &colordef,&ecolordef ); 
    } else {
      XParseColor( XiWin->disp, XiWin->cmap, colornames[i], &colordef );
      /* try to allocate the color in the default-map */
      found = XAllocColor( XiWin->disp, defaultmap, &colordef ); 
      /* use it, if it it exists and is not already used in the new colormap */
      if (found && colordef.pixel < 256  && !cmap_pixvalues_used[colordef.pixel]) {
        cmap_pixvalues_used[colordef.pixel] = 1; 
	/* otherwise search for the next available slot */
      } else {
        while (cmap_pixvalues_used[cmap_base]) cmap_base++;
        colordef.pixel                   = cmap_base;
        cmap_pixvalues_used[cmap_base++] = 1;
      }
      XStoreColor( XiWin->disp, XiWin->cmap, &colordef ); 
    }
    XiWin->cmapping[i]   = colordef.pixel;
  }
  XiWin->background = XiWin->cmapping[DRAW_WHITE];
  XiWin->foreground = XiWin->cmapping[DRAW_BLACK];
  XiWin->maxcolors  = DRAW_BASIC_COLORS;
  PLogInfo(0,"XiInitCmap:Successfully allocated basic colors\n");
  PetscFunctionReturn(0);
}

/*
    The input to this routine is RGB, not HLS.
*/
#undef __FUNC__  
#define __FUNC__ "XiCmap" 
int XiCmap( unsigned char *red,unsigned char *green,unsigned char *blue, 
            int mapsize, Draw_X *XiWin )
{
  int      i, found, ierr, fast;
  XColor   colordef;
  Colormap defaultmap = DefaultColormap( XiWin->disp, XiWin->screen );

  PetscFunctionBegin;
  if (mapsize > XiWin->numcolors) mapsize = XiWin->numcolors;

  XiWin->maxcolors = XiWin->numcolors;

  ierr = OptionsHasName(PETSC_NULL,"-draw_fast",&fast);CHKERRQ(ierr);

  if (!fast) {
    /*
     This is very slow because we have to request from the X server for each
     color. Could not figure out a way to request a large number at the
     same time.
    */
    for (i=DRAW_BASIC_COLORS; i<mapsize+DRAW_BASIC_COLORS; i++) {
      colordef.red    = ((int)red[i-DRAW_BASIC_COLORS]   * 65535) / 255;
      colordef.green  = ((int)green[i-DRAW_BASIC_COLORS] * 65535) / 255;
      colordef.blue   = ((int)blue[i-DRAW_BASIC_COLORS]  * 65535) / 255;
      colordef.flags  = DoRed | DoGreen | DoBlue;
      if (defaultmap == XiWin->cmap) { 
        XAllocColor( XiWin->disp, XiWin->cmap, &colordef ); 
      } else {
        /* try to allocate the color in the default-map */
        found = XAllocColor( XiWin->disp, defaultmap, &colordef ); 
        /* use it, if it it exists and is not already used in the new colormap */
        if (found && colordef.pixel < 256  && !cmap_pixvalues_used[colordef.pixel]) {
          cmap_pixvalues_used[colordef.pixel] = 1; 
          /* otherwise search for the next available slot */
        } else {
          while (cmap_pixvalues_used[cmap_base]) cmap_base++;
          colordef.pixel                   = cmap_base;
          cmap_pixvalues_used[cmap_base++] = 1;
        }
        XStoreColor( XiWin->disp, XiWin->cmap, &colordef ); 
      }
      XiWin->cmapping[i]   = colordef.pixel;
    }
  }

  /*
    The window needs to be told the new background pixel so that things
    like XClearArea will work
  */
  if (XiWin->win) {
    XSetWindowBackground( XiWin->disp, XiWin->win, XiWin->cmapping[0] );
  }

  /*
   Note that since we haven't allocated a range of pixel-values to this
   window, the changes will only take effect with future writes.
   Further, several colors may have been mapped to the same display color.
   We could detect this only by seeing if there are any duplications
   among the XiWin->cmap values.
  */
  PLogInfo(0,"XiCmap:Successfully allocated spectrum colors\n");
  PetscFunctionReturn(0);
}

/*
    Color in X is many-layered.  The first layer is the "visual", a
    immutable attribute of a window set when the window is
    created.

    The next layer is the colormap.  The installation of colormaps is
    the buisness of the window manager (in some distant later release).
*/

/*
    This routine gets the visual class (PseudoColor, etc) and returns
    it.  It finds the default visual.  Possible returns are
	PseudoColor
	StaticColor
	DirectColor
	TrueColor
	GrayScale
	StaticGray
 */
#undef __FUNC__  
#define __FUNC__ "XiSetVisualClass" 
int XiSetVisualClass(Draw_X* XiWin )
{
  XVisualInfo vinfo;

  PetscFunctionBegin;
  if (XMatchVisualInfo( XiWin->disp, XiWin->screen, 24, DirectColor, &vinfo)) {
    XiWin->vis    = vinfo.visual;
    PetscFunctionReturn(0);
  }
  if (XMatchVisualInfo( XiWin->disp, XiWin->screen, 8, PseudoColor, &vinfo)) {
    XiWin->vis    = vinfo.visual;
    PetscFunctionReturn(0);
  }
  if (XMatchVisualInfo( XiWin->disp, XiWin->screen,
    DefaultDepth(XiWin->disp,XiWin->screen), PseudoColor, &vinfo)) {
    XiWin->vis    = vinfo.visual;
    PetscFunctionReturn(0);
  }
  XiWin->vis    = DefaultVisual( XiWin->disp, XiWin->screen );
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "XiGetVisualClass"
int XiGetVisualClass(Draw_X* XiWin )
{
  PetscFunctionBegin;
#if defined(__cplusplus)
  PetscFunctionReturn(XiWin->vis->c_class);
#else
  PetscFunctionReturn(XiWin->vis->class);
#endif
}


#undef __FUNC__  
#define __FUNC__ "XiSetColormap" 
int XiSetColormap(Draw_X* XiWin )
{
  PetscFunctionBegin;
  XSetWindowColormap( XiWin->disp, XiWin->win, XiWin->cmap );
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "XiGetBaseColor" 
int XiGetBaseColor(Draw_X* XiWin,PixVal* white_pix,PixVal* black_pix )
{
  PetscFunctionBegin;
  *white_pix  = XiWin->cmapping[DRAW_WHITE];
  *black_pix  = XiWin->cmapping[DRAW_BLACK];
  PetscFunctionReturn(0);
}

/*
    Set up a color map, using uniform separation in hue space.
    Map entries are Red, Green, Blue.
    Values are "gamma" corrected.
 */

/*  
   Gamma is a monitor dependent value.  The value here is an 
   approximate that gives somewhat better results than Gamma = 1.
 */
static double Gamma = 2.0;

#undef __FUNC__  
#define __FUNC__ "XiSetGamma" 
int XiSetGamma( double g )
{
  PetscFunctionBegin;
  Gamma = g;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "XiSetCmapHue" 
int XiSetCmapHue(unsigned char *red,unsigned char *green,unsigned char * blue,int mapsize )
{
  int     i, hue, lightness, saturation;
  double  igamma = 1.0 / Gamma;

  PetscFunctionBegin;
  red[0]      = 0;
  green[0]    = 0;
  blue[0]     = 0;
  hue         = 0;        /* in 0:359 */
  lightness   = 50;       /* in 0:100 */
  saturation  = 100;      /* in 0:100 */
  for (i = 0; i < mapsize; i++) {
    XiHlsToRgb( hue, lightness, saturation, red + i, green + i, blue + i );
    red[i]   = (int)floor( 255.999 * pow( ((double)  red[i])/255.0, igamma ) );
    blue[i]  = (int)floor( 255.999 * pow( ((double) blue[i])/255.0, igamma ) );
    green[i] = (int)floor( 255.999 * pow( ((double)green[i])/255.0, igamma ) );
    hue     += (359/(mapsize-2));
  }
  PetscFunctionReturn(0);
}

/*
 * This algorithm is from Foley and van Dam, page 616
 * given
 *   (0:359, 0:100, 0:100).
 *      h       l      s
 * set
 *   (0:255, 0:255, 0:255)
 *      r       g      b
 */
#undef __FUNC__  
#define __FUNC__ "XiHlsHelper" 
int XiHlsHelper(int h,int n1,int n2 )
{
  PetscFunctionBegin;
  while (h > 360) h = h - 360;
  while (h < 0)   h = h + 360;
  if (h < 60)  PetscFunctionReturn(n1 + (n2-n1)*h/60);
  if (h < 180) PetscFunctionReturn(n2);
  if (h < 240) PetscFunctionReturn(n1 + (n2-n1)*(240-h)/60);
  PetscFunctionReturn(n1);
}

#undef __FUNC__  
#define __FUNC__ "XiHlsToRgb" 
int XiHlsToRgb(int h,int l,int s,unsigned char *r,unsigned char *g,unsigned char *b)
{
  int m1, m2;         /* in 0 to 100 */

  PetscFunctionBegin;
  if (l <= 50) m2 = l * ( 100 + s ) / 100 ;           /* not sure of "/100" */
  else         m2 = l + s - l*s/100;

  m1  = 2*l - m2;
  if (s == 0) {
    /* ignore h */
    *r  = 255 * l / 100;
    *g  = 255 * l / 100;
    *b  = 255 * l / 100;
  } else {
    *r  = (255 * XiHlsHelper( h+120, m1, m2 ) ) / 100;
    *g  = (255 * XiHlsHelper( h, m1, m2 ) )     / 100;
    *b  = (255 * XiHlsHelper( h-120, m1, m2 ) ) / 100;
  }
  PetscFunctionReturn(0);
}

/*
    This routine returns the pixel value for the specified color
    Returns 0 on failure, <>0 otherwise.
 */
#undef __FUNC__  
#define __FUNC__ "XiFindColor" 
int XiFindColor( Draw_X *XiWin, char *name, PixVal *pixval )
{
  XColor   colordef;
  int      st;

  PetscFunctionBegin;
  st = XParseColor( XiWin->disp, XiWin->cmap, name, &colordef );
  if (st) {
    st  = XAllocColor( XiWin->disp, XiWin->cmap, &colordef );
    if (st)  *pixval = colordef.pixel;
  }
  PetscFunctionReturn(st);
}

/*
    Another real need is to assign "colors" that make sense for
    a monochrome display, without unduely penalizing color displays.
    This routine takes a color name, a window, and a flag that
    indicates whether this is "background" or "foreground".
    In the monchrome case (or if the color is otherwise unavailable),
    the "background" or "foreground" colors will be chosen
 */
#undef __FUNC__  
#define __FUNC__ "XiGetColor" 
PixVal XiGetColor(Draw_X* XiWin, char *name, int is_fore )
{
  PixVal pixval;

  PetscFunctionBegin;
  if (XiWin->numcolors == 2 || !XiFindColor( XiWin, name, &pixval )) {
    pixval  = is_fore ? XiWin->cmapping[DRAW_WHITE] : XiWin->cmapping[DRAW_BLACK];
  }
  PetscFunctionReturn(pixval);
}

/*
   This routine takes a named color and returns a color that is either
   lighter or darker
 */
#undef __FUNC__  
#define __FUNC__ "XiSimColor" 
PixVal XiSimColor(Draw_X *XiWin,PixVal pixel, int intensity, int is_fore)
{
  XColor   colordef, colorsdef;
  char     RGBcolor[20];
  PixVal   red, green, blue;

  PetscFunctionBegin;
  colordef.pixel = pixel;
  XQueryColor( XiWin->disp, XiWin->cmap, &colordef );
  /* Adjust the color value up or down.  Get the RGB values for the color */
  red   = colordef.red;
  green = colordef.green;
  blue  = colordef.blue;
#define WHITE_AMOUNT 5000
  if (intensity > 0) {
    /* Add white to the color */
    red   = PetscMin(65535,red + WHITE_AMOUNT);
    green = PetscMin(65535,green + WHITE_AMOUNT);
    blue  = PetscMin(65535,blue + WHITE_AMOUNT);
  }
  else {
    /* Subtract white from the color */
    red   = (red   < WHITE_AMOUNT) ? 0 : red - WHITE_AMOUNT;
    green = (green < WHITE_AMOUNT) ? 0 : green - WHITE_AMOUNT;
    blue  = (blue  < WHITE_AMOUNT) ? 0 : blue - WHITE_AMOUNT;
  }
  sprintf( RGBcolor, "rgb:%4.4x/%4.4x/%4.4x", (unsigned int)red, 
                     (unsigned int)green, (unsigned int)blue );
  XLookupColor( XiWin->disp, XiWin->cmap, RGBcolor, &colordef, 
                     &colorsdef );
  PetscFunctionReturn(colorsdef.pixel);
}

/*
  XiUniformHues - Set the colormap to a uniform distribution

  This routine sets the colors in the current colormap, if the default
  colormap is used.  The Pixel values chosen are in the cmapping 
  structure; this is used by routines such as the Xi contour plotter.
*/  
#undef __FUNC__  
#define __FUNC__ "XiUniformHues" 
int XiUniformHues( Draw_X *Xiwin, int ncolors )
{
  int           ierr;
  unsigned char *red, *green, *blue;

  PetscFunctionBegin;
  red   = (unsigned char *)PetscMalloc(3*ncolors*sizeof(unsigned char));CHKPTRQ(red);
  green = red + ncolors;
  blue  = green + ncolors;
  ierr = XiSetCmapHue( red, green, blue, ncolors );CHKERRQ(ierr);
  ierr = XiCmap( red, green, blue, ncolors, Xiwin );CHKERRQ(ierr);
  PetscFree( red );
  PetscFunctionReturn(0);
}

/*
  XiSetCmapLight - Create rgb values from a single color by adding white
  
  The initial color is (red[0],green[0],blue[0]).
*/
#undef __FUNC__  
#define __FUNC__ "XiSetCmapLight" 
int XiSetCmapLight(unsigned char *red, unsigned char *green,unsigned char *blue, int mapsize )
{
  int     i ;

  PetscFunctionBegin;
  for (i = 1; i < mapsize-1; i++) {
      blue[i]  = i*(255-(int)blue[0])/(mapsize-2)+blue[0] ;
      green[i] = i*(255-(int)green[0])/(mapsize-2)+green[0] ;
      red[i]   = i*(255-(int)red[0])/(mapsize-2)+red[0] ;
  }
  red[mapsize-1] = green[mapsize-1] = blue[mapsize-1] = 255;
  PetscFunctionReturn(0);
}

int XiGetNumcolors( Draw_X *XiWin )
{
  PetscFunctionBegin;
  PetscFunctionReturn(XiWin->numcolors);
}
#else
int dummy_xcolor(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#endif
