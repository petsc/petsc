#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: xcolor.c,v 1.26 1997/07/29 14:10:43 bsmith Exp bsmith $";
#endif
/*
    Code for managing color the X implementation of the Draw routines.

    Currently we default to using cmapping[0-15] for the basic colors and 
  cmapping[16-255] for a uniform hue of all the colors. But in the contour
  plot we only use from 16-240 since the ones beyond that are too dark.

*/
#if defined(HAVE_X11)
#include "src/draw/impls/x/ximpl.h"

static char *(colornames[]) = { "white", "black", "red", "green", 
                                "cyan", "blue", "magenta", "aquamarine",
                                "forestgreen", "orange", "violet", "brown",
                                "pink", "coral", "gray", "yellow" };

extern int XiInitCmap( Draw_X* );
extern int XiGetVisualClass( Draw_X * );
extern int XiHlsToRgb(int,int,int,unsigned char*,unsigned char*,unsigned char*);
Colormap XiCreateColormap(Draw_X*,Display*,int,Visual *);

#include <X11/Xatom.h>

#undef __FUNC__  
#define __FUNC__ "XiInitColors" 
int XiInitColors(Draw_X* XiWin,Colormap cmap,int nc )
{
  PixVal   white_pixel, black_pixel;

  /* reset the number of colors from info on the display */
  /* This is wrong; it needs to take the value from the visual */
  /* Also, I'd like to be able to set this so as to force B&W behaviour
   on color displays */
  if (nc > 0)   XiWin->numcolors = nc;
  else  XiWin->numcolors = 1 << DefaultDepth( XiWin->disp, XiWin->screen );

  /* we will use the default colormap of the visual */
  if (!XiWin->cmap)
    XiWin->cmap = XiCreateColormap(XiWin, XiWin->disp, XiWin->screen, XiWin->vis );

  /* get the initial colormap */
  if (XiWin->numcolors > 2)  XiInitCmap( XiWin );
  else {
    /* note that the 1-bit colormap is the DEFAULT map */
    white_pixel     = WhitePixel(XiWin->disp,XiWin->screen);
    black_pixel     = BlackPixel(XiWin->disp,XiWin->screen);
    /* the default "colormap";mapping from color indices to X pixel values */
    XiWin->cmapping[DRAW_BLACK]   = black_pixel;
    XiWin->cmapping[DRAW_WHITE]   = white_pixel;
    XiWin->foreground        = black_pixel;
    XiWin->background        = white_pixel;
  }
  return 0;
}

/*
    Keep a record of which pixel numbers in the cmap have been 
  used so far.
*/
static long int cmap_pixvalues_used[256];
static int cmap_base = 0;

/*
    Set the initial color map
 */
#undef __FUNC__  
#define __FUNC__ "XiInitCmap"
int XiInitCmap(Draw_X* XiWin )
{
  XColor   colordef;
  int      i;
  Colormap defaultmap = DefaultColormap( XiWin->disp, XiWin->screen );

  cmap_base = 0;
  PetscMemzero(cmap_pixvalues_used,256*sizeof(long int));

  /* Also, allocate black and white first, in the same order that
   there "pixel" values are, in case the pixel values assigned
   start from 0 */
  /* Look up the colors so that they can be use server standards
    (and be corrected for the monitor) */
  for (i=0; i<16; i++) {
    XParseColor( XiWin->disp, XiWin->cmap, colornames[i], &colordef );
    if (defaultmap != XiWin->cmap) { 
      /* allocate the color in the default-map in case it is already not there */
      XAllocColor( XiWin->disp, defaultmap, &colordef );
      /*  force the new color map to use the same slot as the default colormap  */
      XStoreColor( XiWin->disp, XiWin->cmap, &colordef );
    } else {
      XAllocColor( XiWin->disp, XiWin->cmap, &colordef );
    } 
    XiWin->cmapping[i]                  = colordef.pixel;
    cmap_pixvalues_used[colordef.pixel] = 1;
  }
  XiWin->background = XiWin->cmapping[DRAW_WHITE];
  XiWin->foreground = XiWin->cmapping[DRAW_BLACK];
  XiWin->maxcolors = 16;
  return 0;
}

/*
 * The input to this routine is RGB, not HLS.
 * X colors are 16 bits, not 8, so we have to shift the input by 8.
 */
#undef __FUNC__  
#define __FUNC__ "XiCmap" 
int XiCmap( unsigned char *red,unsigned char *green,unsigned char *blue, 
            int mapsize, Draw_X *XiWin )
{
  int      i, found;
  XColor   colordef;
  Colormap defaultmap = DefaultColormap( XiWin->disp, XiWin->screen );

  if (mapsize > XiWin->numcolors) mapsize = XiWin->numcolors;

  XiWin->maxcolors = XiWin->numcolors;

  for (i=16; i<mapsize+16; i++) {
    colordef.red    = ((int)red[i-16]   * 65535) / 255;
    colordef.green  = ((int)green[i-16] * 65535) / 255;
    colordef.blue   = ((int)blue[i-16]  * 65535) / 255;
    colordef.flags  = DoRed | DoGreen | DoBlue;
    if (defaultmap == XiWin->cmap) { 
      XAllocColor( XiWin->disp, XiWin->cmap, &colordef );
    } else {
      /* try to allocate the color in the default-map */
      found = XAllocColor( XiWin->disp, defaultmap, &colordef );
      /* use it, if it it exists and is not already used in the new colormap */
      if (found && !cmap_pixvalues_used[colordef.pixel]) {
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

  /*
    The window needs to be told the new background pixel so that things
    like XClearArea will work
  */
  if (XiWin->win)
    XSetWindowBackground( XiWin->disp, XiWin->win, XiWin->cmapping[0] );

  /*
   Note that since we haven't allocated a range of pixel-values to this
   window, the changes will only take effect with future writes.
   Further, several colors may have been mapped to the same display color.
   We could detect this only by seeing if there are any duplications
   among the XiWin->cmap values.
  */
  return 0;
}

/*
    Color in X is many-layered.  The first layer is the "visual", a
    immutable attribute of a window set when the window is
    created.

    The next layer is the colormap.  The installation of colormaps is
    the buisness of the window manager (in some distant later release).
    Rather than fight with that, we will use the default colormap.
    This usually does not have many (any?) sharable color entries,
    so we just try to match with the existing entries.
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
  if (XMatchVisualInfo( XiWin->disp, XiWin->screen, 24, DirectColor, &vinfo)) {
    XiWin->vis    = vinfo.visual;
    return 0;
  }
  if (XMatchVisualInfo( XiWin->disp, XiWin->screen, 8, PseudoColor, &vinfo)) {
    XiWin->vis    = vinfo.visual;
    return 0;
  }
  if (XMatchVisualInfo( XiWin->disp, XiWin->screen,
    DefaultDepth(XiWin->disp,XiWin->screen), PseudoColor, &vinfo)) {
    XiWin->vis    = vinfo.visual;
    return 0;
  }
  XiWin->vis    = DefaultVisual( XiWin->disp, XiWin->screen );
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "XiGetVisualClass"
int XiGetVisualClass(Draw_X* XiWin )
{
#if defined(__cplusplus)
  return XiWin->vis->c_class;
#else
  return XiWin->vis->class;
#endif
}

#undef __FUNC__  
#define __FUNC__ "XiCreateColormap" 
Colormap XiCreateColormap(Draw_X* XiWin, Display* display,int screen,Visual *visual )
{
  Colormap Cmap;

  if (DefaultDepth( display, screen ) <= 1)
    Cmap    = DefaultColormap( display, screen );
  else {
    Cmap    = XCreateColormap( display, RootWindow(display,screen),visual, AllocAll );
  }
  return Cmap;
}

#undef __FUNC__  
#define __FUNC__ "XiSetColormap" 
int XiSetColormap(Draw_X* XiWin )
{
  XSetWindowColormap( XiWin->disp, XiWin->win, XiWin->cmap );
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "XiGetBaseColor" 
int XiGetBaseColor(Draw_X* XiWin,PixVal* white_pix,PixVal* black_pix )
{
  *white_pix  = XiWin->cmapping[DRAW_WHITE];
  *black_pix  = XiWin->cmapping[DRAW_BLACK];
  return 0;
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
#include <math.h>

#undef __FUNC__  
#define __FUNC__ "XiSetGamma" 
int XiSetGamma( double g )
{
  Gamma = g;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "XiSetCmapHue" 
int XiSetCmapHue(unsigned char *red,unsigned char *green,unsigned char * blue,
              int mapsize )
{
  int     i, hue, lightness, saturation;
  double  igamma = 1.0 / Gamma;

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
    hue += (359/(mapsize-2));
  }
  return 0;
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
  while (h > 360) h = h - 360;
  while (h < 0)   h = h + 360;
  if (h < 60) return n1 + (n2-n1)*h/60;
  if (h < 180) return n2;
  if (h < 240) return n1 + (n2-n1)*(240-h)/60;
  return n1;
}

#undef __FUNC__  
#define __FUNC__ "XiHlsToRgb" 
int XiHlsToRgb(int h,int l,int s,unsigned char *r,unsigned char *g,
           unsigned char *b )
{
  int m1, m2;         /* in 0 to 100 */
  if (l <= 50) m2 = l * ( 100 + s ) / 100 ;           /* not sure of "/100" */
  else         m2 = l + s - l*s/100;

  m1  = 2*l - m2;
  if (s == 0) {
    /* ignore h */
    *r  = 255 * l / 100;
    *g  = 255 * l / 100;
    *b  = 255 * l / 100;
  }
  else {
    *r  = (255 * XiHlsHelper( h+120, m1, m2 ) ) / 100;
    *g  = (255 * XiHlsHelper( h, m1, m2 ) )     / 100;
    *b  = (255 * XiHlsHelper( h-120, m1, m2 ) ) / 100;
  }
  return 0;
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

  st = XParseColor( XiWin->disp, XiWin->cmap, name, &colordef );
  if (st) {
    st  = XAllocColor( XiWin->disp, XiWin->cmap, &colordef );
    if (st)  *pixval = colordef.pixel;
  }
  return st;
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
  if (XiWin->numcolors == 2 || !XiFindColor( XiWin, name, &pixval ))
    pixval  = is_fore ? XiWin->cmapping[DRAW_WHITE] : 
                                            XiWin->cmapping[DRAW_BLACK];
  return pixval;
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
  return  colorsdef.pixel;
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
  unsigned char *red, *green, *blue;

  red   = (unsigned char *)PetscMalloc(3*ncolors*sizeof(unsigned char));CHKPTRQ(red);
  green = red + ncolors;
  blue  = green + ncolors;
  XiSetCmapHue( red, green, blue, ncolors );
  XiCmap( red, green, blue, ncolors, Xiwin );
  PetscFree( red );
  return 0;
}

/*
  XiSetCmapLight - Create rgb values from a single color by adding white
  
  The initial color is (red[0],green[0],blue[0]).
*/
#undef __FUNC__  
#define __FUNC__ "XiSetCmapLight" 
int XiSetCmapLight(unsigned char *red, unsigned char *green,
                    unsigned char *blue, int mapsize )
{
  int     i ;

  for (i = 1; i < mapsize-1; i++) {
      blue[i]  = i*(255-(int)blue[0])/(mapsize-2)+blue[0] ;
      green[i] = i*(255-(int)green[0])/(mapsize-2)+green[0] ;
      red[i]   = i*(255-(int)red[0])/(mapsize-2)+red[0] ;
  }
  red[mapsize-1] = green[mapsize-1] = blue[mapsize-1] = 255;
  return 0;
}

int XiGetNumcolors( Draw_X *XiWin )
{
  return XiWin->numcolors;
}
#else
int dummy_xcolor()
{
  return 0;
}
#endif
