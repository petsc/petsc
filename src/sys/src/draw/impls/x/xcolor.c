#ifndef lint
static char vcid[] = "$Id: color.c,v 1.8 1995/05/14 16:34:03 bsmith Exp bsmith $";
#endif
#include "ximpl.h"

static char *(colornames[]) = { "white", "black", "red", "yellow", "green", 
                                "cyan", "blue", "magenta", "aquamarine",
                                "forestgreen", "orange", "violet", "brown",
                                "pink", "coral", "gray" };

extern int XiInitCmap( DrawCtx_X* );
extern int XiAllocBW( DrawCtx_X*, PixVal*,PixVal*);
extern int XiGetVisualClass( DrawCtx_X * );
extern int XiHlsToRgb(int,int,int,unsigned char*,unsigned char*,unsigned char*);
Colormap XiCreateColormap(Display*,int,Visual *);

/*
    This file contains routines to provide color support where available.
    This is made difficult by the wide variety of color implementations
    that X11 supports.
*/
#include <X11/Xatom.h>

int XiInitColors(DrawCtx_X* XiWin,Colormap cmap,int nc )
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
    XiWin->cmap = XiCreateColormap( XiWin->disp, XiWin->screen, XiWin->vis );

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
    Set the initial color map
 */
int XiInitCmap(DrawCtx_X* XiWin )
{
  XColor  colordef;
  int     i;
  /* Also, allocate black and white first, in the same order that
   there "pixel" values are, incase the pixel values assigned
   start from 0 */
  XiAllocBW( XiWin, &XiWin->cmapping[DRAW_WHITE],
                    &XiWin->cmapping[DRAW_BLACK] );
  XiWin->background = XiWin->cmapping[DRAW_WHITE];
  XiWin->foreground = XiWin->cmapping[DRAW_BLACK];
  /* Look up the colors so that they can be use server standards
    (and be corrected for the monitor) */
  for (i=2; i<16; i++) {
    XParseColor( XiWin->disp, XiWin->cmap, colornames[i], &colordef );
    XAllocColor( XiWin->disp, XiWin->cmap, &colordef );
    XiWin->cmapping[i]   = colordef.pixel;
  }
  XiWin->maxcolors = 15;
  return 0;
}

/*
 * The input to this routine is RGB, not HLS.
 * X colors are 16 bits, not 8, so we have to shift the input by 8.
 */
int XiCmap( unsigned char *red,unsigned char *green,unsigned char *blue, 
            int mapsize, DrawCtx_X *XiWin )
{
  int         i, err;
  XColor      colordef;
  PixVal      white_pixel, black_pixel, pix, white_pix, black_pix;

  white_pixel     = WhitePixel(XiWin->disp,XiWin->screen);
  black_pixel     = BlackPixel(XiWin->disp,XiWin->screen);

  /*
    Free the old colors if we have the colormap.
  */
  if (XiWin->cmap != DefaultColormap( XiWin->disp, XiWin->screen ) ) {
    if (XiGetVisualClass( XiWin ) == PseudoColor ||
	XiGetVisualClass( XiWin ) == DirectColor )
	XFreeColors( XiWin->disp, XiWin->cmap, XiWin->cmapping,
		     XiWin->maxcolors + 1, (unsigned long)0 );
  }

  /*
     The sun convention is that 0 is the background and 2**depth-1 is
     foreground.  We make these the Xtools conventions (ignoring foreground)
  */

  if (mapsize > XiWin->numcolors) mapsize = XiWin->numcolors;

  XiWin->maxcolors = mapsize - 1;
  /*  Now, set the color values

    Since it is hard (impossible?) to insure that black and white are
    allocated to the SAME pixel values in the default/window manager
    colormap, we ALWAYS allocate black and white FIRST

    Note that we may have allocated more than mapsize colors if the
    map did not include black or white.  We need to handle this later.
 */
  XiAllocBW( XiWin, &white_pix, &black_pix );
  err = 0;
  for (i=16; i<mapsize+16; i++) {
    if (red[i] == 0 && green[i] == 0 && blue[i] == 0)
	XiWin->cmapping[i]   = black_pix;
    else if (red[i] == 255 && green[i] == 255 && blue[i] == 255)
	XiWin->cmapping[i]   = white_pix;
    else {
	colordef.red    = ((int)red[i]   * 65535) / 255;
	colordef.green  = ((int)green[i] * 65535) / 255;
	colordef.blue   = ((int)blue[i]  * 65535) / 255;
	colordef.flags  = DoRed | DoGreen | DoBlue;
	if (!XAllocColor( XiWin->disp, XiWin->cmap, &colordef ))
	    err = 1;
	XiWin->cmapping[i]   = colordef.pixel;
    }
  }

  /* make sure that there are 2 different colors */
  pix             = XiWin->cmapping[0];
  for (i=1; i<mapsize; i++)  if (pix != XiWin->cmapping[i]) break;
  if (i >= mapsize) {
    if (XiWin->cmapping[0] != black_pixel) XiWin->cmapping[0] = black_pixel;
    else	XiWin->cmapping[0]   = white_pixel;
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
  return err;
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
int XiSetVisualClass(DrawCtx_X* XiWin )
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

int XiGetVisualClass(DrawCtx_X* XiWin )
{
#if defined(__cplusplus)
  return XiWin->vis->c_class;
#else
  return XiWin->vis->class;
#endif
}

Colormap XiCreateColormap(Display* display,int screen,Visual *visual )
{
  Colormap Cmap;

  if (DefaultDepth( display, screen ) <= 1)
    Cmap    = DefaultColormap( display, screen );
  else
    Cmap    = XCreateColormap( display, RootWindow(display,screen),
			       visual, AllocNone );
  return Cmap;
}

int XiSetColormap(DrawCtx_X* XiWin )
{
  XSetWindowColormap( XiWin->disp, XiWin->win, XiWin->cmap );
  return 0;
}

int XiAllocBW(DrawCtx_X* XiWin,PixVal* white,PixVal* black )
{
  XColor  bcolor, wcolor;
  XParseColor( XiWin->disp, XiWin->cmap, "black", &bcolor );
  XParseColor( XiWin->disp, XiWin->cmap, "white", &wcolor );
  if (BlackPixel(XiWin->disp,XiWin->screen) == 0) {
    XAllocColor( XiWin->disp, XiWin->cmap, &bcolor );
    XAllocColor( XiWin->disp, XiWin->cmap, &wcolor );
  }
  else {
    XAllocColor( XiWin->disp, XiWin->cmap, &wcolor );
    XAllocColor( XiWin->disp, XiWin->cmap, &bcolor );
  }
  *black = bcolor.pixel;
  *white = wcolor.pixel;
  return 0;
}

int XiGetBaseColor(DrawCtx_X* XiWin,PixVal* white_pix,PixVal* black_pix )
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

int XiSetGamma( double g )
{
  Gamma = g;
  return 0;
}

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
  for (i = 1; i < mapsize-1; i++) {
    XiHlsToRgb( hue, lightness, saturation, red + i, green + i, blue + i );
    red[i]   = (int)floor( 255.999 * pow( ((double)  red[i])/255.0, igamma ) );
    blue[i]  = (int)floor( 255.999 * pow( ((double) blue[i])/255.0, igamma ) );
    green[i] = (int)floor( 255.999 * pow( ((double)green[i])/255.0, igamma ) );
    hue += (359/(mapsize-2));
  }
  red  [mapsize-1]    = 255;
  green[mapsize-1]    = 255;
  blue [mapsize-1]    = 255;
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
int XiHlsHelper(int h,int n1,int n2 )
{
  while (h > 360) h = h - 360;
  while (h < 0)   h = h + 360;
  if (h < 60) return n1 + (n2-n1)*h/60;
  if (h < 180) return n2;
  if (h < 240) return n1 + (n2-n1)*(240-h)/60;
  return n1;
}

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
int XiFindColor( DrawCtx_X *XiWin, char *name, PixVal *pixval )
{
  XColor   colordef;
  int      st;

  st = XParseColor( XiWin->disp, XiWin->cmap, name, &colordef );
  if (st) {
    st  = XAllocColor( XiWin->disp, XiWin->cmap, &colordef );
    if (st)  *pixval = colordef.pixel;
  }
  else    printf( "did not find color %s\n", name );
  return st;
}

/*
    When there are several windows being displayed, it may help to
    merge their colormaps together so that all of the windows
    may be displayed simultaneously with true colors.
    These routines attempt to accomplish this
 */

/*
 * The input to this routine is RGB, not HLS.
 * X colors are 16 bits, not 8, so we have to shift the input by 8.
 * This is like XiCmap, except that it APPENDS to the existing
 * colormap.
 */
int XiAddCmap( unsigned char *red, unsigned char *green, unsigned char *blue,
               int mapsize, DrawCtx_X *XiWin )
{
  int      i, err;
  XColor   colordef;
  int      cmap_start;

  if (mapsize + XiWin->maxcolors > XiWin->numcolors)
     mapsize = XiWin->numcolors - XiWin->maxcolors;

  cmap_start  = XiWin->maxcolors;
  XiWin->maxcolors += mapsize;

  err = 0;
  for (i=0; i<mapsize; i++) {
    colordef.red    = ((int)red[i]   * 65535) / 255;
    colordef.green  = ((int)green[i] * 65535) / 255;
    colordef.blue   = ((int)blue[i]  * 65535) / 255;
    colordef.flags  = DoRed | DoGreen | DoBlue;
    if (!XAllocColor( XiWin->disp, XiWin->cmap, &colordef ))
	err = 1;
    XiWin->cmapping[cmap_start+i]    = colordef.pixel;
  }
  return err;
}

/*
    Another real need is to assign "colors" that make sense for
    a monochrome display, without unduely penalizing color displays.
    This routine takes a color name, a window, and a flag that
    indicates whether this is "background" or "foreground".
    In the monchrome case (or if the color is otherwise unavailable),
    the "background" or "foreground" colors will be chosen
 */
PixVal XiGetColor(DrawCtx_X* XiWin, char *name, int is_fore )
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
PixVal XiSimColor(DrawCtx_X *XiWin,PixVal pixel, int intensity, int is_fore)
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
#define min(a,b) ((a)<(b) ? a : b)
#define max(a,b) ((a)>(b) ? a : b)
#define WHITE_AMOUNT 5000
  if (intensity > 0) {
    /* Add white to the color */
    red   = min(65535,red + WHITE_AMOUNT);
    green = min(65535,green + WHITE_AMOUNT);
    blue  = min(65535,blue + WHITE_AMOUNT);
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
int XiUniformHues( DrawCtx_X *Xiwin, int ncolors )
{
  unsigned char *red, *green, *blue;

  red   = (unsigned char *)MALLOC( 3 * ncolors * sizeof(unsigned char) );   
  CHKPTR(red);
  green = red + ncolors;
  blue  = green + ncolors;
  XiSetCmapHue( red, green, blue, ncolors );
  XiCmap( red, green, blue, ncolors, Xiwin );
  FREE( red );
  return 0;
}

/*
  XiSetCmapLight - Create rgb values from a single color by adding white
  
  The initial color is (red[0],green[0],blue[0]).
*/
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

int XiGetNumcolors( DrawCtx_X *XiWin )
{
  return XiWin->numcolors;
}
