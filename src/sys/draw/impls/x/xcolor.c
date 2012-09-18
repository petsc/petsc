
/*
    Code for managing color the X implementation of the PetscDraw routines.

    Currently we default to using cmapping[0 to PETSC_DRAW_BASIC_COLORS-1] for the basic colors and
  cmapping[DRAW_BASIC_COLORS to 255] for a uniform hue of all the colors. But in the contour
  plot we only use from PETSC_DRAW_BASIC_COLORS to 240 since the ones beyond that are too dark.

*/
#include <../src/sys/draw/impls/x/ximpl.h>
#include <X11/Xatom.h>

static const char *(colornames[PETSC_DRAW_BASIC_COLORS]) = { "white",
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
                                                 "lavenderblush",
                                                 "plum"};

extern PetscErrorCode PetscDrawXiInitCmap(PetscDraw_X*);
extern PetscErrorCode PetscDrawXiGetVisualClass(PetscDraw_X *);

/*
   Sets up a color map for a display. This is shared by all the windows
  opened on that display; this is to save time when windows are open so
  each one does not have to create its own color map which can take 15 to 20 seconds

     This is new code written 2/26/1999 Barry Smith,I hope it can replace
  some older,rather confusing code.

     The calls to XAllocNamedColor() and XAllocColor() are very slow
     because we have to request from the X server for each
     color. Could not figure out a way to request a large number at the
     same time.

   IMPORTANT: this code will fail if user opens windows on two different
  displays: should add error checking to detect this. This is because all windows
  share the same gColormap and gCmapping.

*/
static Colormap  gColormap  = 0;
static PetscDrawXiPixVal    gCmapping[256];

#undef __FUNCT__
#define __FUNCT__ "PetscDrawSetUpColormap_Shared"
PetscErrorCode PetscDrawSetUpColormap_Shared(Display *display,int screen,Visual *visual,Colormap colormap)
{
  XColor         colordef,ecolordef;
  unsigned char *red,*green,*blue;
  int            i,ncolors;
  PetscErrorCode ierr;
  PetscBool      fast = PETSC_FALSE;

  PetscFunctionBegin;
  if (colormap) gColormap = colormap;
  else          gColormap   = DefaultColormap(display,screen);

  /* set the basic colors into the color map */
  for (i=0; i<PETSC_DRAW_BASIC_COLORS; i++) {
    XAllocNamedColor(display,gColormap,colornames[i],&colordef,&ecolordef);
    gCmapping[i] = colordef.pixel;
  }

  /* set the uniform hue colors into the color map */
  ncolors = 256-PETSC_DRAW_BASIC_COLORS;
  ierr    = PetscMalloc3(ncolors,unsigned char,&red,ncolors,unsigned char,&green,ncolors,unsigned char,&blue);CHKERRQ(ierr);
  ierr    = PetscDrawUtilitySetCmapHue(red,green,blue,ncolors);CHKERRQ(ierr);
  ierr    = PetscOptionsGetBool(PETSC_NULL,"-draw_fast",&fast,PETSC_NULL);CHKERRQ(ierr);
  if (!fast) {
    for (i=PETSC_DRAW_BASIC_COLORS; i<ncolors+PETSC_DRAW_BASIC_COLORS; i++) {
      colordef.red    = ((int)red[i-PETSC_DRAW_BASIC_COLORS]   * 65535) / 255;
      colordef.green  = ((int)green[i-PETSC_DRAW_BASIC_COLORS] * 65535) / 255;
      colordef.blue   = ((int)blue[i-PETSC_DRAW_BASIC_COLORS]  * 65535) / 255;
      colordef.flags  = DoRed | DoGreen | DoBlue;
      XAllocColor(display,gColormap,&colordef);
      gCmapping[i]   = colordef.pixel;
    }
  }
  ierr = PetscFree3(red,green,blue);CHKERRQ(ierr);
  ierr = PetscInfo(0,"Successfully allocated colors\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Keep a record of which pixel numbers in the cmap have been
  used so far; this is to allow us to try to reuse as much of the current
  colormap as possible.
*/
static PetscBool  cmap_pixvalues_used[256];
static int        cmap_base = 0;

#undef __FUNCT__
#define __FUNCT__ "PetscDrawSetUpColormap_Private"
PetscErrorCode PetscDrawSetUpColormap_Private(Display *display,int screen,Visual *visual,Colormap colormap)
{
  Colormap       defaultmap = DefaultColormap(display,screen);
  PetscErrorCode ierr;
  int            found,i,ncolors;
  XColor         colordef;
  unsigned char  *red,*green,*blue;
  PetscBool      fast = PETSC_FALSE;

  PetscFunctionBegin;

  if (colormap)  gColormap = colormap;
  else           gColormap = XCreateColormap(display,RootWindow(display,screen),visual,AllocAll);

  cmap_base = 0;
  ierr = PetscMemzero(cmap_pixvalues_used,256*sizeof(PetscBool));CHKERRQ(ierr);

  /* set the basic colors into the color map */
  for (i=0; i<PETSC_DRAW_BASIC_COLORS; i++) {
    XParseColor(display,gColormap,colornames[i],&colordef);
    /* try to allocate the color in the default-map */
    found = XAllocColor(display,defaultmap,&colordef);
    /* use it, if it it exists and is not already used in the new colormap */
    if (found && colordef.pixel < 256  && !cmap_pixvalues_used[colordef.pixel]) {
      cmap_pixvalues_used[colordef.pixel] = PETSC_TRUE;
    /* otherwise search for the next available slot */
    } else {
      while (cmap_pixvalues_used[cmap_base]) cmap_base++;
      colordef.pixel                   = cmap_base;
      cmap_pixvalues_used[cmap_base++] = PETSC_TRUE;
    }
    XStoreColor(display,gColormap,&colordef);
    gCmapping[i] = colordef.pixel;
  }

  /* set the uniform hue colors into the color map */
  ncolors = 256-PETSC_DRAW_BASIC_COLORS;
  ierr    = PetscMalloc3(ncolors,unsigned char,&red,ncolors,unsigned char,&green,ncolors,unsigned char,&blue);CHKERRQ(ierr);
  ierr    = PetscDrawUtilitySetCmapHue(red,green,blue,ncolors);CHKERRQ(ierr);
  ierr    = PetscOptionsGetBool(PETSC_NULL,"-draw_fast",&fast,PETSC_NULL);CHKERRQ(ierr);
  if (!fast) {
    for (i=PETSC_DRAW_BASIC_COLORS; i<ncolors+PETSC_DRAW_BASIC_COLORS; i++) {
      colordef.red    = ((int)red[i-PETSC_DRAW_BASIC_COLORS]   * 65535) / 255;
      colordef.green  = ((int)green[i-PETSC_DRAW_BASIC_COLORS] * 65535) / 255;
      colordef.blue   = ((int)blue[i-PETSC_DRAW_BASIC_COLORS]  * 65535) / 255;
      colordef.flags  = DoRed | DoGreen | DoBlue;
      /* try to allocate the color in the default-map */
      found = XAllocColor(display,defaultmap,&colordef);
      /* use it, if it it exists and is not already used in the new colormap */
      if (found && colordef.pixel < 256  && !cmap_pixvalues_used[colordef.pixel]) {
        cmap_pixvalues_used[colordef.pixel] = PETSC_TRUE;
        /* otherwise search for the next available slot */
      } else {
        while (cmap_pixvalues_used[cmap_base]) cmap_base++;
        colordef.pixel                   = cmap_base;
        cmap_pixvalues_used[cmap_base++] = PETSC_TRUE;
      }
      XStoreColor(display,gColormap,&colordef);
      gCmapping[i]   = colordef.pixel;
    }
  }
  ierr = PetscFree3(red,green,blue);CHKERRQ(ierr);
  ierr = PetscInfo(0,"Successfully allocated colors\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawSetUpColormap_X"
PetscErrorCode PetscDrawSetUpColormap_X(Display *display,int screen,Visual *visual,Colormap colormap)
{
  PetscErrorCode ierr;
  PetscBool      sharedcolormap = PETSC_FALSE;
  XVisualInfo    vinfo;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-draw_x_shared_colormap",&sharedcolormap,PETSC_NULL);CHKERRQ(ierr);
  /*
        Need to determine if window supports allocating a private colormap,
  */
  if (XMatchVisualInfo(display,screen,24,StaticColor,&vinfo) ||
      XMatchVisualInfo(display,screen,24,TrueColor,&vinfo)   ||
      XMatchVisualInfo(display,screen,16,StaticColor,&vinfo) ||
      XMatchVisualInfo(display,screen,16,TrueColor,&vinfo)   ||
      XMatchVisualInfo(display,screen,15,StaticColor,&vinfo) ||
      XMatchVisualInfo(display,screen,15,TrueColor,&vinfo)) {
    sharedcolormap = PETSC_TRUE;
  }

  /* generate the X color map object */
  if (sharedcolormap) {
    ierr = PetscDrawSetUpColormap_Shared(display,screen,visual,colormap);CHKERRQ(ierr);
  } else {
    ierr = PetscDrawSetUpColormap_Private(display,screen,visual,colormap);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawSetColormap_X"
PetscErrorCode PetscDrawSetColormap_X(PetscDraw_X* XiWin,char *host,Colormap colormap)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (XiWin->depth < 8) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"PETSc Graphics require monitors with at least 8 bit color (256 colors)");
  if (!gColormap){
    Display  *display;  /* Private display will exist forever contains colormap shared by all windows */
    int      screen;
    Visual*  vis;

    display = XOpenDisplay(host);
    screen  = DefaultScreen(display);
    vis     = DefaultVisual(display,screen);

    ierr = PetscDrawSetUpColormap_X(display,screen,vis,colormap);CHKERRQ(ierr);
  }
  XiWin->cmap = gColormap;
  ierr = PetscMemcpy(XiWin->cmapping,gCmapping,256*sizeof(PetscDrawXiPixVal));CHKERRQ(ierr);
  XiWin->background = XiWin->cmapping[PETSC_DRAW_WHITE];
  XiWin->foreground = XiWin->cmapping[PETSC_DRAW_BLACK];
  PetscFunctionReturn(0);
}

/*
    Color in X is many-layered.  The first layer is the "visual",a
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
#undef __FUNCT__
#define __FUNCT__ "PetscDrawXiSetVisualClass"
PetscErrorCode PetscDrawXiSetVisualClass(PetscDraw_X* XiWin)
{
  XVisualInfo vinfo;

  PetscFunctionBegin;
  if (XMatchVisualInfo(XiWin->disp,XiWin->screen,24,DirectColor,&vinfo)) {
    XiWin->vis    = vinfo.visual;
  } else if (XMatchVisualInfo(XiWin->disp,XiWin->screen,8,PseudoColor,&vinfo)) {
    XiWin->vis    = vinfo.visual;
  } else if (XMatchVisualInfo(XiWin->disp,XiWin->screen,
    DefaultDepth(XiWin->disp,XiWin->screen),PseudoColor,&vinfo)) {
    XiWin->vis    = vinfo.visual;
  } else {
    XiWin->vis    = DefaultVisual(XiWin->disp,XiWin->screen);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawXiGetVisualClass"
PetscErrorCode PetscDrawXiGetVisualClass(PetscDraw_X* XiWin)
{
  PetscFunctionBegin;
#if defined(__cplusplus)
  PetscFunctionReturn(XiWin->vis->c_class);
#else
  PetscFunctionReturn(XiWin->vis->class);
#endif
}


#undef __FUNCT__
#define __FUNCT__ "PetscDrawXiSetColormap"
PetscErrorCode PetscDrawXiSetColormap(PetscDraw_X* XiWin)
{
  PetscFunctionBegin;
  XSetWindowColormap(XiWin->disp,XiWin->win,XiWin->cmap);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawXiGetBaseColor"
PetscErrorCode PetscDrawXiGetBaseColor(PetscDraw_X* XiWin,PetscDrawXiPixVal* white_pix,PetscDrawXiPixVal* black_pix)
{
  PetscFunctionBegin;
  *white_pix  = XiWin->cmapping[PETSC_DRAW_WHITE];
  *black_pix  = XiWin->cmapping[PETSC_DRAW_BLACK];
  PetscFunctionReturn(0);
}

/*
    This routine returns the pixel value for the specified color
    Returns 0 on failure,<>0 otherwise.
 */
#undef __FUNCT__
#define __FUNCT__ "PetscDrawXiFindColor"
PetscErrorCode PetscDrawXiFindColor(PetscDraw_X *XiWin,char *name,PetscDrawXiPixVal *pixval)
{
  XColor   colordef;
  int      st;

  PetscFunctionBegin;
  st = XParseColor(XiWin->disp,XiWin->cmap,name,&colordef);
  if (st) {
    st  = XAllocColor(XiWin->disp,XiWin->cmap,&colordef);
    if (st)  *pixval = colordef.pixel;
  }
  PetscFunctionReturn(st);
}

/*
    Another real need is to assign "colors" that make sense for
    a monochrome display,without unduely penalizing color displays.
    This routine takes a color name,a window, and a flag that
    indicates whether this is "background" or "foreground".
    In the monchrome case (or if the color is otherwise unavailable),
    the "background" or "foreground" colors will be chosen
 */
#undef __FUNCT__
#define __FUNCT__ "PetscDrawXiGetColor"
PetscDrawXiPixVal PetscDrawXiGetColor(PetscDraw_X* XiWin,char *name,int is_fore)
{
  PetscDrawXiPixVal pixval;

  PetscFunctionBegin;
  if (XiWin->numcolors == 2 || !PetscDrawXiFindColor(XiWin,name,&pixval)) {
    pixval  = is_fore ? XiWin->cmapping[PETSC_DRAW_WHITE] : XiWin->cmapping[PETSC_DRAW_BLACK];
  }
  PetscFunctionReturn(pixval);
}

