
/*
    Code for managing color the X implementation of the PetscDraw routines.

    Currently we default to using cmapping[0 to PETSC_DRAW_BASIC_COLORS-1] for the basic colors and
    cmapping[DRAW_BASIC_COLORS to 255] for countour plots.

*/
#include <../src/sys/classes/draw/impls/x/ximpl.h>
#include <X11/Xatom.h>

static const char *colornames[PETSC_DRAW_BASIC_COLORS] = {
  "white",
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
  "plum"
};

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
static Colormap          gColormap = 0;
static PetscDrawXiPixVal gCmapping[PETSC_DRAW_MAXCOLOR];
static unsigned char     gCpalette[PETSC_DRAW_MAXCOLOR][3];

PetscErrorCode PetscDrawSetUpColormap_Shared(Display *display,int screen,Visual *visual,Colormap colormap)
{
  int            i,k,ncolors = PETSC_DRAW_MAXCOLOR - PETSC_DRAW_BASIC_COLORS;
  unsigned char  R[PETSC_DRAW_MAXCOLOR-PETSC_DRAW_BASIC_COLORS];
  unsigned char  G[PETSC_DRAW_MAXCOLOR-PETSC_DRAW_BASIC_COLORS];
  unsigned char  B[PETSC_DRAW_MAXCOLOR-PETSC_DRAW_BASIC_COLORS];
  XColor         colordef,ecolordef;
  PetscBool      fast = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (colormap) gColormap = colormap;
  else          gColormap = DefaultColormap(display,screen);

  /* set the basic colors into the color map */
  for (i=0; i<PETSC_DRAW_BASIC_COLORS; i++) {
    XAllocNamedColor(display,gColormap,colornames[i],&colordef,&ecolordef);
    gCmapping[i]    = colordef.pixel;
    gCpalette[i][0] = (unsigned char)(colordef.red   >> 8);
    gCpalette[i][1] = (unsigned char)(colordef.green >> 8);
    gCpalette[i][2] = (unsigned char)(colordef.blue  >> 8);
  }

  /* set the contour colors into the colormap */
  ierr = PetscOptionsGetBool(NULL,NULL,"-draw_fast",&fast,NULL);CHKERRQ(ierr);
  ierr = PetscDrawUtilitySetCmap(NULL,ncolors,R,G,B);CHKERRQ(ierr);
  for (i=0, k=PETSC_DRAW_BASIC_COLORS; i<ncolors; i++, k++) {
    colordef.red   = (unsigned short)(R[i] << 8);
    colordef.green = (unsigned short)(G[i] << 8);
    colordef.blue  = (unsigned short)(B[i] << 8);
    colordef.flags = DoRed|DoGreen|DoBlue;
    colordef.pixel = gCmapping[PETSC_DRAW_BLACK];
    if (!fast) XAllocColor(display,gColormap,&colordef);
    gCmapping[k]    = colordef.pixel;
    gCpalette[k][0] = R[i];
    gCpalette[k][1] = G[i];
    gCpalette[k][2] = B[i];
  }

  ierr = PetscInfo(NULL,"Successfully allocated colors\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Keep a record of which pixel numbers in the cmap have been
  used so far; this is to allow us to try to reuse as much of the current
  colormap as possible.
*/
static PetscBool cmap_pixvalues_used[PETSC_DRAW_MAXCOLOR];
static int       cmap_base = 0;

PetscErrorCode PetscDrawSetUpColormap_Private(Display *display,int screen,Visual *visual,Colormap colormap)
{
  int            found,i,k,ncolors = PETSC_DRAW_MAXCOLOR-PETSC_DRAW_BASIC_COLORS;
  unsigned char  R[PETSC_DRAW_MAXCOLOR-PETSC_DRAW_BASIC_COLORS];
  unsigned char  G[PETSC_DRAW_MAXCOLOR-PETSC_DRAW_BASIC_COLORS];
  unsigned char  B[PETSC_DRAW_MAXCOLOR-PETSC_DRAW_BASIC_COLORS];
  Colormap       defaultmap = DefaultColormap(display,screen);
  XColor         colordef;
  PetscBool      fast = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (colormap) gColormap = colormap;
  else          gColormap = XCreateColormap(display,RootWindow(display,screen),visual,AllocAll);

  cmap_base = 0;

  ierr = PetscMemzero(cmap_pixvalues_used,sizeof(cmap_pixvalues_used));CHKERRQ(ierr);

  /* set the basic colors into the color map */
  for (i=0; i<PETSC_DRAW_BASIC_COLORS; i++) {
    XParseColor(display,gColormap,colornames[i],&colordef);
    /* try to allocate the color in the default-map */
    found = XAllocColor(display,defaultmap,&colordef);
    /* use it, if it it exists and is not already used in the new colormap */
    if (found && colordef.pixel < PETSC_DRAW_MAXCOLOR  && !cmap_pixvalues_used[colordef.pixel]) {
      cmap_pixvalues_used[colordef.pixel] = PETSC_TRUE;
      /* otherwise search for the next available slot */
    } else {
      while (cmap_pixvalues_used[cmap_base]) cmap_base++;
      colordef.pixel                   = cmap_base;
      cmap_pixvalues_used[cmap_base++] = PETSC_TRUE;
    }
    XStoreColor(display,gColormap,&colordef);
    gCmapping[i]    = colordef.pixel;
    gCpalette[i][0] = (unsigned char)(colordef.red   >> 8);
    gCpalette[i][1] = (unsigned char)(colordef.green >> 8);
    gCpalette[i][2] = (unsigned char)(colordef.blue  >> 8);
  }

  /* set the contour colors into the colormap */
  ierr = PetscOptionsGetBool(NULL,NULL,"-draw_fast",&fast,NULL);CHKERRQ(ierr);
  ierr = PetscDrawUtilitySetCmap(NULL,ncolors,R,G,B);CHKERRQ(ierr);
  for (i=0, k=PETSC_DRAW_BASIC_COLORS; i<ncolors; i++, k++) {
    colordef.red   = (unsigned short)(R[i] << 8);
    colordef.green = (unsigned short)(G[i] << 8);
    colordef.blue  = (unsigned short)(B[i] << 8);
    colordef.flags = DoRed|DoGreen|DoBlue;
    colordef.pixel = gCmapping[PETSC_DRAW_BLACK];
    if (!fast) {
      /* try to allocate the color in the default-map */
      found = XAllocColor(display,defaultmap,&colordef);
      /* use it, if it it exists and is not already used in the new colormap */
      if (found && colordef.pixel < PETSC_DRAW_MAXCOLOR  && !cmap_pixvalues_used[colordef.pixel]) {
        cmap_pixvalues_used[colordef.pixel] = PETSC_TRUE;
        /* otherwise search for the next available slot */
      } else {
        while (cmap_pixvalues_used[cmap_base]) cmap_base++;
        colordef.pixel                   = cmap_base;
        cmap_pixvalues_used[cmap_base++] = PETSC_TRUE;
      }
      XStoreColor(display,gColormap,&colordef);
    }
    gCmapping[k]    = colordef.pixel;
    gCpalette[k][0] = R[i];
    gCpalette[k][1] = G[i];
    gCpalette[k][2] = B[i];
  }

  ierr = PetscInfo(NULL,"Successfully allocated colors\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDrawSetUpColormap_X(Display *display,int screen,Visual *visual,Colormap colormap)
{
  PetscErrorCode ierr;
  PetscBool      sharedcolormap = PETSC_FALSE;
  XVisualInfo    vinfo;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(NULL,NULL,"-draw_x_shared_colormap",&sharedcolormap,NULL);CHKERRQ(ierr);
  /*
     Need to determine if window supports allocating a private colormap,
  */
  if (XMatchVisualInfo(display,screen,24,StaticColor,&vinfo) ||
      XMatchVisualInfo(display,screen,24,TrueColor,&vinfo)   ||
      XMatchVisualInfo(display,screen,16,StaticColor,&vinfo) ||
      XMatchVisualInfo(display,screen,16,TrueColor,&vinfo)   ||
      XMatchVisualInfo(display,screen,15,StaticColor,&vinfo) ||
      XMatchVisualInfo(display,screen,15,TrueColor,&vinfo)) sharedcolormap = PETSC_TRUE;
  /*
     Generate the X colormap object
  */
  if (sharedcolormap) {
    ierr = PetscDrawSetUpColormap_Shared(display,screen,visual,colormap);CHKERRQ(ierr);
  } else {
    ierr = PetscDrawSetUpColormap_Private(display,screen,visual,colormap);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscDrawSetColormap_X(PetscDraw_X*,Colormap);

PetscErrorCode PetscDrawSetColormap_X(PetscDraw_X *XiWin,Colormap colormap)
{
  PetscBool      fast = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(NULL,NULL,"-draw_fast",&fast,NULL);CHKERRQ(ierr);
  PetscCheckFalse(XiWin->depth < 8,PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"PETSc Graphics require monitors with at least 8 bit color (256 colors)");
  if (!gColormap) {
    ierr = PetscDrawSetUpColormap_X(XiWin->disp,XiWin->screen,XiWin->vis,colormap);CHKERRQ(ierr);
  }
  XiWin->cmap     = gColormap;
  XiWin->cmapsize = fast ? PETSC_DRAW_BASIC_COLORS : PETSC_DRAW_MAXCOLOR;
  ierr = PetscMemcpy(XiWin->cmapping,gCmapping,sizeof(XiWin->cmapping));CHKERRQ(ierr);
  ierr = PetscMemcpy(XiWin->cpalette,gCpalette,sizeof(XiWin->cpalette));CHKERRQ(ierr);
  XiWin->background = XiWin->cmapping[PETSC_DRAW_WHITE];
  XiWin->foreground = XiWin->cmapping[PETSC_DRAW_BLACK];
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDrawXiColormap(PetscDraw_X *XiWin)
{ return PetscDrawSetColormap_X(XiWin,(Colormap)0); }

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
PetscErrorCode PetscDrawXiSetVisualClass(PetscDraw_X *XiWin)
{
  XVisualInfo vinfo;

  PetscFunctionBegin;
  if (XMatchVisualInfo(XiWin->disp,XiWin->screen,24,DirectColor,&vinfo)) {
    XiWin->vis = vinfo.visual;
  } else if (XMatchVisualInfo(XiWin->disp,XiWin->screen,8,PseudoColor,&vinfo)) {
    XiWin->vis = vinfo.visual;
  } else if (XMatchVisualInfo(XiWin->disp,XiWin->screen,DefaultDepth(XiWin->disp,XiWin->screen),PseudoColor,&vinfo)) {
    XiWin->vis = vinfo.visual;
  } else {
    XiWin->vis = DefaultVisual(XiWin->disp,XiWin->screen);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDrawXiSetColormap(PetscDraw_X *XiWin)
{
  PetscFunctionBegin;
  XSetWindowColormap(XiWin->disp,XiWin->win,XiWin->cmap);
  PetscFunctionReturn(0);
}

/*
   Get RGB color entries out of the X colormap
*/
PetscErrorCode PetscDrawXiGetPalette(PetscDraw_X *XiWin,unsigned char palette[PETSC_DRAW_MAXCOLOR][3])
{
  int    k;
  XColor colordef[PETSC_DRAW_MAXCOLOR];

  PetscFunctionBegin;
  for (k=0; k<PETSC_DRAW_MAXCOLOR; k++) {
    colordef[k].pixel = XiWin->cmapping[k];
    colordef[k].flags = DoRed|DoGreen|DoBlue;
  }
  XQueryColors(XiWin->disp,XiWin->cmap,colordef,PETSC_DRAW_MAXCOLOR);
  for (k=0; k<PETSC_DRAW_MAXCOLOR; k++) {
    palette[k][0] = (unsigned char)(colordef[k].red   >> 8);
    palette[k][1] = (unsigned char)(colordef[k].green >> 8);
    palette[k][2] = (unsigned char)(colordef[k].blue  >> 8);
  }
  PetscFunctionReturn(0);
}

