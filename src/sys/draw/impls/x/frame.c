#define PETSC_DLL
/*
   This file contains routines to draw a 3-d like frame about a given 
   box with a given width.  Note that we might like to use a high/low
   color for highlights.

   The region has 6 parameters.  These are the dimensions of the actual frame.
 */

#include "../src/sys/draw/impls/x/ximpl.h"

EXTERN PixVal XiGetColor(PetscDraw_X *,char *,int);

/* 50% grey stipple pattern */
static Pixmap grey50 = (Pixmap)0;         
#define cboard50_width 8
#define cboard50_height 8
static unsigned char cboard50_bits[] = {
   0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa};

static PixVal HiPix=0,LoPix=0;
/* 
   Set the colors for the highlights by name 
 */
#undef __FUNCT__  
#define __FUNCT__ "XiFrameColors" 
PetscErrorCode XiFrameColors(PetscDraw_X* XiWin,XiDecoration *Rgn,char *Hi,char *Lo)
{
  PetscFunctionBegin;
  Rgn->Hi = XiGetColor(XiWin,Hi,1);
  Rgn->Lo = XiGetColor(XiWin,Lo,1);
  Rgn->HasColor = Rgn->Hi != Rgn->Lo;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "XiDrawFrame" 
PetscErrorCode XiDrawFrame(PetscDraw_X *XiWin,XiDecoration *Rgn)
{
  int    xl = Rgn->Box.x,yl = Rgn->Box.y,xh = Rgn->Box.xh,yh = Rgn->Box.yh,
         o = Rgn->width;
  XPoint high[7],low[7];
  PixVal Hi,Lo;

  PetscFunctionBegin;
  /* High polygon */
  high[0].x = xl;            high[0].y = yh;
  high[1].x = xl + o;        high[1].y = yh - o;
  high[2].x = xh - o;        high[2].y = yh - o;
  high[3].x = xh - o;        high[3].y = yl + o;
  high[4].x = xh;            high[4].y = yl;
  high[5].x = xh;            high[5].y = yh;
  high[6].x = xl;            high[6].y = yh;     /* close path */

  low[0].x  = xl;            low[0].y = yh;
  low[1].x  = xl;            low[1].y = yl;
  low[2].x  = xh;            low[2].y = yl;
  low[3].x  = xh - o;        low[3].y = yl + o;
  low[4].x  = xl + o;        low[4].y = yl + o;
  low[5].x  = xl + o;        low[5].y = yh - o;
  low[6].x  = xl;            low[6].y = yh;      /* close path */

  if (Rgn->HasColor) {
    if (Rgn->Hi) Hi = Rgn->Hi;
    else         Hi = HiPix;
    if (Rgn->Lo) Lo = Rgn->Lo;
    else         Lo = LoPix;
    XiSetPixVal(XiWin,(Rgn->is_in !=0) ? Hi : Lo);
    if (o <= 1)
	XDrawLines(XiWin->disp,XiDrawable(XiWin),XiWin->gc.set,
		   high,7,CoordModeOrigin);
    else
	XFillPolygon(XiWin->disp,XiDrawable(XiWin),XiWin->gc.set,
		      high,7,Nonconvex,CoordModeOrigin);
    XiSetPixVal(XiWin,(Rgn->is_in !=0) ? Lo : Hi);
    if (o <= 1)
	XDrawLines(XiWin->disp,XiDrawable(XiWin),XiWin->gc.set,
		    low,7,CoordModeOrigin);
    else
	XFillPolygon(XiWin->disp,XiDrawable(XiWin),XiWin->gc.set,
		      low,7,Nonconvex,CoordModeOrigin);
    /* We could use additional highlights here,such as lines drawn
       connecting the mitred edges. */		 
  }
  else {
    if (!grey50) 
	grey50 = XCreatePixmapFromBitmapData(XiWin->disp,XiWin->win,
					     (char *)cboard50_bits,
					     cboard50_width,
					     cboard50_height,1,0,1);
    XiSetPixVal(XiWin,Rgn->Hi);
    XFillPolygon(XiWin->disp,XiDrawable(XiWin),XiWin->gc.set,
		 high,7,Nonconvex,CoordModeOrigin);
    /* This can actually be done by using a stipple effect */
    XSetFillStyle(XiWin->disp,XiWin->gc.set,FillStippled);
    XSetStipple(XiWin->disp,XiWin->gc.set,grey50);
    XFillPolygon(XiWin->disp,XiDrawable(XiWin),XiWin->gc.set,
		 low,7,Nonconvex,CoordModeOrigin);
    XSetFillStyle(XiWin->disp,XiWin->gc.set,FillSolid);
  }
  PetscFunctionReturn(0);
}


/*
   Set the colors for the highlights by name 
 */
#undef __FUNCT__  
#define __FUNCT__ "XiFrameColorsByName" 
PetscErrorCode XiFrameColorsByName(PetscDraw_X* XiWin,char *Hi,char *Lo)
{
  PetscFunctionBegin;
  if (XiWin->numcolors > 2) {
    HiPix = XiGetColor(XiWin,Hi,1);
    LoPix = XiGetColor(XiWin,Lo,1);
  }
  PetscFunctionReturn(0);
}
