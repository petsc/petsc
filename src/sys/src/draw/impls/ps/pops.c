/* $Id: pops.c,v 1.3 2000/01/11 20:59:17 bsmith Exp bsmith $*/

/*
    Defines the operations for the Postscript Draw implementation.
*/

#include "src/sys/src/draw/impls/ps/psimpl.h"         /*I  "petsc.h" I*/

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawOpenPS" 
/*@C
      DrawOpenPS - Opens a viewer that generates Postscript

  Collective on MPI_Comm

  Input Parameters:
+   comm - communicator that shares the viewer
-   file - name of file where Postscript is to be stored

  Output Parameter:
.   viewer - the viewer object

  Level: beginner

.seealso: DrawDestroy(), DrawOpenX(), DrawCreate(), ViewerDrawOpen(), ViewerDrawGetDraw()
@*/
int DrawOpenPS(MPI_Comm comm,char *filename,Draw *draw)
{
  int ierr;

  PetscFunctionBegin;
  ierr = DrawCreate(comm,filename,0,0,0,0,0,draw);CHKERRQ(ierr);
  ierr = DrawSetType(*draw,DRAW_PS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     These macros transform from the users coordinates to the Postscript
*/
#define WIDTH  8.5*72
#define HEIGHT 11*72
#define XTRANS(win,x) \
   ((WIDTH)*((win)->port_xl+(((x-(win)->coor_xl)*((win)->port_xr-(win)->port_xl))/((win)->coor_xr-(win)->coor_xl))))
#define YTRANS(win,y) \
   ((HEIGHT)*((win)->port_yl+(((y-(win)->coor_yl)*((win)->port_yr-(win)->port_yl))/((win)->coor_yr-(win)->coor_yl))))

/*
    Contains the RGB colors for the PETSc defined colors
*/
static PetscReal  rgb[3][256];
static PetscTruth rgbfilled = PETSC_FALSE;

#define PSSetColor(ps,c)   (((c) == ps->currentcolor) ? 0 : \
(ps->currentcolor = (c),ViewerASCIISynchronizedPrintf(ps->ps_file,"%g %g %g setrgbcolor\n",rgb[0][c],rgb[1][c],rgb[2][c])))

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawPoint_PS" 
static int DrawPoint_PS(Draw draw,PetscReal x,PetscReal  y,int c)
{
  PetscReal   xx,yy;
  int      ierr;
  Draw_PS* ps = (Draw_PS*)draw->data;

  PetscFunctionBegin;
  xx = XTRANS(draw,x);  yy = YTRANS(draw,y);
  ierr = PSSetColor(ps,c);CHKERRQ(ierr);
  ierr = ViewerASCIISynchronizedPrintf(ps->ps_file,"%g %g moveto %g %g lineto stroke\n",xx,yy,xx+1,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawLine_PS" 
static int DrawLine_PS(Draw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int c)
{
  Draw_PS* ps = (Draw_PS*)draw->data;
  PetscReal   x1,y_1,x2,y2;
  int      ierr;

  PetscFunctionBegin;
  x1 = XTRANS(draw,xl);   x2  = XTRANS(draw,xr); 
  y_1 = YTRANS(draw,yl);   y2  = YTRANS(draw,yr); 
  ierr = PSSetColor(ps,c);CHKERRQ(ierr);
  ierr = ViewerASCIISynchronizedPrintf(ps->ps_file,"%g %g moveto %g %g lineto stroke\n",x1,y_1,x2,y2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawStringSetSize_PS" 
static int DrawStringSetSize_PS(Draw draw,PetscReal x,PetscReal  y)
{
  Draw_PS* ps = (Draw_PS*)draw->data;
  int      ierr,w,h;

  PetscFunctionBegin;
  w = (int)((WIDTH)*x*(draw->port_xr - draw->port_xl)/(draw->coor_xr - draw->coor_xl));
  h = (int)((HEIGHT)*y*(draw->port_yr - draw->port_yl)/(draw->coor_yr - draw->coor_yl));
  ierr = ViewerASCIIPrintf(ps->ps_file,"/Helvetica-normal findfont %g scalefont setfont\n",(w+h)/2.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawStringGetSize_PS" 
static int DrawStringGetSize_PS(Draw draw,PetscReal *x,PetscReal  *y)
{
  PetscReal   w = 9,h = 9;

  PetscFunctionBegin;
  *x = w*(draw->coor_xr - draw->coor_xl)/(WIDTH)*(draw->port_xr - draw->port_xl);
  *y = h*(draw->coor_yr - draw->coor_yl)/(HEIGHT)*(draw->port_yr - draw->port_yl);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawString_PS" 
static int DrawString_PS(Draw draw,PetscReal x,PetscReal  y,int c,char *chrs)
{
  Draw_PS* ps = (Draw_PS*)draw->data;
  PetscReal   x1,y_1;
  int      ierr;

  PetscFunctionBegin;
  ierr = PSSetColor(ps,c);CHKERRQ(ierr);
  x1 = XTRANS(draw,x);
  y_1 = YTRANS(draw,y);
  ierr = ViewerASCIISynchronizedPrintf(ps->ps_file,"%g %g moveto (%s) show\n",x1,y_1,chrs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawStringVertical_PS" 
static int DrawStringVertical_PS(Draw draw,PetscReal x,PetscReal  y,int c,char *chrs)
{
  Draw_PS* ps = (Draw_PS*)draw->data;
  PetscReal   x1,y_1;
  int      ierr;

  PetscFunctionBegin;
  ierr = PSSetColor(ps,c);CHKERRQ(ierr);
  x1 = XTRANS(draw,x);
  y_1 = YTRANS(draw,y);
  ierr = ViewerASCIISynchronizedPrintf(ps->ps_file,"gsave %g %g moveto 90 rotate (%s) show grestore\n",x1,y_1,chrs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static int DrawInterpolatedTriangle_PS(Draw_PS*,PetscReal,PetscReal,int,PetscReal,PetscReal,int,PetscReal,PetscReal,int);

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawTriangle_PS" 
static int DrawTriangle_PS(Draw draw,PetscReal X1,PetscReal Y_1,PetscReal X2,
                          PetscReal Y2,PetscReal X3,PetscReal Y3,int c1,int c2,int c3)
{
  Draw_PS* ps = (Draw_PS*)draw->data;
  int      ierr;
  PetscReal   x1,y_1,x2,y2,x3,y3;

  PetscFunctionBegin;
  x1   = XTRANS(draw,X1);
  y_1  = YTRANS(draw,Y_1); 
  x2   = XTRANS(draw,X2);
  y2   = YTRANS(draw,Y2); 
  x3   = XTRANS(draw,X3);
  y3   = YTRANS(draw,Y3); 

  if (c1 == c2 && c2 == c3) {
    ierr = PSSetColor(ps,c1);CHKERRQ(ierr);
    ierr = ViewerASCIISynchronizedPrintf(ps->ps_file,"%g %g moveto %g %g lineto %g %g lineto fill\n",x1,y_1,x2,y2,x3,y3);CHKERRQ(ierr);
  } else {
    ierr = DrawInterpolatedTriangle_PS(ps,x1,y_1,c1,x2,y2,c2,x3,y3,c3);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawDestroy_PS" 
static int DrawDestroy_PS(Draw draw)
{
  Draw_PS    *ps = (Draw_PS*)draw->data;
  int        ierr;
  PetscTruth show;
  char       *filename,par[1024];
 
  PetscFunctionBegin;
  ierr = ViewerASCIIPrintf(ps->ps_file,"\nshowpage\n");
  ierr = OptionsHasName(draw->prefix,"-draw_ps_show",&show);CHKERRQ(ierr);
  if (show) {
    ierr = ViewerGetFilename(ps->ps_file,&filename);CHKERRQ(ierr);    
    ierr = PetscStrcpy(par,"ghostview ");CHKERRQ(ierr);
    ierr = PetscStrcat(par,filename);CHKERRQ(ierr);
    ierr = PetscPOpen(draw->comm,PETSC_NULL,par,"r",PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = ViewerDestroy(ps->ps_file);CHKERRQ(ierr);
  ierr = PetscFree(ps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawSynchronizedFlush_PS" 
static int DrawSynchronizedFlush_PS(Draw draw)
{
  int      ierr;
  Draw_PS* ps = (Draw_PS*)draw->data;

  PetscFunctionBegin;
  ierr = ViewerFlush(ps->ps_file);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawSynchronizedClear_PS" 
static int DrawSynchronizedClear_PS(Draw draw)
{
  int      ierr;
  Draw_PS* ps = (Draw_PS*)draw->data;

  PetscFunctionBegin;
  ierr = ViewerFlush(ps->ps_file);CHKERRQ(ierr);
  ierr = ViewerASCIIPrintf(ps->ps_file,"\nshowpage\n");

  PetscFunctionReturn(0);
}

static struct _DrawOps DvOps = { 0,
                                 0,
                                 DrawLine_PS,
                                 0,
                                 0,
                                 DrawPoint_PS,
                                 0,
                                 DrawString_PS,
                                 DrawStringVertical_PS,
                                 DrawStringSetSize_PS,
                                 DrawStringGetSize_PS,
                                 0,
                                 0,
                                 DrawSynchronizedFlush_PS,
                                 0,
                                 DrawTriangle_PS,
                                 0,
                                 0,
				 DrawSynchronizedClear_PS,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 DrawDestroy_PS,
                                 0,
                                 0,
                                 0 };

EXTERN_C_BEGIN
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawCreate_PS" 
int DrawCreate_PS(Draw draw)
{
  Draw_PS       *ps;
  int           ierr,ncolors,i;
  unsigned char *red,*green,*blue;
  static int    filecount = 0;
  char          buff[32];

  PetscFunctionBegin;
  if (!draw->display) {
    sprintf(buff,"defaultps%d.ps",filecount++);
    ierr = PetscStrallocpy(buff,&draw->display);CHKERRQ(ierr);
  }

  ps   = PetscNew(Draw_PS);CHKPTRQ(ps);
  ierr = PetscMemcpy(draw->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);
  ierr = ViewerASCIIOpen(draw->comm,draw->display,&ps->ps_file);CHKERRQ(ierr);
  ierr = ViewerASCIIPrintf(ps->ps_file,"%%!PS-Adobe-2.0\n");
  ierr = ViewerASCIIPrintf(ps->ps_file,"%%%%Creator: PETSc %s\n",PETSC_VERSION_NUMBER);
  ierr = ViewerASCIIPrintf(ps->ps_file,"%%%%Title: %s\n",draw->display);
  ierr = ViewerASCIIPrintf(ps->ps_file,"%%%%Pages: 1\n");
  ierr = ViewerASCIIPrintf(ps->ps_file,"%%%%PageOrder: Ascend\n");
  ierr = ViewerASCIIPrintf(ps->ps_file,"%%%%BoundingBox: 0 0 612 792\n");
  ierr = ViewerASCIIPrintf(ps->ps_file,"%%%%DocumentFonts: Helvetica-normal Symbol\n");
  ierr = ViewerASCIIPrintf(ps->ps_file,"%%%%EndComments\n");
  ierr = ViewerASCIIPrintf(ps->ps_file,"/Helvetica-normal findfont 10 scalefont setfont\n");
  ierr = ViewerASCIIPrintf(ps->ps_file,"/c {setrgbcolor} def\n");
  ierr = ViewerASCIIPrintf(ps->ps_file,"/l {lineto stroke} def\n");
  ierr = ViewerASCIIPrintf(ps->ps_file,"/m {moveto} def\n");

  ps->currentcolor = DRAW_BLACK;

  if (!rgbfilled) {
    rgbfilled = PETSC_TRUE;
    rgb[0][DRAW_WHITE]       = 255/255;
    rgb[1][DRAW_WHITE]       = 255/255;
    rgb[2][DRAW_WHITE]       = 255/255;
    rgb[0][DRAW_BLACK]       = 0;
    rgb[1][DRAW_BLACK]       = 0;
    rgb[2][DRAW_BLACK]       = 0;
    rgb[0][DRAW_RED]         = 255/255;
    rgb[1][DRAW_RED]         = 0;
    rgb[2][DRAW_RED]         = 0;
    rgb[0][DRAW_GREEN]       = 0;
    rgb[1][DRAW_GREEN]       = 255./255;
    rgb[2][DRAW_GREEN]       = 0;
    rgb[0][DRAW_CYAN]        = 0;
    rgb[1][DRAW_CYAN]        = 255./255;
    rgb[2][DRAW_CYAN]        = 255./255;
    rgb[0][DRAW_BLUE]        = 0;
    rgb[1][DRAW_BLUE]        = 0;
    rgb[2][DRAW_BLUE]        = 255./255;
    rgb[0][DRAW_MAGENTA]     = 255./255;
    rgb[1][DRAW_MAGENTA]     = 0;
    rgb[2][DRAW_MAGENTA]     = 255./255;
    rgb[0][DRAW_AQUAMARINE]  = 127./255;
    rgb[1][DRAW_AQUAMARINE]  = 255./255;
    rgb[2][DRAW_AQUAMARINE]  = 212./255;
    rgb[0][DRAW_FORESTGREEN] = 34./255   ;
    rgb[1][DRAW_FORESTGREEN] = 139./255.;
    rgb[2][DRAW_FORESTGREEN] = 34./255. ;
    rgb[0][DRAW_ORANGE]      = 255./255.    ;
    rgb[1][DRAW_ORANGE]      = 165./255.;
    rgb[2][DRAW_ORANGE]      = 0 ;
    rgb[0][DRAW_VIOLET]      = 238./255.  ;
    rgb[1][DRAW_VIOLET]      = 130./255.;
    rgb[2][DRAW_VIOLET]      = 238./255.;
    rgb[0][DRAW_BROWN]       = 165./255.   ;
    rgb[1][DRAW_BROWN]       =  42./255.;
    rgb[2][DRAW_BROWN]       = 42./255.;
    rgb[0][DRAW_PINK]        = 255./255. ;
    rgb[1][DRAW_PINK]        = 192./255. ;
    rgb[2][DRAW_PINK]        = 203./255.;
    rgb[0][DRAW_CORAL]       = 255./255.;   
    rgb[1][DRAW_CORAL]       = 127./255.;
    rgb[2][DRAW_CORAL]       = 80./255.;
    rgb[0][DRAW_GRAY]        = 190./255.  ;
    rgb[1][DRAW_GRAY]        = 190./255.;
    rgb[2][DRAW_GRAY]        = 190./255.;
    rgb[0][DRAW_YELLOW]      = 255./255.    ;
    rgb[1][DRAW_YELLOW]      = 255./255.;
    rgb[2][DRAW_YELLOW]      = 0/255.;
    rgb[0][DRAW_GOLD]        = 255./255.  ;
    rgb[1][DRAW_GOLD]        =  215./255.;
    rgb[2][DRAW_GOLD]        =  0;
    rgb[0][DRAW_LIGHTPINK]   = 255./255.  ;
    rgb[1][DRAW_LIGHTPINK]   = 182./255.;
    rgb[2][DRAW_LIGHTPINK]   = 193./255.;
    rgb[0][DRAW_MEDIUMTURQUOISE] = 72./255.;
    rgb[1][DRAW_MEDIUMTURQUOISE] =  209./255.;
    rgb[2][DRAW_MEDIUMTURQUOISE] =  204./255.;
    rgb[0][DRAW_KHAKI]           = 240./255.  ;
    rgb[1][DRAW_KHAKI]           = 230./255.;
    rgb[2][DRAW_KHAKI]           = 140./255.;
    rgb[0][DRAW_DIMGRAY]         = 105./255.  ;
    rgb[1][DRAW_DIMGRAY]         = 105./255.;
    rgb[2][DRAW_DIMGRAY]         = 105./255.;
    rgb[0][DRAW_YELLOWGREEN]     = 154./255.   ;
    rgb[1][DRAW_YELLOWGREEN]     = 205./255.;
    rgb[2][DRAW_YELLOWGREEN]     =  50./255.;
    rgb[0][DRAW_SKYBLUE]         = 135./255.  ;
    rgb[1][DRAW_SKYBLUE]         = 206./255.;
    rgb[2][DRAW_SKYBLUE]         = 235./255.;
    rgb[0][DRAW_DARKGREEN]       = 0    ;
    rgb[1][DRAW_DARKGREEN]       = 100./255.;
    rgb[2][DRAW_DARKGREEN]       = 0;
    rgb[0][DRAW_NAVYBLUE]        = 0   ;
    rgb[1][DRAW_NAVYBLUE]        =  0;
    rgb[2][DRAW_NAVYBLUE]        = 128./255.;
    rgb[0][DRAW_SANDYBROWN]      = 244./255.   ;
    rgb[1][DRAW_SANDYBROWN]      = 164./255.;
    rgb[2][DRAW_SANDYBROWN]      = 96./255.;
    rgb[0][DRAW_CADETBLUE]       =  95./255.  ;
    rgb[1][DRAW_CADETBLUE]       = 158./255.;
    rgb[2][DRAW_CADETBLUE]       = 160./255.;
    rgb[0][DRAW_POWDERBLUE]      = 176./255.  ;
    rgb[1][DRAW_POWDERBLUE]      = 224./255.;
    rgb[2][DRAW_POWDERBLUE]      = 230./255.;
    rgb[0][DRAW_DEEPPINK]        = 255./255. ;
    rgb[1][DRAW_DEEPPINK]        =  20./255.;
    rgb[2][DRAW_DEEPPINK]        = 147./255.;
    rgb[0][DRAW_THISTLE]         = 216./255.  ;
    rgb[1][DRAW_THISTLE]         = 191./255.;
    rgb[2][DRAW_THISTLE]         = 216./255.;
    rgb[0][DRAW_LIMEGREEN]       = 50./255.  ;
    rgb[1][DRAW_LIMEGREEN]       = 205./255.;
    rgb[2][DRAW_LIMEGREEN]       =  50./255.;
    rgb[0][DRAW_LAVENDERBLUSH]   = 255./255.  ;
    rgb[1][DRAW_LAVENDERBLUSH]   = 240./255.;
    rgb[2][DRAW_LAVENDERBLUSH]   = 245./255.;

    /* now do the uniform hue part of the colors */
    ncolors = 256-DRAW_BASIC_COLORS;
    red   = (unsigned char *)PetscMalloc(3*ncolors*sizeof(unsigned char));CHKPTRQ(red);
    green = red   + ncolors;
    blue  = green + ncolors;
    ierr = DrawUtilitySetCmapHue(red,green,blue,ncolors);CHKERRQ(ierr);
    for (i=DRAW_BASIC_COLORS; i<ncolors+DRAW_BASIC_COLORS; i++) {
      rgb[0][i]  = ((double)red[i-DRAW_BASIC_COLORS])/255.;
      rgb[1][i]  = ((double)green[i-DRAW_BASIC_COLORS])/255.;
      rgb[2][i]  = ((double)blue[i-DRAW_BASIC_COLORS])/255.;
    }
    ierr = PetscFree(red);CHKERRQ(ierr);
  }

  draw->data    = (void*)ps;
  PetscFunctionReturn(0);
}
EXTERN_C_END



/*
         This works in Postscript coordinates
*/
/*  

   this kind of thing should do contour triangles with Postscript level 3
1000 dict begin
  /ShadingType 4 def
  /ColorSpace /DeviceRGB def
  /DataSource [0 10 10 255 255 255 0 400 10 0 0 0 0 200 400 255 0 0] def
  shfill

   once we have Postscript level 3 we should put this in as an option
*/


#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawInterpolatedTriangle_PS" 
static int DrawInterpolatedTriangle_PS(Draw_PS* ps,PetscReal x1,PetscReal y_1,int t1,
                                PetscReal x2,PetscReal y2,int t2,PetscReal x3,PetscReal y3,int t3)
{
  PetscReal rfrac,lfrac;
  PetscReal lc,rc = 0.0,lx,rx = 0.0,xx,y;
  PetscReal rc_lc,rx_lx,t2_t1,x2_x1,t3_t1,x3_x1,t3_t2,x3_x2;
  PetscReal R_y2_y_1,R_y3_y_1,R_y3_y2;
  int       ierr,c;

  PetscFunctionBegin;
  /*
        Is triangle even visible in window?
  */
  if (x1 < 0 && x2 < 0 && x3 < 0) PetscFunctionReturn(0);
  if (y_1 < 0 && y2 < 0 && y3 < 0) PetscFunctionReturn(0);
  if (x1 > 72*8.5 && x2 > 72*8.5 && x3 > 72*8.5) PetscFunctionReturn(0);
  if (y_1 > 72*11 && y2 > 72*11 && y3 > 72*11) PetscFunctionReturn(0);

  /* scale everything by two to reduce the huge file; note this reduces the quality */
  x1  /= 2.0;
  x2  /= 2.0;
  x3  /= 2.0;
  y_1 /= 2.0;
  y2  /= 2.0;
  y3  /= 2.0;
  ierr = ViewerASCIISynchronizedPrintf(ps->ps_file,"gsave 2 2 scale\n");CHKERRQ(ierr);


  /* Sort the vertices */
#define SWAP(a,b) {PetscReal _a; _a=a; a=b; b=_a;}
#define ISWAP(a,b) {int _a; _a=a; a=b; b=_a;}
  if (y_1 > y2) {
    SWAP(y_1,y2);ISWAP(t1,t2); SWAP(x1,x2);
  }
  if (y_1 > y3) {
    SWAP(y_1,y3);ISWAP(t1,t3); SWAP(x1,x3);
  }
  if (y2 > y3) {
    SWAP(y2,y3);ISWAP(t2,t3); SWAP(x2,x3);
  }
  /* This code is decidely non-optimal; it is intended to be a start at
   an implementation */

  if (y2 != y_1) R_y2_y_1 = 1.0/((y2-y_1)); else R_y2_y_1 = 0.0; 
  if (y3 != y_1) R_y3_y_1 = 1.0/((y3-y_1)); else R_y3_y_1 = 0.0;
  t2_t1   = t2 - t1;
  x2_x1   = x2 - x1;
  t3_t1   = t3 - t1;
  x3_x1   = x3 - x1;
  for (y=y_1; y<=y2; y++) {
    /* Draw a line with the correct color from t1-t2 to t1-t3 */
    /* Left color is (y-y_1)/(y2-y_1) * (t2-t1) + t1 */
    lfrac = ((y-y_1)) * R_y2_y_1; 
    lc    = (lfrac * (t2_t1) + t1);
    lx    = (lfrac * (x2_x1) + x1);
    /* Right color is (y-y_1)/(y3-y_1) * (t3-t1) + t1 */
    rfrac = ((y - y_1)) * R_y3_y_1; 
    rc    = (rfrac * (t3_t1) + t1);
    rx    = (rfrac * (x3_x1) + x1);
    /* Draw the line */
    rc_lc = rc - lc; 
    rx_lx = rx - lx;
    if (rx > lx) {
      for (xx=lx; xx<=rx; xx++) {
        c = (int)(((xx-lx) * (rc_lc)) / (rx_lx) + lc);
        ierr = ViewerASCIISynchronizedPrintf(ps->ps_file,"%g %g %g c\n",rgb[0][c],rgb[1][c],rgb[2][c]);CHKERRQ(ierr);
        ierr = ViewerASCIISynchronizedPrintf(ps->ps_file,"%g %g m %g %g l\n",xx,y,xx+1,y);CHKERRQ(ierr);
      }
    } else if (rx < lx) {
      for (xx=lx; xx>=rx; xx--) {
        c = (int)(((xx-lx) * (rc_lc)) / (rx_lx) + lc);
        ierr = ViewerASCIISynchronizedPrintf(ps->ps_file,"%g %g %g c\n",rgb[0][c],rgb[1][c],rgb[2][c]);CHKERRQ(ierr);
        ierr = ViewerASCIISynchronizedPrintf(ps->ps_file,"%g %g m %g %g l\n",xx,y,xx+1,y);CHKERRQ(ierr);
      }
    } else {
      c = (int)lc;
      ierr = ViewerASCIISynchronizedPrintf(ps->ps_file,"%g %g %g c\n",rgb[0][c],rgb[1][c],rgb[2][c]);CHKERRQ(ierr);
      ierr = ViewerASCIISynchronizedPrintf(ps->ps_file,"%g %g m %g %g l\n",lx,y,lx+1,y);CHKERRQ(ierr);
    }
  }

  /* For simplicity,"move" t1 to the intersection of t1-t3 with the line y=y2.
     We take advantage of the previous iteration. */
  if (y2 >= y3) {
    ierr = ViewerASCIISynchronizedPrintf(ps->ps_file,"grestore\n");CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (y_1 < y2) {
    t1  = (int)rc;
    y_1 = y2;
    x1  = rx;

    t3_t1   = t3 - t1;
    x3_x1   = x3 - x1;    
  }
  t3_t2 = t3 - t2;
  x3_x2 = x3 - x2;
  if (y3 != y2) R_y3_y2 = 1.0/((y3-y2)); else R_y3_y2 = 0.0;
  if (y3 != y_1) R_y3_y_1 = 1.0/((y3-y_1)); else R_y3_y_1 = 0.0;
  for (y=y2; y<=y3; y++) {
    /* Draw a line with the correct color from t2-t3 to t1-t3 */
    /* Left color is (y-y_1)/(y2-y_1) * (t2-t1) + t1 */
    lfrac = ((y-y2)) * R_y3_y2; 
    lc    = (lfrac * (t3_t2) + t2);
    lx    = (lfrac * (x3_x2) + x2);
    /* Right color is (y-y_1)/(y3-y_1) * (t3-t1) + t1 */
    rfrac = ((y - y_1)) * R_y3_y_1; 
    rc    = (rfrac * (t3_t1) + t1);
    rx    = (rfrac * (x3_x1) + x1);
    /* Draw the line */
    rc_lc = rc - lc; 
    rx_lx = rx - lx;
    if (rx > lx) {
      for (xx=lx; xx<=rx; xx++) {
        c = (int)(((xx-lx) * (rc_lc)) / (rx_lx) + lc);
        ierr = ViewerASCIISynchronizedPrintf(ps->ps_file,"%g %g %g c\n",rgb[0][c],rgb[1][c],rgb[2][c]);CHKERRQ(ierr);
        ierr = ViewerASCIISynchronizedPrintf(ps->ps_file,"%g %g m %g %g l\n",xx,y,xx+1,y);CHKERRQ(ierr);
      }
    } else if (rx < lx) {
      for (xx=lx; xx>=rx; xx--) {
        c = (int)(((xx-lx) * (rc_lc)) / (rx_lx) + lc);
        ierr = ViewerASCIISynchronizedPrintf(ps->ps_file,"%g %g %g c\n",rgb[0][c],rgb[1][c],rgb[2][c]);CHKERRQ(ierr);
        ierr = ViewerASCIISynchronizedPrintf(ps->ps_file,"%g %g m %g %g l\n",xx,y,xx+1,y);CHKERRQ(ierr);
      }
    } else {
      c = (int)lc;
      ierr = ViewerASCIISynchronizedPrintf(ps->ps_file,"%g %g %g c\n",rgb[0][c],rgb[1][c],rgb[2][c]);CHKERRQ(ierr);
      ierr = ViewerASCIISynchronizedPrintf(ps->ps_file,"%g %g m %g %g l\n",lx,y,lx+1,y);CHKERRQ(ierr);
    }
  }
  ierr = ViewerASCIISynchronizedPrintf(ps->ps_file,"grestore\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}







