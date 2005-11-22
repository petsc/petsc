#define PETSC_DLL

/*
    Defines the operations for the Postscript PetscDraw implementation.
*/

#include "src/sys/draw/impls/ps/psimpl.h"         /*I  "petsc.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawOpenPS" 
/*@C
      PetscDrawOpenPS - Opens a PetscViewer that generates Postscript

  Collective on MPI_Comm

  Input Parameters:
+   comm - communicator that shares the PetscViewer
-   file - name of file where Postscript is to be stored

  Output Parameter:
.  viewer - the PetscViewer object

  Level: beginner

.seealso: PetscDrawDestroy(), PetscDrawOpenX(), PetscDrawCreate(), PetscViewerDrawOpen(), PetscViewerDrawGetDraw()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscDrawOpenPS(MPI_Comm comm,char *filename,PetscDraw *draw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDrawCreate(comm,filename,0,0,0,0,0,draw);CHKERRQ(ierr);
  ierr = PetscDrawSetType(*draw,PETSC_DRAW_PS);CHKERRQ(ierr);
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
(ps->currentcolor = (c),PetscViewerASCIISynchronizedPrintf(ps->ps_file,"%G %G %G setrgbcolor\n",rgb[0][c],rgb[1][c],rgb[2][c])))

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawPoint_PS" 
static PetscErrorCode PetscDrawPoint_PS(PetscDraw draw,PetscReal x,PetscReal  y,int c)
{
  PetscReal      xx,yy;
  PetscErrorCode ierr;
  PetscDraw_PS*  ps = (PetscDraw_PS*)draw->data;

  PetscFunctionBegin;
  xx = XTRANS(draw,x);  yy = YTRANS(draw,y);
  ierr = PSSetColor(ps,c);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(ps->ps_file,"%G %G moveto %G %G lineto stroke\n",xx,yy,xx+1,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawLine_PS" 
static PetscErrorCode PetscDrawLine_PS(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int c)
{
  PetscDraw_PS*  ps = (PetscDraw_PS*)draw->data;
  PetscReal      x1,y_1,x2,y2;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  x1 = XTRANS(draw,xl);   x2  = XTRANS(draw,xr); 
  y_1 = YTRANS(draw,yl);   y2  = YTRANS(draw,yr); 
  ierr = PSSetColor(ps,c);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(ps->ps_file,"%G %G moveto %G %G lineto stroke\n",x1,y_1,x2,y2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawStringSetSize_PS" 
static PetscErrorCode PetscDrawStringSetSize_PS(PetscDraw draw,PetscReal x,PetscReal  y)
{
  PetscDraw_PS*  ps = (PetscDraw_PS*)draw->data;
  PetscErrorCode ierr;
  int            w,h;

  PetscFunctionBegin;
  w = (int)((WIDTH)*x*(draw->port_xr - draw->port_xl)/(draw->coor_xr - draw->coor_xl));
  h = (int)((HEIGHT)*y*(draw->port_yr - draw->port_yl)/(draw->coor_yr - draw->coor_yl));
  ierr = PetscViewerASCIIPrintf(ps->ps_file,"/Helvetica-normal findfont %G scalefont setfont\n",(w+h)/2.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawStringGetSize_PS" 
static PetscErrorCode PetscDrawStringGetSize_PS(PetscDraw draw,PetscReal *x,PetscReal  *y)
{
  PetscReal   w = 9,h = 9;

  PetscFunctionBegin;
  *x = w*(draw->coor_xr - draw->coor_xl)/(WIDTH)*(draw->port_xr - draw->port_xl);
  *y = h*(draw->coor_yr - draw->coor_yl)/(HEIGHT)*(draw->port_yr - draw->port_yl);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawString_PS" 
static PetscErrorCode PetscDrawString_PS(PetscDraw draw,PetscReal x,PetscReal  y,int c,const char chrs[])
{
  PetscDraw_PS* ps = (PetscDraw_PS*)draw->data;
  PetscReal     x1,y_1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PSSetColor(ps,c);CHKERRQ(ierr);
  x1 = XTRANS(draw,x);
  y_1 = YTRANS(draw,y);
  ierr = PetscViewerASCIISynchronizedPrintf(ps->ps_file,"%G %G moveto (%s) show\n",x1,y_1,chrs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawStringVertical_PS" 
static PetscErrorCode PetscDrawStringVertical_PS(PetscDraw draw,PetscReal x,PetscReal  y,int c,const char chrs[])
{
  PetscDraw_PS*  ps = (PetscDraw_PS*)draw->data;
  PetscReal      x1,y_1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PSSetColor(ps,c);CHKERRQ(ierr);
  x1 = XTRANS(draw,x);
  y_1 = YTRANS(draw,y);
  ierr = PetscViewerASCIISynchronizedPrintf(ps->ps_file,"gsave %G %G moveto 90 rotate (%s) show grestore\n",x1,y_1,chrs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawInterpolatedTriangle_PS(PetscDraw_PS*,PetscReal,PetscReal,int,PetscReal,PetscReal,int,PetscReal,PetscReal,int);

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawTriangle_PS" 
static PetscErrorCode PetscDrawTriangle_PS(PetscDraw draw,PetscReal X1,PetscReal Y_1,PetscReal X2,
                          PetscReal Y2,PetscReal X3,PetscReal Y3,int c1,int c2,int c3)
{
  PetscDraw_PS*  ps = (PetscDraw_PS*)draw->data;
  PetscErrorCode ierr;
  PetscReal      x1,y_1,x2,y2,x3,y3;

  PetscFunctionBegin;
  x1   = XTRANS(draw,X1);
  y_1  = YTRANS(draw,Y_1); 
  x2   = XTRANS(draw,X2);
  y2   = YTRANS(draw,Y2); 
  x3   = XTRANS(draw,X3);
  y3   = YTRANS(draw,Y3); 

  if (c1 == c2 && c2 == c3) {
    ierr = PSSetColor(ps,c1);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(ps->ps_file,"%G %G moveto %G %G lineto %G %G lineto fill\n",x1,y_1,x2,y2,x3,y3);CHKERRQ(ierr);
  } else {
    ierr = PetscDrawInterpolatedTriangle_PS(ps,x1,y_1,c1,x2,y2,c2,x3,y3,c3);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawRectangle_PS" 
static PetscErrorCode PetscDrawRectangle_PS(PetscDraw draw,PetscReal X1,PetscReal Y_1,PetscReal X2,
                          PetscReal Y2,int c1,int c2,int c3,int c4)
{
  PetscDraw_PS*  ps = (PetscDraw_PS*)draw->data;
  PetscErrorCode ierr;
  PetscReal      x1,y_1,x2,y2,x3,y3,x4,y4;

  PetscFunctionBegin;
  x1   = XTRANS(draw,X1);
  y_1  = YTRANS(draw,Y_1); 
  x2   = XTRANS(draw,X2);
  y2   = YTRANS(draw,Y_1); 
  x3   = XTRANS(draw,X2);
  y3   = YTRANS(draw,Y2); 
  x4   = XTRANS(draw,X1);
  y4   = YTRANS(draw,Y2); 

  ierr = PSSetColor(ps,(c1+c2+c3+c4)/4);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(ps->ps_file,"%G %G moveto %G %G lineto %G %G lineto %G %G lineto %G %G lineto fill\n",x1,y_1,x2,y2,x3,y3,x4,y4,x1,y_1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawDestroy_PS" 
static PetscErrorCode PetscDrawDestroy_PS(PetscDraw draw)
{
  PetscDraw_PS   *ps = (PetscDraw_PS*)draw->data;
  PetscErrorCode ierr;
  PetscTruth     show;
  char           *filename,par[PETSC_MAX_PATH_LEN];
 
  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(ps->ps_file,"\nshowpage\n");CHKERRQ(ierr);
  ierr = PetscOptionsHasName(draw->prefix,"-draw_ps_show",&show);CHKERRQ(ierr);
  if (show) {
    ierr = PetscViewerFileGetName(ps->ps_file,&filename);CHKERRQ(ierr);    
    ierr = PetscStrcpy(par,"ghostview ");CHKERRQ(ierr);
    ierr = PetscStrcat(par,filename);CHKERRQ(ierr);
#if defined(PETSC_HAVE_POPEN)    
    ierr = PetscPOpen(draw->comm,PETSC_NULL,par,"r",PETSC_NULL);CHKERRQ(ierr);
#else
    SETERRQ(PETSC_ERR_SUP_SYS,"Cannot run external programs on this machine");
#endif
  }
  ierr = PetscViewerDestroy(ps->ps_file);CHKERRQ(ierr);
  ierr = PetscFree(ps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSynchronizedFlush_PS" 
static PetscErrorCode PetscDrawSynchronizedFlush_PS(PetscDraw draw)
{
  PetscErrorCode ierr;
  PetscDraw_PS*  ps = (PetscDraw_PS*)draw->data;

  PetscFunctionBegin;
  ierr = PetscViewerFlush(ps->ps_file);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSynchronizedClear_PS" 
static PetscErrorCode PetscDrawSynchronizedClear_PS(PetscDraw draw)
{
  PetscErrorCode ierr;
  PetscDraw_PS*  ps = (PetscDraw_PS*)draw->data;

  PetscFunctionBegin;
  ierr = PetscViewerFlush(ps->ps_file);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(ps->ps_file,"\nshowpage\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static struct _PetscDrawOps DvOps = { 0,
                                 0,
                                 PetscDrawLine_PS,
                                 0,
                                 0,
                                 PetscDrawPoint_PS,
                                 0,
                                 PetscDrawString_PS,
                                 PetscDrawStringVertical_PS,
                                 PetscDrawStringSetSize_PS,
                                 PetscDrawStringGetSize_PS,
                                 0,
                                 0,
                                 PetscDrawSynchronizedFlush_PS,
                                 PetscDrawRectangle_PS,
                                 PetscDrawTriangle_PS,
                                 0,
                                 0,
                                 0,
				 PetscDrawSynchronizedClear_PS,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 PetscDrawDestroy_PS,
                                 0,
                                 0,
                                 0 };

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscDrawCreate_PS" 
PetscErrorCode PETSC_DLLEXPORT PetscDrawCreate_PS(PetscDraw draw)
{
  PetscDraw_PS  *ps;
  PetscErrorCode ierr;
  int           ncolors,i;
  unsigned char *red,*green,*blue;
  static int    filecount = 0;
  char          buff[32];
  char          version[256];

  PetscFunctionBegin;
  if (!draw->display) {
    sprintf(buff,"defaultps%d.ps",filecount++);
    ierr = PetscStrallocpy(buff,&draw->display);CHKERRQ(ierr);
  }

  ierr = PetscGetVersion(&version);CHKERRQ(ierr);
  ierr = PetscNew(PetscDraw_PS,&ps);CHKERRQ(ierr);
  ierr = PetscMemcpy(draw->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);
  ierr = PetscViewerASCIIOpen(draw->comm,draw->display,&ps->ps_file);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(ps->ps_file,"%%!PS-Adobe-2.0\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(ps->ps_file,"%%%%Creator: PETSc %s\n",version);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(ps->ps_file,"%%%%Title: %s\n",draw->display);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(ps->ps_file,"%%%%Pages: 1\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(ps->ps_file,"%%%%PageOrder: Ascend\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(ps->ps_file,"%%%%BoundingBox: 0 0 612 792\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(ps->ps_file,"%%%%DocumentFonts: Helvetica-normal Symbol\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(ps->ps_file,"%%%%EndComments\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(ps->ps_file,"/Helvetica-normal findfont 10 scalefont setfont\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(ps->ps_file,"/c {setrgbcolor} def\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(ps->ps_file,"/l {lineto stroke} def\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(ps->ps_file,"/m {moveto} def\n");CHKERRQ(ierr);

  ps->currentcolor = PETSC_DRAW_BLACK;

  if (!rgbfilled) {
    rgbfilled = PETSC_TRUE;
    rgb[0][PETSC_DRAW_WHITE]       = 255/255;
    rgb[1][PETSC_DRAW_WHITE]       = 255/255;
    rgb[2][PETSC_DRAW_WHITE]       = 255/255;
    rgb[0][PETSC_DRAW_BLACK]       = 0;
    rgb[1][PETSC_DRAW_BLACK]       = 0;
    rgb[2][PETSC_DRAW_BLACK]       = 0;
    rgb[0][PETSC_DRAW_RED]         = 255/255;
    rgb[1][PETSC_DRAW_RED]         = 0;
    rgb[2][PETSC_DRAW_RED]         = 0;
    rgb[0][PETSC_DRAW_GREEN]       = 0;
    rgb[1][PETSC_DRAW_GREEN]       = 255./255;
    rgb[2][PETSC_DRAW_GREEN]       = 0;
    rgb[0][PETSC_DRAW_CYAN]        = 0;
    rgb[1][PETSC_DRAW_CYAN]        = 255./255;
    rgb[2][PETSC_DRAW_CYAN]        = 255./255;
    rgb[0][PETSC_DRAW_BLUE]        = 0;
    rgb[1][PETSC_DRAW_BLUE]        = 0;
    rgb[2][PETSC_DRAW_BLUE]        = 255./255;
    rgb[0][PETSC_DRAW_MAGENTA]     = 255./255;
    rgb[1][PETSC_DRAW_MAGENTA]     = 0;
    rgb[2][PETSC_DRAW_MAGENTA]     = 255./255;
    rgb[0][PETSC_DRAW_AQUAMARINE]  = 127./255;
    rgb[1][PETSC_DRAW_AQUAMARINE]  = 255./255;
    rgb[2][PETSC_DRAW_AQUAMARINE]  = 212./255;
    rgb[0][PETSC_DRAW_FORESTGREEN] = 34./255   ;
    rgb[1][PETSC_DRAW_FORESTGREEN] = 139./255.;
    rgb[2][PETSC_DRAW_FORESTGREEN] = 34./255. ;
    rgb[0][PETSC_DRAW_ORANGE]      = 255./255.    ;
    rgb[1][PETSC_DRAW_ORANGE]      = 165./255.;
    rgb[2][PETSC_DRAW_ORANGE]      = 0 ;
    rgb[0][PETSC_DRAW_VIOLET]      = 238./255.  ;
    rgb[1][PETSC_DRAW_VIOLET]      = 130./255.;
    rgb[2][PETSC_DRAW_VIOLET]      = 238./255.;
    rgb[0][PETSC_DRAW_BROWN]       = 165./255.   ;
    rgb[1][PETSC_DRAW_BROWN]       =  42./255.;
    rgb[2][PETSC_DRAW_BROWN]       = 42./255.;
    rgb[0][PETSC_DRAW_PINK]        = 255./255. ;
    rgb[1][PETSC_DRAW_PINK]        = 192./255. ;
    rgb[2][PETSC_DRAW_PINK]        = 203./255.;
    rgb[0][PETSC_DRAW_CORAL]       = 255./255.;   
    rgb[1][PETSC_DRAW_CORAL]       = 127./255.;
    rgb[2][PETSC_DRAW_CORAL]       = 80./255.;
    rgb[0][PETSC_DRAW_GRAY]        = 190./255.  ;
    rgb[1][PETSC_DRAW_GRAY]        = 190./255.;
    rgb[2][PETSC_DRAW_GRAY]        = 190./255.;
    rgb[0][PETSC_DRAW_YELLOW]      = 255./255.    ;
    rgb[1][PETSC_DRAW_YELLOW]      = 255./255.;
    rgb[2][PETSC_DRAW_YELLOW]      = 0/255.;
    rgb[0][PETSC_DRAW_GOLD]        = 255./255.  ;
    rgb[1][PETSC_DRAW_GOLD]        =  215./255.;
    rgb[2][PETSC_DRAW_GOLD]        =  0;
    rgb[0][PETSC_DRAW_LIGHTPINK]   = 255./255.  ;
    rgb[1][PETSC_DRAW_LIGHTPINK]   = 182./255.;
    rgb[2][PETSC_DRAW_LIGHTPINK]   = 193./255.;
    rgb[0][PETSC_DRAW_MEDIUMTURQUOISE] = 72./255.;
    rgb[1][PETSC_DRAW_MEDIUMTURQUOISE] =  209./255.;
    rgb[2][PETSC_DRAW_MEDIUMTURQUOISE] =  204./255.;
    rgb[0][PETSC_DRAW_KHAKI]           = 240./255.  ;
    rgb[1][PETSC_DRAW_KHAKI]           = 230./255.;
    rgb[2][PETSC_DRAW_KHAKI]           = 140./255.;
    rgb[0][PETSC_DRAW_DIMGRAY]         = 105./255.  ;
    rgb[1][PETSC_DRAW_DIMGRAY]         = 105./255.;
    rgb[2][PETSC_DRAW_DIMGRAY]         = 105./255.;
    rgb[0][PETSC_DRAW_YELLOWGREEN]     = 154./255.   ;
    rgb[1][PETSC_DRAW_YELLOWGREEN]     = 205./255.;
    rgb[2][PETSC_DRAW_YELLOWGREEN]     =  50./255.;
    rgb[0][PETSC_DRAW_SKYBLUE]         = 135./255.  ;
    rgb[1][PETSC_DRAW_SKYBLUE]         = 206./255.;
    rgb[2][PETSC_DRAW_SKYBLUE]         = 235./255.;
    rgb[0][PETSC_DRAW_DARKGREEN]       = 0    ;
    rgb[1][PETSC_DRAW_DARKGREEN]       = 100./255.;
    rgb[2][PETSC_DRAW_DARKGREEN]       = 0;
    rgb[0][PETSC_DRAW_NAVYBLUE]        = 0   ;
    rgb[1][PETSC_DRAW_NAVYBLUE]        =  0;
    rgb[2][PETSC_DRAW_NAVYBLUE]        = 128./255.;
    rgb[0][PETSC_DRAW_SANDYBROWN]      = 244./255.   ;
    rgb[1][PETSC_DRAW_SANDYBROWN]      = 164./255.;
    rgb[2][PETSC_DRAW_SANDYBROWN]      = 96./255.;
    rgb[0][PETSC_DRAW_CADETBLUE]       =  95./255.  ;
    rgb[1][PETSC_DRAW_CADETBLUE]       = 158./255.;
    rgb[2][PETSC_DRAW_CADETBLUE]       = 160./255.;
    rgb[0][PETSC_DRAW_POWDERBLUE]      = 176./255.  ;
    rgb[1][PETSC_DRAW_POWDERBLUE]      = 224./255.;
    rgb[2][PETSC_DRAW_POWDERBLUE]      = 230./255.;
    rgb[0][PETSC_DRAW_DEEPPINK]        = 255./255. ;
    rgb[1][PETSC_DRAW_DEEPPINK]        =  20./255.;
    rgb[2][PETSC_DRAW_DEEPPINK]        = 147./255.;
    rgb[0][PETSC_DRAW_THISTLE]         = 216./255.  ;
    rgb[1][PETSC_DRAW_THISTLE]         = 191./255.;
    rgb[2][PETSC_DRAW_THISTLE]         = 216./255.;
    rgb[0][PETSC_DRAW_LIMEGREEN]       = 50./255.  ;
    rgb[1][PETSC_DRAW_LIMEGREEN]       = 205./255.;
    rgb[2][PETSC_DRAW_LIMEGREEN]       =  50./255.;
    rgb[0][PETSC_DRAW_LAVENDERBLUSH]   = 255./255.  ;
    rgb[1][PETSC_DRAW_LAVENDERBLUSH]   = 240./255.;
    rgb[2][PETSC_DRAW_LAVENDERBLUSH]   = 245./255.;

    /* now do the uniform hue part of the colors */
    ncolors = 256-PETSC_DRAW_BASIC_COLORS;
    ierr    = PetscMalloc(3*ncolors*sizeof(unsigned char),&red);CHKERRQ(ierr);
    green   = red   + ncolors;
    blue    = green + ncolors;
    ierr    = PetscDrawUtilitySetCmapHue(red,green,blue,ncolors);CHKERRQ(ierr);
    for (i=PETSC_DRAW_BASIC_COLORS; i<ncolors+PETSC_DRAW_BASIC_COLORS; i++) {
      rgb[0][i]  = ((double)red[i-PETSC_DRAW_BASIC_COLORS])/255.;
      rgb[1][i]  = ((double)green[i-PETSC_DRAW_BASIC_COLORS])/255.;
      rgb[2][i]  = ((double)blue[i-PETSC_DRAW_BASIC_COLORS])/255.;
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


#undef __FUNCT__  
#define __FUNCT__ "PetscDrawInterpolatedTriangle_PS" 
static PetscErrorCode PetscDrawInterpolatedTriangle_PS(PetscDraw_PS* ps,PetscReal x1,PetscReal y_1,int t1,
                                PetscReal x2,PetscReal y2,int t2,PetscReal x3,PetscReal y3,int t3)
{
  PetscReal      rfrac,lfrac;
  PetscReal      lc,rc = 0.0,lx,rx = 0.0,xx,y;
  PetscReal      rc_lc,rx_lx,t2_t1,x2_x1,t3_t1,x3_x1,t3_t2,x3_x2;
  PetscReal      R_y2_y_1,R_y3_y_1,R_y3_y2;
  PetscErrorCode ierr;
  int            c;

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
  ierr = PetscViewerASCIISynchronizedPrintf(ps->ps_file,"gsave 2 2 scale\n");CHKERRQ(ierr);


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
    /* PetscDraw a line with the correct color from t1-t2 to t1-t3 */
    /* Left color is (y-y_1)/(y2-y_1) * (t2-t1) + t1 */
    lfrac = ((y-y_1)) * R_y2_y_1; 
    lc    = (lfrac * (t2_t1) + t1);
    lx    = (lfrac * (x2_x1) + x1);
    /* Right color is (y-y_1)/(y3-y_1) * (t3-t1) + t1 */
    rfrac = ((y - y_1)) * R_y3_y_1; 
    rc    = (rfrac * (t3_t1) + t1);
    rx    = (rfrac * (x3_x1) + x1);
    /* PetscDraw the line */
    rc_lc = rc - lc; 
    rx_lx = rx - lx;
    if (rx > lx) {
      for (xx=lx; xx<=rx; xx++) {
        c = (int)(((xx-lx) * (rc_lc)) / (rx_lx) + lc);
        ierr = PetscViewerASCIISynchronizedPrintf(ps->ps_file,"%G %G %G c\n",rgb[0][c],rgb[1][c],rgb[2][c]);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(ps->ps_file,"%G %G m %G %G l\n",xx,y,xx+1,y);CHKERRQ(ierr);
      }
    } else if (rx < lx) {
      for (xx=lx; xx>=rx; xx--) {
        c = (int)(((xx-lx) * (rc_lc)) / (rx_lx) + lc);
        ierr = PetscViewerASCIISynchronizedPrintf(ps->ps_file,"%G %G %G c\n",rgb[0][c],rgb[1][c],rgb[2][c]);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(ps->ps_file,"%G %G m %G %G l\n",xx,y,xx+1,y);CHKERRQ(ierr);
      }
    } else {
      c = (int)lc;
      ierr = PetscViewerASCIISynchronizedPrintf(ps->ps_file,"%G %G %G c\n",rgb[0][c],rgb[1][c],rgb[2][c]);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(ps->ps_file,"%G %G m %G %G l\n",lx,y,lx+1,y);CHKERRQ(ierr);
    }
  }

  /* For simplicity,"move" t1 to the intersection of t1-t3 with the line y=y2.
     We take advantage of the previous iteration. */
  if (y2 >= y3) {
    ierr = PetscViewerASCIISynchronizedPrintf(ps->ps_file,"grestore\n");CHKERRQ(ierr);
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
    /* PetscDraw a line with the correct color from t2-t3 to t1-t3 */
    /* Left color is (y-y_1)/(y2-y_1) * (t2-t1) + t1 */
    lfrac = ((y-y2)) * R_y3_y2; 
    lc    = (lfrac * (t3_t2) + t2);
    lx    = (lfrac * (x3_x2) + x2);
    /* Right color is (y-y_1)/(y3-y_1) * (t3-t1) + t1 */
    rfrac = ((y - y_1)) * R_y3_y_1; 
    rc    = (rfrac * (t3_t1) + t1);
    rx    = (rfrac * (x3_x1) + x1);
    /* PetscDraw the line */
    rc_lc = rc - lc; 
    rx_lx = rx - lx;
    if (rx > lx) {
      for (xx=lx; xx<=rx; xx++) {
        c = (int)(((xx-lx) * (rc_lc)) / (rx_lx) + lc);
        ierr = PetscViewerASCIISynchronizedPrintf(ps->ps_file,"%G %G %G c\n",rgb[0][c],rgb[1][c],rgb[2][c]);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(ps->ps_file,"%G %G m %G %G l\n",xx,y,xx+1,y);CHKERRQ(ierr);
      }
    } else if (rx < lx) {
      for (xx=lx; xx>=rx; xx--) {
        c = (int)(((xx-lx) * (rc_lc)) / (rx_lx) + lc);
        ierr = PetscViewerASCIISynchronizedPrintf(ps->ps_file,"%G %G %G c\n",rgb[0][c],rgb[1][c],rgb[2][c]);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(ps->ps_file,"%G %G m %G %G l\n",xx,y,xx+1,y);CHKERRQ(ierr);
      }
    } else {
      c = (int)lc;
      ierr = PetscViewerASCIISynchronizedPrintf(ps->ps_file,"%G %G %G c\n",rgb[0][c],rgb[1][c],rgb[2][c]);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(ps->ps_file,"%G %G m %G %G l\n",lx,y,lx+1,y);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerASCIISynchronizedPrintf(ps->ps_file,"grestore\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}







