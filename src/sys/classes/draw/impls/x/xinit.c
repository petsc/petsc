
/*
   This file contains routines to open an X window display and window
   This consists of a number of routines that set the various
   fields in the Window structure, which is passed to
   all of these routines.

   Note that if you use the default visual and colormap, then you
   can use these routines with any X toolkit that will give you the
   Window id of the window that it is managing.  Use that instead of the
   call to PetscDrawXiCreateWindow .  Similarly for the Display.
*/

#include <../src/sys/classes/draw/impls/x/ximpl.h>

extern PetscErrorCode PetscDrawXi_wait_map(PetscDraw_X*);
extern PetscErrorCode PetscDrawXiFontFixed(PetscDraw_X*,int,int,PetscDrawXiFont**);
extern PetscErrorCode PetscDrawXiInitCmap(PetscDraw_X*);
extern PetscErrorCode PetscDrawSetColormap_X(PetscDraw_X*,Colormap);

/*
  PetscDrawXiOpenDisplay - Open and setup a display
*/
#undef __FUNCT__
#define __FUNCT__ "PetscDrawXiOpenDisplay"
PetscErrorCode PetscDrawXiOpenDisplay(PetscDraw_X *XiWin,const char display[])
{
  PetscFunctionBegin;
  XiWin->disp = XOpenDisplay(display);
  if (!XiWin->disp) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Unable to open display on %s\n\
    Make sure your COMPUTE NODES are authorized to connect \n\
    to this X server and either your DISPLAY variable\n\
    is set or you use the -display name option\n",display);
  }
  XiWin->screen     = DefaultScreen(XiWin->disp);
  XiWin->vis        = DefaultVisual(XiWin->disp,XiWin->screen);
  XiWin->depth      = DefaultDepth(XiWin->disp,XiWin->screen);
  XiWin->cmap       = DefaultColormap(XiWin->disp,XiWin->screen);
  XiWin->background = WhitePixel(XiWin->disp,XiWin->screen);
  XiWin->foreground = BlackPixel(XiWin->disp,XiWin->screen);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawXiClose"
PetscErrorCode PetscDrawXiClose(PetscDraw_X *XiWin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!XiWin) PetscFunctionReturn(0);
  ierr = PetscFree(XiWin->font);CHKERRQ(ierr);
  if (XiWin->disp) {
#if defined(PETSC_HAVE_SETJMP_H)
    jmp_buf              jmpbuf;
    PetscXIOErrorHandler xioerrhdl;
    ierr = PetscMemcpy(&jmpbuf,&PetscXIOErrorHandlerJumpBuf,sizeof(jmpbuf));CHKERRQ(ierr);
    xioerrhdl = PetscSetXIOErrorHandler(PetscXIOErrorHandlerJump);
    if (!setjmp(PetscXIOErrorHandlerJumpBuf))
#endif
    {
      XFreeGC(XiWin->disp,XiWin->gc.set);
      XCloseDisplay(XiWin->disp);
    }
    XiWin->disp = NULL;
#if defined(PETSC_HAVE_SETJMP_H)
    (void)PetscSetXIOErrorHandler(xioerrhdl);
    ierr = PetscMemcpy(&PetscXIOErrorHandlerJumpBuf,&jmpbuf,sizeof(jmpbuf));CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

/*
   PetscDrawXiSetGC - setup the GC structure
*/
#undef __FUNCT__
#define __FUNCT__ "PetscDrawXiSetGC"
PetscErrorCode PetscDrawXiSetGC(PetscDraw_X *XiWin,PetscDrawXiPixVal fg)
{
  XGCValues gcvalues;             /* window graphics context values */

  PetscFunctionBegin;
  /* Set the graphics contexts */
  /* create a gc for the ROP_SET operation (writing the fg value to a pixel) */
  /* (do this with function GXcopy; GXset will automatically write 1) */
  gcvalues.function   = GXcopy;
  gcvalues.foreground = fg;
  XiWin->gc.cur_pix   = fg;
  XiWin->gc.set       = XCreateGC(XiWin->disp,RootWindow(XiWin->disp,XiWin->screen),GCFunction|GCForeground,&gcvalues);
  if (!XiWin->gc.set) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Unable to create X graphics context");
  PetscFunctionReturn(0);
}

/*
   PetscDrawXiInit - basic setup the draw (display, graphics context, font)
*/
#undef __FUNCT__
#define __FUNCT__ "PetscDrawXiInit"
PetscErrorCode PetscDrawXiInit(PetscDraw_X *XiWin,const char display[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscDrawXiOpenDisplay(XiWin,display);CHKERRQ(ierr);
  ierr = PetscDrawXiSetGC(XiWin,XiWin->foreground);CHKERRQ(ierr);
  ierr = PetscDrawXiFontFixed(XiWin,6,10,&XiWin->font);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Actually display a window at [x,y] with sizes (w,h)
*/
#undef __FUNCT__
#define __FUNCT__ "PetscDrawXiDisplayWindow"
PetscErrorCode PetscDrawXiDisplayWindow(PetscDraw_X *XiWin,char *label,int x,int y,int w,int h)
{
  unsigned int         wavail,havail;
  XSizeHints           size_hints;
  XWindowAttributes    in_window_attributes;
  XSetWindowAttributes window_attributes;
  unsigned int         border_width = 0;
  unsigned long        backgnd_pixel = WhitePixel(XiWin->disp,XiWin->screen);
  unsigned long        wmask;

  PetscFunctionBegin;
  /* get the available widths */
  wavail = DisplayWidth(XiWin->disp,XiWin->screen);
  havail = DisplayHeight(XiWin->disp,XiWin->screen);
  if (w <= 0 || h <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"X Window display has invalid height or width");
  if ((unsigned int)w > wavail) w = wavail;
  if ((unsigned int)h > havail) h = havail;

  if (x < 0) x = (int)(wavail - (unsigned int)w + (unsigned int)x);
  if (y < 0) y = (int)(havail - (unsigned int)h + (unsigned int)y);
  x = ((unsigned int)x + w > wavail) ? (int)(wavail - (unsigned int)w) : x;
  y = ((unsigned int)y + h > havail) ? (int)(havail - (unsigned int)h) : y;

  /* We need XCreateWindow since we may need an visual other than the default one */
  XGetWindowAttributes(XiWin->disp,RootWindow(XiWin->disp,XiWin->screen),&in_window_attributes);
  window_attributes.background_pixmap = None;
  window_attributes.background_pixel  = backgnd_pixel;
  /* No border for now */
  window_attributes.border_pixmap     = None;
  /*
  window_attributes.border_pixel      = border_pixel;
  */
  window_attributes.bit_gravity       = in_window_attributes.bit_gravity;
  window_attributes.win_gravity       = in_window_attributes.win_gravity;
  /* Backing store is too slow in color systems */
  window_attributes.backing_store     = NotUseful;
  window_attributes.backing_pixel     = backgnd_pixel;
  window_attributes.save_under        = 1;
  window_attributes.event_mask        = 0;
  window_attributes.do_not_propagate_mask = 0;
  window_attributes.override_redirect = 0;
  window_attributes.colormap          = XiWin->cmap;
  /* None for cursor does NOT mean none, it means cursor of Parent */
  window_attributes.cursor            = None;

  wmask = CWBackPixmap | CWBackPixel    | CWBorderPixmap  | CWBitGravity |
          CWWinGravity | CWBackingStore | CWBackingPixel  | CWOverrideRedirect |
          CWSaveUnder  | CWEventMask    | CWDontPropagate |
          CWCursor     | CWColormap;

  XiWin->win = XCreateWindow(XiWin->disp,RootWindow(XiWin->disp,XiWin->screen),x,y,w,h,border_width,XiWin->depth,InputOutput,XiWin->vis,wmask,&window_attributes);
  if (!XiWin->win) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Unable to open X window");

  /* set window manager hints */
  {
    XWMHints      wm_hints;
    XClassHint    class_hints;
    XTextProperty windowname,iconname;

    if (label) XStringListToTextProperty(&label,1,&windowname);
    else       XStringListToTextProperty(&label,0,&windowname);
    if (label) XStringListToTextProperty(&label,1,&iconname);
    else       XStringListToTextProperty(&label,0,&iconname);

    wm_hints.initial_state = NormalState;
    wm_hints.input         = True;
    wm_hints.flags         = StateHint|InputHint;

    /* These properties can be used by window managers to decide how to display a window */
    class_hints.res_name  = (char*)"petsc";
    class_hints.res_class = (char*)"PETSc";

    size_hints.x          = x;
    size_hints.y          = y;
    size_hints.min_width  = 4*border_width;
    size_hints.min_height = 4*border_width;
    size_hints.width      = w;
    size_hints.height     = h;
    size_hints.flags      = USPosition | USSize | PMinSize;

    XSetWMProperties(XiWin->disp,XiWin->win,&windowname,&iconname,0,0,&size_hints,&wm_hints,&class_hints);
    XFree((void*)windowname.value);
    XFree((void*)iconname.value);
  }

  /* make the window visible */
  XSelectInput(XiWin->disp,XiWin->win,ExposureMask|StructureNotifyMask);
  XMapWindow(XiWin->disp,XiWin->win);

  /* some window systems are cruel and interfere with the placement of
     windows.  We wait here for the window to be created or to die */
  if (PetscDrawXi_wait_map(XiWin)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Wait for X window failed");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawXiQuickWindow"
PetscErrorCode PetscDrawXiQuickWindow(PetscDraw_X *XiWin,char *name,int x,int y,int nx,int ny)
{
  Window         root;
  unsigned int   w,h,dummy;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDrawSetColormap_X(XiWin,(Colormap)0);CHKERRQ(ierr);
  ierr = PetscDrawXiDisplayWindow(XiWin,name,x,y,nx,ny);CHKERRQ(ierr);
  XSetWindowBackground(XiWin->disp,XiWin->win,XiWin->background);
  XClearWindow(XiWin->disp,XiWin->win);

  XGetGeometry(XiWin->disp,XiWin->win,&root,&x,&y,&w,&h,&dummy,&dummy);
  XiWin->x = x;
  XiWin->y = y;
  XiWin->w = (int)w;
  XiWin->h = (int)h;
  PetscFunctionReturn(0);
}

/*
   A version from an already defined window
*/
#undef __FUNCT__
#define __FUNCT__ "PetscDrawXiQuickWindowFromWindow"
PetscErrorCode PetscDrawXiQuickWindowFromWindow(PetscDraw_X *XiWin,Window win)
{
  Window            root;
  int               x,y;
  unsigned int      w,h,dummy;
  XWindowAttributes attributes;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  XiWin->win = win;
  XGetWindowAttributes(XiWin->disp,XiWin->win,&attributes);
  ierr = PetscDrawSetColormap_X(XiWin,attributes.colormap);CHKERRQ(ierr);

  XGetGeometry(XiWin->disp,XiWin->win,&root,&x,&y,&w,&h,&dummy,&dummy);
  XiWin->x = x;
  XiWin->y = y;
  XiWin->w = (int)w;
  XiWin->h = (int)h;
  PetscFunctionReturn(0);
}

/*
      PetscDrawXiSetWindowLabel - Sets new label in open window.
*/
#undef __FUNCT__
#define __FUNCT__ "PetscDrawXiSetWindowLabel"
PetscErrorCode PetscDrawXiSetWindowLabel(PetscDraw_X *XiWin,char *label)
{
  XTextProperty  prop;
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrlen(label,&len);CHKERRQ(ierr);
  XGetWMName(XiWin->disp,XiWin->win,&prop);
  prop.value  = (unsigned char*)label;
  prop.nitems = (long)len;
  XSetWMName(XiWin->disp,XiWin->win,&prop);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawSetSave_X"
PetscErrorCode PetscDrawSetSave_X(PetscDraw draw,const char *filename)
{
  PetscErrorCode ierr;
#if defined(PETSC_HAVE_POPEN)
  PetscMPIInt    rank;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
#if defined(PETSC_HAVE_POPEN)
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)draw),&rank);CHKERRQ(ierr);
  if (!rank) {
    char  command[PETSC_MAX_PATH_LEN];
    FILE  *fd;
    int   err;

    ierr = PetscMemzero(command,sizeof(command));CHKERRQ(ierr);
    ierr = PetscSNPrintf(command,PETSC_MAX_PATH_LEN,"rm -fr %s %s.m4v",draw->savefilename,draw->savefilename);CHKERRQ(ierr);
    ierr = PetscPOpen(PETSC_COMM_SELF,NULL,command,"r",&fd);CHKERRQ(ierr);
    ierr = PetscPClose(PETSC_COMM_SELF,fd,&err);CHKERRQ(ierr);
    ierr = PetscSNPrintf(command,PETSC_MAX_PATH_LEN,"mkdir %s",draw->savefilename);CHKERRQ(ierr);
    ierr = PetscPOpen(PETSC_COMM_SELF,NULL,command,"r",&fd);CHKERRQ(ierr);
    ierr = PetscPClose(PETSC_COMM_SELF,fd,&err);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}


#if defined(PETSC_HAVE_AFTERIMAGE)
#include <afterimage.h>

/* String names of possible Afterimage formats */
const char *PetscAfterImageFormats[] = {
        ".Xpm",
	".Xpm.Z",
	".Xpm.gz",
	".Png",
	".Jpeg",
	".Xcf", /* Gimp format */
	".Ppm",
	".Pnm",
	"MS Windows Bitmap",
	"MS Windows Icon",
	"MS Windows Cursor",
	".Gif",
	".Tiff",
	"Afterstep XMLScript",
	"Scalable Vector Graphics (SVG)",
	".Xbm",
	"Targa",
	".Pcx",
	".HTML",
	"XML",
	"Unknown"
};

#undef __FUNCT__
#define __FUNCT__ "PetscAfterimageStringToFormat"
static PetscErrorCode PetscAfterimageStringToFormat(const char *ext,ASImageFileTypes *format)
{
  PetscInt       i;
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscStrcasecmp(".Jpg",ext,&flg);CHKERRQ(ierr);
  if (flg) ext = ".Jpeg";
  for (i=0; i<sizeof(PetscAfterImageFormats)/sizeof(char**); i++) {
    ierr = PetscStrcasecmp(PetscAfterImageFormats[i],ext,&flg);CHKERRQ(ierr);
    if (flg) {
      *format = (ASImageFileTypes)i;
      PetscFunctionReturn(0);
    }
  }
  *format = ASIT_Unknown;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_SAWS)
#include <petscviewersaws.h>
/*
  The PetscAfterimage object and functions are used to maintain a list of file images created by Afterimage that can
  be displayed by the SAWs webserver.
*/
typedef struct _P_PetscAfterimage *PetscAfterimage;
struct _P_PetscAfterimage {
  PetscAfterimage next;
  char            *filename;
  char            *ext;
  PetscInt        cnt;
} ;

static PetscAfterimage afterimages = 0;

#undef __FUNCT__
#define __FUNCT__ "PetscAfterimageDestroy"
static PetscErrorCode PetscAfterimageDestroy(void)
{
  PetscErrorCode ierr;
  PetscAfterimage       afterimage,oafterimage = afterimages;

  PetscFunctionBegin;
  while (oafterimage) {
    afterimage = oafterimage->next;
    ierr = PetscFree(oafterimage->filename);CHKERRQ(ierr);
    ierr = PetscFree(oafterimage->ext);CHKERRQ(ierr);
    ierr = PetscFree(oafterimage);CHKERRQ(ierr);
    oafterimage = afterimage;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscAfterimageAdd"
static PetscErrorCode PetscAfterimageAdd(const char *filename,const char *ext,PetscInt cnt)
{
  PetscErrorCode   ierr;
  PetscAfterimage  afterimage,oafterimage = afterimages;
  PetscBool        flg;

  PetscFunctionBegin;
  if (oafterimage){
    ierr = PetscStrcmp(filename,oafterimage->filename,&flg);CHKERRQ(ierr);
    if (flg) {
      oafterimage->cnt = cnt;
      PetscFunctionReturn(0);
    }
    while (oafterimage->next) {
      oafterimage = oafterimage->next;
      ierr = PetscStrcmp(filename,oafterimage->filename,&flg);CHKERRQ(ierr);
      if (flg) {
        oafterimage->cnt = cnt;
        PetscFunctionReturn(0);
      }
    }
    ierr = PetscNew(&afterimage);CHKERRQ(ierr);
    oafterimage->next = afterimage;
  } else {
    ierr = PetscNew(&afterimage);CHKERRQ(ierr);
    afterimages = afterimage;
  }
  ierr = PetscStrallocpy(filename,&afterimage->filename);CHKERRQ(ierr);
  ierr = PetscStrallocpy(ext,&afterimage->ext);CHKERRQ(ierr);
  afterimage->cnt = cnt;
  ierr = PetscRegisterFinalize(PetscAfterimageDestroy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif

#undef __FUNCT__
#define __FUNCT__ "PetscDrawSave_X"
PetscErrorCode PetscDrawSave_X(PetscDraw draw)
{
  PetscDraw_X      *drawx = (PetscDraw_X*)draw->data;
  XImage           *image;
  ASImage          *asimage;
  struct  ASVisual *asv;
  char             filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode   ierr;
  PetscMPIInt      rank;
  int              depth;
  ASImageFileTypes format;

  PetscFunctionBegin;
  if (!draw->savefilename) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)draw),&rank);CHKERRQ(ierr);

  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  XSync(drawx->disp,True);
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  ierr = MPI_Barrier(PetscObjectComm((PetscObject)draw));CHKERRQ(ierr);

  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  if (rank) goto finally; /* only process 0 handles the saving business */
  depth = DefaultDepth(drawx->disp,drawx->screen);
  asv   = create_asvisual(drawx->disp,drawx->screen,depth,NULL);if (!asv) SETERRQ(PetscObjectComm((PetscObject)draw),PETSC_ERR_PLIB,"Cannot create AfterImage ASVisual");
  image = XGetImage(drawx->disp,PetscDrawXiDrawable(drawx),0,0,drawx->w,drawx->h,AllPlanes,ZPixmap);
  if (!image) SETERRQ(PetscObjectComm((PetscObject)draw),PETSC_ERR_PLIB,"Cannot XGetImage()");
  asimage = picture_ximage2asimage(asv,image,0,0);if (!asimage) SETERRQ(PetscObjectComm((PetscObject)draw),PETSC_ERR_PLIB,"Cannot create AfterImage ASImage");
  if (draw->savesinglefile) {
    ierr = PetscSNPrintf(filename,PETSC_MAX_PATH_LEN,"%s/%s%s",draw->savefilename,draw->savefilename,draw->savefilenameext);CHKERRQ(ierr);
  } else {
    ierr = PetscSNPrintf(filename,PETSC_MAX_PATH_LEN,"%s/%s_%d%s",draw->savefilename,draw->savefilename,draw->savefilecount++,draw->savefilenameext);CHKERRQ(ierr);
  }
  ierr = PetscAfterimageStringToFormat(draw->savefilenameext,&format);CHKERRQ(ierr);
  ASImage2file(asimage,0,filename,format,0);
#if defined(PETSC_HAVE_SAWS)
  {
    char     body[4096];
    PetscAfterimage afterimage;
    size_t   len = 0;

    ierr = PetscAfterimageAdd(draw->savefilename,draw->savefilenameext,draw->savefilecount-1);CHKERRQ(ierr);
    afterimage  = afterimages;
    while (afterimage) {
      if (draw->savesinglefile) {
        ierr = PetscSNPrintf(body+len,4086-len,"<img src=\"%s/%s%s\" alt=\"None\">",afterimage->filename,afterimage->filename,afterimage->ext);CHKERRQ(ierr);
      } else {
        ierr = PetscSNPrintf(body+len,4086-len,"<img src=\"%s/%s_%d%s\" alt=\"None\">",afterimage->filename,afterimage->filename,afterimage->cnt,afterimage->ext);CHKERRQ(ierr);
      }
      ierr = PetscStrlen(body,&len);CHKERRQ(ierr);
      afterimage  = afterimage->next;
    }
    ierr = PetscStrcat(body,"<br>\n");CHKERRQ(ierr);
    if (draw->savefilecount > 0) PetscStackCallSAWs(SAWs_Pop_Body,("index.html",1));
    PetscStackCallSAWs(SAWs_Push_Body,("index.html",1,body));
  }
#endif
  destroy_asvisual(asv,0);
  XDestroyImage(image);
finally:
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   There are routines wanted by AfterImage for PNG files
 */
void crc32(void) {;}
void inflateReset(void) {;}
void deflateReset(void) {;}
void deflateInit2(void) {;}
void deflateInit2_(void) {;}
void deflate(void) {;}
void deflateEnd(void) {;}

#endif



