
/*
    Defines the operations for the OpenGL PetscDraw implementation.

    Currently mixes in glut... calls. 
    The eventual plan is to have a window system independent portion (that only uses gl...() routines) plus
      several implementations for different Window interfaces: 
       --  glut,
       --  Apple IOS  EAGLContext  https://developer.apple.com/library/ios/#documentation/3DDrawing/Conceptual/OpenGLES_ProgrammingGuide/WorkingwithOpenGLESContexts/WorkingwithOpenGLESContexts.html#//apple_ref/doc/uid/TP40008793-CH2-SW1
       --  Apple  NSOpenGLView  http://developer.apple.com/library/mac/#documentation/graphicsimaging/conceptual/OpenGL-MacProgGuide/opengl_pg_concepts/opengl_pg_concepts.html#//apple_ref/doc/uid/TP40001987-CH208-SW1
       --  Apple  CGLContextObj http://developer.apple.com/library/mac/#documentation/graphicsimaging/Reference/CGL_OpenGL/Reference/reference.html#//apple_ref/doc/uid/TP40001186
*/

#include <../src/sys/draw/impls/opengl/openglimpl.h>         /*I  "petscsys.h" I*/

/*
     These macros transform from the users coordinates to the  OpenGL coordinates.
*/
#define XTRANS(draw,xwin,x)  (-1.0 + 2.0*((draw)->port_xl + (((x - (draw)->coor_xl)*((draw)->port_xr - (draw)->port_xl))/((draw)->coor_xr - (draw)->coor_xl))))
#define YTRANS(draw,xwin,y)  (-1.0 + 2.0*((draw)->port_yl + (((y - (draw)->coor_yl)*((draw)->port_yr - (draw)->port_yl))/((draw)->coor_yr - (draw)->coor_yl))))

static unsigned char rcolor[256],gcolor[256],bcolor[256];

static int currentcolor = 0;
PETSC_STATIC_INLINE PetscErrorCode OpenGLColor(int icolor)
{
  if (icolor >= 256 || icolor < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Color value out of range");
  if (icolor == currentcolor) return 0;
  currentcolor = icolor;
  glColor3ub(rcolor[icolor],gcolor[icolor],bcolor[icolor]);
  return 0;
}

/*
    The next two routines depend on the Window system used with OpenGL
*/
static int currentwindow = -1;
PETSC_STATIC_INLINE PetscErrorCode OpenGLWindow(PetscDraw_OpenGL *win)
{
  if (win->win == currentwindow) return 0;
  currentwindow = win->win;
  glutSetWindow(win->win);
  return 0;
}

PETSC_STATIC_INLINE PetscErrorCode OpenGLString(float x,float y, const char *str,size_t len)
{
  PetscInt         i;

  glRasterPos2f(x, y);
  for (i = 0; i < len; i++) {
    glutBitmapCharacter(GLUT_BITMAP_8_BY_13, str[i]);
  }
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSetTitle_OpenGL" 
static PetscErrorCode PetscDrawSetTitle_OpenGL(PetscDraw draw,const char title[])
{
  PetscDraw_OpenGL *win = (PetscDraw_OpenGL*)draw->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = OpenGLWindow(win);CHKERRQ(ierr);
  glutSetWindowTitle(title);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawDestroy_OpenGL" 
PetscErrorCode PetscDrawDestroy_OpenGL(PetscDraw draw)
{
  PetscDraw_OpenGL *win = (PetscDraw_OpenGL*)draw->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDrawSynchronizedClear(draw);CHKERRQ(ierr);
  glutDestroyWindow(win->win);
  ierr = PetscFree(win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscDrawFlush_OpenGL" 
static PetscErrorCode PetscDrawFlush_OpenGL(PetscDraw draw)
{
  PetscDraw_OpenGL* XiWin = (PetscDraw_OpenGL*)draw->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = OpenGLWindow(XiWin);CHKERRQ(ierr);
  glutCheckLoop();
  glFinish();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawStringGetSize_OpenGL" 
PetscErrorCode PetscDrawStringGetSize_OpenGL(PetscDraw draw,PetscReal *x,PetscReal  *y)
{
  PetscDraw_OpenGL* XiWin = (PetscDraw_OpenGL*)draw->data;
  PetscInt          w;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = OpenGLWindow(XiWin);CHKERRQ(ierr);
  w = glutBitmapWidth(GLUT_BITMAP_8_BY_13,'W');
  *x = w*(draw->coor_xr - draw->coor_xl)/((XiWin->w)*(draw->port_xr - draw->port_xl));
  *y = (13./8.0)*w*(draw->coor_yr - draw->coor_yl)/((XiWin->h)*(draw->port_yr - draw->port_yl));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawResizeWindow_OpenGL" 
static PetscErrorCode PetscDrawResizeWindow_OpenGL(PetscDraw draw,int w,int h)
{
  PetscDraw_OpenGL *win = (PetscDraw_OpenGL*)draw->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = OpenGLWindow(win);CHKERRQ(ierr);
  glutReshapeWindow(w,h);
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscBool resized = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawCheckResizedWindow_OpenGL" 
static PetscErrorCode PetscDrawCheckResizedWindow_OpenGL(PetscDraw draw)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (!resized) PetscFunctionReturn(0);
  resized = PETSC_FALSE;
  ierr = PetscDrawClear(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  int button,x,y;
} OpenGLButton;
OpenGLButton Mouse;

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawGetMouseButton_OpenGL" 
static PetscErrorCode PetscDrawGetMouseButton_OpenGL(PetscDraw draw,PetscDrawButton *button,PetscReal* x_user,PetscReal *y_user,PetscReal *x_phys,PetscReal *y_phys)
{
  PetscDraw_OpenGL *win = (PetscDraw_OpenGL*)draw->data;
  int              px,py;

  PetscFunctionBegin;
  while (Mouse.button == -1) {glutCheckLoop();}
  *button = (PetscDrawButton)(Mouse.button + 1);
  px      = Mouse.x;
  py      = Mouse.y;
  Mouse.button = -1;

  if (x_phys) *x_phys = ((double)px)/((double)win->w);
  if (y_phys) *y_phys = 1.0 - ((double)py)/((double)win->h);

  if (x_user) *x_user = draw->coor_xl + ((((double)px)/((double)win->w)-draw->port_xl))*(draw->coor_xr - draw->coor_xl)/(draw->port_xr - draw->port_xl);
  if (y_user) *y_user = draw->coor_yl + ((1.0 - ((double)py)/((double)win->h)-draw->port_yl))*(draw->coor_yr - draw->coor_yl)/(draw->port_yr - draw->port_yl);

  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSynchronizedFlush_OpenGL" 
static PetscErrorCode PetscDrawSynchronizedFlush_OpenGL(PetscDraw draw)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* currently on sequential support */
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawClear_OpenGL" 
static PetscErrorCode PetscDrawClear_OpenGL(PetscDraw draw)
{
  PetscDraw_OpenGL* XiWin = (PetscDraw_OpenGL*)draw->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* currently clear entire window, need to only clear single port */
  ierr = OpenGLWindow(XiWin);CHKERRQ(ierr);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_ACCUM_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSynchronizedClear_OpenGL" 
static PetscErrorCode PetscDrawSynchronizedClear_OpenGL(PetscDraw draw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* currently only sequential support */
  ierr = PetscDrawClear(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscDrawPoint_OpenGL" 
PetscErrorCode PetscDrawPoint_OpenGL(PetscDraw draw,PetscReal xl,PetscReal yl,int cl)
{
  PetscDraw_OpenGL *win = (PetscDraw_OpenGL*)draw->data;
  float             x1,y_1;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  x1  = XTRANS(draw,XiWin,xl);   
  y_1 = YTRANS(draw,XiWin,yl);   

  ierr = OpenGLWindow(win);CHKERRQ(ierr);
  ierr = OpenGLColor(cl);CHKERRQ(ierr);
  glBegin(GL_POINTS);
  glVertex2f(x1,y_1);
  glEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawLine_OpenGL" 
PetscErrorCode PetscDrawLine_OpenGL(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int cl)
{
  PetscDraw_OpenGL *win = (PetscDraw_OpenGL*)draw->data;
  float             x1,y_1,x2,y2;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  x1  = XTRANS(draw,XiWin,xl);   
  x2  = XTRANS(draw,XiWin,xr); 
  y_1 = YTRANS(draw,XiWin,yl);   
  y2  = YTRANS(draw,XiWin,yr); 
  if (x1 == x2 && y_1 == y2) PetscFunctionReturn(0);

  ierr = OpenGLWindow(win);CHKERRQ(ierr);
  ierr = OpenGLColor(cl);CHKERRQ(ierr);
  glBegin(GL_LINES);
  glVertex2f(x1,y_1);
  glVertex2f(x2,y2);
  glEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawTriangle_OpenGL" 
static PetscErrorCode PetscDrawTriangle_OpenGL(PetscDraw draw,PetscReal X1,PetscReal Y_1,PetscReal X2,PetscReal Y2,PetscReal X3,PetscReal Y3,int c1,int c2,int c3)
{
  PetscDraw_OpenGL *win = (PetscDraw_OpenGL*)draw->data;
  PetscErrorCode    ierr;
  float             x1,y_1,x2,y2,x3,y3;

  PetscFunctionBegin;
  x1   = XTRANS(draw,XiWin,X1);
  y_1  = YTRANS(draw,XiWin,Y_1); 
  x2   = XTRANS(draw,XiWin,X2);
  y2   = YTRANS(draw,XiWin,Y2); 
  x3   = XTRANS(draw,XiWin,X3);
  y3   = YTRANS(draw,XiWin,Y3); 

  ierr = OpenGLWindow(win);CHKERRQ(ierr);
  glBegin(GL_TRIANGLES);
  ierr = OpenGLColor(c1);CHKERRQ(ierr);
  glVertex2f(x1,y_1);
  ierr = OpenGLColor(c2);CHKERRQ(ierr);
  glVertex2f(x2,y2);
  ierr = OpenGLColor(c3);CHKERRQ(ierr);
  glVertex2f(x3,y3);
  glEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawRectangle_OpenGL" 
static PetscErrorCode PetscDrawRectangle_OpenGL(PetscDraw draw,PetscReal Xl,PetscReal Yl,PetscReal Xr,PetscReal Yr,int c1,int c2,int c3,int c4)
{
  PetscDraw_OpenGL *win = (PetscDraw_OpenGL*)draw->data;
  PetscErrorCode    ierr;
  float             x1,y_1,x2,y2;
  int               c = (c1 + c2 + c3 + c4)/4;

  PetscFunctionBegin;
  x1   = XTRANS(draw,XiWin,Xl);
  y_1  = YTRANS(draw,XiWin,Yl); 
  x2   = XTRANS(draw,XiWin,Xr);
  y2   = YTRANS(draw,XiWin,Yr); 

  ierr = OpenGLWindow(win);CHKERRQ(ierr);
  ierr = OpenGLColor(c);CHKERRQ(ierr);
  glBegin(GL_QUADS);
  glVertex2f(x1,y_1);
  glVertex2f(x2,y_1);
  glVertex2f(x2,y2);
  glVertex2f(x1,y2);
  glEnd();
  PetscFunctionReturn(0);
}



#undef __FUNCT__  
#define __FUNCT__ "PetscDrawString_OpenGL" 
static PetscErrorCode PetscDrawString_OpenGL(PetscDraw draw,PetscReal x,PetscReal  y,int c,const char chrs[])
{
  PetscErrorCode   ierr;
  float            xx,yy;
  size_t           len;
  PetscDraw_OpenGL *XiWin = (PetscDraw_OpenGL*)draw->data;
  char             *substr;
  PetscToken       token;

  PetscFunctionBegin;
  xx = XTRANS(draw,XiWin,x); 
  yy = YTRANS(draw,XiWin,y);
  ierr = OpenGLWindow(XiWin);CHKERRQ(ierr);
  ierr = OpenGLColor(c);CHKERRQ(ierr);
  
  ierr = PetscTokenCreate(chrs,'\n',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&substr);CHKERRQ(ierr);
  ierr = PetscStrlen(substr,&len);CHKERRQ(ierr);
  ierr = OpenGLString(xx,yy,substr,len);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&substr);CHKERRQ(ierr);
  while (substr) {
    yy  += 16;
    ierr = PetscStrlen(substr,&len);CHKERRQ(ierr);
    ierr = OpenGLString(xx,yy,substr,len);CHKERRQ(ierr);
    ierr = PetscTokenFind(token,&substr);CHKERRQ(ierr);
  }
  ierr = PetscTokenDestroy(&token);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawStringVertical_OpenGL" 
PetscErrorCode PetscDrawStringVertical_OpenGL(PetscDraw draw,PetscReal x,PetscReal  y,int c,const char chrs[])
{
  PetscErrorCode   ierr;
  float            xx,yy;
  PetscDraw_OpenGL *XiWin = (PetscDraw_OpenGL*)draw->data;
  PetscReal        tw,th;
  size_t           i,n;

  PetscFunctionBegin;
  ierr = OpenGLWindow(XiWin);CHKERRQ(ierr);
  ierr = OpenGLColor(c);CHKERRQ(ierr);
  ierr = PetscStrlen(chrs,&n);CHKERRQ(ierr);
  ierr = PetscDrawStringGetSize(draw,&tw,&th);CHKERRQ(ierr);
  xx = XTRANS(draw,XiWin,x);
  for (i=0; i<n; i++) {
    yy = YTRANS(draw,XiWin,y-th*i);
    ierr = OpenGLString(xx,yy,chrs+i,1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscDrawPause_OpenGL" 
static PetscErrorCode PetscDrawPause_OpenGL(PetscDraw draw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (draw->pause > 0) PetscSleep(draw->pause);
  else if (draw->pause < 0) {
    PetscDrawButton button;
    PetscMPIInt     rank;
    ierr = MPI_Comm_rank(((PetscObject)draw)->comm,&rank);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscDrawGetMouseButton(draw,&button,0,0,0,0);CHKERRQ(ierr);
      if (button == PETSC_BUTTON_CENTER) draw->pause = 0;
    }
    ierr = MPI_Bcast(&draw->pause,1,MPI_INT,0,((PetscObject)draw)->comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscDrawGetPopup_OpenGL" 
static PetscErrorCode PetscDrawGetPopup_OpenGL(PetscDraw draw,PetscDraw *popup)
{
  PetscErrorCode   ierr;
  PetscDraw_OpenGL *win = (PetscDraw_OpenGL*)draw->data;

  PetscFunctionBegin;
  ierr = PetscDrawOpenOpenGL(((PetscObject)draw)->comm,PETSC_NULL,PETSC_NULL,win->x,win->y+win->h+36,220,220,popup);CHKERRQ(ierr);
  draw->popup = *popup;
  PetscFunctionReturn(0);
}

static struct _PetscDrawOps DvOps = { 0,
                                 PetscDrawFlush_OpenGL,
                                 PetscDrawLine_OpenGL,
                                 0,
                                 0,
                                 PetscDrawPoint_OpenGL,
                                 0,
                                      PetscDrawString_OpenGL,
                                      PetscDrawStringVertical_OpenGL,
                                      0, /* PetscDrawStringSetSize_OpenGL,*/
                                      PetscDrawStringGetSize_OpenGL,
                                      0, /* PetscDrawSetViewport_OpenGL,*/
                                 PetscDrawClear_OpenGL,
                                 PetscDrawSynchronizedFlush_OpenGL,
                                 PetscDrawRectangle_OpenGL, 
                                 PetscDrawTriangle_OpenGL,
                                      0, /* PetscDrawEllipse_OpenGL,*/
                                      PetscDrawGetMouseButton_OpenGL,
                                      PetscDrawPause_OpenGL,
                                 PetscDrawSynchronizedClear_OpenGL,
				 0,
                                 0,
                                      PetscDrawGetPopup_OpenGL,
                                      PetscDrawSetTitle_OpenGL,
                                      PetscDrawCheckResizedWindow_OpenGL, 
                                      PetscDrawResizeWindow_OpenGL,
                                 PetscDrawDestroy_OpenGL,
                                 0,
                                      0, /*PetscDrawGetSingleton_OpenGL,*/
                                      0, /* PetscDrawRestoreSingleton_OpenGL,*/
                                 0,
                                 0,
                                 0,
                                 0};

/* dummy display required by GLUT */
static void display(void) {;}
static void reshape(int width, int height)
{
  glViewport(0, 0, width, height);
  resized = PETSC_TRUE;
}
static void mouse(int button, int state,int x, int y)
{
  if (state == GLUT_UP) {
    Mouse.button = button;
    Mouse.x      = x;
    Mouse.y      = y;
  }
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscDrawCreate_OpenGL" 
PetscErrorCode  PetscDrawCreate_OpenGL(PetscDraw draw)
{
  PetscDraw_OpenGL *Xwin;
  PetscErrorCode   ierr;
  PetscInt         xywh[4],osize = 4;
  int              x = draw->x,y = draw->y,w = draw->w,h = draw->h;
  static int       xavailable = 0,yavailable = 0,xmax = 0,ymax = 0,ybottom = 0;
  static PetscBool initialized = PETSC_FALSE;
  int              argc;
  char             **argv;

  PetscFunctionBegin;

  /*
      Initialize the display size
  */
  if (!xmax) {
    xmax = glutGet(GLUT_SCREEN_WIDTH);
    ymax = glutGet(GLUT_SCREEN_HEIGHT);
  }

  if (w == PETSC_DECIDE) w = draw->w = 300;
  if (h == PETSC_DECIDE) h = draw->h = 300; 
  switch (w) {
    case PETSC_DRAW_FULL_SIZE: w = draw->w = xmax - 10;
                         break;
    case PETSC_DRAW_HALF_SIZE: w = draw->w = (xmax - 20)/2;
                         break;
    case PETSC_DRAW_THIRD_SIZE: w = draw->w = (xmax - 30)/3;
                         break;
    case PETSC_DRAW_QUARTER_SIZE: w = draw->w = (xmax - 40)/4;
                         break;
  }
  switch (h) {
    case PETSC_DRAW_FULL_SIZE: h = draw->h = ymax - 10;
                         break;
    case PETSC_DRAW_HALF_SIZE: h = draw->h = (ymax - 20)/2;
                         break;
    case PETSC_DRAW_THIRD_SIZE: h = draw->h = (ymax - 30)/3;
                         break;
    case PETSC_DRAW_QUARTER_SIZE: h = draw->h = (ymax - 40)/4;
                         break;
  }

  /* allow user to set location and size of window */
  xywh[0] = x; xywh[1] = y; xywh[2] = w; xywh[3] = h;
  ierr = PetscOptionsGetIntArray(PETSC_NULL,"-geometry",xywh,&osize,PETSC_NULL);CHKERRQ(ierr);
  x = (int) xywh[0]; y = (int) xywh[1]; w = (int) xywh[2]; h = (int) xywh[3];


  if (draw->x == PETSC_DECIDE || draw->y == PETSC_DECIDE) {
    /*
       PETSc tries to place windows starting in the upper left corner and 
       moving across to the right. 
    
              --------------------------------------------
              |  Region used so far +xavailable,yavailable |
              |                     +                      |
              |                     +                      |
              |++++++++++++++++++++++ybottom               |
              |                                            |
              |                                            |
              |--------------------------------------------|
    */
    /*  First: can we add it to the right? */
    if (xavailable+w+10 <= xmax) {
      x       = xavailable;
      y       = yavailable;
      ybottom = PetscMax(ybottom,y + h + 30);
    } else {
      /* No, so add it below on the left */
      xavailable = 0;
      x          = 0;
      yavailable = ybottom;    
      y          = ybottom;
      ybottom    = ybottom + h + 30;
    }
  }
  /* update available region */
  xavailable = PetscMax(xavailable,x + w + 10);
  if (xavailable >= xmax) {
    xavailable = 0;
    yavailable = yavailable + h + 30;
    ybottom    = yavailable;
  }
  if (yavailable >= ymax) {
    y          = 0;
    yavailable = 0;
    ybottom    = 0;
  }

  ierr = PetscMemcpy(draw->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);

  /* actually create and open the window */
  ierr = PetscNew(PetscDraw_OpenGL,&Xwin);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(draw,sizeof(PetscDraw_OpenGL));CHKERRQ(ierr);

  if (x < 0 || y < 0)   SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative corner of window");
  if (w <= 0 || h <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative window width or height");

  Xwin->x      = x;
  Xwin->y      = y;
  Xwin->w      = w;
  Xwin->h      = h;

  if (!initialized) {
    ierr = PetscGetArgs(&argc,&argv);CHKERRQ(ierr);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA /* | GLUT_DOUBLE */| GLUT_DEPTH);
    initialized = PETSC_TRUE;

    rcolor[PETSC_DRAW_WHITE] =           255;
    gcolor[PETSC_DRAW_WHITE] =  255;
    bcolor[PETSC_DRAW_WHITE] =    255;
    rcolor[PETSC_DRAW_BLACK] =           0;
    gcolor[PETSC_DRAW_BLACK] = 0;
    bcolor[PETSC_DRAW_BLACK] = 0;
    rcolor[PETSC_DRAW_RED] =             255;
    gcolor[PETSC_DRAW_RED] =  0;
    bcolor[PETSC_DRAW_RED] =  0;
    rcolor[PETSC_DRAW_GREEN] =           0;
    gcolor[PETSC_DRAW_GREEN] =128;
    bcolor[PETSC_DRAW_GREEN] =  0;
    rcolor[PETSC_DRAW_CYAN] =            0;
    gcolor[PETSC_DRAW_CYAN] = 139;
    bcolor[PETSC_DRAW_CYAN] =    139;
    rcolor[PETSC_DRAW_BLUE] =            0;
    gcolor[PETSC_DRAW_BLUE] = 0;
    bcolor[PETSC_DRAW_BLUE] =     255;
    rcolor[PETSC_DRAW_MAGENTA] =         255;
    gcolor[PETSC_DRAW_MAGENTA] =    0;
    bcolor[PETSC_DRAW_MAGENTA] =    255;
    rcolor[PETSC_DRAW_AQUAMARINE] =      127;
    gcolor[PETSC_DRAW_AQUAMARINE] = 255;
    bcolor[PETSC_DRAW_AQUAMARINE] =   212;
    rcolor[PETSC_DRAW_FORESTGREEN] =     34;
    gcolor[PETSC_DRAW_FORESTGREEN] = 139;
    bcolor[PETSC_DRAW_FORESTGREEN] =  34;
    rcolor[PETSC_DRAW_ORANGE] =          255;
    gcolor[PETSC_DRAW_ORANGE] = 165;
    bcolor[PETSC_DRAW_ORANGE] = 0;
    rcolor[PETSC_DRAW_VIOLET] =          238;
    gcolor[PETSC_DRAW_VIOLET] =  130;
    bcolor[PETSC_DRAW_VIOLET] =  238;
    rcolor[PETSC_DRAW_BROWN] =           165;
    gcolor[PETSC_DRAW_BROWN] = 42;
    bcolor[PETSC_DRAW_BROWN] = 42;
    rcolor[PETSC_DRAW_PINK] =            255;
    gcolor[PETSC_DRAW_PINK] = 192;
    bcolor[PETSC_DRAW_PINK] = 203;
    rcolor[PETSC_DRAW_CORAL] =           255;
    gcolor[PETSC_DRAW_CORAL] = 127;
    bcolor[PETSC_DRAW_CORAL] = 80;
    rcolor[PETSC_DRAW_GRAY] =            128;
    gcolor[PETSC_DRAW_GRAY] = 128;
    bcolor[PETSC_DRAW_GRAY] = 128;
    rcolor[PETSC_DRAW_YELLOW] =          255;
    gcolor[PETSC_DRAW_YELLOW] = 255;
    bcolor[PETSC_DRAW_YELLOW] =   0;
    rcolor[PETSC_DRAW_GOLD] =            255;
    gcolor[PETSC_DRAW_GOLD] =   215;
    bcolor[PETSC_DRAW_GOLD] =   0;
    rcolor[PETSC_DRAW_LIGHTPINK] =       255;
    gcolor[PETSC_DRAW_LIGHTPINK] = 182;
    bcolor[PETSC_DRAW_LIGHTPINK] = 193;
    rcolor[PETSC_DRAW_MEDIUMTURQUOISE] = 72;
    gcolor[PETSC_DRAW_MEDIUMTURQUOISE] =  209;
    bcolor[PETSC_DRAW_MEDIUMTURQUOISE] = 204;
    rcolor[PETSC_DRAW_KHAKI] =           240;
    gcolor[PETSC_DRAW_KHAKI] =230;
    bcolor[PETSC_DRAW_KHAKI] =140;
    rcolor[PETSC_DRAW_DIMGRAY] =         105;
    gcolor[PETSC_DRAW_DIMGRAY] = 105;
    bcolor[PETSC_DRAW_DIMGRAY] = 105;
    rcolor[PETSC_DRAW_YELLOWGREEN] =     54;
    gcolor[PETSC_DRAW_YELLOWGREEN] =   205;
    bcolor[PETSC_DRAW_YELLOWGREEN] =  50;
    rcolor[PETSC_DRAW_SKYBLUE] =         135;
    gcolor[PETSC_DRAW_SKYBLUE] =   206;
    bcolor[PETSC_DRAW_SKYBLUE] = 235;
    rcolor[PETSC_DRAW_DARKGREEN] =       0;
    gcolor[PETSC_DRAW_DARKGREEN] =    100;
    bcolor[PETSC_DRAW_DARKGREEN] = 0;
    rcolor[PETSC_DRAW_NAVYBLUE] =       0;
    gcolor[PETSC_DRAW_NAVYBLUE] =    0;
    bcolor[PETSC_DRAW_NAVYBLUE] = 128;
    rcolor[PETSC_DRAW_SANDYBROWN] =      244;
    gcolor[PETSC_DRAW_SANDYBROWN] =  164;
    bcolor[PETSC_DRAW_SANDYBROWN] =   96;
    rcolor[PETSC_DRAW_CADETBLUE] =      95;
    gcolor[PETSC_DRAW_CADETBLUE] =  158;
    bcolor[PETSC_DRAW_CADETBLUE] =  160;
    rcolor[PETSC_DRAW_POWDERBLUE] =     176;
    gcolor[PETSC_DRAW_POWDERBLUE] =  224;
    bcolor[PETSC_DRAW_POWDERBLUE] =  230;
    rcolor[PETSC_DRAW_DEEPPINK] =       255;
    gcolor[PETSC_DRAW_DEEPPINK] =    20;
    bcolor[PETSC_DRAW_DEEPPINK] =    147;
    rcolor[PETSC_DRAW_THISTLE] =        216;
    gcolor[PETSC_DRAW_THISTLE] =  191;
    bcolor[PETSC_DRAW_THISTLE] =   216;
    rcolor[PETSC_DRAW_LIMEGREEN] =      50;
    gcolor[PETSC_DRAW_LIMEGREEN] =   205;
    bcolor[PETSC_DRAW_LIMEGREEN] =   50;
    rcolor[PETSC_DRAW_LAVENDERBLUSH] =  255;
    gcolor[PETSC_DRAW_LAVENDERBLUSH] =     240;
    bcolor[PETSC_DRAW_LAVENDERBLUSH] =    245;
    rcolor[PETSC_DRAW_PLUM] =           221;
    gcolor[PETSC_DRAW_PLUM] =  160;
    bcolor[PETSC_DRAW_PLUM] =   221;

    ierr    = PetscDrawUtilitySetCmapHue(rcolor+PETSC_DRAW_BASIC_COLORS,gcolor+PETSC_DRAW_BASIC_COLORS,bcolor+PETSC_DRAW_BASIC_COLORS,256-PETSC_DRAW_BASIC_COLORS);CHKERRQ(ierr);
  }
  glutInitWindowSize(w, h);
  Xwin->win = glutCreateWindow(draw->title);
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  Mouse.button = -1;
  Mouse.x      = -1;
  Mouse.y      = -1;
  glutMouseFunc(mouse);

  glClearColor(1.0,1.0,1.0,1.0);
  /*   glClearIndex();*/

  draw->data = Xwin;
  ierr = PetscDrawClear(draw);CHKERRQ(ierr);
  resized = PETSC_FALSE; /* opening the window triggers OpenGL call to reshape so need to cancel that resized flag */
  glutCheckLoop();
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "PetscDrawOpenOpenGL" 
/*@C
   PetscDrawOpenOpenGL - Opens an OpenGL for use with the PetscDraw routines.

   Collective on MPI_Comm

   Input Parameters:
+  comm - the communicator that will share X-window
.  display - the X display on which to open,or null for the local machine
.  title - the title to put in the title bar,or null for no title
.  x,y - the screen coordinates of the upper left corner of window
          may use PETSC_DECIDE for these two arguments, then PETSc places the 
          window
-  w, h - the screen width and height in pixels,  or PETSC_DRAW_HALF_SIZE, PETSC_DRAW_FULL_SIZE,
          or PETSC_DRAW_THIRD_SIZE or PETSC_DRAW_QUARTER_SIZE

   Output Parameters:
.  draw - the drawing context.

   Options Database Keys:
+  -nox - Disables all x-windows output
.  -draw_pause <pause> - Sets time (in seconds) that the
       program pauses after PetscDrawPause() has been called
       (0 is default, -1 implies until user input).

   Level: beginner

   Note:
   When finished with the drawing context, it should be destroyed
   with PetscDrawDestroy().

   Note for Fortran Programmers:
   Whenever indicating null character data in a Fortran code,
   PETSC_NULL_CHARACTER must be employed; using PETSC_NULL is not
   correct for character data!  Thus, PETSC_NULL_CHARACTER can be
   used for the display and title input parameters.

   Concepts: OpenGL^drawing to

.seealso: PetscDrawSynchronizedFlush(), PetscDrawDestroy(), PetscDrawOpenX(), PetscDrawCreate()
@*/
PetscErrorCode  PetscDrawOpenOpenGL(MPI_Comm comm,const char display[],const char title[],int x,int y,int w,int h,PetscDraw* draw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDrawCreate(comm,display,title,x,y,w,h,draw);CHKERRQ(ierr);
  ierr = PetscDrawSetType(*draw,PETSC_DRAW_OPENGL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}







