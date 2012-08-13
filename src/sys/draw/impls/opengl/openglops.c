
/*
    Defines the operations for the OpenGL PetscDraw implementation.
*/

#include <../src/sys/draw/impls/opengl/openglimpl.h>         /*I  "petscsys.h" I*/

/*
     These macros transform from the users coordinates to the  OpenGL coordinates.
*/
#define XTRANS(draw,xwin,x)  (-1.0 + 2.0*((draw)->port_xl + (((x - (draw)->coor_xl)*((draw)->port_xr - (draw)->port_xl))/((draw)->coor_xr - (draw)->coor_xl))))
#define YTRANS(draw,xwin,y)  (-1.0 + 2.0*((draw)->port_yl + (((y - (draw)->coor_yl)*((draw)->port_yr - (draw)->port_yl))/((draw)->coor_yr - (draw)->coor_yl))))

static unsigned char rcolor[256],gcolor[256],bcolor[256];

static int currentcolor = 0;
PETSC_STATIC_INLINE PetscErrorCode OpenGLColor(icolor){
  if (icolor >= 256 || icolor < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Color value out of range");
  if (icolor == currentcolor) return 0;
  currentcolor = icolor;
  glColor3ub(rcolor[icolor],gcolor[icolor],bcolor[icolor]);
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawFlush_OpenGL" 
static PetscErrorCode PetscDrawFlush_OpenGL(PetscDraw draw)
{
  PetscDraw_OpenGL* XiWin = (PetscDraw_OpenGL*)draw->data;

  PetscFunctionBegin;
  glutSetWindow(XiWin->win);
  glutSwapBuffers();
  glutCheckLoop();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSynchronizedFlush_OpenGL" 
static PetscErrorCode PetscDrawSynchronizedFlush_OpenGL(PetscDraw draw)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* currently on sequential support */
  ierr = PetscDrawFlush_OpenGL(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawClear_OpenGL" 
static PetscErrorCode PetscDrawClear_OpenGL(PetscDraw draw)
{
  PetscDraw_OpenGL* XiWin = (PetscDraw_OpenGL*)draw->data;

  PetscFunctionBegin;
  /* currently clear entire window, need to only clear single port */
  glutSetWindow(XiWin->win);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glutSwapBuffers();
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
#define __FUNCT__ "PetscDrawLine_OpenGL" 
PetscErrorCode PetscDrawLine_OpenGL(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int cl)
{
  float             x1,y_1,x2,y2;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  x1 = XTRANS(draw,XiWin,xl);   x2  = XTRANS(draw,XiWin,xr); 
  y_1 = YTRANS(draw,XiWin,yl);   y2  = YTRANS(draw,XiWin,yr); 
  if (x1 == x2 && y_1 == y2) PetscFunctionReturn(0);

  ierr = OpenGLColor(cl);CHKERRQ(ierr);
  glBegin(GL_LINES);
  glVertex3f(x1,y_1,0.0);
  glVertex3f(x2,y2,0.0);
  glEnd();
  PetscFunctionReturn(0);
}




static struct _PetscDrawOps DvOps = { 0,
                                 PetscDrawFlush_OpenGL,
                                 PetscDrawLine_OpenGL,
                                 0,
                                 0,
                                      0, /* PetscDrawPoint_OpenGL,*/
                                 0,
                                      0, /* PetscDrawString_OpenGL,*/
                                      0,  /*PetscDrawStringVertical_OpenGL, */
                                      0, /* PetscDrawStringSetSize_OpenGL,*/
                                      0, /* PetscDrawStringGetSize_OpenGL,*/
                                      0, /* PetscDrawSetViewport_OpenGL,*/
                                 PetscDrawClear_OpenGL,
                                 PetscDrawSynchronizedFlush_OpenGL,
                                      0, /*PetscDrawRectangle_OpenGL, */
                                      0, /*PetscDrawTriangle_OpenGL, */
                                      0, /* PetscDrawEllipse_OpenGL,*/
                                      0, /*PetscDrawGetMouseButton_OpenGL,*/
                                      0, /*PetscDrawPause_OpenGL,*/
                                 PetscDrawSynchronizedClear_OpenGL,
				 0,
                                 0,
                                      0,/*PetscDrawGetPopup_OpenGL,*/
                                      0, /*PetscDrawSetTitle_OpenGL,*/
                                      0, /*PetscDrawCheckResizedWindow_OpenGL, */
                                      0, /* PetscDrawResizeWindow_OpenGL,*/
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
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
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
  Xwin->win = glutCreateWindow("GLUT Program");
  glutDisplayFunc(display);
  glClearColor(1.0,1.0,1.0,1.00);
  /*   glClearIndex();*/

  draw->data = Xwin;
  ierr = PetscDrawClear(draw);CHKERRQ(ierr);
  glutCheckLoop();
  PetscFunctionReturn(0);
}
EXTERN_C_END









