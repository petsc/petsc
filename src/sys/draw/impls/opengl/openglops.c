
/*
    Defines the operations for the OpenGL PetscDraw implementation.
*/

#include <../src/sys/draw/impls/opengl/openglimpl.h>         /*I  "petscsys.h" I*/

/*
     These macros transform from the users coordinates to the  OpenGL coordinates.
*/
#define XTRANS(draw,xwin,x)  (int)(((xwin)->w)*((draw)->port_xl + (((x - (draw)->coor_xl)*((draw)->port_xr - (draw)->port_xl))/((draw)->coor_xr - (draw)->coor_xl))))
#define YTRANS(draw,xwin,y)  (int)(((xwin)->h)*(1.0-(draw)->port_yl - (((y - (draw)->coor_yl)*((draw)->port_yr - (draw)->port_yl))/((draw)->coor_yr - (draw)->coor_yl))))


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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscDrawSave(draw);CHKERRQ(ierr);
  /* currently clear entire window, need to only clear single port */
  glutSetWindow(XiWin->win);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
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

static struct _PetscDrawOps DvOps = { 0,
                                 PetscDrawFlush_OpenGL,
                                      0, /*PetscDrawLine_OpenGL, */
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
  }
  glutInitWindowSize(w, h);
  Xwin->win = glutCreateWindow("GLUT Program");
  glutCheckLoop();
  PetscFunctionReturn(0);
}
EXTERN_C_END









