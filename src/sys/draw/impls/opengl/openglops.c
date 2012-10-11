
/*
    Defines the operations for the OpenGL PetscDraw implementation.

    The eventual plan is to have a window system independent portion (that only uses gl...() routines) plus
      several implementations for different Window interfaces: 
       --  glut (implmented)
       --  Apple IOS  EAGLContext  https://developer.apple.com/library/ios/#documentation/3DDrawing/Conceptual/OpenGLES_ProgrammingGuide/WorkingwithOpenGLESContexts/WorkingwithOpenGLESContexts.html#//apple_ref/doc/uid/TP40008793-CH2-SW1
       --  Apple  NSOpenGLView  http://developer.apple.com/library/mac/#documentation/graphicsimaging/conceptual/OpenGL-MacProgGuide/opengl_pg_concepts/opengl_pg_concepts.html#//apple_ref/doc/uid/TP40001987-CH208-SW1
       --  Apple  CGLContextObj http://developer.apple.com/library/mac/#documentation/graphicsimaging/Reference/CGL_OpenGL/Reference/reference.html#//apple_ref/doc/uid/TP40001186
*/

#include <../src/sys/draw/drawimpl.h>  /*I  "petscsys.h" I*/
#if defined(PETSC_HAVE_OPENGLES)
#import <UIKit/UIKit.h>
#import <GLKit/GLKit.h>
#import <OpenGLES/EAGLDrawable.h>
#import <OpenGLES/ES2/gl.h>
#elif defined(PETSC_HAVE_GLUT)
#if defined(__APPLE__) || defined(MACOSX)
#include <AvailabilityMacros.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif
#endif


/*
     These macros transform from the users coordinates to the OpenGL coordinates of -1,-1 to 1,1.
*/
#define XTRANS(draw,x)  (-1.0 + 2.0*((draw)->port_xl + (((x - (draw)->coor_xl)*((draw)->port_xr - (draw)->port_xl))/((draw)->coor_xr - (draw)->coor_xl))))
#define YTRANS(draw,y)  (-1.0 + 2.0*((draw)->port_yl + (((y - (draw)->coor_yl)*((draw)->port_yr - (draw)->port_yl))/((draw)->coor_yr - (draw)->coor_yl))))

/*
     These macros transform from the users coordinates to pixel coordinates.
*/
#define XPTRANS(draw,win,x) (int)(((win)->w)*((draw)->port_xl + (((x - (draw)->coor_xl)*((draw)->port_xr - (draw)->port_xl))/((draw)->coor_xr - (draw)->coor_xl))))
#define YPTRANS(draw,win,y) (int)(((win)->h)*(1.0-(draw)->port_yl - (((y - (draw)->coor_yl)*((draw)->port_yr - (draw)->port_yl))/((draw)->coor_yr - (draw)->coor_yl))))

static unsigned char rcolor[256],gcolor[256],bcolor[256];
#undef __FUNCT__  
#define __FUNCT__ "InitializeColors" 
static PetscErrorCode InitializeColors(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
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
  PetscFunctionReturn(0);
} 

static GLuint        vertexshader,fragmentshader,shaderprogram;
#undef __FUNCT__  
#define __FUNCT__ "InitializeShader" 
static PetscErrorCode InitializeShader(void)
{
  const char     *vertexsource = "attribute vec2 position;\
                                  attribute vec3 color; \
                                  varying vec4 vColor;\
                                  void main(void)\
                                 {\
                                   vColor = vec4(color,0.50);\
                                   gl_Position = vec4(position,0.0,1.0);\
                                 }\n";
#if defined(PETSC_HAVE_GLUT)
  const char    *fragmentsource = "varying vec4 vColor;\
                                   void main (void)\
                                   {\
                                     gl_FragColor = vColor; \
                                   }\n";
#else
  const char    *fragmentsource = "precision mediump float;\
                                   varying vec4 vColor;\
                                   void main (void)\
                                   {\
                                     gl_FragColor = vColor; \
                                   }\n";
#endif
  int           isCompiled_VS, isCompiled_FS;
  int           isLinked;
  GLenum        err;

  PetscFunctionBegin;
  /*  http://www.opengl.org/wiki/OpenGL_Shading_Language */
  /* Create an empty vertex shader handle */
  vertexshader = glCreateShader(GL_VERTEX_SHADER);
 
  /* Send the vertex shader source code to GL */
  /* Note that the source code is NULL character terminated. */
  /* GL will automatically detect that therefore the length info can be 0 in this case (the last parameter) */
  glShaderSource(vertexshader, 1, (const GLchar**)&vertexsource, 0);
 
  /* Compile the vertex shader */
  glCompileShader(vertexshader);
  glGetShaderiv(vertexshader, GL_COMPILE_STATUS, &isCompiled_VS);
  if (isCompiled_VS == GL_FALSE) {
    PetscErrorCode ierr;
    int            maxLength;
    char           *vertexInfoLog;
    glGetShaderiv(vertexshader, GL_INFO_LOG_LENGTH, &maxLength);
    ierr = PetscMalloc(maxLength*sizeof(char),&vertexInfoLog);CHKERRQ(ierr);
    glGetShaderInfoLog(vertexshader, maxLength, &maxLength, vertexInfoLog);
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Failed to compile vertex shader %s",vertexInfoLog);
  }
 
  /* Create an empty fragment shader handle */
  fragmentshader = glCreateShader(GL_FRAGMENT_SHADER);
 
  /* Send the fragment shader source code to GL */
  /* Note that the source code is NULL character terminated. */
  /* GL will automatically detect that therefore the length info can be 0 in this case (the last parameter) */
  glShaderSource(fragmentshader, 1, (const GLchar**)&fragmentsource, 0);
 
  /* Compile the fragment shader */
  glCompileShader(fragmentshader);
  glGetShaderiv(fragmentshader, GL_COMPILE_STATUS, &isCompiled_FS);
  if (isCompiled_FS == GL_FALSE) {
    PetscErrorCode ierr;
    int            maxLength;
    char          *fragmentInfoLog;
    glGetShaderiv(fragmentshader, GL_INFO_LOG_LENGTH, &maxLength);
    ierr = PetscMalloc(maxLength*sizeof(char),&fragmentInfoLog);CHKERRQ(ierr);
    glGetShaderInfoLog(fragmentshader, maxLength, &maxLength, fragmentInfoLog);
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Failed to compile fragment shader %s",fragmentInfoLog);
  }
 
  /* If we reached this point it means the vertex and fragment shaders compiled and are syntax error free. */
  /* We must link them together to make a GL shader program */
  /* GL shader programs are monolithic. It is a single piece made of 1 vertex shader and 1 fragment shader. */
  /* Assign our program handle a "name" */
  shaderprogram = glCreateProgram();
 
  /* Attach our shaders to our program */
  glAttachShader(shaderprogram, vertexshader);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glAttachShader(shaderprogram, fragmentshader);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
 
  glBindAttribLocation(shaderprogram,0,"position");
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glBindAttribLocation(shaderprogram,1,"color");
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);

  /* Link our program */
  /* At this stage, the vertex and fragment programs are inspected, optimized and a binary code is generated for the shader. */
  /* The binary code is uploaded to the GPU, if there is no error. */
  glLinkProgram(shaderprogram);
 
  /* Again, we must check and make sure that it linked. If it fails, it would mean either there is a mismatch between the vertex */
  /* and fragment shaders. It might be that you have surpassed your GPU's abilities. Perhaps too many ALU operations or */
  /* too many texel fetch instructions or too many interpolators or dynamic loops. */
 
  glGetProgramiv(shaderprogram, GL_LINK_STATUS, &isLinked);
  if (isLinked == GL_FALSE) {
    /* 
    char          *shaderProgramInfoLog;
    glGetProgramiv(shaderprogram, GL_INFO_LOG_LENGTH, &maxLength);
    shaderProgramInfoLog = new char[maxLength];
    glGetProgramInfoLog(shaderprogram, maxLength, &maxLength, shaderProgramInfoLog);
    free(shaderProgramInfoLog);
    */
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Failed to compile fragment shader");
  }
 
  /* In your rendering code, you just need to call glUseProgram, call the various glUniform** to update your uniforms */
  /* and then render. */
  /* Load the shader into the rendering pipeline */
  glUseProgram(shaderprogram);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "FinalizeShader" 
/*
    This is currently all wrong, there is actually a separate shader for each window
   so they cannot be stored as global
*/
static PetscErrorCode FinalizeShader(void)
{
  PetscFunctionBegin;
  /* When the user shuts down your program, you should deallocate all your GL resources. */
  /* Unbind your shader. */
  glUseProgram(0);
  /* Let's detach */
  glDetachShader(shaderprogram, vertexshader);
  glDetachShader(shaderprogram, fragmentshader);
  /* Delete the shaders */
  glDeleteShader(vertexshader);
  glDeleteShader(fragmentshader);
  /* Delete the shader object */
  glDeleteProgram(shaderprogram);
  PetscFunctionReturn(0);
} 

extern PetscErrorCode PetscDrawClear_OpenGL_Base(PetscDraw);

#if defined(PETSC_HAVE_GLUT)
#include <GLUT/glut.h>
typedef struct {
  int  win;          /* OpenGL GLUT window identifier */
  int  x,y,w,h;      /* Size and location of window */
} PetscDraw_OpenGL;

static int currentwindow = -1;
PETSC_STATIC_INLINE PetscErrorCode OpenGLWindow(PetscDraw_OpenGL *win)
{
  if (win->win == currentwindow) return 0;
  currentwindow = win->win;
  glutSetWindow(win->win);
  return 0;
}

PETSC_STATIC_INLINE PetscErrorCode OpenGLString(float x,float y, const char *str,size_t len,int icolor)
{
  PetscInt         i;

  glColor3ub(rcolor[icolor],gcolor[icolor],bcolor[icolor]);
  glRasterPos2f(x, y);
  for (i = 0; i < len; i++) {
    glutBitmapCharacter(GLUT_BITMAP_8_BY_13, str[i]);
  }
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawClear_OpenGL" 
PetscErrorCode PetscDrawClear_OpenGL(PetscDraw draw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDrawClear_OpenGL_Base(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawGetPopup_OpenGL" 
static PetscErrorCode PetscDrawGetPopup_OpenGL(PetscDraw draw,PetscDraw *popup)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscDrawCreate(((PetscObject)draw)->comm,PETSC_NULL,PETSC_NULL,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,popup);CHKERRQ(ierr);
  ierr = PetscDrawSetType(*popup,((PetscObject)draw)->type_name);CHKERRQ(ierr);
  draw->popup = *popup;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawStringVertical_OpenGL" 
static PetscErrorCode PetscDrawStringVertical_OpenGL(PetscDraw draw,PetscReal x,PetscReal  y,int c,const char chrs[])
{
  PetscErrorCode   ierr;
  float            xx,yy;
  PetscDraw_OpenGL *win = (PetscDraw_OpenGL*)draw->data;
  PetscReal        tw,th;
  size_t           i,n;

  PetscFunctionBegin;
  ierr = OpenGLWindow(win);CHKERRQ(ierr);
  ierr = PetscStrlen(chrs,&n);CHKERRQ(ierr);
  ierr = PetscDrawStringGetSize(draw,&tw,&th);CHKERRQ(ierr);
  xx = XTRANS(draw,x);
  for (i=0; i<n; i++) {
    yy = YTRANS(draw,y-th*i);
    ierr = OpenGLString(xx,yy,chrs+i,1,c);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawString_OpenGL" 
static PetscErrorCode PetscDrawString_OpenGL(PetscDraw draw,PetscReal x,PetscReal  y,int c,const char chrs[])
{
  PetscErrorCode   ierr;
  float            xx,yy;
  size_t           len;
  PetscDraw_OpenGL *win = (PetscDraw_OpenGL*)draw->data;
  char             *substr;
  PetscToken       token;

  PetscFunctionBegin;
  xx = XTRANS(draw,x); 
  yy = YTRANS(draw,y);
  ierr = OpenGLWindow(win);CHKERRQ(ierr);
  
  ierr = PetscTokenCreate(chrs,'\n',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&substr);CHKERRQ(ierr);
  ierr = PetscStrlen(substr,&len);CHKERRQ(ierr);
  ierr = OpenGLString(xx,yy,substr,len,c);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&substr);CHKERRQ(ierr);
  while (substr) {
    yy  += 16;
    ierr = PetscStrlen(substr,&len);CHKERRQ(ierr);
    ierr = OpenGLString(xx,yy,substr,len,c);CHKERRQ(ierr);
    ierr = PetscTokenFind(token,&substr);CHKERRQ(ierr);
  }
  ierr = PetscTokenDestroy(&token);CHKERRQ(ierr);
  PetscFunctionReturn(0);
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
static PetscErrorCode PetscDrawDestroy_OpenGL(PetscDraw draw)
{
  PetscDraw_OpenGL *win = (PetscDraw_OpenGL*)draw->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscDrawSynchronizedClear(draw);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&draw->popup);CHKERRQ(ierr);
  glutDestroyWindow(win->win);
  ierr = PetscFree(win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawFlush_OpenGL" 
static PetscErrorCode PetscDrawFlush_OpenGL(PetscDraw draw)
{
  PetscDraw_OpenGL* win = (PetscDraw_OpenGL*)draw->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = OpenGLWindow(win);CHKERRQ(ierr);
  glutCheckLoop();
  glFinish();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawStringGetSize_OpenGL" 
static PetscErrorCode PetscDrawStringGetSize_OpenGL(PetscDraw draw,PetscReal *x,PetscReal  *y)
{
  PetscDraw_OpenGL* win = (PetscDraw_OpenGL*)draw->data;
  PetscInt          w;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = OpenGLWindow(win);CHKERRQ(ierr);
  w = glutBitmapWidth(GLUT_BITMAP_8_BY_13,'W');
  *x = w*(draw->coor_xr - draw->coor_xl)/((win->w)*(draw->port_xr - draw->port_xl));
  *y = (13./8.0)*w*(draw->coor_yr - draw->coor_yl)/((win->h)*(draw->port_yr - draw->port_yl));
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
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  ierr = PetscDrawClear(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  int button,x,y;
} OpenGLButton;
static OpenGLButton Mouse;

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
#elif defined(PETSC_HAVE_OPENGLES)
typedef struct {
  GLint   win;    /* not currently used */
  int     w,h;    /* width and height in pixels */
  GLKView *view;
} PetscDraw_OpenGL;

static GLKView  *globalGLKView[10] = {0,0,0,0,0,0,0,0,0,0};

PETSC_STATIC_INLINE PetscErrorCode OpenGLWindow(PetscDraw_OpenGL *win)
{
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawClear_OpenGL" 
static PetscErrorCode PetscDrawClear_OpenGL(PetscDraw draw)
{
  PetscErrorCode   ierr;
  PetscDraw_OpenGL *win = (PetscDraw_OpenGL*)draw->data;

  PetscFunctionBegin;
  /* remove all UIText added to window */
  for (UIView *view in [win->view subviews]) {[view removeFromSuperview];}
  ierr = PetscDrawClear_OpenGL_Base(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawGetPopup_OpenGL" 
static PetscErrorCode PetscDrawGetPopup_OpenGL(PetscDraw draw,PetscDraw *popup)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  *popup = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawString_OpenGL"
static PetscErrorCode PetscDrawString_OpenGL(PetscDraw draw,PetscReal x,PetscReal  y,int c,const char chrs[])
{
  PetscErrorCode   ierr;
  float            xx,yy;
  PetscDraw_OpenGL *win = (PetscDraw_OpenGL*)draw->data;
  UILabel          *yourLabel;

  PetscFunctionBegin;
  ierr = OpenGLWindow(win);CHKERRQ(ierr);
  xx = XPTRANS(draw,win,x);
  yy = YPTRANS(draw,win,y);
  NSLog(@"Draw string start");
  yourLabel = [[UILabel alloc] initWithFrame:CGRectMake(xx, yy, 300, 20)];
  [yourLabel setTextColor:[UIColor colorWithRed:rcolor[c]/255.0 green:gcolor[c]/255.0 blue:rcolor[c]/255.0 alpha:1.0]];
  [yourLabel setText: [[NSString alloc] initWithCString:chrs encoding:NSMacOSRomanStringEncoding]];
  [yourLabel setBackgroundColor:[UIColor clearColor]];
  /* [yourLabel setFont:[UIFont fontWithName: @"Trebuchet MS" size: 14.0f]]; */
  [win->view addSubview:yourLabel];
  NSLog(@"Draw string end");
  PetscFunctionReturn(0);
}

/*
   I don't understand the rotation. It seems to maybe rotate from the middle of the width?

   It would be nice if the frame could be made to match the text length automatically

   This makes src/sys/draw/examples/tests/ex3.c look good but may be terrible for other locations
*/
#undef __FUNCT__  
#define __FUNCT__ "PetscDrawStringVertical_OpenGL" 
static PetscErrorCode PetscDrawStringVertical_OpenGL(PetscDraw draw,PetscReal x,PetscReal  y,int c,const char chrs[])
{
  PetscErrorCode   ierr;
  float            xx,yy, w = 100,h = 20;
  PetscDraw_OpenGL *win = (PetscDraw_OpenGL*)draw->data;
  UILabel          *yourLabel;

  PetscFunctionBegin;
  ierr = OpenGLWindow(win);CHKERRQ(ierr);
  xx = XPTRANS(draw,win,x);
  yy = YPTRANS(draw,win,y);
  NSLog(@"Draw string vertical start");
  yourLabel = [[UILabel alloc] initWithFrame:CGRectMake(0,0, w, h)];
  [yourLabel setTextColor:[UIColor colorWithRed:rcolor[c]/255.0 green:gcolor[c]/255.0 blue:rcolor[c]/255.0 alpha:1.0]];
  [yourLabel setText: [[NSString alloc] initWithCString:chrs encoding:NSMacOSRomanStringEncoding]];
  [yourLabel setBackgroundColor:[UIColor clearColor]];
  [yourLabel setTransform:CGAffineTransformTranslate(CGAffineTransformMakeRotation(-M_PI / 2),-w/2.0-yy,-h+xx)];
  /* [yourLabel setFont:[UIFont fontWithName: @"Trebuchet MS" size: 14.0f]]; */
  [win->view addSubview:yourLabel];
  NSLog(@"Draw string vertical end");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSetTitle_OpenGL" 
static PetscErrorCode PetscDrawSetTitle_OpenGL(PetscDraw draw,const char title[])
{
  return 0;
}
#undef __FUNCT__  
#define __FUNCT__ "PetscDrawDestroy_OpenGL" 
static PetscErrorCode PetscDrawDestroy_OpenGL(PetscDraw draw)
{
  PetscDraw_OpenGL *win = (PetscDraw_OpenGL*)draw->data;
  PetscErrorCode   ierr;
  PetscInt         i;

  PetscFunctionBegin;
  for (i=0; i<10; i++) {
    if (!globalGLKView[i]) {
      globalGLKView[i] = win->view;
      ierr = PetscFree(win);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Could not locate available GLKView slot");
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "PetscDrawFlush_OpenGL" 
static PetscErrorCode PetscDrawFlush_OpenGL(PetscDraw draw)
{
  PetscDraw_OpenGL* win = (PetscDraw_OpenGL*)draw->data;

  GLenum err;
  glFlush();
  err = glGetError();
  if (err != GL_NO_ERROR) {
    NSLog(@"GL error detected glFlush()");
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unable to flush OpenGL Error Code %d",err);
  }
  [win->view display];
  NSLog(@"Completed display in PetscDrawFlush()");
  return 0;
}
#undef __FUNCT__  
#define __FUNCT__ "PetscDrawStringGetSize_OpenGL" 
static PetscErrorCode PetscDrawStringGetSize_OpenGL(PetscDraw draw,PetscReal *x,PetscReal  *y)
{
  float w = .02;
  *x = w*(draw->coor_xr - draw->coor_xl)/(draw->port_xr - draw->port_xl);
  *y = (13./8.0)*w*(draw->coor_yr - draw->coor_yl)/(draw->port_yr - draw->port_yl);
  return 0;
}
#undef __FUNCT__  
#define __FUNCT__ "PetscDrawResizeWindow_OpenGL" 
static PetscErrorCode PetscDrawResizeWindow_OpenGL(PetscDraw draw,int w,int h)
{
  return 0;
}
#undef __FUNCT__  
#define __FUNCT__ "PetscDrawCheckResizedWindow_OpenGL" 
static PetscErrorCode PetscDrawCheckResizedWindow_OpenGL(PetscDraw draw)
{
  return 0;
}
#undef __FUNCT__  
#define __FUNCT__ "PetscDrawGetMouseButton_OpenGL" 
static PetscErrorCode PetscDrawGetMouseButton_OpenGL(PetscDraw draw,PetscDrawButton *button,PetscReal* x_user,PetscReal *y_user,PetscReal *x_phys,PetscReal *y_phys)
{
  return 0;
}
#endif

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
#define __FUNCT__ "PetscDrawClear_OpenGL_Base" 
PetscErrorCode PetscDrawClear_OpenGL_Base(PetscDraw draw)
{
  PetscDraw_OpenGL* win = (PetscDraw_OpenGL*)draw->data;
  PetscErrorCode    ierr;
  float             xl,yl,xr,yr;
  GLfloat           vertices[12];
  GLfloat           colors[18]; 
  GLuint            positionBufferObject;
  GLuint            colorBufferObject;
  GLenum            err;

  PetscFunctionBegin;
  ierr = OpenGLWindow(win);CHKERRQ(ierr);
  xl = -1.0 + 2.0*((draw)->port_xl);
  xr = -1.0 + 2.0*((draw)->port_xr);
  yl = -1.0 + 2.0*((draw)->port_yl);
  yr = -1.0 + 2.0*((draw)->port_yr);

  vertices[0] = xl;
  vertices[1] = yl;
  vertices[2] = xr;
  vertices[3] = yl;
  vertices[4] = xl;
  vertices[5] = yr;

  vertices[6]  = xl;
  vertices[7]  = yr;
  vertices[8]  = xr;
  vertices[9]  = yr;
  vertices[10] = xr;
  vertices[11] = yl;

  colors[0] = 1.0;  colors[1] = 1.0; colors[2] = 1.0;
  colors[3] = 1.0;  colors[4] = 1.0; colors[5] = 1.0;
  colors[6] = 1.0;  colors[7] = 1.0; colors[8] = 1.0;

  colors[9] = 1.0;  colors[10] = 1.0; colors[11] = 1.0;
  colors[12] = 1.0;  colors[13] = 1.0; colors[14] = 1.0;
  colors[15] = 1.0;  colors[16] = 1.0; colors[17] = 1.0;

  glGenBuffers(1, &positionBufferObject);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glBindBuffer(GL_ARRAY_BUFFER, positionBufferObject);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glEnableVertexAttribArray(0);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);

  glGenBuffers(1, &colorBufferObject);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glBindBuffer(GL_ARRAY_BUFFER, colorBufferObject);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STATIC_DRAW);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glEnableVertexAttribArray(1);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);

  glDrawArrays(GL_TRIANGLES, 0, 6);
  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDeleteBuffers(1, &positionBufferObject);
  glDeleteBuffers(1, &colorBufferObject);
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
static PetscErrorCode PetscDrawPoint_OpenGL(PetscDraw draw,PetscReal xl,PetscReal yl,int cl)
{
  PetscDraw_OpenGL *win = (PetscDraw_OpenGL*)draw->data;
  GLfloat           vertices[2],colors[3];
  PetscErrorCode    ierr;
  GLuint            positionBufferObject;
  GLuint            colorBufferObject;
  GLenum            err;

  PetscFunctionBegin;
  vertices[0] = XTRANS(draw,xl);   
  vertices[1] = YTRANS(draw,yl);   
  colors[0] = rcolor[cl]/255.0;  colors[1] = gcolor[cl]/255.0; colors[2] = bcolor[cl]/255.0;

  ierr = OpenGLWindow(win);CHKERRQ(ierr);
  glGenBuffers(1, &positionBufferObject);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glBindBuffer(GL_ARRAY_BUFFER, positionBufferObject);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glEnableVertexAttribArray(0);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);

  glGenBuffers(1, &colorBufferObject);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glBindBuffer(GL_ARRAY_BUFFER, colorBufferObject);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STATIC_DRAW);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glEnableVertexAttribArray(1);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);

  glDrawArrays(GL_POINTS, 0, 2);
  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDeleteBuffers(1, &positionBufferObject);
  glDeleteBuffers(1, &colorBufferObject);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawLine_OpenGL" 
static PetscErrorCode PetscDrawLine_OpenGL(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int cl)
{
  PetscDraw_OpenGL *win = (PetscDraw_OpenGL*)draw->data;
  GLfloat           linevertices[4],colors[6];
  PetscErrorCode    ierr;
  GLenum            err;
  GLuint            positionBufferObject;
  GLuint            colorBufferObject;


  PetscFunctionBegin;
  linevertices[0]  = XTRANS(draw,xl);
  linevertices[2]  = XTRANS(draw,xr);
  linevertices[1]  = YTRANS(draw,yl);
  linevertices[3]  = YTRANS(draw,yr);
  if (linevertices[0] == linevertices[2] && linevertices[1] == linevertices[3]) PetscFunctionReturn(0);
  colors[0] = rcolor[cl]/255.0;  colors[1] = gcolor[cl]/255.0; colors[2] = bcolor[cl]/255.0;
  colors[3] = rcolor[cl]/255.0;  colors[4] = gcolor[cl]/255.0; colors[5] = bcolor[cl]/255.0;

  ierr = OpenGLWindow(win);CHKERRQ(ierr);
  /* http://arcsynthesis.org/gltut/Basics/Tutorial%2001.html */
  glGenBuffers(1, &positionBufferObject);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glBindBuffer(GL_ARRAY_BUFFER, positionBufferObject);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glBufferData(GL_ARRAY_BUFFER, sizeof(linevertices), linevertices, GL_STATIC_DRAW);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glEnableVertexAttribArray(0);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);

  glGenBuffers(1, &colorBufferObject);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glBindBuffer(GL_ARRAY_BUFFER, colorBufferObject);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STATIC_DRAW);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glEnableVertexAttribArray(1);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);

  glDrawArrays(GL_LINES, 0, 2);
  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDeleteBuffers(1, &positionBufferObject);
  glDeleteBuffers(1, &colorBufferObject);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawTriangle_OpenGL" 
static PetscErrorCode PetscDrawTriangle_OpenGL(PetscDraw draw,PetscReal X1,PetscReal Y_1,PetscReal X2,PetscReal Y2,PetscReal X3,PetscReal Y3,int c1,int c2,int c3)
{
  PetscDraw_OpenGL *win = (PetscDraw_OpenGL*)draw->data;
  PetscErrorCode    ierr;
  GLfloat           vertices[6],colors[9];
  GLenum            err;
  GLuint            positionBufferObject;
  GLuint            colorBufferObject;

  PetscFunctionBegin;
  vertices[0]  = XTRANS(draw,X1);
  vertices[1]  = YTRANS(draw,Y_1); 
  vertices[2]  = XTRANS(draw,X2);
  vertices[3]  = YTRANS(draw,Y2); 
  vertices[4]  = XTRANS(draw,X3);
  vertices[5]  = YTRANS(draw,Y3); 
  colors[0] = rcolor[c1]/255.0;  colors[1] = gcolor[c1]/255.0; colors[2] = bcolor[c1]/255.0;
  colors[3] = rcolor[c2]/255.0;  colors[4] = gcolor[c2]/255.0; colors[5] = bcolor[c2]/255.0;
  colors[6] = rcolor[c3]/255.0;  colors[7] = gcolor[c3]/255.0; colors[8] = bcolor[c3]/255.0;

  ierr = OpenGLWindow(win);CHKERRQ(ierr);
  glGenBuffers(1, &positionBufferObject);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glBindBuffer(GL_ARRAY_BUFFER, positionBufferObject);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glEnableVertexAttribArray(0);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);

  glGenBuffers(1, &colorBufferObject);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glBindBuffer(GL_ARRAY_BUFFER, colorBufferObject);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STATIC_DRAW);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glEnableVertexAttribArray(1);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);

  glDrawArrays(GL_TRIANGLES, 0, 3);
  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDeleteBuffers(1, &positionBufferObject);
  glDeleteBuffers(1, &colorBufferObject);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawRectangle_OpenGL" 
static PetscErrorCode PetscDrawRectangle_OpenGL(PetscDraw draw,PetscReal Xl,PetscReal Yl,PetscReal Xr,PetscReal Yr,int c1,int c2,int c3,int c4)
{
  PetscDraw_OpenGL *win = (PetscDraw_OpenGL*)draw->data;
  PetscErrorCode    ierr;
  GLfloat           vertices[12],colors[18];
  float             x1,y_1,x2,y2;
  GLuint            positionBufferObject;
  GLuint            colorBufferObject;
  GLenum            err;

  PetscFunctionBegin;
  x1   = XTRANS(draw,Xl);
  y_1  = YTRANS(draw,Yl);
  x2   = XTRANS(draw,Xr);
  y2   = YTRANS(draw,Yr);

  ierr = OpenGLWindow(win);CHKERRQ(ierr);
  vertices[0] = x1; colors[0] = rcolor[c1]/255.0;  colors[1] = gcolor[c1]/255.0; colors[2] = bcolor[c1]/255.0;
  vertices[1] = y_1;
  vertices[2] = x2;colors[3] = rcolor[c2]/255.0;  colors[4] = gcolor[c2]/255.0; colors[5] = bcolor[c2]/255.0;
  vertices[3] = y_1;
  vertices[4] = x1;colors[6] = rcolor[c4]/255.0;  colors[7] = gcolor[c4]/255.0; colors[8] = bcolor[c4]/255.0;
  vertices[5] = y2;

  vertices[6]  = x1;colors[9] = rcolor[c4]/255.0;  colors[10] = gcolor[c4]/255.0; colors[11] = bcolor[c4]/255.0;
  vertices[7]  = y2;
  vertices[8]  = x2;colors[12] = rcolor[c3]/255.0;  colors[13] = gcolor[c3]/255.0; colors[14] = bcolor[c3]/255.0;
  vertices[9]  = y2;
  vertices[10] = x2;colors[15] = rcolor[c2]/255.0;  colors[16] = gcolor[c2]/255.0; colors[17] = bcolor[c2]/255.0;
  vertices[11] = y_1;

  glGenBuffers(1, &positionBufferObject);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glBindBuffer(GL_ARRAY_BUFFER, positionBufferObject);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glEnableVertexAttribArray(0);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);

  glGenBuffers(1, &colorBufferObject);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glBindBuffer(GL_ARRAY_BUFFER, colorBufferObject);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STATIC_DRAW);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glEnableVertexAttribArray(1);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
  err = glGetError(); if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"OpenGL error %d\n",err);

  glDrawArrays(GL_TRIANGLES, 0, 6);
  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDeleteBuffers(1, &positionBufferObject);
  glDeleteBuffers(1, &colorBufferObject);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscDrawPause_OpenGL" 
static PetscErrorCode PetscDrawPause_OpenGL(PetscDraw draw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (draw->pause > 0) PetscSleep(draw->pause);
  else if (draw->pause == -1) {
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
                                 PetscDrawLine_OpenGL};

#if defined(PETSC_HAVE_GLUT)
/* callbacks required by GLUT */
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
#define __FUNCT__ "PetscDrawCreate_GLUT" 
PetscErrorCode  PetscDrawCreate_GLUT(PetscDraw draw)
{
  PetscDraw_OpenGL *win;
  PetscErrorCode   ierr;
  PetscInt         xywh[4],osize = 4;
  int              x = draw->x,y = draw->y,w = draw->w,h = draw->h;
  static int       xavailable = 0,yavailable = 0,xmax = 0,ymax = 0,ybottom = 0;
  static PetscBool initialized = PETSC_FALSE;
  int              argc;
  char             **argv;

  PetscFunctionBegin;
  if (!initialized) {
    ierr = PetscGetArgs(&argc,&argv);CHKERRQ(ierr);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA /* | GLUT_DOUBLE */| GLUT_DEPTH);
    ierr = InitializeColors();CHKERRQ(ierr);
  }

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
  ierr = PetscNew(PetscDraw_OpenGL,&win);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(draw,sizeof(PetscDraw_OpenGL));CHKERRQ(ierr);

  if (x < 0 || y < 0)   SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative corner of window");
  if (w <= 0 || h <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative window width or height");

  win->x      = x;
  win->y      = y;
  win->w      = w;
  win->h      = h;

  glutInitWindowSize(w, h);
  glutInitWindowPosition(x,y);
  win->win = glutCreateWindow(draw->title);
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  Mouse.button = -1;
  Mouse.x      = -1;
  Mouse.y      = -1;
  glutMouseFunc(mouse);

  glClearColor(1.0,1.0,1.0,1.0);

  draw->data = win;
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  ierr = InitializeShader();CHKERRQ(ierr);

  ierr = PetscDrawClear(draw);CHKERRQ(ierr);
  resized = PETSC_FALSE; /* opening the window triggers OpenGL call to reshape so need to cancel that resized flag */
  glutCheckLoop();
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawOpenGLUT" 
/*@C
   PetscDrawOpenGLUT - Opens an OpenGL window based on GLUT for use with the PetscDraw routines.

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
PetscErrorCode  PetscDrawOpenGLUT(MPI_Comm comm,const char display[],const char title[],int x,int y,int w,int h,PetscDraw* draw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDrawCreate(comm,display,title,x,y,w,h,draw);CHKERRQ(ierr);
  ierr = PetscDrawSetType(*draw,PETSC_DRAW_GLUT);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#elif defined(PETSC_HAVE_OPENGLES)

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawOpenGLESRegisterGLKView" 
PetscErrorCode  PetscDrawOpenGLESRegisterGLKView(GLKView *view)
{
  PetscInt i;

  PetscFunctionBegin;
  for (i=0; i<10; i++) {
    if (view == globalGLKView[i]) PetscFunctionReturn(0);  /* already registered */
    if (!globalGLKView[i]) {
      globalGLKView[i] = view;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Out of GLKView slots");
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscDrawCreate_OpenGLES" 
PetscErrorCode  PetscDrawCreate_OpenGLES(PetscDraw draw)
{
  PetscDraw_OpenGL *win;
  PetscErrorCode   ierr;
  static PetscBool initialized = PETSC_FALSE;
  PetscInt         i;

  PetscFunctionBegin;
  NSLog(@"Beginning PetscDrawCreate_OpenGLES()");

  ierr = PetscMemcpy(draw->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);
  ierr = PetscNew(PetscDraw_OpenGL,&win);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(draw,sizeof(PetscDraw_OpenGL));CHKERRQ(ierr);
  draw->data = win;
  for (i=0; i<10; i++) {
    if (globalGLKView[i]) {
      win->view = globalGLKView[i];
      win->w = win->view.frame.size.width;
      win->h = win->view.frame.size.height;
      [win->view bindDrawable];
      globalGLKView[i] = 0;
      break;
    }
  }
  if (!win->view) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Requested more OpenGL ES windows then provided with PetscDrawOpenGLRegisterGLKView()");

  if (!initialized) {
    initialized = PETSC_TRUE;
    ierr = InitializeColors();CHKERRQ(ierr);
  }
  ierr = InitializeShader();CHKERRQ(ierr);

  ierr = PetscDrawClear(draw);CHKERRQ(ierr); 
  NSLog(@"Ending PetscDrawCreate_OpenGLES()");
  PetscFunctionReturn(0);
}
EXTERN_C_END
#endif







