
/*
      Defines the internal data structures for the X-windows 
   implementation of the graphics functionality in PETSc.
*/

#include <../src/sys/draw/drawimpl.h>

#if !defined(_OPENGLIMPL_H)
#define _OPENGLIMPL_H

#include <GLUT/glut.h>

typedef struct {
  int  win;          /* OpenGL window identifier */
  int  x,y,w,h;      /* Size and location of window */
} PetscDraw_OpenGL;

#endif
