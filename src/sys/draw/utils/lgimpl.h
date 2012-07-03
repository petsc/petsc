
/*
       Contains the data structure for plotting several line
    graphs in a window with an axis. This is intended for line 
    graphs that change dynamically by adding more points onto 
    the end of the X axis.
*/

#include <petscsys.h>         /*I "petscsys.h" I*/

struct _p_PetscDrawLG {
  PETSCHEADER(int);
  PetscErrorCode (*destroy)(PetscDrawLG);
  PetscErrorCode (*view)(PetscDrawLG,PetscViewer);
  int           len,loc;
  PetscDraw     win;
  PetscDrawAxis axis;
  PetscReal     xmin,xmax,ymin,ymax,*x,*y;
  int           nopts,dim,*colors;
  PetscBool     use_dots;
  char          **legend;
};
#define CHUNCKSIZE 100

