
/*
       Contains the data structure for plotting several line
    graphs in a window with an axis. This is intended for line
    graphs that change dynamically by adding more points onto
    the end of the X axis.
*/

#include <petscdraw.h>         /*I "petscdraw.h" I*/
#include <petsc/private/petscimpl.h>         /*I "petscsys.h" I*/

struct _p_PetscDrawLG {
  PETSCHEADER(int);
  PetscErrorCode (*destroy)(PetscDrawLG);
  PetscErrorCode (*view)(PetscDrawLG,PetscViewer);
  int            len,loc;
  PetscDraw      win;
  PetscDrawAxis  axis;
  PetscReal      xmin,xmax,ymin,ymax,*x,*y;
  int            nopts,dim,*colors;
  PetscBool      use_markers;
  char           **legend;
};
#define CHUNCKSIZE 100

