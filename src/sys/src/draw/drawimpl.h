/*
       Abstract data structure and functions for graphics.
*/

#if !defined(_DRAWIMPL_H)
#define _DRAWIMPL_H

#include "petsc.h"

struct _PetscDrawOps {
  int (*setdoublebuffer)(PetscDraw);
  int (*flush)(PetscDraw);
  int (*line)(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int);
  int (*linesetwidth)(PetscDraw,PetscReal);
  int (*linegetwidth)(PetscDraw,PetscReal*);
  int (*point)(PetscDraw,PetscReal,PetscReal,int);
  int (*pointsetsize)(PetscDraw,PetscReal);
  int (*string)(PetscDraw,PetscReal,PetscReal,int,const char[]);
  int (*stringvertical)(PetscDraw,PetscReal,PetscReal,int,const char[]);
  int (*stringsetsize)(PetscDraw,PetscReal,PetscReal);
  int (*stringgetsize)(PetscDraw,PetscReal*,PetscReal*);
  int (*setviewport)(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal);
  int (*clear)(PetscDraw);
  int (*synchronizedflush)(PetscDraw);
  int (*rectangle)(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int,int,int,int);
  int (*triangle)(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,int,int,int);
  int (*ellipse)(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int);
  int (*getmousebutton)(PetscDraw,PetscDrawButton*,PetscReal *,PetscReal *,PetscReal*,PetscReal*);
  int (*pause)(PetscDraw);
  int (*synchronizedclear)(PetscDraw);
  int (*beginpage)(PetscDraw);
  int (*endpage)(PetscDraw);
  int (*getpopup)(PetscDraw,PetscDraw*);
  int (*settitle)(PetscDraw,const char[]);
  int (*checkresizedwindow)(PetscDraw);
  int (*resizewindow)(PetscDraw,int,int);
  int (*destroy)(PetscDraw);
  int (*view)(PetscDraw,PetscViewer);
  int (*getsingleton)(PetscDraw,PetscDraw*);
  int (*restoresingleton)(PetscDraw,PetscDraw*);
  int (*setcoordinates)(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal);
};

struct _p_PetscDraw {
  PETSCHEADER(struct _PetscDrawOps)
  int             pause;       /* sleep time after a synchronized flush */
  PetscReal       port_xl,port_yl,port_xr,port_yr;
  PetscReal       coor_xl,coor_yl,coor_xr,coor_yr;
  char            *title;
  char            *display;
  PetscDraw       popup;
  int             x,y,h,w;
  void            *data;
};

#endif
