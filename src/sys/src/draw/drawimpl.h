/* $Id: drawimpl.h,v 1.30 1999/11/24 21:52:46 bsmith Exp bsmith $ */
/*
       Abstract data structure and functions for graphics.
*/

#if !defined(_DRAWIMPL_H)
#define _DRAWIMPL_H

#include "petsc.h"

struct _PetscDrawOps {
  int (*setdoublebuffer)(PetscDraw);
  int (*flush)(PetscDraw);
  int (*line)(PetscDraw,double,double,double,double,int);
  int (*linesetwidth)(PetscDraw,double);
  int (*linegetwidth)(PetscDraw,double*);
  int (*point)(PetscDraw,double,double,int);
  int (*pointsetsize)(PetscDraw,double);
  int (*string)(PetscDraw,double,double,int,char*);
  int (*stringvertical)(PetscDraw,double,double,int,char*);
  int (*stringsetsize)(PetscDraw,double,double);
  int (*stringgetsize)(PetscDraw,double*,double*);
  int (*setviewport)(PetscDraw,double,double,double,double);
  int (*clear)(PetscDraw);
  int (*synchronizedflush)(PetscDraw);
  int (*rectangle)(PetscDraw,double,double,double,double,int,int,int,int);
  int (*triangle)(PetscDraw,double,double,double,double,double,double,int,int,int);
  int (*getmousebutton)(PetscDraw,PetscDrawButton*,double *,double *,double*,double*);
  int (*pause)(PetscDraw);
  int (*synchronizedclear)(PetscDraw);
  int (*beginpage)(PetscDraw);
  int (*endpage)(PetscDraw);
  int (*getpopup)(PetscDraw,PetscDraw*);
  int (*settitle)(PetscDraw,char *);
  int (*checkresizedwindow)(PetscDraw);
  int (*resizewindow)(PetscDraw,int,int);
  int (*destroy)(PetscDraw);
  int (*view)(PetscDraw,PetscViewer);
  int (*getsingleton)(PetscDraw,PetscDraw*);
  int (*restoresingleton)(PetscDraw,PetscDraw*);
  int (*setcoordinates)(PetscDraw,double,double,double,double);
};

struct _p_PetscDraw {
  PETSCHEADER(struct _PetscDrawOps)
  int             pause;       /* sleep time after a synchronized flush */
  double          port_xl,port_yl,port_xr,port_yr;
  double          coor_xl,coor_yl,coor_xr,coor_yr;
  char            *title;
  char            *display;
  PetscDraw       popup;
  int             x,y,h,w;
  void            *data;
};

#endif
