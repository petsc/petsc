/* $Id: drawimpl.h,v 1.28 1999/01/12 23:16:27 bsmith Exp bsmith $ */
/*
       Abstract data structure and functions for graphics.
*/

#if !defined(_DRAWIMPL_H)
#define _DRAWIMPL_H

#include "petsc.h"

struct _DrawOps {
  int (*setdoublebuffer)(Draw);
  int (*flush)(Draw);
  int (*line)(Draw,double,double,double,double,int);
  int (*linesetwidth)(Draw,double);
  int (*linegetwidth)(Draw,double*);
  int (*point)(Draw,double,double,int);
  int (*pointsetsize)(Draw,double);
  int (*string)(Draw,double,double,int,char*);
  int (*stringvertical)(Draw,double,double,int,char*);
  int (*stringsetsize)(Draw,double,double);
  int (*stringgetsize)(Draw,double*,double*);
  int (*setviewport)(Draw,double,double,double,double);
  int (*clear)(Draw);
  int (*synchronizedflush)(Draw);
  int (*rectangle)(Draw,double,double,double,double,int,int,int,int);
  int (*triangle)(Draw,double,double,double,double,double,double,int,int,int);
  int (*getmousebutton)(Draw,DrawButton*,double *,double *,double*,double*);
  int (*pause)(Draw);
  int (*synchronizedclear)(Draw);
  int (*beginpage)(Draw);
  int (*endpage)(Draw);
  int (*getpopup)(Draw,Draw*);
  int (*settitle)(Draw,char *);
  int (*checkresizedwindow)(Draw);
  int (*resizewindow)(Draw,int,int);
  int (*destroy)(Draw);
  int (*view)(Draw,Viewer);
  int (*getsingleton)(Draw,Draw*);
  int (*restoresingleton)(Draw,Draw*);
};

struct _p_Draw {
  PETSCHEADER(struct _DrawOps)
  int             pause;       /* sleep time after a synchronized flush */
  double          port_xl,port_yl,port_xr,port_yr;
  double          coor_xl,coor_yl,coor_xr,coor_yr;
  char            *title;
  char            *display;
  Draw            popup;
  int             x,y,h,w;
  void            *data;
};

#endif
