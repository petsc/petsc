/* $Id: vdraw.h,v 1.4 2001/01/15 21:43:15 bsmith Exp $ */
/*
     Data structure for the PetscDraw version of the viewer
*/

#if !defined(__VDRAW_H)
#define __VDRAWL_H

#define PETSC_VIEWER_DRAW_MAX 5

#include "src/sys/src/viewer/viewerimpl.h"
typedef struct {
  PetscDraw      draw[PETSC_VIEWER_DRAW_MAX];
  PetscDrawLG    drawlg[PETSC_VIEWER_DRAW_MAX];
  PetscDrawAxis  drawaxis[PETSC_VIEWER_DRAW_MAX];
  int            w,h;        /* These are saved in case additional windows are opened */
  char           *display;
  PetscTruth     singleton_made;
} PetscViewer_Draw;

#endif
