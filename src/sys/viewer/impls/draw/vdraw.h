/*
     Data structure for the PetscDraw version of the viewer
*/

#if !defined(__VDRAW_H)
#define __VDRAWL_H

#include "private/viewerimpl.h"
typedef struct {
  PetscInt       draw_max;
  PetscInt       draw_base;
  PetscDraw      *draw;
  PetscDrawLG    *drawlg;
  PetscDrawAxis  *drawaxis;
  int            w,h;        /* These are saved in case additional windows are opened */
  char           *display;
  char           *title;
  PetscTruth     singleton_made;
} PetscViewer_Draw;

#endif
