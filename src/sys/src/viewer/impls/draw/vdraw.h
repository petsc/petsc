/*
     Data structure for the PetscDraw version of the viewer
*/

#if !defined(__VDRAW_H)
#define __VDRAWL_H

#include "src/sys/src/viewer/viewerimpl.h"
typedef struct {
  int            draw_max;
  PetscDraw      *draw;
  PetscDrawLG    *drawlg;
  PetscDrawAxis  *drawaxis;
  int            w,h;        /* These are saved in case additional windows are opened */
  char           *display;
  PetscTruth     singleton_made;
} PetscViewer_Draw;

#endif
