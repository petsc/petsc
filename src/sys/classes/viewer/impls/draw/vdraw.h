/*
     Data structure for the PetscDraw version of the viewer
*/

#ifndef __VDRAW_H
#define __VDRAW_H

#include <petscdraw.h>
#include <petsc/private/viewerimpl.h>
typedef struct {
  PetscInt       draw_max;
  PetscInt       draw_base;
  PetscInt       nbounds; /* number of bounds supplied with PetscViewerDrawSetBounds() */
  PetscReal     *bounds;  /* lower and upper bounds for each component to be used in plotting */
  PetscDraw     *draw;
  PetscDrawLG   *drawlg;
  PetscDrawAxis *drawaxis;
  int            w, h; /* These are saved in case additional windows are opened */
  char          *display;
  char          *title;
  PetscBool      singleton_made;
  PetscBool      hold; /* Keep previous image when adding new */
  PetscReal      pause;
  PetscDrawType  drawtype;
} PetscViewer_Draw;

#endif
