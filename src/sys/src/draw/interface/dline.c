#ifndef lint
static char vcid[] = "$Id: dline.c,v 1.2 1996/02/08 18:27:49 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/
  
/*@
   DrawLine - Draws a line onto a drawable.

   Input Parameters:
.   draw - the drawing context
.   xl,yl,xr,yr - the coordinates of the line endpoints
.   cl - the colors of the endpoints

.keywords:  draw, line
@*/
int DrawLine(Draw draw,double xl,double yl,double xr,double yr,int cl)
{
  PETSCVALIDHEADERSPECIFIC(draw,DRAW_COOKIE);
  if (draw->type == NULLWINDOW) return 0;
  return (*draw->ops.line)(draw,xl,yl,xr,yr,cl);
}

/*@
    DrawIsNull - Returns PETSC_TRUE if there is a null draw object.

  Input Parameter:
    draw - the draw context

  Output Parameter:
    yes - PETSC_TRUE if it is a null draw object, else PETSC_FALSE

@*/
int DrawIsNull(Draw draw,PetscTruth *yes)
{
  if (draw->type == NULLWINDOW) *yes = PETSC_TRUE;
  else                          *yes = PETSC_FALSE;
  return 0;
}
