/*$Id: dgcoor.c,v 1.24 2000/05/05 22:13:25 balay Exp bsmith $*/
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawGetCoordinates"></a>*/"DrawGetCoordinates" 
/*@
   DrawGetCoordinates - Gets the application coordinates of the corners of
   the window (or page).

   Not Collective

   Input Parameter:
.  draw - the drawing object

   Level: advanced

   Ouput Parameters:
.  xl,yl,xr,yr - the coordinates of the lower left corner and upper
                 right corner of the drawing region.

.keywords:  draw, get, coordinates

.seealso: DrawSetCoordinates()

@*/
int DrawGetCoordinates(Draw draw,PetscReal *xl,PetscReal *yl,PetscReal *xr,PetscReal *yr)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  PetscValidDoublePointer(xl);
  PetscValidDoublePointer(yl);
  PetscValidDoublePointer(xr);
  PetscValidDoublePointer(yr);
  *xl = draw->coor_xl; *yl = draw->coor_yl;
  *xr = draw->coor_xr; *yr = draw->coor_yr;
  PetscFunctionReturn(0);
}
