#ifndef lint
static char vcid[] = "$Id: draw.c,v 1.32 1996/07/08 22:21:15 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

/*@C
   DrawDestroy - Deletes a draw context.

   Input Parameters:
.  draw - the drawing context

.keywords: draw, destroy
@*/
int DrawDestroy(Draw draw)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->destroy) return (*draw->destroy)((PetscObject)draw);
  return 0;
}

int DrawDestroy_Null(PetscObject obj)
{
  PLogObjectDestroy(obj);
  PetscHeaderDestroy(obj); 
  return 0;
}

/*
  DrawOpenNull - Opens a null drawing context. All draw commands to 
  it are ignored.

  Output Parameter:
. win - the drawing context
*/
int DrawOpenNull(MPI_Comm comm,Draw *win)
{
  Draw draw;
  *win = 0;
  PetscHeaderCreate(draw,_Draw,DRAW_COOKIE,DRAW_NULLWINDOW,comm);
  PLogObjectCreate(draw);
  PetscMemzero(&draw->ops,sizeof(struct _DrawOps));
  draw->destroy = DrawDestroy_Null;
  draw->view    = 0;
  draw->pause   = 0;
  draw->coor_xl = 0.0;  draw->coor_xr = 1.0;
  draw->coor_yl = 0.0;  draw->coor_yr = 1.0;
  draw->port_xl = 0.0;  draw->port_xr = 1.0;
  draw->port_yl = 0.0;  draw->port_yr = 1.0;
  *win = draw;
  return 0;
}



