#ifndef lint
static char vcid[] = "$Id: draw.c,v 1.28 1996/01/27 20:21:28 curfman Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

/*@C
   DrawDestroy - Deletes a draw context.

   Input Parameters:
.  ctx - the drawing context

.keywords: draw, destroy
@*/
int DrawDestroy(Draw ctx)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->destroy) return (*ctx->destroy)((PetscObject)ctx);
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
  Draw ctx;
  *win = 0;
  PetscHeaderCreate(ctx,_Draw,DRAW_COOKIE,NULLWINDOW,comm);
  PLogObjectCreate(ctx);
  PetscMemzero(&ctx->ops,sizeof(struct _DrawOps));
  ctx->destroy = DrawDestroy_Null;
  ctx->view    = 0;
  ctx->pause   = 0;
  ctx->coor_xl = 0.0;  ctx->coor_xr = 1.0;
  ctx->coor_yl = 0.0;  ctx->coor_yr = 1.0;
  ctx->port_xl = 0.0;  ctx->port_xr = 1.0;
  ctx->port_yl = 0.0;  ctx->port_yr = 1.0;
  *win = ctx;
  return 0;
}
