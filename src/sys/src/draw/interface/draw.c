#ifndef lint
static char vcid[] = "$Id: draw.c,v 1.38 1997/05/23 16:31:12 balay Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawCheckResizedWindow" /* ADIC Ignore */
/*@
   DrawCheckResizedWindow - Checks if the user has resized the window.

  Input Parameter:
.  draw - the window

@*/
int DrawCheckResizedWindow(Draw draw)
{
  if (draw->ops.checkresizedwindow) return (*draw->ops.checkresizedwindow)(draw);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "DrawGetTitle" /* ADIC Ignore */
/*@
   DrawGetTitle - Gets pointer to title of a Draw context.

   Input Parameter:
.    draw - the graphics context

   Output Parameter:
.    title - the title
@*/
int DrawGetTitle(Draw draw,char **title)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  *title = draw->title;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "DrawSetTitle" /* ADIC Ignore */
/*@
   DrawSetTitle - Sets the title of a Draw context.

   Input Parameters:
.    draw - the graphics context
.    title - the title

   Note: A copy of the string is made, so you may destroy the 
         title string after calling this routine.
@*/
int DrawSetTitle(Draw draw,char *title)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->title) PetscFree(draw->title);
  if (title) {
    int len = PetscStrlen(title);
    draw->title = (char *) PetscMalloc((len+1)*sizeof(char*));CHKPTRQ(draw->title);
    PLogObjectMemory(draw,(len+1)*sizeof(char*))
    PetscStrcpy(draw->title,title);
  } else {
    draw->title = 0;
  }
  if (draw->ops.settitle) return (*draw->ops.settitle)(draw,title);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "DrawAppendTitle" /* ADIC Ignore */
/*@
   DrawAppendTitle - Appends to the title of a Draw context.

   Input Parameters:
.    draw - the graphics context
.    title - the title

   Note: A copy of the string is made, so you may destroy the 
         title string after calling this routine.
@*/
int DrawAppendTitle(Draw draw,char *title)
{
  char *newtitle;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (!title) return 0;

  if (draw->title) {
    int len  = PetscStrlen(title) + PetscStrlen(draw->title);
    newtitle = (char *) PetscMalloc( (len + 1)*sizeof(char*) );CHKPTRQ(newtitle);
    PLogObjectMemory(draw,(len+1)*sizeof(char*));
    PetscStrcpy(newtitle,draw->title);
    PetscStrcat(newtitle,title);
    PetscFree(draw->title);
    draw->title = newtitle;
  } else {
    int len     = PetscStrlen(title);
    draw->title = (char *) PetscMalloc((len + 1)*sizeof(char*));CHKPTRQ(draw->title);
    PLogObjectMemory(draw,(len+1)*sizeof(char*));
    PetscStrcpy(draw->title,title);
  }
  if (draw->ops.settitle) return (*draw->ops.settitle)(draw,draw->title);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "DrawDestroy" /* ADIC Ignore */
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

#undef __FUNC__  
#define __FUNC__ "DrawCreatePopUp" /* ADIC Ignore */
/*@
   DrawCreatePopUp - Creates a popup window associated with 
      a Draw window.

  Input Parameter:
.  draw - the original window

  Output Parameter:
.  popup - the new popup window
@*/
int DrawCreatePopUp(Draw draw,Draw *popup)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  PetscValidPointer(popup);

  if (draw->popup) {*popup = draw->popup; return 0;}
  if (draw->ops.createpopup) return (*draw->ops.createpopup)(draw,popup);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "DrawDestroy_Null" /* ADIC Ignore */
int DrawDestroy_Null(PetscObject obj)
{
  PLogObjectDestroy(obj);
  PetscHeaderDestroy(obj); 
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "DrawOpenNull" /* ADIC Ignore */
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
  PetscHeaderCreate(draw,_p_Draw,DRAW_COOKIE,DRAW_NULLWINDOW,comm);
  PLogObjectCreate(draw);
  PetscMemzero(&draw->ops,sizeof(struct _DrawOps));
  draw->destroy = DrawDestroy_Null;
  draw->view    = 0;
  draw->pause   = 0;
  draw->coor_xl = 0.0;  draw->coor_xr = 1.0;
  draw->coor_yl = 0.0;  draw->coor_yr = 1.0;
  draw->port_xl = 0.0;  draw->port_xr = 1.0;
  draw->port_yl = 0.0;  draw->port_yr = 1.0;
  draw->popup   = 0;
  *win = draw;
  return 0;
}



