/*$Id: draw.c,v 1.65 2000/01/11 20:59:07 bsmith Exp bsmith $*/
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawResizeWindow" 
/*@
   DrawResizeWindow - Allows one to resize a window from a program.

   Collective on Draw

   Input Parameter:
+  draw - the window
-  w,h - the new width and height of the window

   Level: intermediate

.seealso: DrawCheckResizedWindow()
@*/
int DrawResizeWindow(Draw draw,int w,int h)
{
  int ierr;
  PetscFunctionBegin;
  if (draw->ops->resizewindow) {
    ierr = (*draw->ops->resizewindow)(draw,w,h);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawCheckResizedWindow" 
/*@
   DrawCheckResizedWindow - Checks if the user has resized the window.

   Collective on Draw

   Input Parameter:
.  draw - the window

   Level: advanced

.seealso: DrawResizeWindow()

@*/
int DrawCheckResizedWindow(Draw draw)
{
  int ierr;
  PetscFunctionBegin;
  if (draw->ops->checkresizedwindow) {
    ierr = (*draw->ops->checkresizedwindow)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawGetTitle" 
/*@C
   DrawGetTitle - Gets pointer to title of a Draw context.

   Not collective

   Input Parameter:
.  draw - the graphics context

   Output Parameter:
.  title - the title

   Level: intermediate

.seealso: DrawSetTitle()
@*/
int DrawGetTitle(Draw draw,char **title)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  *title = draw->title;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawSetTitle" 
/*@C
   DrawSetTitle - Sets the title of a Draw context.

   Not collective (any processor or all may call this)

   Input Parameters:
+  draw - the graphics context
-  title - the title

   Level: intermediate

   Note:
   A copy of the string is made, so you may destroy the 
   title string after calling this routine.

.seealso: DrawGetTitle(), DrawAppendTitle()
@*/
int DrawSetTitle(Draw draw,char *title)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  ierr = PetscStrfree(draw->title);CHKERRQ(ierr);
  ierr = PetscStrallocpy(title,&draw->title);CHKERRQ(ierr);
  if (draw->ops->settitle) {
    ierr = (*draw->ops->settitle)(draw,title);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawAppendTitle" 
/*@C
   DrawAppendTitle - Appends to the title of a Draw context.

   Not collective (any processor or all can call this)

   Input Parameters:
+  draw - the graphics context
-  title - the title

   Note:
   A copy of the string is made, so you may destroy the 
   title string after calling this routine.

   Level: advanced

.seealso: DrawSetTitle(), DrawGetTitle()
@*/
int DrawAppendTitle(Draw draw,char *title)
{
  int  ierr,len1,len2,len;
  char *newtitle;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (!title) PetscFunctionReturn(0);

  if (draw->title) {
    ierr = PetscStrlen(title,&len1);CHKERRQ(ierr);
    ierr = PetscStrlen(draw->title,&len2);CHKERRQ(ierr);
    len  = len1 + len2;
    newtitle = (char*)PetscMalloc((len + 1)*sizeof(char*));CHKPTRQ(newtitle);
    ierr = PetscStrcpy(newtitle,draw->title);CHKERRQ(ierr);
    ierr = PetscStrcat(newtitle,title);CHKERRQ(ierr);
    ierr = PetscFree(draw->title);CHKERRQ(ierr);
    draw->title = newtitle;
  } else {
    ierr = PetscStrallocpy(title,&draw->title);CHKERRQ(ierr);
  }
  if (draw->ops->settitle) {
    ierr = (*draw->ops->settitle)(draw,draw->title);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawDestroy" 
/*@C
   DrawDestroy - Deletes a draw context.

   Collective on Draw

   Input Parameters:
.  draw - the drawing context

   Level: beginner

.keywords: draw, destroy

.seealso: DrawCreate()

@*/
int DrawDestroy(Draw draw)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (--draw->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(draw);CHKERRQ(ierr);

  if (draw->ops->destroy) {
    ierr = (*draw->ops->destroy)(draw);CHKERRQ(ierr);
  }
  ierr = PetscStrfree(draw->title);CHKERRQ(ierr);
  ierr = PetscStrfree(draw->display);CHKERRQ(ierr);
  PLogObjectDestroy(draw);
  PetscHeaderDestroy(draw);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawGetPopup" 
/*@C
   DrawGetPopup - Creates a popup window associated with a Draw window.

   Collective on Draw

   Input Parameter:
.  draw - the original window

   Output Parameter:
.  popup - the new popup window

   Level: advanced

@*/
int DrawGetPopup(Draw draw,Draw *popup)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  PetscValidPointer(popup);

  if (draw->popup) {
    *popup = draw->popup; 
  } else if (draw->ops->getpopup) {
      ierr = (*draw->ops->getpopup)(draw,popup);CHKERRQ(ierr);
  } else {
    *popup = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawDestroy_Null" 
int DrawDestroy_Null(Draw draw)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawOpenNull" 
/*
  DrawOpenNull - Opens a null drawing context. All draw commands to 
  it are ignored.

  Output Parameter:
. win - the drawing context

   Level: advanced

*/
int DrawOpenNull(MPI_Comm comm,Draw *win)
{
  int ierr;

  PetscFunctionBegin;
  ierr = DrawCreate(comm,PETSC_NULL,PETSC_NULL,0,0,1,1,win);CHKERRQ(ierr);
  ierr = DrawSetType(*win,DRAW_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawCreate_Null" 
/*
  DrawCreate_Null - Opens a null drawing context. All draw commands to 
  it are ignored.

  Input Parameter:
. win - the drawing context
*/
int DrawCreate_Null(Draw draw)
{
  int ierr;

  PetscFunctionBegin;
  ierr = PetscMemzero(draw->ops,sizeof(struct _DrawOps));CHKERRQ(ierr);
  draw->ops->destroy = DrawDestroy_Null;
  draw->ops->view    = 0;
  draw->pause   = 0;
  draw->coor_xl = 0.0;  draw->coor_xr = 1.0;
  draw->coor_yl = 0.0;  draw->coor_yr = 1.0;
  draw->port_xl = 0.0;  draw->port_xr = 1.0;
  draw->port_yl = 0.0;  draw->port_yr = 1.0;
  draw->popup   = 0;

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawGetSingleton" 
/*@C
   DrawGetSingleton - Gain access to a Draw object as if it were owned 
        by the one process.

   Collective on Draw

   Input Parameter:
.  draw - the original window

   Output Parameter:
.  sdraw - the singleton window

   Level: advanced

.seealso: DrawRestoreSingleton(), ViewerGetSingleton(), ViewerRestoreSingleton()

@*/
int DrawGetSingleton(Draw draw,Draw *sdraw)
{
  int ierr,size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  PetscValidPointer(sdraw);

  ierr = MPI_Comm_size(draw->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    *sdraw = draw;
    PetscFunctionReturn(0);
  }

  if (draw->ops->getsingleton) {
    ierr = (*draw->ops->getsingleton)(draw,sdraw);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Cannot get singleton for this type %s of draw object",draw->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawRestoreSingleton" 
/*@C
   DrawRestoreSingleton - Remove access to a Draw object as if it were owned 
        by the one process.

   Collective on Draw

   Input Parameters:
+  draw - the original window
-  sdraw - the singleton window

   Level: advanced

.seealso: DrawGetSingleton(), ViewerGetSingleton(), ViewerRestoreSingleton()

@*/
int DrawRestoreSingleton(Draw draw,Draw *sdraw)
{
  int ierr,size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  PetscValidPointer(sdraw);
  PetscValidHeaderSpecific(*sdraw,DRAW_COOKIE);

  ierr = MPI_Comm_size(draw->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
     PetscFunctionReturn(0);
  }

  if (draw->ops->restoresingleton) {
    ierr = (*draw->ops->restoresingleton)(draw,sdraw);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Cannot restore singleton for this type %s of draw object",draw->type_name);
  }
  PetscFunctionReturn(0);
}







