/*$Id: draw.c,v 1.73 2001/03/23 23:20:08 balay Exp $*/
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

int PETSC_DRAW_COOKIE = 0;

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawResizeWindow" 
/*@
   PetscDrawResizeWindow - Allows one to resize a window from a program.

   Collective on PetscDraw

   Input Parameter:
+  draw - the window
-  w,h - the new width and height of the window

   Level: intermediate

.seealso: PetscDrawCheckResizedWindow()
@*/
int PetscDrawResizeWindow(PetscDraw draw,int w,int h)
{
  int ierr;
  PetscFunctionBegin;
  if (draw->ops->resizewindow) {
    ierr = (*draw->ops->resizewindow)(draw,w,h);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawCheckResizedWindow" 
/*@
   PetscDrawCheckResizedWindow - Checks if the user has resized the window.

   Collective on PetscDraw

   Input Parameter:
.  draw - the window

   Level: advanced

.seealso: PetscDrawResizeWindow()

@*/
int PetscDrawCheckResizedWindow(PetscDraw draw)
{
  int ierr;
  PetscFunctionBegin;
  if (draw->ops->checkresizedwindow) {
    ierr = (*draw->ops->checkresizedwindow)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawGetTitle" 
/*@C
   PetscDrawGetTitle - Gets pointer to title of a PetscDraw context.

   Not collective

   Input Parameter:
.  draw - the graphics context

   Output Parameter:
.  title - the title

   Level: intermediate

.seealso: PetscDrawSetTitle()
@*/
int PetscDrawGetTitle(PetscDraw draw,char **title)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE);
  *title = draw->title;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSetTitle" 
/*@C
   PetscDrawSetTitle - Sets the title of a PetscDraw context.

   Not collective (any processor or all may call this)

   Input Parameters:
+  draw - the graphics context
-  title - the title

   Level: intermediate

   Note:
   A copy of the string is made, so you may destroy the 
   title string after calling this routine.

.seealso: PetscDrawGetTitle(), PetscDrawAppendTitle()
@*/
int PetscDrawSetTitle(PetscDraw draw,const char title[])
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE);
  ierr = PetscStrfree(draw->title);CHKERRQ(ierr);
  ierr = PetscStrallocpy(title,&draw->title);CHKERRQ(ierr);
  if (draw->ops->settitle) {
    ierr = (*draw->ops->settitle)(draw,title);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawAppendTitle" 
/*@C
   PetscDrawAppendTitle - Appends to the title of a PetscDraw context.

   Not collective (any processor or all can call this)

   Input Parameters:
+  draw - the graphics context
-  title - the title

   Note:
   A copy of the string is made, so you may destroy the 
   title string after calling this routine.

   Level: advanced

.seealso: PetscDrawSetTitle(), PetscDrawGetTitle()
@*/
int PetscDrawAppendTitle(PetscDraw draw,const char title[])
{
  int  ierr,len1,len2,len;
  char *newtitle;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE);
  if (!title) PetscFunctionReturn(0);

  if (draw->title) {
    ierr = PetscStrlen(title,&len1);CHKERRQ(ierr);
    ierr = PetscStrlen(draw->title,&len2);CHKERRQ(ierr);
    len  = len1 + len2;
    ierr = PetscMalloc((len + 1)*sizeof(char*),&newtitle);CHKERRQ(ierr);
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

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawDestroy" 
/*@C
   PetscDrawDestroy - Deletes a draw context.

   Collective on PetscDraw

   Input Parameters:
.  draw - the drawing context

   Level: beginner

.seealso: PetscDrawCreate()

@*/
int PetscDrawDestroy(PetscDraw draw)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE);
  if (--draw->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(draw);CHKERRQ(ierr);

  if (draw->ops->destroy) {
    ierr = (*draw->ops->destroy)(draw);CHKERRQ(ierr);
  }
  ierr = PetscStrfree(draw->title);CHKERRQ(ierr);
  ierr = PetscStrfree(draw->display);CHKERRQ(ierr);
  PetscLogObjectDestroy(draw);
  PetscHeaderDestroy(draw);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawGetPopup" 
/*@C
   PetscDrawGetPopup - Creates a popup window associated with a PetscDraw window.

   Collective on PetscDraw

   Input Parameter:
.  draw - the original window

   Output Parameter:
.  popup - the new popup window

   Level: advanced

@*/
int PetscDrawGetPopup(PetscDraw draw,PetscDraw *popup)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE);
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

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawDestroy_Null" 
int PetscDrawDestroy_Null(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawOpenNull" 
/*
  PetscDrawOpenNull - Opens a null drawing context. All draw commands to 
  it are ignored.

  Output Parameter:
. win - the drawing context

   Level: advanced

*/
int PetscDrawOpenNull(MPI_Comm comm,PetscDraw *win)
{
  int ierr;

  PetscFunctionBegin;
  ierr = PetscDrawCreate(comm,PETSC_NULL,PETSC_NULL,0,0,1,1,win);CHKERRQ(ierr);
  ierr = PetscDrawSetType(*win,PETSC_DRAW_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSetDisplay"
/*@
  PetscDrawSetDisplay - Sets the display where a PetscDraw object will be displayed

  Input Parameter:
+ draw - the drawing context
- display - the X windows display

  Level: advanced

@*/
int PetscDrawSetDisplay(PetscDraw draw,char *display)
{
  int ierr;

  PetscFunctionBegin;
  ierr          = PetscStrfree(draw->display);CHKERRQ(ierr); 
  ierr          = PetscStrallocpy(display,&draw->display);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscDrawCreate_Null" 
/*
  PetscDrawCreate_Null - Opens a null drawing context. All draw commands to 
  it are ignored.

  Input Parameter:
. win - the drawing context
*/
int PetscDrawCreate_Null(PetscDraw draw)
{
  int ierr;

  PetscFunctionBegin;
  ierr = PetscMemzero(draw->ops,sizeof(struct _PetscDrawOps));CHKERRQ(ierr);
  draw->ops->destroy = PetscDrawDestroy_Null;
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

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawGetSingleton" 
/*@C
   PetscDrawGetSingleton - Gain access to a PetscDraw object as if it were owned 
        by the one process.

   Collective on PetscDraw

   Input Parameter:
.  draw - the original window

   Output Parameter:
.  sdraw - the singleton window

   Level: advanced

.seealso: PetscDrawRestoreSingleton(), PetscViewerGetSingleton(), PetscViewerRestoreSingleton()

@*/
int PetscDrawGetSingleton(PetscDraw draw,PetscDraw *sdraw)
{
  int ierr,size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE);
  PetscValidPointer(sdraw);

  ierr = MPI_Comm_size(draw->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    *sdraw = draw;
    PetscFunctionReturn(0);
  }

  if (draw->ops->getsingleton) {
    ierr = (*draw->ops->getsingleton)(draw,sdraw);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"Cannot get singleton for this type %s of draw object",draw->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawRestoreSingleton" 
/*@C
   PetscDrawRestoreSingleton - Remove access to a PetscDraw object as if it were owned 
        by the one process.

   Collective on PetscDraw

   Input Parameters:
+  draw - the original window
-  sdraw - the singleton window

   Level: advanced

.seealso: PetscDrawGetSingleton(), PetscViewerGetSingleton(), PetscViewerRestoreSingleton()

@*/
int PetscDrawRestoreSingleton(PetscDraw draw,PetscDraw *sdraw)
{
  int ierr,size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE);
  PetscValidPointer(sdraw);
  PetscValidHeaderSpecific(*sdraw,PETSC_DRAW_COOKIE);

  ierr = MPI_Comm_size(draw->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
     PetscFunctionReturn(0);
  }

  if (draw->ops->restoresingleton) {
    ierr = (*draw->ops->restoresingleton)(draw,sdraw);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"Cannot restore singleton for this type %s of draw object",draw->type_name);
  }
  PetscFunctionReturn(0);
}







