#define PETSC_DLL
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "../src/sys/draw/drawimpl.h"  /*I "petscdraw.h" I*/

PetscCookie PETSC_DRAW_COOKIE;

static PetscTruth PetscDrawPackageInitialized = PETSC_FALSE;
#undef __FUNCT__  
#define __FUNCT__ "PetscDrawFinalizePackage"
/*@C
  PetscDrawFinalizePackage - This function destroys everything in the Petsc interface to the charactoristics package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package, mathematica
.seealso: PetscFinalize()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscDrawFinalizePackage(void) 
{
  PetscFunctionBegin;
  PetscDrawPackageInitialized = PETSC_FALSE;
  PetscDrawList               = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawInitializePackage" 
/*@C
  PetscInitializeDrawPackage - This function initializes everything in the PetscDraw package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the call to PetscInitialize()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: Petsc, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscDrawInitializePackage(const char path[])
{
  char              logList[256];
  char              *className;
  PetscTruth        opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (PetscDrawPackageInitialized) PetscFunctionReturn(0);
  PetscDrawPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscCookieRegister("Draw",&PETSC_DRAW_COOKIE);CHKERRQ(ierr);
  ierr = PetscCookieRegister("Axis",&DRAWAXIS_COOKIE);CHKERRQ(ierr);
  ierr = PetscCookieRegister("Line Graph",&DRAWLG_COOKIE);CHKERRQ(ierr);
  ierr = PetscCookieRegister("Histogram",&DRAWHG_COOKIE);CHKERRQ(ierr);
  ierr = PetscCookieRegister("Scatter Plot",&DRAWSP_COOKIE);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = PetscDrawRegisterAll(path);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "draw", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(0);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "draw", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(0);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(PetscDrawFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
PetscErrorCode PETSC_DLLEXPORT PetscDrawResizeWindow(PetscDraw draw,int w,int h)
{
  PetscErrorCode ierr;
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
PetscErrorCode PETSC_DLLEXPORT PetscDrawCheckResizedWindow(PetscDraw draw)
{
  PetscErrorCode ierr;
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
PetscErrorCode PETSC_DLLEXPORT PetscDrawGetTitle(PetscDraw draw,char **title)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);
  PetscValidPointer(title,2);
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
PetscErrorCode PETSC_DLLEXPORT PetscDrawSetTitle(PetscDraw draw,const char title[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);
  PetscValidCharPointer(title,2);
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
PetscErrorCode PETSC_DLLEXPORT PetscDrawAppendTitle(PetscDraw draw,const char title[])
{
  PetscErrorCode ierr;
  size_t len1,len2,len;
  char   *newtitle;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);
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
/*@
   PetscDrawDestroy - Deletes a draw context.

   Collective on PetscDraw

   Input Parameters:
.  draw - the drawing context

   Level: beginner

.seealso: PetscDrawCreate()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscDrawDestroy(PetscDraw draw)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);
  if (--((PetscObject)draw)->refct > 0) PetscFunctionReturn(0);

  /* if memory was published then destroy it */
  ierr = PetscObjectDepublish(draw);CHKERRQ(ierr);

  if (draw->ops->destroy) {
    ierr = (*draw->ops->destroy)(draw);CHKERRQ(ierr);
  }
  ierr = PetscStrfree(draw->title);CHKERRQ(ierr);
  ierr = PetscStrfree(draw->display);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawGetPopup" 
/*@
   PetscDrawGetPopup - Creates a popup window associated with a PetscDraw window.

   Collective on PetscDraw

   Input Parameter:
.  draw - the original window

   Output Parameter:
.  popup - the new popup window

   Level: advanced

@*/
PetscErrorCode PETSC_DLLEXPORT PetscDrawGetPopup(PetscDraw draw,PetscDraw *popup)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);
  PetscValidPointer(popup,2);

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
PetscErrorCode PetscDrawDestroy_Null(PetscDraw draw)
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
PetscErrorCode PETSC_DLLEXPORT PetscDrawOpenNull(MPI_Comm comm,PetscDraw *win)
{
  PetscErrorCode ierr;

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
PetscErrorCode PETSC_DLLEXPORT PetscDrawSetDisplay(PetscDraw draw,char *display)
{
  PetscErrorCode ierr;

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
PetscErrorCode PetscDrawCreate_Null(PetscDraw draw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemzero(draw->ops,sizeof(struct _PetscDrawOps));CHKERRQ(ierr);
  draw->ops->destroy = PetscDrawDestroy_Null;
  draw->ops->view    = 0;
  draw->pause   = 0.0;
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
PetscErrorCode PETSC_DLLEXPORT PetscDrawGetSingleton(PetscDraw draw,PetscDraw *sdraw)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);
  PetscValidPointer(sdraw,2);

  ierr = MPI_Comm_size(((PetscObject)draw)->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    *sdraw = draw;
  } else {
    if (draw->ops->getsingleton) {
      ierr = (*draw->ops->getsingleton)(draw,sdraw);CHKERRQ(ierr);
    } else {
      SETERRQ1(PETSC_ERR_SUP,"Cannot get singleton for this type %s of draw object",((PetscObject)draw)->type_name);
    }
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
PetscErrorCode PETSC_DLLEXPORT PetscDrawRestoreSingleton(PetscDraw draw,PetscDraw *sdraw)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);
  PetscValidPointer(sdraw,2);
  PetscValidHeaderSpecific(*sdraw,PETSC_DRAW_COOKIE,2);

  ierr = MPI_Comm_size(((PetscObject)draw)->comm,&size);CHKERRQ(ierr);
  if (size != 1) {
    if (draw->ops->restoresingleton) {
      ierr = (*draw->ops->restoresingleton)(draw,sdraw);CHKERRQ(ierr);
    } else {
      SETERRQ1(PETSC_ERR_SUP,"Cannot restore singleton for this type %s of draw object",((PetscObject)draw)->type_name);
    }
  }
  PetscFunctionReturn(0);
}







