
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <petsc/private/drawimpl.h>  /*I "petscdraw.h" I*/
#include <petscviewer.h>

PetscClassId PETSC_DRAW_CLASSID;

static PetscBool PetscDrawPackageInitialized = PETSC_FALSE;
/*@C
  PetscDrawFinalizePackage - This function destroys everything in the Petsc interface to the Draw package. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode  PetscDrawFinalizePackage(void)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListDestroy(&PetscDrawList));
  PetscDrawPackageInitialized = PETSC_FALSE;
  PetscDrawRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscInitializeDrawPackage - This function initializes everything in the PetscDraw package. It is called
  from PetscDLLibraryRegister_petsc() when using dynamic libraries, and on the call to PetscInitialize()
  when using shared or static libraries.

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode  PetscDrawInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;

  PetscFunctionBegin;
  if (PetscDrawPackageInitialized) PetscFunctionReturn(0);
  PetscDrawPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  CHKERRQ(PetscClassIdRegister("Draw",&PETSC_DRAW_CLASSID));
  CHKERRQ(PetscClassIdRegister("Draw Axis",&PETSC_DRAWAXIS_CLASSID));
  CHKERRQ(PetscClassIdRegister("Line Graph",&PETSC_DRAWLG_CLASSID));
  CHKERRQ(PetscClassIdRegister("Histogram",&PETSC_DRAWHG_CLASSID));
  CHKERRQ(PetscClassIdRegister("Bar Graph",&PETSC_DRAWBAR_CLASSID));
  CHKERRQ(PetscClassIdRegister("Scatter Plot",&PETSC_DRAWSP_CLASSID));
  /* Register Constructors */
  CHKERRQ(PetscDrawRegisterAll());
  /* Process Info */
  {
    PetscClassId  classids[6];

    classids[0] = PETSC_DRAW_CLASSID;
    classids[1] = PETSC_DRAWAXIS_CLASSID;
    classids[2] = PETSC_DRAWLG_CLASSID;
    classids[3] = PETSC_DRAWHG_CLASSID;
    classids[4] = PETSC_DRAWBAR_CLASSID;
    classids[5] = PETSC_DRAWSP_CLASSID;
    CHKERRQ(PetscInfoProcessClass("draw", 6, classids));
  }
  /* Process summary exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    CHKERRQ(PetscStrInList("draw",logList,',',&pkg));
    if (pkg) {
      CHKERRQ(PetscLogEventExcludeClass(PETSC_DRAW_CLASSID));
      CHKERRQ(PetscLogEventExcludeClass(PETSC_DRAWAXIS_CLASSID));
      CHKERRQ(PetscLogEventExcludeClass(PETSC_DRAWLG_CLASSID));
      CHKERRQ(PetscLogEventExcludeClass(PETSC_DRAWHG_CLASSID));
      CHKERRQ(PetscLogEventExcludeClass(PETSC_DRAWBAR_CLASSID));
      CHKERRQ(PetscLogEventExcludeClass(PETSC_DRAWSP_CLASSID));
    }
  }
  /* Register package finalizer */
  CHKERRQ(PetscRegisterFinalize(PetscDrawFinalizePackage));
  PetscFunctionReturn(0);
}

/*@
   PetscDrawResizeWindow - Allows one to resize a window from a program.

   Collective on PetscDraw

   Input Parameters:
+  draw - the window
-  w,h - the new width and height of the window

   Level: intermediate

.seealso: PetscDrawCheckResizedWindow()
@*/
PetscErrorCode  PetscDrawResizeWindow(PetscDraw draw,int w,int h)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidLogicalCollectiveInt(draw,w,2);
  PetscValidLogicalCollectiveInt(draw,h,3);
  if (draw->ops->resizewindow) {
    CHKERRQ((*draw->ops->resizewindow)(draw,w,h));
  }
  PetscFunctionReturn(0);
}

/*@
   PetscDrawGetWindowSize - Gets the size of the window.

   Not collective

   Input Parameter:
.  draw - the window

   Output Parameters:
.  w,h - the window width and height

   Level: intermediate

.seealso: PetscDrawResizeWindow(), PetscDrawCheckResizedWindow()
@*/
PetscErrorCode  PetscDrawGetWindowSize(PetscDraw draw,int *w,int *h)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (w) PetscValidPointer(w,2);
  if (h) PetscValidPointer(h,3);
  if (w) *w = draw->w;
  if (h) *h = draw->h;
  PetscFunctionReturn(0);
}

/*@
   PetscDrawCheckResizedWindow - Checks if the user has resized the window.

   Collective on PetscDraw

   Input Parameter:
.  draw - the window

   Level: advanced

.seealso: PetscDrawResizeWindow()

@*/
PetscErrorCode  PetscDrawCheckResizedWindow(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->ops->checkresizedwindow) {
    CHKERRQ((*draw->ops->checkresizedwindow)(draw));
  }
  PetscFunctionReturn(0);
}

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
PetscErrorCode  PetscDrawGetTitle(PetscDraw draw,const char *title[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidPointer(title,2);
  *title = draw->title;
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawSetTitle - Sets the title of a PetscDraw context.

   Collective on PetscDraw

   Input Parameters:
+  draw - the graphics context
-  title - the title

   Level: intermediate

   Note: The title is positioned in the windowing system title bar for the window. Hence it will not be saved with -draw_save
   in the image.

   A copy of the string is made, so you may destroy the
   title string after calling this routine.

   You can use PetscDrawAxisSetLabels() to indicate a title within the window

.seealso: PetscDrawGetTitle(), PetscDrawAppendTitle()
@*/
PetscErrorCode  PetscDrawSetTitle(PetscDraw draw,const char title[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidCharPointer(title,2);
  CHKERRQ(PetscFree(draw->title));
  CHKERRQ(PetscStrallocpy(title,&draw->title));
  if (draw->ops->settitle) {
    CHKERRQ((*draw->ops->settitle)(draw,draw->title));
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawAppendTitle - Appends to the title of a PetscDraw context.

   Collective on PetscDraw

   Input Parameters:
+  draw - the graphics context
-  title - the title

   Note:
   A copy of the string is made, so you may destroy the
   title string after calling this routine.

   Level: advanced

.seealso: PetscDrawSetTitle(), PetscDrawGetTitle()
@*/
PetscErrorCode  PetscDrawAppendTitle(PetscDraw draw,const char title[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (title) PetscValidCharPointer(title,2);
  if (!title || !title[0]) PetscFunctionReturn(0);

  if (draw->title) {
    size_t len1,len2;
    char   *newtitle;
    CHKERRQ(PetscStrlen(title,&len1));
    CHKERRQ(PetscStrlen(draw->title,&len2));
    CHKERRQ(PetscMalloc1(len1 + len2 + 1,&newtitle));
    CHKERRQ(PetscStrcpy(newtitle,draw->title));
    CHKERRQ(PetscStrcat(newtitle,title));
    CHKERRQ(PetscFree(draw->title));
    draw->title = newtitle;
  } else {
    CHKERRQ(PetscStrallocpy(title,&draw->title));
  }
  if (draw->ops->settitle) {
    CHKERRQ((*draw->ops->settitle)(draw,draw->title));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawDestroy_Private(PetscDraw draw)
{
  PetscFunctionBegin;
  if (!draw->ops->save && !draw->ops->getimage) PetscFunctionReturn(0);
  CHKERRQ(PetscDrawSaveMovie(draw));
  if (draw->savefinalfilename) {
    draw->savesinglefile = PETSC_TRUE;
    CHKERRQ(PetscDrawSetSave(draw,draw->savefinalfilename));
    CHKERRQ(PetscDrawSave(draw));
  }
  CHKERRQ(PetscBarrier((PetscObject)draw));
  PetscFunctionReturn(0);
}

/*@
   PetscDrawDestroy - Deletes a draw context.

   Collective on PetscDraw

   Input Parameters:
.  draw - the drawing context

   Level: beginner

.seealso: PetscDrawCreate()

@*/
PetscErrorCode  PetscDrawDestroy(PetscDraw *draw)
{
  PetscFunctionBegin;
  if (!*draw) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*draw,PETSC_DRAW_CLASSID,1);
  if (--((PetscObject)(*draw))->refct > 0) PetscFunctionReturn(0);

  if ((*draw)->pause == -2) {
    (*draw)->pause = -1;
    CHKERRQ(PetscDrawPause(*draw));
  }

  /* if memory was published then destroy it */
  CHKERRQ(PetscObjectSAWsViewOff((PetscObject)*draw));

  CHKERRQ(PetscDrawDestroy_Private(*draw));

  if ((*draw)->ops->destroy) {
    CHKERRQ((*(*draw)->ops->destroy)(*draw));
  }
  CHKERRQ(PetscDrawDestroy(&(*draw)->popup));
  CHKERRQ(PetscFree((*draw)->title));
  CHKERRQ(PetscFree((*draw)->display));
  CHKERRQ(PetscFree((*draw)->savefilename));
  CHKERRQ(PetscFree((*draw)->saveimageext));
  CHKERRQ(PetscFree((*draw)->savemovieext));
  CHKERRQ(PetscFree((*draw)->savefinalfilename));
  CHKERRQ(PetscHeaderDestroy(draw));
  PetscFunctionReturn(0);
}

/*@
   PetscDrawGetPopup - Creates a popup window associated with a PetscDraw window.

   Collective on PetscDraw

   Input Parameter:
.  draw - the original window

   Output Parameter:
.  popup - the new popup window

   Level: advanced

.seealso: PetscDrawScalePopup(), PetscDrawCreate()

@*/
PetscErrorCode  PetscDrawGetPopup(PetscDraw draw,PetscDraw *popup)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidPointer(popup,2);

  if (draw->popup) *popup = draw->popup;
  else if (draw->ops->getpopup) {
    CHKERRQ((*draw->ops->getpopup)(draw,popup));
    if (*popup) {
      CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)*popup,"popup_"));
      (*popup)->pause = 0.0;
      CHKERRQ(PetscDrawSetFromOptions(*popup));
    }
  } else *popup = NULL;
  PetscFunctionReturn(0);
}

/*@C
  PetscDrawSetDisplay - Sets the display where a PetscDraw object will be displayed

  Input Parameters:
+ draw - the drawing context
- display - the X windows display

  Level: advanced

.seealso: PetscDrawCreate()

@*/
PetscErrorCode  PetscDrawSetDisplay(PetscDraw draw,const char display[])
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(draw->display));
  CHKERRQ(PetscStrallocpy(display,&draw->display));
  PetscFunctionReturn(0);
}

/*@
   PetscDrawSetDoubleBuffer - Sets a window to be double buffered.

   Logically Collective on PetscDraw

   Input Parameter:
.  draw - the drawing context

   Level: intermediate

@*/
PetscErrorCode  PetscDrawSetDoubleBuffer(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->ops->setdoublebuffer) {
    CHKERRQ((*draw->ops->setdoublebuffer)(draw));
  }
  PetscFunctionReturn(0);
}

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
PetscErrorCode  PetscDrawGetSingleton(PetscDraw draw,PetscDraw *sdraw)
{
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidPointer(sdraw,2);

  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)draw),&size));
  if (size == 1) {
    CHKERRQ(PetscObjectReference((PetscObject)draw));
    *sdraw = draw;
  } else {
    if (draw->ops->getsingleton) {
      CHKERRQ((*draw->ops->getsingleton)(draw,sdraw));
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot get singleton for this type %s of draw object",((PetscObject)draw)->type_name);
  }
  PetscFunctionReturn(0);
}

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
PetscErrorCode  PetscDrawRestoreSingleton(PetscDraw draw,PetscDraw *sdraw)
{
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidPointer(sdraw,2);
  PetscValidHeaderSpecific(*sdraw,PETSC_DRAW_CLASSID,2);

  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)draw),&size));
  if (size == 1) {
    if (draw == *sdraw) {
      CHKERRQ(PetscObjectDereference((PetscObject)draw));
      *sdraw = NULL;
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot restore singleton, it is not the parent draw");
  } else {
    if (draw->ops->restoresingleton) {
      CHKERRQ((*draw->ops->restoresingleton)(draw,sdraw));
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot restore singleton for this type %s of draw object",((PetscObject)draw)->type_name);
  }
  PetscFunctionReturn(0);
}
