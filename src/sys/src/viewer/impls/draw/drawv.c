/*$Id: drawv.c,v 1.48 2000/04/12 04:20:56 bsmith Exp bsmith $*/

#include "petsc.h"
#include "src/sys/src/viewer/impls/draw/vdraw.h" /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"ViewerDestroy_Draw" 
int ViewerDestroy_Draw(Viewer v)
{
  int         ierr,i;
  Viewer_Draw *vdraw = (Viewer_Draw*)v->data;

  PetscFunctionBegin;
  if (vdraw->singleton_made) {
    SETERRQ(1,1,"Destroying viewer without first restoring singleton");
  }
  for (i=0; i<VIEWER_DRAW_MAX; i++) {
    if (vdraw->drawaxis[i]) {ierr = DrawAxisDestroy(vdraw->drawaxis[i]);CHKERRQ(ierr);}
    if (vdraw->drawlg[i])   {ierr = DrawLGDestroy(vdraw->drawlg[i]);CHKERRQ(ierr);}
    if (vdraw->draw[i])     {ierr = DrawDestroy(vdraw->draw[i]);CHKERRQ(ierr);}
  }
  ierr = PetscFree(vdraw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"ViewerFlush_Draw" 
int ViewerFlush_Draw(Viewer v)
{
  int         ierr,i;
  Viewer_Draw *vdraw = (Viewer_Draw*)v->data;

  PetscFunctionBegin;
  for (i=0; i<VIEWER_DRAW_MAX; i++) {
    if (vdraw->draw[i]) {ierr = DrawSynchronizedFlush(vdraw->draw[i]);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"ViewerDrawGetDraw" 
/*@C
    ViewerDrawGetDraw - Returns Draw object from Viewer object.
    This Draw object may then be used to perform graphics using 
    DrawXXX() commands.

    Not collective (but Draw returned will be parallel object if Viewer is)

    Input Parameter:
+   viewer - the viewer (created with ViewerDrawOpen()
-   windownumber - indicates which subwindow (usually 0)

    Ouput Parameter:
.   draw - the draw object

    Level: intermediate

.keywords: viewer, draw, get

.seealso: ViewerDrawGetLG(), ViewerDrawGetAxis(), ViewerDrawOpen()
@*/
int ViewerDrawGetDraw(Viewer viewer,int windownumber,Draw *draw)
{
  Viewer_Draw *vdraw;
  int         ierr;
  PetscTruth  isdraw;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  PetscValidPointer(draw);
  ierr = PetscTypeCompare((PetscObject)viewer,DRAW_VIEWER,&isdraw);CHKERRQ(ierr);
  if (!isdraw) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"Must be draw type viewer");
  }
  if (windownumber < 0 || windownumber >= VIEWER_DRAW_MAX) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Window number out of range");
  }

  vdraw = (Viewer_Draw*)viewer->data;
  if (!vdraw->draw[windownumber]) {
    ierr = DrawCreate(viewer->comm,vdraw->display,0,PETSC_DECIDE,PETSC_DECIDE,vdraw->w,vdraw->h,
                     &vdraw->draw[windownumber]);CHKERRQ(ierr);
    ierr = DrawSetFromOptions(vdraw->draw[windownumber]);CHKERRQ(ierr);
  }
  *draw = vdraw->draw[windownumber];
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"ViewerDrawGetDrawLG" 
/*@C
    ViewerDrawGetDrawLG - Returns DrawLG object from Viewer object.
    This DrawLG object may then be used to perform graphics using 
    DrawLGXXX() commands.

    Not Collective (but DrawLG object will be parallel if Viewer is)

    Input Parameter:
+   viewer - the viewer (created with ViewerDrawOpen())
-   windownumber - indicates which subwindow (usually 0)

    Ouput Parameter:
.   draw - the draw line graph object

    Level: intermediate

.keywords: viewer, draw, get, line graph

.seealso: ViewerDrawGetDraw(), ViewerDrawGetAxis(), ViewerDrawOpen()
@*/
int ViewerDrawGetDrawLG(Viewer viewer,int windownumber,DrawLG *drawlg)
{
  int         ierr;
  PetscTruth  isdraw;
  Viewer_Draw *vdraw;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  PetscValidPointer(drawlg);
  ierr = PetscTypeCompare((PetscObject)viewer,DRAW_VIEWER,&isdraw);CHKERRQ(ierr);
  if (!isdraw) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"Must be draw type viewer");
  }
  if (windownumber < 0 || windownumber >= VIEWER_DRAW_MAX) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Window number out of range");
  }
  vdraw = (Viewer_Draw*)viewer->data;
  if (!vdraw->draw[windownumber]) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"No window with that number");
  }

  if (!vdraw->drawlg[windownumber]) {
    ierr = DrawLGCreate(vdraw->draw[windownumber],1,&vdraw->drawlg[windownumber]);CHKERRQ(ierr);
    PLogObjectParent(viewer,vdraw->drawlg[windownumber]);
  }
  *drawlg = vdraw->drawlg[windownumber];
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"ViewerDrawGetDrawAxis" 
/*@C
    ViewerDrawGetDrawAxis - Returns DrawAxis object from Viewer object.
    This DrawAxis object may then be used to perform graphics using 
    DrawAxisXXX() commands.

    Not Collective (but DrawAxis object will be parallel if Viewer is)

    Input Parameter:
+   viewer - the viewer (created with ViewerDrawOpen()
-   windownumber - indicates which subwindow (usually 0)

    Ouput Parameter:
.   drawaxis - the draw axis object

    Level: advanced

.keywords: viewer, draw, get, line graph

.seealso: ViewerDrawGetDraw(), ViewerDrawGetLG(), ViewerDrawOpen()
@*/
int ViewerDrawGetDrawAxis(Viewer viewer,int windownumber,DrawAxis *drawaxis)
{
  int         ierr;
  PetscTruth  isdraw;
  Viewer_Draw *vdraw;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  PetscValidPointer(drawaxis);
  ierr = PetscTypeCompare((PetscObject)viewer,DRAW_VIEWER,&isdraw);CHKERRQ(ierr);
  if (!isdraw) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"Must be draw type viewer");
  }
  if (windownumber < 0 || windownumber >= VIEWER_DRAW_MAX) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Window number out of range");
  }
  vdraw = (Viewer_Draw*)viewer->data;
  if (!vdraw->draw[windownumber]) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"No window with that number");
  }

  if (!vdraw->drawaxis[windownumber]) {
    ierr = DrawAxisCreate(vdraw->draw[windownumber],&vdraw->drawaxis[windownumber]);CHKERRQ(ierr);
    PLogObjectParent(viewer,vdraw->drawaxis[windownumber]);
  }
  *drawaxis = vdraw->drawaxis[windownumber];
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"ViewerDrawSetInfo" 
int ViewerDrawSetInfo(Viewer v,const char display[],const char title[],int x,int y,int w,int h)
{
  int         ierr;
  Viewer_Draw *vdraw = (Viewer_Draw*)v->data;

  PetscFunctionBegin;
  vdraw->h  = h;
  vdraw->w  = w;
  ierr      = PetscStrallocpy(display,&vdraw->display);CHKERRQ(ierr);
  ierr      = DrawCreate(v->comm,display,title,x,y,w,h,&vdraw->draw[0]);CHKERRQ(ierr);
  ierr      = DrawSetFromOptions(vdraw->draw[0]);CHKERRQ(ierr);
  PLogObjectParent(v,vdraw->draw[0]);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"ViewerDrawOpen" 
/*@C
   ViewerDrawOpen - Opens an X window for use as a viewer. If you want to 
   do graphics in this window, you must call ViewerDrawGetDraw() and
   perform the graphics on the Draw object.

   Collective on MPI_Comm

   Input Parameters:
+  comm - communicator that will share window
.  display - the X display on which to open, or null for the local machine
.  title - the title to put in the title bar, or null for no title
.  x, y - the screen coordinates of the upper left corner of window, or use PETSC_DECIDE
-  w, h - window width and height in pixels, or may use PETSC_DECIDE or DRAW_FULL_SIZE, DRAW_HALF_SIZE,
          DRAW_THIRD_SIZE, DRAW_QUARTER_SIZE

   Output Parameters:
.  viewer - the viewer

   Format Options:
+  VIEWER_FORMAT_DRAW_BASIC - displays with basic format
-  VIEWER_FORMAT_DRAW_LG    - displays using a line graph

   Options Database Keys:
   ViewerDrawOpen() calls DrawOpen(), so see the manual page for
   DrawOpen() for runtime options, including
+  -draw_type x or null
.  -nox - Disables all x-windows output
.  -display <name> - Specifies name of machine for the X display
-  -draw_pause <pause> - Sets time (in seconds) that the
     program pauses after DrawPause() has been called
     (0 is default, -1 implies until user input).

   Level: beginner

   Note for Fortran Programmers:
   Whenever indicating null character data in a Fortran code,
   PETSC_NULL_CHARACTER must be employed; using PETSC_NULL is not
   correct for character data!  Thus, PETSC_NULL_CHARACTER can be
   used for the display and title input parameters.

.keywords: draw, open, x, viewer

.seealso: DrawOpen(), ViewerDestroy(), ViewerDrawGetDraw(), ViewerCreate(), VIEWER_DRAW_,
          VIEWER_DRAW_WORLD, VIEWER_DRAW_SELF
@*/
int ViewerDrawOpen(MPI_Comm comm,const char display[],const char title[],int x,int y,int w,int h,Viewer *viewer)
{
  int ierr;

  PetscFunctionBegin;
  ierr = ViewerCreate(comm,viewer);CHKERRQ(ierr);
  ierr = ViewerSetType(*viewer,DRAW_VIEWER);CHKERRQ(ierr);
  ierr = ViewerDrawSetInfo(*viewer,display,title,x,y,w,h);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"ViewerGetSingleton_Draw" 
int ViewerGetSingleton_Draw(Viewer viewer,Viewer *sviewer)
{
  int         ierr,rank,i;
  Viewer_Draw *vdraw = (Viewer_Draw *)viewer->data,*vsdraw;

  PetscFunctionBegin;
  if (vdraw->singleton_made) {
    SETERRQ(1,1,"Trying to get singleton without first restoring previous");
  }

  ierr = MPI_Comm_rank(viewer->comm,&rank);CHKERRQ(ierr);
  if (rank) SETERRQ(1,1,"Cannot get singleton for Draw viewer except on processor 0");

  ierr   = ViewerCreate(PETSC_COMM_SELF,sviewer);CHKERRQ(ierr);
  ierr   = ViewerSetType(*sviewer,DRAW_VIEWER);CHKERRQ(ierr);
  vsdraw = (Viewer_Draw *)(*sviewer)->data;
  for (i=0; i<VIEWER_DRAW_MAX; i++) {
    if (vdraw->draw[i]) {
      ierr = DrawGetSingleton(vdraw->draw[i],&vsdraw->draw[i]);CHKERRQ(ierr);
    }
  }
  vdraw->singleton_made = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"ViewerRestoreSingleton_Draw" 
int ViewerRestoreSingleton_Draw(Viewer viewer,Viewer *sviewer)
{
  int         ierr,rank,i;
  Viewer_Draw *vdraw = (Viewer_Draw *)viewer->data,*vsdraw = (Viewer_Draw *)(*sviewer)->data;

  PetscFunctionBegin;
  if (!vdraw->singleton_made) {
    SETERRQ(1,1,"Trying to restore a singleton that was not gotten");
  }
  ierr = MPI_Comm_rank(viewer->comm,&rank);CHKERRQ(ierr);
  if (rank) SETERRQ(1,1,"Cannot restore singleton for Draw viewer except on processor 0");

  for (i=0; i<VIEWER_DRAW_MAX; i++) {
    if (vdraw->draw[i] && vsdraw->draw[i]) {
       ierr = DrawRestoreSingleton(vdraw->draw[i],&vsdraw->draw[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree((*sviewer)->data);CHKERRQ(ierr);
  PLogObjectDestroy((PetscObject)*sviewer);
  PetscHeaderDestroy((PetscObject)*sviewer);
  vdraw->singleton_made = PETSC_FALSE;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"ViewerCreate_Draw" 
int ViewerCreate_Draw(Viewer viewer)
{
  int         i;
  Viewer_Draw *vdraw;

  PetscFunctionBegin;
  vdraw        = PetscNew(Viewer_Draw);CHKPTRQ(vdraw);
  viewer->data = (void*)vdraw;

  viewer->ops->flush            = ViewerFlush_Draw;
  viewer->ops->destroy          = ViewerDestroy_Draw;
  viewer->ops->getsingleton     = ViewerGetSingleton_Draw;
  viewer->ops->restoresingleton = ViewerRestoreSingleton_Draw;
  viewer->format       = 0;

  /* these are created on the fly if requested */
  for (i=0; i<VIEWER_DRAW_MAX; i++) {
    vdraw->draw[i]     = 0; 
    vdraw->drawlg[i]   = 0; 
    vdraw->drawaxis[i] = 0;
  }
  vdraw->singleton_made = PETSC_FALSE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"ViewerDrawClear" 
/*@
    ViewerDrawClear - Clears a Draw graphic associated with a viewer.

    Not Collective

    Input Parameter:
.   viewer - the viewer 

    Level: intermediate

.seealso: ViewerDrawOpen(), ViewerDrawGetDraw(), 

@*/
int ViewerDrawClear(Viewer viewer)
{
  int         ierr,i;
  PetscTruth  isdraw;
  Viewer_Draw *vdraw;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,DRAW_VIEWER,&isdraw);CHKERRQ(ierr);
  if (isdraw) {
    vdraw = (Viewer_Draw*)viewer->data;
    for (i=0; i<VIEWER_DRAW_MAX; i++) {
      if (vdraw->draw[i]) {ierr = DrawClear(vdraw->draw[i]);CHKERRQ(ierr);}
    }
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------*/
/*
    The variable Petsc_Viewer_Draw_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a Viewer.
*/
static int Petsc_Viewer_Draw_keyval = MPI_KEYVAL_INVALID;

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"VIEWER_DRAW_" 
/*@C
     VIEWER_DRAW_ - Creates a window viewer shared by all processors 
                     in a communicator.

     Collective on MPI_Comm

     Input Parameter:
.    comm - the MPI communicator to share the window viewer

     Level: intermediate

     Notes:
     Unlike almost all other PETSc routines, VIEWER_DRAW_ does not return 
     an error code.  The window viewer is usually used in the form
$       XXXView(XXX object,VIEWER_DRAW_(comm));

.seealso: VIEWER_DRAW_WORLD, VIEWER_DRAW_SELF, ViewerDrawOpen(), 
@*/
Viewer VIEWER_DRAW_(MPI_Comm comm)
{
  int    ierr,flag;
  Viewer viewer;

  PetscFunctionBegin;
  if (Petsc_Viewer_Draw_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Viewer_Draw_keyval,0);
    if (ierr) {PetscError(__LINE__,"VIEWER_DRAW_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  }
  ierr = MPI_Attr_get(comm,Petsc_Viewer_Draw_keyval,(void **)&viewer,&flag);
  if (ierr) {PetscError(__LINE__,"VIEWER_DRAW_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  if (!flag) { /* viewer not yet created */
    ierr = ViewerDrawOpen(comm,0,0,PETSC_DECIDE,PETSC_DECIDE,300,300,&viewer); 
    if (ierr) {PetscError(__LINE__,"VIEWER_DRAW_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
    ierr = PetscObjectRegisterDestroy((PetscObject)viewer);
    if (ierr) {PetscError(__LINE__,"VIEWER_STDOUT_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
    ierr = MPI_Attr_put(comm,Petsc_Viewer_Draw_keyval,(void*)viewer);
    if (ierr) {PetscError(__LINE__,"VIEWER_DRAW_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  } 
  PetscFunctionReturn(viewer);
}


