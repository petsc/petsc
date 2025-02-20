#include <../src/sys/classes/viewer/impls/draw/vdraw.h> /*I "petscdraw.h" I*/
#include <petscviewer.h>                                /*I "petscviewer.h" I*/

/*@
  PetscViewerDrawGetDraw - Returns `PetscDraw` object from `PETSCVIEWERDRAW` `PetscViewer` object.
  This `PetscDraw` object may then be used to perform graphics using `PetscDraw` commands.

  Collective

  Input Parameters:
+ viewer       - the `PetscViewer` (created with `PetscViewerDrawOpen()` of type `PETSCVIEWERDRAW`)
- windownumber - indicates which subwindow (usually 0) to obtain

  Output Parameter:
. draw - the draw object

  Level: intermediate

.seealso: [](sec_viewers), `PETSCVIEWERDRAW`, `PetscViewerDrawGetLG()`, `PetscViewerDrawGetAxis()`, `PetscViewerDrawOpen()`
@*/
PetscErrorCode PetscViewerDrawGetDraw(PetscViewer viewer, PetscInt windownumber, PetscDraw *draw)
{
  PetscViewer_Draw *vdraw;
  PetscBool         isdraw;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidLogicalCollectiveInt(viewer, windownumber, 2);
  if (draw) PetscAssertPointer(draw, 3);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERDRAW, &isdraw));
  PetscCheck(isdraw, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Must be draw type PetscViewer");
  PetscCheck(windownumber >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Window number cannot be negative");
  vdraw = (PetscViewer_Draw *)viewer->data;

  windownumber += vdraw->draw_base;
  if (windownumber >= vdraw->draw_max) {
    /* allocate twice as many slots as needed */
    PetscInt       draw_max = vdraw->draw_max;
    PetscDraw     *tdraw    = vdraw->draw;
    PetscDrawLG   *drawlg   = vdraw->drawlg;
    PetscDrawAxis *drawaxis = vdraw->drawaxis;

    vdraw->draw_max = 2 * windownumber;

    PetscCall(PetscCalloc3(vdraw->draw_max, &vdraw->draw, vdraw->draw_max, &vdraw->drawlg, vdraw->draw_max, &vdraw->drawaxis));
    PetscCall(PetscArraycpy(vdraw->draw, tdraw, draw_max));
    PetscCall(PetscArraycpy(vdraw->drawlg, drawlg, draw_max));
    PetscCall(PetscArraycpy(vdraw->drawaxis, drawaxis, draw_max));
    PetscCall(PetscFree3(tdraw, drawlg, drawaxis));
  }

  if (!vdraw->draw[windownumber]) {
    char *title = vdraw->title, tmp_str[128];
    if (windownumber) {
      PetscCall(PetscSNPrintf(tmp_str, sizeof(tmp_str), "%s:%" PetscInt_FMT, vdraw->title ? vdraw->title : "", windownumber));
      title = tmp_str;
    }
    PetscCall(PetscDrawCreate(PetscObjectComm((PetscObject)viewer), vdraw->display, title, PETSC_DECIDE, PETSC_DECIDE, vdraw->w, vdraw->h, &vdraw->draw[windownumber]));
    if (vdraw->drawtype) PetscCall(PetscDrawSetType(vdraw->draw[windownumber], vdraw->drawtype));
    PetscCall(PetscDrawSetPause(vdraw->draw[windownumber], vdraw->pause));
    PetscCall(PetscDrawSetOptionsPrefix(vdraw->draw[windownumber], ((PetscObject)viewer)->prefix));
    PetscCall(PetscDrawSetFromOptions(vdraw->draw[windownumber]));
  }
  if (draw) *draw = vdraw->draw[windownumber];
  if (draw) PetscValidHeaderSpecific(*draw, PETSC_DRAW_CLASSID, 3);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerDrawGetDrawLG - Returns a `PetscDrawLG` object from `PetscViewer` object of type `PETSCVIEWERDRAW`.
  This `PetscDrawLG` object may then be used to perform graphics using `PetscDrawLG` commands.

  Collective

  Input Parameters:
+ viewer       - the `PetscViewer` (created with `PetscViewerDrawOpen()`)
- windownumber - indicates which subwindow (usually 0)

  Output Parameter:
. drawlg - the draw line graph object

  Level: intermediate

  Note:
  A `PETSCVIEWERDRAW` may have multiple `PetscDraw` subwindows

.seealso: [](sec_viewers), `PetscDrawLG`, `PetscViewerDrawGetDraw()`, `PetscViewerDrawGetAxis()`, `PetscViewerDrawOpen()`
@*/
PetscErrorCode PetscViewerDrawGetDrawLG(PetscViewer viewer, PetscInt windownumber, PetscDrawLG *drawlg)
{
  PetscBool         isdraw;
  PetscViewer_Draw *vdraw;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidLogicalCollectiveInt(viewer, windownumber, 2);
  PetscAssertPointer(drawlg, 3);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERDRAW, &isdraw));
  PetscCheck(isdraw, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Must be draw type PetscViewer");
  PetscCheck(windownumber >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Window number cannot be negative");
  vdraw = (PetscViewer_Draw *)viewer->data;

  if (windownumber + vdraw->draw_base >= vdraw->draw_max || !vdraw->draw[windownumber + vdraw->draw_base]) PetscCall(PetscViewerDrawGetDraw(viewer, windownumber, NULL));
  if (!vdraw->drawlg[windownumber + vdraw->draw_base]) {
    PetscCall(PetscDrawLGCreate(vdraw->draw[windownumber + vdraw->draw_base], 1, &vdraw->drawlg[windownumber + vdraw->draw_base]));
    PetscCall(PetscDrawLGSetFromOptions(vdraw->drawlg[windownumber + vdraw->draw_base]));
  }
  *drawlg = vdraw->drawlg[windownumber + vdraw->draw_base];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerDrawGetDrawAxis - Returns a `PetscDrawAxis` object from a `PetscViewer` object of type `PETSCVIEWERDRAW`.
  This `PetscDrawAxis` object may then be used to perform graphics using `PetscDrawAxis` commands.

  Collective

  Input Parameters:
+ viewer       - the `PetscViewer` (created with `PetscViewerDrawOpen()`)
- windownumber - indicates which subwindow (usually 0)

  Output Parameter:
. drawaxis - the draw axis object

  Level: advanced

  Note:
  A `PETSCVIEWERDRAW` may have multiple `PetscDraw` subwindows

.seealso: [](sec_viewers), `PetscViewerDrawGetDraw()`, `PetscViewerDrawGetLG()`, `PetscViewerDrawOpen()`
@*/
PetscErrorCode PetscViewerDrawGetDrawAxis(PetscViewer viewer, PetscInt windownumber, PetscDrawAxis *drawaxis)
{
  PetscBool         isdraw;
  PetscViewer_Draw *vdraw;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidLogicalCollectiveInt(viewer, windownumber, 2);
  PetscAssertPointer(drawaxis, 3);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERDRAW, &isdraw));
  PetscCheck(isdraw, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Must be draw type PetscViewer");
  PetscCheck(windownumber >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Window number cannot be negative");
  vdraw = (PetscViewer_Draw *)viewer->data;

  if (windownumber + vdraw->draw_base >= vdraw->draw_max || !vdraw->draw[windownumber + vdraw->draw_base]) PetscCall(PetscViewerDrawGetDraw(viewer, windownumber, NULL));
  if (!vdraw->drawaxis[windownumber + vdraw->draw_base]) PetscCall(PetscDrawAxisCreate(vdraw->draw[windownumber + vdraw->draw_base], &vdraw->drawaxis[windownumber + vdraw->draw_base]));
  *drawaxis = vdraw->drawaxis[windownumber + vdraw->draw_base];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerDrawSetDrawType(PetscViewer v, PetscDrawType drawtype)
{
  PetscViewer_Draw *vdraw;
  PetscBool         isdraw;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, PETSC_VIEWER_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERDRAW, &isdraw));
  if (!isdraw) PetscFunctionReturn(PETSC_SUCCESS);
  vdraw = (PetscViewer_Draw *)v->data;

  PetscCall(PetscFree(vdraw->drawtype));
  PetscCall(PetscStrallocpy(drawtype, (char **)&vdraw->drawtype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerDrawGetDrawType(PetscViewer v, PetscDrawType *drawtype)
{
  PetscViewer_Draw *vdraw;
  PetscBool         isdraw;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, PETSC_VIEWER_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERDRAW, &isdraw));
  PetscCheck(isdraw, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Must be draw type PetscViewer");
  vdraw = (PetscViewer_Draw *)v->data;

  *drawtype = vdraw->drawtype;
  PetscFunctionReturn(PETSC_SUCCESS);
}
