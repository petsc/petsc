
#include <petscviewer.h>
#include <petsc/private/drawimpl.h>  /*I   "petscdraw.h"  I*/
PetscClassId PETSC_DRAWLG_CLASSID = 0;

/*@
   PetscDrawLGGetAxis - Gets the axis context associated with a line graph.
   This is useful if one wants to change some axis property, such as
   labels, color, etc. The axis context should not be destroyed by the
   application code.

   Not Collective, if PetscDrawLG is parallel then PetscDrawAxis is parallel

   Input Parameter:
.  lg - the line graph context

   Output Parameter:
.  axis - the axis context

   Level: advanced

.seealso: PetscDrawLGCreate(), PetscDrawAxis

@*/
PetscErrorCode  PetscDrawLGGetAxis(PetscDrawLG lg,PetscDrawAxis *axis)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  PetscValidPointer(axis,2);
  *axis = lg->axis;
  PetscFunctionReturn(0);
}

/*@
   PetscDrawLGGetDraw - Gets the draw context associated with a line graph.

   Not Collective, if PetscDrawLG is parallel then PetscDraw is parallel

   Input Parameter:
.  lg - the line graph context

   Output Parameter:
.  draw - the draw context

   Level: intermediate

.seealso: PetscDrawLGCreate(), PetscDraw
@*/
PetscErrorCode  PetscDrawLGGetDraw(PetscDrawLG lg,PetscDraw *draw)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  PetscValidPointer(draw,2);
  *draw = lg->win;
  PetscFunctionReturn(0);
}

/*@
   PetscDrawLGSPDraw - Redraws a line graph.

   Collective on PetscDrawLG

   Input Parameter:
.  lg - the line graph context

   Level: intermediate

.seealso: PetscDrawLGDraw(), PetscDrawSPDraw()

   Developer Notes:
    This code cheats and uses the fact that the LG and SP structs are the same

@*/
PetscErrorCode  PetscDrawLGSPDraw(PetscDrawLG lg,PetscDrawSP spin)
{
  PetscDrawLG    sp = (PetscDrawLG)spin;
  PetscReal      xmin,xmax,ymin,ymax;
  PetscBool      isnull;
  PetscMPIInt    rank;
  PetscDraw      draw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  PetscValidHeaderSpecific(sp,PETSC_DRAWLG_CLASSID,2);
  CHKERRQ(PetscDrawIsNull(lg->win,&isnull));
  if (isnull) PetscFunctionReturn(0);
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)lg),&rank));

  draw = lg->win;
  CHKERRQ(PetscDrawCheckResizedWindow(draw));
  CHKERRQ(PetscDrawClear(draw));

  xmin = PetscMin(lg->xmin,sp->xmin); ymin = PetscMin(lg->ymin,sp->ymin);
  xmax = PetscMax(lg->xmax,sp->xmax); ymax = PetscMax(lg->ymax,sp->ymax);
  CHKERRQ(PetscDrawAxisSetLimits(lg->axis,xmin,xmax,ymin,ymax));
  CHKERRQ(PetscDrawAxisDraw(lg->axis));

  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  if (rank == 0) {
    int i,j,dim,nopts;
    dim   = lg->dim;
    nopts = lg->nopts;
    for (i=0; i<dim; i++) {
      for (j=1; j<nopts; j++) {
        CHKERRQ(PetscDrawLine(draw,lg->x[(j-1)*dim+i],lg->y[(j-1)*dim+i],lg->x[j*dim+i],lg->y[j*dim+i],PETSC_DRAW_BLACK+i));
        if (lg->use_markers) {
          CHKERRQ(PetscDrawMarker(draw,lg->x[j*dim+i],lg->y[j*dim+i],PETSC_DRAW_RED));
        }
      }
    }
    dim   = sp->dim;
    nopts = sp->nopts;
    for (i=0; i<dim; i++) {
      for (j=0; j<nopts; j++) {
        CHKERRQ(PetscDrawMarker(draw,sp->x[j*dim+i],sp->y[j*dim+i],PETSC_DRAW_RED));
      }
    }
  }
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);

  CHKERRQ(PetscDrawFlush(draw));
  CHKERRQ(PetscDrawPause(draw));
  PetscFunctionReturn(0);
}

/*@
    PetscDrawLGCreate - Creates a line graph data structure.

    Collective on PetscDraw

    Input Parameters:
+   draw - the window where the graph will be made.
-   dim - the number of curves which will be drawn

    Output Parameters:
.   outlg - the line graph context

    Level: intermediate

    Notes:
    The MPI communicator that owns the PetscDraw owns this PetscDrawLG, but the calls to set options and add points are ignored on all processes except the
           zeroth MPI process in the communicator. All MPI processes in the communicator must call PetscDrawLGDraw() to display the updated graph.

.seealso:  PetscDrawLGDestroy(), PetscDrawLGAddPoint(), PetscDrawLGAddCommonPoint(), PetscDrawLGAddPoints(), PetscDrawLGDraw(), PetscDrawLGSave(),
           PetscDrawLGView(), PetscDrawLGReset(), PetscDrawLGSetDimension(), PetscDrawLGGetDimension(), PetscDrawLGSetLegend(), PetscDrawLGGetAxis(),
           PetscDrawLGGetDraw(), PetscDrawLGSetUseMarkers(), PetscDrawLGSetLimits(), PetscDrawLGSetColors(), PetscDrawLGSetOptionsPrefix(), PetscDrawLGSetFromOptions()
@*/
PetscErrorCode  PetscDrawLGCreate(PetscDraw draw,PetscInt dim,PetscDrawLG *outlg)
{
  PetscDrawLG    lg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidLogicalCollectiveInt(draw,dim,2);
  PetscValidPointer(outlg,3);

  CHKERRQ(PetscHeaderCreate(lg,PETSC_DRAWLG_CLASSID,"DrawLG","Line Graph","Draw",PetscObjectComm((PetscObject)draw),PetscDrawLGDestroy,NULL));
  CHKERRQ(PetscLogObjectParent((PetscObject)draw,(PetscObject)lg));
  CHKERRQ(PetscDrawLGSetOptionsPrefix(lg,((PetscObject)draw)->prefix));

  CHKERRQ(PetscObjectReference((PetscObject)draw));
  lg->win = draw;

  lg->view    = NULL;
  lg->destroy = NULL;
  lg->nopts   = 0;
  lg->dim     = dim;
  lg->xmin    = 1.e20;
  lg->ymin    = 1.e20;
  lg->xmax    = -1.e20;
  lg->ymax    = -1.e20;

  CHKERRQ(PetscMalloc2(dim*PETSC_DRAW_LG_CHUNK_SIZE,&lg->x,dim*PETSC_DRAW_LG_CHUNK_SIZE,&lg->y));
  CHKERRQ(PetscLogObjectMemory((PetscObject)lg,2*dim*PETSC_DRAW_LG_CHUNK_SIZE*sizeof(PetscReal)));

  lg->len         = dim*PETSC_DRAW_LG_CHUNK_SIZE;
  lg->loc         = 0;
  lg->use_markers = PETSC_FALSE;

  CHKERRQ(PetscDrawAxisCreate(draw,&lg->axis));
  CHKERRQ(PetscLogObjectParent((PetscObject)lg,(PetscObject)lg->axis));

  *outlg = lg;
  PetscFunctionReturn(0);
}

/*@
   PetscDrawLGSetColors - Sets the color of each line graph drawn

   Logically Collective on PetscDrawLG

   Input Parameters:
+  lg - the line graph context.
-  colors - the colors

   Level: intermediate

.seealso: PetscDrawLGCreate()

@*/
PetscErrorCode  PetscDrawLGSetColors(PetscDrawLG lg,const int colors[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  if (lg->dim) PetscValidIntPointer(colors,2);

  CHKERRQ(PetscFree(lg->colors));
  CHKERRQ(PetscMalloc1(lg->dim,&lg->colors));
  CHKERRQ(PetscArraycpy(lg->colors,colors,lg->dim));
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawLGSetLegend - sets the names of each curve plotted

   Logically Collective on PetscDrawLG

   Input Parameters:
+  lg - the line graph context.
-  names - the names for each curve

   Level: intermediate

   Notes:
    Call PetscDrawLGGetAxis() and then change properties of the PetscDrawAxis for detailed control of the plot

.seealso: PetscDrawLGGetAxis(), PetscDrawAxis, PetscDrawAxisSetColors(), PetscDrawAxisSetLabels(), PetscDrawAxisSetHoldLimits()

@*/
PetscErrorCode  PetscDrawLGSetLegend(PetscDrawLG lg,const char *const *names)
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  if (names) PetscValidPointer(names,2);

  if (lg->legend) {
    for (i=0; i<lg->dim; i++) {
      CHKERRQ(PetscFree(lg->legend[i]));
    }
    CHKERRQ(PetscFree(lg->legend));
  }
  if (names) {
    CHKERRQ(PetscMalloc1(lg->dim,&lg->legend));
    for (i=0; i<lg->dim; i++) {
      CHKERRQ(PetscStrallocpy(names[i],&lg->legend[i]));
    }
  }
  PetscFunctionReturn(0);
}

/*@
   PetscDrawLGGetDimension - Change the number of lines that are to be drawn.

   Not Collective

   Input Parameter:
.  lg - the line graph context.

   Output Parameter:
.  dim - the number of curves.

   Level: intermediate

.seealso: PetscDrawLGCreate(), PetscDrawLGSetDimension()

@*/
PetscErrorCode  PetscDrawLGGetDimension(PetscDrawLG lg,PetscInt *dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  PetscValidIntPointer(dim,2);
  *dim = lg->dim;
  PetscFunctionReturn(0);
}

/*@
   PetscDrawLGSetDimension - Change the number of lines that are to be drawn.

   Logically Collective on PetscDrawLG

   Input Parameters:
+  lg - the line graph context.
-  dim - the number of curves.

   Level: intermediate

.seealso: PetscDrawLGCreate(), PetscDrawLGGetDimension()
@*/
PetscErrorCode  PetscDrawLGSetDimension(PetscDrawLG lg,PetscInt dim)
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  PetscValidLogicalCollectiveInt(lg,dim,2);
  if (lg->dim == dim) PetscFunctionReturn(0);

  CHKERRQ(PetscFree2(lg->x,lg->y));
  if (lg->legend) {
    for (i=0; i<lg->dim; i++) {
      CHKERRQ(PetscFree(lg->legend[i]));
    }
    CHKERRQ(PetscFree(lg->legend));
  }
  CHKERRQ(PetscFree(lg->colors));
  lg->dim = dim;
  CHKERRQ(PetscMalloc2(dim*PETSC_DRAW_LG_CHUNK_SIZE,&lg->x,dim*PETSC_DRAW_LG_CHUNK_SIZE,&lg->y));
  CHKERRQ(PetscLogObjectMemory((PetscObject)lg,2*dim*PETSC_DRAW_LG_CHUNK_SIZE*sizeof(PetscReal)));
  lg->len = dim*PETSC_DRAW_LG_CHUNK_SIZE;
  PetscFunctionReturn(0);
}

/*@
   PetscDrawLGSetLimits - Sets the axis limits for a line graph. If more
   points are added after this call, the limits will be adjusted to
   include those additional points.

   Logically Collective on PetscDrawLG

   Input Parameters:
+  xlg - the line graph context
-  x_min,x_max,y_min,y_max - the limits

   Level: intermediate

.seealso: PetscDrawLGCreate()

@*/
PetscErrorCode  PetscDrawLGSetLimits(PetscDrawLG lg,PetscReal x_min,PetscReal x_max,PetscReal y_min,PetscReal y_max)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);

  (lg)->xmin = x_min;
  (lg)->xmax = x_max;
  (lg)->ymin = y_min;
  (lg)->ymax = y_max;
  PetscFunctionReturn(0);
}

/*@
   PetscDrawLGReset - Clears line graph to allow for reuse with new data.

   Logically Collective on PetscDrawLG

   Input Parameter:
.  lg - the line graph context.

   Level: intermediate

.seealso: PetscDrawLGCreate()
@*/
PetscErrorCode  PetscDrawLGReset(PetscDrawLG lg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  lg->xmin  = 1.e20;
  lg->ymin  = 1.e20;
  lg->xmax  = -1.e20;
  lg->ymax  = -1.e20;
  lg->loc   = 0;
  lg->nopts = 0;
  PetscFunctionReturn(0);
}

/*@
   PetscDrawLGDestroy - Frees all space taken up by line graph data structure.

   Collective on PetscDrawLG

   Input Parameter:
.  lg - the line graph context

   Level: intermediate

.seealso:  PetscDrawLGCreate()
@*/
PetscErrorCode  PetscDrawLGDestroy(PetscDrawLG *lg)
{
  PetscInt       i;

  PetscFunctionBegin;
  if (!*lg) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*lg,PETSC_DRAWLG_CLASSID,1);
  if (--((PetscObject)(*lg))->refct > 0) {*lg = NULL; PetscFunctionReturn(0);}

  if ((*lg)->legend) {
    for (i=0; i<(*lg)->dim; i++) {
      CHKERRQ(PetscFree((*lg)->legend[i]));
    }
    CHKERRQ(PetscFree((*lg)->legend));
  }
  CHKERRQ(PetscFree((*lg)->colors));
  CHKERRQ(PetscFree2((*lg)->x,(*lg)->y));
  CHKERRQ(PetscDrawAxisDestroy(&(*lg)->axis));
  CHKERRQ(PetscDrawDestroy(&(*lg)->win));
  CHKERRQ(PetscHeaderDestroy(lg));
  PetscFunctionReturn(0);
}
/*@
   PetscDrawLGSetUseMarkers - Causes LG to draw a marker for each data-point.

   Logically Collective on PetscDrawLG

   Input Parameters:
+  lg - the linegraph context
-  flg - should mark each data point

   Options Database:
.  -lg_use_markers  <true,false> - true means the graphPetscDrawLG draws a marker for each point

   Level: intermediate

.seealso: PetscDrawLGCreate()
@*/
PetscErrorCode  PetscDrawLGSetUseMarkers(PetscDrawLG lg,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  PetscValidLogicalCollectiveBool(lg,flg,2);
  lg->use_markers = flg;
  PetscFunctionReturn(0);
}

/*@
   PetscDrawLGDraw - Redraws a line graph.

   Collective on PetscDrawLG

   Input Parameter:
.  lg - the line graph context

   Level: intermediate

.seealso: PetscDrawSPDraw(), PetscDrawLGSPDraw(), PetscDrawLGReset()
@*/
PetscErrorCode  PetscDrawLGDraw(PetscDrawLG lg)
{
  PetscReal      xmin,xmax,ymin,ymax;
  PetscMPIInt    rank;
  PetscDraw      draw;
  PetscBool      isnull;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  CHKERRQ(PetscDrawIsNull(lg->win,&isnull));
  if (isnull) PetscFunctionReturn(0);
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)lg),&rank));

  draw = lg->win;
  CHKERRQ(PetscDrawCheckResizedWindow(draw));
  CHKERRQ(PetscDrawClear(draw));

  xmin = lg->xmin; xmax = lg->xmax; ymin = lg->ymin; ymax = lg->ymax;
  CHKERRQ(PetscDrawAxisSetLimits(lg->axis,xmin,xmax,ymin,ymax));
  CHKERRQ(PetscDrawAxisDraw(lg->axis));

  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  if (rank == 0) {
    int i,j,dim=lg->dim,nopts=lg->nopts,cl;
    for (i=0; i<dim; i++) {
      for (j=1; j<nopts; j++) {
        cl   = lg->colors ? lg->colors[i] : ((PETSC_DRAW_BLACK + i) % PETSC_DRAW_MAXCOLOR);
        CHKERRQ(PetscDrawLine(draw,lg->x[(j-1)*dim+i],lg->y[(j-1)*dim+i],lg->x[j*dim+i],lg->y[j*dim+i],cl));
        if (lg->use_markers) CHKERRQ(PetscDrawMarker(draw,lg->x[j*dim+i],lg->y[j*dim+i],cl));
      }
    }
  }
  if (rank == 0 && lg->legend) {
    PetscBool right = PETSC_FALSE;
    int       i,dim=lg->dim,cl;
    PetscReal xl,yl,xr,yr,tw,th;
    size_t    slen,len=0;
    CHKERRQ(PetscDrawAxisGetLimits(lg->axis,&xl,&xr,&yl,&yr));
    CHKERRQ(PetscDrawStringGetSize(draw,&tw,&th));
    for (i=0; i<dim; i++) {
      CHKERRQ(PetscStrlen(lg->legend[i],&slen));
      len = PetscMax(len,slen);
    }
    if (right) {
      xr = xr - 1.5*tw; xl = xr - (len + 7)*tw;
    } else {
      xl = xl + 1.5*tw; xr = xl + (len + 7)*tw;
    }
    yr = yr - 1.0*th; yl = yr - (dim + 1)*th;
    CHKERRQ(PetscDrawLine(draw,xl,yl,xr,yl,PETSC_DRAW_BLACK));
    CHKERRQ(PetscDrawLine(draw,xr,yl,xr,yr,PETSC_DRAW_BLACK));
    CHKERRQ(PetscDrawLine(draw,xr,yr,xl,yr,PETSC_DRAW_BLACK));
    CHKERRQ(PetscDrawLine(draw,xl,yr,xl,yl,PETSC_DRAW_BLACK));
    for  (i=0; i<dim; i++) {
      cl   = lg->colors ? lg->colors[i] : (PETSC_DRAW_BLACK + i);
      CHKERRQ(PetscDrawLine(draw,xl + 1*tw,yr - (i + 1)*th,xl + 5*tw,yr - (i + 1)*th,cl));
      CHKERRQ(PetscDrawString(draw,xl + 6*tw,yr - (i + 1.5)*th,PETSC_DRAW_BLACK,lg->legend[i]));
    }
  }
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);

  CHKERRQ(PetscDrawFlush(draw));
  CHKERRQ(PetscDrawPause(draw));
  PetscFunctionReturn(0);
}

/*@
  PetscDrawLGSave - Saves a drawn image

  Collective on PetscDrawLG

  Input Parameter:
. lg - The line graph context

  Level: intermediate

.seealso:  PetscDrawLGCreate(), PetscDrawLGGetDraw(), PetscDrawSetSave(), PetscDrawSave()
@*/
PetscErrorCode  PetscDrawLGSave(PetscDrawLG lg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  CHKERRQ(PetscDrawSave(lg->win));
  PetscFunctionReturn(0);
}

/*@
  PetscDrawLGView - Prints a line graph.

  Collective on PetscDrawLG

  Input Parameter:
. lg - the line graph context

  Level: beginner

.seealso: PetscDrawLGCreate()

@*/
PetscErrorCode  PetscDrawLGView(PetscDrawLG lg,PetscViewer viewer)
{
  PetscReal      xmin=lg->xmin, xmax=lg->xmax, ymin=lg->ymin, ymax=lg->ymax;
  PetscInt       i, j, dim = lg->dim, nopts = lg->nopts;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);

  if (nopts < 1)                  PetscFunctionReturn(0);
  if (xmin > xmax || ymin > ymax) PetscFunctionReturn(0);

  if (!viewer) {
    CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)lg),&viewer));
  }
  CHKERRQ(PetscObjectPrintClassNamePrefixType((PetscObject)lg,viewer));
  for (i = 0; i < dim; i++) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "Line %" PetscInt_FMT ">\n", i));
    for (j = 0; j < nopts; j++) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "  X: %g Y: %g\n", (double)lg->x[j*dim+i], (double)lg->y[j*dim+i]));
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawLGSetOptionsPrefix - Sets the prefix used for searching for all
   PetscDrawLG options in the database.

   Logically Collective on PetscDrawLG

   Input Parameters:
+  lg - the line graph context
-  prefix - the prefix to prepend to all option names

   Level: advanced

.seealso: PetscDrawLGSetFromOptions(), PetscDrawLGCreate()
@*/
PetscErrorCode  PetscDrawLGSetOptionsPrefix(PetscDrawLG lg,const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)lg,prefix));
  PetscFunctionReturn(0);
}

/*@
    PetscDrawLGSetFromOptions - Sets options related to the PetscDrawLG

    Collective on PetscDrawLG

    Options Database:

    Level: intermediate

.seealso:  PetscDrawLGDestroy(), PetscDrawLGCreate()
@*/
PetscErrorCode  PetscDrawLGSetFromOptions(PetscDrawLG lg)
{
  PetscBool           usemarkers,set;
  PetscDrawMarkerType markertype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);

  CHKERRQ(PetscDrawGetMarkerType(lg->win,&markertype));
  CHKERRQ(PetscOptionsGetEnum(((PetscObject)lg)->options,((PetscObject)lg)->prefix,"-lg_marker_type",PetscDrawMarkerTypes,(PetscEnum*)&markertype,&set));
  if (set) {
    CHKERRQ(PetscDrawLGSetUseMarkers(lg,PETSC_TRUE));
    CHKERRQ(PetscDrawSetMarkerType(lg->win,markertype));
  }
  usemarkers = lg->use_markers;
  CHKERRQ(PetscOptionsGetBool(((PetscObject)lg)->options,((PetscObject)lg)->prefix,"-lg_use_markers",&usemarkers,&set));
  if (set) CHKERRQ(PetscDrawLGSetUseMarkers(lg,usemarkers));
  PetscFunctionReturn(0);
}
