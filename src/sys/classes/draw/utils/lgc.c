
#include <petscviewer.h>
#include <../src/sys/classes/draw/utils/lgimpl.h>  /*I   "petscdraw.h"  I*/
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
  PetscErrorCode ierr;
  PetscBool      isnull;
  PetscMPIInt    rank;
  PetscDraw      draw;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  PetscValidHeaderSpecific(sp,PETSC_DRAWLG_CLASSID,2);
  ierr = PetscDrawIsNull(lg->win,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)lg),&rank);CHKERRMPI(ierr);

  draw = lg->win;
  ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
  ierr = PetscDrawClear(draw);CHKERRQ(ierr);

  xmin = PetscMin(lg->xmin,sp->xmin); ymin = PetscMin(lg->ymin,sp->ymin);
  xmax = PetscMax(lg->xmax,sp->xmax); ymax = PetscMax(lg->ymax,sp->ymax);
  ierr = PetscDrawAxisSetLimits(lg->axis,xmin,xmax,ymin,ymax);CHKERRQ(ierr);
  ierr = PetscDrawAxisDraw(lg->axis);CHKERRQ(ierr);

  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  if (!rank) {
    int i,j,dim,nopts;
    dim   = lg->dim;
    nopts = lg->nopts;
    for (i=0; i<dim; i++) {
      for (j=1; j<nopts; j++) {
        ierr = PetscDrawLine(draw,lg->x[(j-1)*dim+i],lg->y[(j-1)*dim+i],lg->x[j*dim+i],lg->y[j*dim+i],PETSC_DRAW_BLACK+i);CHKERRQ(ierr);
        if (lg->use_markers) {
          ierr = PetscDrawMarker(draw,lg->x[j*dim+i],lg->y[j*dim+i],PETSC_DRAW_RED);CHKERRQ(ierr);
        }
      }
    }
    dim   = sp->dim;
    nopts = sp->nopts;
    for (i=0; i<dim; i++) {
      for (j=0; j<nopts; j++) {
        ierr = PetscDrawMarker(draw,sp->x[j*dim+i],sp->y[j*dim+i],PETSC_DRAW_RED);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);

  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidLogicalCollectiveInt(draw,dim,2);
  PetscValidPointer(outlg,3);

  ierr = PetscHeaderCreate(lg,PETSC_DRAWLG_CLASSID,"DrawLG","Line Graph","Draw",PetscObjectComm((PetscObject)draw),PetscDrawLGDestroy,NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)draw,(PetscObject)lg);CHKERRQ(ierr);
  ierr = PetscDrawLGSetOptionsPrefix(lg,((PetscObject)draw)->prefix);CHKERRQ(ierr);

  ierr = PetscObjectReference((PetscObject)draw);CHKERRQ(ierr);
  lg->win = draw;

  lg->view    = NULL;
  lg->destroy = NULL;
  lg->nopts   = 0;
  lg->dim     = dim;
  lg->xmin    = 1.e20;
  lg->ymin    = 1.e20;
  lg->xmax    = -1.e20;
  lg->ymax    = -1.e20;

  ierr = PetscMalloc2(dim*CHUNCKSIZE,&lg->x,dim*CHUNCKSIZE,&lg->y);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)lg,2*dim*CHUNCKSIZE*sizeof(PetscReal));CHKERRQ(ierr);

  lg->len         = dim*CHUNCKSIZE;
  lg->loc         = 0;
  lg->use_markers = PETSC_FALSE;

  ierr = PetscDrawAxisCreate(draw,&lg->axis);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)lg,(PetscObject)lg->axis);CHKERRQ(ierr);

  *outlg = lg;
  PetscFunctionReturn(0);
}

/*@
   PetscDrawLGSetColors - Sets the color of each line graph drawn

   Logically Collective on PetscDrawLG

   Input Parameter:
+  lg - the line graph context.
-  colors - the colors

   Level: intermediate

.seealso: PetscDrawLGCreate()

@*/
PetscErrorCode  PetscDrawLGSetColors(PetscDrawLG lg,const int colors[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  if (lg->dim) PetscValidIntPointer(colors,2);

  ierr = PetscFree(lg->colors);CHKERRQ(ierr);
  ierr = PetscMalloc1(lg->dim,&lg->colors);CHKERRQ(ierr);
  ierr = PetscArraycpy(lg->colors,colors,lg->dim);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawLGSetLegend - sets the names of each curve plotted

   Logically Collective on PetscDrawLG

   Input Parameter:
+  lg - the line graph context.
-  names - the names for each curve

   Level: intermediate

   Notes:
    Call PetscDrawLGGetAxis() and then change properties of the PetscDrawAxis for detailed control of the plot

.seealso: PetscDrawLGGetAxis(), PetscDrawAxis, PetscDrawAxisSetColors(), PetscDrawAxisSetLabels(), PetscDrawAxisSetHoldLimits()

@*/
PetscErrorCode  PetscDrawLGSetLegend(PetscDrawLG lg,const char *const *names)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  if (names) PetscValidPointer(names,2);

  if (lg->legend) {
    for (i=0; i<lg->dim; i++) {
      ierr = PetscFree(lg->legend[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(lg->legend);CHKERRQ(ierr);
  }
  if (names) {
    ierr = PetscMalloc1(lg->dim,&lg->legend);CHKERRQ(ierr);
    for (i=0; i<lg->dim; i++) {
      ierr = PetscStrallocpy(names[i],&lg->legend[i]);CHKERRQ(ierr);
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

   Input Parameter:
+  lg - the line graph context.
-  dim - the number of curves.

   Level: intermediate

.seealso: PetscDrawLGCreate(), PetscDrawLGGetDimension()
@*/
PetscErrorCode  PetscDrawLGSetDimension(PetscDrawLG lg,PetscInt dim)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  PetscValidLogicalCollectiveInt(lg,dim,2);
  if (lg->dim == dim) PetscFunctionReturn(0);

  ierr = PetscFree2(lg->x,lg->y);CHKERRQ(ierr);
  if (lg->legend) {
    for (i=0; i<lg->dim; i++) {
      ierr = PetscFree(lg->legend[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(lg->legend);CHKERRQ(ierr);
  }
  ierr    = PetscFree(lg->colors);CHKERRQ(ierr);
  lg->dim = dim;
  ierr    = PetscMalloc2(dim*CHUNCKSIZE,&lg->x,dim*CHUNCKSIZE,&lg->y);CHKERRQ(ierr);
  ierr    = PetscLogObjectMemory((PetscObject)lg,2*dim*CHUNCKSIZE*sizeof(PetscReal));CHKERRQ(ierr);
  lg->len = dim*CHUNCKSIZE;
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
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (!*lg) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*lg,PETSC_DRAWLG_CLASSID,1);
  if (--((PetscObject)(*lg))->refct > 0) {*lg = NULL; PetscFunctionReturn(0);}

  if ((*lg)->legend) {
    for (i=0; i<(*lg)->dim; i++) {
      ierr = PetscFree((*lg)->legend[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree((*lg)->legend);CHKERRQ(ierr);
  }
  ierr = PetscFree((*lg)->colors);CHKERRQ(ierr);
  ierr = PetscFree2((*lg)->x,(*lg)->y);CHKERRQ(ierr);
  ierr = PetscDrawAxisDestroy(&(*lg)->axis);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&(*lg)->win);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(lg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*@
   PetscDrawLGSetUseMarkers - Causes LG to draw a marker for each data-point.

   Logically Collective on PetscDrawLG

   Input Parameters:
+  lg - the linegraph context
-  flg - should mark each data point

   Options Database:
.  -lg_use_markers  <true,false>

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
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscDraw      draw;
  PetscBool      isnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  ierr = PetscDrawIsNull(lg->win,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)lg),&rank);CHKERRMPI(ierr);

  draw = lg->win;
  ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
  ierr = PetscDrawClear(draw);CHKERRQ(ierr);

  xmin = lg->xmin; xmax = lg->xmax; ymin = lg->ymin; ymax = lg->ymax;
  ierr = PetscDrawAxisSetLimits(lg->axis,xmin,xmax,ymin,ymax);CHKERRQ(ierr);
  ierr = PetscDrawAxisDraw(lg->axis);CHKERRQ(ierr);

  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  if (!rank) {
    int i,j,dim=lg->dim,nopts=lg->nopts,cl;
    for (i=0; i<dim; i++) {
      for (j=1; j<nopts; j++) {
        cl   = lg->colors ? lg->colors[i] : ((PETSC_DRAW_BLACK + i) % PETSC_DRAW_MAXCOLOR);
        ierr = PetscDrawLine(draw,lg->x[(j-1)*dim+i],lg->y[(j-1)*dim+i],lg->x[j*dim+i],lg->y[j*dim+i],cl);CHKERRQ(ierr);
        if (lg->use_markers) {ierr = PetscDrawMarker(draw,lg->x[j*dim+i],lg->y[j*dim+i],cl);CHKERRQ(ierr);}
      }
    }
  }
  if (!rank && lg->legend) {
    int       i,dim=lg->dim,cl;
    PetscReal xl,yl,xr,yr,tw,th;
    size_t    slen,len=0;
    ierr = PetscDrawAxisGetLimits(lg->axis,&xl,&xr,&yl,&yr);CHKERRQ(ierr);
    ierr = PetscDrawStringGetSize(draw,&tw,&th);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      ierr = PetscStrlen(lg->legend[i],&slen);CHKERRQ(ierr);
      len = PetscMax(len,slen);
    }
    xr = xr - 1.5*tw; xl = xr - (len + 7)*tw;
    yr = yr - 1.0*th; yl = yr - (dim + 1)*th;
    ierr = PetscDrawLine(draw,xl,yl,xr,yl,PETSC_DRAW_BLACK);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xr,yl,xr,yr,PETSC_DRAW_BLACK);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xr,yr,xl,yr,PETSC_DRAW_BLACK);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xl,yr,xl,yl,PETSC_DRAW_BLACK);CHKERRQ(ierr);
    for  (i=0; i<dim; i++) {
      cl   = lg->colors ? lg->colors[i] : (PETSC_DRAW_BLACK + i);
      ierr = PetscDrawLine(draw,xl + 1*tw,yr - (i + 1)*th,xl + 5*tw,yr - (i + 1)*th,cl);CHKERRQ(ierr);
      ierr = PetscDrawString(draw,xl + 6*tw,yr - (i + 1.5)*th,PETSC_DRAW_BLACK,lg->legend[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);

  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  ierr = PetscDrawSave(lg->win);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);

  if (nopts < 1)                  PetscFunctionReturn(0);
  if (xmin > xmax || ymin > ymax) PetscFunctionReturn(0);

  if (!viewer){
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)lg),&viewer);CHKERRQ(ierr);
  }
  ierr = PetscObjectPrintClassNamePrefixType((PetscObject)lg,viewer);CHKERRQ(ierr);
  for (i = 0; i < dim; i++) {
    ierr = PetscViewerASCIIPrintf(viewer, "Line %D>\n", i);CHKERRQ(ierr);
    for (j = 0; j < nopts; j++) {
      ierr = PetscViewerASCIIPrintf(viewer, "  X: %g Y: %g\n", (double)lg->x[j*dim+i], (double)lg->y[j*dim+i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawLGSetOptionsPrefix - Sets the prefix used for searching for all
   PetscDrawLG options in the database.

   Logically Collective on PetscDrawLG

   Input Parameter:
+  lg - the line graph context
-  prefix - the prefix to prepend to all option names

   Level: advanced

.seealso: PetscDrawLGSetFromOptions(), PetscDrawLGCreate()
@*/
PetscErrorCode  PetscDrawLGSetOptionsPrefix(PetscDrawLG lg,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)lg,prefix);CHKERRQ(ierr);
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
  PetscErrorCode      ierr;
  PetscBool           usemarkers,set;
  PetscDrawMarkerType markertype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);

  ierr = PetscDrawGetMarkerType(lg->win,&markertype);CHKERRQ(ierr);
  ierr = PetscOptionsGetEnum(((PetscObject)lg)->options,((PetscObject)lg)->prefix,"-lg_marker_type",PetscDrawMarkerTypes,(PetscEnum*)&markertype,&set);CHKERRQ(ierr);
  if (set) {
    ierr = PetscDrawLGSetUseMarkers(lg,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscDrawSetMarkerType(lg->win,markertype);CHKERRQ(ierr);
  }
  usemarkers = lg->use_markers;
  ierr = PetscOptionsGetBool(((PetscObject)lg)->options,((PetscObject)lg)->prefix,"-lg_use_markers",&usemarkers,&set);CHKERRQ(ierr);
  if (set) {ierr = PetscDrawLGSetUseMarkers(lg,usemarkers);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
