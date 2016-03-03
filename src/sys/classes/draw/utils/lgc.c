
#include <petscviewer.h>
#include <../src/sys/classes/draw/utils/lgimpl.h>  /*I   "petscdraw.h"  I*/
PetscClassId PETSC_DRAWLG_CLASSID = 0;

#undef __FUNCT__
#define __FUNCT__ "PetscDrawLGGetAxis"
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

@*/
PetscErrorCode  PetscDrawLGGetAxis(PetscDrawLG lg,PetscDrawAxis *axis)
{
  PetscFunctionBegin;
  PetscValidPointer(axis,2);
  if (!lg) {*axis = NULL; PetscFunctionReturn(0);}
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  *axis = lg->axis;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawLGGetDraw"
/*@
   PetscDrawLGGetDraw - Gets the draw context associated with a line graph.

   Not Collective, if PetscDrawLG is parallel then PetscDraw is parallel

   Input Parameter:
.  lg - the line graph context

   Output Parameter:
.  draw - the draw context

   Level: intermediate

@*/
PetscErrorCode  PetscDrawLGGetDraw(PetscDrawLG lg,PetscDraw *draw)
{
  PetscFunctionBegin;
  PetscValidPointer(draw,2);
  if (!lg) {*draw = NULL; PetscFunctionReturn(0);}
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  *draw = lg->win;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscDrawLGSPDraw"
/*@
   PetscDrawLGSPDraw - Redraws a line graph.

   Collective on PetscDrawLG

   Input Parameter:
.  lg - the line graph context

   Level: intermediate

.seealso: PetscDrawLGDraw(), PetscDrawSPDraw()

   Developer Notes: This code cheats and uses the fact that the LG and SP structs are the same

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
  if (!lg || !spin) {PetscFunctionReturn(0);}
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  PetscValidHeaderSpecific(sp,PETSC_DRAWSP_CLASSID,2);
  ierr = PetscDrawIsNull(lg->win,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)lg),&rank);CHKERRQ(ierr);

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


#undef __FUNCT__
#define __FUNCT__ "PetscDrawLGCreate"
/*@
    PetscDrawLGCreate - Creates a line graph data structure.

    Collective on PetscDraw

    Input Parameters:
+   draw - the window where the graph will be made.
-   dim - the number of curves which will be drawn

    Output Parameters:
.   outlg - the line graph context

    Level: intermediate

    Concepts: line graph^creating

.seealso:  PetscDrawLGDestroy()
@*/
PetscErrorCode  PetscDrawLGCreate(PetscDraw draw,PetscInt dim,PetscDrawLG *outlg)
{
  PetscBool      isnull;
  PetscDrawLG    lg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidLogicalCollectiveInt(draw,dim,2);
  PetscValidPointer(outlg,3);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) {*outlg = NULL; PetscFunctionReturn(0);}

  ierr = PetscHeaderCreate(lg,PETSC_DRAWLG_CLASSID,"PetscDrawLG","Line graph","Draw",PetscObjectComm((PetscObject)draw),PetscDrawLGDestroy,NULL);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "PetscDrawLGSetColors"
/*@
   PetscDrawLGSetColors - Sets the color of each line graph drawn

   Logically Collective on PetscDrawLG

   Input Parameter:
+  lg - the line graph context.
-  colors - the colors

   Level: intermediate

   Concepts: line graph^setting number of lines

@*/
PetscErrorCode  PetscDrawLGSetColors(PetscDrawLG lg,const int *colors)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!lg) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);

  ierr = PetscFree(lg->colors);CHKERRQ(ierr);
  ierr = PetscMalloc1(lg->dim,&lg->colors);CHKERRQ(ierr);
  ierr = PetscMemcpy(lg->colors,colors,lg->dim*sizeof(int));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#undef __FUNCT__
#define __FUNCT__ "PetscDrawLGSetLegend"
/*@C
   PetscDrawLGSetLegend - sets the names of each curve plotted

   Logically Collective on PetscDrawLG

   Input Parameter:
+  lg - the line graph context.
-  names - the names for each curve

   Level: intermediate

   Concepts: line graph^setting number of lines

@*/
PetscErrorCode  PetscDrawLGSetLegend(PetscDrawLG lg,const char *const *names)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (!lg) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);

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

#undef __FUNCT__
#define __FUNCT__ "PetscDrawLGGetDimension"
/*@
   PetscDrawLGGetDimension - Change the number of lines that are to be drawn.

   Not Collective

   Input Parameter:
.  lg - the line graph context.

   Output Parameter:
.  dim - the number of curves.

   Level: intermediate

   Concepts: line graph^setting number of lines

@*/
PetscErrorCode  PetscDrawLGGetDimension(PetscDrawLG lg,PetscInt *dim)
{
  PetscFunctionBegin;
  if (!lg) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);

  *dim = lg->dim;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawLGSetDimension"
/*@
   PetscDrawLGSetDimension - Change the number of lines that are to be drawn.

   Logically Collective on PetscDrawLG

   Input Parameter:
+  lg - the line graph context.
-  dim - the number of curves.

   Level: intermediate

   Concepts: line graph^setting number of lines

@*/
PetscErrorCode  PetscDrawLGSetDimension(PetscDrawLG lg,PetscInt dim)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (!lg) PetscFunctionReturn(0);
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

#undef __FUNCT__
#define __FUNCT__ "PetscDrawLGReset"
/*@
   PetscDrawLGReset - Clears line graph to allow for reuse with new data.

   Logically Collective on PetscDrawLG

   Input Parameter:
.  lg - the line graph context.

   Level: intermediate

   Concepts: line graph^restarting

@*/
PetscErrorCode  PetscDrawLGReset(PetscDrawLG lg)
{
  PetscFunctionBegin;
  if (!lg) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  lg->xmin  = 1.e20;
  lg->ymin  = 1.e20;
  lg->xmax  = -1.e20;
  lg->ymax  = -1.e20;
  lg->loc   = 0;
  lg->nopts = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawLGDestroy"
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
#undef __FUNCT__
#define __FUNCT__ "PetscDrawLGSetUseMarkers"
/*@
   PetscDrawLGSetUseMarkers - Causes LG to draw a marker for each data-point.

   Logically Collective on PetscDrawLG

   Input Parameters:
+  lg - the linegraph context
-  flg - should mark each data point

   Options Database:
.  -lg_use_markers  <true,false>

   Level: intermediate

   Concepts: line graph^showing points

@*/
PetscErrorCode  PetscDrawLGSetUseMarkers(PetscDrawLG lg,PetscBool flg)
{
  PetscFunctionBegin;
  if (!lg) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);

  lg->use_markers = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawLGDraw"
/*@
   PetscDrawLGDraw - Redraws a line graph.

   Collective on PetscDrawLG

   Input Parameter:
.  lg - the line graph context

   Level: intermediate

.seealso: PetscDrawSPDraw(), PetscDrawLGSPDraw()

@*/
PetscErrorCode  PetscDrawLGDraw(PetscDrawLG lg)
{
  PetscReal      xmin,xmax,ymin,ymax;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscDraw      draw;
  PetscBool      isnull;

  PetscFunctionBegin;
  if (!lg) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  ierr = PetscDrawIsNull(lg->win,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)lg),&rank);CHKERRQ(ierr);

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
        if (lg->colors) cl = lg->colors[i];
        else cl = PETSC_DRAW_BLACK+i;
        ierr = PetscDrawLine(draw,lg->x[(j-1)*dim+i],lg->y[(j-1)*dim+i],lg->x[j*dim+i],lg->y[j*dim+i],cl);CHKERRQ(ierr);
        if (lg->use_markers) {
          ierr = PetscDrawMarker(draw,lg->x[j*dim+i],lg->y[j*dim+i],cl);CHKERRQ(ierr);
        }
      }
    }
  }
  if (!rank && lg->legend) {
    int       i,dim=lg->dim,cl;
    PetscReal xl,yl,xr,yr,tw,th;
    size_t    len,mlen = 0;
    ierr = PetscDrawGetCoordinates(draw,&xl,&yl,&xr,&yr);CHKERRQ(ierr);
    ierr = PetscDrawStringGetSize(draw,&tw,&th);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      ierr = PetscStrlen(lg->legend[i],&len);CHKERRQ(ierr);
      mlen = PetscMax(mlen,len);
    }
    ierr = PetscDrawLine(draw,xr - (mlen + 8)*tw,yr - 3*th,xr - 2*tw,yr - 3*th,PETSC_DRAW_BLACK);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xr - (mlen + 8)*tw,yr - 3*th,xr - (mlen + 8)*tw,yr - (4+lg->dim)*th,PETSC_DRAW_BLACK);CHKERRQ(ierr);
    for  (i=0; i<dim; i++) {
      cl   = (lg->colors ? lg->colors[i] : i + 1);
      ierr = PetscDrawLine(draw,xr - (mlen + 6.7)*tw,yr - (4 + i)*th,xr - (mlen + 3.2)*tw,yr - (4 + i)*th,cl);CHKERRQ(ierr);
      ierr = PetscDrawString(draw,xr - (mlen + 3)*tw,yr - (4.5 + i)*th,PETSC_DRAW_BLACK,lg->legend[i]);CHKERRQ(ierr);
    }
    ierr = PetscDrawLine(draw,xr - 2*tw,yr - 3*th,xr - 2*tw,yr - (4+lg->dim)*th,PETSC_DRAW_BLACK);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xr - (mlen + 8)*tw,yr - (4+lg->dim)*th,xr - 2*tw,yr - (4+lg->dim)*th,PETSC_DRAW_BLACK);CHKERRQ(ierr);
  }
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);

  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawLGSave"
/*@
  PetscDrawLGSave - Saves a drawn image

  Collective on PetscDrawLG

  Input Parameter:
. lg - The line graph context

  Level: intermediate

  Concepts: line graph^saving

.seealso:  PetscDrawLGCreate(), PetscDrawLGGetDraw(), PetscDrawSetSave(), PetscDrawSave()
@*/
PetscErrorCode  PetscDrawLGSave(PetscDrawLG lg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!lg) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  ierr = PetscDrawSave(lg->win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawLGView"
/*@
  PetscDrawLGView - Prints a line graph.

  Collective on PetscDrawLG

  Input Parameter:
. lg - the line graph context

  Level: beginner

.keywords:  draw, line, graph
@*/
PetscErrorCode  PetscDrawLGView(PetscDrawLG lg,PetscViewer viewer)
{
  PetscReal      xmin=lg->xmin, xmax=lg->xmax, ymin=lg->ymin, ymax=lg->ymax;
  PetscInt       i, j, dim = lg->dim, nopts = lg->nopts;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!lg) PetscFunctionReturn(0);
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

#undef __FUNCT__
#define __FUNCT__ "PetscDrawLGSetOptionsPrefix"
/*@C
   PetscDrawLGSetOptionsPrefix - Sets the prefix used for searching for all
   PetscDrawLG options in the database.

   Logically Collective on PetscDrawLG

   Input Parameter:
+  lg - the line graph context
-  prefix - the prefix to prepend to all option names

   Level: advanced

.keywords: PetscDrawLG, set, options, prefix, database

.seealso: PetscDrawLGSetFromOptions()
@*/
PetscErrorCode  PetscDrawLGSetOptionsPrefix(PetscDrawLG lg,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!lg) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)lg,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawLGSetFromOptions"
/*@
    PetscDrawLGSetFromOptions - Sets options related to the PetscDrawLG

    Collective on PetscDrawLG

    Options Database:

    Level: intermediate

    Concepts: line graph^creating

.seealso:  PetscDrawLGDestroy(), PetscDrawLGCreate()
@*/
PetscErrorCode  PetscDrawLGSetFromOptions(PetscDrawLG lg)
{
  PetscErrorCode ierr;
  PetscBool      flg=PETSC_FALSE, set;

  PetscFunctionBegin;
  if (!lg) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,1);

  ierr = PetscOptionsGetBool(((PetscObject)lg)->options,NULL,"-lg_use_markers",&flg,&set);CHKERRQ(ierr);
  if (set) {ierr = PetscDrawLGSetUseMarkers(lg,flg);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
