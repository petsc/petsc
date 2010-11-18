
#include "../src/sys/draw/utils/lgimpl.h"
PetscClassId DRAWLG_CLASSID = 0;

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
PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGGetAxis(PetscDrawLG lg,PetscDrawAxis *axis)
{
  PetscFunctionBegin;
  if (lg && ((PetscObject)lg)->classid == PETSC_DRAW_CLASSID) {
    *axis = 0;
    PetscFunctionReturn(0);
  }
  PetscValidHeaderSpecific(lg,DRAWLG_CLASSID,1);
  PetscValidPointer(axis,2);
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
PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGGetDraw(PetscDrawLG lg,PetscDraw *draw)
{
  PetscFunctionBegin;
  PetscValidHeader(lg,1);
  PetscValidPointer(draw,2);
  if (((PetscObject)lg)->classid == PETSC_DRAW_CLASSID) {
    *draw = (PetscDraw)lg;
  } else {
    PetscValidHeaderSpecific(lg,DRAWLG_CLASSID,1);
    *draw = lg->win;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscDrawLGSPDraw" 
/*@
   PetscDrawLGSPDraw - Redraws a line graph.

   Not Collective,but ignored by all processors except processor 0 in PetscDrawLG

   Input Parameter:
.  lg - the line graph context

   Level: intermediate

.seealso: PetscDrawLGDraw(), PetscDrawSPDraw()

   Developer Notes: This code cheats and uses the fact that the LG and SP structs are the same

@*/
PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGSPDraw(PetscDrawLG lg,PetscDrawSP spin)
{
  PetscDrawLG    sp = (PetscDrawLG)spin;
  PetscReal      xmin,xmax,ymin,ymax;
  PetscErrorCode ierr;
  int            i,j,dim,nopts,rank;
  PetscDraw      draw = lg->win;

  PetscFunctionBegin;
  if (lg && ((PetscObject)lg)->classid == PETSC_DRAW_CLASSID) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(lg,DRAWLG_CLASSID,1);
  PetscValidHeaderSpecific(sp,DRAWSP_CLASSID,2); 

  xmin = PetscMin(lg->xmin,sp->xmin);
  ymin = PetscMin(lg->ymin,sp->ymin);
  xmax = PetscMax(lg->xmax,sp->xmax);
  ymax = PetscMax(lg->ymax,sp->ymax);

  ierr = PetscDrawClear(draw);CHKERRQ(ierr);
  ierr = PetscDrawAxisSetLimits(lg->axis,xmin,xmax,ymin,ymax);CHKERRQ(ierr);
  ierr = PetscDrawAxisDraw(lg->axis);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(((PetscObject)lg)->comm,&rank);CHKERRQ(ierr);
  if (!rank) {
  
    dim   = lg->dim;
    nopts = lg->nopts;
    for (i=0; i<dim; i++) {
      for (j=1; j<nopts; j++) {
        ierr = PetscDrawLine(draw,lg->x[(j-1)*dim+i],lg->y[(j-1)*dim+i],lg->x[j*dim+i],lg->y[j*dim+i],PETSC_DRAW_BLACK+i);CHKERRQ(ierr);
        if (lg->use_dots) {
          ierr = PetscDrawString(draw,lg->x[j*dim+i],lg->y[j*dim+i],PETSC_DRAW_RED,"x");CHKERRQ(ierr);
        }
      }
    }

    dim   = sp->dim;
    nopts = sp->nopts;
    for (i=0; i<dim; i++) {
      for (j=0; j<nopts; j++) {
	ierr = PetscDrawString(draw,sp->x[j*dim+i],sp->y[j*dim+i],PETSC_DRAW_RED,"x");CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscDrawFlush(lg->win);CHKERRQ(ierr);
  ierr = PetscDrawPause(lg->win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 


#undef __FUNCT__  
#define __FUNCT__ "PetscDrawLGCreate" 
/*@
    PetscDrawLGCreate - Creates a line graph data structure.

    Collective over PetscDraw

    Input Parameters:
+   draw - the window where the graph will be made.
-   dim - the number of curves which will be drawn

    Output Parameters:
.   outctx - the line graph context

    Level: intermediate

    Concepts: line graph^creating

.seealso:  PetscDrawLGDestroy()
@*/
PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGCreate(PetscDraw draw,int dim,PetscDrawLG *outctx)
{
  PetscErrorCode ierr;
  PetscBool      isnull;
  PetscObject    obj = (PetscObject)draw;
  PetscDrawLG    lg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidPointer(outctx,2);
  ierr = PetscTypeCompare(obj,PETSC_DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) {
    ierr = PetscDrawOpenNull(((PetscObject)obj)->comm,(PetscDraw*)outctx);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscHeaderCreate(lg,_p_DrawLG,int,DRAWLG_CLASSID,0,"PetscDrawLG",((PetscObject)obj)->comm,PetscDrawLGDestroy,0);CHKERRQ(ierr);
  lg->view    = 0;
  lg->destroy = 0;
  lg->nopts   = 0;
  lg->win     = draw;
  lg->dim     = dim;
  lg->xmin    = 1.e20;
  lg->ymin    = 1.e20;
  lg->xmax    = -1.e20;
  lg->ymax    = -1.e20;
  ierr = PetscMalloc2(dim*CHUNCKSIZE,PetscReal,&lg->x,dim*CHUNCKSIZE,PetscReal,&lg->y);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(lg,2*dim*CHUNCKSIZE*sizeof(PetscReal));CHKERRQ(ierr);
  lg->len     = dim*CHUNCKSIZE;
  lg->loc     = 0;
  lg->use_dots= PETSC_FALSE;
  ierr = PetscDrawAxisCreate(draw,&lg->axis);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(lg,lg->axis);CHKERRQ(ierr);
  *outctx = lg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawLGSetDimension" 
/*@
   PetscDrawLGSetDimension - Change the number of lines that are to be drawn.

   Logically Collective over PetscDrawLG

   Input Parameter:
+  lg - the line graph context.
-  dim - the number of curves.

   Level: intermediate

   Concepts: line graph^setting number of lines

@*/
PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGSetDimension(PetscDrawLG lg,PetscInt dim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (lg && ((PetscObject)lg)->classid == PETSC_DRAW_CLASSID) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(lg,DRAWLG_CLASSID,1);
  PetscValidLogicalCollectiveInt(lg,dim,2);
  if (lg->dim == dim) PetscFunctionReturn(0);

  ierr    = PetscFree2(lg->x,lg->y);CHKERRQ(ierr);
  lg->dim = dim;
  ierr    = PetscMalloc2(dim*CHUNCKSIZE,PetscReal,&lg->x,dim*CHUNCKSIZE,PetscReal,&lg->y);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(lg,2*dim*CHUNCKSIZE*sizeof(PetscReal));CHKERRQ(ierr);
  lg->len     = dim*CHUNCKSIZE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawLGReset" 
/*@
   PetscDrawLGReset - Clears line graph to allow for reuse with new data.

   Logically Collective over PetscDrawLG

   Input Parameter:
.  lg - the line graph context.

   Level: intermediate

   Concepts: line graph^restarting

@*/
PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGReset(PetscDrawLG lg)
{
  PetscFunctionBegin;
  if (lg && ((PetscObject)lg)->classid == PETSC_DRAW_CLASSID) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(lg,DRAWLG_CLASSID,1);
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

   Collective over PetscDrawLG

   Input Parameter:
.  lg - the line graph context

   Level: intermediate

.seealso:  PetscDrawLGCreate()
@*/
PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGDestroy(PetscDrawLG lg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!lg || ((PetscObject)lg)->classid != PETSC_DRAW_CLASSID) {
    PetscValidHeaderSpecific(lg,DRAWLG_CLASSID,1);
  }

  if (--((PetscObject)lg)->refct > 0) PetscFunctionReturn(0);
  if (lg && ((PetscObject)lg)->classid == PETSC_DRAW_CLASSID) {
    ierr = PetscObjectDestroy((PetscObject)lg);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscDrawAxisDestroy(lg->axis);CHKERRQ(ierr);
  ierr = PetscFree2(lg->x,lg->y);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(lg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "PetscDrawLGIndicateDataPoints" 
/*@
   PetscDrawLGIndicateDataPoints - Causes LG to draw a big dot for each data-point.

   Not Collective, but ignored by all processors except processor 0 in PetscDrawLG

   Input Parameters:
.  lg - the linegraph context

   Level: intermediate

   Concepts: line graph^showing points

@*/
PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGIndicateDataPoints(PetscDrawLG lg)
{
  PetscFunctionBegin;
  if (lg && ((PetscObject)lg)->classid == PETSC_DRAW_CLASSID) PetscFunctionReturn(0);

  lg->use_dots = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawLGDraw" 
/*@
   PetscDrawLGDraw - Redraws a line graph.

   Not Collective,but ignored by all processors except processor 0 in PetscDrawLG

   Input Parameter:
.  lg - the line graph context

   Level: intermediate

.seealso: PetscDrawSPDraw(), PetscDrawLGSPDraw()

@*/
PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGDraw(PetscDrawLG lg)
{
  PetscReal      xmin=lg->xmin,xmax=lg->xmax,ymin=lg->ymin,ymax=lg->ymax;
  PetscErrorCode ierr;
  int            i,j,dim = lg->dim,nopts = lg->nopts,rank;
  PetscDraw      draw = lg->win;

  PetscFunctionBegin;
  if (lg && ((PetscObject)lg)->classid == PETSC_DRAW_CLASSID) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(lg,DRAWLG_CLASSID,1);

  ierr = PetscDrawClear(draw);CHKERRQ(ierr);
  ierr = PetscDrawAxisSetLimits(lg->axis,xmin,xmax,ymin,ymax);CHKERRQ(ierr);
  ierr = PetscDrawAxisDraw(lg->axis);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(((PetscObject)lg)->comm,&rank);CHKERRQ(ierr);
  if (!rank) {
  
    for (i=0; i<dim; i++) {
      for (j=1; j<nopts; j++) {
        ierr = PetscDrawLine(draw,lg->x[(j-1)*dim+i],lg->y[(j-1)*dim+i],lg->x[j*dim+i],lg->y[j*dim+i],PETSC_DRAW_BLACK+i);CHKERRQ(ierr);
        if (lg->use_dots) {
          ierr = PetscDrawString(draw,lg->x[j*dim+i],lg->y[j*dim+i],PETSC_DRAW_RED,"x");CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = PetscDrawFlush(lg->win);CHKERRQ(ierr);
  ierr = PetscDrawPause(lg->win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawLGPrint"
/*@
  PetscDrawLGPrint - Prints a line graph.

  Not collective

  Input Parameter:
. lg - the line graph context

  Level: beginner

  Contributed by Matthew Knepley

.keywords:  draw, line, graph
@*/
PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGPrint(PetscDrawLG lg)
{
  PetscReal xmin=lg->xmin, xmax=lg->xmax, ymin=lg->ymin, ymax=lg->ymax;
  int       i, j, dim = lg->dim, nopts = lg->nopts;

  PetscFunctionBegin;
  if (lg && ((PetscObject)lg)->classid == PETSC_DRAW_CLASSID) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(lg, DRAWLG_CLASSID,1);
  if (nopts < 1)                  PetscFunctionReturn(0);
  if (xmin > xmax || ymin > ymax) PetscFunctionReturn(0);

  for(i = 0; i < dim; i++) {
    PetscPrintf(((PetscObject)lg)->comm, "Line %d>\n", i);
    for(j = 0; j < nopts; j++) {
      PetscPrintf(((PetscObject)lg)->comm, "  X: %G Y: %G\n", lg->x[j*dim+i], lg->y[j*dim+i]);
    }
  }
  PetscFunctionReturn(0);
}
