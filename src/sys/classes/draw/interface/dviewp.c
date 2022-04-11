
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <petsc/private/drawimpl.h>  /*I "petscdraw.h" I*/

/*@
   PetscDrawSetViewPort - Sets the portion of the window (page) to which draw
   routines will write.

   Collective on PetscDraw

   Input Parameters:
+  xl - the horizontal coordinate of the lower left corner of the subwindow.
.  yl - the vertical coordinate of the lower left corner of the subwindow.
.  xr - the horizontal coordinate of the upper right corner of the subwindow.
.  yr - the vertical coordinate of the upper right corner of the subwindow.
-  draw - the drawing context

   Notes:
   These numbers must always be between 0.0 and 1.0.
   Lower left corner is (0,0).

   Level: advanced

@*/
PetscErrorCode  PetscDrawSetViewPort(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscCheck(xl >= 0.0 && xr <= 1.0 && yl >= 0.0 && yr <= 1.0 && xr > xl && yr > yl,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"ViewPort values must be >= 0 and <= 1: Instead %g %g %g %g",(double)xl,(double)yl,(double)xr,(double)yr);
  draw->port_xl = xl; draw->port_yl = yl;
  draw->port_xr = xr; draw->port_yr = yr;
  if (draw->ops->setviewport) {
    PetscCall((*draw->ops->setviewport)(draw,xl,yl,xr,yr));
  }
  PetscFunctionReturn(0);
}

/*@
   PetscDrawGetViewPort - Gets the portion of the window (page) to which draw
   routines will write.

   Collective on PetscDraw

   Input Parameter:
.  draw - the drawing context

   Output Parameters:
+  xl - the horizontal coordinate of the lower left corner of the subwindow.
.  yl - the vertical coordinate of the lower left corner of the subwindow.
.  xr - the horizontal coordinate of the upper right corner of the subwindow.
-  yr - the vertical coordinate of the upper right corner of the subwindow.

   Notes:
   These numbers must always be between 0.0 and 1.0.
   Lower left corner is (0,0).

   Level: advanced

@*/
PetscErrorCode  PetscDrawGetViewPort(PetscDraw draw,PetscReal *xl,PetscReal *yl,PetscReal *xr,PetscReal *yr)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidRealPointer(xl,2);
  PetscValidRealPointer(yl,3);
  PetscValidRealPointer(xr,4);
  PetscValidRealPointer(yr,5);
  *xl = draw->port_xl;
  *yl = draw->port_yl;
  *xr = draw->port_xr;
  *yr = draw->port_yr;
  PetscFunctionReturn(0);
}

/*@
   PetscDrawSplitViewPort - Splits a window shared by several processes into smaller
   view ports. One for each process.

   Collective on PetscDraw

   Input Parameter:
.  draw - the drawing context

   Level: advanced

.seealso: PetscDrawDivideViewPort(), PetscDrawSetViewPort()

@*/
PetscErrorCode  PetscDrawSplitViewPort(PetscDraw draw)
{
  PetscMPIInt    rank,size;
  PetscInt       n;
  PetscBool      isnull;
  PetscReal      xl,xr,yl,yr,h;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscCall(PetscDrawIsNull(draw,&isnull));
  if (isnull) PetscFunctionReturn(0);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)draw),&rank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)draw),&size));

  n = (PetscInt)(.1 + PetscSqrtReal((PetscReal)size));
  while (n*n < size) n++;

  h  = 1.0/n;
  xl = (rank % n)*h;
  xr = xl + h;
  yl = (rank / n)*h;
  yr = yl + h;

  PetscDrawCollectiveBegin(draw);
  PetscCall(PetscDrawLine(draw,xl,yl,xl,yr,PETSC_DRAW_BLACK));
  PetscCall(PetscDrawLine(draw,xl,yr,xr,yr,PETSC_DRAW_BLACK));
  PetscCall(PetscDrawLine(draw,xr,yr,xr,yl,PETSC_DRAW_BLACK));
  PetscCall(PetscDrawLine(draw,xr,yl,xl,yl,PETSC_DRAW_BLACK));
  PetscDrawCollectiveEnd(draw);
  PetscCall(PetscDrawFlush(draw));

  draw->port_xl = xl + .05*h;
  draw->port_xr = xr - .05*h;
  draw->port_yl = yl + .05*h;
  draw->port_yr = yr - .05*h;

  if (draw->ops->setviewport) PetscCall((*draw->ops->setviewport)(draw,xl,yl,xr,yr));
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawViewPortsCreate - Splits a window into smaller view ports. Each processor shares all the viewports.

   Collective on PetscDraw

   Input Parameters:
+  draw - the drawing context
-  nports - the number of ports

   Output Parameter:
.  ports - a PetscDrawViewPorts context (C structure)

   Options Database:
.  -draw_ports - display multiple fields in the same window with PetscDrawPorts() instead of in separate windows

   Level: advanced

.seealso: PetscDrawSplitViewPort(), PetscDrawSetViewPort(), PetscDrawViewPortsSet(), PetscDrawViewPortsDestroy()

@*/
PetscErrorCode  PetscDrawViewPortsCreate(PetscDraw draw,PetscInt nports,PetscDrawViewPorts **newports)
{
  PetscDrawViewPorts *ports;
  PetscInt           i,n;
  PetscBool          isnull;
  PetscMPIInt        rank;
  PetscReal          *xl,*xr,*yl,*yr,h;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscCheck(nports >= 1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE, "Number of divisions must be positive: %" PetscInt_FMT, nports);
  PetscValidPointer(newports,3);
  PetscCall(PetscDrawIsNull(draw,&isnull));
  if (isnull) {*newports = NULL; PetscFunctionReturn(0);}
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)draw),&rank));

  PetscCall(PetscNew(&ports)); *newports = ports;
  ports->draw = draw;
  ports->nports = nports;
  PetscCall(PetscObjectReference((PetscObject)draw));
  /* save previous drawport of window */
  PetscCall(PetscDrawGetViewPort(draw,&ports->port_xl,&ports->port_yl,&ports->port_xr,&ports->port_yr));

  n = (PetscInt)(.1 + PetscSqrtReal((PetscReal)nports));
  while (n*n < nports) n++;
  h = 1.0/n;

  PetscCall(PetscMalloc4(n*n,&xl,n*n,&xr,n*n,&yl,n*n,&yr));
  ports->xl = xl;
  ports->xr = xr;
  ports->yl = yl;
  ports->yr = yr;

  PetscCall(PetscDrawSetCoordinates(draw,0.0,0.0,1.0,1.0));
  PetscDrawCollectiveBegin(draw);
  for (i=0; i<n*n; i++) {
    xl[i] = (i % n)*h;
    xr[i] = xl[i] + h;
    yl[i] = (i / n)*h;
    yr[i] = yl[i] + h;

    if (rank == 0) {
      PetscCall(PetscDrawLine(draw,xl[i],yl[i],xl[i],yr[i],PETSC_DRAW_BLACK));
      PetscCall(PetscDrawLine(draw,xl[i],yr[i],xr[i],yr[i],PETSC_DRAW_BLACK));
      PetscCall(PetscDrawLine(draw,xr[i],yr[i],xr[i],yl[i],PETSC_DRAW_BLACK));
      PetscCall(PetscDrawLine(draw,xr[i],yl[i],xl[i],yl[i],PETSC_DRAW_BLACK));
    }

    xl[i] += .05*h;
    xr[i] -= .05*h;
    yl[i] += .05*h;
    yr[i] -= .05*h;
  }
  PetscDrawCollectiveEnd(draw);
  PetscCall(PetscDrawFlush(draw));
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawViewPortsCreateRect - Splits a window into smaller
       view ports. Each processor shares all the viewports. The number
       of views in the x- and y-directions is specified.

   Collective on PetscDraw

   Input Parameters:
+  draw - the drawing context
.  nx - the number of x divisions
-  ny - the number of y divisions

   Output Parameter:
.  ports - a PetscDrawViewPorts context (C structure)

   Level: advanced

.seealso: PetscDrawSplitViewPort(), PetscDrawSetViewPort(), PetscDrawViewPortsSet(), PetscDrawViewPortsDestroy()

@*/
PetscErrorCode  PetscDrawViewPortsCreateRect(PetscDraw draw,PetscInt nx,PetscInt ny,PetscDrawViewPorts **newports)
{
  PetscDrawViewPorts *ports;
  PetscReal          *xl,*xr,*yl,*yr,hx,hy;
  PetscInt           i,j,k,n;
  PetscBool          isnull;
  PetscMPIInt        rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscCheck(nx >= 1 && ny >= 1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE, "Number of divisions must be positive: %" PetscInt_FMT " x %" PetscInt_FMT, nx, ny);
  PetscValidPointer(newports,4);
  PetscCall(PetscDrawIsNull(draw,&isnull));
  if (isnull) {*newports = NULL; PetscFunctionReturn(0);}
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)draw),&rank));

  n  = nx*ny;
  hx = 1.0/nx;
  hy = 1.0/ny;
  PetscCall(PetscNew(&ports)); *newports = ports;
  ports->draw = draw;
  ports->nports = n;
  PetscCall(PetscObjectReference((PetscObject) draw));
  /* save previous drawport of window */
  PetscCall(PetscDrawGetViewPort(draw,&ports->port_xl,&ports->port_yl,&ports->port_xr,&ports->port_yr));

  PetscCall(PetscMalloc4(n,&xl,n,&xr,n,&yl,n,&yr));
  ports->xr = xr;
  ports->xl = xl;
  ports->yl = yl;
  ports->yr = yr;

  PetscCall(PetscDrawSetCoordinates(draw,0.0,0.0,1.0,1.0));
  PetscDrawCollectiveBegin(draw);
  for (i = 0; i < nx; i++) {
    for (j = 0; j < ny; j++) {
      k = j*nx+i;

      xl[k] = i*hx;
      xr[k] = xl[k] + hx;
      yl[k] = j*hy;
      yr[k] = yl[k] + hy;

      if (rank == 0) {
        PetscCall(PetscDrawLine(draw,xl[k],yl[k],xl[k],yr[k],PETSC_DRAW_BLACK));
        PetscCall(PetscDrawLine(draw,xl[k],yr[k],xr[k],yr[k],PETSC_DRAW_BLACK));
        PetscCall(PetscDrawLine(draw,xr[k],yr[k],xr[k],yl[k],PETSC_DRAW_BLACK));
        PetscCall(PetscDrawLine(draw,xr[k],yl[k],xl[k],yl[k],PETSC_DRAW_BLACK));
      }

      xl[k] += .05*hx;
      xr[k] -= .05*hx;
      yl[k] += .05*hy;
      yr[k] -= .05*hy;
    }
  }
  PetscDrawCollectiveEnd(draw);
  PetscCall(PetscDrawFlush(draw));
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawViewPortsDestroy - frees a PetscDrawViewPorts object

   Collective on PetscDraw inside PetscDrawViewPorts

   Input Parameter:
.  ports - the PetscDrawViewPorts object

   Level: advanced

.seealso: PetscDrawSplitViewPort(), PetscDrawSetViewPort(), PetscDrawViewPortsSet(), PetscDrawViewPortsCreate()

@*/
PetscErrorCode  PetscDrawViewPortsDestroy(PetscDrawViewPorts *ports)
{
  PetscFunctionBegin;
  if (!ports) PetscFunctionReturn(0);
  PetscValidPointer(ports,1);
  /* reset Drawport of Window back to previous value */
  PetscCall(PetscDrawSetViewPort(ports->draw,ports->port_xl,ports->port_yl,ports->port_xr,ports->port_yr));
  PetscCall(PetscDrawDestroy(&ports->draw));
  PetscCall(PetscFree4(ports->xl,ports->xr,ports->yl,ports->yr));
  PetscCall(PetscFree(ports));
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawViewPortsSet - sets a draw object to use a particular subport

   Logically Collective on PetscDraw inside PetscDrawViewPorts

   Input Parameters:
+  ports - the PetscDrawViewPorts object
-  port - the port number, from 0 to nports-1

   Level: advanced

.seealso: PetscDrawSplitViewPort(), PetscDrawSetViewPort(), PetscDrawViewPortsDestroy(), PetscDrawViewPortsCreate()

@*/
PetscErrorCode  PetscDrawViewPortsSet(PetscDrawViewPorts *ports,PetscInt port)
{
  PetscFunctionBegin;
  if (!ports) PetscFunctionReturn(0);
  PetscValidPointer(ports,1);
  PetscCheck(port >= 0 && (port <= ports->nports-1),PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Port is out of range requested %" PetscInt_FMT " from 0 to %" PetscInt_FMT,port,ports->nports-1);
  PetscCall(PetscDrawSetViewPort(ports->draw,ports->xl[port],ports->yl[port],ports->xr[port],ports->yr[port]));
  PetscFunctionReturn(0);
}
