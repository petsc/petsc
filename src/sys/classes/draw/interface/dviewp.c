
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <petsc/private/drawimpl.h>  /*I "petscdraw.h" I*/

/*@
   PetscDrawSetViewPort - Sets the portion of the window (page) to which draw
   routines will write.

   Collective on PetscDraw

   Input Parameters:
+  xl,yl,xr,yr - upper right and lower left corners of subwindow
                 These numbers must always be between 0.0 and 1.0.
                 Lower left corner is (0,0).
-  draw - the drawing context

   Level: advanced


@*/
PetscErrorCode  PetscDrawSetViewPort(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (xl < 0.0 || xr > 1.0 || yl < 0.0 || yr > 1.0 || xr <= xl || yr <= yl) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"ViewPort values must be >= 0 and <= 1: Instead %g %g %g %g",(double)xl,(double)yl,(double)xr,(double)yr);
  draw->port_xl = xl; draw->port_yl = yl;
  draw->port_xr = xr; draw->port_yr = yr;
  if (draw->ops->setviewport) {
    ierr = (*draw->ops->setviewport)(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   PetscDrawGetViewPort - Gets the portion of the window (page) to which draw
   routines will write.

   Collective on PetscDraw

   Input Parameter:
.  draw - the drawing context

   Output Parameter:
.  xl,yl,xr,yr - upper right and lower left corners of subwindow
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
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       n;
  PetscBool      isnull;
  PetscReal      xl,xr,yl,yr,h;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)draw),&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)draw),&size);CHKERRMPI(ierr);

  n = (PetscInt)(.1 + PetscSqrtReal((PetscReal)size));
  while (n*n < size) n++;

  h  = 1.0/n;
  xl = (rank % n)*h;
  xr = xl + h;
  yl = (rank / n)*h;
  yr = yl + h;

  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,xl,yl,xl,yr,PETSC_DRAW_BLACK);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,xl,yr,xr,yr,PETSC_DRAW_BLACK);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,xr,yr,xr,yl,PETSC_DRAW_BLACK);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,xr,yl,xl,yl,PETSC_DRAW_BLACK);CHKERRQ(ierr);
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);

  draw->port_xl = xl + .05*h;
  draw->port_xr = xr - .05*h;
  draw->port_yl = yl + .05*h;
  draw->port_yr = yr - .05*h;

  if (draw->ops->setviewport) {
    ierr =  (*draw->ops->setviewport)(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  }
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
.  -draw_ports - display multiple fields in the same window with PetscDrawPorts instead of in separate windows

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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (nports < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE, "Number of divisions must be positive: %d", nports);
  PetscValidPointer(newports,3);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) {*newports = NULL; PetscFunctionReturn(0);}
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)draw),&rank);CHKERRMPI(ierr);

  ierr = PetscNew(&ports);CHKERRQ(ierr); *newports = ports;
  ports->draw = draw;
  ports->nports = nports;
  ierr = PetscObjectReference((PetscObject)draw);CHKERRQ(ierr);
  /* save previous drawport of window */
  ierr = PetscDrawGetViewPort(draw,&ports->port_xl,&ports->port_yl,&ports->port_xr,&ports->port_yr);CHKERRQ(ierr);

  n = (PetscInt)(.1 + PetscSqrtReal((PetscReal)nports));
  while (n*n < nports) n++;
  h = 1.0/n;

  ierr = PetscMalloc4(n*n,&xl,n*n,&xr,n*n,&yl,n*n,&yr);CHKERRQ(ierr);
  ports->xl = xl;
  ports->xr = xr;
  ports->yl = yl;
  ports->yr = yr;

  ierr = PetscDrawSetCoordinates(draw,0.0,0.0,1.0,1.0);CHKERRQ(ierr);
  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  for (i=0; i<n*n; i++) {
    xl[i] = (i % n)*h;
    xr[i] = xl[i] + h;
    yl[i] = (i / n)*h;
    yr[i] = yl[i] + h;

    if (!rank) {
      ierr = PetscDrawLine(draw,xl[i],yl[i],xl[i],yr[i],PETSC_DRAW_BLACK);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw,xl[i],yr[i],xr[i],yr[i],PETSC_DRAW_BLACK);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw,xr[i],yr[i],xr[i],yl[i],PETSC_DRAW_BLACK);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw,xr[i],yl[i],xl[i],yl[i],PETSC_DRAW_BLACK);CHKERRQ(ierr);
    }

    xl[i] += .05*h;
    xr[i] -= .05*h;
    yl[i] += .05*h;
    yr[i] -= .05*h;
  }
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if ((nx < 1) || (ny < 1)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE, "Number of divisions must be positive: %d x %d", nx, ny);
  PetscValidPointer(newports,3);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) {*newports = NULL; PetscFunctionReturn(0);}
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)draw),&rank);CHKERRMPI(ierr);

  n  = nx*ny;
  hx = 1.0/nx;
  hy = 1.0/ny;
  ierr = PetscNew(&ports);CHKERRQ(ierr); *newports = ports;
  ports->draw = draw;
  ports->nports = n;
  ierr = PetscObjectReference((PetscObject) draw);CHKERRQ(ierr);
  /* save previous drawport of window */
  ierr = PetscDrawGetViewPort(draw,&ports->port_xl,&ports->port_yl,&ports->port_xr,&ports->port_yr);CHKERRQ(ierr);

  ierr = PetscMalloc4(n,&xl,n,&xr,n,&yl,n,&yr);CHKERRQ(ierr);
  ports->xr = xr;
  ports->xl = xl;
  ports->yl = yl;
  ports->yr = yr;

  ierr = PetscDrawSetCoordinates(draw,0.0,0.0,1.0,1.0);CHKERRQ(ierr);
  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  for (i = 0; i < nx; i++) {
    for (j = 0; j < ny; j++) {
      k = j*nx+i;

      xl[k] = i*hx;
      xr[k] = xl[k] + hx;
      yl[k] = j*hy;
      yr[k] = yl[k] + hy;

      if (!rank) {
        ierr = PetscDrawLine(draw,xl[k],yl[k],xl[k],yr[k],PETSC_DRAW_BLACK);CHKERRQ(ierr);
        ierr = PetscDrawLine(draw,xl[k],yr[k],xr[k],yr[k],PETSC_DRAW_BLACK);CHKERRQ(ierr);
        ierr = PetscDrawLine(draw,xr[k],yr[k],xr[k],yl[k],PETSC_DRAW_BLACK);CHKERRQ(ierr);
        ierr = PetscDrawLine(draw,xr[k],yl[k],xl[k],yl[k],PETSC_DRAW_BLACK);CHKERRQ(ierr);
      }

      xl[k] += .05*hx;
      xr[k] -= .05*hx;
      yl[k] += .05*hy;
      yr[k] -= .05*hy;
    }
  }
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ports) PetscFunctionReturn(0);
  PetscValidPointer(ports,1);
  /* reset Drawport of Window back to previous value */
  ierr = PetscDrawSetViewPort(ports->draw,ports->port_xl,ports->port_yl,ports->port_xr,ports->port_yr);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&ports->draw);CHKERRQ(ierr);
  ierr = PetscFree4(ports->xl,ports->xr,ports->yl,ports->yr);CHKERRQ(ierr);
  ierr = PetscFree(ports);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawViewPortsSet - sets a draw object to use a particular subport

   Logically Collective on PetscDraw inside PetscDrawViewPorts

   Input Parameter:
+  ports - the PetscDrawViewPorts object
-  port - the port number, from 0 to nports-1

   Level: advanced

.seealso: PetscDrawSplitViewPort(), PetscDrawSetViewPort(), PetscDrawViewPortsDestroy(), PetscDrawViewPortsCreate()

@*/
PetscErrorCode  PetscDrawViewPortsSet(PetscDrawViewPorts *ports,PetscInt port)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ports) PetscFunctionReturn(0);
  PetscValidPointer(ports,1);
  if (port < 0 || port > ports->nports-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Port is out of range requested %d from 0 to %d\n",port,ports->nports-1);
  ierr = PetscDrawSetViewPort(ports->draw,ports->xl[port],ports->yl[port],ports->xr[port],ports->yr[port]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
