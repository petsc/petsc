
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <../src/sys/draw/drawimpl.h>  /*I "petscdraw.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSetViewPort" 
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

   Concepts: drawing^in subset of window
   Concepts: graphics^in subset of window

@*/
PetscErrorCode  PetscDrawSetViewPort(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (xl < 0.0 || xr > 1.0 || yl < 0.0 || yr > 1.0 || xr <= xl || yr <= yl) {
    SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"ViewPort values must be >= 0 and <= 1: Instead %G %G %G %G",xl,yl,xr,yr); 
  }
  draw->port_xl = xl; draw->port_yl = yl;
  draw->port_xr = xr; draw->port_yr = yr;
  if (draw->ops->setviewport) {
    ierr = (*draw->ops->setviewport)(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSplitViewPort" 
/*@
   PetscDrawSplitViewPort - Splits a window shared by several processes into smaller
   view ports. One for each process. 

   Collective on PetscDraw

   Input Parameter:
.  draw - the drawing context

   Level: advanced

   Concepts: drawing^in subset of window

.seealso: PetscDrawDivideViewPort(), PetscDrawSetViewPort()

@*/
PetscErrorCode  PetscDrawSplitViewPort(PetscDraw draw)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  int            n;
  PetscBool      isnull;
  PetscReal      xl,xr,yl,yr,h;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)draw,PETSC_DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  ierr = MPI_Comm_rank(((PetscObject)draw)->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)draw)->comm,&size);CHKERRQ(ierr);

  n = (int)(.1 + sqrt((double)size));
  while (n*n < size) {n++;}

  h  = 1.0/n;
  xl = (rank % n)*h;
  xr = xl + h;
  yl = (rank/n)*h;
  yr = yl + h;

  ierr = PetscDrawLine(draw,xl,yl,xl,yr,PETSC_DRAW_BLACK);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,xl,yr,xr,yr,PETSC_DRAW_BLACK);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,xr,yr,xr,yl,PETSC_DRAW_BLACK);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,xr,yl,xl,yl,PETSC_DRAW_BLACK);CHKERRQ(ierr);
  ierr = PetscDrawSynchronizedFlush(draw);CHKERRQ(ierr);

  draw->port_xl = xl + .1*h;
  draw->port_xr = xr - .1*h;
  draw->port_yl = yl + .1*h;
  draw->port_yr = yr - .1*h;

  if (draw->ops->setviewport) {
    ierr =  (*draw->ops->setviewport)(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawViewPortsCreate" 
/*@C
   PetscDrawViewPortsCreate - Splits a window into smaller
       view ports. Each processor shares all the viewports.

   Collective on PetscDraw

   Input Parameters:
+  draw - the drawing context
-  nports - the number of ports

   Output Parameter:
.  ports - a PetscDrawViewPorts context (C structure)

   Level: advanced

   Concepts: drawing^in subset of window

.seealso: PetscDrawSplitViewPort(), PetscDrawSetViewPort(), PetscDrawViewPortsSet(), PetscDrawViewPortsDestroy()

@*/
PetscErrorCode  PetscDrawViewPortsCreate(PetscDraw draw,PetscInt nports,PetscDrawViewPorts **ports)
{
  int        i,n;
  PetscErrorCode ierr;
  PetscBool  isnull;
  PetscReal  *xl,*xr,*yl,*yr,h;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidPointer(ports,3);
  ierr = PetscObjectTypeCompare((PetscObject)draw,PETSC_DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) {
    *ports = PETSC_NULL;
    PetscFunctionReturn(0);
  }

  ierr             = PetscNew(PetscDrawViewPorts,ports);CHKERRQ(ierr);
  (*ports)->draw   = draw;
  (*ports)->nports = nports;

  ierr = PetscObjectReference((PetscObject)draw);CHKERRQ(ierr);

  n = (int)(.1 + sqrt((double)nports));
  while (n*n < nports) {n++;}
  
  ierr = PetscMalloc(n*n*sizeof(PetscReal),&xl);CHKERRQ(ierr);(*ports)->xl = xl; 
  ierr = PetscMalloc(n*n*sizeof(PetscReal),&xr);CHKERRQ(ierr);(*ports)->xr = xr; 
  ierr = PetscMalloc(n*n*sizeof(PetscReal),&yl);CHKERRQ(ierr);(*ports)->yl = yl; 
  ierr = PetscMalloc(n*n*sizeof(PetscReal),&yr);CHKERRQ(ierr);(*ports)->yr = yr; 

  h  = 1.0/n;

  for (i=0; i<n*n; i++) {
    xl[i] = (i % n)*h;
    xr[i] = xl[i] + h;
    yl[i] = (i/n)*h;
    yr[i] = yl[i] + h;

    ierr = PetscDrawLine(draw,xl[i],yl[i],xl[i],yr[i],PETSC_DRAW_BLACK);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xl[i],yr[i],xr[i],yr[i],PETSC_DRAW_BLACK);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xr[i],yr[i],xr[i],yl[i],PETSC_DRAW_BLACK);CHKERRQ(ierr);
    ierr = PetscDrawLine(draw,xr[i],yl[i],xl[i],yl[i],PETSC_DRAW_BLACK);CHKERRQ(ierr);

    xl[i] += .1*h;
    xr[i] -= .1*h;
    yl[i] += .1*h;
    yr[i] -= .1*h;
  }
  ierr = PetscDrawSynchronizedFlush(draw);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawViewPortsCreateRect" 
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

   Concepts: drawing^in subset of window

.seealso: PetscDrawSplitViewPort(), PetscDrawSetViewPort(), PetscDrawViewPortsSet(), PetscDrawViewPortsDestroy()

@*/
PetscErrorCode  PetscDrawViewPortsCreateRect(PetscDraw draw,PetscInt nx,PetscInt ny,PetscDrawViewPorts **ports)
{
  PetscReal     *xl, *xr, *yl, *yr, hx, hy;
  PetscBool      isnull;
  PetscInt       i, j, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if ((nx < 1) || (ny < 1)) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE, "Number of divisions must be positive: %d x %d", nx, ny);
  }
  ierr = PetscObjectTypeCompare((PetscObject) draw, PETSC_DRAW_NULL, &isnull);CHKERRQ(ierr);
  if (isnull) {
    *ports = PETSC_NULL;
    PetscFunctionReturn(0);
  }
  n  = nx*ny;
  hx = 1.0/nx;
  hy = 1.0/ny;
  ierr = PetscNew(PetscDrawViewPorts, ports);CHKERRQ(ierr);
  (*ports)->draw   = draw;
  (*ports)->nports = n;
  ierr = PetscObjectReference((PetscObject) draw);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscReal), &xl);CHKERRQ(ierr);(*ports)->xl = xl; 
  ierr = PetscMalloc(n*sizeof(PetscReal), &xr);CHKERRQ(ierr);(*ports)->xr = xr; 
  ierr = PetscMalloc(n*sizeof(PetscReal), &yl);CHKERRQ(ierr);(*ports)->yl = yl; 
  ierr = PetscMalloc(n*sizeof(PetscReal), &yr);CHKERRQ(ierr);(*ports)->yr = yr; 
  for(i = 0; i < nx; i++) {
    for(j = 0; j < ny; j++) {
      PetscInt k = j*nx+i;

      xl[k] = i*hx;
      xr[k] = xl[k] + hx;
      yl[k] = j*hy;
      yr[k] = yl[k] + hy;

      ierr = PetscDrawLine(draw,xl[k],yl[k],xl[k],yr[k],PETSC_DRAW_BLACK);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw,xl[k],yr[k],xr[k],yr[k],PETSC_DRAW_BLACK);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw,xr[k],yr[k],xr[k],yl[k],PETSC_DRAW_BLACK);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw,xr[k],yl[k],xl[k],yl[k],PETSC_DRAW_BLACK);CHKERRQ(ierr);

      xl[k] += .01*hx;
      xr[k] -= .01*hx;
      yl[k] += .01*hy;
      yr[k] -= .01*hy;
    }
  }
  ierr = PetscDrawSynchronizedFlush(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawViewPortsDestroy" 
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
  ierr = PetscDrawDestroy(&ports->draw);CHKERRQ(ierr);
  ierr = PetscFree(ports->xl);CHKERRQ(ierr);
  ierr = PetscFree(ports->xr);CHKERRQ(ierr);
  ierr = PetscFree(ports->yl);CHKERRQ(ierr);
  ierr = PetscFree(ports->yr);CHKERRQ(ierr);
  ierr = PetscFree(ports);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawViewPortsSet" 
/*@C
   PetscDrawViewPortsSet - sets a draw object to use a particular subport

   Logically Collective on PetscDraw inside PetscDrawViewPorts

   Input Parameter:
+  ports - the PetscDrawViewPorts object
-  port - the port number, from 0 to nports-1

   Level: advanced

   Concepts: drawing^in subset of window

.seealso: PetscDrawSplitViewPort(), PetscDrawSetViewPort(), PetscDrawViewPortsDestroy(), PetscDrawViewPortsCreate()

@*/
PetscErrorCode  PetscDrawViewPortsSet(PetscDrawViewPorts *ports,PetscInt port)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ports) { 
    if (port < 0 || port > ports->nports-1) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Port is out of range requested %d from 0 to %d\n",port,ports->nports);
    }
    ierr = PetscDrawSetViewPort(ports->draw,ports->xl[port],ports->yl[port],ports->xr[port],ports->yr[port]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
