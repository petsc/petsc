/*$Id: dviewp.c,v 1.38 2000/04/12 04:21:02 bsmith Exp balay $*/
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DrawSetViewPort" 
/*@
   DrawSetViewPort - Sets the portion of the window (page) to which draw
   routines will write.

   Collective on Draw

   Input Parameters:
+  xl,yl,xr,yr - upper right and lower left corners of subwindow
                 These numbers must always be between 0.0 and 1.0.
                 Lower left corner is (0,0).
-  draw - the drawing context

   Level: advanced

.keywords:  draw, set, view, port
@*/
int DrawSetViewPort(Draw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (xl < 0.0 || xr > 1.0 || yl < 0.0 || yr > 1.0 || xr <= xl || yr <= yl) {
    SETERRQ4(PETSC_ERR_ARG_OUTOFRANGE,0,"ViewPort values must be >= 0 and <= 1: Instead %g %g %g %g",xl,yl,xr,yr); 
  }
  draw->port_xl = xl; draw->port_yl = yl;
  draw->port_xr = xr; draw->port_yr = yr;
  if (draw->ops->setviewport) {
    ierr = (*draw->ops->setviewport)(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DrawSplitViewPort" 
/*@
   DrawSplitViewPort - Splits a window shared by several processes into smaller
   view ports. One for each process. 

   Collective on Draw

   Input Parameter:
.  draw - the drawing context

   Level: advanced

.keywords:  draw, set, view, port, split

.seealso: DrawDivideViewPort(), DrawSetViewPort()

@*/
int DrawSplitViewPort(Draw draw)
{
  int        rank,size,n,ierr;
  PetscTruth isnull;
  PetscReal  xl,xr,yl,yr,h;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  ierr = PetscTypeCompare((PetscObject)draw,DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  ierr = MPI_Comm_rank(draw->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(draw->comm,&size);CHKERRQ(ierr);

  n = (int)(.1 + sqrt((double)size));
  while (n*n < size) {n++;}

  h  = 1.0/n;
  xl = (rank % n)*h;
  xr = xl + h;
  yl = (rank/n)*h;
  yr = yl + h;

  ierr = DrawLine(draw,xl,yl,xl,yr,DRAW_BLACK);CHKERRQ(ierr);
  ierr = DrawLine(draw,xl,yr,xr,yr,DRAW_BLACK);CHKERRQ(ierr);
  ierr = DrawLine(draw,xr,yr,xr,yl,DRAW_BLACK);CHKERRQ(ierr);
  ierr = DrawLine(draw,xr,yl,xl,yl,DRAW_BLACK);CHKERRQ(ierr);
  ierr = DrawSynchronizedFlush(draw);CHKERRQ(ierr);

  draw->port_xl = xl + .1*h;
  draw->port_xr = xr - .1*h;
  draw->port_yl = yl + .1*h;
  draw->port_yr = yr - .1*h;

  if (draw->ops->setviewport) {
    ierr =  (*draw->ops->setviewport)(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DrawViewPortsCreate" 
/*@C
   DrawViewPortsCreate - Splits a window into smaller
       view ports. Each processor shares all the viewports.

   Collective on Draw

   Input Parameter:
.  draw - the drawing context

   Output Parameter:
.  divide - a DrawViewPorts context (C structure)

   Level: advanced

.keywords:  draw, set, view, port, split

.seealso: DrawSplitViewPort(), DrawSetViewPort(), DrawViewPortsSet(), DrawViewPortsDestroy()

@*/
int DrawViewPortsCreate(Draw draw,int nports,DrawViewPorts **ports)
{
  int        i,ierr,n;
  PetscTruth isnull;
  PetscReal  *xl,*xr,*yl,*yr,h;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  ierr = PetscTypeCompare((PetscObject)draw,DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) {
    *ports = PETSC_NULL;
    PetscFunctionReturn(0);
  }

  *ports = PetscNew(DrawViewPorts);CHKPTRQ(*ports);
  (*ports)->draw   = draw;
  (*ports)->nports = nports;

  ierr = PetscObjectReference((PetscObject)draw);CHKERRQ(ierr);

  n = (int)(.1 + sqrt((double)nports));
  while (n*n < nports) {n++;}
  
  (*ports)->xl = xl = (PetscReal*)PetscMalloc(n*n*sizeof(PetscReal));CHKPTRQ(xl);
  (*ports)->xr = xr = (PetscReal*)PetscMalloc(n*n*sizeof(PetscReal));CHKPTRQ(xr);
  (*ports)->yl = yl = (PetscReal*)PetscMalloc(n*n*sizeof(PetscReal));CHKPTRQ(yl);
  (*ports)->yr = yr = (PetscReal*)PetscMalloc(n*n*sizeof(PetscReal));CHKPTRQ(yr);

  h  = 1.0/n;

  for (i=0; i<n*n; i++) {
    xl[i] = (i % n)*h;
    xr[i] = xl[i] + h;
    yl[i] = (i/n)*h;
    yr[i] = yl[i] + h;

    ierr = DrawLine(draw,xl[i],yl[i],xl[i],yr[i],DRAW_BLACK);CHKERRQ(ierr);
    ierr = DrawLine(draw,xl[i],yr[i],xr[i],yr[i],DRAW_BLACK);CHKERRQ(ierr);
    ierr = DrawLine(draw,xr[i],yr[i],xr[i],yl[i],DRAW_BLACK);CHKERRQ(ierr);
    ierr = DrawLine(draw,xr[i],yl[i],xl[i],yl[i],DRAW_BLACK);CHKERRQ(ierr);

    xl[i] += .1*h;
    xr[i] -= .1*h;
    yl[i] += .1*h;
    yr[i] -= .1*h;
  }
  ierr = DrawSynchronizedFlush(draw);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DrawViewPortsDestroy" 
/*@C
   DrawViewPortsDestroy - frees a DrawViewPorts object

   Collective on Draw inside DrawViewPorts

   Input Parameter:
.  ports - the DrawViewPorts object

   Level: advanced

.keywords:  draw, set, view, port, split

.seealso: DrawSplitViewPort(), DrawSetViewPort(), DrawViewPortsSet(), DrawViewPortsCreate()

@*/
int DrawViewPortsDestroy(DrawViewPorts *ports)
{
  int        ierr;

  PetscFunctionBegin;

  if (!ports) PetscFunctionReturn(0);  
  if (ports->draw) {ierr = DrawDestroy(ports->draw);CHKERRQ(ierr);}
  ierr = PetscFree(ports->xl);CHKERRQ(ierr);
  ierr = PetscFree(ports->xr);CHKERRQ(ierr);
  ierr = PetscFree(ports->yl);CHKERRQ(ierr);
  ierr = PetscFree(ports->yr);CHKERRQ(ierr);
  ierr = PetscFree(ports);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DrawViewPortsSet" 
/*@C
   DrawViewPortsSet - sets a draw object to use a particular subport

   Collective on Draw inside DrawViewPorts

   Input Parameter:
+  ports - the DrawViewPorts object
-  port - the port number, from 0 to nports-1

   Level: advanced

.keywords:  draw, set, view, port, split

.seealso: DrawSplitViewPort(), DrawSetViewPort(), DrawViewPortsDestroy(), DrawViewPortsCreate()

@*/
int DrawViewPortsSet(DrawViewPorts *ports,int port)
{
  int        ierr;

  PetscFunctionBegin;
  if (ports) { 
    if (port < 0 || port > ports->nports-1) {
      SETERRQ2(1,1,"Port is out of range requested %d from 0 to %d\n",port,ports->nports);
    }
    ierr = DrawSetViewPort(ports->draw,ports->xl[port],ports->yl[port],ports->xr[port],ports->yr[port]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
