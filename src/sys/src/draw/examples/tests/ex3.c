/*$Id: ex3.c,v 1.35 2000/01/11 20:59:19 bsmith Exp bsmith $*/

static char help[] = "Plots a simple line graph\n";

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  PetscDraw           draw;
  PetscDrawLG         lg;
  PetscDrawAxis       axis;
  int            n = 20,i,ierr,x = 0,y = 0,width = 300,height = 300,nports = 1;
  PetscTruth     flg;
  char           *xlabel,*ylabel,*toplabel;
  double         xd,yd;
  PetscDrawViewPorts  *ports;

  xlabel = "X-axis Label";toplabel = "Top Label";ylabel = "Y-axis Label";

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-width",&width,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-height",&height,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-nolabels",&flg);CHKERRA(ierr); 
  if (flg) {
    xlabel = (char *)0; toplabel = (char *)0;
  }
  /* ierr = PetscDrawOpenX(PETSC_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRA(ierr);*/
  ierr = PetscDrawCreate(PETSC_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRA(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRA(ierr);
  
  ierr = PetscOptionsGetInt(PETSC_NULL,"-nports",&nports,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscDrawViewPortsCreate(draw,nports,&ports);CHKERRA(ierr);
  ierr = PetscDrawViewPortsSet(ports,0);CHKERRA(ierr);

  ierr = PetscDrawLGCreate(draw,1,&lg);CHKERRA(ierr);
  ierr = PetscDrawLGGetAxis(lg,&axis);CHKERRA(ierr);
  ierr = PetscDrawAxisSetColors(axis,PETSC_DRAW_BLACK,PETSC_DRAW_RED,PETSC_DRAW_BLUE);CHKERRA(ierr);
  ierr = PetscDrawAxisSetLabels(axis,toplabel,xlabel,ylabel);CHKERRA(ierr);

  for (i=0; i<n ; i++) {
    xd = (double)(i - 5); yd = xd*xd;
    ierr = PetscDrawLGAddPoint(lg,&xd,&yd);CHKERRA(ierr);
  }
  ierr = PetscDrawLGIndicateDataPoints(lg);CHKERRA(ierr);
  ierr = PetscDrawLGDraw(lg);CHKERRA(ierr);
  ierr = PetscDrawFlush(draw);CHKERRA(ierr);
  ierr = PetscSleep(2);CHKERRA(ierr);

  ierr = PetscDrawViewPortsDestroy(ports);CHKERRA(ierr);
  ierr = PetscDrawLGDestroy(lg);CHKERRA(ierr);
  ierr = PetscDrawDestroy(draw);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
