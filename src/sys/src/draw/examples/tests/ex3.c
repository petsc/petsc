/*$Id: ex3.c,v 1.36 2001/01/15 21:43:34 bsmith Exp bsmith $*/

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
  ierr = PetscOptionsGetInt(PETSC_NULL,"-width",&width,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-height",&height,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-nolabels",&flg);CHKERRQ(ierr); 
  if (flg) {
    xlabel = (char *)0; toplabel = (char *)0;
  }
  /* ierr = PetscDrawOpenX(PETSC_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRQ(ierr);*/
  ierr = PetscDrawCreate(PETSC_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);
  
  ierr = PetscOptionsGetInt(PETSC_NULL,"-nports",&nports,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscDrawViewPortsCreate(draw,nports,&ports);CHKERRQ(ierr);
  ierr = PetscDrawViewPortsSet(ports,0);CHKERRQ(ierr);

  ierr = PetscDrawLGCreate(draw,1,&lg);CHKERRQ(ierr);
  ierr = PetscDrawLGGetAxis(lg,&axis);CHKERRQ(ierr);
  ierr = PetscDrawAxisSetColors(axis,PETSC_DRAW_BLACK,PETSC_DRAW_RED,PETSC_DRAW_BLUE);CHKERRQ(ierr);
  ierr = PetscDrawAxisSetLabels(axis,toplabel,xlabel,ylabel);CHKERRQ(ierr);

  for (i=0; i<n ; i++) {
    xd = (double)(i - 5); yd = xd*xd;
    ierr = PetscDrawLGAddPoint(lg,&xd,&yd);CHKERRQ(ierr);
  }
  ierr = PetscDrawLGIndicateDataPoints(lg);CHKERRQ(ierr);
  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscSleep(2);CHKERRQ(ierr);

  ierr = PetscDrawViewPortsDestroy(ports);CHKERRQ(ierr);
  ierr = PetscDrawLGDestroy(lg);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(draw);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
 
