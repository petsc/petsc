
static char help[] = "Plots a simple line graph.\n";

#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscDraw          draw;
  PetscDrawLG        lg;
  PetscDrawAxis      axis;
  PetscInt           n = 20,i,x = 0,y = 0,width = 300,height = 300,nports = 1;
  PetscBool          flg;
  const char         *xlabel,*ylabel,*toplabel;
  PetscReal          xd,yd;
  PetscDrawViewPorts *ports;
  PetscErrorCode     ierr;

  xlabel = "X-axis Label";toplabel = "Top Label";ylabel = "Y-axis Label";

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = PetscOptionsGetInt(PETSC_NULL,"-width",&width,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-height",&height,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-nolabels",&flg);CHKERRQ(ierr); 
  if (flg) {
    xlabel = (char *)0; toplabel = (char *)0;
  }
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
    xd = (PetscReal)(i - 5); yd = xd*xd;
    ierr = PetscDrawLGAddPoint(lg,&xd,&yd);CHKERRQ(ierr);
  }
  ierr = PetscDrawLGIndicateDataPoints(lg);CHKERRQ(ierr);
  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  ierr = PetscDrawString(draw,-3.,150.0,PETSC_DRAW_BLUE,"A legend");CHKERRQ(ierr);
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscSleep(2);CHKERRQ(ierr);

  ierr = PetscDrawViewPortsDestroy(ports);CHKERRQ(ierr);
  ierr = PetscDrawLGDestroy(&lg);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
 
