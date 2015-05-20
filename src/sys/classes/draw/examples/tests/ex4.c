
static char help[] = "Demonstrates use of PetscDrawZoom()\n";

#if defined(PETSC_APPLE_FRAMEWORK)
#include <PETSc/petscsys.h>
#include <PETSc/petscdraw.h>
#else
#include <petscsys.h>
#include <petscdraw.h>
#endif

#undef __FUNCT__
#define __FUNCT__ "zoomfunction"
PetscErrorCode zoomfunction(PetscDraw draw,void *dummy)
{
  PetscErrorCode ierr;
  int            i;

  for (i=0; i<256; i++) {
    ierr = PetscDrawLine(draw,0.0,((PetscReal)i)/256.,1.0,((PetscReal)i)/256.,i);CHKERRQ(ierr);
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscDraw      draw;
  int            x = 0,y = 0,width = 256,height = 256;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = PetscDrawCreate(PETSC_COMM_SELF,0,"Title",x,y,width,height,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);
  ierr = PetscDrawZoom(draw,zoomfunction,NULL);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}


