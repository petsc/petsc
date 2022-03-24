
static char help[] = "Tests repeatedly setting a window type.\n";

#include <petscsys.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscDraw draw;
  int       x = 0,y = 0,width = 300,height = 300;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  CHKERRQ(PetscDrawCreate(PETSC_COMM_WORLD,0,"Title",x,y,width,height,&draw));
#if defined(PETSC_HAVE_X)
  CHKERRQ(PetscDrawSetType(draw,"x"));
  CHKERRQ(PetscDrawSetType(draw,"null"));
  CHKERRQ(PetscDrawSetType(draw,"x"));
#else
  CHKERRQ(PetscDrawSetType(draw,"null"));
#endif
  CHKERRQ(PetscDrawSetFromOptions(draw));
  CHKERRQ(PetscDrawSetViewPort(draw,.25,.25,.75,.75));
  CHKERRQ(PetscDrawClear(draw));
  CHKERRQ(PetscDrawLine(draw,0.0,0.0,1.0,1.0,PETSC_DRAW_BLACK));
  CHKERRQ(PetscDrawString(draw,.2,.2,PETSC_DRAW_RED,"Some Text"));
  CHKERRQ(PetscDrawStringSetSize(draw,.5,.5));
  CHKERRQ(PetscDrawString(draw,.2,.2,PETSC_DRAW_BLUE,"Some Text"));
  CHKERRQ(PetscDrawFlush(draw));
  CHKERRQ(PetscSleep(2));
  CHKERRQ(PetscDrawResizeWindow(draw,600,600));
  CHKERRQ(PetscDrawClear(draw));
  CHKERRQ(PetscSleep(2));
  CHKERRQ(PetscDrawLine(draw,0.0,1.0,1.0,0.0,PETSC_DRAW_BLUE));
  CHKERRQ(PetscDrawFlush(draw));
  CHKERRQ(PetscSleep(2));
  CHKERRQ(PetscDrawDestroy(&draw));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: x

   test:
     output_file: output/ex1_1.out

TEST*/
