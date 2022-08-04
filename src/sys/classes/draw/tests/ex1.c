
static char help[] = "Demonstrates opening and drawing in a window\n";

#include <petscsys.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscDraw draw;
  int       x = 0,y = 0,width = 300,height = 300;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));

  PetscCall(PetscDrawCreate(PETSC_COMM_WORLD,0,"Title",x,y,width,height,&draw));
  PetscCall(PetscDrawSetPause(draw,2.0));
  PetscCall(PetscDrawSetFromOptions(draw));
  PetscCall(PetscDrawSetViewPort(draw,.25,.25,.75,.75));

  PetscCall(PetscDrawLine(draw,0.0,0.0,1.0,1.0,PETSC_DRAW_BLACK));
  PetscCall(PetscDrawString(draw,.2,.2,PETSC_DRAW_RED,"Some Text"));
  PetscCall(PetscDrawString(draw,.5,.5,PETSC_DRAW_GREEN,"Some Text"));
  PetscCall(PetscDrawString(draw,.2,.8,PETSC_DRAW_BLUE,"Some Text"));
  PetscCall(PetscDrawFlush(draw));
  PetscCall(PetscDrawPause(draw));
  PetscCall(PetscDrawSave(draw));

  PetscCall(PetscDrawClear(draw));
  /*PetscCall(PetscDrawStringSetSize(draw,.5,.5));*/
  PetscCall(PetscDrawString(draw,.2,.2,PETSC_DRAW_RED,"Some Text"));
  PetscCall(PetscDrawString(draw,.5,.5,PETSC_DRAW_GREEN,"Some Text"));
  PetscCall(PetscDrawString(draw,.2,.8,PETSC_DRAW_BLUE,"Some Text"));
  PetscCall(PetscDrawFlush(draw));
  PetscCall(PetscDrawPause(draw));
  PetscCall(PetscDrawSave(draw));

  PetscCall(PetscDrawResizeWindow(draw,600,600));
  PetscCall(PetscDrawGetWindowSize(draw,&width,&height));
  PetscCall(PetscDrawSetViewPort(draw,0,0,1,1));
  PetscCall(PetscDrawClear(draw));
  /*PetscCall(PetscDrawLine(draw,0.0,0.0,1.0,1.0,PETSC_DRAW_RED));*/
  /*PetscCall(PetscDrawLine(draw,0.0,1.0,1.0,0.0,PETSC_DRAW_BLUE));*/
  PetscCall(PetscDrawString(draw,.2,.2,PETSC_DRAW_RED,"Some Text\n  Some Other Text"));
  PetscCall(PetscDrawString(draw,.5,.5,PETSC_DRAW_RED,"ABCygj\n()[]F$"));
  PetscCall(PetscDrawString(draw,0,0,PETSC_DRAW_RED,"Horizontal Text (ABCygj)"));
  PetscCall(PetscDrawStringVertical(draw,0,1,PETSC_DRAW_RED,"Vertical Text"));
  PetscCall(PetscDrawFlush(draw));
  PetscCall(PetscDrawPause(draw));
  PetscCall(PetscDrawSave(draw));

  PetscCall(PetscDrawDestroy(&draw));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: x

   test:
     output_file: output/ex1_1.out

TEST*/
