
static char help[] = "Tests repeatedly setting a window type.\n";

#include <petscsys.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscDraw draw;
  int       x = 0,y = 0,width = 300,height = 300;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCall(PetscDrawCreate(PETSC_COMM_WORLD,0,"Title",x,y,width,height,&draw));
#if defined(PETSC_HAVE_X)
  PetscCall(PetscDrawSetType(draw,"x"));
  PetscCall(PetscDrawSetType(draw,"null"));
  PetscCall(PetscDrawSetType(draw,"x"));
#else
  PetscCall(PetscDrawSetType(draw,"null"));
#endif
  PetscCall(PetscDrawSetFromOptions(draw));
  PetscCall(PetscDrawSetViewPort(draw,.25,.25,.75,.75));
  PetscCall(PetscDrawClear(draw));
  PetscCall(PetscDrawLine(draw,0.0,0.0,1.0,1.0,PETSC_DRAW_BLACK));
  PetscCall(PetscDrawString(draw,.2,.2,PETSC_DRAW_RED,"Some Text"));
  PetscCall(PetscDrawStringSetSize(draw,.5,.5));
  PetscCall(PetscDrawString(draw,.2,.2,PETSC_DRAW_BLUE,"Some Text"));
  PetscCall(PetscDrawFlush(draw));
  PetscCall(PetscSleep(2));
  PetscCall(PetscDrawResizeWindow(draw,600,600));
  PetscCall(PetscDrawClear(draw));
  PetscCall(PetscSleep(2));
  PetscCall(PetscDrawLine(draw,0.0,1.0,1.0,0.0,PETSC_DRAW_BLUE));
  PetscCall(PetscDrawFlush(draw));
  PetscCall(PetscSleep(2));
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
