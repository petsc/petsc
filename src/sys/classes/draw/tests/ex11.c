
static char help[] = "Demonstrates use of color map\n";

#include <petscsys.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscDraw draw;

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));

  PetscCall(PetscDrawCreate(PETSC_COMM_SELF,0,"Title",0,0,256,256,&draw));
  PetscCall(PetscDrawSetFromOptions(draw));

  PetscCall(PetscDrawStringBoxed(draw,.5,.5,PETSC_DRAW_BLUE,PETSC_DRAW_RED,"Greetings",NULL,NULL));

  PetscCall(PetscDrawStringBoxed(draw,.25,.75,PETSC_DRAW_BLUE,PETSC_DRAW_RED,"How\nare\nyou?",NULL,NULL));
  PetscCall(PetscDrawStringBoxed(draw,.25,.25,PETSC_DRAW_GREEN,PETSC_DRAW_RED,"Long line followed by a very\nshort line",NULL,NULL));
  PetscCall(PetscDrawFlush(draw));
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
