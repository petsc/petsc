
static char help[] = "Demonstrates use of color map\n";

#include <petscsys.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscDraw      draw;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  ierr = PetscDrawCreate(PETSC_COMM_SELF,0,"Title",0,0,256,256,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);

  ierr = PetscDrawStringBoxed(draw,.5,.5,PETSC_DRAW_BLUE,PETSC_DRAW_RED,"Greetings",NULL,NULL);CHKERRQ(ierr);

  ierr = PetscDrawStringBoxed(draw,.25,.75,PETSC_DRAW_BLUE,PETSC_DRAW_RED,"How\nare\nyou?",NULL,NULL);CHKERRQ(ierr);
  ierr = PetscDrawStringBoxed(draw,.25,.25,PETSC_DRAW_GREEN,PETSC_DRAW_RED,"Long line followed by a very\nshort line",NULL,NULL);CHKERRQ(ierr);
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   build:
     requires: x

   test:
     output_file: output/ex1_1.out

TEST*/
