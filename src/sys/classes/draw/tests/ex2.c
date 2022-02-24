
static char help[] = "Demonstrates use of color map\n";

#include <petscsys.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscDraw      draw;
  PetscMPIInt    size,rank;
  PetscErrorCode ierr;
  int            x = 0,y = 0,width = 256,height = 256,i;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  CHKERRQ(PetscDrawCreate(PETSC_COMM_WORLD,0,"Title",x,y,width,height,&draw));
  CHKERRQ(PetscDrawSetFromOptions(draw));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  for (i=rank; i<height; i+=size) {
    PetscReal y = ((PetscReal)i)/(height-1);
    CHKERRQ(PetscDrawLine(draw,0.0,y,1.0,y,i%256));
  }
  CHKERRQ(PetscDrawFlush(draw));
  CHKERRQ(PetscDrawPause(draw));
  CHKERRQ(PetscDrawSave(draw));
  CHKERRQ(PetscDrawDestroy(&draw));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: x

   test:
     output_file: output/ex1_1.out

TEST*/
