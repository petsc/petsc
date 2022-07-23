
static char help[] = "Demonstrates use of color map\n";

#include <petscsys.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscDraw   draw;
  PetscMPIInt size,rank;
  int         x = 0,y = 0,width = 256,height = 256,i;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCall(PetscDrawCreate(PETSC_COMM_WORLD,0,"Title",x,y,width,height,&draw));
  PetscCall(PetscDrawSetFromOptions(draw));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  for (i=rank; i<height; i+=size) {
    PetscReal y = ((PetscReal)i)/(height-1);
    PetscCall(PetscDrawLine(draw,0.0,y,1.0,y,i%256));
  }
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
