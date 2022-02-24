
static char help[] = "Demonstrates use of PetscDrawZoom()\n";

#if defined(PETSC_APPLE_FRAMEWORK)

#include <PETSc/petscsys.h>
#include <PETSc/petscdraw.h>
#else
#include <petscsys.h>
#include <petscdraw.h>
#endif

PetscErrorCode zoomfunction(PetscDraw draw,void *dummy)
{
  int         i;
  MPI_Comm    comm = PetscObjectComm((PetscObject)draw);
  PetscMPIInt size,rank;

  CHKERRMPI(MPI_Comm_size(comm,&size));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  for (i=rank; i<256; i+=size) {
    PetscReal y = ((PetscReal)i)/(256-1);
    CHKERRQ(PetscDrawLine(draw,0.0,y,1.0,y,i));
  }
  return 0;
}

int main(int argc,char **argv)
{
  int            x = 0,y = 0,width = 256,height = 256;
  PetscDraw      draw;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  CHKERRQ(PetscDrawCreate(PETSC_COMM_WORLD,NULL,"Title",x,y,width,height,&draw));
  CHKERRQ(PetscDrawSetFromOptions(draw));
  CHKERRQ(PetscDrawZoom(draw,zoomfunction,NULL));
  CHKERRQ(PetscDrawDestroy(&draw));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: x

   test:
     output_file: output/ex1_1.out

   test:
     suffix: db
     args: -draw_double_buffer 0
     output_file: output/ex1_1.out

   test:
     suffix: df
     args: -draw_fast
     output_file: output/ex1_1.out

   test:
     suffix: dv
     args: -draw_virtual
     output_file: output/ex1_1.out

TEST*/
