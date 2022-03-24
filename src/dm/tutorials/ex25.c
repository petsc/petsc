
static char help[] = "Takes a patch of a large DMDA vector to one process.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmpatch.h>
#include <petscsf.h>

typedef struct {
  PetscScalar x,y;
} Field;

int main(int argc,char **argv)
{
  Vec            xy,sxy;
  DM             da,sda = NULL;
  PetscSF        sf;
  PetscInt       m = 10, n = 10, dof = 2;
  MatStencil     lower = {0,3,2,0}, upper = {0,7,8,0}; /* These are in the order of the z, y, x, logical coordinates, the fourth entry is ignored */
  MPI_Comm       comm;
  PetscMPIInt    rank;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));

  /* create the large DMDA and set coordinates (which we will copy down to the small DA). */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,m,n,PETSC_DECIDE,PETSC_DECIDE,dof,1,0,0,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0));
  /* Just as a simple example we use the coordinates as the variables in the vectors we wish to examine. */
  CHKERRQ(DMGetCoordinates(da,&xy));
  /* The vector entries are displayed in the "natural" ordering on the two dimensional grid; interlaced x and y with with the x variable increasing more rapidly than the y */
  CHKERRQ(VecView(xy,0));

  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if (rank == 0) comm = MPI_COMM_SELF;
  else comm = MPI_COMM_NULL;

  CHKERRQ(DMPatchZoom(da,lower,upper,comm,&sda, NULL,&sf));
  if (rank == 0) {
    CHKERRQ(DMCreateGlobalVector(sda,&sxy));
  } else {
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,0,&sxy));
  }
  /*  A PetscSF can also be used as a VecScatter context */
  CHKERRQ(VecScatterBegin(sf,xy,sxy,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(sf,xy,sxy,INSERT_VALUES,SCATTER_FORWARD));
  /* Only rank == 0 has the entries of the patch, so run code only at that rank */
  if (rank == 0) {
    Field         **vars;
    DMDALocalInfo info;
    PetscInt      i,j;
    PetscScalar   sum = 0;

    /* The vector entries of the patch are displayed in the "natural" ordering on the two grid; interlaced x and y with with the x variable increasing more rapidly */
    CHKERRQ(VecView(sxy,PETSC_VIEWER_STDOUT_SELF));
    /* Compute some trivial statistic of the coordinates */
    CHKERRQ(DMDAGetLocalInfo(sda,&info));
    CHKERRQ(DMDAVecGetArray(sda,sxy,&vars));
    /* Loop over the patch of the entire domain */
    for (j=info.ys; j<info.ys+info.ym; j++) {
      for (i=info.xs; i<info.xs+info.xm; i++) {
        sum += vars[j][i].x;
      }
    }
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"The sum of the x coordinates is %g\n",(double)PetscRealPart(sum)));
    CHKERRQ(DMDAVecRestoreArray(sda,sxy,&vars));
  }

  CHKERRQ(VecDestroy(&sxy));
  CHKERRQ(PetscSFDestroy(&sf));
  CHKERRQ(DMDestroy(&sda));
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

   test:
     nsize: 4
     suffix: 2

TEST*/
