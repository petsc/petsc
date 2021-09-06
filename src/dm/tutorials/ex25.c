
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
  PetscErrorCode ierr;
  PetscInt       m = 10, n = 10, dof = 2;
  MatStencil     lower = {0,3,2,0}, upper = {0,7,8,0}; /* These are in the order of the z, y, x, logical coordinates, the fourth entry is ignored */
  MPI_Comm       comm;
  PetscMPIInt    rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* create the large DMDA and set coordinates (which we will copy down to the small DA). */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,m,n,PETSC_DECIDE,PETSC_DECIDE,dof,1,0,0,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
  /* Just as a simple example we use the coordinates as the variables in the vectors we wish to examine. */
  ierr = DMGetCoordinates(da,&xy);CHKERRQ(ierr);
  /* The vector entries are displayed in the "natural" ordering on the two dimensional grid; interlaced x and y with with the x variable increasing more rapidly than the y */
  ierr = VecView(xy,0);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  if (!rank) comm = MPI_COMM_SELF;
  else comm = MPI_COMM_NULL;

  ierr = DMPatchZoom(da,lower,upper,comm,&sda, NULL,&sf);CHKERRQ(ierr);
  if (!rank) {
    ierr = DMCreateGlobalVector(sda,&sxy);CHKERRQ(ierr);
  } else {
    ierr = VecCreateSeq(PETSC_COMM_SELF,0,&sxy);CHKERRQ(ierr);
  }
  /*  A PetscSF can also be used as a VecScatter context */
  ierr = VecScatterBegin(sf,xy,sxy,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(sf,xy,sxy,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* Only rank == 0 has the entries of the patch, so run code only at that rank */
  if (!rank) {
    Field         **vars;
    DMDALocalInfo info;
    PetscInt      i,j;
    PetscScalar   sum = 0;

    /* The vector entries of the patch are displayed in the "natural" ordering on the two grid; interlaced x and y with with the x variable increasing more rapidly */
    ierr = VecView(sxy,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    /* Compute some trivial statistic of the coordinates */
    ierr = DMDAGetLocalInfo(sda,&info);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(sda,sxy,&vars);CHKERRQ(ierr);
    /* Loop over the patch of the entire domain */
    for (j=info.ys; j<info.ys+info.ym; j++) {
      for (i=info.xs; i<info.xs+info.xm; i++) {
        sum += vars[j][i].x;
      }
    }
    ierr = PetscPrintf(PETSC_COMM_SELF,"The sum of the x coordinates is %g\n",(double)PetscRealPart(sum));CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(sda,sxy,&vars);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&sxy);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = DMDestroy(&sda);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

   test:
     nsize: 4
     suffix: 2

TEST*/
