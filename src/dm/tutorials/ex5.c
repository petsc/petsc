
static char help[] = "Tests DMDAGetElements() and VecView() contour plotting for 2d DMDAs.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscInt         M = 10,N = 8,ne,nc,i;
  const PetscInt   *e;
  PetscErrorCode   ierr;
  PetscBool        flg = PETSC_FALSE;
  DM               da;
  PetscViewer      viewer;
  Vec              local,global;
  PetscScalar      value;
  DMBoundaryType   bx    = DM_BOUNDARY_NONE,by = DM_BOUNDARY_NONE;
  DMDAStencilType  stype = DMDA_STENCIL_BOX;
  PetscScalar      *lv;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,300,300,&viewer));

  /* Read options */
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-star_stencil",&flg,NULL));
  if (flg) stype = DMDA_STENCIL_STAR;

  /* Create distributed array and get vectors */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,bx,by,stype,M,N,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMCreateGlobalVector(da,&global));
  CHKERRQ(DMCreateLocalVector(da,&local));

  value = -3.0;
  CHKERRQ(VecSet(global,value));
  CHKERRQ(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  CHKERRQ(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));

  CHKERRQ(DMDASetElementType(da,DMDA_ELEMENT_P1));
  CHKERRQ(DMDAGetElements(da,&ne,&nc,&e));
  CHKERRQ(VecGetArray(local,&lv));
  for (i=0; i<ne; i++) {
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"i %D e[3*i] %D %D %D\n",i,e[3*i],e[3*i+1],e[3*i+2]));
    lv[e[3*i]] = i;
  }
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,stdout));
  CHKERRQ(VecRestoreArray(local,&lv));
  CHKERRQ(DMDARestoreElements(da,&ne,&nc,&e));

  CHKERRQ(DMLocalToGlobalBegin(da,local,ADD_VALUES,global));
  CHKERRQ(DMLocalToGlobalEnd(da,local,ADD_VALUES,global));

  CHKERRQ(DMView(da,viewer));
  CHKERRQ(VecView(global,viewer));

  /* Free memory */
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(VecDestroy(&local));
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(DMDestroy(&da));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     requires: x

   test:
     suffix: 2
     nsize: 2
     requires: x

TEST*/
