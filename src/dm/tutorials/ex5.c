
static char help[] = "Tests DMDAGetElements() and VecView() contour plotting for 2d DMDAs.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscInt         M = 10,N = 8,ne,nc,i;
  const PetscInt   *e;
  PetscBool        flg = PETSC_FALSE;
  DM               da;
  PetscViewer      viewer;
  Vec              local,global;
  PetscScalar      value;
  DMBoundaryType   bx    = DM_BOUNDARY_NONE,by = DM_BOUNDARY_NONE;
  DMDAStencilType  stype = DMDA_STENCIL_BOX;
  PetscScalar      *lv;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,300,300,&viewer));

  /* Read options */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-star_stencil",&flg,NULL));
  if (flg) stype = DMDA_STENCIL_STAR;

  /* Create distributed array and get vectors */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,bx,by,stype,M,N,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMCreateGlobalVector(da,&global));
  PetscCall(DMCreateLocalVector(da,&local));

  value = -3.0;
  PetscCall(VecSet(global,value));
  PetscCall(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  PetscCall(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));

  PetscCall(DMDASetElementType(da,DMDA_ELEMENT_P1));
  PetscCall(DMDAGetElements(da,&ne,&nc,&e));
  PetscCall(VecGetArray(local,&lv));
  for (i=0; i<ne; i++) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"i %" PetscInt_FMT " e[3*i] %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n",i,e[3*i],e[3*i+1],e[3*i+2]));
    lv[e[3*i]] = i;
  }
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,stdout));
  PetscCall(VecRestoreArray(local,&lv));
  PetscCall(DMDARestoreElements(da,&ne,&nc,&e));

  PetscCall(DMLocalToGlobalBegin(da,local,ADD_VALUES,global));
  PetscCall(DMLocalToGlobalEnd(da,local,ADD_VALUES,global));

  PetscCall(DMView(da,viewer));
  PetscCall(VecView(global,viewer));

  /* Free memory */
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&local));
  PetscCall(VecDestroy(&global));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     requires: x

   test:
     suffix: 2
     nsize: 2
     requires: x

TEST*/
