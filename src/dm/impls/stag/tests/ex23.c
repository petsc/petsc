static char help[] = "Test modifying DMStag coordinates, when represented as a product of 1d coordinate arrays\n\n";

#include <petscdm.h>
#include <petscdmstag.h>

int main(int argc,char **argv)
{
  DM             dm,cdm;
  PetscInt       ex,ey,ez,n[3],start[3],nExtra[3],iNext,iPrev,iCenter,d,round;
  PetscScalar    **cArrX,**cArrY,**cArrZ;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  PetscCall(DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_PERIODIC,4,3,2,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,1,1,DMSTAG_STENCIL_BOX,2,NULL,NULL,NULL,&dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm,-1.0,0.0,-2.0,0.0,-3.0,0.0));

  PetscCall(DMStagGetCorners(dm,&start[0],&start[1],&start[2],&n[0],&n[1],&n[2],&nExtra[0],&nExtra[1],&nExtra[2]));

  for (round=1; round<=2; ++round) {
    PetscCall(DMStagGetProductCoordinateArrays(dm,&cArrX,&cArrY,&cArrZ));
    PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_LEFT,&iPrev));
    PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_RIGHT,&iNext));
    PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&iCenter));
    if (round == 1) {
      /* On first round, do a stretching operation */
      for (ex=start[0]; ex<start[0]+n[0]; ++ex) {
        cArrX[ex][iPrev] *= 1.1;
        cArrX[ex][iNext] = cArrX[ex][iPrev] + 0.1;
        cArrX[ex][iCenter] = 0.5 * (cArrX[ex][iPrev] + cArrX[ex][iNext]);
      }
      for (ey=start[1]; ey<start[1]+n[1]; ++ey) {
        cArrY[ey][iPrev] *= 1.1;
        cArrY[ey][iNext] = cArrY[ey][iPrev] + 0.1;
        cArrY[ey][iCenter] = 0.5 * (cArrY[ey][iPrev] + cArrY[ey][iNext]);
      }
      for (ez=start[2]; ez<start[2]+n[2]; ++ez) {
        cArrZ[ez][iPrev] *= 1.1;
        cArrZ[ez][iNext] = cArrZ[ez][iPrev] + 0.1;
        cArrZ[ez][iCenter] = 0.5 * (cArrZ[ez][iPrev] + cArrZ[ez][iNext]);
      }
    } else {
      /* On second round, set everything to 2.0 */
      for (ex=start[0]; ex<start[0]+n[0]; ++ex) {
        cArrX[ex][iPrev]   = 2.0;
        cArrX[ex][iNext]   = 2.0;
        cArrX[ex][iCenter] = 2.0;
      }
      for (ey=start[1]; ey<start[1]+n[1]; ++ey) {
        cArrY[ey][iPrev]   = 2.0;
        cArrY[ey][iNext]   = 2.0;
        cArrY[ey][iCenter] = 2.0;
      }
      for (ez=start[2]; ez<start[2]+n[2]; ++ez) {
        cArrZ[ez][iPrev]   = 2.0;
        cArrZ[ez][iNext]   = 2.0;
        cArrZ[ez][iCenter] = 2.0;
      }
    }
    PetscCall(DMStagRestoreProductCoordinateArrays(dm,&cArrX,&cArrY,&cArrZ));

    /* View the global coordinates, after explicitly calling a local-global scatter */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"####### Round %D #######\n",round));
    PetscCall(DMGetCoordinateDM(dm,&cdm));
    for (d=0; d<3; ++d) {
      DM subdm;
      Vec coor,coor_local;

      PetscCall(DMProductGetDM(cdm,d,&subdm));
      PetscCall(DMGetCoordinates(subdm,&coor));
      PetscCall(DMGetCoordinatesLocal(subdm,&coor_local));
      PetscCall(DMLocalToGlobal(subdm,coor_local,INSERT_VALUES,coor));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Coordinates dim %D:\n",d));
      PetscCall(VecView(coor,PETSC_VIEWER_STDOUT_WORLD));
    }
  }

  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 1

   test:
      suffix: 2
      nsize: 2

TEST*/
