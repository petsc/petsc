static char help[] = "Test DMStagPopulateLocalToGlobalInjective.\n\n";

#include <petscdm.h>
#include <petscdmstag.h>

static PetscErrorCode Test1(DM dm);
static PetscErrorCode Test2_1d(DM dm);
static PetscErrorCode Test2_2d(DM dm);
static PetscErrorCode Test2_3d(DM dm);

int main(int argc,char **argv)
{
  DM             dm;
  PetscInt       dim;
  PetscBool      setSizes,useInjective;

  /* Initialize PETSc and process command line arguments */
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  dim = 2;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL));
  setSizes = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-setsizes",&setSizes,NULL));
  useInjective = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-useinjective",&useInjective,NULL));

  /* Creation (normal) */
  if (!setSizes) {
    switch (dim) {
      case 1:
        PetscCall(DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,3,1,1,DMSTAG_STENCIL_BOX,1,NULL,&dm));
        break;
      case 2:
        PetscCall(DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,3,2,PETSC_DECIDE,PETSC_DECIDE,1,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,&dm));
        break;
      case 3:
        PetscCall(DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,3,2,4,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dm));
        break;
      default:
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"No support for dimension %" PetscInt_FMT,dim);
    }
  } else {
    /* Creation (test providing decomp exactly)*/
    PetscMPIInt size;
    PetscInt lx[4] = {2,3,4}, ranksx = 3, mx = 9;
    PetscInt ly[3] = {4,5},   ranksy = 2, my = 9;
    PetscInt lz[2] = {6,7},   ranksz = 2, mz = 13;

    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
    switch (dim) {
      case 1:
        PetscCheck(size == ranksx,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Must run on %" PetscInt_FMT " ranks with -dim 1 -setSizes",ranksx);
        PetscCall(DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,mx,1,1,DMSTAG_STENCIL_BOX,1,lx,&dm));
        break;
      case 2:
        PetscCheckFalse(size != ranksx * ranksy,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Must run on %" PetscInt_FMT " ranks with -dim 2 -setSizes",ranksx * ranksy);
        PetscCall(DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,mx,my,ranksx,ranksy,1,1,1,DMSTAG_STENCIL_BOX,1,lx,ly,&dm));
        break;
      case 3:
        PetscCheckFalse(size != ranksx * ranksy * ranksz,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Must run on %" PetscInt_FMT " ranks with -dim 3 -setSizes", ranksx * ranksy * ranksz);
        PetscCall(DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,mx,my,mz,ranksx,ranksy,ranksz,1,1,1,1,DMSTAG_STENCIL_BOX,1,lx,ly,lz,&dm));
        break;
      default:
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"No support for dimension %" PetscInt_FMT,dim);
    }
  }

  /* Setup */
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));

  /* Populate Additional Injective Local-to-Global Map */
  if (useInjective) {
    PetscCall(DMStagPopulateLocalToGlobalInjective(dm));
  }

  /* Test: Make sure L2G inverts G2L */
  PetscCall(Test1(dm));

  /* Test: Make sure that G2L inverts L2G, on its domain */
  PetscCall(DMGetDimension(dm,&dim));
  switch (dim) {
    case 1: PetscCall(Test2_1d(dm)); break;
    case 2: PetscCall(Test2_2d(dm)); break;
    case 3: PetscCall(Test2_3d(dm)); break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented for dimension %" PetscInt_FMT,dim);
  }

  /* Clean up and finalize PETSc */
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

static PetscErrorCode Test1(DM dm)
{
  Vec            vecLocal,vecGlobal,vecGlobalCheck;
  PetscRandom    rctx;
  PetscBool      equal;

  PetscFunctionBeginUser;
  PetscCall(DMCreateLocalVector(dm,&vecLocal));
  PetscCall(DMCreateGlobalVector(dm,&vecGlobal));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
  PetscCall(VecSetRandom(vecGlobal,rctx));
  PetscCall(VecSetRandom(vecLocal,rctx)); /* garbage */
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(VecDuplicate(vecGlobal,&vecGlobalCheck));
  PetscCall(DMGlobalToLocal(dm,vecGlobal,INSERT_VALUES,vecLocal));
  PetscCall(DMLocalToGlobal(dm,vecLocal,INSERT_VALUES,vecGlobalCheck));
  PetscCall(VecEqual(vecGlobal,vecGlobalCheck,&equal));
  PetscCheck(equal,PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Check failed - vectors should be bitwise identical");
  PetscCall(VecDestroy(&vecLocal));
  PetscCall(VecDestroy(&vecGlobal));
  PetscCall(VecDestroy(&vecGlobalCheck));
  PetscFunctionReturn(0);
}

/* Test function with positive values for positive arguments */
#define TEST_FUNCTION(i,j,k,idx,c) (8.33 * i + 7.343 * j + 1.234 * idx + 99.011 * c)

/* Helper function to check */
static PetscErrorCode CompareValues(PetscInt i,PetscInt j, PetscInt k, PetscInt c, PetscScalar val, PetscScalar valRef)
{
  PetscFunctionBeginUser;
  if (val != valRef && PetscAbsScalar(val-valRef)/PetscAbsScalar(valRef) > 10*PETSC_MACHINE_EPSILON)
  {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"(%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ") Value %.17g does not match the expected %.17g",i,j,k,c,(double)PetscRealPart(val),(double)PetscRealPart(valRef));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode Test2_1d(DM dm)
{
  Vec            vecLocal,vecLocalCheck,vecGlobal;
  PetscInt       i,startx,nx,nExtrax,dof0,dof1,c,idxLeft,idxElement;
  PetscScalar    **arr;
  const PetscInt j=-1,k=-1;

  PetscFunctionBeginUser;
  PetscCall(DMCreateLocalVector(dm,&vecLocal));
  PetscCall(VecSet(vecLocal,-1.0));
  PetscCall(DMStagGetCorners(dm,&startx,NULL,NULL,&nx,NULL,NULL,&nExtrax,NULL,NULL));
  PetscCall(DMStagGetDOF(dm,&dof0,&dof1,NULL,NULL));
  PetscCall(DMStagVecGetArray(dm,vecLocal,&arr));
  if (dof0 > 0) {
    PetscCall(DMStagGetLocationSlot(dm,DMSTAG_LEFT,0,&idxLeft));
  }
  if (dof1 > 0) {
    PetscCall(DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,0,&idxElement));
  }
  for (i=startx; i<startx+nx+nExtrax; ++i) {
    for (c=0; c<dof0; ++c) {
      const PetscScalar valRef = TEST_FUNCTION(i,0,0,idxLeft,c);
      arr[i][idxLeft+c] = valRef;
    }
    if (i < startx+nx) {
      for (c=0; c<dof1; ++c) {
        const PetscScalar valRef = TEST_FUNCTION(i,0,0,idxElement,c);
        arr[i][idxElement+c] = valRef;
      }
    }
  }
  PetscCall(DMStagVecRestoreArray(dm,vecLocal,&arr));
  PetscCall(DMCreateGlobalVector(dm,&vecGlobal));
  PetscCall(DMLocalToGlobal(dm,vecLocal,INSERT_VALUES,vecGlobal));
  PetscCall(VecDuplicate(vecLocal,&vecLocalCheck));
  PetscCall(VecSet(vecLocalCheck,-1.0));
  PetscCall(DMGlobalToLocal(dm,vecGlobal,INSERT_VALUES,vecLocalCheck));
  PetscCall(DMStagVecGetArrayRead(dm,vecLocalCheck,&arr));
  for (i=startx; i<startx+nx+nExtrax; ++i) {
    for (c=0; c<dof0; ++c) {
      const PetscScalar valRef = TEST_FUNCTION(i,0,0,idxLeft,c);
      const PetscScalar val    = arr[i][idxLeft+c];
      PetscCall(CompareValues(i,j,k,c,val,valRef));
    }
    if (i < startx+nx) {
      for (c=0; c<dof1; ++c) {
        const PetscScalar valRef = TEST_FUNCTION(i,0,0,idxElement,c);
        const PetscScalar val    = arr[i][idxElement+c];
        PetscCall(CompareValues(i,j,k,c,val,valRef));
      }
    } else {
      for (c=0; c<dof1; ++c) {
        const PetscScalar valRef = -1.0;
        const PetscScalar val    = arr[i][idxElement+c];
        PetscCall(CompareValues(i,j,k,c,val,valRef));
      }
    }
  }
  PetscCall(DMStagVecRestoreArrayRead(dm,vecLocalCheck,&arr));
  PetscCall(VecDestroy(&vecLocal));
  PetscCall(VecDestroy(&vecLocalCheck));
  PetscCall(VecDestroy(&vecGlobal));
  PetscFunctionReturn(0);
}

static PetscErrorCode Test2_2d(DM dm)
{
  Vec            vecLocal,vecLocalCheck,vecGlobal;
  PetscInt       i,j,startx,starty,nx,ny,nExtrax,nExtray,dof0,dof1,dof2,c,idxLeft,idxDown,idxDownLeft,idxElement;
  PetscScalar    ***arr;
  const PetscInt k=-1;

  PetscFunctionBeginUser;
  PetscCall(DMCreateLocalVector(dm,&vecLocal));
  PetscCall(VecSet(vecLocal,-1.0));
  PetscCall(DMStagGetCorners(dm,&startx,&starty,NULL,&nx,&ny,NULL,&nExtrax,&nExtray,NULL));
  PetscCall(DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL));
  PetscCall(DMStagVecGetArray(dm,vecLocal,&arr));
  if (dof0 > 0) {
    PetscCall(DMStagGetLocationSlot(dm,DMSTAG_DOWN_LEFT,0,&idxDownLeft));
  }
  if (dof1 > 0) {
    PetscCall(DMStagGetLocationSlot(dm,DMSTAG_LEFT,0,&idxLeft));
    PetscCall(DMStagGetLocationSlot(dm,DMSTAG_DOWN,0,&idxDown));
  }
  if (dof2 > 0) {
    PetscCall(DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,0,&idxElement));
  }
  for (j=starty; j<starty+ny+nExtray; ++j) {
    for (i=startx; i<startx+nx+nExtrax; ++i) {
      for (c=0; c<dof0; ++c) {
        const PetscScalar valRef = TEST_FUNCTION(i,j,0,idxDownLeft,c);
        arr[j][i][idxDownLeft+c] = valRef;
      }
      if (j < starty+ny) {
        for (c=0; c<dof1; ++c) {
          const PetscScalar valRef = TEST_FUNCTION(i,j,0,idxLeft,c);
          arr[j][i][idxLeft+c] = valRef;
        }
      }
      if (i < startx+nx) {
        for (c=0; c<dof1; ++c) {
          const PetscScalar valRef = TEST_FUNCTION(i,j,0,idxDown,c);
          arr[j][i][idxDown+c] = valRef;
        }
      }
      if (i < startx+nx && j < starty+ny) {
        for (c=0; c<dof2; ++c) {
          const PetscScalar valRef = TEST_FUNCTION(i,j,0,idxElement,c);
          arr[j][i][idxElement+c] = valRef;
        }
      }
    }
  }
  PetscCall(DMStagVecRestoreArray(dm,vecLocal,&arr));
  PetscCall(DMCreateGlobalVector(dm,&vecGlobal));
  PetscCall(DMLocalToGlobal(dm,vecLocal,INSERT_VALUES,vecGlobal));
  PetscCall(VecDuplicate(vecLocal,&vecLocalCheck));
  PetscCall(VecSet(vecLocalCheck,-1.0));
  PetscCall(DMGlobalToLocal(dm,vecGlobal,INSERT_VALUES,vecLocalCheck));
  PetscCall(DMStagVecGetArrayRead(dm,vecLocalCheck,&arr));
  for (j=starty; j<starty+ny+nExtray; ++j) {
    for (i=startx; i<startx+nx+nExtrax; ++i) {
      for (c=0; c<dof0; ++c) {
        const PetscScalar valRef = TEST_FUNCTION(i,j,0,idxDownLeft,c);
        const PetscScalar val    = arr[j][i][idxDownLeft+c];
        PetscCall(CompareValues(i,j,k,c,val,valRef));
      }
      if (j < starty+ny) {
        for (c=0; c<dof1; ++c) {
          const PetscScalar valRef = TEST_FUNCTION(i,j,0,idxLeft,c);
          const PetscScalar val    = arr[j][i][idxLeft+c];
          PetscCall(CompareValues(i,j,k,c,val,valRef));
        }
      } else {
        for (c=0; c<dof1; ++c) {
          const PetscScalar valRef = -1.0;
          const PetscScalar val    = arr[j][i][idxLeft+c];
          PetscCall(CompareValues(i,j,k,c,val,valRef));
        }
      }
      if (i < startx+nx) {
        for (c=0; c<dof1; ++c) {
          const PetscScalar valRef = TEST_FUNCTION(i,j,0,idxDown,c);
          const PetscScalar val    = arr[j][i][idxDown+c];
          PetscCall(CompareValues(i,j,k,c,val,valRef));
        }
      } else {
        for (c=0; c<dof1; ++c) {
          const PetscScalar valRef = -1.0;
          const PetscScalar val    = arr[j][i][idxDown+c];
          PetscCall(CompareValues(i,j,k,c,val,valRef));
        }
      }
      if (i < startx+nx && j < starty+ny) {
        for (c=0; c<dof2; ++c) {
          const PetscScalar valRef = TEST_FUNCTION(i,j,0,idxElement,c);
          const PetscScalar val    = arr[j][i][idxElement+c];
          PetscCall(CompareValues(i,j,k,c,val,valRef));
        }
      } else {
        for (c=0; c<dof2; ++c) {
          const PetscScalar valRef = -1.0;
          const PetscScalar val    = arr[j][i][idxElement+c];
          PetscCall(CompareValues(i,j,k,c,val,valRef));
        }
      }
    }
  }
  PetscCall(DMStagVecRestoreArrayRead(dm,vecLocalCheck,&arr));
  PetscCall(VecDestroy(&vecLocal));
  PetscCall(VecDestroy(&vecLocalCheck));
  PetscCall(VecDestroy(&vecGlobal));
  PetscFunctionReturn(0);
}

static PetscErrorCode Test2_3d(DM dm)
{
  Vec            vecLocal,vecLocalCheck,vecGlobal;
  PetscInt       i,j,k,startx,starty,startz,nx,ny,nz,nExtrax,nExtray,nExtraz,dof0,dof1,dof2,dof3,c,idxLeft,idxDown,idxDownLeft,idxBackDownLeft,idxBackDown,idxBack,idxBackLeft,idxElement;
  PetscScalar    ****arr;

  PetscFunctionBeginUser;
  PetscCall(DMCreateLocalVector(dm,&vecLocal));
  PetscCall(VecSet(vecLocal,-1.0));
  PetscCall(DMStagGetCorners(dm,&startx,&starty,&startz,&nx,&ny,&nz,&nExtrax,&nExtray,&nExtraz));
  PetscCall(DMStagGetDOF(dm,&dof0,&dof1,&dof2,&dof3));
  PetscCall(DMStagVecGetArray(dm,vecLocal,&arr));
  if (dof0 > 0) {
    PetscCall(DMStagGetLocationSlot(dm,DMSTAG_BACK_DOWN_LEFT,0,&idxBackDownLeft));
  }
  if (dof1 > 0) {
    PetscCall(DMStagGetLocationSlot(dm,DMSTAG_BACK_LEFT,0,&idxBackLeft));
    PetscCall(DMStagGetLocationSlot(dm,DMSTAG_BACK_DOWN,0,&idxBackDown));
    PetscCall(DMStagGetLocationSlot(dm,DMSTAG_DOWN_LEFT,0,&idxDownLeft));
  }
  if (dof2 > 0) {
    PetscCall(DMStagGetLocationSlot(dm,DMSTAG_LEFT,0,&idxLeft));
    PetscCall(DMStagGetLocationSlot(dm,DMSTAG_DOWN,0,&idxDown));
    PetscCall(DMStagGetLocationSlot(dm,DMSTAG_BACK,0,&idxBack));
  }
  if (dof3 > 0) {
    PetscCall(DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,0,&idxElement));
  }
  for (k=startz; k<startz+nz+nExtraz; ++k) {
    for (j=starty; j<starty+ny+nExtray; ++j) {
      for (i=startx; i<startx+nx+nExtrax; ++i) {
        for (c=0; c<dof0; ++c) {
          const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxBackDownLeft,c);
          arr[k][j][i][idxBackDownLeft+c] = valRef;
        }
        if (k < startz+nz) {
          for (c=0; c<dof1; ++c) {
            const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxDownLeft,c);
            arr[k][j][i][idxDownLeft+c] = valRef;
          }
        }
        if (j < starty+ny) {
          for (c=0; c<dof1; ++c) {
            const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxBackLeft,c);
            arr[k][j][i][idxBackLeft+c] = valRef;
          }
        }
        if (i < startx+nx) {
          for (c=0; c<dof1; ++c) {
            const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxBackDown,c);
            arr[k][j][i][idxBackDown+c] = valRef;
          }
        }
        if (j < starty+ny && k < startz+nz) {
          for (c=0; c<dof2; ++c) {
            const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxLeft,c);
            arr[k][j][i][idxLeft+c] = valRef;
          }
        }
        if (i < startx+nx && k < startz+nz) {
          for (c=0; c<dof2; ++c) {
            const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxDown,c);
            arr[k][j][i][idxDown+c] = valRef;
          }
        }
        if (i < startx+nx && j < starty+ny) {
          for (c=0; c<dof2; ++c) {
            const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxBack,c);
            arr[k][j][i][idxBack+c] = valRef;
          }
        }
        if (i < startx+nx && j < starty+ny && k < startz+nz) {
          for (c=0; c<dof3; ++c) {
            const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxElement,c);
            arr[k][j][i][idxElement+c] = valRef;
          }
        }
      }
    }
  }
  PetscCall(DMStagVecRestoreArray(dm,vecLocal,&arr));
  PetscCall(DMCreateGlobalVector(dm,&vecGlobal));
  PetscCall(DMLocalToGlobal(dm,vecLocal,INSERT_VALUES,vecGlobal));
  PetscCall(VecDuplicate(vecLocal,&vecLocalCheck));
  PetscCall(VecSet(vecLocalCheck,-1.0));
  PetscCall(DMGlobalToLocal(dm,vecGlobal,INSERT_VALUES,vecLocalCheck));
  PetscCall(DMStagVecGetArrayRead(dm,vecLocalCheck,&arr));
  for (k=startz; k<startz+nz+nExtraz; ++k) {
    for (j=starty; j<starty+ny+nExtray; ++j) {
      for (i=startx; i<startx+nx+nExtrax; ++i) {
        for (c=0; c<dof0; ++c) {
          const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxBackDownLeft,c);
          const PetscScalar val    = arr[k][j][i][idxBackDownLeft+c];
          PetscCall(CompareValues(i,j,k,c,val,valRef));
        }
        if (k < startz+nz) {
          for (c=0; c<dof1; ++c) {
            const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxDownLeft,c);
            const PetscScalar val    =  arr[k][j][i][idxDownLeft+c];
            PetscCall(CompareValues(i,j,k,c,val,valRef));
          }
        } else {
          for (c=0; c<dof1; ++c) {
            const PetscScalar valRef = -1.0;
            const PetscScalar val    = arr[k][j][i][idxDownLeft+c];
            PetscCall(CompareValues(i,j,k,c,val,valRef));
          }
        }
        if (j < starty+ny) {
          for (c=0; c<dof1; ++c) {
            const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxBackLeft,c);
            const PetscScalar val    = arr[k][j][i][idxBackLeft+c];
            PetscCall(CompareValues(i,j,k,c,val,valRef));
          }
        } else {
          for (c=0; c<dof1; ++c) {
            const PetscScalar valRef = -1.0;
            const PetscScalar val    = arr[k][j][i][idxBackLeft+c];
            PetscCall(CompareValues(i,j,k,c,val,valRef));
          }
        }
        if (i < startx+nx) {
          for (c=0; c<dof1; ++c) {
            const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxBackDown,c);
            const PetscScalar val    = arr[k][j][i][idxBackDown+c];
            PetscCall(CompareValues(i,j,k,c,val,valRef));
          }
        } else {
          for (c=0; c<dof1; ++c) {
            const PetscScalar valRef = -1.0;
            const PetscScalar val    = arr[k][j][i][idxBackDown+c];
            PetscCall(CompareValues(i,j,k,c,val,valRef));
          }
        }
        if (j < starty+ny && k < startz+nz) {
          for (c=0; c<dof2; ++c) {
            const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxLeft,c);
            const PetscScalar val    = arr[k][j][i][idxLeft+c];
            PetscCall(CompareValues(i,j,k,c,val,valRef));
          }
        } else {
          for (c=0; c<dof2; ++c) {
            const PetscScalar valRef = -1.0;
            const PetscScalar val    = arr[k][j][i][idxLeft+c];
            PetscCall(CompareValues(i,j,k,c,val,valRef));
          }
        }
        if (i < startx+nx && k < startz+nz) {
          for (c=0; c<dof2; ++c) {
            const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxDown,c);
            const PetscScalar val    = arr[k][j][i][idxDown+c];
            PetscCall(CompareValues(i,j,k,c,val,valRef));
          }
        } else {
          for (c=0; c<dof2; ++c) {
            const PetscScalar valRef = -1.0;
            const PetscScalar val    = arr[k][j][i][idxDown+c];
            PetscCall(CompareValues(i,j,k,c,val,valRef));
          }
        }
        if (i < startx+nx && j < starty+ny) {
          for (c=0; c<dof2; ++c) {
            const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxBack,c);
            const PetscScalar val    = arr[k][j][i][idxBack+c];
            PetscCall(CompareValues(i,j,k,c,val,valRef));
          }
        } else {
          for (c=0; c<dof2; ++c) {
            const PetscScalar valRef = -1.0;
            const PetscScalar val    = arr[k][j][i][idxBack+c];
            PetscCall(CompareValues(i,j,k,c,val,valRef));
          }
        }
        if (i < startx+nx && j < starty+ny && k < startz+nz) {
          for (c=0; c<dof3; ++c) {
            const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxElement,c);
            const PetscScalar val    = arr[k][j][i][idxElement+c];
            PetscCall(CompareValues(i,j,k,c,val,valRef));
          }
        } else {
          for (c=0; c<dof3; ++c) {
            const PetscScalar valRef = -1.0;
            const PetscScalar val    = arr[k][j][i][idxElement+c];
            PetscCall(CompareValues(i,j,k,c,val,valRef));
          }
        }
      }
    }
  }
  PetscCall(DMStagVecRestoreArrayRead(dm,vecLocalCheck,&arr));
  PetscCall(VecDestroy(&vecLocal));
  PetscCall(VecDestroy(&vecLocalCheck));
  PetscCall(VecDestroy(&vecGlobal));
  PetscFunctionReturn(0);
}
#undef TEST_FUNCTION

/*TEST

   testset:
      suffix: periodic_1d_seq
      nsize: 1
      args:  -dm_view -dim 1 -stag_grid_x 4 -stag_boundary_type_x periodic -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: ghosted_1d_seq
      nsize: 1
      args:  -dm_view -dim 1 -stag_grid_x 4 -stag_boundary_type_x ghosted -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: none_1d_seq
      nsize: 1
      args:  -dm_view -dim 1 -stag_grid_x 4 -stag_boundary_type_x ghosted -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: periodic_1d_par
      nsize: 3
      args:  -dm_view -dim 1 -setsizes -stag_boundary_type_x periodic -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: ghosted_1d_par
      nsize: 3
      args:  -dm_view -dim 1 -setsizes -stag_boundary_type_x ghosted -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: none_1d_par
      nsize: 3
      args:  -dm_view -dim 1 -setsizes -stag_boundary_type_x ghosted -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: periodic_periodic_2d_seq
      nsize: 1
      args:  -dm_view -dim 2 -stag_grid_x 4 -stag_grid_y 5 -stag_boundary_type_x periodic -stag_boundary_type_y periodic -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: periodic_ghosted_2d_seq
      nsize: 1
      args:  -dm_view -dim 2 -stag_grid_x 4 -stag_grid_y 5 -stag_boundary_type_x periodic -stag_boundary_type_y ghosted -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: none_none_2d_seq
      nsize: 1
      args:  -dm_view -dim 2 -stag_grid_x 4 -stag_grid_y 5 -stag_boundary_type_x none -stag_boundary_type_y none -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: none_ghosted_2d_seq
      nsize: 1
      args:  -dm_view -dim 2 -stag_grid_x 4 -stag_grid_y 5 -stag_boundary_type_x none -stag_boundary_type_y ghosted -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: none_periodic_2d_seq
      nsize: 1
      args:  -dm_view -dim 2 -stag_grid_x 4 -stag_grid_y 5 -stag_boundary_type_x none -stag_boundary_type_y periodic -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: periodic_periodic_2d_par
      nsize: 6
      args:  -dm_view -dim 2 -setsizes -stag_boundary_type_x periodic -stag_boundary_type_y periodic -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: periodic_ghosted_2d_par
      nsize: 6
      args:  -dm_view -dim 2 -setsizes -stag_boundary_type_x periodic -stag_boundary_type_y ghosted -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: none_none_2d_par
      nsize: 6
      args:  -dm_view -dim 2 -setsizes -stag_boundary_type_x none -stag_boundary_type_y none -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: none_ghosted_2d_par
      nsize: 6
      args:  -dm_view -dim 2 -setsizes -stag_boundary_type_x none -stag_boundary_type_y ghosted -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: none_periodic_2d_par
      nsize: 6
      args:  -dm_view -dim 2 -setsizes -stag_boundary_type_x none -stag_boundary_type_y periodic -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: periodic_periodic_periodic_3d_seq
      nsize: 1
      args:  -dm_view -dim 3 -stag_grid_x 4 -stag_grid_y 5 -stag_grid_z 3 -stag_boundary_type_x periodic -stag_boundary_type_y periodic -stag_boundary_type_z periodic -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: periodic_ghosted_periodic_3d_seq
      nsize: 1
      args:  -dm_view -dim 3 -stag_grid_x 4 -stag_grid_y 5 -stag_grid_z 3 -stag_boundary_type_x periodic -stag_boundary_type_y ghosted -stag_boundary_type_z periodic -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: none_periodic_ghosted_3d_seq
      nsize: 1
      args:  -dm_view -dim 3 -stag_grid_x 4 -stag_grid_y 5 -stag_grid_z 3 -stag_boundary_type_x none -stag_boundary_type_y periodic -stag_boundary_type_z ghosted -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: none_none_none_3d_seq
      nsize: 1
      args:  -dm_view -dim 3 -stag_grid_x 4 -stag_grid_y 5 -stag_grid_z 3 -stag_boundary_type_x none -stag_boundary_type_y none -stag_boundary_type_z none -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: periodic_periodic_periodic_3d_par
      nsize: 12
      args:  -dm_view -dim 3 -setsizes -stag_boundary_type_x periodic -stag_boundary_type_y periodic -stag_boundary_type_z periodic -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: periodic_ghosted_ghosted_3d_par
      nsize: 12
      args:  -dm_view -dim 3 -setsizes -stag_boundary_type_x periodic -stag_boundary_type_y ghosted -stag_boundary_type_z ghosted -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: ghosted_periodic_periodic_3d_par
      nsize: 12
      args:  -dm_view -dim 3 -setsizes -stag_boundary_type_x ghosted -stag_boundary_type_y periodic -stag_boundary_type_z periodic -stag_stencil_width {{0 1 2}separate output}

   testset:
      suffix: none_none_none_3d_par
      nsize: 12
      args:  -dm_view -dim 3 -setsizes -stag_boundary_type_x none -stag_boundary_type_y none -stag_boundary_type_z none -stag_stencil_width {{0 1 2}separate output}

   test:
      suffix: periodic_none_none_3d_skinny_seq
      nsize: 1
      args:  -dm_view -dim 3 -stag_boundary_type_x periodic -stag_boundary_type_y none -stag_boundary_type_z none -stag_grid_x 3 -stag_grid_y 6 -stag_grid_z 5 -stag_stencil_width 1 -useinjective 0

   test:
      suffix: periodic_none_none_3d_skinny_par
      nsize: 4
      args:  -dm_view -dim 3 -stag_boundary_type_x periodic -stag_boundary_type_y none -stag_boundary_type_z none -stag_grid_x 3 -stag_grid_y 6 -stag_grid_z 5 -stag_ranks_x 1 -stag_ranks_y 2 -stag_ranks_z 2 -stag_stencil_width 1 -useinjective 0

TEST*/
