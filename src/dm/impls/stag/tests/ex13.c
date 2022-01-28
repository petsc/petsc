static char help[] = "Test DMStagPopulateLocalToGlobalInjective.\n\n";

#include <petscdm.h>
#include <petscdmstag.h>

static PetscErrorCode Test1(DM dm);
static PetscErrorCode Test2_1d(DM dm);
static PetscErrorCode Test2_2d(DM dm);
static PetscErrorCode Test2_3d(DM dm);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             dm;
  PetscInt       dim;
  PetscBool      setSizes,useInjective;

  /* Initialize PETSc and process command line arguments */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  dim = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL);CHKERRQ(ierr);
  setSizes = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-setsizes",&setSizes,NULL);CHKERRQ(ierr);
  useInjective = PETSC_TRUE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-useinjective",&useInjective,NULL);CHKERRQ(ierr);

  /* Creation (normal) */
  if (!setSizes) {
    switch (dim) {
      case 1:
        ierr = DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,3,1,1,DMSTAG_STENCIL_BOX,1,NULL,&dm);CHKERRQ(ierr);
        break;
      case 2:
        ierr = DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,3,2,PETSC_DECIDE,PETSC_DECIDE,1,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,&dm);CHKERRQ(ierr);
        break;
      case 3:
        ierr = DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,3,2,4,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dm);CHKERRQ(ierr);
        break;
      default:
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"No support for dimension %D",dim);
    }
  } else {
    /* Creation (test providing decomp exactly)*/
    PetscMPIInt size;
    PetscInt lx[4] = {2,3,4}, ranksx = 3, mx = 9;
    PetscInt ly[3] = {4,5},   ranksy = 2, my = 9;
    PetscInt lz[2] = {6,7},   ranksz = 2, mz = 13;

    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
    switch (dim) {
      case 1:
        PetscAssertFalse(size != ranksx,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Must run on %D ranks with -dim 1 -setSizes",ranksx);
        ierr = DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,mx,1,1,DMSTAG_STENCIL_BOX,1,lx,&dm);CHKERRQ(ierr);
        break;
      case 2:
        PetscAssertFalse(size != ranksx * ranksy,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Must run on %D ranks with -dim 2 -setSizes",ranksx * ranksy);
        ierr = DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,mx,my,ranksx,ranksy,1,1,1,DMSTAG_STENCIL_BOX,1,lx,ly,&dm);CHKERRQ(ierr);
        break;
      case 3:
        PetscAssertFalse(size != ranksx * ranksy * ranksz,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Must run on %D ranks with -dim 3 -setSizes", ranksx * ranksy * ranksz);
        ierr = DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,mx,my,mz,ranksx,ranksy,ranksz,1,1,1,1,DMSTAG_STENCIL_BOX,1,lx,ly,lz,&dm);CHKERRQ(ierr);
        break;
      default:
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"No support for dimension %D",dim);
    }
  }

  /* Setup */
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);

  /* Populate Additional Injective Local-to-Global Map */
  if (useInjective) {
    ierr = DMStagPopulateLocalToGlobalInjective(dm);CHKERRQ(ierr);
  }

  /* Test: Make sure L2G inverts G2L */
  ierr = Test1(dm);CHKERRQ(ierr);

  /* Test: Make sure that G2L inverts L2G, on its domain */
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 1: ierr = Test2_1d(dm);CHKERRQ(ierr); break;
    case 2: ierr = Test2_2d(dm);CHKERRQ(ierr); break;
    case 3: ierr = Test2_3d(dm);CHKERRQ(ierr); break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented for dimension %D",dim);
  }

  /* Clean up and finalize PETSc */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

static PetscErrorCode Test1(DM dm)
{
  PetscErrorCode ierr;
  Vec            vecLocal,vecGlobal,vecGlobalCheck;
  PetscRandom    rctx;
  PetscBool      equal;

  PetscFunctionBeginUser;
  ierr = DMCreateLocalVector(dm,&vecLocal);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&vecGlobal);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = VecSetRandom(vecGlobal,rctx);CHKERRQ(ierr);
  ierr = VecSetRandom(vecLocal,rctx);CHKERRQ(ierr); /* garbage */
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = VecDuplicate(vecGlobal,&vecGlobalCheck);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,vecGlobal,INSERT_VALUES,vecLocal);CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm,vecLocal,INSERT_VALUES,vecGlobalCheck);CHKERRQ(ierr);
  ierr = VecEqual(vecGlobal,vecGlobalCheck,&equal);CHKERRQ(ierr);
  PetscAssertFalse(!equal,PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Check failed - vectors should be bitwise identical");
  ierr = VecDestroy(&vecLocal);CHKERRQ(ierr);
  ierr = VecDestroy(&vecGlobal);CHKERRQ(ierr);
  ierr = VecDestroy(&vecGlobalCheck);CHKERRQ(ierr);
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
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"(%D,%D,%D,%D) Value %.17g does not match the expected %.17g",i,j,k,c,(double)PetscRealPart(val),(double)PetscRealPart(valRef));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode Test2_1d(DM dm)
{
  PetscErrorCode ierr;
  Vec            vecLocal,vecLocalCheck,vecGlobal;
  PetscInt       i,startx,nx,nExtrax,dof0,dof1,c,idxLeft,idxElement;
  PetscScalar    **arr;
  const PetscInt j=-1,k=-1;

  PetscFunctionBeginUser;
  ierr = DMCreateLocalVector(dm,&vecLocal);CHKERRQ(ierr);
  ierr = VecSet(vecLocal,-1.0);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm,&startx,NULL,NULL,&nx,NULL,NULL,&nExtrax,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetDOF(dm,&dof0,&dof1,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,vecLocal,&arr);CHKERRQ(ierr);
  if (dof0 > 0) {
    ierr = DMStagGetLocationSlot(dm,DMSTAG_LEFT,0,&idxLeft);CHKERRQ(ierr);
  }
  if (dof1 > 0) {
    ierr = DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,0,&idxElement);CHKERRQ(ierr);
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
  ierr = DMStagVecRestoreArray(dm,vecLocal,&arr);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&vecGlobal);CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm,vecLocal,INSERT_VALUES,vecGlobal);CHKERRQ(ierr);
  ierr = VecDuplicate(vecLocal,&vecLocalCheck);CHKERRQ(ierr);
  ierr = VecSet(vecLocalCheck,-1.0);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,vecGlobal,INSERT_VALUES,vecLocalCheck);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dm,vecLocalCheck,&arr);CHKERRQ(ierr);
  for (i=startx; i<startx+nx+nExtrax; ++i) {
    for (c=0; c<dof0; ++c) {
      const PetscScalar valRef = TEST_FUNCTION(i,0,0,idxLeft,c);
      const PetscScalar val    = arr[i][idxLeft+c];
      ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
    }
    if (i < startx+nx) {
      for (c=0; c<dof1; ++c) {
        const PetscScalar valRef = TEST_FUNCTION(i,0,0,idxElement,c);
        const PetscScalar val    = arr[i][idxElement+c];
        ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
      }
    } else {
      for (c=0; c<dof1; ++c) {
        const PetscScalar valRef = -1.0;
        const PetscScalar val    = arr[i][idxElement+c];
        ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMStagVecRestoreArrayRead(dm,vecLocalCheck,&arr);CHKERRQ(ierr);
  ierr = VecDestroy(&vecLocal);CHKERRQ(ierr);
  ierr = VecDestroy(&vecLocalCheck);CHKERRQ(ierr);
  ierr = VecDestroy(&vecGlobal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode Test2_2d(DM dm)
{
  PetscErrorCode ierr;
  Vec            vecLocal,vecLocalCheck,vecGlobal;
  PetscInt       i,j,startx,starty,nx,ny,nExtrax,nExtray,dof0,dof1,dof2,c,idxLeft,idxDown,idxDownLeft,idxElement;
  PetscScalar    ***arr;
  const PetscInt k=-1;

  PetscFunctionBeginUser;
  ierr = DMCreateLocalVector(dm,&vecLocal);CHKERRQ(ierr);
  ierr = VecSet(vecLocal,-1.0);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm,&startx,&starty,NULL,&nx,&ny,NULL,&nExtrax,&nExtray,NULL);CHKERRQ(ierr);
  ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL);CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,vecLocal,&arr);CHKERRQ(ierr);
  if (dof0 > 0) {
    ierr = DMStagGetLocationSlot(dm,DMSTAG_DOWN_LEFT,0,&idxDownLeft);CHKERRQ(ierr);
  }
  if (dof1 > 0) {
    ierr = DMStagGetLocationSlot(dm,DMSTAG_LEFT,0,&idxLeft);CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(dm,DMSTAG_DOWN,0,&idxDown);CHKERRQ(ierr);
  }
  if (dof2 > 0) {
    ierr = DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,0,&idxElement);CHKERRQ(ierr);
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
  ierr = DMStagVecRestoreArray(dm,vecLocal,&arr);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&vecGlobal);CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm,vecLocal,INSERT_VALUES,vecGlobal);CHKERRQ(ierr);
  ierr = VecDuplicate(vecLocal,&vecLocalCheck);CHKERRQ(ierr);
  ierr = VecSet(vecLocalCheck,-1.0);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,vecGlobal,INSERT_VALUES,vecLocalCheck);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dm,vecLocalCheck,&arr);CHKERRQ(ierr);
  for (j=starty; j<starty+ny+nExtray; ++j) {
    for (i=startx; i<startx+nx+nExtrax; ++i) {
      for (c=0; c<dof0; ++c) {
        const PetscScalar valRef = TEST_FUNCTION(i,j,0,idxDownLeft,c);
        const PetscScalar val    = arr[j][i][idxDownLeft+c];
        ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
      }
      if (j < starty+ny) {
        for (c=0; c<dof1; ++c) {
          const PetscScalar valRef = TEST_FUNCTION(i,j,0,idxLeft,c);
          const PetscScalar val    = arr[j][i][idxLeft+c];
          ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
        }
      } else {
        for (c=0; c<dof1; ++c) {
          const PetscScalar valRef = -1.0;
          const PetscScalar val    = arr[j][i][idxLeft+c];
          ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
        }
      }
      if (i < startx+nx) {
        for (c=0; c<dof1; ++c) {
          const PetscScalar valRef = TEST_FUNCTION(i,j,0,idxDown,c);
          const PetscScalar val    = arr[j][i][idxDown+c];
          ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
        }
      } else {
        for (c=0; c<dof1; ++c) {
          const PetscScalar valRef = -1.0;
          const PetscScalar val    = arr[j][i][idxDown+c];
          ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
        }
      }
      if (i < startx+nx && j < starty+ny) {
        for (c=0; c<dof2; ++c) {
          const PetscScalar valRef = TEST_FUNCTION(i,j,0,idxElement,c);
          const PetscScalar val    = arr[j][i][idxElement+c];
          ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
        }
      } else {
        for (c=0; c<dof2; ++c) {
          const PetscScalar valRef = -1.0;
          const PetscScalar val    = arr[j][i][idxElement+c];
          ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = DMStagVecRestoreArrayRead(dm,vecLocalCheck,&arr);CHKERRQ(ierr);
  ierr = VecDestroy(&vecLocal);CHKERRQ(ierr);
  ierr = VecDestroy(&vecLocalCheck);CHKERRQ(ierr);
  ierr = VecDestroy(&vecGlobal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode Test2_3d(DM dm)
{
  PetscErrorCode ierr;
  Vec            vecLocal,vecLocalCheck,vecGlobal;
  PetscInt       i,j,k,startx,starty,startz,nx,ny,nz,nExtrax,nExtray,nExtraz,dof0,dof1,dof2,dof3,c,idxLeft,idxDown,idxDownLeft,idxBackDownLeft,idxBackDown,idxBack,idxBackLeft,idxElement;
  PetscScalar    ****arr;

  PetscFunctionBeginUser;
  ierr = DMCreateLocalVector(dm,&vecLocal);CHKERRQ(ierr);
  ierr = VecSet(vecLocal,-1.0);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm,&startx,&starty,&startz,&nx,&ny,&nz,&nExtrax,&nExtray,&nExtraz);CHKERRQ(ierr);
  ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,&dof3);CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,vecLocal,&arr);CHKERRQ(ierr);
  if (dof0 > 0) {
    ierr = DMStagGetLocationSlot(dm,DMSTAG_BACK_DOWN_LEFT,0,&idxBackDownLeft);CHKERRQ(ierr);
  }
  if (dof1 > 0) {
    ierr = DMStagGetLocationSlot(dm,DMSTAG_BACK_LEFT,0,&idxBackLeft);CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(dm,DMSTAG_BACK_DOWN,0,&idxBackDown);CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(dm,DMSTAG_DOWN_LEFT,0,&idxDownLeft);CHKERRQ(ierr);
  }
  if (dof2 > 0) {
    ierr = DMStagGetLocationSlot(dm,DMSTAG_LEFT,0,&idxLeft);CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(dm,DMSTAG_DOWN,0,&idxDown);CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(dm,DMSTAG_BACK,0,&idxBack);CHKERRQ(ierr);
  }
  if (dof3 > 0) {
    ierr = DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,0,&idxElement);CHKERRQ(ierr);
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
  ierr = DMStagVecRestoreArray(dm,vecLocal,&arr);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&vecGlobal);CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm,vecLocal,INSERT_VALUES,vecGlobal);CHKERRQ(ierr);
  ierr = VecDuplicate(vecLocal,&vecLocalCheck);CHKERRQ(ierr);
  ierr = VecSet(vecLocalCheck,-1.0);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,vecGlobal,INSERT_VALUES,vecLocalCheck);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dm,vecLocalCheck,&arr);CHKERRQ(ierr);
  for (k=startz; k<startz+nz+nExtraz; ++k) {
    for (j=starty; j<starty+ny+nExtray; ++j) {
      for (i=startx; i<startx+nx+nExtrax; ++i) {
        for (c=0; c<dof0; ++c) {
          const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxBackDownLeft,c);
          const PetscScalar val    = arr[k][j][i][idxBackDownLeft+c];
          ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
        }
        if (k < startz+nz) {
          for (c=0; c<dof1; ++c) {
            const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxDownLeft,c);
            const PetscScalar val    =  arr[k][j][i][idxDownLeft+c];
            ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
          }
        } else {
          for (c=0; c<dof1; ++c) {
            const PetscScalar valRef = -1.0;
            const PetscScalar val    = arr[k][j][i][idxDownLeft+c];
            ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
          }
        }
        if (j < starty+ny) {
          for (c=0; c<dof1; ++c) {
            const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxBackLeft,c);
            const PetscScalar val    = arr[k][j][i][idxBackLeft+c];
            ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
          }
        } else {
          for (c=0; c<dof1; ++c) {
            const PetscScalar valRef = -1.0;
            const PetscScalar val    = arr[k][j][i][idxBackLeft+c];
            ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
          }
        }
        if (i < startx+nx) {
          for (c=0; c<dof1; ++c) {
            const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxBackDown,c);
            const PetscScalar val    = arr[k][j][i][idxBackDown+c];
            ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
          }
        } else {
          for (c=0; c<dof1; ++c) {
            const PetscScalar valRef = -1.0;
            const PetscScalar val    = arr[k][j][i][idxBackDown+c];
            ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
          }
        }
        if (j < starty+ny && k < startz+nz) {
          for (c=0; c<dof2; ++c) {
            const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxLeft,c);
            const PetscScalar val    = arr[k][j][i][idxLeft+c];
            ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
          }
        } else {
          for (c=0; c<dof2; ++c) {
            const PetscScalar valRef = -1.0;
            const PetscScalar val    = arr[k][j][i][idxLeft+c];
            ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
          }
        }
        if (i < startx+nx && k < startz+nz) {
          for (c=0; c<dof2; ++c) {
            const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxDown,c);
            const PetscScalar val    = arr[k][j][i][idxDown+c];
            ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
          }
        } else {
          for (c=0; c<dof2; ++c) {
            const PetscScalar valRef = -1.0;
            const PetscScalar val    = arr[k][j][i][idxDown+c];
            ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
          }
        }
        if (i < startx+nx && j < starty+ny) {
          for (c=0; c<dof2; ++c) {
            const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxBack,c);
            const PetscScalar val    = arr[k][j][i][idxBack+c];
            ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
          }
        } else {
          for (c=0; c<dof2; ++c) {
            const PetscScalar valRef = -1.0;
            const PetscScalar val    = arr[k][j][i][idxBack+c];
            ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
          }
        }
        if (i < startx+nx && j < starty+ny && k < startz+nz) {
          for (c=0; c<dof3; ++c) {
            const PetscScalar valRef = TEST_FUNCTION(i,j,k,idxElement,c);
            const PetscScalar val    = arr[k][j][i][idxElement+c];
            ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
          }
        } else {
          for (c=0; c<dof3; ++c) {
            const PetscScalar valRef = -1.0;
            const PetscScalar val    = arr[k][j][i][idxElement+c];
            ierr = CompareValues(i,j,k,c,val,valRef);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  ierr = DMStagVecRestoreArrayRead(dm,vecLocalCheck,&arr);CHKERRQ(ierr);
  ierr = VecDestroy(&vecLocal);CHKERRQ(ierr);
  ierr = VecDestroy(&vecLocalCheck);CHKERRQ(ierr);
  ierr = VecDestroy(&vecGlobal);CHKERRQ(ierr);
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
