/* Routines to convert between a (subset of) DMStag and DMDA */

#include <petscdmda.h>
#include <petsc/private/dmstagimpl.h>
#include <petscdmdatypes.h>

static PetscErrorCode DMStagCreateCompatibleDMDA(DM dm,DMStagStencilLocation loc,PetscInt c,DM *dmda)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*) dm->data;
  PetscInt        dim,i,j,stencilWidth,dof,N[DMSTAG_MAX_DIM];
  DMDAStencilType stencilType;
  PetscInt        *l[DMSTAG_MAX_DIM];

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);

  /* Create grid decomposition (to be adjusted later) */
  for (i=0; i<dim; ++i) {
    ierr = PetscMalloc1(stag->nRanks[i],&l[i]);CHKERRQ(ierr);
    for (j=0; j<stag->nRanks[i]; ++j) l[i][j] = stag->l[i][j];
    N[i] = stag->N[i];
  }

  /* dof */
  dof = c < 0 ? -c : 1;

  /* Determine/adjust sizes */
  switch (loc) {
    case DMSTAG_ELEMENT:
      break;
    case DMSTAG_LEFT:
    case DMSTAG_RIGHT:
      if (dim<1) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Incompatible dim (%d) and loc(%s) combination",dim,DMStagStencilLocations[loc]);
      l[0][stag->nRanks[0]-1] += 1; /* extra vertex in direction 0 on last rank in dimension 0 */
      N[0] += 1;
      break;
    case DMSTAG_UP:
    case DMSTAG_DOWN:
      if (dim < 2) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Incompatible dim (%d) and loc(%s) combination",dim,DMStagStencilLocations[loc]);
      l[1][stag->nRanks[1]-1] += 1; /* extra vertex in direction 1 on last rank in dimension 1 */
      N[1] += 1;
      break;
    case DMSTAG_BACK:
    case DMSTAG_FRONT:
      if (dim < 3) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Incompatible dim (%d) and loc(%s) combination",dim,DMStagStencilLocations[loc]);
      l[2][stag->nRanks[2]-1] += 1; /* extra vertex in direction 2 on last rank in dimension 2 */
      N[2] += 1;
      break;
    case DMSTAG_DOWN_LEFT :
    case DMSTAG_DOWN_RIGHT :
    case DMSTAG_UP_LEFT :
    case DMSTAG_UP_RIGHT :
      if (dim < 2) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Incompatible dim (%d) and loc(%s) combination",dim,DMStagStencilLocations[loc]);
      for (i=0; i<2; ++i) { /* extra vertex in direction i on last rank in dimension i = 0,1 */
        l[i][stag->nRanks[i]-1] += 1;
        N[i] += 1;
      }
      break;
    case DMSTAG_BACK_LEFT:
    case DMSTAG_BACK_RIGHT:
    case DMSTAG_FRONT_LEFT:
    case DMSTAG_FRONT_RIGHT:
      if (dim < 3) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Incompatible dim (%d) and loc(%s) combination",dim,DMStagStencilLocations[loc]);
      for (i=0; i<3; i+=2) { /* extra vertex in direction i on last rank in dimension i = 0,2 */
        l[i][stag->nRanks[i]-1] += 1;
        N[i] += 1;
      }
      break;
    case DMSTAG_BACK_DOWN:
    case DMSTAG_BACK_UP:
    case DMSTAG_FRONT_DOWN:
    case DMSTAG_FRONT_UP:
      if (dim < 3) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Incompatible dim (%d) and loc(%s) combination",dim,DMStagStencilLocations[loc]);
      for (i=1; i<3; ++i) { /* extra vertex in direction i on last rank in dimension i = 1,2 */
        l[i][stag->nRanks[i]-1] += 1;
        N[i] += 1;
      }
      break;
    case DMSTAG_BACK_DOWN_LEFT:
    case DMSTAG_BACK_DOWN_RIGHT:
    case DMSTAG_BACK_UP_LEFT:
    case DMSTAG_BACK_UP_RIGHT:
    case DMSTAG_FRONT_DOWN_LEFT:
    case DMSTAG_FRONT_DOWN_RIGHT:
    case DMSTAG_FRONT_UP_LEFT:
    case DMSTAG_FRONT_UP_RIGHT:
      if (dim < 3) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Incompatible dim (%d) and loc(%s) combination",dim,DMStagStencilLocations[loc]);
      for (i=0; i<3; ++i) { /* extra vertex in direction i on last rank in dimension i = 0,1,2 */
        l[i][stag->nRanks[i]-1] += 1;
        N[i] += 1;
      }
      break;
      break;
    default : SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented for location %s",DMStagStencilLocations[loc]);
  }

  /* Use the same stencil type */
  switch (stag->stencilType) {
    case DMSTAG_STENCIL_STAR: stencilType = DMDA_STENCIL_STAR; stencilWidth = stag->stencilWidth; break;
    case DMSTAG_STENCIL_BOX : stencilType = DMDA_STENCIL_BOX ; stencilWidth = stag->stencilWidth; break;
    default: SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported Stencil Type %d",stag->stencilType);
  }

  /* Create DMDA, using same boundary type */
  switch (dim) {
    case 1:
      ierr = DMDACreate1d(PetscObjectComm((PetscObject)dm),stag->boundaryType[0],N[0],dof,stencilWidth,l[0],dmda);CHKERRQ(ierr);
      break;
    case 2:
      ierr = DMDACreate2d(PetscObjectComm((PetscObject)dm),stag->boundaryType[0],stag->boundaryType[1],stencilType,N[0],N[1],stag->nRanks[0],stag->nRanks[1],dof,stencilWidth,l[0],l[1],dmda);CHKERRQ(ierr);
      break;
    case 3:
      ierr = DMDACreate3d(PetscObjectComm((PetscObject)dm),stag->boundaryType[0],stag->boundaryType[1],stag->boundaryType[2],stencilType,N[0],N[1],N[2],stag->nRanks[0],stag->nRanks[1],stag->nRanks[2],dof,stencilWidth,l[0],l[1],l[2],dmda);CHKERRQ(ierr);
      break;
    default: SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"not implemented for dim %d",dim);
  }
  for (i=0; i<dim; ++i) {
    ierr = PetscFree(l[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
Helper function to get the number of extra points in a DMDA representation for a given canonical location.
*/
static PetscErrorCode DMStagDMDAGetExtraPoints(DM dm,DMStagStencilLocation locCanonical,PetscInt *extraPoint)
{
  PetscErrorCode ierr;
  PetscInt       dim,d,nExtra[DMSTAG_MAX_DIM];

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dim > DMSTAG_MAX_DIM) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented for %D dimensions",dim);
  ierr = DMStagGetCorners(dm,NULL,NULL,NULL,NULL,NULL,NULL,&nExtra[0],&nExtra[1],&nExtra[2]);CHKERRQ(ierr);
  for (d=0; d<dim; ++d) extraPoint[d] = 0;
  switch (locCanonical) {
    case DMSTAG_ELEMENT:
      break; /* no extra points */
    case DMSTAG_LEFT:
      extraPoint[0] = nExtra[0]; break; /* only extra point in x */
    case DMSTAG_DOWN:
      extraPoint[1] = nExtra[1]; break; /* only extra point in y */
    case DMSTAG_BACK:
      extraPoint[2] = nExtra[2]; break; /* only extra point in z */
    case DMSTAG_DOWN_LEFT:
      extraPoint[0] = nExtra[0]; extraPoint[1] = nExtra[1]; break; /* extra point in both x and y  */
    case DMSTAG_BACK_LEFT:
      extraPoint[0] = nExtra[0]; extraPoint[2] = nExtra[2]; break; /* extra point in both x and z  */
    case DMSTAG_BACK_DOWN:
      extraPoint[1] = nExtra[1]; extraPoint[2] = nExtra[2]; break; /* extra point in both y and z  */
    case DMSTAG_BACK_DOWN_LEFT:
     extraPoint[0] = nExtra[0]; extraPoint[1] = nExtra[1]; extraPoint[2] = nExtra[2]; break; /* extra points in x,y,z */
    default : SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented for location (perhaps not canonical) %s",DMStagStencilLocations[locCanonical]);
  }
  PetscFunctionReturn(0);
}

/*
Function much like DMStagMigrateVec(), but which accepts an additional position argument to disambiguate which
type of DMDA to migrate to.
*/

static PetscErrorCode DMStagMigrateVecDMDA(DM dm,Vec vec,DMStagStencilLocation loc,PetscInt c,DM dmTo,Vec vecTo)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,d,dim,dof,dofToMax,start[DMSTAG_MAX_DIM],n[DMSTAG_MAX_DIM],extraPoint[DMSTAG_MAX_DIM];
  Vec            vecLocal;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidHeaderSpecificType(dmTo,DM_CLASSID,4,DMDA);
  PetscValidHeaderSpecific(vecTo,VEC_CLASSID,5);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMDAGetDof(dmTo,&dofToMax);CHKERRQ(ierr);
  if (-c > dofToMax) SETERRQ1(PetscObjectComm((PetscObject)dmTo),PETSC_ERR_ARG_OUTOFRANGE,"Invalid negative component value. Must be >= -%D",dofToMax);
  ierr = DMStagGetCorners(dm,&start[0],&start[1],&start[2],&n[0],&n[1],&n[2],NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagDMDAGetExtraPoints(dm,loc,extraPoint);CHKERRQ(ierr);
  ierr = DMStagGetLocationDOF(dm,loc,&dof);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&vecLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,vec,INSERT_VALUES,vecLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,vec,INSERT_VALUES,vecLocal);CHKERRQ(ierr);
  if (dim == 1) {
    PetscScalar **arrTo;
    ierr = DMDAVecGetArrayDOF(dmTo,vecTo,&arrTo);CHKERRQ(ierr);
    if (c < 0) {
      const PetscInt dofTo = -c;
      for (i=start[0]; i<start[0] + n[0] + extraPoint[0]; ++i) {
        for (d=0; d<PetscMin(dof,dofTo); ++d) {
          DMStagStencil pos;
          pos.i = i; pos.loc = loc; pos.c = d;
          ierr = DMStagVecGetValuesStencil(dm,vecLocal,1,&pos,&arrTo[i][d]);CHKERRQ(ierr);
        }
        for (;d<dofTo; ++d) {
          arrTo[i][d] = 0.0; /* Pad extra dof with zeros */
        }
      }
    } else {
      for (i=start[0]; i<start[0] + n[0] + extraPoint[0]; ++i) {
        DMStagStencil pos;
        pos.i = i; pos.loc = loc; pos.c = c;
        ierr = DMStagVecGetValuesStencil(dm,vecLocal,1,&pos,&arrTo[i][0]);CHKERRQ(ierr);
      }
    }
    ierr = DMDAVecRestoreArrayDOF(dmTo,vecTo,&arrTo);CHKERRQ(ierr);
  } else if (dim == 2) {
    PetscScalar ***arrTo;
    ierr = DMDAVecGetArrayDOF(dmTo,vecTo,&arrTo);CHKERRQ(ierr);
    if (c < 0) {
      const PetscInt dofTo = -c;
      for (j=start[1]; j<start[1] + n[1] + extraPoint[1]; ++j) {
        for (i=start[0]; i<start[0] + n[0] + extraPoint[0]; ++i) {
          for (d=0; d<PetscMin(dof,dofTo); ++d) {
            DMStagStencil pos;
            pos.i = i; pos.j = j; pos.loc = loc; pos.c = d;
            ierr = DMStagVecGetValuesStencil(dm,vecLocal,1,&pos,&arrTo[j][i][d]);CHKERRQ(ierr);
          }
          for (;d<dofTo; ++d) {
            arrTo[j][i][d] = 0.0; /* Pad extra dof with zeros */
          }
        }
      }
    } else {
      for (j=start[1]; j<start[1] + n[1] + extraPoint[1]; ++j) {
        for (i=start[0]; i<start[0] + n[0] + extraPoint[0]; ++i) {
          DMStagStencil pos;
          pos.i = i; pos.j = j; pos.loc = loc; pos.c = c;
          ierr = DMStagVecGetValuesStencil(dm,vecLocal,1,&pos,&arrTo[j][i][0]);CHKERRQ(ierr);
        }
      }
    }
    ierr = DMDAVecRestoreArrayDOF(dmTo,vecTo,&arrTo);CHKERRQ(ierr);
  } else if (dim == 3) {
    PetscScalar ****arrTo;
    ierr = DMDAVecGetArrayDOF(dmTo,vecTo,&arrTo);CHKERRQ(ierr);
    if (c < 0) {
      const PetscInt dofTo = -c;
      for (k=start[2]; k<start[2] + n[2] + extraPoint[2]; ++k) {
        for (j=start[1]; j<start[1] + n[1] + extraPoint[1]; ++j) {
          for (i=start[0]; i<start[0] + n[0] + extraPoint[0]; ++i) {
            for (d=0; d<PetscMin(dof,dofTo); ++d) {
              DMStagStencil pos;
              pos.i = i; pos.j = j; pos.k = k; pos.loc = loc; pos.c = d;
              ierr = DMStagVecGetValuesStencil(dm,vecLocal,1,&pos,&arrTo[k][j][i][d]);CHKERRQ(ierr);
            }
            for (;d<dofTo; ++d) {
              arrTo[k][j][i][d] = 0.0; /* Pad extra dof with zeros */
            }
          }
        }
      }
    } else {
      for (k=start[2]; k<start[2] + n[2] + extraPoint[2]; ++k) {
        for (j=start[1]; j<start[1] + n[1] + extraPoint[1]; ++j) {
          for (i=start[0]; i<start[0] + n[0] + extraPoint[0]; ++i) {
            DMStagStencil pos;
            pos.i = i; pos.j = j; pos.k = k; pos.loc = loc; pos.c = c;
            ierr = DMStagVecGetValuesStencil(dm,vecLocal,1,&pos,&arrTo[k][j][i][0]);CHKERRQ(ierr);
          }
        }
      }
    }
    ierr = DMDAVecRestoreArrayDOF(dmTo,vecTo,&arrTo);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported dimension %d",dim);
  ierr = DMRestoreLocalVector(dm,&vecLocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Transfer coordinates from a DMStag to a DMDA, specifying which location */
static PetscErrorCode DMStagTransferCoordinatesToDMDA(DM dmstag,DMStagStencilLocation loc,DM dmda)
{
  PetscErrorCode ierr;
  PetscInt       dim,start[DMSTAG_MAX_DIM],n[DMSTAG_MAX_DIM],extraPoint[DMSTAG_MAX_DIM],d;
  DM             dmstagCoord,dmdaCoord;
  DMType         dmstagCoordType;
  Vec            stagCoord,daCoord;
  PetscBool      daCoordIsStag,daCoordIsProduct;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dmstag,DM_CLASSID,1,DMSTAG);
  PetscValidHeaderSpecificType(dmda,DM_CLASSID,3,DMDA);
  ierr = DMGetDimension(dmstag,&dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dmstag,&dmstagCoord);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dmstag,&stagCoord);CHKERRQ(ierr); /* Note local */
  ierr = DMGetCoordinateDM(dmda,&dmdaCoord);CHKERRQ(ierr);
  daCoord = NULL;
  ierr = DMGetCoordinates(dmda,&daCoord);CHKERRQ(ierr);
  if (!daCoord) {
    ierr = DMCreateGlobalVector(dmdaCoord,&daCoord);CHKERRQ(ierr);
    ierr = DMSetCoordinates(dmda,daCoord);CHKERRQ(ierr);
    ierr = VecDestroy(&daCoord);CHKERRQ(ierr);
    ierr = DMGetCoordinates(dmda,&daCoord);CHKERRQ(ierr);
  }
  ierr = DMGetType(dmstagCoord,&dmstagCoordType);CHKERRQ(ierr);
  ierr = PetscStrcmp(dmstagCoordType,DMSTAG,&daCoordIsStag);CHKERRQ(ierr);
  ierr = PetscStrcmp(dmstagCoordType,DMPRODUCT,&daCoordIsProduct);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmstag,&start[0],&start[1],&start[2],&n[0],&n[1],&n[2],NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagDMDAGetExtraPoints(dmstag,loc,extraPoint);CHKERRQ(ierr);
  if (dim == 1) {
    PetscInt ex;
    PetscScalar **cArrDa;
    ierr = DMDAVecGetArrayDOF(dmdaCoord,daCoord,&cArrDa);CHKERRQ(ierr);
    if (daCoordIsStag)  {
      PetscInt slot;
      PetscScalar **cArrStag;
      ierr = DMStagGetLocationSlot(dmstagCoord,loc,0,&slot);CHKERRQ(ierr);
      ierr = DMStagVecGetArrayRead(dmstagCoord,stagCoord,&cArrStag);CHKERRQ(ierr);
      for (ex=start[0]; ex<start[0] + n[0] + extraPoint[0]; ++ex) {
        cArrDa[ex][0] = cArrStag[ex][slot];
      }
      ierr = DMStagVecRestoreArrayRead(dmstagCoord,stagCoord,&cArrStag);CHKERRQ(ierr);
    } else if (daCoordIsProduct) {
      PetscScalar **cArrX;
      ierr = DMStagGetProductCoordinateArraysRead(dmstag,&cArrX,NULL,NULL);CHKERRQ(ierr);
      for (ex=start[0]; ex<start[0] + n[0] + extraPoint[0]; ++ex) {
        cArrDa[ex][0] = cArrX[ex][0];
      }
      ierr = DMStagRestoreProductCoordinateArraysRead(dmstag,&cArrX,NULL,NULL);CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)dmstag),PETSC_ERR_SUP,"Stag to DA coordinate transfer only supported for DMStag coordinate DM of type DMstag or DMProduct");
    ierr = DMDAVecRestoreArrayDOF(dmdaCoord,daCoord,&cArrDa);CHKERRQ(ierr);
  } else if (dim == 2) {
    PetscInt ex,ey;
    PetscScalar ***cArrDa;
    ierr = DMDAVecGetArrayDOF(dmdaCoord,daCoord,&cArrDa);CHKERRQ(ierr);
    if (daCoordIsStag)  {
      PetscInt slot;
      PetscScalar ***cArrStag;
      ierr = DMStagGetLocationSlot(dmstagCoord,loc,0,&slot);CHKERRQ(ierr);
      ierr = DMStagVecGetArrayRead(dmstagCoord,stagCoord,&cArrStag);CHKERRQ(ierr);
      for (ey=start[1]; ey<start[1] + n[1] + extraPoint[1]; ++ey) {
        for (ex=start[0]; ex<start[0] + n[0] + extraPoint[0]; ++ex) {
          for (d=0; d<2; ++d) {
            cArrDa[ey][ex][d] = cArrStag[ey][ex][slot+d];
          }
        }
      }
      ierr = DMStagVecRestoreArrayRead(dmstagCoord,stagCoord,&cArrStag);CHKERRQ(ierr);
    } else if (daCoordIsProduct) {
      PetscScalar **cArrX,**cArrY;
      ierr = DMStagGetProductCoordinateArraysRead(dmstag,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
      for (ey=start[1]; ey<start[1] + n[1] + extraPoint[1]; ++ey) {
        for (ex=start[0]; ex<start[0] + n[0] + extraPoint[0]; ++ex) {
          cArrDa[ey][ex][0] = cArrX[ex][0];
          cArrDa[ey][ex][1] = cArrY[ey][0];
        }
      }
      ierr = DMStagRestoreProductCoordinateArraysRead(dmstag,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)dmstag),PETSC_ERR_SUP,"Stag to DA coordinate transfer only supported for DMStag coordinate DM of type DMstag or DMProduct");
    ierr = DMDAVecRestoreArrayDOF(dmdaCoord,daCoord,&cArrDa);CHKERRQ(ierr);
  }  else if (dim == 3) {
    PetscInt ex,ey,ez;
    PetscScalar ****cArrDa;
    ierr = DMDAVecGetArrayDOF(dmdaCoord,daCoord,&cArrDa);CHKERRQ(ierr);
    if (daCoordIsStag)  {
      PetscInt slot;
      PetscScalar ****cArrStag;
      ierr = DMStagGetLocationSlot(dmstagCoord,loc,0,&slot);CHKERRQ(ierr);
      ierr = DMStagVecGetArrayRead(dmstagCoord,stagCoord,&cArrStag);CHKERRQ(ierr);
      for (ez=start[2]; ez<start[2] + n[2] + extraPoint[2]; ++ez) {
        for (ey=start[1]; ey<start[1] + n[1] + extraPoint[1]; ++ey) {
          for (ex=start[0]; ex<start[0] + n[0] + extraPoint[0]; ++ex) {
            for (d=0; d<3; ++d) {
              cArrDa[ez][ey][ex][d] = cArrStag[ez][ey][ex][slot+d];
            }
          }
        }
      }
      ierr = DMStagVecRestoreArrayRead(dmstagCoord,stagCoord,&cArrStag);CHKERRQ(ierr);
    } else if (daCoordIsProduct) {
      PetscScalar **cArrX,**cArrY,**cArrZ;
      ierr = DMStagGetProductCoordinateArraysRead(dmstag,&cArrX,&cArrY,&cArrZ);CHKERRQ(ierr);
      for (ez=start[2]; ez<start[2] + n[2] + extraPoint[2]; ++ez) {
        for (ey=start[1]; ey<start[1] + n[1] + extraPoint[1]; ++ey) {
          for (ex=start[0]; ex<start[0] + n[0] + extraPoint[0]; ++ex) {
            cArrDa[ez][ey][ex][0] = cArrX[ex][0];
            cArrDa[ez][ey][ex][1] = cArrY[ey][0];
            cArrDa[ez][ey][ex][2] = cArrZ[ez][0];
          }
        }
      }
      ierr = DMStagRestoreProductCoordinateArraysRead(dmstag,&cArrX,&cArrY,&cArrZ);CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)dmstag),PETSC_ERR_SUP,"Stag to DA coordinate transfer only supported for DMStag coordinate DM of type DMstag or DMProduct");
    ierr = DMDAVecRestoreArrayDOF(dmdaCoord,daCoord,&cArrDa);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported dimension %d",dim);
  PetscFunctionReturn(0);
}

/*
Convert to a location value with only BACK, DOWN, LEFT, and ELEMENT involved (makes looping easier)
*/
static PetscErrorCode DMStagStencilLocationCanonicalize(DMStagStencilLocation loc,DMStagStencilLocation *locCanonical)
{
  PetscFunctionBegin;
  switch (loc) {
    case DMSTAG_ELEMENT:
      *locCanonical = DMSTAG_ELEMENT;
      break;
    case DMSTAG_LEFT:
    case DMSTAG_RIGHT:
      *locCanonical = DMSTAG_LEFT;
      break;
    case DMSTAG_DOWN:
    case DMSTAG_UP:
      *locCanonical = DMSTAG_DOWN;
      break;
    case DMSTAG_BACK:
    case DMSTAG_FRONT:
      *locCanonical = DMSTAG_BACK;
      break;
    case DMSTAG_DOWN_LEFT :
    case DMSTAG_DOWN_RIGHT :
    case DMSTAG_UP_LEFT :
    case DMSTAG_UP_RIGHT :
      *locCanonical = DMSTAG_DOWN_LEFT;
      break;
    case DMSTAG_BACK_LEFT:
    case DMSTAG_BACK_RIGHT:
    case DMSTAG_FRONT_LEFT:
    case DMSTAG_FRONT_RIGHT:
      *locCanonical = DMSTAG_BACK_LEFT;
      break;
    case DMSTAG_BACK_DOWN:
    case DMSTAG_BACK_UP:
    case DMSTAG_FRONT_DOWN:
    case DMSTAG_FRONT_UP:
      *locCanonical = DMSTAG_BACK_DOWN;
      break;
    case DMSTAG_BACK_DOWN_LEFT:
    case DMSTAG_BACK_DOWN_RIGHT:
    case DMSTAG_BACK_UP_LEFT:
    case DMSTAG_BACK_UP_RIGHT:
    case DMSTAG_FRONT_DOWN_LEFT:
    case DMSTAG_FRONT_DOWN_RIGHT:
    case DMSTAG_FRONT_UP_LEFT:
    case DMSTAG_FRONT_UP_RIGHT:
      *locCanonical = DMSTAG_BACK_DOWN_LEFT;
      break;
    default :
      *locCanonical = DMSTAG_NULL_LOCATION;
      break;
  }
  PetscFunctionReturn(0);
}

/*@C
  DMStagVecSplitToDMDA - create a DMDA and Vec from a DMStag and Vec

  Logically Collective

  High-level helper function which accepts a DMStag, a global vector, and location/dof,
  and generates a corresponding DMDA and Vec.

  Input Parameters:
+ dm - the DMStag object
. vec- Vec object associated with dm
. loc - which subgrid to extract (see DMStagStencilLocation)
- c - which component to extract (see note below)

  Output Parameters:
+ pda - the new DMDA
- pdavec - the new Vec

  Notes:
  If a c value of -k is provided, the first k dof for that position are extracted,
  padding with zero values if needbe. If a non-negative value is provided, a single
  dof is extracted.

  The caller is responsible for destroying the created DMDA and Vec.

  Level: advanced

.seealso: DMSTAG, DMDA, DMStagMigrateVec(), DMStagCreateCompatibleDMStag()
@*/
PetscErrorCode DMStagVecSplitToDMDA(DM dm,Vec vec,DMStagStencilLocation loc,PetscInt c,DM *pda,Vec *pdavec)
{
  PetscErrorCode  ierr;
  PetscInt        dim,locdof;
  DM              da,coordDM;
  Vec             davec;
  DMStagStencilLocation locCanonical;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMStagGetLocationDOF(dm,loc,&locdof);CHKERRQ(ierr);
  if (c >= locdof) SETERRQ3(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Location %s has %D dof, but component %D requested\n",DMStagStencilLocations[loc],locdof,c);
  ierr = DMStagStencilLocationCanonicalize(loc,&locCanonical);CHKERRQ(ierr);
  ierr = DMStagCreateCompatibleDMDA(dm,locCanonical,c,pda);CHKERRQ(ierr);
  da = *pda;
  ierr = DMSetUp(*pda);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm,&coordDM);CHKERRQ(ierr);
  if (coordDM) {
    ierr = DMStagTransferCoordinatesToDMDA(dm,locCanonical,da);CHKERRQ(ierr);
  }
  ierr = DMCreateGlobalVector(da,pdavec);CHKERRQ(ierr);
  davec = *pdavec;
  ierr = DMStagMigrateVecDMDA(dm,vec,locCanonical,c,da,davec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
