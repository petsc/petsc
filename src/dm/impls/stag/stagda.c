/* Routines to convert between a (subset of) DMStag and DMDA */

#include <petscdmda.h>
#include <petsc/private/dmstagimpl.h>
#include <petscdmdatypes.h>

static PetscErrorCode DMStagCreateCompatibleDMDA(DM dm,DMStagStencilLocation loc,PetscInt c,DM *dmda)
{
  DM_Stag * const stag = (DM_Stag*) dm->data;
  PetscInt        dim,i,j,stencilWidth,dof,N[DMSTAG_MAX_DIM];
  DMDAStencilType stencilType;
  PetscInt        *l[DMSTAG_MAX_DIM];

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscCall(DMGetDimension(dm,&dim));

  /* Create grid decomposition (to be adjusted later) */
  for (i=0; i<dim; ++i) {
    PetscCall(PetscMalloc1(stag->nRanks[i],&l[i]));
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
      PetscCheck(dim>=1,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Incompatible dim (%" PetscInt_FMT ") and loc(%s) combination",dim,DMStagStencilLocations[loc]);
      l[0][stag->nRanks[0]-1] += 1; /* extra vertex in direction 0 on last rank in dimension 0 */
      N[0] += 1;
      break;
    case DMSTAG_UP:
    case DMSTAG_DOWN:
      PetscCheck(dim >= 2,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Incompatible dim (%" PetscInt_FMT ") and loc(%s) combination",dim,DMStagStencilLocations[loc]);
      l[1][stag->nRanks[1]-1] += 1; /* extra vertex in direction 1 on last rank in dimension 1 */
      N[1] += 1;
      break;
    case DMSTAG_BACK:
    case DMSTAG_FRONT:
      PetscCheck(dim >= 3,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Incompatible dim (%" PetscInt_FMT ") and loc(%s) combination",dim,DMStagStencilLocations[loc]);
      l[2][stag->nRanks[2]-1] += 1; /* extra vertex in direction 2 on last rank in dimension 2 */
      N[2] += 1;
      break;
    case DMSTAG_DOWN_LEFT :
    case DMSTAG_DOWN_RIGHT :
    case DMSTAG_UP_LEFT :
    case DMSTAG_UP_RIGHT :
      PetscCheck(dim >= 2,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Incompatible dim (%" PetscInt_FMT ") and loc(%s) combination",dim,DMStagStencilLocations[loc]);
      for (i=0; i<2; ++i) { /* extra vertex in direction i on last rank in dimension i = 0,1 */
        l[i][stag->nRanks[i]-1] += 1;
        N[i] += 1;
      }
      break;
    case DMSTAG_BACK_LEFT:
    case DMSTAG_BACK_RIGHT:
    case DMSTAG_FRONT_LEFT:
    case DMSTAG_FRONT_RIGHT:
      PetscCheck(dim >= 3,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Incompatible dim (%" PetscInt_FMT ") and loc(%s) combination",dim,DMStagStencilLocations[loc]);
      for (i=0; i<3; i+=2) { /* extra vertex in direction i on last rank in dimension i = 0,2 */
        l[i][stag->nRanks[i]-1] += 1;
        N[i] += 1;
      }
      break;
    case DMSTAG_BACK_DOWN:
    case DMSTAG_BACK_UP:
    case DMSTAG_FRONT_DOWN:
    case DMSTAG_FRONT_UP:
      PetscCheck(dim >= 3,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Incompatible dim (%" PetscInt_FMT ") and loc(%s) combination",dim,DMStagStencilLocations[loc]);
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
      PetscCheck(dim >= 3,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Incompatible dim (%" PetscInt_FMT ") and loc(%s) combination",dim,DMStagStencilLocations[loc]);
      for (i=0; i<3; ++i) { /* extra vertex in direction i on last rank in dimension i = 0,1,2 */
        l[i][stag->nRanks[i]-1] += 1;
        N[i] += 1;
      }
      break;
    default : SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented for location %s",DMStagStencilLocations[loc]);
  }

  /* Use the same stencil type */
  switch (stag->stencilType) {
    case DMSTAG_STENCIL_STAR: stencilType = DMDA_STENCIL_STAR; stencilWidth = stag->stencilWidth; break;
    case DMSTAG_STENCIL_BOX : stencilType = DMDA_STENCIL_BOX ; stencilWidth = stag->stencilWidth; break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported Stencil Type %d",stag->stencilType);
  }

  /* Create DMDA, using same boundary type */
  switch (dim) {
    case 1:
      PetscCall(DMDACreate1d(PetscObjectComm((PetscObject)dm),stag->boundaryType[0],N[0],dof,stencilWidth,l[0],dmda));
      break;
    case 2:
      PetscCall(DMDACreate2d(PetscObjectComm((PetscObject)dm),stag->boundaryType[0],stag->boundaryType[1],stencilType,N[0],N[1],stag->nRanks[0],stag->nRanks[1],dof,stencilWidth,l[0],l[1],dmda));
      break;
    case 3:
      PetscCall(DMDACreate3d(PetscObjectComm((PetscObject)dm),stag->boundaryType[0],stag->boundaryType[1],stag->boundaryType[2],stencilType,N[0],N[1],N[2],stag->nRanks[0],stag->nRanks[1],stag->nRanks[2],dof,stencilWidth,l[0],l[1],l[2],dmda));
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"not implemented for dim %" PetscInt_FMT,dim);
  }
  for (i=0; i<dim; ++i) {
    PetscCall(PetscFree(l[i]));
  }
  PetscFunctionReturn(0);
}

/*
Helper function to get the number of extra points in a DMDA representation for a given canonical location.
*/
static PetscErrorCode DMStagDMDAGetExtraPoints(DM dm,DMStagStencilLocation locCanonical,PetscInt *extraPoint)
{
  PetscInt       dim,d,nExtra[DMSTAG_MAX_DIM];

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscCall(DMGetDimension(dm,&dim));
  PetscCheck(dim <= DMSTAG_MAX_DIM,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented for %" PetscInt_FMT " dimensions",dim);
  PetscCall(DMStagGetCorners(dm,NULL,NULL,NULL,NULL,NULL,NULL,&nExtra[0],&nExtra[1],&nExtra[2]));
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
    default : SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented for location (perhaps not canonical) %s",DMStagStencilLocations[locCanonical]);
  }
  PetscFunctionReturn(0);
}

/*
Function much like DMStagMigrateVec(), but which accepts an additional position argument to disambiguate which
type of DMDA to migrate to.
*/

static PetscErrorCode DMStagMigrateVecDMDA(DM dm,Vec vec,DMStagStencilLocation loc,PetscInt c,DM dmTo,Vec vecTo)
{
  PetscInt       i,j,k,d,dim,dof,dofToMax,start[DMSTAG_MAX_DIM],n[DMSTAG_MAX_DIM],extraPoint[DMSTAG_MAX_DIM];
  Vec            vecLocal;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidHeaderSpecificType(dmTo,DM_CLASSID,5,DMDA);
  PetscValidHeaderSpecific(vecTo,VEC_CLASSID,6);
  PetscCall(DMGetDimension(dm,&dim));
  PetscCall(DMDAGetDof(dmTo,&dofToMax));
  PetscCheck(-c <= dofToMax,PetscObjectComm((PetscObject)dmTo),PETSC_ERR_ARG_OUTOFRANGE,"Invalid negative component value. Must be >= -%" PetscInt_FMT,dofToMax);
  PetscCall(DMStagGetCorners(dm,&start[0],&start[1],&start[2],&n[0],&n[1],&n[2],NULL,NULL,NULL));
  PetscCall(DMStagDMDAGetExtraPoints(dm,loc,extraPoint));
  PetscCall(DMStagGetLocationDOF(dm,loc,&dof));
  PetscCall(DMGetLocalVector(dm,&vecLocal));
  PetscCall(DMGlobalToLocalBegin(dm,vec,INSERT_VALUES,vecLocal));
  PetscCall(DMGlobalToLocalEnd(dm,vec,INSERT_VALUES,vecLocal));
  if (dim == 1) {
    PetscScalar **arrTo;
    PetscCall(DMDAVecGetArrayDOF(dmTo,vecTo,&arrTo));
    if (c < 0) {
      const PetscInt dofTo = -c;
      for (i=start[0]; i<start[0] + n[0] + extraPoint[0]; ++i) {
        for (d=0; d<PetscMin(dof,dofTo); ++d) {
          DMStagStencil pos;
          pos.i = i; pos.loc = loc; pos.c = d;
          PetscCall(DMStagVecGetValuesStencil(dm,vecLocal,1,&pos,&arrTo[i][d]));
        }
        for (;d<dofTo; ++d) {
          arrTo[i][d] = 0.0; /* Pad extra dof with zeros */
        }
      }
    } else {
      for (i=start[0]; i<start[0] + n[0] + extraPoint[0]; ++i) {
        DMStagStencil pos;
        pos.i = i; pos.loc = loc; pos.c = c;
        PetscCall(DMStagVecGetValuesStencil(dm,vecLocal,1,&pos,&arrTo[i][0]));
      }
    }
    PetscCall(DMDAVecRestoreArrayDOF(dmTo,vecTo,&arrTo));
  } else if (dim == 2) {
    PetscScalar ***arrTo;
    PetscCall(DMDAVecGetArrayDOF(dmTo,vecTo,&arrTo));
    if (c < 0) {
      const PetscInt dofTo = -c;
      for (j=start[1]; j<start[1] + n[1] + extraPoint[1]; ++j) {
        for (i=start[0]; i<start[0] + n[0] + extraPoint[0]; ++i) {
          for (d=0; d<PetscMin(dof,dofTo); ++d) {
            DMStagStencil pos;
            pos.i = i; pos.j = j; pos.loc = loc; pos.c = d;
            PetscCall(DMStagVecGetValuesStencil(dm,vecLocal,1,&pos,&arrTo[j][i][d]));
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
          PetscCall(DMStagVecGetValuesStencil(dm,vecLocal,1,&pos,&arrTo[j][i][0]));
        }
      }
    }
    PetscCall(DMDAVecRestoreArrayDOF(dmTo,vecTo,&arrTo));
  } else if (dim == 3) {
    PetscScalar ****arrTo;
    PetscCall(DMDAVecGetArrayDOF(dmTo,vecTo,&arrTo));
    if (c < 0) {
      const PetscInt dofTo = -c;
      for (k=start[2]; k<start[2] + n[2] + extraPoint[2]; ++k) {
        for (j=start[1]; j<start[1] + n[1] + extraPoint[1]; ++j) {
          for (i=start[0]; i<start[0] + n[0] + extraPoint[0]; ++i) {
            for (d=0; d<PetscMin(dof,dofTo); ++d) {
              DMStagStencil pos;
              pos.i = i; pos.j = j; pos.k = k; pos.loc = loc; pos.c = d;
              PetscCall(DMStagVecGetValuesStencil(dm,vecLocal,1,&pos,&arrTo[k][j][i][d]));
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
            PetscCall(DMStagVecGetValuesStencil(dm,vecLocal,1,&pos,&arrTo[k][j][i][0]));
          }
        }
      }
    }
    PetscCall(DMDAVecRestoreArrayDOF(dmTo,vecTo,&arrTo));
  } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported dimension %" PetscInt_FMT,dim);
  PetscCall(DMRestoreLocalVector(dm,&vecLocal));
  PetscFunctionReturn(0);
}

/* Transfer coordinates from a DMStag to a DMDA, specifying which location */
static PetscErrorCode DMStagTransferCoordinatesToDMDA(DM dmstag,DMStagStencilLocation loc,DM dmda)
{
  PetscInt       dim,start[DMSTAG_MAX_DIM],n[DMSTAG_MAX_DIM],extraPoint[DMSTAG_MAX_DIM],d;
  DM             dmstagCoord,dmdaCoord;
  DMType         dmstagCoordType;
  Vec            stagCoord,daCoord;
  PetscBool      daCoordIsStag,daCoordIsProduct;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dmstag,DM_CLASSID,1,DMSTAG);
  PetscValidHeaderSpecificType(dmda,DM_CLASSID,3,DMDA);
  PetscCall(DMGetDimension(dmstag,&dim));
  PetscCall(DMGetCoordinateDM(dmstag,&dmstagCoord));
  PetscCall(DMGetCoordinatesLocal(dmstag,&stagCoord)); /* Note local */
  PetscCall(DMGetCoordinateDM(dmda,&dmdaCoord));
  daCoord = NULL;
  PetscCall(DMGetCoordinates(dmda,&daCoord));
  if (!daCoord) {
    PetscCall(DMCreateGlobalVector(dmdaCoord,&daCoord));
    PetscCall(DMSetCoordinates(dmda,daCoord));
    PetscCall(VecDestroy(&daCoord));
    PetscCall(DMGetCoordinates(dmda,&daCoord));
  }
  PetscCall(DMGetType(dmstagCoord,&dmstagCoordType));
  PetscCall(PetscStrcmp(dmstagCoordType,DMSTAG,&daCoordIsStag));
  PetscCall(PetscStrcmp(dmstagCoordType,DMPRODUCT,&daCoordIsProduct));
  PetscCall(DMStagGetCorners(dmstag,&start[0],&start[1],&start[2],&n[0],&n[1],&n[2],NULL,NULL,NULL));
  PetscCall(DMStagDMDAGetExtraPoints(dmstag,loc,extraPoint));
  if (dim == 1) {
    PetscInt ex;
    PetscScalar **cArrDa;
    PetscCall(DMDAVecGetArrayDOF(dmdaCoord,daCoord,&cArrDa));
    if (daCoordIsStag)  {
      PetscInt slot;
      PetscScalar **cArrStag;
      PetscCall(DMStagGetLocationSlot(dmstagCoord,loc,0,&slot));
      PetscCall(DMStagVecGetArrayRead(dmstagCoord,stagCoord,&cArrStag));
      for (ex=start[0]; ex<start[0] + n[0] + extraPoint[0]; ++ex) {
        cArrDa[ex][0] = cArrStag[ex][slot];
      }
      PetscCall(DMStagVecRestoreArrayRead(dmstagCoord,stagCoord,&cArrStag));
    } else if (daCoordIsProduct) {
      PetscScalar **cArrX;
      PetscCall(DMStagGetProductCoordinateArraysRead(dmstag,&cArrX,NULL,NULL));
      for (ex=start[0]; ex<start[0] + n[0] + extraPoint[0]; ++ex) {
        cArrDa[ex][0] = cArrX[ex][0];
      }
      PetscCall(DMStagRestoreProductCoordinateArraysRead(dmstag,&cArrX,NULL,NULL));
    } else SETERRQ(PetscObjectComm((PetscObject)dmstag),PETSC_ERR_SUP,"Stag to DA coordinate transfer only supported for DMStag coordinate DM of type DMstag or DMProduct");
    PetscCall(DMDAVecRestoreArrayDOF(dmdaCoord,daCoord,&cArrDa));
  } else if (dim == 2) {
    PetscInt ex,ey;
    PetscScalar ***cArrDa;
    PetscCall(DMDAVecGetArrayDOF(dmdaCoord,daCoord,&cArrDa));
    if (daCoordIsStag)  {
      PetscInt slot;
      PetscScalar ***cArrStag;
      PetscCall(DMStagGetLocationSlot(dmstagCoord,loc,0,&slot));
      PetscCall(DMStagVecGetArrayRead(dmstagCoord,stagCoord,&cArrStag));
      for (ey=start[1]; ey<start[1] + n[1] + extraPoint[1]; ++ey) {
        for (ex=start[0]; ex<start[0] + n[0] + extraPoint[0]; ++ex) {
          for (d=0; d<2; ++d) {
            cArrDa[ey][ex][d] = cArrStag[ey][ex][slot+d];
          }
        }
      }
      PetscCall(DMStagVecRestoreArrayRead(dmstagCoord,stagCoord,&cArrStag));
    } else if (daCoordIsProduct) {
      PetscScalar **cArrX,**cArrY;
      PetscCall(DMStagGetProductCoordinateArraysRead(dmstag,&cArrX,&cArrY,NULL));
      for (ey=start[1]; ey<start[1] + n[1] + extraPoint[1]; ++ey) {
        for (ex=start[0]; ex<start[0] + n[0] + extraPoint[0]; ++ex) {
          cArrDa[ey][ex][0] = cArrX[ex][0];
          cArrDa[ey][ex][1] = cArrY[ey][0];
        }
      }
      PetscCall(DMStagRestoreProductCoordinateArraysRead(dmstag,&cArrX,&cArrY,NULL));
    } else SETERRQ(PetscObjectComm((PetscObject)dmstag),PETSC_ERR_SUP,"Stag to DA coordinate transfer only supported for DMStag coordinate DM of type DMstag or DMProduct");
    PetscCall(DMDAVecRestoreArrayDOF(dmdaCoord,daCoord,&cArrDa));
  }  else if (dim == 3) {
    PetscInt ex,ey,ez;
    PetscScalar ****cArrDa;
    PetscCall(DMDAVecGetArrayDOF(dmdaCoord,daCoord,&cArrDa));
    if (daCoordIsStag)  {
      PetscInt slot;
      PetscScalar ****cArrStag;
      PetscCall(DMStagGetLocationSlot(dmstagCoord,loc,0,&slot));
      PetscCall(DMStagVecGetArrayRead(dmstagCoord,stagCoord,&cArrStag));
      for (ez=start[2]; ez<start[2] + n[2] + extraPoint[2]; ++ez) {
        for (ey=start[1]; ey<start[1] + n[1] + extraPoint[1]; ++ey) {
          for (ex=start[0]; ex<start[0] + n[0] + extraPoint[0]; ++ex) {
            for (d=0; d<3; ++d) {
              cArrDa[ez][ey][ex][d] = cArrStag[ez][ey][ex][slot+d];
            }
          }
        }
      }
      PetscCall(DMStagVecRestoreArrayRead(dmstagCoord,stagCoord,&cArrStag));
    } else if (daCoordIsProduct) {
      PetscScalar **cArrX,**cArrY,**cArrZ;
      PetscCall(DMStagGetProductCoordinateArraysRead(dmstag,&cArrX,&cArrY,&cArrZ));
      for (ez=start[2]; ez<start[2] + n[2] + extraPoint[2]; ++ez) {
        for (ey=start[1]; ey<start[1] + n[1] + extraPoint[1]; ++ey) {
          for (ex=start[0]; ex<start[0] + n[0] + extraPoint[0]; ++ex) {
            cArrDa[ez][ey][ex][0] = cArrX[ex][0];
            cArrDa[ez][ey][ex][1] = cArrY[ey][0];
            cArrDa[ez][ey][ex][2] = cArrZ[ez][0];
          }
        }
      }
      PetscCall(DMStagRestoreProductCoordinateArraysRead(dmstag,&cArrX,&cArrY,&cArrZ));
    } else SETERRQ(PetscObjectComm((PetscObject)dmstag),PETSC_ERR_SUP,"Stag to DA coordinate transfer only supported for DMStag coordinate DM of type DMstag or DMProduct");
    PetscCall(DMDAVecRestoreArrayDOF(dmdaCoord,daCoord,&cArrDa));
  } else SETERRQ(PetscObjectComm((PetscObject)dmstag),PETSC_ERR_SUP,"Unsupported dimension %" PetscInt_FMT,dim);
  PetscFunctionReturn(0);
}

/*@C
  DMStagVecSplitToDMDA - create a DMDA and `Vec` from a DMStag and `Vec`

  Logically Collective

  High-level helper function which accepts a DMStag, a global vector, and location/dof,
  and generates a corresponding DMDA and Vec.

  Input Parameters:
+ dm - the DMStag object
. vec- Vec object associated with `dm`
. loc - which subgrid to extract (see `DMStagStencilLocation`)
- c - which component to extract (see note below)

  Output Parameters:
+ pda - the new `DMDA`
- pdavec - the new `Vec`

  Notes:
  If a `c` value of `-k` is provided, the first `k` DOF for that position are extracted,
  padding with zero values if needbe. If a non-negative value is provided, a single
  DOF is extracted.

  The caller is responsible for destroying the created `DMDA` and `Vec`.

  Level: advanced

.seealso: `DMSTAG`, `DMDA`, `DMStagMigrateVec()`, `DMStagCreateCompatibleDMStag()`
@*/
PetscErrorCode DMStagVecSplitToDMDA(DM dm,Vec vec,DMStagStencilLocation loc,PetscInt c,DM *pda,Vec *pdavec)
{
  PetscInt        dim,locdof;
  DM              da,coordDM;
  Vec             davec;
  DMStagStencilLocation locCanonical;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscCall(DMGetDimension(dm,&dim));
  PetscCall(DMStagGetLocationDOF(dm,loc,&locdof));
  PetscCheck(c < locdof,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Location %s has %" PetscInt_FMT " dof, but component %" PetscInt_FMT " requested",DMStagStencilLocations[loc],locdof,c);
  PetscCall(DMStagStencilLocationCanonicalize(loc,&locCanonical));
  PetscCall(DMStagCreateCompatibleDMDA(dm,locCanonical,c,pda));
  da = *pda;
  PetscCall(DMSetUp(*pda));
  if (dm->coordinateDM != NULL) {
    PetscCall(DMGetCoordinateDM(dm,&coordDM));
    PetscCall(DMStagTransferCoordinatesToDMDA(dm,locCanonical,da));
  }
  PetscCall(DMCreateGlobalVector(da,pdavec));
  davec = *pdavec;
  PetscCall(DMStagMigrateVecDMDA(dm,vec,locCanonical,c,da,davec));
  PetscFunctionReturn(0);
}
