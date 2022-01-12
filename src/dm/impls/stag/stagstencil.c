/* Functions concerning getting and setting Vec and Mat values with DMStagStencil */
#include <petsc/private/dmstagimpl.h>

/* Strings corresponding the the types defined in $PETSC_DIR/include/petscdmstag.h */
const char *const DMStagStencilTypes[] = {"NONE","STAR","BOX","DMStagStencilType","DM_STAG_STENCIL_",NULL};

/* Strings corresponding the positions in $PETSC_DIR/include/petscdmstag.h */
const char * const DMStagStencilLocations[] = {"NONE","BACK_DOWN_LEFT","BACK_DOWN","BACK_DOWN_RIGHT","BACK_LEFT","BACK","BACK_RIGHT","BACK_UP_LEFT","BACK_UP","BACK_UP_RIGHT","DOWN_LEFT","DOWN","DOWN_RIGHT","LEFT","ELEMENT","RIGHT","UP_LEFT","UP","UP_RIGHT","FRONT_DOWN_LEFT","FRONT_DOWN","FRONT_DOWN_RIGHT","FRONT_LEFT","FRONT","FRONT_RIGHT","FRONT_UP_LEFT","FRONT_UP","FRONT_UP_RIGHT"};
/*@C
  DMStagGetLocationDOF - Get number of DOF associated with a given point in a DMStag grid

  Not Collective

  Input Parameters:
+ dm - the DMStag object
- loc - grid point (see DMStagStencilLocation)

  Output Parameter:
. dof - the number of dof (components) living at loc in dm

  Level: intermediate

.seealso: DMSTAG, DMStagStencilLocation, DMStagStencil, DMDAGetDof()
@*/
PetscErrorCode DMStagGetLocationDOF(DM dm,DMStagStencilLocation loc,PetscInt *dof)
{
  PetscErrorCode        ierr;
  const DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt              dim;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 1:
      switch (loc) {
        case DMSTAG_LEFT:
        case DMSTAG_RIGHT:
          *dof = stag->dof[0]; break;
        case DMSTAG_ELEMENT:
          *dof = stag->dof[1]; break;
        default : SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented for location %s",DMStagStencilLocations[loc]);
      }
      break;
    case 2:
      switch (loc) {
        case DMSTAG_DOWN_LEFT:
        case DMSTAG_DOWN_RIGHT:
        case DMSTAG_UP_LEFT:
        case DMSTAG_UP_RIGHT:
          *dof = stag->dof[0]; break;
        case DMSTAG_LEFT:
        case DMSTAG_RIGHT:
        case DMSTAG_UP:
        case DMSTAG_DOWN:
          *dof = stag->dof[1]; break;
        case DMSTAG_ELEMENT:
          *dof = stag->dof[2]; break;
        default : SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented for location %s",DMStagStencilLocations[loc]);
      }
      break;
    case 3:
      switch (loc) {
        case DMSTAG_BACK_DOWN_LEFT:
        case DMSTAG_BACK_DOWN_RIGHT:
        case DMSTAG_BACK_UP_LEFT:
        case DMSTAG_BACK_UP_RIGHT:
        case DMSTAG_FRONT_DOWN_LEFT:
        case DMSTAG_FRONT_DOWN_RIGHT:
        case DMSTAG_FRONT_UP_LEFT:
        case DMSTAG_FRONT_UP_RIGHT:
          *dof = stag->dof[0]; break;
        case DMSTAG_BACK_DOWN:
        case DMSTAG_BACK_LEFT:
        case DMSTAG_BACK_RIGHT:
        case DMSTAG_BACK_UP:
        case DMSTAG_DOWN_LEFT:
        case DMSTAG_DOWN_RIGHT:
        case DMSTAG_UP_LEFT:
        case DMSTAG_UP_RIGHT:
        case DMSTAG_FRONT_DOWN:
        case DMSTAG_FRONT_LEFT:
        case DMSTAG_FRONT_RIGHT:
        case DMSTAG_FRONT_UP:
          *dof = stag->dof[1]; break;
        case DMSTAG_LEFT:
        case DMSTAG_RIGHT:
        case DMSTAG_DOWN:
        case DMSTAG_UP:
        case DMSTAG_BACK:
        case DMSTAG_FRONT:
          *dof = stag->dof[2]; break;
        case DMSTAG_ELEMENT:
          *dof = stag->dof[3]; break;
        default : SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented for location %s",DMStagStencilLocations[loc]);
      }
      break;
    default : SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported dimension %D",dim);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMStagMatGetValuesStencil - retrieve local matrix entries using grid indexing

  Not Collective

  Input Parameters:
+ dm - the DMStag object
. mat - the matrix
. nRow - number of rows
. posRow - grid locations (including components) of rows
. nCol - number of columns
- posCol - grid locations (including components) of columns

  Output Parameter:
. val - logically two-dimensional array of values

  Level: advanced

.seealso: DMSTAG, DMStagStencil, DMStagStencilLocation, DMStagVecGetValuesStencil(), DMStagVecSetValuesStencil(), DMStagMatSetValuesStencil(), MatSetValuesStencil(), MatAssemblyBegin(), MatAssemblyEnd(), DMCreateMatrix()
@*/
PetscErrorCode DMStagMatGetValuesStencil(DM dm,Mat mat,PetscInt nRow,const DMStagStencil *posRow,PetscInt nCol,const DMStagStencil *posCol,PetscScalar *val)
{
  PetscErrorCode ierr;
  PetscInt       dim;
  PetscInt       *ir,*ic;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = PetscMalloc2(nRow,&ir,nCol,&ic);CHKERRQ(ierr);
  ierr = DMStagStencilToIndexLocal(dm,dim,nRow,posRow,ir);CHKERRQ(ierr);
  ierr = DMStagStencilToIndexLocal(dm,dim,nCol,posCol,ic);CHKERRQ(ierr);
  ierr = MatGetValuesLocal(mat,nRow,ir,nCol,ic,val);CHKERRQ(ierr);
  ierr = PetscFree2(ir,ic);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMStagMatSetValuesStencil - insert or add matrix entries using grid indexing

  Not Collective

  Input Parameters:
+ dm - the DMStag object
. mat - the matrix
. nRow - number of rows
. posRow - grid locations (including components) of rows
. nCol - number of columns
. posCol - grid locations (including components) of columns
. val - logically two-dimensional array of values
- insertmode - INSERT_VALUES or ADD_VALUES

  Notes:
  See notes for MatSetValuesStencil()

  Level: intermediate

.seealso: DMSTAG, DMStagStencil, DMStagStencilLocation, DMStagVecGetValuesStencil(), DMStagVecSetValuesStencil(), DMStagMatGetValuesStencil(), MatSetValuesStencil(), MatAssemblyBegin(), MatAssemblyEnd(), DMCreateMatrix()
@*/
PetscErrorCode DMStagMatSetValuesStencil(DM dm,Mat mat,PetscInt nRow,const DMStagStencil *posRow,PetscInt nCol,const DMStagStencil *posCol,const PetscScalar *val,InsertMode insertMode)
{
  PetscErrorCode ierr;
  PetscInt       dim;
  PetscInt       *ir,*ic;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = PetscMalloc2(nRow,&ir,nCol,&ic);CHKERRQ(ierr);
  ierr = DMStagStencilToIndexLocal(dm,dim,nRow,posRow,ir);CHKERRQ(ierr);
  ierr = DMStagStencilToIndexLocal(dm,dim,nCol,posCol,ic);CHKERRQ(ierr);
  ierr = MatSetValuesLocal(mat,nRow,ir,nCol,ic,val,insertMode);CHKERRQ(ierr);
  ierr = PetscFree2(ir,ic);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMStagStencilToIndexLocal - Convert an array of DMStagStencil objects to an array of indices into a local vector.

  Not Collective

  Input Parameters:
+ dm - the DMStag object
. dim - the dimension of the DMStag object
. n - the number of DMStagStencil objects
- pos - an array of n DMStagStencil objects

  Output Parameter:
. ix - output array of n indices

  Notes:
  The DMStagStencil objects in pos use global element indices.

  The .c fields in pos must always be set (even if to 0).

  Developer Notes:
  This is a "hot" function, and accepts the dimension redundantly to avoid having to perform any error checking inside the function.

  Level: developer

.seealso: DMSTAG, DMStagStencilLocation, DMStagStencil, DMGetLocalVector, DMCreateLocalVector
@*/
PetscErrorCode DMStagStencilToIndexLocal(DM dm,PetscInt dim,PetscInt n,const DMStagStencil *pos,PetscInt *ix)
{
  const DM_Stag * const stag = (DM_Stag*)dm->data;
  const PetscInt        epe = stag->entriesPerElement;

  PetscFunctionBeginHot;
  if (dim == 1) {
    for (PetscInt idx=0; idx<n; ++idx) {
      const PetscInt eLocal = pos[idx].i - stag->startGhost[0];

      ix[idx] = eLocal * epe + stag->locationOffsets[pos[idx].loc] + pos[idx].c;
    }
  } else if (dim == 2) {
    const PetscInt epr = stag->nGhost[0];

    for (PetscInt idx=0; idx<n; ++idx) {
      const PetscInt eLocalx = pos[idx].i - stag->startGhost[0];
      const PetscInt eLocaly = pos[idx].j - stag->startGhost[1];
      const PetscInt eLocal = eLocalx + epr*eLocaly;

      ix[idx] = eLocal * epe + stag->locationOffsets[pos[idx].loc] + pos[idx].c;
    }
  } else if (dim == 3) {
    const PetscInt epr = stag->nGhost[0];
    const PetscInt epl = stag->nGhost[0]*stag->nGhost[1];

    for (PetscInt idx=0; idx<n; ++idx) {
      const PetscInt eLocalx = pos[idx].i - stag->startGhost[0];
      const PetscInt eLocaly = pos[idx].j - stag->startGhost[1];
      const PetscInt eLocalz = pos[idx].k - stag->startGhost[2];
      const PetscInt eLocal  = epl*eLocalz + epr*eLocaly + eLocalx;

      ix[idx] = eLocal * epe + stag->locationOffsets[pos[idx].loc] + pos[idx].c;
    }
  } else SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %d",dim);
  PetscFunctionReturn(0);
}

/*@C
  DMStagVecGetValuesStencil - get vector values using grid indexing

  Not Collective

  Input Parameters:
+ dm - the DMStag object
. vec - the vector object
. n - the number of values to obtain
- pos - locations to obtain values from (as an array of DMStagStencil values)

  Output Parameter:
. val - value at the point

  Notes:
  Accepts stencils which refer to global element numbers, but
  only allows access to entries in the local representation (including ghosts).

  This approach is not as efficient as setting values directly with DMStagVecGetArray(), which is recommended for matrix free operators.

  Level: advanced

.seealso: DMSTAG, DMStagStencil, DMStagStencilLocation, DMStagVecSetValuesStencil(), DMStagMatSetValuesStencil(), DMStagVecGetArray()
@*/
PetscErrorCode DMStagVecGetValuesStencil(DM dm, Vec vec,PetscInt n,const DMStagStencil *pos,PetscScalar *val)
{
  PetscErrorCode    ierr;
  DM_Stag * const   stag = (DM_Stag*)dm->data;
  PetscInt          nLocal,dim,idx;
  PetscInt          *ix;
  PetscScalar const *arr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vec,&nLocal);CHKERRQ(ierr);
  if (nLocal != stag->entriesGhost) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Vector should be a local vector. Local size %d does not match expected %d",nLocal,stag->entriesGhost);
  ierr = PetscMalloc1(n,&ix);CHKERRQ(ierr);
  ierr = DMStagStencilToIndexLocal(dm,dim,n,pos,ix);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vec,&arr);CHKERRQ(ierr);
  for (idx=0; idx<n; ++idx) val[idx] = arr[ix[idx]];
  ierr = VecRestoreArrayRead(vec,&arr);CHKERRQ(ierr);
  ierr = PetscFree(ix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMStagVecSetValuesStencil - Set Vec values using global grid indexing

  Not Collective

  Input Parameters:
+ dm - the DMStag object
. vec - the Vec
. n - the number of values to set
. pos - the locations to set values, as an array of DMStagStencil structs
. val - the values to set
- insertMode - INSERT_VALUES or ADD_VALUES

  Notes:
  The vector is expected to be a global vector compatible with the DM (usually obtained by DMGetGlobalVector() or DMCreateGlobalVector()).

  This approach is not as efficient as setting values directly with DMStagVecGetArray(), which is recommended for matrix-free operators.
  For assembling systems, where overhead may be less important than convenience, this routine could be helpful in assembling a righthand side and a matrix (using DMStagMatSetValuesStencil()).

  Level: advanced

.seealso: DMSTAG, DMStagStencil, DMStagStencilLocation, DMStagVecGetValuesStencil(), DMStagMatSetValuesStencil(), DMCreateGlobalVector(), DMGetLocalVector(), DMStagVecGetArray()
@*/
PetscErrorCode DMStagVecSetValuesStencil(DM dm,Vec vec,PetscInt n,const DMStagStencil *pos,const PetscScalar *val,InsertMode insertMode)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        dim,nLocal;
  PetscInt        *ix;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vec,&nLocal);CHKERRQ(ierr);
  if (nLocal != stag->entries) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONG,"Provided vec has a different number of local entries (%D) than expected (%D). It should be a global vector",nLocal,stag->entries);
  ierr = PetscMalloc1(n,&ix);CHKERRQ(ierr);
  ierr = DMStagStencilToIndexLocal(dm,dim,n,pos,ix);CHKERRQ(ierr);
  ierr = VecSetValuesLocal(vec,n,ix,val,insertMode);CHKERRQ(ierr);
  ierr = PetscFree(ix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
