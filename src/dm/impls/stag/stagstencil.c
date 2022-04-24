/* Functions concerning getting and setting Vec and Mat values with DMStagStencil */
#include <petsc/private/dmstagimpl.h>

/* Strings corresponding the the types defined in $PETSC_DIR/include/petscdmstag.h */
const char *const DMStagStencilTypes[] = {"NONE","STAR","BOX","DMStagStencilType","DM_STAG_STENCIL_",NULL};

/* Strings corresponding the positions in $PETSC_DIR/include/petscdmstag.h */
const char * const DMStagStencilLocations[] = {"NONE","BACK_DOWN_LEFT","BACK_DOWN","BACK_DOWN_RIGHT","BACK_LEFT","BACK","BACK_RIGHT","BACK_UP_LEFT","BACK_UP","BACK_UP_RIGHT","DOWN_LEFT","DOWN","DOWN_RIGHT","LEFT","ELEMENT","RIGHT","UP_LEFT","UP","UP_RIGHT","FRONT_DOWN_LEFT","FRONT_DOWN","FRONT_DOWN_RIGHT","FRONT_LEFT","FRONT","FRONT_RIGHT","FRONT_UP_LEFT","FRONT_UP","FRONT_UP_RIGHT","DMStagStencilLocation","",NULL};

/*@C
  DMStagCreateISFromStencils - Create an IS, using global numberings, for a subset of DOF in a DMStag object

  Collective

  Input Parameters:
+ dm - the DMStag object
. nStencil - the number of stencils provided
- stencils - an array of DMStagStencil objects (i,j, and k are ignored)

  Output Parameter:
. is - the global IS

  Note:
  Redundant entries in s are ignored

  Level: advanced

.seealso: DMSTAG, IS, DMStagStencil, DMCreateGlobalVector
@*/
PetscErrorCode DMStagCreateISFromStencils(DM dm,PetscInt nStencil,DMStagStencil* stencils,IS *is)
{
  DMStagStencil          *ss;
  PetscInt               *idx,*idxLocal;
  const PetscInt         *ltogidx;
  PetscInt               p,p2,pmax,i,j,k,d,dim,count,nidx;
  ISLocalToGlobalMapping ltog;
  PetscInt               start[DMSTAG_MAX_DIM],n[DMSTAG_MAX_DIM],extraPoint[DMSTAG_MAX_DIM];

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm,&dim));
  PetscCheck(dim >= 1 && dim <= 3,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported dimension %" PetscInt_FMT,dim);

  /* Only use non-redundant stencils */
  PetscCall(PetscMalloc1(nStencil,&ss));
  pmax = 0;
  for (p=0; p<nStencil; ++p) {
    PetscBool skip = PETSC_FALSE;
    DMStagStencil stencilPotential = stencils[p];
    PetscCall(DMStagStencilLocationCanonicalize(stencils[p].loc,&stencilPotential.loc));
    for (p2=0; p2<pmax; ++p2) { /* Quadratic complexity algorithm in nStencil */
      if (stencilPotential.loc == ss[p2].loc && stencilPotential.c == ss[p2].c) {
        skip = PETSC_TRUE;
        break;
      }
    }
    if (!skip) {
      ss[pmax] = stencilPotential;
      ++pmax;
    }
  }

  PetscCall(PetscMalloc1(pmax,&idxLocal));
  PetscCall(DMGetLocalToGlobalMapping(dm,&ltog));
  PetscCall(ISLocalToGlobalMappingGetIndices(ltog,&ltogidx));
  PetscCall(DMStagGetCorners(dm,&start[0],&start[1],&start[2],&n[0],&n[1],&n[2],&extraPoint[0],&extraPoint[1],&extraPoint[2]));
  for (d=dim; d<DMSTAG_MAX_DIM; ++d) {
    start[d]      = 0;
    n[d]          = 1; /* To allow for a single loop nest below */
    extraPoint[d] = 0;
  }
  nidx = pmax; for (d=0; d<dim; ++d) nidx *= (n[d]+1); /* Overestimate (always assumes extraPoint) */
  PetscCall(PetscMalloc1(nidx,&idx));
  count = 0;
  /* Note that unused loop variables are not accessed, for lower dimensions */
  for (k=start[2]; k<start[2]+n[2]+extraPoint[2]; ++k) {
    for (j=start[1]; j<start[1]+n[1]+extraPoint[1]; ++j) {
      for (i=start[0]; i<start[0]+n[0]+extraPoint[0]; ++i) {
        for (p=0; p<pmax; ++p) {
          ss[p].i = i; ss[p].j = j; ss[p].k = k;
        }
        PetscCall(DMStagStencilToIndexLocal(dm,dim,pmax,ss,idxLocal));
        for (p=0; p<pmax; ++p) {
          const PetscInt gidx = ltogidx[idxLocal[p]];
          if (gidx >= 0) {
            idx[count] = gidx;
            ++count;
          }
        }
      }
    }
  }
  PetscCall(ISLocalToGlobalMappingRestoreIndices(ltog,&ltogidx));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm),count,idx,PETSC_OWN_POINTER,is));

  PetscCall(PetscFree(ss));
  PetscCall(PetscFree(idxLocal));
  PetscFunctionReturn(0);
}

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
  const DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt              dim;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscCall(DMGetDimension(dm,&dim));
  switch (dim) {
    case 1:
      switch (loc) {
        case DMSTAG_LEFT:
        case DMSTAG_RIGHT:
          *dof = stag->dof[0]; break;
        case DMSTAG_ELEMENT:
          *dof = stag->dof[1]; break;
        default : SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented for location %s",DMStagStencilLocations[loc]);
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
        default : SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented for location %s",DMStagStencilLocations[loc]);
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
        default : SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented for location %s",DMStagStencilLocations[loc]);
      }
      break;
    default : SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported dimension %" PetscInt_FMT,dim);
  }
  PetscFunctionReturn(0);
}

/*
Convert to a location value with only BACK, DOWN, LEFT, and ELEMENT involved
*/
PETSC_INTERN PetscErrorCode DMStagStencilLocationCanonicalize(DMStagStencilLocation loc,DMStagStencilLocation *locCanonical)
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
  PetscInt       dim;
  PetscInt       *ir,*ic;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  PetscCall(DMGetDimension(dm,&dim));
  PetscCall(PetscMalloc2(nRow,&ir,nCol,&ic));
  PetscCall(DMStagStencilToIndexLocal(dm,dim,nRow,posRow,ir));
  PetscCall(DMStagStencilToIndexLocal(dm,dim,nCol,posCol,ic));
  PetscCall(MatGetValuesLocal(mat,nRow,ir,nCol,ic,val));
  PetscCall(PetscFree2(ir,ic));
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
  PetscInt       *ir,*ic;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  PetscCall(PetscMalloc2(nRow,&ir,nCol,&ic));
  PetscCall(DMStagStencilToIndexLocal(dm,dm->dim,nRow,posRow,ir));
  PetscCall(DMStagStencilToIndexLocal(dm,dm->dim,nCol,posCol,ic));
  PetscCall(MatSetValuesLocal(mat,nRow,ir,nCol,ic,val,insertMode));
  PetscCall(PetscFree2(ir,ic));
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
  } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %" PetscInt_FMT,dim);
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

  This approach is not as efficient as getting values directly with DMStagVecGetArray(), which is recommended for matrix free operators.

  Level: advanced

.seealso: DMSTAG, DMStagStencil, DMStagStencilLocation, DMStagVecSetValuesStencil(), DMStagMatSetValuesStencil(), DMStagVecGetArray()
@*/
PetscErrorCode DMStagVecGetValuesStencil(DM dm, Vec vec,PetscInt n,const DMStagStencil *pos,PetscScalar *val)
{
  DM_Stag * const   stag = (DM_Stag*)dm->data;
  PetscInt          nLocal,idx;
  PetscInt          *ix;
  PetscScalar const *arr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscCall(VecGetLocalSize(vec,&nLocal));
  PetscCheck(nLocal == stag->entriesGhost,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Vector should be a local vector. Local size %" PetscInt_FMT " does not match expected %" PetscInt_FMT,nLocal,stag->entriesGhost);
  PetscCall(PetscMalloc1(n,&ix));
  PetscCall(DMStagStencilToIndexLocal(dm,dm->dim,n,pos,ix));
  PetscCall(VecGetArrayRead(vec,&arr));
  for (idx=0; idx<n; ++idx) val[idx] = arr[ix[idx]];
  PetscCall(VecRestoreArrayRead(vec,&arr));
  PetscCall(PetscFree(ix));
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
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        nLocal;
  PetscInt        *ix;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscCall(VecGetLocalSize(vec,&nLocal));
  PetscCheck(nLocal == stag->entries,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONG,"Provided vec has a different number of local entries (%" PetscInt_FMT ") than expected (%" PetscInt_FMT "). It should be a global vector",nLocal,stag->entries);
  PetscCall(PetscMalloc1(n,&ix));
  PetscCall(DMStagStencilToIndexLocal(dm,dm->dim,n,pos,ix));
  PetscCall(VecSetValuesLocal(vec,n,ix,val,insertMode));
  PetscCall(PetscFree(ix));
  PetscFunctionReturn(0);
}
