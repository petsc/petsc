#include <petsc-private/dmmbimpl.h> /*I  "petscdm.h"   I*/

#include <petscdmmoab.h>

#undef __FUNCT__
#define __FUNCT__ "DMMoabSetFieldVector"
PetscErrorCode DMMoabSetFieldVector(DM dm, PetscInt ifield, Vec fvec)
{
  DM_Moab        *dmmoab;
  moab::Tag     vtag,ntag;
  const PetscScalar *varray;
  PetscScalar *farray;
  moab::ErrorCode merr;
  PetscErrorCode  ierr;
  std::string tag_name;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dmmoab = (DM_Moab*)(dm)->data;

  if ((ifield < 0) || (ifield >= dmmoab->numFields)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "The field %d should be positive and less than %d.", ifield, dmmoab->numFields);

  /* Create a tag in MOAB mesh to index and keep track of number of Petsc vec tags */
  merr = dmmoab->mbiface->tag_get_handle(dmmoab->fieldNames[ifield],1,moab::MB_TYPE_DOUBLE,ntag,
                                          moab::MB_TAG_DENSE|moab::MB_TAG_CREAT);MBERRNM(merr);

  ierr = DMMoabGetVecTag(fvec,&vtag);CHKERRQ(ierr);

  merr = dmmoab->mbiface->tag_get_name(vtag, tag_name);
  if (!tag_name.length() && merr !=moab::MB_SUCCESS) {
    ierr = VecGetArrayRead(fvec,&varray);CHKERRQ(ierr);
    /* use the entity handle and the Dof index to set the right value */
    merr = dmmoab->mbiface->tag_set_data(ntag, *dmmoab->vowned, (const void*)varray);MBERRNM(merr);
    ierr = VecRestoreArrayRead(fvec,&varray);CHKERRQ(ierr);
  }
  else {
    ierr = PetscMalloc(dmmoab->nloc*sizeof(PetscScalar),&farray);CHKERRQ(ierr);
    /* we are using a MOAB Vec - directly copy the tag data to new one */
    merr = dmmoab->mbiface->tag_get_data(vtag, *dmmoab->vowned, (void*)farray);MBERRNM(merr);
    merr = dmmoab->mbiface->tag_set_data(ntag, *dmmoab->vowned, (const void*)farray);MBERRNM(merr);
    /* make sure the parallel exchange for ghosts are done appropriately */
    ierr = PetscFree(farray);CHKERRQ(ierr);
  }
  merr = dmmoab->pcomm->exchange_tags(ntag, *dmmoab->vowned);MBERRNM(merr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabSetGlobalFieldVector"
PetscErrorCode DMMoabSetGlobalFieldVector(DM dm, Vec fvec)
{
  DM_Moab        *dmmoab;
  moab::Tag     vtag,ntag;
  const PetscScalar   *varray;
  PetscScalar   *farray;
  moab::ErrorCode merr;
  PetscErrorCode  ierr;
  PetscInt i,ifield;
  std::string tag_name;
  moab::Range::iterator iter;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dmmoab = (DM_Moab*)(dm)->data;

  /* get the Tag corresponding to the global vector - possible that there is no tag associated.. */
  ierr = DMMoabGetVecTag(fvec,&vtag);CHKERRQ(ierr);
  merr = dmmoab->mbiface->tag_get_name(vtag, tag_name);
  ierr = PetscMalloc(dmmoab->nloc*sizeof(PetscScalar),&farray);CHKERRQ(ierr);
  if (!tag_name.length() && merr !=moab::MB_SUCCESS) {
    /* not a MOAB vector - use VecGetSubVector to get the parts as needed */

    ierr = VecGetArrayRead(fvec,&varray);CHKERRQ(ierr);
    for (ifield=0; ifield<dmmoab->numFields; ++ifield) {

      /* Create a tag in MOAB mesh to index and keep track of number of Petsc vec tags */
      merr = dmmoab->mbiface->tag_get_handle(dmmoab->fieldNames[ifield],1,moab::MB_TYPE_DOUBLE,ntag,
                                            moab::MB_TAG_DENSE|moab::MB_TAG_CREAT);MBERRNM(merr);

      for(i=0;i<dmmoab->nloc;i++) {
        if (dmmoab->bs == 1)
          farray[i]=varray[ifield*dmmoab->nloc+i];
        else
          farray[i]=varray[i*dmmoab->numFields+ifield];
      }

      /* use the entity handle and the Dof index to set the right value */
      merr = dmmoab->mbiface->tag_set_data(ntag, *dmmoab->vowned, (const void*)farray);MBERRNM(merr);
    }
    ierr = VecRestoreArrayRead(fvec,&varray);CHKERRQ(ierr);
  }
  else {
    ierr = PetscMalloc(dmmoab->nloc*dmmoab->bs*sizeof(PetscScalar),&varray);CHKERRQ(ierr);

    /* we are using a MOAB Vec - directly copy the tag data to new one */
    merr = dmmoab->mbiface->tag_get_data(vtag, *dmmoab->vowned, (void*)varray);MBERRNM(merr);
    for (ifield=0; ifield<dmmoab->numFields; ++ifield) {

      /* Create a tag in MOAB mesh to index and keep track of number of Petsc vec tags */
      merr = dmmoab->mbiface->tag_get_handle(dmmoab->fieldNames[ifield],1,moab::MB_TYPE_DOUBLE,ntag,
                                            moab::MB_TAG_DENSE|moab::MB_TAG_CREAT);MBERRNM(merr);

      /* we are using a MOAB Vec - directly copy the tag data to new one */
      for(i=0; i < dmmoab->nloc; i++) {
        farray[i] = varray[i*dmmoab->bs+ifield];
      }

      merr = dmmoab->mbiface->tag_set_data(ntag, *dmmoab->vowned, (const void*)farray);MBERRNM(merr);
      /* make sure the parallel exchange for ghosts are done appropriately */
      merr = dmmoab->pcomm->exchange_tags(ntag, *dmmoab->vlocal);MBERRNM(merr);
    }
    ierr = PetscFree(varray);CHKERRQ(ierr);
  }
  ierr = PetscFree(farray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabSetFields"
PetscErrorCode DMMoabSetFields(DM dm,PetscInt numFields,const char** fields)
{
  PetscErrorCode ierr;
  PetscInt       i;
  DM_Moab        *dmmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dmmoab = (DM_Moab*)(dm)->data;

  /* first deallocate the existing field structure */
  if (dmmoab->fieldNames) {
    for(i=0; i<dmmoab->numFields; i++) {
      ierr = PetscFree(dmmoab->fieldNames[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(dmmoab->fieldNames);CHKERRQ(ierr);
  }

  /* now re-allocate and assign field names  */
  dmmoab->numFields = numFields;
  ierr = PetscMalloc(sizeof(char*)*numFields,&dmmoab->fieldNames);CHKERRQ(ierr);
  if (fields) {
    for(i=0; i<dmmoab->numFields; i++) {
      ierr = PetscStrallocpy(fields[i], (char**) &dmmoab->fieldNames[i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMoabSetFieldName"
/*@C
  DMMoabSetFieldName - Sets the name of a field in the DM

  Not Collective

  Input Parameters:
+ dm     - the DM object
. field - the field number
- fieldName - the field name

  Level: developer
  Note: Needs to be called after DMMoabSetFields with correct numFields

.seealso: DMMoabSetFields()
@*/
PetscErrorCode DMMoabSetFieldName(DM dm, PetscInt field, const char fieldName[])
{
  PetscErrorCode ierr;
  DM_Moab        *dmmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidCharPointer(fieldName,3);
  dmmoab = (DM_Moab*)(dm)->data;

  if ((field < 0) || (field >= dmmoab->numFields)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "DM field %d should be in [%d, %d)", field, 0, dmmoab->numFields);
  ierr = PetscFree(dmmoab->fieldNames[field]);CHKERRQ(ierr);
  ierr = PetscStrallocpy(fieldName, (char**) &dmmoab->fieldNames[field]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetFieldDof"
PetscErrorCode DMMoabGetFieldDof(DM dm,moab::EntityHandle point,PetscInt field,PetscInt* dof)
{
  DM_Moab        *dmmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dmmoab = (DM_Moab*)(dm)->data;

  *dof=dmmoab->gidmap[(PetscInt)point];
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetFieldDofs"
PetscErrorCode DMMoabGetFieldDofs(DM dm,PetscInt npoints,const moab::EntityHandle* points,PetscInt field,PetscInt* dof)
{
  PetscInt        i;
  PetscErrorCode  ierr;
  DM_Moab        *dmmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(points,2);
  dmmoab = (DM_Moab*)(dm)->data;

  if (!dof) {
    ierr = PetscMalloc(sizeof(PetscInt)*npoints, &dof);CHKERRQ(ierr);
  }

  /* first get the local indices */
  if (dmmoab->bs > 1) {
    for (i=0; i<npoints; ++i)
      dof[i] = dmmoab->gidmap[(PetscInt)points[i]]*dmmoab->numFields+field;
  }
  else {
    /* assume all fields have equal distribution */
    for (i=0; i<npoints; ++i)
      dof[i] = dmmoab->gidmap[(PetscInt)points[i]]+field*dmmoab->n;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetFieldDofsLocal"
PetscErrorCode DMMoabGetFieldDofsLocal(DM dm,PetscInt npoints,const moab::EntityHandle* points,PetscInt field,PetscInt* dof)
{
  PetscInt i,offset;
  PetscErrorCode  ierr;
  DM_Moab        *dmmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(points,2);
  dmmoab = (DM_Moab*)(dm)->data;

  if (!dof) {
    ierr = PetscMalloc(sizeof(PetscInt)*npoints, &dof);CHKERRQ(ierr);
  }

  if (dmmoab->bs > 1) {
    for (i=0; i<npoints; ++i)
      dof[i] = dmmoab->lidmap[(PetscInt)points[i]]*dmmoab->numFields+field;
  }
  else {
    offset = field*dmmoab->n; /* assume all fields have equal distribution */
    for (i=0; i<npoints; ++i)
      dof[i] = dmmoab->lidmap[(PetscInt)points[i]]+offset;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetDofs"
PetscErrorCode DMMoabGetDofs(DM dm,PetscInt npoints,const moab::EntityHandle* points,PetscInt* dof)
{
  PetscInt        i,field,offset;  
  PetscErrorCode  ierr;
  DM_Moab        *dmmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(points,2);
  dmmoab = (DM_Moab*)(dm)->data;

  if (!dof) {
    ierr = PetscMalloc(sizeof(PetscInt)*dmmoab->numFields*npoints, &dof);CHKERRQ(ierr);
  }
  
  if (dmmoab->bs > 1) {
    for (field=0; field<dmmoab->numFields; ++field) {
      for (i=0; i<npoints; ++i)
        dof[i*dmmoab->numFields+field] = dmmoab->gidmap[(PetscInt)points[i]]*dmmoab->numFields+field;
    }
  }
  else {
    for (field=0; field<dmmoab->numFields; ++field) {
      offset = field*dmmoab->n; /* assume all fields have equal distribution */
      for (i=0; i<npoints; ++i)
        dof[i*dmmoab->numFields+field] = dmmoab->gidmap[(PetscInt)points[i]]+offset;
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetDofsLocal"
PetscErrorCode DMMoabGetDofsLocal(DM dm,PetscInt npoints,const moab::EntityHandle* points,PetscInt* dof)
{
  PetscInt        i,field,offset;
  PetscErrorCode  ierr;
  DM_Moab        *dmmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(points,2);
  dmmoab = (DM_Moab*)(dm)->data;

  if (!dof) {
    ierr = PetscMalloc(sizeof(PetscInt)*dmmoab->numFields*npoints, &dof);CHKERRQ(ierr);
  }

  if (dmmoab->bs > 1) {
    for (field=0; field<dmmoab->numFields; ++field) {
      for (i=0; i<npoints; ++i)
        dof[i*dmmoab->numFields+field] = dmmoab->lidmap[(PetscInt)points[i]]*dmmoab->numFields+field;
    }
  }
  else {
    for (field=0; field<dmmoab->numFields; ++field) {
      offset = field*dmmoab->n; /* assume all fields have equal distribution */
      for (i=0; i<npoints; ++i)
        dof[i*dmmoab->numFields+field] = dmmoab->lidmap[(PetscInt)points[i]]+offset;
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetDofsBlocked"
PetscErrorCode DMMoabGetDofsBlocked(DM dm,PetscInt npoints,const moab::EntityHandle* points,PetscInt* dof)
{
  PetscInt        i;
  DM_Moab        *dmmoab;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(points,2);
  dmmoab = (DM_Moab*)(dm)->data;

  if (!dof) {
    ierr = PetscMalloc(sizeof(PetscInt)*npoints, &dof);CHKERRQ(ierr);
  }

  for (i=0; i<npoints; ++i) {
    dof[i]=dmmoab->gidmap[(PetscInt)points[i]];
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetDofsBlockedLocal"
PetscErrorCode DMMoabGetDofsBlockedLocal(DM dm,PetscInt npoints,const moab::EntityHandle* points,PetscInt* dof)
{
  PetscInt        i;
  DM_Moab        *dmmoab;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(points,2);
  dmmoab = (DM_Moab*)(dm)->data;

  if (!dof) {
    ierr = PetscMalloc(sizeof(PetscInt)*npoints, &dof);CHKERRQ(ierr);
  }

  for (i=0; i<npoints; ++i)
    dof[i] = dmmoab->lidmap[(PetscInt)points[i]];
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetVertexDofsBlocked"
PetscErrorCode DMMoabGetVertexDofsBlocked(DM dm,PetscInt** dof)
{
  DM_Moab        *dmmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dmmoab = (DM_Moab*)(dm)->data;

  *dof = dmmoab->gidmap;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetVertexDofsBlockedLocal"
PetscErrorCode DMMoabGetVertexDofsBlockedLocal(DM dm,PetscInt** dof)
{
  DM_Moab        *dmmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(dof,2);
  dmmoab = (DM_Moab*)(dm)->data;

  *dof = dmmoab->lidmap;
  PetscFunctionReturn(0);
}



