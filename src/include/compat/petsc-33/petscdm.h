#undef  __FUNCT__
#define __FUNCT__ "DMGetCoordinateDM"
PetscErrorCode DMGetCoordinateDM(DM dm,DM *cdm)
{
  PetscBool match;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(cdm,2);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMDA,&match);CHKERRQ(ierr);
  if (!match) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  ierr = DMDAGetCoordinateDA(dm,cdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMSetCoordinates"
PetscErrorCode DMSetCoordinates(DM dm,Vec c)
{
  PetscBool match;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(c,VEC_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMDA,&match);CHKERRQ(ierr);
  if (!match) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  ierr = DMDASetCoordinates(dm,c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMGetCoordinates"
PetscErrorCode DMGetCoordinates(DM dm,Vec *c)
{
  PetscBool match;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(c,2);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMDA,&match);CHKERRQ(ierr);
  if (!match) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  ierr = DMDAGetCoordinates(dm,c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMSetCoordinatesLocal"
PetscErrorCode DMSetCoordinatesLocal(DM dm,Vec c)
{
  PetscBool match;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(c,VEC_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMDA,&match);CHKERRQ(ierr);
  if (!match) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  ierr = DMDASetGhostedCoordinates(dm,c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMGetCoordinatesLocal"
PetscErrorCode DMGetCoordinatesLocal(DM dm,Vec *c)
{
  PetscBool match;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(c,2);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMDA,&match);CHKERRQ(ierr);
  if (!match) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  ierr = DMDAGetGhostedCoordinates(dm,c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMCompositeScatterArray"
PetscErrorCode DMCompositeScatterArray(DM dm,Vec g,Vec *lvecs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(g,VEC_CLASSID,2);
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMCompositeGatherArray"
PetscErrorCode DMCompositeGatherArray(DM dm,Vec g,InsertMode imode,Vec *lvecs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(g,VEC_CLASSID,2);
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMCompositeGetAccessArray"
PetscErrorCode DMCompositeGetAccessArray(DM dm,Vec g,PetscInt n,const PetscInt *locs,Vec *vecs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(g,VEC_CLASSID,2);
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMCompositeRestoreAccessArray"
PetscErrorCode DMCompositeRestoreAccessArray(DM dm,Vec g,PetscInt n,const PetscInt *locs,Vec *vecs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(g,VEC_CLASSID,2);
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(0);
}
