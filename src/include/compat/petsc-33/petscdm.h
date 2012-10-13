#undef  __FUNC__
#define __FUNC__ ""
PetscErrorCode DMGetCoordinateDM(DM dm,DM *cdm)
{
  PetscBool match;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(cdm,2);
  ierr = PetscObjectTypeCompare((PetscObject)dm, DMDA, &match);CHKERRQ(ierr);
  if (!match) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  ierr = DMDAGetCoordinateDA(dm,cdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNC__
#define __FUNC__ ""
PetscErrorCode DMSetCoordinates(DM dm,Vec c)
{
  PetscBool match;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(c,VEC_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)dm, DMDA, &match);CHKERRQ(ierr);
  if (!match) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  ierr = DMDASetCoordinates(dm,c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNC__
#define __FUNC__ ""
PetscErrorCode DMGetCoordinates(DM dm,Vec *c)
{
  PetscBool match;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(c,2);
  ierr = PetscObjectTypeCompare((PetscObject)dm, DMDA, &match);CHKERRQ(ierr);
  if (!match) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  ierr = DMDAGetCoordinates(dm,c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNC__
#define __FUNC__ ""
PetscErrorCode DMSetCoordinatesLocal(DM dm,Vec c)
{
  PetscBool match;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(c,VEC_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)dm, DMDA, &match);CHKERRQ(ierr);
  if (!match) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  ierr = DMDASetGhostedCoordinates(dm,c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNC__
#define __FUNC__ ""
PetscErrorCode DMGetCoordinatesLocal(DM dm,Vec *c)
{
  PetscBool match;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(c,2);
  ierr = PetscObjectTypeCompare((PetscObject)dm, DMDA, &match);CHKERRQ(ierr);
  if (!match) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  ierr = DMDAGetGhostedCoordinates(dm,c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
