#undef __FUNCT__
#define __FUNCT__ "DMCreateMatrix_Compat"
static PetscErrorCode DMCreateMatrix_Compat(DM dm,Mat *A)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMCreateMatrix(dm,NULL,A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define DMCreateMatrix DMCreateMatrix_Compat

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

/* --- */

#define DMPATCH "patch"
#define DMMOAB  "moab"

/* --- */

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

/* --- */

#define DMPLEX "plex"

#define DMPlexError do {                                                \
    PetscFunctionBegin;                                                 \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version"); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

#undef  __FUNCT__
#define __FUNCT__ "DMPlexCreate"
PetscErrorCode DMPlexCreate(PETSC_UNUSED MPI_Comm comm,PETSC_UNUSED DM *dm){DMPlexError;}
#undef  __FUNCT__
#define __FUNCT__ "DMPlexClone"
PetscErrorCode DMPlexClone(PETSC_UNUSED DM dm,...){DMPlexError;}
#undef  __FUNCT__
#define __FUNCT__ "DMPlexCreateFromCellList"
PetscErrorCode DMPlexCreateFromCellList(PETSC_UNUSED MPI_Comm comm,...){DMPlexError;}
#undef  __FUNCT__
#define __FUNCT__ "DMPlexGetDimension"
PetscErrorCode DMPlexGetDimension(PETSC_UNUSED DM dm,...) {DMPlexError;}
#undef  __FUNCT__
#define __FUNCT__ "DMPlexSetDimension"
PetscErrorCode DMPlexSetDimension(PETSC_UNUSED DM dm,...) {DMPlexError;}
#undef  __FUNCT__
#define __FUNCT__ "DMPlexGetChart"
PetscErrorCode DMPlexGetChart(PETSC_UNUSED DM dm,...) {DMPlexError;}
#undef  __FUNCT__
#define __FUNCT__ "DMPlexSetChart"
PetscErrorCode DMPlexSetChart(PETSC_UNUSED DM dm,...) {DMPlexError;}
#undef  __FUNCT__
#define __FUNCT__ "DMPlexGetConeSize"
PetscErrorCode DMPlexGetConeSize(PETSC_UNUSED DM dm,...) {DMPlexError;}
#undef  __FUNCT__
#define __FUNCT__ "DMPlexSetConeSize"
PetscErrorCode DMPlexSetConeSize(PETSC_UNUSED DM dm,...) {DMPlexError;}
#undef  __FUNCT__
#define __FUNCT__ "DMPlexGetCone"
PetscErrorCode DMPlexGetCone(PETSC_UNUSED DM dm,...) {DMPlexError;}
#undef  __FUNCT__
#define __FUNCT__ "DMPlexSetCone"
PetscErrorCode DMPlexSetCone(PETSC_UNUSED DM dm,...) {DMPlexError;}
#undef  __FUNCT__
#define __FUNCT__ "DMPlexSetConeOrientation"
PetscErrorCode DMPlexSetConeOrientation(PETSC_UNUSED DM dm,...) {DMPlexError;}
#undef  __FUNCT__
#define __FUNCT__ "DMPlexGetConeOrientation"
PetscErrorCode DMPlexGetConeOrientation(PETSC_UNUSED DM dm,...) {DMPlexError;}
#undef  __FUNCT__
#define __FUNCT__ "DMPlexGetSupportSize"
PetscErrorCode DMPlexGetSupportSize(PETSC_UNUSED DM dm,...) {DMPlexError;}
#undef  __FUNCT__
#define __FUNCT__ "DMPlexSetSupportSize"
PetscErrorCode DMPlexSetSupportSize(PETSC_UNUSED DM dm,...) {DMPlexError;}
#undef  __FUNCT__
#define __FUNCT__ "DMPlexGetSupport"
PetscErrorCode DMPlexGetSupport(PETSC_UNUSED DM dm,...) {DMPlexError;}
#undef  __FUNCT__
#define __FUNCT__ "DMPlexSetSupport"
PetscErrorCode DMPlexSetSupport(PETSC_UNUSED DM dm,...) {DMPlexError;}
#undef  __FUNCT__
#define __FUNCT__ "DMPlexGetMaxSizes"
PetscErrorCode DMPlexGetMaxSizes(PETSC_UNUSED DM dm,...) {DMPlexError;}
#undef  __FUNCT__
#define __FUNCT__ "DMPlexSymmetrize"
PetscErrorCode DMPlexSymmetrize(PETSC_UNUSED DM dm,...) {DMPlexError;}
#undef  __FUNCT__
#define __FUNCT__ "DMPlexStratify"
PetscErrorCode DMPlexStratify(PETSC_UNUSED DM dm,...) {DMPlexError;}
#undef  __FUNCT__
#define __FUNCT__ "DMPlexOrient"
PetscErrorCode DMPlexOrient(PETSC_UNUSED DM dm,...) {DMPlexError;}

#undef DMPlexError
