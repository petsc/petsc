#include <petsc/private/dmimpl.h>           /*I      "petscdm.h"          I*/
#include <petsc/private/dmlabelimpl.h>      /*I      "petscdmlabel.h"     I*/
#include <petsc/private/petscdsimpl.h>      /*I      "petscds.h"     I*/
#include <petscdmplex.h>
#include <petscdmfield.h>
#include <petscsf.h>
#include <petscds.h>

PetscClassId  DM_CLASSID;
PetscClassId  DMLABEL_CLASSID;
PetscLogEvent DM_Convert, DM_GlobalToLocal, DM_LocalToGlobal, DM_LocalToLocal, DM_LocatePoints, DM_Coarsen, DM_Refine, DM_CreateInterpolation, DM_CreateRestriction, DM_CreateInjection, DM_CreateMatrix, DM_Load;

const char *const DMBoundaryTypes[] = {"NONE","GHOSTED","MIRROR","PERIODIC","TWIST","DMBoundaryType","DM_BOUNDARY_",0};
const char *const DMBoundaryConditionTypes[] = {"INVALID","ESSENTIAL","NATURAL","INVALID","INVALID","ESSENTIAL_FIELD","NATURAL_FIELD","INVALID","INVALID","INVALID","NATURAL_RIEMANN","DMBoundaryConditionType","DM_BC_",0};
const char *const DMPolytopeTypes[] = {"point", "segment", "triangle", "quadrilateral", "segment tensor prism", "tetrahedron", "hexahedron", "triangular prism", "triangular tensor prism", "quadrilateral tensor prism", "unknown", "DMPolytopeType", "DM_POLYTOPE_", 0};
/*@
  DMCreate - Creates an empty DM object. The type can then be set with DMSetType().

   If you never  call DMSetType()  it will generate an
   error when you try to use the vector.

  Collective

  Input Parameter:
. comm - The communicator for the DM object

  Output Parameter:
. dm - The DM object

  Level: beginner

.seealso: DMSetType(), DMDA, DMSLICED, DMCOMPOSITE, DMPLEX, DMMOAB, DMNETWORK
@*/
PetscErrorCode  DMCreate(MPI_Comm comm,DM *dm)
{
  DM             v;
  PetscDS        ds;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm,2);
  *dm = NULL;
  ierr = DMInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(v, DM_CLASSID, "DM", "Distribution Manager", "DM", comm, DMDestroy, DMView);CHKERRQ(ierr);

  v->setupcalled              = PETSC_FALSE;
  v->setfromoptionscalled     = PETSC_FALSE;
  v->ltogmap                  = NULL;
  v->bs                       = 1;
  v->coloringtype             = IS_COLORING_GLOBAL;
  ierr                        = PetscSFCreate(comm, &v->sf);CHKERRQ(ierr);
  ierr                        = PetscSFCreate(comm, &v->sectionSF);CHKERRQ(ierr);
  v->labels                   = NULL;
  v->adjacency[0]             = PETSC_FALSE;
  v->adjacency[1]             = PETSC_TRUE;
  v->depthLabel               = NULL;
  v->celltypeLabel            = NULL;
  v->localSection             = NULL;
  v->globalSection            = NULL;
  v->defaultConstraintSection = NULL;
  v->defaultConstraintMat     = NULL;
  v->L                        = NULL;
  v->maxCell                  = NULL;
  v->bdtype                   = NULL;
  v->dimEmbed                 = PETSC_DEFAULT;
  v->dim                      = PETSC_DETERMINE;
  {
    PetscInt i;
    for (i = 0; i < 10; ++i) {
      v->nullspaceConstructors[i] = NULL;
      v->nearnullspaceConstructors[i] = NULL;
    }
  }
  ierr = PetscDSCreate(PetscObjectComm((PetscObject) v), &ds);CHKERRQ(ierr);
  ierr = DMSetRegionDS(v, NULL, NULL, ds);CHKERRQ(ierr);
  ierr = PetscDSDestroy(&ds);CHKERRQ(ierr);
  v->dmBC = NULL;
  v->coarseMesh = NULL;
  v->outputSequenceNum = -1;
  v->outputSequenceVal = 0.0;
  ierr = DMSetVecType(v,VECSTANDARD);CHKERRQ(ierr);
  ierr = DMSetMatType(v,MATAIJ);CHKERRQ(ierr);

  *dm = v;
  PetscFunctionReturn(0);
}

/*@
  DMClone - Creates a DM object with the same topology as the original.

  Collective

  Input Parameter:
. dm - The original DM object

  Output Parameter:
. newdm  - The new DM object

  Level: beginner

  Notes: For some DM this is a shallow clone, the result of which may share (referent counted) information with its parent. For example,
         DMClone() applied to a DMPLEX object will result in a new DMPLEX that shares the topology with the original DMPLEX. It does
         share the PetscSection of the original DM

  The clone is considered set up iff the original is.

.seealso: DMDestroy(), DMCreate(), DMSetType(), DMSetLocalSection(), DMSetGlobalSection()

@*/
PetscErrorCode DMClone(DM dm, DM *newdm)
{
  PetscSF        sf;
  Vec            coords;
  void          *ctx;
  PetscInt       dim, cdim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(newdm,2);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), newdm);CHKERRQ(ierr);
  ierr = DMCopyLabels(dm, *newdm, PETSC_COPY_VALUES, PETSC_TRUE);CHKERRQ(ierr);
  (*newdm)->leveldown  = dm->leveldown;
  (*newdm)->levelup    = dm->levelup;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMSetDimension(*newdm, dim);CHKERRQ(ierr);
  if (dm->ops->clone) {
    ierr = (*dm->ops->clone)(dm, newdm);CHKERRQ(ierr);
  }
  (*newdm)->setupcalled = dm->setupcalled;
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  ierr = DMSetPointSF(*newdm, sf);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*newdm, ctx);CHKERRQ(ierr);
  if (dm->coordinateDM) {
    DM           ncdm;
    PetscSection cs;
    PetscInt     pEnd = -1, pEndMax = -1;

    ierr = DMGetLocalSection(dm->coordinateDM, &cs);CHKERRQ(ierr);
    if (cs) {ierr = PetscSectionGetChart(cs, NULL, &pEnd);CHKERRQ(ierr);}
    ierr = MPI_Allreduce(&pEnd,&pEndMax,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
    if (pEndMax >= 0) {
      ierr = DMClone(dm->coordinateDM, &ncdm);CHKERRQ(ierr);
      ierr = DMCopyDisc(dm->coordinateDM, ncdm);CHKERRQ(ierr);
      ierr = DMSetLocalSection(ncdm, cs);CHKERRQ(ierr);
      ierr = DMSetCoordinateDM(*newdm, ncdm);CHKERRQ(ierr);
      ierr = DMDestroy(&ncdm);CHKERRQ(ierr);
    }
  }
  ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
  ierr = DMSetCoordinateDim(*newdm, cdim);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coords);CHKERRQ(ierr);
  if (coords) {
    ierr = DMSetCoordinatesLocal(*newdm, coords);CHKERRQ(ierr);
  } else {
    ierr = DMGetCoordinates(dm, &coords);CHKERRQ(ierr);
    if (coords) {ierr = DMSetCoordinates(*newdm, coords);CHKERRQ(ierr);}
  }
  {
    PetscBool             isper;
    const PetscReal      *maxCell, *L;
    const DMBoundaryType *bd;
    ierr = DMGetPeriodicity(dm, &isper, &maxCell, &L, &bd);CHKERRQ(ierr);
    ierr = DMSetPeriodicity(*newdm, isper, maxCell,  L,  bd);CHKERRQ(ierr);
  }
  {
    PetscBool useCone, useClosure;

    ierr = DMGetAdjacency(dm, PETSC_DEFAULT, &useCone, &useClosure);CHKERRQ(ierr);
    ierr = DMSetAdjacency(*newdm, PETSC_DEFAULT, useCone, useClosure);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
       DMSetVecType - Sets the type of vector created with DMCreateLocalVector() and DMCreateGlobalVector()

   Logically Collective on da

   Input Parameter:
+  da - initial distributed array
.  ctype - the vector type, currently either VECSTANDARD, VECCUDA, or VECVIENNACL

   Options Database:
.   -dm_vec_type ctype

   Level: intermediate

.seealso: DMCreate(), DMDestroy(), DM, DMDAInterpolationType, VecType, DMGetVecType(), DMSetMatType(), DMGetMatType()
@*/
PetscErrorCode  DMSetVecType(DM da,VecType ctype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  ierr = PetscFree(da->vectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(ctype,(char**)&da->vectype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
       DMGetVecType - Gets the type of vector created with DMCreateLocalVector() and DMCreateGlobalVector()

   Logically Collective on da

   Input Parameter:
.  da - initial distributed array

   Output Parameter:
.  ctype - the vector type

   Level: intermediate

.seealso: DMCreate(), DMDestroy(), DM, DMDAInterpolationType, VecType, DMSetMatType(), DMGetMatType(), DMSetVecType()
@*/
PetscErrorCode  DMGetVecType(DM da,VecType *ctype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  *ctype = da->vectype;
  PetscFunctionReturn(0);
}

/*@
  VecGetDM - Gets the DM defining the data layout of the vector

  Not collective

  Input Parameter:
. v - The Vec

  Output Parameter:
. dm - The DM

  Level: intermediate

.seealso: VecSetDM(), DMGetLocalVector(), DMGetGlobalVector(), DMSetVecType()
@*/
PetscErrorCode VecGetDM(Vec v, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(dm,2);
  ierr = PetscObjectQuery((PetscObject) v, "__PETSc_dm", (PetscObject*) dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  VecSetDM - Sets the DM defining the data layout of the vector.

  Not collective

  Input Parameters:
+ v - The Vec
- dm - The DM

  Note: This is NOT the same as DMCreateGlobalVector() since it does not change the view methods or perform other customization, but merely sets the DM member.

  Level: intermediate

.seealso: VecGetDM(), DMGetLocalVector(), DMGetGlobalVector(), DMSetVecType()
@*/
PetscErrorCode VecSetDM(Vec v, DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  if (dm) PetscValidHeaderSpecific(dm,DM_CLASSID,2);
  ierr = PetscObjectCompose((PetscObject) v, "__PETSc_dm", (PetscObject) dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
       DMSetISColoringType - Sets the type of coloring, global or local, that is created by the DM

   Logically Collective on dm

   Input Parameters:
+  dm - the DM context
-  ctype - the matrix type

   Options Database:
.   -dm_is_coloring_type - global or local

   Level: intermediate

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMCreateMatrix(), DMSetMatrixPreallocateOnly(), MatType, DMGetMatType(),
          DMGetISColoringType()
@*/
PetscErrorCode  DMSetISColoringType(DM dm,ISColoringType ctype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->coloringtype = ctype;
  PetscFunctionReturn(0);
}

/*@C
       DMGetISColoringType - Gets the type of coloring, global or local, that is created by the DM

   Logically Collective on dm

   Input Parameter:
.  dm - the DM context

   Output Parameter:
.  ctype - the matrix type

   Options Database:
.   -dm_is_coloring_type - global or local

   Level: intermediate

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMCreateMatrix(), DMSetMatrixPreallocateOnly(), MatType, DMGetMatType(),
          DMGetISColoringType()
@*/
PetscErrorCode  DMGetISColoringType(DM dm,ISColoringType *ctype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *ctype = dm->coloringtype;
  PetscFunctionReturn(0);
}

/*@C
       DMSetMatType - Sets the type of matrix created with DMCreateMatrix()

   Logically Collective on dm

   Input Parameters:
+  dm - the DM context
-  ctype - the matrix type

   Options Database:
.   -dm_mat_type ctype

   Level: intermediate

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMCreateMatrix(), DMSetMatrixPreallocateOnly(), MatType, DMGetMatType(), DMSetMatType(), DMGetMatType()
@*/
PetscErrorCode  DMSetMatType(DM dm,MatType ctype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscFree(dm->mattype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(ctype,(char**)&dm->mattype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
       DMGetMatType - Gets the type of matrix created with DMCreateMatrix()

   Logically Collective on dm

   Input Parameter:
.  dm - the DM context

   Output Parameter:
.  ctype - the matrix type

   Options Database:
.   -dm_mat_type ctype

   Level: intermediate

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMCreateMatrix(), DMSetMatrixPreallocateOnly(), MatType, DMSetMatType(), DMSetMatType(), DMGetMatType()
@*/
PetscErrorCode  DMGetMatType(DM dm,MatType *ctype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *ctype = dm->mattype;
  PetscFunctionReturn(0);
}

/*@
  MatGetDM - Gets the DM defining the data layout of the matrix

  Not collective

  Input Parameter:
. A - The Mat

  Output Parameter:
. dm - The DM

  Level: intermediate

  Developer Note: Since the Mat class doesn't know about the DM class the DM object is associated with
                  the Mat through a PetscObjectCompose() operation

.seealso: MatSetDM(), DMCreateMatrix(), DMSetMatType()
@*/
PetscErrorCode MatGetDM(Mat A, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(dm,2);
  ierr = PetscObjectQuery((PetscObject) A, "__PETSc_dm", (PetscObject*) dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  MatSetDM - Sets the DM defining the data layout of the matrix

  Not collective

  Input Parameters:
+ A - The Mat
- dm - The DM

  Level: intermediate

  Developer Note: Since the Mat class doesn't know about the DM class the DM object is associated with
                  the Mat through a PetscObjectCompose() operation


.seealso: MatGetDM(), DMCreateMatrix(), DMSetMatType()
@*/
PetscErrorCode MatSetDM(Mat A, DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  if (dm) PetscValidHeaderSpecific(dm,DM_CLASSID,2);
  ierr = PetscObjectCompose((PetscObject) A, "__PETSc_dm", (PetscObject) dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   DMSetOptionsPrefix - Sets the prefix used for searching for all
   DM options in the database.

   Logically Collective on dm

   Input Parameter:
+  da - the DM context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: DMSetFromOptions()
@*/
PetscErrorCode  DMSetOptionsPrefix(DM dm,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)dm,prefix);CHKERRQ(ierr);
  if (dm->sf) {
    ierr = PetscObjectSetOptionsPrefix((PetscObject)dm->sf,prefix);CHKERRQ(ierr);
  }
  if (dm->sectionSF) {
    ierr = PetscObjectSetOptionsPrefix((PetscObject)dm->sectionSF,prefix);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   DMAppendOptionsPrefix - Appends to the prefix used for searching for all
   DM options in the database.

   Logically Collective on dm

   Input Parameters:
+  dm - the DM context
-  prefix - the prefix string to prepend to all DM option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: DMSetOptionsPrefix(), DMGetOptionsPrefix()
@*/
PetscErrorCode  DMAppendOptionsPrefix(DM dm,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)dm,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   DMGetOptionsPrefix - Gets the prefix used for searching for all
   DM options in the database.

   Not Collective

   Input Parameters:
.  dm - the DM context

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

   Notes:
    On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: DMSetOptionsPrefix(), DMAppendOptionsPrefix()
@*/
PetscErrorCode  DMGetOptionsPrefix(DM dm,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)dm,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCountNonCyclicReferences(DM dm, PetscBool recurseCoarse, PetscBool recurseFine, PetscInt *ncrefct)
{
  PetscInt i, refct = ((PetscObject) dm)->refct;
  DMNamedVecLink nlink;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *ncrefct = 0;
  /* count all the circular references of DM and its contained Vecs */
  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    if (dm->localin[i])  refct--;
    if (dm->globalin[i]) refct--;
  }
  for (nlink=dm->namedglobal; nlink; nlink=nlink->next) refct--;
  for (nlink=dm->namedlocal; nlink; nlink=nlink->next) refct--;
  if (dm->x) {
    DM obj;
    ierr = VecGetDM(dm->x, &obj);CHKERRQ(ierr);
    if (obj == dm) refct--;
  }
  if (dm->coarseMesh && dm->coarseMesh->fineMesh == dm) {
    refct--;
    if (recurseCoarse) {
      PetscInt coarseCount;

      ierr = DMCountNonCyclicReferences(dm->coarseMesh, PETSC_TRUE, PETSC_FALSE,&coarseCount);CHKERRQ(ierr);
      refct += coarseCount;
    }
  }
  if (dm->fineMesh && dm->fineMesh->coarseMesh == dm) {
    refct--;
    if (recurseFine) {
      PetscInt fineCount;

      ierr = DMCountNonCyclicReferences(dm->fineMesh, PETSC_FALSE, PETSC_TRUE,&fineCount);CHKERRQ(ierr);
      refct += fineCount;
    }
  }
  *ncrefct = refct;
  PetscFunctionReturn(0);
}

PetscErrorCode DMDestroyLabelLinkList_Internal(DM dm)
{
  DMLabelLink    next = dm->labels;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* destroy the labels */
  while (next) {
    DMLabelLink tmp = next->next;

    if (next->label == dm->depthLabel)    dm->depthLabel    = NULL;
    if (next->label == dm->celltypeLabel) dm->celltypeLabel = NULL;
    ierr = DMLabelDestroy(&next->label);CHKERRQ(ierr);
    ierr = PetscFree(next);CHKERRQ(ierr);
    next = tmp;
  }
  dm->labels = NULL;
  PetscFunctionReturn(0);
}

/*@
    DMDestroy - Destroys a vector packer or DM.

    Collective on dm

    Input Parameter:
.   dm - the DM object to destroy

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix()

@*/
PetscErrorCode  DMDestroy(DM *dm)
{
  PetscInt       i, cnt;
  DMNamedVecLink nlink,nnext;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*dm) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*dm),DM_CLASSID,1);

  /* count all non-cyclic references in the doubly-linked list of coarse<->fine meshes */
  ierr = DMCountNonCyclicReferences(*dm,PETSC_TRUE,PETSC_TRUE,&cnt);CHKERRQ(ierr);
  --((PetscObject)(*dm))->refct;
  if (--cnt > 0) {*dm = 0; PetscFunctionReturn(0);}
  /*
     Need this test because the dm references the vectors that
     reference the dm, so destroying the dm calls destroy on the
     vectors that cause another destroy on the dm
  */
  if (((PetscObject)(*dm))->refct < 0) PetscFunctionReturn(0);
  ((PetscObject) (*dm))->refct = 0;
  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    if ((*dm)->localout[i]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Destroying a DM that has a local vector obtained with DMGetLocalVector()");
    ierr = VecDestroy(&(*dm)->localin[i]);CHKERRQ(ierr);
  }
  nnext=(*dm)->namedglobal;
  (*dm)->namedglobal = NULL;
  for (nlink=nnext; nlink; nlink=nnext) { /* Destroy the named vectors */
    nnext = nlink->next;
    if (nlink->status != DMVEC_STATUS_IN) SETERRQ1(((PetscObject)*dm)->comm,PETSC_ERR_ARG_WRONGSTATE,"DM still has Vec named '%s' checked out",nlink->name);
    ierr = PetscFree(nlink->name);CHKERRQ(ierr);
    ierr = VecDestroy(&nlink->X);CHKERRQ(ierr);
    ierr = PetscFree(nlink);CHKERRQ(ierr);
  }
  nnext=(*dm)->namedlocal;
  (*dm)->namedlocal = NULL;
  for (nlink=nnext; nlink; nlink=nnext) { /* Destroy the named local vectors */
    nnext = nlink->next;
    if (nlink->status != DMVEC_STATUS_IN) SETERRQ1(((PetscObject)*dm)->comm,PETSC_ERR_ARG_WRONGSTATE,"DM still has Vec named '%s' checked out",nlink->name);
    ierr = PetscFree(nlink->name);CHKERRQ(ierr);
    ierr = VecDestroy(&nlink->X);CHKERRQ(ierr);
    ierr = PetscFree(nlink);CHKERRQ(ierr);
  }

  /* Destroy the list of hooks */
  {
    DMCoarsenHookLink link,next;
    for (link=(*dm)->coarsenhook; link; link=next) {
      next = link->next;
      ierr = PetscFree(link);CHKERRQ(ierr);
    }
    (*dm)->coarsenhook = NULL;
  }
  {
    DMRefineHookLink link,next;
    for (link=(*dm)->refinehook; link; link=next) {
      next = link->next;
      ierr = PetscFree(link);CHKERRQ(ierr);
    }
    (*dm)->refinehook = NULL;
  }
  {
    DMSubDomainHookLink link,next;
    for (link=(*dm)->subdomainhook; link; link=next) {
      next = link->next;
      ierr = PetscFree(link);CHKERRQ(ierr);
    }
    (*dm)->subdomainhook = NULL;
  }
  {
    DMGlobalToLocalHookLink link,next;
    for (link=(*dm)->gtolhook; link; link=next) {
      next = link->next;
      ierr = PetscFree(link);CHKERRQ(ierr);
    }
    (*dm)->gtolhook = NULL;
  }
  {
    DMLocalToGlobalHookLink link,next;
    for (link=(*dm)->ltoghook; link; link=next) {
      next = link->next;
      ierr = PetscFree(link);CHKERRQ(ierr);
    }
    (*dm)->ltoghook = NULL;
  }
  /* Destroy the work arrays */
  {
    DMWorkLink link,next;
    if ((*dm)->workout) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Work array still checked out");
    for (link=(*dm)->workin; link; link=next) {
      next = link->next;
      ierr = PetscFree(link->mem);CHKERRQ(ierr);
      ierr = PetscFree(link);CHKERRQ(ierr);
    }
    (*dm)->workin = NULL;
  }
  /* destroy the labels */
  ierr = DMDestroyLabelLinkList_Internal(*dm);CHKERRQ(ierr);
  /* destroy the fields */
  ierr = DMClearFields(*dm);CHKERRQ(ierr);
  /* destroy the boundaries */
  {
    DMBoundary next = (*dm)->boundary;
    while (next) {
      DMBoundary b = next;

      next = b->next;
      ierr = PetscFree(b);CHKERRQ(ierr);
    }
  }

  ierr = PetscObjectDestroy(&(*dm)->dmksp);CHKERRQ(ierr);
  ierr = PetscObjectDestroy(&(*dm)->dmsnes);CHKERRQ(ierr);
  ierr = PetscObjectDestroy(&(*dm)->dmts);CHKERRQ(ierr);

  if ((*dm)->ctx && (*dm)->ctxdestroy) {
    ierr = (*(*dm)->ctxdestroy)(&(*dm)->ctx);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&(*dm)->x);CHKERRQ(ierr);
  ierr = MatFDColoringDestroy(&(*dm)->fd);CHKERRQ(ierr);
  ierr = DMClearGlobalVectors(*dm);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&(*dm)->ltogmap);CHKERRQ(ierr);
  ierr = PetscFree((*dm)->vectype);CHKERRQ(ierr);
  ierr = PetscFree((*dm)->mattype);CHKERRQ(ierr);

  ierr = PetscSectionDestroy(&(*dm)->localSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&(*dm)->globalSection);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&(*dm)->map);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&(*dm)->defaultConstraintSection);CHKERRQ(ierr);
  ierr = MatDestroy(&(*dm)->defaultConstraintMat);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&(*dm)->sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&(*dm)->sectionSF);CHKERRQ(ierr);
  if ((*dm)->useNatural) {
    if ((*dm)->sfNatural) {
      ierr = PetscSFDestroy(&(*dm)->sfNatural);CHKERRQ(ierr);
    }
    ierr = PetscObjectDereference((PetscObject) (*dm)->sfMigration);CHKERRQ(ierr);
  }
  if ((*dm)->coarseMesh && (*dm)->coarseMesh->fineMesh == *dm) {
    ierr = DMSetFineDM((*dm)->coarseMesh,NULL);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&(*dm)->coarseMesh);CHKERRQ(ierr);
  if ((*dm)->fineMesh && (*dm)->fineMesh->coarseMesh == *dm) {
    ierr = DMSetCoarseDM((*dm)->fineMesh,NULL);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&(*dm)->fineMesh);CHKERRQ(ierr);
  ierr = DMFieldDestroy(&(*dm)->coordinateField);CHKERRQ(ierr);
  ierr = DMDestroy(&(*dm)->coordinateDM);CHKERRQ(ierr);
  ierr = VecDestroy(&(*dm)->coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&(*dm)->coordinatesLocal);CHKERRQ(ierr);
  ierr = PetscFree3((*dm)->L,(*dm)->maxCell,(*dm)->bdtype);CHKERRQ(ierr);
  if ((*dm)->transformDestroy) {ierr = (*(*dm)->transformDestroy)(*dm, (*dm)->transformCtx);CHKERRQ(ierr);}
  ierr = DMDestroy(&(*dm)->transformDM);CHKERRQ(ierr);
  ierr = VecDestroy(&(*dm)->transform);CHKERRQ(ierr);

  ierr = DMClearDS(*dm);CHKERRQ(ierr);
  ierr = DMDestroy(&(*dm)->dmBC);CHKERRQ(ierr);
  /* if memory was published with SAWs then destroy it */
  ierr = PetscObjectSAWsViewOff((PetscObject)*dm);CHKERRQ(ierr);

  if ((*dm)->ops->destroy) {
    ierr = (*(*dm)->ops->destroy)(*dm);CHKERRQ(ierr);
  }
  ierr = DMMonitorCancel(*dm);CHKERRQ(ierr);
  /* We do not destroy (*dm)->data here so that we can reference count backend objects */
  ierr = PetscHeaderDestroy(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    DMSetUp - sets up the data structures inside a DM object

    Collective on dm

    Input Parameter:
.   dm - the DM object to setup

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix()

@*/
PetscErrorCode  DMSetUp(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (dm->setupcalled) PetscFunctionReturn(0);
  if (dm->ops->setup) {
    ierr = (*dm->ops->setup)(dm);CHKERRQ(ierr);
  }
  dm->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
    DMSetFromOptions - sets parameters in a DM from the options database

    Collective on dm

    Input Parameter:
.   dm - the DM object to set options for

    Options Database:
+   -dm_preallocate_only - Only preallocate the matrix for DMCreateMatrix(), but do not fill it with zeros
.   -dm_vec_type <type>  - type of vector to create inside DM
.   -dm_mat_type <type>  - type of matrix to create inside DM
-   -dm_is_coloring_type - <global or local>

    DMPLEX Specific Checks
+   -dm_plex_check_symmetry        - Check that the adjacency information in the mesh is symmetric - DMPlexCheckSymmetry()
.   -dm_plex_check_skeleton        - Check that each cell has the correct number of vertices (only for homogeneous simplex or tensor meshes) - DMPlexCheckSkeleton()
.   -dm_plex_check_faces           - Check that the faces of each cell give a vertex order this is consistent with what we expect from the cell type - DMPlexCheckFaces()
.   -dm_plex_check_geometry        - Check that cells have positive volume - DMPlexCheckGeometry()
.   -dm_plex_check_pointsf         - Check some necessary conditions for PointSF - DMPlexCheckPointSF()
.   -dm_plex_check_interface_cones - Check points on inter-partition interfaces have conforming order of cone points - DMPlexCheckInterfaceCones()
-   -dm_plex_check_all             - Perform all the checks above

    Level: intermediate

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(),
    DMPlexCheckSymmetry(), DMPlexCheckSkeleton(), DMPlexCheckFaces(), DMPlexCheckGeometry(), DMPlexCheckPointSF(), DMPlexCheckInterfaceCones()

@*/
PetscErrorCode DMSetFromOptions(DM dm)
{
  char           typeName[256];
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->setfromoptionscalled = PETSC_TRUE;
  if (dm->sf) {ierr = PetscSFSetFromOptions(dm->sf);CHKERRQ(ierr);}
  if (dm->sectionSF) {ierr = PetscSFSetFromOptions(dm->sectionSF);CHKERRQ(ierr);}
  ierr = PetscObjectOptionsBegin((PetscObject)dm);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_preallocate_only","only preallocate matrix, but do not set column indices","DMSetMatrixPreallocateOnly",dm->prealloc_only,&dm->prealloc_only,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-dm_vec_type","Vector type used for created vectors","DMSetVecType",VecList,dm->vectype,typeName,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMSetVecType(dm,typeName);CHKERRQ(ierr);
  }
  ierr = PetscOptionsFList("-dm_mat_type","Matrix type used for created matrices","DMSetMatType",MatList,dm->mattype ? dm->mattype : typeName,typeName,sizeof(typeName),&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMSetMatType(dm,typeName);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnum("-dm_is_coloring_type","Global or local coloring of Jacobian","DMSetISColoringType",ISColoringTypes,(PetscEnum)dm->coloringtype,(PetscEnum*)&dm->coloringtype,NULL);CHKERRQ(ierr);
  if (dm->ops->setfromoptions) {
    ierr = (*dm->ops->setfromoptions)(PetscOptionsObject,dm);CHKERRQ(ierr);
  }
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject) dm);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   DMViewFromOptions - View from Options

   Collective on DM

   Input Parameters:
+  dm - the DM object
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso:  DM, DMView, PetscObjectViewFromOptions(), DMCreate()
@*/
PetscErrorCode  DMViewFromOptions(DM dm,PetscObject obj,const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectViewFromOptions((PetscObject)dm,obj,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    DMView - Views a DM

    Collective on dm

    Input Parameter:
+   dm - the DM object to view
-   v - the viewer

    Level: beginner

.seealso DMDestroy(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix()

@*/
PetscErrorCode  DMView(DM dm,PetscViewer v)
{
  PetscErrorCode    ierr;
  PetscBool         isbinary;
  PetscMPIInt       size;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (!v) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)dm),&v);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);
  /* Ideally, we would like to have this test on.
     However, it currently breaks socket viz via GLVis.
     During DMView(parallel_mesh,glvis_viewer), each
     process opens a sequential ASCII socket to visualize
     the local mesh, and PetscObjectView(dm,local_socket)
     is internally called inside VecView_GLVis, incurring
     in an error here */
  /* PetscCheckSameComm(dm,1,v,2); */
  ierr = PetscViewerCheckWritable(v);CHKERRQ(ierr);

  ierr = PetscViewerGetFormat(v,&format);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm),&size);CHKERRQ(ierr);
  if (size == 1 && format == PETSC_VIEWER_LOAD_BALANCE) PetscFunctionReturn(0);
  ierr = PetscObjectPrintClassNamePrefixType((PetscObject)dm,v);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (isbinary) {
    PetscInt classid = DM_FILE_CLASSID;
    char     type[256];

    ierr = PetscViewerBinaryWrite(v,&classid,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscStrncpy(type,((PetscObject)dm)->type_name,256);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(v,type,256,PETSC_CHAR,PETSC_FALSE);CHKERRQ(ierr);
  }
  if (dm->ops->view) {
    ierr = (*dm->ops->view)(dm,v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
    DMCreateGlobalVector - Creates a global vector from a DM object

    Collective on dm

    Input Parameter:
.   dm - the DM object

    Output Parameter:
.   vec - the global vector

    Level: beginner

.seealso DMDestroy(), DMView(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix()

@*/
PetscErrorCode  DMCreateGlobalVector(DM dm,Vec *vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(vec,2);
  if (!dm->ops->createglobalvector) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMCreateGlobalVector",((PetscObject)dm)->type_name);
  ierr = (*dm->ops->createglobalvector)(dm,vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    DMCreateLocalVector - Creates a local vector from a DM object

    Not Collective

    Input Parameter:
.   dm - the DM object

    Output Parameter:
.   vec - the local vector

    Level: beginner

.seealso DMDestroy(), DMView(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix()

@*/
PetscErrorCode  DMCreateLocalVector(DM dm,Vec *vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(vec,2);
  if (!dm->ops->createlocalvector) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMCreateLocalVector",((PetscObject)dm)->type_name);
  ierr = (*dm->ops->createlocalvector)(dm,vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   DMGetLocalToGlobalMapping - Accesses the local-to-global mapping in a DM.

   Collective on dm

   Input Parameter:
.  dm - the DM that provides the mapping

   Output Parameter:
.  ltog - the mapping

   Level: intermediate

   Notes:
   This mapping can then be used by VecSetLocalToGlobalMapping() or
   MatSetLocalToGlobalMapping().

.seealso: DMCreateLocalVector()
@*/
PetscErrorCode DMGetLocalToGlobalMapping(DM dm,ISLocalToGlobalMapping *ltog)
{
  PetscInt       bs = -1, bsLocal[2], bsMinMax[2];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(ltog,2);
  if (!dm->ltogmap) {
    PetscSection section, sectionGlobal;

    ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
    if (section) {
      const PetscInt *cdofs;
      PetscInt       *ltog;
      PetscInt        pStart, pEnd, n, p, k, l;

      ierr = DMGetGlobalSection(dm, &sectionGlobal);CHKERRQ(ierr);
      ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
      ierr = PetscSectionGetStorageSize(section, &n);CHKERRQ(ierr);
      ierr = PetscMalloc1(n, &ltog);CHKERRQ(ierr); /* We want the local+overlap size */
      for (p = pStart, l = 0; p < pEnd; ++p) {
        PetscInt bdof, cdof, dof, off, c, cind = 0;

        /* Should probably use constrained dofs */
        ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
        ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
        ierr = PetscSectionGetConstraintIndices(section, p, &cdofs);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(sectionGlobal, p, &off);CHKERRQ(ierr);
        /* If you have dofs, and constraints, and they are unequal, we set the blocksize to 1 */
        bdof = cdof && (dof-cdof) ? 1 : dof;
        if (dof) {
          if (bs < 0)          {bs = bdof;}
          else if (bs != bdof) {bs = 1;}
        }
        for (c = 0; c < dof; ++c, ++l) {
          if ((cind < cdof) && (c == cdofs[cind])) ltog[l] = off < 0 ? off-c : off+c;
          else                                     ltog[l] = (off < 0 ? -(off+1) : off) + c;
        }
      }
      /* Must have same blocksize on all procs (some might have no points) */
      bsLocal[0] = bs < 0 ? PETSC_MAX_INT : bs; bsLocal[1] = bs;
      ierr = PetscGlobalMinMaxInt(PetscObjectComm((PetscObject) dm), bsLocal, bsMinMax);CHKERRQ(ierr);
      if (bsMinMax[0] != bsMinMax[1]) {bs = 1;}
      else                            {bs = bsMinMax[0];}
      bs = bs < 0 ? 1 : bs;
      /* Must reduce indices by blocksize */
      if (bs > 1) {
        for (l = 0, k = 0; l < n; l += bs, ++k) ltog[k] = ltog[l]/bs;
        n /= bs;
      }
      ierr = ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)dm), bs, n, ltog, PETSC_OWN_POINTER, &dm->ltogmap);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)dm, (PetscObject)dm->ltogmap);CHKERRQ(ierr);
    } else {
      if (!dm->ops->getlocaltoglobalmapping) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMGetLocalToGlobalMapping",((PetscObject)dm)->type_name);
      ierr = (*dm->ops->getlocaltoglobalmapping)(dm);CHKERRQ(ierr);
    }
  }
  *ltog = dm->ltogmap;
  PetscFunctionReturn(0);
}

/*@
   DMGetBlockSize - Gets the inherent block size associated with a DM

   Not Collective

   Input Parameter:
.  dm - the DM with block structure

   Output Parameter:
.  bs - the block size, 1 implies no exploitable block structure

   Level: intermediate

.seealso: ISCreateBlock(), VecSetBlockSize(), MatSetBlockSize(), DMGetLocalToGlobalMapping()
@*/
PetscErrorCode  DMGetBlockSize(DM dm,PetscInt *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidIntPointer(bs,2);
  if (dm->bs < 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"DM does not have enough information to provide a block size yet");
  *bs = dm->bs;
  PetscFunctionReturn(0);
}

/*@
    DMCreateInterpolation - Gets interpolation matrix between two DM objects

    Collective on dm1

    Input Parameter:
+   dm1 - the DM object
-   dm2 - the second, finer DM object

    Output Parameter:
+  mat - the interpolation
-  vec - the scaling (optional)

    Level: developer

    Notes:
    For DMDA objects this only works for "uniform refinement", that is the refined mesh was obtained DMRefine() or the coarse mesh was obtained by
        DMCoarsen(). The coordinates set into the DMDA are completely ignored in computing the interpolation.

        For DMDA objects you can use this interpolation (more precisely the interpolation from the DMGetCoordinateDM()) to interpolate the mesh coordinate vectors
        EXCEPT in the periodic case where it does not make sense since the coordinate vectors are not periodic.


.seealso DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateColoring(), DMCreateMatrix(), DMRefine(), DMCoarsen(), DMCreateRestriction(), DMCreateInterpolationScale()

@*/
PetscErrorCode  DMCreateInterpolation(DM dm1,DM dm2,Mat *mat,Vec *vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm1,DM_CLASSID,1);
  PetscValidHeaderSpecific(dm2,DM_CLASSID,2);
  PetscValidPointer(mat,3);
  if (!dm1->ops->createinterpolation) SETERRQ1(PetscObjectComm((PetscObject)dm1),PETSC_ERR_SUP,"DM type %s does not implement DMCreateInterpolation",((PetscObject)dm1)->type_name);
  ierr = PetscLogEventBegin(DM_CreateInterpolation,dm1,dm2,0,0);CHKERRQ(ierr);
  ierr = (*dm1->ops->createinterpolation)(dm1,dm2,mat,vec);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DM_CreateInterpolation,dm1,dm2,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    DMCreateInterpolationScale - Forms L = 1/(R*1) such that diag(L)*R preserves scale and is thus suitable for state (versus residual) restriction.

  Input Parameters:
+      dac - DM that defines a coarse mesh
.      daf - DM that defines a fine mesh
-      mat - the restriction (or interpolation operator) from fine to coarse

  Output Parameter:
.    scale - the scaled vector

  Level: developer

.seealso: DMCreateInterpolation()

@*/
PetscErrorCode  DMCreateInterpolationScale(DM dac,DM daf,Mat mat,Vec *scale)
{
  PetscErrorCode ierr;
  Vec            fine;
  PetscScalar    one = 1.0;

  PetscFunctionBegin;
  ierr = DMCreateGlobalVector(daf,&fine);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dac,scale);CHKERRQ(ierr);
  ierr = VecSet(fine,one);CHKERRQ(ierr);
  ierr = MatRestrict(mat,fine,*scale);CHKERRQ(ierr);
  ierr = VecDestroy(&fine);CHKERRQ(ierr);
  ierr = VecReciprocal(*scale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    DMCreateRestriction - Gets restriction matrix between two DM objects

    Collective on dm1

    Input Parameter:
+   dm1 - the DM object
-   dm2 - the second, finer DM object

    Output Parameter:
.  mat - the restriction


    Level: developer

    Notes:
    For DMDA objects this only works for "uniform refinement", that is the refined mesh was obtained DMRefine() or the coarse mesh was obtained by
        DMCoarsen(). The coordinates set into the DMDA are completely ignored in computing the interpolation.


.seealso DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateColoring(), DMCreateMatrix(), DMRefine(), DMCoarsen(), DMCreateInterpolation()

@*/
PetscErrorCode  DMCreateRestriction(DM dm1,DM dm2,Mat *mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm1,DM_CLASSID,1);
  PetscValidHeaderSpecific(dm2,DM_CLASSID,2);
  PetscValidPointer(mat,3);
  if (!dm1->ops->createrestriction) SETERRQ1(PetscObjectComm((PetscObject)dm1),PETSC_ERR_SUP,"DM type %s does not implement DMCreateRestriction",((PetscObject)dm1)->type_name);
  ierr = PetscLogEventBegin(DM_CreateRestriction,dm1,dm2,0,0);CHKERRQ(ierr);
  ierr = (*dm1->ops->createrestriction)(dm1,dm2,mat);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DM_CreateRestriction,dm1,dm2,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    DMCreateInjection - Gets injection matrix between two DM objects

    Collective on dm1

    Input Parameter:
+   dm1 - the DM object
-   dm2 - the second, finer DM object

    Output Parameter:
.   mat - the injection

    Level: developer

   Notes:
    For DMDA objects this only works for "uniform refinement", that is the refined mesh was obtained DMRefine() or the coarse mesh was obtained by
        DMCoarsen(). The coordinates set into the DMDA are completely ignored in computing the injection.

.seealso DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateColoring(), DMCreateMatrix(), DMCreateInterpolation()

@*/
PetscErrorCode  DMCreateInjection(DM dm1,DM dm2,Mat *mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm1,DM_CLASSID,1);
  PetscValidHeaderSpecific(dm2,DM_CLASSID,2);
  PetscValidPointer(mat,3);
  if (!dm1->ops->createinjection) SETERRQ1(PetscObjectComm((PetscObject)dm1),PETSC_ERR_SUP,"DM type %s does not implement DMCreateInjection",((PetscObject)dm1)->type_name);
  ierr = PetscLogEventBegin(DM_CreateInjection,dm1,dm2,0,0);CHKERRQ(ierr);
  ierr = (*dm1->ops->createinjection)(dm1,dm2,mat);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DM_CreateInjection,dm1,dm2,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMCreateMassMatrix - Gets mass matrix between two DM objects, M_ij = \int \phi_i \psi_j

  Collective on dm1

  Input Parameter:
+ dm1 - the DM object
- dm2 - the second, finer DM object

  Output Parameter:
. mat - the interpolation

  Level: developer

.seealso DMCreateMatrix(), DMRefine(), DMCoarsen(), DMCreateRestriction(), DMCreateInterpolation(), DMCreateInjection()
@*/
PetscErrorCode DMCreateMassMatrix(DM dm1, DM dm2, Mat *mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm1, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dm2, DM_CLASSID, 2);
  PetscValidPointer(mat,3);
  if (!dm1->ops->createmassmatrix) SETERRQ1(PetscObjectComm((PetscObject)dm1),PETSC_ERR_SUP,"DM type %s does not implement DMCreateMassMatrix",((PetscObject)dm1)->type_name);
  ierr = (*dm1->ops->createmassmatrix)(dm1, dm2, mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    DMCreateColoring - Gets coloring for a DM

    Collective on dm

    Input Parameter:
+   dm - the DM object
-   ctype - IS_COLORING_LOCAL or IS_COLORING_GLOBAL

    Output Parameter:
.   coloring - the coloring

    Notes:
       Coloring of matrices can be computed directly from the sparse matrix nonzero structure via the MatColoring object or from the mesh from which the
       matrix comes from. In general using the mesh produces a more optimal coloring (fewer colors).

       This produces a coloring with the distance of 2, see MatSetColoringDistance() which can be used for efficiently computing Jacobians with MatFDColoringCreate()

    Level: developer

.seealso DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateMatrix(), DMSetMatType(), MatColoring, MatFDColoringCreate()

@*/
PetscErrorCode  DMCreateColoring(DM dm,ISColoringType ctype,ISColoring *coloring)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(coloring,3);
  if (!dm->ops->getcoloring) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMCreateColoring",((PetscObject)dm)->type_name);
  ierr = (*dm->ops->getcoloring)(dm,ctype,coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    DMCreateMatrix - Gets empty Jacobian for a DM

    Collective on dm

    Input Parameter:
.   dm - the DM object

    Output Parameter:
.   mat - the empty Jacobian

    Level: beginner

    Notes:
    This properly preallocates the number of nonzeros in the sparse matrix so you
       do not need to do it yourself.

       By default it also sets the nonzero structure and puts in the zero entries. To prevent setting
       the nonzero pattern call DMSetMatrixPreallocateOnly()

       For structured grid problems, when you call MatView() on this matrix it is displayed using the global natural ordering, NOT in the ordering used
       internally by PETSc.

       For structured grid problems, in general it is easiest to use MatSetValuesStencil() or MatSetValuesLocal() to put values into the matrix because MatSetValues() requires
       the indices for the global numbering for DMDAs which is complicated.

.seealso DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMSetMatType()

@*/
PetscErrorCode  DMCreateMatrix(DM dm,Mat *mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(mat,3);
  if (!dm->ops->creatematrix) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMCreateMatrix",((PetscObject)dm)->type_name);
  ierr = MatInitializePackage();CHKERRQ(ierr);
  ierr = PetscLogEventBegin(DM_CreateMatrix,0,0,0,0);CHKERRQ(ierr);
  ierr = (*dm->ops->creatematrix)(dm,mat);CHKERRQ(ierr);
  /* Handle nullspace and near nullspace */
  if (dm->Nf) {
    MatNullSpace nullSpace;
    PetscInt     Nf;

    ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
    if (Nf == 1) {
      if (dm->nullspaceConstructors[0]) {
        ierr = (*dm->nullspaceConstructors[0])(dm, 0, &nullSpace);CHKERRQ(ierr);
        ierr = MatSetNullSpace(*mat, nullSpace);CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&nullSpace);CHKERRQ(ierr);
      }
      if (dm->nearnullspaceConstructors[0]) {
        ierr = (*dm->nearnullspaceConstructors[0])(dm, 0, &nullSpace);CHKERRQ(ierr);
        ierr = MatSetNearNullSpace(*mat, nullSpace);CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&nullSpace);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscLogEventEnd(DM_CreateMatrix,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMSetMatrixPreallocateOnly - When DMCreateMatrix() is called the matrix will be properly
    preallocated but the nonzero structure and zero values will not be set.

  Logically Collective on dm

  Input Parameter:
+ dm - the DM
- only - PETSC_TRUE if only want preallocation

  Level: developer
.seealso DMCreateMatrix(), DMSetMatrixStructureOnly()
@*/
PetscErrorCode DMSetMatrixPreallocateOnly(DM dm, PetscBool only)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->prealloc_only = only;
  PetscFunctionReturn(0);
}

/*@
  DMSetMatrixStructureOnly - When DMCreateMatrix() is called, the matrix structure will be created
    but the array for values will not be allocated.

  Logically Collective on dm

  Input Parameter:
+ dm - the DM
- only - PETSC_TRUE if only want matrix stucture

  Level: developer
.seealso DMCreateMatrix(), DMSetMatrixPreallocateOnly()
@*/
PetscErrorCode DMSetMatrixStructureOnly(DM dm, PetscBool only)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->structure_only = only;
  PetscFunctionReturn(0);
}

/*@C
  DMGetWorkArray - Gets a work array guaranteed to be at least the input size, restore with DMRestoreWorkArray()

  Not Collective

  Input Parameters:
+ dm - the DM object
. count - The minium size
- dtype - MPI data type, often MPIU_REAL, MPIU_SCALAR, MPIU_INT)

  Output Parameter:
. array - the work array

  Level: developer

.seealso DMDestroy(), DMCreate()
@*/
PetscErrorCode DMGetWorkArray(DM dm,PetscInt count,MPI_Datatype dtype,void *mem)
{
  PetscErrorCode ierr;
  DMWorkLink     link;
  PetscMPIInt    dsize;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(mem,4);
  if (dm->workin) {
    link       = dm->workin;
    dm->workin = dm->workin->next;
  } else {
    ierr = PetscNewLog(dm,&link);CHKERRQ(ierr);
  }
  ierr = MPI_Type_size(dtype,&dsize);CHKERRQ(ierr);
  if (((size_t)dsize*count) > link->bytes) {
    ierr        = PetscFree(link->mem);CHKERRQ(ierr);
    ierr        = PetscMalloc(dsize*count,&link->mem);CHKERRQ(ierr);
    link->bytes = dsize*count;
  }
  link->next   = dm->workout;
  dm->workout  = link;
  *(void**)mem = link->mem;
  PetscFunctionReturn(0);
}

/*@C
  DMRestoreWorkArray - Restores a work array guaranteed to be at least the input size, restore with DMRestoreWorkArray()

  Not Collective

  Input Parameters:
+ dm - the DM object
. count - The minium size
- dtype - MPI data type, often MPIU_REAL, MPIU_SCALAR, MPIU_INT

  Output Parameter:
. array - the work array

  Level: developer

  Developer Notes:
    count and dtype are ignored, they are only needed for DMGetWorkArray()
.seealso DMDestroy(), DMCreate()
@*/
PetscErrorCode DMRestoreWorkArray(DM dm,PetscInt count,MPI_Datatype dtype,void *mem)
{
  DMWorkLink *p,link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(mem,4);
  for (p=&dm->workout; (link=*p); p=&link->next) {
    if (link->mem == *(void**)mem) {
      *p           = link->next;
      link->next   = dm->workin;
      dm->workin   = link;
      *(void**)mem = NULL;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Array was not checked out");
}

PetscErrorCode DMSetNullSpaceConstructor(DM dm, PetscInt field, PetscErrorCode (*nullsp)(DM dm, PetscInt field, MatNullSpace *nullSpace))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (field >= 10) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle %d >= 10 fields", field);
  dm->nullspaceConstructors[field] = nullsp;
  PetscFunctionReturn(0);
}

PetscErrorCode DMGetNullSpaceConstructor(DM dm, PetscInt field, PetscErrorCode (**nullsp)(DM dm, PetscInt field, MatNullSpace *nullSpace))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(nullsp, 3);
  if (field >= 10) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle %d >= 10 fields", field);
  *nullsp = dm->nullspaceConstructors[field];
  PetscFunctionReturn(0);
}

PetscErrorCode DMSetNearNullSpaceConstructor(DM dm, PetscInt field, PetscErrorCode (*nullsp)(DM dm, PetscInt field, MatNullSpace *nullSpace))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (field >= 10) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle %d >= 10 fields", field);
  dm->nearnullspaceConstructors[field] = nullsp;
  PetscFunctionReturn(0);
}

PetscErrorCode DMGetNearNullSpaceConstructor(DM dm, PetscInt field, PetscErrorCode (**nullsp)(DM dm, PetscInt field, MatNullSpace *nullSpace))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(nullsp, 3);
  if (field >= 10) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle %d >= 10 fields", field);
  *nullsp = dm->nearnullspaceConstructors[field];
  PetscFunctionReturn(0);
}

/*@C
  DMCreateFieldIS - Creates a set of IS objects with the global indices of dofs for each field

  Not collective

  Input Parameter:
. dm - the DM object

  Output Parameters:
+ numFields  - The number of fields (or NULL if not requested)
. fieldNames - The name for each field (or NULL if not requested)
- fields     - The global indices for each field (or NULL if not requested)

  Level: intermediate

  Notes:
  The user is responsible for freeing all requested arrays. In particular, every entry of names should be freed with
  PetscFree(), every entry of fields should be destroyed with ISDestroy(), and both arrays should be freed with
  PetscFree().

.seealso DMDestroy(), DMView(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix()
@*/
PetscErrorCode DMCreateFieldIS(DM dm, PetscInt *numFields, char ***fieldNames, IS **fields)
{
  PetscSection   section, sectionGlobal;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (numFields) {
    PetscValidIntPointer(numFields,2);
    *numFields = 0;
  }
  if (fieldNames) {
    PetscValidPointer(fieldNames,3);
    *fieldNames = NULL;
  }
  if (fields) {
    PetscValidPointer(fields,4);
    *fields = NULL;
  }
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  if (section) {
    PetscInt *fieldSizes, *fieldNc, **fieldIndices;
    PetscInt nF, f, pStart, pEnd, p;

    ierr = DMGetGlobalSection(dm, &sectionGlobal);CHKERRQ(ierr);
    ierr = PetscSectionGetNumFields(section, &nF);CHKERRQ(ierr);
    ierr = PetscMalloc3(nF,&fieldSizes,nF,&fieldNc,nF,&fieldIndices);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(sectionGlobal, &pStart, &pEnd);CHKERRQ(ierr);
    for (f = 0; f < nF; ++f) {
      fieldSizes[f] = 0;
      ierr          = PetscSectionGetFieldComponents(section, f, &fieldNc[f]);CHKERRQ(ierr);
    }
    for (p = pStart; p < pEnd; ++p) {
      PetscInt gdof;

      ierr = PetscSectionGetDof(sectionGlobal, p, &gdof);CHKERRQ(ierr);
      if (gdof > 0) {
        for (f = 0; f < nF; ++f) {
          PetscInt fdof, fcdof, fpdof;

          ierr  = PetscSectionGetFieldDof(section, p, f, &fdof);CHKERRQ(ierr);
          ierr  = PetscSectionGetFieldConstraintDof(section, p, f, &fcdof);CHKERRQ(ierr);
          fpdof = fdof-fcdof;
          if (fpdof && fpdof != fieldNc[f]) {
            /* Layout does not admit a pointwise block size */
            fieldNc[f] = 1;
          }
          fieldSizes[f] += fpdof;
        }
      }
    }
    for (f = 0; f < nF; ++f) {
      ierr          = PetscMalloc1(fieldSizes[f], &fieldIndices[f]);CHKERRQ(ierr);
      fieldSizes[f] = 0;
    }
    for (p = pStart; p < pEnd; ++p) {
      PetscInt gdof, goff;

      ierr = PetscSectionGetDof(sectionGlobal, p, &gdof);CHKERRQ(ierr);
      if (gdof > 0) {
        ierr = PetscSectionGetOffset(sectionGlobal, p, &goff);CHKERRQ(ierr);
        for (f = 0; f < nF; ++f) {
          PetscInt fdof, fcdof, fc;

          ierr = PetscSectionGetFieldDof(section, p, f, &fdof);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldConstraintDof(section, p, f, &fcdof);CHKERRQ(ierr);
          for (fc = 0; fc < fdof-fcdof; ++fc, ++fieldSizes[f]) {
            fieldIndices[f][fieldSizes[f]] = goff++;
          }
        }
      }
    }
    if (numFields) *numFields = nF;
    if (fieldNames) {
      ierr = PetscMalloc1(nF, fieldNames);CHKERRQ(ierr);
      for (f = 0; f < nF; ++f) {
        const char *fieldName;

        ierr = PetscSectionGetFieldName(section, f, &fieldName);CHKERRQ(ierr);
        ierr = PetscStrallocpy(fieldName, (char**) &(*fieldNames)[f]);CHKERRQ(ierr);
      }
    }
    if (fields) {
      ierr = PetscMalloc1(nF, fields);CHKERRQ(ierr);
      for (f = 0; f < nF; ++f) {
        PetscInt bs, in[2], out[2];

        ierr  = ISCreateGeneral(PetscObjectComm((PetscObject)dm), fieldSizes[f], fieldIndices[f], PETSC_OWN_POINTER, &(*fields)[f]);CHKERRQ(ierr);
        in[0] = -fieldNc[f];
        in[1] = fieldNc[f];
        ierr  = MPIU_Allreduce(in, out, 2, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
        bs    = (-out[0] == out[1]) ? out[1] : 1;
        ierr  = ISSetBlockSize((*fields)[f], bs);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree3(fieldSizes,fieldNc,fieldIndices);CHKERRQ(ierr);
  } else if (dm->ops->createfieldis) {
    ierr = (*dm->ops->createfieldis)(dm, numFields, fieldNames, fields);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/*@C
  DMCreateFieldDecomposition - Returns a list of IS objects defining a decomposition of a problem into subproblems
                          corresponding to different fields: each IS contains the global indices of the dofs of the
                          corresponding field. The optional list of DMs define the DM for each subproblem.
                          Generalizes DMCreateFieldIS().

  Not collective

  Input Parameter:
. dm - the DM object

  Output Parameters:
+ len       - The number of subproblems in the field decomposition (or NULL if not requested)
. namelist  - The name for each field (or NULL if not requested)
. islist    - The global indices for each field (or NULL if not requested)
- dmlist    - The DMs for each field subproblem (or NULL, if not requested; if NULL is returned, no DMs are defined)

  Level: intermediate

  Notes:
  The user is responsible for freeing all requested arrays. In particular, every entry of names should be freed with
  PetscFree(), every entry of is should be destroyed with ISDestroy(), every entry of dm should be destroyed with DMDestroy(),
  and all of the arrays should be freed with PetscFree().

.seealso DMDestroy(), DMView(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMCreateFieldIS()
@*/
PetscErrorCode DMCreateFieldDecomposition(DM dm, PetscInt *len, char ***namelist, IS **islist, DM **dmlist)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (len) {
    PetscValidIntPointer(len,2);
    *len = 0;
  }
  if (namelist) {
    PetscValidPointer(namelist,3);
    *namelist = 0;
  }
  if (islist) {
    PetscValidPointer(islist,4);
    *islist = 0;
  }
  if (dmlist) {
    PetscValidPointer(dmlist,5);
    *dmlist = 0;
  }
  /*
   Is it a good idea to apply the following check across all impls?
   Perhaps some impls can have a well-defined decomposition before DMSetUp?
   This, however, follows the general principle that accessors are not well-behaved until the object is set up.
   */
  if (!dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE, "Decomposition defined only after DMSetUp");
  if (!dm->ops->createfielddecomposition) {
    PetscSection section;
    PetscInt     numFields, f;

    ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
    if (section) {ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);}
    if (section && numFields && dm->ops->createsubdm) {
      if (len) *len = numFields;
      if (namelist) {ierr = PetscMalloc1(numFields,namelist);CHKERRQ(ierr);}
      if (islist)   {ierr = PetscMalloc1(numFields,islist);CHKERRQ(ierr);}
      if (dmlist)   {ierr = PetscMalloc1(numFields,dmlist);CHKERRQ(ierr);}
      for (f = 0; f < numFields; ++f) {
        const char *fieldName;

        ierr = DMCreateSubDM(dm, 1, &f, islist ? &(*islist)[f] : NULL, dmlist ? &(*dmlist)[f] : NULL);CHKERRQ(ierr);
        if (namelist) {
          ierr = PetscSectionGetFieldName(section, f, &fieldName);CHKERRQ(ierr);
          ierr = PetscStrallocpy(fieldName, (char**) &(*namelist)[f]);CHKERRQ(ierr);
        }
      }
    } else {
      ierr = DMCreateFieldIS(dm, len, namelist, islist);CHKERRQ(ierr);
      /* By default there are no DMs associated with subproblems. */
      if (dmlist) *dmlist = NULL;
    }
  } else {
    ierr = (*dm->ops->createfielddecomposition)(dm,len,namelist,islist,dmlist);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  DMCreateSubDM - Returns an IS and DM encapsulating a subproblem defined by the fields passed in.
                  The fields are defined by DMCreateFieldIS().

  Not collective

  Input Parameters:
+ dm        - The DM object
. numFields - The number of fields in this subproblem
- fields    - The field numbers of the selected fields

  Output Parameters:
+ is - The global indices for the subproblem
- subdm - The DM for the subproblem

  Note: You need to call DMPlexSetMigrationSF() on the original DM if you want the Global-To-Natural map to be automatically constructed

  Level: intermediate

.seealso DMPlexSetMigrationSF(), DMDestroy(), DMView(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMCreateFieldIS()
@*/
PetscErrorCode DMCreateSubDM(DM dm, PetscInt numFields, const PetscInt fields[], IS *is, DM *subdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(fields,3);
  if (is) PetscValidPointer(is,4);
  if (subdm) PetscValidPointer(subdm,5);
  if (!dm->ops->createsubdm) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMCreateSubDM",((PetscObject)dm)->type_name);
  ierr = (*dm->ops->createsubdm)(dm, numFields, fields, is, subdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMCreateSuperDM - Returns an arrays of ISes and DM encapsulating a superproblem defined by the DMs passed in.

  Not collective

  Input Parameter:
+ dms - The DM objects
- len - The number of DMs

  Output Parameters:
+ is - The global indices for the subproblem, or NULL
- superdm - The DM for the superproblem

  Note: You need to call DMPlexSetMigrationSF() on the original DM if you want the Global-To-Natural map to be automatically constructed

  Level: intermediate

.seealso DMPlexSetMigrationSF(), DMDestroy(), DMView(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMCreateFieldIS()
@*/
PetscErrorCode DMCreateSuperDM(DM dms[], PetscInt len, IS **is, DM *superdm)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dms,1);
  for (i = 0; i < len; ++i) {PetscValidHeaderSpecific(dms[i],DM_CLASSID,1);}
  if (is) PetscValidPointer(is,3);
  PetscValidPointer(superdm,4);
  if (len < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of DMs must be nonnegative: %D", len);
  if (len) {
    DM dm = dms[0];
    if (!dm->ops->createsuperdm) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMCreateSuperDM",((PetscObject)dm)->type_name);
    ierr = (*dm->ops->createsuperdm)(dms, len, is, superdm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/*@C
  DMCreateDomainDecomposition - Returns lists of IS objects defining a decomposition of a problem into subproblems
                          corresponding to restrictions to pairs nested subdomains: each IS contains the global
                          indices of the dofs of the corresponding subdomains.  The inner subdomains conceptually
                          define a nonoverlapping covering, while outer subdomains can overlap.
                          The optional list of DMs define the DM for each subproblem.

  Not collective

  Input Parameter:
. dm - the DM object

  Output Parameters:
+ len         - The number of subproblems in the domain decomposition (or NULL if not requested)
. namelist    - The name for each subdomain (or NULL if not requested)
. innerislist - The global indices for each inner subdomain (or NULL, if not requested)
. outerislist - The global indices for each outer subdomain (or NULL, if not requested)
- dmlist      - The DMs for each subdomain subproblem (or NULL, if not requested; if NULL is returned, no DMs are defined)

  Level: intermediate

  Notes:
  The user is responsible for freeing all requested arrays. In particular, every entry of names should be freed with
  PetscFree(), every entry of is should be destroyed with ISDestroy(), every entry of dm should be destroyed with DMDestroy(),
  and all of the arrays should be freed with PetscFree().

.seealso DMDestroy(), DMView(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMCreateDomainDecompositionDM(), DMCreateFieldDecomposition()
@*/
PetscErrorCode DMCreateDomainDecomposition(DM dm, PetscInt *len, char ***namelist, IS **innerislist, IS **outerislist, DM **dmlist)
{
  PetscErrorCode      ierr;
  DMSubDomainHookLink link;
  PetscInt            i,l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (len)           {PetscValidPointer(len,2);            *len         = 0;}
  if (namelist)      {PetscValidPointer(namelist,3);       *namelist    = NULL;}
  if (innerislist)   {PetscValidPointer(innerislist,4);    *innerislist = NULL;}
  if (outerislist)   {PetscValidPointer(outerislist,5);    *outerislist = NULL;}
  if (dmlist)        {PetscValidPointer(dmlist,6);         *dmlist      = NULL;}
  /*
   Is it a good idea to apply the following check across all impls?
   Perhaps some impls can have a well-defined decomposition before DMSetUp?
   This, however, follows the general principle that accessors are not well-behaved until the object is set up.
   */
  if (!dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE, "Decomposition defined only after DMSetUp");
  if (dm->ops->createdomaindecomposition) {
    ierr = (*dm->ops->createdomaindecomposition)(dm,&l,namelist,innerislist,outerislist,dmlist);CHKERRQ(ierr);
    /* copy subdomain hooks and context over to the subdomain DMs */
    if (dmlist && *dmlist) {
      for (i = 0; i < l; i++) {
        for (link=dm->subdomainhook; link; link=link->next) {
          if (link->ddhook) {ierr = (*link->ddhook)(dm,(*dmlist)[i],link->ctx);CHKERRQ(ierr);}
        }
        if (dm->ctx) (*dmlist)[i]->ctx = dm->ctx;
      }
    }
    if (len) *len = l;
  }
  PetscFunctionReturn(0);
}


/*@C
  DMCreateDomainDecompositionScatters - Returns scatters to the subdomain vectors from the global vector

  Not collective

  Input Parameters:
+ dm - the DM object
. n  - the number of subdomain scatters
- subdms - the local subdomains

  Output Parameters:
+ n     - the number of scatters returned
. iscat - scatter from global vector to nonoverlapping global vector entries on subdomain
. oscat - scatter from global vector to overlapping global vector entries on subdomain
- gscat - scatter from global vector to local vector on subdomain (fills in ghosts)

  Notes:
    This is an alternative to the iis and ois arguments in DMCreateDomainDecomposition that allow for the solution
  of general nonlinear problems with overlapping subdomain methods.  While merely having index sets that enable subsets
  of the residual equations to be created is fine for linear problems, nonlinear problems require local assembly of
  solution and residual data.

  Level: developer

.seealso DMDestroy(), DMView(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMCreateFieldIS()
@*/
PetscErrorCode DMCreateDomainDecompositionScatters(DM dm,PetscInt n,DM *subdms,VecScatter **iscat,VecScatter **oscat,VecScatter **gscat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(subdms,3);
  if (!dm->ops->createddscatters) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMCreateDomainDecompositionScatters",((PetscObject)dm)->type_name);
  ierr = (*dm->ops->createddscatters)(dm,n,subdms,iscat,oscat,gscat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMRefine - Refines a DM object

  Collective on dm

  Input Parameter:
+ dm   - the DM object
- comm - the communicator to contain the new DM object (or MPI_COMM_NULL)

  Output Parameter:
. dmf - the refined DM, or NULL

  Note: If no refinement was done, the return value is NULL

  Level: developer

.seealso DMCoarsen(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation()
@*/
PetscErrorCode  DMRefine(DM dm,MPI_Comm comm,DM *dmf)
{
  PetscErrorCode   ierr;
  DMRefineHookLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (!dm->ops->refine) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMRefine",((PetscObject)dm)->type_name);
  ierr = PetscLogEventBegin(DM_Refine,dm,0,0,0);CHKERRQ(ierr);
  ierr = (*dm->ops->refine)(dm,comm,dmf);CHKERRQ(ierr);
  if (*dmf) {
    (*dmf)->ops->creatematrix = dm->ops->creatematrix;

    ierr = PetscObjectCopyFortranFunctionPointers((PetscObject)dm,(PetscObject)*dmf);CHKERRQ(ierr);

    (*dmf)->ctx       = dm->ctx;
    (*dmf)->leveldown = dm->leveldown;
    (*dmf)->levelup   = dm->levelup + 1;

    ierr = DMSetMatType(*dmf,dm->mattype);CHKERRQ(ierr);
    for (link=dm->refinehook; link; link=link->next) {
      if (link->refinehook) {
        ierr = (*link->refinehook)(dm,*dmf,link->ctx);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscLogEventEnd(DM_Refine,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   DMRefineHookAdd - adds a callback to be run when interpolating a nonlinear problem to a finer grid

   Logically Collective

   Input Arguments:
+  coarse - nonlinear solver context on which to run a hook when restricting to a coarser level
.  refinehook - function to run when setting up a coarser level
.  interphook - function to run to update data on finer levels (once per SNESSolve())
-  ctx - [optional] user-defined context for provide data for the hooks (may be NULL)

   Calling sequence of refinehook:
$    refinehook(DM coarse,DM fine,void *ctx);

+  coarse - coarse level DM
.  fine - fine level DM to interpolate problem to
-  ctx - optional user-defined function context

   Calling sequence for interphook:
$    interphook(DM coarse,Mat interp,DM fine,void *ctx)

+  coarse - coarse level DM
.  interp - matrix interpolating a coarse-level solution to the finer grid
.  fine - fine level DM to update
-  ctx - optional user-defined function context

   Level: advanced

   Notes:
   This function is only needed if auxiliary data needs to be passed to fine grids while grid sequencing

   If this function is called multiple times, the hooks will be run in the order they are added.

   This function is currently not available from Fortran.

.seealso: DMCoarsenHookAdd(), SNESFASGetInterpolation(), SNESFASGetInjection(), PetscObjectCompose(), PetscContainerCreate()
@*/
PetscErrorCode DMRefineHookAdd(DM coarse,PetscErrorCode (*refinehook)(DM,DM,void*),PetscErrorCode (*interphook)(DM,Mat,DM,void*),void *ctx)
{
  PetscErrorCode   ierr;
  DMRefineHookLink link,*p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarse,DM_CLASSID,1);
  for (p=&coarse->refinehook; *p; p=&(*p)->next) { /* Scan to the end of the current list of hooks */
    if ((*p)->refinehook == refinehook && (*p)->interphook == interphook && (*p)->ctx == ctx) PetscFunctionReturn(0);
  }
  ierr             = PetscNew(&link);CHKERRQ(ierr);
  link->refinehook = refinehook;
  link->interphook = interphook;
  link->ctx        = ctx;
  link->next       = NULL;
  *p               = link;
  PetscFunctionReturn(0);
}

/*@C
   DMRefineHookRemove - remove a callback from the list of hooks to be run when interpolating a nonlinear problem to a finer grid

   Logically Collective

   Input Arguments:
+  coarse - nonlinear solver context on which to run a hook when restricting to a coarser level
.  refinehook - function to run when setting up a coarser level
.  interphook - function to run to update data on finer levels (once per SNESSolve())
-  ctx - [optional] user-defined context for provide data for the hooks (may be NULL)

   Level: advanced

   Notes:
   This function does nothing if the hook is not in the list.

   This function is currently not available from Fortran.

.seealso: DMCoarsenHookRemove(), SNESFASGetInterpolation(), SNESFASGetInjection(), PetscObjectCompose(), PetscContainerCreate()
@*/
PetscErrorCode DMRefineHookRemove(DM coarse,PetscErrorCode (*refinehook)(DM,DM,void*),PetscErrorCode (*interphook)(DM,Mat,DM,void*),void *ctx)
{
  PetscErrorCode   ierr;
  DMRefineHookLink link,*p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarse,DM_CLASSID,1);
  for (p=&coarse->refinehook; *p; p=&(*p)->next) { /* Search the list of current hooks */
    if ((*p)->refinehook == refinehook && (*p)->interphook == interphook && (*p)->ctx == ctx) {
      link = *p;
      *p = link->next;
      ierr = PetscFree(link);CHKERRQ(ierr);
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*@
   DMInterpolate - interpolates user-defined problem data to a finer DM by running hooks registered by DMRefineHookAdd()

   Collective if any hooks are

   Input Arguments:
+  coarse - coarser DM to use as a base
.  interp - interpolation matrix, apply using MatInterpolate()
-  fine - finer DM to update

   Level: developer

.seealso: DMRefineHookAdd(), MatInterpolate()
@*/
PetscErrorCode DMInterpolate(DM coarse,Mat interp,DM fine)
{
  PetscErrorCode   ierr;
  DMRefineHookLink link;

  PetscFunctionBegin;
  for (link=fine->refinehook; link; link=link->next) {
    if (link->interphook) {
      ierr = (*link->interphook)(coarse,interp,fine,link->ctx);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@
    DMGetRefineLevel - Get's the number of refinements that have generated this DM.

    Not Collective

    Input Parameter:
.   dm - the DM object

    Output Parameter:
.   level - number of refinements

    Level: developer

.seealso DMCoarsen(), DMGetCoarsenLevel(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation()

@*/
PetscErrorCode  DMGetRefineLevel(DM dm,PetscInt *level)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *level = dm->levelup;
  PetscFunctionReturn(0);
}

/*@
    DMSetRefineLevel - Set's the number of refinements that have generated this DM.

    Not Collective

    Input Parameter:
+   dm - the DM object
-   level - number of refinements

    Level: advanced

    Notes:
    This value is used by PCMG to determine how many multigrid levels to use

.seealso DMCoarsen(), DMGetCoarsenLevel(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation()

@*/
PetscErrorCode  DMSetRefineLevel(DM dm,PetscInt level)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->levelup = level;
  PetscFunctionReturn(0);
}

PetscErrorCode DMGetBasisTransformDM_Internal(DM dm, DM *tdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(tdm, 2);
  *tdm = dm->transformDM;
  PetscFunctionReturn(0);
}

PetscErrorCode DMGetBasisTransformVec_Internal(DM dm, Vec *tv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(tv, 2);
  *tv = dm->transform;
  PetscFunctionReturn(0);
}

/*@
  DMHasBasisTransform - Whether we employ a basis transformation from functions in global vectors to functions in local vectors

  Input Parameter:
. dm - The DM

  Output Parameter:
. flg - PETSC_TRUE if a basis transformation should be done

  Level: developer

.seealso: DMPlexGlobalToLocalBasis(), DMPlexLocalToGlobalBasis()()
@*/
PetscErrorCode DMHasBasisTransform(DM dm, PetscBool *flg)
{
  Vec            tv;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidBoolPointer(flg, 2);
  ierr = DMGetBasisTransformVec_Internal(dm, &tv);CHKERRQ(ierr);
  *flg = tv ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode DMConstructBasisTransform_Internal(DM dm)
{
  PetscSection   s, ts;
  PetscScalar   *ta;
  PetscInt       cdim, pStart, pEnd, p, Nf, f, Nc, dof;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(s, &Nf);CHKERRQ(ierr);
  ierr = DMClone(dm, &dm->transformDM);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm->transformDM, &ts);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(ts, Nf);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(ts, pStart, pEnd);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    ierr = PetscSectionGetFieldComponents(s, f, &Nc);CHKERRQ(ierr);
    /* We could start to label fields by their transformation properties */
    if (Nc != cdim) continue;
    for (p = pStart; p < pEnd; ++p) {
      ierr = PetscSectionGetFieldDof(s, p, f, &dof);CHKERRQ(ierr);
      if (!dof) continue;
      ierr = PetscSectionSetFieldDof(ts, p, f, PetscSqr(cdim));CHKERRQ(ierr);
      ierr = PetscSectionAddDof(ts, p, PetscSqr(cdim));CHKERRQ(ierr);
    }
  }
  ierr = PetscSectionSetUp(ts);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm->transformDM, &dm->transform);CHKERRQ(ierr);
  ierr = VecGetArray(dm->transform, &ta);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    for (f = 0; f < Nf; ++f) {
      ierr = PetscSectionGetFieldDof(ts, p, f, &dof);CHKERRQ(ierr);
      if (dof) {
        PetscReal          x[3] = {0.0, 0.0, 0.0};
        PetscScalar       *tva;
        const PetscScalar *A;

        /* TODO Get quadrature point for this dual basis vector for coordinate */
        ierr = (*dm->transformGetMatrix)(dm, x, PETSC_TRUE, &A, dm->transformCtx);CHKERRQ(ierr);
        ierr = DMPlexPointLocalFieldRef(dm->transformDM, p, f, ta, (void *) &tva);CHKERRQ(ierr);
        ierr = PetscArraycpy(tva, A, PetscSqr(cdim));CHKERRQ(ierr);
      }
    }
  }
  ierr = VecRestoreArray(dm->transform, &ta);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCopyTransform(DM dm, DM newdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(newdm, DM_CLASSID, 2);
  newdm->transformCtx       = dm->transformCtx;
  newdm->transformSetUp     = dm->transformSetUp;
  newdm->transformDestroy   = NULL;
  newdm->transformGetMatrix = dm->transformGetMatrix;
  if (newdm->transformSetUp) {ierr = DMConstructBasisTransform_Internal(newdm);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
   DMGlobalToLocalHookAdd - adds a callback to be run when global to local is called

   Logically Collective

   Input Arguments:
+  dm - the DM
.  beginhook - function to run at the beginning of DMGlobalToLocalBegin()
.  endhook - function to run after DMGlobalToLocalEnd() has completed
-  ctx - [optional] user-defined context for provide data for the hooks (may be NULL)

   Calling sequence for beginhook:
$    beginhook(DM fine,VecScatter out,VecScatter in,DM coarse,void *ctx)

+  dm - global DM
.  g - global vector
.  mode - mode
.  l - local vector
-  ctx - optional user-defined function context


   Calling sequence for endhook:
$    endhook(DM fine,VecScatter out,VecScatter in,DM coarse,void *ctx)

+  global - global DM
-  ctx - optional user-defined function context

   Level: advanced

.seealso: DMRefineHookAdd(), SNESFASGetInterpolation(), SNESFASGetInjection(), PetscObjectCompose(), PetscContainerCreate()
@*/
PetscErrorCode DMGlobalToLocalHookAdd(DM dm,PetscErrorCode (*beginhook)(DM,Vec,InsertMode,Vec,void*),PetscErrorCode (*endhook)(DM,Vec,InsertMode,Vec,void*),void *ctx)
{
  PetscErrorCode          ierr;
  DMGlobalToLocalHookLink link,*p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  for (p=&dm->gtolhook; *p; p=&(*p)->next) {} /* Scan to the end of the current list of hooks */
  ierr            = PetscNew(&link);CHKERRQ(ierr);
  link->beginhook = beginhook;
  link->endhook   = endhook;
  link->ctx       = ctx;
  link->next      = NULL;
  *p              = link;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGlobalToLocalHook_Constraints(DM dm, Vec g, InsertMode mode, Vec l, void *ctx)
{
  Mat cMat;
  Vec cVec;
  PetscSection section, cSec;
  PetscInt pStart, pEnd, p, dof;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDefaultConstraints(dm,&cSec,&cMat);CHKERRQ(ierr);
  if (cMat && (mode == INSERT_VALUES || mode == INSERT_ALL_VALUES || mode == INSERT_BC_VALUES)) {
    PetscInt nRows;

    ierr = MatGetSize(cMat,&nRows,NULL);CHKERRQ(ierr);
    if (nRows <= 0) PetscFunctionReturn(0);
    ierr = DMGetLocalSection(dm,&section);CHKERRQ(ierr);
    ierr = MatCreateVecs(cMat,NULL,&cVec);CHKERRQ(ierr);
    ierr = MatMult(cMat,l,cVec);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(cSec,&pStart,&pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; p++) {
      ierr = PetscSectionGetDof(cSec,p,&dof);CHKERRQ(ierr);
      if (dof) {
        PetscScalar *vals;
        ierr = VecGetValuesSection(cVec,cSec,p,&vals);CHKERRQ(ierr);
        ierr = VecSetValuesSection(l,section,p,vals,INSERT_ALL_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = VecDestroy(&cVec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
    DMGlobalToLocal - update local vectors from global vector

    Neighbor-wise Collective on dm

    Input Parameters:
+   dm - the DM object
.   g - the global vector
.   mode - INSERT_VALUES or ADD_VALUES
-   l - the local vector

    Notes:
    The communication involved in this update can be overlapped with computation by using
    DMGlobalToLocalBegin() and DMGlobalToLocalEnd().

    Level: beginner

.seealso DMCoarsen(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMGlobalToLocalEnd(), DMLocalToGlobalBegin(), DMLocalToGlobal(), DMLocalToGlobalBegin(), DMLocalToGlobalEnd()

@*/
PetscErrorCode DMGlobalToLocal(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGlobalToLocalBegin(dm,g,mode,l);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,g,mode,l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    DMGlobalToLocalBegin - Begins updating local vectors from global vector

    Neighbor-wise Collective on dm

    Input Parameters:
+   dm - the DM object
.   g - the global vector
.   mode - INSERT_VALUES or ADD_VALUES
-   l - the local vector

    Level: intermediate

.seealso DMCoarsen(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMGlobalToLocal(), DMGlobalToLocalEnd(), DMLocalToGlobalBegin(), DMLocalToGlobal(), DMLocalToGlobalBegin(), DMLocalToGlobalEnd()

@*/
PetscErrorCode  DMGlobalToLocalBegin(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscSF                 sf;
  PetscErrorCode          ierr;
  DMGlobalToLocalHookLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  for (link=dm->gtolhook; link; link=link->next) {
    if (link->beginhook) {
      ierr = (*link->beginhook)(dm,g,mode,l,link->ctx);CHKERRQ(ierr);
    }
  }
  ierr = DMGetSectionSF(dm, &sf);CHKERRQ(ierr);
  if (sf) {
    const PetscScalar *gArray;
    PetscScalar       *lArray;

    if (mode == ADD_VALUES) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %D", mode);
    ierr = VecGetArray(l, &lArray);CHKERRQ(ierr);
    ierr = VecGetArrayRead(g, &gArray);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(sf, MPIU_SCALAR, gArray, lArray);CHKERRQ(ierr);
    ierr = VecRestoreArray(l, &lArray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(g, &gArray);CHKERRQ(ierr);
  } else {
    if (!dm->ops->globaltolocalbegin) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Missing DMGlobalToLocalBegin() for type %s",((PetscObject)dm)->type_name);
    ierr = (*dm->ops->globaltolocalbegin)(dm,g,mode == INSERT_ALL_VALUES ? INSERT_VALUES : (mode == ADD_ALL_VALUES ? ADD_VALUES : mode),l);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
    DMGlobalToLocalEnd - Ends updating local vectors from global vector

    Neighbor-wise Collective on dm

    Input Parameters:
+   dm - the DM object
.   g - the global vector
.   mode - INSERT_VALUES or ADD_VALUES
-   l - the local vector

    Level: intermediate

.seealso DMCoarsen(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMGlobalToLocal(), DMLocalToGlobalBegin(), DMLocalToGlobal(), DMLocalToGlobalBegin(), DMLocalToGlobalEnd()

@*/
PetscErrorCode  DMGlobalToLocalEnd(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscSF                 sf;
  PetscErrorCode          ierr;
  const PetscScalar      *gArray;
  PetscScalar            *lArray;
  PetscBool               transform;
  DMGlobalToLocalHookLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetSectionSF(dm, &sf);CHKERRQ(ierr);
  ierr = DMHasBasisTransform(dm, &transform);CHKERRQ(ierr);
  if (sf) {
    if (mode == ADD_VALUES) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %D", mode);

    ierr = VecGetArray(l, &lArray);CHKERRQ(ierr);
    ierr = VecGetArrayRead(g, &gArray);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf, MPIU_SCALAR, gArray, lArray);CHKERRQ(ierr);
    ierr = VecRestoreArray(l, &lArray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(g, &gArray);CHKERRQ(ierr);
    if (transform) {ierr = DMPlexGlobalToLocalBasis(dm, l);CHKERRQ(ierr);}
  } else {
    if (!dm->ops->globaltolocalend) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Missing DMGlobalToLocalEnd() for type %s",((PetscObject)dm)->type_name);
    ierr = (*dm->ops->globaltolocalend)(dm,g,mode == INSERT_ALL_VALUES ? INSERT_VALUES : (mode == ADD_ALL_VALUES ? ADD_VALUES : mode),l);CHKERRQ(ierr);
  }
  ierr = DMGlobalToLocalHook_Constraints(dm,g,mode,l,NULL);CHKERRQ(ierr);
  for (link=dm->gtolhook; link; link=link->next) {
    if (link->endhook) {ierr = (*link->endhook)(dm,g,mode,l,link->ctx);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

/*@C
   DMLocalToGlobalHookAdd - adds a callback to be run when a local to global is called

   Logically Collective

   Input Arguments:
+  dm - the DM
.  beginhook - function to run at the beginning of DMLocalToGlobalBegin()
.  endhook - function to run after DMLocalToGlobalEnd() has completed
-  ctx - [optional] user-defined context for provide data for the hooks (may be NULL)

   Calling sequence for beginhook:
$    beginhook(DM fine,Vec l,InsertMode mode,Vec g,void *ctx)

+  dm - global DM
.  l - local vector
.  mode - mode
.  g - global vector
-  ctx - optional user-defined function context


   Calling sequence for endhook:
$    endhook(DM fine,Vec l,InsertMode mode,Vec g,void *ctx)

+  global - global DM
.  l - local vector
.  mode - mode
.  g - global vector
-  ctx - optional user-defined function context

   Level: advanced

.seealso: DMRefineHookAdd(), SNESFASGetInterpolation(), SNESFASGetInjection(), PetscObjectCompose(), PetscContainerCreate()
@*/
PetscErrorCode DMLocalToGlobalHookAdd(DM dm,PetscErrorCode (*beginhook)(DM,Vec,InsertMode,Vec,void*),PetscErrorCode (*endhook)(DM,Vec,InsertMode,Vec,void*),void *ctx)
{
  PetscErrorCode          ierr;
  DMLocalToGlobalHookLink link,*p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  for (p=&dm->ltoghook; *p; p=&(*p)->next) {} /* Scan to the end of the current list of hooks */
  ierr            = PetscNew(&link);CHKERRQ(ierr);
  link->beginhook = beginhook;
  link->endhook   = endhook;
  link->ctx       = ctx;
  link->next      = NULL;
  *p              = link;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMLocalToGlobalHook_Constraints(DM dm, Vec l, InsertMode mode, Vec g, void *ctx)
{
  Mat cMat;
  Vec cVec;
  PetscSection section, cSec;
  PetscInt pStart, pEnd, p, dof;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDefaultConstraints(dm,&cSec,&cMat);CHKERRQ(ierr);
  if (cMat && (mode == ADD_VALUES || mode == ADD_ALL_VALUES || mode == ADD_BC_VALUES)) {
    PetscInt nRows;

    ierr = MatGetSize(cMat,&nRows,NULL);CHKERRQ(ierr);
    if (nRows <= 0) PetscFunctionReturn(0);
    ierr = DMGetLocalSection(dm,&section);CHKERRQ(ierr);
    ierr = MatCreateVecs(cMat,NULL,&cVec);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(cSec,&pStart,&pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; p++) {
      ierr = PetscSectionGetDof(cSec,p,&dof);CHKERRQ(ierr);
      if (dof) {
        PetscInt d;
        PetscScalar *vals;
        ierr = VecGetValuesSection(l,section,p,&vals);CHKERRQ(ierr);
        ierr = VecSetValuesSection(cVec,cSec,p,vals,mode);CHKERRQ(ierr);
        /* for this to be the true transpose, we have to zero the values that
         * we just extracted */
        for (d = 0; d < dof; d++) {
          vals[d] = 0.;
        }
      }
    }
    ierr = MatMultTransposeAdd(cMat,cVec,l,l);CHKERRQ(ierr);
    ierr = VecDestroy(&cVec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/*@
    DMLocalToGlobal - updates global vectors from local vectors

    Neighbor-wise Collective on dm

    Input Parameters:
+   dm - the DM object
.   l - the local vector
.   mode - if INSERT_VALUES then no parallel communication is used, if ADD_VALUES then all ghost points from the same base point accumulate into that base point.
-   g - the global vector

    Notes:
    The communication involved in this update can be overlapped with computation by using
    DMLocalToGlobalBegin() and DMLocalToGlobalEnd().

    In the ADD_VALUES case you normally would zero the receiving vector before beginning this operation.
           INSERT_VALUES is not supported for DMDA; in that case simply compute the values directly into a global vector instead of a local one.

    Level: beginner

.seealso DMLocalToGlobalBegin(), DMLocalToGlobalEnd(), DMCoarsen(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMGlobalToLocal(), DMGlobalToLocalEnd(), DMGlobalToLocalBegin()

@*/
PetscErrorCode DMLocalToGlobal(DM dm,Vec l,InsertMode mode,Vec g)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMLocalToGlobalBegin(dm,l,mode,g);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm,l,mode,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    DMLocalToGlobalBegin - begins updating global vectors from local vectors

    Neighbor-wise Collective on dm

    Input Parameters:
+   dm - the DM object
.   l - the local vector
.   mode - if INSERT_VALUES then no parallel communication is used, if ADD_VALUES then all ghost points from the same base point accumulate into that base point.
-   g - the global vector

    Notes:
    In the ADD_VALUES case you normally would zero the receiving vector before beginning this operation.
           INSERT_VALUES is not supported for DMDA, in that case simply compute the values directly into a global vector instead of a local one.

    Level: intermediate

.seealso DMLocalToGlobal(), DMLocalToGlobalEnd(), DMCoarsen(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMGlobalToLocal(), DMGlobalToLocalEnd(), DMGlobalToLocalBegin()

@*/
PetscErrorCode  DMLocalToGlobalBegin(DM dm,Vec l,InsertMode mode,Vec g)
{
  PetscSF                 sf;
  PetscSection            s, gs;
  DMLocalToGlobalHookLink link;
  Vec                     tmpl;
  const PetscScalar      *lArray;
  PetscScalar            *gArray;
  PetscBool               isInsert, transform;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  for (link=dm->ltoghook; link; link=link->next) {
    if (link->beginhook) {
      ierr = (*link->beginhook)(dm,l,mode,g,link->ctx);CHKERRQ(ierr);
    }
  }
  ierr = DMLocalToGlobalHook_Constraints(dm,l,mode,g,NULL);CHKERRQ(ierr);
  ierr = DMGetSectionSF(dm, &sf);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
  switch (mode) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
  case INSERT_BC_VALUES:
    isInsert = PETSC_TRUE; break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
  case ADD_BC_VALUES:
    isInsert = PETSC_FALSE; break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %D", mode);
  }
  if ((sf && !isInsert) || (s && isInsert)) {
    ierr = DMHasBasisTransform(dm, &transform);CHKERRQ(ierr);
    if (transform) {
      ierr = DMGetNamedLocalVector(dm, "__petsc_dm_transform_local_copy", &tmpl);CHKERRQ(ierr);
      ierr = VecCopy(l, tmpl);CHKERRQ(ierr);
      ierr = DMPlexLocalToGlobalBasis(dm, tmpl);CHKERRQ(ierr);
      ierr = VecGetArrayRead(tmpl, &lArray);CHKERRQ(ierr);
    } else {
      ierr = VecGetArrayRead(l, &lArray);CHKERRQ(ierr);
    }
    ierr = VecGetArray(g, &gArray);CHKERRQ(ierr);
    if (sf && !isInsert) {
      ierr = PetscSFReduceBegin(sf, MPIU_SCALAR, lArray, gArray, MPIU_SUM);CHKERRQ(ierr);
    } else if (s && isInsert) {
      PetscInt gStart, pStart, pEnd, p;

      ierr = DMGetGlobalSection(dm, &gs);CHKERRQ(ierr);
      ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(g, &gStart, NULL);CHKERRQ(ierr);
      for (p = pStart; p < pEnd; ++p) {
        PetscInt dof, gdof, cdof, gcdof, off, goff, d, e;

        ierr = PetscSectionGetDof(s, p, &dof);CHKERRQ(ierr);
        ierr = PetscSectionGetDof(gs, p, &gdof);CHKERRQ(ierr);
        ierr = PetscSectionGetConstraintDof(s, p, &cdof);CHKERRQ(ierr);
        ierr = PetscSectionGetConstraintDof(gs, p, &gcdof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(s, p, &off);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(gs, p, &goff);CHKERRQ(ierr);
        /* Ignore off-process data and points with no global data */
        if (!gdof || goff < 0) continue;
        if (dof != gdof) SETERRQ5(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Inconsistent sizes, p: %d dof: %d gdof: %d cdof: %d gcdof: %d", p, dof, gdof, cdof, gcdof);
        /* If no constraints are enforced in the global vector */
        if (!gcdof) {
          for (d = 0; d < dof; ++d) gArray[goff-gStart+d] = lArray[off+d];
          /* If constraints are enforced in the global vector */
        } else if (cdof == gcdof) {
          const PetscInt *cdofs;
          PetscInt        cind = 0;

          ierr = PetscSectionGetConstraintIndices(s, p, &cdofs);CHKERRQ(ierr);
          for (d = 0, e = 0; d < dof; ++d) {
            if ((cind < cdof) && (d == cdofs[cind])) {++cind; continue;}
            gArray[goff-gStart+e++] = lArray[off+d];
          }
        } else SETERRQ5(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Inconsistent sizes, p: %d dof: %d gdof: %d cdof: %d gcdof: %d", p, dof, gdof, cdof, gcdof);
      }
    }
    ierr = VecRestoreArray(g, &gArray);CHKERRQ(ierr);
    if (transform) {
      ierr = VecRestoreArrayRead(tmpl, &lArray);CHKERRQ(ierr);
      ierr = DMRestoreNamedLocalVector(dm, "__petsc_dm_transform_local_copy", &tmpl);CHKERRQ(ierr);
    } else {
      ierr = VecRestoreArrayRead(l, &lArray);CHKERRQ(ierr);
    }
  } else {
    if (!dm->ops->localtoglobalbegin) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Missing DMLocalToGlobalBegin() for type %s",((PetscObject)dm)->type_name);
    ierr = (*dm->ops->localtoglobalbegin)(dm,l,mode == INSERT_ALL_VALUES ? INSERT_VALUES : (mode == ADD_ALL_VALUES ? ADD_VALUES : mode),g);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
    DMLocalToGlobalEnd - updates global vectors from local vectors

    Neighbor-wise Collective on dm

    Input Parameters:
+   dm - the DM object
.   l - the local vector
.   mode - INSERT_VALUES or ADD_VALUES
-   g - the global vector

    Level: intermediate

.seealso DMCoarsen(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMGlobalToLocalEnd(), DMGlobalToLocalEnd()

@*/
PetscErrorCode  DMLocalToGlobalEnd(DM dm,Vec l,InsertMode mode,Vec g)
{
  PetscSF                 sf;
  PetscSection            s;
  DMLocalToGlobalHookLink link;
  PetscBool               isInsert, transform;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetSectionSF(dm, &sf);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
  switch (mode) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    isInsert = PETSC_TRUE; break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    isInsert = PETSC_FALSE; break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %D", mode);
  }
  if (sf && !isInsert) {
    const PetscScalar *lArray;
    PetscScalar       *gArray;
    Vec                tmpl;

    ierr = DMHasBasisTransform(dm, &transform);CHKERRQ(ierr);
    if (transform) {
      ierr = DMGetNamedLocalVector(dm, "__petsc_dm_transform_local_copy", &tmpl);CHKERRQ(ierr);
      ierr = VecGetArrayRead(tmpl, &lArray);CHKERRQ(ierr);
    } else {
      ierr = VecGetArrayRead(l, &lArray);CHKERRQ(ierr);
    }
    ierr = VecGetArray(g, &gArray);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(sf, MPIU_SCALAR, lArray, gArray, MPIU_SUM);CHKERRQ(ierr);
    if (transform) {
      ierr = VecRestoreArrayRead(tmpl, &lArray);CHKERRQ(ierr);
      ierr = DMRestoreNamedLocalVector(dm, "__petsc_dm_transform_local_copy", &tmpl);CHKERRQ(ierr);
    } else {
      ierr = VecRestoreArrayRead(l, &lArray);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(g, &gArray);CHKERRQ(ierr);
  } else if (s && isInsert) {
  } else {
    if (!dm->ops->localtoglobalend) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Missing DMLocalToGlobalEnd() for type %s",((PetscObject)dm)->type_name);
    ierr = (*dm->ops->localtoglobalend)(dm,l,mode == INSERT_ALL_VALUES ? INSERT_VALUES : (mode == ADD_ALL_VALUES ? ADD_VALUES : mode),g);CHKERRQ(ierr);
  }
  for (link=dm->ltoghook; link; link=link->next) {
    if (link->endhook) {ierr = (*link->endhook)(dm,g,mode,l,link->ctx);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

/*@
   DMLocalToLocalBegin - Maps from a local vector (including ghost points
   that contain irrelevant values) to another local vector where the ghost
   points in the second are set correctly. Must be followed by DMLocalToLocalEnd().

   Neighbor-wise Collective on dm

   Input Parameters:
+  dm - the DM object
.  g - the original local vector
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the local vector with correct ghost values

   Level: intermediate

   Notes:
   The local vectors used here need not be the same as those
   obtained from DMCreateLocalVector(), BUT they
   must have the same parallel data layout; they could, for example, be
   obtained with VecDuplicate() from the DM originating vectors.

.seealso DMCoarsen(), DMDestroy(), DMView(), DMCreateLocalVector(), DMCreateGlobalVector(), DMCreateInterpolation(), DMLocalToLocalEnd(), DMGlobalToLocalEnd(), DMLocalToGlobalBegin()

@*/
PetscErrorCode  DMLocalToLocalBegin(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (!dm->ops->localtolocalbegin) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"This DM does not support local to local maps");
  ierr = (*dm->ops->localtolocalbegin)(dm,g,mode == INSERT_ALL_VALUES ? INSERT_VALUES : (mode == ADD_ALL_VALUES ? ADD_VALUES : mode),l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   DMLocalToLocalEnd - Maps from a local vector (including ghost points
   that contain irrelevant values) to another local vector where the ghost
   points in the second are set correctly. Must be preceded by DMLocalToLocalBegin().

   Neighbor-wise Collective on dm

   Input Parameters:
+  da - the DM object
.  g - the original local vector
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the local vector with correct ghost values

   Level: intermediate

   Notes:
   The local vectors used here need not be the same as those
   obtained from DMCreateLocalVector(), BUT they
   must have the same parallel data layout; they could, for example, be
   obtained with VecDuplicate() from the DM originating vectors.

.seealso DMCoarsen(), DMDestroy(), DMView(), DMCreateLocalVector(), DMCreateGlobalVector(), DMCreateInterpolation(), DMLocalToLocalBegin(), DMGlobalToLocalEnd(), DMLocalToGlobalBegin()

@*/
PetscErrorCode  DMLocalToLocalEnd(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (!dm->ops->localtolocalend) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"This DM does not support local to local maps");
  ierr = (*dm->ops->localtolocalend)(dm,g,mode == INSERT_ALL_VALUES ? INSERT_VALUES : (mode == ADD_ALL_VALUES ? ADD_VALUES : mode),l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*@
    DMCoarsen - Coarsens a DM object

    Collective on dm

    Input Parameter:
+   dm - the DM object
-   comm - the communicator to contain the new DM object (or MPI_COMM_NULL)

    Output Parameter:
.   dmc - the coarsened DM

    Level: developer

.seealso DMRefine(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation()

@*/
PetscErrorCode DMCoarsen(DM dm, MPI_Comm comm, DM *dmc)
{
  PetscErrorCode    ierr;
  DMCoarsenHookLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (!dm->ops->coarsen) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMCoarsen",((PetscObject)dm)->type_name);
  ierr = PetscLogEventBegin(DM_Coarsen,dm,0,0,0);CHKERRQ(ierr);
  ierr = (*dm->ops->coarsen)(dm, comm, dmc);CHKERRQ(ierr);
  if (*dmc) {
    ierr = DMSetCoarseDM(dm,*dmc);CHKERRQ(ierr);
    (*dmc)->ops->creatematrix = dm->ops->creatematrix;
    ierr                      = PetscObjectCopyFortranFunctionPointers((PetscObject)dm,(PetscObject)*dmc);CHKERRQ(ierr);
    (*dmc)->ctx               = dm->ctx;
    (*dmc)->levelup           = dm->levelup;
    (*dmc)->leveldown         = dm->leveldown + 1;
    ierr                      = DMSetMatType(*dmc,dm->mattype);CHKERRQ(ierr);
    for (link=dm->coarsenhook; link; link=link->next) {
      if (link->coarsenhook) {ierr = (*link->coarsenhook)(dm,*dmc,link->ctx);CHKERRQ(ierr);}
    }
  }
  ierr = PetscLogEventEnd(DM_Coarsen,dm,0,0,0);CHKERRQ(ierr);
  if (!(*dmc)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "NULL coarse mesh produced");
  PetscFunctionReturn(0);
}

/*@C
   DMCoarsenHookAdd - adds a callback to be run when restricting a nonlinear problem to the coarse grid

   Logically Collective

   Input Arguments:
+  fine - nonlinear solver context on which to run a hook when restricting to a coarser level
.  coarsenhook - function to run when setting up a coarser level
.  restricthook - function to run to update data on coarser levels (once per SNESSolve())
-  ctx - [optional] user-defined context for provide data for the hooks (may be NULL)

   Calling sequence of coarsenhook:
$    coarsenhook(DM fine,DM coarse,void *ctx);

+  fine - fine level DM
.  coarse - coarse level DM to restrict problem to
-  ctx - optional user-defined function context

   Calling sequence for restricthook:
$    restricthook(DM fine,Mat mrestrict,Vec rscale,Mat inject,DM coarse,void *ctx)

+  fine - fine level DM
.  mrestrict - matrix restricting a fine-level solution to the coarse grid
.  rscale - scaling vector for restriction
.  inject - matrix restricting by injection
.  coarse - coarse level DM to update
-  ctx - optional user-defined function context

   Level: advanced

   Notes:
   This function is only needed if auxiliary data needs to be set up on coarse grids.

   If this function is called multiple times, the hooks will be run in the order they are added.

   In order to compose with nonlinear preconditioning without duplicating storage, the hook should be implemented to
   extract the finest level information from its context (instead of from the SNES).

   This function is currently not available from Fortran.

.seealso: DMCoarsenHookRemove(), DMRefineHookAdd(), SNESFASGetInterpolation(), SNESFASGetInjection(), PetscObjectCompose(), PetscContainerCreate()
@*/
PetscErrorCode DMCoarsenHookAdd(DM fine,PetscErrorCode (*coarsenhook)(DM,DM,void*),PetscErrorCode (*restricthook)(DM,Mat,Vec,Mat,DM,void*),void *ctx)
{
  PetscErrorCode    ierr;
  DMCoarsenHookLink link,*p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fine,DM_CLASSID,1);
  for (p=&fine->coarsenhook; *p; p=&(*p)->next) { /* Scan to the end of the current list of hooks */
    if ((*p)->coarsenhook == coarsenhook && (*p)->restricthook == restricthook && (*p)->ctx == ctx) PetscFunctionReturn(0);
  }
  ierr               = PetscNew(&link);CHKERRQ(ierr);
  link->coarsenhook  = coarsenhook;
  link->restricthook = restricthook;
  link->ctx          = ctx;
  link->next         = NULL;
  *p                 = link;
  PetscFunctionReturn(0);
}

/*@C
   DMCoarsenHookRemove - remove a callback from the list of hooks to be run when restricting a nonlinear problem to the coarse grid

   Logically Collective

   Input Arguments:
+  fine - nonlinear solver context on which to run a hook when restricting to a coarser level
.  coarsenhook - function to run when setting up a coarser level
.  restricthook - function to run to update data on coarser levels (once per SNESSolve())
-  ctx - [optional] user-defined context for provide data for the hooks (may be NULL)

   Level: advanced

   Notes:
   This function does nothing if the hook is not in the list.

   This function is currently not available from Fortran.

.seealso: DMCoarsenHookAdd(), DMRefineHookAdd(), SNESFASGetInterpolation(), SNESFASGetInjection(), PetscObjectCompose(), PetscContainerCreate()
@*/
PetscErrorCode DMCoarsenHookRemove(DM fine,PetscErrorCode (*coarsenhook)(DM,DM,void*),PetscErrorCode (*restricthook)(DM,Mat,Vec,Mat,DM,void*),void *ctx)
{
  PetscErrorCode    ierr;
  DMCoarsenHookLink link,*p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fine,DM_CLASSID,1);
  for (p=&fine->coarsenhook; *p; p=&(*p)->next) { /* Search the list of current hooks */
    if ((*p)->coarsenhook == coarsenhook && (*p)->restricthook == restricthook && (*p)->ctx == ctx) {
      link = *p;
      *p = link->next;
      ierr = PetscFree(link);CHKERRQ(ierr);
      break;
    }
  }
  PetscFunctionReturn(0);
}


/*@
   DMRestrict - restricts user-defined problem data to a coarser DM by running hooks registered by DMCoarsenHookAdd()

   Collective if any hooks are

   Input Arguments:
+  fine - finer DM to use as a base
.  restrct - restriction matrix, apply using MatRestrict()
.  rscale - scaling vector for restriction
.  inject - injection matrix, also use MatRestrict()
-  coarse - coarser DM to update

   Level: developer

.seealso: DMCoarsenHookAdd(), MatRestrict()
@*/
PetscErrorCode DMRestrict(DM fine,Mat restrct,Vec rscale,Mat inject,DM coarse)
{
  PetscErrorCode    ierr;
  DMCoarsenHookLink link;

  PetscFunctionBegin;
  for (link=fine->coarsenhook; link; link=link->next) {
    if (link->restricthook) {
      ierr = (*link->restricthook)(fine,restrct,rscale,inject,coarse,link->ctx);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   DMSubDomainHookAdd - adds a callback to be run when restricting a problem to the coarse grid

   Logically Collective on global

   Input Arguments:
+  global - global DM
.  ddhook - function to run to pass data to the decomposition DM upon its creation
.  restricthook - function to run to update data on block solve (at the beginning of the block solve)
-  ctx - [optional] user-defined context for provide data for the hooks (may be NULL)


   Calling sequence for ddhook:
$    ddhook(DM global,DM block,void *ctx)

+  global - global DM
.  block  - block DM
-  ctx - optional user-defined function context

   Calling sequence for restricthook:
$    restricthook(DM global,VecScatter out,VecScatter in,DM block,void *ctx)

+  global - global DM
.  out    - scatter to the outer (with ghost and overlap points) block vector
.  in     - scatter to block vector values only owned locally
.  block  - block DM
-  ctx - optional user-defined function context

   Level: advanced

   Notes:
   This function is only needed if auxiliary data needs to be set up on subdomain DMs.

   If this function is called multiple times, the hooks will be run in the order they are added.

   In order to compose with nonlinear preconditioning without duplicating storage, the hook should be implemented to
   extract the global information from its context (instead of from the SNES).

   This function is currently not available from Fortran.

.seealso: DMRefineHookAdd(), SNESFASGetInterpolation(), SNESFASGetInjection(), PetscObjectCompose(), PetscContainerCreate()
@*/
PetscErrorCode DMSubDomainHookAdd(DM global,PetscErrorCode (*ddhook)(DM,DM,void*),PetscErrorCode (*restricthook)(DM,VecScatter,VecScatter,DM,void*),void *ctx)
{
  PetscErrorCode      ierr;
  DMSubDomainHookLink link,*p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(global,DM_CLASSID,1);
  for (p=&global->subdomainhook; *p; p=&(*p)->next) { /* Scan to the end of the current list of hooks */
    if ((*p)->ddhook == ddhook && (*p)->restricthook == restricthook && (*p)->ctx == ctx) PetscFunctionReturn(0);
  }
  ierr               = PetscNew(&link);CHKERRQ(ierr);
  link->restricthook = restricthook;
  link->ddhook       = ddhook;
  link->ctx          = ctx;
  link->next         = NULL;
  *p                 = link;
  PetscFunctionReturn(0);
}

/*@C
   DMSubDomainHookRemove - remove a callback from the list to be run when restricting a problem to the coarse grid

   Logically Collective

   Input Arguments:
+  global - global DM
.  ddhook - function to run to pass data to the decomposition DM upon its creation
.  restricthook - function to run to update data on block solve (at the beginning of the block solve)
-  ctx - [optional] user-defined context for provide data for the hooks (may be NULL)

   Level: advanced

   Notes:

   This function is currently not available from Fortran.

.seealso: DMSubDomainHookAdd(), SNESFASGetInterpolation(), SNESFASGetInjection(), PetscObjectCompose(), PetscContainerCreate()
@*/
PetscErrorCode DMSubDomainHookRemove(DM global,PetscErrorCode (*ddhook)(DM,DM,void*),PetscErrorCode (*restricthook)(DM,VecScatter,VecScatter,DM,void*),void *ctx)
{
  PetscErrorCode      ierr;
  DMSubDomainHookLink link,*p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(global,DM_CLASSID,1);
  for (p=&global->subdomainhook; *p; p=&(*p)->next) { /* Search the list of current hooks */
    if ((*p)->ddhook == ddhook && (*p)->restricthook == restricthook && (*p)->ctx == ctx) {
      link = *p;
      *p = link->next;
      ierr = PetscFree(link);CHKERRQ(ierr);
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*@
   DMSubDomainRestrict - restricts user-defined problem data to a block DM by running hooks registered by DMSubDomainHookAdd()

   Collective if any hooks are

   Input Arguments:
+  fine - finer DM to use as a base
.  oscatter - scatter from domain global vector filling subdomain global vector with overlap
.  gscatter - scatter from domain global vector filling subdomain local vector with ghosts
-  coarse - coarer DM to update

   Level: developer

.seealso: DMCoarsenHookAdd(), MatRestrict()
@*/
PetscErrorCode DMSubDomainRestrict(DM global,VecScatter oscatter,VecScatter gscatter,DM subdm)
{
  PetscErrorCode      ierr;
  DMSubDomainHookLink link;

  PetscFunctionBegin;
  for (link=global->subdomainhook; link; link=link->next) {
    if (link->restricthook) {
      ierr = (*link->restricthook)(global,oscatter,gscatter,subdm,link->ctx);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@
    DMGetCoarsenLevel - Get's the number of coarsenings that have generated this DM.

    Not Collective

    Input Parameter:
.   dm - the DM object

    Output Parameter:
.   level - number of coarsenings

    Level: developer

.seealso DMCoarsen(), DMGetRefineLevel(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation()

@*/
PetscErrorCode  DMGetCoarsenLevel(DM dm,PetscInt *level)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidIntPointer(level,2);
  *level = dm->leveldown;
  PetscFunctionReturn(0);
}

/*@
    DMSetCoarsenLevel - Sets the number of coarsenings that have generated this DM.

    Not Collective

    Input Parameters:
+   dm - the DM object
-   level - number of coarsenings

    Level: developer

.seealso DMCoarsen(), DMGetCoarsenLevel(), DMGetRefineLevel(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation()
@*/
PetscErrorCode DMSetCoarsenLevel(DM dm,PetscInt level)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->leveldown = level;
  PetscFunctionReturn(0);
}



/*@C
    DMRefineHierarchy - Refines a DM object, all levels at once

    Collective on dm

    Input Parameter:
+   dm - the DM object
-   nlevels - the number of levels of refinement

    Output Parameter:
.   dmf - the refined DM hierarchy

    Level: developer

.seealso DMCoarsenHierarchy(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation()

@*/
PetscErrorCode  DMRefineHierarchy(DM dm,PetscInt nlevels,DM dmf[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (nlevels < 0) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"nlevels cannot be negative");
  if (nlevels == 0) PetscFunctionReturn(0);
  PetscValidPointer(dmf,3);
  if (dm->ops->refinehierarchy) {
    ierr = (*dm->ops->refinehierarchy)(dm,nlevels,dmf);CHKERRQ(ierr);
  } else if (dm->ops->refine) {
    PetscInt i;

    ierr = DMRefine(dm,PetscObjectComm((PetscObject)dm),&dmf[0]);CHKERRQ(ierr);
    for (i=1; i<nlevels; i++) {
      ierr = DMRefine(dmf[i-1],PetscObjectComm((PetscObject)dm),&dmf[i]);CHKERRQ(ierr);
    }
  } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"No RefineHierarchy for this DM yet");
  PetscFunctionReturn(0);
}

/*@C
    DMCoarsenHierarchy - Coarsens a DM object, all levels at once

    Collective on dm

    Input Parameter:
+   dm - the DM object
-   nlevels - the number of levels of coarsening

    Output Parameter:
.   dmc - the coarsened DM hierarchy

    Level: developer

.seealso DMRefineHierarchy(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateInterpolation()

@*/
PetscErrorCode  DMCoarsenHierarchy(DM dm, PetscInt nlevels, DM dmc[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (nlevels < 0) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"nlevels cannot be negative");
  if (nlevels == 0) PetscFunctionReturn(0);
  PetscValidPointer(dmc,3);
  if (dm->ops->coarsenhierarchy) {
    ierr = (*dm->ops->coarsenhierarchy)(dm, nlevels, dmc);CHKERRQ(ierr);
  } else if (dm->ops->coarsen) {
    PetscInt i;

    ierr = DMCoarsen(dm,PetscObjectComm((PetscObject)dm),&dmc[0]);CHKERRQ(ierr);
    for (i=1; i<nlevels; i++) {
      ierr = DMCoarsen(dmc[i-1],PetscObjectComm((PetscObject)dm),&dmc[i]);CHKERRQ(ierr);
    }
  } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"No CoarsenHierarchy for this DM yet");
  PetscFunctionReturn(0);
}

/*@C
    DMSetApplicationContextDestroy - Sets a user function that will be called to destroy the application context when the DM is destroyed

    Not Collective

    Input Parameters:
+   dm - the DM object
-   destroy - the destroy function

    Level: intermediate

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext()

@*/
PetscErrorCode  DMSetApplicationContextDestroy(DM dm,PetscErrorCode (*destroy)(void**))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->ctxdestroy = destroy;
  PetscFunctionReturn(0);
}

/*@
    DMSetApplicationContext - Set a user context into a DM object

    Not Collective

    Input Parameters:
+   dm - the DM object
-   ctx - the user context

    Level: intermediate

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext()

@*/
PetscErrorCode  DMSetApplicationContext(DM dm,void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->ctx = ctx;
  PetscFunctionReturn(0);
}

/*@
    DMGetApplicationContext - Gets a user context from a DM object

    Not Collective

    Input Parameter:
.   dm - the DM object

    Output Parameter:
.   ctx - the user context

    Level: intermediate

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext()

@*/
PetscErrorCode  DMGetApplicationContext(DM dm,void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *(void**)ctx = dm->ctx;
  PetscFunctionReturn(0);
}

/*@C
    DMSetVariableBounds - sets a function to compute the lower and upper bound vectors for SNESVI.

    Logically Collective on dm

    Input Parameter:
+   dm - the DM object
-   f - the function that computes variable bounds used by SNESVI (use NULL to cancel a previous function that was set)

    Level: intermediate

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext(),
         DMSetJacobian()

@*/
PetscErrorCode DMSetVariableBounds(DM dm,PetscErrorCode (*f)(DM,Vec,Vec))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->ops->computevariablebounds = f;
  PetscFunctionReturn(0);
}

/*@
    DMHasVariableBounds - does the DM object have a variable bounds function?

    Not Collective

    Input Parameter:
.   dm - the DM object to destroy

    Output Parameter:
.   flg - PETSC_TRUE if the variable bounds function exists

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext()

@*/
PetscErrorCode DMHasVariableBounds(DM dm,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  *flg =  (dm->ops->computevariablebounds) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
    DMComputeVariableBounds - compute variable bounds used by SNESVI.

    Logically Collective on dm

    Input Parameters:
.   dm - the DM object

    Output parameters:
+   xl - lower bound
-   xu - upper bound

    Level: advanced

    Notes:
    This is generally not called by users. It calls the function provided by the user with DMSetVariableBounds()

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext()

@*/
PetscErrorCode DMComputeVariableBounds(DM dm, Vec xl, Vec xu)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(xl,VEC_CLASSID,2);
  PetscValidHeaderSpecific(xu,VEC_CLASSID,3);
  if (!dm->ops->computevariablebounds) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMComputeVariableBounds",((PetscObject)dm)->type_name);
  ierr = (*dm->ops->computevariablebounds)(dm, xl,xu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    DMHasColoring - does the DM object have a method of providing a coloring?

    Not Collective

    Input Parameter:
.   dm - the DM object

    Output Parameter:
.   flg - PETSC_TRUE if the DM has facilities for DMCreateColoring().

    Level: developer

.seealso DMCreateColoring()

@*/
PetscErrorCode DMHasColoring(DM dm,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  *flg =  (dm->ops->getcoloring) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
    DMHasCreateRestriction - does the DM object have a method of providing a restriction?

    Not Collective

    Input Parameter:
.   dm - the DM object

    Output Parameter:
.   flg - PETSC_TRUE if the DM has facilities for DMCreateRestriction().

    Level: developer

.seealso DMCreateRestriction()

@*/
PetscErrorCode DMHasCreateRestriction(DM dm,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  *flg =  (dm->ops->createrestriction) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}


/*@
    DMHasCreateInjection - does the DM object have a method of providing an injection?

    Not Collective

    Input Parameter:
.   dm - the DM object

    Output Parameter:
.   flg - PETSC_TRUE if the DM has facilities for DMCreateInjection().

    Level: developer

.seealso DMCreateInjection()

@*/
PetscErrorCode DMHasCreateInjection(DM dm,PetscBool *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  if (dm->ops->hascreateinjection) {
    ierr = (*dm->ops->hascreateinjection)(dm,flg);CHKERRQ(ierr);
  } else {
    *flg = (dm->ops->createinjection) ? PETSC_TRUE : PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}


/*@C
    DMSetVec - set the vector at which to compute residual, Jacobian and VI bounds, if the problem is nonlinear.

    Collective on dm

    Input Parameter:
+   dm - the DM object
-   x - location to compute residual and Jacobian, if NULL is passed to those routines; will be NULL for linear problems.

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMGetApplicationContext()

@*/
PetscErrorCode  DMSetVec(DM dm,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (x) {
    if (!dm->x) {
      ierr = DMCreateGlobalVector(dm,&dm->x);CHKERRQ(ierr);
    }
    ierr = VecCopy(x,dm->x);CHKERRQ(ierr);
  } else if (dm->x) {
    ierr = VecDestroy(&dm->x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscFunctionList DMList              = NULL;
PetscBool         DMRegisterAllCalled = PETSC_FALSE;

/*@C
  DMSetType - Builds a DM, for a particular DM implementation.

  Collective on dm

  Input Parameters:
+ dm     - The DM object
- method - The name of the DM type

  Options Database Key:
. -dm_type <type> - Sets the DM type; use -help for a list of available types

  Notes:
  See "petsc/include/petscdm.h" for available DM types (for instance, DM1D, DM2D, or DM3D).

  Level: intermediate

.seealso: DMGetType(), DMCreate()
@*/
PetscErrorCode  DMSetType(DM dm, DMType method)
{
  PetscErrorCode (*r)(DM);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject) dm, method, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = DMRegisterAll();CHKERRQ(ierr);
  ierr = PetscFunctionListFind(DMList,method,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown DM type: %s", method);

  if (dm->ops->destroy) {
    ierr = (*dm->ops->destroy)(dm);CHKERRQ(ierr);
  }
  ierr = PetscMemzero(dm->ops,sizeof(*dm->ops));CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)dm,method);CHKERRQ(ierr);
  ierr = (*r)(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMGetType - Gets the DM type name (as a string) from the DM.

  Not Collective

  Input Parameter:
. dm  - The DM

  Output Parameter:
. type - The DM type name

  Level: intermediate

.seealso: DMSetType(), DMCreate()
@*/
PetscErrorCode  DMGetType(DM dm, DMType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID,1);
  PetscValidPointer(type,2);
  ierr = DMRegisterAll();CHKERRQ(ierr);
  *type = ((PetscObject)dm)->type_name;
  PetscFunctionReturn(0);
}

/*@C
  DMConvert - Converts a DM to another DM, either of the same or different type.

  Collective on dm

  Input Parameters:
+ dm - the DM
- newtype - new DM type (use "same" for the same type)

  Output Parameter:
. M - pointer to new DM

  Notes:
  Cannot be used to convert a sequential DM to parallel or parallel to sequential,
  the MPI communicator of the generated DM is always the same as the communicator
  of the input DM.

  Level: intermediate

.seealso: DMCreate()
@*/
PetscErrorCode DMConvert(DM dm, DMType newtype, DM *M)
{
  DM             B;
  char           convname[256];
  PetscBool      sametype/*, issame */;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidType(dm,1);
  PetscValidPointer(M,3);
  ierr = PetscObjectTypeCompare((PetscObject) dm, newtype, &sametype);CHKERRQ(ierr);
  /* ierr = PetscStrcmp(newtype, "same", &issame);CHKERRQ(ierr); */
  if (sametype) {
    *M   = dm;
    ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else {
    PetscErrorCode (*conv)(DM, DMType, DM*) = NULL;

    /*
       Order of precedence:
       1) See if a specialized converter is known to the current DM.
       2) See if a specialized converter is known to the desired DM class.
       3) See if a good general converter is registered for the desired class
       4) See if a good general converter is known for the current matrix.
       5) Use a really basic converter.
    */

    /* 1) See if a specialized converter is known to the current DM and the desired class */
    ierr = PetscStrncpy(convname,"DMConvert_",sizeof(convname));CHKERRQ(ierr);
    ierr = PetscStrlcat(convname,((PetscObject) dm)->type_name,sizeof(convname));CHKERRQ(ierr);
    ierr = PetscStrlcat(convname,"_",sizeof(convname));CHKERRQ(ierr);
    ierr = PetscStrlcat(convname,newtype,sizeof(convname));CHKERRQ(ierr);
    ierr = PetscStrlcat(convname,"_C",sizeof(convname));CHKERRQ(ierr);
    ierr = PetscObjectQueryFunction((PetscObject)dm,convname,&conv);CHKERRQ(ierr);
    if (conv) goto foundconv;

    /* 2)  See if a specialized converter is known to the desired DM class. */
    ierr = DMCreate(PetscObjectComm((PetscObject)dm), &B);CHKERRQ(ierr);
    ierr = DMSetType(B, newtype);CHKERRQ(ierr);
    ierr = PetscStrncpy(convname,"DMConvert_",sizeof(convname));CHKERRQ(ierr);
    ierr = PetscStrlcat(convname,((PetscObject) dm)->type_name,sizeof(convname));CHKERRQ(ierr);
    ierr = PetscStrlcat(convname,"_",sizeof(convname));CHKERRQ(ierr);
    ierr = PetscStrlcat(convname,newtype,sizeof(convname));CHKERRQ(ierr);
    ierr = PetscStrlcat(convname,"_C",sizeof(convname));CHKERRQ(ierr);
    ierr = PetscObjectQueryFunction((PetscObject)B,convname,&conv);CHKERRQ(ierr);
    if (conv) {
      ierr = DMDestroy(&B);CHKERRQ(ierr);
      goto foundconv;
    }

#if 0
    /* 3) See if a good general converter is registered for the desired class */
    conv = B->ops->convertfrom;
    ierr = DMDestroy(&B);CHKERRQ(ierr);
    if (conv) goto foundconv;

    /* 4) See if a good general converter is known for the current matrix */
    if (dm->ops->convert) {
      conv = dm->ops->convert;
    }
    if (conv) goto foundconv;
#endif

    /* 5) Use a really basic converter. */
    SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "No conversion possible between DM types %s and %s", ((PetscObject) dm)->type_name, newtype);

foundconv:
    ierr = PetscLogEventBegin(DM_Convert,dm,0,0,0);CHKERRQ(ierr);
    ierr = (*conv)(dm,newtype,M);CHKERRQ(ierr);
    /* Things that are independent of DM type: We should consult DMClone() here */
    {
      PetscBool             isper;
      const PetscReal      *maxCell, *L;
      const DMBoundaryType *bd;
      ierr = DMGetPeriodicity(dm, &isper, &maxCell, &L, &bd);CHKERRQ(ierr);
      ierr = DMSetPeriodicity(*M, isper, maxCell,  L,  bd);CHKERRQ(ierr);
    }
    ierr = PetscLogEventEnd(DM_Convert,dm,0,0,0);CHKERRQ(ierr);
  }
  ierr = PetscObjectStateIncrease((PetscObject) *M);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*--------------------------------------------------------------------------------------------------------------------*/

/*@C
  DMRegister -  Adds a new DM component implementation

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  DMRegister() may be called multiple times to add several user-defined DMs


  Sample usage:
.vb
    DMRegister("my_da", MyDMCreate);
.ve

  Then, your DM type can be chosen with the procedural interface via
.vb
    DMCreate(MPI_Comm, DM *);
    DMSetType(DM,"my_da");
.ve
   or at runtime via the option
.vb
    -da_type my_da
.ve

  Level: advanced

.seealso: DMRegisterAll(), DMRegisterDestroy()

@*/
PetscErrorCode  DMRegister(const char sname[],PetscErrorCode (*function)(DM))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&DMList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMLoad - Loads a DM that has been stored in binary  with DMView().

  Collective on viewer

  Input Parameters:
+ newdm - the newly loaded DM, this needs to have been created with DMCreate() or
           some related function before a call to DMLoad().
- viewer - binary file viewer, obtained from PetscViewerBinaryOpen() or
           HDF5 file viewer, obtained from PetscViewerHDF5Open()

   Level: intermediate

  Notes:
   The type is determined by the data in the file, any type set into the DM before this call is ignored.

  Notes for advanced users:
  Most users should not need to know the details of the binary storage
  format, since DMLoad() and DMView() completely hide these details.
  But for anyone who's interested, the standard binary matrix storage
  format is
.vb
     has not yet been determined
.ve

.seealso: PetscViewerBinaryOpen(), DMView(), MatLoad(), VecLoad()
@*/
PetscErrorCode  DMLoad(DM newdm, PetscViewer viewer)
{
  PetscBool      isbinary, ishdf5;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(newdm,DM_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = PetscViewerCheckReadable(viewer);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(DM_Load,viewer,0,0,0);CHKERRQ(ierr);
  if (isbinary) {
    PetscInt classid;
    char     type[256];

    ierr = PetscViewerBinaryRead(viewer,&classid,1,NULL,PETSC_INT);CHKERRQ(ierr);
    if (classid != DM_FILE_CLASSID) SETERRQ1(PetscObjectComm((PetscObject)newdm),PETSC_ERR_ARG_WRONG,"Not DM next in file, classid found %d",(int)classid);
    ierr = PetscViewerBinaryRead(viewer,type,256,NULL,PETSC_CHAR);CHKERRQ(ierr);
    ierr = DMSetType(newdm, type);CHKERRQ(ierr);
    if (newdm->ops->load) {ierr = (*newdm->ops->load)(newdm,viewer);CHKERRQ(ierr);}
  } else if (ishdf5) {
    if (newdm->ops->load) {ierr = (*newdm->ops->load)(newdm,viewer);CHKERRQ(ierr);}
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid viewer; open viewer with PetscViewerBinaryOpen() or PetscViewerHDF5Open()");
  ierr = PetscLogEventEnd(DM_Load,viewer,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMGetLocalBoundingBox - Returns the bounding box for the piece of the DM on this process.

  Not collective

  Input Parameter:
. dm - the DM

  Output Parameters:
+ lmin - local minimum coordinates (length coord dim, optional)
- lmax - local maximim coordinates (length coord dim, optional)

  Level: beginner

  Note: If the DM is a DMDA and has no coordinates, the index bounds are returned instead.


.seealso: DMGetCoordinates(), DMGetCoordinatesLocal(), DMGetBoundingBox()
@*/
PetscErrorCode DMGetLocalBoundingBox(DM dm, PetscReal lmin[], PetscReal lmax[])
{
  Vec                coords = NULL;
  PetscReal          min[3] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL};
  PetscReal          max[3] = {PETSC_MIN_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL};
  const PetscScalar *local_coords;
  PetscInt           N, Ni;
  PetscInt           cdim, i, j;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
  ierr = DMGetCoordinates(dm, &coords);CHKERRQ(ierr);
  if (coords) {
    ierr = VecGetArrayRead(coords, &local_coords);CHKERRQ(ierr);
    ierr = VecGetLocalSize(coords, &N);CHKERRQ(ierr);
    Ni   = N/cdim;
    for (i = 0; i < Ni; ++i) {
      for (j = 0; j < 3; ++j) {
        min[j] = j < cdim ? PetscMin(min[j], PetscRealPart(local_coords[i*cdim+j])) : 0;
        max[j] = j < cdim ? PetscMax(max[j], PetscRealPart(local_coords[i*cdim+j])) : 0;
      }
    }
    ierr = VecRestoreArrayRead(coords, &local_coords);CHKERRQ(ierr);
  } else {
    PetscBool isda;

    ierr = PetscObjectTypeCompare((PetscObject) dm, DMDA, &isda);CHKERRQ(ierr);
    if (isda) {ierr = DMGetLocalBoundingIndices_DMDA(dm, min, max);CHKERRQ(ierr);}
  }
  if (lmin) {ierr = PetscArraycpy(lmin, min, cdim);CHKERRQ(ierr);}
  if (lmax) {ierr = PetscArraycpy(lmax, max, cdim);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  DMGetBoundingBox - Returns the global bounding box for the DM.

  Collective

  Input Parameter:
. dm - the DM

  Output Parameters:
+ gmin - global minimum coordinates (length coord dim, optional)
- gmax - global maximim coordinates (length coord dim, optional)

  Level: beginner

.seealso: DMGetLocalBoundingBox(), DMGetCoordinates(), DMGetCoordinatesLocal()
@*/
PetscErrorCode DMGetBoundingBox(DM dm, PetscReal gmin[], PetscReal gmax[])
{
  PetscReal      lmin[3], lmax[3];
  PetscInt       cdim;
  PetscMPIInt    count;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(cdim, &count);CHKERRQ(ierr);
  ierr = DMGetLocalBoundingBox(dm, lmin, lmax);CHKERRQ(ierr);
  if (gmin) {ierr = MPIU_Allreduce(lmin, gmin, count, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject) dm));CHKERRQ(ierr);}
  if (gmax) {ierr = MPIU_Allreduce(lmax, gmax, count, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject) dm));CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/******************************** FEM Support **********************************/

PetscErrorCode DMPrintCellVector(PetscInt c, const char name[], PetscInt len, const PetscScalar x[])
{
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPrintf(PETSC_COMM_SELF, "Cell %D Element %s\n", c, name);CHKERRQ(ierr);
  for (f = 0; f < len; ++f) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "  | %g |\n", (double)PetscRealPart(x[f]));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPrintCellMatrix(PetscInt c, const char name[], PetscInt rows, PetscInt cols, const PetscScalar A[])
{
  PetscInt       f, g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPrintf(PETSC_COMM_SELF, "Cell %D Element %s\n", c, name);CHKERRQ(ierr);
  for (f = 0; f < rows; ++f) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "  |");CHKERRQ(ierr);
    for (g = 0; g < cols; ++g) {
      ierr = PetscPrintf(PETSC_COMM_SELF, " % 9.5g", PetscRealPart(A[f*cols+g]));CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_SELF, " |\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPrintLocalVec(DM dm, const char name[], PetscReal tol, Vec X)
{
  PetscInt          localSize, bs;
  PetscMPIInt       size;
  Vec               x, xglob;
  const PetscScalar *xarray;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm),&size);CHKERRQ(ierr);
  ierr = VecDuplicate(X, &x);CHKERRQ(ierr);
  ierr = VecCopy(X, x);CHKERRQ(ierr);
  ierr = VecChop(x, tol);CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject) dm),"%s:\n",name);CHKERRQ(ierr);
  if (size > 1) {
    ierr = VecGetLocalSize(x,&localSize);CHKERRQ(ierr);
    ierr = VecGetArrayRead(x,&xarray);CHKERRQ(ierr);
    ierr = VecGetBlockSize(x,&bs);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject) dm),bs,localSize,PETSC_DETERMINE,xarray,&xglob);CHKERRQ(ierr);
  } else {
    xglob = x;
  }
  ierr = VecView(xglob,PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject) dm)));CHKERRQ(ierr);
  if (size > 1) {
    ierr = VecDestroy(&xglob);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(x,&xarray);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMGetSection - Get the PetscSection encoding the local data layout for the DM.   This is equivalent to DMGetLocalSection(). Deprecated in v3.12

  Input Parameter:
. dm - The DM

  Output Parameter:
. section - The PetscSection

  Options Database Keys:
. -dm_petscsection_view - View the Section created by the DM

  Level: advanced

  Notes:
  Use DMGetLocalSection() in new code.

  This gets a borrowed reference, so the user should not destroy this PetscSection.

.seealso: DMGetLocalSection(), DMSetLocalSection(), DMGetGlobalSection()
@*/
PetscErrorCode DMGetSection(DM dm, PetscSection *section)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalSection(dm,section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMGetLocalSection - Get the PetscSection encoding the local data layout for the DM.

  Input Parameter:
. dm - The DM

  Output Parameter:
. section - The PetscSection

  Options Database Keys:
. -dm_petscsection_view - View the Section created by the DM

  Level: intermediate

  Note: This gets a borrowed reference, so the user should not destroy this PetscSection.

.seealso: DMSetLocalSection(), DMGetGlobalSection()
@*/
PetscErrorCode DMGetLocalSection(DM dm, PetscSection *section)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(section, 2);
  if (!dm->localSection && dm->ops->createlocalsection) {
    PetscInt d;

    if (dm->setfromoptionscalled) for (d = 0; d < dm->Nds; ++d) {ierr = PetscDSSetFromOptions(dm->probs[d].ds);CHKERRQ(ierr);}
    ierr = (*dm->ops->createlocalsection)(dm);CHKERRQ(ierr);
    if (dm->localSection) {ierr = PetscObjectViewFromOptions((PetscObject) dm->localSection, NULL, "-dm_petscsection_view");CHKERRQ(ierr);}
  }
  *section = dm->localSection;
  PetscFunctionReturn(0);
}

/*@
  DMSetSection - Set the PetscSection encoding the local data layout for the DM.  This is equivalent to DMSetLocalSection(). Deprecated in v3.12

  Input Parameters:
+ dm - The DM
- section - The PetscSection

  Level: advanced

  Notes:
  Use DMSetLocalSection() in new code.

  Any existing Section will be destroyed

.seealso: DMSetLocalSection(), DMGetLocalSection(), DMSetGlobalSection()
@*/
PetscErrorCode DMSetSection(DM dm, PetscSection section)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMSetLocalSection(dm,section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMSetLocalSection - Set the PetscSection encoding the local data layout for the DM.

  Input Parameters:
+ dm - The DM
- section - The PetscSection

  Level: intermediate

  Note: Any existing Section will be destroyed

.seealso: DMGetLocalSection(), DMSetGlobalSection()
@*/
PetscErrorCode DMSetLocalSection(DM dm, PetscSection section)
{
  PetscInt       numFields = 0;
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (section) PetscValidHeaderSpecific(section,PETSC_SECTION_CLASSID,2);
  ierr = PetscObjectReference((PetscObject)section);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&dm->localSection);CHKERRQ(ierr);
  dm->localSection = section;
  if (section) {ierr = PetscSectionGetNumFields(dm->localSection, &numFields);CHKERRQ(ierr);}
  if (numFields) {
    ierr = DMSetNumFields(dm, numFields);CHKERRQ(ierr);
    for (f = 0; f < numFields; ++f) {
      PetscObject disc;
      const char *name;

      ierr = PetscSectionGetFieldName(dm->localSection, f, &name);CHKERRQ(ierr);
      ierr = DMGetField(dm, f, NULL, &disc);CHKERRQ(ierr);
      ierr = PetscObjectSetName(disc, name);CHKERRQ(ierr);
    }
  }
  /* The global section will be rebuilt in the next call to DMGetGlobalSection(). */
  ierr = PetscSectionDestroy(&dm->globalSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMGetDefaultConstraints - Get the PetscSection and Mat that specify the local constraint interpolation. See DMSetDefaultConstraints() for a description of the purpose of constraint interpolation.

  not collective

  Input Parameter:
. dm - The DM

  Output Parameter:
+ section - The PetscSection describing the range of the constraint matrix: relates rows of the constraint matrix to dofs of the default section.  Returns NULL if there are no local constraints.
- mat - The Mat that interpolates local constraints: its width should be the layout size of the default section.  Returns NULL if there are no local constraints.

  Level: advanced

  Note: This gets borrowed references, so the user should not destroy the PetscSection or the Mat.

.seealso: DMSetDefaultConstraints()
@*/
PetscErrorCode DMGetDefaultConstraints(DM dm, PetscSection *section, Mat *mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!dm->defaultConstraintSection && !dm->defaultConstraintMat && dm->ops->createdefaultconstraints) {ierr = (*dm->ops->createdefaultconstraints)(dm);CHKERRQ(ierr);}
  if (section) {*section = dm->defaultConstraintSection;}
  if (mat) {*mat = dm->defaultConstraintMat;}
  PetscFunctionReturn(0);
}

/*@
  DMSetDefaultConstraints - Set the PetscSection and Mat that specify the local constraint interpolation.

  If a constraint matrix is specified, then it is applied during DMGlobalToLocalEnd() when mode is INSERT_VALUES, INSERT_BC_VALUES, or INSERT_ALL_VALUES.  Without a constraint matrix, the local vector l returned by DMGlobalToLocalEnd() contains values that have been scattered from a global vector without modification; with a constraint matrix A, l is modified by computing c = A * l, l[s[i]] = c[i], where the scatter s is defined by the PetscSection returned by DMGetDefaultConstraintMatrix().

  If a constraint matrix is specified, then its adjoint is applied during DMLocalToGlobalBegin() when mode is ADD_VALUES, ADD_BC_VALUES, or ADD_ALL_VALUES.  Without a constraint matrix, the local vector l is accumulated into a global vector without modification; with a constraint matrix A, l is first modified by computing c[i] = l[s[i]], l[s[i]] = 0, l = l + A'*c, which is the adjoint of the operation described above.

  collective on dm

  Input Parameters:
+ dm - The DM
+ section - The PetscSection describing the range of the constraint matrix: relates rows of the constraint matrix to dofs of the default section.  Must have a local communicator (PETSC_COMM_SELF or derivative).
- mat - The Mat that interpolates local constraints: its width should be the layout size of the default section:  NULL indicates no constraints.  Must have a local communicator (PETSC_COMM_SELF or derivative).

  Level: advanced

  Note: This increments the references of the PetscSection and the Mat, so they user can destroy them

.seealso: DMGetDefaultConstraints()
@*/
PetscErrorCode DMSetDefaultConstraints(DM dm, PetscSection section, Mat mat)
{
  PetscMPIInt result;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (section) {
    PetscValidHeaderSpecific(section,PETSC_SECTION_CLASSID,2);
    ierr = MPI_Comm_compare(PETSC_COMM_SELF,PetscObjectComm((PetscObject)section),&result);CHKERRQ(ierr);
    if (result != MPI_CONGRUENT && result != MPI_IDENT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMECOMM,"constraint section must have local communicator");
  }
  if (mat) {
    PetscValidHeaderSpecific(mat,MAT_CLASSID,3);
    ierr = MPI_Comm_compare(PETSC_COMM_SELF,PetscObjectComm((PetscObject)mat),&result);CHKERRQ(ierr);
    if (result != MPI_CONGRUENT && result != MPI_IDENT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMECOMM,"constraint matrix must have local communicator");
  }
  ierr = PetscObjectReference((PetscObject)section);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&dm->defaultConstraintSection);CHKERRQ(ierr);
  dm->defaultConstraintSection = section;
  ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
  ierr = MatDestroy(&dm->defaultConstraintMat);CHKERRQ(ierr);
  dm->defaultConstraintMat = mat;
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_DEBUG)
/*
  DMDefaultSectionCheckConsistency - Check the consistentcy of the global and local sections.

  Input Parameters:
+ dm - The DM
. localSection - PetscSection describing the local data layout
- globalSection - PetscSection describing the global data layout

  Level: intermediate

.seealso: DMGetSectionSF(), DMSetSectionSF()
*/
static PetscErrorCode DMDefaultSectionCheckConsistency_Internal(DM dm, PetscSection localSection, PetscSection globalSection)
{
  MPI_Comm        comm;
  PetscLayout     layout;
  const PetscInt *ranges;
  PetscInt        pStart, pEnd, p, nroots;
  PetscMPIInt     size, rank;
  PetscBool       valid = PETSC_TRUE, gvalid;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(globalSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(globalSection, &nroots);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(comm, &layout);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(layout, 1);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(layout, nroots);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(layout);CHKERRQ(ierr);
  ierr = PetscLayoutGetRanges(layout, &ranges);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt       dof, cdof, off, gdof, gcdof, goff, gsize, d;

    ierr = PetscSectionGetDof(localSection, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(localSection, p, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(localSection, p, &cdof);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(globalSection, p, &gdof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(globalSection, p, &gcdof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(globalSection, p, &goff);CHKERRQ(ierr);
    if (!gdof) continue; /* Censored point */
    if ((gdof < 0 ? -(gdof+1) : gdof) != dof) {ierr = PetscSynchronizedPrintf(comm, "[%d]Global dof %d for point %d not equal to local dof %d\n", rank, gdof, p, dof);CHKERRQ(ierr); valid = PETSC_FALSE;}
    if (gcdof && (gcdof != cdof)) {ierr = PetscSynchronizedPrintf(comm, "[%d]Global constraints %d for point %d not equal to local constraints %d\n", rank, gcdof, p, cdof);CHKERRQ(ierr); valid = PETSC_FALSE;}
    if (gdof < 0) {
      gsize = gdof < 0 ? -(gdof+1)-gcdof : gdof-gcdof;
      for (d = 0; d < gsize; ++d) {
        PetscInt offset = -(goff+1) + d, r;

        ierr = PetscFindInt(offset,size+1,ranges,&r);CHKERRQ(ierr);
        if (r < 0) r = -(r+2);
        if ((r < 0) || (r >= size)) {ierr = PetscSynchronizedPrintf(comm, "[%d]Point %d mapped to invalid process %d (%d, %d)\n", rank, p, r, gdof, goff);CHKERRQ(ierr); valid = PETSC_FALSE;break;}
      }
    }
  }
  ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(comm, NULL);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&valid, &gvalid, 1, MPIU_BOOL, MPI_LAND, comm);CHKERRQ(ierr);
  if (!gvalid) {
    ierr = DMView(dm, NULL);CHKERRQ(ierr);
    SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Inconsistent local and global sections");
  }
  PetscFunctionReturn(0);
}
#endif

/*@
  DMGetGlobalSection - Get the PetscSection encoding the global data layout for the DM.

  Collective on dm

  Input Parameter:
. dm - The DM

  Output Parameter:
. section - The PetscSection

  Level: intermediate

  Note: This gets a borrowed reference, so the user should not destroy this PetscSection.

.seealso: DMSetLocalSection(), DMGetLocalSection()
@*/
PetscErrorCode DMGetGlobalSection(DM dm, PetscSection *section)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(section, 2);
  if (!dm->globalSection) {
    PetscSection s;

    ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
    if (!s)  SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONGSTATE, "DM must have a default PetscSection in order to create a global PetscSection");
    if (!dm->sf) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM must have a point PetscSF in order to create a global PetscSection");
    ierr = PetscSectionCreateGlobalSection(s, dm->sf, PETSC_FALSE, PETSC_FALSE, &dm->globalSection);CHKERRQ(ierr);
    ierr = PetscLayoutDestroy(&dm->map);CHKERRQ(ierr);
    ierr = PetscSectionGetValueLayout(PetscObjectComm((PetscObject)dm), dm->globalSection, &dm->map);CHKERRQ(ierr);
    ierr = PetscSectionViewFromOptions(dm->globalSection, NULL, "-global_section_view");CHKERRQ(ierr);
  }
  *section = dm->globalSection;
  PetscFunctionReturn(0);
}

/*@
  DMSetGlobalSection - Set the PetscSection encoding the global data layout for the DM.

  Input Parameters:
+ dm - The DM
- section - The PetscSection, or NULL

  Level: intermediate

  Note: Any existing Section will be destroyed

.seealso: DMGetGlobalSection(), DMSetLocalSection()
@*/
PetscErrorCode DMSetGlobalSection(DM dm, PetscSection section)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (section) PetscValidHeaderSpecific(section,PETSC_SECTION_CLASSID,2);
  ierr = PetscObjectReference((PetscObject)section);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&dm->globalSection);CHKERRQ(ierr);
  dm->globalSection = section;
#if defined(PETSC_USE_DEBUG)
  if (section) {ierr = DMDefaultSectionCheckConsistency_Internal(dm, dm->localSection, section);CHKERRQ(ierr);}
#endif
  PetscFunctionReturn(0);
}

/*@
  DMGetSectionSF - Get the PetscSF encoding the parallel dof overlap for the DM. If it has not been set,
  it is created from the default PetscSection layouts in the DM.

  Input Parameter:
. dm - The DM

  Output Parameter:
. sf - The PetscSF

  Level: intermediate

  Note: This gets a borrowed reference, so the user should not destroy this PetscSF.

.seealso: DMSetSectionSF(), DMCreateSectionSF()
@*/
PetscErrorCode DMGetSectionSF(DM dm, PetscSF *sf)
{
  PetscInt       nroots;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(sf, 2);
  if (!dm->sectionSF) {
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)dm),&dm->sectionSF);CHKERRQ(ierr);
  }
  ierr = PetscSFGetGraph(dm->sectionSF, &nroots, NULL, NULL, NULL);CHKERRQ(ierr);
  if (nroots < 0) {
    PetscSection section, gSection;

    ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
    if (section) {
      ierr = DMGetGlobalSection(dm, &gSection);CHKERRQ(ierr);
      ierr = DMCreateSectionSF(dm, section, gSection);CHKERRQ(ierr);
    } else {
      *sf = NULL;
      PetscFunctionReturn(0);
    }
  }
  *sf = dm->sectionSF;
  PetscFunctionReturn(0);
}

/*@
  DMSetSectionSF - Set the PetscSF encoding the parallel dof overlap for the DM

  Input Parameters:
+ dm - The DM
- sf - The PetscSF

  Level: intermediate

  Note: Any previous SF is destroyed

.seealso: DMGetSectionSF(), DMCreateSectionSF()
@*/
PetscErrorCode DMSetSectionSF(DM dm, PetscSF sf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (sf) PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 2);
  ierr = PetscObjectReference((PetscObject) sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&dm->sectionSF);CHKERRQ(ierr);
  dm->sectionSF = sf;
  PetscFunctionReturn(0);
}

/*@C
  DMCreateSectionSF - Create the PetscSF encoding the parallel dof overlap for the DM based upon the PetscSections
  describing the data layout.

  Input Parameters:
+ dm - The DM
. localSection - PetscSection describing the local data layout
- globalSection - PetscSection describing the global data layout

  Notes: One usually uses DMGetSectionSF() to obtain the PetscSF

  Level: developer

  Developer Note: Since this routine has for arguments the two sections from the DM and puts the resulting PetscSF
                  directly into the DM, perhaps this function should not take the local and global sections as
                  input and should just obtain them from the DM?

.seealso: DMGetSectionSF(), DMSetSectionSF(), DMGetLocalSection(), DMGetGlobalSection()
@*/
PetscErrorCode DMCreateSectionSF(DM dm, PetscSection localSection, PetscSection globalSection)
{
  MPI_Comm       comm;
  PetscLayout    layout;
  const PetscInt *ranges;
  PetscInt       *local;
  PetscSFNode    *remote;
  PetscInt       pStart, pEnd, p, nroots, nleaves = 0, l;
  PetscMPIInt    size, rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(globalSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(globalSection, &nroots);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(comm, &layout);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(layout, 1);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(layout, nroots);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(layout);CHKERRQ(ierr);
  ierr = PetscLayoutGetRanges(layout, &ranges);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt gdof, gcdof;

    ierr = PetscSectionGetDof(globalSection, p, &gdof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(globalSection, p, &gcdof);CHKERRQ(ierr);
    if (gcdof > (gdof < 0 ? -(gdof+1) : gdof)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point %d has %d constraints > %d dof", p, gcdof, (gdof < 0 ? -(gdof+1) : gdof));
    nleaves += gdof < 0 ? -(gdof+1)-gcdof : gdof-gcdof;
  }
  ierr = PetscMalloc1(nleaves, &local);CHKERRQ(ierr);
  ierr = PetscMalloc1(nleaves, &remote);CHKERRQ(ierr);
  for (p = pStart, l = 0; p < pEnd; ++p) {
    const PetscInt *cind;
    PetscInt       dof, cdof, off, gdof, gcdof, goff, gsize, d, c;

    ierr = PetscSectionGetDof(localSection, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(localSection, p, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(localSection, p, &cdof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintIndices(localSection, p, &cind);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(globalSection, p, &gdof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(globalSection, p, &gcdof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(globalSection, p, &goff);CHKERRQ(ierr);
    if (!gdof) continue; /* Censored point */
    gsize = gdof < 0 ? -(gdof+1)-gcdof : gdof-gcdof;
    if (gsize != dof-cdof) {
      if (gsize != dof) SETERRQ4(comm, PETSC_ERR_ARG_WRONG, "Global dof %d for point %d is neither the constrained size %d, nor the unconstrained %d", gsize, p, dof-cdof, dof);
      cdof = 0; /* Ignore constraints */
    }
    for (d = 0, c = 0; d < dof; ++d) {
      if ((c < cdof) && (cind[c] == d)) {++c; continue;}
      local[l+d-c] = off+d;
    }
    if (gdof < 0) {
      for (d = 0; d < gsize; ++d, ++l) {
        PetscInt offset = -(goff+1) + d, r;

        ierr = PetscFindInt(offset,size+1,ranges,&r);CHKERRQ(ierr);
        if (r < 0) r = -(r+2);
        if ((r < 0) || (r >= size)) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point %d mapped to invalid process %d (%d, %d)", p, r, gdof, goff);
        remote[l].rank  = r;
        remote[l].index = offset - ranges[r];
      }
    } else {
      for (d = 0; d < gsize; ++d, ++l) {
        remote[l].rank  = rank;
        remote[l].index = goff+d - ranges[rank];
      }
    }
  }
  if (l != nleaves) SETERRQ2(comm, PETSC_ERR_PLIB, "Iteration error, l %d != nleaves %d", l, nleaves);
  ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(dm->sectionSF, nroots, nleaves, local, PETSC_OWN_POINTER, remote, PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMGetPointSF - Get the PetscSF encoding the parallel section point overlap for the DM.

  Input Parameter:
. dm - The DM

  Output Parameter:
. sf - The PetscSF

  Level: intermediate

  Note: This gets a borrowed reference, so the user should not destroy this PetscSF.

.seealso: DMSetPointSF(), DMGetSectionSF(), DMSetSectionSF(), DMCreateSectionSF()
@*/
PetscErrorCode DMGetPointSF(DM dm, PetscSF *sf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(sf, 2);
  *sf = dm->sf;
  PetscFunctionReturn(0);
}

/*@
  DMSetPointSF - Set the PetscSF encoding the parallel section point overlap for the DM.

  Input Parameters:
+ dm - The DM
- sf - The PetscSF

  Level: intermediate

.seealso: DMGetPointSF(), DMGetSectionSF(), DMSetSectionSF(), DMCreateSectionSF()
@*/
PetscErrorCode DMSetPointSF(DM dm, PetscSF sf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (sf) PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 2);
  ierr = PetscObjectReference((PetscObject) sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&dm->sf);CHKERRQ(ierr);
  dm->sf = sf;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetDefaultAdjacency_Private(DM dm, PetscInt f, PetscObject disc)
{
  PetscClassId   id;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetClassId(disc, &id);CHKERRQ(ierr);
  if (id == PETSCFE_CLASSID) {
    ierr = DMSetAdjacency(dm, f, PETSC_FALSE, PETSC_TRUE);CHKERRQ(ierr);
  } else if (id == PETSCFV_CLASSID) {
    ierr = DMSetAdjacency(dm, f, PETSC_TRUE, PETSC_FALSE);CHKERRQ(ierr);
  } else {
    ierr = DMSetAdjacency(dm, f, PETSC_FALSE, PETSC_TRUE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldEnlarge_Static(DM dm, PetscInt NfNew)
{
  RegionField   *tmpr;
  PetscInt       Nf = dm->Nf, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (Nf >= NfNew) PetscFunctionReturn(0);
  ierr = PetscMalloc1(NfNew, &tmpr);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) tmpr[f] = dm->fields[f];
  for (f = Nf; f < NfNew; ++f) {tmpr[f].disc = NULL; tmpr[f].label = NULL;}
  ierr = PetscFree(dm->fields);CHKERRQ(ierr);
  dm->Nf     = NfNew;
  dm->fields = tmpr;
  PetscFunctionReturn(0);
}

/*@
  DMClearFields - Remove all fields from the DM

  Logically collective on dm

  Input Parameter:
. dm - The DM

  Level: intermediate

.seealso: DMGetNumFields(), DMSetNumFields(), DMSetField()
@*/
PetscErrorCode DMClearFields(DM dm)
{
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  for (f = 0; f < dm->Nf; ++f) {
    ierr = PetscObjectDestroy(&dm->fields[f].disc);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&dm->fields[f].label);CHKERRQ(ierr);
  }
  ierr = PetscFree(dm->fields);CHKERRQ(ierr);
  dm->fields = NULL;
  dm->Nf     = 0;
  PetscFunctionReturn(0);
}

/*@
  DMGetNumFields - Get the number of fields in the DM

  Not collective

  Input Parameter:
. dm - The DM

  Output Parameter:
. Nf - The number of fields

  Level: intermediate

.seealso: DMSetNumFields(), DMSetField()
@*/
PetscErrorCode DMGetNumFields(DM dm, PetscInt *numFields)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(numFields, 2);
  *numFields = dm->Nf;
  PetscFunctionReturn(0);
}

/*@
  DMSetNumFields - Set the number of fields in the DM

  Logically collective on dm

  Input Parameters:
+ dm - The DM
- Nf - The number of fields

  Level: intermediate

.seealso: DMGetNumFields(), DMSetField()
@*/
PetscErrorCode DMSetNumFields(DM dm, PetscInt numFields)
{
  PetscInt       Nf, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  for (f = Nf; f < numFields; ++f) {
    PetscContainer obj;

    ierr = PetscContainerCreate(PetscObjectComm((PetscObject) dm), &obj);CHKERRQ(ierr);
    ierr = DMAddField(dm, NULL, (PetscObject) obj);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&obj);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  DMGetField - Return the discretization object for a given DM field

  Not collective

  Input Parameters:
+ dm - The DM
- f  - The field number

  Output Parameters:
+ label - The label indicating the support of the field, or NULL for the entire mesh
- field - The discretization object

  Level: intermediate

.seealso: DMAddField(), DMSetField()
@*/
PetscErrorCode DMGetField(DM dm, PetscInt f, DMLabel *label, PetscObject *field)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(field, 3);
  if ((f < 0) || (f >= dm->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, dm->Nf);
  if (label) *label = dm->fields[f].label;
  if (field) *field = dm->fields[f].disc;
  PetscFunctionReturn(0);
}

/*@
  DMSetField - Set the discretization object for a given DM field

  Logically collective on dm

  Input Parameters:
+ dm    - The DM
. f     - The field number
. label - The label indicating the support of the field, or NULL for the entire mesh
- field - The discretization object

  Level: intermediate

.seealso: DMAddField(), DMGetField()
@*/
PetscErrorCode DMSetField(DM dm, PetscInt f, DMLabel label, PetscObject field)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (label) PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 3);
  PetscValidHeader(field, 4);
  if (f < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  ierr = DMFieldEnlarge_Static(dm, f+1);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&dm->fields[f].label);CHKERRQ(ierr);
  ierr = PetscObjectDestroy(&dm->fields[f].disc);CHKERRQ(ierr);
  dm->fields[f].label = label;
  dm->fields[f].disc  = field;
  ierr = PetscObjectReference((PetscObject) label);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) field);CHKERRQ(ierr);
  ierr = DMSetDefaultAdjacency_Private(dm, f, field);CHKERRQ(ierr);
  ierr = DMClearDS(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMAddField - Add the discretization object for the given DM field

  Logically collective on dm

  Input Parameters:
+ dm    - The DM
. label - The label indicating the support of the field, or NULL for the entire mesh
- field - The discretization object

  Level: intermediate

.seealso: DMSetField(), DMGetField()
@*/
PetscErrorCode DMAddField(DM dm, DMLabel label, PetscObject field)
{
  PetscInt       Nf = dm->Nf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (label) PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 3);
  PetscValidHeader(field, 3);
  ierr = DMFieldEnlarge_Static(dm, Nf+1);CHKERRQ(ierr);
  dm->fields[Nf].label = label;
  dm->fields[Nf].disc  = field;
  ierr = PetscObjectReference((PetscObject) label);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) field);CHKERRQ(ierr);
  ierr = DMSetDefaultAdjacency_Private(dm, Nf, field);CHKERRQ(ierr);
  ierr = DMClearDS(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMCopyFields - Copy the discretizations for the DM into another DM

  Collective on dm

  Input Parameter:
. dm - The DM

  Output Parameter:
. newdm - The DM

  Level: advanced

.seealso: DMGetField(), DMSetField(), DMAddField(), DMCopyDS(), DMGetDS(), DMGetCellDS()
@*/
PetscErrorCode DMCopyFields(DM dm, DM newdm)
{
  PetscInt       Nf, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dm == newdm) PetscFunctionReturn(0);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  ierr = DMClearFields(newdm);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    DMLabel     label;
    PetscObject field;
    PetscBool   useCone, useClosure;

    ierr = DMGetField(dm, f, &label, &field);CHKERRQ(ierr);
    ierr = DMSetField(newdm, f, label, field);CHKERRQ(ierr);
    ierr = DMGetAdjacency(dm, f, &useCone, &useClosure);CHKERRQ(ierr);
    ierr = DMSetAdjacency(newdm, f, useCone, useClosure);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  DMGetAdjacency - Returns the flags for determining variable influence

  Not collective

  Input Parameters:
+ dm - The DM object
- f  - The field number, or PETSC_DEFAULT for the default adjacency

  Output Parameter:
+ useCone    - Flag for variable influence starting with the cone operation
- useClosure - Flag for variable influence using transitive closure

  Notes:
$     FEM:   Two points p and q are adjacent if q \in closure(star(p)),   useCone = PETSC_FALSE, useClosure = PETSC_TRUE
$     FVM:   Two points p and q are adjacent if q \in support(p+cone(p)), useCone = PETSC_TRUE,  useClosure = PETSC_FALSE
$     FVM++: Two points p and q are adjacent if q \in star(closure(p)),   useCone = PETSC_TRUE,  useClosure = PETSC_TRUE
  Further explanation can be found in the User's Manual Section on the Influence of Variables on One Another.

  Level: developer

.seealso: DMSetAdjacency(), DMGetField(), DMSetField()
@*/
PetscErrorCode DMGetAdjacency(DM dm, PetscInt f, PetscBool *useCone, PetscBool *useClosure)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (useCone)    PetscValidBoolPointer(useCone, 3);
  if (useClosure) PetscValidBoolPointer(useClosure, 4);
  if (f < 0) {
    if (useCone)    *useCone    = dm->adjacency[0];
    if (useClosure) *useClosure = dm->adjacency[1];
  } else {
    PetscInt       Nf;
    PetscErrorCode ierr;

    ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
    if (f >= Nf) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, Nf);
    if (useCone)    *useCone    = dm->fields[f].adjacency[0];
    if (useClosure) *useClosure = dm->fields[f].adjacency[1];
  }
  PetscFunctionReturn(0);
}

/*@
  DMSetAdjacency - Set the flags for determining variable influence

  Not collective

  Input Parameters:
+ dm         - The DM object
. f          - The field number
. useCone    - Flag for variable influence starting with the cone operation
- useClosure - Flag for variable influence using transitive closure

  Notes:
$     FEM:   Two points p and q are adjacent if q \in closure(star(p)),   useCone = PETSC_FALSE, useClosure = PETSC_TRUE
$     FVM:   Two points p and q are adjacent if q \in support(p+cone(p)), useCone = PETSC_TRUE,  useClosure = PETSC_FALSE
$     FVM++: Two points p and q are adjacent if q \in star(closure(p)),   useCone = PETSC_TRUE,  useClosure = PETSC_TRUE
  Further explanation can be found in the User's Manual Section on the Influence of Variables on One Another.

  Level: developer

.seealso: DMGetAdjacency(), DMGetField(), DMSetField()
@*/
PetscErrorCode DMSetAdjacency(DM dm, PetscInt f, PetscBool useCone, PetscBool useClosure)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (f < 0) {
    dm->adjacency[0] = useCone;
    dm->adjacency[1] = useClosure;
  } else {
    PetscInt       Nf;
    PetscErrorCode ierr;

    ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
    if (f >= Nf) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, Nf);
    dm->fields[f].adjacency[0] = useCone;
    dm->fields[f].adjacency[1] = useClosure;
  }
  PetscFunctionReturn(0);
}

/*@
  DMGetBasicAdjacency - Returns the flags for determining variable influence, using either the default or field 0 if it is defined

  Not collective

  Input Parameters:
. dm - The DM object

  Output Parameter:
+ useCone    - Flag for variable influence starting with the cone operation
- useClosure - Flag for variable influence using transitive closure

  Notes:
$     FEM:   Two points p and q are adjacent if q \in closure(star(p)),   useCone = PETSC_FALSE, useClosure = PETSC_TRUE
$     FVM:   Two points p and q are adjacent if q \in support(p+cone(p)), useCone = PETSC_TRUE,  useClosure = PETSC_FALSE
$     FVM++: Two points p and q are adjacent if q \in star(closure(p)),   useCone = PETSC_TRUE,  useClosure = PETSC_TRUE

  Level: developer

.seealso: DMSetBasicAdjacency(), DMGetField(), DMSetField()
@*/
PetscErrorCode DMGetBasicAdjacency(DM dm, PetscBool *useCone, PetscBool *useClosure)
{
  PetscInt       Nf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (useCone)    PetscValidBoolPointer(useCone, 3);
  if (useClosure) PetscValidBoolPointer(useClosure, 4);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  if (!Nf) {
    ierr = DMGetAdjacency(dm, PETSC_DEFAULT, useCone, useClosure);CHKERRQ(ierr);
  } else {
    ierr = DMGetAdjacency(dm, 0, useCone, useClosure);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  DMSetBasicAdjacency - Set the flags for determining variable influence, using either the default or field 0 if it is defined

  Not collective

  Input Parameters:
+ dm         - The DM object
. useCone    - Flag for variable influence starting with the cone operation
- useClosure - Flag for variable influence using transitive closure

  Notes:
$     FEM:   Two points p and q are adjacent if q \in closure(star(p)),   useCone = PETSC_FALSE, useClosure = PETSC_TRUE
$     FVM:   Two points p and q are adjacent if q \in support(p+cone(p)), useCone = PETSC_TRUE,  useClosure = PETSC_FALSE
$     FVM++: Two points p and q are adjacent if q \in star(closure(p)),   useCone = PETSC_TRUE,  useClosure = PETSC_TRUE

  Level: developer

.seealso: DMGetBasicAdjacency(), DMGetField(), DMSetField()
@*/
PetscErrorCode DMSetBasicAdjacency(DM dm, PetscBool useCone, PetscBool useClosure)
{
  PetscInt       Nf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  if (!Nf) {
    ierr = DMSetAdjacency(dm, PETSC_DEFAULT, useCone, useClosure);CHKERRQ(ierr);
  } else {
    ierr = DMSetAdjacency(dm, 0, useCone, useClosure);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDSEnlarge_Static(DM dm, PetscInt NdsNew)
{
  DMSpace       *tmpd;
  PetscInt       Nds = dm->Nds, s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (Nds >= NdsNew) PetscFunctionReturn(0);
  ierr = PetscMalloc1(NdsNew, &tmpd);CHKERRQ(ierr);
  for (s = 0; s < Nds; ++s) tmpd[s] = dm->probs[s];
  for (s = Nds; s < NdsNew; ++s) {tmpd[s].ds = NULL; tmpd[s].label = NULL; tmpd[s].fields = NULL;}
  ierr = PetscFree(dm->probs);CHKERRQ(ierr);
  dm->Nds   = NdsNew;
  dm->probs = tmpd;
  PetscFunctionReturn(0);
}

/*@
  DMGetNumDS - Get the number of discrete systems in the DM

  Not collective

  Input Parameter:
. dm - The DM

  Output Parameter:
. Nds - The number of PetscDS objects

  Level: intermediate

.seealso: DMGetDS(), DMGetCellDS()
@*/
PetscErrorCode DMGetNumDS(DM dm, PetscInt *Nds)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(Nds, 2);
  *Nds = dm->Nds;
  PetscFunctionReturn(0);
}

/*@
  DMClearDS - Remove all discrete systems from the DM

  Logically collective on dm

  Input Parameter:
. dm - The DM

  Level: intermediate

.seealso: DMGetNumDS(), DMGetDS(), DMSetField()
@*/
PetscErrorCode DMClearDS(DM dm)
{
  PetscInt       s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  for (s = 0; s < dm->Nds; ++s) {
    ierr = PetscDSDestroy(&dm->probs[s].ds);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&dm->probs[s].label);CHKERRQ(ierr);
    ierr = ISDestroy(&dm->probs[s].fields);CHKERRQ(ierr);
  }
  ierr = PetscFree(dm->probs);CHKERRQ(ierr);
  dm->probs = NULL;
  dm->Nds   = 0;
  PetscFunctionReturn(0);
}

/*@
  DMGetDS - Get the default PetscDS

  Not collective

  Input Parameter:
. dm    - The DM

  Output Parameter:
. prob - The default PetscDS

  Level: intermediate

.seealso: DMGetCellDS(), DMGetRegionDS()
@*/
PetscErrorCode DMGetDS(DM dm, PetscDS *prob)
{
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(prob, 2);
  if (dm->Nds <= 0) {
    PetscDS ds;

    ierr = PetscDSCreate(PetscObjectComm((PetscObject) dm), &ds);CHKERRQ(ierr);
    ierr = DMSetRegionDS(dm, NULL, NULL, ds);CHKERRQ(ierr);
    ierr = PetscDSDestroy(&ds);CHKERRQ(ierr);
  }
  *prob = dm->probs[0].ds;
  PetscFunctionReturn(0);
}

/*@
  DMGetCellDS - Get the PetscDS defined on a given cell

  Not collective

  Input Parameters:
+ dm    - The DM
- point - Cell for the DS

  Output Parameter:
. prob - The PetscDS defined on the given cell

  Level: developer

.seealso: DMGetDS(), DMSetRegionDS()
@*/
PetscErrorCode DMGetCellDS(DM dm, PetscInt point, PetscDS *prob)
{
  PetscDS        probDef = NULL;
  PetscInt       s;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(prob, 3);
  *prob = NULL;
  for (s = 0; s < dm->Nds; ++s) {
    PetscInt val;

    if (!dm->probs[s].label) {probDef = dm->probs[s].ds;}
    else {
      ierr = DMLabelGetValue(dm->probs[s].label, point, &val);CHKERRQ(ierr);
      if (val >= 0) {*prob = dm->probs[s].ds; break;}
    }
  }
  if (!*prob) *prob = probDef;
  PetscFunctionReturn(0);
}

/*@
  DMGetRegionDS - Get the PetscDS for a given mesh region, defined by a DMLabel

  Not collective

  Input Parameters:
+ dm    - The DM
- label - The DMLabel defining the mesh region, or NULL for the entire mesh

  Output Parameters:
+ fields - The IS containing the DM field numbers for the fields in this DS, or NULL
- prob - The PetscDS defined on the given region, or NULL

  Note: If the label is missing, this function returns an error

  Level: advanced

.seealso: DMGetRegionNumDS(), DMSetRegionDS(), DMGetDS(), DMGetCellDS()
@*/
PetscErrorCode DMGetRegionDS(DM dm, DMLabel label, IS *fields, PetscDS *ds)
{
  PetscInt Nds = dm->Nds, s;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (label) PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 2);
  if (fields) {PetscValidPointer(fields, 3); *fields = NULL;}
  if (ds)     {PetscValidPointer(ds, 4);     *ds     = NULL;}
  for (s = 0; s < Nds; ++s) {
    if (dm->probs[s].label == label) {
      if (fields) *fields = dm->probs[s].fields;
      if (ds)     *ds     = dm->probs[s].ds;
      PetscFunctionReturn(0);
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMGetRegionNumDS - Get the PetscDS for a given mesh region, defined by the region number

  Not collective

  Input Parameters:
+ dm  - The DM
- num - The region number, in [0, Nds)

  Output Parameters:
+ label  - The region label, or NULL
. fields - The IS containing the DM field numbers for the fields in this DS, or NULL
- prob   - The PetscDS defined on the given region, or NULL

  Level: advanced

.seealso: DMGetRegionDS(), DMSetRegionDS(), DMGetDS(), DMGetCellDS()
@*/
PetscErrorCode DMGetRegionNumDS(DM dm, PetscInt num, DMLabel *label, IS *fields, PetscDS *ds)
{
  PetscInt       Nds;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetNumDS(dm, &Nds);CHKERRQ(ierr);
  if ((num < 0) || (num >= Nds)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Region number %D is not in [0, %D)", num, Nds);
  if (label) {
    PetscValidPointer(label, 3);
    *label = dm->probs[num].label;
  }
  if (fields) {
    PetscValidPointer(fields, 4);
    *fields = dm->probs[num].fields;
  }
  if (ds) {
    PetscValidPointer(ds, 5);
    *ds = dm->probs[num].ds;
  }
  PetscFunctionReturn(0);
}

/*@
  DMSetRegionDS - Set the PetscDS for a given mesh region, defined by a DMLabel

  Collective on dm

  Input Parameters:
+ dm     - The DM
. label  - The DMLabel defining the mesh region, or NULL for the entire mesh
. fields - The IS containing the DM field numbers for the fields in this DS, or NULL for all fields
- prob   - The PetscDS defined on the given cell

  Note: If the label has a DS defined, it will be replaced. Otherwise, it will be added to the DM. If DS is replaced,
  the fields argument is ignored.

  Level: advanced

.seealso: DMGetRegionDS(), DMGetDS(), DMGetCellDS()
@*/
PetscErrorCode DMSetRegionDS(DM dm, DMLabel label, IS fields, PetscDS ds)
{
  PetscInt       Nds = dm->Nds, s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (label) PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 2);
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 3);
  for (s = 0; s < Nds; ++s) {
    if (dm->probs[s].label == label) {
      ierr = PetscDSDestroy(&dm->probs[s].ds);CHKERRQ(ierr);
      dm->probs[s].ds = ds;
      PetscFunctionReturn(0);
    }
  }
  ierr = DMDSEnlarge_Static(dm, Nds+1);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) label);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) fields);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) ds);CHKERRQ(ierr);
  if (!label) {
    /* Put the NULL label at the front, so it is returned as the default */
    for (s = Nds-1; s >=0; --s) dm->probs[s+1] = dm->probs[s];
    Nds = 0;
  }
  dm->probs[Nds].label  = label;
  dm->probs[Nds].fields = fields;
  dm->probs[Nds].ds     = ds;
  PetscFunctionReturn(0);
}

/*@
  DMCreateDS - Create the discrete systems for the DM based upon the fields added to the DM

  Collective on dm

  Input Parameter:
. dm - The DM

  Note: If the label has a DS defined, it will be replaced. Otherwise, it will be added to the DM.

  Level: intermediate

.seealso: DMSetField, DMAddField(), DMGetDS(), DMGetCellDS(), DMGetRegionDS(), DMSetRegionDS()
@*/
PetscErrorCode DMCreateDS(DM dm)
{
  MPI_Comm       comm;
  PetscDS        prob, probh = NULL;
  PetscInt       dimEmbed, Nf = dm->Nf, f, s, field = 0, fieldh = 0;
  PetscBool      doSetup = PETSC_TRUE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!dm->fields) PetscFunctionReturn(0);
  /* Can only handle two label cases right now:
   1) NULL
   2) Hybrid cells
  */
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dimEmbed);CHKERRQ(ierr);
  /* Create default DS */
  ierr = DMGetRegionDS(dm, NULL, NULL, &prob);CHKERRQ(ierr);
  if (!prob) {
    IS        fields;
    PetscInt *fld, nf;

    for (f = 0, nf = 0; f < Nf; ++f) if (!dm->fields[f].label) ++nf;
    ierr = PetscMalloc1(nf, &fld);CHKERRQ(ierr);
    for (f = 0, nf = 0; f < Nf; ++f) if (!dm->fields[f].label) fld[nf++] = f;
    ierr = ISCreate(PETSC_COMM_SELF, &fields);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) fields, "dm_fields_");CHKERRQ(ierr);
    ierr = ISSetType(fields, ISGENERAL);CHKERRQ(ierr);
    ierr = ISGeneralSetIndices(fields, nf, fld, PETSC_OWN_POINTER);CHKERRQ(ierr);

    ierr = PetscDSCreate(comm, &prob);CHKERRQ(ierr);
    ierr = DMSetRegionDS(dm, NULL, fields, prob);CHKERRQ(ierr);
    ierr = PetscDSDestroy(&prob);CHKERRQ(ierr);
    ierr = ISDestroy(&fields);CHKERRQ(ierr);
    ierr = DMGetRegionDS(dm, NULL, NULL, &prob);CHKERRQ(ierr);
  }
  ierr = PetscDSSetCoordinateDimension(prob, dimEmbed);CHKERRQ(ierr);
  /* Optionally create hybrid DS */
  for (f = 0; f < Nf; ++f) {
    DMLabel  label = dm->fields[f].label;
    PetscInt lStart, lEnd;

    if (label) {
      DM        plex;
      IS        fields;
      PetscInt *fld;
      PetscInt  depth, pMax[4];

      ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
      ierr = DMPlexGetDepth(plex, &depth);CHKERRQ(ierr);
      ierr = DMPlexGetHybridBounds(plex, depth >= 0 ? &pMax[depth] : NULL, depth>1 ? &pMax[depth-1] : NULL, depth>2 ? &pMax[1] : NULL, &pMax[0]);CHKERRQ(ierr);
      ierr = DMDestroy(&plex);CHKERRQ(ierr);

      ierr = DMLabelGetBounds(label, &lStart, &lEnd);CHKERRQ(ierr);
      if (lStart < pMax[depth]) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only support labels over hybrid cells right now");
      ierr = PetscDSCreate(comm, &probh);CHKERRQ(ierr);
      ierr = PetscMalloc1(1, &fld);CHKERRQ(ierr);
      fld[0] = f;
      ierr = ISCreate(PETSC_COMM_SELF, &fields);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject) fields, "dm_fields_");CHKERRQ(ierr);
      ierr = ISSetType(fields, ISGENERAL);CHKERRQ(ierr);
      ierr = ISGeneralSetIndices(fields, 1, fld, PETSC_OWN_POINTER);CHKERRQ(ierr);
      ierr = DMSetRegionDS(dm, label, fields, probh);CHKERRQ(ierr);
      ierr = ISDestroy(&fields);CHKERRQ(ierr);
      ierr = PetscDSSetHybrid(probh, PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscDSSetCoordinateDimension(probh, dimEmbed);CHKERRQ(ierr);
      break;
    }
  }
  /* Set fields in DSes */
  for (f = 0; f < Nf; ++f) {
    DMLabel     label = dm->fields[f].label;
    PetscObject disc  = dm->fields[f].disc;

    if (!label) {
      ierr = PetscDSSetDiscretization(prob,  field++,  disc);CHKERRQ(ierr);
      if (probh) {
        PetscFE subfe;

        ierr = PetscFEGetHeightSubspace((PetscFE) disc, 1, &subfe);CHKERRQ(ierr);
        ierr = PetscDSSetDiscretization(probh, fieldh++, (PetscObject) subfe);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscDSSetDiscretization(probh, fieldh++, disc);CHKERRQ(ierr);
    }
    /* We allow people to have placeholder fields and construct the Section by hand */
    {
      PetscClassId id;

      ierr = PetscObjectGetClassId(disc, &id);CHKERRQ(ierr);
      if ((id != PETSCFE_CLASSID) && (id != PETSCFV_CLASSID)) doSetup = PETSC_FALSE;
    }
  }
  ierr = PetscDSDestroy(&probh);CHKERRQ(ierr);
  /* Setup DSes */
  if (doSetup) {
    for (s = 0; s < dm->Nds; ++s) {ierr = PetscDSSetUp(dm->probs[s].ds);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

/*@
  DMCopyDS - Copy the discrete systems for the DM into another DM

  Collective on dm

  Input Parameter:
. dm - The DM

  Output Parameter:
. newdm - The DM

  Level: advanced

.seealso: DMCopyFields(), DMAddField(), DMGetDS(), DMGetCellDS(), DMGetRegionDS(), DMSetRegionDS()
@*/
PetscErrorCode DMCopyDS(DM dm, DM newdm)
{
  PetscInt       Nds, s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dm == newdm) PetscFunctionReturn(0);
  ierr = DMGetNumDS(dm, &Nds);CHKERRQ(ierr);
  ierr = DMClearDS(newdm);CHKERRQ(ierr);
  for (s = 0; s < Nds; ++s) {
    DMLabel label;
    IS      fields;
    PetscDS ds;

    ierr = DMGetRegionNumDS(dm, s, &label, &fields, &ds);CHKERRQ(ierr);
    ierr = DMSetRegionDS(newdm, label, fields, ds);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  DMCopyDisc - Copy the fields and discrete systems for the DM into another DM

  Collective on dm

  Input Parameter:
. dm - The DM

  Output Parameter:
. newdm - The DM

  Level: advanced

.seealso: DMCopyFields(), DMCopyDS()
@*/
PetscErrorCode DMCopyDisc(DM dm, DM newdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dm == newdm) PetscFunctionReturn(0);
  ierr = DMCopyFields(dm, newdm);CHKERRQ(ierr);
  ierr = DMCopyDS(dm, newdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMRestrictHook_Coordinates(DM dm,DM dmc,void *ctx)
{
  DM dm_coord,dmc_coord;
  PetscErrorCode ierr;
  Vec coords,ccoords;
  Mat inject;
  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(dm,&dm_coord);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dmc,&dmc_coord);CHKERRQ(ierr);
  ierr = DMGetCoordinates(dm,&coords);CHKERRQ(ierr);
  ierr = DMGetCoordinates(dmc,&ccoords);CHKERRQ(ierr);
  if (coords && !ccoords) {
    ierr = DMCreateGlobalVector(dmc_coord,&ccoords);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)ccoords,"coordinates");CHKERRQ(ierr);
    ierr = DMCreateInjection(dmc_coord,dm_coord,&inject);CHKERRQ(ierr);
    ierr = MatRestrict(inject,coords,ccoords);CHKERRQ(ierr);
    ierr = MatDestroy(&inject);CHKERRQ(ierr);
    ierr = DMSetCoordinates(dmc,ccoords);CHKERRQ(ierr);
    ierr = VecDestroy(&ccoords);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSubDomainHook_Coordinates(DM dm,DM subdm,void *ctx)
{
  DM dm_coord,subdm_coord;
  PetscErrorCode ierr;
  Vec coords,ccoords,clcoords;
  VecScatter *scat_i,*scat_g;
  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(dm,&dm_coord);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(subdm,&subdm_coord);CHKERRQ(ierr);
  ierr = DMGetCoordinates(dm,&coords);CHKERRQ(ierr);
  ierr = DMGetCoordinates(subdm,&ccoords);CHKERRQ(ierr);
  if (coords && !ccoords) {
    ierr = DMCreateGlobalVector(subdm_coord,&ccoords);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)ccoords,"coordinates");CHKERRQ(ierr);
    ierr = DMCreateLocalVector(subdm_coord,&clcoords);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)clcoords,"coordinates");CHKERRQ(ierr);
    ierr = DMCreateDomainDecompositionScatters(dm_coord,1,&subdm_coord,NULL,&scat_i,&scat_g);CHKERRQ(ierr);
    ierr = VecScatterBegin(scat_i[0],coords,ccoords,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_i[0],coords,ccoords,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterBegin(scat_g[0],coords,clcoords,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_g[0],coords,clcoords,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = DMSetCoordinates(subdm,ccoords);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(subdm,clcoords);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scat_i[0]);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scat_g[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&ccoords);CHKERRQ(ierr);
    ierr = VecDestroy(&clcoords);CHKERRQ(ierr);
    ierr = PetscFree(scat_i);CHKERRQ(ierr);
    ierr = PetscFree(scat_g);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  DMGetDimension - Return the topological dimension of the DM

  Not collective

  Input Parameter:
. dm - The DM

  Output Parameter:
. dim - The topological dimension

  Level: beginner

.seealso: DMSetDimension(), DMCreate()
@*/
PetscErrorCode DMGetDimension(DM dm, PetscInt *dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(dim, 2);
  *dim = dm->dim;
  PetscFunctionReturn(0);
}

/*@
  DMSetDimension - Set the topological dimension of the DM

  Collective on dm

  Input Parameters:
+ dm - The DM
- dim - The topological dimension

  Level: beginner

.seealso: DMGetDimension(), DMCreate()
@*/
PetscErrorCode DMSetDimension(DM dm, PetscInt dim)
{
  PetscDS        ds;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidLogicalCollectiveInt(dm, dim, 2);
  dm->dim = dim;
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  if (ds->dimEmbed < 0) {ierr = PetscDSSetCoordinateDimension(ds, dm->dim);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  DMGetDimPoints - Get the half-open interval for all points of a given dimension

  Collective on dm

  Input Parameters:
+ dm - the DM
- dim - the dimension

  Output Parameters:
+ pStart - The first point of the given dimension
- pEnd - The first point following points of the given dimension

  Note:
  The points are vertices in the Hasse diagram encoding the topology. This is explained in
  https://arxiv.org/abs/0908.4427. If no points exist of this dimension in the storage scheme,
  then the interval is empty.

  Level: intermediate

.seealso: DMPLEX, DMPlexGetDepthStratum(), DMPlexGetHeightStratum()
@*/
PetscErrorCode DMGetDimPoints(DM dm, PetscInt dim, PetscInt *pStart, PetscInt *pEnd)
{
  PetscInt       d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDimension(dm, &d);CHKERRQ(ierr);
  if ((dim < 0) || (dim > d)) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d 1", dim, d);
  if (!dm->ops->getdimpoints) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "DM type %s does not implement DMGetDimPoints",((PetscObject)dm)->type_name);
  ierr = (*dm->ops->getdimpoints)(dm, dim, pStart, pEnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMSetCoordinates - Sets into the DM a global vector that holds the coordinates

  Collective on dm

  Input Parameters:
+ dm - the DM
- c - coordinate vector

  Notes:
  The coordinates do include those for ghost points, which are in the local vector.

  The vector c should be destroyed by the caller.

  Level: intermediate

.seealso: DMSetCoordinatesLocal(), DMGetCoordinates(), DMGetCoordinatesLocal(), DMGetCoordinateDM()
@*/
PetscErrorCode DMSetCoordinates(DM dm, Vec c)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(c,VEC_CLASSID,2);
  ierr            = PetscObjectReference((PetscObject) c);CHKERRQ(ierr);
  ierr            = VecDestroy(&dm->coordinates);CHKERRQ(ierr);
  dm->coordinates = c;
  ierr            = VecDestroy(&dm->coordinatesLocal);CHKERRQ(ierr);
  ierr            = DMCoarsenHookAdd(dm,DMRestrictHook_Coordinates,NULL,NULL);CHKERRQ(ierr);
  ierr            = DMSubDomainHookAdd(dm,DMSubDomainHook_Coordinates,NULL,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMSetCoordinatesLocal - Sets into the DM a local vector that holds the coordinates

  Not collective

   Input Parameters:
+  dm - the DM
-  c - coordinate vector

  Notes:
  The coordinates of ghost points can be set using DMSetCoordinates()
  followed by DMGetCoordinatesLocal(). This is intended to enable the
  setting of ghost coordinates outside of the domain.

  The vector c should be destroyed by the caller.

  Level: intermediate

.seealso: DMGetCoordinatesLocal(), DMSetCoordinates(), DMGetCoordinates(), DMGetCoordinateDM()
@*/
PetscErrorCode DMSetCoordinatesLocal(DM dm, Vec c)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(c,VEC_CLASSID,2);
  ierr = PetscObjectReference((PetscObject) c);CHKERRQ(ierr);
  ierr = VecDestroy(&dm->coordinatesLocal);CHKERRQ(ierr);

  dm->coordinatesLocal = c;

  ierr = VecDestroy(&dm->coordinates);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMGetCoordinates - Gets a global vector with the coordinates associated with the DM.

  Collective on dm

  Input Parameter:
. dm - the DM

  Output Parameter:
. c - global coordinate vector

  Note:
  This is a borrowed reference, so the user should NOT destroy this vector

  Each process has only the local coordinates (does NOT have the ghost coordinates).

  For DMDA, in two and three dimensions coordinates are interlaced (x_0,y_0,x_1,y_1,...)
  and (x_0,y_0,z_0,x_1,y_1,z_1...)

  Level: intermediate

.seealso: DMSetCoordinates(), DMGetCoordinatesLocal(), DMGetCoordinateDM()
@*/
PetscErrorCode DMGetCoordinates(DM dm, Vec *c)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(c,2);
  if (!dm->coordinates && dm->coordinatesLocal) {
    DM        cdm = NULL;
    PetscBool localized;

    ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(cdm, &dm->coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocalized(dm, &localized);CHKERRQ(ierr);
    /* Block size is not correctly set by CreateGlobalVector() if coordinates are localized */
    if (localized) {
      PetscInt cdim;

      ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
      ierr = VecSetBlockSize(dm->coordinates, cdim);CHKERRQ(ierr);
    }
    ierr = PetscObjectSetName((PetscObject) dm->coordinates, "coordinates");CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(cdm, dm->coordinatesLocal, INSERT_VALUES, dm->coordinates);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(cdm, dm->coordinatesLocal, INSERT_VALUES, dm->coordinates);CHKERRQ(ierr);
  }
  *c = dm->coordinates;
  PetscFunctionReturn(0);
}

/*@
  DMGetCoordinatesLocalSetUp - Prepares a local vector of coordinates, so that DMGetCoordinatesLocalNoncollective() can be used as non-collective afterwards.

  Collective on dm

  Input Parameter:
. dm - the DM

  Level: advanced

.seealso: DMGetCoordinatesLocalNoncollective()
@*/
PetscErrorCode DMGetCoordinatesLocalSetUp(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (!dm->coordinatesLocal && dm->coordinates) {
    DM        cdm = NULL;
    PetscBool localized;

    ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
    ierr = DMCreateLocalVector(cdm, &dm->coordinatesLocal);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocalized(dm, &localized);CHKERRQ(ierr);
    /* Block size is not correctly set by CreateLocalVector() if coordinates are localized */
    if (localized) {
      PetscInt cdim;

      ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
      ierr = VecSetBlockSize(dm->coordinates, cdim);CHKERRQ(ierr);
    }
    ierr = PetscObjectSetName((PetscObject) dm->coordinatesLocal, "coordinates");CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(cdm, dm->coordinates, INSERT_VALUES, dm->coordinatesLocal);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(cdm, dm->coordinates, INSERT_VALUES, dm->coordinatesLocal);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  DMGetCoordinatesLocal - Gets a local vector with the coordinates associated with the DM.

  Collective on dm

  Input Parameter:
. dm - the DM

  Output Parameter:
. c - coordinate vector

  Note:
  This is a borrowed reference, so the user should NOT destroy this vector

  Each process has the local and ghost coordinates

  For DMDA, in two and three dimensions coordinates are interlaced (x_0,y_0,x_1,y_1,...)
  and (x_0,y_0,z_0,x_1,y_1,z_1...)

  Level: intermediate

.seealso: DMSetCoordinatesLocal(), DMGetCoordinates(), DMSetCoordinates(), DMGetCoordinateDM(), DMGetCoordinatesLocalNoncollective()
@*/
PetscErrorCode DMGetCoordinatesLocal(DM dm, Vec *c)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(c,2);
  ierr = DMGetCoordinatesLocalSetUp(dm);CHKERRQ(ierr);
  *c = dm->coordinatesLocal;
  PetscFunctionReturn(0);
}

/*@
  DMGetCoordinatesLocalNoncollective - Non-collective version of DMGetCoordinatesLocal(). Fails if global coordinates have been set and DMGetCoordinatesLocalSetUp() not called.

  Not collective

  Input Parameter:
. dm - the DM

  Output Parameter:
. c - coordinate vector

  Level: advanced

.seealso: DMGetCoordinatesLocalSetUp(), DMGetCoordinatesLocal(), DMSetCoordinatesLocal(), DMGetCoordinates(), DMSetCoordinates(), DMGetCoordinateDM()
@*/
PetscErrorCode DMGetCoordinatesLocalNoncollective(DM dm, Vec *c)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(c,2);
  if (!dm->coordinatesLocal && dm->coordinates) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DMGetCoordinatesLocalSetUp() has not been called");
  *c = dm->coordinatesLocal;
  PetscFunctionReturn(0);
}

/*@
  DMGetCoordinatesLocalTuple - Gets a local vector with the coordinates of specified points and section describing its layout.

  Not collective

  Input Parameter:
+ dm - the DM
- p - the IS of points whose coordinates will be returned

  Output Parameter:
+ pCoordSection - the PetscSection describing the layout of pCoord, i.e. each point corresponds to one point in p, and DOFs correspond to coordinates
- pCoord - the Vec with coordinates of points in p

  Note:
  DMGetCoordinatesLocalSetUp() must be called first. This function employs DMGetCoordinatesLocalNoncollective() so it is not collective.

  This creates a new vector, so the user SHOULD destroy this vector

  Each process has the local and ghost coordinates

  For DMDA, in two and three dimensions coordinates are interlaced (x_0,y_0,x_1,y_1,...)
  and (x_0,y_0,z_0,x_1,y_1,z_1...)

  Level: advanced

.seealso: DMSetCoordinatesLocal(), DMGetCoordinatesLocal(), DMGetCoordinatesLocalNoncollective(), DMGetCoordinatesLocalSetUp(), DMGetCoordinates(), DMSetCoordinates(), DMGetCoordinateDM()
@*/
PetscErrorCode DMGetCoordinatesLocalTuple(DM dm, IS p, PetscSection *pCoordSection, Vec *pCoord)
{
  PetscSection        cs, newcs;
  Vec                 coords;
  const PetscScalar   *arr;
  PetscScalar         *newarr=NULL;
  PetscInt            n;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(p, IS_CLASSID, 2);
  if (pCoordSection) PetscValidPointer(pCoordSection, 3);
  if (pCoord) PetscValidPointer(pCoord, 4);
  if (!dm->coordinatesLocal) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DMGetCoordinatesLocalSetUp() has not been called or coordinates not set");
  if (!dm->coordinateDM || !dm->coordinateDM->localSection) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM not supported");
  cs = dm->coordinateDM->localSection;
  coords = dm->coordinatesLocal;
  ierr = VecGetArrayRead(coords, &arr);CHKERRQ(ierr);
  ierr = PetscSectionExtractDofsFromArray(cs, MPIU_SCALAR, arr, p, &newcs, pCoord ? ((void**)&newarr) : NULL);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coords, &arr);CHKERRQ(ierr);
  if (pCoord) {
    ierr = PetscSectionGetStorageSize(newcs, &n);CHKERRQ(ierr);
    /* set array in two steps to mimic PETSC_OWN_POINTER */
    ierr = VecCreateSeqWithArray(PetscObjectComm((PetscObject)p), 1, n, NULL, pCoord);CHKERRQ(ierr);
    ierr = VecReplaceArray(*pCoord, newarr);CHKERRQ(ierr);
  } else {
    ierr = PetscFree(newarr);CHKERRQ(ierr);
  }
  if (pCoordSection) {*pCoordSection = newcs;}
  else               {ierr = PetscSectionDestroy(&newcs);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode DMGetCoordinateField(DM dm, DMField *field)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(field,2);
  if (!dm->coordinateField) {
    if (dm->ops->createcoordinatefield) {
      ierr = (*dm->ops->createcoordinatefield)(dm,&dm->coordinateField);CHKERRQ(ierr);
    }
  }
  *field = dm->coordinateField;
  PetscFunctionReturn(0);
}

PetscErrorCode DMSetCoordinateField(DM dm, DMField field)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (field) PetscValidHeaderSpecific(field,DMFIELD_CLASSID,2);
  ierr = PetscObjectReference((PetscObject)field);CHKERRQ(ierr);
  ierr = DMFieldDestroy(&dm->coordinateField);CHKERRQ(ierr);
  dm->coordinateField = field;
  PetscFunctionReturn(0);
}

/*@
  DMGetCoordinateDM - Gets the DM that prescribes coordinate layout and scatters between global and local coordinates

  Collective on dm

  Input Parameter:
. dm - the DM

  Output Parameter:
. cdm - coordinate DM

  Level: intermediate

.seealso: DMSetCoordinateDM(), DMSetCoordinates(), DMSetCoordinatesLocal(), DMGetCoordinates(), DMGetCoordinatesLocal()
@*/
PetscErrorCode DMGetCoordinateDM(DM dm, DM *cdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(cdm,2);
  if (!dm->coordinateDM) {
    DM cdm;

    if (!dm->ops->createcoordinatedm) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unable to create coordinates for this DM");
    ierr = (*dm->ops->createcoordinatedm)(dm, &cdm);CHKERRQ(ierr);
    /* Just in case the DM sets the coordinate DM when creating it (DMP4est can do this, because it may not setup
     * until the call to CreateCoordinateDM) */
    ierr = DMDestroy(&dm->coordinateDM);CHKERRQ(ierr);
    dm->coordinateDM = cdm;
  }
  *cdm = dm->coordinateDM;
  PetscFunctionReturn(0);
}

/*@
  DMSetCoordinateDM - Sets the DM that prescribes coordinate layout and scatters between global and local coordinates

  Logically Collective on dm

  Input Parameters:
+ dm - the DM
- cdm - coordinate DM

  Level: intermediate

.seealso: DMGetCoordinateDM(), DMSetCoordinates(), DMSetCoordinatesLocal(), DMGetCoordinates(), DMGetCoordinatesLocal()
@*/
PetscErrorCode DMSetCoordinateDM(DM dm, DM cdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(cdm,DM_CLASSID,2);
  ierr = PetscObjectReference((PetscObject)cdm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm->coordinateDM);CHKERRQ(ierr);
  dm->coordinateDM = cdm;
  PetscFunctionReturn(0);
}

/*@
  DMGetCoordinateDim - Retrieve the dimension of embedding space for coordinate values.

  Not Collective

  Input Parameter:
. dm - The DM object

  Output Parameter:
. dim - The embedding dimension

  Level: intermediate

.seealso: DMSetCoordinateDim(), DMGetCoordinateSection(), DMGetCoordinateDM(), DMGetLocalSection(), DMSetLocalSection()
@*/
PetscErrorCode DMGetCoordinateDim(DM dm, PetscInt *dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(dim, 2);
  if (dm->dimEmbed == PETSC_DEFAULT) {
    dm->dimEmbed = dm->dim;
  }
  *dim = dm->dimEmbed;
  PetscFunctionReturn(0);
}

/*@
  DMSetCoordinateDim - Set the dimension of the embedding space for coordinate values.

  Not Collective

  Input Parameters:
+ dm  - The DM object
- dim - The embedding dimension

  Level: intermediate

.seealso: DMGetCoordinateDim(), DMSetCoordinateSection(), DMGetCoordinateSection(), DMGetLocalSection(), DMSetLocalSection()
@*/
PetscErrorCode DMSetCoordinateDim(DM dm, PetscInt dim)
{
  PetscDS        ds;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->dimEmbed = dim;
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscDSSetCoordinateDimension(ds, dim);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMGetCoordinateSection - Retrieve the layout of coordinate values over the mesh.

  Collective on dm

  Input Parameter:
. dm - The DM object

  Output Parameter:
. section - The PetscSection object

  Level: intermediate

.seealso: DMGetCoordinateDM(), DMGetLocalSection(), DMSetLocalSection()
@*/
PetscErrorCode DMGetCoordinateSection(DM dm, PetscSection *section)
{
  DM             cdm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(section, 2);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetLocalSection(cdm, section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMSetCoordinateSection - Set the layout of coordinate values over the mesh.

  Not Collective

  Input Parameters:
+ dm      - The DM object
. dim     - The embedding dimension, or PETSC_DETERMINE
- section - The PetscSection object

  Level: intermediate

.seealso: DMGetCoordinateSection(), DMGetLocalSection(), DMSetLocalSection()
@*/
PetscErrorCode DMSetCoordinateSection(DM dm, PetscInt dim, PetscSection section)
{
  DM             cdm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(section,PETSC_SECTION_CLASSID,3);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMSetLocalSection(cdm, section);CHKERRQ(ierr);
  if (dim == PETSC_DETERMINE) {
    PetscInt d = PETSC_DEFAULT;
    PetscInt pStart, pEnd, vStart, vEnd, v, dd;

    ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMGetDimPoints(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    pStart = PetscMax(vStart, pStart);
    pEnd   = PetscMin(vEnd, pEnd);
    for (v = pStart; v < pEnd; ++v) {
      ierr = PetscSectionGetDof(section, v, &dd);CHKERRQ(ierr);
      if (dd) {d = dd; break;}
    }
    if (d >= 0) {ierr = DMSetCoordinateDim(dm, d);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

/*@C
  DMGetPeriodicity - Get the description of mesh periodicity

  Input Parameters:
. dm      - The DM object

  Output Parameters:
+ per     - Whether the DM is periodic or not
. maxCell - Over distances greater than this, we can assume a point has crossed over to another sheet, when trying to localize cell coordinates
. L       - If we assume the mesh is a torus, this is the length of each coordinate
- bd      - This describes the type of periodicity in each topological dimension

  Level: developer

.seealso: DMGetPeriodicity()
@*/
PetscErrorCode DMGetPeriodicity(DM dm, PetscBool *per, const PetscReal **maxCell, const PetscReal **L, const DMBoundaryType **bd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (per)     *per     = dm->periodic;
  if (L)       *L       = dm->L;
  if (maxCell) *maxCell = dm->maxCell;
  if (bd)      *bd      = dm->bdtype;
  PetscFunctionReturn(0);
}

/*@C
  DMSetPeriodicity - Set the description of mesh periodicity

  Input Parameters:
+ dm      - The DM object
. per     - Whether the DM is periodic or not. If maxCell is not provided, coordinates need to be localized
. maxCell - Over distances greater than this, we can assume a point has crossed over to another sheet, when trying to localize cell coordinates
. L       - If we assume the mesh is a torus, this is the length of each coordinate
- bd      - This describes the type of periodicity in each topological dimension

  Level: developer

.seealso: DMGetPeriodicity()
@*/
PetscErrorCode DMSetPeriodicity(DM dm, PetscBool per, const PetscReal maxCell[], const PetscReal L[], const DMBoundaryType bd[])
{
  PetscInt       dim, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidLogicalCollectiveBool(dm,per,2);
  if (maxCell) {
    PetscValidRealPointer(maxCell,3);
    PetscValidRealPointer(L,4);
    PetscValidPointer(bd,5);
  }
  ierr = PetscFree3(dm->L,dm->maxCell,dm->bdtype);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (maxCell) {
    ierr = PetscMalloc3(dim,&dm->L,dim,&dm->maxCell,dim,&dm->bdtype);CHKERRQ(ierr);
    for (d = 0; d < dim; ++d) {dm->L[d] = L[d]; dm->maxCell[d] = maxCell[d]; dm->bdtype[d] = bd[d];}
  }
  dm->periodic = per;
  PetscFunctionReturn(0);
}

/*@
  DMLocalizeCoordinate - If a mesh is periodic (a torus with lengths L_i, some of which can be infinite), project the coordinate onto [0, L_i) in each dimension.

  Input Parameters:
+ dm     - The DM
. in     - The input coordinate point (dim numbers)
- endpoint - Include the endpoint L_i

  Output Parameter:
. out - The localized coordinate point

  Level: developer

.seealso: DMLocalizeCoordinates(), DMLocalizeAddCoordinate()
@*/
PetscErrorCode DMLocalizeCoordinate(DM dm, const PetscScalar in[], PetscBool endpoint, PetscScalar out[])
{
  PetscInt       dim, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDim(dm, &dim);CHKERRQ(ierr);
  if (!dm->maxCell) {
    for (d = 0; d < dim; ++d) out[d] = in[d];
  } else {
    if (endpoint) {
      for (d = 0; d < dim; ++d) {
        if ((PetscAbsReal(PetscRealPart(in[d])/dm->L[d] - PetscFloorReal(PetscRealPart(in[d])/dm->L[d])) < PETSC_SMALL) && (PetscRealPart(in[d])/dm->L[d] > PETSC_SMALL)) {
          out[d] = in[d] - dm->L[d]*(PetscFloorReal(PetscRealPart(in[d])/dm->L[d]) - 1);
        } else {
          out[d] = in[d] - dm->L[d]*PetscFloorReal(PetscRealPart(in[d])/dm->L[d]);
        }
      }
    } else {
      for (d = 0; d < dim; ++d) {
        out[d] = in[d] - dm->L[d]*PetscFloorReal(PetscRealPart(in[d])/dm->L[d]);
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
  DMLocalizeCoordinate_Internal - If a mesh is periodic, and the input point is far from the anchor, pick the coordinate sheet of the torus which moves it closer.

  Input Parameters:
+ dm     - The DM
. dim    - The spatial dimension
. anchor - The anchor point, the input point can be no more than maxCell away from it
- in     - The input coordinate point (dim numbers)

  Output Parameter:
. out - The localized coordinate point

  Level: developer

  Note: This is meant to get a set of coordinates close to each other, as in a cell. The anchor is usually the one of the vertices on a containing cell

.seealso: DMLocalizeCoordinates(), DMLocalizeAddCoordinate()
*/
PetscErrorCode DMLocalizeCoordinate_Internal(DM dm, PetscInt dim, const PetscScalar anchor[], const PetscScalar in[], PetscScalar out[])
{
  PetscInt d;

  PetscFunctionBegin;
  if (!dm->maxCell) {
    for (d = 0; d < dim; ++d) out[d] = in[d];
  } else {
    for (d = 0; d < dim; ++d) {
      if ((dm->bdtype[d] != DM_BOUNDARY_NONE) && (PetscAbsScalar(anchor[d] - in[d]) > dm->maxCell[d])) {
        out[d] = PetscRealPart(anchor[d]) > PetscRealPart(in[d]) ? dm->L[d] + in[d] : in[d] - dm->L[d];
      } else {
        out[d] = in[d];
      }
    }
  }
  PetscFunctionReturn(0);
}
PetscErrorCode DMLocalizeCoordinateReal_Internal(DM dm, PetscInt dim, const PetscReal anchor[], const PetscReal in[], PetscReal out[])
{
  PetscInt d;

  PetscFunctionBegin;
  if (!dm->maxCell) {
    for (d = 0; d < dim; ++d) out[d] = in[d];
  } else {
    for (d = 0; d < dim; ++d) {
      if ((dm->bdtype[d] != DM_BOUNDARY_NONE) && (PetscAbsReal(anchor[d] - in[d]) > dm->maxCell[d])) {
        out[d] = anchor[d] > in[d] ? dm->L[d] + in[d] : in[d] - dm->L[d];
      } else {
        out[d] = in[d];
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
  DMLocalizeAddCoordinate_Internal - If a mesh is periodic, and the input point is far from the anchor, pick the coordinate sheet of the torus which moves it closer.

  Input Parameters:
+ dm     - The DM
. dim    - The spatial dimension
. anchor - The anchor point, the input point can be no more than maxCell away from it
. in     - The input coordinate delta (dim numbers)
- out    - The input coordinate point (dim numbers)

  Output Parameter:
. out    - The localized coordinate in + out

  Level: developer

  Note: This is meant to get a set of coordinates close to each other, as in a cell. The anchor is usually the one of the vertices on a containing cell

.seealso: DMLocalizeCoordinates(), DMLocalizeCoordinate()
*/
PetscErrorCode DMLocalizeAddCoordinate_Internal(DM dm, PetscInt dim, const PetscScalar anchor[], const PetscScalar in[], PetscScalar out[])
{
  PetscInt d;

  PetscFunctionBegin;
  if (!dm->maxCell) {
    for (d = 0; d < dim; ++d) out[d] += in[d];
  } else {
    for (d = 0; d < dim; ++d) {
      if ((dm->bdtype[d] != DM_BOUNDARY_NONE) && (PetscAbsScalar(anchor[d] - in[d]) > dm->maxCell[d])) {
        out[d] += PetscRealPart(anchor[d]) > PetscRealPart(in[d]) ? dm->L[d] + in[d] : in[d] - dm->L[d];
      } else {
        out[d] += in[d];
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMGetCoordinatesLocalizedLocal - Check if the DM coordinates have been localized for cells on this process

  Not collective

  Input Parameter:
. dm - The DM

  Output Parameter:
  areLocalized - True if localized

  Level: developer

.seealso: DMLocalizeCoordinates(), DMGetCoordinatesLocalized(), DMSetPeriodicity()
@*/
PetscErrorCode DMGetCoordinatesLocalizedLocal(DM dm,PetscBool *areLocalized)
{
  DM             cdm;
  PetscSection   coordSection;
  PetscInt       cStart, cEnd, sStart, sEnd, c, dof;
  PetscBool      isPlex, alreadyLocalized;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidBoolPointer(areLocalized, 2);
  *areLocalized = PETSC_FALSE;

  /* We need some generic way of refering to cells/vertices */
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) cdm, DMPLEX, &isPlex);CHKERRQ(ierr);
  if (!isPlex) PetscFunctionReturn(0);

  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(cdm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(coordSection, &sStart, &sEnd);CHKERRQ(ierr);
  alreadyLocalized = PETSC_FALSE;
  for (c = cStart; c < cEnd; ++c) {
    if (c < sStart || c >= sEnd) continue;
    ierr = PetscSectionGetDof(coordSection, c, &dof);CHKERRQ(ierr);
    if (dof) { alreadyLocalized = PETSC_TRUE; break; }
  }
  *areLocalized = alreadyLocalized;
  PetscFunctionReturn(0);
}

/*@
  DMGetCoordinatesLocalized - Check if the DM coordinates have been localized for cells

  Collective on dm

  Input Parameter:
. dm - The DM

  Output Parameter:
  areLocalized - True if localized

  Level: developer

.seealso: DMLocalizeCoordinates(), DMSetPeriodicity(), DMGetCoordinatesLocalizedLocal()
@*/
PetscErrorCode DMGetCoordinatesLocalized(DM dm,PetscBool *areLocalized)
{
  PetscBool      localized;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidBoolPointer(areLocalized, 2);
  ierr = DMGetCoordinatesLocalizedLocal(dm,&localized);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&localized,areLocalized,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMLocalizeCoordinates - If a mesh is periodic, create local coordinates for cells having periodic faces

  Collective on dm

  Input Parameter:
. dm - The DM

  Level: developer

.seealso: DMSetPeriodicity(), DMLocalizeCoordinate(), DMLocalizeAddCoordinate()
@*/
PetscErrorCode DMLocalizeCoordinates(DM dm)
{
  DM             cdm;
  PetscSection   coordSection, cSection;
  Vec            coordinates,  cVec;
  PetscScalar   *coords, *coords2, *anchor, *localized;
  PetscInt       Nc, vStart, vEnd, v, sStart, sEnd, newStart = PETSC_MAX_INT, newEnd = PETSC_MIN_INT, dof, d, off, off2, bs, coordSize;
  PetscBool      alreadyLocalized, alreadyLocalizedGlobal;
  PetscInt       maxHeight = 0, h;
  PetscInt       *pStart = NULL, *pEnd = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!dm->periodic) PetscFunctionReturn(0);
  ierr = DMGetCoordinatesLocalized(dm, &alreadyLocalized);CHKERRQ(ierr);
  if (alreadyLocalized) PetscFunctionReturn(0);

  /* We need some generic way of refering to cells/vertices */
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  {
    PetscBool isplex;

    ierr = PetscObjectTypeCompare((PetscObject) cdm, DMPLEX, &isplex);CHKERRQ(ierr);
    if (isplex) {
      ierr = DMPlexGetDepthStratum(cdm, 0, &vStart, &vEnd);CHKERRQ(ierr);
      ierr = DMPlexGetMaxProjectionHeight(cdm,&maxHeight);CHKERRQ(ierr);
      ierr = DMGetWorkArray(dm,2*(maxHeight + 1),MPIU_INT,&pStart);CHKERRQ(ierr);
      pEnd = &pStart[maxHeight + 1];
      newStart = vStart;
      newEnd   = vEnd;
      for (h = 0; h <= maxHeight; h++) {
        ierr = DMPlexGetHeightStratum(cdm, h, &pStart[h], &pEnd[h]);CHKERRQ(ierr);
        newStart = PetscMin(newStart,pStart[h]);
        newEnd   = PetscMax(newEnd,pEnd[h]);
      }
    } else SETERRQ(PetscObjectComm((PetscObject) cdm), PETSC_ERR_ARG_WRONG, "Coordinate localization requires a DMPLEX coordinate DM");
  }
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  if (!coordinates) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Missing local coordinates vector");
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coordinates, &bs);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(coordSection,&sStart,&sEnd);CHKERRQ(ierr);

  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &cSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(cSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionGetFieldComponents(coordSection, 0, &Nc);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(cSection, 0, Nc);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(cSection, newStart, newEnd);CHKERRQ(ierr);

  ierr = DMGetWorkArray(dm, 2 * bs, MPIU_SCALAR, &anchor);CHKERRQ(ierr);
  localized = &anchor[bs];
  alreadyLocalized = alreadyLocalizedGlobal = PETSC_TRUE;
  for (h = 0; h <= maxHeight; h++) {
    PetscInt cStart = pStart[h], cEnd = pEnd[h], c;

    for (c = cStart; c < cEnd; ++c) {
      PetscScalar *cellCoords = NULL;
      PetscInt     b;

      if (c < sStart || c >= sEnd) alreadyLocalized = PETSC_FALSE;
      ierr = DMPlexVecGetClosure(cdm, coordSection, coordinates, c, &dof, &cellCoords);CHKERRQ(ierr);
      for (b = 0; b < bs; ++b) anchor[b] = cellCoords[b];
      for (d = 0; d < dof/bs; ++d) {
        ierr = DMLocalizeCoordinate_Internal(dm, bs, anchor, &cellCoords[d*bs], localized);CHKERRQ(ierr);
        for (b = 0; b < bs; b++) {
          if (cellCoords[d*bs + b] != localized[b]) break;
        }
        if (b < bs) break;
      }
      if (d < dof/bs) {
        if (c >= sStart && c < sEnd) {
          PetscInt cdof;

          ierr = PetscSectionGetDof(coordSection, c, &cdof);CHKERRQ(ierr);
          if (cdof != dof) alreadyLocalized = PETSC_FALSE;
        }
        ierr = PetscSectionSetDof(cSection, c, dof);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldDof(cSection, c, 0, dof);CHKERRQ(ierr);
      }
      ierr = DMPlexVecRestoreClosure(cdm, coordSection, coordinates, c, &dof, &cellCoords);CHKERRQ(ierr);
    }
  }
  ierr = MPI_Allreduce(&alreadyLocalized,&alreadyLocalizedGlobal,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  if (alreadyLocalizedGlobal) {
    ierr = DMRestoreWorkArray(dm, 2 * bs, MPIU_SCALAR, &anchor);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&cSection);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm,2*(maxHeight + 1),MPIU_INT,&pStart);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  for (v = vStart; v < vEnd; ++v) {
    ierr = PetscSectionGetDof(coordSection, v, &dof);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(cSection, v, dof);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(cSection, v, 0, dof);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(cSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(cSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &cVec);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)cVec,"coordinates");CHKERRQ(ierr);
  ierr = VecSetBlockSize(cVec, bs);CHKERRQ(ierr);
  ierr = VecSetSizes(cVec, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetType(cVec, VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, (const PetscScalar**)&coords);CHKERRQ(ierr);
  ierr = VecGetArray(cVec, &coords2);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    ierr = PetscSectionGetDof(coordSection, v, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(cSection,     v, &off2);CHKERRQ(ierr);
    for (d = 0; d < dof; ++d) coords2[off2+d] = coords[off+d];
  }
  for (h = 0; h <= maxHeight; h++) {
    PetscInt cStart = pStart[h], cEnd = pEnd[h], c;

    for (c = cStart; c < cEnd; ++c) {
      PetscScalar *cellCoords = NULL;
      PetscInt     b, cdof;

      ierr = PetscSectionGetDof(cSection,c,&cdof);CHKERRQ(ierr);
      if (!cdof) continue;
      ierr = DMPlexVecGetClosure(cdm, coordSection, coordinates, c, &dof, &cellCoords);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(cSection, c, &off2);CHKERRQ(ierr);
      for (b = 0; b < bs; ++b) anchor[b] = cellCoords[b];
      for (d = 0; d < dof/bs; ++d) {ierr = DMLocalizeCoordinate_Internal(dm, bs, anchor, &cellCoords[d*bs], &coords2[off2+d*bs]);CHKERRQ(ierr);}
      ierr = DMPlexVecRestoreClosure(cdm, coordSection, coordinates, c, &dof, &cellCoords);CHKERRQ(ierr);
    }
  }
  ierr = DMRestoreWorkArray(dm, 2 * bs, MPIU_SCALAR, &anchor);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm,2*(maxHeight + 1),MPIU_INT,&pStart);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coordinates, (const PetscScalar**)&coords);CHKERRQ(ierr);
  ierr = VecRestoreArray(cVec, &coords2);CHKERRQ(ierr);
  ierr = DMSetCoordinateSection(dm, PETSC_DETERMINE, cSection);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, cVec);CHKERRQ(ierr);
  ierr = VecDestroy(&cVec);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&cSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMLocatePoints - Locate the points in v in the mesh and return a PetscSF of the containing cells

  Collective on v (see explanation below)

  Input Parameters:
+ dm - The DM
. v - The Vec of points
. ltype - The type of point location, e.g. DM_POINTLOCATION_NONE or DM_POINTLOCATION_NEAREST
- cells - Points to either NULL, or a PetscSF with guesses for which cells contain each point.

  Output Parameter:
+ v - The Vec of points, which now contains the nearest mesh points to the given points if DM_POINTLOCATION_NEAREST is used
- cells - The PetscSF containing the ranks and local indices of the containing points.


  Level: developer

  Notes:
  To do a search of the local cells of the mesh, v should have PETSC_COMM_SELF as its communicator.
  To do a search of all the cells in the distributed mesh, v should have the same communicator as dm.

  If *cellSF is NULL on input, a PetscSF will be created.
  If *cellSF is not NULL on input, it should point to an existing PetscSF, whose graph will be used as initial guesses.

  An array that maps each point to its containing cell can be obtained with

$    const PetscSFNode *cells;
$    PetscInt           nFound;
$    const PetscInt    *found;
$
$    PetscSFGetGraph(cellSF,NULL,&nFound,&found,&cells);

  Where cells[i].rank is the rank of the cell containing point found[i] (or i if found == NULL), and cells[i].index is
  the index of the cell in its rank's local numbering.

.seealso: DMSetCoordinates(), DMSetCoordinatesLocal(), DMGetCoordinates(), DMGetCoordinatesLocal(), DMPointLocationType
@*/
PetscErrorCode DMLocatePoints(DM dm, Vec v, DMPointLocationType ltype, PetscSF *cellSF)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  PetscValidPointer(cellSF,4);
  if (*cellSF) {
    PetscMPIInt result;

    PetscValidHeaderSpecific(*cellSF,PETSCSF_CLASSID,4);
    ierr = MPI_Comm_compare(PetscObjectComm((PetscObject)v),PetscObjectComm((PetscObject)*cellSF),&result);CHKERRQ(ierr);
    if (result != MPI_IDENT && result != MPI_CONGRUENT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"cellSF must have a communicator congruent to v's");
  } else {
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)v),cellSF);CHKERRQ(ierr);
  }
  if (!dm->ops->locatepoints) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Point location not available for this DM");
  ierr = PetscLogEventBegin(DM_LocatePoints,dm,0,0,0);CHKERRQ(ierr);
  ierr = (*dm->ops->locatepoints)(dm,v,ltype,*cellSF);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DM_LocatePoints,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMGetOutputDM - Retrieve the DM associated with the layout for output

  Collective on dm

  Input Parameter:
. dm - The original DM

  Output Parameter:
. odm - The DM which provides the layout for output

  Level: intermediate

.seealso: VecView(), DMGetGlobalSection()
@*/
PetscErrorCode DMGetOutputDM(DM dm, DM *odm)
{
  PetscSection   section;
  PetscBool      hasConstraints, ghasConstraints;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(odm,2);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionHasConstraints(section, &hasConstraints);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&hasConstraints, &ghasConstraints, 1, MPIU_BOOL, MPI_LOR, PetscObjectComm((PetscObject) dm));CHKERRQ(ierr);
  if (!ghasConstraints) {
    *odm = dm;
    PetscFunctionReturn(0);
  }
  if (!dm->dmBC) {
    PetscSection newSection, gsection;
    PetscSF      sf;

    ierr = DMClone(dm, &dm->dmBC);CHKERRQ(ierr);
    ierr = DMCopyDisc(dm, dm->dmBC);CHKERRQ(ierr);
    ierr = PetscSectionClone(section, &newSection);CHKERRQ(ierr);
    ierr = DMSetLocalSection(dm->dmBC, newSection);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&newSection);CHKERRQ(ierr);
    ierr = DMGetPointSF(dm->dmBC, &sf);CHKERRQ(ierr);
    ierr = PetscSectionCreateGlobalSection(section, sf, PETSC_TRUE, PETSC_FALSE, &gsection);CHKERRQ(ierr);
    ierr = DMSetGlobalSection(dm->dmBC, gsection);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&gsection);CHKERRQ(ierr);
  }
  *odm = dm->dmBC;
  PetscFunctionReturn(0);
}

/*@
  DMGetOutputSequenceNumber - Retrieve the sequence number/value for output

  Input Parameter:
. dm - The original DM

  Output Parameters:
+ num - The output sequence number
- val - The output sequence value

  Level: intermediate

  Note: This is intended for output that should appear in sequence, for instance
  a set of timesteps in an HDF5 file, or a set of realizations of a stochastic system.

.seealso: VecView()
@*/
PetscErrorCode DMGetOutputSequenceNumber(DM dm, PetscInt *num, PetscReal *val)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (num) {PetscValidIntPointer(num,2); *num = dm->outputSequenceNum;}
  if (val) {PetscValidRealPointer(val,3);*val = dm->outputSequenceVal;}
  PetscFunctionReturn(0);
}

/*@
  DMSetOutputSequenceNumber - Set the sequence number/value for output

  Input Parameters:
+ dm - The original DM
. num - The output sequence number
- val - The output sequence value

  Level: intermediate

  Note: This is intended for output that should appear in sequence, for instance
  a set of timesteps in an HDF5 file, or a set of realizations of a stochastic system.

.seealso: VecView()
@*/
PetscErrorCode DMSetOutputSequenceNumber(DM dm, PetscInt num, PetscReal val)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->outputSequenceNum = num;
  dm->outputSequenceVal = val;
  PetscFunctionReturn(0);
}

/*@C
  DMOutputSequenceLoad - Retrieve the sequence value from a Viewer

  Input Parameters:
+ dm   - The original DM
. name - The sequence name
- num  - The output sequence number

  Output Parameter:
. val  - The output sequence value

  Level: intermediate

  Note: This is intended for output that should appear in sequence, for instance
  a set of timesteps in an HDF5 file, or a set of realizations of a stochastic system.

.seealso: DMGetOutputSequenceNumber(), DMSetOutputSequenceNumber(), VecView()
@*/
PetscErrorCode DMOutputSequenceLoad(DM dm, PetscViewer viewer, const char *name, PetscInt num, PetscReal *val)
{
  PetscBool      ishdf5;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscValidRealPointer(val,4);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5, &ishdf5);CHKERRQ(ierr);
  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    PetscScalar value;

    ierr = DMSequenceLoad_HDF5_Internal(dm, name, num, &value, viewer);CHKERRQ(ierr);
    *val = PetscRealPart(value);
#endif
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid viewer; open viewer with PetscViewerHDF5Open()");
  PetscFunctionReturn(0);
}

/*@
  DMGetUseNatural - Get the flag for creating a mapping to the natural order on distribution

  Not collective

  Input Parameter:
. dm - The DM

  Output Parameter:
. useNatural - The flag to build the mapping to a natural order during distribution

  Level: beginner

.seealso: DMSetUseNatural(), DMCreate()
@*/
PetscErrorCode DMGetUseNatural(DM dm, PetscBool *useNatural)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidBoolPointer(useNatural, 2);
  *useNatural = dm->useNatural;
  PetscFunctionReturn(0);
}

/*@
  DMSetUseNatural - Set the flag for creating a mapping to the natural order after distribution

  Collective on dm

  Input Parameters:
+ dm - The DM
- useNatural - The flag to build the mapping to a natural order during distribution

  Note: This also causes the map to be build after DMCreateSubDM() and DMCreateSuperDM()

  Level: beginner

.seealso: DMGetUseNatural(), DMCreate(), DMPlexDistribute(), DMCreateSubDM(), DMCreateSuperDM()
@*/
PetscErrorCode DMSetUseNatural(DM dm, PetscBool useNatural)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidLogicalCollectiveBool(dm, useNatural, 2);
  dm->useNatural = useNatural;
  PetscFunctionReturn(0);
}


/*@C
  DMCreateLabel - Create a label of the given name if it does not already exist

  Not Collective

  Input Parameters:
+ dm   - The DM object
- name - The label name

  Level: intermediate

.seealso: DMLabelCreate(), DMHasLabel(), DMGetLabelValue(), DMSetLabelValue(), DMGetStratumIS()
@*/
PetscErrorCode DMCreateLabel(DM dm, const char name[])
{
  PetscBool      flg;
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  ierr = DMHasLabel(dm, name, &flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = DMLabelCreate(PETSC_COMM_SELF, name, &label);CHKERRQ(ierr);
    ierr = DMAddLabel(dm, label);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&label);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMGetLabelValue - Get the value in a Sieve Label for the given point, with 0 as the default

  Not Collective

  Input Parameters:
+ dm   - The DM object
. name - The label name
- point - The mesh point

  Output Parameter:
. value - The label value for this point, or -1 if the point is not in the label

  Level: beginner

.seealso: DMLabelGetValue(), DMSetLabelValue(), DMGetStratumIS()
@*/
PetscErrorCode DMGetLabelValue(DM dm, const char name[], PetscInt point, PetscInt *value)
{
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
  if (!label) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No label named %s was found", name);
  ierr = DMLabelGetValue(label, point, value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMSetLabelValue - Add a point to a Sieve Label with given value

  Not Collective

  Input Parameters:
+ dm   - The DM object
. name - The label name
. point - The mesh point
- value - The label value for this point

  Output Parameter:

  Level: beginner

.seealso: DMLabelSetValue(), DMGetStratumIS(), DMClearLabelValue()
@*/
PetscErrorCode DMSetLabelValue(DM dm, const char name[], PetscInt point, PetscInt value)
{
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
  if (!label) {
    ierr = DMCreateLabel(dm, name);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
  }
  ierr = DMLabelSetValue(label, point, value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMClearLabelValue - Remove a point from a Sieve Label with given value

  Not Collective

  Input Parameters:
+ dm   - The DM object
. name - The label name
. point - The mesh point
- value - The label value for this point

  Output Parameter:

  Level: beginner

.seealso: DMLabelClearValue(), DMSetLabelValue(), DMGetStratumIS()
@*/
PetscErrorCode DMClearLabelValue(DM dm, const char name[], PetscInt point, PetscInt value)
{
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
  if (!label) PetscFunctionReturn(0);
  ierr = DMLabelClearValue(label, point, value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMGetLabelSize - Get the number of different integer ids in a Label

  Not Collective

  Input Parameters:
+ dm   - The DM object
- name - The label name

  Output Parameter:
. size - The number of different integer ids, or 0 if the label does not exist

  Level: beginner

.seealso: DMLabelGetNumValues(), DMSetLabelValue()
@*/
PetscErrorCode DMGetLabelSize(DM dm, const char name[], PetscInt *size)
{
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidIntPointer(size, 3);
  ierr  = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
  *size = 0;
  if (!label) PetscFunctionReturn(0);
  ierr = DMLabelGetNumValues(label, size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMGetLabelIdIS - Get the integer ids in a label

  Not Collective

  Input Parameters:
+ mesh - The DM object
- name - The label name

  Output Parameter:
. ids - The integer ids, or NULL if the label does not exist

  Level: beginner

.seealso: DMLabelGetValueIS(), DMGetLabelSize()
@*/
PetscErrorCode DMGetLabelIdIS(DM dm, const char name[], IS *ids)
{
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(ids, 3);
  ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
  *ids = NULL;
 if (label) {
    ierr = DMLabelGetValueIS(label, ids);CHKERRQ(ierr);
  } else {
    /* returning an empty IS */
    ierr = ISCreateGeneral(PETSC_COMM_SELF,0,NULL,PETSC_USE_POINTER,ids);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMGetStratumSize - Get the number of points in a label stratum

  Not Collective

  Input Parameters:
+ dm - The DM object
. name - The label name
- value - The stratum value

  Output Parameter:
. size - The stratum size

  Level: beginner

.seealso: DMLabelGetStratumSize(), DMGetLabelSize(), DMGetLabelIds()
@*/
PetscErrorCode DMGetStratumSize(DM dm, const char name[], PetscInt value, PetscInt *size)
{
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidIntPointer(size, 4);
  ierr  = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
  *size = 0;
  if (!label) PetscFunctionReturn(0);
  ierr = DMLabelGetStratumSize(label, value, size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMGetStratumIS - Get the points in a label stratum

  Not Collective

  Input Parameters:
+ dm - The DM object
. name - The label name
- value - The stratum value

  Output Parameter:
. points - The stratum points, or NULL if the label does not exist or does not have that value

  Level: beginner

.seealso: DMLabelGetStratumIS(), DMGetStratumSize()
@*/
PetscErrorCode DMGetStratumIS(DM dm, const char name[], PetscInt value, IS *points)
{
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(points, 4);
  ierr    = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
  *points = NULL;
  if (!label) PetscFunctionReturn(0);
  ierr = DMLabelGetStratumIS(label, value, points);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMSetStratumIS - Set the points in a label stratum

  Not Collective

  Input Parameters:
+ dm - The DM object
. name - The label name
. value - The stratum value
- points - The stratum points

  Level: beginner

.seealso: DMLabelSetStratumIS(), DMGetStratumSize()
@*/
PetscErrorCode DMSetStratumIS(DM dm, const char name[], PetscInt value, IS points)
{
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(points, 4);
  ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
  if (!label) PetscFunctionReturn(0);
  ierr = DMLabelSetStratumIS(label, value, points);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMClearLabelStratum - Remove all points from a stratum from a Sieve Label

  Not Collective

  Input Parameters:
+ dm   - The DM object
. name - The label name
- value - The label value for this point

  Output Parameter:

  Level: beginner

.seealso: DMLabelClearStratum(), DMSetLabelValue(), DMGetStratumIS(), DMClearLabelValue()
@*/
PetscErrorCode DMClearLabelStratum(DM dm, const char name[], PetscInt value)
{
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
  if (!label) PetscFunctionReturn(0);
  ierr = DMLabelClearStratum(label, value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMGetNumLabels - Return the number of labels defined by the mesh

  Not Collective

  Input Parameter:
. dm   - The DM object

  Output Parameter:
. numLabels - the number of Labels

  Level: intermediate

.seealso: DMGetLabelValue(), DMSetLabelValue(), DMGetStratumIS()
@*/
PetscErrorCode DMGetNumLabels(DM dm, PetscInt *numLabels)
{
  DMLabelLink next = dm->labels;
  PetscInt  n    = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(numLabels, 2);
  while (next) {++n; next = next->next;}
  *numLabels = n;
  PetscFunctionReturn(0);
}

/*@C
  DMGetLabelName - Return the name of nth label

  Not Collective

  Input Parameters:
+ dm - The DM object
- n  - the label number

  Output Parameter:
. name - the label name

  Level: intermediate

.seealso: DMGetLabelValue(), DMSetLabelValue(), DMGetStratumIS()
@*/
PetscErrorCode DMGetLabelName(DM dm, PetscInt n, const char **name)
{
  DMLabelLink    next = dm->labels;
  PetscInt       l    = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(name, 3);
  while (next) {
    if (l == n) {
      ierr = PetscObjectGetName((PetscObject) next->label, name);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ++l;
    next = next->next;
  }
  SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label %D does not exist in this DM", n);
}

/*@C
  DMHasLabel - Determine whether the mesh has a label of a given name

  Not Collective

  Input Parameters:
+ dm   - The DM object
- name - The label name

  Output Parameter:
. hasLabel - PETSC_TRUE if the label is present

  Level: intermediate

.seealso: DMCreateLabel(), DMGetLabelValue(), DMSetLabelValue(), DMGetStratumIS()
@*/
PetscErrorCode DMHasLabel(DM dm, const char name[], PetscBool *hasLabel)
{
  DMLabelLink    next = dm->labels;
  const char    *lname;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidBoolPointer(hasLabel, 3);
  *hasLabel = PETSC_FALSE;
  while (next) {
    ierr = PetscObjectGetName((PetscObject) next->label, &lname);CHKERRQ(ierr);
    ierr = PetscStrcmp(name, lname, hasLabel);CHKERRQ(ierr);
    if (*hasLabel) break;
    next = next->next;
  }
  PetscFunctionReturn(0);
}

/*@C
  DMGetLabel - Return the label of a given name, or NULL

  Not Collective

  Input Parameters:
+ dm   - The DM object
- name - The label name

  Output Parameter:
. label - The DMLabel, or NULL if the label is absent

  Level: intermediate

.seealso: DMCreateLabel(), DMHasLabel(), DMGetLabelValue(), DMSetLabelValue(), DMGetStratumIS()
@*/
PetscErrorCode DMGetLabel(DM dm, const char name[], DMLabel *label)
{
  DMLabelLink    next = dm->labels;
  PetscBool      hasLabel;
  const char    *lname;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(label, 3);
  *label = NULL;
  while (next) {
    ierr = PetscObjectGetName((PetscObject) next->label, &lname);CHKERRQ(ierr);
    ierr = PetscStrcmp(name, lname, &hasLabel);CHKERRQ(ierr);
    if (hasLabel) {
      *label = next->label;
      break;
    }
    next = next->next;
  }
  PetscFunctionReturn(0);
}

/*@C
  DMGetLabelByNum - Return the nth label

  Not Collective

  Input Parameters:
+ dm - The DM object
- n  - the label number

  Output Parameter:
. label - the label

  Level: intermediate

.seealso: DMGetLabelValue(), DMSetLabelValue(), DMGetStratumIS()
@*/
PetscErrorCode DMGetLabelByNum(DM dm, PetscInt n, DMLabel *label)
{
  DMLabelLink next = dm->labels;
  PetscInt    l    = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(label, 3);
  while (next) {
    if (l == n) {
      *label = next->label;
      PetscFunctionReturn(0);
    }
    ++l;
    next = next->next;
  }
  SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label %D does not exist in this DM", n);
}

/*@C
  DMAddLabel - Add the label to this mesh

  Not Collective

  Input Parameters:
+ dm   - The DM object
- label - The DMLabel

  Level: developer

.seealso: DMCreateLabel(), DMHasLabel(), DMGetLabelValue(), DMSetLabelValue(), DMGetStratumIS()
@*/
PetscErrorCode DMAddLabel(DM dm, DMLabel label)
{
  DMLabelLink    l, *p, tmpLabel;
  PetscBool      hasLabel;
  const char    *lname;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscObjectGetName((PetscObject) label, &lname);CHKERRQ(ierr);
  ierr = DMHasLabel(dm, lname, &hasLabel);CHKERRQ(ierr);
  if (hasLabel) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label %s already exists in this DM", lname);
  ierr = PetscCalloc1(1, &tmpLabel);CHKERRQ(ierr);
  tmpLabel->label  = label;
  tmpLabel->output = PETSC_TRUE;
  for (p=&dm->labels; (l=*p); p=&l->next) {}
  *p = tmpLabel;
  ierr = PetscObjectReference((PetscObject)label);CHKERRQ(ierr);
  ierr = PetscStrcmp(lname, "depth", &flg);CHKERRQ(ierr);
  if (flg) dm->depthLabel = label;
  ierr = PetscStrcmp(lname, "celltype", &flg);CHKERRQ(ierr);
  if (flg) dm->celltypeLabel = label;
  PetscFunctionReturn(0);
}

/*@C
  DMRemoveLabel - Remove the label given by name from this mesh

  Not Collective

  Input Parameters:
+ dm   - The DM object
- name - The label name

  Output Parameter:
. label - The DMLabel, or NULL if the label is absent

  Level: developer

  Notes:
  DMRemoveLabel(dm,name,NULL) removes the label from dm and calls
  DMLabelDestroy() on the label.

  DMRemoveLabel(dm,name,&label) removes the label from dm, but it DOES NOT
  call DMLabelDestroy(). Instead, the label is returned and the user is
  responsible of calling DMLabelDestroy() at some point.

.seealso: DMCreateLabel(), DMHasLabel(), DMGetLabel(), DMGetLabelValue(), DMSetLabelValue(), DMLabelDestroy(), DMRemoveLabelBySelf()
@*/
PetscErrorCode DMRemoveLabel(DM dm, const char name[], DMLabel *label)
{
  DMLabelLink    link, *pnext;
  PetscBool      hasLabel;
  const char    *lname;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  if (label) {
    PetscValidPointer(label, 3);
    *label = NULL;
  }
  for (pnext=&dm->labels; (link=*pnext); pnext=&link->next) {
    ierr = PetscObjectGetName((PetscObject) link->label, &lname);CHKERRQ(ierr);
    ierr = PetscStrcmp(name, lname, &hasLabel);CHKERRQ(ierr);
    if (hasLabel) {
      *pnext = link->next; /* Remove from list */
      ierr = PetscStrcmp(name, "depth", &hasLabel);CHKERRQ(ierr);
      if (hasLabel) dm->depthLabel = NULL;
      ierr = PetscStrcmp(name, "celltype", &hasLabel);CHKERRQ(ierr);
      if (hasLabel) dm->celltypeLabel = NULL;
      if (label) *label = link->label;
      else       {ierr = DMLabelDestroy(&link->label);CHKERRQ(ierr);}
      ierr = PetscFree(link);CHKERRQ(ierr);
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMRemoveLabelBySelf - Remove the label from this mesh

  Not Collective

  Input Parameters:
+ dm   - The DM object
. label - (Optional) The DMLabel to be removed from the DM
- failNotFound - Should it fail if the label is not found in the DM?

  Level: developer

  Notes:
  Only exactly the same instance is removed if found, name match is ignored.
  If the DM has an exclusive reference to the label, it gets destroyed and
  *label nullified.

.seealso: DMCreateLabel(), DMHasLabel(), DMGetLabel() DMGetLabelValue(), DMSetLabelValue(), DMLabelDestroy(), DMRemoveLabel()
@*/
PetscErrorCode DMRemoveLabelBySelf(DM dm, DMLabel *label, PetscBool failNotFound)
{
  DMLabelLink    link, *pnext;
  PetscBool      hasLabel = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(label, 2);
  if (!*label && !failNotFound) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*label, DMLABEL_CLASSID, 2);
  PetscValidLogicalCollectiveBool(dm,failNotFound,3);
  for (pnext=&dm->labels; (link=*pnext); pnext=&link->next) {
    if (*label == link->label) {
      hasLabel = PETSC_TRUE;
      *pnext = link->next; /* Remove from list */
      if (*label == dm->depthLabel) dm->depthLabel = NULL;
      if (*label == dm->celltypeLabel) dm->celltypeLabel = NULL;
      if (((PetscObject) link->label)->refct < 2) *label = NULL; /* nullify if exclusive reference */
      ierr = DMLabelDestroy(&link->label);CHKERRQ(ierr);
      ierr = PetscFree(link);CHKERRQ(ierr);
      break;
    }
  }
  if (!hasLabel && failNotFound) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Given label not found in DM");
  PetscFunctionReturn(0);
}

/*@C
  DMGetLabelOutput - Get the output flag for a given label

  Not Collective

  Input Parameters:
+ dm   - The DM object
- name - The label name

  Output Parameter:
. output - The flag for output

  Level: developer

.seealso: DMSetLabelOutput(), DMCreateLabel(), DMHasLabel(), DMGetLabelValue(), DMSetLabelValue(), DMGetStratumIS()
@*/
PetscErrorCode DMGetLabelOutput(DM dm, const char name[], PetscBool *output)
{
  DMLabelLink    next = dm->labels;
  const char    *lname;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(name, 2);
  PetscValidPointer(output, 3);
  while (next) {
    PetscBool flg;

    ierr = PetscObjectGetName((PetscObject) next->label, &lname);CHKERRQ(ierr);
    ierr = PetscStrcmp(name, lname, &flg);CHKERRQ(ierr);
    if (flg) {*output = next->output; PetscFunctionReturn(0);}
    next = next->next;
  }
  SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No label named %s was present in this dm", name);
}

/*@C
  DMSetLabelOutput - Set the output flag for a given label

  Not Collective

  Input Parameters:
+ dm     - The DM object
. name   - The label name
- output - The flag for output

  Level: developer

.seealso: DMGetLabelOutput(), DMCreateLabel(), DMHasLabel(), DMGetLabelValue(), DMSetLabelValue(), DMGetStratumIS()
@*/
PetscErrorCode DMSetLabelOutput(DM dm, const char name[], PetscBool output)
{
  DMLabelLink    next = dm->labels;
  const char    *lname;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  while (next) {
    PetscBool flg;

    ierr = PetscObjectGetName((PetscObject) next->label, &lname);CHKERRQ(ierr);
    ierr = PetscStrcmp(name, lname, &flg);CHKERRQ(ierr);
    if (flg) {next->output = output; PetscFunctionReturn(0);}
    next = next->next;
  }
  SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No label named %s was present in this dm", name);
}

/*@
  DMCopyLabels - Copy labels from one mesh to another with a superset of the points

  Collective on dmA

  Input Parameter:
+ dmA - The DM object with initial labels
. dmB - The DM object with copied labels
. mode - Copy labels by pointers (PETSC_OWN_POINTER) or duplicate them (PETSC_COPY_VALUES)
- all  - Copy all labels including "depth", "dim", and "celltype" (PETSC_TRUE) which are otherwise ignored (PETSC_FALSE)

  Level: intermediate

  Note: This is typically used when interpolating or otherwise adding to a mesh

.seealso: DMGetCoordinates(), DMGetCoordinatesLocal(), DMGetCoordinateDM(), DMGetCoordinateSection(), DMShareLabels()
@*/
PetscErrorCode DMCopyLabels(DM dmA, DM dmB, PetscCopyMode mode, PetscBool all)
{
  DMLabel        label, labelNew;
  const char    *name;
  PetscBool      flg;
  DMLabelLink    link;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmA, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmB, DM_CLASSID, 2);
  PetscValidLogicalCollectiveEnum(dmA, mode,3);
  PetscValidLogicalCollectiveBool(dmA, all, 4);
  if (mode==PETSC_USE_POINTER) SETERRQ(PetscObjectComm((PetscObject)dmA), PETSC_ERR_SUP, "PETSC_USE_POINTER not supported for objects");
  if (dmA == dmB) PetscFunctionReturn(0);
  for (link=dmA->labels; link; link=link->next) {
    label=link->label;
    ierr = PetscObjectGetName((PetscObject)label, &name);CHKERRQ(ierr);
    if (!all) {
      ierr = PetscStrcmp(name, "depth", &flg);CHKERRQ(ierr);
      if (flg) continue;
      ierr = PetscStrcmp(name, "dim", &flg);CHKERRQ(ierr);
      if (flg) continue;
      ierr = PetscStrcmp(name, "celltype", &flg);CHKERRQ(ierr);
      if (flg) continue;
    } else {
      dmB->depthLabel    = dmA->depthLabel;
      dmB->celltypeLabel = dmA->celltypeLabel;
    }
    if (mode==PETSC_COPY_VALUES) {
      ierr = DMLabelDuplicate(label, &labelNew);CHKERRQ(ierr);
    } else {
      labelNew = label;
    }
    ierr = DMAddLabel(dmB, labelNew);CHKERRQ(ierr);
    if (mode==PETSC_COPY_VALUES) {ierr = DMLabelDestroy(&labelNew);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

/*@
  DMGetCoarseDM - Get the coarse mesh from which this was obtained by refinement

  Input Parameter:
. dm - The DM object

  Output Parameter:
. cdm - The coarse DM

  Level: intermediate

.seealso: DMSetCoarseDM()
@*/
PetscErrorCode DMGetCoarseDM(DM dm, DM *cdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(cdm, 2);
  *cdm = dm->coarseMesh;
  PetscFunctionReturn(0);
}

/*@
  DMSetCoarseDM - Set the coarse mesh from which this was obtained by refinement

  Input Parameters:
+ dm - The DM object
- cdm - The coarse DM

  Level: intermediate

.seealso: DMGetCoarseDM()
@*/
PetscErrorCode DMSetCoarseDM(DM dm, DM cdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (cdm) PetscValidHeaderSpecific(cdm, DM_CLASSID, 2);
  ierr = PetscObjectReference((PetscObject)cdm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm->coarseMesh);CHKERRQ(ierr);
  dm->coarseMesh = cdm;
  PetscFunctionReturn(0);
}

/*@
  DMGetFineDM - Get the fine mesh from which this was obtained by refinement

  Input Parameter:
. dm - The DM object

  Output Parameter:
. fdm - The fine DM

  Level: intermediate

.seealso: DMSetFineDM()
@*/
PetscErrorCode DMGetFineDM(DM dm, DM *fdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(fdm, 2);
  *fdm = dm->fineMesh;
  PetscFunctionReturn(0);
}

/*@
  DMSetFineDM - Set the fine mesh from which this was obtained by refinement

  Input Parameters:
+ dm - The DM object
- fdm - The fine DM

  Level: intermediate

.seealso: DMGetFineDM()
@*/
PetscErrorCode DMSetFineDM(DM dm, DM fdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (fdm) PetscValidHeaderSpecific(fdm, DM_CLASSID, 2);
  ierr = PetscObjectReference((PetscObject)fdm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm->fineMesh);CHKERRQ(ierr);
  dm->fineMesh = fdm;
  PetscFunctionReturn(0);
}

/*=== DMBoundary code ===*/

PetscErrorCode DMCopyBoundary(DM dm, DM dmNew)
{
  PetscInt       d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (d = 0; d < dm->Nds; ++d) {
    ierr = PetscDSCopyBoundary(dm->probs[d].ds, dmNew->probs[d].ds);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMAddBoundary - Add a boundary condition to the model

  Input Parameters:
+ dm          - The DM, with a PetscDS that matches the problem being constrained
. type        - The type of condition, e.g. DM_BC_ESSENTIAL_ANALYTIC/DM_BC_ESSENTIAL_FIELD (Dirichlet), or DM_BC_NATURAL (Neumann)
. name        - The BC name
. labelname   - The label defining constrained points
. field       - The field to constrain
. numcomps    - The number of constrained field components (0 will constrain all fields)
. comps       - An array of constrained component numbers
. bcFunc      - A pointwise function giving boundary values
. numids      - The number of DMLabel ids for constrained points
. ids         - An array of ids for constrained points
- ctx         - An optional user context for bcFunc

  Options Database Keys:
+ -bc_<boundary name> <num> - Overrides the boundary ids
- -bc_<boundary name>_comp <num> - Overrides the boundary components

  Level: developer

.seealso: DMGetBoundary()
@*/
PetscErrorCode DMAddBoundary(DM dm, DMBoundaryConditionType type, const char name[], const char labelname[], PetscInt field, PetscInt numcomps, const PetscInt *comps, void (*bcFunc)(void), PetscInt numids, const PetscInt *ids, void *ctx)
{
  PetscDS        ds;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscDSAddBoundary(ds, type,name, labelname, field, numcomps, comps, bcFunc, numids, ids, ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMGetNumBoundary - Get the number of registered BC

  Input Parameters:
. dm - The mesh object

  Output Parameters:
. numBd - The number of BC

  Level: intermediate

.seealso: DMAddBoundary(), DMGetBoundary()
@*/
PetscErrorCode DMGetNumBoundary(DM dm, PetscInt *numBd)
{
  PetscDS        ds;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscDSGetNumBoundary(ds, numBd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMGetBoundary - Get a model boundary condition

  Input Parameters:
+ dm          - The mesh object
- bd          - The BC number

  Output Parameters:
+ type        - The type of condition, e.g. DM_BC_ESSENTIAL_ANALYTIC/DM_BC_ESSENTIAL_FIELD (Dirichlet), or DM_BC_NATURAL (Neumann)
. name        - The BC name
. labelname   - The label defining constrained points
. field       - The field to constrain
. numcomps    - The number of constrained field components
. comps       - An array of constrained component numbers
. bcFunc      - A pointwise function giving boundary values
. numids      - The number of DMLabel ids for constrained points
. ids         - An array of ids for constrained points
- ctx         - An optional user context for bcFunc

  Options Database Keys:
+ -bc_<boundary name> <num> - Overrides the boundary ids
- -bc_<boundary name>_comp <num> - Overrides the boundary components

  Level: developer

.seealso: DMAddBoundary()
@*/
PetscErrorCode DMGetBoundary(DM dm, PetscInt bd, DMBoundaryConditionType *type, const char **name, const char **labelname, PetscInt *field, PetscInt *numcomps, const PetscInt **comps, void (**func)(void), PetscInt *numids, const PetscInt **ids, void **ctx)
{
  PetscDS        ds;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscDSGetBoundary(ds, bd, type, name, labelname, field, numcomps, comps, func, numids, ids, ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPopulateBoundary(DM dm)
{
  PetscDS        ds;
  DMBoundary    *lastnext;
  DSBoundary     dsbound;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  dsbound = ds->boundary;
  if (dm->boundary) {
    DMBoundary next = dm->boundary;

    /* quick check to see if the PetscDS has changed */
    if (next->dsboundary == dsbound) PetscFunctionReturn(0);
    /* the PetscDS has changed: tear down and rebuild */
    while (next) {
      DMBoundary b = next;

      next = b->next;
      ierr = PetscFree(b);CHKERRQ(ierr);
    }
    dm->boundary = NULL;
  }

  lastnext = &(dm->boundary);
  while (dsbound) {
    DMBoundary dmbound;

    ierr = PetscNew(&dmbound);CHKERRQ(ierr);
    dmbound->dsboundary = dsbound;
    ierr = DMGetLabel(dm, dsbound->labelname, &(dmbound->label));CHKERRQ(ierr);
    if (!dmbound->label) {ierr = PetscInfo2(dm, "DSBoundary %s wants label %s, which is not in this dm.\n",dsbound->name,dsbound->labelname);CHKERRQ(ierr);}
    /* push on the back instead of the front so that it is in the same order as in the PetscDS */
    *lastnext = dmbound;
    lastnext = &(dmbound->next);
    dsbound = dsbound->next;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMIsBoundaryPoint(DM dm, PetscInt point, PetscBool *isBd)
{
  DMBoundary     b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidBoolPointer(isBd, 3);
  *isBd = PETSC_FALSE;
  ierr = DMPopulateBoundary(dm);CHKERRQ(ierr);
  b = dm->boundary;
  while (b && !(*isBd)) {
    DMLabel    label = b->label;
    DSBoundary dsb = b->dsboundary;

    if (label) {
      PetscInt i;

      for (i = 0; i < dsb->numids && !(*isBd); ++i) {
        ierr = DMLabelStratumHasPoint(label, dsb->ids[i], point, isBd);CHKERRQ(ierr);
      }
    }
    b = b->next;
  }
  PetscFunctionReturn(0);
}

/*@C
  DMProjectFunction - This projects the given function into the function space provided, putting the coefficients in a global vector.

  Collective on DM

  Input Parameters:
+ dm      - The DM
. time    - The time
. funcs   - The coordinate functions to evaluate, one per field
. ctxs    - Optional array of contexts to pass to each coordinate function.  ctxs itself may be null.
- mode    - The insertion mode for values

  Output Parameter:
. X - vector

   Calling sequence of func:
$    func(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx);

+  dim - The spatial dimension
.  x   - The coordinates
.  Nf  - The number of fields
.  u   - The output field values
-  ctx - optional user-defined function context

  Level: developer

.seealso: DMProjectFunctionLocal(), DMProjectFunctionLabel(), DMComputeL2Diff()
@*/
PetscErrorCode DMProjectFunction(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, InsertMode mode, Vec X)
{
  Vec            localX;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(dm, time, funcs, ctxs, mode, localX);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, localX, mode, X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, localX, mode, X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMProjectFunctionLocal - This projects the given function into the function space provided, putting the coefficients in a local vector.

  Not collective

  Input Parameters:
+ dm      - The DM
. time    - The time
. funcs   - The coordinate functions to evaluate, one per field
. ctxs    - Optional array of contexts to pass to each coordinate function.  ctxs itself may be null.
- mode    - The insertion mode for values

  Output Parameter:
. localX - vector

   Calling sequence of func:
$    func(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx);

+  dim - The spatial dimension
.  x   - The coordinates
.  Nf  - The number of fields
.  u   - The output field values
-  ctx - optional user-defined function context

  Level: developer

.seealso: DMProjectFunction(), DMProjectFunctionLabel(), DMComputeL2Diff()
@*/
PetscErrorCode DMProjectFunctionLocal(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, InsertMode mode, Vec localX)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(localX,VEC_CLASSID,5);
  if (!dm->ops->projectfunctionlocal) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMProjectFunctionLocal",((PetscObject)dm)->type_name);
  ierr = (dm->ops->projectfunctionlocal) (dm, time, funcs, ctxs, mode, localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMProjectFunctionLabel - This projects the given function into the function space provided, putting the coefficients in a global vector, setting values only for points in the given label.

  Collective on DM

  Input Parameters:
+ dm      - The DM
. time    - The time
. label   - The DMLabel selecting the portion of the mesh for projection
. funcs   - The coordinate functions to evaluate, one per field
. ctxs    - Optional array of contexts to pass to each coordinate function.  ctxs itself may be null.
- mode    - The insertion mode for values

  Output Parameter:
. X - vector

   Calling sequence of func:
$    func(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx);

+  dim - The spatial dimension
.  x   - The coordinates
.  Nf  - The number of fields
.  u   - The output field values
-  ctx - optional user-defined function context

  Level: developer

.seealso: DMProjectFunction(), DMProjectFunctionLocal(), DMProjectFunctionLabelLocal(), DMComputeL2Diff()
@*/
PetscErrorCode DMProjectFunctionLabel(DM dm, PetscReal time, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscInt Nc, const PetscInt comps[], PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, InsertMode mode, Vec X)
{
  Vec            localX;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMProjectFunctionLabelLocal(dm, time, label, numIds, ids, Nc, comps, funcs, ctxs, mode, localX);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, localX, mode, X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, localX, mode, X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMProjectFunctionLabelLocal - This projects the given function into the function space provided, putting the coefficients in a local vector, setting values only for points in the given label.

  Not collective

  Input Parameters:
+ dm      - The DM
. time    - The time
. label   - The DMLabel selecting the portion of the mesh for projection
. funcs   - The coordinate functions to evaluate, one per field
. ctxs    - Optional array of contexts to pass to each coordinate function.  ctxs itself may be null.
- mode    - The insertion mode for values

  Output Parameter:
. localX - vector

   Calling sequence of func:
$    func(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx);

+  dim - The spatial dimension
.  x   - The coordinates
.  Nf  - The number of fields
.  u   - The output field values
-  ctx - optional user-defined function context

  Level: developer

.seealso: DMProjectFunction(), DMProjectFunctionLocal(), DMProjectFunctionLabel(), DMComputeL2Diff()
@*/
PetscErrorCode DMProjectFunctionLabelLocal(DM dm, PetscReal time, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscInt Nc, const PetscInt comps[], PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, InsertMode mode, Vec localX)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(localX,VEC_CLASSID,5);
  if (!dm->ops->projectfunctionlabellocal) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMProjectFunctionLabelLocal",((PetscObject)dm)->type_name);
  ierr = (dm->ops->projectfunctionlabellocal) (dm, time, label, numIds, ids, Nc, comps, funcs, ctxs, mode, localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMProjectFieldLocal - This projects the given function of the input fields into the function space provided, putting the coefficients in a local vector.

  Not collective

  Input Parameters:
+ dm      - The DM
. time    - The time
. localU  - The input field vector
. funcs   - The functions to evaluate, one per field
- mode    - The insertion mode for values

  Output Parameter:
. localX  - The output vector

   Calling sequence of func:
$    func(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$         const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$         const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$         PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[]);

+  dim          - The spatial dimension
.  Nf           - The number of input fields
.  NfAux        - The number of input auxiliary fields
.  uOff         - The offset of each field in u[]
.  uOff_x       - The offset of each field in u_x[]
.  u            - The field values at this point in space
.  u_t          - The field time derivative at this point in space (or NULL)
.  u_x          - The field derivatives at this point in space
.  aOff         - The offset of each auxiliary field in u[]
.  aOff_x       - The offset of each auxiliary field in u_x[]
.  a            - The auxiliary field values at this point in space
.  a_t          - The auxiliary field time derivative at this point in space (or NULL)
.  a_x          - The auxiliary field derivatives at this point in space
.  t            - The current time
.  x            - The coordinates of this point
.  numConstants - The number of constants
.  constants    - The value of each constant
-  f            - The value of the function at this point in space

  Note: There are three different DMs that potentially interact in this function. The output DM, dm, specifies the layout of the values calculates by funcs.
  The input DM, attached to U, may be different. For example, you can input the solution over the full domain, but output over a piece of the boundary, or
  a subdomain. You can also output a different number of fields than the input, with different discretizations. Last the auxiliary DM, attached to the
  auxiliary field vector, which is attached to dm, can also be different. It can have a different topology, number of fields, and discretizations.

  Level: intermediate

.seealso: DMProjectField(), DMProjectFieldLabelLocal(), DMProjectFunction(), DMComputeL2Diff()
@*/
PetscErrorCode DMProjectFieldLocal(DM dm, PetscReal time, Vec localU,
                                   void (**funcs)(PetscInt, PetscInt, PetscInt,
                                                  const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                  const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                  PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                   InsertMode mode, Vec localX)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(localU,VEC_CLASSID,3);
  PetscValidHeaderSpecific(localX,VEC_CLASSID,6);
  if (!dm->ops->projectfieldlocal) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMProjectFieldLocal",((PetscObject)dm)->type_name);
  ierr = (dm->ops->projectfieldlocal) (dm, time, localU, funcs, mode, localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMProjectFieldLabelLocal - This projects the given function of the input fields into the function space provided, putting the coefficients in a local vector, calculating only over the portion of the domain specified by the label.

  Not collective

  Input Parameters:
+ dm      - The DM
. time    - The time
. label   - The DMLabel marking the portion of the domain to output
. numIds  - The number of label ids to use
. ids     - The label ids to use for marking
. Nc      - The number of components to set in the output, or PETSC_DETERMINE for all components
. comps   - The components to set in the output, or NULL for all components
. localU  - The input field vector
. funcs   - The functions to evaluate, one per field
- mode    - The insertion mode for values

  Output Parameter:
. localX  - The output vector

   Calling sequence of func:
$    func(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$         const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$         const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$         PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[]);

+  dim          - The spatial dimension
.  Nf           - The number of input fields
.  NfAux        - The number of input auxiliary fields
.  uOff         - The offset of each field in u[]
.  uOff_x       - The offset of each field in u_x[]
.  u            - The field values at this point in space
.  u_t          - The field time derivative at this point in space (or NULL)
.  u_x          - The field derivatives at this point in space
.  aOff         - The offset of each auxiliary field in u[]
.  aOff_x       - The offset of each auxiliary field in u_x[]
.  a            - The auxiliary field values at this point in space
.  a_t          - The auxiliary field time derivative at this point in space (or NULL)
.  a_x          - The auxiliary field derivatives at this point in space
.  t            - The current time
.  x            - The coordinates of this point
.  numConstants - The number of constants
.  constants    - The value of each constant
-  f            - The value of the function at this point in space

  Note: There are three different DMs that potentially interact in this function. The output DM, dm, specifies the layout of the values calculates by funcs.
  The input DM, attached to U, may be different. For example, you can input the solution over the full domain, but output over a piece of the boundary, or
  a subdomain. You can also output a different number of fields than the input, with different discretizations. Last the auxiliary DM, attached to the
  auxiliary field vector, which is attached to dm, can also be different. It can have a different topology, number of fields, and discretizations.

  Level: intermediate

.seealso: DMProjectField(), DMProjectFieldLabelLocal(), DMProjectFunction(), DMComputeL2Diff()
@*/
PetscErrorCode DMProjectFieldLabelLocal(DM dm, PetscReal time, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscInt Nc, const PetscInt comps[], Vec localU,
                                        void (**funcs)(PetscInt, PetscInt, PetscInt,
                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                       PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        InsertMode mode, Vec localX)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(localU,VEC_CLASSID,6);
  PetscValidHeaderSpecific(localX,VEC_CLASSID,9);
  if (!dm->ops->projectfieldlabellocal) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMProjectFieldLocal",((PetscObject)dm)->type_name);
  ierr = (dm->ops->projectfieldlabellocal)(dm, time, label, numIds, ids, Nc, comps, localU, funcs, mode, localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMComputeL2Diff - This function computes the L_2 difference between a function u and an FEM interpolant solution u_h.

  Input Parameters:
+ dm    - The DM
. time  - The time
. funcs - The functions to evaluate for each field component
. ctxs  - Optional array of contexts to pass to each function, or NULL.
- X     - The coefficient vector u_h, a global vector

  Output Parameter:
. diff - The diff ||u - u_h||_2

  Level: developer

.seealso: DMProjectFunction(), DMComputeL2FieldDiff(), DMComputeL2GradientDiff()
@*/
PetscErrorCode DMComputeL2Diff(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, Vec X, PetscReal *diff)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,5);
  if (!dm->ops->computel2diff) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMComputeL2Diff",((PetscObject)dm)->type_name);
  ierr = (dm->ops->computel2diff)(dm,time,funcs,ctxs,X,diff);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMComputeL2GradientDiff - This function computes the L_2 difference between the gradient of a function u and an FEM interpolant solution grad u_h.

  Collective on dm

  Input Parameters:
+ dm    - The DM
, time  - The time
. funcs - The gradient functions to evaluate for each field component
. ctxs  - Optional array of contexts to pass to each function, or NULL.
. X     - The coefficient vector u_h, a global vector
- n     - The vector to project along

  Output Parameter:
. diff - The diff ||(grad u - grad u_h) . n||_2

  Level: developer

.seealso: DMProjectFunction(), DMComputeL2Diff()
@*/
PetscErrorCode DMComputeL2GradientDiff(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], const PetscReal[], PetscInt, PetscScalar *, void *), void **ctxs, Vec X, const PetscReal n[], PetscReal *diff)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,5);
  if (!dm->ops->computel2gradientdiff) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMComputeL2GradientDiff",((PetscObject)dm)->type_name);
  ierr = (dm->ops->computel2gradientdiff)(dm,time,funcs,ctxs,X,n,diff);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMComputeL2FieldDiff - This function computes the L_2 difference between a function u and an FEM interpolant solution u_h, separated into field components.

  Collective on dm

  Input Parameters:
+ dm    - The DM
. time  - The time
. funcs - The functions to evaluate for each field component
. ctxs  - Optional array of contexts to pass to each function, or NULL.
- X     - The coefficient vector u_h, a global vector

  Output Parameter:
. diff - The array of differences, ||u^f - u^f_h||_2

  Level: developer

.seealso: DMProjectFunction(), DMComputeL2FieldDiff(), DMComputeL2GradientDiff()
@*/
PetscErrorCode DMComputeL2FieldDiff(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, Vec X, PetscReal diff[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,5);
  if (!dm->ops->computel2fielddiff) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMComputeL2FieldDiff",((PetscObject)dm)->type_name);
  ierr = (dm->ops->computel2fielddiff)(dm,time,funcs,ctxs,X,diff);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMAdaptLabel - Adapt a dm based on a label with values interpreted as coarsening and refining flags.  Specific implementations of DM maybe have
                 specialized flags, but all implementations should accept flag values DM_ADAPT_DETERMINE, DM_ADAPT_KEEP, DM_ADAPT_REFINE, and DM_ADAPT_COARSEN.

  Collective on dm

  Input parameters:
+ dm - the pre-adaptation DM object
- label - label with the flags

  Output parameters:
. dmAdapt - the adapted DM object: may be NULL if an adapted DM could not be produced.

  Level: intermediate

.seealso: DMAdaptMetric(), DMCoarsen(), DMRefine()
@*/
PetscErrorCode DMAdaptLabel(DM dm, DMLabel label, DM *dmAdapt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(label,2);
  PetscValidPointer(dmAdapt,3);
  *dmAdapt = NULL;
  if (!dm->ops->adaptlabel) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMAdaptLabel",((PetscObject)dm)->type_name);
  ierr = (dm->ops->adaptlabel)(dm, label, dmAdapt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMAdaptMetric - Generates a mesh adapted to the specified metric field using the pragmatic library.

  Input Parameters:
+ dm - The DM object
. metric - The metric to which the mesh is adapted, defined vertex-wise.
- bdLabel - Label for boundary tags, which will be preserved in the output mesh. bdLabel should be NULL if there is no such label, and should be different from "_boundary_".

  Output Parameter:
. dmAdapt  - Pointer to the DM object containing the adapted mesh

  Note: The label in the adapted mesh will be registered under the name of the input DMLabel object

  Level: advanced

.seealso: DMAdaptLabel(), DMCoarsen(), DMRefine()
@*/
PetscErrorCode DMAdaptMetric(DM dm, Vec metric, DMLabel bdLabel, DM *dmAdapt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(metric, VEC_CLASSID, 2);
  if (bdLabel) PetscValidPointer(bdLabel, 3);
  PetscValidPointer(dmAdapt, 4);
  *dmAdapt = NULL;
  if (!dm->ops->adaptmetric) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMAdaptMetric",((PetscObject)dm)->type_name);
  ierr = (dm->ops->adaptmetric)(dm, metric, bdLabel, dmAdapt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
 DMGetNeighbors - Gets an array containing the MPI rank of all the processes neighbors

 Not Collective

 Input Parameter:
 . dm    - The DM

 Output Parameter:
 . nranks - the number of neighbours
 . ranks - the neighbors ranks

 Notes:
 Do not free the array, it is freed when the DM is destroyed.

 Level: beginner

 .seealso: DMDAGetNeighbors(), PetscSFGetRootRanks()
@*/
PetscErrorCode DMGetNeighbors(DM dm,PetscInt *nranks,const PetscMPIInt *ranks[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (!dm->ops->getneighbors) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMGetNeighbors",((PetscObject)dm)->type_name);
  ierr = (dm->ops->getneighbors)(dm,nranks,ranks);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petsc/private/matimpl.h> /* Needed because of coloring->ctype below */

/*
    Converts the input vector to a ghosted vector and then calls the standard coloring code.
    This has be a different function because it requires DM which is not defined in the Mat library
*/
PetscErrorCode  MatFDColoringApply_AIJDM(Mat J,MatFDColoring coloring,Vec x1,void *sctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (coloring->ctype == IS_COLORING_LOCAL) {
    Vec x1local;
    DM  dm;
    ierr = MatGetDM(J,&dm);CHKERRQ(ierr);
    if (!dm) SETERRQ(PetscObjectComm((PetscObject)J),PETSC_ERR_ARG_INCOMP,"IS_COLORING_LOCAL requires a DM");
    ierr = DMGetLocalVector(dm,&x1local);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,x1,INSERT_VALUES,x1local);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,x1,INSERT_VALUES,x1local);CHKERRQ(ierr);
    x1   = x1local;
  }
  ierr = MatFDColoringApply_AIJ(J,coloring,x1,sctx);CHKERRQ(ierr);
  if (coloring->ctype == IS_COLORING_LOCAL) {
    DM  dm;
    ierr = MatGetDM(J,&dm);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&x1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
    MatFDColoringUseDM - allows a MatFDColoring object to use the DM associated with the matrix to use a IS_COLORING_LOCAL coloring

    Input Parameter:
.    coloring - the MatFDColoring object

    Developer Notes:
    this routine exists because the PETSc Mat library does not know about the DM objects

    Level: advanced

.seealso: MatFDColoring, MatFDColoringCreate(), ISColoringType
@*/
PetscErrorCode  MatFDColoringUseDM(Mat coloring,MatFDColoring fdcoloring)
{
  PetscFunctionBegin;
  coloring->ops->fdcoloringapply = MatFDColoringApply_AIJDM;
  PetscFunctionReturn(0);
}

/*@
    DMGetCompatibility - determine if two DMs are compatible

    Collective

    Input Parameters:
+    dm - the first DM
-    dm2 - the second DM

    Output Parameters:
+    compatible - whether or not the two DMs are compatible
-    set - whether or not the compatible value was set

    Notes:
    Two DMs are deemed compatible if they represent the same parallel decomposition
    of the same topology. This implies that the section (field data) on one
    "makes sense" with respect to the topology and parallel decomposition of the other.
    Loosely speaking, compatible DMs represent the same domain and parallel
    decomposition, but hold different data.

    Typically, one would confirm compatibility if intending to simultaneously iterate
    over a pair of vectors obtained from different DMs.

    For example, two DMDA objects are compatible if they have the same local
    and global sizes and the same stencil width. They can have different numbers
    of degrees of freedom per node. Thus, one could use the node numbering from
    either DM in bounds for a loop over vectors derived from either DM.

    Consider the operation of summing data living on a 2-dof DMDA to data living
    on a 1-dof DMDA, which should be compatible, as in the following snippet.
.vb
  ...
  ierr = DMGetCompatibility(da1,da2,&compatible,&set);CHKERRQ(ierr);
  if (set && compatible)  {
    ierr = DMDAVecGetArrayDOF(da1,vec1,&arr1);CHKERRQ(ierr);
    ierr = DMDAVecGetArrayDOF(da2,vec2,&arr2);CHKERRQ(ierr);
    ierr = DMDAGetCorners(da1,&x,&y,NULL,&m,&n,NULL);CHKERRQ(ierr);
    for (j=y; j<y+n; ++j) {
      for (i=x; i<x+m, ++i) {
        arr1[j][i][0] = arr2[j][i][0] + arr2[j][i][1];
      }
    }
    ierr = DMDAVecRestoreArrayDOF(da1,vec1,&arr1);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayDOF(da2,vec2,&arr2);CHKERRQ(ierr);
  } else {
    SETERRQ(PetscObjectComm((PetscObject)da1,PETSC_ERR_ARG_INCOMP,"DMDA objects incompatible");
  }
  ...
.ve

    Checking compatibility might be expensive for a given implementation of DM,
    or might be impossible to unambiguously confirm or deny. For this reason,
    this function may decline to determine compatibility, and hence users should
    always check the "set" output parameter.

    A DM is always compatible with itself.

    In the current implementation, DMs which live on "unequal" communicators
    (MPI_UNEQUAL in the terminology of MPI_Comm_compare()) are always deemed
    incompatible.

    This function is labeled "Collective," as information about all subdomains
    is required on each rank. However, in DM implementations which store all this
    information locally, this function may be merely "Logically Collective".

    Developer Notes:
    Compatibility is assumed to be a symmetric concept; DM A is compatible with DM B
    iff B is compatible with A. Thus, this function checks the implementations
    of both dm and dm2 (if they are of different types), attempting to determine
    compatibility. It is left to DM implementers to ensure that symmetry is
    preserved. The simplest way to do this is, when implementing type-specific
    logic for this function, is to check for existing logic in the implementation
    of other DM types and let *set = PETSC_FALSE if found.

    Level: advanced

.seealso: DM, DMDACreateCompatibleDMDA(), DMStagCreateCompatibleDMStag()
@*/

PetscErrorCode DMGetCompatibility(DM dm,DM dm2,PetscBool *compatible,PetscBool *set)
{
  PetscErrorCode ierr;
  PetscMPIInt    compareResult;
  DMType         type,type2;
  PetscBool      sameType;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(dm2,DM_CLASSID,2);

  /* Declare a DM compatible with itself */
  if (dm == dm2) {
    *set = PETSC_TRUE;
    *compatible = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  /* Declare a DM incompatible with a DM that lives on an "unequal"
     communicator. Note that this does not preclude compatibility with
     DMs living on "congruent" or "similar" communicators, but this must be
     determined by the implementation-specific logic */
  ierr = MPI_Comm_compare(PetscObjectComm((PetscObject)dm),PetscObjectComm((PetscObject)dm2),&compareResult);CHKERRQ(ierr);
  if (compareResult == MPI_UNEQUAL) {
    *set = PETSC_TRUE;
    *compatible = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  /* Pass to the implementation-specific routine, if one exists. */
  if (dm->ops->getcompatibility) {
    ierr = (*dm->ops->getcompatibility)(dm,dm2,compatible,set);CHKERRQ(ierr);
    if (*set) PetscFunctionReturn(0);
  }

  /* If dm and dm2 are of different types, then attempt to check compatibility
     with an implementation of this function from dm2 */
  ierr = DMGetType(dm,&type);CHKERRQ(ierr);
  ierr = DMGetType(dm2,&type2);CHKERRQ(ierr);
  ierr = PetscStrcmp(type,type2,&sameType);CHKERRQ(ierr);
  if (!sameType && dm2->ops->getcompatibility) {
    ierr = (*dm2->ops->getcompatibility)(dm2,dm,compatible,set);CHKERRQ(ierr); /* Note argument order */
  } else {
    *set = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*@C
  DMMonitorSet - Sets an ADDITIONAL function that is to be used after a solve to monitor discretization performance.

  Logically Collective on DM

  Input Parameters:
+ DM - the DM
. f - the monitor function
. mctx - [optional] user-defined context for private data for the monitor routine (use NULL if no context is desired)
- monitordestroy - [optional] routine that frees monitor context (may be NULL)

  Options Database Keys:
- -dm_monitor_cancel - cancels all monitors that have been hardwired into a code by calls to DMMonitorSet(), but
                            does not cancel those set via the options database.

  Notes:
  Several different monitoring routines may be set by calling
  DMMonitorSet() multiple times; all will be called in the
  order in which they were set.

  Fortran Notes:
  Only a single monitor function can be set for each DM object

  Level: intermediate

.seealso: DMMonitorCancel()
@*/
PetscErrorCode DMMonitorSet(DM dm, PetscErrorCode (*f)(DM, void *), void *mctx, PetscErrorCode (*monitordestroy)(void**))
{
  PetscInt       m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  for (m = 0; m < dm->numbermonitors; ++m) {
    PetscBool identical;

    ierr = PetscMonitorCompare((PetscErrorCode (*)(void)) f, mctx, monitordestroy, (PetscErrorCode (*)(void)) dm->monitor[m], dm->monitorcontext[m], dm->monitordestroy[m], &identical);CHKERRQ(ierr);
    if (identical) PetscFunctionReturn(0);
  }
  if (dm->numbermonitors >= MAXDMMONITORS) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Too many monitors set");
  dm->monitor[dm->numbermonitors]          = f;
  dm->monitordestroy[dm->numbermonitors]   = monitordestroy;
  dm->monitorcontext[dm->numbermonitors++] = (void *) mctx;
  PetscFunctionReturn(0);
}

/*@
  DMMonitorCancel - Clears all the monitor functions for a DM object.

  Logically Collective on DM

  Input Parameter:
. dm - the DM

  Options Database Key:
. -dm_monitor_cancel - cancels all monitors that have been hardwired
  into a code by calls to DMonitorSet(), but does not cancel those
  set via the options database

  Notes:
  There is no way to clear one specific monitor from a DM object.

  Level: intermediate

.seealso: DMMonitorSet()
@*/
PetscErrorCode DMMonitorCancel(DM dm)
{
  PetscErrorCode ierr;
  PetscInt       m;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  for (m = 0; m < dm->numbermonitors; ++m) {
    if (dm->monitordestroy[m]) {ierr = (*dm->monitordestroy[m])(&dm->monitorcontext[m]);CHKERRQ(ierr);}
  }
  dm->numbermonitors = 0;
  PetscFunctionReturn(0);
}

/*@C
  DMMonitorSetFromOptions - Sets a monitor function and viewer appropriate for the type indicated by the user

  Collective on DM

  Input Parameters:
+ dm   - DM object you wish to monitor
. name - the monitor type one is seeking
. help - message indicating what monitoring is done
. manual - manual page for the monitor
. monitor - the monitor function
- monitorsetup - a function that is called once ONLY if the user selected this monitor that may set additional features of the DM or PetscViewer objects

  Output Parameter:
. flg - Flag set if the monitor was created

  Level: developer

.seealso: PetscOptionsGetViewer(), PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode DMMonitorSetFromOptions(DM dm, const char name[], const char help[], const char manual[], PetscErrorCode (*monitor)(DM, void *), PetscErrorCode (*monitorsetup)(DM, PetscViewerAndFormat *), PetscBool *flg)
{
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject) dm), ((PetscObject) dm)->options, ((PetscObject) dm)->prefix, name, &viewer, &format, flg);CHKERRQ(ierr);
  if (*flg) {
    PetscViewerAndFormat *vf;

    ierr = PetscViewerAndFormatCreate(viewer, format, &vf);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject) viewer);CHKERRQ(ierr);
    if (monitorsetup) {ierr = (*monitorsetup)(dm, vf);CHKERRQ(ierr);}
    ierr = DMMonitorSet(dm,(PetscErrorCode (*)(DM, void *)) monitor, vf, (PetscErrorCode (*)(void **)) PetscViewerAndFormatDestroy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   DMMonitor - runs the user provided monitor routines, if they exist

   Collective on DM

   Input Parameters:
.  dm - The DM

   Level: developer

.seealso: DMMonitorSet()
@*/
PetscErrorCode DMMonitor(DM dm)
{
  PetscInt       m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!dm) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  for (m = 0; m < dm->numbermonitors; ++m) {
    ierr = (*dm->monitor[m])(dm, dm->monitorcontext[m]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
