#include <petscvec.h>
#include <petsc/private/dmimpl.h>           /*I      "petscdm.h"          I*/
#include <petsc/private/dmlabelimpl.h>      /*I      "petscdmlabel.h"     I*/
#include <petsc/private/petscdsimpl.h>      /*I      "petscds.h"     I*/
#include <petscdmplex.h>
#include <petscdmfield.h>
#include <petscsf.h>
#include <petscds.h>

#ifdef PETSC_HAVE_LIBCEED
#include <petscfeceed.h>
#endif

#if defined(PETSC_HAVE_VALGRIND)
#  include <valgrind/memcheck.h>
#endif

PetscClassId  DM_CLASSID;
PetscClassId  DMLABEL_CLASSID;
PetscLogEvent DM_Convert, DM_GlobalToLocal, DM_LocalToGlobal, DM_LocalToLocal, DM_LocatePoints, DM_Coarsen, DM_Refine, DM_CreateInterpolation, DM_CreateRestriction, DM_CreateInjection, DM_CreateMatrix, DM_Load, DM_AdaptInterpolator;

const char *const DMBoundaryTypes[] = {"NONE","GHOSTED","MIRROR","PERIODIC","TWIST","DMBoundaryType","DM_BOUNDARY_", NULL};
const char *const DMBoundaryConditionTypes[] = {"INVALID","ESSENTIAL","NATURAL","INVALID","INVALID","ESSENTIAL_FIELD","NATURAL_FIELD","INVALID","INVALID","ESSENTIAL_BD_FIELD","NATURAL_RIEMANN","DMBoundaryConditionType","DM_BC_", NULL};
const char *const DMPolytopeTypes[] = {"vertex", "segment", "tensor_segment", "triangle", "quadrilateral", "tensor_quad", "tetrahedron", "hexahedron", "triangular_prism", "tensor_triangular_prism", "tensor_quadrilateral_prism", "pyramid", "FV_ghost_cell", "interior_ghost_cell", "unknown", "invalid", "DMPolytopeType", "DM_POLYTOPE_", NULL};
const char *const DMCopyLabelsModes[] = {"replace","keep","fail","DMCopyLabelsMode","DM_COPY_LABELS_", NULL};

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
  v->bind_below               = 0;
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
  ierr = PetscDSCreate(PETSC_COMM_SELF, &ds);CHKERRQ(ierr);
  ierr = DMSetRegionDS(v, NULL, NULL, ds);CHKERRQ(ierr);
  ierr = PetscDSDestroy(&ds);CHKERRQ(ierr);
  ierr = PetscHMapAuxCreate(&v->auxData);CHKERRQ(ierr);
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

  Notes:
  For some DM implementations this is a shallow clone, the result of which may share (referent counted) information with its parent. For example,
  DMClone() applied to a DMPLEX object will result in a new DMPLEX that shares the topology with the original DMPLEX. It does not
  share the PetscSection of the original DM.

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
  ierr = DMCopyLabels(dm, *newdm, PETSC_COPY_VALUES, PETSC_TRUE, DM_COPY_LABELS_FAIL);CHKERRQ(ierr);
  (*newdm)->leveldown  = dm->leveldown;
  (*newdm)->levelup    = dm->levelup;
  (*newdm)->prealloc_only = dm->prealloc_only;
  ierr = PetscFree((*newdm)->vectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(dm->vectype,(char**)&(*newdm)->vectype);CHKERRQ(ierr);
  ierr = PetscFree((*newdm)->mattype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(dm->mattype,(char**)&(*newdm)->mattype);CHKERRQ(ierr);
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
    ierr = MPI_Allreduce(&pEnd,&pEndMax,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)dm));CHKERRMPI(ierr);
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

   Input Parameters:
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
  PetscInt       refct = ((PetscObject) dm)->refct;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *ncrefct = 0;
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

/*@C
    DMDestroy - Destroys a vector packer or DM.

    Collective on dm

    Input Parameter:
.   dm - the DM object to destroy

    Level: developer

.seealso DMView(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix()

@*/
PetscErrorCode  DMDestroy(DM *dm)
{
  PetscInt       cnt;
  DMNamedVecLink nlink,nnext;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*dm) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*dm),DM_CLASSID,1);

  /* count all non-cyclic references in the doubly-linked list of coarse<->fine meshes */
  ierr = DMCountNonCyclicReferences(*dm,PETSC_TRUE,PETSC_TRUE,&cnt);CHKERRQ(ierr);
  --((PetscObject)(*dm))->refct;
  if (--cnt > 0) {*dm = NULL; PetscFunctionReturn(0);}
  if (((PetscObject)(*dm))->refct < 0) PetscFunctionReturn(0);
  ((PetscObject)(*dm))->refct = 0;

  ierr = DMClearGlobalVectors(*dm);CHKERRQ(ierr);
  ierr = DMClearLocalVectors(*dm);CHKERRQ(ierr);

  nnext=(*dm)->namedglobal;
  (*dm)->namedglobal = NULL;
  for (nlink=nnext; nlink; nlink=nnext) { /* Destroy the named vectors */
    nnext = nlink->next;
    PetscCheckFalse(nlink->status != DMVEC_STATUS_IN,((PetscObject)*dm)->comm,PETSC_ERR_ARG_WRONGSTATE,"DM still has Vec named '%s' checked out",nlink->name);
    ierr = PetscFree(nlink->name);CHKERRQ(ierr);
    ierr = VecDestroy(&nlink->X);CHKERRQ(ierr);
    ierr = PetscFree(nlink);CHKERRQ(ierr);
  }
  nnext=(*dm)->namedlocal;
  (*dm)->namedlocal = NULL;
  for (nlink=nnext; nlink; nlink=nnext) { /* Destroy the named local vectors */
    nnext = nlink->next;
    PetscCheckFalse(nlink->status != DMVEC_STATUS_IN,((PetscObject)*dm)->comm,PETSC_ERR_ARG_WRONGSTATE,"DM still has Vec named '%s' checked out",nlink->name);
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
    PetscCheckFalse((*dm)->workout,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Work array still checked out");
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
  ierr = MatFDColoringDestroy(&(*dm)->fd);CHKERRQ(ierr);
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
  {
    Vec     *auxData;
    PetscInt n, i, off = 0;

    ierr = PetscHMapAuxGetSize((*dm)->auxData, &n);CHKERRQ(ierr);
    ierr = PetscMalloc1(n, &auxData);CHKERRQ(ierr);
    ierr = PetscHMapAuxGetVals((*dm)->auxData, &off, auxData);CHKERRQ(ierr);
    for (i = 0; i < n; ++i) {ierr = VecDestroy(&auxData[i]);CHKERRQ(ierr);}
    ierr = PetscFree(auxData);CHKERRQ(ierr);
    ierr = PetscHMapAuxDestroy(&(*dm)->auxData);CHKERRQ(ierr);
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
  ierr = PetscFree((*dm)->L);CHKERRQ(ierr);
  ierr = PetscFree((*dm)->maxCell);CHKERRQ(ierr);
  ierr = PetscFree((*dm)->bdtype);CHKERRQ(ierr);
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
#ifdef PETSC_HAVE_LIBCEED
  ierr = CeedElemRestrictionDestroy(&(*dm)->ceedERestrict);CHKERRQ_CEED(ierr);
  ierr = CeedDestroy(&(*dm)->ceed);CHKERRQ_CEED(ierr);
#endif
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
.   -dm_is_coloring_type - <global or local>
-   -dm_bind_below <n>   - bind (force execution on CPU) for Vec and Mat objects with local size (number of vector entries or matrix rows) below n; currently only supported for DMDA

    DMPLEX Specific creation options
+ -dm_plex_filename <str>           - File containing a mesh
. -dm_plex_boundary_filename <str>  - File containing a mesh boundary
. -dm_plex_name <str>               - Name of the mesh in the file
. -dm_plex_shape <shape>            - The domain shape, such as DM_SHAPE_BOX, DM_SHAPE_SPHERE, etc.
. -dm_plex_cell <ct>                - Cell shape
. -dm_plex_reference_cell_domain <bool> - Use a reference cell domain
. -dm_plex_dim <dim>                - Set the topological dimension
. -dm_plex_simplex <bool>           - PETSC_TRUE for simplex elements, PETSC_FALSE for tensor elements
. -dm_plex_interpolate <bool>       - PETSC_TRUE turns on topological interpolation (creating edges and faces)
. -dm_plex_scale <sc>               - Scale factor for mesh coordinates
. -dm_plex_box_faces <m,n,p>        - Number of faces along each dimension
. -dm_plex_box_lower <x,y,z>        - Specify lower-left-bottom coordinates for the box
. -dm_plex_box_upper <x,y,z>        - Specify upper-right-top coordinates for the box
. -dm_plex_box_bd <bx,by,bz>        - Specify the DMBoundaryType for each direction
. -dm_plex_sphere_radius <r>        - The sphere radius
. -dm_plex_ball_radius <r>          - Radius of the ball
. -dm_plex_cylinder_bd <bz>         - Boundary type in the z direction
. -dm_plex_cylinder_num_wedges <n>  - Number of wedges around the cylinder
. -dm_plex_reorder <order>          - Reorder the mesh using the specified algorithm
. -dm_refine_pre <n>                - The number of refinements before distribution
. -dm_refine_uniform_pre <bool>     - Flag for uniform refinement before distribution
. -dm_refine_volume_limit_pre <v>   - The maximum cell volume after refinement before distribution
. -dm_refine <n>                    - The number of refinements after distribution
. -dm_extrude <l>                   - Activate extrusion and specify the number of layers to extrude
. -dm_plex_transform_extrude_thickness <t>           - The total thickness of extruded layers
. -dm_plex_transform_extrude_use_tensor <bool>       - Use tensor cells when extruding
. -dm_plex_transform_extrude_symmetric <bool>        - Extrude layers symmetrically about the surface
. -dm_plex_transform_extrude_normal <n0,...,nd>      - Specify the extrusion direction
. -dm_plex_transform_extrude_thicknesses <t0,...,tl> - Specify thickness of each layer
. -dm_plex_create_fv_ghost_cells    - Flag to create finite volume ghost cells on the boundary
. -dm_plex_fv_ghost_cells_label <name> - Label name for ghost cells boundary
. -dm_distribute <bool>             - Flag to redistribute a mesh among processes
. -dm_distribute_overlap <n>        - The size of the overlap halo
. -dm_plex_adj_cone <bool>          - Set adjacency direction
- -dm_plex_adj_closure <bool>       - Set adjacency size

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
  ierr = PetscOptionsInt("-dm_bind_below","Set the size threshold (in entries) below which the Vec is bound to the CPU","VecBindToCPU",dm->bind_below,&dm->bind_below,&flg);CHKERRQ(ierr);
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

    Input Parameters:
+   dm - the DM object to view
-   v - the viewer

    Notes:
    Using PETSCVIEWERHDF5 type with PETSC_VIEWER_HDF5_PETSC format, one can save multiple DMPlex
    meshes in a single HDF5 file. This in turn requires one to name the DMPlex object with PetscObjectSetName()
    before saving it with DMView() and before loading it with DMLoad() for identification of the mesh object.

    Level: beginner

.seealso DMDestroy(), DMCreateGlobalVector(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMLoad(), PetscObjectSetName()

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
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm),&size);CHKERRMPI(ierr);
  if (size == 1 && format == PETSC_VIEWER_LOAD_BALANCE) PetscFunctionReturn(0);
  ierr = PetscObjectPrintClassNamePrefixType((PetscObject)dm,v);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (isbinary) {
    PetscInt classid = DM_FILE_CLASSID;
    char     type[256];

    ierr = PetscViewerBinaryWrite(v,&classid,1,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscStrncpy(type,((PetscObject)dm)->type_name,256);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(v,type,256,PETSC_CHAR);CHKERRQ(ierr);
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

.seealso DMCreateLocalVector(), DMGetGlobalVector(), DMDestroy(), DMView(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix()

@*/
PetscErrorCode  DMCreateGlobalVector(DM dm,Vec *vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(vec,2);
  PetscCheckFalse(!dm->ops->createglobalvector,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMCreateGlobalVector",((PetscObject)dm)->type_name);
  ierr = (*dm->ops->createglobalvector)(dm,vec);CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {
    DM vdm;

    ierr = VecGetDM(*vec,&vdm);CHKERRQ(ierr);
    PetscCheckFalse(!vdm,PETSC_COMM_SELF,PETSC_ERR_PLIB,"DM type '%s' did not attach the DM to the vector",((PetscObject)dm)->type_name);
  }
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

.seealso DMCreateGlobalVector(), DMGetLocalVector(), DMDestroy(), DMView(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix()

@*/
PetscErrorCode  DMCreateLocalVector(DM dm,Vec *vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(vec,2);
  PetscCheckFalse(!dm->ops->createlocalvector,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMCreateLocalVector",((PetscObject)dm)->type_name);
  ierr = (*dm->ops->createlocalvector)(dm,vec);CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {
    DM vdm;

    ierr = VecGetDM(*vec,&vdm);CHKERRQ(ierr);
    PetscCheckFalse(!vdm,PETSC_COMM_SELF,PETSC_ERR_LIB,"DM type '%s' did not attach the DM to the vector",((PetscObject)dm)->type_name);
  }
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
        PetscInt bdof, cdof, dof, off, c, cind;

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
        for (c = 0, cind = 0; c < dof; ++c, ++l) {
          if ((cind < cdof) && (c == cdofs[cind])) {
            ltog[l] = off < 0 ? off-c : -(off+c+1);
            cind++;
          } else {
            ltog[l] = (off < 0 ? -(off+1) : off) + c;
          }
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
        for (l = 0, k = 0; l < n; l += bs, ++k) {
          // Integer division of negative values truncates toward zero(!), not toward negative infinity
          ltog[k] = ltog[l] >= 0 ? ltog[l]/bs : -(-(ltog[l]+1)/bs + 1);
        }
        n /= bs;
      }
      ierr = ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)dm), bs, n, ltog, PETSC_OWN_POINTER, &dm->ltogmap);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)dm, (PetscObject)dm->ltogmap);CHKERRQ(ierr);
    } else {
      PetscCheckFalse(!dm->ops->getlocaltoglobalmapping,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMGetLocalToGlobalMapping",((PetscObject)dm)->type_name);
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
  PetscCheckFalse(dm->bs < 1,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"DM does not have enough information to provide a block size yet");
  *bs = dm->bs;
  PetscFunctionReturn(0);
}

/*@C
    DMCreateInterpolation - Gets interpolation matrix between two DM objects

    Collective on dmc

    Input Parameters:
+   dmc - the DM object
-   dmf - the second, finer DM object

    Output Parameters:
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
PetscErrorCode  DMCreateInterpolation(DM dmc,DM dmf,Mat *mat,Vec *vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmc,DM_CLASSID,1);
  PetscValidHeaderSpecific(dmf,DM_CLASSID,2);
  PetscValidPointer(mat,3);
  PetscCheckFalse(!dmc->ops->createinterpolation,PetscObjectComm((PetscObject)dmc),PETSC_ERR_SUP,"DM type %s does not implement DMCreateInterpolation",((PetscObject)dmc)->type_name);
  ierr = PetscLogEventBegin(DM_CreateInterpolation,dmc,dmf,0,0);CHKERRQ(ierr);
  ierr = (*dmc->ops->createinterpolation)(dmc,dmf,mat,vec);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DM_CreateInterpolation,dmc,dmf,0,0);CHKERRQ(ierr);
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

  Developer Notes:
  If the fine-scale DMDA has the -dm_bind_below option set to true, then DMCreateInterpolationScale() calls MatSetBindingPropagates()
  on the restriction/interpolation operator to set the bindingpropagates flag to true.

.seealso: DMCreateInterpolation()

@*/
PetscErrorCode  DMCreateInterpolationScale(DM dac,DM daf,Mat mat,Vec *scale)
{
  PetscErrorCode ierr;
  Vec            fine;
  PetscScalar    one = 1.0;
#if defined(PETSC_HAVE_CUDA)
  PetscBool      bindingpropagates,isbound;
#endif

  PetscFunctionBegin;
  ierr = DMCreateGlobalVector(daf,&fine);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dac,scale);CHKERRQ(ierr);
  ierr = VecSet(fine,one);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  /* If the 'fine' Vec is bound to the CPU, it makes sense to bind 'mat' as well.
   * Note that we only do this for the CUDA case, right now, but if we add support for MatMultTranspose() via ViennaCL,
   * we'll need to do it for that case, too.*/
  ierr = VecGetBindingPropagates(fine,&bindingpropagates);CHKERRQ(ierr);
  if (bindingpropagates) {
    ierr = MatSetBindingPropagates(mat,PETSC_TRUE);CHKERRQ(ierr);
    ierr = VecBoundToCPU(fine,&isbound);CHKERRQ(ierr);
    ierr = MatBindToCPU(mat,isbound);CHKERRQ(ierr);
  }
#endif
  ierr = MatRestrict(mat,fine,*scale);CHKERRQ(ierr);
  ierr = VecDestroy(&fine);CHKERRQ(ierr);
  ierr = VecReciprocal(*scale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    DMCreateRestriction - Gets restriction matrix between two DM objects

    Collective on dmc

    Input Parameters:
+   dmc - the DM object
-   dmf - the second, finer DM object

    Output Parameter:
.  mat - the restriction

    Level: developer

    Notes:
    For DMDA objects this only works for "uniform refinement", that is the refined mesh was obtained DMRefine() or the coarse mesh was obtained by
        DMCoarsen(). The coordinates set into the DMDA are completely ignored in computing the interpolation.

.seealso DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateColoring(), DMCreateMatrix(), DMRefine(), DMCoarsen(), DMCreateInterpolation()

@*/
PetscErrorCode  DMCreateRestriction(DM dmc,DM dmf,Mat *mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmc,DM_CLASSID,1);
  PetscValidHeaderSpecific(dmf,DM_CLASSID,2);
  PetscValidPointer(mat,3);
  PetscCheckFalse(!dmc->ops->createrestriction,PetscObjectComm((PetscObject)dmc),PETSC_ERR_SUP,"DM type %s does not implement DMCreateRestriction",((PetscObject)dmc)->type_name);
  ierr = PetscLogEventBegin(DM_CreateRestriction,dmc,dmf,0,0);CHKERRQ(ierr);
  ierr = (*dmc->ops->createrestriction)(dmc,dmf,mat);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DM_CreateRestriction,dmc,dmf,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    DMCreateInjection - Gets injection matrix between two DM objects

    Collective on dac

    Input Parameters:
+   dac - the DM object
-   daf - the second, finer DM object

    Output Parameter:
.   mat - the injection

    Level: developer

   Notes:
    For DMDA objects this only works for "uniform refinement", that is the refined mesh was obtained DMRefine() or the coarse mesh was obtained by
        DMCoarsen(). The coordinates set into the DMDA are completely ignored in computing the injection.

.seealso DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateColoring(), DMCreateMatrix(), DMCreateInterpolation()

@*/
PetscErrorCode  DMCreateInjection(DM dac,DM daf,Mat *mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dac,DM_CLASSID,1);
  PetscValidHeaderSpecific(daf,DM_CLASSID,2);
  PetscValidPointer(mat,3);
  PetscCheckFalse(!dac->ops->createinjection,PetscObjectComm((PetscObject)dac),PETSC_ERR_SUP,"DM type %s does not implement DMCreateInjection",((PetscObject)dac)->type_name);
  ierr = PetscLogEventBegin(DM_CreateInjection,dac,daf,0,0);CHKERRQ(ierr);
  ierr = (*dac->ops->createinjection)(dac,daf,mat);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DM_CreateInjection,dac,daf,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMCreateMassMatrix - Gets mass matrix between two DM objects, M_ij = \int \phi_i \psi_j

  Collective on dac

  Input Parameters:
+ dmc - the target DM object
- dmf - the source DM object

  Output Parameter:
. mat - the mass matrix

  Level: developer

.seealso DMCreateMassMatrixLumped(), DMCreateMatrix(), DMRefine(), DMCoarsen(), DMCreateRestriction(), DMCreateInterpolation(), DMCreateInjection()
@*/
PetscErrorCode DMCreateMassMatrix(DM dmc, DM dmf, Mat *mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmc, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmf, DM_CLASSID, 2);
  PetscValidPointer(mat,3);
  PetscCheck(dmc->ops->createmassmatrix,PetscObjectComm((PetscObject)dmc),PETSC_ERR_SUP,"DM type %s does not implement DMCreateMassMatrix",((PetscObject)dmc)->type_name);
  ierr = (*dmc->ops->createmassmatrix)(dmc, dmf, mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMCreateMassMatrixLumped - Gets the lumped mass matrix for a given DM

  Collective on dm

  Input Parameter:
. dm - the DM object

  Output Parameter:
. lm - the lumped mass matrix

  Level: developer

.seealso DMCreateMassMatrix(), DMCreateMatrix(), DMRefine(), DMCoarsen(), DMCreateRestriction(), DMCreateInterpolation(), DMCreateInjection()
@*/
PetscErrorCode DMCreateMassMatrixLumped(DM dm, Vec *lm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(lm,2);
  PetscCheck(dm->ops->createmassmatrixlumped,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMCreateMassMatrixLumped",((PetscObject)dm)->type_name);
  ierr = (*dm->ops->createmassmatrixlumped)(dm, lm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    DMCreateColoring - Gets coloring for a DM

    Collective on dm

    Input Parameters:
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
  PetscCheckFalse(!dm->ops->getcoloring,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMCreateColoring",((PetscObject)dm)->type_name);
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

    Options Database Keys:
. -dm_preallocate_only - Only preallocate the matrix for DMCreateMatrix(), but do not fill it with zeros

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
  PetscValidPointer(mat,2);
  PetscCheckFalse(!dm->ops->creatematrix,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMCreateMatrix",((PetscObject)dm)->type_name);
  ierr = MatInitializePackage();CHKERRQ(ierr);
  ierr = PetscLogEventBegin(DM_CreateMatrix,0,0,0,0);CHKERRQ(ierr);
  ierr = (*dm->ops->creatematrix)(dm,mat);CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {
    DM mdm;

    ierr = MatGetDM(*mat,&mdm);CHKERRQ(ierr);
    PetscCheckFalse(!mdm,PETSC_COMM_SELF,PETSC_ERR_PLIB,"DM type '%s' did not attach the DM to the matrix",((PetscObject)dm)->type_name);
  }
  /* Handle nullspace and near nullspace */
  if (dm->Nf) {
    MatNullSpace nullSpace;
    PetscInt     Nf, f;

    ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {
      if (dm->nullspaceConstructors[f]) {
        ierr = (*dm->nullspaceConstructors[f])(dm, f, f, &nullSpace);CHKERRQ(ierr);
        ierr = MatSetNullSpace(*mat, nullSpace);CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&nullSpace);CHKERRQ(ierr);
        break;
      }
    }
    for (f = 0; f < Nf; ++f) {
      if (dm->nearnullspaceConstructors[f]) {
        ierr = (*dm->nearnullspaceConstructors[f])(dm, f, f, &nullSpace);CHKERRQ(ierr);
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

  Input Parameters:
+ dm - the DM
- only - PETSC_TRUE if only want preallocation

  Level: developer

  Options Database Keys:
. -dm_preallocate_only - Only preallocate the matrix for DMCreateMatrix(), but do not fill it with zeros

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

  Input Parameters:
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
. count - The minimum size
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
  ierr = MPI_Type_size(dtype,&dsize);CHKERRMPI(ierr);
  if (((size_t)dsize*count) > link->bytes) {
    ierr        = PetscFree(link->mem);CHKERRQ(ierr);
    ierr        = PetscMalloc(dsize*count,&link->mem);CHKERRQ(ierr);
    link->bytes = dsize*count;
  }
  link->next   = dm->workout;
  dm->workout  = link;
#if defined(PETSC_HAVE_VALGRIND)
  VALGRIND_MAKE_MEM_NOACCESS((char*)link->mem + (size_t)dsize*count, link->bytes - (size_t)dsize*count);
  VALGRIND_MAKE_MEM_UNDEFINED(link->mem, (size_t)dsize*count);
#endif
  *(void**)mem = link->mem;
  PetscFunctionReturn(0);
}

/*@C
  DMRestoreWorkArray - Restores a work array guaranteed to be at least the input size, restore with DMRestoreWorkArray()

  Not Collective

  Input Parameters:
+ dm - the DM object
. count - The minimum size
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

/*@C
  DMSetNullSpaceConstructor - Provide a callback function which constructs the nullspace for a given field

  Logically collective on DM

  Input Parameters:
+ dm     - The DM
. field  - The field number for the nullspace
- nullsp - A callback to create the nullspace

  Notes:
  The callback is intended to provide nullspaces when function spaces are joined or split, such as in DMCreateSubDM(). The calling sequence is
$ PetscErrorCode nullsp(DM dm, PetscInt origField, PetscInt field, MatNullSpace *nullSpace)
$ dm        - The present DM
$ origField - The field number given above, in the original DM
$ field     - The field number in dm
$ nullSpace - The nullspace for the given field

  This function is currently not available from Fortran.

.seealso: DMGetNullSpaceConstructor(), DMSetNearNullSpaceConstructor(), DMGetNearNullSpaceConstructor(), DMCreateSubDM(), DMCreateSuperDM()
*/
PetscErrorCode DMSetNullSpaceConstructor(DM dm, PetscInt field, PetscErrorCode (*nullsp)(DM dm, PetscInt origField, PetscInt field, MatNullSpace *nullSpace))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCheckFalse(field >= 10,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle %d >= 10 fields", field);
  dm->nullspaceConstructors[field] = nullsp;
  PetscFunctionReturn(0);
}

/*@C
  DMGetNullSpaceConstructor - Return the callback function which constructs the nullspace for a given field, or NULL

  Not collective

  Input Parameters:
+ dm     - The DM
- field  - The field number for the nullspace

  Output Parameter:
. nullsp - A callback to create the nullspace

  Notes:
  The callback is intended to provide nullspaces when function spaces are joined or split, such as in DMCreateSubDM(). The calling sequence is
$ PetscErrorCode nullsp(DM dm, PetscInt origField, PetscInt field, MatNullSpace *nullSpace)
$ dm        - The present DM
$ origField - The field number given above, in the original DM
$ field     - The field number in dm
$ nullSpace - The nullspace for the given field

  This function is currently not available from Fortran.

.seealso: DMSetNullSpaceConstructor(), DMSetNearNullSpaceConstructor(), DMGetNearNullSpaceConstructor(), DMCreateSubDM(), DMCreateSuperDM()
*/
PetscErrorCode DMGetNullSpaceConstructor(DM dm, PetscInt field, PetscErrorCode (**nullsp)(DM dm, PetscInt origField, PetscInt field, MatNullSpace *nullSpace))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(nullsp, 3);
  PetscCheckFalse(field >= 10,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle %d >= 10 fields", field);
  *nullsp = dm->nullspaceConstructors[field];
  PetscFunctionReturn(0);
}

/*@C
  DMSetNearNullSpaceConstructor - Provide a callback function which constructs the near-nullspace for a given field

  Logically collective on DM

  Input Parameters:
+ dm     - The DM
. field  - The field number for the nullspace
- nullsp - A callback to create the near-nullspace

  Notes:
  The callback is intended to provide nullspaces when function spaces are joined or split, such as in DMCreateSubDM(). The calling sequence is
$ PetscErrorCode nullsp(DM dm, PetscInt origField, PetscInt field, MatNullSpace *nullSpace)
$ dm        - The present DM
$ origField - The field number given above, in the original DM
$ field     - The field number in dm
$ nullSpace - The nullspace for the given field

  This function is currently not available from Fortran.

.seealso: DMGetNearNullSpaceConstructor(), DMSetNullSpaceConstructor(), DMGetNullSpaceConstructor(), DMCreateSubDM(), DMCreateSuperDM()
*/
PetscErrorCode DMSetNearNullSpaceConstructor(DM dm, PetscInt field, PetscErrorCode (*nullsp)(DM dm, PetscInt origField, PetscInt field, MatNullSpace *nullSpace))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCheckFalse(field >= 10,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle %d >= 10 fields", field);
  dm->nearnullspaceConstructors[field] = nullsp;
  PetscFunctionReturn(0);
}

/*@C
  DMGetNearNullSpaceConstructor - Return the callback function which constructs the near-nullspace for a given field, or NULL

  Not collective

  Input Parameters:
+ dm     - The DM
- field  - The field number for the nullspace

  Output Parameter:
. nullsp - A callback to create the near-nullspace

  Notes:
  The callback is intended to provide nullspaces when function spaces are joined or split, such as in DMCreateSubDM(). The calling sequence is
$ PetscErrorCode nullsp(DM dm, PetscInt origField, PetscInt field, MatNullSpace *nullSpace)
$ dm        - The present DM
$ origField - The field number given above, in the original DM
$ field     - The field number in dm
$ nullSpace - The nullspace for the given field

  This function is currently not available from Fortran.

.seealso: DMSetNearNullSpaceConstructor(), DMSetNullSpaceConstructor(), DMGetNullSpaceConstructor(), DMCreateSubDM(), DMCreateSuperDM()
*/
PetscErrorCode DMGetNearNullSpaceConstructor(DM dm, PetscInt field, PetscErrorCode (**nullsp)(DM dm, PetscInt origField, PetscInt field, MatNullSpace *nullSpace))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(nullsp, 3);
  PetscCheckFalse(field >= 10,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle %d >= 10 fields", field);
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
        ierr  = MPIU_Allreduce(in, out, 2, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm));CHKERRMPI(ierr);
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
    *namelist = NULL;
  }
  if (islist) {
    PetscValidPointer(islist,4);
    *islist = NULL;
  }
  if (dmlist) {
    PetscValidPointer(dmlist,5);
    *dmlist = NULL;
  }
  /*
   Is it a good idea to apply the following check across all impls?
   Perhaps some impls can have a well-defined decomposition before DMSetUp?
   This, however, follows the general principle that accessors are not well-behaved until the object is set up.
   */
  PetscCheckFalse(!dm->setupcalled,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE, "Decomposition defined only after DMSetUp");
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
  PetscCheckFalse(!dm->ops->createsubdm,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMCreateSubDM",((PetscObject)dm)->type_name);
  ierr = (*dm->ops->createsubdm)(dm, numFields, fields, is, subdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMCreateSuperDM - Returns an arrays of ISes and DM encapsulating a superproblem defined by the DMs passed in.

  Not collective

  Input Parameters:
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
  PetscCheckFalse(len < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of DMs must be nonnegative: %D", len);
  if (len) {
    DM dm = dms[0];
    PetscCheckFalse(!dm->ops->createsuperdm,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMCreateSuperDM",((PetscObject)dm)->type_name);
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

.seealso DMDestroy(), DMView(), DMCreateInterpolation(), DMCreateColoring(), DMCreateMatrix(), DMCreateFieldDecomposition()
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
  PetscCheckFalse(!dm->setupcalled,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE, "Decomposition defined only after DMSetUp");
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
+ iscat - scatter from global vector to nonoverlapping global vector entries on subdomain
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
  PetscCheckFalse(!dm->ops->createddscatters,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMCreateDomainDecompositionScatters",((PetscObject)dm)->type_name);
  ierr = (*dm->ops->createddscatters)(dm,n,subdms,iscat,oscat,gscat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMRefine - Refines a DM object

  Collective on dm

  Input Parameters:
+ dm   - the DM object
- comm - the communicator to contain the new DM object (or MPI_COMM_NULL)

  Output Parameter:
. dmf - the refined DM, or NULL

  Options Database Keys:
. -dm_plex_cell_refiner <strategy> - chooses the refinement strategy, e.g. regular, tohex

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
  PetscCheckFalse(!dm->ops->refine,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMRefine",((PetscObject)dm)->type_name);
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

   Input Parameters:
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

   Input Parameters:
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

   Input Parameters:
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
   DMInterpolateSolution - Interpolates a solution from a coarse mesh to a fine mesh.

   Collective on DM

   Input Parameters:
+  coarse - coarse DM
.  fine   - fine DM
.  interp - (optional) the matrix computed by DMCreateInterpolation().  Implementations may not need this, but if it
            is available it can avoid some recomputation.  If it is provided, MatInterpolate() will be used if
            the coarse DM does not have a specialized implementation.
-  coarseSol - solution on the coarse mesh

   Output Parameter:
.  fineSol - the interpolation of coarseSol to the fine mesh

   Level: developer

   Note: This function exists because the interpolation of a solution vector between meshes is not always a linear
   map.  For example, if a boundary value problem has an inhomogeneous Dirichlet boundary condition that is compressed
   out of the solution vector.  Or if interpolation is inherently a nonlinear operation, such as a method using
   slope-limiting reconstruction.

.seealso DMInterpolate(), DMCreateInterpolation()
@*/
PetscErrorCode DMInterpolateSolution(DM coarse, DM fine, Mat interp, Vec coarseSol, Vec fineSol)
{
  PetscErrorCode (*interpsol)(DM,DM,Mat,Vec,Vec) = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarse,DM_CLASSID,1);
  if (interp) PetscValidHeaderSpecific(interp,MAT_CLASSID,3);
  PetscValidHeaderSpecific(coarseSol,VEC_CLASSID,4);
  PetscValidHeaderSpecific(fineSol,VEC_CLASSID,5);

  ierr = PetscObjectQueryFunction((PetscObject)coarse,"DMInterpolateSolution_C", &interpsol);CHKERRQ(ierr);
  if (interpsol) {
    ierr = (*interpsol)(coarse, fine, interp, coarseSol, fineSol);CHKERRQ(ierr);
  } else if (interp) {
    ierr = MatInterpolate(interp, coarseSol, fineSol);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)coarse), PETSC_ERR_SUP, "DM %s does not implement DMInterpolateSolution()", ((PetscObject)coarse)->type_name);
  PetscFunctionReturn(0);
}

/*@
    DMGetRefineLevel - Gets the number of refinements that have generated this DM.

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
    DMSetRefineLevel - Sets the number of refinements that have generated this DM.

    Not Collective

    Input Parameters:
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

/*@
  DMExtrude - Extrude a DM object from a surface

  Collective on dm

  Input Parameter:
+ dm     - the DM object
- layers - the number of extruded cell layers

  Output Parameter:
. dme - the extruded DM, or NULL

  Note: If no extrusion was done, the return value is NULL

  Level: developer

.seealso DMRefine(), DMCoarsen(), DMDestroy(), DMView(), DMCreateGlobalVector()
@*/
PetscErrorCode DMExtrude(DM dm, PetscInt layers, DM *dme)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCheckFalse(!dm->ops->extrude,PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "DM type %s does not implement DMExtrude", ((PetscObject) dm)->type_name);
  ierr = (*dm->ops->extrude)(dm, layers, dme);CHKERRQ(ierr);
  if (*dme) {
    (*dme)->ops->creatematrix = dm->ops->creatematrix;
    ierr = PetscObjectCopyFortranFunctionPointers((PetscObject) dm, (PetscObject) *dme);CHKERRQ(ierr);
    (*dme)->ctx = dm->ctx;
    ierr = DMSetMatType(*dme, dm->mattype);CHKERRQ(ierr);
  }
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

.seealso: DMPlexGlobalToLocalBasis(), DMPlexLocalToGlobalBasis(), DMPlexCreateBasisRotation()
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

   Input Parameters:
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
    PetscMemType      lmtype,gmtype;

    PetscCheckFalse(mode == ADD_VALUES,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %D", mode);
    ierr = VecGetArrayAndMemType(l, &lArray, &lmtype);CHKERRQ(ierr);
    ierr = VecGetArrayReadAndMemType(g, &gArray, &gmtype);CHKERRQ(ierr);
    ierr = PetscSFBcastWithMemTypeBegin(sf, MPIU_SCALAR, gmtype, gArray, lmtype, lArray, MPI_REPLACE);CHKERRQ(ierr);
    ierr = VecRestoreArrayAndMemType(l, &lArray);CHKERRQ(ierr);
    ierr = VecRestoreArrayReadAndMemType(g, &gArray);CHKERRQ(ierr);
  } else {
    PetscCheckFalse(!dm->ops->globaltolocalbegin,PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Missing DMGlobalToLocalBegin() for type %s",((PetscObject)dm)->type_name);
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
  PetscMemType            lmtype,gmtype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetSectionSF(dm, &sf);CHKERRQ(ierr);
  ierr = DMHasBasisTransform(dm, &transform);CHKERRQ(ierr);
  if (sf) {
    PetscCheckFalse(mode == ADD_VALUES,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %D", mode);

    ierr = VecGetArrayAndMemType(l, &lArray, &lmtype);CHKERRQ(ierr);
    ierr = VecGetArrayReadAndMemType(g, &gArray, &gmtype);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf, MPIU_SCALAR, gArray, lArray,MPI_REPLACE);CHKERRQ(ierr);
    ierr = VecRestoreArrayAndMemType(l, &lArray);CHKERRQ(ierr);
    ierr = VecRestoreArrayReadAndMemType(g, &gArray);CHKERRQ(ierr);
    if (transform) {ierr = DMPlexGlobalToLocalBasis(dm, l);CHKERRQ(ierr);}
  } else {
    PetscCheckFalse(!dm->ops->globaltolocalend,PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Missing DMGlobalToLocalEnd() for type %s",((PetscObject)dm)->type_name);
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

   Input Parameters:
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
  PetscBool               isInsert, transform, l_inplace = PETSC_FALSE, g_inplace = PETSC_FALSE;
  PetscErrorCode          ierr;
  PetscMemType            lmtype=PETSC_MEMTYPE_HOST,gmtype=PETSC_MEMTYPE_HOST;

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
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %D", mode);
  }
  if ((sf && !isInsert) || (s && isInsert)) {
    ierr = DMHasBasisTransform(dm, &transform);CHKERRQ(ierr);
    if (transform) {
      ierr = DMGetNamedLocalVector(dm, "__petsc_dm_transform_local_copy", &tmpl);CHKERRQ(ierr);
      ierr = VecCopy(l, tmpl);CHKERRQ(ierr);
      ierr = DMPlexLocalToGlobalBasis(dm, tmpl);CHKERRQ(ierr);
      ierr = VecGetArrayRead(tmpl, &lArray);CHKERRQ(ierr);
    } else if (isInsert) {
      ierr = VecGetArrayRead(l, &lArray);CHKERRQ(ierr);
    } else {
      ierr = VecGetArrayReadAndMemType(l, &lArray, &lmtype);CHKERRQ(ierr);
      l_inplace = PETSC_TRUE;
    }
    if (s && isInsert) {
      ierr = VecGetArray(g, &gArray);CHKERRQ(ierr);
    } else {
      ierr = VecGetArrayAndMemType(g, &gArray, &gmtype);CHKERRQ(ierr);
      g_inplace = PETSC_TRUE;
    }
    if (sf && !isInsert) {
      ierr = PetscSFReduceWithMemTypeBegin(sf, MPIU_SCALAR, lmtype, lArray, gmtype, gArray, MPIU_SUM);CHKERRQ(ierr);
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
        PetscCheckFalse(dof != gdof,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Inconsistent sizes, p: %d dof: %d gdof: %d cdof: %d gcdof: %d", p, dof, gdof, cdof, gcdof);
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
        } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Inconsistent sizes, p: %d dof: %d gdof: %d cdof: %d gcdof: %d", p, dof, gdof, cdof, gcdof);
      }
    }
    if (g_inplace) {
      ierr = VecRestoreArrayAndMemType(g, &gArray);CHKERRQ(ierr);
    } else {
      ierr = VecRestoreArray(g, &gArray);CHKERRQ(ierr);
    }
    if (transform) {
      ierr = VecRestoreArrayRead(tmpl, &lArray);CHKERRQ(ierr);
      ierr = DMRestoreNamedLocalVector(dm, "__petsc_dm_transform_local_copy", &tmpl);CHKERRQ(ierr);
    } else if (l_inplace) {
      ierr = VecRestoreArrayReadAndMemType(l, &lArray);CHKERRQ(ierr);
    } else {
      ierr = VecRestoreArrayRead(l, &lArray);CHKERRQ(ierr);
    }
  } else {
    PetscCheckFalse(!dm->ops->localtoglobalbegin,PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Missing DMLocalToGlobalBegin() for type %s",((PetscObject)dm)->type_name);
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
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %D", mode);
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
      ierr = VecGetArrayReadAndMemType(l, &lArray, NULL);CHKERRQ(ierr);
    }
    ierr = VecGetArrayAndMemType(g, &gArray, NULL);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(sf, MPIU_SCALAR, lArray, gArray, MPIU_SUM);CHKERRQ(ierr);
    if (transform) {
      ierr = VecRestoreArrayRead(tmpl, &lArray);CHKERRQ(ierr);
      ierr = DMRestoreNamedLocalVector(dm, "__petsc_dm_transform_local_copy", &tmpl);CHKERRQ(ierr);
    } else {
      ierr = VecRestoreArrayReadAndMemType(l, &lArray);CHKERRQ(ierr);
    }
    ierr = VecRestoreArrayAndMemType(g, &gArray);CHKERRQ(ierr);
  } else if (s && isInsert) {
  } else {
    PetscCheckFalse(!dm->ops->localtoglobalend,PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Missing DMLocalToGlobalEnd() for type %s",((PetscObject)dm)->type_name);
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
  PetscCheckFalse(!dm->ops->localtolocalbegin,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"This DM does not support local to local maps");
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
  PetscCheckFalse(!dm->ops->localtolocalend,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"This DM does not support local to local maps");
  ierr = (*dm->ops->localtolocalend)(dm,g,mode == INSERT_ALL_VALUES ? INSERT_VALUES : (mode == ADD_ALL_VALUES ? ADD_VALUES : mode),l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    DMCoarsen - Coarsens a DM object

    Collective on dm

    Input Parameters:
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
  PetscCheckFalse(!dm->ops->coarsen,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMCoarsen",((PetscObject)dm)->type_name);
  ierr = PetscLogEventBegin(DM_Coarsen,dm,0,0,0);CHKERRQ(ierr);
  ierr = (*dm->ops->coarsen)(dm, comm, dmc);CHKERRQ(ierr);
  if (*dmc) {
    (*dmc)->bind_below = dm->bind_below; /* Propagate this from parent DM; otherwise -dm_bind_below will be useless for multigrid cases. */
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
  PetscCheckFalse(!(*dmc),PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "NULL coarse mesh produced");
  PetscFunctionReturn(0);
}

/*@C
   DMCoarsenHookAdd - adds a callback to be run when restricting a nonlinear problem to the coarse grid

   Logically Collective

   Input Parameters:
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

   Input Parameters:
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

   Input Parameters:
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

   Input Parameters:
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

   Input Parameters:
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

   Input Parameters:
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

    Input Parameters:
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
  PetscCheckFalse(nlevels < 0,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"nlevels cannot be negative");
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

    Input Parameters:
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
  PetscCheckFalse(nlevels < 0,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"nlevels cannot be negative");
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

    Input Parameters:
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

    Input Parameter:
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
  PetscCheckFalse(!dm->ops->computevariablebounds,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMComputeVariableBounds",((PetscObject)dm)->type_name);
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
  PetscCheckFalse(!r,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown DM type: %s", method);

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
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "No conversion possible between DM types %s and %s", ((PetscObject) dm)->type_name, newtype);

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
      (*M)->prealloc_only = dm->prealloc_only;
      ierr = PetscFree((*M)->vectype);CHKERRQ(ierr);
      ierr = PetscStrallocpy(dm->vectype,(char**)&(*M)->vectype);CHKERRQ(ierr);
      ierr = PetscFree((*M)->mattype);CHKERRQ(ierr);
      ierr = PetscStrallocpy(dm->mattype,(char**)&(*M)->mattype);CHKERRQ(ierr);
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

  Using PETSCVIEWERHDF5 type with PETSC_VIEWER_HDF5_PETSC format, one can save multiple DMPlex
  meshes in a single HDF5 file. This in turn requires one to name the DMPlex object with PetscObjectSetName()
  before saving it with DMView() and before loading it with DMLoad() for identification of the mesh object.

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
    PetscCheckFalse(classid != DM_FILE_CLASSID,PetscObjectComm((PetscObject)newdm),PETSC_ERR_ARG_WRONG,"Not DM next in file, classid found %d",(int)classid);
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
  if (gmin) {ierr = MPIU_Allreduce(lmin, gmin, count, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject) dm));CHKERRMPI(ierr);}
  if (gmax) {ierr = MPIU_Allreduce(lmax, gmax, count, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject) dm));CHKERRMPI(ierr);}
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
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm),&size);CHKERRMPI(ierr);
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

    if (dm->setfromoptionscalled) {
      PetscObject       obj = (PetscObject) dm;
      PetscViewer       viewer;
      PetscViewerFormat format;
      PetscBool         flg;

      ierr = PetscOptionsGetViewer(PetscObjectComm(obj), obj->options, obj->prefix, "-dm_petscds_view", &viewer, &format, &flg);CHKERRQ(ierr);
      if (flg) {ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);}
      for (d = 0; d < dm->Nds; ++d) {
        ierr = PetscDSSetFromOptions(dm->probs[d].ds);CHKERRQ(ierr);
        if (flg) {ierr = PetscDSView(dm->probs[d].ds, viewer);CHKERRQ(ierr);}
      }
      if (flg) {
        ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
        ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      }
    }
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

  Output Parameters:
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
    ierr = MPI_Comm_compare(PETSC_COMM_SELF,PetscObjectComm((PetscObject)section),&result);CHKERRMPI(ierr);
    PetscCheckFalse(result != MPI_CONGRUENT && result != MPI_IDENT,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMECOMM,"constraint section must have local communicator");
  }
  if (mat) {
    PetscValidHeaderSpecific(mat,MAT_CLASSID,3);
    ierr = MPI_Comm_compare(PETSC_COMM_SELF,PetscObjectComm((PetscObject)mat),&result);CHKERRMPI(ierr);
    PetscCheckFalse(result != MPI_CONGRUENT && result != MPI_IDENT,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMECOMM,"constraint matrix must have local communicator");
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
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
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
  ierr = MPIU_Allreduce(&valid, &gvalid, 1, MPIU_BOOL, MPI_LAND, comm);CHKERRMPI(ierr);
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
    PetscCheckFalse(!s,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONGSTATE, "DM must have a default PetscSection in order to create a global PetscSection");
    PetscCheckFalse(!dm->sf,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM must have a point PetscSF in order to create a global PetscSection");
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSFSetGraphSection(dm->sectionSF, localSection, globalSection);CHKERRQ(ierr);
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

/*@
  DMGetNaturalSF - Get the PetscSF encoding the map back to the original mesh ordering

  Input Parameter:
. dm - The DM

  Output Parameter:
. sf - The PetscSF

  Level: intermediate

  Note: This gets a borrowed reference, so the user should not destroy this PetscSF.

.seealso: DMSetNaturalSF(), DMSetUseNatural(), DMGetUseNatural(), DMPlexCreateGlobalToNaturalSF(), DMPlexDistribute()
@*/
PetscErrorCode DMGetNaturalSF(DM dm, PetscSF *sf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(sf, 2);
  *sf = dm->sfNatural;
  PetscFunctionReturn(0);
}

/*@
  DMSetNaturalSF - Set the PetscSF encoding the map back to the original mesh ordering

  Input Parameters:
+ dm - The DM
- sf - The PetscSF

  Level: intermediate

.seealso: DMGetNaturalSF(), DMSetUseNatural(), DMGetUseNatural(), DMPlexCreateGlobalToNaturalSF(), DMPlexDistribute()
@*/
PetscErrorCode DMSetNaturalSF(DM dm, PetscSF sf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (sf) PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 2);
  ierr = PetscObjectReference((PetscObject) sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&dm->sfNatural);CHKERRQ(ierr);
  dm->sfNatural = sf;
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
  for (f = Nf; f < NfNew; ++f) {tmpr[f].disc = NULL; tmpr[f].label = NULL; tmpr[f].avoidTensor = PETSC_FALSE;}
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
  PetscValidPointer(field, 4);
  PetscCheckFalse((f < 0) || (f >= dm->Nf),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, dm->Nf);
  if (label) *label = dm->fields[f].label;
  if (field) *field = dm->fields[f].disc;
  PetscFunctionReturn(0);
}

/* Does not clear the DS */
PetscErrorCode DMSetField_Internal(DM dm, PetscInt f, DMLabel label, PetscObject field)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMFieldEnlarge_Static(dm, f+1);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&dm->fields[f].label);CHKERRQ(ierr);
  ierr = PetscObjectDestroy(&dm->fields[f].disc);CHKERRQ(ierr);
  dm->fields[f].label = label;
  dm->fields[f].disc  = field;
  ierr = PetscObjectReference((PetscObject) label);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) field);CHKERRQ(ierr);
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
  PetscCheckFalse(f < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be non-negative", f);
  ierr = DMSetField_Internal(dm, f, label, field);CHKERRQ(ierr);
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
  if (label) PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 2);
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
  DMSetFieldAvoidTensor - Set flag to avoid defining the field on tensor cells

  Logically collective on dm

  Input Parameters:
+ dm          - The DM
. f           - The field index
- avoidTensor - The flag to avoid defining the field on tensor cells

  Level: intermediate

.seealso: DMGetFieldAvoidTensor(), DMSetField(), DMGetField()
@*/
PetscErrorCode DMSetFieldAvoidTensor(DM dm, PetscInt f, PetscBool avoidTensor)
{
  PetscFunctionBegin;
  PetscCheckFalse((f < 0) || (f >= dm->Nf),PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Field %D is not in [0, %D)", f, dm->Nf);
  dm->fields[f].avoidTensor = avoidTensor;
  PetscFunctionReturn(0);
}

/*@
  DMGetFieldAvoidTensor - Get flag to avoid defining the field on tensor cells

  Logically collective on dm

  Input Parameters:
+ dm          - The DM
- f           - The field index

  Output Parameter:
. avoidTensor - The flag to avoid defining the field on tensor cells

  Level: intermediate

.seealso: DMSetFieldAvoidTensor(), DMSetField(), DMGetField()
@*/
PetscErrorCode DMGetFieldAvoidTensor(DM dm, PetscInt f, PetscBool *avoidTensor)
{
  PetscFunctionBegin;
  PetscCheckFalse((f < 0) || (f >= dm->Nf),PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Field %D is not in [0, %D)", f, dm->Nf);
  *avoidTensor = dm->fields[f].avoidTensor;
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

  Output Parameters:
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
    PetscCheckFalse(f >= Nf,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, Nf);
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
    PetscCheckFalse(f >= Nf,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %d must be in [0, %d)", f, Nf);
    dm->fields[f].adjacency[0] = useCone;
    dm->fields[f].adjacency[1] = useClosure;
  }
  PetscFunctionReturn(0);
}

/*@
  DMGetBasicAdjacency - Returns the flags for determining variable influence, using either the default or field 0 if it is defined

  Not collective

  Input Parameter:
. dm - The DM object

  Output Parameters:
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
  if (useCone)    PetscValidBoolPointer(useCone, 2);
  if (useClosure) PetscValidBoolPointer(useClosure, 3);
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

/* Complete labels that are being used for FEM BC */
static PetscErrorCode DMCompleteBoundaryLabel_Internal(DM dm, PetscDS ds, PetscInt field, PetscInt bdNum, DMLabel label)
{
  PetscObject    obj;
  PetscClassId   id;
  PetscInt       Nbd, bd;
  PetscBool      isFE      = PETSC_FALSE;
  PetscBool      duplicate = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetField(dm, field, NULL, &obj);CHKERRQ(ierr);
  ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
  if (id == PETSCFE_CLASSID) isFE = PETSC_TRUE;
  if (isFE && label) {
    /* Only want to modify label once */
    ierr = PetscDSGetNumBoundary(ds, &Nbd);CHKERRQ(ierr);
    for (bd = 0; bd < PetscMin(Nbd, bdNum); ++bd) {
      DMLabel l;

      ierr = PetscDSGetBoundary(ds, bd, NULL, NULL, NULL, &l, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
      duplicate = l == label ? PETSC_TRUE : PETSC_FALSE;
      if (duplicate) break;
    }
    if (!duplicate) {
      DM plex;

      ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
      if (plex) {ierr = DMPlexLabelComplete(plex, label);CHKERRQ(ierr);}
      ierr = DMDestroy(&plex);CHKERRQ(ierr);
    }
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

    ierr = PetscDSCreate(PETSC_COMM_SELF, &ds);CHKERRQ(ierr);
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
  PetscCheckFalse(point < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Mesh point cannot be negative: %D", point);
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

  Note:
  If a non-NULL label is given, but there is no PetscDS on that specific label,
  the PetscDS for the full domain (if present) is returned. Returns with
  fields=NULL and prob=NULL if there is no PetscDS for the full domain.

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
    if (dm->probs[s].label == label || !dm->probs[s].label) {
      if (fields) *fields = dm->probs[s].fields;
      if (ds)     *ds     = dm->probs[s].ds;
      if (dm->probs[s].label) PetscFunctionReturn(0);
    }
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

.seealso: DMGetRegionDS(), DMSetRegionNumDS(), DMGetDS(), DMGetCellDS()
@*/
PetscErrorCode DMSetRegionDS(DM dm, DMLabel label, IS fields, PetscDS ds)
{
  PetscInt       Nds = dm->Nds, s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (label) PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 2);
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 4);
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
  DMGetRegionNumDS - Get the PetscDS for a given mesh region, defined by the region number

  Not collective

  Input Parameters:
+ dm  - The DM
- num - The region number, in [0, Nds)

  Output Parameters:
+ label  - The region label, or NULL
. fields - The IS containing the DM field numbers for the fields in this DS, or NULL
- ds     - The PetscDS defined on the given region, or NULL

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
  PetscCheckFalse((num < 0) || (num >= Nds),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Region number %D is not in [0, %D)", num, Nds);
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
  DMSetRegionNumDS - Set the PetscDS for a given mesh region, defined by the region number

  Not collective

  Input Parameters:
+ dm     - The DM
. num    - The region number, in [0, Nds)
. label  - The region label, or NULL
. fields - The IS containing the DM field numbers for the fields in this DS, or NULL to prevent setting
- ds     - The PetscDS defined on the given region, or NULL to prevent setting

  Level: advanced

.seealso: DMGetRegionDS(), DMSetRegionDS(), DMGetDS(), DMGetCellDS()
@*/
PetscErrorCode DMSetRegionNumDS(DM dm, PetscInt num, DMLabel label, IS fields, PetscDS ds)
{
  PetscInt       Nds;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (label) {PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 3);}
  ierr = DMGetNumDS(dm, &Nds);CHKERRQ(ierr);
  PetscCheckFalse((num < 0) || (num >= Nds),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Region number %D is not in [0, %D)", num, Nds);
  ierr = PetscObjectReference((PetscObject) label);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&dm->probs[num].label);CHKERRQ(ierr);
  dm->probs[num].label = label;
  if (fields) {
    PetscValidHeaderSpecific(fields, IS_CLASSID, 4);
    ierr = PetscObjectReference((PetscObject) fields);CHKERRQ(ierr);
    ierr = ISDestroy(&dm->probs[num].fields);CHKERRQ(ierr);
    dm->probs[num].fields = fields;
  }
  if (ds) {
    PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 5);
    ierr = PetscObjectReference((PetscObject) ds);CHKERRQ(ierr);
    ierr = PetscDSDestroy(&dm->probs[num].ds);CHKERRQ(ierr);
    dm->probs[num].ds = ds;
  }
  PetscFunctionReturn(0);
}

/*@
  DMFindRegionNum - Find the region number for a given PetscDS, or -1 if it is not found.

  Not collective

  Input Parameters:
+ dm  - The DM
- ds  - The PetscDS defined on the given region

  Output Parameter:
. num - The region number, in [0, Nds), or -1 if not found

  Level: advanced

.seealso: DMGetRegionNumDS(), DMGetRegionDS(), DMSetRegionDS(), DMGetDS(), DMGetCellDS()
@*/
PetscErrorCode DMFindRegionNum(DM dm, PetscDS ds, PetscInt *num)
{
  PetscInt       Nds, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 2);
  PetscValidPointer(num, 3);
  ierr = DMGetNumDS(dm, &Nds);CHKERRQ(ierr);
  for (n = 0; n < Nds; ++n) if (ds == dm->probs[n].ds) break;
  if (n >= Nds) *num = -1;
  else          *num = n;
  PetscFunctionReturn(0);
}

/*@C
  DMCreateFEDefault - Create a PetscFE based on the celltype for the mesh

  Not collective

  Input Parameter:
+ dm     - The DM
. Nc     - The number of components for the field
. prefix - The options prefix for the output PetscFE, or NULL
- qorder - The quadrature order or PETSC_DETERMINE to use PetscSpace polynomial degree

  Output Parameter:
. fem - The PetscFE

  Note: This is a convenience method that just calls PetscFECreateByCell() underneath.

  Level: intermediate

.seealso: PetscFECreateByCell(), DMAddField(), DMCreateDS(), DMGetCellDS(), DMGetRegionDS()
@*/
PetscErrorCode DMCreateFEDefault(DM dm, PetscInt Nc, const char prefix[], PetscInt qorder, PetscFE *fem)
{
  DMPolytopeType ct;
  PetscInt       dim, cStart;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidLogicalCollectiveInt(dm, Nc, 2);
  if (prefix) PetscValidCharPointer(prefix, 3);
  PetscValidLogicalCollectiveInt(dm, qorder, 4);
  PetscValidPointer(fem, 5);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
  ierr = PetscFECreateByCell(PETSC_COMM_SELF, dim, Nc, ct, prefix, qorder, fem);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMCreateDS - Create the discrete systems for the DM based upon the fields added to the DM

  Collective on dm

  Input Parameter:
. dm - The DM

  Options Database Keys:
. -dm_petscds_view - View all the PetscDS objects in this DM

  Note: If the label has a DS defined, it will be replaced. Otherwise, it will be added to the DM.

  Level: intermediate

.seealso: DMSetField, DMAddField(), DMGetDS(), DMGetCellDS(), DMGetRegionDS(), DMSetRegionDS()
@*/
PetscErrorCode DMCreateDS(DM dm)
{
  MPI_Comm       comm;
  PetscDS        dsDef;
  DMLabel       *labelSet;
  PetscInt       dE, Nf = dm->Nf, f, s, Nl, l, Ndef, k;
  PetscBool      doSetup = PETSC_TRUE, flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!dm->fields) PetscFunctionReturn(0);
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dE);CHKERRQ(ierr);
  /* Determine how many regions we have */
  ierr = PetscMalloc1(Nf, &labelSet);CHKERRQ(ierr);
  Nl   = 0;
  Ndef = 0;
  for (f = 0; f < Nf; ++f) {
    DMLabel  label = dm->fields[f].label;
    PetscInt l;

#ifdef PETSC_HAVE_LIBCEED
    /* Move CEED context to discretizations */
    {
      PetscClassId id;

      ierr = PetscObjectGetClassId(dm->fields[f].disc, &id);CHKERRQ(ierr);
      if (id == PETSCFE_CLASSID) {
        Ceed ceed;

        ierr = DMGetCeed(dm, &ceed);CHKERRQ(ierr);
        ierr = PetscFESetCeed((PetscFE) dm->fields[f].disc, ceed);CHKERRQ(ierr);
      }
    }
#endif
    if (!label) {++Ndef; continue;}
    for (l = 0; l < Nl; ++l) if (label == labelSet[l]) break;
    if (l < Nl) continue;
    labelSet[Nl++] = label;
  }
  /* Create default DS if there are no labels to intersect with */
  ierr = DMGetRegionDS(dm, NULL, NULL, &dsDef);CHKERRQ(ierr);
  if (!dsDef && Ndef && !Nl) {
    IS        fields;
    PetscInt *fld, nf;

    for (f = 0, nf = 0; f < Nf; ++f) if (!dm->fields[f].label) ++nf;
    PetscCheckFalse(!nf,comm, PETSC_ERR_PLIB, "All fields have labels, but we are trying to create a default DS");
    ierr = PetscMalloc1(nf, &fld);CHKERRQ(ierr);
    for (f = 0, nf = 0; f < Nf; ++f) if (!dm->fields[f].label) fld[nf++] = f;
    ierr = ISCreate(PETSC_COMM_SELF, &fields);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) fields, "dm_fields_");CHKERRQ(ierr);
    ierr = ISSetType(fields, ISGENERAL);CHKERRQ(ierr);
    ierr = ISGeneralSetIndices(fields, nf, fld, PETSC_OWN_POINTER);CHKERRQ(ierr);

    ierr = PetscDSCreate(PETSC_COMM_SELF, &dsDef);CHKERRQ(ierr);
    ierr = DMSetRegionDS(dm, NULL, fields, dsDef);CHKERRQ(ierr);
    ierr = PetscDSDestroy(&dsDef);CHKERRQ(ierr);
    ierr = ISDestroy(&fields);CHKERRQ(ierr);
  }
  ierr = DMGetRegionDS(dm, NULL, NULL, &dsDef);CHKERRQ(ierr);
  if (dsDef) {ierr = PetscDSSetCoordinateDimension(dsDef, dE);CHKERRQ(ierr);}
  /* Intersect labels with default fields */
  if (Ndef && Nl) {
    DM              plex;
    DMLabel         cellLabel;
    IS              fieldIS, allcellIS, defcellIS = NULL;
    PetscInt       *fields;
    const PetscInt *cells;
    PetscInt        depth, nf = 0, n, c;

    ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
    ierr = DMPlexGetDepth(plex, &depth);CHKERRQ(ierr);
    ierr = DMGetStratumIS(plex, "dim", depth, &allcellIS);CHKERRQ(ierr);
    if (!allcellIS) {ierr = DMGetStratumIS(plex, "depth", depth, &allcellIS);CHKERRQ(ierr);}
    /* TODO This looks like it only works for one label */
    for (l = 0; l < Nl; ++l) {
      DMLabel label = labelSet[l];
      IS      pointIS;

      ierr = ISDestroy(&defcellIS);CHKERRQ(ierr);
      ierr = DMLabelGetStratumIS(label, 1, &pointIS);CHKERRQ(ierr);
      ierr = ISDifference(allcellIS, pointIS, &defcellIS);CHKERRQ(ierr);
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&allcellIS);CHKERRQ(ierr);

    ierr = DMLabelCreate(PETSC_COMM_SELF, "defaultCells", &cellLabel);CHKERRQ(ierr);
    ierr = ISGetLocalSize(defcellIS, &n);CHKERRQ(ierr);
    ierr = ISGetIndices(defcellIS, &cells);CHKERRQ(ierr);
    for (c = 0; c < n; ++c) {ierr = DMLabelSetValue(cellLabel, cells[c], 1);CHKERRQ(ierr);}
    ierr = ISRestoreIndices(defcellIS, &cells);CHKERRQ(ierr);
    ierr = ISDestroy(&defcellIS);CHKERRQ(ierr);
    ierr = DMPlexLabelComplete(plex, cellLabel);CHKERRQ(ierr);

    ierr = PetscMalloc1(Ndef, &fields);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) if (!dm->fields[f].label) fields[nf++] = f;
    ierr = ISCreate(PETSC_COMM_SELF, &fieldIS);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) fieldIS, "dm_fields_");CHKERRQ(ierr);
    ierr = ISSetType(fieldIS, ISGENERAL);CHKERRQ(ierr);
    ierr = ISGeneralSetIndices(fieldIS, nf, fields, PETSC_OWN_POINTER);CHKERRQ(ierr);

    ierr = PetscDSCreate(PETSC_COMM_SELF, &dsDef);CHKERRQ(ierr);
    ierr = DMSetRegionDS(dm, cellLabel, fieldIS, dsDef);CHKERRQ(ierr);
    ierr = PetscDSSetCoordinateDimension(dsDef, dE);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&cellLabel);CHKERRQ(ierr);
    ierr = PetscDSDestroy(&dsDef);CHKERRQ(ierr);
    ierr = ISDestroy(&fieldIS);CHKERRQ(ierr);
    ierr = DMDestroy(&plex);CHKERRQ(ierr);
  }
  /* Create label DSes
     - WE ONLY SUPPORT IDENTICAL OR DISJOINT LABELS
  */
  /* TODO Should check that labels are disjoint */
  for (l = 0; l < Nl; ++l) {
    DMLabel   label = labelSet[l];
    PetscDS   ds;
    IS        fields;
    PetscInt *fld, nf;

    ierr = PetscDSCreate(PETSC_COMM_SELF, &ds);CHKERRQ(ierr);
    for (f = 0, nf = 0; f < Nf; ++f) if (label == dm->fields[f].label || !dm->fields[f].label) ++nf;
    ierr = PetscMalloc1(nf, &fld);CHKERRQ(ierr);
    for (f = 0, nf  = 0; f < Nf; ++f) if (label == dm->fields[f].label || !dm->fields[f].label) fld[nf++] = f;
    ierr = ISCreate(PETSC_COMM_SELF, &fields);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) fields, "dm_fields_");CHKERRQ(ierr);
    ierr = ISSetType(fields, ISGENERAL);CHKERRQ(ierr);
    ierr = ISGeneralSetIndices(fields, nf, fld, PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = DMSetRegionDS(dm, label, fields, ds);CHKERRQ(ierr);
    ierr = ISDestroy(&fields);CHKERRQ(ierr);
    ierr = PetscDSSetCoordinateDimension(ds, dE);CHKERRQ(ierr);
    {
      DMPolytopeType ct;
      PetscInt       lStart, lEnd;
      PetscBool      isCohesiveLocal = PETSC_FALSE, isCohesive;

      ierr = DMLabelGetBounds(label, &lStart, &lEnd);CHKERRQ(ierr);
      if (lStart >= 0) {
        ierr = DMPlexGetCellType(dm, lStart, &ct);CHKERRQ(ierr);
        switch (ct) {
          case DM_POLYTOPE_POINT_PRISM_TENSOR:
          case DM_POLYTOPE_SEG_PRISM_TENSOR:
          case DM_POLYTOPE_TRI_PRISM_TENSOR:
          case DM_POLYTOPE_QUAD_PRISM_TENSOR:
            isCohesiveLocal = PETSC_TRUE;break;
          default: break;
        }
      }
      ierr = MPI_Allreduce(&isCohesiveLocal, &isCohesive, 1, MPIU_BOOL, MPI_LOR, comm);CHKERRMPI(ierr);
      for (f = 0, nf  = 0; f < Nf; ++f) {
        if (label == dm->fields[f].label || !dm->fields[f].label) {
          if (label == dm->fields[f].label) {
            ierr = PetscDSSetDiscretization(ds, nf, NULL);CHKERRQ(ierr);
            ierr = PetscDSSetCohesive(ds, nf, isCohesive);CHKERRQ(ierr);
          }
          ++nf;
        }
      }
    }
    ierr = PetscDSDestroy(&ds);CHKERRQ(ierr);
  }
  ierr = PetscFree(labelSet);CHKERRQ(ierr);
  /* Set fields in DSes */
  for (s = 0; s < dm->Nds; ++s) {
    PetscDS         ds     = dm->probs[s].ds;
    IS              fields = dm->probs[s].fields;
    const PetscInt *fld;
    PetscInt        nf, dsnf;
    PetscBool       isCohesive;

    ierr = PetscDSGetNumFields(ds, &dsnf);CHKERRQ(ierr);
    ierr = PetscDSIsCohesive(ds, &isCohesive);CHKERRQ(ierr);
    ierr = ISGetLocalSize(fields, &nf);CHKERRQ(ierr);
    ierr = ISGetIndices(fields, &fld);CHKERRQ(ierr);
    for (f = 0; f < nf; ++f) {
      PetscObject  disc  = dm->fields[fld[f]].disc;
      PetscBool    isCohesiveField;
      PetscClassId id;

      /* Handle DS with no fields */
      if (dsnf) {ierr = PetscDSGetCohesive(ds, f, &isCohesiveField);CHKERRQ(ierr);}
      /* If this is a cohesive cell, then regular fields need the lower dimensional discretization */
      if (isCohesive && !isCohesiveField) {ierr = PetscFEGetHeightSubspace((PetscFE) disc, 1, (PetscFE *) &disc);CHKERRQ(ierr);}
      ierr = PetscDSSetDiscretization(ds, f, disc);CHKERRQ(ierr);
      /* We allow people to have placeholder fields and construct the Section by hand */
      ierr = PetscObjectGetClassId(disc, &id);CHKERRQ(ierr);
      if ((id != PETSCFE_CLASSID) && (id != PETSCFV_CLASSID)) doSetup = PETSC_FALSE;
    }
    ierr = ISRestoreIndices(fields, &fld);CHKERRQ(ierr);
  }
  /* Allow k-jet tabulation */
  ierr = PetscOptionsGetInt(NULL, ((PetscObject) dm)->prefix, "-dm_ds_jet_degree", &k, &flg);CHKERRQ(ierr);
  if (flg) {
    for (s = 0; s < dm->Nds; ++s) {
      PetscDS  ds = dm->probs[s].ds;
      PetscInt Nf, f;

      ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
      for (f = 0; f < Nf; ++f) {ierr = PetscDSSetJetDegree(ds, f, k);CHKERRQ(ierr);}
    }
  }
  /* Setup DSes */
  if (doSetup) {
    for (s = 0; s < dm->Nds; ++s) {ierr = PetscDSSetUp(dm->probs[s].ds);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

/*@
  DMComputeExactSolution - Compute the exact solution for a given DM, using the PetscDS information.

  Collective on DM

  Input Parameters:
+ dm   - The DM
- time - The time

  Output Parameters:
+ u    - The vector will be filled with exact solution values, or NULL
- u_t  - The vector will be filled with the time derivative of exact solution values, or NULL

  Note: The user must call PetscDSSetExactSolution() beforehand

  Level: developer

.seealso: PetscDSSetExactSolution()
@*/
PetscErrorCode DMComputeExactSolution(DM dm, PetscReal time, Vec u, Vec u_t)
{
  PetscErrorCode (**exacts)(PetscInt, PetscReal, const PetscReal x[], PetscInt, PetscScalar *u, void *ctx);
  void            **ectxs;
  PetscInt          Nf, Nds, s;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (u)   PetscValidHeaderSpecific(u, VEC_CLASSID, 3);
  if (u_t) PetscValidHeaderSpecific(u_t, VEC_CLASSID, 4);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  ierr = PetscMalloc2(Nf, &exacts, Nf, &ectxs);CHKERRQ(ierr);
  ierr = DMGetNumDS(dm, &Nds);CHKERRQ(ierr);
  for (s = 0; s < Nds; ++s) {
    PetscDS         ds;
    DMLabel         label;
    IS              fieldIS;
    const PetscInt *fields, id = 1;
    PetscInt        dsNf, f;

    ierr = DMGetRegionNumDS(dm, s, &label, &fieldIS, &ds);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(ds, &dsNf);CHKERRQ(ierr);
    ierr = ISGetIndices(fieldIS, &fields);CHKERRQ(ierr);
    ierr = PetscArrayzero(exacts, Nf);CHKERRQ(ierr);
    ierr = PetscArrayzero(ectxs, Nf);CHKERRQ(ierr);
    if (u) {
      for (f = 0; f < dsNf; ++f) {
        const PetscInt field = fields[f];
        ierr = PetscDSGetExactSolution(ds, field, &exacts[field], &ectxs[field]);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(fieldIS, &fields);CHKERRQ(ierr);
      if (label) {
        ierr = DMProjectFunctionLabel(dm, time, label, 1, &id, 0, NULL, exacts, ectxs, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
      } else {
        ierr = DMProjectFunction(dm, time, exacts, ectxs, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
      }
    }
    if (u_t) {
      ierr = PetscArrayzero(exacts, Nf);CHKERRQ(ierr);
      ierr = PetscArrayzero(ectxs, Nf);CHKERRQ(ierr);
      for (f = 0; f < dsNf; ++f) {
        const PetscInt field = fields[f];
        ierr = PetscDSGetExactSolutionTimeDerivative(ds, field, &exacts[field], &ectxs[field]);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(fieldIS, &fields);CHKERRQ(ierr);
      if (label) {
        ierr = DMProjectFunctionLabel(dm, time, label, 1, &id, 0, NULL, exacts, ectxs, INSERT_ALL_VALUES, u_t);CHKERRQ(ierr);
      } else {
        ierr = DMProjectFunction(dm, time, exacts, ectxs, INSERT_ALL_VALUES, u_t);CHKERRQ(ierr);
      }
    }
  }
  if (u) {
    ierr = PetscObjectSetName((PetscObject) u, "Exact Solution");CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) u, "exact_");CHKERRQ(ierr);
  }
  if (u_t) {
    ierr = PetscObjectSetName((PetscObject) u, "Exact Solution Time Derivative");CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) u_t, "exact_t_");CHKERRQ(ierr);
  }
  ierr = PetscFree2(exacts, ectxs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMTransferDS_Internal(DM dm, DMLabel label, IS fields, PetscDS ds)
{
  PetscDS        dsNew;
  DSBoundary     b;
  PetscInt       cdim, Nf, f;
  PetscBool      isCohesive;
  void          *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDSCreate(PetscObjectComm((PetscObject) ds), &dsNew);CHKERRQ(ierr);
  ierr = PetscDSCopyConstants(ds, dsNew);CHKERRQ(ierr);
  ierr = PetscDSCopyExactSolutions(ds, dsNew);CHKERRQ(ierr);
  ierr = PetscDSSelectDiscretizations(ds, PETSC_DETERMINE, NULL, dsNew);CHKERRQ(ierr);
  ierr = PetscDSCopyEquations(ds, dsNew);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    ierr = PetscDSGetContext(ds, f, &ctx);CHKERRQ(ierr);
    ierr = PetscDSSetContext(dsNew, f, ctx);CHKERRQ(ierr);
    ierr = PetscDSGetCohesive(ds, f, &isCohesive);CHKERRQ(ierr);
    ierr = PetscDSSetCohesive(dsNew, f, isCohesive);CHKERRQ(ierr);
  }
  if (Nf) {
    ierr = PetscDSGetCoordinateDimension(ds, &cdim);CHKERRQ(ierr);
    ierr = PetscDSSetCoordinateDimension(dsNew, cdim);CHKERRQ(ierr);
  }
  ierr = PetscDSCopyBoundary(ds, PETSC_DETERMINE, NULL, dsNew);CHKERRQ(ierr);
  for (b = dsNew->boundary; b; b = b->next) {
    ierr = DMGetLabel(dm, b->lname, &b->label);CHKERRQ(ierr);
    /* Do not check if label exists here, since p4est calls this for the reference tree which does not have the labels */
    //PetscCheckFalse(!b->label,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Label %s missing in new DM", name);
  }

  ierr = DMSetRegionDS(dm, label, fields, dsNew);CHKERRQ(ierr);
  ierr = PetscDSDestroy(&dsNew);CHKERRQ(ierr);
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
    DMLabel  label;
    IS       fields;
    PetscDS  ds, newds;
    PetscInt Nbd, bd;

    ierr = DMGetRegionNumDS(dm, s, &label, &fields, &ds);CHKERRQ(ierr);
    /* TODO: We need to change all keys from labels in the old DM to labels in the new DM */
    ierr = DMTransferDS_Internal(newdm, label, fields, ds);CHKERRQ(ierr);
    /* Commplete new labels in the new DS */
    ierr = DMGetRegionDS(newdm, label, NULL, &newds);CHKERRQ(ierr);
    ierr = PetscDSGetNumBoundary(newds, &Nbd);CHKERRQ(ierr);
    for (bd = 0; bd < Nbd; ++bd) {
      PetscWeakForm wf;
      DMLabel       label;
      PetscInt      field;

      ierr = PetscDSGetBoundary(newds, bd, &wf, NULL, NULL, &label, NULL, NULL, &field, NULL, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
      ierr = DMCompleteBoundaryLabel_Internal(newdm, newds, field, bd, label);CHKERRQ(ierr);
      ierr = PetscWeakFormReplaceLabel(wf, label);CHKERRQ(ierr);
    }
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
  PetscInt       Nds, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidLogicalCollectiveInt(dm, dim, 2);
  dm->dim = dim;
  if (dm->dim >= 0) {
    ierr = DMGetNumDS(dm, &Nds);CHKERRQ(ierr);
    for (n = 0; n < Nds; ++n) {
      ierr = DMGetRegionNumDS(dm, n, NULL, NULL, &ds);CHKERRQ(ierr);
      if (ds->dimEmbed < 0) {ierr = PetscDSSetCoordinateDimension(ds, dim);CHKERRQ(ierr);}
    }
  }
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
  PetscCheckFalse((dim < 0) || (dim > d),PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d 1", dim, d);
  PetscCheckFalse(!dm->ops->getdimpoints,PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "DM type %s does not implement DMGetDimPoints",((PetscObject)dm)->type_name);
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

.seealso: DMSetCoordinatesLocal(), DMGetCoordinates(), DMGetCoordinatesLocal(), DMGetCoordinateDM(), DMDASetUniformCoordinates()
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
  if (c) PetscValidHeaderSpecific(c,VEC_CLASSID,2);
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
  This is a borrowed reference, so the user should NOT destroy this vector. When the DM is
  destroyed the array will no longer be valid.

  Each process has only the locally-owned portion of the global coordinates (does NOT have the ghost coordinates).

  For DMDA, in two and three dimensions coordinates are interlaced (x_0,y_0,x_1,y_1,...)
  and (x_0,y_0,z_0,x_1,y_1,z_1...)

  Level: intermediate

.seealso: DMSetCoordinates(), DMGetCoordinatesLocal(), DMGetCoordinateDM(), DMDASetUniformCoordinates()
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
  PetscCheckFalse(!dm->coordinatesLocal && dm->coordinates,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DMGetCoordinatesLocalSetUp() has not been called");
  *c = dm->coordinatesLocal;
  PetscFunctionReturn(0);
}

/*@
  DMGetCoordinatesLocalTuple - Gets a local vector with the coordinates of specified points and section describing its layout.

  Not collective

  Input Parameters:
+ dm - the DM
- p - the IS of points whose coordinates will be returned

  Output Parameters:
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
  PetscCheckFalse(!dm->coordinatesLocal,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DMGetCoordinatesLocalSetUp() has not been called or coordinates not set");
  PetscCheckFalse(!dm->coordinateDM || !dm->coordinateDM->localSection,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM not supported");
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

    PetscCheckFalse(!dm->ops->createcoordinatedm,PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unable to create coordinates for this DM");
    ierr = (*dm->ops->createcoordinatedm)(dm, &cdm);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)cdm, "coordinateDM");CHKERRQ(ierr);
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
  PetscInt       Nds, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->dimEmbed = dim;
  if (dm->dim >= 0) {
    ierr = DMGetNumDS(dm, &Nds);CHKERRQ(ierr);
    for (n = 0; n < Nds; ++n) {
      ierr = DMGetRegionNumDS(dm, n, NULL, NULL, &ds);CHKERRQ(ierr);
      ierr = PetscDSSetCoordinateDimension(ds, dim);CHKERRQ(ierr);
    }
  }
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

/*@
  DMProjectCoordinates - Project coordinates to a different space

  Input Parameters:
+ dm      - The DM object
- disc    - The new coordinate discretization

  Level: intermediate

.seealso: DMGetCoordinateField()
@*/
PetscErrorCode DMProjectCoordinates(DM dm, PetscFE disc)
{
  PetscObject    discOld;
  PetscClassId   classid;
  DM             cdmOld,cdmNew;
  Vec            coordsOld,coordsNew;
  Mat            matInterp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(disc,PETSCFE_CLASSID,2);

  ierr = DMGetCoordinateDM(dm, &cdmOld);CHKERRQ(ierr);
  /* Check current discretization is compatible */
  ierr = DMGetField(cdmOld, 0, NULL, &discOld);CHKERRQ(ierr);
  ierr = PetscObjectGetClassId(discOld, &classid);CHKERRQ(ierr);
  if (classid != PETSCFE_CLASSID) {
    if (classid == PETSC_CONTAINER_CLASSID) {
      PetscFE        feLinear;
      DMPolytopeType ct;
      PetscInt       dim, dE, cStart;
      PetscBool      simplex;

      /* Assume linear vertex coordinates */
      ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
      ierr = DMGetCoordinateDim(dm, &dE);CHKERRQ(ierr);
      ierr = DMPlexGetHeightStratum(cdmOld, 0, &cStart, NULL);CHKERRQ(ierr);
      ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
      switch (ct) {
        case DM_POLYTOPE_TRI_PRISM:
        case DM_POLYTOPE_TRI_PRISM_TENSOR:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot autoamtically create coordinate space for prisms");
        default: break;
      }
      simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
      ierr = PetscFECreateLagrange(PETSC_COMM_SELF, dim, dE, simplex, 1, -1, &feLinear);CHKERRQ(ierr);
      ierr = DMSetField(cdmOld, 0, NULL, (PetscObject) feLinear);CHKERRQ(ierr);
      ierr = PetscFEDestroy(&feLinear);CHKERRQ(ierr);
      ierr = DMCreateDS(cdmOld);CHKERRQ(ierr);
    } else {
      const char *discname;

      ierr = PetscObjectGetType(discOld, &discname);CHKERRQ(ierr);
      SETERRQ(PetscObjectComm(discOld), PETSC_ERR_SUP, "Discretization type %s not supported", discname);
    }
  }
  /* Make a fresh clone of the coordinate DM */
  ierr = DMClone(cdmOld, &cdmNew);CHKERRQ(ierr);
  ierr = DMSetField(cdmNew, 0, NULL, (PetscObject) disc);CHKERRQ(ierr);
  ierr = DMCreateDS(cdmNew);CHKERRQ(ierr);
  /* Project the coordinate vector from old to new space  */
  ierr = DMGetCoordinates(dm, &coordsOld);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(cdmNew, &coordsNew);CHKERRQ(ierr);
  ierr = DMCreateInterpolation(cdmOld, cdmNew, &matInterp, NULL);CHKERRQ(ierr);
  ierr = MatInterpolate(matInterp, coordsOld, coordsNew);CHKERRQ(ierr);
  ierr = MatDestroy(&matInterp);CHKERRQ(ierr);
  /* Set new coordinate structures */
  ierr = DMSetCoordinateField(dm, NULL);CHKERRQ(ierr);
  ierr = DMSetCoordinateDM(dm, cdmNew);CHKERRQ(ierr);
  ierr = DMSetCoordinates(dm, coordsNew);CHKERRQ(ierr);
  ierr = VecDestroy(&coordsNew);CHKERRQ(ierr);
  ierr = DMDestroy(&cdmNew);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMGetPeriodicity - Get the description of mesh periodicity

  Input Parameter:
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
. per     - Whether the DM is periodic or not.
. maxCell - Over distances greater than this, we can assume a point has crossed over to another sheet, when trying to localize cell coordinates. Pass NULL to remove such information.
. L       - If we assume the mesh is a torus, this is the length of each coordinate
- bd      - This describes the type of periodicity in each topological dimension

  Notes: If per is PETSC_TRUE and maxCell is not provided, coordinates need to be already localized, or must be localized by hand by the user.

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
  if (maxCell) {PetscValidRealPointer(maxCell,3);}
  if (L)       {PetscValidRealPointer(L,4);}
  if (bd)      {PetscValidPointer(bd,5);}
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (maxCell) {
    if (!dm->maxCell) {ierr = PetscMalloc1(dim, &dm->maxCell);CHKERRQ(ierr);}
    for (d = 0; d < dim; ++d) dm->maxCell[d] = maxCell[d];
  } else { /* remove maxCell information to disable automatic computation of localized vertices */
    ierr = PetscFree(dm->maxCell);CHKERRQ(ierr);
  }

  if (L) {
    if (!dm->L) {ierr = PetscMalloc1(dim, &dm->L);CHKERRQ(ierr);}
    for (d = 0; d < dim; ++d) dm->L[d] = L[d];
  }
  if (bd) {
    if (!dm->bdtype) {ierr = PetscMalloc1(dim, &dm->bdtype);CHKERRQ(ierr);}
    for (d = 0; d < dim; ++d) dm->bdtype[d] = bd[d];
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
      const PetscReal maxC = dm->maxCell[d];

      if ((dm->bdtype[d] != DM_BOUNDARY_NONE) && (PetscAbsScalar(anchor[d] - in[d]) > maxC)) {
        const PetscScalar newCoord = PetscRealPart(anchor[d]) > PetscRealPart(in[d]) ? dm->L[d] + in[d] : in[d] - dm->L[d];

        if (PetscAbsScalar(newCoord - anchor[d]) > maxC)
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "%D-Coordinate %g more than %g away from anchor %g", d, (double) PetscRealPart(in[d]), (double) maxC, (double) PetscRealPart(anchor[d]));
        out[d] += newCoord;
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
  PetscInt       depth, cStart, cEnd, sStart, sEnd, c, dof;
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
  ierr = DMPlexGetDepth(cdm, &depth);CHKERRQ(ierr);
  if (!depth) PetscFunctionReturn(0);

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
  ierr = MPIU_Allreduce(&localized,areLocalized,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)dm));CHKERRMPI(ierr);
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
  PetscCheckFalse(!coordinates,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Missing local coordinates vector");
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
  ierr = MPI_Allreduce(&alreadyLocalized,&alreadyLocalizedGlobal,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)dm));CHKERRMPI(ierr);
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
- ltype - The type of point location, e.g. DM_POINTLOCATION_NONE or DM_POINTLOCATION_NEAREST

  Input/Output Parameters:
+ v - The Vec of points, on output contains the nearest mesh points to the given points if DM_POINTLOCATION_NEAREST is used
- cellSF - Points to either NULL, or a PetscSF with guesses for which cells contain each point;
           on output, the PetscSF containing the ranks and local indices of the containing points

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
    ierr = MPI_Comm_compare(PetscObjectComm((PetscObject)v),PetscObjectComm((PetscObject)*cellSF),&result);CHKERRMPI(ierr);
    PetscCheckFalse(result != MPI_IDENT && result != MPI_CONGRUENT,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"cellSF must have a communicator congruent to v's");
  } else {
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)v),cellSF);CHKERRQ(ierr);
  }
  PetscCheckFalse(!dm->ops->locatepoints,PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Point location not available for this DM");
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
  ierr = MPI_Allreduce(&hasConstraints, &ghasConstraints, 1, MPIU_BOOL, MPI_LOR, PetscObjectComm((PetscObject) dm));CHKERRMPI(ierr);
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
  PetscValidRealPointer(val,5);
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
  DMCreateLabelAtIndex - Create a label of the given name at the iven index. If it already exists, move it to this index.

  Not Collective

  Input Parameters:
+ dm   - The DM object
. l    - The index for the label
- name - The label name

  Level: intermediate

.seealso: DMCreateLabel(), DMLabelCreate(), DMHasLabel(), DMGetLabelValue(), DMSetLabelValue(), DMGetStratumIS()
@*/
PetscErrorCode DMCreateLabelAtIndex(DM dm, PetscInt l, const char name[])
{
  DMLabelLink    orig, prev = NULL;
  DMLabel        label;
  PetscInt       Nl, m;
  PetscBool      flg, match;
  const char    *lname;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 3);
  ierr = DMHasLabel(dm, name, &flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = DMLabelCreate(PETSC_COMM_SELF, name, &label);CHKERRQ(ierr);
    ierr = DMAddLabel(dm, label);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&label);CHKERRQ(ierr);
  }
  ierr = DMGetNumLabels(dm, &Nl);CHKERRQ(ierr);
  PetscCheckFalse(l >= Nl,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label index %D must be in [0, %D)", l, Nl);
  for (m = 0, orig = dm->labels; m < Nl; ++m, prev = orig, orig = orig->next) {
    ierr = PetscObjectGetName((PetscObject) orig->label, &lname);CHKERRQ(ierr);
    ierr = PetscStrcmp(name, lname, &match);CHKERRQ(ierr);
    if (match) break;
  }
  if (m == l) PetscFunctionReturn(0);
  if (!m) dm->labels = orig->next;
  else    prev->next = orig->next;
  if (!l) {
    orig->next = dm->labels;
    dm->labels = orig;
  } else {
    for (m = 0, prev = dm->labels; m < l-1; ++m, prev = prev->next);
    orig->next = prev->next;
    prev->next = orig;
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
  PetscCheckFalse(!label,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No label named %s was found", name);
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
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label %D does not exist in this DM", n);
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

  Note: Some of the default labels in a DMPlex will be
$ "depth"       - Holds the depth (co-dimension) of each mesh point
$ "celltype"    - Holds the topological type of each cell
$ "ghost"       - If the DM is distributed with overlap, this marks the cells and faces in the overlap
$ "Cell Sets"   - Mirrors the cell sets defined by GMsh and ExodusII
$ "Face Sets"   - Mirrors the face sets defined by GMsh and ExodusII
$ "Vertex Sets" - Mirrors the vertex sets defined by GMsh

  Level: intermediate

.seealso: DMCreateLabel(), DMHasLabel(), DMPlexGetDepthLabel(), DMPlexGetCellType()
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
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label %D does not exist in this DM", n);
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
  PetscCheckFalse(hasLabel,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label %s already exists in this DM", lname);
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
  DMSetLabel - Replaces the label of a given name, or ignores it if the name is not present

  Not Collective

  Input Parameters:
+ dm    - The DM object
- label - The DMLabel, having the same name, to substitute

  Note: Some of the default labels in a DMPlex will be
$ "depth"       - Holds the depth (co-dimension) of each mesh point
$ "celltype"    - Holds the topological type of each cell
$ "ghost"       - If the DM is distributed with overlap, this marks the cells and faces in the overlap
$ "Cell Sets"   - Mirrors the cell sets defined by GMsh and ExodusII
$ "Face Sets"   - Mirrors the face sets defined by GMsh and ExodusII
$ "Vertex Sets" - Mirrors the vertex sets defined by GMsh

  Level: intermediate

.seealso: DMCreateLabel(), DMHasLabel(), DMPlexGetDepthLabel(), DMPlexGetCellType()
@*/
PetscErrorCode DMSetLabel(DM dm, DMLabel label)
{
  DMLabelLink    next = dm->labels;
  PetscBool      hasLabel, flg;
  const char    *name, *lname;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 2);
  ierr = PetscObjectGetName((PetscObject) label, &name);CHKERRQ(ierr);
  while (next) {
    ierr = PetscObjectGetName((PetscObject) next->label, &lname);CHKERRQ(ierr);
    ierr = PetscStrcmp(name, lname, &hasLabel);CHKERRQ(ierr);
    if (hasLabel) {
      ierr = PetscObjectReference((PetscObject) label);CHKERRQ(ierr);
      ierr = PetscStrcmp(lname, "depth", &flg);CHKERRQ(ierr);
      if (flg) dm->depthLabel = label;
      ierr = PetscStrcmp(lname, "celltype", &flg);CHKERRQ(ierr);
      if (flg) dm->celltypeLabel = label;
      ierr = DMLabelDestroy(&next->label);CHKERRQ(ierr);
      next->label = label;
      break;
    }
    next = next->next;
  }
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
. label - The DMLabel to be removed from the DM
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
  PetscCheckFalse(!hasLabel && failNotFound,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Given label not found in DM");
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
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No label named %s was present in this dm", name);
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
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No label named %s was present in this dm", name);
}

/*@
  DMCopyLabels - Copy labels from one mesh to another with a superset of the points

  Collective on dmA

  Input Parameters:
+ dmA - The DM object with initial labels
. dmB - The DM object to which labels are copied
. mode - Copy labels by pointers (PETSC_OWN_POINTER) or duplicate them (PETSC_COPY_VALUES)
. all  - Copy all labels including "depth", "dim", and "celltype" (PETSC_TRUE) which are otherwise ignored (PETSC_FALSE)
- emode - How to behave when a DMLabel in the source and destination DMs with the same name is encountered (see DMCopyLabelsMode)

  Level: intermediate

  Notes:
  This is typically used when interpolating or otherwise adding to a mesh, or testing.

.seealso: DMAddLabel(), DMCopyLabelsMode
@*/
PetscErrorCode DMCopyLabels(DM dmA, DM dmB, PetscCopyMode mode, PetscBool all, DMCopyLabelsMode emode)
{
  DMLabel        label, labelNew, labelOld;
  const char    *name;
  PetscBool      flg;
  DMLabelLink    link;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmA, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmB, DM_CLASSID, 2);
  PetscValidLogicalCollectiveEnum(dmA, mode,3);
  PetscValidLogicalCollectiveBool(dmA, all, 4);
  PetscCheckFalse(mode==PETSC_USE_POINTER,PetscObjectComm((PetscObject)dmA), PETSC_ERR_SUP, "PETSC_USE_POINTER not supported for objects");
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
    }
    ierr = DMGetLabel(dmB, name, &labelOld);CHKERRQ(ierr);
    if (labelOld) {
      switch (emode) {
        case DM_COPY_LABELS_KEEP:
          continue;
        case DM_COPY_LABELS_REPLACE:
          ierr = DMRemoveLabelBySelf(dmB, &labelOld, PETSC_TRUE);CHKERRQ(ierr);
          break;
        case DM_COPY_LABELS_FAIL:
          SETERRQ(PetscObjectComm((PetscObject)dmA), PETSC_ERR_ARG_OUTOFRANGE, "Label %s already exists in destination DM", name);
        default:
          SETERRQ(PetscObjectComm((PetscObject)dmA), PETSC_ERR_ARG_OUTOFRANGE, "Unhandled DMCopyLabelsMode %d", emode);
      }
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

/*@C
  DMCompareLabels - Compare labels of two DMPlex meshes

  Collective

  Input Parameters:
+ dm0 - First DM object
- dm1 - Second DM object

  Output Parameters
+ equal   - (Optional) Flag whether labels of dm0 and dm1 are the same
- message - (Optional) Message describing the difference, or NULL if there is no difference

  Level: intermediate

  Notes:
  The output flag equal is the same on all processes.
  If it is passed as NULL and difference is found, an error is thrown on all processes.
  Make sure to pass NULL on all processes.

  The output message is set independently on each rank.
  It is set to NULL if no difference was found on the current rank. It must be freed by user.
  If message is passed as NULL and difference is found, the difference description is printed to stderr in synchronized manner.
  Make sure to pass NULL on all processes.

  Labels are matched by name. If the number of labels and their names are equal,
  DMLabelCompare() is used to compare each pair of labels with the same name.

  Fortran Notes:
  This function is currently not available from Fortran.

.seealso: DMAddLabel(), DMCopyLabelsMode, DMLabelCompare()
@*/
PetscErrorCode DMCompareLabels(DM dm0, DM dm1, PetscBool *equal, char **message)
{
  PetscInt        n, i;
  char            msg[PETSC_MAX_PATH_LEN] = "";
  PetscBool       eq;
  MPI_Comm        comm;
  PetscMPIInt     rank;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm0,DM_CLASSID,1);
  PetscValidHeaderSpecific(dm1,DM_CLASSID,2);
  PetscCheckSameComm(dm0,1,dm1,2);
  if (equal) PetscValidBoolPointer(equal,3);
  if (message) PetscValidPointer(message, 4);
  ierr = PetscObjectGetComm((PetscObject)dm0, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  {
    PetscInt n1;

    ierr = DMGetNumLabels(dm0, &n);CHKERRQ(ierr);
    ierr = DMGetNumLabels(dm1, &n1);CHKERRQ(ierr);
    eq = (PetscBool) (n == n1);
    if (!eq) {
      ierr = PetscSNPrintf(msg, sizeof(msg), "Number of labels in dm0 = %D != %D = Number of labels in dm1", n, n1);CHKERRQ(ierr);
    }
    ierr = MPI_Allreduce(MPI_IN_PLACE, &eq, 1, MPIU_BOOL, MPI_LAND, comm);CHKERRMPI(ierr);
    if (!eq) goto finish;
  }
  for (i=0; i<n; i++) {
    DMLabel     l0, l1;
    const char *name;
    char       *msgInner;

    /* Ignore label order */
    ierr = DMGetLabelByNum(dm0, i, &l0);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject)l0, &name);CHKERRQ(ierr);
    ierr = DMGetLabel(dm1, name, &l1);CHKERRQ(ierr);
    if (!l1) {
      ierr = PetscSNPrintf(msg, sizeof(msg), "Label \"%s\" (#%D in dm0) not found in dm1", name, i);CHKERRQ(ierr);
      eq = PETSC_FALSE;
      break;
    }
    ierr = DMLabelCompare(comm, l0, l1, &eq, &msgInner);CHKERRQ(ierr);
    ierr = PetscStrncpy(msg, msgInner, sizeof(msg));CHKERRQ(ierr);
    ierr = PetscFree(msgInner);CHKERRQ(ierr);
    if (!eq) break;
  }
  ierr = MPI_Allreduce(MPI_IN_PLACE, &eq, 1, MPIU_BOOL, MPI_LAND, comm);CHKERRMPI(ierr);
finish:
  /* If message output arg not set, print to stderr */
  if (message) {
    *message = NULL;
    if (msg[0]) {
      ierr = PetscStrallocpy(msg, message);CHKERRQ(ierr);
    }
  } else {
    if (msg[0]) {
      ierr = PetscSynchronizedFPrintf(comm, PETSC_STDERR, "[%d] %s\n", rank, msg);CHKERRQ(ierr);
    }
    ierr = PetscSynchronizedFlush(comm, PETSC_STDERR);CHKERRQ(ierr);
  }
  /* If same output arg not ser and labels are not equal, throw error */
  if (equal) *equal = eq;
  else PetscCheckFalse(!eq,comm, PETSC_ERR_ARG_INCOMP, "DMLabels are not the same in dm0 and dm1");
  PetscFunctionReturn(0);
}

PetscErrorCode DMSetLabelValue_Fast(DM dm, DMLabel *label, const char name[], PetscInt point, PetscInt value)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(label,2);
  if (!*label) {
    ierr = DMCreateLabel(dm, name);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, name, label);CHKERRQ(ierr);
  }
  ierr = DMLabelSetValue(*label, point, value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Many mesh programs, such as Triangle and TetGen, allow only a single label for each mesh point. Therefore, we would
  like to encode all label IDs using a single, universal label. We can do this by assigning an integer to every
  (label, id) pair in the DM.

  However, a mesh point can have multiple labels, so we must separate all these values. We will assign a bit range to
  each label.
*/
PetscErrorCode DMUniversalLabelCreate(DM dm, DMUniversalLabel *universal)
{
  DMUniversalLabel ul;
  PetscBool       *active;
  PetscInt         pStart, pEnd, p, Nl, l, m;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(1, &ul);CHKERRQ(ierr);
  ierr = DMLabelCreate(PETSC_COMM_SELF, "universal", &ul->label);CHKERRQ(ierr);
  ierr = DMGetNumLabels(dm, &Nl);CHKERRQ(ierr);
  ierr = PetscCalloc1(Nl, &active);CHKERRQ(ierr);
  ul->Nl = 0;
  for (l = 0; l < Nl; ++l) {
    PetscBool   isdepth, iscelltype;
    const char *name;

    ierr = DMGetLabelName(dm, l, &name);CHKERRQ(ierr);
    ierr = PetscStrncmp(name, "depth", 6, &isdepth);CHKERRQ(ierr);
    ierr = PetscStrncmp(name, "celltype", 9, &iscelltype);CHKERRQ(ierr);
    active[l] = !(isdepth || iscelltype) ? PETSC_TRUE : PETSC_FALSE;
    if (active[l]) ++ul->Nl;
  }
  ierr = PetscCalloc5(ul->Nl, &ul->names, ul->Nl, &ul->indices, ul->Nl+1, &ul->offsets, ul->Nl+1, &ul->bits, ul->Nl, &ul->masks);CHKERRQ(ierr);
  ul->Nv = 0;
  for (l = 0, m = 0; l < Nl; ++l) {
    DMLabel     label;
    PetscInt    nv;
    const char *name;

    if (!active[l]) continue;
    ierr = DMGetLabelName(dm, l, &name);CHKERRQ(ierr);
    ierr = DMGetLabelByNum(dm, l, &label);CHKERRQ(ierr);
    ierr = DMLabelGetNumValues(label, &nv);CHKERRQ(ierr);
    ierr = PetscStrallocpy(name, &ul->names[m]);CHKERRQ(ierr);
    ul->indices[m]   = l;
    ul->Nv          += nv;
    ul->offsets[m+1] = nv;
    ul->bits[m+1]    = PetscCeilReal(PetscLog2Real(nv+1));
    ++m;
  }
  for (l = 1; l <= ul->Nl; ++l) {
    ul->offsets[l] = ul->offsets[l-1] + ul->offsets[l];
    ul->bits[l]    = ul->bits[l-1]    + ul->bits[l];
  }
  for (l = 0; l < ul->Nl; ++l) {
    PetscInt b;

    ul->masks[l] = 0;
    for (b = ul->bits[l]; b < ul->bits[l+1]; ++b) ul->masks[l] |= 1 << b;
  }
  ierr = PetscMalloc1(ul->Nv, &ul->values);CHKERRQ(ierr);
  for (l = 0, m = 0; l < Nl; ++l) {
    DMLabel         label;
    IS              valueIS;
    const PetscInt *varr;
    PetscInt        nv, v;

    if (!active[l]) continue;
    ierr = DMGetLabelByNum(dm, l, &label);CHKERRQ(ierr);
    ierr = DMLabelGetNumValues(label, &nv);CHKERRQ(ierr);
    ierr = DMLabelGetValueIS(label, &valueIS);CHKERRQ(ierr);
    ierr = ISGetIndices(valueIS, &varr);CHKERRQ(ierr);
    for (v = 0; v < nv; ++v) {
      ul->values[ul->offsets[m]+v] = varr[v];
    }
    ierr = ISRestoreIndices(valueIS, &varr);CHKERRQ(ierr);
    ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
    ierr = PetscSortInt(nv, &ul->values[ul->offsets[m]]);CHKERRQ(ierr);
    ++m;
  }
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt  uval = 0;
    PetscBool marked = PETSC_FALSE;

    for (l = 0, m = 0; l < Nl; ++l) {
      DMLabel  label;
      PetscInt val, defval, loc, nv;

      if (!active[l]) continue;
      ierr = DMGetLabelByNum(dm, l, &label);CHKERRQ(ierr);
      ierr = DMLabelGetValue(label, p, &val);CHKERRQ(ierr);
      ierr = DMLabelGetDefaultValue(label, &defval);CHKERRQ(ierr);
      if (val == defval) {++m; continue;}
      nv = ul->offsets[m+1]-ul->offsets[m];
      marked = PETSC_TRUE;
      ierr = PetscFindInt(val, nv, &ul->values[ul->offsets[m]], &loc);CHKERRQ(ierr);
      PetscCheckFalse(loc < 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Label value %D not found in compression array", val);
      uval += (loc+1) << ul->bits[m];
      ++m;
    }
    if (marked) {ierr = DMLabelSetValue(ul->label, p, uval);CHKERRQ(ierr);}
  }
  ierr = PetscFree(active);CHKERRQ(ierr);
  *universal = ul;
  PetscFunctionReturn(0);
}

PetscErrorCode DMUniversalLabelDestroy(DMUniversalLabel *universal)
{
  PetscInt       l;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (l = 0; l < (*universal)->Nl; ++l) {ierr = PetscFree((*universal)->names[l]);CHKERRQ(ierr);}
  ierr = DMLabelDestroy(&(*universal)->label);CHKERRQ(ierr);
  ierr = PetscFree5((*universal)->names, (*universal)->indices, (*universal)->offsets, (*universal)->bits, (*universal)->masks);CHKERRQ(ierr);
  ierr = PetscFree((*universal)->values);CHKERRQ(ierr);
  ierr = PetscFree(*universal);CHKERRQ(ierr);
  *universal = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode DMUniversalLabelGetLabel(DMUniversalLabel ul, DMLabel *ulabel)
{
  PetscFunctionBegin;
  PetscValidPointer(ulabel, 2);
  *ulabel = ul->label;
  PetscFunctionReturn(0);
}

PetscErrorCode DMUniversalLabelCreateLabels(DMUniversalLabel ul, PetscBool preserveOrder, DM dm)
{
  PetscInt       Nl = ul->Nl, l;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 3);
  for (l = 0; l < Nl; ++l) {
    if (preserveOrder) {ierr = DMCreateLabelAtIndex(dm, ul->indices[l], ul->names[l]);CHKERRQ(ierr);}
    else               {ierr = DMCreateLabel(dm, ul->names[l]);CHKERRQ(ierr);}
  }
  if (preserveOrder) {
    for (l = 0; l < ul->Nl; ++l) {
      const char *name;
      PetscBool   match;

      ierr = DMGetLabelName(dm, ul->indices[l], &name);CHKERRQ(ierr);
      ierr = PetscStrcmp(name, ul->names[l], &match);CHKERRQ(ierr);
      PetscCheckFalse(!match,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Label %D name %s does not match new name %s", l, name, ul->names[l]);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMUniversalLabelSetLabelValue(DMUniversalLabel ul, DM dm, PetscBool useIndex, PetscInt p, PetscInt value)
{
  PetscInt       l;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (l = 0; l < ul->Nl; ++l) {
    DMLabel  label;
    PetscInt lval = (value & ul->masks[l]) >> ul->bits[l];

    if (lval) {
      if (useIndex) {ierr = DMGetLabelByNum(dm, ul->indices[l], &label);CHKERRQ(ierr);}
      else          {ierr = DMGetLabel(dm, ul->names[l], &label);CHKERRQ(ierr);}
      ierr = DMLabelSetValue(label, p, ul->values[ul->offsets[l]+lval-1]);CHKERRQ(ierr);
    }
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

/*@C
  DMAddBoundary - Add a boundary condition to the model

  Collective on dm

  Input Parameters:
+ dm       - The DM, with a PetscDS that matches the problem being constrained
. type     - The type of condition, e.g. DM_BC_ESSENTIAL_ANALYTIC/DM_BC_ESSENTIAL_FIELD (Dirichlet), or DM_BC_NATURAL (Neumann)
. name     - The BC name
. label    - The label defining constrained points
. Nv       - The number of DMLabel values for constrained points
. values   - An array of values for constrained points
. field    - The field to constrain
. Nc       - The number of constrained field components (0 will constrain all fields)
. comps    - An array of constrained component numbers
. bcFunc   - A pointwise function giving boundary values
. bcFunc_t - A pointwise function giving the time deriative of the boundary values, or NULL
- ctx      - An optional user context for bcFunc

  Output Parameter:
. bd          - (Optional) Boundary number

  Options Database Keys:
+ -bc_<boundary name> <num> - Overrides the boundary ids
- -bc_<boundary name>_comp <num> - Overrides the boundary components

  Note:
  Both bcFunc abd bcFunc_t will depend on the boundary condition type. If the type if DM_BC_ESSENTIAL, Then the calling sequence is:

$ bcFunc(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar bcval[])

  If the type is DM_BC_ESSENTIAL_FIELD or other _FIELD value, then the calling sequence is:

$ bcFunc(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$        PetscReal time, const PetscReal x[], PetscScalar bcval[])

+ dim - the spatial dimension
. Nf - the number of fields
. uOff - the offset into u[] and u_t[] for each field
. uOff_x - the offset into u_x[] for each field
. u - each field evaluated at the current point
. u_t - the time derivative of each field evaluated at the current point
. u_x - the gradient of each field evaluated at the current point
. aOff - the offset into a[] and a_t[] for each auxiliary field
. aOff_x - the offset into a_x[] for each auxiliary field
. a - each auxiliary field evaluated at the current point
. a_t - the time derivative of each auxiliary field evaluated at the current point
. a_x - the gradient of auxiliary each field evaluated at the current point
. t - current time
. x - coordinates of the current point
. numConstants - number of constant parameters
. constants - constant parameters
- bcval - output values at the current point

  Level: developer

.seealso: DSGetBoundary(), PetscDSAddBoundary()
@*/
PetscErrorCode DMAddBoundary(DM dm, DMBoundaryConditionType type, const char name[], DMLabel label, PetscInt Nv, const PetscInt values[], PetscInt field, PetscInt Nc, const PetscInt comps[], void (*bcFunc)(void), void (*bcFunc_t)(void), void *ctx, PetscInt *bd)
{
  PetscDS        ds;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(dm, type, 2);
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 4);
  PetscValidLogicalCollectiveInt(dm, Nv, 5);
  PetscValidLogicalCollectiveInt(dm, field, 7);
  PetscValidLogicalCollectiveInt(dm, Nc, 8);
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = DMCompleteBoundaryLabel_Internal(dm, ds, field, PETSC_MAX_INT, label);CHKERRQ(ierr);
  ierr = PetscDSAddBoundary(ds, type, name, label, Nv, values, field, Nc, comps, bcFunc, bcFunc_t, ctx, bd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* TODO Remove this since now the structures are the same */
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
    dmbound->label      = dsbound->label;
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
    DSBoundary dsb   = b->dsboundary;
    PetscInt   i;

    if (label) {
      for (i = 0; i < dsb->Nv && !(*isBd); ++i) {ierr = DMLabelStratumHasPoint(label, dsb->values[i], point, isBd);CHKERRQ(ierr);}
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
$    func(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx);

+  dim - The spatial dimension
.  time - The time at which to sample
.  x   - The coordinates
.  Nc  - The number of components
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
$    func(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx);

+  dim - The spatial dimension
.  x   - The coordinates
.  Nc  - The number of components
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
  PetscValidHeaderSpecific(localX,VEC_CLASSID,6);
  PetscCheckFalse(!dm->ops->projectfunctionlocal,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMProjectFunctionLocal",((PetscObject)dm)->type_name);
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
$    func(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx);

+  dim - The spatial dimension
.  x   - The coordinates
.  Nc  - The number of components
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
$    func(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx);

+  dim - The spatial dimension
.  x   - The coordinates
.  Nc  - The number of components
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
  PetscValidHeaderSpecific(localX,VEC_CLASSID,11);
  PetscCheckFalse(!dm->ops->projectfunctionlabellocal,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMProjectFunctionLabelLocal",((PetscObject)dm)->type_name);
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
  PetscCheckFalse(!dm->ops->projectfieldlocal,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMProjectFieldLocal",((PetscObject)dm)->type_name);
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
  PetscValidHeaderSpecific(localU,VEC_CLASSID,8);
  PetscValidHeaderSpecific(localX,VEC_CLASSID,11);
  PetscCheckFalse(!dm->ops->projectfieldlabellocal,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMProjectFieldLabelLocal",((PetscObject)dm)->type_name);
  ierr = (dm->ops->projectfieldlabellocal)(dm, time, label, numIds, ids, Nc, comps, localU, funcs, mode, localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMProjectBdFieldLabelLocal - This projects the given function of the input fields into the function space provided, putting the coefficients in a local vector, calculating only over the portion of the domain boundary specified by the label.

  Not collective

  Input Parameters:
+ dm      - The DM
. time    - The time
. label   - The DMLabel marking the portion of the domain boundary to output
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
.  n            - The face normal
.  numConstants - The number of constants
.  constants    - The value of each constant
-  f            - The value of the function at this point in space

  Note:
  There are three different DMs that potentially interact in this function. The output DM, dm, specifies the layout of the values calculates by funcs.
  The input DM, attached to U, may be different. For example, you can input the solution over the full domain, but output over a piece of the boundary, or
  a subdomain. You can also output a different number of fields than the input, with different discretizations. Last the auxiliary DM, attached to the
  auxiliary field vector, which is attached to dm, can also be different. It can have a different topology, number of fields, and discretizations.

  Level: intermediate

.seealso: DMProjectField(), DMProjectFieldLabelLocal(), DMProjectFunction(), DMComputeL2Diff()
@*/
PetscErrorCode DMProjectBdFieldLabelLocal(DM dm, PetscReal time, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscInt Nc, const PetscInt comps[], Vec localU,
                                          void (**funcs)(PetscInt, PetscInt, PetscInt,
                                                         const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                         const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                         PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                          InsertMode mode, Vec localX)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(localU,VEC_CLASSID,8);
  PetscValidHeaderSpecific(localX,VEC_CLASSID,11);
  PetscCheckFalse(!dm->ops->projectbdfieldlabellocal,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMProjectBdFieldLabelLocal",((PetscObject)dm)->type_name);
  ierr = (dm->ops->projectbdfieldlabellocal)(dm, time, label, numIds, ids, Nc, comps, localU, funcs, mode, localX);CHKERRQ(ierr);
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
  PetscCheckFalse(!dm->ops->computel2diff,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMComputeL2Diff",((PetscObject)dm)->type_name);
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
  PetscCheckFalse(!dm->ops->computel2gradientdiff,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMComputeL2GradientDiff",((PetscObject)dm)->type_name);
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
  PetscCheckFalse(!dm->ops->computel2fielddiff,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMComputeL2FieldDiff",((PetscObject)dm)->type_name);
  ierr = (dm->ops->computel2fielddiff)(dm,time,funcs,ctxs,X,diff);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
 DMGetNeighbors - Gets an array containing the MPI rank of all the processes neighbors

 Not Collective

 Input Parameter:
.  dm    - The DM

 Output Parameters:
+  nranks - the number of neighbours
-  ranks - the neighbors ranks

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
  PetscCheckFalse(!dm->ops->getneighbors,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM type %s does not implement DMGetNeighbors",((PetscObject)dm)->type_name);
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
    PetscCheckFalse(!dm,PetscObjectComm((PetscObject)J),PETSC_ERR_ARG_INCOMP,"IS_COLORING_LOCAL requires a DM");
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
+    dm1 - the first DM
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
    of both dm and dmc (if they are of different types), attempting to determine
    compatibility. It is left to DM implementers to ensure that symmetry is
    preserved. The simplest way to do this is, when implementing type-specific
    logic for this function, is to check for existing logic in the implementation
    of other DM types and let *set = PETSC_FALSE if found.

    Level: advanced

.seealso: DM, DMDACreateCompatibleDMDA(), DMStagCreateCompatibleDMStag()
@*/

PetscErrorCode DMGetCompatibility(DM dm1,DM dm2,PetscBool *compatible,PetscBool *set)
{
  PetscErrorCode ierr;
  PetscMPIInt    compareResult;
  DMType         type,type2;
  PetscBool      sameType;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm1,DM_CLASSID,1);
  PetscValidHeaderSpecific(dm2,DM_CLASSID,2);

  /* Declare a DM compatible with itself */
  if (dm1 == dm2) {
    *set = PETSC_TRUE;
    *compatible = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  /* Declare a DM incompatible with a DM that lives on an "unequal"
     communicator. Note that this does not preclude compatibility with
     DMs living on "congruent" or "similar" communicators, but this must be
     determined by the implementation-specific logic */
  ierr = MPI_Comm_compare(PetscObjectComm((PetscObject)dm1),PetscObjectComm((PetscObject)dm2),&compareResult);CHKERRMPI(ierr);
  if (compareResult == MPI_UNEQUAL) {
    *set = PETSC_TRUE;
    *compatible = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  /* Pass to the implementation-specific routine, if one exists. */
  if (dm1->ops->getcompatibility) {
    ierr = (*dm1->ops->getcompatibility)(dm1,dm2,compatible,set);CHKERRQ(ierr);
    if (*set) PetscFunctionReturn(0);
  }

  /* If dm1 and dm2 are of different types, then attempt to check compatibility
     with an implementation of this function from dm2 */
  ierr = DMGetType(dm1,&type);CHKERRQ(ierr);
  ierr = DMGetType(dm2,&type2);CHKERRQ(ierr);
  ierr = PetscStrcmp(type,type2,&sameType);CHKERRQ(ierr);
  if (!sameType && dm2->ops->getcompatibility) {
    ierr = (*dm2->ops->getcompatibility)(dm2,dm1,compatible,set);CHKERRQ(ierr); /* Note argument order */
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
  PetscCheckFalse(dm->numbermonitors >= MAXDMMONITORS,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Too many monitors set");
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

/*@
  DMComputeError - Computes the error assuming the user has given exact solution functions

  Collective on DM

  Input Parameters:
+ dm     - The DM
- sol    - The solution vector

  Input/Output Parameter:
. errors - An array of length Nf, the number of fields, or NULL for no output; on output
           contains the error in each field

  Output Parameter:
. errorVec - A vector to hold the cellwise error (may be NULL)

  Note: The exact solutions come from the PetscDS object, and the time comes from DMGetOutputSequenceNumber().

  Level: developer

.seealso: DMMonitorSet(), DMGetRegionNumDS(), PetscDSGetExactSolution(), DMGetOutputSequenceNumber()
@*/
PetscErrorCode DMComputeError(DM dm, Vec sol, PetscReal errors[], Vec *errorVec)
{
  PetscErrorCode (**exactSol)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
  void            **ctxs;
  PetscReal         time;
  PetscInt          Nf, f, Nds, s;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  ierr = PetscCalloc2(Nf, &exactSol, Nf, &ctxs);CHKERRQ(ierr);
  ierr = DMGetNumDS(dm, &Nds);CHKERRQ(ierr);
  for (s = 0; s < Nds; ++s) {
    PetscDS         ds;
    DMLabel         label;
    IS              fieldIS;
    const PetscInt *fields;
    PetscInt        dsNf;

    ierr = DMGetRegionNumDS(dm, s, &label, &fieldIS, &ds);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(ds, &dsNf);CHKERRQ(ierr);
    if (fieldIS) {ierr = ISGetIndices(fieldIS, &fields);CHKERRQ(ierr);}
    for (f = 0; f < dsNf; ++f) {
      const PetscInt field = fields[f];
      ierr = PetscDSGetExactSolution(ds, field, &exactSol[field], &ctxs[field]);CHKERRQ(ierr);
    }
    if (fieldIS) {ierr = ISRestoreIndices(fieldIS, &fields);CHKERRQ(ierr);}
  }
  for (f = 0; f < Nf; ++f) {
    PetscCheckFalse(!exactSol[f],PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "DS must contain exact solution functions in order to calculate error, missing for field %D", f);
  }
  ierr = DMGetOutputSequenceNumber(dm, NULL, &time);CHKERRQ(ierr);
  if (errors) {ierr = DMComputeL2FieldDiff(dm, time, exactSol, ctxs, sol, errors);CHKERRQ(ierr);}
  if (errorVec) {
    DM             edm;
    DMPolytopeType ct;
    PetscBool      simplex;
    PetscInt       dim, cStart, Nf;

    ierr = DMClone(dm, &edm);CHKERRQ(ierr);
    ierr = DMGetDimension(edm, &dim);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
    ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
    simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
    ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {
      PetscFE         fe, efe;
      PetscQuadrature q;
      const char     *name;

      ierr = DMGetField(dm, f, NULL, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscFECreateLagrange(PETSC_COMM_SELF, dim, Nf, simplex, 0, PETSC_DETERMINE, &efe);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject) fe, &name);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) efe, name);CHKERRQ(ierr);
      ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
      ierr = PetscFESetQuadrature(efe, q);CHKERRQ(ierr);
      ierr = DMSetField(edm, f, NULL, (PetscObject) efe);CHKERRQ(ierr);
      ierr = PetscFEDestroy(&efe);CHKERRQ(ierr);
    }
    ierr = DMCreateDS(edm);CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(edm, errorVec);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) *errorVec, "Error");CHKERRQ(ierr);
    ierr = DMPlexComputeL2DiffVec(dm, time, exactSol, ctxs, sol, *errorVec);CHKERRQ(ierr);
    ierr = DMDestroy(&edm);CHKERRQ(ierr);
  }
  ierr = PetscFree2(exactSol, ctxs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMGetNumAuxiliaryVec - Get the number of auxiliary vectors associated with this DM

  Not collective

  Input Parameter:
. dm     - The DM

  Output Parameter:
. numAux - The number of auxiliary data vectors

  Level: advanced

.seealso: DMGetAuxiliaryLabels(), DMGetAuxiliaryVec(), DMSetAuxiliaryVec()
@*/
PetscErrorCode DMGetNumAuxiliaryVec(DM dm, PetscInt *numAux)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscHMapAuxGetSize(dm->auxData, numAux);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMGetAuxiliaryVec - Get the auxiliary vector for region specified by the given label and value

  Not collective

  Input Parameters:
+ dm     - The DM
. label  - The DMLabel
- value  - The label value indicating the region

  Output Parameter:
. aux    - The Vec holding auxiliary field data

  Note: If no auxiliary vector is found for this (label, value), (NULL, 0) is checked as well.

  Level: advanced

.seealso: DMSetAuxiliaryVec(), DMGetNumAuxiliaryVec()
@*/
PetscErrorCode DMGetAuxiliaryVec(DM dm, DMLabel label, PetscInt value, Vec *aux)
{
  PetscHashAuxKey key, wild = {NULL, 0};
  PetscBool       has;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (label) PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 2);
  key.label = label;
  key.value = value;
  ierr = PetscHMapAuxHas(dm->auxData, key, &has);CHKERRQ(ierr);
  if (has) {ierr = PetscHMapAuxGet(dm->auxData, key,  aux);CHKERRQ(ierr);}
  else     {ierr = PetscHMapAuxGet(dm->auxData, wild, aux);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  DMSetAuxiliaryVec - Set the auxiliary vector for region specified by the given label and value

  Not collective

  Input Parameters:
+ dm     - The DM
. label  - The DMLabel
. value  - The label value indicating the region
- aux    - The Vec holding auxiliary field data

  Level: advanced

.seealso: DMGetAuxiliaryVec()
@*/
PetscErrorCode DMSetAuxiliaryVec(DM dm, DMLabel label, PetscInt value, Vec aux)
{
  Vec             old;
  PetscHashAuxKey key;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (label) PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 2);
  key.label = label;
  key.value = value;
  ierr = PetscHMapAuxGet(dm->auxData, key, &old);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) aux);CHKERRQ(ierr);
  ierr = PetscObjectDereference((PetscObject) old);CHKERRQ(ierr);
  if (!aux) {ierr = PetscHMapAuxDel(dm->auxData, key);CHKERRQ(ierr);}
  else      {ierr = PetscHMapAuxSet(dm->auxData, key, aux);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
  DMGetAuxiliaryLabels - Get the labels and values for all auxiliary vectors in this DM

  Not collective

  Input Parameter:
. dm      - The DM

  Output Parameters:
+ labels  - The DMLabels for each Vec
- values  - The label values for each Vec

  Note: The arrays passed in must be at least as large as DMGetNumAuxiliaryVec().

  Level: advanced

.seealso: DMGetNumAuxiliaryVec(), DMGetAuxiliaryVec(), DMSetAuxiliaryVec()
@*/
PetscErrorCode DMGetAuxiliaryLabels(DM dm, DMLabel labels[], PetscInt values[])
{
  PetscHashAuxKey *keys;
  PetscInt         n, i, off = 0;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(labels, 2);
  PetscValidPointer(values, 3);
  ierr = DMGetNumAuxiliaryVec(dm, &n);CHKERRQ(ierr);
  ierr = PetscMalloc1(n, &keys);CHKERRQ(ierr);
  ierr = PetscHMapAuxGetKeys(dm->auxData, &off, keys);CHKERRQ(ierr);
  for (i = 0; i < n; ++i) {labels[i] = keys[i].label; values[i] = keys[i].value;}
  ierr = PetscFree(keys);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMCopyAuxiliaryVec - Copy the auxiliary data to a new DM

  Not collective

  Input Parameter:
. dm    - The DM

  Output Parameter:
. dmNew - The new DM, now with the same auxiliary data

  Level: advanced

.seealso: DMGetNumAuxiliaryVec(), DMGetAuxiliaryVec(), DMSetAuxiliaryVec()
@*/
PetscErrorCode DMCopyAuxiliaryVec(DM dm, DM dmNew)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscHMapAuxDestroy(&dmNew->auxData);CHKERRQ(ierr);
  ierr = PetscHMapAuxDuplicate(dm->auxData, &dmNew->auxData);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPolytopeMatchOrientation - Determine an orientation that takes the source face arrangement to the target face arrangement

  Not collective

  Input Parameters:
+ ct         - The DMPolytopeType
. sourceCone - The source arrangement of faces
- targetCone - The target arrangement of faces

  Output Parameters:
+ ornt  - The orientation which will take the source arrangement to the target arrangement
- found - Flag indicating that a suitable orientation was found

  Level: advanced

.seealso: DMPolytopeGetOrientation(), DMPolytopeMatchVertexOrientation()
@*/
PetscErrorCode DMPolytopeMatchOrientation(DMPolytopeType ct, const PetscInt sourceCone[], const PetscInt targetCone[], PetscInt *ornt, PetscBool *found)
{
  const PetscInt cS = DMPolytopeTypeGetConeSize(ct);
  const PetscInt nO = DMPolytopeTypeGetNumArrangments(ct)/2;
  PetscInt       o, c;

  PetscFunctionBegin;
  if (!nO) {*ornt = 0; *found = PETSC_TRUE; PetscFunctionReturn(0);}
  for (o = -nO; o < nO; ++o) {
    const PetscInt *arr = DMPolytopeTypeGetArrangment(ct, o);

    for (c = 0; c < cS; ++c) if (sourceCone[arr[c*2]] != targetCone[c]) break;
    if (c == cS) {*ornt = o; break;}
  }
  *found = o == nO ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  DMPolytopeGetOrientation - Determine an orientation that takes the source face arrangement to the target face arrangement

  Not collective

  Input Parameters:
+ ct         - The DMPolytopeType
. sourceCone - The source arrangement of faces
- targetCone - The target arrangement of faces

  Output Parameters:
. ornt  - The orientation which will take the source arrangement to the target arrangement

  Note: This function will fail if no suitable orientation can be found.

  Level: advanced

.seealso: DMPolytopeMatchOrientation(), DMPolytopeGetVertexOrientation()
@*/
PetscErrorCode DMPolytopeGetOrientation(DMPolytopeType ct, const PetscInt sourceCone[], const PetscInt targetCone[], PetscInt *ornt)
{
  PetscBool      found;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPolytopeMatchOrientation(ct, sourceCone, targetCone, ornt, &found);CHKERRQ(ierr);
  PetscCheckFalse(!found,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not find orientation for %s", DMPolytopeTypes[ct]);
  PetscFunctionReturn(0);
}

/*@C
  DMPolytopeMatchVertexOrientation - Determine an orientation that takes the source vertex arrangement to the target vertex arrangement

  Not collective

  Input Parameters:
+ ct         - The DMPolytopeType
. sourceVert - The source arrangement of vertices
- targetVert - The target arrangement of vertices

  Output Parameters:
+ ornt  - The orientation which will take the source arrangement to the target arrangement
- found - Flag indicating that a suitable orientation was found

  Level: advanced

.seealso: DMPolytopeGetOrientation(), DMPolytopeMatchOrientation()
@*/
PetscErrorCode DMPolytopeMatchVertexOrientation(DMPolytopeType ct, const PetscInt sourceVert[], const PetscInt targetVert[], PetscInt *ornt, PetscBool *found)
{
  const PetscInt cS = DMPolytopeTypeGetNumVertices(ct);
  const PetscInt nO = DMPolytopeTypeGetNumArrangments(ct)/2;
  PetscInt       o, c;

  PetscFunctionBegin;
  if (!nO) {*ornt = 0; *found = PETSC_TRUE; PetscFunctionReturn(0);}
  for (o = -nO; o < nO; ++o) {
    const PetscInt *arr = DMPolytopeTypeGetVertexArrangment(ct, o);

    for (c = 0; c < cS; ++c) if (sourceVert[arr[c]] != targetVert[c]) break;
    if (c == cS) {*ornt = o; break;}
  }
  *found = o == nO ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  DMPolytopeGetVertexOrientation - Determine an orientation that takes the source vertex arrangement to the target vertex arrangement

  Not collective

  Input Parameters:
+ ct         - The DMPolytopeType
. sourceCone - The source arrangement of vertices
- targetCone - The target arrangement of vertices

  Output Parameters:
. ornt  - The orientation which will take the source arrangement to the target arrangement

  Note: This function will fail if no suitable orientation can be found.

  Level: advanced

.seealso: DMPolytopeMatchVertexOrientation(), DMPolytopeGetOrientation()
@*/
PetscErrorCode DMPolytopeGetVertexOrientation(DMPolytopeType ct, const PetscInt sourceCone[], const PetscInt targetCone[], PetscInt *ornt)
{
  PetscBool      found;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPolytopeMatchVertexOrientation(ct, sourceCone, targetCone, ornt, &found);CHKERRQ(ierr);
  PetscCheckFalse(!found,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not find orientation for %s", DMPolytopeTypes[ct]);
  PetscFunctionReturn(0);
}

/*@C
  DMPolytopeInCellTest - Check whether a point lies inside the reference cell of given type

  Not collective

  Input Parameters:
+ ct    - The DMPolytopeType
- point - Coordinates of the point

  Output Parameters:
. inside  - Flag indicating whether the point is inside the reference cell of given type

  Level: advanced

.seealso: DMLocatePoints()
@*/
PetscErrorCode DMPolytopeInCellTest(DMPolytopeType ct, const PetscReal point[], PetscBool *inside)
{
  PetscReal sum = 0.0;
  PetscInt  d;

  PetscFunctionBegin;
  *inside = PETSC_TRUE;
  switch (ct) {
  case DM_POLYTOPE_TRIANGLE:
  case DM_POLYTOPE_TETRAHEDRON:
    for (d = 0; d < DMPolytopeTypeGetDim(ct); ++d) {
      if (point[d] < -1.0) {*inside = PETSC_FALSE; break;}
      sum += point[d];
    }
    if (sum > PETSC_SMALL) {*inside = PETSC_FALSE; break;}
    break;
  case DM_POLYTOPE_QUADRILATERAL:
  case DM_POLYTOPE_HEXAHEDRON:
    for (d = 0; d < DMPolytopeTypeGetDim(ct); ++d)
      if (PetscAbsReal(point[d]) > 1.+PETSC_SMALL) {*inside = PETSC_FALSE; break;}
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported polytope type %s", DMPolytopeTypes[ct]);
  }
  PetscFunctionReturn(0);
}
