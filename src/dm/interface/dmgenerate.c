#include <petsc/private/dmimpl.h>           /*I      "petscdm.h"          I*/

PETSC_EXTERN PetscErrorCode DMIsForest(DM,PetscBool*);

DMGeneratorFunctionList DMGenerateList = NULL;
PetscBool DMGenerateRegisterAllCalled = PETSC_FALSE;

#if defined(PETSC_HAVE_TRIANGLE)
PETSC_EXTERN PetscErrorCode DMPlexGenerate_Triangle(DM, PetscBool, DM*);
PETSC_EXTERN PetscErrorCode DMPlexRefine_Triangle(DM, double*, DM*);
#endif
#if defined(PETSC_HAVE_TETGEN)
PETSC_EXTERN PetscErrorCode DMPlexGenerate_Tetgen(DM, PetscBool, DM*);
PETSC_EXTERN PetscErrorCode DMPlexRefine_Tetgen(DM, double*, DM*);
#endif
#if defined(PETSC_HAVE_CTETGEN)
PETSC_EXTERN PetscErrorCode DMPlexGenerate_CTetgen(DM, PetscBool, DM*);
PETSC_EXTERN PetscErrorCode DMPlexRefine_CTetgen(DM, double*, DM*);
#endif
#if defined(PETSC_HAVE_PRAGMATIC)
PETSC_EXTERN PetscErrorCode DMAdaptMetric_Pragmatic_Plex(DM, Vec, DMLabel, DMLabel, DM*);
#endif
#if defined(PETSC_HAVE_MMG)
PETSC_EXTERN PetscErrorCode DMAdaptMetric_Mmg_Plex(DM, Vec, DMLabel, DMLabel, DM*);
#endif
#if defined(PETSC_HAVE_PARMMG)
PETSC_EXTERN PetscErrorCode DMAdaptMetric_ParMmg_Plex(DM, Vec, DMLabel, DMLabel, DM*);
#endif
PETSC_EXTERN PetscErrorCode DMPlexTransformAdaptLabel(DM, Vec, DMLabel, DMLabel, DM*);
PETSC_EXTERN PetscErrorCode DMAdaptLabel_Forest(DM, Vec, DMLabel, DMLabel, DM*);

/*@C
  DMGenerateRegisterAll - Registers all of the mesh generation methods in the DM package.

  Not Collective

  Level: advanced

.seealso:  DMGenerateRegisterDestroy()
@*/
PetscErrorCode DMGenerateRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (DMGenerateRegisterAllCalled) PetscFunctionReturn(0);
  DMGenerateRegisterAllCalled = PETSC_TRUE;
#if defined(PETSC_HAVE_TRIANGLE)
  ierr = DMGenerateRegister("triangle",DMPlexGenerate_Triangle,DMPlexRefine_Triangle,NULL,1);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_CTETGEN)
  ierr = DMGenerateRegister("ctetgen",DMPlexGenerate_CTetgen,DMPlexRefine_CTetgen,NULL,2);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_TETGEN)
  ierr = DMGenerateRegister("tetgen",DMPlexGenerate_Tetgen,DMPlexRefine_Tetgen,NULL,2);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_PRAGMATIC)
  ierr = DMGenerateRegister("pragmatic",NULL,NULL,DMAdaptMetric_Pragmatic_Plex,-1);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MMG)
  ierr = DMGenerateRegister("mmg",NULL,NULL,DMAdaptMetric_Mmg_Plex,-1);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_PARMMG)
  ierr = DMGenerateRegister("parmmg",NULL,NULL,DMAdaptMetric_ParMmg_Plex,-1);CHKERRQ(ierr);
#endif
  ierr = DMGenerateRegister("cellrefiner",NULL,NULL,DMPlexTransformAdaptLabel,-1);CHKERRQ(ierr);
  ierr = DMGenerateRegister("forest",NULL,NULL,DMAdaptLabel_Forest,-1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMGenerateRegister -  Adds a grid generator to DM

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined grid generator
.  fnc - generator function
.  rfnc - refinement function
.  alfnc - adapt by label function
-  dim - dimension of boundary of domain

   Notes:
   DMGenerateRegister() may be called multiple times to add several user-defined solvers.

   Sample usage:
.vb
   DMGenerateRegister("my_generator",MyGeneratorCreate,MyGeneratorRefiner,MyGeneratorAdaptor,dim);
.ve

   Then, your generator can be chosen with the procedural interface via
$     DMGenerate(dm,"my_generator",...)
   or at runtime via the option
$     -dm_generator my_generator

   Level: advanced

.seealso: DMGenerateRegisterAll(), DMPlexGenerate(), DMGenerateRegisterDestroy()

@*/
PetscErrorCode DMGenerateRegister(const char sname[], PetscErrorCode (*fnc)(DM, PetscBool, DM*), PetscErrorCode (*rfnc)(DM, PetscReal*, DM*), PetscErrorCode (*alfnc)(DM, Vec, DMLabel, DMLabel, DM*), PetscInt dim)
{
  DMGeneratorFunctionList entry;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&entry);CHKERRQ(ierr);
  ierr = PetscStrallocpy(sname,&entry->name);CHKERRQ(ierr);
  entry->generate = fnc;
  entry->refine   = rfnc;
  entry->adapt    = alfnc;
  entry->dim      = dim;
  entry->next     = NULL;
  if (!DMGenerateList) DMGenerateList = entry;
  else {
    DMGeneratorFunctionList fl = DMGenerateList;
    while (fl->next) fl = fl->next;
    fl->next = entry;
  }
  PetscFunctionReturn(0);
}

extern PetscBool DMGenerateRegisterAllCalled;

PetscErrorCode DMGenerateRegisterDestroy(void)
{
  DMGeneratorFunctionList next, fl;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  next = fl = DMGenerateList;
  while (next) {
    next = fl ? fl->next : NULL;
    if (fl) {ierr = PetscFree(fl->name);CHKERRQ(ierr);}
    ierr = PetscFree(fl);CHKERRQ(ierr);
    fl = next;
  }
  DMGenerateList              = NULL;
  DMGenerateRegisterAllCalled = PETSC_FALSE;
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
  DMGeneratorFunctionList fl;
  char                    adaptname[PETSC_MAX_PATH_LEN];
  const char             *name;
  PetscInt                dim;
  PetscBool               flg, isForest, found = PETSC_FALSE;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (label) PetscValidPointer(label, 2);
  PetscValidPointer(dmAdapt, 3);
  *dmAdapt = NULL;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMIsForest(dm, &isForest);CHKERRQ(ierr);
  name = isForest ? "forest" : "cellrefiner";
  ierr = PetscOptionsGetString(((PetscObject) dm)->options, ((PetscObject) dm)->prefix, "-dm_adaptor", adaptname, sizeof(adaptname), &flg);CHKERRQ(ierr);
  if (flg) name = adaptname;

  fl = DMGenerateList;
  while (fl) {
    ierr = PetscStrcmp(fl->name, name, &flg);CHKERRQ(ierr);
    if (flg) {
      ierr = (*fl->adapt)(dm, NULL, label, NULL, dmAdapt);CHKERRQ(ierr);
      found = PETSC_TRUE;
    }
    fl = fl->next;
  }
  PetscAssertFalse(!found,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Grid adaptor %s not registered; you may need to add --download-%s to your ./configure options", name, name);
  if (*dmAdapt) {
    (*dmAdapt)->prealloc_only = dm->prealloc_only;  /* maybe this should go .... */
    ierr = PetscFree((*dmAdapt)->vectype);CHKERRQ(ierr);
    ierr = PetscStrallocpy(dm->vectype,(char**)&(*dmAdapt)->vectype);CHKERRQ(ierr);
    ierr = PetscFree((*dmAdapt)->mattype);CHKERRQ(ierr);
    ierr = PetscStrallocpy(dm->mattype,(char**)&(*dmAdapt)->mattype);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMAdaptMetric - Generates a mesh adapted to the specified metric field.

  Input Parameters:
+ dm - The DM object
. metric - The metric to which the mesh is adapted, defined vertex-wise.
. bdLabel - Label for boundary tags, which will be preserved in the output mesh. bdLabel should be NULL if there is no such label, and should be different from "_boundary_".
- rgLabel - Label for cell tags, which will be preserved in the output mesh. rgLabel should be NULL if there is no such label, and should be different from "_regions_".

  Output Parameter:
. dmAdapt  - Pointer to the DM object containing the adapted mesh

  Note: The label in the adapted mesh will be registered under the name of the input DMLabel object

  Level: advanced

.seealso: DMAdaptLabel(), DMCoarsen(), DMRefine()
@*/
PetscErrorCode DMAdaptMetric(DM dm, Vec metric, DMLabel bdLabel, DMLabel rgLabel, DM *dmAdapt)
{
  DMGeneratorFunctionList fl;
  char                    adaptname[PETSC_MAX_PATH_LEN];
  const char             *name;
  const char * const      adaptors[3] = {"pragmatic", "mmg", "parmmg"};
  PetscInt                dim;
  PetscBool               flg, found = PETSC_FALSE;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(metric, VEC_CLASSID, 2);
  if (bdLabel) PetscValidPointer(bdLabel, 3);
  if (rgLabel) PetscValidPointer(rgLabel, 4);
  PetscValidPointer(dmAdapt, 5);
  *dmAdapt = NULL;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(((PetscObject) dm)->options, ((PetscObject) dm)->prefix, "-dm_adaptor", adaptname, sizeof(adaptname), &flg);CHKERRQ(ierr);

  /* Default to Mmg in serial and ParMmg in parallel */
  if (flg) name = adaptname;
  else {
    MPI_Comm                comm;
    PetscMPIInt             size;

    ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
    if (size == 1) name = adaptors[1];
    else           name = adaptors[2];
  }

  fl = DMGenerateList;
  while (fl) {
    ierr = PetscStrcmp(fl->name, name, &flg);CHKERRQ(ierr);
    if (flg) {
      ierr = (*fl->adapt)(dm, metric, bdLabel, rgLabel, dmAdapt);CHKERRQ(ierr);
      found = PETSC_TRUE;
    }
    fl = fl->next;
  }
  PetscAssertFalse(!found,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Grid adaptor %s not registered; you may need to add --download-%s to your ./configure options", name, name);
  if (*dmAdapt) {
    (*dmAdapt)->prealloc_only = dm->prealloc_only;  /* maybe this should go .... */
    ierr = PetscFree((*dmAdapt)->vectype);CHKERRQ(ierr);
    ierr = PetscStrallocpy(dm->vectype,(char**)&(*dmAdapt)->vectype);CHKERRQ(ierr);
    ierr = PetscFree((*dmAdapt)->mattype);CHKERRQ(ierr);
    ierr = PetscStrallocpy(dm->mattype,(char**)&(*dmAdapt)->mattype);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
