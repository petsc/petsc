#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <../src/sys/utils/hash.h>
#include <petsc-private/isimpl.h>
#include <petscsf.h>

/** hierarchy routines */

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetReferenceTree"
/*@
  DMPlexSetReferenceTree - set the reference tree for hierarchically non-conforming meshes.

  Not collective

  Input Parameters:
+ dm - The DMPlex object
- ref - The reference tree DMPlex object

  Level: intermediate

.seealso: DMPlexGetReferenceTree()
@*/
PetscErrorCode DMPlexSetReferenceTree(DM dm, DM ref)
{
  DM_Plex        *mesh = (DM_Plex *)dm->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(ref, DM_CLASSID, 2);
  ierr = PetscObjectReference((PetscObject)ref);CHKERRQ(ierr);
  ierr = DMDestroy(&mesh->referenceTree);CHKERRQ(ierr);
  mesh->referenceTree = ref;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetReferenceTree"
/*@
  DMPlexGetReferenceTree - get the reference tree for hierarchically non-conforming meshes.

  Not collective

  Input Parameters:
. dm - The DMPlex object

  Output Parameters
. ref - The reference tree DMPlex object

  Level: intermediate

.seealso: DMPlexSetReferenceTree()
@*/
PetscErrorCode DMPlexGetReferenceTree(DM dm, DM *ref)
{
  DM_Plex        *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(ref,2);
  *ref = mesh->referenceTree;
  PetscFunctionReturn(0);
}

