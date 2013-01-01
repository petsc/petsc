#include <petsc-private/dmimpl.h>     /*I      "petscdm.h"     I*/

extern PetscErrorCode VecView_Seq(Vec, PetscViewer);
extern PetscErrorCode VecView_MPI(Vec, PetscViewer);

#undef __FUNCT__
#define __FUNCT__ "DMCreateGlobalVector_Section_Private"
PetscErrorCode DMCreateGlobalVector_Section_Private(DM dm,Vec *vec)
{
  PetscErrorCode ierr;
  PetscSection gSection;
  PetscInt     localSize, blockSize = -1, pStart, pEnd, p;

  PetscFunctionBegin;
  ierr = DMGetDefaultGlobalSection(dm, &gSection);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(dm->defaultSection, &pStart, &pEnd);CHKERRQ(ierr);
  for(p = pStart; p < pEnd; ++p) {
    PetscInt dof, cdof;

    ierr = PetscSectionGetDof(dm->defaultSection, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(dm->defaultSection, p, &cdof);CHKERRQ(ierr);
    if ((blockSize < 0) && (dof > 0)) blockSize = dof-cdof;
    if ((dof > 0) && (dof-cdof != blockSize)) {
      blockSize = 1;
      break;
    }
  }
  ierr = PetscSectionGetConstrainedStorageSize(dm->defaultGlobalSection, &localSize);CHKERRQ(ierr);
  if (localSize%blockSize) SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "Mismatch between blocksize %d and local storage size %d", blockSize, localSize);
  ierr = VecCreate(((PetscObject) dm)->comm, vec);CHKERRQ(ierr);
  ierr = VecSetSizes(*vec, localSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*vec, blockSize);CHKERRQ(ierr);
  /* ierr = VecSetType(*vec, dm->vectype);CHKERRQ(ierr); */
  ierr = VecSetFromOptions(*vec);CHKERRQ(ierr);
  ierr = VecSetDM(*vec, dm);CHKERRQ(ierr);
  /* ierr = VecSetLocalToGlobalMapping(*vec, dm->ltogmap);CHKERRQ(ierr); */
  /* ierr = VecSetLocalToGlobalMappingBlock(*vec, dm->ltogmapb);CHKERRQ(ierr); */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateLocalVector_Section_Private"
PetscErrorCode DMCreateLocalVector_Section_Private(DM dm,Vec *vec)
{
  PetscErrorCode ierr;
  PetscInt localSize, blockSize = -1, pStart, pEnd, p;

  PetscFunctionBegin;
  ierr = PetscSectionGetChart(dm->defaultSection, &pStart, &pEnd);CHKERRQ(ierr);
  for(p = pStart; p < pEnd; ++p) {
    PetscInt dof;

    ierr = PetscSectionGetDof(dm->defaultSection, p, &dof);CHKERRQ(ierr);
    if ((blockSize < 0) && (dof > 0)) blockSize = dof;
    if ((dof > 0) && (dof != blockSize)) {
      blockSize = 1;
      break;
    }
  }
  ierr = PetscSectionGetStorageSize(dm->defaultSection, &localSize);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, vec);CHKERRQ(ierr);
  ierr = VecSetSizes(*vec, localSize, localSize);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*vec, blockSize);CHKERRQ(ierr);
  ierr = VecSetFromOptions(*vec);CHKERRQ(ierr);
  ierr = VecSetDM(*vec, dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
