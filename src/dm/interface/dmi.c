#include <petsc-private/dmimpl.h>     /*I      "petscdm.h"     I*/

#undef __FUNCT__
#define __FUNCT__ "DMCreateGlobalVector_Section_Private"
PetscErrorCode DMCreateGlobalVector_Section_Private(DM dm,Vec *vec)
{
  PetscSection   gSection;
  PetscInt       localSize, bs, blockSize = -1, pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDefaultGlobalSection(dm, &gSection);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(gSection, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, cdof;

    ierr = PetscSectionGetDof(gSection, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(gSection, p, &cdof);CHKERRQ(ierr);
    if ((blockSize < 0) && (dof > 0) && (dof-cdof > 0)) blockSize = dof-cdof;
    if ((dof > 0) && (dof-cdof != blockSize)) {
      blockSize = 1;
      break;
    }
  }
  if (blockSize < 0) blockSize = 1;
  ierr = MPI_Allreduce(&blockSize, &bs, 1, MPIU_INT, MPI_MIN, PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(gSection, &localSize);CHKERRQ(ierr);
  if (localSize%blockSize) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Mismatch between blocksize %d and local storage size %d", blockSize, localSize);
  ierr = VecCreate(PetscObjectComm((PetscObject)dm), vec);CHKERRQ(ierr);
  ierr = VecSetSizes(*vec, localSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*vec, bs);CHKERRQ(ierr);
  ierr = VecSetType(*vec,dm->vectype);CHKERRQ(ierr);
  ierr = VecSetDM(*vec, dm);CHKERRQ(ierr);
  /* ierr = VecSetLocalToGlobalMapping(*vec, dm->ltogmap);CHKERRQ(ierr); */
  /* ierr = VecSetLocalToGlobalMappingBlock(*vec, dm->ltogmapb);CHKERRQ(ierr); */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateLocalVector_Section_Private"
PetscErrorCode DMCreateLocalVector_Section_Private(DM dm,Vec *vec)
{
  PetscSection   section;
  PetscInt       localSize, blockSize = -1, pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    if ((blockSize < 0) && (dof > 0)) blockSize = dof;
    if ((dof > 0) && (dof != blockSize)) {
      blockSize = 1;
      break;
    }
  }
  ierr = PetscSectionGetStorageSize(section, &localSize);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, vec);CHKERRQ(ierr);
  ierr = VecSetSizes(*vec, localSize, localSize);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*vec, blockSize);CHKERRQ(ierr);
  ierr = VecSetType(*vec,dm->vectype);CHKERRQ(ierr);
  ierr = VecSetDM(*vec, dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateSubDM_Section_Private"
/* This assumes that the DM has been cloned prior to the call */
PetscErrorCode DMCreateSubDM_Section_Private(DM dm, PetscInt numFields, PetscInt fields[], IS *is, DM *subdm)
{
  PetscSection   section, sectionGlobal;
  PetscInt      *subIndices;
  PetscInt       subSize = 0, subOff = 0, nF, f, pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!numFields) PetscFunctionReturn(0);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(dm, &sectionGlobal);CHKERRQ(ierr);
  if (!section) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Must set default section for DM before splitting fields");
  if (!sectionGlobal) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Must set default global section for DM before splitting fields");
  ierr = PetscSectionGetNumFields(section, &nF);CHKERRQ(ierr);
  if (numFields > nF) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Number of requested fields %d greater than number of DM fields %d", numFields, nF);
  if (is) {
    ierr = PetscSectionGetChart(sectionGlobal, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt gdof;

      ierr = PetscSectionGetDof(sectionGlobal, p, &gdof);CHKERRQ(ierr);
      if (gdof > 0) {
        for (f = 0; f < numFields; ++f) {
          PetscInt fdof, fcdof;

          ierr     = PetscSectionGetFieldDof(section, p, fields[f], &fdof);CHKERRQ(ierr);
          ierr     = PetscSectionGetFieldConstraintDof(section, p, fields[f], &fcdof);CHKERRQ(ierr);
          subSize += fdof-fcdof;
        }
      }
    }
    ierr = PetscMalloc(subSize * sizeof(PetscInt), &subIndices);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt gdof, goff;

      ierr = PetscSectionGetDof(sectionGlobal, p, &gdof);CHKERRQ(ierr);
      if (gdof > 0) {
        ierr = PetscSectionGetOffset(sectionGlobal, p, &goff);CHKERRQ(ierr);
        for (f = 0; f < numFields; ++f) {
          PetscInt fdof, fcdof, fc, f2, poff = 0;

          /* Can get rid of this loop by storing field information in the global section */
          for (f2 = 0; f2 < fields[f]; ++f2) {
            ierr  = PetscSectionGetFieldDof(section, p, f2, &fdof);CHKERRQ(ierr);
            ierr  = PetscSectionGetFieldConstraintDof(section, p, f2, &fcdof);CHKERRQ(ierr);
            poff += fdof-fcdof;
          }
          ierr = PetscSectionGetFieldDof(section, p, fields[f], &fdof);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldConstraintDof(section, p, fields[f], &fcdof);CHKERRQ(ierr);
          for (fc = 0; fc < fdof-fcdof; ++fc, ++subOff) {
            subIndices[subOff] = goff+poff+fc;
          }
        }
      }
    }
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm), subSize, subIndices, PETSC_OWN_POINTER, is);CHKERRQ(ierr);
  }
  if (subdm) {
    PetscSection subsection;
    PetscBool    haveNull = PETSC_FALSE;
    PetscInt     f, nf = 0;

    ierr = PetscSectionCreateSubsection(section, numFields, fields, &subsection);CHKERRQ(ierr);
    ierr = DMSetDefaultSection(*subdm, subsection);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&subsection);CHKERRQ(ierr);
    for (f = 0; f < numFields; ++f) {
      (*subdm)->nullspaceConstructors[f] = dm->nullspaceConstructors[fields[f]];
      if ((*subdm)->nullspaceConstructors[f]) {
        haveNull = PETSC_TRUE;
        nf       = f;
      }
    }
    if (haveNull) {
      MatNullSpace nullSpace;

      ierr = (*(*subdm)->nullspaceConstructors[nf])(*subdm, nf, &nullSpace);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) *is, "nullspace", (PetscObject) nullSpace);CHKERRQ(ierr);
      ierr = MatNullSpaceDestroy(&nullSpace);CHKERRQ(ierr);
    }
    if (dm->fields) {
      if (nF != dm->numFields) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "The number of DM fields %d does not match the number of Section fields %d", dm->numFields, nF);
      ierr = DMSetNumFields(*subdm, numFields);CHKERRQ(ierr);
      for (f = 0; f < numFields; ++f) {
        ierr = PetscObjectListDuplicate(dm->fields[fields[f]]->olist, &(*subdm)->fields[f]->olist);CHKERRQ(ierr);
      }
      if (numFields == 1) {
        MatNullSpace space;
        Mat          pmat;

        ierr = PetscObjectQuery((*subdm)->fields[0], "nullspace", (PetscObject*) &space);CHKERRQ(ierr);
        if (space) {ierr = PetscObjectCompose((PetscObject) *is, "nullspace", (PetscObject) space);CHKERRQ(ierr);}
        ierr = PetscObjectQuery((*subdm)->fields[0], "nearnullspace", (PetscObject*) &space);CHKERRQ(ierr);
        if (space) {ierr = PetscObjectCompose((PetscObject) *is, "nearnullspace", (PetscObject) space);CHKERRQ(ierr);}
        ierr = PetscObjectQuery((*subdm)->fields[0], "pmat", (PetscObject*) &pmat);CHKERRQ(ierr);
        if (pmat) {ierr = PetscObjectCompose((PetscObject) *is, "pmat", (PetscObject) pmat);CHKERRQ(ierr);}
      }
    }
  }
  PetscFunctionReturn(0);
}
