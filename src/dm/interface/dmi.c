#include <petsc/private/dmimpl.h>     /*I      "petscdm.h"     I*/
#include <petscds.h>

PetscErrorCode DMCreateGlobalVector_Section_Private(DM dm,Vec *vec)
{
  PetscSection   gSection;
  PetscInt       localSize, bs, blockSize = -1, pStart, pEnd, p;
  PetscErrorCode ierr;
  PetscInt       in[2],out[2];

  PetscFunctionBegin;
  ierr = DMGetDefaultGlobalSection(dm, &gSection);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(gSection, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, cdof;

    ierr = PetscSectionGetDof(gSection, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(gSection, p, &cdof);CHKERRQ(ierr);

    if (dof > 0) {
      if (blockSize < 0 && dof-cdof > 0) {
        /* set blockSize */
        blockSize = dof-cdof;
      } else if (dof-cdof != blockSize) {
        /* non-identical blockSize, set it as 1 */
        blockSize = 1;
        break;
      }
    }
  }

  in[0] = blockSize < 0 ? PETSC_MIN_INT : -blockSize;
  in[1] = blockSize;
  ierr = MPIU_Allreduce(in,out,2,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  /* -out[0] = min(blockSize), out[1] = max(blockSize) */
  if (-out[0] == out[1]) {
    bs = out[1];
  } else bs = 1;

  if (bs < 0) { /* Everyone was empty */
    blockSize = 1;
    bs        = 1;
  }

  ierr = PetscSectionGetConstrainedStorageSize(gSection, &localSize);CHKERRQ(ierr);
  if (localSize%blockSize) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Mismatch between blocksize %d and local storage size %d", blockSize, localSize);
  ierr = VecCreate(PetscObjectComm((PetscObject)dm), vec);CHKERRQ(ierr);
  ierr = VecSetSizes(*vec, localSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*vec, bs);CHKERRQ(ierr);
  ierr = VecSetType(*vec,dm->vectype);CHKERRQ(ierr);
  ierr = VecSetDM(*vec, dm);CHKERRQ(ierr);
  /* ierr = VecSetLocalToGlobalMapping(*vec, dm->ltogmap);CHKERRQ(ierr); */
  PetscFunctionReturn(0);
}

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
    PetscInt bs = -1, bsLocal, bsMax, bsMin;
    ierr = PetscSectionGetChart(sectionGlobal, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt gdof, pSubSize  = 0;

      ierr = PetscSectionGetDof(sectionGlobal, p, &gdof);CHKERRQ(ierr);
      if (gdof > 0) {
        for (f = 0; f < numFields; ++f) {
          PetscInt fdof, fcdof;

          ierr     = PetscSectionGetFieldDof(section, p, fields[f], &fdof);CHKERRQ(ierr);
          ierr     = PetscSectionGetFieldConstraintDof(section, p, fields[f], &fcdof);CHKERRQ(ierr);
          pSubSize += fdof-fcdof;
        }
        subSize += pSubSize;
        if (pSubSize) {
          if (bs < 0) {
            bs = pSubSize;
          } else if (bs != pSubSize) {
            /* Layout does not admit a pointwise block size */
            bs = 1;
          }
        }
      }
    }
    /* Must have same blocksize on all procs (some might have no points) */
    bsLocal = bs;
    ierr = MPIU_Allreduce(&bsLocal, &bsMax, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
    bsLocal = bs < 0 ? bsMax : bs;
    ierr = MPIU_Allreduce(&bsLocal, &bsMin, 1, MPIU_INT, MPI_MIN, PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
    if (bsMin != bsMax) {
      bs = 1;
    } else {
      bs = bsMax;
    }
    ierr = PetscMalloc1(subSize, &subIndices);CHKERRQ(ierr);
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
    ierr = ISSetBlockSize(*is, bs);CHKERRQ(ierr);
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
    if (haveNull && is) {
      MatNullSpace nullSpace;

      ierr = (*(*subdm)->nullspaceConstructors[nf])(*subdm, nf, &nullSpace);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) *is, "nullspace", (PetscObject) nullSpace);CHKERRQ(ierr);
      ierr = MatNullSpaceDestroy(&nullSpace);CHKERRQ(ierr);
    }
    if (dm->prob) {
      PetscInt Nf;

      ierr = PetscDSGetNumFields(dm->prob, &Nf);CHKERRQ(ierr);
      if (nF != Nf) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "The number of DM fields %d does not match the number of Section fields %d", Nf, nF);
      ierr = DMSetNumFields(*subdm, numFields);CHKERRQ(ierr);
      for (f = 0; f < numFields; ++f) {
        PetscObject disc;

        ierr = DMGetField(dm, fields[f], &disc);CHKERRQ(ierr);
        ierr = DMSetField(*subdm, f, disc);CHKERRQ(ierr);
      }
      if (numFields == 1 && is) {
        PetscObject disc, space, pmat;

        ierr = DMGetField(*subdm, 0, &disc);CHKERRQ(ierr);
        ierr = PetscObjectQuery(disc, "nullspace", &space);CHKERRQ(ierr);
        if (space) {ierr = PetscObjectCompose((PetscObject) *is, "nullspace", space);CHKERRQ(ierr);}
        ierr = PetscObjectQuery(disc, "nearnullspace", &space);CHKERRQ(ierr);
        if (space) {ierr = PetscObjectCompose((PetscObject) *is, "nearnullspace", space);CHKERRQ(ierr);}
        ierr = PetscObjectQuery(disc, "pmat", &pmat);CHKERRQ(ierr);
        if (pmat) {ierr = PetscObjectCompose((PetscObject) *is, "pmat", pmat);CHKERRQ(ierr);}
      }
    }
  }
#if 0
  /* We need a way to filter the original SF for given fields:
       - Keeping the original section around it too much I think
       - We could keep the distributed section, and subset it
   */
  if (dm->sfNatural) {
    PetscSF sfNatural;

    ierr = PetscSectionCreateSubsection(dm->originalSection, numFields, fields, &(*subdm)->originalSection);CHKERRQ(ierr);
    ierr = DMPlexCreateGlobalToNaturalPetscSF(*subdm, &sfNatural);CHKERRQ(ierr);
    ierr = DMPlexSetGlobalToNaturalPetscSF(*subdm, sfNatural);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}
