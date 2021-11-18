#include <petsc/private/dmimpl.h>     /*I      "petscdm.h"     I*/
#include <petscds.h>

PetscErrorCode DMCreateGlobalVector_Section_Private(DM dm,Vec *vec)
{
  PetscSection   gSection;
  PetscInt       localSize, bs, blockSize = -1, pStart, pEnd, p;
  PetscErrorCode ierr;
  PetscInt       in[2],out[2];

  PetscFunctionBegin;
  ierr = DMGetGlobalSection(dm, &gSection);CHKERRQ(ierr);
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
  ierr = MPIU_Allreduce(in,out,2,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)dm));CHKERRMPI(ierr);
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
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
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

/*@C
  DMCreateSectionSubDM - Returns an IS and subDM+subSection encapsulating a subproblem defined by the fields in a PetscSection in the DM.

  Not collective

  Input Parameters:
+ dm        - The DM object
. numFields - The number of fields in this subproblem
- fields    - The field numbers of the selected fields

  Output Parameters:
+ is - The global indices for the subproblem
- subdm - The DM for the subproblem, which must already have be cloned from dm

  Note: This handles all information in the DM class and the PetscSection. This is used as the basis for creating subDMs in specialized classes,
  such as Plex and Forest.

  Level: intermediate

.seealso DMCreateSubDM(), DMGetLocalSection(), DMPlexSetMigrationSF(), DMView()
@*/
PetscErrorCode DMCreateSectionSubDM(DM dm, PetscInt numFields, const PetscInt fields[], IS *is, DM *subdm)
{
  PetscSection   section, sectionGlobal;
  PetscInt      *subIndices;
  PetscInt       subSize = 0, subOff = 0, Nf, f, pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!numFields) PetscFunctionReturn(0);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dm, &sectionGlobal);CHKERRQ(ierr);
  if (!section) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Must set default section for DM before splitting fields");
  if (!sectionGlobal) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Must set default global section for DM before splitting fields");
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  if (numFields > Nf) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Number of requested fields %d greater than number of DM fields %d", numFields, Nf);
  if (is) {
    PetscInt bs, bsLocal[2], bsMinMax[2];

    for (f = 0, bs = 0; f < numFields; ++f) {
      PetscInt Nc;

      ierr = PetscSectionGetFieldComponents(section, fields[f], &Nc);CHKERRQ(ierr);
      bs  += Nc;
    }
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
        if (pSubSize && bs != pSubSize) {
          /* Layout does not admit a pointwise block size */
          bs = 1;
        }
      }
    }
    /* Must have same blocksize on all procs (some might have no points) */
    bsLocal[0] = bs < 0 ? PETSC_MAX_INT : bs; bsLocal[1] = bs;
    ierr = PetscGlobalMinMaxInt(PetscObjectComm((PetscObject) dm), bsLocal, bsMinMax);CHKERRQ(ierr);
    if (bsMinMax[0] != bsMinMax[1]) {bs = 1;}
    else                            {bs = bsMinMax[0];}
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
    if (bs > 1) {
      /* We need to check that the block size does not come from non-contiguous fields */
      PetscInt i, j, set = 1;
      for (i = 0; i < subSize; i += bs) {
        for (j = 0; j < bs; ++j) {
          if (subIndices[i+j] != subIndices[i]+j) {set = 0; break;}
        }
      }
      if (set) {ierr = ISSetBlockSize(*is, bs);CHKERRQ(ierr);}
    }
  }
  if (subdm) {
    PetscSection subsection;
    PetscBool    haveNull = PETSC_FALSE;
    PetscInt     f, nf = 0, of = 0;

    ierr = PetscSectionCreateSubsection(section, numFields, fields, &subsection);CHKERRQ(ierr);
    ierr = DMSetLocalSection(*subdm, subsection);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&subsection);CHKERRQ(ierr);
    for (f = 0; f < numFields; ++f) {
      (*subdm)->nullspaceConstructors[f] = dm->nullspaceConstructors[fields[f]];
      if ((*subdm)->nullspaceConstructors[f]) {
        haveNull = PETSC_TRUE;
        nf       = f;
        of       = fields[f];
      }
    }
    if (dm->probs) {
      ierr = DMSetNumFields(*subdm, numFields);CHKERRQ(ierr);
      for (f = 0; f < numFields; ++f) {
        PetscObject disc;

        ierr = DMGetField(dm, fields[f], NULL, &disc);CHKERRQ(ierr);
        ierr = DMSetField(*subdm, f, NULL, disc);CHKERRQ(ierr);
      }
      ierr = DMCreateDS(*subdm);CHKERRQ(ierr);
      if (numFields == 1 && is) {
        PetscObject disc, space, pmat;

        ierr = DMGetField(*subdm, 0, NULL, &disc);CHKERRQ(ierr);
        ierr = PetscObjectQuery(disc, "nullspace", &space);CHKERRQ(ierr);
        if (space) {ierr = PetscObjectCompose((PetscObject) *is, "nullspace", space);CHKERRQ(ierr);}
        ierr = PetscObjectQuery(disc, "nearnullspace", &space);CHKERRQ(ierr);
        if (space) {ierr = PetscObjectCompose((PetscObject) *is, "nearnullspace", space);CHKERRQ(ierr);}
        ierr = PetscObjectQuery(disc, "pmat", &pmat);CHKERRQ(ierr);
        if (pmat) {ierr = PetscObjectCompose((PetscObject) *is, "pmat", pmat);CHKERRQ(ierr);}
      }
      /* Check if DSes record their DM fields */
      if (dm->probs[0].fields) {
        PetscInt d, e;

        for (d = 0, e = 0; d < dm->Nds && e < (*subdm)->Nds; ++d) {
          const PetscInt  Nf = dm->probs[d].ds->Nf;
          const PetscInt *fld;
          PetscInt        f, g;

          ierr = ISGetIndices(dm->probs[d].fields, &fld);CHKERRQ(ierr);
          for (f = 0; f < Nf; ++f) {
            for (g = 0; g < numFields; ++g) if (fld[f] == fields[g]) break;
            if (g < numFields) break;
          }
          ierr = ISRestoreIndices(dm->probs[d].fields, &fld);CHKERRQ(ierr);
          if (f == Nf) continue;
          ierr = PetscDSCopyConstants(dm->probs[d].ds, (*subdm)->probs[e].ds);CHKERRQ(ierr);
          ierr = PetscDSCopyBoundary(dm->probs[d].ds, numFields, fields, (*subdm)->probs[e].ds);CHKERRQ(ierr);
          /* Translate DM fields to DS fields */
          {
            IS              infields, dsfields;
            const PetscInt *fld, *ofld;
            PetscInt       *fidx;
            PetscInt        onf, nf, f, g;

            ierr = ISCreateGeneral(PETSC_COMM_SELF, numFields, fields, PETSC_USE_POINTER, &infields);CHKERRQ(ierr);
            ierr = ISIntersect(infields, dm->probs[d].fields, &dsfields);CHKERRQ(ierr);
            ierr = ISDestroy(&infields);CHKERRQ(ierr);
            ierr = ISGetLocalSize(dsfields, &nf);CHKERRQ(ierr);
            if (!nf) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "DS cannot be supported on 0 fields");
            ierr = ISGetIndices(dsfields, &fld);CHKERRQ(ierr);
            ierr = ISGetLocalSize(dm->probs[d].fields, &onf);CHKERRQ(ierr);
            ierr = ISGetIndices(dm->probs[d].fields, &ofld);CHKERRQ(ierr);
            ierr = PetscMalloc1(nf, &fidx);CHKERRQ(ierr);
            for (f = 0, g = 0; f < onf && g < nf; ++f) {
              if (ofld[f] == fld[g]) fidx[g++] = f;
            }
            ierr = ISRestoreIndices(dm->probs[d].fields, &ofld);CHKERRQ(ierr);
            ierr = ISRestoreIndices(dsfields, &fld);CHKERRQ(ierr);
            ierr = ISDestroy(&dsfields);CHKERRQ(ierr);
            ierr = PetscDSSelectDiscretizations(dm->probs[0].ds, nf, fidx, (*subdm)->probs[0].ds);CHKERRQ(ierr);
            ierr = PetscDSSelectEquations(dm->probs[0].ds, nf, fidx, (*subdm)->probs[0].ds);CHKERRQ(ierr);
            ierr = PetscFree(fidx);CHKERRQ(ierr);
          }
          ++e;
        }
      } else {
        ierr = PetscDSCopyConstants(dm->probs[0].ds, (*subdm)->probs[0].ds);CHKERRQ(ierr);
        ierr = PetscDSCopyBoundary(dm->probs[0].ds, PETSC_DETERMINE, NULL, (*subdm)->probs[0].ds);CHKERRQ(ierr);
        ierr = PetscDSSelectDiscretizations(dm->probs[0].ds, numFields, fields, (*subdm)->probs[0].ds);CHKERRQ(ierr);
        ierr = PetscDSSelectEquations(dm->probs[0].ds, numFields, fields, (*subdm)->probs[0].ds);CHKERRQ(ierr);
      }
    }
    if (haveNull && is) {
      MatNullSpace nullSpace;

      ierr = (*(*subdm)->nullspaceConstructors[nf])(*subdm, of, nf, &nullSpace);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) *is, "nullspace", (PetscObject) nullSpace);CHKERRQ(ierr);
      ierr = MatNullSpaceDestroy(&nullSpace);CHKERRQ(ierr);
    }
    if (dm->coarseMesh) {
      ierr = DMCreateSubDM(dm->coarseMesh, numFields, fields, NULL, &(*subdm)->coarseMesh);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  DMCreateSectionSuperDM - Returns an arrays of ISes and DM+Section encapsulating a superproblem defined by the DM+Sections passed in.

  Not collective

  Input Parameters:
+ dms - The DM objects
- len - The number of DMs

  Output Parameters:
+ is - The global indices for the subproblem, or NULL
- superdm - The DM for the superproblem, which must already have be cloned

  Note: This handles all information in the DM class and the PetscSection. This is used as the basis for creating subDMs in specialized classes,
  such as Plex and Forest.

  Level: intermediate

.seealso DMCreateSuperDM(), DMGetLocalSection(), DMPlexSetMigrationSF(), DMView()
@*/
PetscErrorCode DMCreateSectionSuperDM(DM dms[], PetscInt len, IS **is, DM *superdm)
{
  MPI_Comm       comm;
  PetscSection   supersection, *sections, *sectionGlobals;
  PetscInt      *Nfs, Nf = 0, f, supf, oldf = -1, nullf = -1, i;
  PetscBool      haveNull = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dms[0], &comm);CHKERRQ(ierr);
  /* Pull out local and global sections */
  ierr = PetscMalloc3(len, &Nfs, len, &sections, len, &sectionGlobals);CHKERRQ(ierr);
  for (i = 0 ; i < len; ++i) {
    ierr = DMGetLocalSection(dms[i], &sections[i]);CHKERRQ(ierr);
    ierr = DMGetGlobalSection(dms[i], &sectionGlobals[i]);CHKERRQ(ierr);
    if (!sections[i]) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Must set default section for DM before splitting fields");
    if (!sectionGlobals[i]) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Must set default global section for DM before splitting fields");
    ierr = PetscSectionGetNumFields(sections[i], &Nfs[i]);CHKERRQ(ierr);
    Nf += Nfs[i];
  }
  /* Create the supersection */
  ierr = PetscSectionCreateSupersection(sections, len, &supersection);CHKERRQ(ierr);
  ierr = DMSetLocalSection(*superdm, supersection);CHKERRQ(ierr);
  /* Create ISes */
  if (is) {
    PetscSection supersectionGlobal;
    PetscInt     bs = -1, startf = 0;

    ierr = PetscMalloc1(len, is);CHKERRQ(ierr);
    ierr = DMGetGlobalSection(*superdm, &supersectionGlobal);CHKERRQ(ierr);
    for (i = 0 ; i < len; startf += Nfs[i], ++i) {
      PetscInt *subIndices;
      PetscInt  subSize, subOff, pStart, pEnd, p, start, end, dummy;

      ierr = PetscSectionGetChart(sectionGlobals[i], &pStart, &pEnd);CHKERRQ(ierr);
      ierr = PetscSectionGetConstrainedStorageSize(sectionGlobals[i], &subSize);CHKERRQ(ierr);
      ierr = PetscMalloc1(subSize, &subIndices);CHKERRQ(ierr);
      for (p = pStart, subOff = 0; p < pEnd; ++p) {
        PetscInt gdof, gcdof, gtdof, d;

        ierr = PetscSectionGetDof(sectionGlobals[i], p, &gdof);CHKERRQ(ierr);
        ierr = PetscSectionGetConstraintDof(sections[i], p, &gcdof);CHKERRQ(ierr);
        gtdof = gdof - gcdof;
        if (gdof > 0 && gtdof) {
          if (bs < 0)           {bs = gtdof;}
          else if (bs != gtdof) {bs = 1;}
          ierr = DMGetGlobalFieldOffset_Private(*superdm, p, startf, &start, &dummy);CHKERRQ(ierr);
          ierr = DMGetGlobalFieldOffset_Private(*superdm, p, startf+Nfs[i]-1, &dummy, &end);CHKERRQ(ierr);
          if (end-start != gtdof) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid number of global dofs %D != %D for dm %D on point %D", end-start, gtdof, i, p);
          for (d = start; d < end; ++d, ++subOff) subIndices[subOff] = d;
        }
      }
      ierr = ISCreateGeneral(comm, subSize, subIndices, PETSC_OWN_POINTER, &(*is)[i]);CHKERRQ(ierr);
      /* Must have same blocksize on all procs (some might have no points) */
      {
        PetscInt bs = -1, bsLocal[2], bsMinMax[2];

        bsLocal[0] = bs < 0 ? PETSC_MAX_INT : bs; bsLocal[1] = bs;
        ierr = PetscGlobalMinMaxInt(comm, bsLocal, bsMinMax);CHKERRQ(ierr);
        if (bsMinMax[0] != bsMinMax[1]) {bs = 1;}
        else                            {bs = bsMinMax[0];}
        ierr = ISSetBlockSize((*is)[i], bs);CHKERRQ(ierr);
      }
    }
  }
  /* Preserve discretizations */
  if (len && dms[0]->probs) {
    ierr = DMSetNumFields(*superdm, Nf);CHKERRQ(ierr);
    for (i = 0, supf = 0; i < len; ++i) {
      for (f = 0; f < Nfs[i]; ++f, ++supf) {
        PetscObject disc;

        ierr = DMGetField(dms[i], f, NULL, &disc);CHKERRQ(ierr);
        ierr = DMSetField(*superdm, supf, NULL, disc);CHKERRQ(ierr);
      }
    }
    ierr = DMCreateDS(*superdm);CHKERRQ(ierr);
  }
  /* Preserve nullspaces */
  for (i = 0, supf = 0; i < len; ++i) {
    for (f = 0; f < Nfs[i]; ++f, ++supf) {
      (*superdm)->nullspaceConstructors[supf] = dms[i]->nullspaceConstructors[f];
      if ((*superdm)->nullspaceConstructors[supf]) {
        haveNull = PETSC_TRUE;
        nullf    = supf;
        oldf     = f;
      }
    }
  }
  /* Attach nullspace to IS */
  if (haveNull && is) {
    MatNullSpace nullSpace;

    ierr = (*(*superdm)->nullspaceConstructors[nullf])(*superdm, oldf, nullf, &nullSpace);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) (*is)[nullf], "nullspace", (PetscObject) nullSpace);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullSpace);CHKERRQ(ierr);
  }
  ierr = PetscSectionDestroy(&supersection);CHKERRQ(ierr);
  ierr = PetscFree3(Nfs, sections, sectionGlobals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
