#include <petsc/private/dmimpl.h>     /*I      "petscdm.h"     I*/
#include <petscds.h>

// Greatest common divisor of two nonnegative integers
PetscInt PetscGCD(PetscInt a, PetscInt b) {
  while (b != 0) {
    PetscInt tmp = b;
    b = a % b;
    a = tmp;
  }
  return a;
}

PetscErrorCode DMCreateGlobalVector_Section_Private(DM dm,Vec *vec)
{
  PetscSection   gSection;
  PetscInt       localSize, bs, blockSize = -1, pStart, pEnd, p;
  PetscInt       in[2],out[2];

  PetscFunctionBegin;
  PetscCall(DMGetGlobalSection(dm, &gSection));
  PetscCall(PetscSectionGetChart(gSection, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, cdof;

    PetscCall(PetscSectionGetDof(gSection, p, &dof));
    PetscCall(PetscSectionGetConstraintDof(gSection, p, &cdof));

    if (dof - cdof > 0) {
      if (blockSize < 0) {
        /* set blockSize */
        blockSize = dof-cdof;
      } else {
        blockSize = PetscGCD(dof - cdof, blockSize);
      }
    }
  }

  in[0] = blockSize < 0 ? PETSC_MIN_INT : -blockSize;
  in[1] = blockSize;
  PetscCall(MPIU_Allreduce(in,out,2,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)dm)));
  /* -out[0] = min(blockSize), out[1] = max(blockSize) */
  if (-out[0] == out[1]) {
    bs = out[1];
  } else bs = 1;

  if (bs < 0) { /* Everyone was empty */
    blockSize = 1;
    bs        = 1;
  }

  PetscCall(PetscSectionGetConstrainedStorageSize(gSection, &localSize));
  PetscCheck(localSize%blockSize == 0,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Mismatch between blocksize %" PetscInt_FMT " and local storage size %" PetscInt_FMT, blockSize, localSize);
  PetscCall(VecCreate(PetscObjectComm((PetscObject)dm), vec));
  PetscCall(VecSetSizes(*vec, localSize, PETSC_DETERMINE));
  PetscCall(VecSetBlockSize(*vec, bs));
  PetscCall(VecSetType(*vec,dm->vectype));
  PetscCall(VecSetDM(*vec, dm));
  /* PetscCall(VecSetLocalToGlobalMapping(*vec, dm->ltogmap)); */
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateLocalVector_Section_Private(DM dm,Vec *vec)
{
  PetscSection   section;
  PetscInt       localSize, blockSize = -1, pStart, pEnd, p;

  PetscFunctionBegin;
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionGetChart(section, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof;

    PetscCall(PetscSectionGetDof(section, p, &dof));
    if ((blockSize < 0) && (dof > 0)) blockSize = dof;
    if (dof > 0) blockSize = PetscGCD(dof, blockSize);
  }
  PetscCall(PetscSectionGetStorageSize(section, &localSize));
  PetscCall(VecCreate(PETSC_COMM_SELF, vec));
  PetscCall(VecSetSizes(*vec, localSize, localSize));
  PetscCall(VecSetBlockSize(*vec, blockSize));
  PetscCall(VecSetType(*vec,dm->vectype));
  PetscCall(VecSetDM(*vec, dm));
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

.seealso `DMCreateSubDM()`, `DMGetLocalSection()`, `DMPlexSetMigrationSF()`, `DMView()`
@*/
PetscErrorCode DMCreateSectionSubDM(DM dm, PetscInt numFields, const PetscInt fields[], IS *is, DM *subdm)
{
  PetscSection   section, sectionGlobal;
  PetscInt      *subIndices;
  PetscInt       subSize = 0, subOff = 0, Nf, f, pStart, pEnd, p;

  PetscFunctionBegin;
  if (!numFields) PetscFunctionReturn(0);
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetGlobalSection(dm, &sectionGlobal));
  PetscCheck(section,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Must set default section for DM before splitting fields");
  PetscCheck(sectionGlobal,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Must set default global section for DM before splitting fields");
  PetscCall(PetscSectionGetNumFields(section, &Nf));
  PetscCheck(numFields <= Nf,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Number of requested fields %" PetscInt_FMT " greater than number of DM fields %" PetscInt_FMT, numFields, Nf);
  if (is) {
    PetscInt bs, bsLocal[2], bsMinMax[2];

    for (f = 0, bs = 0; f < numFields; ++f) {
      PetscInt Nc;

      PetscCall(PetscSectionGetFieldComponents(section, fields[f], &Nc));
      bs  += Nc;
    }
    PetscCall(PetscSectionGetChart(sectionGlobal, &pStart, &pEnd));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt gdof, pSubSize  = 0;

      PetscCall(PetscSectionGetDof(sectionGlobal, p, &gdof));
      if (gdof > 0) {
        for (f = 0; f < numFields; ++f) {
          PetscInt fdof, fcdof;

          PetscCall(PetscSectionGetFieldDof(section, p, fields[f], &fdof));
          PetscCall(PetscSectionGetFieldConstraintDof(section, p, fields[f], &fcdof));
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
    PetscCall(PetscGlobalMinMaxInt(PetscObjectComm((PetscObject) dm), bsLocal, bsMinMax));
    if (bsMinMax[0] != bsMinMax[1]) {bs = 1;}
    else                            {bs = bsMinMax[0];}
    PetscCall(PetscMalloc1(subSize, &subIndices));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt gdof, goff;

      PetscCall(PetscSectionGetDof(sectionGlobal, p, &gdof));
      if (gdof > 0) {
        PetscCall(PetscSectionGetOffset(sectionGlobal, p, &goff));
        for (f = 0; f < numFields; ++f) {
          PetscInt fdof, fcdof, fc, f2, poff = 0;

          /* Can get rid of this loop by storing field information in the global section */
          for (f2 = 0; f2 < fields[f]; ++f2) {
            PetscCall(PetscSectionGetFieldDof(section, p, f2, &fdof));
            PetscCall(PetscSectionGetFieldConstraintDof(section, p, f2, &fcdof));
            poff += fdof-fcdof;
          }
          PetscCall(PetscSectionGetFieldDof(section, p, fields[f], &fdof));
          PetscCall(PetscSectionGetFieldConstraintDof(section, p, fields[f], &fcdof));
          for (fc = 0; fc < fdof-fcdof; ++fc, ++subOff) {
            subIndices[subOff] = goff+poff+fc;
          }
        }
      }
    }
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), subSize, subIndices, PETSC_OWN_POINTER, is));
    if (bs > 1) {
      /* We need to check that the block size does not come from non-contiguous fields */
      PetscInt i, j, set = 1, rset = 1;
      for (i = 0; i < subSize; i += bs) {
        for (j = 0; j < bs; ++j) {
          if (subIndices[i+j] != subIndices[i]+j) {set = 0; break;}
        }
      }
      PetscCallMPI(MPI_Allreduce(&set, &rset, 1, MPIU_INT, MPI_PROD, PetscObjectComm((PetscObject)dm)));
      if (rset) PetscCall(ISSetBlockSize(*is, bs));
    }
  }
  if (subdm) {
    PetscSection subsection;
    PetscBool    haveNull = PETSC_FALSE;
    PetscInt     f, nf = 0, of = 0;

    PetscCall(PetscSectionCreateSubsection(section, numFields, fields, &subsection));
    PetscCall(DMSetLocalSection(*subdm, subsection));
    PetscCall(PetscSectionDestroy(&subsection));
    for (f = 0; f < numFields; ++f) {
      (*subdm)->nullspaceConstructors[f] = dm->nullspaceConstructors[fields[f]];
      if ((*subdm)->nullspaceConstructors[f]) {
        haveNull = PETSC_TRUE;
        nf       = f;
        of       = fields[f];
      }
    }
    if (dm->probs) {
      PetscCall(DMSetNumFields(*subdm, numFields));
      for (f = 0; f < numFields; ++f) {
        PetscObject disc;

        PetscCall(DMGetField(dm, fields[f], NULL, &disc));
        PetscCall(DMSetField(*subdm, f, NULL, disc));
      }
      PetscCall(DMCreateDS(*subdm));
      if (numFields == 1 && is) {
        PetscObject disc, space, pmat;

        PetscCall(DMGetField(*subdm, 0, NULL, &disc));
        PetscCall(PetscObjectQuery(disc, "nullspace", &space));
        if (space) PetscCall(PetscObjectCompose((PetscObject) *is, "nullspace", space));
        PetscCall(PetscObjectQuery(disc, "nearnullspace", &space));
        if (space) PetscCall(PetscObjectCompose((PetscObject) *is, "nearnullspace", space));
        PetscCall(PetscObjectQuery(disc, "pmat", &pmat));
        if (pmat) PetscCall(PetscObjectCompose((PetscObject) *is, "pmat", pmat));
      }
      /* Check if DSes record their DM fields */
      if (dm->probs[0].fields) {
        PetscInt d, e;

        for (d = 0, e = 0; d < dm->Nds && e < (*subdm)->Nds; ++d) {
          const PetscInt  Nf = dm->probs[d].ds->Nf;
          const PetscInt *fld;
          PetscInt        f, g;

          PetscCall(ISGetIndices(dm->probs[d].fields, &fld));
          for (f = 0; f < Nf; ++f) {
            for (g = 0; g < numFields; ++g) if (fld[f] == fields[g]) break;
            if (g < numFields) break;
          }
          PetscCall(ISRestoreIndices(dm->probs[d].fields, &fld));
          if (f == Nf) continue;
          PetscCall(PetscDSCopyConstants(dm->probs[d].ds, (*subdm)->probs[e].ds));
          PetscCall(PetscDSCopyBoundary(dm->probs[d].ds, numFields, fields, (*subdm)->probs[e].ds));
          /* Translate DM fields to DS fields */
          {
            IS              infields, dsfields;
            const PetscInt *fld, *ofld;
            PetscInt       *fidx;
            PetscInt        onf, nf, f, g;

            PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numFields, fields, PETSC_USE_POINTER, &infields));
            PetscCall(ISIntersect(infields, dm->probs[d].fields, &dsfields));
            PetscCall(ISDestroy(&infields));
            PetscCall(ISGetLocalSize(dsfields, &nf));
            PetscCheck(nf,PETSC_COMM_SELF, PETSC_ERR_PLIB, "DS cannot be supported on 0 fields");
            PetscCall(ISGetIndices(dsfields, &fld));
            PetscCall(ISGetLocalSize(dm->probs[d].fields, &onf));
            PetscCall(ISGetIndices(dm->probs[d].fields, &ofld));
            PetscCall(PetscMalloc1(nf, &fidx));
            for (f = 0, g = 0; f < onf && g < nf; ++f) {
              if (ofld[f] == fld[g]) fidx[g++] = f;
            }
            PetscCall(ISRestoreIndices(dm->probs[d].fields, &ofld));
            PetscCall(ISRestoreIndices(dsfields, &fld));
            PetscCall(ISDestroy(&dsfields));
            PetscCall(PetscDSSelectDiscretizations(dm->probs[0].ds, nf, fidx, (*subdm)->probs[0].ds));
            PetscCall(PetscDSSelectEquations(dm->probs[0].ds, nf, fidx, (*subdm)->probs[0].ds));
            PetscCall(PetscFree(fidx));
          }
          ++e;
        }
      } else {
        PetscCall(PetscDSCopyConstants(dm->probs[0].ds, (*subdm)->probs[0].ds));
        PetscCall(PetscDSCopyBoundary(dm->probs[0].ds, PETSC_DETERMINE, NULL, (*subdm)->probs[0].ds));
        PetscCall(PetscDSSelectDiscretizations(dm->probs[0].ds, numFields, fields, (*subdm)->probs[0].ds));
        PetscCall(PetscDSSelectEquations(dm->probs[0].ds, numFields, fields, (*subdm)->probs[0].ds));
      }
    }
    if (haveNull && is) {
      MatNullSpace nullSpace;

      PetscCall((*(*subdm)->nullspaceConstructors[nf])(*subdm, of, nf, &nullSpace));
      PetscCall(PetscObjectCompose((PetscObject) *is, "nullspace", (PetscObject) nullSpace));
      PetscCall(MatNullSpaceDestroy(&nullSpace));
    }
    if (dm->coarseMesh) {
      PetscCall(DMCreateSubDM(dm->coarseMesh, numFields, fields, NULL, &(*subdm)->coarseMesh));
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

.seealso `DMCreateSuperDM()`, `DMGetLocalSection()`, `DMPlexSetMigrationSF()`, `DMView()`
@*/
PetscErrorCode DMCreateSectionSuperDM(DM dms[], PetscInt len, IS **is, DM *superdm)
{
  MPI_Comm       comm;
  PetscSection   supersection, *sections, *sectionGlobals;
  PetscInt      *Nfs, Nf = 0, f, supf, oldf = -1, nullf = -1, i;
  PetscBool      haveNull = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dms[0], &comm));
  /* Pull out local and global sections */
  PetscCall(PetscMalloc3(len, &Nfs, len, &sections, len, &sectionGlobals));
  for (i = 0 ; i < len; ++i) {
    PetscCall(DMGetLocalSection(dms[i], &sections[i]));
    PetscCall(DMGetGlobalSection(dms[i], &sectionGlobals[i]));
    PetscCheck(sections[i],comm, PETSC_ERR_ARG_WRONG, "Must set default section for DM before splitting fields");
    PetscCheck(sectionGlobals[i],comm, PETSC_ERR_ARG_WRONG, "Must set default global section for DM before splitting fields");
    PetscCall(PetscSectionGetNumFields(sections[i], &Nfs[i]));
    Nf += Nfs[i];
  }
  /* Create the supersection */
  PetscCall(PetscSectionCreateSupersection(sections, len, &supersection));
  PetscCall(DMSetLocalSection(*superdm, supersection));
  /* Create ISes */
  if (is) {
    PetscSection supersectionGlobal;
    PetscInt     bs = -1, startf = 0;

    PetscCall(PetscMalloc1(len, is));
    PetscCall(DMGetGlobalSection(*superdm, &supersectionGlobal));
    for (i = 0 ; i < len; startf += Nfs[i], ++i) {
      PetscInt *subIndices;
      PetscInt  subSize, subOff, pStart, pEnd, p, start, end, dummy;

      PetscCall(PetscSectionGetChart(sectionGlobals[i], &pStart, &pEnd));
      PetscCall(PetscSectionGetConstrainedStorageSize(sectionGlobals[i], &subSize));
      PetscCall(PetscMalloc1(subSize, &subIndices));
      for (p = pStart, subOff = 0; p < pEnd; ++p) {
        PetscInt gdof, gcdof, gtdof, d;

        PetscCall(PetscSectionGetDof(sectionGlobals[i], p, &gdof));
        PetscCall(PetscSectionGetConstraintDof(sections[i], p, &gcdof));
        gtdof = gdof - gcdof;
        if (gdof > 0 && gtdof) {
          if (bs < 0)           {bs = gtdof;}
          else if (bs != gtdof) {bs = 1;}
          PetscCall(DMGetGlobalFieldOffset_Private(*superdm, p, startf, &start, &dummy));
          PetscCall(DMGetGlobalFieldOffset_Private(*superdm, p, startf+Nfs[i]-1, &dummy, &end));
          PetscCheck(end-start == gtdof,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid number of global dofs %" PetscInt_FMT " != %" PetscInt_FMT " for dm %" PetscInt_FMT " on point %" PetscInt_FMT, end-start, gtdof, i, p);
          for (d = start; d < end; ++d, ++subOff) subIndices[subOff] = d;
        }
      }
      PetscCall(ISCreateGeneral(comm, subSize, subIndices, PETSC_OWN_POINTER, &(*is)[i]));
      /* Must have same blocksize on all procs (some might have no points) */
      {
        PetscInt bs = -1, bsLocal[2], bsMinMax[2];

        bsLocal[0] = bs < 0 ? PETSC_MAX_INT : bs; bsLocal[1] = bs;
        PetscCall(PetscGlobalMinMaxInt(comm, bsLocal, bsMinMax));
        if (bsMinMax[0] != bsMinMax[1]) {bs = 1;}
        else                            {bs = bsMinMax[0];}
        PetscCall(ISSetBlockSize((*is)[i], bs));
      }
    }
  }
  /* Preserve discretizations */
  if (len && dms[0]->probs) {
    PetscCall(DMSetNumFields(*superdm, Nf));
    for (i = 0, supf = 0; i < len; ++i) {
      for (f = 0; f < Nfs[i]; ++f, ++supf) {
        PetscObject disc;

        PetscCall(DMGetField(dms[i], f, NULL, &disc));
        PetscCall(DMSetField(*superdm, supf, NULL, disc));
      }
    }
    PetscCall(DMCreateDS(*superdm));
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

    PetscCall((*(*superdm)->nullspaceConstructors[nullf])(*superdm, oldf, nullf, &nullSpace));
    PetscCall(PetscObjectCompose((PetscObject) (*is)[nullf], "nullspace", (PetscObject) nullSpace));
    PetscCall(MatNullSpaceDestroy(&nullSpace));
  }
  PetscCall(PetscSectionDestroy(&supersection));
  PetscCall(PetscFree3(Nfs, sections, sectionGlobals));
  PetscFunctionReturn(0);
}
