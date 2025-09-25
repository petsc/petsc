#include <petsc/private/dmimpl.h> /*I      "petscdm.h"     I*/
#include <petscds.h>

PetscErrorCode DMCreateGlobalVector_Section_Private(DM dm, Vec *vec)
{
  PetscSection gSection;
  PetscInt     localSize, bs, blockSize = -1, pStart, pEnd, p;
  PetscInt     in[2], out[2];

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
        blockSize = dof - cdof;
      } else {
        blockSize = PetscGCD(dof - cdof, blockSize);
      }
    }
  }

  // You cannot negate PETSC_INT_MIN
  in[0] = blockSize < 0 ? -PETSC_INT_MAX : -blockSize;
  in[1] = blockSize;
  PetscCallMPI(MPIU_Allreduce(in, out, 2, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm)));
  /* -out[0] = min(blockSize), out[1] = max(blockSize) */
  if (-out[0] == out[1]) {
    bs = out[1];
  } else bs = 1;

  if (bs < 0) { /* Everyone was empty */
    blockSize = 1;
    bs        = 1;
  }

  PetscCall(PetscSectionGetConstrainedStorageSize(gSection, &localSize));
  PetscCheck(localSize % blockSize == 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Mismatch between blocksize %" PetscInt_FMT " and local storage size %" PetscInt_FMT, blockSize, localSize);
  PetscCall(VecCreate(PetscObjectComm((PetscObject)dm), vec));
  PetscCall(VecSetSizes(*vec, localSize, PETSC_DETERMINE));
  PetscCall(VecSetBlockSize(*vec, bs));
  PetscCall(VecSetType(*vec, dm->vectype));
  PetscCall(VecSetDM(*vec, dm));
  /* PetscCall(VecSetLocalToGlobalMapping(*vec, dm->ltogmap)); */
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateLocalVector_Section_Private(DM dm, Vec *vec)
{
  PetscSection section;
  PetscInt     localSize, blockSize = -1, pStart, pEnd, p;

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
  PetscCall(VecSetBlockSize(*vec, PetscAbs(blockSize)));
  PetscCall(VecSetType(*vec, dm->vectype));
  PetscCall(VecSetDM(*vec, dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSectionSelectFields_Private(PetscSection s, PetscSection gs, PetscInt numFields, const PetscInt fields[], const PetscInt numComps[], const PetscInt comps[], IS *is)
{
  IS              permutation;
  const PetscInt *perm = NULL;
  PetscInt       *subIndices;
  PetscInt        mbs, bs = 0, bsLocal[2], bsMinMax[2];
  PetscInt        pStart, pEnd, Nc, subSize = 0, subOff = 0;

  PetscFunctionBegin;
  PetscCall(PetscSectionGetChart(gs, &pStart, &pEnd));
  PetscCall(PetscSectionGetPermutation(s, &permutation));
  if (permutation) PetscCall(ISGetIndices(permutation, &perm));
  if (numComps) {
    for (PetscInt f = 0, off = 0; f < numFields; ++f) {
      PetscInt Nc;

      if (numComps[f] < 0) continue;
      PetscCall(PetscSectionGetFieldComponents(s, f, &Nc));
      PetscCheck(numComps[f] <= Nc, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field %" PetscInt_FMT ": Number of selected components %" PetscInt_FMT " > %" PetscInt_FMT " number of field components", f, numComps[f], Nc);
      for (PetscInt c = 0; c < numComps[f]; ++c, ++off) PetscCheck(comps[off] < Nc, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field %" PetscInt_FMT ": component %" PetscInt_FMT " not in [0, %" PetscInt_FMT ")", f, comps[off], Nc);
      bs += numComps[f];
    }
  } else {
    for (PetscInt f = 0; f < numFields; ++f) {
      PetscInt Nc;

      PetscCall(PetscSectionGetFieldComponents(s, fields[f], &Nc));
      bs += Nc;
    }
  }
  mbs = -1; /* multiple of block size not set */
  for (PetscInt p = pStart; p < pEnd; ++p) {
    const PetscInt point = perm ? perm[p - pStart] : p;
    PetscInt       gdof, pSubSize = 0;

    PetscCall(PetscSectionGetDof(gs, point, &gdof));
    if (gdof > 0) {
      PetscInt off = 0;

      for (PetscInt f = 0; f < numFields; ++f) {
        PetscInt fdof, fcdof, sfdof, sfcdof = 0;

        PetscCall(PetscSectionGetFieldComponents(s, f, &Nc));
        PetscCall(PetscSectionGetFieldDof(s, point, fields[f], &fdof));
        PetscCall(PetscSectionGetFieldConstraintDof(s, point, fields[f], &fcdof));
        if (numComps && numComps[f] >= 0) {
          const PetscInt *ind;

          // Assume sets of dofs on points are of size Nc
          PetscCheck(!(fdof % Nc), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components %" PetscInt_FMT " should evenly divide the dofs %" PetscInt_FMT " on point %" PetscInt_FMT, Nc, fdof, point);
          sfdof = (fdof / Nc) * numComps[f];
          PetscCall(PetscSectionGetFieldConstraintIndices(s, point, fields[f], &ind));
          for (PetscInt i = 0; i < (fdof / Nc); ++i) {
            for (PetscInt c = 0, fcc = 0; c < Nc; ++c) {
              if (c == comps[off + fcc]) {
                ++fcc;
                ++sfcdof;
              }
            }
          }
          pSubSize += sfdof - sfcdof;
          off += numComps[f];
        } else {
          pSubSize += fdof - fcdof;
        }
      }
      subSize += pSubSize;
      if (pSubSize && pSubSize % bs) {
        // Layout does not admit a pointwise block size -> set mbs to 0
        mbs = 0;
      } else if (pSubSize) {
        if (mbs == -1) mbs = pSubSize / bs;
        else mbs = PetscMin(mbs, pSubSize / bs);
      }
    }
  }

  // Must have same blocksize on all procs (some might have no points)
  bsLocal[0] = mbs < 0 ? PETSC_INT_MAX : mbs;
  bsLocal[1] = mbs;
  PetscCall(PetscGlobalMinMaxInt(PetscObjectComm((PetscObject)gs), bsLocal, bsMinMax));
  if (bsMinMax[0] != bsMinMax[1]) { /* different multiple of block size -> set bs to 1 */
    bs = 1;
  } else { /* same multiple */
    mbs = bsMinMax[0];
    bs *= mbs;
  }
  PetscCall(PetscMalloc1(subSize, &subIndices));
  for (PetscInt p = pStart; p < pEnd; ++p) {
    const PetscInt point = perm ? perm[p - pStart] : p;
    PetscInt       gdof, goff;

    PetscCall(PetscSectionGetDof(gs, point, &gdof));
    if (gdof > 0) {
      PetscInt off = 0;

      PetscCall(PetscSectionGetOffset(gs, point, &goff));
      for (PetscInt f = 0; f < numFields; ++f) {
        PetscInt fdof, fcdof, poff = 0;

        /* Can get rid of this loop by storing field information in the global section */
        for (PetscInt f2 = 0; f2 < fields[f]; ++f2) {
          PetscCall(PetscSectionGetFieldDof(s, point, f2, &fdof));
          PetscCall(PetscSectionGetFieldConstraintDof(s, point, f2, &fcdof));
          poff += fdof - fcdof;
        }
        PetscCall(PetscSectionGetFieldDof(s, point, fields[f], &fdof));
        PetscCall(PetscSectionGetFieldConstraintDof(s, point, fields[f], &fcdof));

        if (numComps && numComps[f] >= 0) {
          const PetscInt *ind;

          // Assume sets of dofs on points are of size Nc
          PetscCall(PetscSectionGetFieldConstraintIndices(s, point, fields[f], &ind));
          for (PetscInt i = 0, fcoff = 0, pfoff = 0; i < (fdof / Nc); ++i) {
            for (PetscInt c = 0, fcc = 0; c < Nc; ++c) {
              const PetscInt k = i * Nc + c;

              if (ind[fcoff] == k) {
                ++fcoff;
                continue;
              }
              if (c == comps[off + fcc]) {
                ++fcc;
                subIndices[subOff++] = goff + poff + pfoff;
              }
              ++pfoff;
            }
          }
          off += numComps[f];
        } else {
          for (PetscInt fc = 0; fc < fdof - fcdof; ++fc, ++subOff) subIndices[subOff] = goff + poff + fc;
        }
      }
    }
  }
  if (permutation) PetscCall(ISRestoreIndices(permutation, &perm));
  PetscCheck(subSize == subOff, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "The offset array size %" PetscInt_FMT " != %" PetscInt_FMT " the number of indices", subSize, subOff);
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)gs), subSize, subIndices, PETSC_OWN_POINTER, is));
  if (bs > 1) {
    // We need to check that the block size does not come from non-contiguous fields
    PetscInt set = 1, rset = 1;
    for (PetscInt i = 0; i < subSize; i += bs) {
      for (PetscInt j = 0; j < bs; ++j) {
        if (subIndices[i + j] != subIndices[i] + j) {
          set = 0;
          break;
        }
      }
    }
    PetscCallMPI(MPIU_Allreduce(&set, &rset, 1, MPIU_INT, MPI_PROD, PetscObjectComm((PetscObject)gs)));
    if (rset) PetscCall(ISSetBlockSize(*is, bs));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSelectFields_Private(DM dm, PetscSection section, PetscInt numFields, const PetscInt fields[], const PetscInt numComps[], const PetscInt comps[], IS *is, DM *subdm)
{
  PetscSection subsection;
  PetscBool    haveNull = PETSC_FALSE;
  PetscInt     nf = 0, of = 0;

  PetscFunctionBegin;
  // Create nullspace constructor slots
  if (dm->nullspaceConstructors) {
    PetscCall(PetscFree2((*subdm)->nullspaceConstructors, (*subdm)->nearnullspaceConstructors));
    PetscCall(PetscCalloc2(numFields, &(*subdm)->nullspaceConstructors, numFields, &(*subdm)->nearnullspaceConstructors));
  }
  if (numComps) {
    const PetscInt field = fields[0];

    PetscCheck(numFields == 1, PETSC_COMM_SELF, PETSC_ERR_SUP, "We only support a single field for component selection right now");
    PetscCall(PetscSectionCreateComponentSubsection(section, numComps[field], comps, &subsection));
    PetscCall(DMSetLocalSection(*subdm, subsection));
    PetscCall(PetscSectionDestroy(&subsection));
    if (dm->nullspaceConstructors) (*subdm)->nullspaceConstructors[field] = dm->nullspaceConstructors[field];
    if (dm->probs) {
      PetscFV  fv, fvNew;
      PetscInt fnum[1] = {field};

      PetscCall(DMSetNumFields(*subdm, 1));
      PetscCall(DMGetField(dm, field, NULL, (PetscObject *)&fv));
      PetscCall(PetscFVClone(fv, &fvNew));
      PetscCall(PetscFVSetNumComponents(fvNew, numComps[0]));
      PetscCall(DMSetField(*subdm, 0, NULL, (PetscObject)fvNew));
      PetscCall(PetscFVDestroy(&fvNew));
      PetscCall(DMCreateDS(*subdm));
      if (numComps[0] == 1 && is) {
        PetscObject disc, space, pmat;

        PetscCall(DMGetField(*subdm, field, NULL, &disc));
        PetscCall(PetscObjectQuery(disc, "nullspace", &space));
        if (space) PetscCall(PetscObjectCompose((PetscObject)*is, "nullspace", space));
        PetscCall(PetscObjectQuery(disc, "nearnullspace", &space));
        if (space) PetscCall(PetscObjectCompose((PetscObject)*is, "nearnullspace", space));
        PetscCall(PetscObjectQuery(disc, "pmat", &pmat));
        if (pmat) PetscCall(PetscObjectCompose((PetscObject)*is, "pmat", pmat));
      }
      PetscCall(PetscDSCopyConstants(dm->probs[field].ds, (*subdm)->probs[0].ds));
      PetscCall(PetscDSCopyBoundary(dm->probs[field].ds, 1, fnum, (*subdm)->probs[0].ds));
      PetscCall(PetscDSSelectEquations(dm->probs[field].ds, 1, fnum, (*subdm)->probs[0].ds));
    }
    if ((*subdm)->nullspaceConstructors && (*subdm)->nullspaceConstructors[0] && is) {
      MatNullSpace nullSpace;

      PetscCall((*(*subdm)->nullspaceConstructors[0])(*subdm, 0, 0, &nullSpace));
      PetscCall(PetscObjectCompose((PetscObject)*is, "nullspace", (PetscObject)nullSpace));
      PetscCall(MatNullSpaceDestroy(&nullSpace));
    }
    if (dm->coarseMesh) PetscCall(DMCreateSubDM(dm->coarseMesh, numFields, fields, NULL, &(*subdm)->coarseMesh));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscSectionCreateSubsection(section, numFields, fields, &subsection));
  PetscCall(DMSetLocalSection(*subdm, subsection));
  PetscCall(PetscSectionDestroy(&subsection));
  if (dm->probs) {
    PetscCall(DMSetNumFields(*subdm, numFields));
    for (PetscInt f = 0; f < numFields; ++f) {
      PetscObject disc;

      PetscCall(DMGetField(dm, fields[f], NULL, &disc));
      PetscCall(DMSetField(*subdm, f, NULL, disc));
    }
    // TODO: if only FV, then cut down the components
    PetscCall(DMCreateDS(*subdm));
    if (numFields == 1 && is) {
      PetscObject disc, space, pmat;

      PetscCall(DMGetField(*subdm, 0, NULL, &disc));
      PetscCall(PetscObjectQuery(disc, "nullspace", &space));
      if (space) PetscCall(PetscObjectCompose((PetscObject)*is, "nullspace", space));
      PetscCall(PetscObjectQuery(disc, "nearnullspace", &space));
      if (space) PetscCall(PetscObjectCompose((PetscObject)*is, "nearnullspace", space));
      PetscCall(PetscObjectQuery(disc, "pmat", &pmat));
      if (pmat) PetscCall(PetscObjectCompose((PetscObject)*is, "pmat", pmat));
    }
    // Check if DSes record their DM fields
    if (dm->probs[0].fields) {
      PetscInt d, e;

      for (d = 0, e = 0; d < dm->Nds && e < (*subdm)->Nds; ++d) {
        const PetscInt  Nf = dm->probs[d].ds->Nf;
        const PetscInt *fld;
        PetscInt        f, g;

        PetscCall(ISGetIndices(dm->probs[d].fields, &fld));
        for (f = 0; f < Nf; ++f) {
          for (g = 0; g < numFields; ++g)
            if (fld[f] == fields[g]) break;
          if (g < numFields) break;
        }
        PetscCall(ISRestoreIndices(dm->probs[d].fields, &fld));
        if (f == Nf) continue;
        PetscCall(PetscDSCopyConstants(dm->probs[d].ds, (*subdm)->probs[e].ds));
        PetscCall(PetscDSCopyBoundary(dm->probs[d].ds, numFields, fields, (*subdm)->probs[e].ds));
        // Translate DM fields to DS fields
        {
          IS              infields, dsfields;
          const PetscInt *fld, *ofld;
          PetscInt       *fidx;
          PetscInt        onf, nf;

          PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numFields, fields, PETSC_USE_POINTER, &infields));
          PetscCall(ISIntersect(infields, dm->probs[d].fields, &dsfields));
          PetscCall(ISDestroy(&infields));
          PetscCall(ISGetLocalSize(dsfields, &nf));
          PetscCheck(nf, PETSC_COMM_SELF, PETSC_ERR_PLIB, "DS cannot be supported on 0 fields");
          PetscCall(ISGetIndices(dsfields, &fld));
          PetscCall(ISGetLocalSize(dm->probs[d].fields, &onf));
          PetscCall(ISGetIndices(dm->probs[d].fields, &ofld));
          PetscCall(PetscMalloc1(nf, &fidx));
          for (PetscInt f = 0, g = 0; f < onf && g < nf; ++f) {
            if (ofld[f] == fld[g]) fidx[g++] = f;
          }
          PetscCall(ISRestoreIndices(dm->probs[d].fields, &ofld));
          PetscCall(ISRestoreIndices(dsfields, &fld));
          PetscCall(ISDestroy(&dsfields));
          PetscCall(PetscDSSelectDiscretizations(dm->probs[0].ds, nf, fidx, PETSC_DETERMINE, PETSC_DETERMINE, (*subdm)->probs[0].ds));
          PetscCall(PetscDSSelectEquations(dm->probs[0].ds, nf, fidx, (*subdm)->probs[0].ds));
          PetscCall(PetscFree(fidx));
        }
        ++e;
      }
    } else {
      PetscCall(PetscDSCopyConstants(dm->probs[0].ds, (*subdm)->probs[0].ds));
      PetscCall(PetscDSCopyBoundary(dm->probs[0].ds, PETSC_DETERMINE, NULL, (*subdm)->probs[0].ds));
      PetscCall(PetscDSSelectDiscretizations(dm->probs[0].ds, numFields, fields, PETSC_DETERMINE, PETSC_DETERMINE, (*subdm)->probs[0].ds));
      PetscCall(PetscDSSelectEquations(dm->probs[0].ds, numFields, fields, (*subdm)->probs[0].ds));
    }
  }
  for (PetscInt f = 0; f < numFields; ++f) {
    if (dm->nullspaceConstructors) {
      (*subdm)->nullspaceConstructors[f] = dm->nullspaceConstructors[fields[f]];
      if ((*subdm)->nullspaceConstructors[f]) {
        haveNull = PETSC_TRUE;
        nf       = f;
        of       = fields[f];
      }
    }
  }
  if (haveNull && is) {
    MatNullSpace nullSpace;

    PetscCall((*(*subdm)->nullspaceConstructors[nf])(*subdm, of, nf, &nullSpace));
    PetscCall(PetscObjectCompose((PetscObject)*is, "nullspace", (PetscObject)nullSpace));
    PetscCall(MatNullSpaceDestroy(&nullSpace));
  }
  if (dm->coarseMesh) PetscCall(DMCreateSubDM(dm->coarseMesh, numFields, fields, NULL, &(*subdm)->coarseMesh));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMCreateSectionSubDM - Returns an `IS` and `subDM` containing a `PetscSection` that encapsulates a subproblem defined by a subset of the fields in a `PetscSection` in the `DM`.

  Not Collective

  Input Parameters:
+ dm        - The `DM` object
. numFields - The number of fields to incorporate into `subdm`
. fields    - The field numbers of the selected fields
. numComps  - The number of components from each field to incorporate into `subdm`, or PETSC_DECIDE for all components
- comps     - The component numbers of the selected fields (omitted for PTESC_DECIDE fields)

  Output Parameters:
+ is    - The global indices for the subproblem or `NULL`
- subdm - The `DM` for the subproblem, which must already have be cloned from `dm` or `NULL`

  Level: intermediate

  Notes:
  If `is` and `subdm` are both `NULL` this does nothing

.seealso: `DMCreateSubDM()`, `DMGetLocalSection()`, `DMPlexSetMigrationSF()`, `DMView()`
@*/
PetscErrorCode DMCreateSectionSubDM(DM dm, PetscInt numFields, const PetscInt fields[], const PetscInt numComps[], const PetscInt comps[], IS *is, DM *subdm)
{
  PetscSection section, sectionGlobal;
  PetscInt     Nf;

  PetscFunctionBegin;
  if (!numFields) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetGlobalSection(dm, &sectionGlobal));
  PetscCheck(section, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Must set default section for DM before splitting fields");
  PetscCheck(sectionGlobal, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Must set default global section for DM before splitting fields");
  PetscCall(PetscSectionGetNumFields(section, &Nf));
  PetscCheck(numFields <= Nf, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Number of requested fields %" PetscInt_FMT " greater than number of DM fields %" PetscInt_FMT, numFields, Nf);

  if (is) PetscCall(PetscSectionSelectFields_Private(section, sectionGlobal, numFields, fields, numComps, comps, is));
  if (subdm) PetscCall(DMSelectFields_Private(dm, section, numFields, fields, numComps, comps, is, subdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMCreateSectionSuperDM - Returns an arrays of `IS` and a `DM` containing a `PetscSection` that encapsulates a superproblem defined by the array of `DM` and their `PetscSection`

  Not Collective

  Input Parameters:
+ dms - The `DM` objects, the must all have the same topology; for example obtained with `DMClone()`
- len - The number of `DM` in `dms`

  Output Parameters:
+ is      - The global indices for the subproblem, or `NULL`
- superdm - The `DM` for the superproblem, which must already have be cloned and contain the same topology as the `dms`

  Level: intermediate

.seealso: `DMCreateSuperDM()`, `DMGetLocalSection()`, `DMPlexSetMigrationSF()`, `DMView()`
@*/
PetscErrorCode DMCreateSectionSuperDM(DM dms[], PetscInt len, IS *is[], DM *superdm)
{
  MPI_Comm     comm;
  PetscSection supersection, *sections, *sectionGlobals;
  PetscInt    *Nfs, Nf = 0, f, supf, oldf = -1, nullf = -1, i;
  PetscBool    haveNull = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dms[0], &comm));
  /* Pull out local and global sections */
  PetscCall(PetscMalloc3(len, &Nfs, len, &sections, len, &sectionGlobals));
  for (i = 0; i < len; ++i) {
    PetscCall(DMGetLocalSection(dms[i], &sections[i]));
    PetscCall(DMGetGlobalSection(dms[i], &sectionGlobals[i]));
    PetscCheck(sections[i], comm, PETSC_ERR_ARG_WRONG, "Must set default section for DM before splitting fields");
    PetscCheck(sectionGlobals[i], comm, PETSC_ERR_ARG_WRONG, "Must set default global section for DM before splitting fields");
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
    for (i = 0; i < len; startf += Nfs[i], ++i) {
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
          if (bs < 0) {
            bs = gtdof;
          } else if (bs != gtdof) {
            bs = 1;
          }
          PetscCall(DMGetGlobalFieldOffset_Private(*superdm, p, startf, &start, &dummy));
          PetscCall(DMGetGlobalFieldOffset_Private(*superdm, p, startf + Nfs[i] - 1, &dummy, &end));
          PetscCheck(end - start == gtdof, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid number of global dofs %" PetscInt_FMT " != %" PetscInt_FMT " for dm %" PetscInt_FMT " on point %" PetscInt_FMT, end - start, gtdof, i, p);
          for (d = start; d < end; ++d, ++subOff) subIndices[subOff] = d;
        }
      }
      PetscCall(ISCreateGeneral(comm, subSize, subIndices, PETSC_OWN_POINTER, &(*is)[i]));
      /* Must have same blocksize on all procs (some might have no points) */
      {
        PetscInt bs = -1, bsLocal[2], bsMinMax[2];

        bsLocal[0] = bs < 0 ? PETSC_INT_MAX : bs;
        bsLocal[1] = bs;
        PetscCall(PetscGlobalMinMaxInt(comm, bsLocal, bsMinMax));
        if (bsMinMax[0] != bsMinMax[1]) {
          bs = 1;
        } else {
          bs = bsMinMax[0];
        }
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
  // Create nullspace constructor slots
  PetscCall(PetscFree2((*superdm)->nullspaceConstructors, (*superdm)->nearnullspaceConstructors));
  PetscCall(PetscCalloc2(Nf, &(*superdm)->nullspaceConstructors, Nf, &(*superdm)->nearnullspaceConstructors));
  /* Preserve nullspaces */
  for (i = 0, supf = 0; i < len; ++i) {
    for (f = 0; f < Nfs[i]; ++f, ++supf) {
      if (dms[i]->nullspaceConstructors) {
        (*superdm)->nullspaceConstructors[supf] = dms[i]->nullspaceConstructors[f];
        if ((*superdm)->nullspaceConstructors[supf]) {
          haveNull = PETSC_TRUE;
          nullf    = supf;
          oldf     = f;
        }
      }
    }
  }
  /* Attach nullspace to IS */
  if (haveNull && is) {
    MatNullSpace nullSpace;

    PetscCall((*(*superdm)->nullspaceConstructors[nullf])(*superdm, oldf, nullf, &nullSpace));
    PetscCall(PetscObjectCompose((PetscObject)(*is)[nullf], "nullspace", (PetscObject)nullSpace));
    PetscCall(MatNullSpaceDestroy(&nullSpace));
  }
  PetscCall(PetscSectionDestroy(&supersection));
  PetscCall(PetscFree3(Nfs, sections, sectionGlobals));
  PetscFunctionReturn(PETSC_SUCCESS);
}
