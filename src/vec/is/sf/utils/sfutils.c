#include <petsc/private/sfimpl.h>       /*I  "petscsf.h"   I*/
#include <petsc/private/sectionimpl.h>

/*@C
   PetscSFSetGraphLayout - Set a parallel star forest via global indices and a PetscLayout

   Collective

   Input Parameters:
+  sf - star forest
.  layout - PetscLayout defining the global space for roots
.  nleaves - number of leaf vertices on the current process, each of these references a root on any process
.  ilocal - locations of leaves in leafdata buffers, pass NULL for contiguous storage
.  localmode - copy mode for ilocal
-  iremote - root vertices in global numbering corresponding to leaves in ilocal

   Level: intermediate

   Notes:
   Global indices must lie in [0, N) where N is the global size of layout.
   Leaf indices in ilocal get sorted; this means the user-provided array gets sorted if localmode is PETSC_OWN_POINTER.

   Developer Notes:
   Local indices which are the identity permutation in the range [0,nleaves) are discarded as they
   encode contiguous storage. In such case, if localmode is PETSC_OWN_POINTER, the memory is deallocated as it is not
   needed

.seealso: PetscSFCreate(), PetscSFView(), PetscSFSetGraph(), PetscSFGetGraph()
@*/
PetscErrorCode PetscSFSetGraphLayout(PetscSF sf,PetscLayout layout,PetscInt nleaves,PetscInt *ilocal,PetscCopyMode localmode,const PetscInt *iremote)
{
  PetscErrorCode ierr;
  const PetscInt *range;
  PetscInt       i, nroots, ls = -1, ln = -1;
  PetscMPIInt    lr = -1;
  PetscSFNode    *remote;

  PetscFunctionBegin;
  ierr = PetscLayoutGetLocalSize(layout,&nroots);CHKERRQ(ierr);
  ierr = PetscLayoutGetRanges(layout,&range);CHKERRQ(ierr);
  ierr = PetscMalloc1(nleaves,&remote);CHKERRQ(ierr);
  if (nleaves) { ls = iremote[0] + 1; }
  for (i=0; i<nleaves; i++) {
    const PetscInt idx = iremote[i] - ls;
    if (idx < 0 || idx >= ln) { /* short-circuit the search */
      ierr = PetscLayoutFindOwnerIndex(layout,iremote[i],&lr,&remote[i].index);CHKERRQ(ierr);
      remote[i].rank = lr;
      ls = range[lr];
      ln = range[lr+1] - ls;
    } else {
      remote[i].rank  = lr;
      remote[i].index = idx;
    }
  }
  ierr = PetscSFSetGraph(sf,nroots,nleaves,ilocal,localmode,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSFSetGraphSection - Sets the PetscSF graph encoding the parallel dof overlap based upon the PetscSections
  describing the data layout.

  Input Parameters:
+ sf - The SF
. localSection - PetscSection describing the local data layout
- globalSection - PetscSection describing the global data layout

  Level: developer

.seealso: PetscSFSetGraph(), PetscSFSetGraphLayout()
@*/
PetscErrorCode PetscSFSetGraphSection(PetscSF sf, PetscSection localSection, PetscSection globalSection)
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
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscValidHeaderSpecific(localSection, PETSC_SECTION_CLASSID, 2);
  PetscValidHeaderSpecific(globalSection, PETSC_SECTION_CLASSID, 3);

  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
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
    PetscInt gdof, gcdof;

    ierr = PetscSectionGetDof(globalSection, p, &gdof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(globalSection, p, &gcdof);CHKERRQ(ierr);
    PetscCheckFalse(gcdof > (gdof < 0 ? -(gdof+1) : gdof),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point %" PetscInt_FMT " has %" PetscInt_FMT " constraints > %" PetscInt_FMT " dof", p, gcdof, (gdof < 0 ? -(gdof+1) : gdof));
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
      PetscCheckFalse(gsize != dof,comm, PETSC_ERR_ARG_WRONG, "Global dof %" PetscInt_FMT " for point %" PetscInt_FMT " is neither the constrained size %" PetscInt_FMT ", nor the unconstrained %" PetscInt_FMT, gsize, p, dof-cdof, dof);
      cdof = 0; /* Ignore constraints */
    }
    for (d = 0, c = 0; d < dof; ++d) {
      if ((c < cdof) && (cind[c] == d)) {++c; continue;}
      local[l+d-c] = off+d;
    }
    PetscCheckFalse(d-c != gsize,comm, PETSC_ERR_ARG_WRONG, "Point %" PetscInt_FMT ": Global dof %" PetscInt_FMT " != %" PetscInt_FMT " size - number of constraints", p, gsize, d-c);
    if (gdof < 0) {
      for (d = 0; d < gsize; ++d, ++l) {
        PetscInt offset = -(goff+1) + d, r;

        ierr = PetscFindInt(offset,size+1,ranges,&r);CHKERRQ(ierr);
        if (r < 0) r = -(r+2);
        PetscCheckFalse((r < 0) || (r >= size),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point %" PetscInt_FMT " mapped to invalid process %" PetscInt_FMT " (%" PetscInt_FMT ", %" PetscInt_FMT ")", p, r, gdof, goff);
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
  PetscCheckFalse(l != nleaves,comm, PETSC_ERR_PLIB, "Iteration error, l %" PetscInt_FMT " != nleaves %" PetscInt_FMT, l, nleaves);
  ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf, nroots, nleaves, local, PETSC_OWN_POINTER, remote, PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSectionCheckConstraints_Static(PetscSection s)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!s->bc) {
    ierr = PetscSectionCreate(PETSC_COMM_SELF, &s->bc);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(s->bc, s->pStart, s->pEnd);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscSFDistributeSection - Create a new PetscSection reorganized, moving from the root to the leaves of the SF

  Collective on sf

  Input Parameters:
+ sf - The SF
- rootSection - Section defined on root space

  Output Parameters:
+ remoteOffsets - root offsets in leaf storage, or NULL
- leafSection - Section defined on the leaf space

  Level: advanced

.seealso: PetscSFCreate()
@*/
PetscErrorCode PetscSFDistributeSection(PetscSF sf, PetscSection rootSection, PetscInt **remoteOffsets, PetscSection leafSection)
{
  PetscSF        embedSF;
  const PetscInt *indices;
  IS             selected;
  PetscInt       numFields, nroots, rpStart, rpEnd, lpStart = PETSC_MAX_INT, lpEnd = -1, f, c;
  PetscBool      *sub, hasc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PETSCSF_DistSect,sf,0,0,0);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(rootSection, &numFields);CHKERRQ(ierr);
  if (numFields) {
    IS perm;

    /* PetscSectionSetNumFields() calls PetscSectionReset(), which destroys
       leafSection->perm. To keep this permutation set by the user, we grab
       the reference before calling PetscSectionSetNumFields() and set it
       back after. */
    ierr = PetscSectionGetPermutation(leafSection, &perm);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr);
    ierr = PetscSectionSetNumFields(leafSection, numFields);CHKERRQ(ierr);
    ierr = PetscSectionSetPermutation(leafSection, perm);CHKERRQ(ierr);
    ierr = ISDestroy(&perm);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(numFields+2, &sub);CHKERRQ(ierr);
  sub[1] = rootSection->bc ? PETSC_TRUE : PETSC_FALSE;
  for (f = 0; f < numFields; ++f) {
    PetscSectionSym sym, dsym = NULL;
    const char      *name   = NULL;
    PetscInt        numComp = 0;

    sub[2 + f] = rootSection->field[f]->bc ? PETSC_TRUE : PETSC_FALSE;
    ierr = PetscSectionGetFieldComponents(rootSection, f, &numComp);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldName(rootSection, f, &name);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldSym(rootSection, f, &sym);CHKERRQ(ierr);
    if (sym) {ierr = PetscSectionSymDistribute(sym, sf, &dsym);CHKERRQ(ierr);}
    ierr = PetscSectionSetFieldComponents(leafSection, f, numComp);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(leafSection, f, name);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldSym(leafSection, f, dsym);CHKERRQ(ierr);
    ierr = PetscSectionSymDestroy(&dsym);CHKERRQ(ierr);
    for (c = 0; c < rootSection->numFieldComponents[f]; ++c) {
      ierr = PetscSectionGetComponentName(rootSection, f, c, &name);CHKERRQ(ierr);
      ierr = PetscSectionSetComponentName(leafSection, f, c, name);CHKERRQ(ierr);
    }
  }
  ierr = PetscSectionGetChart(rootSection, &rpStart, &rpEnd);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf,&nroots,NULL,NULL,NULL);CHKERRQ(ierr);
  rpEnd = PetscMin(rpEnd,nroots);
  rpEnd = PetscMax(rpStart,rpEnd);
  /* see if we can avoid creating the embedded SF, since it can cost more than an allreduce */
  sub[0] = (PetscBool)(nroots != rpEnd - rpStart);
  ierr = MPIU_Allreduce(MPI_IN_PLACE, sub, 2+numFields, MPIU_BOOL, MPI_LOR, PetscObjectComm((PetscObject)sf));CHKERRMPI(ierr);
  if (sub[0]) {
    ierr = ISCreateStride(PETSC_COMM_SELF, rpEnd - rpStart, rpStart, 1, &selected);CHKERRQ(ierr);
    ierr = ISGetIndices(selected, &indices);CHKERRQ(ierr);
    ierr = PetscSFCreateEmbeddedRootSF(sf, rpEnd - rpStart, indices, &embedSF);CHKERRQ(ierr);
    ierr = ISRestoreIndices(selected, &indices);CHKERRQ(ierr);
    ierr = ISDestroy(&selected);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectReference((PetscObject)sf);CHKERRQ(ierr);
    embedSF = sf;
  }
  ierr = PetscSFGetLeafRange(embedSF, &lpStart, &lpEnd);CHKERRQ(ierr);
  lpEnd++;

  ierr = PetscSectionSetChart(leafSection, lpStart, lpEnd);CHKERRQ(ierr);

  /* Constrained dof section */
  hasc = sub[1];
  for (f = 0; f < numFields; ++f) hasc = (PetscBool)(hasc || sub[2+f]);

  /* Could fuse these at the cost of copies and extra allocation */
  ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->atlasDof[-rpStart], &leafSection->atlasDof[-lpStart],MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->atlasDof[-rpStart], &leafSection->atlasDof[-lpStart],MPI_REPLACE);CHKERRQ(ierr);
  if (sub[1]) {
    ierr = PetscSectionCheckConstraints_Static(rootSection);CHKERRQ(ierr);
    ierr = PetscSectionCheckConstraints_Static(leafSection);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->bc->atlasDof[-rpStart], &leafSection->bc->atlasDof[-lpStart],MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->bc->atlasDof[-rpStart], &leafSection->bc->atlasDof[-lpStart],MPI_REPLACE);CHKERRQ(ierr);
  }
  for (f = 0; f < numFields; ++f) {
    ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->field[f]->atlasDof[-rpStart], &leafSection->field[f]->atlasDof[-lpStart],MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->field[f]->atlasDof[-rpStart], &leafSection->field[f]->atlasDof[-lpStart],MPI_REPLACE);CHKERRQ(ierr);
    if (sub[2+f]) {
      ierr = PetscSectionCheckConstraints_Static(rootSection->field[f]);CHKERRQ(ierr);
      ierr = PetscSectionCheckConstraints_Static(leafSection->field[f]);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->field[f]->bc->atlasDof[-rpStart], &leafSection->field[f]->bc->atlasDof[-lpStart],MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->field[f]->bc->atlasDof[-rpStart], &leafSection->field[f]->bc->atlasDof[-lpStart],MPI_REPLACE);CHKERRQ(ierr);
    }
  }
  if (remoteOffsets) {
    ierr = PetscMalloc1(lpEnd - lpStart, remoteOffsets);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->atlasOff[-rpStart], &(*remoteOffsets)[-lpStart],MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->atlasOff[-rpStart], &(*remoteOffsets)[-lpStart],MPI_REPLACE);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(leafSection);CHKERRQ(ierr);
  if (hasc) { /* need to communicate bcIndices */
    PetscSF  bcSF;
    PetscInt *rOffBc;

    ierr = PetscMalloc1(lpEnd - lpStart, &rOffBc);CHKERRQ(ierr);
    if (sub[1]) {
      ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->bc->atlasOff[-rpStart], &rOffBc[-lpStart],MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->bc->atlasOff[-rpStart], &rOffBc[-lpStart],MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFCreateSectionSF(embedSF, rootSection->bc, rOffBc, leafSection->bc, &bcSF);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(bcSF, MPIU_INT, rootSection->bcIndices, leafSection->bcIndices,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(bcSF, MPIU_INT, rootSection->bcIndices, leafSection->bcIndices,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFDestroy(&bcSF);CHKERRQ(ierr);
    }
    for (f = 0; f < numFields; ++f) {
      if (sub[2+f]) {
        ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->field[f]->bc->atlasOff[-rpStart], &rOffBc[-lpStart],MPI_REPLACE);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->field[f]->bc->atlasOff[-rpStart], &rOffBc[-lpStart],MPI_REPLACE);CHKERRQ(ierr);
        ierr = PetscSFCreateSectionSF(embedSF, rootSection->field[f]->bc, rOffBc, leafSection->field[f]->bc, &bcSF);CHKERRQ(ierr);
        ierr = PetscSFBcastBegin(bcSF, MPIU_INT, rootSection->field[f]->bcIndices, leafSection->field[f]->bcIndices,MPI_REPLACE);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(bcSF, MPIU_INT, rootSection->field[f]->bcIndices, leafSection->field[f]->bcIndices,MPI_REPLACE);CHKERRQ(ierr);
        ierr = PetscSFDestroy(&bcSF);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(rOffBc);CHKERRQ(ierr);
  }
  ierr = PetscSFDestroy(&embedSF);CHKERRQ(ierr);
  ierr = PetscFree(sub);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSCSF_DistSect,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscSFCreateRemoteOffsets - Create offsets for point data on remote processes

  Collective on sf

  Input Parameters:
+ sf          - The SF
. rootSection - Data layout of remote points for outgoing data (this is layout for SF roots)
- leafSection - Data layout of local points for incoming data  (this is layout for SF leaves)

  Output Parameter:
. remoteOffsets - Offsets for point data on remote processes (these are offsets from the root section), or NULL

  Level: developer

.seealso: PetscSFCreate()
@*/
PetscErrorCode PetscSFCreateRemoteOffsets(PetscSF sf, PetscSection rootSection, PetscSection leafSection, PetscInt **remoteOffsets)
{
  PetscSF         embedSF;
  const PetscInt *indices;
  IS              selected;
  PetscInt        numRoots, rpStart = 0, rpEnd = 0, lpStart = 0, lpEnd = 0;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  *remoteOffsets = NULL;
  ierr = PetscSFGetGraph(sf, &numRoots, NULL, NULL, NULL);CHKERRQ(ierr);
  if (numRoots < 0) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(PETSCSF_RemoteOff,sf,0,0,0);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(rootSection, &rpStart, &rpEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(leafSection, &lpStart, &lpEnd);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF, rpEnd - rpStart, rpStart, 1, &selected);CHKERRQ(ierr);
  ierr = ISGetIndices(selected, &indices);CHKERRQ(ierr);
  ierr = PetscSFCreateEmbeddedRootSF(sf, rpEnd - rpStart, indices, &embedSF);CHKERRQ(ierr);
  ierr = ISRestoreIndices(selected, &indices);CHKERRQ(ierr);
  ierr = ISDestroy(&selected);CHKERRQ(ierr);
  ierr = PetscCalloc1(lpEnd - lpStart, remoteOffsets);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->atlasOff[-rpStart], &(*remoteOffsets)[-lpStart],MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->atlasOff[-rpStart], &(*remoteOffsets)[-lpStart],MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&embedSF);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSCSF_RemoteOff,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscSFCreateSectionSF - Create an expanded SF of dofs, assuming the input SF relates points

  Collective on sf

  Input Parameters:
+ sf - The SF
. rootSection - Data layout of remote points for outgoing data (this is usually the serial section)
. remoteOffsets - Offsets for point data on remote processes (these are offsets from the root section), or NULL
- leafSection - Data layout of local points for incoming data  (this is the distributed section)

  Output Parameters:
- sectionSF - The new SF

  Note: Either rootSection or remoteOffsets can be specified

  Level: advanced

.seealso: PetscSFCreate()
@*/
PetscErrorCode PetscSFCreateSectionSF(PetscSF sf, PetscSection rootSection, PetscInt remoteOffsets[], PetscSection leafSection, PetscSF *sectionSF)
{
  MPI_Comm          comm;
  const PetscInt    *localPoints;
  const PetscSFNode *remotePoints;
  PetscInt          lpStart, lpEnd;
  PetscInt          numRoots, numSectionRoots, numPoints, numIndices = 0;
  PetscInt          *localIndices;
  PetscSFNode       *remoteIndices;
  PetscInt          i, ind;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidPointer(rootSection,2);
  /* Cannot check PetscValidIntPointer(remoteOffsets,3) because it can be NULL if sf does not reference any points in leafSection */
  PetscValidPointer(leafSection,4);
  PetscValidPointer(sectionSF,5);
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm, sectionSF);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(leafSection, &lpStart, &lpEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(rootSection, &numSectionRoots);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, &numRoots, &numPoints, &localPoints, &remotePoints);CHKERRQ(ierr);
  if (numRoots < 0) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(PETSCSF_SectSF,sf,0,0,0);CHKERRQ(ierr);
  for (i = 0; i < numPoints; ++i) {
    PetscInt localPoint = localPoints ? localPoints[i] : i;
    PetscInt dof;

    if ((localPoint >= lpStart) && (localPoint < lpEnd)) {
      ierr = PetscSectionGetDof(leafSection, localPoint, &dof);CHKERRQ(ierr);
      numIndices += dof;
    }
  }
  ierr = PetscMalloc1(numIndices, &localIndices);CHKERRQ(ierr);
  ierr = PetscMalloc1(numIndices, &remoteIndices);CHKERRQ(ierr);
  /* Create new index graph */
  for (i = 0, ind = 0; i < numPoints; ++i) {
    PetscInt localPoint = localPoints ? localPoints[i] : i;
    PetscInt rank       = remotePoints[i].rank;

    if ((localPoint >= lpStart) && (localPoint < lpEnd)) {
      PetscInt remoteOffset = remoteOffsets[localPoint-lpStart];
      PetscInt loff, dof, d;

      ierr = PetscSectionGetOffset(leafSection, localPoint, &loff);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(leafSection, localPoint, &dof);CHKERRQ(ierr);
      for (d = 0; d < dof; ++d, ++ind) {
        localIndices[ind]        = loff+d;
        remoteIndices[ind].rank  = rank;
        remoteIndices[ind].index = remoteOffset+d;
      }
    }
  }
  PetscCheckFalse(numIndices != ind,comm, PETSC_ERR_PLIB, "Inconsistency in indices, %" PetscInt_FMT " should be %" PetscInt_FMT, ind, numIndices);
  ierr = PetscSFSetGraph(*sectionSF, numSectionRoots, numIndices, localIndices, PETSC_OWN_POINTER, remoteIndices, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFSetUp(*sectionSF);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSCSF_SectSF,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFCreateFromLayouts - Creates a parallel star forest mapping two PetscLayout objects

   Collective

   Input Parameters:
+  rmap - PetscLayout defining the global root space
-  lmap - PetscLayout defining the global leaf space

   Output Parameter:
.  sf - The parallel star forest

   Level: intermediate

.seealso: PetscSFCreate(), PetscLayoutCreate(), PetscSFSetGraphLayout()
@*/
PetscErrorCode PetscSFCreateFromLayouts(PetscLayout rmap, PetscLayout lmap, PetscSF* sf)
{
  PetscErrorCode ierr;
  PetscInt       i,nroots,nleaves = 0;
  PetscInt       rN, lst, len;
  PetscMPIInt    owner = -1;
  PetscSFNode    *remote;
  MPI_Comm       rcomm = rmap->comm;
  MPI_Comm       lcomm = lmap->comm;
  PetscMPIInt    flg;

  PetscFunctionBegin;
  PetscValidPointer(sf,3);
  PetscCheckFalse(!rmap->setupcalled,rcomm,PETSC_ERR_ARG_WRONGSTATE,"Root layout not setup");
  PetscCheckFalse(!lmap->setupcalled,lcomm,PETSC_ERR_ARG_WRONGSTATE,"Leaf layout not setup");
  ierr = MPI_Comm_compare(rcomm,lcomm,&flg);CHKERRMPI(ierr);
  PetscCheckFalse(flg != MPI_CONGRUENT && flg != MPI_IDENT,rcomm,PETSC_ERR_SUP,"cannot map two layouts with non-matching communicators");
  ierr = PetscSFCreate(rcomm,sf);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(rmap,&nroots);CHKERRQ(ierr);
  ierr = PetscLayoutGetSize(rmap,&rN);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(lmap,&lst,&len);CHKERRQ(ierr);
  ierr = PetscMalloc1(len-lst,&remote);CHKERRQ(ierr);
  for (i = lst; i < len && i < rN; i++) {
    if (owner < -1 || i >= rmap->range[owner+1]) {
      ierr = PetscLayoutFindOwner(rmap,i,&owner);CHKERRQ(ierr);
    }
    remote[nleaves].rank  = owner;
    remote[nleaves].index = i - rmap->range[owner];
    nleaves++;
  }
  ierr = PetscSFSetGraph(*sf,nroots,nleaves,NULL,PETSC_OWN_POINTER,remote,PETSC_COPY_VALUES);CHKERRQ(ierr);
  ierr = PetscFree(remote);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* TODO: handle nooffprocentries like MatZeroRowsMapLocal_Private, since this code is the same */
PetscErrorCode PetscLayoutMapLocal(PetscLayout map,PetscInt N,const PetscInt idxs[], PetscInt *on,PetscInt **oidxs,PetscInt **ogidxs)
{
  PetscInt      *owners = map->range;
  PetscInt       n      = map->n;
  PetscSF        sf;
  PetscInt      *lidxs,*work = NULL;
  PetscSFNode   *ridxs;
  PetscMPIInt    rank, p = 0;
  PetscInt       r, len = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (on) *on = 0;              /* squelch -Wmaybe-uninitialized */
  /* Create SF where leaves are input idxs and roots are owned idxs */
  ierr = MPI_Comm_rank(map->comm,&rank);CHKERRMPI(ierr);
  ierr = PetscMalloc1(n,&lidxs);CHKERRQ(ierr);
  for (r = 0; r < n; ++r) lidxs[r] = -1;
  ierr = PetscMalloc1(N,&ridxs);CHKERRQ(ierr);
  for (r = 0; r < N; ++r) {
    const PetscInt idx = idxs[r];
    PetscCheckFalse(idx < 0 || map->N <= idx,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index %" PetscInt_FMT " out of range [0,%" PetscInt_FMT ")",idx,map->N);
    if (idx < owners[p] || owners[p+1] <= idx) { /* short-circuit the search if the last p owns this idx too */
      ierr = PetscLayoutFindOwner(map,idx,&p);CHKERRQ(ierr);
    }
    ridxs[r].rank = p;
    ridxs[r].index = idxs[r] - owners[p];
  }
  ierr = PetscSFCreate(map->comm,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,n,N,NULL,PETSC_OWN_POINTER,ridxs,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(sf,MPIU_INT,(PetscInt*)idxs,lidxs,MPI_LOR);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(sf,MPIU_INT,(PetscInt*)idxs,lidxs,MPI_LOR);CHKERRQ(ierr);
  if (ogidxs) { /* communicate global idxs */
    PetscInt cum = 0,start,*work2;

    ierr = PetscMalloc1(n,&work);CHKERRQ(ierr);
    ierr = PetscCalloc1(N,&work2);CHKERRQ(ierr);
    for (r = 0; r < N; ++r) if (idxs[r] >=0) cum++;
    ierr = MPI_Scan(&cum,&start,1,MPIU_INT,MPI_SUM,map->comm);CHKERRMPI(ierr);
    start -= cum;
    cum = 0;
    for (r = 0; r < N; ++r) if (idxs[r] >=0) work2[r] = start+cum++;
    ierr = PetscSFReduceBegin(sf,MPIU_INT,work2,work,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(sf,MPIU_INT,work2,work,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscFree(work2);CHKERRQ(ierr);
  }
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  /* Compress and put in indices */
  for (r = 0; r < n; ++r)
    if (lidxs[r] >= 0) {
      if (work) work[len] = work[r];
      lidxs[len++] = r;
    }
  if (on) *on = len;
  if (oidxs) *oidxs = lidxs;
  if (ogidxs) *ogidxs = work;
  PetscFunctionReturn(0);
}

/*@
  PetscSFCreateByMatchingIndices - Create SF by matching root and leaf indices

  Collective

  Input Parameters:
+ layout - PetscLayout defining the global index space and the rank that brokers each index
. numRootIndices - size of rootIndices
. rootIndices - PetscInt array of global indices of which this process requests ownership
. rootLocalIndices - root local index permutation (NULL if no permutation)
. rootLocalOffset - offset to be added to root local indices
. numLeafIndices - size of leafIndices
. leafIndices - PetscInt array of global indices with which this process requires data associated
. leafLocalIndices - leaf local index permutation (NULL if no permutation)
- leafLocalOffset - offset to be added to leaf local indices

  Output Parameters:
+ sfA - star forest representing the communication pattern from the layout space to the leaf space (NULL if not needed)
- sf - star forest representing the communication pattern from the root space to the leaf space

  Example 1:
$
$  rank             : 0            1            2
$  rootIndices      : [1 0 2]      [3]          [3]
$  rootLocalOffset  : 100          200          300
$  layout           : [0 1]        [2]          [3]
$  leafIndices      : [0]          [2]          [0 3]
$  leafLocalOffset  : 400          500          600
$
would build the following SF
$
$  [0] 400 <- (0,101)
$  [1] 500 <- (0,102)
$  [2] 600 <- (0,101)
$  [2] 601 <- (2,300)
$
  Example 2:
$
$  rank             : 0               1               2
$  rootIndices      : [1 0 2]         [3]             [3]
$  rootLocalOffset  : 100             200             300
$  layout           : [0 1]           [2]             [3]
$  leafIndices      : rootIndices     rootIndices     rootIndices
$  leafLocalOffset  : rootLocalOffset rootLocalOffset rootLocalOffset
$
would build the following SF
$
$  [1] 200 <- (2,300)
$
  Example 3:
$
$  No process requests ownership of global index 1, but no process needs it.
$
$  rank             : 0            1            2
$  numRootIndices   : 2            1            1
$  rootIndices      : [0 2]        [3]          [3]
$  rootLocalOffset  : 100          200          300
$  layout           : [0 1]        [2]          [3]
$  numLeafIndices   : 1            1            2
$  leafIndices      : [0]          [2]          [0 3]
$  leafLocalOffset  : 400          500          600
$
would build the following SF
$
$  [0] 400 <- (0,100)
$  [1] 500 <- (0,101)
$  [2] 600 <- (0,100)
$  [2] 601 <- (2,300)
$

  Notes:
  The layout parameter represents any partitioning of [0, N), where N is the total number of global indices, and its
  local size can be set to PETSC_DECIDE.
  If a global index x lies in the partition owned by process i, each process whose rootIndices contains x requests
  ownership of x and sends its own rank and the local index of x to process i.
  If multiple processes request ownership of x, the one with the highest rank is to own x.
  Process i then broadcasts the ownership information, so that each process whose leafIndices contains x knows the
  ownership information of x.
  The output sf is constructed by associating each leaf point to a root point in this way.

  Suppose there is point data ordered according to the global indices and partitioned according to the given layout.
  The optional output PetscSF object sfA can be used to push such data to leaf points.

  All indices in rootIndices and leafIndices must lie in the layout range. The union (over all processes) of rootIndices
  must cover that of leafIndices, but need not cover the entire layout.

  If (leafIndices, leafLocalIndices, leafLocalOffset) == (rootIndices, rootLocalIndices, rootLocalOffset), the output
  star forest is almost identity, so will only include non-trivial part of the map.

  Developer Notes:
  Current approach of a process of the highest rank gaining the ownership may cause load imbalance; consider using
  hash(rank, root_local_index) as the bid for the ownership determination.

  Level: advanced

.seealso: PetscSFCreate()
@*/
PetscErrorCode PetscSFCreateByMatchingIndices(PetscLayout layout, PetscInt numRootIndices, const PetscInt *rootIndices, const PetscInt *rootLocalIndices, PetscInt rootLocalOffset, PetscInt numLeafIndices, const PetscInt *leafIndices, const PetscInt *leafLocalIndices, PetscInt leafLocalOffset, PetscSF *sfA, PetscSF *sf)
{
  MPI_Comm        comm = layout->comm;
  PetscMPIInt     size, rank;
  PetscSF         sf1;
  PetscSFNode    *owners, *buffer, *iremote;
  PetscInt       *ilocal, nleaves, N, n, i;
#if defined(PETSC_USE_DEBUG)
  PetscInt        N1;
#endif
  PetscBool       flag;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (rootIndices)      PetscValidIntPointer(rootIndices,3);
  if (rootLocalIndices) PetscValidIntPointer(rootLocalIndices,4);
  if (leafIndices)      PetscValidIntPointer(leafIndices,7);
  if (leafLocalIndices) PetscValidIntPointer(leafLocalIndices,8);
  if (sfA)              PetscValidPointer(sfA,10);
  PetscValidPointer(sf,11);
  PetscCheckFalse(numRootIndices < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "numRootIndices (%" PetscInt_FMT ") must be non-negative", numRootIndices);
  PetscCheckFalse(numLeafIndices < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "numLeafIndices (%" PetscInt_FMT ") must be non-negative", numLeafIndices);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = PetscLayoutSetUp(layout);CHKERRQ(ierr);
  ierr = PetscLayoutGetSize(layout, &N);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(layout, &n);CHKERRQ(ierr);
  flag = (PetscBool)(leafIndices == rootIndices);
  ierr = MPIU_Allreduce(MPI_IN_PLACE, &flag, 1, MPIU_BOOL, MPI_LAND, comm);CHKERRMPI(ierr);
  PetscCheckFalse(flag && numLeafIndices != numRootIndices,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "leafIndices == rootIndices, but numLeafIndices (%" PetscInt_FMT ") != numRootIndices(%" PetscInt_FMT ")", numLeafIndices, numRootIndices);
#if defined(PETSC_USE_DEBUG)
  N1 = PETSC_MIN_INT;
  for (i = 0; i < numRootIndices; i++) if (rootIndices[i] > N1) N1 = rootIndices[i];
  ierr = MPI_Allreduce(MPI_IN_PLACE, &N1, 1, MPIU_INT, MPI_MAX, comm);CHKERRMPI(ierr);
  PetscCheckFalse(N1 >= N,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Max. root index (%" PetscInt_FMT ") out of layout range [0,%" PetscInt_FMT ")", N1, N);
  if (!flag) {
    N1 = PETSC_MIN_INT;
    for (i = 0; i < numLeafIndices; i++) if (leafIndices[i] > N1) N1 = leafIndices[i];
    ierr = MPI_Allreduce(MPI_IN_PLACE, &N1, 1, MPIU_INT, MPI_MAX, comm);CHKERRMPI(ierr);
    PetscCheckFalse(N1 >= N,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Max. leaf index (%" PetscInt_FMT ") out of layout range [0,%" PetscInt_FMT ")", N1, N);
  }
#endif
  /* Reduce: owners -> buffer */
  ierr = PetscMalloc1(n, &buffer);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm, &sf1);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf1);CHKERRQ(ierr);
  ierr = PetscSFSetGraphLayout(sf1, layout, numRootIndices, NULL, PETSC_OWN_POINTER, rootIndices);CHKERRQ(ierr);
  ierr = PetscMalloc1(numRootIndices, &owners);CHKERRQ(ierr);
  for (i = 0; i < numRootIndices; ++i) {
    owners[i].rank = rank;
    owners[i].index = rootLocalOffset + (rootLocalIndices ? rootLocalIndices[i] : i);
  }
  for (i = 0; i < n; ++i) {
    buffer[i].index = -1;
    buffer[i].rank = -1;
  }
  ierr = PetscSFReduceBegin(sf1, MPIU_2INT, owners, buffer, MPI_MAXLOC);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(sf1, MPIU_2INT, owners, buffer, MPI_MAXLOC);CHKERRQ(ierr);
  /* Bcast: buffer -> owners */
  if (!flag) {
    /* leafIndices is different from rootIndices */
    ierr = PetscFree(owners);CHKERRQ(ierr);
    ierr = PetscSFSetGraphLayout(sf1, layout, numLeafIndices, NULL, PETSC_OWN_POINTER, leafIndices);CHKERRQ(ierr);
    ierr = PetscMalloc1(numLeafIndices, &owners);CHKERRQ(ierr);
  }
  ierr = PetscSFBcastBegin(sf1, MPIU_2INT, buffer, owners, MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf1, MPIU_2INT, buffer, owners, MPI_REPLACE);CHKERRQ(ierr);
  for (i = 0; i < numLeafIndices; ++i) PetscCheckFalse(owners[i].rank < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Global point %" PetscInt_FMT " was unclaimed", leafIndices[i]);
  ierr = PetscFree(buffer);CHKERRQ(ierr);
  if (sfA) {*sfA = sf1;}
  else     {ierr = PetscSFDestroy(&sf1);CHKERRQ(ierr);}
  /* Create sf */
  if (flag && rootLocalIndices == leafLocalIndices && leafLocalOffset == rootLocalOffset) {
    /* leaf space == root space */
    for (i = 0, nleaves = 0; i < numLeafIndices; ++i) if (owners[i].rank != rank) ++nleaves;
    ierr = PetscMalloc1(nleaves, &ilocal);CHKERRQ(ierr);
    ierr = PetscMalloc1(nleaves, &iremote);CHKERRQ(ierr);
    for (i = 0, nleaves = 0; i < numLeafIndices; ++i) {
      if (owners[i].rank != rank) {
        ilocal[nleaves]        = leafLocalOffset + i;
        iremote[nleaves].rank  = owners[i].rank;
        iremote[nleaves].index = owners[i].index;
        ++nleaves;
      }
    }
    ierr = PetscFree(owners);CHKERRQ(ierr);
  } else {
    nleaves = numLeafIndices;
    ierr = PetscMalloc1(nleaves, &ilocal);CHKERRQ(ierr);
    for (i = 0; i < nleaves; ++i) {ilocal[i] = leafLocalOffset + (leafLocalIndices ? leafLocalIndices[i] : i);}
    iremote = owners;
  }
  ierr = PetscSFCreate(comm, sf);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(*sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*sf, rootLocalOffset + numRootIndices, nleaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
