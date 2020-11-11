#include <petsc/private/sfimpl.h>       /*I  "petscsf.h"   I*/
#include <petsc/private/sectionimpl.h>

/*@C
   PetscSFSetGraphLayout - Set a parallel star forest via global indices and a PetscLayout

   Collective

   Input Arguments:
+  sf - star forest
.  layout - PetscLayout defining the global space
.  nleaves - number of leaf vertices on the current process, each of these references a root on any process
.  ilocal - locations of leaves in leafdata buffers, pass NULL for contiguous storage
.  localmode - copy mode for ilocal
-  iremote - remote locations of root vertices for each leaf on the current process

   Level: intermediate

   Developers Note: Local indices which are the identity permutation in the range [0,nleaves) are discarded as they
   encode contiguous storage. In such case, if localmode is PETSC_OWN_POINTER, the memory is deallocated as it is not
   needed

.seealso: PetscSFCreate(), PetscSFView(), PetscSFSetGraph(), PetscSFGetGraph()
@*/
PetscErrorCode PetscSFSetGraphLayout(PetscSF sf,PetscLayout layout,PetscInt nleaves,const PetscInt *ilocal,PetscCopyMode localmode,const PetscInt *iremote)
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
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
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
    if (gcdof > (gdof < 0 ? -(gdof+1) : gdof)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point %D has %D constraints > %D dof", p, gcdof, (gdof < 0 ? -(gdof+1) : gdof));
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
      if (gsize != dof) SETERRQ4(comm, PETSC_ERR_ARG_WRONG, "Global dof %D for point %D is neither the constrained size %D, nor the unconstrained %D", gsize, p, dof-cdof, dof);
      cdof = 0; /* Ignore constraints */
    }
    for (d = 0, c = 0; d < dof; ++d) {
      if ((c < cdof) && (cind[c] == d)) {++c; continue;}
      local[l+d-c] = off+d;
    }
    if (gdof < 0) {
      for (d = 0; d < gsize; ++d, ++l) {
        PetscInt offset = -(goff+1) + d, r;

        ierr = PetscFindInt(offset,size+1,ranges,&r);CHKERRQ(ierr);
        if (r < 0) r = -(r+2);
        if ((r < 0) || (r >= size)) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point %D mapped to invalid process %D (%D, %D)", p, r, gdof, goff);
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
  if (l != nleaves) SETERRQ2(comm, PETSC_ERR_PLIB, "Iteration error, l %D != nleaves %D", l, nleaves);
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
  if (numFields) {ierr = PetscSectionSetNumFields(leafSection, numFields);CHKERRQ(ierr);}
  ierr = PetscMalloc1(numFields+2, &sub);CHKERRQ(ierr);
  sub[1] = rootSection->bc ? PETSC_TRUE : PETSC_FALSE;
  for (f = 0; f < numFields; ++f) {
    PetscSectionSym sym;
    const char      *name   = NULL;
    PetscInt        numComp = 0;

    sub[2 + f] = rootSection->field[f]->bc ? PETSC_TRUE : PETSC_FALSE;
    ierr = PetscSectionGetFieldComponents(rootSection, f, &numComp);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldName(rootSection, f, &name);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldSym(rootSection, f, &sym);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(leafSection, f, numComp);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(leafSection, f, name);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldSym(leafSection, f, sym);CHKERRQ(ierr);
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
  ierr = MPIU_Allreduce(MPI_IN_PLACE, sub, 2+numFields, MPIU_BOOL, MPI_LOR, PetscObjectComm((PetscObject)sf));CHKERRQ(ierr);
  if (sub[0]) {
    ierr = ISCreateStride(PETSC_COMM_SELF, rpEnd - rpStart, rpStart, 1, &selected);CHKERRQ(ierr);
    ierr = ISGetIndices(selected, &indices);CHKERRQ(ierr);
    ierr = PetscSFCreateEmbeddedSF(sf, rpEnd - rpStart, indices, &embedSF);CHKERRQ(ierr);
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
  ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->atlasDof[-rpStart], &leafSection->atlasDof[-lpStart]);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->atlasDof[-rpStart], &leafSection->atlasDof[-lpStart]);CHKERRQ(ierr);
  if (sub[1]) {
    ierr = PetscSectionCheckConstraints_Static(rootSection);CHKERRQ(ierr);
    ierr = PetscSectionCheckConstraints_Static(leafSection);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->bc->atlasDof[-rpStart], &leafSection->bc->atlasDof[-lpStart]);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->bc->atlasDof[-rpStart], &leafSection->bc->atlasDof[-lpStart]);CHKERRQ(ierr);
  }
  for (f = 0; f < numFields; ++f) {
    ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->field[f]->atlasDof[-rpStart], &leafSection->field[f]->atlasDof[-lpStart]);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->field[f]->atlasDof[-rpStart], &leafSection->field[f]->atlasDof[-lpStart]);CHKERRQ(ierr);
    if (sub[2+f]) {
      ierr = PetscSectionCheckConstraints_Static(rootSection->field[f]);CHKERRQ(ierr);
      ierr = PetscSectionCheckConstraints_Static(leafSection->field[f]);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->field[f]->bc->atlasDof[-rpStart], &leafSection->field[f]->bc->atlasDof[-lpStart]);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->field[f]->bc->atlasDof[-rpStart], &leafSection->field[f]->bc->atlasDof[-lpStart]);CHKERRQ(ierr);
    }
  }
  if (remoteOffsets) {
    ierr = PetscMalloc1(lpEnd - lpStart, remoteOffsets);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->atlasOff[-rpStart], &(*remoteOffsets)[-lpStart]);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->atlasOff[-rpStart], &(*remoteOffsets)[-lpStart]);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(leafSection);CHKERRQ(ierr);
  if (hasc) { /* need to communicate bcIndices */
    PetscSF  bcSF;
    PetscInt *rOffBc;

    ierr = PetscMalloc1(lpEnd - lpStart, &rOffBc);CHKERRQ(ierr);
    if (sub[1]) {
      ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->bc->atlasOff[-rpStart], &rOffBc[-lpStart]);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->bc->atlasOff[-rpStart], &rOffBc[-lpStart]);CHKERRQ(ierr);
      ierr = PetscSFCreateSectionSF(embedSF, rootSection->bc, rOffBc, leafSection->bc, &bcSF);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(bcSF, MPIU_INT, rootSection->bcIndices, leafSection->bcIndices);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(bcSF, MPIU_INT, rootSection->bcIndices, leafSection->bcIndices);CHKERRQ(ierr);
      ierr = PetscSFDestroy(&bcSF);CHKERRQ(ierr);
    }
    for (f = 0; f < numFields; ++f) {
      if (sub[2+f]) {
        ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->field[f]->bc->atlasOff[-rpStart], &rOffBc[-lpStart]);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->field[f]->bc->atlasOff[-rpStart], &rOffBc[-lpStart]);CHKERRQ(ierr);
        ierr = PetscSFCreateSectionSF(embedSF, rootSection->field[f]->bc, rOffBc, leafSection->field[f]->bc, &bcSF);CHKERRQ(ierr);
        ierr = PetscSFBcastBegin(bcSF, MPIU_INT, rootSection->field[f]->bcIndices, leafSection->field[f]->bcIndices);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(bcSF, MPIU_INT, rootSection->field[f]->bcIndices, leafSection->field[f]->bcIndices);CHKERRQ(ierr);
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
  ierr = PetscSFCreateEmbeddedSF(sf, rpEnd - rpStart, indices, &embedSF);CHKERRQ(ierr);
  ierr = ISRestoreIndices(selected, &indices);CHKERRQ(ierr);
  ierr = ISDestroy(&selected);CHKERRQ(ierr);
  ierr = PetscCalloc1(lpEnd - lpStart, remoteOffsets);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->atlasOff[-rpStart], &(*remoteOffsets)[-lpStart]);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->atlasOff[-rpStart], &(*remoteOffsets)[-lpStart]);CHKERRQ(ierr);
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
  if (numIndices != ind) SETERRQ2(comm, PETSC_ERR_PLIB, "Inconsistency in indices, %D should be %D", ind, numIndices);
  ierr = PetscSFSetGraph(*sectionSF, numSectionRoots, numIndices, localIndices, PETSC_OWN_POINTER, remoteIndices, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFSetUp(*sectionSF);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSCSF_SectSF,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
