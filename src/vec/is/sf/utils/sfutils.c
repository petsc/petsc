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
  const PetscInt *range;
  PetscInt       i, nroots, ls = -1, ln = -1;
  PetscMPIInt    lr = -1;
  PetscSFNode    *remote;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetLocalSize(layout,&nroots));
  PetscCall(PetscLayoutGetRanges(layout,&range));
  PetscCall(PetscMalloc1(nleaves,&remote));
  if (nleaves) { ls = iremote[0] + 1; }
  for (i=0; i<nleaves; i++) {
    const PetscInt idx = iremote[i] - ls;
    if (idx < 0 || idx >= ln) { /* short-circuit the search */
      PetscCall(PetscLayoutFindOwnerIndex(layout,iremote[i],&lr,&remote[i].index));
      remote[i].rank = lr;
      ls = range[lr];
      ln = range[lr+1] - ls;
    } else {
      remote[i].rank  = lr;
      remote[i].index = idx;
    }
  }
  PetscCall(PetscSFSetGraph(sf,nroots,nleaves,ilocal,localmode,remote,PETSC_OWN_POINTER));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscValidHeaderSpecific(localSection, PETSC_SECTION_CLASSID, 2);
  PetscValidHeaderSpecific(globalSection, PETSC_SECTION_CLASSID, 3);

  PetscCall(PetscObjectGetComm((PetscObject)sf,&comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscSectionGetChart(globalSection, &pStart, &pEnd));
  PetscCall(PetscSectionGetConstrainedStorageSize(globalSection, &nroots));
  PetscCall(PetscLayoutCreate(comm, &layout));
  PetscCall(PetscLayoutSetBlockSize(layout, 1));
  PetscCall(PetscLayoutSetLocalSize(layout, nroots));
  PetscCall(PetscLayoutSetUp(layout));
  PetscCall(PetscLayoutGetRanges(layout, &ranges));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt gdof, gcdof;

    PetscCall(PetscSectionGetDof(globalSection, p, &gdof));
    PetscCall(PetscSectionGetConstraintDof(globalSection, p, &gcdof));
    PetscCheckFalse(gcdof > (gdof < 0 ? -(gdof+1) : gdof),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point %" PetscInt_FMT " has %" PetscInt_FMT " constraints > %" PetscInt_FMT " dof", p, gcdof, (gdof < 0 ? -(gdof+1) : gdof));
    nleaves += gdof < 0 ? -(gdof+1)-gcdof : gdof-gcdof;
  }
  PetscCall(PetscMalloc1(nleaves, &local));
  PetscCall(PetscMalloc1(nleaves, &remote));
  for (p = pStart, l = 0; p < pEnd; ++p) {
    const PetscInt *cind;
    PetscInt       dof, cdof, off, gdof, gcdof, goff, gsize, d, c;

    PetscCall(PetscSectionGetDof(localSection, p, &dof));
    PetscCall(PetscSectionGetOffset(localSection, p, &off));
    PetscCall(PetscSectionGetConstraintDof(localSection, p, &cdof));
    PetscCall(PetscSectionGetConstraintIndices(localSection, p, &cind));
    PetscCall(PetscSectionGetDof(globalSection, p, &gdof));
    PetscCall(PetscSectionGetConstraintDof(globalSection, p, &gcdof));
    PetscCall(PetscSectionGetOffset(globalSection, p, &goff));
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

        PetscCall(PetscFindInt(offset,size+1,ranges,&r));
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
  PetscCall(PetscLayoutDestroy(&layout));
  PetscCall(PetscSFSetGraph(sf, nroots, nleaves, local, PETSC_OWN_POINTER, remote, PETSC_OWN_POINTER));
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

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(PETSCSF_DistSect,sf,0,0,0));
  PetscCall(PetscSectionGetNumFields(rootSection, &numFields));
  if (numFields) {
    IS perm;

    /* PetscSectionSetNumFields() calls PetscSectionReset(), which destroys
       leafSection->perm. To keep this permutation set by the user, we grab
       the reference before calling PetscSectionSetNumFields() and set it
       back after. */
    PetscCall(PetscSectionGetPermutation(leafSection, &perm));
    PetscCall(PetscObjectReference((PetscObject)perm));
    PetscCall(PetscSectionSetNumFields(leafSection, numFields));
    PetscCall(PetscSectionSetPermutation(leafSection, perm));
    PetscCall(ISDestroy(&perm));
  }
  PetscCall(PetscMalloc1(numFields+2, &sub));
  sub[1] = rootSection->bc ? PETSC_TRUE : PETSC_FALSE;
  for (f = 0; f < numFields; ++f) {
    PetscSectionSym sym, dsym = NULL;
    const char      *name   = NULL;
    PetscInt        numComp = 0;

    sub[2 + f] = rootSection->field[f]->bc ? PETSC_TRUE : PETSC_FALSE;
    PetscCall(PetscSectionGetFieldComponents(rootSection, f, &numComp));
    PetscCall(PetscSectionGetFieldName(rootSection, f, &name));
    PetscCall(PetscSectionGetFieldSym(rootSection, f, &sym));
    if (sym) PetscCall(PetscSectionSymDistribute(sym, sf, &dsym));
    PetscCall(PetscSectionSetFieldComponents(leafSection, f, numComp));
    PetscCall(PetscSectionSetFieldName(leafSection, f, name));
    PetscCall(PetscSectionSetFieldSym(leafSection, f, dsym));
    PetscCall(PetscSectionSymDestroy(&dsym));
    for (c = 0; c < rootSection->numFieldComponents[f]; ++c) {
      PetscCall(PetscSectionGetComponentName(rootSection, f, c, &name));
      PetscCall(PetscSectionSetComponentName(leafSection, f, c, name));
    }
  }
  PetscCall(PetscSectionGetChart(rootSection, &rpStart, &rpEnd));
  PetscCall(PetscSFGetGraph(sf,&nroots,NULL,NULL,NULL));
  rpEnd = PetscMin(rpEnd,nroots);
  rpEnd = PetscMax(rpStart,rpEnd);
  /* see if we can avoid creating the embedded SF, since it can cost more than an allreduce */
  sub[0] = (PetscBool)(nroots != rpEnd - rpStart);
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE, sub, 2+numFields, MPIU_BOOL, MPI_LOR, PetscObjectComm((PetscObject)sf)));
  if (sub[0]) {
    PetscCall(ISCreateStride(PETSC_COMM_SELF, rpEnd - rpStart, rpStart, 1, &selected));
    PetscCall(ISGetIndices(selected, &indices));
    PetscCall(PetscSFCreateEmbeddedRootSF(sf, rpEnd - rpStart, indices, &embedSF));
    PetscCall(ISRestoreIndices(selected, &indices));
    PetscCall(ISDestroy(&selected));
  } else {
    PetscCall(PetscObjectReference((PetscObject)sf));
    embedSF = sf;
  }
  PetscCall(PetscSFGetLeafRange(embedSF, &lpStart, &lpEnd));
  lpEnd++;

  PetscCall(PetscSectionSetChart(leafSection, lpStart, lpEnd));

  /* Constrained dof section */
  hasc = sub[1];
  for (f = 0; f < numFields; ++f) hasc = (PetscBool)(hasc || sub[2+f]);

  /* Could fuse these at the cost of copies and extra allocation */
  PetscCall(PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->atlasDof[-rpStart], &leafSection->atlasDof[-lpStart],MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->atlasDof[-rpStart], &leafSection->atlasDof[-lpStart],MPI_REPLACE));
  if (sub[1]) {
    PetscCall(PetscSectionCheckConstraints_Private(rootSection));
    PetscCall(PetscSectionCheckConstraints_Private(leafSection));
    PetscCall(PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->bc->atlasDof[-rpStart], &leafSection->bc->atlasDof[-lpStart],MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->bc->atlasDof[-rpStart], &leafSection->bc->atlasDof[-lpStart],MPI_REPLACE));
  }
  for (f = 0; f < numFields; ++f) {
    PetscCall(PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->field[f]->atlasDof[-rpStart], &leafSection->field[f]->atlasDof[-lpStart],MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->field[f]->atlasDof[-rpStart], &leafSection->field[f]->atlasDof[-lpStart],MPI_REPLACE));
    if (sub[2+f]) {
      PetscCall(PetscSectionCheckConstraints_Private(rootSection->field[f]));
      PetscCall(PetscSectionCheckConstraints_Private(leafSection->field[f]));
      PetscCall(PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->field[f]->bc->atlasDof[-rpStart], &leafSection->field[f]->bc->atlasDof[-lpStart],MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->field[f]->bc->atlasDof[-rpStart], &leafSection->field[f]->bc->atlasDof[-lpStart],MPI_REPLACE));
    }
  }
  if (remoteOffsets) {
    PetscCall(PetscMalloc1(lpEnd - lpStart, remoteOffsets));
    PetscCall(PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->atlasOff[-rpStart], &(*remoteOffsets)[-lpStart],MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->atlasOff[-rpStart], &(*remoteOffsets)[-lpStart],MPI_REPLACE));
  }
  PetscCall(PetscSectionInvalidateMaxDof_Internal(leafSection));
  PetscCall(PetscSectionSetUp(leafSection));
  if (hasc) { /* need to communicate bcIndices */
    PetscSF  bcSF;
    PetscInt *rOffBc;

    PetscCall(PetscMalloc1(lpEnd - lpStart, &rOffBc));
    if (sub[1]) {
      PetscCall(PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->bc->atlasOff[-rpStart], &rOffBc[-lpStart],MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->bc->atlasOff[-rpStart], &rOffBc[-lpStart],MPI_REPLACE));
      PetscCall(PetscSFCreateSectionSF(embedSF, rootSection->bc, rOffBc, leafSection->bc, &bcSF));
      PetscCall(PetscSFBcastBegin(bcSF, MPIU_INT, rootSection->bcIndices, leafSection->bcIndices,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(bcSF, MPIU_INT, rootSection->bcIndices, leafSection->bcIndices,MPI_REPLACE));
      PetscCall(PetscSFDestroy(&bcSF));
    }
    for (f = 0; f < numFields; ++f) {
      if (sub[2+f]) {
        PetscCall(PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->field[f]->bc->atlasOff[-rpStart], &rOffBc[-lpStart],MPI_REPLACE));
        PetscCall(PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->field[f]->bc->atlasOff[-rpStart], &rOffBc[-lpStart],MPI_REPLACE));
        PetscCall(PetscSFCreateSectionSF(embedSF, rootSection->field[f]->bc, rOffBc, leafSection->field[f]->bc, &bcSF));
        PetscCall(PetscSFBcastBegin(bcSF, MPIU_INT, rootSection->field[f]->bcIndices, leafSection->field[f]->bcIndices,MPI_REPLACE));
        PetscCall(PetscSFBcastEnd(bcSF, MPIU_INT, rootSection->field[f]->bcIndices, leafSection->field[f]->bcIndices,MPI_REPLACE));
        PetscCall(PetscSFDestroy(&bcSF));
      }
    }
    PetscCall(PetscFree(rOffBc));
  }
  PetscCall(PetscSFDestroy(&embedSF));
  PetscCall(PetscFree(sub));
  PetscCall(PetscLogEventEnd(PETSCSF_DistSect,sf,0,0,0));
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

  PetscFunctionBegin;
  *remoteOffsets = NULL;
  PetscCall(PetscSFGetGraph(sf, &numRoots, NULL, NULL, NULL));
  if (numRoots < 0) PetscFunctionReturn(0);
  PetscCall(PetscLogEventBegin(PETSCSF_RemoteOff,sf,0,0,0));
  PetscCall(PetscSectionGetChart(rootSection, &rpStart, &rpEnd));
  PetscCall(PetscSectionGetChart(leafSection, &lpStart, &lpEnd));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, rpEnd - rpStart, rpStart, 1, &selected));
  PetscCall(ISGetIndices(selected, &indices));
  PetscCall(PetscSFCreateEmbeddedRootSF(sf, rpEnd - rpStart, indices, &embedSF));
  PetscCall(ISRestoreIndices(selected, &indices));
  PetscCall(ISDestroy(&selected));
  PetscCall(PetscCalloc1(lpEnd - lpStart, remoteOffsets));
  PetscCall(PetscSFBcastBegin(embedSF, MPIU_INT, &rootSection->atlasOff[-rpStart], &(*remoteOffsets)[-lpStart],MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(embedSF, MPIU_INT, &rootSection->atlasOff[-rpStart], &(*remoteOffsets)[-lpStart],MPI_REPLACE));
  PetscCall(PetscSFDestroy(&embedSF));
  PetscCall(PetscLogEventEnd(PETSCSF_RemoteOff,sf,0,0,0));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidPointer(rootSection,2);
  /* Cannot check PetscValidIntPointer(remoteOffsets,3) because it can be NULL if sf does not reference any points in leafSection */
  PetscValidPointer(leafSection,4);
  PetscValidPointer(sectionSF,5);
  PetscCall(PetscObjectGetComm((PetscObject)sf,&comm));
  PetscCall(PetscSFCreate(comm, sectionSF));
  PetscCall(PetscSectionGetChart(leafSection, &lpStart, &lpEnd));
  PetscCall(PetscSectionGetStorageSize(rootSection, &numSectionRoots));
  PetscCall(PetscSFGetGraph(sf, &numRoots, &numPoints, &localPoints, &remotePoints));
  if (numRoots < 0) PetscFunctionReturn(0);
  PetscCall(PetscLogEventBegin(PETSCSF_SectSF,sf,0,0,0));
  for (i = 0; i < numPoints; ++i) {
    PetscInt localPoint = localPoints ? localPoints[i] : i;
    PetscInt dof;

    if ((localPoint >= lpStart) && (localPoint < lpEnd)) {
      PetscCall(PetscSectionGetDof(leafSection, localPoint, &dof));
      numIndices += dof;
    }
  }
  PetscCall(PetscMalloc1(numIndices, &localIndices));
  PetscCall(PetscMalloc1(numIndices, &remoteIndices));
  /* Create new index graph */
  for (i = 0, ind = 0; i < numPoints; ++i) {
    PetscInt localPoint = localPoints ? localPoints[i] : i;
    PetscInt rank       = remotePoints[i].rank;

    if ((localPoint >= lpStart) && (localPoint < lpEnd)) {
      PetscInt remoteOffset = remoteOffsets[localPoint-lpStart];
      PetscInt loff, dof, d;

      PetscCall(PetscSectionGetOffset(leafSection, localPoint, &loff));
      PetscCall(PetscSectionGetDof(leafSection, localPoint, &dof));
      for (d = 0; d < dof; ++d, ++ind) {
        localIndices[ind]        = loff+d;
        remoteIndices[ind].rank  = rank;
        remoteIndices[ind].index = remoteOffset+d;
      }
    }
  }
  PetscCheckFalse(numIndices != ind,comm, PETSC_ERR_PLIB, "Inconsistency in indices, %" PetscInt_FMT " should be %" PetscInt_FMT, ind, numIndices);
  PetscCall(PetscSFSetGraph(*sectionSF, numSectionRoots, numIndices, localIndices, PETSC_OWN_POINTER, remoteIndices, PETSC_OWN_POINTER));
  PetscCall(PetscSFSetUp(*sectionSF));
  PetscCall(PetscLogEventEnd(PETSCSF_SectSF,sf,0,0,0));
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
  PetscInt       i,nroots,nleaves = 0;
  PetscInt       rN, lst, len;
  PetscMPIInt    owner = -1;
  PetscSFNode    *remote;
  MPI_Comm       rcomm = rmap->comm;
  MPI_Comm       lcomm = lmap->comm;
  PetscMPIInt    flg;

  PetscFunctionBegin;
  PetscValidPointer(sf,3);
  PetscCheck(rmap->setupcalled,rcomm,PETSC_ERR_ARG_WRONGSTATE,"Root layout not setup");
  PetscCheck(lmap->setupcalled,lcomm,PETSC_ERR_ARG_WRONGSTATE,"Leaf layout not setup");
  PetscCallMPI(MPI_Comm_compare(rcomm,lcomm,&flg));
  PetscCheckFalse(flg != MPI_CONGRUENT && flg != MPI_IDENT,rcomm,PETSC_ERR_SUP,"cannot map two layouts with non-matching communicators");
  PetscCall(PetscSFCreate(rcomm,sf));
  PetscCall(PetscLayoutGetLocalSize(rmap,&nroots));
  PetscCall(PetscLayoutGetSize(rmap,&rN));
  PetscCall(PetscLayoutGetRange(lmap,&lst,&len));
  PetscCall(PetscMalloc1(len-lst,&remote));
  for (i = lst; i < len && i < rN; i++) {
    if (owner < -1 || i >= rmap->range[owner+1]) {
      PetscCall(PetscLayoutFindOwner(rmap,i,&owner));
    }
    remote[nleaves].rank  = owner;
    remote[nleaves].index = i - rmap->range[owner];
    nleaves++;
  }
  PetscCall(PetscSFSetGraph(*sf,nroots,nleaves,NULL,PETSC_OWN_POINTER,remote,PETSC_COPY_VALUES));
  PetscCall(PetscFree(remote));
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

  PetscFunctionBegin;
  if (on) *on = 0;              /* squelch -Wmaybe-uninitialized */
  /* Create SF where leaves are input idxs and roots are owned idxs */
  PetscCallMPI(MPI_Comm_rank(map->comm,&rank));
  PetscCall(PetscMalloc1(n,&lidxs));
  for (r = 0; r < n; ++r) lidxs[r] = -1;
  PetscCall(PetscMalloc1(N,&ridxs));
  for (r = 0; r < N; ++r) {
    const PetscInt idx = idxs[r];
    PetscCheckFalse(idx < 0 || map->N <= idx,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index %" PetscInt_FMT " out of range [0,%" PetscInt_FMT ")",idx,map->N);
    if (idx < owners[p] || owners[p+1] <= idx) { /* short-circuit the search if the last p owns this idx too */
      PetscCall(PetscLayoutFindOwner(map,idx,&p));
    }
    ridxs[r].rank = p;
    ridxs[r].index = idxs[r] - owners[p];
  }
  PetscCall(PetscSFCreate(map->comm,&sf));
  PetscCall(PetscSFSetGraph(sf,n,N,NULL,PETSC_OWN_POINTER,ridxs,PETSC_OWN_POINTER));
  PetscCall(PetscSFReduceBegin(sf,MPIU_INT,(PetscInt*)idxs,lidxs,MPI_LOR));
  PetscCall(PetscSFReduceEnd(sf,MPIU_INT,(PetscInt*)idxs,lidxs,MPI_LOR));
  if (ogidxs) { /* communicate global idxs */
    PetscInt cum = 0,start,*work2;

    PetscCall(PetscMalloc1(n,&work));
    PetscCall(PetscCalloc1(N,&work2));
    for (r = 0; r < N; ++r) if (idxs[r] >=0) cum++;
    PetscCallMPI(MPI_Scan(&cum,&start,1,MPIU_INT,MPI_SUM,map->comm));
    start -= cum;
    cum = 0;
    for (r = 0; r < N; ++r) if (idxs[r] >=0) work2[r] = start+cum++;
    PetscCall(PetscSFReduceBegin(sf,MPIU_INT,work2,work,MPI_REPLACE));
    PetscCall(PetscSFReduceEnd(sf,MPIU_INT,work2,work,MPI_REPLACE));
    PetscCall(PetscFree(work2));
  }
  PetscCall(PetscSFDestroy(&sf));
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

  PetscFunctionBegin;
  if (rootIndices)      PetscValidIntPointer(rootIndices,3);
  if (rootLocalIndices) PetscValidIntPointer(rootLocalIndices,4);
  if (leafIndices)      PetscValidIntPointer(leafIndices,7);
  if (leafLocalIndices) PetscValidIntPointer(leafLocalIndices,8);
  if (sfA)              PetscValidPointer(sfA,10);
  PetscValidPointer(sf,11);
  PetscCheckFalse(numRootIndices < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "numRootIndices (%" PetscInt_FMT ") must be non-negative", numRootIndices);
  PetscCheckFalse(numLeafIndices < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "numLeafIndices (%" PetscInt_FMT ") must be non-negative", numLeafIndices);
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscLayoutSetUp(layout));
  PetscCall(PetscLayoutGetSize(layout, &N));
  PetscCall(PetscLayoutGetLocalSize(layout, &n));
  flag = (PetscBool)(leafIndices == rootIndices);
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE, &flag, 1, MPIU_BOOL, MPI_LAND, comm));
  PetscCheckFalse(flag && numLeafIndices != numRootIndices,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "leafIndices == rootIndices, but numLeafIndices (%" PetscInt_FMT ") != numRootIndices(%" PetscInt_FMT ")", numLeafIndices, numRootIndices);
#if defined(PETSC_USE_DEBUG)
  N1 = PETSC_MIN_INT;
  for (i = 0; i < numRootIndices; i++) if (rootIndices[i] > N1) N1 = rootIndices[i];
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &N1, 1, MPIU_INT, MPI_MAX, comm));
  PetscCheckFalse(N1 >= N,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Max. root index (%" PetscInt_FMT ") out of layout range [0,%" PetscInt_FMT ")", N1, N);
  if (!flag) {
    N1 = PETSC_MIN_INT;
    for (i = 0; i < numLeafIndices; i++) if (leafIndices[i] > N1) N1 = leafIndices[i];
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &N1, 1, MPIU_INT, MPI_MAX, comm));
    PetscCheckFalse(N1 >= N,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Max. leaf index (%" PetscInt_FMT ") out of layout range [0,%" PetscInt_FMT ")", N1, N);
  }
#endif
  /* Reduce: owners -> buffer */
  PetscCall(PetscMalloc1(n, &buffer));
  PetscCall(PetscSFCreate(comm, &sf1));
  PetscCall(PetscSFSetFromOptions(sf1));
  PetscCall(PetscSFSetGraphLayout(sf1, layout, numRootIndices, NULL, PETSC_OWN_POINTER, rootIndices));
  PetscCall(PetscMalloc1(numRootIndices, &owners));
  for (i = 0; i < numRootIndices; ++i) {
    owners[i].rank = rank;
    owners[i].index = rootLocalOffset + (rootLocalIndices ? rootLocalIndices[i] : i);
  }
  for (i = 0; i < n; ++i) {
    buffer[i].index = -1;
    buffer[i].rank = -1;
  }
  PetscCall(PetscSFReduceBegin(sf1, MPIU_2INT, owners, buffer, MPI_MAXLOC));
  PetscCall(PetscSFReduceEnd(sf1, MPIU_2INT, owners, buffer, MPI_MAXLOC));
  /* Bcast: buffer -> owners */
  if (!flag) {
    /* leafIndices is different from rootIndices */
    PetscCall(PetscFree(owners));
    PetscCall(PetscSFSetGraphLayout(sf1, layout, numLeafIndices, NULL, PETSC_OWN_POINTER, leafIndices));
    PetscCall(PetscMalloc1(numLeafIndices, &owners));
  }
  PetscCall(PetscSFBcastBegin(sf1, MPIU_2INT, buffer, owners, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf1, MPIU_2INT, buffer, owners, MPI_REPLACE));
  for (i = 0; i < numLeafIndices; ++i) PetscCheckFalse(owners[i].rank < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Global point %" PetscInt_FMT " was unclaimed", leafIndices[i]);
  PetscCall(PetscFree(buffer));
  if (sfA) {*sfA = sf1;}
  else     PetscCall(PetscSFDestroy(&sf1));
  /* Create sf */
  if (flag && rootLocalIndices == leafLocalIndices && leafLocalOffset == rootLocalOffset) {
    /* leaf space == root space */
    for (i = 0, nleaves = 0; i < numLeafIndices; ++i) if (owners[i].rank != rank) ++nleaves;
    PetscCall(PetscMalloc1(nleaves, &ilocal));
    PetscCall(PetscMalloc1(nleaves, &iremote));
    for (i = 0, nleaves = 0; i < numLeafIndices; ++i) {
      if (owners[i].rank != rank) {
        ilocal[nleaves]        = leafLocalOffset + i;
        iremote[nleaves].rank  = owners[i].rank;
        iremote[nleaves].index = owners[i].index;
        ++nleaves;
      }
    }
    PetscCall(PetscFree(owners));
  } else {
    nleaves = numLeafIndices;
    PetscCall(PetscMalloc1(nleaves, &ilocal));
    for (i = 0; i < nleaves; ++i) {ilocal[i] = leafLocalOffset + (leafLocalIndices ? leafLocalIndices[i] : i);}
    iremote = owners;
  }
  PetscCall(PetscSFCreate(comm, sf));
  PetscCall(PetscSFSetFromOptions(*sf));
  PetscCall(PetscSFSetGraph(*sf, rootLocalOffset + numRootIndices, nleaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  PetscFunctionReturn(0);
}
