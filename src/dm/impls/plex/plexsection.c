#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

/* Set the number of dof on each point and separate by fields */
static PetscErrorCode DMPlexCreateSectionInitial(DM dm, PetscInt dim, PetscInt numFields,const PetscInt numComp[],const PetscInt numDof[], PetscSection *section)
{
  PetscInt      *pMax;
  PetscInt       depth, cellHeight, pStart = 0, pEnd = 0;
  PetscInt       Nf, p, d, dep, f;
  PetscBool     *isFE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(numFields, &isFE);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  for (f = 0; f < numFields; ++f) {
    PetscObject  obj;
    PetscClassId id;

    isFE[f] = PETSC_FALSE;
    if (f >= Nf) continue;
    ierr = DMGetField(dm, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID)      {isFE[f] = PETSC_TRUE;}
    else if (id == PETSCFV_CLASSID) {isFE[f] = PETSC_FALSE;}
  }
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), section);CHKERRQ(ierr);
  if (numFields > 0) {
    ierr = PetscSectionSetNumFields(*section, numFields);CHKERRQ(ierr);
    if (numComp) {
      for (f = 0; f < numFields; ++f) {
        ierr = PetscSectionSetFieldComponents(*section, f, numComp[f]);CHKERRQ(ierr);
        if (isFE[f]) {
          PetscFE           fe;
          PetscDualSpace    dspace;
          const PetscInt    ***perms;
          const PetscScalar ***flips;
          const PetscInt    *numDof;

          ierr = DMGetField(dm,f,(PetscObject *) &fe);CHKERRQ(ierr);
          ierr = PetscFEGetDualSpace(fe,&dspace);CHKERRQ(ierr);
          ierr = PetscDualSpaceGetSymmetries(dspace,&perms,&flips);CHKERRQ(ierr);
          ierr = PetscDualSpaceGetNumDof(dspace,&numDof);CHKERRQ(ierr);
          if (perms || flips) {
            DM               K;
            DMLabel          depthLabel;
            PetscInt         depth, h;
            PetscSectionSym  sym;

            ierr = PetscDualSpaceGetDM(dspace,&K);CHKERRQ(ierr);
            ierr = DMPlexGetDepthLabel(dm,&depthLabel);CHKERRQ(ierr);
            ierr = DMPlexGetDepth(dm,&depth);CHKERRQ(ierr);
            ierr = PetscSectionSymCreateLabel(PetscObjectComm((PetscObject)*section),depthLabel,&sym);CHKERRQ(ierr);
            for (h = 0; h <= depth; h++) {
              PetscDualSpace    hspace;
              PetscInt          kStart, kEnd;
              PetscInt          kConeSize;
              const PetscInt    **perms0 = NULL;
              const PetscScalar **flips0 = NULL;

              ierr = PetscDualSpaceGetHeightSubspace(dspace,h,&hspace);CHKERRQ(ierr);
              ierr = DMPlexGetHeightStratum(K,h,&kStart,&kEnd);CHKERRQ(ierr);
              if (!hspace) continue;
              ierr = PetscDualSpaceGetSymmetries(hspace,&perms,&flips);CHKERRQ(ierr);
              if (perms) perms0 = perms[0];
              if (flips) flips0 = flips[0];
              if (!(perms0 || flips0)) continue;
              ierr = DMPlexGetConeSize(K,kStart,&kConeSize);CHKERRQ(ierr);
              ierr = PetscSectionSymLabelSetStratum(sym,depth - h,numDof[depth - h],-kConeSize,kConeSize,PETSC_USE_POINTER,perms0 ? &perms0[-kConeSize] : NULL,flips0 ? &flips0[-kConeSize] : NULL);CHKERRQ(ierr);
            }
            ierr = PetscSectionSetFieldSym(*section,f,sym);CHKERRQ(ierr);
            ierr = PetscSectionSymDestroy(&sym);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*section, pStart, pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = PetscMalloc1(depth+1,&pMax);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, depth >= 0 ? &pMax[depth] : NULL, depth>1 ? &pMax[depth-1] : NULL, depth>2 ? &pMax[1] : NULL, &pMax[0]);CHKERRQ(ierr);
  ierr = DMPlexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
  for (dep = 0; dep <= depth - cellHeight; ++dep) {
    d    = dim == depth ? dep : (!dep ? 0 : dim);
    ierr = DMPlexGetDepthStratum(dm, dep, &pStart, &pEnd);CHKERRQ(ierr);
    pMax[dep] = pMax[dep] < 0 ? pEnd : pMax[dep];
    for (p = pStart; p < pEnd; ++p) {
      PetscInt tot = 0;

      for (f = 0; f < numFields; ++f) {
        if (isFE[f] && p >= pMax[dep]) continue;
        ierr = PetscSectionSetFieldDof(*section, p, f, numDof[f*(dim+1)+d]);CHKERRQ(ierr);
        tot += numDof[f*(dim+1)+d];
      }
      ierr = PetscSectionSetDof(*section, p, tot);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(pMax);CHKERRQ(ierr);
  ierr = PetscFree(isFE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Set the number of dof on each point and separate by fields
   If bcComps is NULL or the IS is NULL, constrain every dof on the point
*/
static PetscErrorCode DMPlexCreateSectionBCDof(DM dm, PetscInt numBC, const PetscInt bcField[], const IS bcComps[], const IS bcPoints[], PetscSection section)
{
  PetscInt       numFields;
  PetscInt       bc;
  PetscSection   aSec;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  for (bc = 0; bc < numBC; ++bc) {
    PetscInt        field = 0;
    const PetscInt *comp;
    const PetscInt *idx;
    PetscInt        Nc = -1, n, i;

    if (numFields) field = bcField[bc];
    if (bcComps && bcComps[bc]) {ierr = ISGetLocalSize(bcComps[bc], &Nc);CHKERRQ(ierr);}
    if (bcComps && bcComps[bc]) {ierr = ISGetIndices(bcComps[bc], &comp);CHKERRQ(ierr);}
    ierr = ISGetLocalSize(bcPoints[bc], &n);CHKERRQ(ierr);
    ierr = ISGetIndices(bcPoints[bc], &idx);CHKERRQ(ierr);
    for (i = 0; i < n; ++i) {
      const PetscInt p = idx[i];
      PetscInt       numConst;

      if (numFields) {
        ierr = PetscSectionGetFieldDof(section, p, field, &numConst);CHKERRQ(ierr);
      } else {
        ierr = PetscSectionGetDof(section, p, &numConst);CHKERRQ(ierr);
      }
      /* If Nc < 0, constrain every dof on the point */
      /* TODO: Matt, this only works if there is one node on the point.  We need to handle numDofs > NumComponents */
      if (Nc > 0) numConst = PetscMin(numConst, Nc);
      if (numFields) {ierr = PetscSectionAddFieldConstraintDof(section, p, field, numConst);CHKERRQ(ierr);}
      ierr = PetscSectionAddConstraintDof(section, p, numConst);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(bcPoints[bc], &idx);CHKERRQ(ierr);
    if (bcComps && bcComps[bc]) {ierr = ISRestoreIndices(bcComps[bc], &comp);CHKERRQ(ierr);}
  }
  ierr = DMPlexGetAnchors(dm, &aSec, NULL);CHKERRQ(ierr);
  if (aSec) {
    PetscInt aStart, aEnd, a;

    ierr = PetscSectionGetChart(aSec, &aStart, &aEnd);CHKERRQ(ierr);
    for (a = aStart; a < aEnd; a++) {
      PetscInt dof, f;

      ierr = PetscSectionGetDof(aSec, a, &dof);CHKERRQ(ierr);
      if (dof) {
        /* if there are point-to-point constraints, then all dofs are constrained */
        ierr = PetscSectionGetDof(section, a, &dof);CHKERRQ(ierr);
        ierr = PetscSectionSetConstraintDof(section, a, dof);CHKERRQ(ierr);
        for (f = 0; f < numFields; f++) {
          ierr = PetscSectionGetFieldDof(section, a, f, &dof);CHKERRQ(ierr);
          ierr = PetscSectionSetFieldConstraintDof(section, a, f, dof);CHKERRQ(ierr);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/* Set the constrained field indices on each point
   If bcComps is NULL or the IS is NULL, constrain every dof on the point
*/
static PetscErrorCode DMPlexCreateSectionBCIndicesField(DM dm, PetscInt numBC,const PetscInt bcField[], const IS bcComps[], const IS bcPoints[], PetscSection section)
{
  PetscSection   aSec;
  PetscInt      *indices;
  PetscInt       numFields, cdof, maxDof = 0, pStart, pEnd, p, bc, f, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  if (!numFields) PetscFunctionReturn(0);
  /* Initialize all field indices to -1 */
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr); maxDof = PetscMax(maxDof, cdof);}
  ierr = PetscMalloc1(maxDof, &indices);CHKERRQ(ierr);
  for (d = 0; d < maxDof; ++d) indices[d] = -1;
  for (p = pStart; p < pEnd; ++p) for (f = 0; f < numFields; ++f) {ierr = PetscSectionSetFieldConstraintIndices(section, p, f, indices);CHKERRQ(ierr);}
  /* Handle BC constraints */
  for (bc = 0; bc < numBC; ++bc) {
    const PetscInt  field = bcField[bc];
    const PetscInt *comp, *idx;
    PetscInt        Nc = -1, n, i;

    if (bcComps && bcComps[bc]) {ierr = ISGetLocalSize(bcComps[bc], &Nc);CHKERRQ(ierr);}
    if (bcComps && bcComps[bc]) {ierr = ISGetIndices(bcComps[bc], &comp);CHKERRQ(ierr);}
    ierr = ISGetLocalSize(bcPoints[bc], &n);CHKERRQ(ierr);
    ierr = ISGetIndices(bcPoints[bc], &idx);CHKERRQ(ierr);
    for (i = 0; i < n; ++i) {
      const PetscInt  p = idx[i];
      const PetscInt *find;
      PetscInt        fdof, fcdof, c;

      ierr = PetscSectionGetFieldDof(section, p, field, &fdof);CHKERRQ(ierr);
      if (!fdof) continue;
      if (Nc < 0) {
        for (d = 0; d < fdof; ++d) indices[d] = d;
        fcdof = fdof;
      } else {
        ierr = PetscSectionGetFieldConstraintDof(section, p, field, &fcdof);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldConstraintIndices(section, p, field, &find);CHKERRQ(ierr);
        for (d = 0; d < fcdof; ++d) {if (find[d] < 0) break; indices[d] = find[d];}
        for (c = 0; c < Nc; ++c) indices[d++] = comp[c];
        ierr = PetscSortRemoveDupsInt(&d, indices);CHKERRQ(ierr);
        for (c = d; c < fcdof; ++c) indices[c] = -1;
        fcdof = d;
      }
      ierr = PetscSectionSetFieldConstraintDof(section, p, field, fcdof);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldConstraintIndices(section, p, field, indices);CHKERRQ(ierr);
    }
    if (bcComps && bcComps[bc]) {ierr = ISRestoreIndices(bcComps[bc], &comp);CHKERRQ(ierr);}
    ierr = ISRestoreIndices(bcPoints[bc], &idx);CHKERRQ(ierr);
  }
  /* Handle anchors */
  ierr = DMPlexGetAnchors(dm, &aSec, NULL);CHKERRQ(ierr);
  if (aSec) {
    PetscInt aStart, aEnd, a;

    for (d = 0; d < maxDof; ++d) indices[d] = d;
    ierr = PetscSectionGetChart(aSec, &aStart, &aEnd);CHKERRQ(ierr);
    for (a = aStart; a < aEnd; a++) {
      PetscInt dof, f;

      ierr = PetscSectionGetDof(aSec, a, &dof);CHKERRQ(ierr);
      if (dof) {
        /* if there are point-to-point constraints, then all dofs are constrained */
        for (f = 0; f < numFields; f++) {
          ierr = PetscSectionSetFieldConstraintIndices(section, a, f, indices);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = PetscFree(indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Set the constrained indices on each point */
static PetscErrorCode DMPlexCreateSectionBCIndices(DM dm, PetscSection section)
{
  PetscInt      *indices;
  PetscInt       numFields, maxDof, pStart, pEnd, p, f, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = PetscSectionGetMaxDof(section, &maxDof);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxDof, &indices);CHKERRQ(ierr);
  for (d = 0; d < maxDof; ++d) indices[d] = -1;
  for (p = pStart; p < pEnd; ++p) {
    PetscInt cdof, d;

    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    if (cdof) {
      if (numFields) {
        PetscInt numConst = 0, foff = 0;

        for (f = 0; f < numFields; ++f) {
          const PetscInt *find;
          PetscInt        fcdof, fdof;

          ierr = PetscSectionGetFieldDof(section, p, f, &fdof);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldConstraintDof(section, p, f, &fcdof);CHKERRQ(ierr);
          /* Change constraint numbering from field component to local dof number */
          ierr = PetscSectionGetFieldConstraintIndices(section, p, f, &find);CHKERRQ(ierr);
          for (d = 0; d < fcdof; ++d) indices[numConst+d] = find[d] + foff;
          numConst += fcdof;
          foff     += fdof;
        }
        if (cdof != numConst) {ierr = PetscSectionSetConstraintDof(section, p, numConst);CHKERRQ(ierr);}
      } else {
        for (d = 0; d < cdof; ++d) indices[d] = d;
      }
      ierr = PetscSectionSetConstraintIndices(section, p, indices);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreateSection - Create a PetscSection based upon the dof layout specification provided.

  Not Collective

  Input Parameters:
+ dm        - The DMPlex object
. dim       - The spatial dimension of the problem
. numFields - The number of fields in the problem
. numComp   - An array of size numFields that holds the number of components for each field
. numDof    - An array of size numFields*(dim+1) which holds the number of dof for each field on a mesh piece of dimension d
. numBC     - The number of boundary conditions
. bcField   - An array of size numBC giving the field number for each boundry condition
. bcComps   - [Optional] An array of size numBC giving an IS holding the field components to which each boundary condition applies
. bcPoints  - An array of size numBC giving an IS holding the Plex points to which each boundary condition applies
- perm      - Optional permutation of the chart, or NULL

  Output Parameter:
. section - The PetscSection object

  Notes:
    numDof[f*(dim+1)+d] gives the number of dof for field f on points of dimension d. For instance, numDof[1] is the
  number of dof for field 0 on each edge.

  The chart permutation is the same one set using PetscSectionSetPermutation()

  Level: developer

  Fortran Notes:
  A Fortran 90 version is available as DMPlexCreateSectionF90()

.keywords: mesh, elements
.seealso: DMPlexCreate(), PetscSectionCreate(), PetscSectionSetPermutation()
@*/
PetscErrorCode DMPlexCreateSection(DM dm, PetscInt dim, PetscInt numFields,const PetscInt numComp[],const PetscInt numDof[], PetscInt numBC,const PetscInt bcField[], const IS bcComps[], const IS bcPoints[], IS perm, PetscSection *section)
{
  PetscSection   aSec;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexCreateSectionInitial(dm, dim, numFields, numComp, numDof, section);CHKERRQ(ierr);
  ierr = DMPlexCreateSectionBCDof(dm, numBC, bcField, bcComps, bcPoints, *section);CHKERRQ(ierr);
  if (perm) {ierr = PetscSectionSetPermutation(*section, perm);CHKERRQ(ierr);}
  ierr = PetscSectionSetFromOptions(*section);CHKERRQ(ierr);
  ierr = PetscSectionSetUp(*section);CHKERRQ(ierr);
  ierr = DMPlexGetAnchors(dm,&aSec,NULL);CHKERRQ(ierr);
  if (numBC || aSec) {
    ierr = DMPlexCreateSectionBCIndicesField(dm, numBC, bcField, bcComps, bcPoints, *section);CHKERRQ(ierr);
    ierr = DMPlexCreateSectionBCIndices(dm, *section);CHKERRQ(ierr);
  }
  ierr = PetscSectionViewFromOptions(*section,NULL,"-section_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateDefaultSection_Plex(DM dm)
{
  PetscSection   section;
  IS            *bcPoints, *bcComps;
  PetscBool     *isFE;
  PetscInt      *bcFields, *numComp, *numDof;
  PetscInt       depth, dim, numBd, numBC = 0, numFields, bd, bc = 0, f;
  PetscInt       cStart, cEnd, cEndInterior;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetNumFields(dm, &numFields);CHKERRQ(ierr);
  /* FE and FV boundary conditions are handled slightly differently */
  ierr = PetscMalloc1(numFields, &isFE);CHKERRQ(ierr);
  for (f = 0; f < numFields; ++f) {
    PetscObject  obj;
    PetscClassId id;

    ierr = DMGetField(dm, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID)      {isFE[f] = PETSC_TRUE;}
    else if (id == PETSCFV_CLASSID) {isFE[f] = PETSC_FALSE;}
    else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", f);
  }
  /* Allocate boundary point storage for FEM boundaries */
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetNumBoundary(dm->prob, &numBd);CHKERRQ(ierr);
  if (!numFields && numBd) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "number of fields is zero and number of boundary conditions is nonzero (this should never happen)");
  for (bd = 0; bd < numBd; ++bd) {
    PetscInt                field;
    DMBoundaryConditionType type;
    const char             *labelName;
    DMLabel                 label;

    ierr = PetscDSGetBoundary(dm->prob, bd, &type, NULL, &labelName, &field, NULL, NULL, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = DMGetLabel(dm,labelName,&label);CHKERRQ(ierr);
    if (label && isFE[field] && (type & DM_BC_ESSENTIAL)) ++numBC;
  }
  /* Add ghost cell boundaries for FVM */
  for (f = 0; f < numFields; ++f) if (!isFE[f] && cEndInterior >= 0) ++numBC;
  ierr = PetscCalloc3(numBC,&bcFields,numBC,&bcPoints,numBC,&bcComps);CHKERRQ(ierr);
  /* Constrain ghost cells for FV */
  for (f = 0; f < numFields; ++f) {
    PetscInt *newidx, c;

    if (isFE[f] || cEndInterior < 0) continue;
    ierr = PetscMalloc1(cEnd-cEndInterior,&newidx);CHKERRQ(ierr);
    for (c = cEndInterior; c < cEnd; ++c) newidx[c-cEndInterior] = c;
    bcFields[bc] = f;
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), cEnd-cEndInterior, newidx, PETSC_OWN_POINTER, &bcPoints[bc++]);CHKERRQ(ierr);
  }
  /* Handle FEM Dirichlet boundaries */
  for (bd = 0; bd < numBd; ++bd) {
    const char             *bdLabel;
    DMLabel                 label;
    const PetscInt         *comps;
    const PetscInt         *values;
    PetscInt                bd2, field, numComps, numValues;
    DMBoundaryConditionType type;
    PetscBool               duplicate = PETSC_FALSE;

    ierr = PetscDSGetBoundary(dm->prob, bd, &type, NULL, &bdLabel, &field, &numComps, &comps, NULL, &numValues, &values, NULL);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, bdLabel, &label);CHKERRQ(ierr);
    if (!isFE[field] || !label) continue;
    /* Only want to modify label once */
    for (bd2 = 0; bd2 < bd; ++bd2) {
      const char *bdname;
      ierr = PetscDSGetBoundary(dm->prob, bd2, NULL, NULL, &bdname, NULL, NULL, NULL, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscStrcmp(bdname, bdLabel, &duplicate);CHKERRQ(ierr);
      if (duplicate) break;
    }
    if (!duplicate && (isFE[field])) {
      /* don't complete cells, which are just present to give orientation to the boundary */
      ierr = DMPlexLabelComplete(dm, label);CHKERRQ(ierr);
    }
    /* Filter out cells, if you actually want to constrain cells you need to do things by hand right now */
    if (type & DM_BC_ESSENTIAL) {
      PetscInt       *newidx;
      PetscInt        n, newn = 0, p, v;

      bcFields[bc] = field;
      if (numComps) {ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), numComps, comps, PETSC_COPY_VALUES, &bcComps[bc]);CHKERRQ(ierr);}
      for (v = 0; v < numValues; ++v) {
        IS              tmp;
        const PetscInt *idx;

        ierr = DMGetStratumIS(dm, bdLabel, values[v], &tmp);CHKERRQ(ierr);
        if (!tmp) continue;
        ierr = ISGetLocalSize(tmp, &n);CHKERRQ(ierr);
        ierr = ISGetIndices(tmp, &idx);CHKERRQ(ierr);
        if (isFE[field]) {
          for (p = 0; p < n; ++p) if ((idx[p] < cStart) || (idx[p] >= cEnd)) ++newn;
        } else {
          for (p = 0; p < n; ++p) if ((idx[p] >= cStart) || (idx[p] < cEnd)) ++newn;
        }
        ierr = ISRestoreIndices(tmp, &idx);CHKERRQ(ierr);
        ierr = ISDestroy(&tmp);CHKERRQ(ierr);
      }
      ierr = PetscMalloc1(newn,&newidx);CHKERRQ(ierr);
      newn = 0;
      for (v = 0; v < numValues; ++v) {
        IS              tmp;
        const PetscInt *idx;

        ierr = DMGetStratumIS(dm, bdLabel, values[v], &tmp);CHKERRQ(ierr);
        if (!tmp) continue;
        ierr = ISGetLocalSize(tmp, &n);CHKERRQ(ierr);
        ierr = ISGetIndices(tmp, &idx);CHKERRQ(ierr);
        if (isFE[field]) {
          for (p = 0; p < n; ++p) if ((idx[p] < cStart) || (idx[p] >= cEnd)) newidx[newn++] = idx[p];
        } else {
          for (p = 0; p < n; ++p) if ((idx[p] >= cStart) || (idx[p] < cEnd)) newidx[newn++] = idx[p];
        }
        ierr = ISRestoreIndices(tmp, &idx);CHKERRQ(ierr);
        ierr = ISDestroy(&tmp);CHKERRQ(ierr);
      }
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), newn, newidx, PETSC_OWN_POINTER, &bcPoints[bc++]);CHKERRQ(ierr);
    }
  }
  /* Handle discretization */
  ierr = PetscCalloc2(numFields,&numComp,numFields*(dim+1),&numDof);CHKERRQ(ierr);
  for (f = 0; f < numFields; ++f) {
    PetscObject obj;

    ierr = DMGetField(dm, f, &obj);CHKERRQ(ierr);
    if (isFE[f]) {
      PetscFE         fe = (PetscFE) obj;
      const PetscInt *numFieldDof;
      PetscInt        d;

      ierr = PetscFEGetNumComponents(fe, &numComp[f]);CHKERRQ(ierr);
      ierr = PetscFEGetNumDof(fe, &numFieldDof);CHKERRQ(ierr);
      for (d = 0; d < dim+1; ++d) numDof[f*(dim+1)+d] = numFieldDof[d];
    } else {
      PetscFV fv = (PetscFV) obj;

      ierr = PetscFVGetNumComponents(fv, &numComp[f]);CHKERRQ(ierr);
      numDof[f*(dim+1)+dim] = numComp[f];
    }
  }
  for (f = 0; f < numFields; ++f) {
    PetscInt d;
    for (d = 1; d < dim; ++d) {
      if ((numDof[f*(dim+1)+d] > 0) && (depth < dim)) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Mesh must be interpolated when unknowns are specified on edges or faces.");
    }
  }
  ierr = DMPlexCreateSection(dm, dim, numFields, numComp, numDof, numBC, bcFields, bcComps, bcPoints, NULL, &section);CHKERRQ(ierr);
  for (f = 0; f < numFields; ++f) {
    PetscFE     fe;
    const char *name;

    ierr = DMGetField(dm, f, (PetscObject *) &fe);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) fe, &name);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(section, f, name);CHKERRQ(ierr);
  }
  ierr = DMSetSection(dm, section);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  for (bc = 0; bc < numBC; ++bc) {ierr = ISDestroy(&bcPoints[bc]);CHKERRQ(ierr);ierr = ISDestroy(&bcComps[bc]);CHKERRQ(ierr);}
  ierr = PetscFree3(bcFields,bcPoints,bcComps);CHKERRQ(ierr);
  ierr = PetscFree2(numComp,numDof);CHKERRQ(ierr);
  ierr = PetscFree(isFE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
