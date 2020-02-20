#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

/* Set the number of dof on each point and separate by fields */
static PetscErrorCode DMPlexCreateSectionFields(DM dm, const PetscInt numComp[], PetscSection *section)
{
  DMLabel        depthLabel;
  PetscInt       depth, Nf, f, pStart, pEnd;
  PetscBool     *isFE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm,&depthLabel);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  ierr = PetscCalloc1(Nf, &isFE);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    ierr = DMGetField(dm, f, NULL, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID)      {isFE[f] = PETSC_TRUE;}
    else if (id == PETSCFV_CLASSID) {isFE[f] = PETSC_FALSE;}
  }

  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), section);CHKERRQ(ierr);
  if (Nf > 0) {
    ierr = PetscSectionSetNumFields(*section, Nf);CHKERRQ(ierr);
    if (numComp) {
      for (f = 0; f < Nf; ++f) {
        ierr = PetscSectionSetFieldComponents(*section, f, numComp[f]);CHKERRQ(ierr);
        if (isFE[f]) {
          PetscFE           fe;
          PetscDualSpace    dspace;
          const PetscInt    ***perms;
          const PetscScalar ***flips;
          const PetscInt    *numDof;

          ierr = DMGetField(dm,f,NULL,(PetscObject *) &fe);CHKERRQ(ierr);
          ierr = PetscFEGetDualSpace(fe,&dspace);CHKERRQ(ierr);
          ierr = PetscDualSpaceGetSymmetries(dspace,&perms,&flips);CHKERRQ(ierr);
          ierr = PetscDualSpaceGetNumDof(dspace,&numDof);CHKERRQ(ierr);
          if (perms || flips) {
            DM              K;
            PetscInt        h;
            PetscSectionSym sym;

            ierr = PetscDualSpaceGetDM(dspace,&K);CHKERRQ(ierr);
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
  ierr = PetscFree(isFE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Set the number of dof on each point and separate by fields */
static PetscErrorCode DMPlexCreateSectionDof(DM dm, DMLabel label[],const PetscInt numDof[], PetscSection section)
{
  DMLabel        depthLabel;
  PetscInt      *pMax;
  PetscInt       depth, cellHeight, pStart = 0, pEnd = 0;
  PetscInt       Nf, f, dim, d, dep, p;
  PetscBool     *isFE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm,&depthLabel);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nf, &isFE);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    ierr = DMGetField(dm, f, NULL, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    /* User is allowed to put a "placeholder" field in (c.f. DMCreateDS) */
    isFE[f] = id == PETSCFE_CLASSID ? PETSC_TRUE : PETSC_FALSE;
  }

  ierr = PetscMalloc1(depth+1, &pMax);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, depth >= 0 ? &pMax[depth] : NULL, depth>1 ? &pMax[depth-1] : NULL, depth>2 ? &pMax[1] : NULL, &pMax[0]);CHKERRQ(ierr);
  ierr = DMPlexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    if (label && label[f]) {
      IS              pointIS;
      const PetscInt *points;
      PetscInt        n;

      ierr = DMLabelGetStratumIS(label[f], 1, &pointIS);CHKERRQ(ierr);
      if (!pointIS) continue;
      ierr = ISGetLocalSize(pointIS, &n);CHKERRQ(ierr);
      ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
      for (p = 0; p < n; ++p) {
        const PetscInt point = points[p];
        PetscInt       dof, d;

        ierr = DMLabelGetValue(depthLabel, point, &d);CHKERRQ(ierr);
        ierr = DMPlexGetDepthStratum(dm, d, NULL, &pEnd);CHKERRQ(ierr);
        /* If this is a hybrid point, use dof for one dimension lower */
        if ((point >= pMax[d]) && (point < pEnd)) --d;
        dof  = d < 0 ? 0 : numDof[f*(dim+1)+d];
        ierr = PetscSectionSetFieldDof(section, point, f, dof);CHKERRQ(ierr);
        ierr = PetscSectionAddDof(section, point, dof);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    } else {
      for (dep = 0; dep <= depth - cellHeight; ++dep) {
        /* Cases: dim > depth (cell-vertex mesh), dim == depth (fully interpolated), dim < depth (interpolated submesh) */
        d    = dim <= depth ? dep : (!dep ? 0 : dim);
        ierr = DMPlexGetDepthStratum(dm, dep, &pStart, &pEnd);CHKERRQ(ierr);
        pMax[dep] = pMax[dep] < 0 ? pEnd : pMax[dep];
        for (p = pStart; p < pEnd; ++p) {
          const PetscInt dof = numDof[f*(dim+1)+d];

          if (isFE[f] && p >= pMax[dep]) continue;
          ierr = PetscSectionSetFieldDof(section, p, f, dof);CHKERRQ(ierr);
          ierr = PetscSectionAddDof(section, p, dof);CHKERRQ(ierr);
        }
      }
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
  PetscInt       Nf;
  PetscInt       bc;
  PetscSection   aSec;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  for (bc = 0; bc < numBC; ++bc) {
    PetscInt        field = 0;
    const PetscInt *comp;
    const PetscInt *idx;
    PetscInt        Nc = 0, cNc = -1, n, i;

    if (Nf) {
      field = bcField[bc];
      ierr = PetscSectionGetFieldComponents(section, field, &Nc);CHKERRQ(ierr);
    }
    if (bcComps && bcComps[bc]) {ierr = ISGetLocalSize(bcComps[bc], &cNc);CHKERRQ(ierr);}
    if (bcComps && bcComps[bc]) {ierr = ISGetIndices(bcComps[bc], &comp);CHKERRQ(ierr);}
    ierr = ISGetLocalSize(bcPoints[bc], &n);CHKERRQ(ierr);
    ierr = ISGetIndices(bcPoints[bc], &idx);CHKERRQ(ierr);
    for (i = 0; i < n; ++i) {
      const PetscInt p = idx[i];
      PetscInt       numConst;

      if (Nf) {
        ierr = PetscSectionGetFieldDof(section, p, field, &numConst);CHKERRQ(ierr);
      } else {
        ierr = PetscSectionGetDof(section, p, &numConst);CHKERRQ(ierr);
      }
      /* If Nc <= 0, constrain every dof on the point */
      if (cNc > 0) {
        /* We assume that a point may have multiple "nodes", which are collections of Nc dofs,
           and that those dofs are numbered n*Nc+c */
        if (Nf) {
          if (numConst % Nc) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Point %D has %D dof which is not divisible by %D field components", p, numConst, Nc);
          numConst = (numConst/Nc) * cNc;
        } else {
          numConst = PetscMin(numConst, cNc);
        }
      }
      if (Nf) {ierr = PetscSectionAddFieldConstraintDof(section, p, field, numConst);CHKERRQ(ierr);}
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
        for (f = 0; f < Nf; f++) {
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
  PetscInt       Nf, cdof, maxDof = 0, pStart, pEnd, p, bc, f, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  if (!Nf) PetscFunctionReturn(0);
  /* Initialize all field indices to -1 */
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr); maxDof = PetscMax(maxDof, cdof);}
  ierr = PetscMalloc1(maxDof, &indices);CHKERRQ(ierr);
  for (d = 0; d < maxDof; ++d) indices[d] = -1;
  for (p = pStart; p < pEnd; ++p) for (f = 0; f < Nf; ++f) {ierr = PetscSectionSetFieldConstraintIndices(section, p, f, indices);CHKERRQ(ierr);}
  /* Handle BC constraints */
  for (bc = 0; bc < numBC; ++bc) {
    const PetscInt  field = bcField[bc];
    const PetscInt *comp, *idx;
    PetscInt        Nc, cNc = -1, n, i;

    ierr = PetscSectionGetFieldComponents(section, field, &Nc);CHKERRQ(ierr);
    if (bcComps && bcComps[bc]) {ierr = ISGetLocalSize(bcComps[bc], &cNc);CHKERRQ(ierr);}
    if (bcComps && bcComps[bc]) {ierr = ISGetIndices(bcComps[bc], &comp);CHKERRQ(ierr);}
    ierr = ISGetLocalSize(bcPoints[bc], &n);CHKERRQ(ierr);
    ierr = ISGetIndices(bcPoints[bc], &idx);CHKERRQ(ierr);
    for (i = 0; i < n; ++i) {
      const PetscInt  p = idx[i];
      const PetscInt *find;
      PetscInt        fdof, fcdof, c, j;

      ierr = PetscSectionGetFieldDof(section, p, field, &fdof);CHKERRQ(ierr);
      if (!fdof) continue;
      if (cNc < 0) {
        for (d = 0; d < fdof; ++d) indices[d] = d;
        fcdof = fdof;
      } else {
        /* We assume that a point may have multiple "nodes", which are collections of Nc dofs,
           and that those dofs are numbered n*Nc+c */
        ierr = PetscSectionGetFieldConstraintDof(section, p, field, &fcdof);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldConstraintIndices(section, p, field, &find);CHKERRQ(ierr);
        /* Get indices constrained by previous bcs */
        for (d = 0; d < fcdof; ++d) {if (find[d] < 0) break; indices[d] = find[d];}
        for (j = 0; j < fdof/Nc; ++j) for (c = 0; c < cNc; ++c) indices[d++] = j*Nc + comp[c];
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
        for (f = 0; f < Nf; f++) {
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
  PetscInt       Nf, maxDof, pStart, pEnd, p, f, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  ierr = PetscSectionGetMaxDof(section, &maxDof);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxDof, &indices);CHKERRQ(ierr);
  for (d = 0; d < maxDof; ++d) indices[d] = -1;
  for (p = pStart; p < pEnd; ++p) {
    PetscInt cdof, d;

    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    if (cdof) {
      if (Nf) {
        PetscInt numConst = 0, foff = 0;

        for (f = 0; f < Nf; ++f) {
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
. label     - The label indicating the mesh support of each field, or NULL for the whole mesh
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

  TODO: How is this related to DMCreateLocalSection()

.seealso: DMPlexCreate(), PetscSectionCreate(), PetscSectionSetPermutation()
@*/
PetscErrorCode DMPlexCreateSection(DM dm, DMLabel label[], const PetscInt numComp[],const PetscInt numDof[], PetscInt numBC, const PetscInt bcField[], const IS bcComps[], const IS bcPoints[], IS perm, PetscSection *section)
{
  PetscSection   aSec;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexCreateSectionFields(dm, numComp, section);CHKERRQ(ierr);
  ierr = DMPlexCreateSectionDof(dm, label, numDof, *section);CHKERRQ(ierr);
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

PetscErrorCode DMCreateLocalSection_Plex(DM dm)
{
  PetscDS        probBC;
  PetscSection   section;
  DMLabel       *labels;
  IS            *bcPoints, *bcComps;
  PetscBool     *isFE;
  PetscInt      *bcFields, *numComp, *numDof;
  PetscInt       depth, dim, numBd, numBC = 0, Nf, bd, bc = 0, f;
  PetscInt       cStart, cEnd, cEndInterior;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  /* FE and FV boundary conditions are handled slightly differently */
  ierr = PetscMalloc1(Nf, &isFE);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    ierr = DMGetField(dm, f, NULL, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID)      {isFE[f] = PETSC_TRUE;}
    else if (id == PETSCFV_CLASSID) {isFE[f] = PETSC_FALSE;}
    else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", f);
  }
  /* Allocate boundary point storage for FEM boundaries */
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetGhostCellStratum(dm, &cEndInterior, NULL);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &probBC);CHKERRQ(ierr);
  ierr = PetscDSGetNumBoundary(probBC, &numBd);CHKERRQ(ierr);
  if (!Nf && numBd) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "number of fields is zero and number of boundary conditions is nonzero (this should never happen)");
  for (bd = 0; bd < numBd; ++bd) {
    PetscInt                field;
    DMBoundaryConditionType type;
    const char             *labelName;
    DMLabel                 label;

    ierr = PetscDSGetBoundary(probBC, bd, &type, NULL, &labelName, &field, NULL, NULL, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = DMGetLabel(dm,labelName,&label);CHKERRQ(ierr);
    if (label && isFE[field] && (type & DM_BC_ESSENTIAL)) ++numBC;
  }
  /* Add ghost cell boundaries for FVM */
  for (f = 0; f < Nf; ++f) if (!isFE[f] && cEndInterior >= 0) ++numBC;
  ierr = PetscCalloc3(numBC,&bcFields,numBC,&bcPoints,numBC,&bcComps);CHKERRQ(ierr);
  /* Constrain ghost cells for FV */
  for (f = 0; f < Nf; ++f) {
    PetscInt *newidx, c;

    if (isFE[f] || cEndInterior < 0) continue;
    ierr = PetscMalloc1(cEnd-cEndInterior,&newidx);CHKERRQ(ierr);
    for (c = cEndInterior; c < cEnd; ++c) newidx[c-cEndInterior] = c;
    bcFields[bc] = f;
    ierr = ISCreateGeneral(PETSC_COMM_SELF, cEnd-cEndInterior, newidx, PETSC_OWN_POINTER, &bcPoints[bc++]);CHKERRQ(ierr);
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

    ierr = PetscDSGetBoundary(probBC, bd, &type, NULL, &bdLabel, &field, &numComps, &comps, NULL, &numValues, &values, NULL);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, bdLabel, &label);CHKERRQ(ierr);
    if (!isFE[field] || !label) continue;
    /* Only want to modify label once */
    for (bd2 = 0; bd2 < bd; ++bd2) {
      const char *bdname;
      ierr = PetscDSGetBoundary(probBC, bd2, NULL, NULL, &bdname, NULL, NULL, NULL, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
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
      if (numComps) {ierr = ISCreateGeneral(PETSC_COMM_SELF, numComps, comps, PETSC_COPY_VALUES, &bcComps[bc]);CHKERRQ(ierr);}
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
      ierr = ISCreateGeneral(PETSC_COMM_SELF, newn, newidx, PETSC_OWN_POINTER, &bcPoints[bc++]);CHKERRQ(ierr);
    }
  }
  /* Handle discretization */
  ierr = PetscCalloc3(Nf,&labels,Nf,&numComp,Nf*(dim+1),&numDof);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    labels[f] = dm->fields[f].label;
    if (isFE[f]) {
      PetscFE         fe = (PetscFE) dm->fields[f].disc;
      const PetscInt *numFieldDof;
      PetscInt        fedim, d;

      ierr = PetscFEGetNumComponents(fe, &numComp[f]);CHKERRQ(ierr);
      ierr = PetscFEGetNumDof(fe, &numFieldDof);CHKERRQ(ierr);
      ierr = PetscFEGetSpatialDimension(fe, &fedim);CHKERRQ(ierr);
      for (d = 0; d < PetscMin(dim, fedim)+1; ++d) numDof[f*(dim+1)+d] = numFieldDof[d];
    } else {
      PetscFV fv = (PetscFV) dm->fields[f].disc;

      ierr = PetscFVGetNumComponents(fv, &numComp[f]);CHKERRQ(ierr);
      numDof[f*(dim+1)+dim] = numComp[f];
    }
  }
  for (f = 0; f < Nf; ++f) {
    PetscInt d;
    for (d = 1; d < dim; ++d) {
      if ((numDof[f*(dim+1)+d] > 0) && (depth < dim)) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Mesh must be interpolated when unknowns are specified on edges or faces.");
    }
  }
  ierr = DMPlexCreateSection(dm, labels, numComp, numDof, numBC, bcFields, bcComps, bcPoints, NULL, &section);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscFE     fe;
    const char *name;

    if (isFE[f]) {
      ierr = DMGetField(dm, f, NULL, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject) fe, &name);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldName(section, f, name);CHKERRQ(ierr);
    }
  }
  ierr = DMSetLocalSection(dm, section);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  for (bc = 0; bc < numBC; ++bc) {ierr = ISDestroy(&bcPoints[bc]);CHKERRQ(ierr);ierr = ISDestroy(&bcComps[bc]);CHKERRQ(ierr);}
  ierr = PetscFree3(bcFields,bcPoints,bcComps);CHKERRQ(ierr);
  ierr = PetscFree3(labels,numComp,numDof);CHKERRQ(ierr);
  ierr = PetscFree(isFE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
