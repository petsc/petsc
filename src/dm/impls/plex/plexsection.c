#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

/* Set the number of dof on each point and separate by fields */
static PetscErrorCode DMPlexCreateSectionFields(DM dm, const PetscInt numComp[], PetscSection *section)
{
  DMLabel        depthLabel;
  PetscInt       depth, Nf, f, pStart, pEnd;
  PetscBool     *isFE;

  PetscFunctionBegin;
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMPlexGetDepthLabel(dm,&depthLabel));
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCall(PetscCalloc1(Nf, &isFE));
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    PetscCall(DMGetField(dm, f, NULL, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID)      {isFE[f] = PETSC_TRUE;}
    else if (id == PETSCFV_CLASSID) {isFE[f] = PETSC_FALSE;}
  }

  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), section));
  if (Nf > 0) {
    PetscCall(PetscSectionSetNumFields(*section, Nf));
    if (numComp) {
      for (f = 0; f < Nf; ++f) {
        PetscCall(PetscSectionSetFieldComponents(*section, f, numComp[f]));
        if (isFE[f]) {
          PetscFE           fe;
          PetscDualSpace    dspace;
          const PetscInt    ***perms;
          const PetscScalar ***flips;
          const PetscInt    *numDof;

          PetscCall(DMGetField(dm,f,NULL,(PetscObject *) &fe));
          PetscCall(PetscFEGetDualSpace(fe,&dspace));
          PetscCall(PetscDualSpaceGetSymmetries(dspace,&perms,&flips));
          PetscCall(PetscDualSpaceGetNumDof(dspace,&numDof));
          if (perms || flips) {
            DM              K;
            PetscInt        sph, spdepth;
            PetscSectionSym sym;

            PetscCall(PetscDualSpaceGetDM(dspace,&K));
            PetscCall(DMPlexGetDepth(K, &spdepth));
            PetscCall(PetscSectionSymCreateLabel(PetscObjectComm((PetscObject)*section),depthLabel,&sym));
            for (sph = 0; sph <= spdepth; sph++) {
              PetscDualSpace    hspace;
              PetscInt          kStart, kEnd;
              PetscInt          kConeSize, h = sph + (depth - spdepth);
              const PetscInt    **perms0 = NULL;
              const PetscScalar **flips0 = NULL;

              PetscCall(PetscDualSpaceGetHeightSubspace(dspace, sph, &hspace));
              PetscCall(DMPlexGetHeightStratum(K, h, &kStart, &kEnd));
              if (!hspace) continue;
              PetscCall(PetscDualSpaceGetSymmetries(hspace,&perms,&flips));
              if (perms) perms0 = perms[0];
              if (flips) flips0 = flips[0];
              if (!(perms0 || flips0)) continue;
              {
                DMPolytopeType ct;
                /* The number of arrangements is no longer based on the number of faces */
                PetscCall(DMPlexGetCellType(K, kStart, &ct));
                kConeSize = DMPolytopeTypeGetNumArrangments(ct) / 2;
              }
              PetscCall(PetscSectionSymLabelSetStratum(sym,depth - h,numDof[depth - h],-kConeSize,kConeSize,PETSC_USE_POINTER,perms0 ? &perms0[-kConeSize] : NULL,flips0 ? &flips0[-kConeSize] : NULL));
            }
            PetscCall(PetscSectionSetFieldSym(*section,f,sym));
            PetscCall(PetscSectionSymDestroy(&sym));
          }
        }
      }
    }
  }
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(PetscSectionSetChart(*section, pStart, pEnd));
  PetscCall(PetscFree(isFE));
  PetscFunctionReturn(0);
}

/* Set the number of dof on each point and separate by fields */
static PetscErrorCode DMPlexCreateSectionDof(DM dm, DMLabel label[],const PetscInt numDof[], PetscSection section)
{
  DMLabel        depthLabel;
  DMPolytopeType ct;
  PetscInt       depth, cellHeight, pStart = 0, pEnd = 0;
  PetscInt       Nf, f, Nds, n, dim, d, dep, p;
  PetscBool     *isFE, hasCohesive = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMPlexGetDepthLabel(dm,&depthLabel));
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCall(DMGetNumDS(dm, &Nds));
  for (n = 0; n < Nds; ++n) {
    PetscDS   ds;
    PetscBool isCohesive;

    PetscCall(DMGetRegionNumDS(dm, n, NULL, NULL, &ds));
    PetscCall(PetscDSIsCohesive(ds, &isCohesive));
    if (isCohesive) {hasCohesive = PETSC_TRUE; break;}
  }
  PetscCall(PetscMalloc1(Nf, &isFE));
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    PetscCall(DMGetField(dm, f, NULL, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    /* User is allowed to put a "placeholder" field in (c.f. DMCreateDS) */
    isFE[f] = id == PETSCFE_CLASSID ? PETSC_TRUE : PETSC_FALSE;
  }

  PetscCall(DMPlexGetVTKCellHeight(dm, &cellHeight));
  for (f = 0; f < Nf; ++f) {
    PetscBool avoidTensor;

    PetscCall(DMGetFieldAvoidTensor(dm, f, &avoidTensor));
    avoidTensor = (avoidTensor || hasCohesive) ? PETSC_TRUE : PETSC_FALSE;
    if (label && label[f]) {
      IS              pointIS;
      const PetscInt *points;
      PetscInt        n;

      PetscCall(DMLabelGetStratumIS(label[f], 1, &pointIS));
      if (!pointIS) continue;
      PetscCall(ISGetLocalSize(pointIS, &n));
      PetscCall(ISGetIndices(pointIS, &points));
      for (p = 0; p < n; ++p) {
        const PetscInt point = points[p];
        PetscInt       dof, d;

        PetscCall(DMPlexGetCellType(dm, point, &ct));
        PetscCall(DMLabelGetValue(depthLabel, point, &d));
        /* If this is a tensor prism point, use dof for one dimension lower */
        switch (ct) {
          case DM_POLYTOPE_POINT_PRISM_TENSOR:
          case DM_POLYTOPE_SEG_PRISM_TENSOR:
          case DM_POLYTOPE_TRI_PRISM_TENSOR:
          case DM_POLYTOPE_QUAD_PRISM_TENSOR:
            if (hasCohesive) {--d;} break;
          default: break;
        }
        dof  = d < 0 ? 0 : numDof[f*(dim+1)+d];
        PetscCall(PetscSectionSetFieldDof(section, point, f, dof));
        PetscCall(PetscSectionAddDof(section, point, dof));
      }
      PetscCall(ISRestoreIndices(pointIS, &points));
      PetscCall(ISDestroy(&pointIS));
    } else {
      for (dep = 0; dep <= depth - cellHeight; ++dep) {
        /* Cases: dim > depth (cell-vertex mesh), dim == depth (fully interpolated), dim < depth (interpolated submesh) */
        d    = dim <= depth ? dep : (!dep ? 0 : dim);
        PetscCall(DMPlexGetDepthStratum(dm, dep, &pStart, &pEnd));
        for (p = pStart; p < pEnd; ++p) {
          const PetscInt dof = numDof[f*(dim+1)+d];

          PetscCall(DMPlexGetCellType(dm, p, &ct));
          switch (ct) {
            case DM_POLYTOPE_POINT_PRISM_TENSOR:
            case DM_POLYTOPE_SEG_PRISM_TENSOR:
            case DM_POLYTOPE_TRI_PRISM_TENSOR:
            case DM_POLYTOPE_QUAD_PRISM_TENSOR:
              if (avoidTensor && isFE[f]) continue;
            default: break;
          }
          PetscCall(PetscSectionSetFieldDof(section, p, f, dof));
          PetscCall(PetscSectionAddDof(section, p, dof));
        }
      }
    }
  }
  PetscCall(PetscFree(isFE));
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

  PetscFunctionBegin;
  PetscCall(PetscSectionGetNumFields(section, &Nf));
  for (bc = 0; bc < numBC; ++bc) {
    PetscInt        field = 0;
    const PetscInt *comp;
    const PetscInt *idx;
    PetscInt        Nc = 0, cNc = -1, n, i;

    if (Nf) {
      field = bcField[bc];
      PetscCall(PetscSectionGetFieldComponents(section, field, &Nc));
    }
    if (bcComps && bcComps[bc]) PetscCall(ISGetLocalSize(bcComps[bc], &cNc));
    if (bcComps && bcComps[bc]) PetscCall(ISGetIndices(bcComps[bc], &comp));
    if (bcPoints[bc]) {
      PetscCall(ISGetLocalSize(bcPoints[bc], &n));
      PetscCall(ISGetIndices(bcPoints[bc], &idx));
      for (i = 0; i < n; ++i) {
        const PetscInt p = idx[i];
        PetscInt       numConst;

        if (Nf) {
          PetscCall(PetscSectionGetFieldDof(section, p, field, &numConst));
        } else {
          PetscCall(PetscSectionGetDof(section, p, &numConst));
        }
        /* If Nc <= 0, constrain every dof on the point */
        if (cNc > 0) {
          /* We assume that a point may have multiple "nodes", which are collections of Nc dofs,
             and that those dofs are numbered n*Nc+c */
          if (Nf) {
            PetscCheck(numConst % Nc == 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Point %" PetscInt_FMT " has %" PetscInt_FMT " dof which is not divisible by %" PetscInt_FMT " field components", p, numConst, Nc);
            numConst = (numConst/Nc) * cNc;
          } else {
            numConst = PetscMin(numConst, cNc);
          }
        }
        if (Nf) PetscCall(PetscSectionAddFieldConstraintDof(section, p, field, numConst));
        PetscCall(PetscSectionAddConstraintDof(section, p, numConst));
      }
      PetscCall(ISRestoreIndices(bcPoints[bc], &idx));
    }
    if (bcComps && bcComps[bc]) PetscCall(ISRestoreIndices(bcComps[bc], &comp));
  }
  PetscCall(DMPlexGetAnchors(dm, &aSec, NULL));
  if (aSec) {
    PetscInt aStart, aEnd, a;

    PetscCall(PetscSectionGetChart(aSec, &aStart, &aEnd));
    for (a = aStart; a < aEnd; a++) {
      PetscInt dof, f;

      PetscCall(PetscSectionGetDof(aSec, a, &dof));
      if (dof) {
        /* if there are point-to-point constraints, then all dofs are constrained */
        PetscCall(PetscSectionGetDof(section, a, &dof));
        PetscCall(PetscSectionSetConstraintDof(section, a, dof));
        for (f = 0; f < Nf; f++) {
          PetscCall(PetscSectionGetFieldDof(section, a, f, &dof));
          PetscCall(PetscSectionSetFieldConstraintDof(section, a, f, dof));
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

  PetscFunctionBegin;
  PetscCall(PetscSectionGetNumFields(section, &Nf));
  if (!Nf) PetscFunctionReturn(0);
  /* Initialize all field indices to -1 */
  PetscCall(PetscSectionGetChart(section, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {PetscCall(PetscSectionGetConstraintDof(section, p, &cdof)); maxDof = PetscMax(maxDof, cdof);}
  PetscCall(PetscMalloc1(maxDof, &indices));
  for (d = 0; d < maxDof; ++d) indices[d] = -1;
  for (p = pStart; p < pEnd; ++p) for (f = 0; f < Nf; ++f) PetscCall(PetscSectionSetFieldConstraintIndices(section, p, f, indices));
  /* Handle BC constraints */
  for (bc = 0; bc < numBC; ++bc) {
    const PetscInt  field = bcField[bc];
    const PetscInt *comp, *idx;
    PetscInt        Nc, cNc = -1, n, i;

    PetscCall(PetscSectionGetFieldComponents(section, field, &Nc));
    if (bcComps && bcComps[bc]) PetscCall(ISGetLocalSize(bcComps[bc], &cNc));
    if (bcComps && bcComps[bc]) PetscCall(ISGetIndices(bcComps[bc], &comp));
    if (bcPoints[bc]) {
      PetscCall(ISGetLocalSize(bcPoints[bc], &n));
      PetscCall(ISGetIndices(bcPoints[bc], &idx));
      for (i = 0; i < n; ++i) {
        const PetscInt  p = idx[i];
        const PetscInt *find;
        PetscInt        fdof, fcdof, c, j;

        PetscCall(PetscSectionGetFieldDof(section, p, field, &fdof));
        if (!fdof) continue;
        if (cNc < 0) {
          for (d = 0; d < fdof; ++d) indices[d] = d;
          fcdof = fdof;
        } else {
          /* We assume that a point may have multiple "nodes", which are collections of Nc dofs,
             and that those dofs are numbered n*Nc+c */
          PetscCall(PetscSectionGetFieldConstraintDof(section, p, field, &fcdof));
          PetscCall(PetscSectionGetFieldConstraintIndices(section, p, field, &find));
          /* Get indices constrained by previous bcs */
          for (d = 0; d < fcdof; ++d) {if (find[d] < 0) break; indices[d] = find[d];}
          for (j = 0; j < fdof/Nc; ++j) for (c = 0; c < cNc; ++c) indices[d++] = j*Nc + comp[c];
          PetscCall(PetscSortRemoveDupsInt(&d, indices));
          for (c = d; c < fcdof; ++c) indices[c] = -1;
          fcdof = d;
        }
        PetscCall(PetscSectionSetFieldConstraintDof(section, p, field, fcdof));
        PetscCall(PetscSectionSetFieldConstraintIndices(section, p, field, indices));
      }
      PetscCall(ISRestoreIndices(bcPoints[bc], &idx));
    }
    if (bcComps && bcComps[bc]) PetscCall(ISRestoreIndices(bcComps[bc], &comp));
  }
  /* Handle anchors */
  PetscCall(DMPlexGetAnchors(dm, &aSec, NULL));
  if (aSec) {
    PetscInt aStart, aEnd, a;

    for (d = 0; d < maxDof; ++d) indices[d] = d;
    PetscCall(PetscSectionGetChart(aSec, &aStart, &aEnd));
    for (a = aStart; a < aEnd; a++) {
      PetscInt dof, f;

      PetscCall(PetscSectionGetDof(aSec, a, &dof));
      if (dof) {
        /* if there are point-to-point constraints, then all dofs are constrained */
        for (f = 0; f < Nf; f++) {
          PetscCall(PetscSectionSetFieldConstraintIndices(section, a, f, indices));
        }
      }
    }
  }
  PetscCall(PetscFree(indices));
  PetscFunctionReturn(0);
}

/* Set the constrained indices on each point */
static PetscErrorCode DMPlexCreateSectionBCIndices(DM dm, PetscSection section)
{
  PetscInt      *indices;
  PetscInt       Nf, maxDof, pStart, pEnd, p, f, d;

  PetscFunctionBegin;
  PetscCall(PetscSectionGetNumFields(section, &Nf));
  PetscCall(PetscSectionGetMaxDof(section, &maxDof));
  PetscCall(PetscSectionGetChart(section, &pStart, &pEnd));
  PetscCall(PetscMalloc1(maxDof, &indices));
  for (d = 0; d < maxDof; ++d) indices[d] = -1;
  for (p = pStart; p < pEnd; ++p) {
    PetscInt cdof, d;

    PetscCall(PetscSectionGetConstraintDof(section, p, &cdof));
    if (cdof) {
      if (Nf) {
        PetscInt numConst = 0, foff = 0;

        for (f = 0; f < Nf; ++f) {
          const PetscInt *find;
          PetscInt        fcdof, fdof;

          PetscCall(PetscSectionGetFieldDof(section, p, f, &fdof));
          PetscCall(PetscSectionGetFieldConstraintDof(section, p, f, &fcdof));
          /* Change constraint numbering from field component to local dof number */
          PetscCall(PetscSectionGetFieldConstraintIndices(section, p, f, &find));
          for (d = 0; d < fcdof; ++d) indices[numConst+d] = find[d] + foff;
          numConst += fcdof;
          foff     += fdof;
        }
        if (cdof != numConst) PetscCall(PetscSectionSetConstraintDof(section, p, numConst));
      } else {
        for (d = 0; d < cdof; ++d) indices[d] = d;
      }
      PetscCall(PetscSectionSetConstraintIndices(section, p, indices));
    }
  }
  PetscCall(PetscFree(indices));
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

.seealso: `DMPlexCreate()`, `PetscSectionCreate()`, `PetscSectionSetPermutation()`
@*/
PetscErrorCode DMPlexCreateSection(DM dm, DMLabel label[], const PetscInt numComp[],const PetscInt numDof[], PetscInt numBC, const PetscInt bcField[], const IS bcComps[], const IS bcPoints[], IS perm, PetscSection *section)
{
  PetscSection   aSec;

  PetscFunctionBegin;
  PetscCall(DMPlexCreateSectionFields(dm, numComp, section));
  PetscCall(DMPlexCreateSectionDof(dm, label, numDof, *section));
  PetscCall(DMPlexCreateSectionBCDof(dm, numBC, bcField, bcComps, bcPoints, *section));
  if (perm) PetscCall(PetscSectionSetPermutation(*section, perm));
  PetscCall(PetscSectionSetFromOptions(*section));
  PetscCall(PetscSectionSetUp(*section));
  PetscCall(DMPlexGetAnchors(dm,&aSec,NULL));
  if (numBC || aSec) {
    PetscCall(DMPlexCreateSectionBCIndicesField(dm, numBC, bcField, bcComps, bcPoints, *section));
    PetscCall(DMPlexCreateSectionBCIndices(dm, *section));
  }
  PetscCall(PetscSectionViewFromOptions(*section,NULL,"-section_view"));
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateLocalSection_Plex(DM dm)
{
  PetscSection   section;
  DMLabel       *labels;
  IS            *bcPoints, *bcComps;
  PetscBool     *isFE;
  PetscInt      *bcFields, *numComp, *numDof;
  PetscInt       depth, dim, numBC = 0, Nf, Nds, s, bc = 0, f;
  PetscInt       cStart, cEnd, cEndInterior;

  PetscFunctionBegin;
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  /* FE and FV boundary conditions are handled slightly differently */
  PetscCall(PetscMalloc1(Nf, &isFE));
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    PetscCall(DMGetField(dm, f, NULL, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID)      {isFE[f] = PETSC_TRUE;}
    else if (id == PETSCFV_CLASSID) {isFE[f] = PETSC_FALSE;}
    else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, f);
  }
  /* Allocate boundary point storage for FEM boundaries */
  PetscCall(DMGetNumDS(dm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS  dsBC;
    PetscInt numBd, bd;

    PetscCall(DMGetRegionNumDS(dm, s, NULL, NULL, &dsBC));
    PetscCall(PetscDSGetNumBoundary(dsBC, &numBd));
    PetscCheck(Nf || !numBd,PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "number of fields is zero and number of boundary conditions is nonzero (this should never happen)");
    for (bd = 0; bd < numBd; ++bd) {
      PetscInt                field;
      DMBoundaryConditionType type;
      DMLabel                 label;

      PetscCall(PetscDSGetBoundary(dsBC, bd, NULL, &type, NULL, &label, NULL, NULL, &field, NULL, NULL, NULL, NULL, NULL));
      if (label && isFE[field] && (type & DM_BC_ESSENTIAL)) ++numBC;
    }
  }
  /* Add ghost cell boundaries for FVM */
  PetscCall(DMPlexGetGhostCellStratum(dm, &cEndInterior, NULL));
  for (f = 0; f < Nf; ++f) if (!isFE[f] && cEndInterior >= 0) ++numBC;
  PetscCall(PetscCalloc3(numBC, &bcFields, numBC, &bcPoints, numBC, &bcComps));
  /* Constrain ghost cells for FV */
  for (f = 0; f < Nf; ++f) {
    PetscInt *newidx, c;

    if (isFE[f] || cEndInterior < 0) continue;
    PetscCall(PetscMalloc1(cEnd-cEndInterior,&newidx));
    for (c = cEndInterior; c < cEnd; ++c) newidx[c-cEndInterior] = c;
    bcFields[bc] = f;
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, cEnd-cEndInterior, newidx, PETSC_OWN_POINTER, &bcPoints[bc++]));
  }
  /* Complete labels for boundary conditions */
  PetscCall(DMCompleteBCLabels_Internal(dm));
  /* Handle FEM Dirichlet boundaries */
  PetscCall(DMGetNumDS(dm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS  dsBC;
    PetscInt numBd, bd;

    PetscCall(DMGetRegionNumDS(dm, s, NULL, NULL, &dsBC));
    PetscCall(PetscDSGetNumBoundary(dsBC, &numBd));
    for (bd = 0; bd < numBd; ++bd) {
      DMLabel                 label;
      const PetscInt         *comps;
      const PetscInt         *values;
      PetscInt                bd2, field, numComps, numValues;
      DMBoundaryConditionType type;
      PetscBool               duplicate = PETSC_FALSE;

      PetscCall(PetscDSGetBoundary(dsBC, bd, NULL, &type, NULL, &label, &numValues, &values, &field, &numComps, &comps, NULL, NULL, NULL));
      if (!isFE[field] || !label) continue;
      /* Only want to modify label once */
      for (bd2 = 0; bd2 < bd; ++bd2) {
        DMLabel l;

        PetscCall(PetscDSGetBoundary(dsBC, bd2, NULL, NULL, NULL, &l, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
        duplicate = l == label ? PETSC_TRUE : PETSC_FALSE;
        if (duplicate) break;
      }
      /* Filter out cells, if you actually want to constrain cells you need to do things by hand right now */
      if (type & DM_BC_ESSENTIAL) {
        PetscInt       *newidx;
        PetscInt        n, newn = 0, p, v;

        bcFields[bc] = field;
        if (numComps) PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numComps, comps, PETSC_COPY_VALUES, &bcComps[bc]));
        for (v = 0; v < numValues; ++v) {
          IS              tmp;
          const PetscInt *idx;

          PetscCall(DMLabelGetStratumIS(label, values[v], &tmp));
          if (!tmp) continue;
          PetscCall(ISGetLocalSize(tmp, &n));
          PetscCall(ISGetIndices(tmp, &idx));
          if (isFE[field]) {
            for (p = 0; p < n; ++p) if ((idx[p] < cStart) || (idx[p] >= cEnd)) ++newn;
          } else {
            for (p = 0; p < n; ++p) if ((idx[p] >= cStart) || (idx[p] < cEnd)) ++newn;
          }
          PetscCall(ISRestoreIndices(tmp, &idx));
          PetscCall(ISDestroy(&tmp));
        }
        PetscCall(PetscMalloc1(newn, &newidx));
        newn = 0;
        for (v = 0; v < numValues; ++v) {
          IS              tmp;
          const PetscInt *idx;

          PetscCall(DMLabelGetStratumIS(label, values[v], &tmp));
          if (!tmp) continue;
          PetscCall(ISGetLocalSize(tmp, &n));
          PetscCall(ISGetIndices(tmp, &idx));
          if (isFE[field]) {
            for (p = 0; p < n; ++p) if ((idx[p] < cStart) || (idx[p] >= cEnd)) newidx[newn++] = idx[p];
          } else {
            for (p = 0; p < n; ++p) if ((idx[p] >= cStart) || (idx[p] < cEnd)) newidx[newn++] = idx[p];
          }
          PetscCall(ISRestoreIndices(tmp, &idx));
          PetscCall(ISDestroy(&tmp));
        }
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF, newn, newidx, PETSC_OWN_POINTER, &bcPoints[bc++]));
      }
    }
  }
  /* Handle discretization */
  PetscCall(PetscCalloc3(Nf,&labels,Nf,&numComp,Nf*(dim+1),&numDof));
  for (f = 0; f < Nf; ++f) {
    labels[f] = dm->fields[f].label;
    if (isFE[f]) {
      PetscFE         fe = (PetscFE) dm->fields[f].disc;
      const PetscInt *numFieldDof;
      PetscInt        fedim, d;

      PetscCall(PetscFEGetNumComponents(fe, &numComp[f]));
      PetscCall(PetscFEGetNumDof(fe, &numFieldDof));
      PetscCall(PetscFEGetSpatialDimension(fe, &fedim));
      for (d = 0; d < PetscMin(dim, fedim)+1; ++d) numDof[f*(dim+1)+d] = numFieldDof[d];
    } else {
      PetscFV fv = (PetscFV) dm->fields[f].disc;

      PetscCall(PetscFVGetNumComponents(fv, &numComp[f]));
      numDof[f*(dim+1)+dim] = numComp[f];
    }
  }
  PetscCall(DMPlexGetDepth(dm, &depth));
  for (f = 0; f < Nf; ++f) {
    PetscInt d;
    for (d = 1; d < dim; ++d) {
      PetscCheck(numDof[f*(dim+1)+d] <= 0 || depth >= dim,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Mesh must be interpolated when unknowns are specified on edges or faces.");
    }
  }
  PetscCall(DMPlexCreateSection(dm, labels, numComp, numDof, numBC, bcFields, bcComps, bcPoints, NULL, &section));
  for (f = 0; f < Nf; ++f) {
    PetscFE     fe;
    const char *name;

    PetscCall(DMGetField(dm, f, NULL, (PetscObject *) &fe));
    if (!((PetscObject) fe)->name) continue;
    PetscCall(PetscObjectGetName((PetscObject) fe, &name));
    PetscCall(PetscSectionSetFieldName(section, f, name));
  }
  PetscCall(DMSetLocalSection(dm, section));
  PetscCall(PetscSectionDestroy(&section));
  for (bc = 0; bc < numBC; ++bc) {PetscCall(ISDestroy(&bcPoints[bc]));PetscCall(ISDestroy(&bcComps[bc]));}
  PetscCall(PetscFree3(bcFields,bcPoints,bcComps));
  PetscCall(PetscFree3(labels,numComp,numDof));
  PetscCall(PetscFree(isFE));
  PetscFunctionReturn(0);
}
