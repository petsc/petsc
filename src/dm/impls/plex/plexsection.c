#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

/* Set the number of dof on each point and separate by fields */
static PetscErrorCode DMPlexCreateSectionFields(DM dm, const PetscInt numComp[], PetscSection *section)
{
  DMLabel        depthLabel;
  PetscInt       depth, Nf, f, pStart, pEnd;
  PetscBool     *isFE;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(DMPlexGetDepthLabel(dm,&depthLabel));
  CHKERRQ(DMGetNumFields(dm, &Nf));
  CHKERRQ(PetscCalloc1(Nf, &isFE));
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    CHKERRQ(DMGetField(dm, f, NULL, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID)      {isFE[f] = PETSC_TRUE;}
    else if (id == PETSCFV_CLASSID) {isFE[f] = PETSC_FALSE;}
  }

  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)dm), section));
  if (Nf > 0) {
    CHKERRQ(PetscSectionSetNumFields(*section, Nf));
    if (numComp) {
      for (f = 0; f < Nf; ++f) {
        CHKERRQ(PetscSectionSetFieldComponents(*section, f, numComp[f]));
        if (isFE[f]) {
          PetscFE           fe;
          PetscDualSpace    dspace;
          const PetscInt    ***perms;
          const PetscScalar ***flips;
          const PetscInt    *numDof;

          CHKERRQ(DMGetField(dm,f,NULL,(PetscObject *) &fe));
          CHKERRQ(PetscFEGetDualSpace(fe,&dspace));
          CHKERRQ(PetscDualSpaceGetSymmetries(dspace,&perms,&flips));
          CHKERRQ(PetscDualSpaceGetNumDof(dspace,&numDof));
          if (perms || flips) {
            DM              K;
            PetscInt        sph, spdepth;
            PetscSectionSym sym;

            CHKERRQ(PetscDualSpaceGetDM(dspace,&K));
            CHKERRQ(DMPlexGetDepth(K, &spdepth));
            CHKERRQ(PetscSectionSymCreateLabel(PetscObjectComm((PetscObject)*section),depthLabel,&sym));
            for (sph = 0; sph <= spdepth; sph++) {
              PetscDualSpace    hspace;
              PetscInt          kStart, kEnd;
              PetscInt          kConeSize, h = sph + (depth - spdepth);
              const PetscInt    **perms0 = NULL;
              const PetscScalar **flips0 = NULL;

              CHKERRQ(PetscDualSpaceGetHeightSubspace(dspace, sph, &hspace));
              CHKERRQ(DMPlexGetHeightStratum(K, h, &kStart, &kEnd));
              if (!hspace) continue;
              CHKERRQ(PetscDualSpaceGetSymmetries(hspace,&perms,&flips));
              if (perms) perms0 = perms[0];
              if (flips) flips0 = flips[0];
              if (!(perms0 || flips0)) continue;
              {
                DMPolytopeType ct;
                /* The number of arrangements is no longer based on the number of faces */
                CHKERRQ(DMPlexGetCellType(K, kStart, &ct));
                kConeSize = DMPolytopeTypeGetNumArrangments(ct) / 2;
              }
              CHKERRQ(PetscSectionSymLabelSetStratum(sym,depth - h,numDof[depth - h],-kConeSize,kConeSize,PETSC_USE_POINTER,perms0 ? &perms0[-kConeSize] : NULL,flips0 ? &flips0[-kConeSize] : NULL));
            }
            CHKERRQ(PetscSectionSetFieldSym(*section,f,sym));
            CHKERRQ(PetscSectionSymDestroy(&sym));
          }
        }
      }
    }
  }
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  CHKERRQ(PetscSectionSetChart(*section, pStart, pEnd));
  CHKERRQ(PetscFree(isFE));
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
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(DMPlexGetDepthLabel(dm,&depthLabel));
  CHKERRQ(DMGetNumFields(dm, &Nf));
  CHKERRQ(DMGetNumDS(dm, &Nds));
  for (n = 0; n < Nds; ++n) {
    PetscDS   ds;
    PetscBool isCohesive;

    CHKERRQ(DMGetRegionNumDS(dm, n, NULL, NULL, &ds));
    CHKERRQ(PetscDSIsCohesive(ds, &isCohesive));
    if (isCohesive) {hasCohesive = PETSC_TRUE; break;}
  }
  CHKERRQ(PetscMalloc1(Nf, &isFE));
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    CHKERRQ(DMGetField(dm, f, NULL, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    /* User is allowed to put a "placeholder" field in (c.f. DMCreateDS) */
    isFE[f] = id == PETSCFE_CLASSID ? PETSC_TRUE : PETSC_FALSE;
  }

  CHKERRQ(DMPlexGetVTKCellHeight(dm, &cellHeight));
  for (f = 0; f < Nf; ++f) {
    PetscBool avoidTensor;

    CHKERRQ(DMGetFieldAvoidTensor(dm, f, &avoidTensor));
    avoidTensor = (avoidTensor || hasCohesive) ? PETSC_TRUE : PETSC_FALSE;
    if (label && label[f]) {
      IS              pointIS;
      const PetscInt *points;
      PetscInt        n;

      CHKERRQ(DMLabelGetStratumIS(label[f], 1, &pointIS));
      if (!pointIS) continue;
      CHKERRQ(ISGetLocalSize(pointIS, &n));
      CHKERRQ(ISGetIndices(pointIS, &points));
      for (p = 0; p < n; ++p) {
        const PetscInt point = points[p];
        PetscInt       dof, d;

        CHKERRQ(DMPlexGetCellType(dm, point, &ct));
        CHKERRQ(DMLabelGetValue(depthLabel, point, &d));
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
        CHKERRQ(PetscSectionSetFieldDof(section, point, f, dof));
        CHKERRQ(PetscSectionAddDof(section, point, dof));
      }
      CHKERRQ(ISRestoreIndices(pointIS, &points));
      CHKERRQ(ISDestroy(&pointIS));
    } else {
      for (dep = 0; dep <= depth - cellHeight; ++dep) {
        /* Cases: dim > depth (cell-vertex mesh), dim == depth (fully interpolated), dim < depth (interpolated submesh) */
        d    = dim <= depth ? dep : (!dep ? 0 : dim);
        CHKERRQ(DMPlexGetDepthStratum(dm, dep, &pStart, &pEnd));
        for (p = pStart; p < pEnd; ++p) {
          const PetscInt dof = numDof[f*(dim+1)+d];

          CHKERRQ(DMPlexGetCellType(dm, p, &ct));
          switch (ct) {
            case DM_POLYTOPE_POINT_PRISM_TENSOR:
            case DM_POLYTOPE_SEG_PRISM_TENSOR:
            case DM_POLYTOPE_TRI_PRISM_TENSOR:
            case DM_POLYTOPE_QUAD_PRISM_TENSOR:
              if (avoidTensor && isFE[f]) continue;
            default: break;
          }
          CHKERRQ(PetscSectionSetFieldDof(section, p, f, dof));
          CHKERRQ(PetscSectionAddDof(section, p, dof));
        }
      }
    }
  }
  CHKERRQ(PetscFree(isFE));
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
  CHKERRQ(PetscSectionGetNumFields(section, &Nf));
  for (bc = 0; bc < numBC; ++bc) {
    PetscInt        field = 0;
    const PetscInt *comp;
    const PetscInt *idx;
    PetscInt        Nc = 0, cNc = -1, n, i;

    if (Nf) {
      field = bcField[bc];
      CHKERRQ(PetscSectionGetFieldComponents(section, field, &Nc));
    }
    if (bcComps && bcComps[bc]) CHKERRQ(ISGetLocalSize(bcComps[bc], &cNc));
    if (bcComps && bcComps[bc]) CHKERRQ(ISGetIndices(bcComps[bc], &comp));
    CHKERRQ(ISGetLocalSize(bcPoints[bc], &n));
    CHKERRQ(ISGetIndices(bcPoints[bc], &idx));
    for (i = 0; i < n; ++i) {
      const PetscInt p = idx[i];
      PetscInt       numConst;

      if (Nf) {
        CHKERRQ(PetscSectionGetFieldDof(section, p, field, &numConst));
      } else {
        CHKERRQ(PetscSectionGetDof(section, p, &numConst));
      }
      /* If Nc <= 0, constrain every dof on the point */
      if (cNc > 0) {
        /* We assume that a point may have multiple "nodes", which are collections of Nc dofs,
           and that those dofs are numbered n*Nc+c */
        if (Nf) {
          PetscCheckFalse(numConst % Nc,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Point %D has %D dof which is not divisible by %D field components", p, numConst, Nc);
          numConst = (numConst/Nc) * cNc;
        } else {
          numConst = PetscMin(numConst, cNc);
        }
      }
      if (Nf) CHKERRQ(PetscSectionAddFieldConstraintDof(section, p, field, numConst));
      CHKERRQ(PetscSectionAddConstraintDof(section, p, numConst));
    }
    CHKERRQ(ISRestoreIndices(bcPoints[bc], &idx));
    if (bcComps && bcComps[bc]) CHKERRQ(ISRestoreIndices(bcComps[bc], &comp));
  }
  CHKERRQ(DMPlexGetAnchors(dm, &aSec, NULL));
  if (aSec) {
    PetscInt aStart, aEnd, a;

    CHKERRQ(PetscSectionGetChart(aSec, &aStart, &aEnd));
    for (a = aStart; a < aEnd; a++) {
      PetscInt dof, f;

      CHKERRQ(PetscSectionGetDof(aSec, a, &dof));
      if (dof) {
        /* if there are point-to-point constraints, then all dofs are constrained */
        CHKERRQ(PetscSectionGetDof(section, a, &dof));
        CHKERRQ(PetscSectionSetConstraintDof(section, a, dof));
        for (f = 0; f < Nf; f++) {
          CHKERRQ(PetscSectionGetFieldDof(section, a, f, &dof));
          CHKERRQ(PetscSectionSetFieldConstraintDof(section, a, f, dof));
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
  CHKERRQ(PetscSectionGetNumFields(section, &Nf));
  if (!Nf) PetscFunctionReturn(0);
  /* Initialize all field indices to -1 */
  CHKERRQ(PetscSectionGetChart(section, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {CHKERRQ(PetscSectionGetConstraintDof(section, p, &cdof)); maxDof = PetscMax(maxDof, cdof);}
  CHKERRQ(PetscMalloc1(maxDof, &indices));
  for (d = 0; d < maxDof; ++d) indices[d] = -1;
  for (p = pStart; p < pEnd; ++p) for (f = 0; f < Nf; ++f) CHKERRQ(PetscSectionSetFieldConstraintIndices(section, p, f, indices));
  /* Handle BC constraints */
  for (bc = 0; bc < numBC; ++bc) {
    const PetscInt  field = bcField[bc];
    const PetscInt *comp, *idx;
    PetscInt        Nc, cNc = -1, n, i;

    CHKERRQ(PetscSectionGetFieldComponents(section, field, &Nc));
    if (bcComps && bcComps[bc]) CHKERRQ(ISGetLocalSize(bcComps[bc], &cNc));
    if (bcComps && bcComps[bc]) CHKERRQ(ISGetIndices(bcComps[bc], &comp));
    CHKERRQ(ISGetLocalSize(bcPoints[bc], &n));
    CHKERRQ(ISGetIndices(bcPoints[bc], &idx));
    for (i = 0; i < n; ++i) {
      const PetscInt  p = idx[i];
      const PetscInt *find;
      PetscInt        fdof, fcdof, c, j;

      CHKERRQ(PetscSectionGetFieldDof(section, p, field, &fdof));
      if (!fdof) continue;
      if (cNc < 0) {
        for (d = 0; d < fdof; ++d) indices[d] = d;
        fcdof = fdof;
      } else {
        /* We assume that a point may have multiple "nodes", which are collections of Nc dofs,
           and that those dofs are numbered n*Nc+c */
        CHKERRQ(PetscSectionGetFieldConstraintDof(section, p, field, &fcdof));
        CHKERRQ(PetscSectionGetFieldConstraintIndices(section, p, field, &find));
        /* Get indices constrained by previous bcs */
        for (d = 0; d < fcdof; ++d) {if (find[d] < 0) break; indices[d] = find[d];}
        for (j = 0; j < fdof/Nc; ++j) for (c = 0; c < cNc; ++c) indices[d++] = j*Nc + comp[c];
        CHKERRQ(PetscSortRemoveDupsInt(&d, indices));
        for (c = d; c < fcdof; ++c) indices[c] = -1;
        fcdof = d;
      }
      CHKERRQ(PetscSectionSetFieldConstraintDof(section, p, field, fcdof));
      CHKERRQ(PetscSectionSetFieldConstraintIndices(section, p, field, indices));
    }
    if (bcComps && bcComps[bc]) CHKERRQ(ISRestoreIndices(bcComps[bc], &comp));
    CHKERRQ(ISRestoreIndices(bcPoints[bc], &idx));
  }
  /* Handle anchors */
  CHKERRQ(DMPlexGetAnchors(dm, &aSec, NULL));
  if (aSec) {
    PetscInt aStart, aEnd, a;

    for (d = 0; d < maxDof; ++d) indices[d] = d;
    CHKERRQ(PetscSectionGetChart(aSec, &aStart, &aEnd));
    for (a = aStart; a < aEnd; a++) {
      PetscInt dof, f;

      CHKERRQ(PetscSectionGetDof(aSec, a, &dof));
      if (dof) {
        /* if there are point-to-point constraints, then all dofs are constrained */
        for (f = 0; f < Nf; f++) {
          CHKERRQ(PetscSectionSetFieldConstraintIndices(section, a, f, indices));
        }
      }
    }
  }
  CHKERRQ(PetscFree(indices));
  PetscFunctionReturn(0);
}

/* Set the constrained indices on each point */
static PetscErrorCode DMPlexCreateSectionBCIndices(DM dm, PetscSection section)
{
  PetscInt      *indices;
  PetscInt       Nf, maxDof, pStart, pEnd, p, f, d;

  PetscFunctionBegin;
  CHKERRQ(PetscSectionGetNumFields(section, &Nf));
  CHKERRQ(PetscSectionGetMaxDof(section, &maxDof));
  CHKERRQ(PetscSectionGetChart(section, &pStart, &pEnd));
  CHKERRQ(PetscMalloc1(maxDof, &indices));
  for (d = 0; d < maxDof; ++d) indices[d] = -1;
  for (p = pStart; p < pEnd; ++p) {
    PetscInt cdof, d;

    CHKERRQ(PetscSectionGetConstraintDof(section, p, &cdof));
    if (cdof) {
      if (Nf) {
        PetscInt numConst = 0, foff = 0;

        for (f = 0; f < Nf; ++f) {
          const PetscInt *find;
          PetscInt        fcdof, fdof;

          CHKERRQ(PetscSectionGetFieldDof(section, p, f, &fdof));
          CHKERRQ(PetscSectionGetFieldConstraintDof(section, p, f, &fcdof));
          /* Change constraint numbering from field component to local dof number */
          CHKERRQ(PetscSectionGetFieldConstraintIndices(section, p, f, &find));
          for (d = 0; d < fcdof; ++d) indices[numConst+d] = find[d] + foff;
          numConst += fcdof;
          foff     += fdof;
        }
        if (cdof != numConst) CHKERRQ(PetscSectionSetConstraintDof(section, p, numConst));
      } else {
        for (d = 0; d < cdof; ++d) indices[d] = d;
      }
      CHKERRQ(PetscSectionSetConstraintIndices(section, p, indices));
    }
  }
  CHKERRQ(PetscFree(indices));
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

  PetscFunctionBegin;
  CHKERRQ(DMPlexCreateSectionFields(dm, numComp, section));
  CHKERRQ(DMPlexCreateSectionDof(dm, label, numDof, *section));
  CHKERRQ(DMPlexCreateSectionBCDof(dm, numBC, bcField, bcComps, bcPoints, *section));
  if (perm) CHKERRQ(PetscSectionSetPermutation(*section, perm));
  CHKERRQ(PetscSectionSetFromOptions(*section));
  CHKERRQ(PetscSectionSetUp(*section));
  CHKERRQ(DMPlexGetAnchors(dm,&aSec,NULL));
  if (numBC || aSec) {
    CHKERRQ(DMPlexCreateSectionBCIndicesField(dm, numBC, bcField, bcComps, bcPoints, *section));
    CHKERRQ(DMPlexCreateSectionBCIndices(dm, *section));
  }
  CHKERRQ(PetscSectionViewFromOptions(*section,NULL,"-section_view"));
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
  CHKERRQ(DMGetNumFields(dm, &Nf));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  /* FE and FV boundary conditions are handled slightly differently */
  CHKERRQ(PetscMalloc1(Nf, &isFE));
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    CHKERRQ(DMGetField(dm, f, NULL, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID)      {isFE[f] = PETSC_TRUE;}
    else if (id == PETSCFV_CLASSID) {isFE[f] = PETSC_FALSE;}
    else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", f);
  }
  /* Allocate boundary point storage for FEM boundaries */
  CHKERRQ(DMGetNumDS(dm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS  dsBC;
    PetscInt numBd, bd;

    CHKERRQ(DMGetRegionNumDS(dm, s, NULL, NULL, &dsBC));
    CHKERRQ(PetscDSGetNumBoundary(dsBC, &numBd));
    PetscCheckFalse(!Nf && numBd,PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "number of fields is zero and number of boundary conditions is nonzero (this should never happen)");
    for (bd = 0; bd < numBd; ++bd) {
      PetscInt                field;
      DMBoundaryConditionType type;
      DMLabel                 label;

      CHKERRQ(PetscDSGetBoundary(dsBC, bd, NULL, &type, NULL, &label, NULL, NULL, &field, NULL, NULL, NULL, NULL, NULL));
      if (label && isFE[field] && (type & DM_BC_ESSENTIAL)) ++numBC;
    }
  }
  /* Add ghost cell boundaries for FVM */
  CHKERRQ(DMPlexGetGhostCellStratum(dm, &cEndInterior, NULL));
  for (f = 0; f < Nf; ++f) if (!isFE[f] && cEndInterior >= 0) ++numBC;
  CHKERRQ(PetscCalloc3(numBC, &bcFields, numBC, &bcPoints, numBC, &bcComps));
  /* Constrain ghost cells for FV */
  for (f = 0; f < Nf; ++f) {
    PetscInt *newidx, c;

    if (isFE[f] || cEndInterior < 0) continue;
    CHKERRQ(PetscMalloc1(cEnd-cEndInterior,&newidx));
    for (c = cEndInterior; c < cEnd; ++c) newidx[c-cEndInterior] = c;
    bcFields[bc] = f;
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, cEnd-cEndInterior, newidx, PETSC_OWN_POINTER, &bcPoints[bc++]));
  }
  /* Handle FEM Dirichlet boundaries */
  CHKERRQ(DMGetNumDS(dm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS  dsBC;
    PetscInt numBd, bd;

    CHKERRQ(DMGetRegionNumDS(dm, s, NULL, NULL, &dsBC));
    CHKERRQ(PetscDSGetNumBoundary(dsBC, &numBd));
    for (bd = 0; bd < numBd; ++bd) {
      DMLabel                 label;
      const PetscInt         *comps;
      const PetscInt         *values;
      PetscInt                bd2, field, numComps, numValues;
      DMBoundaryConditionType type;
      PetscBool               duplicate = PETSC_FALSE;

      CHKERRQ(PetscDSGetBoundary(dsBC, bd, NULL, &type, NULL, &label, &numValues, &values, &field, &numComps, &comps, NULL, NULL, NULL));
      if (!isFE[field] || !label) continue;
      /* Only want to modify label once */
      for (bd2 = 0; bd2 < bd; ++bd2) {
        DMLabel l;

        CHKERRQ(PetscDSGetBoundary(dsBC, bd2, NULL, NULL, NULL, &l, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
        duplicate = l == label ? PETSC_TRUE : PETSC_FALSE;
        if (duplicate) break;
      }
      if (!duplicate && (isFE[field])) {
        /* don't complete cells, which are just present to give orientation to the boundary */
        CHKERRQ(DMPlexLabelComplete(dm, label));
      }
      /* Filter out cells, if you actually want to constrain cells you need to do things by hand right now */
      if (type & DM_BC_ESSENTIAL) {
        PetscInt       *newidx;
        PetscInt        n, newn = 0, p, v;

        bcFields[bc] = field;
        if (numComps) CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject) dm), numComps, comps, PETSC_COPY_VALUES, &bcComps[bc]));
        for (v = 0; v < numValues; ++v) {
          IS              tmp;
          const PetscInt *idx;

          CHKERRQ(DMLabelGetStratumIS(label, values[v], &tmp));
          if (!tmp) continue;
          CHKERRQ(ISGetLocalSize(tmp, &n));
          CHKERRQ(ISGetIndices(tmp, &idx));
          if (isFE[field]) {
            for (p = 0; p < n; ++p) if ((idx[p] < cStart) || (idx[p] >= cEnd)) ++newn;
          } else {
            for (p = 0; p < n; ++p) if ((idx[p] >= cStart) || (idx[p] < cEnd)) ++newn;
          }
          CHKERRQ(ISRestoreIndices(tmp, &idx));
          CHKERRQ(ISDestroy(&tmp));
        }
        CHKERRQ(PetscMalloc1(newn, &newidx));
        newn = 0;
        for (v = 0; v < numValues; ++v) {
          IS              tmp;
          const PetscInt *idx;

          CHKERRQ(DMLabelGetStratumIS(label, values[v], &tmp));
          if (!tmp) continue;
          CHKERRQ(ISGetLocalSize(tmp, &n));
          CHKERRQ(ISGetIndices(tmp, &idx));
          if (isFE[field]) {
            for (p = 0; p < n; ++p) if ((idx[p] < cStart) || (idx[p] >= cEnd)) newidx[newn++] = idx[p];
          } else {
            for (p = 0; p < n; ++p) if ((idx[p] >= cStart) || (idx[p] < cEnd)) newidx[newn++] = idx[p];
          }
          CHKERRQ(ISRestoreIndices(tmp, &idx));
          CHKERRQ(ISDestroy(&tmp));
        }
        CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, newn, newidx, PETSC_OWN_POINTER, &bcPoints[bc++]));
      }
    }
  }
  /* Handle discretization */
  CHKERRQ(PetscCalloc3(Nf,&labels,Nf,&numComp,Nf*(dim+1),&numDof));
  for (f = 0; f < Nf; ++f) {
    labels[f] = dm->fields[f].label;
    if (isFE[f]) {
      PetscFE         fe = (PetscFE) dm->fields[f].disc;
      const PetscInt *numFieldDof;
      PetscInt        fedim, d;

      CHKERRQ(PetscFEGetNumComponents(fe, &numComp[f]));
      CHKERRQ(PetscFEGetNumDof(fe, &numFieldDof));
      CHKERRQ(PetscFEGetSpatialDimension(fe, &fedim));
      for (d = 0; d < PetscMin(dim, fedim)+1; ++d) numDof[f*(dim+1)+d] = numFieldDof[d];
    } else {
      PetscFV fv = (PetscFV) dm->fields[f].disc;

      CHKERRQ(PetscFVGetNumComponents(fv, &numComp[f]));
      numDof[f*(dim+1)+dim] = numComp[f];
    }
  }
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  for (f = 0; f < Nf; ++f) {
    PetscInt d;
    for (d = 1; d < dim; ++d) {
      PetscCheckFalse((numDof[f*(dim+1)+d] > 0) && (depth < dim),PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Mesh must be interpolated when unknowns are specified on edges or faces.");
    }
  }
  CHKERRQ(DMPlexCreateSection(dm, labels, numComp, numDof, numBC, bcFields, bcComps, bcPoints, NULL, &section));
  for (f = 0; f < Nf; ++f) {
    PetscFE     fe;
    const char *name;

    CHKERRQ(DMGetField(dm, f, NULL, (PetscObject *) &fe));
    if (!((PetscObject) fe)->name) continue;
    CHKERRQ(PetscObjectGetName((PetscObject) fe, &name));
    CHKERRQ(PetscSectionSetFieldName(section, f, name));
  }
  CHKERRQ(DMSetLocalSection(dm, section));
  CHKERRQ(PetscSectionDestroy(&section));
  for (bc = 0; bc < numBC; ++bc) {CHKERRQ(ISDestroy(&bcPoints[bc]));CHKERRQ(ISDestroy(&bcComps[bc]));}
  CHKERRQ(PetscFree3(bcFields,bcPoints,bcComps));
  CHKERRQ(PetscFree3(labels,numComp,numDof));
  CHKERRQ(PetscFree(isFE));
  PetscFunctionReturn(0);
}
