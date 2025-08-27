#include <petsc/private/dmpleximpl.h> /*I      "petscdmplex.h"          I*/

PetscErrorCode DMGetPoints_Internal(DM dm, DMLabel domainLabel, PetscInt labelVal, PetscInt height, IS *pointIS)
{
  PetscInt depth;
  DMLabel  depthLabel;
  IS       depthIS;

  PetscFunctionBegin;
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMPlexGetDepthLabel(dm, &depthLabel));
  PetscCall(DMLabelGetStratumIS(depthLabel, depth - height, &depthIS));
  if (domainLabel) {
    IS domainIS;

    PetscCall(DMLabelGetStratumIS(domainLabel, labelVal, &domainIS));
    if (domainIS) { // domainIS is non-empty
      PetscCall(ISIntersect(depthIS, domainIS, pointIS));
      PetscCall(ISDestroy(&domainIS));
    } else { // domainIS is NULL (empty)
      *pointIS = NULL;
    }
    PetscCall(ISDestroy(&depthIS));
  } else {
    *pointIS = depthIS;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetLocalOffsets - Allocate and populate array of local offsets for each cell closure.

  Not collective

  Input Parameters:
+ dm           - The `DMPLEX` object
. domain_label - label for `DMPLEX` domain, or NULL for whole domain
. label_value  - Stratum value
. height       - Height of target cells in `DMPLEX` topology
- dm_field     - Index of `DMPLEX` field

  Output Parameters:
+ num_cells - Number of local cells
. cell_size - Size of each cell, given by cell_size * num_comp = num_dof
. num_comp  - Number of components per dof
. l_size    - Size of local vector
- offsets   - Allocated offsets array for cells

  Level: developer

  Notes:
  Allocate and populate array of shape [num_cells, cell_size] defining offsets for each value (cell, node) for local vector of the `DMPLEX` field. All offsets are in the range [0, l_size - 1].

  Caller is responsible for freeing the offsets array using `PetscFree()`.

.seealso: [](ch_unstructured), `DMPlexGetLocalOffsetsSupport()`, `DM`, `DMPLEX`, `DMLabel`, `DMPlexGetClosureIndices()`, `DMPlexSetClosurePermutationTensor()`, `DMPlexGetCeedRestriction()`
@*/
PetscErrorCode DMPlexGetLocalOffsets(DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, PetscInt dm_field, PetscInt *num_cells, PetscInt *cell_size, PetscInt *num_comp, PetscInt *l_size, PetscInt *offsets[])
{
  PetscDS         ds = NULL;
  PetscFE         fe;
  PetscSection    section;
  PetscInt        dim, ds_field = -1;
  PetscInt       *restr_indices;
  const PetscInt *iter_indices;
  IS              iter_is;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscLogEventBegin(DMPLEX_GetLocalOffsets, dm, 0, 0, 0));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionGetStorageSize(section, l_size));
  {
    IS              field_is;
    const PetscInt *fields;
    PetscInt        num_fields;

    PetscCall(DMGetRegionDS(dm, domain_label, &field_is, &ds, NULL));
    PetscCheck(field_is, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Domain label does not have any fields associated with it");
    // Translate dm_field to ds_field
    PetscCall(ISGetIndices(field_is, &fields));
    PetscCall(ISGetSize(field_is, &num_fields));
    for (PetscInt i = 0; i < num_fields; i++) {
      if (dm_field == fields[i]) {
        ds_field = i;
        break;
      }
    }
    PetscCall(ISRestoreIndices(field_is, &fields));
  }
  PetscCheck(ds_field != -1, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Could not find dm_field %" PetscInt_FMT " in DS", dm_field);

  PetscCall(DMGetPoints_Internal(dm, domain_label, label_value, height, &iter_is));
  if (iter_is) {
    PetscCall(ISGetLocalSize(iter_is, num_cells));
    PetscCall(ISGetIndices(iter_is, &iter_indices));
  } else {
    *num_cells   = 0;
    iter_indices = NULL;
  }

  {
    PetscDualSpace dual_space;
    PetscInt       num_dual_basis_vectors;

    PetscCall(PetscDSGetDiscretization(ds, ds_field, (PetscObject *)&fe));
    PetscCall(PetscFEGetHeightSubspace(fe, height, &fe));
    PetscCheck(fe, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Height %" PetscInt_FMT " is invalid for DG discretizations", height);
    PetscCall(PetscFEGetDualSpace(fe, &dual_space));
    PetscCall(PetscDualSpaceGetDimension(dual_space, &num_dual_basis_vectors));
    PetscCall(PetscDualSpaceGetNumComponents(dual_space, num_comp));
    PetscCheck(num_dual_basis_vectors % *num_comp == 0, PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for number of dual basis vectors %" PetscInt_FMT " not divisible by %" PetscInt_FMT " components", num_dual_basis_vectors, *num_comp);
    *cell_size = num_dual_basis_vectors / *num_comp;
  }
  PetscInt restr_size = (*num_cells) * (*cell_size);
  PetscCall(PetscMalloc1(restr_size, &restr_indices));
  PetscInt cell_offset = 0;

  PetscInt P = dim - height ? (PetscInt)PetscPowReal(*cell_size, 1.0 / (dim - height)) : 0;
  for (PetscInt p = 0; p < *num_cells; p++) {
    PetscBool flip = PETSC_FALSE;
    PetscInt  c    = iter_indices[p];
    PetscInt  num_indices, *indices;
    PetscInt  field_offsets[17]; // max number of fields plus 1
    PetscCall(DMPlexGetClosureIndices(dm, section, section, c, PETSC_TRUE, &num_indices, &indices, field_offsets, NULL));
    if (height > 0) {
      PetscInt        num_cells_support, num_faces, start = -1;
      const PetscInt *orients, *faces, *cells;
      PetscCall(DMPlexGetSupport(dm, c, &cells));
      PetscCall(DMPlexGetSupportSize(dm, c, &num_cells_support));
      PetscCheck(num_cells_support == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Expected one cell in support of exterior face, but got %" PetscInt_FMT " cells", num_cells_support);
      PetscCall(DMPlexGetCone(dm, cells[0], &faces));
      PetscCall(DMPlexGetConeSize(dm, cells[0], &num_faces));
      for (PetscInt i = 0; i < num_faces; i++) {
        if (faces[i] == c) start = i;
      }
      PetscCheck(start >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT, "Could not find face %" PetscInt_FMT " in cone of its support", c);
      PetscCall(DMPlexGetConeOrientation(dm, cells[0], &orients));
      if (orients[start] < 0) flip = PETSC_TRUE;
    }

    for (PetscInt i = 0; i < *cell_size; i++) {
      PetscInt ii = i;
      if (flip) {
        if (*cell_size == P) ii = *cell_size - 1 - i;
        else if (*cell_size == P * P) {
          PetscInt row = i / P, col = i % P;
          ii = row + col * P;
        } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for flipping point with cell size %" PetscInt_FMT " != P (%" PetscInt_FMT ") or P^2", *cell_size, P);
      }
      // Essential boundary conditions are encoded as -(loc+1), but we don't care so we decode.
      PetscInt loc                 = indices[field_offsets[dm_field] + ii * (*num_comp)];
      loc                          = loc < 0 ? -(loc + 1) : loc;
      restr_indices[cell_offset++] = loc;
      PetscCheck(loc >= 0 && loc < *l_size, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Location %" PetscInt_FMT " not in [0, %" PetscInt_FMT ") local vector", loc, *l_size);
    }
    PetscCall(DMPlexRestoreClosureIndices(dm, section, section, c, PETSC_TRUE, &num_indices, &indices, field_offsets, NULL));
  }
  PetscCheck(cell_offset == restr_size, PETSC_COMM_SELF, PETSC_ERR_SUP, "Shape mismatch, offsets array of shape (%" PetscInt_FMT ", %" PetscInt_FMT ") initialized for %" PetscInt_FMT " nodes", *num_cells, *cell_size, cell_offset);
  if (iter_is) PetscCall(ISRestoreIndices(iter_is, &iter_indices));
  PetscCall(ISDestroy(&iter_is));

  *offsets = restr_indices;
  PetscCall(PetscLogEventEnd(DMPLEX_GetLocalOffsets, dm, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetLocalOffsetsSupport - Allocate and populate arrays of local offsets for each face support.

  Not collective

  Input Parameters:
+ dm           - The `DMPLEX` object
. domain_label - label for `DMPLEX` domain, or NULL for whole domain
- label_value  - Stratum value

  Output Parameters:
+ num_faces  - Number of local, non-boundary faces
. num_comp   - Number of components per dof
. l_size     - Size of local vector
. offsetsNeg - Allocated offsets array for cells on the inward normal side of each face
- offsetsPos - Allocated offsets array for cells on the outward normal side of each face

  Level: developer

  Notes:
  Allocate and populate array of shape [num_cells, num_comp] defining offsets for each cell for local vector of the `DMPLEX` field. All offsets are in the range [0, l_size - 1].

  Caller is responsible for freeing the offsets array using `PetscFree()`.

.seealso: [](ch_unstructured), `DMPlexGetLocalOffsets()`, `DM`, `DMPLEX`, `DMLabel`, `DMPlexGetClosureIndices()`, `DMPlexSetClosurePermutationTensor()`, `DMPlexGetCeedRestriction()`
@*/
PetscErrorCode DMPlexGetLocalOffsetsSupport(DM dm, DMLabel domain_label, PetscInt label_value, PetscInt *num_faces, PetscInt *num_comp, PetscInt *l_size, PetscInt **offsetsNeg, PetscInt **offsetsPos)
{
  PetscDS         ds = NULL;
  PetscFV         fv;
  PetscSection    section;
  PetscInt        dim, height = 1, dm_field = 0, ds_field = 0, Nf, NfInt = 0, Nc;
  PetscInt       *restr_indices_neg, *restr_indices_pos;
  const PetscInt *iter_indices;
  IS              iter_is;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetRegionDS(dm, domain_label, NULL, &ds, NULL));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionGetStorageSize(section, l_size));

  PetscCall(DMGetPoints_Internal(dm, domain_label, label_value, height, &iter_is));
  if (iter_is) {
    PetscCall(ISGetIndices(iter_is, &iter_indices));
    PetscCall(ISGetLocalSize(iter_is, &Nf));
    for (PetscInt p = 0, Ns; p < Nf; ++p) {
      PetscCall(DMPlexGetSupportSize(dm, iter_indices[p], &Ns));
      if (Ns == 2) ++NfInt;
    }
    *num_faces = NfInt;
  } else {
    *num_faces   = 0;
    iter_indices = NULL;
  }

  PetscCall(PetscDSGetDiscretization(ds, ds_field, (PetscObject *)&fv));
  PetscCall(PetscFVGetNumComponents(fv, &Nc));
  PetscCall(PetscMalloc1(NfInt, &restr_indices_neg));
  PetscCall(PetscMalloc1(NfInt, &restr_indices_pos));
  PetscInt face_offset_neg = 0, face_offset_pos = 0;

  for (PetscInt p = 0; p < Nf; ++p) {
    const PetscInt  face = iter_indices[p];
    PetscInt        num_indices, *indices;
    PetscInt        field_offsets[17]; // max number of fields plus 1
    const PetscInt *supp;
    PetscInt        Ns, loc;

    PetscCall(DMPlexGetSupport(dm, face, &supp));
    PetscCall(DMPlexGetSupportSize(dm, face, &Ns));
    // Ignore boundary faces
    //   TODO check for face on parallel boundary
    if (Ns == 2) {
      // Essential boundary conditions are encoded as -(loc+1), but we don't care so we decode.
      PetscCall(DMPlexGetClosureIndices(dm, section, section, supp[0], PETSC_TRUE, &num_indices, &indices, field_offsets, NULL));
      PetscCheck(num_indices == Nc, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Number of closure indices %" PetscInt_FMT " != %" PetscInt_FMT " number of FV components", num_indices, Nc);
      loc                                  = indices[field_offsets[dm_field]];
      loc                                  = loc < 0 ? -(loc + 1) : loc;
      restr_indices_neg[face_offset_neg++] = loc;
      PetscCheck(loc >= 0 && loc + Nc <= *l_size, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Location %" PetscInt_FMT " + Nc not in [0, %" PetscInt_FMT ") local vector", loc, *l_size);
      PetscCall(DMPlexRestoreClosureIndices(dm, section, section, supp[0], PETSC_TRUE, &num_indices, &indices, field_offsets, NULL));
      PetscCall(DMPlexGetClosureIndices(dm, section, section, supp[1], PETSC_TRUE, &num_indices, &indices, field_offsets, NULL));
      PetscCheck(num_indices == Nc, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Number of closure indices %" PetscInt_FMT " != %" PetscInt_FMT " number of FV components", num_indices, Nc);
      loc                                  = indices[field_offsets[dm_field]];
      loc                                  = loc < 0 ? -(loc + 1) : loc;
      restr_indices_pos[face_offset_pos++] = loc;
      PetscCheck(loc >= 0 && loc + Nc <= *l_size, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Location %" PetscInt_FMT " + Nc not in [0, %" PetscInt_FMT ") local vector", loc, *l_size);
      PetscCall(DMPlexRestoreClosureIndices(dm, section, section, supp[1], PETSC_TRUE, &num_indices, &indices, field_offsets, NULL));
    }
  }
  PetscCheck(face_offset_neg == NfInt, PETSC_COMM_SELF, PETSC_ERR_SUP, "Shape mismatch, neg offsets array of shape (%" PetscInt_FMT ") initialized for %" PetscInt_FMT " nodes", NfInt, face_offset_neg);
  PetscCheck(face_offset_pos == NfInt, PETSC_COMM_SELF, PETSC_ERR_SUP, "Shape mismatch, pos offsets array of shape (%" PetscInt_FMT ") initialized for %" PetscInt_FMT " nodes", NfInt, face_offset_pos);
  if (iter_is) PetscCall(ISRestoreIndices(iter_is, &iter_indices));
  PetscCall(ISDestroy(&iter_is));

  *num_comp   = Nc;
  *offsetsNeg = restr_indices_neg;
  *offsetsPos = restr_indices_pos;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_LIBCEED)
  #include <petscdmplexceed.h>

// Consumes the input petsc_indices and provides the output ceed_indices; no-copy when the sizes match.
static PetscErrorCode PetscIntArrayIntoCeedInt_Private(PetscInt length, PetscInt max_bound, const char *max_bound_name, PetscInt **petsc_indices, CeedInt **ceed_indices)
{
  PetscFunctionBegin;
  if (length) PetscAssertPointer(petsc_indices, 3);
  PetscAssertPointer(ceed_indices, 4);
  #if defined(PETSC_USE_64BIT_INDICES)
  PetscCheck(max_bound <= PETSC_INT32_MAX, PETSC_COMM_SELF, PETSC_ERR_SUP, "%s %" PetscInt_FMT " does not fit in int32_t", max_bound_name, max_bound);
  {
    CeedInt *ceed_ind;
    PetscCall(PetscMalloc1(length, &ceed_ind));
    for (PetscInt i = 0; i < length; i++) ceed_ind[i] = (*petsc_indices)[i];
    *ceed_indices = ceed_ind;
    PetscCall(PetscFree(*petsc_indices));
  }
  #else
  *ceed_indices  = *petsc_indices;
  *petsc_indices = NULL;
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetCeedRestriction - Define the libCEED map from the local vector (Lvector) to the cells (Evector)

  Input Parameters:
+  dm - The `DMPLEX` object
.  domain_label - label for `DMPLEX` domain, or NULL for the whole domain
.  label_value - Stratum value
.  height - Height of target cells in `DMPLEX` topology
-  dm_field - Index of `DMPLEX` field

  Output Parameter:
.  ERestrict - libCEED restriction from local vector to the cells

  Level: developer

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMLabel`, `DMPlexGetLocalOffsets()`
@*/
PetscErrorCode DMPlexGetCeedRestriction(DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, PetscInt dm_field, CeedElemRestriction *ERestrict)
{
  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(ERestrict, 6);
  if (!dm->ceedERestrict) {
    PetscInt            num_cells, cell_size, num_comp, lvec_size, *restr_indices;
    CeedElemRestriction elem_restr;
    Ceed                ceed;

    PetscCall(DMPlexGetLocalOffsets(dm, domain_label, label_value, height, dm_field, &num_cells, &cell_size, &num_comp, &lvec_size, &restr_indices));
    PetscCall(DMGetCeed(dm, &ceed));
    {
      CeedInt *ceed_indices;
      PetscCall(PetscIntArrayIntoCeedInt_Private(num_cells * cell_size, lvec_size, "lvec_size", &restr_indices, &ceed_indices));
      PetscCallCEED(CeedElemRestrictionCreate(ceed, num_cells, cell_size, num_comp, 1, lvec_size, CEED_MEM_HOST, CEED_COPY_VALUES, ceed_indices, &elem_restr));
      PetscCall(PetscFree(ceed_indices));
    }
    dm->ceedERestrict = elem_restr;
  }
  *ERestrict = dm->ceedERestrict;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexCreateCeedRestrictionFVM(DM dm, CeedElemRestriction *erL, CeedElemRestriction *erR)
{
  Ceed      ceed;
  PetscInt *offL, *offR;
  PetscInt  num_faces, num_comp, lvec_size;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(erL, 2);
  PetscAssertPointer(erR, 3);
  PetscCall(DMGetCeed(dm, &ceed));
  PetscCall(DMPlexGetLocalOffsetsSupport(dm, NULL, 0, &num_faces, &num_comp, &lvec_size, &offL, &offR));
  {
    CeedInt *ceed_off;
    PetscCall(PetscIntArrayIntoCeedInt_Private(num_faces * 1, lvec_size, "lvec_size", &offL, &ceed_off));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_faces, 1, num_comp, 1, lvec_size, CEED_MEM_HOST, CEED_COPY_VALUES, ceed_off, erL));
    PetscCall(PetscFree(ceed_off));
    PetscCall(PetscIntArrayIntoCeedInt_Private(num_faces * 1, lvec_size, "lvec_size", &offR, &ceed_off));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_faces, 1, num_comp, 1, lvec_size, CEED_MEM_HOST, CEED_COPY_VALUES, ceed_off, erR));
    PetscCall(PetscFree(ceed_off));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// TODO DMPlexComputeGeometryFVM() also computes centroids and minimum radius
// TODO DMPlexComputeGeometryFVM() flips normal to match support orientation
// This function computes area-weights normals
PetscErrorCode DMPlexCeedComputeGeometryFVM(DM dm, CeedVector qd)
{
  DMLabel         domain_label = NULL;
  PetscInt        label_value = 0, height = 1, Nf, cdim;
  const PetscInt *iter_indices;
  IS              iter_is;
  CeedScalar     *qdata;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMGetPoints_Internal(dm, domain_label, label_value, height, &iter_is));
  if (iter_is) {
    PetscCall(ISGetIndices(iter_is, &iter_indices));
    PetscCall(ISGetLocalSize(iter_is, &Nf));
    for (PetscInt p = 0, Ns; p < Nf; ++p) PetscCall(DMPlexGetSupportSize(dm, iter_indices[p], &Ns));
  } else {
    iter_indices = NULL;
  }

  PetscCallCEED(CeedVectorSetValue(qd, 0.));
  PetscCallCEED(CeedVectorGetArray(qd, CEED_MEM_HOST, &qdata));
  for (PetscInt p = 0, off = 0; p < Nf; ++p) {
    const PetscInt  face = iter_indices[p];
    const PetscInt *supp;
    PetscInt        suppSize;

    PetscCall(DMPlexGetSupport(dm, face, &supp));
    PetscCall(DMPlexGetSupportSize(dm, face, &suppSize));
    // Ignore boundary faces
    //   TODO check for face on parallel boundary
    if (suppSize == 2) {
      DMPolytopeType ct;
      PetscReal      area, fcentroid[3], centroids[2][3];

      PetscCall(DMPlexComputeCellGeometryFVM(dm, face, &area, fcentroid, &qdata[off]));
      for (PetscInt d = 0; d < cdim; ++d) qdata[off + d] *= area;
      off += cdim;
      for (PetscInt s = 0; s < suppSize; ++s) {
        PetscCall(DMPlexGetCellType(dm, supp[s], &ct));
        if (ct == DM_POLYTOPE_FV_GHOST) continue;
        PetscCall(DMPlexComputeCellGeometryFVM(dm, supp[s], &qdata[off + s], centroids[s], NULL));
      }
      // Give FV ghosts the same volume as the opposite cell
      for (PetscInt s = 0; s < suppSize; ++s) {
        PetscCall(DMPlexGetCellType(dm, supp[s], &ct));
        if (ct != DM_POLYTOPE_FV_GHOST) continue;
        qdata[off + s] = qdata[off + (1 - s)];
        for (PetscInt d = 0; d < cdim; ++d) centroids[s][d] = fcentroid[d];
      }
      // Flip normal orientation if necessary to match ordering in support
      {
        CeedScalar *normal = &qdata[off - cdim];
        PetscReal   l[3], r[3], v[3];

        PetscCall(DMLocalizeCoordinateReal_Internal(dm, cdim, fcentroid, centroids[0], l));
        PetscCall(DMLocalizeCoordinateReal_Internal(dm, cdim, fcentroid, centroids[1], r));
        DMPlex_WaxpyD_Internal(cdim, -1, l, r, v);
        if (DMPlex_DotRealD_Internal(cdim, normal, v) < 0) {
          for (PetscInt d = 0; d < cdim; ++d) normal[d] = -normal[d];
        }
        if (DMPlex_DotRealD_Internal(cdim, normal, v) <= 0) {
          PetscCheck(cdim != 2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Direction for face %" PetscInt_FMT " could not be fixed, normal (%g,%g) v (%g,%g)", face, (double)normal[0], (double)normal[1], (double)v[0], (double)v[1]);
          PetscCheck(cdim != 3, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Direction for face %" PetscInt_FMT " could not be fixed, normal (%g,%g,%g) v (%g,%g,%g)", face, (double)normal[0], (double)normal[1], (double)normal[2], (double)v[0], (double)v[1], (double)v[2]);
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Direction for face %" PetscInt_FMT " could not be fixed", face);
        }
      }
      off += suppSize;
    }
  }
  PetscCallCEED(CeedVectorRestoreArray(qd, &qdata));
  if (iter_is) PetscCall(ISRestoreIndices(iter_is, &iter_indices));
  PetscCall(ISDestroy(&iter_is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
