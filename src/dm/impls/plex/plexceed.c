#include <petsc/private/dmpleximpl.h>           /*I      "petscdmplex.h"          I*/

/*@C
  DMPlexGetLocalOffsets - Allocate and populate array of local offsets.

  Input Parameters:
  dm - The DMPlex object
  domain_label - label for DMPlex domain, or NULL for whole domain
  label_value - Stratum value
  height - Height of target cells in DMPlex topology
  dm_field - Index of DMPlex field

  Output Parameters:
  num_cells - Number of local cells
  cell_size - Size of each cell, given by cell_size * num_comp = num_dof
  num_comp - Number of components per dof
  l_size - Size of local vector
  offsets - Allocated offsets array for cells

  Notes: Allocate and populate array of shape [num_cells, cell_size] defining offsets for each value (cell, node) for local vector of the DMPlex field. All offsets are in the range [0, l_size - 1]. Caller is responsible for freeing the offsets array using PetscFree().

  Level: developer

.seealso: DMPlexGetClosureIndices(), DMPlexSetClosurePermutationTensor(), DMPlexGetCeedRestriction()
@*/
PetscErrorCode DMPlexGetLocalOffsets(DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, PetscInt dm_field, PetscInt *num_cells, PetscInt *cell_size, PetscInt *num_comp, PetscInt *l_size, PetscInt **offsets)
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
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetDimension(dm, &dim));
  {
    IS              field_is;
    const PetscInt *fields;
    PetscInt        num_fields;

    PetscCall(DMGetRegionDS(dm, domain_label, &field_is, &ds));
    // Translate dm_field to ds_field
    PetscCall(ISGetIndices(field_is, &fields));
    PetscCall(ISGetSize(field_is, &num_fields));
    for (PetscInt i=0; i<num_fields; i++) {
      if (dm_field == fields[i]) {
        ds_field = i;
        break;
      }
    }
    PetscCall(ISRestoreIndices(field_is, &fields));
  }
  PetscCheck(ds_field != -1,PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Could not find dm_field %D in DS", dm_field);

  {
    PetscInt depth;
    DMLabel  depth_label;
    IS       depth_is;

    PetscCall(DMPlexGetDepth(dm, &depth));
    PetscCall(DMPlexGetDepthLabel(dm, &depth_label));
    PetscCall(DMLabelGetStratumIS(depth_label, depth - height, &depth_is));
    if (domain_label) {
      IS domain_is;

      PetscCall(DMLabelGetStratumIS(domain_label, label_value, &domain_is));
      if (domain_is) { // domainIS is non-empty
        PetscCall(ISIntersect(depth_is, domain_is, &iter_is));
        PetscCall(ISDestroy(&domain_is));
      } else { // domainIS is NULL (empty)
        iter_is = NULL;
      }
      PetscCall(ISDestroy(&depth_is));
    } else {
      iter_is = depth_is;
    }
    if (iter_is) {
      PetscCall(ISGetLocalSize(iter_is, num_cells));
      PetscCall(ISGetIndices(iter_is, &iter_indices));
    } else {
      *num_cells = 0;
      iter_indices = NULL;
    }
  }

  {
    PetscDualSpace dual_space;
    PetscInt       num_dual_basis_vectors;

    PetscCall(PetscDSGetDiscretization(ds, ds_field, (PetscObject*)&fe));
    PetscCall(PetscFEGetHeightSubspace(fe, height, &fe));
    PetscCall(PetscFEGetDualSpace(fe, &dual_space));
    PetscCall(PetscDualSpaceGetDimension(dual_space, &num_dual_basis_vectors));
    PetscCall(PetscDualSpaceGetNumComponents(dual_space, num_comp));
    PetscCheck(num_dual_basis_vectors % *num_comp == 0,PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for number of dual basis vectors %D not divisible by %D components", num_dual_basis_vectors, *num_comp);
    *cell_size = num_dual_basis_vectors / *num_comp;
  }
  PetscInt restr_size = (*num_cells)*(*cell_size);
  PetscCall(PetscMalloc1(restr_size, &restr_indices));
  PetscInt cell_offset = 0;

  PetscInt P = (PetscInt) pow(*cell_size, 1.0 / (dim - height));
  for (PetscInt p = 0; p < *num_cells; p++) {
    PetscBool flip = PETSC_FALSE;
    PetscInt c = iter_indices[p];
    PetscInt num_indices, *indices;
    PetscInt field_offsets[17]; // max number of fields plus 1
    PetscCall(DMPlexGetClosureIndices(dm, section, section, c, PETSC_TRUE, &num_indices, &indices, field_offsets, NULL));
    if (height > 0) {
      PetscInt num_cells_support, num_faces, start = -1;
      const PetscInt *orients, *faces, *cells;
      PetscCall(DMPlexGetSupport(dm, c, &cells));
      PetscCall(DMPlexGetSupportSize(dm, c, &num_cells_support));
      PetscCheck(num_cells_support == 1,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Expected one cell in support of exterior face, but got %" PetscInt_FMT " cells", num_cells_support);
      PetscCall(DMPlexGetCone(dm, cells[0], &faces));
      PetscCall(DMPlexGetConeSize(dm, cells[0], &num_faces));
      for (PetscInt i=0; i<num_faces; i++) {if (faces[i] == c) start = i;}
      PetscCheck(start >= 0,PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT, "Could not find face %" PetscInt_FMT " in cone of its support", c);
      PetscCall(DMPlexGetConeOrientation(dm, cells[0], &orients));
      if (orients[start] < 0) flip = PETSC_TRUE;
    }

    for (PetscInt i = 0; i < *cell_size; i++) {
      PetscInt ii = i;
      if (flip) {
        if (*cell_size == P) ii = *cell_size - 1 - i;
        else if (*cell_size == P*P) {
          PetscInt row = i / P, col = i % P;
          ii = row + col * P;
        } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for flipping point with cell size %D != P (%D) or P^2", *cell_size, P);
      }
      // Essential boundary conditions are encoded as -(loc+1), but we don't care so we decode.
      PetscInt loc = indices[field_offsets[dm_field] + ii*(*num_comp)];
      restr_indices[cell_offset++] = loc >= 0 ? loc : -(loc + 1);
    }
    PetscCall(DMPlexRestoreClosureIndices(dm, section, section, c, PETSC_TRUE, &num_indices, &indices, field_offsets, NULL));
  }
  PetscCheck(cell_offset == restr_size,PETSC_COMM_SELF, PETSC_ERR_SUP, "Shape mismatch, offsets array of shape (%D, %D) initialized for %D nodes", *num_cells, (*cell_size), cell_offset);
  if (iter_is) PetscCall(ISRestoreIndices(iter_is, &iter_indices));
  PetscCall(ISDestroy(&iter_is));

  *offsets = restr_indices;
  PetscCall(PetscSectionGetStorageSize(section, l_size));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_LIBCEED)
#include <petscdmplexceed.h>

/*@C
  DMPlexGetCeedRestriction - Define the libCEED map from the local vector (Lvector) to the cells (Evector)

  Input Parameters:
  dm - The DMPlex object
  domain_label - label for DMPlex domain, or NULL for the whole domain
  label_value - Stratum value
  height - Height of target cells in DMPlex topology
  dm_field - Index of DMPlex field

  Output Parameters:
  ERestrict - libCEED restriction from local vector to to the cells

  Level: developer
@*/
PetscErrorCode DMPlexGetCeedRestriction(DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, PetscInt dm_field, CeedElemRestriction *ERestrict)
{
  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(ERestrict, 2);
  if (!dm->ceedERestrict) {
    PetscInt     num_cells, cell_size, num_comp, lvec_size, *restr_indices;
    CeedElemRestriction elem_restr;
    Ceed         ceed;

    PetscCall(DMPlexGetLocalOffsets(dm, domain_label, label_value, height, dm_field, &num_cells, &cell_size, &num_comp, &lvec_size, &restr_indices));
    PetscCall(DMGetCeed(dm, &ceed));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_cells, cell_size, num_comp, 1, lvec_size, CEED_MEM_HOST, CEED_COPY_VALUES, restr_indices, &elem_restr));
    PetscCall(PetscFree(restr_indices));
    dm->ceedERestrict = elem_restr;
  }
  *ERestrict = dm->ceedERestrict;
  PetscFunctionReturn(0);
}

#endif
