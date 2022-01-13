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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscDS      ds = NULL;
  PetscFE      fe;
  PetscSection section;
  PetscInt     dim;
  PetscInt    *restr_indices;
  const PetscInt *iter_indices;
  IS           iter_is;

  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  if (domain_label) {
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = DMGetFirstLabelEntry_Internal(dm, dm, domain_label, 1, &label_value, dim, NULL, &ds);CHKERRQ(ierr);
  }

  // Translate dm_field to ds_field
  PetscInt ds_field = -1;
  for (PetscInt i=0; i<dm->Nds; i++) {
    if (!domain_label || domain_label == dm->probs[i].label) {
      ds = dm->probs[i].ds;
    }
    if (ds == dm->probs[i].ds) {
      const PetscInt *arr;
      PetscInt nf;
      IS is = dm->probs[i].fields;
      ierr = ISGetIndices(is, &arr);CHKERRQ(ierr);
      ierr = ISGetSize(is, &nf);CHKERRQ(ierr);
      for (PetscInt j=0; j<nf; j++) {
        if (dm_field == arr[j]) {
          ds_field = j;
          break;
        }
      }
      ierr = ISRestoreIndices(is, &arr);CHKERRQ(ierr);
    }
  }
  if (ds_field == -1) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Could not find dm_field %D in DS", dm_field);

  {
    PetscInt depth;
    DMLabel depth_label;
    IS depth_is;
    ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
    ierr = DMPlexGetDepthLabel(dm, &depth_label);CHKERRQ(ierr);
    ierr = DMLabelGetStratumIS(depth_label, depth - height, &depth_is);CHKERRQ(ierr);
    if (domain_label) {
      IS domain_is;
      ierr = DMLabelGetStratumIS(domain_label, label_value, &domain_is);CHKERRQ(ierr);
      if (domain_is) { // domainIS is non-empty
        ierr = ISIntersect(depth_is, domain_is, &iter_is);CHKERRQ(ierr);
        ierr = ISDestroy(&domain_is);CHKERRQ(ierr);
      } else { // domainIS is NULL (empty)
        iter_is = NULL;
      }
      ierr = ISDestroy(&depth_is);CHKERRQ(ierr);
    } else {
      iter_is = depth_is;
    }
    if (iter_is) {
      ierr = ISGetLocalSize(iter_is, num_cells);CHKERRQ(ierr);
      ierr = ISGetIndices(iter_is, &iter_indices);CHKERRQ(ierr);
    } else {
      *num_cells = 0;
      iter_indices = NULL;
    }
  }

  {
    PetscDualSpace dual_space;
    PetscInt num_dual_basis_vectors;
    ierr = PetscDSGetDiscretization(ds, ds_field, (PetscObject*)&fe);CHKERRQ(ierr);
    ierr = PetscFEGetHeightSubspace(fe, height, &fe);CHKERRQ(ierr);
    ierr = PetscFEGetDualSpace(fe, &dual_space);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetDimension(dual_space, &num_dual_basis_vectors);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetNumComponents(dual_space, num_comp);CHKERRQ(ierr);
    if (num_dual_basis_vectors % *num_comp != 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for number of dual basis vectors %D not divisible by %D components", num_dual_basis_vectors, *num_comp);
    *cell_size = num_dual_basis_vectors / *num_comp;
  }
  PetscInt restr_size = (*num_cells)*(*cell_size);
  ierr = PetscMalloc1(restr_size, &restr_indices);CHKERRQ(ierr);
  PetscInt cell_offset = 0;

  PetscInt P = (PetscInt) pow(*cell_size, 1.0 / (dim - height));
  for (PetscInt p = 0; p < *num_cells; p++) {
    PetscBool flip = PETSC_FALSE;
    PetscInt c = iter_indices[p];
    PetscInt num_indices, *indices;
    PetscInt field_offsets[17]; // max number of fields plus 1
    ierr = DMPlexGetClosureIndices(dm, section, section, c, PETSC_TRUE, &num_indices, &indices, field_offsets, NULL);CHKERRQ(ierr);
    if (height > 0) {
      PetscInt num_cells_support, num_faces, start = -1;
      const PetscInt *orients, *faces, *cells;
      ierr = DMPlexGetSupport(dm, c, &cells);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, c, &num_cells_support);CHKERRQ(ierr);
      if (num_cells_support != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Expected one cell in support of exterior face, but got %D cells", num_cells_support);
      ierr = DMPlexGetCone(dm, cells[0], &faces);CHKERRQ(ierr);
      ierr = DMPlexGetConeSize(dm, cells[0], &num_faces);CHKERRQ(ierr);
      for (PetscInt i=0; i<num_faces; i++) {if (faces[i] == c) start = i;}
      if (start < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT, "Could not find face %D in cone of its support", c);
      ierr = DMPlexGetConeOrientation(dm, cells[0], &orients);CHKERRQ(ierr);
      if (orients[start] < 0) flip = PETSC_TRUE;
    }

    for (PetscInt i = 0; i < *cell_size; i++) {
      PetscInt ii = i;
      if (flip) {
        if (*cell_size == P) ii = *cell_size - 1 - i;
        else if (*cell_size == P*P) {
          PetscInt row = i / P, col = i % P;
          ii = row + col * P;
        } else SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for flipping point with cell size %D != P (%D) or P^2", *cell_size, P);
      }
      // Essential boundary conditions are encoded as -(loc+1), but we don't care so we decode.
      PetscInt loc = indices[field_offsets[dm_field] + ii*(*num_comp)];
      restr_indices[cell_offset++] = loc >= 0 ? loc : -(loc + 1);
    }
    ierr = DMPlexRestoreClosureIndices(dm, section, section, c, PETSC_TRUE, &num_indices, &indices, field_offsets, NULL);CHKERRQ(ierr);
  }
  if (cell_offset != restr_size) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_SUP, "Shape mismatch, offsets array of shape (%D, %D) initialized for %D nodes", *num_cells, (*cell_size), cell_offset);
  if (iter_is) { ierr = ISRestoreIndices(iter_is, &iter_indices);CHKERRQ(ierr); }
  ierr = ISDestroy(&iter_is);CHKERRQ(ierr);

  *offsets = restr_indices;
  ierr = PetscSectionGetStorageSize(section, l_size);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(ERestrict, 2);
  if (!dm->ceedERestrict) {
    PetscInt     num_cells, cell_size, num_comp, lvec_size, *restr_indices;
    CeedElemRestriction elem_restr;
    Ceed         ceed;

    ierr = DMPlexGetLocalOffsets(dm, domain_label, label_value, height, dm_field, &num_cells, &cell_size, &num_comp, &lvec_size, &restr_indices);CHKERRQ(ierr);

    ierr = DMGetCeed(dm, &ceed);CHKERRQ(ierr);
    ierr = CeedElemRestrictionCreate(ceed, num_cells, cell_size, num_comp, 1, lvec_size, CEED_MEM_HOST, CEED_COPY_VALUES, restr_indices, &elem_restr);CHKERRQ_CEED(ierr);
    ierr = PetscFree(restr_indices);CHKERRQ(ierr);
    dm->ceedERestrict = elem_restr;
  }
  *ERestrict = dm->ceedERestrict;
  PetscFunctionReturn(0);
}

#endif
