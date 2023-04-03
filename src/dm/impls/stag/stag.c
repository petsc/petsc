/*
   Implementation of DMStag, defining dimension-independent functions in the
   DM API. stag1d.c, stag2d.c, and stag3d.c may include dimension-specific
   implementations of DM API functions, and other files here contain additional
   DMStag-specific API functions, as well as internal functions.
*/
#include <petsc/private/dmstagimpl.h>
#include <petscsf.h>

static PetscErrorCode DMCreateFieldDecomposition_Stag(DM dm, PetscInt *len, char ***namelist, IS **islist, DM **dmlist)
{
  PetscInt       f0, f1, f2, f3, dof0, dof1, dof2, dof3, n_entries, k, d, cnt, n_fields, dim;
  DMStagStencil *stencil0, *stencil1, *stencil2, *stencil3;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMStagGetDOF(dm, &dof0, &dof1, &dof2, &dof3));
  PetscCall(DMStagGetEntriesPerElement(dm, &n_entries));

  f0 = 1;
  f1 = f2 = f3 = 0;
  if (dim == 1) {
    f1 = 1;
  } else if (dim == 2) {
    f1 = 2;
    f2 = 1;
  } else if (dim == 3) {
    f1 = 3;
    f2 = 3;
    f3 = 1;
  } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported dimension %" PetscInt_FMT, dim);

  PetscCall(PetscCalloc1(f0 * dof0, &stencil0));
  PetscCall(PetscCalloc1(f1 * dof1, &stencil1));
  if (dim >= 2) PetscCall(PetscCalloc1(f2 * dof2, &stencil2));
  if (dim >= 3) PetscCall(PetscCalloc1(f3 * dof3, &stencil3));
  for (k = 0; k < f0; ++k) {
    for (d = 0; d < dof0; ++d) {
      stencil0[dof0 * k + d].i = 0;
      stencil0[dof0 * k + d].j = 0;
      stencil0[dof0 * k + d].j = 0;
    }
  }
  for (k = 0; k < f1; ++k) {
    for (d = 0; d < dof1; ++d) {
      stencil1[dof1 * k + d].i = 0;
      stencil1[dof1 * k + d].j = 0;
      stencil1[dof1 * k + d].j = 0;
    }
  }
  if (dim >= 2) {
    for (k = 0; k < f2; ++k) {
      for (d = 0; d < dof2; ++d) {
        stencil2[dof2 * k + d].i = 0;
        stencil2[dof2 * k + d].j = 0;
        stencil2[dof2 * k + d].j = 0;
      }
    }
  }
  if (dim >= 3) {
    for (k = 0; k < f3; ++k) {
      for (d = 0; d < dof3; ++d) {
        stencil3[dof3 * k + d].i = 0;
        stencil3[dof3 * k + d].j = 0;
        stencil3[dof3 * k + d].j = 0;
      }
    }
  }

  n_fields = 0;
  if (dof0 != 0) ++n_fields;
  if (dof1 != 0) ++n_fields;
  if (dim >= 2 && dof2 != 0) ++n_fields;
  if (dim >= 3 && dof3 != 0) ++n_fields;
  if (len) *len = n_fields;

  if (islist) {
    PetscCall(PetscMalloc1(n_fields, islist));

    if (dim == 1) {
      /* face, element */
      for (d = 0; d < dof0; ++d) {
        stencil0[d].loc = DMSTAG_LEFT;
        stencil0[d].c   = d;
      }
      for (d = 0; d < dof1; ++d) {
        stencil1[d].loc = DMSTAG_ELEMENT;
        stencil1[d].c   = d;
      }
    } else if (dim == 2) {
      /* vertex, edge(down,left), element */
      for (d = 0; d < dof0; ++d) {
        stencil0[d].loc = DMSTAG_DOWN_LEFT;
        stencil0[d].c   = d;
      }
      /* edge */
      cnt = 0;
      for (d = 0; d < dof1; ++d) {
        stencil1[cnt].loc = DMSTAG_DOWN;
        stencil1[cnt].c   = d;
        ++cnt;
      }
      for (d = 0; d < dof1; ++d) {
        stencil1[cnt].loc = DMSTAG_LEFT;
        stencil1[cnt].c   = d;
        ++cnt;
      }
      /* element */
      for (d = 0; d < dof2; ++d) {
        stencil2[d].loc = DMSTAG_ELEMENT;
        stencil2[d].c   = d;
      }
    } else if (dim == 3) {
      /* vertex, edge(down,left), face(down,left,back), element */
      for (d = 0; d < dof0; ++d) {
        stencil0[d].loc = DMSTAG_BACK_DOWN_LEFT;
        stencil0[d].c   = d;
      }
      /* edges */
      cnt = 0;
      for (d = 0; d < dof1; ++d) {
        stencil1[cnt].loc = DMSTAG_BACK_DOWN;
        stencil1[cnt].c   = d;
        ++cnt;
      }
      for (d = 0; d < dof1; ++d) {
        stencil1[cnt].loc = DMSTAG_BACK_LEFT;
        stencil1[cnt].c   = d;
        ++cnt;
      }
      for (d = 0; d < dof1; ++d) {
        stencil1[cnt].loc = DMSTAG_DOWN_LEFT;
        stencil1[cnt].c   = d;
        ++cnt;
      }
      /* faces */
      cnt = 0;
      for (d = 0; d < dof2; ++d) {
        stencil2[cnt].loc = DMSTAG_BACK;
        stencil2[cnt].c   = d;
        ++cnt;
      }
      for (d = 0; d < dof2; ++d) {
        stencil2[cnt].loc = DMSTAG_DOWN;
        stencil2[cnt].c   = d;
        ++cnt;
      }
      for (d = 0; d < dof2; ++d) {
        stencil2[cnt].loc = DMSTAG_LEFT;
        stencil2[cnt].c   = d;
        ++cnt;
      }
      /* elements */
      for (d = 0; d < dof3; ++d) {
        stencil3[d].loc = DMSTAG_ELEMENT;
        stencil3[d].c   = d;
      }
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported dimension %" PetscInt_FMT, dim);

    cnt = 0;
    if (dof0 != 0) {
      PetscCall(DMStagCreateISFromStencils(dm, f0 * dof0, stencil0, &(*islist)[cnt]));
      ++cnt;
    }
    if (dof1 != 0) {
      PetscCall(DMStagCreateISFromStencils(dm, f1 * dof1, stencil1, &(*islist)[cnt]));
      ++cnt;
    }
    if (dim >= 2 && dof2 != 0) {
      PetscCall(DMStagCreateISFromStencils(dm, f2 * dof2, stencil2, &(*islist)[cnt]));
      ++cnt;
    }
    if (dim >= 3 && dof3 != 0) {
      PetscCall(DMStagCreateISFromStencils(dm, f3 * dof3, stencil3, &(*islist)[cnt]));
      ++cnt;
    }
  }

  if (namelist) {
    PetscCall(PetscMalloc1(n_fields, namelist));
    cnt = 0;
    if (dim == 1) {
      if (dof0 != 0) {
        PetscCall(PetscStrallocpy("vertex", &(*namelist)[cnt]));
        ++cnt;
      }
      if (dof1 != 0) {
        PetscCall(PetscStrallocpy("element", &(*namelist)[cnt]));
        ++cnt;
      }
    } else if (dim == 2) {
      if (dof0 != 0) {
        PetscCall(PetscStrallocpy("vertex", &(*namelist)[cnt]));
        ++cnt;
      }
      if (dof1 != 0) {
        PetscCall(PetscStrallocpy("face", &(*namelist)[cnt]));
        ++cnt;
      }
      if (dof2 != 0) {
        PetscCall(PetscStrallocpy("element", &(*namelist)[cnt]));
        ++cnt;
      }
    } else if (dim == 3) {
      if (dof0 != 0) {
        PetscCall(PetscStrallocpy("vertex", &(*namelist)[cnt]));
        ++cnt;
      }
      if (dof1 != 0) {
        PetscCall(PetscStrallocpy("edge", &(*namelist)[cnt]));
        ++cnt;
      }
      if (dof2 != 0) {
        PetscCall(PetscStrallocpy("face", &(*namelist)[cnt]));
        ++cnt;
      }
      if (dof3 != 0) {
        PetscCall(PetscStrallocpy("element", &(*namelist)[cnt]));
        ++cnt;
      }
    }
  } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported dimension %" PetscInt_FMT, dim);
  if (dmlist) {
    PetscCall(PetscMalloc1(n_fields, dmlist));
    cnt = 0;
    if (dof0 != 0) {
      PetscCall(DMStagCreateCompatibleDMStag(dm, dof0, 0, 0, 0, &(*dmlist)[cnt]));
      ++cnt;
    }
    if (dof1 != 0) {
      PetscCall(DMStagCreateCompatibleDMStag(dm, 0, dof1, 0, 0, &(*dmlist)[cnt]));
      ++cnt;
    }
    if (dim >= 2 && dof2 != 0) {
      PetscCall(DMStagCreateCompatibleDMStag(dm, 0, 0, dof2, 0, &(*dmlist)[cnt]));
      ++cnt;
    }
    if (dim >= 3 && dof3 != 0) {
      PetscCall(DMStagCreateCompatibleDMStag(dm, 0, 0, 0, dof3, &(*dmlist)[cnt]));
      ++cnt;
    }
  }
  PetscCall(PetscFree(stencil0));
  PetscCall(PetscFree(stencil1));
  if (dim >= 2) PetscCall(PetscFree(stencil2));
  if (dim >= 3) PetscCall(PetscFree(stencil3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMClone_Stag(DM dm, DM *newdm)
{
  PetscFunctionBegin;
  /* Destroy the DM created by generic logic in DMClone() */
  if (*newdm) PetscCall(DMDestroy(newdm));
  PetscCall(DMStagDuplicateWithoutSetup(dm, PetscObjectComm((PetscObject)dm), newdm));
  PetscCall(DMSetUp(*newdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCoarsen_Stag(DM dm, MPI_Comm comm, DM *dmc)
{
  const DM_Stag *const stag = (DM_Stag *)dm->data;
  PetscInt             d, dim;

  PetscFunctionBegin;
  PetscCall(DMStagDuplicateWithoutSetup(dm, comm, dmc));
  PetscCall(DMSetOptionsPrefix(*dmc, ((PetscObject)dm)->prefix));
  PetscCall(DMGetDimension(dm, &dim));
  for (d = 0; d < dim; ++d) PetscCheck(stag->N[d] % 2 == 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "coarsening not supported except for even numbers of elements in each dimension ");
  PetscCall(DMStagSetGlobalSizes(*dmc, stag->N[0] / 2, stag->N[1] / 2, stag->N[2] / 2));
  {
    PetscInt *l[DMSTAG_MAX_DIM];
    for (d = 0; d < dim; ++d) {
      PetscInt i;
      PetscCall(PetscMalloc1(stag->nRanks[d], &l[d]));
      for (i = 0; i < stag->nRanks[d]; ++i) {
        PetscCheck(stag->l[d][i] % 2 == 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "coarsening not supported except for an even number of elements in each direction on each rank");
        l[d][i] = stag->l[d][i] / 2; /* Just halve everything */
      }
    }
    PetscCall(DMStagSetOwnershipRanges(*dmc, l[0], l[1], l[2]));
    for (d = 0; d < dim; ++d) PetscCall(PetscFree(l[d]));
  }
  PetscCall(DMSetUp(*dmc));

  if (dm->coordinates[0].dm) { /* Note that with product coordinates, dm->coordinates = NULL, so we check the DM */
    DM        coordinate_dm, coordinate_dmc;
    PetscBool isstag, isprod;

    PetscCall(DMGetCoordinateDM(dm, &coordinate_dm));
    PetscCall(PetscObjectTypeCompare((PetscObject)coordinate_dm, DMSTAG, &isstag));
    PetscCall(PetscObjectTypeCompare((PetscObject)coordinate_dm, DMPRODUCT, &isprod));
    if (isstag) {
      PetscCall(DMStagSetUniformCoordinatesExplicit(*dmc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)); /* Coordinates will be overwritten */
      PetscCall(DMGetCoordinateDM(*dmc, &coordinate_dmc));
      PetscCall(DMStagRestrictSimple(coordinate_dm, dm->coordinates[0].x, coordinate_dmc, (*dmc)->coordinates[0].x));
    } else if (isprod) {
      PetscCall(DMStagSetUniformCoordinatesProduct(*dmc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)); /* Coordinates will be overwritten */
      PetscCall(DMGetCoordinateDM(*dmc, &coordinate_dmc));
      for (d = 0; d < dim; ++d) {
        DM subdm_coarse, subdm_coord_coarse, subdm_fine, subdm_coord_fine;

        PetscCall(DMProductGetDM(coordinate_dm, d, &subdm_fine));
        PetscCall(DMGetCoordinateDM(subdm_fine, &subdm_coord_fine));
        PetscCall(DMProductGetDM(coordinate_dmc, d, &subdm_coarse));
        PetscCall(DMGetCoordinateDM(subdm_coarse, &subdm_coord_coarse));
        PetscCall(DMStagRestrictSimple(subdm_coord_fine, subdm_fine->coordinates[0].xl, subdm_coord_coarse, subdm_coarse->coordinates[0].xl));
      }
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unknown coordinate DM type");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMRefine_Stag(DM dm, MPI_Comm comm, DM *dmc)
{
  const DM_Stag *const stag = (DM_Stag *)dm->data;

  PetscFunctionBegin;
  PetscCall(DMStagDuplicateWithoutSetup(dm, comm, dmc));
  PetscCall(DMSetOptionsPrefix(*dmc, ((PetscObject)dm)->prefix));
  PetscCall(DMStagSetGlobalSizes(*dmc, stag->N[0] * 2, stag->N[1] * 2, stag->N[2] * 2));
  {
    PetscInt  dim, d;
    PetscInt *l[DMSTAG_MAX_DIM];
    PetscCall(DMGetDimension(dm, &dim));
    for (d = 0; d < dim; ++d) {
      PetscInt i;
      PetscCall(PetscMalloc1(stag->nRanks[d], &l[d]));
      for (i = 0; i < stag->nRanks[d]; ++i) { l[d][i] = stag->l[d][i] * 2; /* Just double everything */ }
    }
    PetscCall(DMStagSetOwnershipRanges(*dmc, l[0], l[1], l[2]));
    for (d = 0; d < dim; ++d) PetscCall(PetscFree(l[d]));
  }
  PetscCall(DMSetUp(*dmc));
  /* Note: For now, we do not refine coordinates */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMDestroy_Stag(DM dm)
{
  DM_Stag *stag;
  PetscInt i;

  PetscFunctionBegin;
  stag = (DM_Stag *)dm->data;
  for (i = 0; i < DMSTAG_MAX_DIM; ++i) PetscCall(PetscFree(stag->l[i]));
  PetscCall(VecScatterDestroy(&stag->gtol));
  PetscCall(VecScatterDestroy(&stag->ltog_injective));
  PetscCall(PetscFree(stag->neighbors));
  PetscCall(PetscFree(stag->locationOffsets));
  PetscCall(PetscFree(stag->coordinateDMType));
  PetscCall(PetscFree(stag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCreateGlobalVector_Stag(DM dm, Vec *vec)
{
  DM_Stag *const stag = (DM_Stag *)dm->data;

  PetscFunctionBegin;
  PetscCheck(dm->setupcalled, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "This function must be called after DMSetUp()");
  PetscCall(VecCreate(PetscObjectComm((PetscObject)dm), vec));
  PetscCall(VecSetSizes(*vec, stag->entries, PETSC_DETERMINE));
  PetscCall(VecSetType(*vec, dm->vectype));
  PetscCall(VecSetDM(*vec, dm));
  /* Could set some ops, as DMDA does */
  PetscCall(VecSetLocalToGlobalMapping(*vec, dm->ltogmap));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCreateLocalVector_Stag(DM dm, Vec *vec)
{
  DM_Stag *const stag = (DM_Stag *)dm->data;

  PetscFunctionBegin;
  PetscCheck(dm->setupcalled, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "This function must be called after DMSetUp()");
  PetscCall(VecCreate(PETSC_COMM_SELF, vec));
  PetscCall(VecSetSizes(*vec, stag->entriesGhost, PETSC_DETERMINE));
  PetscCall(VecSetType(*vec, dm->vectype));
  PetscCall(VecSetBlockSize(*vec, stag->entriesPerElement));
  PetscCall(VecSetDM(*vec, dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Helper function to check for the limited situations for which interpolation
   and restriction functions are implemented */
static PetscErrorCode CheckTransferOperatorRequirements_Private(DM dmc, DM dmf)
{
  PetscInt dim, stencilWidthc, stencilWidthf, nf[DMSTAG_MAX_DIM], nc[DMSTAG_MAX_DIM], doff[DMSTAG_MAX_STRATA], dofc[DMSTAG_MAX_STRATA];

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dmc, &dim));
  PetscCall(DMStagGetStencilWidth(dmc, &stencilWidthc));
  PetscCheck(stencilWidthc >= 1, PetscObjectComm((PetscObject)dmc), PETSC_ERR_SUP, "DMCreateRestriction not implemented for coarse grid stencil width < 1");
  PetscCall(DMStagGetStencilWidth(dmf, &stencilWidthf));
  PetscCheck(stencilWidthf >= 1, PetscObjectComm((PetscObject)dmf), PETSC_ERR_SUP, "DMCreateRestriction not implemented for fine grid stencil width < 1");
  PetscCall(DMStagGetLocalSizes(dmf, &nf[0], &nf[1], &nf[2]));
  PetscCall(DMStagGetLocalSizes(dmc, &nc[0], &nc[1], &nc[2]));
  for (PetscInt d = 0; d < dim; ++d)
    PetscCheck(nf[d] == 2 * nc[d], PetscObjectComm((PetscObject)dmc), PETSC_ERR_SUP, "No support for fine to coarse ratio other than 2 (it is %" PetscInt_FMT " to %" PetscInt_FMT " in dimension %" PetscInt_FMT ")", nf[d], nc[d], d);
  PetscCall(DMStagGetDOF(dmc, &dofc[0], &dofc[1], &dofc[2], &dofc[3]));
  PetscCall(DMStagGetDOF(dmf, &doff[0], &doff[1], &doff[2], &doff[3]));
  for (PetscInt d = 0; d < dim + 1; ++d)
    PetscCheck(dofc[d] == doff[d], PetscObjectComm((PetscObject)dmc), PETSC_ERR_SUP, "No support for different numbers of dof per stratum between coarse and fine DMStag objects: dof%" PetscInt_FMT " is %" PetscInt_FMT " (fine) but %" PetscInt_FMT "(coarse))", d, doff[d], dofc[d]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Since the interpolation uses MATMAIJ for dof > 0 we convert requests for non-MATAIJ baseded matrices to MATAIJ.
   This is a bit of a hack; the reason for it is partially because -dm_mat_type defines the
   matrix type for both the operator matrices and the interpolation matrices so that users
   can select matrix types of base MATAIJ for accelerators

   Note: The ConvertToAIJ() code below *has been copied from dainterp.c*! ConvertToAIJ() should perhaps be placed somewhere
   in mat/utils to avoid code duplication, but then the DMStag and DMDA code would need to include the private Mat headers.
   Since it is only used in two places, I have simply duplicated the code to avoid the need to exposure the private
   Mat routines in parts of DM. If we find a need for ConvertToAIJ() elsewhere, then we should consolidate it to one
   place in mat/utils.
*/
static PetscErrorCode ConvertToAIJ(MatType intype, MatType *outtype)
{
  PetscInt    i;
  char const *types[3] = {MATAIJ, MATSEQAIJ, MATMPIAIJ};
  PetscBool   flg;

  PetscFunctionBegin;
  *outtype = MATAIJ;
  for (i = 0; i < 3; i++) {
    PetscCall(PetscStrbeginswith(intype, types[i], &flg));
    if (flg) {
      *outtype = intype;
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCreateInterpolation_Stag(DM dmc, DM dmf, Mat *A, Vec *vec)
{
  PetscInt               dim, entriesf, entriesc, doff[DMSTAG_MAX_STRATA];
  ISLocalToGlobalMapping ltogmf, ltogmc;
  MatType                mattype;

  PetscFunctionBegin;
  PetscCall(CheckTransferOperatorRequirements_Private(dmc, dmf));

  PetscCall(DMStagGetEntries(dmf, &entriesf));
  PetscCall(DMStagGetEntries(dmc, &entriesc));
  PetscCall(DMGetLocalToGlobalMapping(dmf, &ltogmf));
  PetscCall(DMGetLocalToGlobalMapping(dmc, &ltogmc));

  PetscCall(MatCreate(PetscObjectComm((PetscObject)dmc), A));
  PetscCall(MatSetSizes(*A, entriesf, entriesc, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(ConvertToAIJ(dmc->mattype, &mattype));
  PetscCall(MatSetType(*A, mattype));
  PetscCall(MatSetLocalToGlobalMapping(*A, ltogmf, ltogmc));

  PetscCall(DMGetDimension(dmc, &dim));
  PetscCall(DMStagGetDOF(dmf, &doff[0], &doff[1], &doff[2], &doff[3]));
  if (dim == 1) {
    PetscCall(DMStagPopulateInterpolation1d_a_b_Private(dmc, dmf, *A));
  } else if (dim == 2) {
    if (doff[0] == 0) {
      PetscCall(DMStagPopulateInterpolation2d_0_a_b_Private(dmc, dmf, *A));
    } else
      SETERRQ(PetscObjectComm((PetscObject)dmc), PETSC_ERR_SUP, "No default interpolation available between 2d DMStag objects with %" PetscInt_FMT " dof/vertex, %" PetscInt_FMT " dof/face and %" PetscInt_FMT " dof/element", doff[0], doff[1], doff[2]);
  } else if (dim == 3) {
    if (doff[0] == 0 && doff[1] == 0) {
      PetscCall(DMStagPopulateInterpolation3d_0_0_a_b_Private(dmc, dmf, *A));
    } else
      SETERRQ(PetscObjectComm((PetscObject)dmc), PETSC_ERR_SUP, "No default interpolation available between 3d DMStag objects with %" PetscInt_FMT " dof/vertex, %" PetscInt_FMT " dof/edge, %" PetscInt_FMT " dof/face and %" PetscInt_FMT " dof/element", doff[0], doff[1], doff[2], doff[3]);
  } else SETERRQ(PetscObjectComm((PetscObject)dmc), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dimension %" PetscInt_FMT, dim);
  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));

  if (vec) *vec = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCreateRestriction_Stag(DM dmc, DM dmf, Mat *A)
{
  PetscInt               dim, entriesf, entriesc, doff[DMSTAG_MAX_STRATA];
  ISLocalToGlobalMapping ltogmf, ltogmc;
  MatType                mattype;

  PetscFunctionBegin;
  PetscCall(CheckTransferOperatorRequirements_Private(dmc, dmf));

  PetscCall(DMStagGetEntries(dmf, &entriesf));
  PetscCall(DMStagGetEntries(dmc, &entriesc));
  PetscCall(DMGetLocalToGlobalMapping(dmf, &ltogmf));
  PetscCall(DMGetLocalToGlobalMapping(dmc, &ltogmc));

  PetscCall(MatCreate(PetscObjectComm((PetscObject)dmc), A));
  PetscCall(MatSetSizes(*A, entriesc, entriesf, PETSC_DECIDE, PETSC_DECIDE)); /* Note transpose wrt interpolation */
  PetscCall(ConvertToAIJ(dmc->mattype, &mattype));
  PetscCall(MatSetType(*A, mattype));
  PetscCall(MatSetLocalToGlobalMapping(*A, ltogmc, ltogmf)); /* Note transpose wrt interpolation */

  PetscCall(DMGetDimension(dmc, &dim));
  PetscCall(DMStagGetDOF(dmf, &doff[0], &doff[1], &doff[2], &doff[3]));
  if (dim == 1) {
    PetscCall(DMStagPopulateRestriction1d_a_b_Private(dmc, dmf, *A));
  } else if (dim == 2) {
    if (doff[0] == 0) {
      PetscCall(DMStagPopulateRestriction2d_0_a_b_Private(dmc, dmf, *A));
    } else
      SETERRQ(PetscObjectComm((PetscObject)dmc), PETSC_ERR_SUP, "No default restriction available between 2d DMStag objects with %" PetscInt_FMT " dof/vertex, %" PetscInt_FMT " dof/face and %" PetscInt_FMT " dof/element", doff[0], doff[1], doff[2]);
  } else if (dim == 3) {
    if (doff[0] == 0 && doff[0] == 0) {
      PetscCall(DMStagPopulateRestriction3d_0_0_a_b_Private(dmc, dmf, *A));
    } else
      SETERRQ(PetscObjectComm((PetscObject)dmc), PETSC_ERR_SUP, "No default restriction available between 3d DMStag objects with %" PetscInt_FMT " dof/vertex, %" PetscInt_FMT " dof/edge, %" PetscInt_FMT " dof/face and %" PetscInt_FMT " dof/element", doff[0], doff[1], doff[2], doff[3]);
  } else SETERRQ(PetscObjectComm((PetscObject)dmc), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dimension %" PetscInt_FMT, dim);

  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCreateMatrix_Stag(DM dm, Mat *mat)
{
  MatType                mat_type;
  PetscBool              is_shell, is_aij;
  PetscInt               dim, entries;
  ISLocalToGlobalMapping ltogmap;

  PetscFunctionBegin;
  PetscCheck(dm->setupcalled, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "This function must be called after DMSetUp()");
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetMatType(dm, &mat_type));
  PetscCall(DMStagGetEntries(dm, &entries));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)dm), mat));
  PetscCall(MatSetSizes(*mat, entries, entries, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(*mat, mat_type));
  PetscCall(MatSetUp(*mat));
  PetscCall(DMGetLocalToGlobalMapping(dm, &ltogmap));
  PetscCall(MatSetLocalToGlobalMapping(*mat, ltogmap, ltogmap));
  PetscCall(MatSetDM(*mat, dm));

  /* Compare to similar and perhaps superior logic in DMCreateMatrix_DA, which creates
     the matrix first and then performs this logic by checking for preallocation functions */
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)*mat, MATAIJ, &is_aij));
  if (!is_aij) PetscCall(PetscObjectBaseTypeCompare((PetscObject)*mat, MATSEQAIJ, &is_aij));
  if (!is_aij) PetscCall(PetscObjectBaseTypeCompare((PetscObject)*mat, MATMPIAIJ, &is_aij));
  PetscCall(PetscStrcmp(mat_type, MATSHELL, &is_shell));
  if (is_aij) {
    Mat             preallocator;
    PetscInt        m, n;
    const PetscBool fill_with_zeros = PETSC_FALSE;

    PetscCall(MatCreate(PetscObjectComm((PetscObject)dm), &preallocator));
    PetscCall(MatSetType(preallocator, MATPREALLOCATOR));
    PetscCall(MatGetLocalSize(*mat, &m, &n));
    PetscCall(MatSetSizes(preallocator, m, n, PETSC_DECIDE, PETSC_DECIDE));
    PetscCall(MatSetLocalToGlobalMapping(preallocator, ltogmap, ltogmap));
    PetscCall(MatSetUp(preallocator));
    switch (dim) {
    case 1:
      PetscCall(DMCreateMatrix_Stag_1D_AIJ_Assemble(dm, preallocator));
      break;
    case 2:
      PetscCall(DMCreateMatrix_Stag_2D_AIJ_Assemble(dm, preallocator));
      break;
    case 3:
      PetscCall(DMCreateMatrix_Stag_3D_AIJ_Assemble(dm, preallocator));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dimension %" PetscInt_FMT, dim);
    }
    PetscCall(MatPreallocatorPreallocate(preallocator, fill_with_zeros, *mat));
    PetscCall(MatDestroy(&preallocator));

    if (!dm->prealloc_only) {
      /* Bind to CPU before assembly, to prevent unnecessary copies of zero entries from CPU to GPU */
      PetscCall(MatBindToCPU(*mat, PETSC_TRUE));
      switch (dim) {
      case 1:
        PetscCall(DMCreateMatrix_Stag_1D_AIJ_Assemble(dm, *mat));
        break;
      case 2:
        PetscCall(DMCreateMatrix_Stag_2D_AIJ_Assemble(dm, *mat));
        break;
      case 3:
        PetscCall(DMCreateMatrix_Stag_3D_AIJ_Assemble(dm, *mat));
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dimension %" PetscInt_FMT, dim);
      }
      PetscCall(MatBindToCPU(*mat, PETSC_FALSE));
    }
  } else if (is_shell) {
    /* nothing more to do */
  } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Not implemented for Mattype %s", mat_type);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMGetCompatibility_Stag(DM dm, DM dm2, PetscBool *compatible, PetscBool *set)
{
  const DM_Stag *const stag  = (DM_Stag *)dm->data;
  const DM_Stag *const stag2 = (DM_Stag *)dm2->data;
  PetscInt             dim, dim2, i;
  MPI_Comm             comm;
  PetscMPIInt          sameComm;
  DMType               type2;
  PetscBool            sameType;

  PetscFunctionBegin;
  PetscCall(DMGetType(dm2, &type2));
  PetscCall(PetscStrcmp(DMSTAG, type2, &sameType));
  if (!sameType) {
    PetscCall(PetscInfo((PetscObject)dm, "DMStag compatibility check not implemented with DM of type %s\n", type2));
    *set = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_compare(comm, PetscObjectComm((PetscObject)dm2), &sameComm));
  if (sameComm != MPI_IDENT) {
    PetscCall(PetscInfo((PetscObject)dm, "DMStag objects have different communicators: %" PETSC_INTPTR_T_FMT " != %" PETSC_INTPTR_T_FMT "\n", (PETSC_INTPTR_T)comm, (PETSC_INTPTR_T)PetscObjectComm((PetscObject)dm2)));
    *set = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetDimension(dm2, &dim2));
  if (dim != dim2) {
    PetscCall(PetscInfo((PetscObject)dm, "DMStag objects have different dimensions"));
    *set        = PETSC_TRUE;
    *compatible = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  for (i = 0; i < dim; ++i) {
    if (stag->N[i] != stag2->N[i]) {
      PetscCall(PetscInfo((PetscObject)dm, "DMStag objects have different global numbers of elements in dimension %" PetscInt_FMT ": %" PetscInt_FMT " != %" PetscInt_FMT "\n", i, stag->n[i], stag2->n[i]));
      *set        = PETSC_TRUE;
      *compatible = PETSC_FALSE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    if (stag->n[i] != stag2->n[i]) {
      PetscCall(PetscInfo((PetscObject)dm, "DMStag objects have different local numbers of elements in dimension %" PetscInt_FMT ": %" PetscInt_FMT " != %" PetscInt_FMT "\n", i, stag->n[i], stag2->n[i]));
      *set        = PETSC_TRUE;
      *compatible = PETSC_FALSE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    if (stag->boundaryType[i] != stag2->boundaryType[i]) {
      PetscCall(PetscInfo((PetscObject)dm, "DMStag objects have different boundary types in dimension %" PetscInt_FMT ": %s != %s\n", i, DMBoundaryTypes[stag->boundaryType[i]], DMBoundaryTypes[stag2->boundaryType[i]]));
      *set        = PETSC_TRUE;
      *compatible = PETSC_FALSE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  /* Note: we include stencil type and width in the notion of compatibility, as this affects
     the "atlas" (local subdomains). This might be irritating in legitimate cases
     of wanting to transfer between two other-wise compatible DMs with different
     stencil characteristics. */
  if (stag->stencilType != stag2->stencilType) {
    PetscCall(PetscInfo((PetscObject)dm, "DMStag objects have different ghost stencil types: %s != %s\n", DMStagStencilTypes[stag->stencilType], DMStagStencilTypes[stag2->stencilType]));
    *set        = PETSC_TRUE;
    *compatible = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (stag->stencilWidth != stag2->stencilWidth) {
    PetscCall(PetscInfo((PetscObject)dm, "DMStag objects have different ghost stencil widths: %" PetscInt_FMT " != %" PetscInt_FMT "\n", stag->stencilWidth, stag->stencilWidth));
    *set        = PETSC_TRUE;
    *compatible = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  *set        = PETSC_TRUE;
  *compatible = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMHasCreateInjection_Stag(DM dm, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidBoolPointer(flg, 2);
  *flg = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
Note there are several orderings in play here.
In all cases, non-element dof are associated with the element that they are below/left/behind, and the order in 2D proceeds vertex/bottom edge/left edge/element (with all dof on each together).
Also in all cases, only subdomains which are the last in their dimension have partial elements.

1) "Natural" Ordering (not used). Number adding each full or partial (on the right or top) element, starting at the bottom left (i=0,j=0) and proceeding across the entire domain, row by row to get a global numbering.
2) Global ("PETSc") ordering. The same as natural, but restricted to each domain. So, traverse all elements (again starting at the bottom left and going row-by-row) on rank 0, then continue numbering with rank 1, and so on.
3) Local ordering. Including ghost elements (both interior and on the right/top/front to complete partial elements), use the same convention to create a local numbering.
*/

static PetscErrorCode DMLocalToGlobalBegin_Stag(DM dm, Vec l, InsertMode mode, Vec g)
{
  DM_Stag *const stag = (DM_Stag *)dm->data;

  PetscFunctionBegin;
  if (mode == ADD_VALUES) {
    PetscCall(VecScatterBegin(stag->gtol, l, g, mode, SCATTER_REVERSE));
  } else if (mode == INSERT_VALUES) {
    if (stag->ltog_injective) {
      PetscCall(VecScatterBegin(stag->ltog_injective, l, g, mode, SCATTER_FORWARD));
    } else {
      PetscCall(VecScatterBegin(stag->gtol, l, g, mode, SCATTER_REVERSE_LOCAL));
    }
  } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported InsertMode");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMLocalToGlobalEnd_Stag(DM dm, Vec l, InsertMode mode, Vec g)
{
  DM_Stag *const stag = (DM_Stag *)dm->data;

  PetscFunctionBegin;
  if (mode == ADD_VALUES) {
    PetscCall(VecScatterEnd(stag->gtol, l, g, mode, SCATTER_REVERSE));
  } else if (mode == INSERT_VALUES) {
    if (stag->ltog_injective) {
      PetscCall(VecScatterEnd(stag->ltog_injective, l, g, mode, SCATTER_FORWARD));
    } else {
      PetscCall(VecScatterEnd(stag->gtol, l, g, mode, SCATTER_REVERSE_LOCAL));
    }
  } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported InsertMode");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMGlobalToLocalBegin_Stag(DM dm, Vec g, InsertMode mode, Vec l)
{
  DM_Stag *const stag = (DM_Stag *)dm->data;

  PetscFunctionBegin;
  PetscCall(VecScatterBegin(stag->gtol, g, l, mode, SCATTER_FORWARD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMGlobalToLocalEnd_Stag(DM dm, Vec g, InsertMode mode, Vec l)
{
  DM_Stag *const stag = (DM_Stag *)dm->data;

  PetscFunctionBegin;
  PetscCall(VecScatterEnd(stag->gtol, g, l, mode, SCATTER_FORWARD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
If a stratum is active (non-zero dof), make it active in the coordinate DM.
*/
static PetscErrorCode DMCreateCoordinateDM_Stag(DM dm, DM *dmc)
{
  DM_Stag *const stag = (DM_Stag *)dm->data;
  PetscInt       dim;
  PetscBool      isstag, isproduct;
  const char    *prefix;

  PetscFunctionBegin;

  PetscCheck(stag->coordinateDMType, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Before creating a coordinate DM, a type must be specified with DMStagSetCoordinateDMType()");

  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscStrcmp(stag->coordinateDMType, DMSTAG, &isstag));
  PetscCall(PetscStrcmp(stag->coordinateDMType, DMPRODUCT, &isproduct));
  if (isstag) {
    PetscCall(DMStagCreateCompatibleDMStag(dm, stag->dof[0] > 0 ? dim : 0, stag->dof[1] > 0 ? dim : 0, stag->dof[2] > 0 ? dim : 0, stag->dof[3] > 0 ? dim : 0, dmc));
  } else if (isproduct) {
    PetscCall(DMCreate(PETSC_COMM_WORLD, dmc));
    PetscCall(DMSetType(*dmc, DMPRODUCT));
    PetscCall(DMSetDimension(*dmc, dim));
  } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported coordinate DM type %s", stag->coordinateDMType);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dm, &prefix));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*dmc, prefix));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)*dmc, "cdm_"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMGetNeighbors_Stag(DM dm, PetscInt *nRanks, const PetscMPIInt *ranks[])
{
  DM_Stag *const stag = (DM_Stag *)dm->data;
  PetscInt       dim;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  switch (dim) {
  case 1:
    *nRanks = 3;
    break;
  case 2:
    *nRanks = 9;
    break;
  case 3:
    *nRanks = 27;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Get neighbors not implemented for dim = %" PetscInt_FMT, dim);
  }
  *ranks = stag->neighbors;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMView_Stag(DM dm, PetscViewer viewer)
{
  DM_Stag *const stag = (DM_Stag *)dm->data;
  PetscBool      isascii, viewAllRanks;
  PetscMPIInt    rank, size;
  PetscInt       dim, maxRanksToView, i;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Dimension: %" PetscInt_FMT "\n", dim));
    switch (dim) {
    case 1:
      PetscCall(PetscViewerASCIIPrintf(viewer, "Global size: %" PetscInt_FMT "\n", stag->N[0]));
      break;
    case 2:
      PetscCall(PetscViewerASCIIPrintf(viewer, "Global sizes: %" PetscInt_FMT " x %" PetscInt_FMT "\n", stag->N[0], stag->N[1]));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Parallel decomposition: %" PetscInt_FMT " x %" PetscInt_FMT " ranks\n", stag->nRanks[0], stag->nRanks[1]));
      break;
    case 3:
      PetscCall(PetscViewerASCIIPrintf(viewer, "Global sizes: %" PetscInt_FMT " x %" PetscInt_FMT " x %" PetscInt_FMT "\n", stag->N[0], stag->N[1], stag->N[2]));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Parallel decomposition: %" PetscInt_FMT " x %" PetscInt_FMT " x %" PetscInt_FMT " ranks\n", stag->nRanks[0], stag->nRanks[1], stag->nRanks[2]));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "not implemented for dim==%" PetscInt_FMT, dim);
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, "Boundary ghosting:"));
    for (i = 0; i < dim; ++i) PetscCall(PetscViewerASCIIPrintf(viewer, " %s", DMBoundaryTypes[stag->boundaryType[i]]));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Elementwise ghost stencil: %s", DMStagStencilTypes[stag->stencilType]));
    if (stag->stencilType != DMSTAG_STENCIL_NONE) {
      PetscCall(PetscViewerASCIIPrintf(viewer, ", width %" PetscInt_FMT "\n", stag->stencilWidth));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT " DOF per vertex (0D)\n", stag->dof[0]));
    if (dim == 3) PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT " DOF per edge (1D)\n", stag->dof[1]));
    if (dim > 1) PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT " DOF per face (%" PetscInt_FMT "D)\n", stag->dof[dim - 1], dim - 1));
    PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT " DOF per element (%" PetscInt_FMT "D)\n", stag->dof[dim], dim));
    if (dm->coordinates[0].dm) PetscCall(PetscViewerASCIIPrintf(viewer, "Has coordinate DM\n"));
    maxRanksToView = 16;
    viewAllRanks   = (PetscBool)(size <= maxRanksToView);
    if (viewAllRanks) {
      PetscCall(PetscViewerASCIIPushSynchronized(viewer));
      switch (dim) {
      case 1:
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Local elements : %" PetscInt_FMT " (%" PetscInt_FMT " with ghosts)\n", rank, stag->n[0], stag->nGhost[0]));
        break;
      case 2:
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Rank coordinates (%d,%d)\n", rank, stag->rank[0], stag->rank[1]));
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Local elements : %" PetscInt_FMT " x %" PetscInt_FMT " (%" PetscInt_FMT " x %" PetscInt_FMT " with ghosts)\n", rank, stag->n[0], stag->n[1], stag->nGhost[0], stag->nGhost[1]));
        break;
      case 3:
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Rank coordinates (%d,%d,%d)\n", rank, stag->rank[0], stag->rank[1], stag->rank[2]));
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Local elements : %" PetscInt_FMT " x %" PetscInt_FMT " x %" PetscInt_FMT " (%" PetscInt_FMT " x %" PetscInt_FMT " x %" PetscInt_FMT " with ghosts)\n", rank, stag->n[0], stag->n[1],
                                                     stag->n[2], stag->nGhost[0], stag->nGhost[1], stag->nGhost[2]));
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "not implemented for dim==%" PetscInt_FMT, dim);
      }
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Local native entries: %" PetscInt_FMT "\n", rank, stag->entries));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Local entries total : %" PetscInt_FMT "\n", rank, stag->entriesGhost));
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerASCIIPopSynchronized(viewer));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "(Per-rank information omitted since >%" PetscInt_FMT " ranks used)\n", maxRanksToView));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSetFromOptions_Stag(DM dm, PetscOptionItems *PetscOptionsObject)
{
  DM_Stag *const stag = (DM_Stag *)dm->data;
  PetscInt       dim;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscOptionsHeadBegin(PetscOptionsObject, "DMStag Options");
  PetscCall(PetscOptionsInt("-stag_grid_x", "Number of grid points in x direction", "DMStagSetGlobalSizes", stag->N[0], &stag->N[0], NULL));
  if (dim > 1) PetscCall(PetscOptionsInt("-stag_grid_y", "Number of grid points in y direction", "DMStagSetGlobalSizes", stag->N[1], &stag->N[1], NULL));
  if (dim > 2) PetscCall(PetscOptionsInt("-stag_grid_z", "Number of grid points in z direction", "DMStagSetGlobalSizes", stag->N[2], &stag->N[2], NULL));
  PetscCall(PetscOptionsInt("-stag_ranks_x", "Number of ranks in x direction", "DMStagSetNumRanks", stag->nRanks[0], &stag->nRanks[0], NULL));
  if (dim > 1) PetscCall(PetscOptionsInt("-stag_ranks_y", "Number of ranks in y direction", "DMStagSetNumRanks", stag->nRanks[1], &stag->nRanks[1], NULL));
  if (dim > 2) PetscCall(PetscOptionsInt("-stag_ranks_z", "Number of ranks in z direction", "DMStagSetNumRanks", stag->nRanks[2], &stag->nRanks[2], NULL));
  PetscCall(PetscOptionsInt("-stag_stencil_width", "Elementwise stencil width", "DMStagSetStencilWidth", stag->stencilWidth, &stag->stencilWidth, NULL));
  PetscCall(PetscOptionsEnum("-stag_stencil_type", "Elementwise stencil stype", "DMStagSetStencilType", DMStagStencilTypes, (PetscEnum)stag->stencilType, (PetscEnum *)&stag->stencilType, NULL));
  PetscCall(PetscOptionsEnum("-stag_boundary_type_x", "Treatment of (physical) boundaries in x direction", "DMStagSetBoundaryTypes", DMBoundaryTypes, (PetscEnum)stag->boundaryType[0], (PetscEnum *)&stag->boundaryType[0], NULL));
  PetscCall(PetscOptionsEnum("-stag_boundary_type_y", "Treatment of (physical) boundaries in y direction", "DMStagSetBoundaryTypes", DMBoundaryTypes, (PetscEnum)stag->boundaryType[1], (PetscEnum *)&stag->boundaryType[1], NULL));
  PetscCall(PetscOptionsEnum("-stag_boundary_type_z", "Treatment of (physical) boundaries in z direction", "DMStagSetBoundaryTypes", DMBoundaryTypes, (PetscEnum)stag->boundaryType[2], (PetscEnum *)&stag->boundaryType[2], NULL));
  PetscCall(PetscOptionsInt("-stag_dof_0", "Number of dof per 0-cell (vertex)", "DMStagSetDOF", stag->dof[0], &stag->dof[0], NULL));
  PetscCall(PetscOptionsInt("-stag_dof_1", "Number of dof per 1-cell (element in 1D, face in 2D, edge in 3D)", "DMStagSetDOF", stag->dof[1], &stag->dof[1], NULL));
  PetscCall(PetscOptionsInt("-stag_dof_2", "Number of dof per 2-cell (element in 2D, face in 3D)", "DMStagSetDOF", stag->dof[2], &stag->dof[2], NULL));
  PetscCall(PetscOptionsInt("-stag_dof_3", "Number of dof per 3-cell (element in 3D)", "DMStagSetDOF", stag->dof[3], &stag->dof[3], NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  DMSTAG - `"stag"` - A `DM` object representing a "staggered grid" or a structured cell complex.

  Level: beginner

  Notes:
  This implementation parallels the `DMDA` implementation in many ways, but allows degrees of freedom
  to be associated with all "strata" in a logically-rectangular grid.

  Each stratum can be characterized by the dimension of the entities ("points", to borrow the `DMPLEX`
  terminology), from 0- to 3-dimensional.

  In some cases this numbering is used directly, for example with `DMStagGetDOF()`.
  To allow easier reading and to some extent more similar code between different-dimensional implementations
  of the same problem, we associate canonical names for each type of point, for each dimension of DMStag.

  * 1-dimensional `DMSTAG` objects have vertices (0D) and elements (1D).
  * 2-dimensional `DMSTAG` objects have vertices (0D), faces (1D), and elements (2D).
  * 3-dimensional `DMSTAG` objects have vertices (0D), edges (1D), faces (2D), and elements (3D).

  This naming is reflected when viewing a `DMSTAG` object with `DMView()`, and in forming
  convenient options prefixes when creating a decomposition with `DMCreateFieldDecomposition()`.

.seealso: [](chapter_stag), `DM`, `DMPRODUCT`, `DMDA`, `DMPLEX`, `DMStagCreate1d()`, `DMStagCreate2d()`, `DMStagCreate3d()`, `DMType`, `DMCreate()`,
          `DMSetType()`, `DMStagVecSplitToDMDA()`
M*/

PETSC_EXTERN PetscErrorCode DMCreate_Stag(DM dm)
{
  DM_Stag *stag;
  PetscInt i, dim;

  PetscFunctionBegin;
  PetscValidPointer(dm, 1);
  PetscCall(PetscNew(&stag));
  dm->data = stag;

  stag->gtol           = NULL;
  stag->ltog_injective = NULL;
  for (i = 0; i < DMSTAG_MAX_STRATA; ++i) stag->dof[i] = 0;
  for (i = 0; i < DMSTAG_MAX_DIM; ++i) stag->l[i] = NULL;
  stag->stencilType  = DMSTAG_STENCIL_NONE;
  stag->stencilWidth = 0;
  for (i = 0; i < DMSTAG_MAX_DIM; ++i) stag->nRanks[i] = -1;
  stag->coordinateDMType = NULL;

  PetscCall(DMGetDimension(dm, &dim));
  PetscCheck(dim == 1 || dim == 2 || dim == 3, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DMSetDimension() must be called to set a dimension with value 1, 2, or 3");

  PetscCall(PetscMemzero(dm->ops, sizeof(*(dm->ops))));
  dm->ops->createcoordinatedm  = DMCreateCoordinateDM_Stag;
  dm->ops->createglobalvector  = DMCreateGlobalVector_Stag;
  dm->ops->createlocalvector   = DMCreateLocalVector_Stag;
  dm->ops->creatematrix        = DMCreateMatrix_Stag;
  dm->ops->hascreateinjection  = DMHasCreateInjection_Stag;
  dm->ops->refine              = DMRefine_Stag;
  dm->ops->coarsen             = DMCoarsen_Stag;
  dm->ops->createinterpolation = DMCreateInterpolation_Stag;
  dm->ops->createrestriction   = DMCreateRestriction_Stag;
  dm->ops->destroy             = DMDestroy_Stag;
  dm->ops->getneighbors        = DMGetNeighbors_Stag;
  dm->ops->globaltolocalbegin  = DMGlobalToLocalBegin_Stag;
  dm->ops->globaltolocalend    = DMGlobalToLocalEnd_Stag;
  dm->ops->localtoglobalbegin  = DMLocalToGlobalBegin_Stag;
  dm->ops->localtoglobalend    = DMLocalToGlobalEnd_Stag;
  dm->ops->setfromoptions      = DMSetFromOptions_Stag;
  switch (dim) {
  case 1:
    dm->ops->setup = DMSetUp_Stag_1d;
    break;
  case 2:
    dm->ops->setup = DMSetUp_Stag_2d;
    break;
  case 3:
    dm->ops->setup = DMSetUp_Stag_3d;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dimension %" PetscInt_FMT, dim);
  }
  dm->ops->clone                    = DMClone_Stag;
  dm->ops->view                     = DMView_Stag;
  dm->ops->getcompatibility         = DMGetCompatibility_Stag;
  dm->ops->createfielddecomposition = DMCreateFieldDecomposition_Stag;
  PetscFunctionReturn(PETSC_SUCCESS);
}
