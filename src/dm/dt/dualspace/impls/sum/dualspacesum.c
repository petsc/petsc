#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

/*@
  PetscDualSpaceSumGetNumSubspaces - Get the number of spaces in the sum space

  Input Parameter:
. sp - the dual space object

  Output Parameter:
. numSumSpaces - the number of spaces

  Level: intermediate

  Note:
  The name NumSubspaces is slightly misleading because it is actually getting the number of defining spaces of the sum, not a number of Subspaces of it

.seealso: `PETSCDUALSPACESUM`, `PetscDualSpace`, `PetscDualSpaceSumSetNumSubspaces()`
@*/
PetscErrorCode PetscDualSpaceSumGetNumSubspaces(PetscDualSpace sp, PetscInt *numSumSpaces)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscAssertPointer(numSumSpaces, 2);
  PetscTryMethod(sp, "PetscDualSpaceSumGetNumSubspaces_C", (PetscDualSpace, PetscInt *), (sp, numSumSpaces));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDualSpaceSumSetNumSubspaces - Set the number of spaces in the sum space

  Input Parameters:
+ sp           - the dual space object
- numSumSpaces - the number of spaces

  Level: intermediate

  Note:
  The name NumSubspaces is slightly misleading because it is actually setting the number of defining spaces of the sum, not a number of Subspaces of it

.seealso: `PETSCDUALSPACESUM`, `PetscDualSpace`, `PetscDualSpaceSumGetNumSubspaces()`
@*/
PetscErrorCode PetscDualSpaceSumSetNumSubspaces(PetscDualSpace sp, PetscInt numSumSpaces)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscTryMethod(sp, "PetscDualSpaceSumSetNumSubspaces_C", (PetscDualSpace, PetscInt), (sp, numSumSpaces));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDualSpaceSumGetConcatenate - Get the concatenate flag for this space.

  Input Parameter:
. sp - the dual space object

  Output Parameter:
. concatenate - flag indicating whether subspaces are concatenated.

  Level: intermediate

  Note:
  A concatenated sum space will have the number of components equal to the sum of the number of
  components of all subspaces. A non-concatenated, or direct sum space will have the same
  number of components as its subspaces.

.seealso: `PETSCDUALSPACESUM`, `PetscDualSpace`, `PetscDualSpaceSumSetConcatenate()`
@*/
PetscErrorCode PetscDualSpaceSumGetConcatenate(PetscDualSpace sp, PetscBool *concatenate)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscTryMethod(sp, "PetscDualSpaceSumGetConcatenate_C", (PetscDualSpace, PetscBool *), (sp, concatenate));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDualSpaceSumSetConcatenate - Sets the concatenate flag for this space.

  Input Parameters:
+ sp          - the dual space object
- concatenate - are subspaces concatenated components (true) or direct summands (false)

  Level: intermediate

  Notes:
  A concatenated sum space will have the number of components equal to the sum of the number of
  components of all subspaces. A non-concatenated, or direct sum space will have the same
  number of components as its subspaces .

.seealso: `PETSCDUALSPACESUM`, `PetscDualSpace`, `PetscDualSpaceSumGetConcatenate()`
@*/
PetscErrorCode PetscDualSpaceSumSetConcatenate(PetscDualSpace sp, PetscBool concatenate)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscTryMethod(sp, "PetscDualSpaceSumSetConcatenate_C", (PetscDualSpace, PetscBool), (sp, concatenate));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDualSpaceSumGetSubspace - Get a space in the sum space

  Input Parameters:
+ sp - the dual space object
- s  - The space number

  Output Parameter:
. subsp - the `PetscDualSpace`

  Level: intermediate

  Note:
  The name GetSubspace is slightly misleading because it is actually getting one of the defining spaces of the sum, not a Subspace of it

.seealso: `PETSCDUALSPACESUM`, `PetscDualSpace`, `PetscDualSpaceSumSetSubspace()`
@*/
PetscErrorCode PetscDualSpaceSumGetSubspace(PetscDualSpace sp, PetscInt s, PetscDualSpace *subsp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscAssertPointer(subsp, 3);
  PetscTryMethod(sp, "PetscDualSpaceSumGetSubspace_C", (PetscDualSpace, PetscInt, PetscDualSpace *), (sp, s, subsp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDualSpaceSumSetSubspace - Set a space in the sum space

  Input Parameters:
+ sp    - the dual space object
. s     - The space number
- subsp - the number of spaces

  Level: intermediate

  Note:
  The name SetSubspace is slightly misleading because it is actually setting one of the defining spaces of the sum, not a Subspace of it

.seealso: `PETSCDUALSPACESUM`, `PetscDualSpace`, `PetscDualSpaceSumGetSubspace()`
@*/
PetscErrorCode PetscDualSpaceSumSetSubspace(PetscDualSpace sp, PetscInt s, PetscDualSpace subsp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  if (subsp) PetscValidHeaderSpecific(subsp, PETSCDUALSPACE_CLASSID, 3);
  PetscTryMethod(sp, "PetscDualSpaceSumSetSubspace_C", (PetscDualSpace, PetscInt, PetscDualSpace), (sp, s, subsp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumGetNumSubspaces_Sum(PetscDualSpace space, PetscInt *numSumSpaces)
{
  PetscDualSpace_Sum *sum = (PetscDualSpace_Sum *)space->data;

  PetscFunctionBegin;
  *numSumSpaces = sum->numSumSpaces;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumSetNumSubspaces_Sum(PetscDualSpace space, PetscInt numSumSpaces)
{
  PetscDualSpace_Sum *sum = (PetscDualSpace_Sum *)space->data;
  PetscInt            Ns  = sum->numSumSpaces;

  PetscFunctionBegin;
  PetscCheck(!sum->setupcalled, PetscObjectComm((PetscObject)space), PETSC_ERR_ARG_WRONGSTATE, "Cannot change number of subspaces after setup called");
  if (numSumSpaces == Ns) PetscFunctionReturn(PETSC_SUCCESS);
  for (PetscInt s = 0; s < Ns; ++s) PetscCall(PetscDualSpaceDestroy(&sum->sumspaces[s]));
  PetscCall(PetscFree(sum->sumspaces));

  Ns = sum->numSumSpaces = numSumSpaces;
  PetscCall(PetscCalloc1(Ns, &sum->sumspaces));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumGetConcatenate_Sum(PetscDualSpace sp, PetscBool *concatenate)
{
  PetscDualSpace_Sum *sum = (PetscDualSpace_Sum *)sp->data;

  PetscFunctionBegin;
  *concatenate = sum->concatenate;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumSetConcatenate_Sum(PetscDualSpace sp, PetscBool concatenate)
{
  PetscDualSpace_Sum *sum = (PetscDualSpace_Sum *)sp->data;

  PetscFunctionBegin;
  PetscCheck(!sum->setupcalled, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONGSTATE, "Cannot change space concatenation after setup called.");

  sum->concatenate = concatenate;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumGetSubspace_Sum(PetscDualSpace space, PetscInt s, PetscDualSpace *subspace)
{
  PetscDualSpace_Sum *sum = (PetscDualSpace_Sum *)space->data;
  PetscInt            Ns  = sum->numSumSpaces;

  PetscFunctionBegin;
  PetscCheck(Ns >= 0, PetscObjectComm((PetscObject)space), PETSC_ERR_ARG_WRONGSTATE, "Must call PetscDualSpaceSumSetNumSubspaces() first");
  PetscCheck(s >= 0 && s < Ns, PetscObjectComm((PetscObject)space), PETSC_ERR_ARG_OUTOFRANGE, "Invalid subspace number %" PetscInt_FMT, s);

  *subspace = sum->sumspaces[s];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumSetSubspace_Sum(PetscDualSpace space, PetscInt s, PetscDualSpace subspace)
{
  PetscDualSpace_Sum *sum = (PetscDualSpace_Sum *)space->data;
  PetscInt            Ns  = sum->numSumSpaces;

  PetscFunctionBegin;
  PetscCheck(!sum->setupcalled, PetscObjectComm((PetscObject)space), PETSC_ERR_ARG_WRONGSTATE, "Cannot change subspace after setup called");
  PetscCheck(Ns >= 0, PetscObjectComm((PetscObject)space), PETSC_ERR_ARG_WRONGSTATE, "Must call PetscDualSpaceSumSetNumSubspaces() first");
  PetscCheck(s >= 0 && s < Ns, PetscObjectComm((PetscObject)space), PETSC_ERR_ARG_OUTOFRANGE, "Invalid subspace number %" PetscInt_FMT, s);

  PetscCall(PetscObjectReference((PetscObject)subspace));
  PetscCall(PetscDualSpaceDestroy(&sum->sumspaces[s]));
  sum->sumspaces[s] = subspace;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceDuplicate_Sum(PetscDualSpace sp, PetscDualSpace spNew)
{
  PetscInt       num_subspaces, Nc;
  PetscBool      concatenate, interleave_basis, interleave_components;
  PetscDualSpace subsp_first     = NULL;
  PetscDualSpace subsp_dup_first = NULL;
  DM             K;

  PetscFunctionBegin;
  PetscCall(PetscDualSpaceSumGetNumSubspaces(sp, &num_subspaces));
  PetscCall(PetscDualSpaceSumSetNumSubspaces(spNew, num_subspaces));
  PetscCall(PetscDualSpaceSumGetConcatenate(sp, &concatenate));
  PetscCall(PetscDualSpaceSumSetConcatenate(spNew, concatenate));
  PetscCall(PetscDualSpaceSumGetInterleave(sp, &interleave_basis, &interleave_components));
  PetscCall(PetscDualSpaceSumSetInterleave(spNew, interleave_basis, interleave_components));
  PetscCall(PetscDualSpaceGetDM(sp, &K));
  PetscCall(PetscDualSpaceSetDM(spNew, K));
  PetscCall(PetscDualSpaceGetNumComponents(sp, &Nc));
  PetscCall(PetscDualSpaceSetNumComponents(spNew, Nc));
  for (PetscInt s = 0; s < num_subspaces; s++) {
    PetscDualSpace subsp, subspNew;

    PetscCall(PetscDualSpaceSumGetSubspace(sp, s, &subsp));
    if (s == 0) {
      subsp_first = subsp;
      PetscCall(PetscDualSpaceDuplicate(subsp, &subsp_dup_first));
      subspNew = subsp_dup_first;
    } else if (subsp == subsp_first) {
      PetscCall(PetscObjectReference((PetscObject)subsp_dup_first));
      subspNew = subsp_dup_first;
    } else {
      PetscCall(PetscDualSpaceDuplicate(subsp, &subspNew));
    }
    PetscCall(PetscDualSpaceSumSetSubspace(spNew, s, subspNew));
    PetscCall(PetscDualSpaceDestroy(&subspNew));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumCreateQuadrature(PetscInt Ns, PetscInt cdim, PetscBool uniform_points, PetscQuadrature subquads[], PetscQuadrature *fullquad)
{
  PetscReal *points;
  PetscInt   Npoints;

  PetscFunctionBegin;
  if (uniform_points) {
    PetscCall(PetscObjectReference((PetscObject)subquads[0]));
    *fullquad = subquads[0];
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  Npoints = 0;
  for (PetscInt s = 0; s < Ns; s++) {
    PetscInt subNpoints;

    if (!subquads[s]) continue;
    PetscCall(PetscQuadratureGetData(subquads[s], NULL, NULL, &subNpoints, NULL, NULL));
    Npoints += subNpoints;
  }
  *fullquad = NULL;
  if (!Npoints) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscMalloc1(Npoints * cdim, &points));
  for (PetscInt s = 0, offset = 0; s < Ns; s++) {
    PetscInt         subNpoints;
    const PetscReal *subpoints;

    PetscCall(PetscQuadratureGetData(subquads[s], NULL, NULL, &subNpoints, &subpoints, NULL));
    PetscCall(PetscArraycpy(&points[offset], subpoints, cdim * subNpoints));
    offset += cdim * subNpoints;
  }
  PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, fullquad));
  PetscCall(PetscQuadratureSetData(*fullquad, cdim, 0, Npoints, points, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumCreateMatrix(PetscDualSpace sp, Mat submats[], ISLocalToGlobalMapping map_rows[], ISLocalToGlobalMapping map_cols[], PetscQuadrature fullquad, Mat *fullmat)
{
  Mat          mat;
  PetscInt    *i = NULL, *j = NULL;
  PetscScalar *v = NULL;
  PetscInt     npoints;
  PetscInt     nrows = 0, ncols, nnz = 0;
  PetscInt     Nc, num_subspaces;

  PetscFunctionBegin;
  PetscCall(PetscDualSpaceGetNumComponents(sp, &Nc));
  PetscCall(PetscDualSpaceSumGetNumSubspaces(sp, &num_subspaces));
  PetscCall(PetscQuadratureGetData(fullquad, NULL, NULL, &npoints, NULL, NULL));
  ncols = Nc * npoints;
  for (PetscInt s = 0; s < num_subspaces; s++) {
    // Get the COO data for each matrix, map the is and js, and append to growing COO data
    PetscInt        sNrows, sNcols;
    Mat             smat;
    const PetscInt *si;
    const PetscInt *sj;
    PetscScalar    *sv;
    PetscMemType    memtype;
    PetscInt        snz;
    PetscInt        snz_actual;
    PetscInt       *cooi;
    PetscInt       *cooj;
    PetscScalar    *coov;
    PetscScalar    *v_new;
    PetscInt       *i_new;
    PetscInt       *j_new;

    if (!submats[s]) continue;
    PetscCall(MatGetSize(submats[s], &sNrows, &sNcols));
    nrows += sNrows;
    PetscCall(MatConvert(submats[s], MATSEQAIJ, MAT_INITIAL_MATRIX, &smat));
    PetscCall(MatBindToCPU(smat, PETSC_TRUE));
    PetscCall(MatSeqAIJGetCSRAndMemType(smat, &si, &sj, &sv, &memtype));
    PetscCheck(memtype == PETSC_MEMTYPE_HOST, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not convert matrix to host memory");
    snz = si[sNrows];

    PetscCall(PetscMalloc1(nnz + snz, &v_new));
    PetscCall(PetscArraycpy(v_new, v, nnz));
    PetscCall(PetscFree(v));
    v = v_new;

    PetscCall(PetscMalloc1(nnz + snz, &i_new));
    PetscCall(PetscArraycpy(i_new, i, nnz));
    PetscCall(PetscFree(i));
    i = i_new;

    PetscCall(PetscMalloc1(nnz + snz, &j_new));
    PetscCall(PetscArraycpy(j_new, j, nnz));
    PetscCall(PetscFree(j));
    j = j_new;

    PetscCall(PetscMalloc2(snz, &cooi, snz, &cooj));

    coov = &v[nnz];

    snz_actual = 0;
    for (PetscInt row = 0; row < sNrows; row++) {
      for (PetscInt k = si[row]; k < si[row + 1]; k++, snz_actual++) {
        cooi[snz_actual] = row;
        cooj[snz_actual] = sj[k];
        coov[snz_actual] = sv[k];
      }
    }
    PetscCall(MatDestroy(&smat));

    if (snz_actual > 0) {
      PetscCall(ISLocalToGlobalMappingApply(map_rows[s], snz_actual, cooi, &i[nnz]));
      PetscCall(ISLocalToGlobalMappingApply(map_cols[s], snz_actual, cooj, &j[nnz]));
      nnz += snz_actual;
    }
    PetscCall(PetscFree2(cooi, cooj));
  }
  PetscCall(MatCreate(PETSC_COMM_SELF, fullmat));
  mat = *fullmat;
  PetscCall(MatSetSizes(mat, nrows, ncols, nrows, ncols));
  PetscCall(MatSetType(mat, MATSEQAIJ));
  PetscCall(MatSetPreallocationCOO(mat, nnz, i, j));
  PetscCall(MatSetValuesCOO(mat, v, INSERT_VALUES));
  PetscCall(PetscFree(i));
  PetscCall(PetscFree(j));
  PetscCall(PetscFree(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumCreateMappings(PetscDualSpace sp, PetscBool interior, PetscBool uniform_points, ISLocalToGlobalMapping map_row[], ISLocalToGlobalMapping map_col[])
{
  PetscSection section;
  PetscInt     pStart, pEnd, Ns, Nc;
  PetscInt     num_points = 0;
  PetscBool    interleave_basis, interleave_components, concatenate;
  PetscInt    *roffset;

  PetscFunctionBegin;
  if (interior) {
    PetscCall(PetscDualSpaceGetInteriorSection(sp, &section));
  } else {
    PetscCall(PetscDualSpaceGetSection(sp, &section));
  }
  PetscCall(PetscDualSpaceGetNumComponents(sp, &Nc));
  PetscCall(PetscDualSpaceSumGetNumSubspaces(sp, &Ns));
  PetscCall(PetscDualSpaceSumGetInterleave(sp, &interleave_basis, &interleave_components));
  PetscCall(PetscDualSpaceSumGetConcatenate(sp, &concatenate));
  PetscCall(PetscSectionGetChart(section, &pStart, &pEnd));
  for (PetscInt p = pStart; p < pEnd; p++) {
    PetscInt dof;

    PetscCall(PetscSectionGetDof(section, p, &dof));
    num_points += (dof > 0);
  }
  PetscCall(PetscCalloc1(pEnd - pStart, &roffset));
  for (PetscInt s = 0, coffset = 0; s < Ns; s++) {
    PetscDualSpace  subsp;
    PetscSection    subsection;
    IS              is_row, is_col;
    PetscInt        subNb;
    PetscInt        sNc, sNpoints, sNcols;
    PetscQuadrature quad;

    PetscCall(PetscDualSpaceSumGetSubspace(sp, s, &subsp));
    PetscCall(PetscDualSpaceGetNumComponents(subsp, &sNc));
    if (interior) {
      PetscCall(PetscDualSpaceGetInteriorSection(subsp, &subsection));
      PetscCall(PetscDualSpaceGetInteriorData(subsp, &quad, NULL));
    } else {
      PetscCall(PetscDualSpaceGetSection(subsp, &subsection));
      PetscCall(PetscDualSpaceGetAllData(subsp, &quad, NULL));
    }
    PetscCall(PetscSectionGetStorageSize(subsection, &subNb));
    if (num_points == 1) {
      PetscInt rstride;

      rstride = interleave_basis ? Ns : 1;
      PetscCall(ISCreateStride(PETSC_COMM_SELF, subNb, roffset[0], rstride, &is_row));
      roffset[0] += interleave_basis ? 1 : subNb;
    } else {
      PetscInt *rows;

      PetscCall(PetscMalloc1(subNb, &rows));
      for (PetscInt p = pStart; p < pEnd; p++) {
        PetscInt subdof, dof, off, suboff, stride;

        PetscCall(PetscSectionGetOffset(subsection, p, &suboff));
        PetscCall(PetscSectionGetDof(subsection, p, &subdof));
        PetscCall(PetscSectionGetOffset(section, p, &off));
        PetscCall(PetscSectionGetDof(section, p, &dof));
        PetscCheck(subdof * Ns == dof || !interleave_basis, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Basis cannot be interleaved");
        stride = interleave_basis ? Ns : 1;
        for (PetscInt k = 0; k < subdof; k++) rows[suboff + k] = off + roffset[p - pStart] + k * stride;
        roffset[p - pStart] += interleave_basis ? 1 : subdof;
      }
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF, subNb, rows, PETSC_OWN_POINTER, &is_row));
    }

    sNpoints = 0;
    if (quad) PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &sNpoints, NULL, NULL));
    sNcols = sNpoints * sNc;

    if (!concatenate) {
      PetscCall(ISCreateStride(PETSC_COMM_SELF, sNcols, coffset, 1, &is_col));
      coffset += uniform_points ? 0 : sNcols;
    } else {
      if (uniform_points && interleave_components) {
        PetscCall(ISCreateStride(PETSC_COMM_SELF, sNcols, coffset, Ns, &is_col));
        coffset += 1;
      } else {
        PetscInt *cols;

        PetscCall(PetscMalloc1(sNcols, &cols));
        for (PetscInt p = 0, r = 0; p < sNpoints; p++) {
          for (PetscInt c = 0; c < sNc; c++, r++) cols[r] = coffset + p * Nc + c;
        }
        coffset += uniform_points ? sNc : Nc * sNpoints + sNc;
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF, sNcols, cols, PETSC_OWN_POINTER, &is_col));
      }
    }
    PetscCall(ISLocalToGlobalMappingCreateIS(is_row, &map_row[s]));
    PetscCall(ISLocalToGlobalMappingCreateIS(is_col, &map_col[s]));
    PetscCall(ISDestroy(&is_row));
    PetscCall(ISDestroy(&is_col));
  }
  PetscCall(PetscFree(roffset));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceCreateFacetSubspace_Sum(PetscDualSpace sp, PetscInt f, PetscDualSpace *bdsp)
{
  PetscInt       k, Nc, Nk, Nknew, Ncnew, Ns;
  PetscInt       dim, pointDim = -1;
  PetscInt       depth, Ncopies;
  PetscBool      any;
  DM             dm, K;
  DMPolytopeType ct;

  PetscFunctionBegin;
  PetscCall(PetscDualSpaceSumGetNumSubspaces(sp, &Ns));
  any = PETSC_FALSE;
  for (PetscInt s = 0; s < Ns; s++) {
    PetscDualSpace subsp, subpointsp;

    PetscCall(PetscDualSpaceSumGetSubspace(sp, s, &subsp));
    PetscCall(PetscDualSpaceGetPointSubspace(subsp, f, &subpointsp));
    if (subpointsp) any = PETSC_TRUE;
  }
  if (!any) {
    *bdsp = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscDualSpaceGetDM(sp, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(PetscDualSpaceDuplicate(sp, bdsp));
  PetscCheck((depth == dim) || (depth == 1), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unsupported interpolation state of reference element");
  pointDim = (depth == dim) ? (dim - 1) : 0;
  PetscCall(DMPlexGetCellType(dm, f, &ct));
  PetscCall(DMPlexCreateReferenceCell(PETSC_COMM_SELF, ct, &K));
  PetscCall(PetscDualSpaceSetDM(*bdsp, K));
  PetscCall(PetscDualSpaceGetFormDegree(sp, &k));
  PetscCall(PetscDualSpaceGetNumComponents(sp, &Nc));
  PetscCall(PetscDTBinomialInt(dim, PetscAbsInt(k), &Nk));
  Ncopies = Nc / Nk;
  PetscCall(PetscDTBinomialInt(pointDim, PetscAbsInt(k), &Nknew));
  Ncnew = Nknew * Ncopies;
  PetscCall(PetscDualSpaceSetNumComponents(*bdsp, Ncnew));
  for (PetscInt s = 0; s < Ns; s++) {
    PetscDualSpace subsp, subpointsp;

    PetscCall(PetscDualSpaceSumGetSubspace(sp, s, &subsp));
    PetscCall(PetscDualSpaceGetPointSubspace(subsp, f, &subpointsp));
    if (subpointsp) {
      PetscCall(PetscObjectReference((PetscObject)subpointsp));
    } else {
      // make an empty dual space
      PetscInt subNc, subNcopies, subpointNc;

      PetscCall(PetscDualSpaceCreate(PetscObjectComm((PetscObject)subsp), &subpointsp));
      PetscCall(PetscDualSpaceGetNumComponents(subsp, &subNc));
      subNcopies = subNc / Nk;
      subpointNc = subNcopies * Nknew;
      PetscCall(PetscDualSpaceSetType(subpointsp, PETSCDUALSPACESIMPLE));
      PetscCall(PetscDualSpaceSimpleSetDimension(subpointsp, 0));
      PetscCall(PetscDualSpaceSetFormDegree(subpointsp, k));
      PetscCall(PetscDualSpaceSetNumComponents(subpointsp, subpointNc));
    }
    // K should be equal to subpointsp->dm, but we want it to be exactly the same
    PetscCall(PetscObjectReference((PetscObject)K));
    PetscCall(DMDestroy(&subpointsp->dm));
    subpointsp->dm = K;
    PetscCall(PetscDualSpaceSetUp(subpointsp));
    PetscCall(PetscDualSpaceSumSetSubspace(*bdsp, s, subpointsp));
    PetscCall(PetscDualSpaceDestroy(&subpointsp));
  }
  PetscCall(DMDestroy(&K));
  PetscCall(PetscDualSpaceSetUp(*bdsp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumIsUniform(PetscDualSpace sp, PetscBool *is_uniform)
{
  PetscDualSpace_Sum *sum     = (PetscDualSpace_Sum *)sp->data;
  PetscBool           uniform = PETSC_TRUE;

  PetscFunctionBegin;
  for (PetscInt s = 1; s < sum->numSumSpaces; s++) {
    if (sum->sumspaces[s] != sum->sumspaces[0]) {
      uniform = PETSC_FALSE;
      break;
    }
  }
  *is_uniform = uniform;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceGetSymmetries_Sum(PetscDualSpace sp, const PetscInt ****perms, const PetscScalar ****flips)
{
  PetscDualSpace_Sum *sum = (PetscDualSpace_Sum *)sp->data;

  PetscFunctionBegin;
  if (!sum->symComputed) {
    PetscInt       Ns;
    PetscBool      any_perms = PETSC_FALSE;
    PetscBool      any_flips = PETSC_FALSE;
    PetscInt    ***symperms  = NULL;
    PetscScalar ***symflips  = NULL;

    sum->symComputed = PETSC_TRUE;
    PetscCall(PetscDualSpaceSumGetNumSubspaces(sp, &Ns));
    for (PetscInt s = 0; s < Ns; s++) {
      PetscDualSpace       subsp;
      const PetscInt    ***sub_perms;
      const PetscScalar ***sub_flips;

      PetscCall(PetscDualSpaceSumGetSubspace(sp, s, &subsp));
      PetscCall(PetscDualSpaceGetSymmetries(subsp, &sub_perms, &sub_flips));
      if (sub_perms) any_perms = PETSC_TRUE;
      if (sub_flips) any_flips = PETSC_TRUE;
    }
    if (any_perms || any_flips) {
      DM       K;
      PetscInt pStart, pEnd, numPoints;
      PetscInt spintdim;

      PetscCall(PetscDualSpaceGetDM(sp, &K));
      PetscCall(DMPlexGetChart(K, &pStart, &pEnd));
      numPoints = pEnd - pStart;
      PetscCall(PetscCalloc1(numPoints, &symperms));
      PetscCall(PetscCalloc1(numPoints, &symflips));
      PetscCall(PetscDualSpaceGetBoundarySymmetries_Internal(sp, symperms, symflips));
      // get interior symmetries
      PetscCall(PetscDualSpaceGetInteriorDimension(sp, &spintdim));
      if (spintdim) {
        PetscInt       groupSize;
        PetscInt     **cellPerms;
        PetscScalar  **cellFlips;
        DMPolytopeType ct;

        PetscCall(DMPlexGetCellType(K, 0, &ct));
        groupSize       = DMPolytopeTypeGetNumArrangements(ct);
        sum->numSelfSym = groupSize;
        sum->selfSymOff = groupSize / 2;
        PetscCall(PetscCalloc1(groupSize, &cellPerms));
        PetscCall(PetscCalloc1(groupSize, &cellFlips));
        symperms[0] = &cellPerms[groupSize / 2];
        symflips[0] = &cellFlips[groupSize / 2];
        for (PetscInt o = -groupSize / 2; o < groupSize / 2; o++) {
          PetscBool any_o_perms = PETSC_FALSE;
          PetscBool any_o_flips = PETSC_FALSE;

          for (PetscInt s = 0; s < Ns; s++) {
            PetscDualSpace       subsp;
            const PetscInt    ***sub_perms;
            const PetscScalar ***sub_flips;

            PetscCall(PetscDualSpaceSumGetSubspace(sp, s, &subsp));
            PetscCall(PetscDualSpaceGetSymmetries(subsp, &sub_perms, &sub_flips));
            if (sub_perms && sub_perms[0] && sub_perms[0][o]) any_o_perms = PETSC_TRUE;
            if (sub_flips && sub_flips[0] && sub_flips[0][o]) any_o_flips = PETSC_TRUE;
          }
          if (any_o_perms) {
            PetscInt *o_perm;

            PetscCall(PetscMalloc1(spintdim, &o_perm));
            for (PetscInt i = 0; i < spintdim; i++) o_perm[i] = i;
            for (PetscInt s = 0; s < Ns; s++) {
              PetscDualSpace       subsp;
              const PetscInt    ***sub_perms;
              const PetscScalar ***sub_flips;

              PetscCall(PetscDualSpaceSumGetSubspace(sp, s, &subsp));
              PetscCall(PetscDualSpaceGetSymmetries(subsp, &sub_perms, &sub_flips));
              if (sub_perms && sub_perms[0] && sub_perms[0][o]) {
                PetscInt  subspdim;
                PetscInt *range, *domain;
                PetscInt *range_mapped, *domain_mapped;

                PetscCall(PetscDualSpaceGetInteriorDimension(subsp, &subspdim));
                PetscCall(PetscMalloc4(subspdim, &range, subspdim, &range_mapped, subspdim, &domain, subspdim, &domain_mapped));
                for (PetscInt i = 0; i < subspdim; i++) domain[i] = i;
                PetscCall(PetscArraycpy(range, sub_perms[0][o], subspdim));
                PetscCall(ISLocalToGlobalMappingApply(sum->int_rows[s], subspdim, domain, domain_mapped));
                PetscCall(ISLocalToGlobalMappingApply(sum->int_rows[s], subspdim, range, range_mapped));
                for (PetscInt i = 0; i < subspdim; i++) o_perm[domain_mapped[i]] = range_mapped[i];
                PetscCall(PetscFree4(range, range_mapped, domain, domain_mapped));
              }
            }
            symperms[0][o] = o_perm;
          }
          if (any_o_flips) {
            PetscScalar *o_flip;

            PetscCall(PetscMalloc1(spintdim, &o_flip));
            for (PetscInt i = 0; i < spintdim; i++) o_flip[i] = 1.0;
            for (PetscInt s = 0; s < Ns; s++) {
              PetscDualSpace       subsp;
              const PetscInt    ***sub_perms;
              const PetscScalar ***sub_flips;

              PetscCall(PetscDualSpaceSumGetSubspace(sp, s, &subsp));
              PetscCall(PetscDualSpaceGetSymmetries(subsp, &sub_perms, &sub_flips));
              if (sub_perms && sub_perms[0] && sub_perms[0][o]) {
                PetscInt  subspdim;
                PetscInt *domain;
                PetscInt *domain_mapped;

                PetscCall(PetscDualSpaceGetInteriorDimension(subsp, &subspdim));
                PetscCall(PetscMalloc2(subspdim, &domain, subspdim, &domain_mapped));
                for (PetscInt i = 0; i < subspdim; i++) domain[i] = i;
                PetscCall(ISLocalToGlobalMappingApply(sum->int_rows[s], subspdim, domain, domain_mapped));
                for (PetscInt i = 0; i < subspdim; i++) o_flip[domain_mapped[i]] = sub_perms[0][o][i];
                PetscCall(PetscFree2(domain, domain_mapped));
              }
            }
            symflips[0][o] = o_flip;
          }
        }
        {
          PetscBool any_perms = PETSC_FALSE;
          PetscBool any_flips = PETSC_FALSE;
          for (PetscInt o = -groupSize / 2; o < groupSize / 2; o++) {
            if (symperms[0][o]) any_perms = PETSC_TRUE;
            if (symflips[0][o]) any_flips = PETSC_TRUE;
          }
          if (!any_perms) {
            PetscCall(PetscFree(cellPerms));
            symperms[0] = NULL;
          }
          if (!any_flips) {
            PetscCall(PetscFree(cellFlips));
            symflips[0] = NULL;
          }
        }
      }
      if (!any_perms) {
        PetscCall(PetscFree(symperms));
        symperms = NULL;
      }
      if (!any_flips) {
        PetscCall(PetscFree(symflips));
        symflips = NULL;
      }
    }
    sum->symperms = symperms;
    sum->symflips = symflips;
  }
  if (perms) *perms = (const PetscInt ***)sum->symperms;
  if (flips) *flips = (const PetscScalar ***)sum->symflips;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSetUp_Sum(PetscDualSpace sp)
{
  PetscDualSpace_Sum *sum         = (PetscDualSpace_Sum *)sp->data;
  PetscBool           concatenate = PETSC_TRUE;
  PetscBool           uniform;
  PetscInt            Ns, Nc, i, sum_Nc = 0;
  PetscInt            minNc, maxNc;
  PetscInt            minForm, maxForm, cdim, depth;
  DM                  K;
  PetscQuadrature    *all_quads = NULL;
  PetscQuadrature    *int_quads = NULL;
  Mat                *all_mats  = NULL;
  Mat                *int_mats  = NULL;

  PetscFunctionBegin;
  if (sum->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  sum->setupcalled = PETSC_TRUE;

  PetscCall(PetscDualSpaceSumGetNumSubspaces(sp, &Ns));
  PetscCheck(Ns >= 0, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_OUTOFRANGE, "Cannot have %" PetscInt_FMT " subspaces", Ns);

  // step 1: make sure they share a DM
  PetscCall(PetscDualSpaceGetDM(sp, &K));
  PetscCall(DMGetCoordinateDim(K, &cdim));
  PetscCall(DMPlexGetDepth(K, &depth));
  PetscCall(PetscDualSpaceSumIsUniform(sp, &sp->uniform));
  uniform = sp->uniform;
  {
    for (PetscInt s = 0; s < Ns; s++) {
      PetscDualSpace subsp;
      DM             sub_K;

      PetscCall(PetscDualSpaceSumGetSubspace(sp, s, &subsp));
      PetscCall(PetscDualSpaceSetUp(subsp));
      PetscCall(PetscDualSpaceGetDM(subsp, &sub_K));
      if (s == 0 && K == NULL) {
        PetscCall(PetscDualSpaceSetDM(sp, sub_K));
        K = sub_K;
      }
      PetscCheck(sub_K == K, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONGSTATE, "Subspace %" PetscInt_FMT " does not have the same DM as the sum space", s);
    }
  }

  // step 2: count components
  PetscCall(PetscDualSpaceGetNumComponents(sp, &Nc));
  PetscCall(PetscDualSpaceSumGetConcatenate(sp, &concatenate));
  minNc   = Nc;
  maxNc   = Nc;
  minForm = sp->k == PETSC_FORM_DEGREE_UNDEFINED ? PETSC_INT_MAX : sp->k;
  maxForm = sp->k == PETSC_FORM_DEGREE_UNDEFINED ? PETSC_INT_MIN : sp->k;
  for (i = 0; i < Ns; ++i) {
    PetscInt       sNc, formDegree;
    PetscDualSpace si;

    PetscCall(PetscDualSpaceSumGetSubspace(sp, i, &si));
    PetscCall(PetscDualSpaceSetUp(si));
    PetscCall(PetscDualSpaceGetNumComponents(si, &sNc));
    if (sNc != Nc) PetscCheck(concatenate, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONGSTATE, "Subspace as a different number of components but space does not concatenate components");
    minNc = PetscMin(minNc, sNc);
    maxNc = PetscMax(maxNc, sNc);
    sum_Nc += sNc;
    PetscCall(PetscDualSpaceGetFormDegree(si, &formDegree));
    minForm = PetscMin(minForm, formDegree);
    maxForm = PetscMax(maxForm, formDegree);
  }
  sp->k = (minForm != maxForm) ? PETSC_FORM_DEGREE_UNDEFINED : minForm;

  if (concatenate) PetscCheck(sum_Nc == Nc, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_OUTOFRANGE, "Total number of subspace components (%" PetscInt_FMT ") does not match number of target space components (%" PetscInt_FMT ").", sum_Nc, Nc);
  else PetscCheck(minNc == Nc && maxNc == Nc, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_OUTOFRANGE, "Subspaces must have same number of components as the target space.");

  /* A PetscDualSpace should have a fixed number of components, but
     if the spaces we are combining have different form degrees, they will not
     have the same number of components on subcomponents of the boundary,
     so we do not try to create boundary dual spaces in this case */
  if (sp->k != PETSC_FORM_DEGREE_UNDEFINED && depth > 0) {
    PetscInt  pStart, pEnd;
    PetscInt *pStratStart, *pStratEnd;

    PetscCall(DMPlexGetDepth(K, &depth));
    PetscCall(DMPlexGetChart(K, &pStart, &pEnd));
    PetscCall(PetscCalloc1(pEnd, &sp->pointSpaces));
    PetscCall(PetscMalloc2(depth + 1, &pStratStart, depth + 1, &pStratEnd));
    for (PetscInt d = 0; d <= depth; ++d) PetscCall(DMPlexGetDepthStratum(K, d, &pStratStart[d], &pStratEnd[d]));

    for (PetscInt p = pStratStart[depth - 1]; p < pStratEnd[depth - 1]; p++) { /* calculate the facet dual spaces */
      PetscReal      v0[3];
      DMPolytopeType ptype;
      PetscReal      J[9], detJ;
      PetscInt       q;

      PetscCall(DMPlexComputeCellGeometryAffineFEM(K, p, v0, J, NULL, &detJ));
      PetscCall(DMPlexGetCellType(K, p, &ptype));

      /* compare to previous facets: if computed, reference that dualspace */
      for (q = pStratStart[depth - 1]; q < p; q++) {
        DMPolytopeType qtype;

        PetscCall(DMPlexGetCellType(K, q, &qtype));
        if (qtype == ptype) break;
      }
      if (q < p) { /* this facet has the same dual space as that one */
        PetscCall(PetscObjectReference((PetscObject)sp->pointSpaces[q]));
        sp->pointSpaces[p] = sp->pointSpaces[q];
        continue;
      }
      /* if not, recursively compute this dual space */
      PetscCall(PetscDualSpaceCreateFacetSubspace_Sum(sp, p, &sp->pointSpaces[p]));
    }
    for (PetscInt h = 2; h <= depth; h++) { /* get the higher subspaces from the facet subspaces */
      PetscInt hd = depth - h;

      for (PetscInt p = pStratStart[hd]; p < pStratEnd[hd]; p++) {
        PetscInt        suppSize;
        const PetscInt *supp;
        DM              qdm;
        PetscDualSpace  qsp, psp;
        PetscInt        c, coneSize, q;
        const PetscInt *cone;
        const PetscInt *refCone;

        PetscCall(DMPlexGetSupportSize(K, p, &suppSize));
        PetscCall(DMPlexGetSupport(K, p, &supp));
        q   = supp[0];
        qsp = sp->pointSpaces[q];
        PetscCall(DMPlexGetConeSize(K, q, &coneSize));
        PetscCall(DMPlexGetCone(K, q, &cone));
        for (c = 0; c < coneSize; c++)
          if (cone[c] == p) break;
        PetscCheck(c != coneSize, PetscObjectComm((PetscObject)K), PETSC_ERR_PLIB, "cone/support mismatch");
        if (!qsp) {
          sp->pointSpaces[p] = NULL;
          continue;
        }
        PetscCall(PetscDualSpaceGetDM(qsp, &qdm));
        PetscCall(DMPlexGetCone(qdm, 0, &refCone));
        /* get the equivalent dual space from the support dual space */
        PetscCall(PetscDualSpaceGetPointSubspace(qsp, refCone[c], &psp));
        PetscCall(PetscObjectReference((PetscObject)psp));
        sp->pointSpaces[p] = psp;
      }
    }
    PetscCall(PetscFree2(pStratStart, pStratEnd));
  }

  sum->uniform = uniform;
  PetscCall(PetscCalloc1(Ns, &sum->all_rows));
  PetscCall(PetscCalloc1(Ns, &sum->all_cols));
  PetscCall(PetscCalloc1(Ns, &sum->int_rows));
  PetscCall(PetscCalloc1(Ns, &sum->int_cols));
  PetscCall(PetscMalloc4(Ns, &all_quads, Ns, &all_mats, Ns, &int_quads, Ns, &int_mats));
  {
    // test for uniform all points and uniform interior points
    PetscBool       uniform_all         = PETSC_TRUE;
    PetscBool       uniform_interior    = PETSC_TRUE;
    PetscQuadrature quad_all_first      = NULL;
    PetscQuadrature quad_interior_first = NULL;
    PetscInt        pStart, pEnd;

    PetscCall(DMPlexGetChart(K, &pStart, &pEnd));
    PetscCall(PetscDualSpaceSectionCreate_Internal(sp, &sp->pointSection));

    for (PetscInt p = pStart; p < pEnd; p++) {
      PetscInt full_dof = 0;

      for (PetscInt s = 0; s < Ns; s++) {
        PetscDualSpace subsp;
        PetscSection   subsection;
        PetscInt       dof;

        PetscCall(PetscDualSpaceSumGetSubspace(sp, s, &subsp));
        PetscCall(PetscDualSpaceGetSection(subsp, &subsection));
        PetscCall(PetscSectionGetDof(subsection, p, &dof));
        full_dof += dof;
      }
      PetscCall(PetscSectionSetDof(sp->pointSection, p, full_dof));
    }
    PetscCall(PetscDualSpaceSectionSetUp_Internal(sp, sp->pointSection));

    for (PetscInt s = 0; s < Ns; s++) {
      PetscDualSpace  subsp;
      PetscQuadrature subquad_all;
      PetscQuadrature subquad_interior;
      Mat             submat_all;
      Mat             submat_interior;

      PetscCall(PetscDualSpaceSumGetSubspace(sp, s, &subsp));
      PetscCall(PetscDualSpaceGetAllData(subsp, &subquad_all, &submat_all));
      PetscCall(PetscDualSpaceGetInteriorData(subsp, &subquad_interior, &submat_interior));
      if (!s) {
        quad_all_first      = subquad_all;
        quad_interior_first = subquad_interior;
      } else {
        if (subquad_all != quad_all_first) uniform_all = PETSC_FALSE;
        if (subquad_interior != quad_interior_first) uniform_interior = PETSC_FALSE;
      }
      all_quads[s] = subquad_all;
      int_quads[s] = subquad_interior;
      all_mats[s]  = submat_all;
      int_mats[s]  = submat_interior;
    }
    sum->uniform_all_points      = uniform_all;
    sum->uniform_interior_points = uniform_interior;

    PetscCall(PetscDualSpaceSumCreateMappings(sp, PETSC_TRUE, uniform_interior, sum->int_rows, sum->int_cols));
    PetscCall(PetscDualSpaceSumCreateQuadrature(Ns, cdim, uniform_interior, int_quads, &sp->intNodes));
    if (sp->intNodes) PetscCall(PetscDualSpaceSumCreateMatrix(sp, int_mats, sum->int_rows, sum->int_cols, sp->intNodes, &sp->intMat));

    PetscCall(PetscDualSpaceSumCreateMappings(sp, PETSC_FALSE, uniform_all, sum->all_rows, sum->all_cols));
    PetscCall(PetscDualSpaceSumCreateQuadrature(Ns, cdim, uniform_all, all_quads, &sp->allNodes));
    if (sp->allNodes) PetscCall(PetscDualSpaceSumCreateMatrix(sp, all_mats, sum->all_rows, sum->all_cols, sp->allNodes, &sp->allMat));
  }
  PetscCall(PetscFree4(all_quads, all_mats, int_quads, int_mats));
  PetscCall(PetscDualSpaceComputeFunctionalsFromAllData(sp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumView_Ascii(PetscDualSpace sp, PetscViewer v)
{
  PetscDualSpace_Sum *sum         = (PetscDualSpace_Sum *)sp->data;
  PetscBool           concatenate = sum->concatenate;
  PetscInt            i, Ns = sum->numSumSpaces;

  PetscFunctionBegin;
  if (concatenate) PetscCall(PetscViewerASCIIPrintf(v, "Sum dual space of %" PetscInt_FMT " concatenated subspaces%s\n", Ns, sum->uniform ? " (all identical)" : ""));
  else PetscCall(PetscViewerASCIIPrintf(v, "Sum dual space of %" PetscInt_FMT " subspaces%s\n", Ns, sum->uniform ? " (all identical)" : ""));
  for (i = 0; i < (sum->uniform ? (Ns > 0 ? 1 : 0) : Ns); ++i) {
    PetscCall(PetscViewerASCIIPushTab(v));
    PetscCall(PetscDualSpaceView(sum->sumspaces[i], v));
    PetscCall(PetscViewerASCIIPopTab(v));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceView_Sum(PetscDualSpace sp, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) PetscCall(PetscDualSpaceSumView_Ascii(sp, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceDestroy_Sum(PetscDualSpace sp)
{
  PetscDualSpace_Sum *sum = (PetscDualSpace_Sum *)sp->data;
  PetscInt            i, Ns = sum->numSumSpaces;

  PetscFunctionBegin;
  if (sum->symperms) {
    PetscInt **selfSyms = sum->symperms[0];

    if (selfSyms) {
      PetscInt i, **allocated = &selfSyms[-sum->selfSymOff];

      for (i = 0; i < sum->numSelfSym; i++) PetscCall(PetscFree(allocated[i]));
      PetscCall(PetscFree(allocated));
    }
    PetscCall(PetscFree(sum->symperms));
  }
  if (sum->symflips) {
    PetscScalar **selfSyms = sum->symflips[0];

    if (selfSyms) {
      PetscInt      i;
      PetscScalar **allocated = &selfSyms[-sum->selfSymOff];

      for (i = 0; i < sum->numSelfSym; i++) PetscCall(PetscFree(allocated[i]));
      PetscCall(PetscFree(allocated));
    }
    PetscCall(PetscFree(sum->symflips));
  }
  for (i = 0; i < Ns; ++i) {
    PetscCall(PetscDualSpaceDestroy(&sum->sumspaces[i]));
    if (sum->all_rows) PetscCall(ISLocalToGlobalMappingDestroy(&sum->all_rows[i]));
    if (sum->all_cols) PetscCall(ISLocalToGlobalMappingDestroy(&sum->all_cols[i]));
    if (sum->int_rows) PetscCall(ISLocalToGlobalMappingDestroy(&sum->int_rows[i]));
    if (sum->int_cols) PetscCall(ISLocalToGlobalMappingDestroy(&sum->int_cols[i]));
  }
  PetscCall(PetscFree(sum->sumspaces));
  PetscCall(PetscFree(sum->all_rows));
  PetscCall(PetscFree(sum->all_cols));
  PetscCall(PetscFree(sum->int_rows));
  PetscCall(PetscFree(sum->int_cols));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumSetSubspace_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumGetSubspace_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumSetNumSubspaces_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumGetNumSubspaces_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumGetConcatenate_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumSetConcatenate_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumGetInterleave_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumSetInterleave_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeGetContinuity_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeSetContinuity_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeGetMomentOrder_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeSetMomentOrder_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeGetNodeType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeSetNodeType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeGetTensor_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeSetTensor_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeGetTrimmed_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeSetTrimmed_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeGetUseMoments_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeSetUseMoments_C", NULL));
  PetscCall(PetscFree(sum));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDualSpaceSumSetInterleave - Set whether the basis functions and components of a uniform sum are interleaved

  Logically collective

  Input Parameters:
+ sp                    - a `PetscDualSpace` of type `PETSCDUALSPACESUM`
. interleave_basis      - if `PETSC_TRUE`, the basis vectors of the subspaces are interleaved
- interleave_components - if `PETSC_TRUE` and the space concatenates components (`PetscDualSpaceSumGetConcatenate()`),
                          interleave the concatenated components

  Level: developer

.seealso: `PetscDualSpace`, `PETSCDUALSPACESUM`, `PETSCFEVECTOR`, `PetscDualSpaceSumGetInterleave()`
@*/
PetscErrorCode PetscDualSpaceSumSetInterleave(PetscDualSpace sp, PetscBool interleave_basis, PetscBool interleave_components)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscTryMethod(sp, "PetscDualSpaceSumSetInterleave_C", (PetscDualSpace, PetscBool, PetscBool), (sp, interleave_basis, interleave_components));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumSetInterleave_Sum(PetscDualSpace sp, PetscBool interleave_basis, PetscBool interleave_components)
{
  PetscDualSpace_Sum *sum = (PetscDualSpace_Sum *)sp->data;

  PetscFunctionBegin;
  sum->interleave_basis      = interleave_basis;
  sum->interleave_components = interleave_components;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDualSpaceSumGetInterleave - Get whether the basis functions and components of a uniform sum are interleaved

  Logically collective

  Input Parameter:
. sp - a `PetscDualSpace` of type `PETSCDUALSPACESUM`

  Output Parameters:
+ interleave_basis      - if `PETSC_TRUE`, the basis vectors of the subspaces are interleaved
- interleave_components - if `PETSC_TRUE` and the space concatenates components (`PetscDualSpaceSumGetConcatenate()`),
                          interleave the concatenated components

  Level: developer

.seealso: `PetscDualSpace`, `PETSCDUALSPACESUM`, `PETSCFEVECTOR`, `PetscDualSpaceSumSetInterleave()`
@*/
PetscErrorCode PetscDualSpaceSumGetInterleave(PetscDualSpace sp, PeOp PetscBool *interleave_basis, PeOp PetscBool *interleave_components)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  if (interleave_basis) PetscAssertPointer(interleave_basis, 2);
  if (interleave_components) PetscAssertPointer(interleave_components, 3);
  PetscTryMethod(sp, "PetscDualSpaceSumGetInterleave_C", (PetscDualSpace, PetscBool *, PetscBool *), (sp, interleave_basis, interleave_components));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumGetInterleave_Sum(PetscDualSpace sp, PetscBool *interleave_basis, PetscBool *interleave_components)
{
  PetscDualSpace_Sum *sum = (PetscDualSpace_Sum *)sp->data;

  PetscFunctionBegin;
  if (interleave_basis) *interleave_basis = sum->interleave_basis;
  if (interleave_components) *interleave_components = sum->interleave_components;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define PetscDualSpaceSumPassthrough(sp, func, ...) \
  do { \
    PetscDualSpace_Sum *sum = (PetscDualSpace_Sum *)sp->data; \
    PetscBool           is_uniform; \
    PetscCall(PetscDualSpaceSumIsUniform(sp, &is_uniform)); \
    if (is_uniform && sum->numSumSpaces > 0) { \
      PetscDualSpace subsp; \
      PetscCall(PetscDualSpaceSumGetSubspace(sp, 0, &subsp)); \
      PetscCall(func(subsp, __VA_ARGS__)); \
    } \
  } while (0)

static PetscErrorCode PetscDualSpaceLagrangeGetContinuity_Sum(PetscDualSpace sp, PetscBool *value)
{
  PetscFunctionBegin;
  PetscDualSpaceSumPassthrough(sp, PetscDualSpaceLagrangeGetContinuity, value);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceLagrangeSetContinuity_Sum(PetscDualSpace sp, PetscBool value)
{
  PetscFunctionBegin;
  PetscDualSpaceSumPassthrough(sp, PetscDualSpaceLagrangeSetContinuity, value);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceLagrangeGetTensor_Sum(PetscDualSpace sp, PetscBool *value)
{
  PetscFunctionBegin;
  PetscDualSpaceSumPassthrough(sp, PetscDualSpaceLagrangeGetTensor, value);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceLagrangeSetTensor_Sum(PetscDualSpace sp, PetscBool value)
{
  PetscFunctionBegin;
  PetscDualSpaceSumPassthrough(sp, PetscDualSpaceLagrangeSetTensor, value);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceLagrangeGetTrimmed_Sum(PetscDualSpace sp, PetscBool *value)
{
  PetscFunctionBegin;
  PetscDualSpaceSumPassthrough(sp, PetscDualSpaceLagrangeGetTrimmed, value);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceLagrangeSetTrimmed_Sum(PetscDualSpace sp, PetscBool value)
{
  PetscFunctionBegin;
  PetscDualSpaceSumPassthrough(sp, PetscDualSpaceLagrangeSetTrimmed, value);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceLagrangeGetUseMoments_Sum(PetscDualSpace sp, PetscBool *value)
{
  PetscFunctionBegin;
  PetscDualSpaceSumPassthrough(sp, PetscDualSpaceLagrangeGetUseMoments, value);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceLagrangeSetUseMoments_Sum(PetscDualSpace sp, PetscBool value)
{
  PetscFunctionBegin;
  PetscDualSpaceSumPassthrough(sp, PetscDualSpaceLagrangeSetUseMoments, value);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceLagrangeGetMomentOrder_Sum(PetscDualSpace sp, PetscInt *value)
{
  PetscFunctionBegin;
  PetscDualSpaceSumPassthrough(sp, PetscDualSpaceLagrangeGetMomentOrder, value);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceLagrangeSetMomentOrder_Sum(PetscDualSpace sp, PetscInt value)
{
  PetscFunctionBegin;
  PetscDualSpaceSumPassthrough(sp, PetscDualSpaceLagrangeSetMomentOrder, value);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceLagrangeGetNodeType_Sum(PetscDualSpace sp, PetscDTNodeType *node_type, PetscBool *include_endpoints, PetscReal *exponent)
{
  PetscFunctionBegin;
  PetscDualSpaceSumPassthrough(sp, PetscDualSpaceLagrangeGetNodeType, node_type, include_endpoints, exponent);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceLagrangeSetNodeType_Sum(PetscDualSpace sp, PetscDTNodeType node_type, PetscBool include_endpoints, PetscReal exponent)
{
  PetscFunctionBegin;
  PetscDualSpaceSumPassthrough(sp, PetscDualSpaceLagrangeSetNodeType, node_type, include_endpoints, exponent);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceInitialize_Sum(PetscDualSpace sp)
{
  PetscFunctionBegin;
  sp->ops->destroy              = PetscDualSpaceDestroy_Sum;
  sp->ops->view                 = PetscDualSpaceView_Sum;
  sp->ops->setfromoptions       = NULL;
  sp->ops->duplicate            = PetscDualSpaceDuplicate_Sum;
  sp->ops->setup                = PetscDualSpaceSetUp_Sum;
  sp->ops->createheightsubspace = NULL;
  sp->ops->createpointsubspace  = NULL;
  sp->ops->getsymmetries        = PetscDualSpaceGetSymmetries_Sum;
  sp->ops->apply                = PetscDualSpaceApplyDefault;
  sp->ops->applyall             = PetscDualSpaceApplyAllDefault;
  sp->ops->applyint             = PetscDualSpaceApplyInteriorDefault;
  sp->ops->createalldata        = PetscDualSpaceCreateAllDataDefault;
  sp->ops->createintdata        = PetscDualSpaceCreateInteriorDataDefault;

  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumGetNumSubspaces_C", PetscDualSpaceSumGetNumSubspaces_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumSetNumSubspaces_C", PetscDualSpaceSumSetNumSubspaces_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumGetSubspace_C", PetscDualSpaceSumGetSubspace_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumSetSubspace_C", PetscDualSpaceSumSetSubspace_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumGetConcatenate_C", PetscDualSpaceSumGetConcatenate_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumSetConcatenate_C", PetscDualSpaceSumSetConcatenate_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumGetInterleave_C", PetscDualSpaceSumGetInterleave_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumSetInterleave_C", PetscDualSpaceSumSetInterleave_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeGetContinuity_C", PetscDualSpaceLagrangeGetContinuity_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeSetContinuity_C", PetscDualSpaceLagrangeSetContinuity_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeGetMomentOrder_C", PetscDualSpaceLagrangeGetMomentOrder_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeSetMomentOrder_C", PetscDualSpaceLagrangeSetMomentOrder_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeGetNodeType_C", PetscDualSpaceLagrangeGetNodeType_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeSetNodeType_C", PetscDualSpaceLagrangeSetNodeType_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeGetTensor_C", PetscDualSpaceLagrangeGetTensor_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeSetTensor_C", PetscDualSpaceLagrangeSetTensor_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeGetTrimmed_C", PetscDualSpaceLagrangeGetTrimmed_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeSetTrimmed_C", PetscDualSpaceLagrangeSetTrimmed_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeGetUseMoments_C", PetscDualSpaceLagrangeGetUseMoments_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeSetUseMoments_C", PetscDualSpaceLagrangeSetUseMoments_Sum));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCDUALSPACESUM = "sum" - A `PetscDualSpace` object that encapsulates a sum of subspaces.

  Level: intermediate

  Note:
  That sum can either be direct or a concatenation. For example if A and B are spaces each with 2 components,
  the direct sum of A and B will also have 2 components while the concatenated sum will have 4 components. In both cases A and B must be defined over the
  same reference element.

.seealso: `PetscDualSpace`, `PetscDualSpaceType`, `PetscDualSpaceCreate()`, `PetscDualSpaceSetType()`, `PetscDualSpaceSumGetNumSubspaces()`, `PetscDualSpaceSumSetNumSubspaces()`,
          `PetscDualSpaceSumGetConcatenate()`, `PetscDualSpaceSumSetConcatenate()`, `PetscDualSpaceSumSetInterleave()`, `PetscDualSpaceSumGetInterleave()`
M*/
PETSC_EXTERN PetscErrorCode PetscDualSpaceCreate_Sum(PetscDualSpace sp)
{
  PetscDualSpace_Sum *sum;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscCall(PetscNew(&sum));
  sum->numSumSpaces = PETSC_DEFAULT;
  sp->data          = sum;
  sp->k             = PETSC_FORM_DEGREE_UNDEFINED;
  PetscCall(PetscDualSpaceInitialize_Sum(sp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDualSpaceCreateSum - Create a finite element dual basis that is the sum of other dual bases

  Collective

  Input Parameters:
+ numSubspaces - the number of spaces that will be added together
. subspaces    - an array of length `numSubspaces` of spaces
- concatenate  - if `PETSC_FALSE`, the sum-space has the same components as the individual dual spaces (`PetscDualSpaceGetNumComponents()`); if `PETSC_TRUE`, the individual components are concatenated to create a dual space with more components

  Output Parameter:
. sumSpace - a `PetscDualSpace` of type `PETSCDUALSPACESUM`

  Level: advanced

.seealso: `PetscDualSpace`, `PETSCDUALSPACESUM`, `PETSCSPACESUM`
@*/
PetscErrorCode PetscDualSpaceCreateSum(PetscInt numSubspaces, const PetscDualSpace subspaces[], PetscBool concatenate, PetscDualSpace *sumSpace)
{
  PetscInt i, Nc = 0;

  PetscFunctionBegin;
  PetscCall(PetscDualSpaceCreate(PetscObjectComm((PetscObject)subspaces[0]), sumSpace));
  PetscCall(PetscDualSpaceSetType(*sumSpace, PETSCDUALSPACESUM));
  PetscCall(PetscDualSpaceSumSetNumSubspaces(*sumSpace, numSubspaces));
  PetscCall(PetscDualSpaceSumSetConcatenate(*sumSpace, concatenate));
  for (i = 0; i < numSubspaces; ++i) {
    PetscInt sNc;

    PetscCall(PetscDualSpaceSumSetSubspace(*sumSpace, i, subspaces[i]));
    PetscCall(PetscDualSpaceGetNumComponents(subspaces[i], &sNc));
    if (concatenate) Nc += sNc;
    else Nc = sNc;

    if (i == 0) {
      DM dm;

      PetscCall(PetscDualSpaceGetDM(subspaces[i], &dm));
      PetscCall(PetscDualSpaceSetDM(*sumSpace, dm));
    }
  }
  PetscCall(PetscDualSpaceSetNumComponents(*sumSpace, Nc));
  PetscFunctionReturn(PETSC_SUCCESS);
}
