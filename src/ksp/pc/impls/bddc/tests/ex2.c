static char help[] = "Tests BDDC Nedelec support and user-defined primal vertices.\n\n";

#include <petsc/private/pcbddcimpl.h>

int main(int argc, char **args)
{
  Mat                    A, local, G;
  ISLocalToGlobalMapping map;
  IS                     primal = NULL, stored;
  KSP                    ksp;
  PC                     pc;
  PC_BDDC               *bddc;
  Vec                    x, b, exact, residual;
  PetscInt              *indices, *eoffset, *voffset, *orders, *selected, *requested, *input;
  PetscInt               start, end, n, nedge, nlocal, nbase, nmesh, nv, nselected, nrequested = 0, ninput = 0, order = 1, copies = 1, field = PETSC_DECIDE, unselected = -1;
  PetscMPIInt            rank, size, active;
  PetscBool             *expected;
  PetscBool              local_primal = PETSC_FALSE, closed_loop = PETSC_FALSE, periodic = PETSC_FALSE, branch = PETSC_FALSE, mixed_order = PETSC_FALSE;
  PetscBool              faces = PETSC_FALSE, permuted = PETSC_FALSE, empty_rank = PETSC_FALSE, distributed_primal = PETSC_FALSE, no_primal = PETSC_FALSE, other_only = PETSC_FALSE;
  PetscBool              explicit_fields = PETSC_FALSE, gradient_global = PETSC_TRUE, conforming = PETSC_TRUE;
  PetscBool              dirichlet = PETSC_FALSE, neumann = PETSC_FALSE, nullspace = PETSC_FALSE, nullspace_explicit = PETSC_FALSE;
  PetscBool              near_nullspace = PETSC_FALSE, near_nullspace_explicit = PETSC_FALSE, transpose = PETSC_TRUE, setprimal = PETSC_FALSE, check_multilevel = PETSC_FALSE, flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-local_primal", &local_primal, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-closed_loop", &closed_loop, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-periodic", &periodic, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-branch", &branch, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-mixed_order", &mixed_order, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-faces", &faces, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-permuted", &permuted, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-empty_rank", &empty_rank, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-distributed_primal", &distributed_primal, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-no_primal", &no_primal, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-other_only", &other_only, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-explicit_fields", &explicit_fields, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-gradient_global", &gradient_global, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-conforming", &conforming, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-dirichlet", &dirichlet, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-neumann", &neumann, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-nullspace", &nullspace, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-nullspace_explicit", &nullspace_explicit, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-near_nullspace", &near_nullspace, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-near_nullspace_explicit", &near_nullspace_explicit, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-transpose", &transpose, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-check_multilevel", &check_multilevel, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-pc_bddc_nedelec_field_primal", &setprimal, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-order", &order, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-copies", &copies, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-field", &field, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-unselected_edge", &unselected, NULL));
  PetscCheck(order >= 1 && order <= 3, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Use orders 1 through 3");
  PetscCheck(copies > 0, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "The number of local subdomains must be positive");
  active = size - (empty_rank ? 1 : 0);
  PetscCheck(active >= 2, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "At least two nonempty subdomains are required");
  if (!gradient_global) explicit_fields = PETSC_TRUE;

  /* Seven fine edges form a path, ring, branched tree, or path plus a separate triangle.
     Optional face edges are shared only by ranks 0 and 1 and meet the path at node 2. */
  nmesh = faces ? 11 : 7;
  nv    = faces ? 12 : 8;
  PetscCall(PetscMalloc3(nmesh + 1, &eoffset, nmesh + 1, &voffset, nmesh, &orders));
  eoffset[0] = 0;
  for (PetscInt e = 0; e < nmesh; e++) {
    orders[e]      = mixed_order ? 1 + e % 3 : order;
    eoffset[e + 1] = eoffset[e] + orders[e];
    voffset[e]     = nv;
    nv += orders[e] - 1;
  }
  nedge = eoffset[nmesh];
  PetscCheck(unselected >= -1 && unselected < nmesh, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Invalid unselected mesh edge");
  if (unselected >= 0) unselected = eoffset[unselected] + orders[unselected] / 2;

  /* Local numbering interleaves fields and may differ on each rank. */
  nbase  = rank < active ? (rank < 2 ? nedge : eoffset[7]) + 4 : 0;
  nlocal = copies * nbase;
  PetscCall(PetscMalloc1(nlocal, &indices));
  if (nlocal) {
    PetscInt ne = nbase - 4;

    indices[0] = nedge + 2;
    for (PetscInt i = 0; i < ne; i++) indices[i + 1] = ne - i - 1;
    indices[ne + 1] = nedge;
    indices[ne + 2] = nedge + 1;
    indices[ne + 3] = nedge + 3 + rank;
    if (permuted) {
      for (PetscInt i = 0; i < nbase; i++) {
        PetscInt j = (3 * i + rank + 1) % nbase, tmp = indices[i];

        indices[i] = indices[j];
        indices[j] = tmp;
      }
    }
    for (PetscInt c = 1; c < copies; c++) PetscCall(PetscArraycpy(indices + c * nbase, indices, nbase));
  }
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, nlocal, indices, PETSC_COPY_VALUES, &map));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, nedge + 3 + active, nedge + 3 + active));
  PetscCall(MatSetType(A, MATIS));
  PetscCall(MatISSetAllowRepeated(A, (PetscBool)(copies > 1)));
  PetscCall(MatSetLocalToGlobalMapping(A, map, map));
  PetscCall(MatISGetLocalMat(A, &local));
  PetscCall(MatSeqAIJSetPreallocation(local, nbase, NULL));
  for (PetscInt i = 0; i < nlocal; i++) {
    for (PetscInt j = i / nbase * nbase; j < (i / nbase + 1) * nbase; j++) PetscCall(MatSetValue(local, i, j, i == j ? 2.0 + 0.1 * i : -1.0 / nbase, INSERT_VALUES));
  }
  if (copies > 1) {
    PetscInt *sizes;

    PetscCall(PetscMalloc1(copies, &sizes));
    for (PetscInt c = 0; c < copies; c++) sizes[c] = nbase;
    PetscCall(MatSetVariableBlockSizes(local, copies, sizes));
    PetscCall(PetscFree(sizes));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  if (dirichlet) {
    PetscInt *rows;

    PetscCall(PetscMalloc1(orders[0], &rows));
    for (PetscInt i = 0; i < orders[0]; i++) rows[i] = i;
    PetscCall(MatZeroRowsColumns(A, orders[0], rows, 1.0, NULL, NULL));
    PetscCall(PetscFree(rows));
  }

  /* Differentiate degree-p Lagrange polynomials at p points on each mesh edge.
     All rows on an edge have the same p+1 nodal columns, without stored zeros. */
  PetscCall(MatGetLocalSize(A, &n, NULL));
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, gradient_global ? n : PETSC_DECIDE, PETSC_DECIDE, gradient_global ? nedge + 3 + active : nedge, nv, 4, NULL, 4, NULL, &G));
  PetscCall(MatGetOwnershipRange(G, &start, &end));
  for (PetscInt e = 0; e < nmesh; e++) {
    PetscInt p = orders[e], v0 = e, v1 = e + 1;

    if (closed_loop && e >= 4 && e < 7) {
      v0 = e + 1;
      v1 = e == 6 ? 5 : e + 2;
    }
    if (periodic && e == 6) v1 = 0;
    if (branch && e >= 4 && e < 7) v0 = e == 4 ? 2 : e;
    if (e >= 7) {
      v0 = e == 7 ? 2 : e;
      v1 = e + 1;
    }
    for (PetscInt r = 0; r < p; r++) {
      PetscInt  row = eoffset[e] + r;
      PetscReal t   = p == 1 ? 0.5 : (PetscReal)r / (p - 1);

      if (row < start || row >= end) continue;
      for (PetscInt j = 0; j <= p; j++) {
        PetscInt  col   = j == 0 ? v0 : (j == p ? v1 : voffset[e] + j - 1);
        PetscReal value = 0.0, denominator = 1.0;

        for (PetscInt k = 0; k <= p; k++)
          if (k != j) denominator *= (PetscReal)(j - k) / p;
        for (PetscInt k = 0; k <= p; k++) {
          PetscReal term = 1.0;

          if (k == j) continue;
          for (PetscInt l = 0; l <= p; l++)
            if (l != j && l != k) term *= t - (PetscReal)l / p;
          value += term / denominator;
        }
        PetscCheck(value != 0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Zero entry in test gradient");
        PetscCall(MatSetValue(G, row, col, value, INSERT_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(G, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY));
  if (nullspace || nullspace_explicit) {
    MatNullSpace nsp;
    Vec          nodes, edges;
    PetscReal    norm;

    PetscCall(MatCreateVecs(G, &nodes, &edges));
    PetscCall(VecSet(nodes, 1.0));
    PetscCall(MatMult(G, nodes, edges));
    PetscCall(VecNorm(edges, NORM_INFINITY, &norm));
    PetscCheck(norm < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Test gradient does not annihilate constants");
    if (nullspace_explicit) {
      PetscCall(VecNormalize(nodes, NULL));
      PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, &nodes, &nsp));
    } else PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nsp));
    PetscCall(VecDestroy(&nodes));
    PetscCall(VecDestroy(&edges));
    PetscCall(MatSetNullSpace(G, nsp));
    PetscCall(MatNullSpaceDestroy(&nsp));
  }
  if (near_nullspace || near_nullspace_explicit) {
    MatNullSpace nsp;

    if (near_nullspace_explicit) {
      Vec mode;

      PetscCall(MatCreateVecs(A, &mode, NULL));
      PetscCall(VecSet(mode, 1.0));
      PetscCall(VecNormalize(mode, NULL));
      PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, &mode, &nsp));
      PetscCall(VecDestroy(&mode));
    } else PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nsp));
    PetscCall(MatSetNearNullSpace(A, nsp));
    PetscCall(MatNullSpaceDestroy(&nsp));
  }

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCBDDC));
  if (explicit_fields) {
    IS        fields[2];
    PetscInt *rows;
    PetscInt  counts[2] = {0, 0};

    PetscCall(PetscMalloc1(nlocal, &rows));
    for (PetscInt i = 0; i < nlocal; i++)
      if (indices[i] >= nedge) rows[counts[0]++] = i;
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, counts[0], rows, PETSC_COPY_VALUES, &fields[0]));
    for (PetscInt i = 0; i < nlocal; i++)
      if (indices[i] < nedge) rows[counts[1]++] = i;
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, counts[1], rows, PETSC_COPY_VALUES, &fields[1]));
    PetscCall(PCBDDCSetDofsSplittingLocal(pc, 2, fields));
    PetscCall(ISDestroy(&fields[0]));
    PetscCall(ISDestroy(&fields[1]));
    PetscCall(PetscFree(rows));
    field = 1;
  }
  PetscCall(PCBDDCSetDiscreteGradient(pc, G, mixed_order ? 0 : order, field, gradient_global, conforming));
  if (dirichlet || neumann) {
    IS        boundary;
    PetscInt *rows;
    PetscInt  nr = 0;

    PetscCall(PetscMalloc1(nlocal, &rows));
    if (dirichlet) {
      for (PetscInt i = 0; i < nlocal; i++)
        if (indices[i] < eoffset[1]) rows[nr++] = i;
      PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, nr, rows, PETSC_COPY_VALUES, &boundary));
      PetscCall(PCBDDCSetDirichletBoundariesLocal(pc, boundary));
      PetscCall(ISDestroy(&boundary));
    }
    if (neumann) {
      nr = 0;
      for (PetscInt i = 0; i < nlocal; i++)
        if (indices[i] >= eoffset[7] && indices[i] < nedge) rows[nr++] = i;
      PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, nr, rows, PETSC_COPY_VALUES, &boundary));
      PetscCall(PCBDDCSetNeumannBoundariesLocal(pc, boundary));
      PetscCall(ISDestroy(&boundary));
    }
    PetscCall(PetscFree(rows));
  }

  /* Include another field, select a non-first edge dof, and allow unsorted duplicates.
     Global input may come from a process that does not own any of the selected dofs. */
  PetscCall(PetscMalloc1(nedge + 3, &selected));
  nselected = nedge + 3;
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-primal_edges", selected, &nselected, &flg));
  if (!flg) {
    selected[0] = 3;
    selected[1] = 8;
    nselected   = faces ? 2 : 1;
  }
  if (other_only) nselected = 0;
  PetscCall(PetscMalloc2(nselected + 1, &requested, nselected + 1, &input));
  PetscCall(PetscCalloc1(nedge + 3, &expected));
  if (!no_primal) {
    requested[nrequested++] = nedge;
    expected[nedge]         = PETSC_TRUE;
    for (PetscInt i = 0; i < nselected; i++) {
      PetscInt e = selected[i], row;

      PetscCheck(e >= 0 && e < nmesh, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Invalid mesh edge %" PetscInt_FMT, e);
      row                     = eoffset[e] + orders[e] / 2;
      requested[nrequested++] = row;
      expected[row]           = PETSC_TRUE;
      if (copies * active > 2 && !setprimal)
        for (PetscInt j = eoffset[e]; j < eoffset[e + 1]; j++) expected[j] = PETSC_TRUE;
    }
    for (PetscInt i = 0; i < nrequested; i++) {
      PetscInt    row         = requested[i], localrow;
      PetscMPIInt contributor = distributed_primal ? (local_primal ? (i % 2 ? 1 : 0) : size - 1) : 0;

      if (rank != contributor) continue;
      if (local_primal) {
        PetscCall(ISGlobalToLocalMappingApply(map, IS_GTOLM_MASK, 1, &row, NULL, &localrow));
        PetscCheck(localrow >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Contributor does not own the requested primal dof");
        input[ninput++] = localrow;
      } else input[ninput++] = row;
    }
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, ninput, input, PETSC_COPY_VALUES, &primal));
    if (local_primal) PetscCall(PCBDDCSetPrimalVerticesLocalIS(pc, primal));
    else PetscCall(PCBDDCSetPrimalVerticesIS(pc, primal));
  }
  if (setprimal && copies * active > 2)
    for (PetscInt i = 0; i < eoffset[7]; i++) expected[i] = PETSC_TRUE;
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));

  if (primal && !local_primal) {
    PetscCall(PCBDDCGetPrimalVerticesIS(pc, &stored));
    PetscCheck(stored, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Global user primal vertices were lost during setup");
    PetscCall(ISEqual(primal, stored, &flg));
    PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Global user primal vertices changed during setup");
  }
  PetscCall(ISDestroy(&primal));

  /* Check the stored vertices and actual point constraints in each rank's numbering. */
  PetscCall(PCBDDCGetPrimalVerticesLocalIS(pc, &stored));
  bddc = (PC_BDDC *)pc->data;
  if (check_multilevel) {
    if (bddc->coarse_ksp) {
      PC        coarsepc;
      PetscBool isbddc;

      PetscCall(KSPGetPC(bddc->coarse_ksp, &coarsepc));
      PetscCall(PetscObjectTypeCompare((PetscObject)coarsepc, PCBDDC, &isbddc));
      PetscCheck(isbddc, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing coarse BDDC level");
      PetscCheck(((PC_BDDC *)coarsepc->data)->discretegradient, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing coarse discrete gradient");
    }
    PetscCheck(bddc->nedcG, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing generated coarse gradient");
  }
  PetscCall(MatGetSize(bddc->ConstraintMatrix, &n, NULL));
  for (PetscInt localrow = 0; localrow < nlocal; localrow++) {
    PetscInt  i     = indices[localrow], pos;
    PetscBool found = PETSC_FALSE;

    if (i >= nedge + 3 || !expected[i]) continue;
    PetscCall(ISLocate(stored, localrow, &pos));
    PetscCheck(pos >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Primal dof %" PetscInt_FMT " was lost", i);
    for (PetscInt j = 0; j < n; j++) {
      const PetscInt    *cols;
      const PetscScalar *vals;
      PetscInt           nc;

      PetscCall(MatGetRow(bddc->ConstraintMatrix, j, &nc, &cols, &vals));
      if (nc == 1 && cols[0] == localrow && vals[0] == 1.0) found = PETSC_TRUE;
      PetscCall(MatRestoreRow(bddc->ConstraintMatrix, j, &nc, &cols, &vals));
    }
    PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing point constraint for primal dof %" PetscInt_FMT, i);
  }
  PetscCheck((PetscBool)!!bddc->user_ChangeOfBasisMatrix == (PetscBool)(copies * active > 2 && !setprimal), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Unexpected Nedelec change-of-basis path");
  if (bddc->user_ChangeOfBasisMatrix) {
    PetscCall(MatGetOwnershipRange(bddc->user_ChangeOfBasisMatrix, &start, &end));
    for (PetscInt i = start; i < PetscMin(end, nedge + 3); i++) {
      const PetscInt    *cols;
      const PetscScalar *vals;
      PetscInt           nc;
      PetscBool          identity = PETSC_TRUE, diagonal = PETSC_FALSE;

      if (!expected[i] && i != unselected) continue;
      PetscCall(MatGetRow(bddc->user_ChangeOfBasisMatrix, i, &nc, &cols, &vals));
      for (PetscInt j = 0; j < nc; j++) {
        if (cols[j] == i) {
          diagonal = PETSC_TRUE;
          if (PetscAbsScalar(vals[j] - 1.0) > PETSC_SMALL) identity = PETSC_FALSE;
        } else if (PetscAbsScalar(vals[j]) > PETSC_SMALL) identity = PETSC_FALSE;
      }
      PetscCall(MatRestoreRow(bddc->user_ChangeOfBasisMatrix, i, &nc, &cols, &vals));
      if (expected[i]) PetscCheck(identity && diagonal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Change of basis modifies primal dof %" PetscInt_FMT, i);
      else PetscCheck(!identity || !diagonal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Change of basis leaves unselected edge dof %" PetscInt_FMT " unchanged", i);
    }
  }

  /* Two nonconstant right-hand sides verify the solve and reuse of the setup. */
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecDuplicate(x, &exact));
  PetscCall(VecDuplicate(x, &residual));
  PetscCall(VecGetOwnershipRange(exact, &start, &end));
  for (PetscInt solve = 0; solve < 2; solve++) {
    PetscScalar *values;
    PetscReal    error, norm;

    PetscCall(VecGetArray(exact, &values));
    for (PetscInt i = start; i < end; i++) values[i - start] = PetscSinReal((solve + 1) * 0.17 * (i + 1)) + 0.03 * (i % 5);
    PetscCall(VecRestoreArray(exact, &values));
    PetscCall(KSPSetUp(ksp));
    if (solve && transpose) {
      PetscCall(MatMultTranspose(A, exact, b));
      PetscCall(KSPSolveTranspose(ksp, b, x));
      PetscCall(MatMultTranspose(A, x, residual));
    } else {
      PetscCall(MatMult(A, exact, b));
      PetscCall(KSPSolve(ksp, b, x));
      PetscCall(MatMult(A, x, residual));
    }
    PetscCall(VecAXPY(residual, -1.0, b));
    PetscCall(VecNorm(residual, NORM_INFINITY, &error));
    PetscCall(VecNorm(b, NORM_INFINITY, &norm));
    PetscCheck(error < PETSC_SMALL * norm, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Relative residual %g", (double)(error / norm));
    PetscCall(VecAXPY(x, -1.0, exact));
    PetscCall(VecNorm(x, NORM_INFINITY, &error));
    PetscCheck(error < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Solution error %g", (double)error);
  }

  PetscCall(PetscFree(indices));
  PetscCall(PetscFree3(eoffset, voffset, orders));
  PetscCall(PetscFree(selected));
  PetscCall(PetscFree2(requested, input));
  PetscCall(PetscFree(expected));
  PetscCall(ISLocalToGlobalMappingDestroy(&map));
  PetscCall(VecDestroy(&residual));
  PetscCall(VecDestroy(&exact));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&G));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    nsize: 3
    requires: double
    args: -local_primal {{0 1}} -ksp_error_if_not_converged -ksp_rtol 1e-12
    output_file: output/empty.out
    test:
      suffix: path
      args: -unselected_edge 1 -pc_bddc_nedelec_field_primal {{0 1}}
    test:
      suffix: loop
      args: -closed_loop
    test:
      suffix: monolithic
      args: -field 0
    test:
      suffix: quadratic
      args: -order 2 -unselected_edge 1 -periodic {{0 1}} -pc_bddc_nedelec_order {{0 2}}
    test:
      suffix: cubic
      args: -order 3 -permuted -distributed_primal -primal_edges 5,3,5,1,3
    test:
      suffix: mixed_order
      args: -mixed_order -periodic -permuted -explicit_fields -distributed_primal
    test:
      suffix: fields
      args: -order 2 -explicit_fields -gradient_global {{0 1}} -permuted
    test:
      suffix: faces
      args: -order 2 -faces -permuted -distributed_primal -pc_bddc_nedelec_field_primal {{0 1}}
    test:
      suffix: boundaries
      args: -order 2 -faces -dirichlet -neumann -permuted -conforming {{0 1}}
    test:
      suffix: branch
      args: -order 2 -branch -permuted
    test:
      suffix: all_primal
      args: -order 2 -primal_edges 6,4,2,0,5,3,1 -permuted -nullspace
    test:
      suffix: other_field
      args: -order 2 -other_only -permuted -distributed_primal
    test:
      suffix: near_nullspace
      args: -order 2 -near_nullspace -faces
    test:
      suffix: combined_nullspaces
      args: -order 2 -nullspace -near_nullspace -near_nullspace_explicit {{0 1}}
    test:
      suffix: nullspace_fields
      args: -order 2 -nullspace -nullspace_explicit {{0 1}} -explicit_fields -gradient_global {{0 1}} -permuted -distributed_primal
    test:
      suffix: multilevel
      nsize: 4
      args: -order 2 -nullspace -nullspace_explicit {{0 1}} -check_multilevel -pc_bddc_levels 2 -pc_bddc_coarse_eqs_limit 0 -pc_bddc_coarsening_ratio 2 -pc_bddc_aggregator_0_mat_partitioning_type average
    test:
      suffix: no_coarse_edges
      nsize: 2
      args: -order 2 -faces -permuted -distributed_primal
    test:
      suffix: empty_rank
      nsize: 4
      args: -order 2 -empty_rank -permuted -distributed_primal -nullspace
    test:
      suffix: repeated
      args: -order 2 -copies 2 -explicit_fields -permuted -distributed_primal

  test:
    suffix: no_user
    nsize: 3
    requires: double
    output_file: output/empty.out
    args: -order 2 -no_primal -explicit_fields -nullspace {{0 1}} -ksp_error_if_not_converged -ksp_rtol 1e-12

TEST*/
