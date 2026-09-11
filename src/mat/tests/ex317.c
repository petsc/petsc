static char help[] = "Tests MatGetLocalSubMatrix() on a MATIS with fields of different block sizes.\n\n";

#include <petscmat.h>

/* Deterministic entry k of the element matrix coupling node i of field fi with node j of field fj */
static PetscScalar ElementValue(PetscInt fi, PetscInt i, PetscInt fj, PetscInt j, PetscInt k)
{
  return (PetscScalar)(1 + ((3 * fi + 5 * i + 7 * fj + 11 * j + 13 * k) % 17));
}

/*
  The contract MatGetLocalSubMatrix() owes its caller, which MATIS and MATLOCALREF both honour: the
  submatrix is sized by its index sets and carries their block size, on the matrix and on its local to
  global maps alike. Which rows and columns those maps reach is left to the assembly below, where the same
  operator is built twice through them, a node at a time and a degree of freedom at a time.
*/
static PetscErrorCode CheckLocalSubMatrix(Mat sub, IS row, IS col)
{
  ISLocalToGlobalMapping rl2g, cl2g;
  PetscInt               nrl, ncl, rbs, cbs, m, n, mbs, nbs;

  PetscFunctionBeginUser;
  PetscCall(ISGetLocalSize(row, &nrl));
  PetscCall(ISGetLocalSize(col, &ncl));
  PetscCall(ISGetBlockSize(row, &rbs));
  PetscCall(ISGetBlockSize(col, &cbs));
  PetscCall(MatGetLocalSize(sub, &m, &n));
  PetscCheck(m == nrl && n == ncl, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Submatrix has local size %" PetscInt_FMT " x %" PetscInt_FMT ", its index sets have %" PetscInt_FMT " x %" PetscInt_FMT, m, n, nrl, ncl);
  PetscCall(MatGetBlockSizes(sub, &mbs, &nbs));
  PetscCheck(mbs == rbs && nbs == cbs, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Submatrix has block size %" PetscInt_FMT " x %" PetscInt_FMT ", its index sets have %" PetscInt_FMT " x %" PetscInt_FMT, mbs, nbs, rbs, cbs);
  PetscCall(MatGetLocalToGlobalMapping(sub, &rl2g, &cl2g));
  PetscCall(ISLocalToGlobalMappingGetBlockSize(rl2g, &mbs));
  PetscCall(ISLocalToGlobalMappingGetBlockSize(cl2g, &nbs));
  PetscCheck(mbs == rbs && nbs == cbs, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Maps of the submatrix have block size %" PetscInt_FMT " x %" PetscInt_FMT ", its index sets have %" PetscInt_FMT " x %" PetscInt_FMT, mbs, nbs, rbs, cbs);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Assemble a two field operator through the local submatrices of a MATIS.
  With blocked, each field is addressed one node at a time, through index sets that carry its block
  size; otherwise every degree of freedom is addressed individually. Both paths insert the same values.
*/
static PetscErrorCode AssembleMixed(Mat A, const PetscInt bs[], const PetscInt nn[], const PetscInt off[], PetscBool blocked)
{
  IS           is[2];
  Mat          sub[2][2];
  PetscScalar *vals;
  PetscInt    *rows, *cols;
  PetscInt     fi, fj, i, j, k, mbs;

  PetscFunctionBeginUser;
  mbs = PetscMax(bs[0], bs[1]);
  PetscCall(PetscMalloc3(mbs, &rows, mbs, &cols, mbs * mbs, &vals));
  for (fi = 0; fi < 2; fi++) {
    PetscCall(ISCreateStride(PETSC_COMM_SELF, nn[fi] * bs[fi], off[fi], 1, &is[fi]));
    if (blocked) PetscCall(ISSetBlockSize(is[fi], bs[fi]));
  }
  for (fi = 0; fi < 2; fi++) {
    for (fj = 0; fj < 2; fj++) {
      PetscCall(MatGetLocalSubMatrix(A, is[fi], is[fj], &sub[fi][fj]));
      PetscCall(CheckLocalSubMatrix(sub[fi][fj], is[fi], is[fj]));
    }
  }

  /* every field couples its local nodes in a chain, and node i of a field to node i of the other one */
  for (fi = 0; fi < 2; fi++) {
    for (fj = 0; fj < 2; fj++) {
      for (i = 0; i < nn[fi]; i++) {
        for (j = PetscMax(i - 1, 0); j <= PetscMin(fi == fj ? i + 1 : i, nn[fj] - 1); j++) {
          for (k = 0; k < bs[fi] * bs[fj]; k++) vals[k] = ElementValue(fi, i, fj, j, k);
          /* the blocked path reaches the diagonal blocks by node and the off diagonal ones by degree of freedom */
          if (blocked && fi == fj) PetscCall(MatSetValuesBlockedLocal(sub[fi][fj], 1, &i, 1, &j, vals, ADD_VALUES));
          else {
            for (k = 0; k < bs[fi]; k++) rows[k] = i * bs[fi] + k;
            for (k = 0; k < bs[fj]; k++) cols[k] = j * bs[fj] + k;
            PetscCall(MatSetValuesLocal(sub[fi][fj], bs[fi], rows, bs[fj], cols, vals, ADD_VALUES));
          }
        }
      }
    }
  }

  for (fi = 0; fi < 2; fi++) {
    for (fj = 0; fj < 2; fj++) PetscCall(MatRestoreLocalSubMatrix(A, is[fi], is[fj], &sub[fi][fj]));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  for (fi = 0; fi < 2; fi++) PetscCall(ISDestroy(&is[fi]));
  PetscCall(PetscFree3(rows, cols, vals));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat                    A, B, A2, B2;
  ISLocalToGlobalMapping map;
  PetscReal              nrm, ref;
  PetscInt              *idxs, bs[2], nn[2], off[2], goff[2], gn, n, N, nl = 4, pbs, f, i, j;
  PetscMPIInt            rank, size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  bs[0] = 2;
  bs[1] = 3;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-bs0", &bs[0], NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-bs1", &bs[1], NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nl", &nl, NULL));
  PetscCheck(nl > 2, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Need at least 3 local nodes per subdomain");

  /*
    Two fields laid out as in a mixed finite element space: neighboring subdomains share one node of
    each field, and the second field starts wherever the first one ends. Neither that offset nor the
    global size need be a multiple of the block size of a field, which is what its local submatrices
    have to cope with.
  */
  nn[0]   = nl;
  nn[1]   = nl - 1;
  off[0]  = 0;
  off[1]  = nn[0] * bs[0];
  goff[0] = 0;
  goff[1] = (size * (nn[0] - 1) + 1) * bs[0];
  n       = off[1] + nn[1] * bs[1];
  N       = goff[1] + (size * (nn[1] - 1) + 1) * bs[1];
  PetscCall(PetscMalloc1(n, &idxs));
  for (f = 0; f < 2; f++) {
    for (i = 0; i < nn[f]; i++) {
      gn = rank * (nn[f] - 1) + i;
      for (j = 0; j < bs[f]; j++) idxs[off[f] + i * bs[f] + j] = goff[f] + gn * bs[f] + j;
    }
  }
  /*
    The whole space is blocked only when the two fields agree on a block size, and then the local
    submatrices can be reached through the ordinary blocked maps; with fields of different block sizes
    they cannot, and the two representations of a local submatrix are both exercised by the tests below.
  */
  pbs = bs[0] == bs[1] ? bs[0] : 1;
  for (i = 0; i < n / pbs; i++) idxs[i] = idxs[i * pbs] / pbs;
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, pbs, n / pbs, idxs, PETSC_OWN_POINTER, &map));

  PetscCall(MatCreateIS(PETSC_COMM_WORLD, pbs, PETSC_DECIDE, PETSC_DECIDE, N, N, map, map, &A));
  PetscCall(MatISSetPreallocation(A, 3 * (bs[0] + bs[1]), NULL, 3 * (bs[0] + bs[1]), NULL));
  PetscCall(MatCreateIS(PETSC_COMM_WORLD, pbs, PETSC_DECIDE, PETSC_DECIDE, N, N, map, map, &B));
  PetscCall(MatISSetPreallocation(B, 3 * (bs[0] + bs[1]), NULL, 3 * (bs[0] + bs[1]), NULL));
  PetscCall(AssembleMixed(A, bs, nn, off, PETSC_TRUE));
  PetscCall(AssembleMixed(B, bs, nn, off, PETSC_FALSE));

  PetscCall(MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, &A2));
  PetscCall(MatConvert(B, MATAIJ, MAT_INITIAL_MATRIX, &B2));
  PetscCall(MatNorm(B2, NORM_INFINITY, &ref));
  PetscCall(MatAXPY(A2, -1.0, B2, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatNorm(A2, NORM_INFINITY, &nrm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Reference operator is nonzero: %s\n", ref > 0.0 ? "yes" : "no"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Blocked and scalar assembly agree: %s\n", nrm < PETSC_SMALL * ref ? "yes" : "no"));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A2));
  PetscCall(MatDestroy(&B2));
  PetscCall(ISLocalToGlobalMappingDestroy(&map));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: {{1 3}}
      args: -bs0 {{1 2 3}} -bs1 {{1 2 3}}
      output_file: output/ex317_1.out

TEST*/
