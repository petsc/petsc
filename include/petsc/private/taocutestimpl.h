#pragma once

#include <petsctao.h>
EXTERN_C_BEGIN
#include <cutest.h>
EXTERN_C_END

typedef struct {
  integer      n, lh;
  integer     *rows, *cols;
  PetscScalar *values;
  Vec          hessian_x;
  logical      goth;
} CUTEstCtx;

#define PetscCallCUTEst(routine, ...) \
  do { \
    integer status; \
    routine(&status, __VA_ARGS__); \
    PetscCheck(!status, PETSC_COMM_SELF, PETSC_ERR_LIB, "%s() returned CUTEst status %d", #routine, status); \
  } while (0)

static PetscErrorCode CUTEstFormObjective(Tao tao, Vec X, PetscReal *f, PetscCtx ctx)
{
  CUTEstCtx         *user = (CUTEstCtx *)ctx;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  user->goth = false;
  PetscCall(VecGetArrayRead(X, &x));
  PetscCallCUTEst(CUTEST_ufn, &user->n, x, f);
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CUTEstFormGradient(Tao tao, Vec X, Vec G, PetscCtx ctx)
{
  CUTEstCtx         *user = (CUTEstCtx *)ctx;
  const PetscScalar *x;
  PetscScalar       *g;

  PetscFunctionBeginUser;
  user->goth = false;
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArrayWrite(G, &g));
  PetscCallCUTEst(CUTEST_ugr, &user->n, x, g);
  PetscCall(VecRestoreArrayWrite(G, &g));
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CUTEstFormObjectiveGradient(Tao tao, Vec X, PetscReal *f, Vec G, PetscCtx ctx)
{
  CUTEstCtx         *user = (CUTEstCtx *)ctx;
  const PetscScalar *x;
  PetscScalar       *g;
  logical            grad = true;

  PetscFunctionBeginUser;
  user->goth = false;
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArrayWrite(G, &g));
  PetscCallCUTEst(CUTEST_uofg, &user->n, x, f, g, &grad);
  PetscCall(VecRestoreArrayWrite(G, &g));
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CUTEstHessianMult(Mat H, Vec V, Vec W)
{
  CUTEstCtx         *user;
  const PetscScalar *x, *v;
  PetscScalar       *w;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(H, &user));
  PetscCall(VecGetArrayRead(user->hessian_x, &x));
  PetscCall(VecGetArrayRead(V, &v));
  PetscCall(VecGetArrayWrite(W, &w));
  PetscCallCUTEst(CUTEST_uhprod, &user->n, &user->goth, x, v, w);
  user->goth = true;
  PetscCall(VecRestoreArrayWrite(W, &w));
  PetscCall(VecRestoreArrayRead(V, &v));
  PetscCall(VecRestoreArrayRead(user->hessian_x, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CUTEstFormHessian(Tao tao, Vec X, Mat H, Mat Hpre, PetscCtx ctx)
{
  CUTEstCtx *user = (CUTEstCtx *)ctx;

  PetscFunctionBeginUser;
  user->goth = false;
  if (user->hessian_x) { /* MATSHELL */
    PetscCall(VecCopy(X, user->hessian_x));
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  } else {
    const PetscScalar *x;
    PetscCount         nz;
    integer            nnz;

    PetscCall(VecGetArrayRead(X, &x));
    PetscCallCUTEst(CUTEST_ush, &user->n, x, &nnz, &user->lh, user->values, user->rows, user->cols);
    PetscCall(VecRestoreArrayRead(X, &x));
    /* CUTEST_ush() uses the fixed ordering from CUTEST_ushp(). Preserve its first nnz values and append the mirrored entries. */
    nz = nnz;
    for (PetscInt k = 0; k < nnz; ++k) {
      if (user->rows[k] != user->cols[k]) user->values[nz++] = user->values[k];
    }
    PetscCall(MatSetValuesCOO(H, user->values, INSERT_VALUES));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CUTEstCreateHessian(Vec X, PetscBool shell, CUTEstCtx *user, Mat *H)
{
  PetscFunctionBeginUser;
  if (shell) {
    PetscCall(VecDuplicate(X, &user->hessian_x));
    PetscCall(MatCreateShell(PETSC_COMM_SELF, user->n, user->n, user->n, user->n, user, H));
    PetscCall(MatShellSetOperation(*H, MATOP_MULT, (PetscErrorCodeFn *)CUTEstHessianMult));
    PetscCall(MatShellSetOperation(*H, MATOP_MULT_TRANSPOSE, (PetscErrorCodeFn *)CUTEstHessianMult));
  } else {
    PetscInt  *rows, *cols;
    PetscCount nz;
    integer    nnz;

    PetscCallCUTEst(CUTEST_udimsh, &user->lh);
    nz = 2 * (PetscCount)user->lh + user->n;
    PetscCall(PetscCalloc3(user->lh, &user->rows, user->lh, &user->cols, nz, &user->values));
    PetscCallCUTEst(CUTEST_ushp, &user->n, &nnz, &user->lh, user->rows, user->cols);
    PetscCall(PetscMalloc2(nz, &rows, nz, &cols));
    for (PetscInt k = 0; k < nnz; ++k) {
      rows[k] = user->rows[k] - 1;
      cols[k] = user->cols[k] - 1;
    }
    nz = nnz;
    for (PetscInt k = 0; k < nnz; ++k) {
      if (rows[k] != cols[k]) {
        rows[nz]   = cols[k];
        cols[nz++] = rows[k];
      }
    }
    /* CUTEst can omit diagonals of linear variables. Keep MatShift() from changing the COO structure. */
    for (PetscInt i = 0; i < user->n; ++i) {
      rows[nz]   = i;
      cols[nz++] = i;
    }
    PetscCall(MatCreate(PETSC_COMM_SELF, H));
    PetscCall(MatSetSizes(*H, user->n, user->n, user->n, user->n));
    PetscCall(MatSetType(*H, MATAIJ));
    PetscCall(MatSetFromOptions(*H));
    PetscCall(MatSetPreallocationCOO(*H, nz, rows, cols));
    PetscCall(PetscFree2(rows, cols));
  }
  PetscCall(MatSetOption(*H, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(*H, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CUTEstLoadProblem(const char library[], const char data[], CUTEstCtx *user, PetscDLLibrary *dll, Vec *X)
{
  const char  *symbols[] = {"elfun_", "group_", "range_"};
  char         fullpath[PETSC_MAX_PATH_LEN], name[FSTRING_LEN + 1];
  PetscScalar *x, *lower, *upper;
  integer     *types;
  integer      input = 55, output = 6, buffer = 77, m, status;

  PetscFunctionBeginUser;
  PetscCheck(library[0], PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Specify the decoded problem library with -cutest_lib");
  PetscCall(PetscGetFullPath(library, fullpath, sizeof(fullpath)));
  /* CUTEst's loader has no status argument, so validate its required symbols first. */
  for (PetscInt i = 0; i < 3; ++i) {
    void *symbol;

    PetscCall(PetscDLLibrarySym(PETSC_COMM_SELF, dll, fullpath, symbols[i], &symbol));
    PetscCheck(symbol, PETSC_COMM_SELF, PETSC_ERR_LIB, "Decoded problem library %s does not provide %s", library, symbols[i]);
  }
  CUTEST_load_routines(fullpath);
  FORTRAN_open(&input, data, &status);
  PetscCheck(!status, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open CUTEst data file %s: status %d", data, status);
  PetscCallCUTEst(CUTEST_cdimen, &input, &user->n, &m);
  PetscCheck(!m, PETSC_COMM_SELF, PETSC_ERR_SUP, "This driver supports unconstrained problems; the problem has %d constraints", m);
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user->n, X));
  PetscCall(PetscMalloc2(user->n, &lower, user->n, &upper));
  PetscCall(VecGetArray(*X, &x));
  PetscCallCUTEst(CUTEST_usetup, &input, &output, &buffer, &user->n, x, lower, upper);
  PetscCall(VecRestoreArray(*X, &x));
  FORTRAN_close(&input, &status);
  PetscCheck(!status, PETSC_COMM_SELF, PETSC_ERR_LIB, "Cannot close CUTEst data file: status %d", status);
  for (PetscInt i = 0; i < user->n; ++i) PetscCheck(lower[i] <= -CUTE_INF && upper[i] >= CUTE_INF, PETSC_COMM_SELF, PETSC_ERR_SUP, "Variable %" PetscInt_FMT " has finite bounds; this driver supports unconstrained problems", i);
  PetscCall(PetscFree2(lower, upper));
  PetscCall(PetscMalloc1(user->n, &types));
  PetscCallCUTEst(CUTEST_uvartype, &user->n, types);
  for (PetscInt i = 0; i < user->n; ++i) PetscCheck(!types[i], PETSC_COMM_SELF, PETSC_ERR_SUP, "Variable %" PetscInt_FMT " is discrete; this driver supports continuous problems", i);
  PetscCall(PetscFree(types));
  PetscCallCUTEst(CUTEST_probname, name);
  name[FSTRING_LEN] = '\0';
  for (PetscInt i = FSTRING_LEN; i > 0 && name[i - 1] == ' '; --i) name[i - 1] = '\0';
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "CUTEst problem %s: %d variables\n", name, user->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CUTEstUnloadProblem(PetscDLLibrary dll)
{
  integer status;

  PetscFunctionBeginUser;
  CUTEST_uterminate(&status);
  PetscCheck(!status, PETSC_COMM_SELF, PETSC_ERR_LIB, "CUTEST_uterminate() returned status %d", status);
  CUTEST_unload_routines();
  PetscCall(PetscDLLibraryClose(dll));
  PetscFunctionReturn(PETSC_SUCCESS);
}
