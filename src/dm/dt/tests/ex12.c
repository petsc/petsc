static char help[] = "Tests for PetscWeakForm.\n\n";

#include <petscds.h>

static void f0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
               PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 0.0;
}

static void f1(PetscInt dim, PetscInt Nf, PetscInt NfAux,
               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
               PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 0.0;
}

static void f2(PetscInt dim, PetscInt Nf, PetscInt NfAux,
               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
               PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 0.0;
}

static void f3(PetscInt dim, PetscInt Nf, PetscInt NfAux,
               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
               PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 0.0;
}

static PetscErrorCode CheckResidual(PetscWeakForm wf, PetscFormKey key, PetscInt in0, PetscPointFunc if0[], PetscInt in1, PetscPointFunc if1[])
{
  PetscPointFunc *f0, *f1;
  PetscInt        n0, n1, i;

  PetscFunctionBegin;
  PetscCall(PetscWeakFormGetResidual(wf, key.label, key.value, key.field, key.part, &n0, &f0, &n1, &f1));
  PetscCheck(n0 == in0,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Found %" PetscInt_FMT " f0 functions != %" PetscInt_FMT " functions input", n0, in0);
  for (i = 0; i < n0; ++i) {PetscCheck(f0[i] == if0[i],PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "f0[%" PetscInt_FMT "] != input f0[%" PetscInt_FMT "]", i, i);}
  PetscCheck(n1 == in1,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Found %" PetscInt_FMT " f1 functions != %" PetscInt_FMT " functions input", n0, in0);
  for (i = 0; i < n1; ++i) {PetscCheck(f1[i] == if1[i],PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "f1[%" PetscInt_FMT "] != input f1[%" PetscInt_FMT "]", i, i);}
  PetscFunctionReturn(0);
}

static PetscErrorCode TestSetIndex(PetscWeakForm wf)
{
  PetscPointFunc   f[4] = {f0, f1, f2, f3};
  DMLabel          label;
  const PetscInt   value = 3, field = 1, part = 2;
  PetscFormKey key;
  PetscInt         i, j, k, l;

  PetscFunctionBegin;
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Test", &label));
  key.label = label; key.value = value; key.field = field; key.part = part;
  /* Check f0 */
  for (i = 0; i < 4; ++i) {
    for (j = 0; j < 4; ++j) {
      if (j == i) continue;
      for (k = 0; k < 4; ++k) {
        if ((k == i) || (k == j)) continue;
        for (l = 0; l < 4; ++l) {
          if ((l == i) || (l == j) || (l == k)) continue;
          PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, i, f[i], 0, NULL));
          PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, j, f[j], 0, NULL));
          PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, k, f[k], 0, NULL));
          PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, l, f[l], 0, NULL));
          PetscCall(CheckResidual(wf, key, 4, f, 0, NULL));
          PetscCall(PetscWeakFormClear(wf));
        }
      }
    }
  }
  /* Check f1 */
  for (i = 0; i < 4; ++i) {
    for (j = 0; j < 4; ++j) {
      if (j == i) continue;
      for (k = 0; k < 4; ++k) {
        if ((k == i) || (k == j)) continue;
        for (l = 0; l < 4; ++l) {
          if ((l == i) || (l == j) || (l == k)) continue;
          PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, i, f[i]));
          PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, j, f[j]));
          PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, k, f[k]));
          PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, l, f[l]));
          PetscCall(CheckResidual(wf, key, 0, NULL, 4, f));
          PetscCall(PetscWeakFormClear(wf));
        }
      }
    }
  }
  /* Check f0 and f1 */
  for (i = 0; i < 4; ++i) {
    for (j = 0; j < 4; ++j) {
      if (j == i) continue;
      for (k = 0; k < 4; ++k) {
        if ((k == i) || (k == j)) continue;
        for (l = 0; l < 4; ++l) {
          if ((l == i) || (l == j) || (l == k)) continue;
          PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, i, f[i], i, f[i]));
          PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, j, f[j], j, f[j]));
          PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, k, f[k], k, f[k]));
          PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, l, f[l], l, f[l]));
          PetscCall(CheckResidual(wf, key, 4, f, 4, f));
          PetscCall(PetscWeakFormClear(wf));
        }
      }
    }
  }
  /* Check f0 and f1 in different orders */
  for (i = 0; i < 4; ++i) {
    for (j = 0; j < 4; ++j) {
      if (j == i) continue;
      for (k = 0; k < 4; ++k) {
        if ((k == i) || (k == j)) continue;
        for (l = 0; l < 4; ++l) {
          if ((l == i) || (l == j) || (l == k)) continue;
          PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, l, f[l], i, f[i]));
          PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, k, f[k], j, f[j]));
          PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, j, f[j], k, f[k]));
          PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, i, f[i], l, f[l]));
          PetscCall(CheckResidual(wf, key, 4, f, 4, f));
          PetscCall(PetscWeakFormClear(wf));
        }
      }
    }
  }
  PetscCall(DMLabelDestroy(&label));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestAdd(PetscWeakForm wf)
{
  PetscPointFunc   f[4] = {f0, f1, f2, f3}, fp[4];
  DMLabel          label;
  const PetscInt   value = 3, field = 1, part = 2;
  PetscFormKey key;
  PetscInt         i, j, k, l;

  PetscFunctionBegin;
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Test", &label));
  key.label = label; key.value = value; key.field = field; key.part = part;
  /* Check f0 */
  for (i = 0; i < 4; ++i) {
    for (j = 0; j < 4; ++j) {
      if (j == i) continue;
      for (k = 0; k < 4; ++k) {
        if ((k == i) || (k == j)) continue;
        for (l = 0; l < 4; ++l) {
          if ((l == i) || (l == j) || (l == k)) continue;
          PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[i], NULL));
          PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[j], NULL));
          PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[k], NULL));
          PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[l], NULL));
          fp[0] = f[i]; fp[1] = f[j]; fp[2] = f[k]; fp[3] = f[l];
          PetscCall(CheckResidual(wf, key, 4, fp, 0, NULL));
          PetscCall(PetscWeakFormClear(wf));
        }
      }
    }
  }
  /* Check f1 */
  for (i = 0; i < 4; ++i) {
    for (j = 0; j < 4; ++j) {
      if (j == i) continue;
      for (k = 0; k < 4; ++k) {
        if ((k == i) || (k == j)) continue;
        for (l = 0; l < 4; ++l) {
          if ((l == i) || (l == j) || (l == k)) continue;
          PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[i]));
          PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[j]));
          PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[k]));
          PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[l]));
          fp[0] = f[i]; fp[1] = f[j]; fp[2] = f[k]; fp[3] = f[l];
          PetscCall(CheckResidual(wf, key, 0, NULL, 4, fp));
          PetscCall(PetscWeakFormClear(wf));
        }
      }
    }
  }
  /* Check f0 and f1 */
  for (i = 0; i < 4; ++i) {
    for (j = 0; j < 4; ++j) {
      if (j == i) continue;
      for (k = 0; k < 4; ++k) {
        if ((k == i) || (k == j)) continue;
        for (l = 0; l < 4; ++l) {
          if ((l == i) || (l == j) || (l == k)) continue;
          PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[i], f[i]));
          PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[j], f[j]));
          PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[k], f[k]));
          PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[l], f[l]));
          fp[0] = f[i]; fp[1] = f[j]; fp[2] = f[k]; fp[3] = f[l];
          PetscCall(CheckResidual(wf, key, 4, fp, 4, fp));
          PetscCall(PetscWeakFormClear(wf));
        }
      }
    }
  }
  PetscCall(DMLabelDestroy(&label));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestSetIndexAdd(PetscWeakForm wf)
{
  PetscPointFunc   f[4] = {f0, f1, f2, f3};
  DMLabel          label;
  const PetscInt   value = 3, field = 1, part = 2;
  PetscFormKey key;

  PetscFunctionBegin;
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Test", &label));
  key.label = label; key.value = value; key.field = field; key.part = part;
  /* Check f0 */
  PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, f[0], 0, NULL));
  PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 1, f[1], 0, NULL));
  PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[2], NULL));
  PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[3], NULL));
  PetscCall(CheckResidual(wf, key, 4, f, 0, NULL));
  PetscCall(PetscWeakFormClear(wf));
  PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, f[0], 0, NULL));
  PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[1], NULL));
  PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 2, f[2], 0, NULL));
  PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[3], NULL));
  PetscCall(CheckResidual(wf, key, 4, f, 0, NULL));
  PetscCall(PetscWeakFormClear(wf));
  PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, f[0], 0, NULL));
  PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[1], NULL));
  PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[2], NULL));
  PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 3, f[3], 0, NULL));
  PetscCall(CheckResidual(wf, key, 4, f, 0, NULL));
  PetscCall(PetscWeakFormClear(wf));
  PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[0], NULL));
  PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 1, f[1], 0, NULL));
  PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[2], NULL));
  PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 3, f[3], 0, NULL));
  PetscCall(CheckResidual(wf, key, 4, f, 0, NULL));
  PetscCall(PetscWeakFormClear(wf));
  PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[0], NULL));
  PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[1], NULL));
  PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 2, f[2], 0, NULL));
  PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 3, f[3], 0, NULL));
  PetscCall(CheckResidual(wf, key, 4, f, 0, NULL));
  PetscCall(PetscWeakFormClear(wf));
  /* Check f1 */
  PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, 0, f[0]));
  PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, 1, f[1]));
  PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[2]));
  PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[3]));
  PetscCall(CheckResidual(wf, key, 0, NULL, 4, f));
  PetscCall(PetscWeakFormClear(wf));
  PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, 0, f[0]));
  PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[1]));
  PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, 2, f[2]));
  PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[3]));
  PetscCall(CheckResidual(wf, key, 0, NULL, 4, f));
  PetscCall(PetscWeakFormClear(wf));
  PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, 0, f[0]));
  PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[1]));
  PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[2]));
  PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, 3, f[3]));
  PetscCall(CheckResidual(wf, key, 0, NULL, 4, f));
  PetscCall(PetscWeakFormClear(wf));
  PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[0]));
  PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, 1, f[1]));
  PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[2]));
  PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, 3, f[3]));
  PetscCall(CheckResidual(wf, key, 0, NULL, 4, f));
  PetscCall(PetscWeakFormClear(wf));
  PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[0]));
  PetscCall(PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[1]));
  PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, 2, f[2]));
  PetscCall(PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, 3, f[3]));
  PetscCall(CheckResidual(wf, key, 0, NULL, 4, f));
  PetscCall(PetscWeakFormClear(wf));

  PetscCall(DMLabelDestroy(&label));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscWeakForm  wf;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscWeakFormCreate(PETSC_COMM_SELF, &wf));
  PetscCall(TestSetIndex(wf));
  PetscCall(TestAdd(wf));
  PetscCall(TestSetIndexAdd(wf));
  PetscCall(PetscWeakFormDestroy(&wf));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

TEST*/
