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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscWeakFormGetResidual(wf, key.label, key.value, key.field, key.part, &n0, &f0, &n1, &f1);CHKERRQ(ierr);
  if (n0 != in0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Found %D f0 functions != %D functions input", n0, in0);
  for (i = 0; i < n0; ++i) {if (f0[i] != if0[i]) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "f0[%D] != input f0[%D]", i, i);}
  if (n1 != in1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Found %D f1 functions != %D functions input", n0, in0);
  for (i = 0; i < n1; ++i) {if (f1[i] != if1[i]) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "f1[%D] != input f1[%D]", i, i);}
  PetscFunctionReturn(0);
}

static PetscErrorCode TestSetIndex(PetscWeakForm wf)
{
  PetscPointFunc   f[4] = {f0, f1, f2, f3};
  DMLabel          label;
  const PetscInt   value = 3, field = 1, part = 2;
  PetscFormKey key;
  PetscInt         i, j, k, l;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMLabelCreate(PETSC_COMM_SELF, "Test", &label);CHKERRQ(ierr);
  key.label = label; key.value = value; key.field = field; key.part = part;
  /* Check f0 */
  for (i = 0; i < 4; ++i) {
    for (j = 0; j < 4; ++j) {
      if (j == i) continue;
      for (k = 0; k < 4; ++k) {
        if ((k == i) || (k == j)) continue;
        for (l = 0; l < 4; ++l) {
          if ((l == i) || (l == j) || (l == k)) continue;
          ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, i, f[i], 0, NULL);CHKERRQ(ierr);
          ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, j, f[j], 0, NULL);CHKERRQ(ierr);
          ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, k, f[k], 0, NULL);CHKERRQ(ierr);
          ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, l, f[l], 0, NULL);CHKERRQ(ierr);
          ierr = CheckResidual(wf, key, 4, f, 0, NULL);CHKERRQ(ierr);
          ierr = PetscWeakFormClear(wf);CHKERRQ(ierr);
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
          ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, i, f[i]);CHKERRQ(ierr);
          ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, j, f[j]);CHKERRQ(ierr);
          ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, k, f[k]);CHKERRQ(ierr);
          ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, l, f[l]);CHKERRQ(ierr);
          ierr = CheckResidual(wf, key, 0, NULL, 4, f);CHKERRQ(ierr);
          ierr = PetscWeakFormClear(wf);CHKERRQ(ierr);
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
          ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, i, f[i], i, f[i]);CHKERRQ(ierr);
          ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, j, f[j], j, f[j]);CHKERRQ(ierr);
          ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, k, f[k], k, f[k]);CHKERRQ(ierr);
          ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, l, f[l], l, f[l]);CHKERRQ(ierr);
          ierr = CheckResidual(wf, key, 4, f, 4, f);CHKERRQ(ierr);
          ierr = PetscWeakFormClear(wf);CHKERRQ(ierr);
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
          ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, l, f[l], i, f[i]);CHKERRQ(ierr);
          ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, k, f[k], j, f[j]);CHKERRQ(ierr);
          ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, j, f[j], k, f[k]);CHKERRQ(ierr);
          ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, i, f[i], l, f[l]);CHKERRQ(ierr);
          ierr = CheckResidual(wf, key, 4, f, 4, f);CHKERRQ(ierr);
          ierr = PetscWeakFormClear(wf);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = DMLabelDestroy(&label);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestAdd(PetscWeakForm wf)
{
  PetscPointFunc   f[4] = {f0, f1, f2, f3}, fp[4];
  DMLabel          label;
  const PetscInt   value = 3, field = 1, part = 2;
  PetscFormKey key;
  PetscInt         i, j, k, l;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMLabelCreate(PETSC_COMM_SELF, "Test", &label);CHKERRQ(ierr);
  key.label = label; key.value = value; key.field = field; key.part = part;
  /* Check f0 */
  for (i = 0; i < 4; ++i) {
    for (j = 0; j < 4; ++j) {
      if (j == i) continue;
      for (k = 0; k < 4; ++k) {
        if ((k == i) || (k == j)) continue;
        for (l = 0; l < 4; ++l) {
          if ((l == i) || (l == j) || (l == k)) continue;
          ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[i], NULL);CHKERRQ(ierr);
          ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[j], NULL);CHKERRQ(ierr);
          ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[k], NULL);CHKERRQ(ierr);
          ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[l], NULL);CHKERRQ(ierr);
          fp[0] = f[i]; fp[1] = f[j]; fp[2] = f[k]; fp[3] = f[l];
          ierr = CheckResidual(wf, key, 4, fp, 0, NULL);CHKERRQ(ierr);
          ierr = PetscWeakFormClear(wf);CHKERRQ(ierr);
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
          ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[i]);CHKERRQ(ierr);
          ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[j]);CHKERRQ(ierr);
          ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[k]);CHKERRQ(ierr);
          ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[l]);CHKERRQ(ierr);
          fp[0] = f[i]; fp[1] = f[j]; fp[2] = f[k]; fp[3] = f[l];
          ierr = CheckResidual(wf, key, 0, NULL, 4, fp);CHKERRQ(ierr);
          ierr = PetscWeakFormClear(wf);CHKERRQ(ierr);
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
          ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[i], f[i]);CHKERRQ(ierr);
          ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[j], f[j]);CHKERRQ(ierr);
          ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[k], f[k]);CHKERRQ(ierr);
          ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[l], f[l]);CHKERRQ(ierr);
          fp[0] = f[i]; fp[1] = f[j]; fp[2] = f[k]; fp[3] = f[l];
          ierr = CheckResidual(wf, key, 4, fp, 4, fp);CHKERRQ(ierr);
          ierr = PetscWeakFormClear(wf);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = DMLabelDestroy(&label);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestSetIndexAdd(PetscWeakForm wf)
{
  PetscPointFunc   f[4] = {f0, f1, f2, f3};
  DMLabel          label;
  const PetscInt   value = 3, field = 1, part = 2;
  PetscFormKey key;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMLabelCreate(PETSC_COMM_SELF, "Test", &label);CHKERRQ(ierr);
  key.label = label; key.value = value; key.field = field; key.part = part;
  /* Check f0 */
  ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, f[0], 0, NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 1, f[1], 0, NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[2], NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[3], NULL);CHKERRQ(ierr);
  ierr = CheckResidual(wf, key, 4, f, 0, NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormClear(wf);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, f[0], 0, NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[1], NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 2, f[2], 0, NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[3], NULL);CHKERRQ(ierr);
  ierr = CheckResidual(wf, key, 4, f, 0, NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormClear(wf);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, f[0], 0, NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[1], NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[2], NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 3, f[3], 0, NULL);CHKERRQ(ierr);
  ierr = CheckResidual(wf, key, 4, f, 0, NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormClear(wf);CHKERRQ(ierr);
  ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[0], NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 1, f[1], 0, NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[2], NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 3, f[3], 0, NULL);CHKERRQ(ierr);
  ierr = CheckResidual(wf, key, 4, f, 0, NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormClear(wf);CHKERRQ(ierr);
  ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[0], NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, f[1], NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 2, f[2], 0, NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 3, f[3], 0, NULL);CHKERRQ(ierr);
  ierr = CheckResidual(wf, key, 4, f, 0, NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormClear(wf);CHKERRQ(ierr);
  /* Check f1 */
  ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, 0, f[0]);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, 1, f[1]);CHKERRQ(ierr);
  ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[2]);CHKERRQ(ierr);
  ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[3]);CHKERRQ(ierr);
  ierr = CheckResidual(wf, key, 0, NULL, 4, f);CHKERRQ(ierr);
  ierr = PetscWeakFormClear(wf);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, 0, f[0]);CHKERRQ(ierr);
  ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[1]);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, 2, f[2]);CHKERRQ(ierr);
  ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[3]);CHKERRQ(ierr);
  ierr = CheckResidual(wf, key, 0, NULL, 4, f);CHKERRQ(ierr);
  ierr = PetscWeakFormClear(wf);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, 0, f[0]);CHKERRQ(ierr);
  ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[1]);CHKERRQ(ierr);
  ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[2]);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, 3, f[3]);CHKERRQ(ierr);
  ierr = CheckResidual(wf, key, 0, NULL, 4, f);CHKERRQ(ierr);
  ierr = PetscWeakFormClear(wf);CHKERRQ(ierr);
  ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[0]);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, 1, f[1]);CHKERRQ(ierr);
  ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[2]);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, 3, f[3]);CHKERRQ(ierr);
  ierr = CheckResidual(wf, key, 0, NULL, 4, f);CHKERRQ(ierr);
  ierr = PetscWeakFormClear(wf);CHKERRQ(ierr);
  ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[0]);CHKERRQ(ierr);
  ierr = PetscWeakFormAddResidual(wf, key.label, key.value, key.field, key.part, NULL, f[1]);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, 2, f[2]);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexResidual(wf, key.label, key.value, key.field, key.part, 0, NULL, 3, f[3]);CHKERRQ(ierr);
  ierr = CheckResidual(wf, key, 0, NULL, 4, f);CHKERRQ(ierr);
  ierr = PetscWeakFormClear(wf);CHKERRQ(ierr);

  ierr = DMLabelDestroy(&label);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscWeakForm  wf;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = PetscWeakFormCreate(PETSC_COMM_SELF, &wf);CHKERRQ(ierr);
  ierr = TestSetIndex(wf);CHKERRQ(ierr);
  ierr = TestAdd(wf);CHKERRQ(ierr);
  ierr = TestSetIndexAdd(wf);CHKERRQ(ierr);
  ierr = PetscWeakFormDestroy(&wf);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0

TEST*/
