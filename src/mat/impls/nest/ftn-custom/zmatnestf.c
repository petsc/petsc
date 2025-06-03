#include <petsc/private/ftnimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matcreatenest_     MATCREATENEST
  #define matnestsetsubmats_ MATNESTSETSUBMATS
  #define matnestgetsubmats_ MATNESTGETSUBMATS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matcreatenest_     matcreatenest
  #define matnestsetsubmats_ matnestsetsubmats
  #define matnestgetsubmats_ matnestgetsubmats
#endif

PETSC_EXTERN void matcreatenest_(MPI_Fint *comm, PetscInt *nr, IS is_row[], PetscInt *nc, IS is_col[], Mat a[], Mat *B, PetscErrorCode *ierr)
{
  Mat     *m, *tmp;
  PetscInt i;

  CHKFORTRANNULLOBJECT(is_row);
  CHKFORTRANNULLOBJECT(is_col);

  *ierr = PetscMalloc1((*nr) * (*nc), &m);
  if (*ierr) return;
  for (i = 0; i < (*nr) * (*nc); i++) {
    tmp = &a[i];
    CHKFORTRANNULLOBJECT(tmp);
    if (a[i] == (Mat)-2 || a[i] == (Mat)-3) {
      (void)PetscError(MPI_Comm_f2c(*comm), __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_WRONG, PETSC_ERROR_INITIAL, "Use PETSC_NULL_MAT for missing blocks");
      *ierr = PETSC_ERR_ARG_WRONG;
      return;
    }
    m[i] = (tmp == NULL ? NULL : a[i]);
  }
  *ierr = MatCreateNest(MPI_Comm_f2c(*comm), *nr, is_row, *nc, is_col, m, B);
  if (*ierr) return;
  *ierr = PetscFree(m);
}

PETSC_EXTERN void matnestsetsubmats_(Mat *B, PetscInt *nr, IS is_row[], PetscInt *nc, IS is_col[], Mat a[], PetscErrorCode *ierr)
{
  Mat     *m, *tmp;
  PetscInt i;
  MPI_Comm comm;

  CHKFORTRANNULLOBJECT(is_row);
  CHKFORTRANNULLOBJECT(is_col);

  *ierr = PetscMalloc1((*nr) * (*nc), &m);
  if (*ierr) return;
  for (i = 0; i < (*nr) * (*nc); i++) {
    tmp = &a[i];
    CHKFORTRANNULLOBJECT(tmp);
    if (a[i] == (Mat)-2 || a[i] == (Mat)-3) {
      *ierr = PetscObjectGetComm((PetscObject)*B, &comm);
      if (*ierr) return;
      (void)PetscError(comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_WRONG, PETSC_ERROR_INITIAL, "Use PETSC_NULL_MAT for missing blocks");
      *ierr = PETSC_ERR_ARG_WRONG;
      return;
    }
    m[i] = (tmp == NULL ? NULL : a[i]);
  }
  *ierr = MatNestSetSubMats(*B, *nr, is_row, *nc, is_col, m);
  if (*ierr) return;
  *ierr = PetscFree(m);
}

PETSC_EXTERN void matnestgetsubmats_(Mat *A, PetscInt *M, PetscInt *N, Mat *sub, PetscErrorCode *ierr)
{
  PetscInt i, j, m, n;
  Mat    **mat;

  CHKFORTRANNULLINTEGER(M);
  CHKFORTRANNULLINTEGER(N);
  CHKFORTRANNULLOBJECT(sub);

  *ierr = MatNestGetSubMats(*A, &m, &n, &mat);

  if (M) *M = m;
  if (N) *N = n;
  if (sub) {
    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
        if (mat[i][j]) {
          sub[j + n * i] = mat[i][j];
        } else {
          sub[j + n * i] = (Mat)-1;
        }
      }
    }
  }
}
