#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matcreatenest_     MATCREATENEST
  #define matnestgetiss_     MATNESTGETISS
  #define matnestgetsubmats_ MATNESTGETSUBMATS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matcreatenest_     matcreatenest
  #define matnestgetiss_     matnestgetiss
  #define matnestgetsubmats_ matnestgetsubmats
#endif

PETSC_EXTERN void matcreatenest_(MPI_Fint *comm, PetscInt *nr, IS is_row[], PetscInt *nc, IS is_col[], Mat a[], Mat *B, int *ierr)
{
  Mat     *m, *tmp;
  PetscInt i;

  CHKFORTRANNULLOBJECT(is_row);
  CHKFORTRANNULLOBJECT(is_col);

  *ierr = PetscMalloc1((*nr) * (*nc), &m);
  if (*ierr) return;
  for (i = 0; i < (*nr) * (*nc); i++) {
    tmp = &(a[i]);
    CHKFORTRANNULLOBJECT(tmp);
    m[i] = (tmp == NULL ? NULL : a[i]);
  }
  *ierr = MatCreateNest(MPI_Comm_f2c(*comm), *nr, is_row, *nc, is_col, m, B);
  if (*ierr) return;
  *ierr = PetscFree(m);
}

PETSC_EXTERN void matnestgetiss_(Mat *A, IS rows[], IS cols[], int *ierr)
{
  CHKFORTRANNULLOBJECT(rows);
  CHKFORTRANNULLOBJECT(cols);
  *ierr = MatNestGetISs(*A, rows, cols);
}

PETSC_EXTERN void matnestgetsubmats_(Mat *A, PetscInt *M, PetscInt *N, Mat *sub, int *ierr)
{
  PetscInt i, j, m, n;
  Mat    **mat;

  CHKFORTRANNULLINTEGER(M);
  CHKFORTRANNULLINTEGER(N);
  CHKFORTRANNULLOBJECT(sub);

  *ierr = MatNestGetSubMats(*A, &m, &n, &mat);

  if (M) { *M = m; }
  if (N) { *N = n; }
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
