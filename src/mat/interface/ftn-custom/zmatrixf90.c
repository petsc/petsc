#include <petscmat.h>
#include <petsc/private/ftnimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matgetrow_                   MATGETROW
  #define matrestorerow_               MATRESTOREROW
  #define matmpiaijgetseqaij_          MATMPIAIJGETSEQAIJ
  #define matmpiaijrestoreseqaij_      MATMPIAIJRESTORESEQAIJ
  #define matdensegetarray1d_          MATDENSEGETARRAY1D
  #define matdenserestorearray1d_      MATDENSERESTOREARRAY1D
  #define matdensegetarrayread1d_      MATDENSEGETARRAYREAD1D
  #define matdenserestorearrayread1d_  MATDENSERESTOREARRAYREAD1D
  #define matdensegetarraywrite1d_     MATDENSEGETARRAYWRITE1D
  #define matdenserestorearraywrite1d_ MATDENSERESTOREARRAYWRITE1D
  #define matdensegetarray2d_          MATDENSEGETARRAY2D
  #define matdenserestorearray2d_      MATDENSERESTOREARRAY2D
  #define matdensegetarrayread2d_      MATDENSEGETARRAYREAD2D
  #define matdenserestorearrayread2d_  MATDENSERESTOREARRAYREAD2D
  #define matdensegetarraywrite2d_     MATDENSEGETARRAYWRITE2D
  #define matdenserestorearraywrite2d_ MATDENSERESTOREARRAYWRITE2D
  #define matdensegetcolumn_           MATDENSEGETCOLUMN
  #define matdenserestorecolumn_       MATDENSERESTORECOLUMN
  #define matseqaijgetarray_           MATSEQAIJGETARRAY
  #define matseqaijrestorearray_       MATSEQAIJRESTOREARRAY
  #define matgetghosts_                MATGETGHOSTS
  #define matgetrowij_                 MATGETROWIJ
  #define matrestorerowij_             MATRESTOREROWIJ
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matgetrow_                   matgetrow
  #define matrestorerow_               matrestorerow
  #define matmpiaijgetseqaij_          matmpiaijgetseqaij
  #define matmpiaijrestoreseqaij_      matmpiaijrestoreseqaij
  #define matdensegetarray1d_          matdensegetarray1d
  #define matdenserestorearray1d_      matdenserestorearray1d
  #define matdensegetarrayread1d_      matdensegetarrayread1d
  #define matdenserestorearrayread1d_  matdenserestorearrayread1d
  #define matdensegetarraywrite1d_     matdensegetarraywrite1d
  #define matdenserestorearraywrite1d_ matdenserestorearraywrite1d
  #define matdensegetarray2d_          matdensegetarray2d
  #define matdenserestorearray2d_      matdenserestorearray2d
  #define matdensegetarrayread2d_      matdensegetarrayread2d
  #define matdenserestorearrayread2d_  matdenserestorearrayread2d
  #define matdensegetarraywrite2d_     matdensegetarraywrite2d
  #define matdenserestorearraywrite2d_ matdenserestorearraywrite2d
  #define matdensegetcolumn_           matdensegetcolumn
  #define matdenserestorecolumn_       matdenserestorecolumn
  #define matseqaijgetarray_           matseqaijgetarray
  #define matseqaijrestorearray_       matseqaijrestorearray
  #define matgetghosts_                matgetghosts
  #define matgetrowij_                 matgetrowij
  #define matrestorerowij_             matrestorerowij
#endif

PETSC_EXTERN void matgetrow_(Mat *B, PetscInt *row, PetscInt *N, F90Array1d *ia, F90Array1d *a, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(iad) PETSC_F90_2PTR_PROTO(jad))
{
  PetscInt           n;
  const PetscInt    *II = NULL;
  const PetscScalar *A  = NULL;

  if (FORTRANNULLINTEGERPOINTER(ia) && FORTRANNULLSCALARPOINTER(a)) {
    *ierr = MatGetRow(*B, *row, &n, NULL, NULL);
  } else if (FORTRANNULLINTEGERPOINTER(ia)) {
    *ierr = MatGetRow(*B, *row, &n, NULL, &A);
  } else if (FORTRANNULLSCALARPOINTER(a)) {
    *ierr = MatGetRow(*B, *row, &n, &II, NULL);
  } else {
    *ierr = MatGetRow(*B, *row, &n, &II, &A);
  }
  if (*ierr) return;
  if (II) *ierr = F90Array1dCreate((void *)II, MPIU_INT, 1, n, ia PETSC_F90_2PTR_PARAM(iad));
  if (A) *ierr = F90Array1dCreate((void *)A, MPIU_SCALAR, 1, n, a PETSC_F90_2PTR_PARAM(jad));
  if (!FORTRANNULLINTEGER(N)) *N = n;
}
PETSC_EXTERN void matrestorerow_(Mat *B, PetscInt *row, PetscInt *N, F90Array1d *ia, F90Array1d *a, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(iad) PETSC_F90_2PTR_PROTO(jad))
{
  const PetscInt    *IA = NULL;
  const PetscScalar *A  = NULL;
  PetscInt           n;

  if (FORTRANNULLINTEGERPOINTER(ia) && FORTRANNULLSCALARPOINTER(a)) {
    *ierr = MatRestoreRow(*B, *row, &n, NULL, NULL);
    return;
  }
  if (!FORTRANNULLINTEGERPOINTER(ia)) {
    *ierr = F90Array1dAccess(ia, MPIU_INT, (void **)&IA PETSC_F90_2PTR_PARAM(iad));
    if (*ierr) return;
    *ierr = F90Array1dDestroy(ia, MPIU_INT PETSC_F90_2PTR_PARAM(iad));
    if (*ierr) return;
  }
  if (!FORTRANNULLSCALARPOINTER(a)) {
    *ierr = F90Array1dAccess(a, MPIU_SCALAR, (void **)&A PETSC_F90_2PTR_PARAM(jad));
    if (*ierr) return;
    *ierr = F90Array1dDestroy(a, MPIU_INT PETSC_F90_2PTR_PARAM(jad));
    if (*ierr) return;
  }
  if (FORTRANNULLINTEGERPOINTER(ia)) {
    *ierr = MatRestoreRow(*B, *row, &n, NULL, &A);
  } else if (FORTRANNULLSCALARPOINTER(a)) {
    *ierr = MatRestoreRow(*B, *row, &n, &IA, NULL);
  } else {
    *ierr = MatRestoreRow(*B, *row, &n, &IA, &A);
  }
}
PETSC_EXTERN void matgetghosts_(Mat *mat, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *ghosts;
  PetscInt        N;

  *ierr = MatGetGhosts(*mat, &N, &ghosts);
  if (*ierr) return;
  *ierr = F90Array1dCreate((PetscInt *)ghosts, MPIU_INT, 1, N, ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matdensegetarray2d_(Mat *mat, F90Array2d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  PetscInt     m, N, lda;
  *ierr = MatDenseGetArray(*mat, &fa);
  if (*ierr) return;
  *ierr = MatGetLocalSize(*mat, &m, NULL);
  if (*ierr) return;
  *ierr = MatGetSize(*mat, NULL, &N);
  if (*ierr) return;
  *ierr = MatDenseGetLDA(*mat, &lda);
  if (*ierr) return;
  if (m != lda) { // TODO: add F90Array2dLDACreate()
    *ierr = PetscError(((PetscObject)*mat)->comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_BADPTR, PETSC_ERROR_INITIAL, "Array lda %" PetscInt_FMT " must match number of local rows %" PetscInt_FMT, lda, m);
    return;
  }
  *ierr = F90Array2dCreate(fa, MPIU_SCALAR, 1, m, 1, N, ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matdenserestorearray2d_(Mat *mat, F90Array2d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array2dAccess(ptr, MPIU_SCALAR, (void **)&fa PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = F90Array2dDestroy(ptr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = MatDenseRestoreArray(*mat, &fa);
}
PETSC_EXTERN void matdensegetarrayread2d_(Mat *mat, F90Array2d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscScalar *fa;
  PetscInt           m, N, lda;
  *ierr = MatDenseGetArrayRead(*mat, &fa);
  if (*ierr) return;
  *ierr = MatGetLocalSize(*mat, &m, NULL);
  if (*ierr) return;
  *ierr = MatGetSize(*mat, NULL, &N);
  if (*ierr) return;
  *ierr = MatDenseGetLDA(*mat, &lda);
  if (*ierr) return;
  if (m != lda) { // TODO: add F90Array2dLDACreate()
    *ierr = PetscError(((PetscObject)*mat)->comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_BADPTR, PETSC_ERROR_INITIAL, "Array lda %" PetscInt_FMT " must match number of local rows %" PetscInt_FMT, lda, m);
    return;
  }
  *ierr = F90Array2dCreate((void **)fa, MPIU_SCALAR, 1, m, 1, N, ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matdenserestorearrayread2d_(Mat *mat, F90Array2d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscScalar *fa;
  *ierr = F90Array2dAccess(ptr, MPIU_SCALAR, (void **)&fa PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = F90Array2dDestroy(ptr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = MatDenseRestoreArrayRead(*mat, &fa);
}
PETSC_EXTERN void matdensegetarraywrite2d_(Mat *mat, F90Array2d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  PetscInt     m, N, lda;
  *ierr = MatDenseGetArrayWrite(*mat, &fa);
  if (*ierr) return;
  *ierr = MatGetLocalSize(*mat, &m, NULL);
  if (*ierr) return;
  *ierr = MatGetSize(*mat, NULL, &N);
  if (*ierr) return;
  *ierr = MatDenseGetLDA(*mat, &lda);
  if (*ierr) return;
  if (m != lda) { // TODO: add F90Array2dLDACreate()
    *ierr = PetscError(((PetscObject)*mat)->comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_BADPTR, PETSC_ERROR_INITIAL, "Array lda %" PetscInt_FMT " must match number of local rows %" PetscInt_FMT, lda, m);
    return;
  }
  *ierr = F90Array2dCreate(fa, MPIU_SCALAR, 1, m, 1, N, ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matdenserestorearraywrite2d_(Mat *mat, F90Array2d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array2dAccess(ptr, MPIU_SCALAR, (void **)&fa PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = F90Array2dDestroy(ptr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = MatDenseRestoreArrayWrite(*mat, &fa);
}
PETSC_EXTERN void matdensegetarray1d_(Mat *mat, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  PetscInt     m, N, lda;
  *ierr = MatDenseGetArray(*mat, &fa);
  if (*ierr) return;
  *ierr = MatGetLocalSize(*mat, &m, NULL);
  if (*ierr) return;
  *ierr = MatGetSize(*mat, NULL, &N);
  if (*ierr) return;
  *ierr = MatDenseGetLDA(*mat, &lda);
  if (*ierr) return;
  if (m != lda) { // TODO: add F90Array1dLDACreate()
    *ierr = PetscError(((PetscObject)*mat)->comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_BADPTR, PETSC_ERROR_INITIAL, "Array lda %" PetscInt_FMT " must match number of local rows %" PetscInt_FMT, lda, m);
    return;
  }
  *ierr = F90Array1dCreate(fa, MPIU_SCALAR, 1, m * N, ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matdenserestorearray1d_(Mat *mat, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array1dAccess(ptr, MPIU_SCALAR, (void **)&fa PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = MatDenseRestoreArray(*mat, &fa);
}
PETSC_EXTERN void matdensegetarrayread1d_(Mat *mat, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscScalar *fa;
  PetscInt           m, N, lda;
  *ierr = MatDenseGetArrayRead(*mat, &fa);
  if (*ierr) return;
  *ierr = MatGetLocalSize(*mat, &m, NULL);
  if (*ierr) return;
  *ierr = MatGetSize(*mat, NULL, &N);
  if (*ierr) return;
  *ierr = MatDenseGetLDA(*mat, &lda);
  if (*ierr) return;
  if (m != lda) { // TODO: add F90Array1dLDACreate()
    *ierr = PetscError(((PetscObject)*mat)->comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_BADPTR, PETSC_ERROR_INITIAL, "Array lda %" PetscInt_FMT " must match number of local rows %" PetscInt_FMT, lda, m);
    return;
  }
  *ierr = F90Array1dCreate((void **)fa, MPIU_SCALAR, 1, m * N, ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matdenserestorearrayread1d_(Mat *mat, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscScalar *fa;
  *ierr = F90Array1dAccess(ptr, MPIU_SCALAR, (void **)&fa PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = MatDenseRestoreArrayRead(*mat, &fa);
}
PETSC_EXTERN void matdensegetarraywrite1d_(Mat *mat, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  PetscInt     m, N, lda;
  *ierr = MatDenseGetArrayWrite(*mat, &fa);
  if (*ierr) return;
  *ierr = MatGetLocalSize(*mat, &m, NULL);
  if (*ierr) return;
  *ierr = MatGetSize(*mat, NULL, &N);
  if (*ierr) return;
  *ierr = MatDenseGetLDA(*mat, &lda);
  if (*ierr) return;
  if (m != lda) { // TODO: add F90Array1dLDACreate()
    *ierr = PetscError(((PetscObject)*mat)->comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_BADPTR, PETSC_ERROR_INITIAL, "Array lda %" PetscInt_FMT " must match number of local rows %" PetscInt_FMT, lda, m);
    return;
  }
  *ierr = F90Array1dCreate(fa, MPIU_SCALAR, 1, m * N, ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matdenserestorearraywrite1d_(Mat *mat, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array1dAccess(ptr, MPIU_SCALAR, (void **)&fa PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = MatDenseRestoreArrayWrite(*mat, &fa);
}
PETSC_EXTERN void matdensegetcolumn_(Mat *mat, PetscInt *col, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  PetscInt     m;
  *ierr = MatDenseGetColumn(*mat, *col, &fa);
  if (*ierr) return;
  *ierr = MatGetLocalSize(*mat, &m, NULL);
  if (*ierr) return;
  *ierr = F90Array1dCreate(fa, MPIU_SCALAR, 1, m, ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matdenserestorecolumn_(Mat *mat, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array1dDestroy(ptr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = MatDenseRestoreColumn(*mat, &fa);
}

#include <../src/mat/impls/aij/seq/aij.h>
PETSC_EXTERN void matseqaijgetarray_(Mat *mat, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  Mat_SeqAIJ  *a  = (Mat_SeqAIJ *)(*mat)->data;
  PetscInt     nz = (*mat)->rmap->n ? a->i[(*mat)->rmap->n] : 0;

  *ierr = MatSeqAIJGetArray(*mat, &fa);
  if (*ierr) return;
  *ierr = F90Array1dCreate(fa, MPIU_SCALAR, 1, nz, ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matseqaijrestorearray_(Mat *mat, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array1dAccess(ptr, MPIU_SCALAR, (void **)&fa PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = MatSeqAIJRestoreArray(*mat, &fa);
}
PETSC_EXTERN void matgetrowij_(Mat *B, PetscInt *shift, PetscBool *sym, PetscBool *blockcompressed, PetscInt *n, F90Array1d *ia, F90Array1d *ja, PetscBool *done, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(iad) PETSC_F90_2PTR_PROTO(jad))
{
  const PetscInt *IA, *JA;
  *ierr = MatGetRowIJ(*B, *shift, *sym, *blockcompressed, n, &IA, &JA, done);
  if (*ierr) return;
  if (!*done) return;
  *ierr = F90Array1dCreate((PetscInt *)IA, MPIU_INT, 1, *n + 1, ia PETSC_F90_2PTR_PARAM(iad));
  *ierr = F90Array1dCreate((PetscInt *)JA, MPIU_INT, 1, IA[*n], ja PETSC_F90_2PTR_PARAM(jad));
}
PETSC_EXTERN void matrestorerowij_(Mat *B, PetscInt *shift, PetscBool *sym, PetscBool *blockcompressed, PetscInt *n, F90Array1d *ia, F90Array1d *ja, PetscBool *done, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(iad) PETSC_F90_2PTR_PROTO(jad))
{
  const PetscInt *IA, *JA;
  *ierr = F90Array1dAccess(ia, MPIU_INT, (void **)&IA PETSC_F90_2PTR_PARAM(iad));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ia, MPIU_INT PETSC_F90_2PTR_PARAM(iad));
  if (*ierr) return;
  *ierr = F90Array1dAccess(ja, MPIU_INT, (void **)&JA PETSC_F90_2PTR_PARAM(jad));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ja, MPIU_INT PETSC_F90_2PTR_PARAM(jad));
  if (*ierr) return;
  *ierr = MatRestoreRowIJ(*B, *shift, *sym, *blockcompressed, n, &IA, &JA, done);
}
PETSC_EXTERN void matmpiaijgetseqaij_(Mat *mat, Mat *A, Mat *B, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;
  PetscInt        n;
  *ierr = MatMPIAIJGetSeqAIJ(*mat, A, B, &fa);
  if (*ierr) return;
  *ierr = MatGetLocalSize(*B, NULL, &n);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)fa, MPIU_INT, 1, n, ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matmpiaijrestoreseqaij_(Mat *mat, Mat *A, Mat *B, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(ptr, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
}
