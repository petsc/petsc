#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matshellsetoperation_ MATSHELLSETOPERATION
  #define matcreateshell_       MATCREATESHELL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matcreateshell_       matcreateshell
  #define matshellsetoperation_ matshellsetoperation
#endif

/**
 * Subset of MatOperation that is supported by the Fortran wrappers.
 */
enum FortranMatOperation {
  FORTRAN_MATOP_MULT               = 0,
  FORTRAN_MATOP_MULT_ADD           = 1,
  FORTRAN_MATOP_MULT_TRANSPOSE     = 2,
  FORTRAN_MATOP_MULT_TRANSPOSE_ADD = 3,
  FORTRAN_MATOP_SOR                = 4,
  FORTRAN_MATOP_TRANSPOSE          = 5,
  FORTRAN_MATOP_GET_DIAGONAL       = 6,
  FORTRAN_MATOP_DIAGONAL_SCALE     = 7,
  FORTRAN_MATOP_ZERO_ENTRIES       = 8,
  FORTRAN_MATOP_AXPY               = 9,
  FORTRAN_MATOP_SHIFT              = 10,
  FORTRAN_MATOP_DIAGONAL_SET       = 11,
  FORTRAN_MATOP_DESTROY            = 12,
  FORTRAN_MATOP_VIEW               = 13,
  FORTRAN_MATOP_CREATE_VECS        = 14,
  FORTRAN_MATOP_GET_DIAGONAL_BLOCK = 15,
  FORTRAN_MATOP_COPY               = 16,
  FORTRAN_MATOP_SCALE              = 17,
  FORTRAN_MATOP_SET_RANDOM         = 18,
  FORTRAN_MATOP_ASSEMBLY_BEGIN     = 19,
  FORTRAN_MATOP_ASSEMBLY_END       = 20,
  FORTRAN_MATOP_SIZE               = 21
};

/*
  The MatShell Matrix Vector product requires a C routine.
  This C routine then calls the corresponding Fortran routine that was
  set by the user.
*/
PETSC_EXTERN void matcreateshell_(MPI_Comm *comm, PetscInt *m, PetscInt *n, PetscInt *M, PetscInt *N, void *ctx, Mat *mat, PetscErrorCode *ierr)
{
  *ierr = MatCreateShell(MPI_Comm_f2c(*(MPI_Fint *)&*comm), *m, *n, *M, *N, ctx, mat);
}

static PetscErrorCode ourmult(Mat mat, Vec x, Vec y)
{
  PetscCallFortranVoidFunction((*(void (*)(Mat *, Vec *, Vec *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[FORTRAN_MATOP_MULT]))(&mat, &x, &y, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourmultadd(Mat mat, Vec x, Vec y, Vec z)
{
  PetscCallFortranVoidFunction((*(void (*)(Mat *, Vec *, Vec *, Vec *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[FORTRAN_MATOP_MULT_ADD]))(&mat, &x, &y, &z, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourmulttranspose(Mat mat, Vec x, Vec y)
{
  PetscCallFortranVoidFunction((*(void (*)(Mat *, Vec *, Vec *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[FORTRAN_MATOP_MULT_TRANSPOSE]))(&mat, &x, &y, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourmulttransposeadd(Mat mat, Vec x, Vec y, Vec z)
{
  PetscCallFortranVoidFunction((*(void (*)(Mat *, Vec *, Vec *, Vec *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[FORTRAN_MATOP_MULT_TRANSPOSE_ADD]))(&mat, &x, &y, &z, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode oursor(Mat mat, Vec b, PetscReal omega, MatSORType flg, PetscReal shift, PetscInt its, PetscInt lits, Vec x)
{
  PetscErrorCode ierr = PETSC_SUCCESS;

  (*(void (*)(Mat *, Vec *, PetscReal *, MatSORType *, PetscReal *, PetscInt *, PetscInt *, Vec *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[FORTRAN_MATOP_SOR]))(&mat, &b, &omega, &flg, &shift, &its, &lits, &x, &ierr);
  return ierr;
}

static PetscErrorCode ourtranspose(Mat mat, MatReuse reuse, Mat *B)
{
  Mat  bb = (Mat)-1;
  Mat *b  = (!B ? &bb : B);

  PetscCallFortranVoidFunction((*(void (*)(Mat *, MatReuse *, Mat *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[FORTRAN_MATOP_TRANSPOSE]))(&mat, &reuse, b, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourgetdiagonal(Mat mat, Vec x)
{
  PetscCallFortranVoidFunction((*(void (*)(Mat *, Vec *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[FORTRAN_MATOP_GET_DIAGONAL]))(&mat, &x, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourdiagonalscale(Mat mat, Vec l, Vec r)
{
  Vec  aa = (Vec)-1;
  Vec *a  = (!l ? &aa : &l);
  Vec *b  = (!r ? &aa : &r);

  PetscCallFortranVoidFunction((*(void (*)(Mat *, Vec *, Vec *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[FORTRAN_MATOP_DIAGONAL_SCALE]))(&mat, a, b, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourzeroentries(Mat mat)
{
  PetscCallFortranVoidFunction((*(void (*)(Mat *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[FORTRAN_MATOP_ZERO_ENTRIES]))(&mat, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ouraxpy(Mat mat, PetscScalar a, Mat X, MatStructure str)
{
  PetscCallFortranVoidFunction((*(void (*)(Mat *, PetscScalar *, Mat *, MatStructure *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[FORTRAN_MATOP_AXPY]))(&mat, &a, &X, &str, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourshift(Mat mat, PetscScalar a)
{
  PetscCallFortranVoidFunction((*(void (*)(Mat *, PetscScalar *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[FORTRAN_MATOP_SHIFT]))(&mat, &a, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourdiagonalset(Mat mat, Vec x, InsertMode ins)
{
  PetscCallFortranVoidFunction((*(void (*)(Mat *, Vec *, InsertMode *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[FORTRAN_MATOP_DIAGONAL_SET]))(&mat, &x, &ins, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourdestroy(Mat mat)
{
  PetscCallFortranVoidFunction((*(void (*)(Mat *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[FORTRAN_MATOP_DESTROY]))(&mat, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourview(Mat mat, PetscViewer v)
{
  PetscCallFortranVoidFunction((*(void (*)(Mat *, PetscViewer *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[FORTRAN_MATOP_VIEW]))(&mat, &v, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourgetvecs(Mat mat, Vec *l, Vec *r)
{
  Vec  aa = (Vec)-1;
  Vec *a  = (!l ? &aa : l);
  Vec *b  = (!r ? &aa : r);

  PetscCallFortranVoidFunction((*(void (*)(Mat *, Vec *, Vec *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[FORTRAN_MATOP_CREATE_VECS]))(&mat, a, b, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourgetdiagonalblock(Mat mat, Mat *l)
{
  PetscCallFortranVoidFunction((*(void (*)(Mat *, Mat *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[FORTRAN_MATOP_GET_DIAGONAL_BLOCK]))(&mat, l, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourcopy(Mat mat, Mat B, MatStructure str)
{
  PetscCallFortranVoidFunction((*(void (*)(Mat *, Mat *, MatStructure *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[FORTRAN_MATOP_COPY]))(&mat, &B, &str, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourscale(Mat mat, PetscScalar a)
{
  PetscCallFortranVoidFunction((*(void (*)(Mat *, PetscScalar *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[FORTRAN_MATOP_SCALE]))(&mat, &a, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode oursetrandom(Mat mat, PetscRandom ctx)
{
  PetscCallFortranVoidFunction((*(void (*)(Mat *, PetscRandom *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[FORTRAN_MATOP_SET_RANDOM]))(&mat, &ctx, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourassemblybegin(Mat mat, MatAssemblyType type)
{
  PetscCallFortranVoidFunction((*(void (*)(Mat *, MatAssemblyType *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[FORTRAN_MATOP_ASSEMBLY_BEGIN]))(&mat, &type, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourassemblyend(Mat mat, MatAssemblyType type)
{
  PetscCallFortranVoidFunction((*(void (*)(Mat *, MatAssemblyType *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[FORTRAN_MATOP_ASSEMBLY_END]))(&mat, &type, &ierr));
  return PETSC_SUCCESS;
}

PETSC_EXTERN void matshellsetoperation_(Mat *mat, MatOperation *op, PetscErrorCode (*f)(Mat *, Vec *, Vec *, PetscErrorCode *), PetscErrorCode *ierr)
{
  MPI_Comm comm;

  *ierr = PetscObjectGetComm((PetscObject)*mat, &comm);
  if (*ierr) return;
  PetscObjectAllocateFortranPointers(*mat, FORTRAN_MATOP_SIZE);

  switch (*op) {
  case MATOP_MULT:
    *ierr                                                          = MatShellSetOperation(*mat, *op, (PetscVoidFunction)ourmult);
    ((PetscObject)*mat)->fortran_func_pointers[FORTRAN_MATOP_MULT] = (PetscVoidFunction)f;
    break;
  case MATOP_MULT_ADD:
    *ierr                                                              = MatShellSetOperation(*mat, *op, (PetscVoidFunction)ourmultadd);
    ((PetscObject)*mat)->fortran_func_pointers[FORTRAN_MATOP_MULT_ADD] = (PetscVoidFunction)f;
    break;
  case MATOP_MULT_TRANSPOSE:
    *ierr                                                                    = MatShellSetOperation(*mat, *op, (PetscVoidFunction)ourmulttranspose);
    ((PetscObject)*mat)->fortran_func_pointers[FORTRAN_MATOP_MULT_TRANSPOSE] = (PetscVoidFunction)f;
    break;
  case MATOP_MULT_TRANSPOSE_ADD:
    *ierr                                                                        = MatShellSetOperation(*mat, *op, (PetscVoidFunction)ourmulttransposeadd);
    ((PetscObject)*mat)->fortran_func_pointers[FORTRAN_MATOP_MULT_TRANSPOSE_ADD] = (PetscVoidFunction)f;
    break;
  case MATOP_SOR:
    *ierr                                                         = MatShellSetOperation(*mat, *op, (PetscVoidFunction)oursor);
    ((PetscObject)*mat)->fortran_func_pointers[FORTRAN_MATOP_SOR] = (PetscVoidFunction)f;
    break;
  case MATOP_TRANSPOSE:
    *ierr                                                               = MatShellSetOperation(*mat, *op, (PetscVoidFunction)ourtranspose);
    ((PetscObject)*mat)->fortran_func_pointers[FORTRAN_MATOP_TRANSPOSE] = (PetscVoidFunction)f;
    break;
  case MATOP_GET_DIAGONAL:
    *ierr                                                                  = MatShellSetOperation(*mat, *op, (PetscVoidFunction)ourgetdiagonal);
    ((PetscObject)*mat)->fortran_func_pointers[FORTRAN_MATOP_GET_DIAGONAL] = (PetscVoidFunction)f;
    break;
  case MATOP_DIAGONAL_SCALE:
    *ierr                                                                    = MatShellSetOperation(*mat, *op, (PetscVoidFunction)ourdiagonalscale);
    ((PetscObject)*mat)->fortran_func_pointers[FORTRAN_MATOP_DIAGONAL_SCALE] = (PetscVoidFunction)f;
    break;
  case MATOP_ZERO_ENTRIES:
    *ierr                                                                  = MatShellSetOperation(*mat, *op, (PetscVoidFunction)ourzeroentries);
    ((PetscObject)*mat)->fortran_func_pointers[FORTRAN_MATOP_ZERO_ENTRIES] = (PetscVoidFunction)f;
    break;
  case MATOP_AXPY:
    *ierr                                                          = MatShellSetOperation(*mat, *op, (PetscVoidFunction)ouraxpy);
    ((PetscObject)*mat)->fortran_func_pointers[FORTRAN_MATOP_AXPY] = (PetscVoidFunction)f;
    break;
  case MATOP_SHIFT:
    *ierr                                                           = MatShellSetOperation(*mat, *op, (PetscVoidFunction)ourshift);
    ((PetscObject)*mat)->fortran_func_pointers[FORTRAN_MATOP_SHIFT] = (PetscVoidFunction)f;
    break;
  case MATOP_DIAGONAL_SET:
    *ierr                                                                  = MatShellSetOperation(*mat, *op, (PetscVoidFunction)ourdiagonalset);
    ((PetscObject)*mat)->fortran_func_pointers[FORTRAN_MATOP_DIAGONAL_SET] = (PetscVoidFunction)f;
    break;
  case MATOP_DESTROY:
    *ierr                                                             = MatShellSetOperation(*mat, *op, (PetscVoidFunction)ourdestroy);
    ((PetscObject)*mat)->fortran_func_pointers[FORTRAN_MATOP_DESTROY] = (PetscVoidFunction)f;
    break;
  case MATOP_VIEW:
    *ierr                                                          = MatShellSetOperation(*mat, *op, (PetscVoidFunction)ourview);
    ((PetscObject)*mat)->fortran_func_pointers[FORTRAN_MATOP_VIEW] = (PetscVoidFunction)f;
    break;
  case MATOP_CREATE_VECS:
    *ierr                                                                 = MatShellSetOperation(*mat, *op, (PetscVoidFunction)ourgetvecs);
    ((PetscObject)*mat)->fortran_func_pointers[FORTRAN_MATOP_CREATE_VECS] = (PetscVoidFunction)f;
    break;
  case MATOP_GET_DIAGONAL_BLOCK:
    *ierr                                                                        = MatShellSetOperation(*mat, *op, (PetscVoidFunction)ourgetdiagonalblock);
    ((PetscObject)*mat)->fortran_func_pointers[FORTRAN_MATOP_GET_DIAGONAL_BLOCK] = (PetscVoidFunction)f;
    break;
  case MATOP_COPY:
    *ierr                                                          = MatShellSetOperation(*mat, *op, (PetscVoidFunction)ourcopy);
    ((PetscObject)*mat)->fortran_func_pointers[FORTRAN_MATOP_COPY] = (PetscVoidFunction)f;
    break;
  case MATOP_SCALE:
    *ierr                                                           = MatShellSetOperation(*mat, *op, (PetscVoidFunction)ourscale);
    ((PetscObject)*mat)->fortran_func_pointers[FORTRAN_MATOP_SCALE] = (PetscVoidFunction)f;
    break;
  case MATOP_SET_RANDOM:
    *ierr                                                                = MatShellSetOperation(*mat, *op, (PetscVoidFunction)oursetrandom);
    ((PetscObject)*mat)->fortran_func_pointers[FORTRAN_MATOP_SET_RANDOM] = (PetscVoidFunction)f;
    break;
  case MATOP_ASSEMBLY_BEGIN:
    *ierr                                                                    = MatShellSetOperation(*mat, *op, (PetscVoidFunction)ourassemblybegin);
    ((PetscObject)*mat)->fortran_func_pointers[FORTRAN_MATOP_ASSEMBLY_BEGIN] = (PetscVoidFunction)f;
    break;
  case MATOP_ASSEMBLY_END:
    *ierr                                                                  = MatShellSetOperation(*mat, *op, (PetscVoidFunction)ourassemblyend);
    ((PetscObject)*mat)->fortran_func_pointers[FORTRAN_MATOP_ASSEMBLY_END] = (PetscVoidFunction)f;
    break;
  default:
    *ierr = PetscError(comm, __LINE__, "MatShellSetOperation_Fortran", __FILE__, PETSC_ERR_ARG_WRONG, PETSC_ERROR_INITIAL, "Cannot set that matrix operation");
    *ierr = PETSC_ERR_ARG_WRONG;
  }
}
