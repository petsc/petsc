#include <petsc-private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matshellsetoperation_            MATSHELLSETOPERATION
#define matcreateshell_                  MATCREATESHELL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matcreateshell_                  matcreateshell
#define matshellsetoperation_            matshellsetoperation
#endif

EXTERN_C_BEGIN

/*
      The MatShell Matrix Vector product requires a C routine.
   This C routine then calls the corresponding Fortran routine that was
   set by the user.
*/
void PETSC_STDCALL matcreateshell_(MPI_Comm *comm,PetscInt *m,PetscInt *n,PetscInt *M,PetscInt *N,void *ctx,Mat *mat,PetscErrorCode *ierr)
{
  *ierr = MatCreateShell(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*m,*n,*M,*N,ctx,mat);
}

static PetscErrorCode ourmult(Mat mat,Vec x,Vec y)
{
  PetscErrorCode ierr = 0;
  (*(PetscErrorCode (PETSC_STDCALL *)(Mat*,Vec*,Vec*,PetscErrorCode*))(((PetscObject)mat)->fortran_func_pointers[0]))(&mat,&x,&y,&ierr);
  return ierr;
}

static PetscErrorCode ourmulttranspose(Mat mat,Vec x,Vec y)
{
  PetscErrorCode ierr = 0;
  (*(PetscErrorCode (PETSC_STDCALL *)(Mat*,Vec*,Vec*,PetscErrorCode*))(((PetscObject)mat)->fortran_func_pointers[2]))(&mat,&x,&y,&ierr);
  return ierr;
}

static PetscErrorCode ourmultadd(Mat mat,Vec x,Vec y,Vec z)
{
  PetscErrorCode ierr = 0;
  (*(PetscErrorCode (PETSC_STDCALL *)(Mat*,Vec*,Vec*,Vec*,PetscErrorCode*))(((PetscObject)mat)->fortran_func_pointers[1]))(&mat,&x,&y,&z,&ierr);
  return ierr;
}

static PetscErrorCode ourmulttransposeadd(Mat mat,Vec x,Vec y,Vec z)
{
  PetscErrorCode ierr = 0;
  (*(PetscErrorCode (PETSC_STDCALL *)(Mat*,Vec*,Vec*,Vec*,PetscErrorCode*))(((PetscObject)mat)->fortran_func_pointers[3]))(&mat,&x,&y,&z,&ierr);
  return ierr;
}

static PetscErrorCode ourgetdiagonal(Mat mat,Vec x)
{
  PetscErrorCode ierr = 0;
  (*(PetscErrorCode (PETSC_STDCALL *)(Mat*,Vec*,PetscErrorCode*))(((PetscObject)mat)->fortran_func_pointers[4]))(&mat,&x,&ierr);
  return ierr;
}

static PetscErrorCode ourdiagonalscale(Mat mat,Vec l,Vec r)
{
  PetscErrorCode ierr = 0;
  if (!l) {
    (*(PetscErrorCode (PETSC_STDCALL *)(Mat*,Vec*,Vec*,PetscErrorCode*))(((PetscObject)mat)->fortran_func_pointers[5]))(&mat,(Vec*)PETSC_NULL_OBJECT_Fortran,&r,&ierr);
  } else if (!r) {
    (*(PetscErrorCode (PETSC_STDCALL *)(Mat*,Vec*,Vec*,PetscErrorCode*))(((PetscObject)mat)->fortran_func_pointers[5]))(&mat,&l,(Vec*)PETSC_NULL_OBJECT_Fortran,&ierr);
  } else {
    (*(PetscErrorCode (PETSC_STDCALL *)(Mat*,Vec*,Vec*,PetscErrorCode*))(((PetscObject)mat)->fortran_func_pointers[5]))(&mat,&l,&r,&ierr);
  }
  return ierr;
}

static PetscErrorCode ourgetvecs(Mat mat,Vec *l,Vec *r)
{
  PetscErrorCode ierr = 0;
  PetscInt none = -1;
  if (!l) {
    (*(PetscErrorCode (PETSC_STDCALL *)(Mat*,Vec*,Vec*,PetscErrorCode*))(((PetscObject)mat)->fortran_func_pointers[7]))(&mat,(Vec*)&none,r,&ierr);
  } else if (!r) {
    (*(PetscErrorCode (PETSC_STDCALL *)(Mat*,Vec*,Vec*,PetscErrorCode*))(((PetscObject)mat)->fortran_func_pointers[7]))(&mat,l,(Vec*)&none,&ierr);
  } else {
    (*(PetscErrorCode (PETSC_STDCALL *)(Mat*,Vec*,Vec*,PetscErrorCode*))(((PetscObject)mat)->fortran_func_pointers[7]))(&mat,l,r,&ierr);
  }
  return ierr;
}

static PetscErrorCode ourdiagonalset(Mat mat,Vec x,InsertMode ins)
{
  PetscErrorCode ierr = 0;
  (*(PetscErrorCode (PETSC_STDCALL *)(Mat*,Vec*,InsertMode*,PetscErrorCode*))(((PetscObject)mat)->fortran_func_pointers[6]))(&mat,&x,&ins,&ierr);
  return ierr;
}

static PetscErrorCode ourview(Mat mat,PetscViewer v)
{
  PetscErrorCode ierr = 0;
  (*(PetscErrorCode (PETSC_STDCALL *)(Mat*,PetscViewer*,PetscErrorCode*))(((PetscObject)mat)->fortran_func_pointers[8]))(&mat,&v,&ierr);
  return ierr;
}

static PetscErrorCode oursor(Mat mat,Vec b,PetscReal omega,MatSORType flg,PetscReal shift,PetscInt its,PetscInt lits,Vec x)
{
  PetscErrorCode ierr = 0;
  (*(PetscErrorCode (PETSC_STDCALL *)(Mat*,Vec*,PetscReal*,MatSORType*,PetscReal*,PetscInt*,PetscInt*,Vec*,PetscErrorCode*))(((PetscObject)mat)->fortran_func_pointers[9]))(&mat,&b,&omega,&flg,&shift,&its,&lits,&x,&ierr);
  return ierr;
}

static PetscErrorCode ourshift(Mat mat, PetscScalar a)
{
  PetscErrorCode ierr = 0;
  (*(PetscErrorCode (PETSC_STDCALL *)(Mat*,PetscScalar*,PetscErrorCode*))(((PetscObject)mat)->fortran_func_pointers[10]))(&mat,&a,&ierr);
  return ierr;
}

void PETSC_STDCALL matshellsetoperation_(Mat *mat,MatOperation *op,PetscErrorCode (PETSC_STDCALL *f)(Mat*,Vec*,Vec*,PetscErrorCode*),PetscErrorCode *ierr)
{
  MPI_Comm comm;

  *ierr = PetscObjectGetComm((PetscObject)*mat,&comm);if (*ierr) return;
  PetscObjectAllocateFortranPointers(*mat,11);
  if (*op == MATOP_MULT) {
    *ierr = MatShellSetOperation(*mat,*op,(PetscVoidFunction)ourmult);
    ((PetscObject)*mat)->fortran_func_pointers[0] = (PetscVoidFunction)f;
  } else if (*op == MATOP_MULT_TRANSPOSE) {
    *ierr = MatShellSetOperation(*mat,*op,(PetscVoidFunction)ourmulttranspose);
    ((PetscObject)*mat)->fortran_func_pointers[2] = (PetscVoidFunction)f;
  } else if (*op == MATOP_MULT_ADD) {
    *ierr = MatShellSetOperation(*mat,*op,(PetscVoidFunction)ourmultadd);
    ((PetscObject)*mat)->fortran_func_pointers[1] = (PetscVoidFunction)f;
  } else if (*op == MATOP_MULT_TRANSPOSE_ADD) {
    *ierr = MatShellSetOperation(*mat,*op,(PetscVoidFunction)ourmulttransposeadd);
    ((PetscObject)*mat)->fortran_func_pointers[3] = (PetscVoidFunction)f;
  } else if (*op == MATOP_GET_DIAGONAL) {
    *ierr = MatShellSetOperation(*mat,*op,(PetscVoidFunction)ourgetdiagonal);
    ((PetscObject)*mat)->fortran_func_pointers[4] = (PetscVoidFunction)f;
  } else if (*op == MATOP_DIAGONAL_SCALE) {
    *ierr = MatShellSetOperation(*mat,*op,(PetscVoidFunction)ourdiagonalscale);
    ((PetscObject)*mat)->fortran_func_pointers[5] = (PetscVoidFunction)f;
  } else if (*op == MATOP_DIAGONAL_SET) {
    *ierr = MatShellSetOperation(*mat,*op,(PetscVoidFunction)ourdiagonalset);
    ((PetscObject)*mat)->fortran_func_pointers[6] = (PetscVoidFunction)f;
  } else if (*op == MATOP_GET_VECS) {
    *ierr = MatShellSetOperation(*mat,*op,(PetscVoidFunction)ourgetvecs);
    ((PetscObject)*mat)->fortran_func_pointers[7] = (PetscVoidFunction)f;
  } else if (*op == MATOP_VIEW) {
    *ierr = MatShellSetOperation(*mat,*op,(PetscVoidFunction)ourview);
    ((PetscObject)*mat)->fortran_func_pointers[8] = (PetscVoidFunction)f;
  } else if (*op == MATOP_SOR) {
    *ierr = MatShellSetOperation(*mat,*op,(PetscVoidFunction)oursor);
    ((PetscObject)*mat)->fortran_func_pointers[9] = (PetscVoidFunction)f;
  } else if (*op == MATOP_SHIFT) {
    *ierr = MatShellSetOperation(*mat,*op,(PetscVoidFunction)ourshift);
    ((PetscObject)*mat)->fortran_func_pointers[10] = (PetscVoidFunction)f;
  } else {
    PetscError(comm,__LINE__,"MatShellSetOperation_Fortran",__FILE__,__SDIR__,PETSC_ERR_ARG_WRONG,PETSC_ERROR_INITIAL,
               "Cannot set that matrix operation");
    *ierr = 1;
  }
}

EXTERN_C_END
