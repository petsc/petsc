#include "private/fortranimpl.h"
#include "petscmat.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matshellsetoperation_            MATSHELLSETOPERATION
#define matcreateshell_                  MATCREATESHELL
#define matshellgetcontext_              MATSHELLGETCONTEXT
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matcreateshell_                  matcreateshell
#define matshellsetoperation_            matshellsetoperation
#define matshellgetcontext_              matshellgetcontext
#endif

EXTERN_C_BEGIN

/*
      The MatShell Matrix Vector product requires a C routine.
   This C routine then calls the corresponding Fortran routine that was
   set by the user.
*/
void PETSC_STDCALL matcreateshell_(MPI_Comm *comm,PetscInt *m,PetscInt *n,PetscInt *M,PetscInt *N,void **ctx,Mat *mat,PetscErrorCode *ierr)
{
  *ierr = MatCreateShell(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*m,*n,*M,*N,*ctx,mat);

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

static PetscErrorCode ourdiagonalset(Mat mat,Vec x,InsertMode ins)
{
  PetscErrorCode ierr = 0;
  (*(PetscErrorCode (PETSC_STDCALL *)(Mat*,Vec*,InsertMode*,PetscErrorCode*))(((PetscObject)mat)->fortran_func_pointers[6]))(&mat,&x,&ins,&ierr);
  return ierr;
}

void PETSC_STDCALL matshellsetoperation_(Mat *mat,MatOperation *op,PetscErrorCode (PETSC_STDCALL *f)(Mat*,Vec*,Vec*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*mat,7);
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
  } else {
    PetscError(__LINE__,"MatShellSetOperation_Fortran",__FILE__,__SDIR__,1,0,
               "Cannot set that matrix operation");
    *ierr = 1;
  }
}

void PETSC_STDCALL matshellgetcontext_(Mat *mat,void **ctx,PetscErrorCode *ierr)
{
  *ierr = MatShellGetContext(*mat,ctx);
}


EXTERN_C_END
