#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matcreatenest_                   MATCREATENEST
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matcreatenest_                   matcreatenest
#endif

PETSC_EXTERN void PETSC_STDCALL matcreatenest_(MPI_Fint *comm,PetscInt *nr,IS is_row[],PetscInt *nc,IS is_col[],Mat a[],Mat *B,int *ierr)
{
  CHKFORTRANNULLOBJECT(is_row);
  CHKFORTRANNULLOBJECT(is_col);
  *ierr = MatCreateNest(MPI_Comm_f2c(*comm),*nr,is_row,*nc,is_col,a,B);
}
