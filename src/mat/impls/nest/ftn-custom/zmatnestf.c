#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matcreatenest_                   MATCREATENEST
#define matnestgetiss_                   MATNESTGETISS
#define matnestgetsubmats_               MATNESTGETSUBMATS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matcreatenest_                   matcreatenest
#define matnestgetiss_                   matnestgetiss
#define matnestgetsubmats_               matnestgetsubmats
#endif

PETSC_EXTERN void PETSC_STDCALL matcreatenest_(MPI_Fint *comm,PetscInt *nr,IS is_row[],PetscInt *nc,IS is_col[],Mat a[],Mat *B,int *ierr)
{
  CHKFORTRANNULLOBJECT(is_row);
  CHKFORTRANNULLOBJECT(is_col);
  *ierr = MatCreateNest(MPI_Comm_f2c(*comm),*nr,is_row,*nc,is_col,a,B);
}

PETSC_EXTERN void PETSC_STDCALL  matnestgetiss_(Mat *A,IS rows[],IS cols[], int *ierr )
{
  CHKFORTRANNULLOBJECT(rows);
  CHKFORTRANNULLOBJECT(cols);
  *ierr = MatNestGetISs(*A,rows,cols);
}

PETSC_EXTERN void PETSC_STDCALL matnestgetsubmats_(Mat *A,PetscInt *M,PetscInt *N,Mat *sub,int *ierr)
{
  PetscInt i,j,m,n;
  Mat **mat;

  CHKFORTRANNULLINTEGER(M);
  CHKFORTRANNULLINTEGER(N);
  CHKFORTRANNULLOBJECT(sub);

  *ierr = MatNestGetSubMats(*A,&m,&n,&mat);

  if (M) {
    *M = m;
  }
  if (N) {
    *N = n;
  }
  if (sub) {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        sub[j + n * i] = mat[i][j];
      }
    }
  }
}
