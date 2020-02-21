#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matmpiaijgetseqaij_              MATMPIAIJGETSEQAIJ
#define matcreateaij_                    MATCREATEAIJ
#define matmpiaijsetpreallocation_       MATMPIAIJSETPREALLOCATION
#define matxaijsetpreallocation_         MATXAIJSETPREALLOCATION
#define matcreatempiaijwithsplitarrays_ MATCREATEMPIAIJWITHSPLITARRAYS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matmpiaijgetseqaij_              matmpiaijgetseqaij
#define matcreateaij_                    matcreateaij
#define matmpiaijsetpreallocation_       matmpiaijsetpreallocation
#define matxaijsetpreallocation_         matxaijsetpreallocation
#define matcreatempiaijwithsplitarrays_  matcreatempiaijwithsplitarrays
#endif

PETSC_EXTERN void  matcreatempiaijwithsplitarrays_(MPI_Comm *comm,PetscInt *m,PetscInt *n,PetscInt *M,PetscInt *N,PetscInt i[],PetscInt j[],PetscScalar a[],PetscInt oi[],PetscInt oj[],PetscScalar oa[],Mat *mat, int *ierr )
{
  *ierr = MatCreateMPIAIJWithSplitArrays(MPI_Comm_f2c(*(MPI_Fint*)&*comm),*m,*n,*M,*N,i,j,a,oi,oj,oa,mat);
}

PETSC_EXTERN void matmpiaijgetseqaij_(Mat *A,Mat *Ad,Mat *Ao,PetscInt *ic,size_t *iic,PetscErrorCode *ierr)
{
  const PetscInt *i;
  *ierr = MatMPIAIJGetSeqAIJ(*A,Ad,Ao,&i);if (*ierr) return;
  *iic  = PetscIntAddressToFortran(ic,(PetscInt*)i);
}

PETSC_EXTERN void matcreateaij_(MPI_Comm *comm,PetscInt *m,PetscInt *n,PetscInt *M,PetscInt *N,PetscInt *d_nz,PetscInt *d_nnz,PetscInt *o_nz,PetscInt *o_nnz,Mat *newmat,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(d_nnz);
  CHKFORTRANNULLINTEGER(o_nnz);

  *ierr = MatCreateAIJ(MPI_Comm_f2c(*(MPI_Fint*)&*comm),*m,*n,*M,*N,*d_nz,d_nnz,*o_nz,o_nnz,newmat);
}

PETSC_EXTERN void matmpiaijsetpreallocation_(Mat *mat,PetscInt *d_nz,PetscInt *d_nnz,PetscInt *o_nz,PetscInt *o_nnz,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(d_nnz);
  CHKFORTRANNULLINTEGER(o_nnz);
  *ierr = MatMPIAIJSetPreallocation(*mat,*d_nz,d_nnz,*o_nz,o_nnz);
}

PETSC_EXTERN void  matxaijsetpreallocation_(Mat *A,PetscInt *bs,PetscInt dnnz[],PetscInt onnz[],PetscInt dnnzu[],PetscInt onnzu[],PetscErrorCode *ierr )
{
  CHKFORTRANNULLINTEGER(dnnz);
  CHKFORTRANNULLINTEGER(onnz);
  CHKFORTRANNULLINTEGER(dnnzu);
  CHKFORTRANNULLINTEGER(onnzu);
  *ierr = MatXAIJSetPreallocation(*A,*bs,dnnz,onnz,dnnzu,onnzu);
}

