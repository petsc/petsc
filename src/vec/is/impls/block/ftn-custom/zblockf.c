#include "zpetsc.h"
#include "petscis.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define isblockgetindices_     ISBLOCKGETINDICES
#define isblockrestoreindices_ ISBLOCKRESTOREINDICES
#define iscreateblock_         ISCREATEBLOCK
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define isblockgetindices_     isblockgetindices
#define isblockrestoreindices_ isblockrestoreindices
#define iscreateblock_         iscreateblock
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL isblockgetindices_(IS *x,PetscInt *fa,size_t *ia,PetscErrorCode *ierr)
{
  PetscInt   *lx;

  *ierr = ISBlockGetIndices(*x,&lx); if (*ierr) return;
  *ia      = PetscIntAddressToFortran(fa,lx);
}

void PETSC_STDCALL isblockrestoreindices_(IS *x,PetscInt *fa,size_t *ia,PetscErrorCode *ierr)
{
  PetscInt *lx = PetscIntAddressFromFortran(fa,*ia);
  *ierr = ISBlockRestoreIndices(*x,&lx);
}

void PETSC_STDCALL iscreateblock_(MPI_Comm *comm,PetscInt *bs,PetscInt *n,PetscInt *idx,IS *is,PetscErrorCode *ierr)
{
  *ierr = ISCreateBlock((MPI_Comm)PetscToPointerComm(*comm),*bs,*n,idx,is);
}

EXTERN_C_END
