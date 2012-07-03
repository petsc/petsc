#include <petsc-private/fortranimpl.h>
#include <petscvec.h>
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define veccreatempiwitharray_        VECCREATEMPIWITHARRAY
#define veccreateghostblockwitharray_ VECCREATEGHOSTBLOCKWITHARRAY
#define veccreateghostwitharray_      VECCREATEGHOSTWITHARRAY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define veccreatempiwitharray_        veccreatempiwitharray
#define veccreateghostblockwitharray_ veccreateghostblockwitharray
#define veccreateghostwitharray_      veccreateghostwitharray
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL veccreatempiwitharray_(MPI_Comm *comm,PetscInt *bs,PetscInt *n,PetscInt *N,PetscScalar *s,Vec *V,PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(s);
  *ierr = VecCreateMPIWithArray(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*bs,*n,*N,s,V);
}

void PETSC_STDCALL veccreateghostblockwitharray_(MPI_Comm *comm,PetscInt *bs,PetscInt *n,PetscInt *N,PetscInt *nghost,PetscInt *ghosts,
                              PetscScalar *array,Vec *vv,PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(array);
  *ierr = VecCreateGhostBlockWithArray(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*bs,*n,*N,*nghost,
                                    ghosts,array,vv);
}

void PETSC_STDCALL veccreateghostwitharray_(MPI_Comm *comm,PetscInt *n,PetscInt *N,PetscInt *nghost,PetscInt *ghosts,PetscScalar *array,
                              Vec *vv,PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(array);
  *ierr = VecCreateGhostWithArray(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*n,*N,*nghost,
                                    ghosts,array,vv);
}

EXTERN_C_END
