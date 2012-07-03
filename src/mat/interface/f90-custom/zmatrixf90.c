
#include <petscmat.h>
#include <../src/sys/f90-src/f90impl.h>

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define matgetarrayf90_            MATGETARRAYF90
#define matrestorearrayf90_        MATRESTOREARRAYF90
#define matgetghostsf90_           MATGETGHOSTSF90
#define matgetrowijf90_            MATGETROWIJF90
#define matrestorerowijf90_        MATRESTOREROWIJF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matgetarrayf90_            matgetarrayf90
#define matrestorearrayf90_        matrestorearrayf90
#define matgetghostsf90_           matgetghostsf90
#define matgetrowijf90_            matgetrowijf90
#define matrestorerowijf90_        matrestorerowijf90
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL matgetghostsf90_(Mat *mat,F90Array1d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *ghosts;
  PetscInt       N;

  *ierr = MatGetGhosts(*mat,&N,&ghosts); if (*ierr) return;
  *ierr = F90Array1dCreate((PetscInt *)ghosts,PETSC_INT,1,N,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
void PETSC_STDCALL matgetarrayf90_(Mat *mat,F90Array2d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  PetscInt     m,n;
  *ierr = MatGetArray(*mat,&fa);       if (*ierr) return;
  *ierr = MatGetLocalSize(*mat,&m,&n); if (*ierr) return;
  *ierr = F90Array2dCreate(fa,PETSC_SCALAR,1,m,1,n,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
void PETSC_STDCALL matrestorearrayf90_(Mat *mat,F90Array2d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array2dAccess(ptr,PETSC_SCALAR,(void **)&fa PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = F90Array2dDestroy(ptr,PETSC_SCALAR PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = MatRestoreArray(*mat,&fa);
}
void PETSC_STDCALL matgetrowijf90_(Mat *B,PetscInt *shift,PetscBool  *sym,PetscBool  *blockcompressed,PetscInt *n,F90Array1d *ia,
                                F90Array1d *ja,PetscBool  *done,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(iad)  PETSC_F90_2PTR_PROTO(jad))
{
  PetscInt *IA,*JA;
  *ierr = MatGetRowIJ(*B,*shift,*sym,*blockcompressed,n,&IA,&JA,done);if (*ierr) return; if (!*done) return;
  *ierr = F90Array1dCreate((PetscInt *)IA,PETSC_INT,1,*n+1,ia PETSC_F90_2PTR_PARAM(iad));
  *ierr = F90Array1dCreate((PetscInt *)JA,PETSC_INT,1,IA[*n],ja PETSC_F90_2PTR_PARAM(jad));
}

void PETSC_STDCALL matrestorerowijf90_(Mat *B,PetscInt *shift,PetscBool  *sym,PetscBool  *blockcompressed, PetscInt *n,F90Array1d *ia,
                                F90Array1d *ja,PetscBool  *done,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(iad)  PETSC_F90_2PTR_PROTO(jad))
{
  PetscInt *IA,*JA;
  *ierr = F90Array1dAccess(ia,PETSC_INT,(void **)&IA PETSC_F90_2PTR_PARAM(iad));if (*ierr) return;
  *ierr = F90Array1dDestroy(ia,PETSC_INT PETSC_F90_2PTR_PARAM(iad));if (*ierr) return;
  *ierr = F90Array1dAccess(ja,PETSC_INT,(void **)&JA PETSC_F90_2PTR_PARAM(jad));if (*ierr) return;
  *ierr = F90Array1dDestroy(ja,PETSC_INT PETSC_F90_2PTR_PARAM(jad));if (*ierr) return;
  *ierr = MatRestoreRowIJ(*B,*shift,*sym,*blockcompressed,n,&IA,&JA,done);
}

EXTERN_C_END


