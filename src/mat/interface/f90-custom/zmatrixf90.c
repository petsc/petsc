
#include <petscmat.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matdensegetarrayf90_       MATDENSEGETARRAYF90
#define matdenserestorearrayf90_   MATDENSERESTOREARRAYF90
#define matdensegetcolumnf90_      MATDENSEGETCOLUMNF90
#define matdenserestorecolumnf90_  MATDENSERESTORECOLUMNF90
#define matseqaijgetarrayf90_      MATSEQAIJGETARRAYF90
#define matseqaijrestorearrayf90_  MATSEQAIJRESTOREARRAYF90
#define matgetghostsf90_           MATGETGHOSTSF90
#define matgetrowijf90_            MATGETROWIJF90
#define matrestorerowijf90_        MATRESTOREROWIJF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matdensegetarrayf90_       matdensegetarrayf90
#define matdenserestorearrayf90_   matdenserestorearrayf90
#define matdensegetcolumnf90_      matdensegetcolumnf90
#define matdenserestorecolumnf90_  matdenserestorecolumnf90
#define matseqaijgetarrayf90_      matseqaijgetarrayf90
#define matseqaijrestorearrayf90_  matseqaijrestorearrayf90
#define matgetghostsf90_           matgetghostsf90
#define matgetrowijf90_            matgetrowijf90
#define matrestorerowijf90_        matrestorerowijf90
#endif

PETSC_EXTERN void matgetghostsf90_(Mat *mat,F90Array1d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *ghosts;
  PetscInt       N;

  *ierr = MatGetGhosts(*mat,&N,&ghosts); if (*ierr) return;
  *ierr = F90Array1dCreate((PetscInt*)ghosts,MPIU_INT,1,N,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matdensegetarrayf90_(Mat *mat,F90Array2d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  PetscInt    m,N;
  *ierr = MatDenseGetArray(*mat,&fa); if (*ierr) return;
  *ierr = MatGetLocalSize(*mat,&m,NULL); if (*ierr) return;
  *ierr = MatGetSize(*mat,NULL,&N); if (*ierr) return;
  *ierr = F90Array2dCreate(fa,MPIU_SCALAR,1,m,1,N,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matdenserestorearrayf90_(Mat *mat,F90Array2d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array2dAccess(ptr,MPIU_SCALAR,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = F90Array2dDestroy(ptr,MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = MatDenseRestoreArray(*mat,&fa);
}
PETSC_EXTERN void matdensegetcolumnf90_(Mat *mat,PetscInt *col,F90Array1d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  PetscInt    m;
  *ierr = MatDenseGetColumn(*mat,*col,&fa); if (*ierr) return;
  *ierr = MatGetLocalSize(*mat,&m,NULL); if (*ierr) return;
  *ierr = F90Array1dCreate(fa,MPIU_SCALAR,1,m,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matdenserestorecolumnf90_(Mat *mat,F90Array1d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array1dDestroy(ptr,MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = MatDenseRestoreColumn(*mat,&fa);
}
PETSC_EXTERN void matseqaijgetarrayf90_(Mat *mat,F90Array1d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  PetscInt    m,n;
  *ierr = MatSeqAIJGetArray(*mat,&fa); if (*ierr) return;
  *ierr = MatGetLocalSize(*mat,&m,&n); if (*ierr) return;
  *ierr = F90Array1dCreate(fa,MPIU_SCALAR,1,m*n,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matseqaijrestorearrayf90_(Mat *mat,F90Array1d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array1dAccess(ptr,MPIU_SCALAR,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr,MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = MatSeqAIJRestoreArray(*mat,&fa);
}
PETSC_EXTERN void matgetrowijf90_(Mat *B,PetscInt *shift,PetscBool *sym,PetscBool *blockcompressed,PetscInt *n,F90Array1d *ia,
                                F90Array1d *ja,PetscBool  *done,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(iad)  PETSC_F90_2PTR_PROTO(jad))
{
  const PetscInt *IA,*JA;
  *ierr = MatGetRowIJ(*B,*shift,*sym,*blockcompressed,n,&IA,&JA,done);if (*ierr) return;
  if (!*done) return;
  *ierr = F90Array1dCreate((PetscInt*)IA,MPIU_INT,1,*n+1,ia PETSC_F90_2PTR_PARAM(iad));
  *ierr = F90Array1dCreate((PetscInt*)JA,MPIU_INT,1,IA[*n],ja PETSC_F90_2PTR_PARAM(jad));
}

PETSC_EXTERN void matrestorerowijf90_(Mat *B,PetscInt *shift,PetscBool *sym,PetscBool *blockcompressed, PetscInt *n,F90Array1d *ia,
                                F90Array1d *ja,PetscBool  *done,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(iad)  PETSC_F90_2PTR_PROTO(jad))
{
  const PetscInt *IA,*JA;
  *ierr = F90Array1dAccess(ia,MPIU_INT,(void**)&IA PETSC_F90_2PTR_PARAM(iad));if (*ierr) return;
  *ierr = F90Array1dDestroy(ia,MPIU_INT PETSC_F90_2PTR_PARAM(iad));if (*ierr) return;
  *ierr = F90Array1dAccess(ja,MPIU_INT,(void**)&JA PETSC_F90_2PTR_PARAM(jad));if (*ierr) return;
  *ierr = F90Array1dDestroy(ja,MPIU_INT PETSC_F90_2PTR_PARAM(jad));if (*ierr) return;
  *ierr = MatRestoreRowIJ(*B,*shift,*sym,*blockcompressed,n,&IA,&JA,done);
}
