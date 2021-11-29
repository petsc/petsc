#include <petsc/private/f90impl.h>
#include <petsc/private/sfimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscsfview_            PETSCSFVIEW
#define petscsfgetgraph_        PETSCSFGETGRAPH
#define petscsfbcastbegin_      PETSCSFBCASTBEGIN
#define petscsfbcastend_        PETSCSFBCASTEND
#define f90arraysfnodecreate_   F90ARRAYSFNODECREATE
#define petscsfviewfromoptions_ PETSCSFVIEWFROMOPTIONS
#define petscsfdestroy_         PETSCSFDESTROY
#define petscsfsetgraph_        PETSCSFSETGRAPH
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscsfgetgraph_        petscsfgetgraph
#define petscsfview_            petscsfview
#define petscsfbcastbegin_      petscsfbcastbegin
#define petscsfbcastend_        petscsfbcastend
#define f90arraysfnodecreate_   f90arraysfnodecreate
#define petscsfviewfromoptions_ petscsfviewfromoptions
#define petscsfdestroy_         petscsfdestroy
#define petscsfsetgraph_        petscsfsetgraph
#endif

PETSC_EXTERN void f90arraysfnodecreate_(const PetscInt *,PetscInt *,void * PETSC_F90_2PTR_PROTO_NOVAR);

PETSC_EXTERN void  petscsfsetgraph_(PetscSF *sf,PetscInt *nroots,PetscInt *nleaves, PetscInt *ilocal,PetscCopyMode *localmode, PetscSFNode *iremote,PetscCopyMode *remotemode, int *ierr)
{
  if (ilocal == PETSC_NULL_INTEGER_Fortran) ilocal = NULL;
  *ierr = PetscSFSetGraph(*sf,*nroots,*nleaves,ilocal,*localmode,iremote,*remotemode);
}

PETSC_EXTERN void petscsfview_(PetscSF *sf, PetscViewer *vin, PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscSFView(*sf, v);
}

PETSC_EXTERN void  petscsfgetgraph_(PetscSF *sf,PetscInt *nroots,PetscInt *nleaves, F90Array1d  *ailocal, F90Array1d  *airemote, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(pilocal) PETSC_F90_2PTR_PROTO(piremote))
{
  const PetscInt    *ilocal;
  const PetscSFNode *iremote;
  PetscInt          nl;

  *ierr = PetscSFGetGraph(*sf,nroots,nleaves,&ilocal,&iremote);if (*ierr) return;
  nl = *nleaves;
  if (!ilocal) nl = 0;
  *ierr = F90Array1dCreate((void*)ilocal,MPIU_INT,1,nl, ailocal PETSC_F90_2PTR_PARAM(pilocal));
  /* this creates a memory leak */
  f90arraysfnodecreate_((PetscInt*)iremote,nleaves, airemote PETSC_F90_2PTR_PARAM(piremote));
}

#if defined(PETSC_HAVE_F90_ASSUMED_TYPE_NOT_PTR)
PETSC_EXTERN void petscsfbcastbegin_(PetscSF *sf, MPI_Fint *unit, const void *rptr, void *lptr, MPI_Fint *op, PetscErrorCode *ierr)
{
  MPI_Datatype dtype;
  MPI_Op       cop = MPI_Op_f2c(*op);

  *ierr = PetscMPIFortranDatatypeToC(*unit,&dtype);if (*ierr) return;
  *ierr = PetscSFBcastBegin(*sf, dtype, rptr, lptr, cop);
}

PETSC_EXTERN void petscsfbcastend_(PetscSF *sf, MPI_Fint *unit, const void *rptr, void *lptr, MPI_Fint *op, PetscErrorCode *ierr)
{
  MPI_Datatype dtype;
  MPI_Op       cop = MPI_Op_f2c(*op);

  *ierr = PetscMPIFortranDatatypeToC(*unit,&dtype);if (*ierr) return;
  *ierr = PetscSFBcastEnd(*sf, dtype, rptr, lptr, cop);
}

#else

PETSC_EXTERN void petscsfbcastbegin_(PetscSF *sf, MPI_Fint *unit,F90Array1d *rptr, F90Array1d *lptr, MPI_Fint *op, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(rptrd) PETSC_F90_2PTR_PROTO(lptrd))
{
  MPI_Datatype dtype;
  const void   *rootdata;
  void         *leafdata;
  MPI_Op       cop = MPI_Op_f2c(*op);

  *ierr = PetscMPIFortranDatatypeToC(*unit,&dtype);if (*ierr) return;
  *ierr = F90Array1dAccess(rptr, dtype, (void**) &rootdata PETSC_F90_2PTR_PARAM(rptrd));if (*ierr) return;
  *ierr = F90Array1dAccess(lptr, dtype, (void**) &leafdata PETSC_F90_2PTR_PARAM(lptrd));if (*ierr) return;
  *ierr = PetscSFBcastBegin(*sf, dtype, rootdata, leafdata, cop);
}

PETSC_EXTERN void petscsfbcastend_(PetscSF *sf, MPI_Fint *unit,F90Array1d *rptr, F90Array1d *lptr, MPI_Fint *op, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(rptrd) PETSC_F90_2PTR_PROTO(lptrd))
{
  MPI_Datatype dtype;
  const void   *rootdata;
  void         *leafdata;
  MPI_Op       cop = MPI_Op_f2c(*op);

  *ierr = PetscMPIFortranDatatypeToC(*unit,&dtype);if (*ierr) return;
  *ierr = F90Array1dAccess(rptr, dtype, (void**) &rootdata PETSC_F90_2PTR_PARAM(rptrd));if (*ierr) return;
  *ierr = F90Array1dAccess(lptr, dtype, (void**) &leafdata PETSC_F90_2PTR_PARAM(lptrd));if (*ierr) return;
  *ierr = PetscSFBcastEnd(*sf, dtype, rootdata, leafdata, cop);
}
PETSC_EXTERN void petscsfviewfromoptions_(PetscSF *ao,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = PetscSFViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}

PETSC_EXTERN void petscsfdestroy_(PetscSF *x,int *ierr)
{
  PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(x);
  *ierr = PetscSFDestroy(x); if (*ierr) return;
  PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(x);
}

#endif
