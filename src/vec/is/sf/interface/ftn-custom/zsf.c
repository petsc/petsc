#include <petsc/private/ftnimpl.h>
#include <petsc/private/sfimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscsfrestoregraph_     PETSCSFRESTOREGRAPH
  #define petscsfgetgraph_         PETSCSFGETGRAPH
  #define petscsfbcastbegin_       PETSCSFBCASTBEGIN
  #define petscsfbcastend_         PETSCSFBCASTEND
  #define petscsfreducebegin_      PETSCSFREDUCEBEGIN
  #define petscsfreduceend_        PETSCSFREDUCEEND
  #define petscsfgetleafranks_     PETSCSFGETLEAFRANKS
  #define petscsfgetrootranks_     PETSCSFGETROOTRANKS
  #define f90array1dcreatesfnode_  F90ARRAY1DCREATESFNODE
  #define f90array1ddestroysfnode_ F90ARRAY1DDESTROYSFNODE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscsfrestoregraph_     petscsfrestoregraph
  #define petscsfgetgraph_         petscsfgetgraph
  #define petscsfbcastbegin_       petscsfbcastbegin
  #define petscsfbcastend_         petscsfbcastend
  #define petscsfreducebegin_      petscsfreducebegin
  #define petscsfreduceend_        petscsfreduceend
  #define petscsfgetleafranks_     petscsfgetleafranks
  #define petscsfgetrootranks_     petscsfgetrootranks
  #define f90array1dcreatesfnode_  f90array1dcreatesfnode
  #define f90array1ddestroysfnode_ f90array1ddestroysfnode
#endif

PETSC_EXTERN void f90array1dcreatesfnode_(const PetscSFNode *, PetscInt *, PetscInt *, void *PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array1ddestroysfnode_(void *PETSC_F90_2PTR_PROTO_NOVAR);

PETSC_EXTERN void petscsfgetgraph_(PetscSF *sf, PetscInt *nroots, PetscInt *nleaves, F90Array1d *ailocal, F90Array1d *airemote, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(pilocal) PETSC_F90_2PTR_PROTO(piremote))
{
  const PetscInt    *ilocal;
  const PetscSFNode *iremote;
  PetscInt           nl, one = 1;

  *ierr = PetscSFGetGraph(*sf, nroots, nleaves, &ilocal, &iremote);
  if (*ierr) return;
  nl = *nleaves;
  if (!ilocal) nl = 0;
  *ierr = F90Array1dCreate((void *)ilocal, MPIU_INT, 1, nl, ailocal PETSC_F90_2PTR_PARAM(pilocal));
  /* this creates a memory leak */
  f90array1dcreatesfnode_(iremote, &one, nleaves, airemote PETSC_F90_2PTR_PARAM(piremote));
}

PETSC_EXTERN void petscsfrestoregraph_(PetscSF *sf, PetscInt *nroots, PetscInt *nleaves, F90Array1d *ailocal, F90Array1d *airemote, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(pilocal) PETSC_F90_2PTR_PROTO(piremote))
{
  *ierr = F90Array1dDestroy(ailocal, MPIU_INT PETSC_F90_2PTR_PARAM(pilocal));
  if (*ierr) return;
  f90array1ddestroysfnode_(airemote PETSC_F90_2PTR_PARAM(piremote));
}

PETSC_EXTERN void petscsfgetleafranks_(PetscSF *sf, PetscMPIInt *niranks, F90Array1d *airanks, F90Array1d *aioffset, F90Array1d *airootloc, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(piranks) PETSC_F90_2PTR_PROTO(pioffset) PETSC_F90_2PTR_PROTO(pirootloc))
{
  const PetscMPIInt *iranks   = NULL;
  const PetscInt    *ioffset  = NULL;
  const PetscInt    *irootloc = NULL;

  *ierr = PetscSFGetLeafRanks(*sf, niranks, &iranks, &ioffset, &irootloc);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)irootloc, MPIU_INT, 1, ioffset[*niranks], airootloc PETSC_F90_2PTR_PARAM(pirootloc));
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)iranks, MPI_INT, 1, *niranks, airanks PETSC_F90_2PTR_PARAM(piranks));
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)ioffset, MPIU_INT, 1, *niranks + 1, aioffset PETSC_F90_2PTR_PARAM(pioffset));
  if (*ierr) return;
}

PETSC_EXTERN void petscsfgetrootranks_(PetscSF *sf, PetscMPIInt *nranks, F90Array1d *aranks, F90Array1d *aroffset, F90Array1d *armine, F90Array1d *arremote, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(pranks) PETSC_F90_2PTR_PROTO(proffset) PETSC_F90_2PTR_PROTO(prmine) PETSC_F90_2PTR_PROTO(prremote))
{
  const PetscMPIInt *ranks   = NULL;
  const PetscInt    *roffset = NULL;
  const PetscInt    *rmine   = NULL;
  const PetscInt    *rremote = NULL;

  *ierr = PetscSFGetRootRanks(*sf, nranks, &ranks, &roffset, &rmine, &rremote);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)ranks, MPI_INT, 1, *nranks, aranks PETSC_F90_2PTR_PARAM(pranks));
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)roffset, MPIU_INT, 1, *nranks + 1, aroffset PETSC_F90_2PTR_PARAM(proffset));
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)rmine, MPIU_INT, 1, roffset[*nranks], armine PETSC_F90_2PTR_PARAM(prmine));
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)rremote, MPIU_INT, 1, roffset[*nranks], arremote PETSC_F90_2PTR_PARAM(prremote));
  if (*ierr) return;
}

#if defined(PETSC_HAVE_F90_ASSUMED_TYPE_NOT_PTR)
PETSC_EXTERN void petscsfbcastbegin_(PetscSF *sf, MPI_Fint *unit, const void *rptr, void *lptr, MPI_Fint *op, PetscErrorCode *ierr)
{
  MPI_Datatype dtype;
  MPI_Op       cop = MPI_Op_f2c(*op);

  *ierr = PetscMPIFortranDatatypeToC(*unit, &dtype);
  if (*ierr) return;
  *ierr = PetscSFBcastBegin(*sf, dtype, rptr, lptr, cop);
}

PETSC_EXTERN void petscsfbcastend_(PetscSF *sf, MPI_Fint *unit, const void *rptr, void *lptr, MPI_Fint *op, PetscErrorCode *ierr)
{
  MPI_Datatype dtype;
  MPI_Op       cop = MPI_Op_f2c(*op);

  *ierr = PetscMPIFortranDatatypeToC(*unit, &dtype);
  if (*ierr) return;
  *ierr = PetscSFBcastEnd(*sf, dtype, rptr, lptr, cop);
}

PETSC_EXTERN void petscsfreducebegin_(PetscSF *sf, MPI_Fint *unit, const void *lptr, void *rptr, MPI_Fint *op, PetscErrorCode *ierr)
{
  MPI_Datatype dtype;
  MPI_Op       cop = MPI_Op_f2c(*op);

  *ierr = PetscMPIFortranDatatypeToC(*unit, &dtype);
  if (*ierr) return;
  *ierr = PetscSFReduceBegin(*sf, dtype, lptr, rptr, cop);
}

PETSC_EXTERN void petscsfreduceend_(PetscSF *sf, MPI_Fint *unit, const void *lptr, void *rptr, MPI_Fint *op, PetscErrorCode *ierr)
{
  MPI_Datatype dtype;
  MPI_Op       cop = MPI_Op_f2c(*op);

  *ierr = PetscMPIFortranDatatypeToC(*unit, &dtype);
  if (*ierr) return;
  *ierr = PetscSFReduceEnd(*sf, dtype, lptr, rptr, cop);
}

#else

PETSC_EXTERN void petscsfbcastbegin_(PetscSF *sf, MPI_Fint *unit, F90Array1d *rptr, F90Array1d *lptr, MPI_Fint *op, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(rptrd) PETSC_F90_2PTR_PROTO(lptrd))
{
  MPI_Datatype dtype;
  const void  *rootdata;
  void        *leafdata;
  MPI_Op       cop = MPI_Op_f2c(*op);

  *ierr = PetscMPIFortranDatatypeToC(*unit, &dtype);
  if (*ierr) return;
  *ierr = F90Array1dAccess(rptr, dtype, (void **)&rootdata PETSC_F90_2PTR_PARAM(rptrd));
  if (*ierr) return;
  *ierr = F90Array1dAccess(lptr, dtype, (void **)&leafdata PETSC_F90_2PTR_PARAM(lptrd));
  if (*ierr) return;
  *ierr = PetscSFBcastBegin(*sf, dtype, rootdata, leafdata, cop);
}

PETSC_EXTERN void petscsfbcastend_(PetscSF *sf, MPI_Fint *unit, F90Array1d *rptr, F90Array1d *lptr, MPI_Fint *op, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(rptrd) PETSC_F90_2PTR_PROTO(lptrd))
{
  MPI_Datatype dtype;
  const void  *rootdata;
  void        *leafdata;
  MPI_Op       cop = MPI_Op_f2c(*op);

  *ierr = PetscMPIFortranDatatypeToC(*unit, &dtype);
  if (*ierr) return;
  *ierr = F90Array1dAccess(rptr, dtype, (void **)&rootdata PETSC_F90_2PTR_PARAM(rptrd));
  if (*ierr) return;
  *ierr = F90Array1dAccess(lptr, dtype, (void **)&leafdata PETSC_F90_2PTR_PARAM(lptrd));
  if (*ierr) return;
  *ierr = PetscSFBcastEnd(*sf, dtype, rootdata, leafdata, cop);
}

PETSC_EXTERN void petscsfreducebegin_(PetscSF *sf, MPI_Fint *unit, F90Array1d *lptr, F90Array1d *rptr, MPI_Fint *op, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(lptrd) PETSC_F90_2PTR_PROTO(rptrd))
{
  MPI_Datatype dtype;
  const void  *leafdata;
  void        *rootdata;
  MPI_Op       cop = MPI_Op_f2c(*op);

  *ierr = PetscMPIFortranDatatypeToC(*unit, &dtype);
  if (*ierr) return;
  *ierr = F90Array1dAccess(rptr, dtype, (void **)&rootdata PETSC_F90_2PTR_PARAM(rptrd));
  if (*ierr) return;
  *ierr = F90Array1dAccess(lptr, dtype, (void **)&leafdata PETSC_F90_2PTR_PARAM(lptrd));
  if (*ierr) return;
  *ierr = PetscSFReduceBegin(*sf, dtype, leafdata, rootdata, cop);
}

PETSC_EXTERN void petscsfreduceend_(PetscSF *sf, MPI_Fint *unit, F90Array1d *lptr, F90Array1d *rptr, MPI_Fint *op, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(lptrd) PETSC_F90_2PTR_PROTO(rptrd))
{
  MPI_Datatype dtype;
  const void  *leafdata;
  void        *rootdata;
  MPI_Op       cop = MPI_Op_f2c(*op);

  *ierr = PetscMPIFortranDatatypeToC(*unit, &dtype);
  if (*ierr) return;
  *ierr = F90Array1dAccess(rptr, dtype, (void **)&rootdata PETSC_F90_2PTR_PARAM(rptrd));
  if (*ierr) return;
  *ierr = F90Array1dAccess(lptr, dtype, (void **)&leafdata PETSC_F90_2PTR_PARAM(lptrd));
  if (*ierr) return;
  *ierr = PetscSFReduceEnd(*sf, dtype, leafdata, rootdata, cop);
}
#endif
