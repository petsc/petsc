#include <petscdmcomposite.h>
#include <petsc/private/ftnimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define dmcompositegetentries1_          DMCOMPOSITEGETENTRIES1
  #define dmcompositegetentries2_          DMCOMPOSITEGETENTRIES2
  #define dmcompositegetentries3_          DMCOMPOSITEGETENTRIES3
  #define dmcompositegetentries4_          DMCOMPOSITEGETENTRIES4
  #define dmcompositegetentries5_          DMCOMPOSITEGETENTRIES5
  #define dmcompositegetaccess4_           DMCOMPOSITEGETACCESS4
  #define dmcompositescatter4_             DMCOMPOSITESCATTER4
  #define dmcompositerestoreaccess4_       DMCOMPOSITERESTOREACCESS4
  #define dmcompositegetlocalvectors4_     DMCOMPOSITEGETLOCALVECTORS4
  #define dmcompositerestorelocalvectors4_ DMCOMPOSITERESTORELOCALVECTORS4
  #define dmcompositegetglobaliss_         DMCOMPOSITEGETGLOBALISS
  #define dmcompositerestoreglobaliss_     DMCOMPOSITERESTOREGLOBALISS
  #define dmcompositegetlocaliss_          DMCOMPOSITEGETLOCALISS
  #define dmcompositerestorelocaliss_      DMCOMPOSITERESTORELOCALISS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define dmcompositegetentries1_          dmcompositegetentries1
  #define dmcompositegetentries2_          dmcompositegetentries2
  #define dmcompositegetentries3_          dmcompositegetentries3
  #define dmcompositegetentries4_          dmcompositegetentries4
  #define dmcompositegetentries5_          dmcompositegetentries5
  #define dmcompositegetaccess4_           dmcompositegetaccess4
  #define dmcompositescatter4_             dmcompositescatter4
  #define dmcompositerestoreaccess4_       dmcompositerestoreaccess4
  #define dmcompositegetlocalvectors4_     dmcompositegetlocalvectors4
  #define dmcompositerestorelocalvectors4_ dmcompositerestorelocalvectors4
  #define dmcompositegetglobaliss_         dmcompositegetglobaliss
  #define dmcompositerestoreglobaliss_     dmcompositerestoreglobaliss
  #define dmcompositegetlocaliss_          dmcompositegetlocaliss
  #define dmcompositerestorelocaliss_      dmcompositerestorelocaliss
#endif

PETSC_EXTERN void dmcompositegetentries1_(DM *dm, DM *da1, PetscErrorCode *ierr)
{
  *ierr = DMCompositeGetEntries(*dm, da1);
}

PETSC_EXTERN void dmcompositegetentries2_(DM *dm, DM *da1, DM *da2, PetscErrorCode *ierr)
{
  *ierr = DMCompositeGetEntries(*dm, da1, da2);
}

PETSC_EXTERN void dmcompositegetentries3_(DM *dm, DM *da1, DM *da2, DM *da3, PetscErrorCode *ierr)
{
  *ierr = DMCompositeGetEntries(*dm, da1, da2, da3);
}

PETSC_EXTERN void dmcompositegetentries4_(DM *dm, DM *da1, DM *da2, DM *da3, DM *da4, PetscErrorCode *ierr)
{
  *ierr = DMCompositeGetEntries(*dm, da1, da2, da3, da4);
}

PETSC_EXTERN void dmcompositegetentries5_(DM *dm, DM *da1, DM *da2, DM *da3, DM *da4, DM *da5, PetscErrorCode *ierr)
{
  *ierr = DMCompositeGetEntries(*dm, da1, da2, da3, da4, da5);
}

PETSC_EXTERN void dmcompositegetglobaliss_(DM *dm, F90Array1d *ptr, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  IS      *ais;
  PetscInt ndm;

  *ierr = DMCompositeGetGlobalISs(*dm, &ais);
  if (*ierr) return;
  *ierr = DMCompositeGetNumberDM(*dm, &ndm);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)ais, MPIU_FORTRANADDR, 1, ndm, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void dmcompositerestoreglobaliss_(DM *dm, F90Array1d *ptr, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  IS      *ais;
  PetscInt ndm;

  *ierr = F90Array1dAccess(ptr, MPIU_FORTRANADDR, (void **)&ais PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = DMCompositeGetNumberDM(*dm, &ndm);
  for (PetscInt i = 0; i < ndm; i++) {
    *ierr = ISDestroy(&ais[i]);
    if (*ierr) return;
  }
  *ierr = PetscFree(ais);
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void dmcompositegetlocaliss_(DM *dm, F90Array1d *ptr, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  IS      *ais;
  PetscInt ndm;

  *ierr = DMCompositeGetLocalISs(*dm, &ais);
  if (*ierr) return;
  *ierr = DMCompositeGetNumberDM(*dm, &ndm);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)ais, MPIU_FORTRANADDR, 1, ndm, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void dmcompositerestorelocaliss_(DM *dm, F90Array1d *ptr, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  IS      *ais;
  PetscInt ndm;

  *ierr = F90Array1dAccess(ptr, MPIU_FORTRANADDR, (void **)&ais PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = DMCompositeGetNumberDM(*dm, &ndm);
  for (PetscInt i = 0; i < ndm; i++) {
    *ierr = ISDestroy(&ais[i]);
    if (*ierr) return;
  }
  *ierr = PetscFree(ais);
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd));
}
