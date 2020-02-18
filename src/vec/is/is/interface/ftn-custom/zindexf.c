#include <petsc/private/fortranimpl.h>
#include <petscis.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petsclayoutfindowner_                      PETSCLAYOUTFINDOWNER
#define petsclayoutfindownerindex_                 PETSCLAYOUTFINDOWNERINDEX
#define isview_                                    ISVIEW
#define isgetindices_                              ISGETINDICES
#define isrestoreindices_                          ISRESTOREINDICES
#define isgettotalindices_                         ISGETTOTALINDICES
#define isrestoretotalindices_                     ISRESTORETOTALINDICES
#define isgetnonlocalindices_                      ISGETNONLOCALINDICES
#define isrestorenonlocalindices_                  ISRESTORENONLOCALINDICES
#define islocaltoglobalmappinggetindices_          ISLOCALTOGLOBALMAPPINGGETINDICES
#define islocaltoglobalmappingrestoreindices_      ISLOCALTOGLOBALMAPPINGRESTOREINDICES
#define islocaltoglobalmappinggetblockindices_     ISLOCALTOGLOBALMAPPINGGETBLOCKINDICES
#define islocaltoglobalmappingrestoreblockindices_ ISLOCALTOGLOBALMAPPINGRESTOREBLOCKINDICES
#define isviewfromoptions_                         ISVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petsclayoutfindowner_                      petsclayoutfindowner
#define petsclayoutfindownerindex_                 petsclayoutfindownerindex
#define isview_                                    isview
#define isgetindices_                              isgetindices
#define isrestoreindices_                          isrestoreindices
#define isgettotalindices_                         isgettotalindices
#define isrestoretotalindices_                     isrestoretotalindices
#define isgetnonlocalindices_                      isgetnonlocalindices
#define isrestorenonlocalindices_                  isrestorenonlocalindices
#define islocaltoglobalmappinggetindices_          islocaltoglobalmappinggetindices
#define islocaltoglobalmappingrestoreindices_      islocaltoglobalmappingrestoreindices
#define islocaltoglobalmappinggetblockindices_     islocaltoglobalmappinggetblockindices
#define islocaltoglobalmappingrestoreblockindices_ islocaltoglobalmappingrestoreblockindices
#define isviewfromoptions_                         isviewfromoptions
#endif

PETSC_EXTERN void PETSC_STDCALL petsclayoutfindowner_(PetscLayout *map,PetscInt *idx,PetscMPIInt *owner,int *ierr)
{
  *ierr = PetscLayoutFindOwner(*map,*idx,owner);
}

PETSC_EXTERN void PETSC_STDCALL petsclayoutfindownerindex_(PetscLayout *map,PetscInt *idx,PetscMPIInt *owner,PetscInt *ridx,int *ierr)
{
  *ierr = PetscLayoutFindOwnerIndex(*map,*idx,owner,ridx);
}

PETSC_EXTERN void PETSC_STDCALL isview_(IS *is,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = ISView(*is,v);
}

PETSC_EXTERN void PETSC_STDCALL isgetindices_(IS *x,PetscInt *fa,size_t *ia,PetscErrorCode *ierr)
{
  const PetscInt *lx;

  *ierr = ISGetIndices(*x,&lx); if (*ierr) return;
  *ia   = PetscIntAddressToFortran(fa,(PetscInt*)lx);
}

PETSC_EXTERN void PETSC_STDCALL isrestoreindices_(IS *x,PetscInt *fa,size_t *ia,PetscErrorCode *ierr)
{
  const PetscInt *lx = PetscIntAddressFromFortran(fa,*ia);
  *ierr = ISRestoreIndices(*x,&lx);
}

PETSC_EXTERN void PETSC_STDCALL isgettotalindices_(IS *x,PetscInt *fa,size_t *ia,PetscErrorCode *ierr)
{
  const PetscInt *lx;

  *ierr = ISGetTotalIndices(*x,&lx); if (*ierr) return;
  *ia   = PetscIntAddressToFortran(fa,(PetscInt*)lx);
}

PETSC_EXTERN void PETSC_STDCALL isrestoretotalindices_(IS *x,PetscInt *fa,size_t *ia,PetscErrorCode *ierr)
{
  const PetscInt *lx = PetscIntAddressFromFortran(fa,*ia);
  *ierr = ISRestoreTotalIndices(*x,&lx);
}

PETSC_EXTERN void PETSC_STDCALL isgetnonlocalindices_(IS *x,PetscInt *fa,size_t *ia,PetscErrorCode *ierr)
{
  const PetscInt *lx;

  *ierr = ISGetNonlocalIndices(*x,&lx); if (*ierr) return;
  *ia   = PetscIntAddressToFortran(fa,(PetscInt*)lx);
}

PETSC_EXTERN void PETSC_STDCALL isrestorenonlocalindices_(IS *x,PetscInt *fa,size_t *ia,PetscErrorCode *ierr)
{
  const PetscInt *lx = PetscIntAddressFromFortran(fa,*ia);
  *ierr = ISRestoreNonlocalIndices(*x,&lx);
}

PETSC_EXTERN void PETSC_STDCALL islocaltoglobalmappinggetindices_(ISLocalToGlobalMapping *x,PetscInt *fa,size_t *ia,PetscErrorCode *ierr)
{
  const PetscInt *lx;

  *ierr = ISLocalToGlobalMappingGetIndices(*x,&lx); if (*ierr) return;
  *ia   = PetscIntAddressToFortran(fa,(PetscInt*)lx);
}

PETSC_EXTERN void PETSC_STDCALL islocaltoglobalmappingrestoreindices_(ISLocalToGlobalMapping *x,PetscInt *fa,size_t *ia,PetscErrorCode *ierr)
{
  const PetscInt *lx = PetscIntAddressFromFortran(fa,*ia);
  *ierr = ISLocalToGlobalMappingRestoreIndices(*x,&lx);
}

PETSC_EXTERN void PETSC_STDCALL islocaltoglobalmappinggetblockindices_(ISLocalToGlobalMapping *x,PetscInt *fa,size_t *ia,PetscErrorCode *ierr)
{
  const PetscInt *lx;

  *ierr = ISLocalToGlobalMappingGetBlockIndices(*x,&lx); if (*ierr) return;
  *ia   = PetscIntAddressToFortran(fa,(PetscInt*)lx);
}

PETSC_EXTERN void PETSC_STDCALL islocaltoglobalmappingrestoreblockindices_(ISLocalToGlobalMapping *x,PetscInt *fa,size_t *ia,PetscErrorCode *ierr)
{
  const PetscInt *lx = PetscIntAddressFromFortran(fa,*ia);
  *ierr = ISLocalToGlobalMappingRestoreBlockIndices(*x,&lx);
}

PETSC_EXTERN void PETSC_STDCALL isviewfromoptions_(IS *ao,PetscObject obj,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = ISViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}
