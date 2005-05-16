
#include "zpetsc.h"
#include "petscis.h"
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define isduplicate_           ISDUPLICATE
#define ispartitioningcount_   ISPARTITIONINGCOUNT
#define isdestroy_             ISDESTROY
#define iscreatestride_        ISCREATESTRIDE
#define iscreategeneral_       ISCREATEGENERAL
#define isgetindices_          ISGETINDICES
#define isrestoreindices_      ISRESTOREINDICES
#define isblockgetindices_     ISBLOCKGETINDICES
#define isblockrestoreindices_ ISBLOCKRESTOREINDICES
#define iscreateblock_         ISCREATEBLOCK
#define isblock_               ISBLOCK
#define isstride_              ISSTRIDE
#define ispermutation_         ISPERMUTATION
#define isidentity_            ISIDENTITY
#define issorted_              ISSORTED
#define isequal_               ISEQUAL
#define isinvertpermutation_   ISINVERTPERMUTATION
#define isview_                ISVIEW
#define iscoloringcreate_      ISCOLORINGCREATE
#define islocaltoglobalmappingcreate_ ISLOCALTOGLOBALMAPPINGCREATE
#define islocaltoglobalmappingblock_ ISLOCALTOGLOBALMAPPINGBLOCK
#define isallgather_                  ISALLGATHER
#define iscoloringdestroy_            ISCOLORINGDESTROY
#define iscoloringview_               ISCOLORINGVIEW
#define ispartitioningtonumbering_    ISPARTITIONINGTONUMBERING
#define islocaltoglobalmappingapply_  ISLOCALTOGLOBALMAPPINGAPPLY
#define islocaltoglobalmappingview_  ISLOCALTOGLOBALMAPPINGVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define isduplicate_           isduplicate
#define islocaltoglobalmappingview_   islocaltoglobalmappingview
#define islocaltoglobalmappingapply_  islocaltoglobalmappingapply
#define iscoloringview_        iscoloringview
#define iscoloringdestroy_     iscoloringdestroy
#define isview_                isview
#define isinvertpermutation_   isinvertpermutation
#define isdestroy_             isdestroy
#define iscreatestride_        iscreatestride
#define iscreategeneral_       iscreategeneral
#define isgetindices_          isgetindices
#define isrestoreindices_      isrestoreindices
#define isblockgetindices_     isblockgetindices
#define isblockrestoreindices_ isblockrestoreindices
#define iscreateblock_         iscreateblock
#define isblock_               isblock
#define isstride_              isstride
#define ispermutation_         ispermutation
#define isidentity_            isidentity
#define issorted_              issorted
#define isequal_               isequal
#define iscoloringcreate_      iscoloringcreate
#define islocaltoglobalmappingcreate_ islocaltoglobalmappingcreate
#define islocaltoglobalmappingblock_ islocaltoglobalmappingblock
#define isallgather_                  isallgather
#define ispartitioningcount_          ispartitioningcount
#define ispartitioningtonumbering_    ispartitioningtonumbering
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL isduplicate_(IS *is,IS *newis,PetscErrorCode *ierr)
{
  *ierr = ISDuplicate(*is,newis);
}

void PETSC_STDCALL islocaltoglobalmappingview_(ISLocalToGlobalMapping *mapping,PetscViewer *viewer,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECTDEREFERENCE(viewer);
  *ierr = ISLocalToGlobalMappingView(*mapping,*viewer);
}

/*
   This is the same as the macro ISLocalToGlobalMappingApply() except it does not
  return error codes.
*/
void PETSC_STDCALL islocaltoglobalmappingapply_(ISLocalToGlobalMapping *mapping,PetscInt *N,PetscInt *in,PetscInt *out,PetscErrorCode *ierr)
{
  PetscInt i,*idx = (*mapping)->indices,Nmax = (*mapping)->n;
  for (i=0; i<(*N); i++) {
    if (in[i] < 0) {out[i] = in[i]; continue;}
    if (in[i] >= Nmax) {
      *ierr = PetscError(__LINE__,"ISLocalToGlobalMappingApply_Fortran",__FILE__,__SDIR__,1,1,"Index out of range");
      return;
    }
    out[i] = idx[in[i]];
  }
}

void PETSC_STDCALL ispartitioningtonumbering_(IS *is,IS *isout,PetscErrorCode *ierr)
{
  *ierr = ISPartitioningToNumbering(*is,isout);
}

void PETSC_STDCALL ispartitioningcount_(IS *is,PetscInt *count,PetscErrorCode *ierr)
{
  *ierr = ISPartitioningCount(*is,count);
}

void PETSC_STDCALL iscoloringdestroy_(ISColoring *iscoloring,PetscErrorCode *ierr)
{
  *ierr = ISColoringDestroy(*iscoloring);
}

void PETSC_STDCALL iscoloringview_(ISColoring *iscoloring,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = ISColoringView(*iscoloring,v);
}

void PETSC_STDCALL isview_(IS *is,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = ISView(*is,v);
}

void PETSC_STDCALL isequal_(IS *is1,IS *is2,PetscTruth *flg,PetscErrorCode *ierr)
{
  *ierr = ISEqual(*is1,*is2,flg);
}

void PETSC_STDCALL isidentity_(IS *is,PetscTruth *ident,PetscErrorCode *ierr)
{
  *ierr = ISIdentity(*is,ident);
}

void PETSC_STDCALL issorted_(IS *is,PetscTruth *flg,PetscErrorCode *ierr)
{
  *ierr = ISSorted(*is,flg);
}

void PETSC_STDCALL ispermutation_(IS *is,PetscTruth *perm,PetscErrorCode *ierr){
  *ierr = ISPermutation(*is,perm);
}

void PETSC_STDCALL isstride_(IS *is,PetscTruth *flag,PetscErrorCode *ierr)
{
  *ierr = ISStride(*is,flag);
}

void PETSC_STDCALL isblockgetindices_(IS *x,PetscInt *fa,size_t *ia,PetscErrorCode *ierr)
{
  PetscInt   *lx;

  *ierr = ISGetIndices(*x,&lx); if (*ierr) return;
  *ia      = PetscIntAddressToFortran(fa,lx);
}

void PETSC_STDCALL isblockrestoreindices_(IS *x,PetscInt *fa,size_t *ia,PetscErrorCode *ierr)
{
  PetscInt *lx = PetscIntAddressFromFortran(fa,*ia);

  *ierr = ISRestoreIndices(*x,&lx);
}

void PETSC_STDCALL isblock_(IS *is,PetscTruth *flag,PetscErrorCode *ierr)
{
  *ierr = ISBlock(*is,flag);
}

void PETSC_STDCALL isgetindices_(IS *x,PetscInt *fa,size_t *ia,PetscErrorCode *ierr)
{
  PetscInt   *lx;

  *ierr = ISGetIndices(*x,&lx); if (*ierr) return;
  *ia      = PetscIntAddressToFortran(fa,lx);
}

void PETSC_STDCALL isrestoreindices_(IS *x,PetscInt *fa,size_t *ia,PetscErrorCode *ierr)
{
  PetscInt *lx = PetscIntAddressFromFortran(fa,*ia);

  *ierr = ISRestoreIndices(*x,&lx);
}

void PETSC_STDCALL iscreategeneral_(MPI_Comm *comm,PetscInt *n,PetscInt *idx,IS *is,PetscErrorCode *ierr)
{
  *ierr = ISCreateGeneral((MPI_Comm)PetscToPointerComm(*comm),*n,idx,is);
}

void PETSC_STDCALL isinvertpermutation_(IS *is,PetscInt *nlocal,IS *isout,PetscErrorCode *ierr)
{
  *ierr = ISInvertPermutation(*is,*nlocal,isout);
}

void PETSC_STDCALL iscreateblock_(MPI_Comm *comm,PetscInt *bs,PetscInt *n,PetscInt *idx,IS *is,PetscErrorCode *ierr)
{
  *ierr = ISCreateBlock((MPI_Comm)PetscToPointerComm(*comm),*bs,*n,idx,is);
}

void PETSC_STDCALL iscreatestride_(MPI_Comm *comm,PetscInt *n,PetscInt *first,PetscInt *step,
                               IS *is,PetscErrorCode *ierr)
{
  *ierr = ISCreateStride((MPI_Comm)PetscToPointerComm(*comm),*n,*first,*step,is);
}

void PETSC_STDCALL isdestroy_(IS *is,PetscErrorCode *ierr)
{
  *ierr = ISDestroy(*is);
}

void PETSC_STDCALL iscoloringcreate_(MPI_Comm *comm,PetscInt *n,PetscInt *colors,ISColoring *iscoloring,PetscErrorCode *ierr)
{
  ISColoringValue *color;
  PetscInt             i;

  /* copies the colors[] array since that is kept by the ISColoring that is created */
  *ierr = PetscMalloc((*n+1)*sizeof(ISColoringValue),&color);if (*ierr) return;
  for (i=0; i<(*n); i++) {
    if (colors[i] > IS_COLORING_MAX) {
      *ierr = PetscError(__LINE__,"ISColoringCreate_Fortran",__FILE__,__SDIR__,1,1,"Color too large");
      return;
    }
    if (colors[i] < 0) {
      *ierr = PetscError(__LINE__,"ISColoringCreate_Fortran",__FILE__,__SDIR__,1,1,"Color cannot be negative");
      return;
    }
    color[i] = (ISColoringValue)colors[i];
  }
  *ierr = ISColoringCreate((MPI_Comm)PetscToPointerComm(*comm),*n,color,iscoloring);
}

void PETSC_STDCALL islocaltoglobalmappingcreate_(MPI_Comm *comm,PetscInt *n,PetscInt *indices,ISLocalToGlobalMapping *mapping,PetscErrorCode *ierr)
{
  *ierr = ISLocalToGlobalMappingCreate((MPI_Comm)PetscToPointerComm(*comm),*n,indices,mapping);
}

void PETSC_STDCALL islocaltoglobalmappingblock_(ISLocalToGlobalMapping *inmap,PetscInt bs,ISLocalToGlobalMapping *outmap,PetscErrorCode *ierr)
{
  *ierr = ISLocalToGlobalMappingBlock(*inmap,bs,outmap);
}

void PETSC_STDCALL isallgather_(IS *is,IS *isout,PetscErrorCode *ierr)
{
  *ierr = ISAllGather(*is,isout);

}

EXTERN_C_END

