/*$Id: zis.c,v 1.41 2001/06/21 21:19:50 bsmith Exp $*/

#include "src/fortran/custom/zpetsc.h"
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

void PETSC_STDCALL isduplicate_(IS *is,IS *newis,int *ierr)
{
  *ierr = ISDuplicate(*is,newis);
}

void PETSC_STDCALL islocaltoglobalmappingview_(ISLocalToGlobalMapping *mapping,PetscViewer *viewer,int *ierr)
{
  CHKFORTRANNULLOBJECT(viewer);
  *ierr = ISLocalToGlobalMappingView(*mapping,*viewer);
}

/*
   This is the same as the macro ISLocalToGlobalMappingApply() except it does not
  return error codes.
*/
void PETSC_STDCALL islocaltoglobalmappingapply_(ISLocalToGlobalMapping *mapping,int *N,int *in,int *out,int *ierr)
{
  int i,*idx = (*mapping)->indices,Nmax = (*mapping)->n;
  for (i=0; i<(*N); i++) {
    if (in[i] < 0) {out[i] = in[i]; continue;}
    if (in[i] >= Nmax) {
      *ierr = PetscError(__LINE__,"ISLocalToGlobalMappingApply_Fortran",__FILE__,__SDIR__,1,1,"Index out of range");
      return;
    }
    out[i] = idx[in[i]];
  }
}

void PETSC_STDCALL ispartitioningtonumbering_(IS *is,IS *isout,int *ierr)
{
  *ierr = ISPartitioningToNumbering(*is,isout);
}

void PETSC_STDCALL ispartitioningcount_(IS *is,int *count,int *ierr)
{
  *ierr = ISPartitioningCount(*is,count);
}

void PETSC_STDCALL iscoloringdestroy_(ISColoring *iscoloring,int *ierr)
{
  *ierr = ISColoringDestroy(*iscoloring);
}

void PETSC_STDCALL iscoloringview_(ISColoring *iscoloring,PetscViewer *viewer,int *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = ISColoringView(*iscoloring,v);
}

void PETSC_STDCALL isview_(IS *is,PetscViewer *vin,int *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = ISView(*is,v);
}

void PETSC_STDCALL isequal_(IS *is1,IS *is2,PetscTruth *flg,int *ierr)
{
  *ierr = ISEqual(*is1,*is2,flg);
}

void PETSC_STDCALL isidentity_(IS *is,PetscTruth *ident,int *ierr)
{
  *ierr = ISIdentity(*is,ident);
}

void PETSC_STDCALL issorted_(IS *is,PetscTruth *flg,int *ierr)
{
  *ierr = ISSorted(*is,flg);
}

void PETSC_STDCALL ispermutation_(IS *is,PetscTruth *perm,int *ierr){
  *ierr = ISPermutation(*is,perm);
}

void PETSC_STDCALL isstride_(IS *is,PetscTruth *flag,int *ierr)
{
  *ierr = ISStride(*is,flag);
}

void PETSC_STDCALL isblockgetindices_(IS *x,int *fa,long *ia,int *ierr)
{
  int   *lx;

  *ierr = ISGetIndices(*x,&lx); if (*ierr) return;
  *ia      = PetscIntAddressToFortran(fa,lx);
}

void PETSC_STDCALL isblockrestoreindices_(IS *x,int *fa,long *ia,int *ierr)
{
  int *lx = PetscIntAddressFromFortran(fa,*ia);

  *ierr = ISRestoreIndices(*x,&lx);
}

void PETSC_STDCALL isblock_(IS *is,PetscTruth *flag,int *ierr)
{
  *ierr = ISBlock(*is,flag);
}

void PETSC_STDCALL isgetindices_(IS *x,int *fa,long *ia,int *ierr)
{
  int   *lx;

  *ierr = ISGetIndices(*x,&lx); if (*ierr) return;
  *ia      = PetscIntAddressToFortran(fa,lx);
}

void PETSC_STDCALL isrestoreindices_(IS *x,int *fa,long *ia,int *ierr)
{
  int *lx = PetscIntAddressFromFortran(fa,*ia);

  *ierr = ISRestoreIndices(*x,&lx);
}

void PETSC_STDCALL iscreategeneral_(MPI_Comm *comm,int *n,int *idx,IS *is,int *ierr)
{
  *ierr = ISCreateGeneral((MPI_Comm)PetscToPointerComm(*comm),*n,idx,is);
}

void PETSC_STDCALL isinvertpermutation_(IS *is,int *nlocal,IS *isout,int *ierr)
{
  *ierr = ISInvertPermutation(*is,*nlocal,isout);
}

void PETSC_STDCALL iscreateblock_(MPI_Comm *comm,int *bs,int *n,int *idx,IS *is,int *ierr)
{
  *ierr = ISCreateBlock((MPI_Comm)PetscToPointerComm(*comm),*bs,*n,idx,is);
}

void PETSC_STDCALL iscreatestride_(MPI_Comm *comm,int *n,int *first,int *step,
                               IS *is,int *ierr)
{
  *ierr = ISCreateStride((MPI_Comm)PetscToPointerComm(*comm),*n,*first,*step,is);
}

void PETSC_STDCALL isdestroy_(IS *is,int *ierr)
{
  *ierr = ISDestroy(*is);
}

void PETSC_STDCALL iscoloringcreate_(MPI_Comm *comm,int *n,int *colors,ISColoring *iscoloring,int *ierr)
{
  int *color;
  /* copies the colors[] array since that is kept by the ISColoring that is created */
  *ierr = PetscMalloc((*n+1)*sizeof(int),&color);if (*ierr) return;
  *ierr = PetscMemcpy(color,colors,(*n)*sizeof(int));if (*ierr) return;
  *ierr = ISColoringCreate((MPI_Comm)PetscToPointerComm(*comm),*n,color,iscoloring);
}

void PETSC_STDCALL islocaltoglobalmappingcreate_(MPI_Comm *comm,int *n,int *indices,ISLocalToGlobalMapping *mapping,int *ierr)
{
  *ierr = ISLocalToGlobalMappingCreate((MPI_Comm)PetscToPointerComm(*comm),*n,indices,mapping);
}

void PETSC_STDCALL islocaltoglobalmappingblock_(ISLocalToGlobalMapping *inmap,int bs,ISLocalToGlobalMapping *outmap,int *ierr)
{
  *ierr = ISLocalToGlobalMappingBlock(*inmap,bs,outmap);
}

void PETSC_STDCALL isallgather_(IS *is,IS *isout,int *ierr)
{
  *ierr = ISAllGather(*is,isout);

}

EXTERN_C_END

