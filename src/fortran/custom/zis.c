/*$Id: zis.c,v 1.29 1999/10/04 22:51:03 balay Exp bsmith $*/

#include "src/fortran/custom/zpetsc.h"
#include "is.h"
#ifdef PETSC_HAVE_FORTRAN_CAPS
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
#define isallgather_                  ISALLGATHER
#define iscoloringdestroy_            ISCOLORINGDESTROY
#define iscoloringview_               ISCOLORINGVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
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
#define isallgather_                  isallgather
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL iscoloringdestroy_(ISColoring *iscoloring, int *__ierr )
{
  *__ierr = ISColoringDestroy(*iscoloring);
}

void PETSC_STDCALL iscoloringview_(ISColoring *iscoloring,Viewer *viewer, int *__ierr )
{
  *__ierr = ISColoringView(*iscoloring,*viewer);
}

void PETSC_STDCALL isview_(IS *is,Viewer *vin, int *__ierr )
{
  Viewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *__ierr = ISView(*is,v);
}

void PETSC_STDCALL isequal_(IS *is1,IS *is2,PetscTruth *flg, int *__ierr )
{
  *__ierr = ISEqual(*is1,*is2, flg);
}

void PETSC_STDCALL isidentity_(IS *is,PetscTruth *ident, int *__ierr )
{
  *__ierr = ISIdentity(*is,ident);
}

void PETSC_STDCALL issorted_(IS *is,PetscTruth *flg, int *__ierr )
{
  *__ierr = ISSorted(*is,flg);
}

void PETSC_STDCALL ispermutation_(IS *is,PetscTruth *perm, int *__ierr ){
  *__ierr = ISPermutation(*is,perm);
}

void PETSC_STDCALL isstride_(IS *is,PetscTruth *flag, int *__ierr )
{
  *__ierr = ISStride(*is,flag);
}

void PETSC_STDCALL isblockgetindices_(IS *x,int *fa,long *ia,int *__ierr)
{
  int   *lx;

  *__ierr = ISGetIndices(*x,&lx); if (*__ierr) return;
  *ia      = PetscIntAddressToFortran(fa,lx);
}

void PETSC_STDCALL isblockrestoreindices_(IS *x,int *fa,long *ia,int *__ierr)
{
  int *lx = PetscIntAddressFromFortran(fa,*ia);

  *__ierr = ISRestoreIndices(*x,&lx);
}

void PETSC_STDCALL isblock_(IS *is,PetscTruth *flag, int *__ierr )
{
  *__ierr = ISBlock(*is,flag);
}

void PETSC_STDCALL isgetindices_(IS *x,int *fa,long *ia,int *__ierr)
{
  int   *lx;

  *__ierr = ISGetIndices(*x,&lx); if (*__ierr) return;
  *ia      = PetscIntAddressToFortran(fa,lx);
}

void PETSC_STDCALL isrestoreindices_(IS *x,int *fa,long *ia,int *__ierr)
{
  int *lx = PetscIntAddressFromFortran(fa,*ia);

  *__ierr = ISRestoreIndices(*x,&lx);
}

void PETSC_STDCALL iscreategeneral_(MPI_Comm *comm,int *n,int *idx,IS *is, int *__ierr )
{
  *__ierr = ISCreateGeneral((MPI_Comm)PetscToPointerComm( *comm ),*n,idx,is);
}

void PETSC_STDCALL isinvertpermutation_(IS *is,IS *isout, int *__ierr )
{
  *__ierr = ISInvertPermutation(*is,isout);
}

void PETSC_STDCALL iscreateblock_(MPI_Comm *comm,int *bs,int *n,int *idx,IS *is, int *__ierr )
{
  *__ierr = ISCreateBlock((MPI_Comm)PetscToPointerComm(*comm),*bs,*n,idx,is);
}

void PETSC_STDCALL iscreatestride_(MPI_Comm *comm,int *n,int *first,int *step,
                               IS *is, int *__ierr )
{
  *__ierr = ISCreateStride((MPI_Comm)PetscToPointerComm( *comm ),*n,*first,*step,is);
}

void PETSC_STDCALL isdestroy_(IS *is, int *__ierr )
{
  *__ierr = ISDestroy(*is);
}

void PETSC_STDCALL iscoloringcreate_(MPI_Comm *comm,int *n,int *colors,ISColoring *iscoloring, int *__ierr )
{
  *__ierr = ISColoringCreate((MPI_Comm)PetscToPointerComm( *comm ),*n,colors,iscoloring);
}

void PETSC_STDCALL islocaltoglobalmappingcreate_(MPI_Comm *comm,int *n,int *indices,ISLocalToGlobalMapping 
                                   *mapping, int *__ierr )
{
  *__ierr = ISLocalToGlobalMappingCreate((MPI_Comm)PetscToPointerComm(*comm),*n,indices,mapping);
}

void PETSC_STDCALL isallgather_(IS *is,IS *isout, int *__ierr )
{
  *__ierr = ISAllGather(*is,isout);

}

EXTERN_C_END

