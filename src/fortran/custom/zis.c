#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zis.c,v 1.27 1998/10/19 22:15:08 bsmith Exp bsmith $";
#endif

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

void iscoloringdestroy_(ISColoring *iscoloring, int *__ierr )
{
  *__ierr = ISColoringDestroy(*iscoloring);
}

void iscoloringview_(ISColoring *iscoloring,Viewer *viewer, int *__ierr )
{
  *__ierr = ISColoringView(*iscoloring,*viewer);
}

void isview_(IS *is,Viewer *vin, int *__ierr )
{
  Viewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *__ierr = ISView(*is,v);
}

void isequal_(IS *is1,IS *is2,PetscTruth *flg, int *__ierr )
{
  *__ierr = ISEqual(*is1,*is2, flg);
}

void isidentity_(IS *is,PetscTruth *ident, int *__ierr )
{
  *__ierr = ISIdentity(*is,ident);
}

void issorted_(IS *is,PetscTruth *flg, int *__ierr )
{
  *__ierr = ISSorted(*is,flg);
}

void ispermutation_(IS *is,PetscTruth *perm, int *__ierr ){
  *__ierr = ISPermutation(*is,perm);
}

void isstride_(IS *is,PetscTruth *flag, int *__ierr )
{
  *__ierr = ISStride(*is,flag);
}

void isblockgetindices_(IS *x,int *fa,long *ia,int *__ierr)
{
  int   *lx;

  *__ierr = ISGetIndices(*x,&lx); if (*__ierr) return;
  *ia      = PetscIntAddressToFortran(fa,lx);
}

void isblockrestoreindices_(IS *x,int *fa,long *ia,int *__ierr)
{
  int *lx = PetscIntAddressFromFortran(fa,*ia);

  *__ierr = ISRestoreIndices(*x,&lx);
}

void isblock_(IS *is,PetscTruth *flag, int *__ierr )
{
  *__ierr = ISBlock(*is,flag);
}

void isgetindices_(IS *x,int *fa,long *ia,int *__ierr)
{
  int   *lx;

  *__ierr = ISGetIndices(*x,&lx); if (*__ierr) return;
  *ia      = PetscIntAddressToFortran(fa,lx);
}

void isrestoreindices_(IS *x,int *fa,long *ia,int *__ierr)
{
  int *lx = PetscIntAddressFromFortran(fa,*ia);

  *__ierr = ISRestoreIndices(*x,&lx);
}

void iscreategeneral_(MPI_Comm *comm,int *n,int *idx,IS *is, int *__ierr )
{
  *__ierr = ISCreateGeneral((MPI_Comm)PetscToPointerComm( *comm ),*n,idx,is);
}

void isinvertpermutation_(IS *is,IS *isout, int *__ierr )
{
  *__ierr = ISInvertPermutation(*is,isout);
}

void iscreateblock_(MPI_Comm *comm,int *bs,int *n,int *idx,IS *is, int *__ierr )
{
  *__ierr = ISCreateBlock((MPI_Comm)PetscToPointerComm(*comm),*bs,*n,idx,is);
}

void iscreatestride_(MPI_Comm *comm,int *n,int *first,int *step,
                               IS *is, int *__ierr )
{
  *__ierr = ISCreateStride((MPI_Comm)PetscToPointerComm( *comm ),*n,*first,*step,is);
}

void isdestroy_(IS *is, int *__ierr )
{
  *__ierr = ISDestroy(*is);
}

void iscoloringcreate_(MPI_Comm *comm,int *n,int *colors,ISColoring *iscoloring, int *__ierr )
{
  *__ierr = ISColoringCreate((MPI_Comm)PetscToPointerComm( *comm ),*n,colors,iscoloring);
}

void islocaltoglobalmappingcreate_(MPI_Comm *comm,int *n,int *indices,ISLocalToGlobalMapping 
                                   *mapping, int *__ierr )
{
  *__ierr = ISLocalToGlobalMappingCreate((MPI_Comm)PetscToPointerComm(*comm),*n,indices,mapping);
}

void isallgather_(IS *is,IS *isout, int *__ierr )
{
  *__ierr = ISAllGather(*is,isout);

}

EXTERN_C_END

