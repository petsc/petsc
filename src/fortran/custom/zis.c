#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zis.c,v 1.18 1997/11/24 15:44:03 bsmith Exp bsmith $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "is.h"
#ifdef HAVE_FORTRAN_CAPS
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

#elif !defined(HAVE_FORTRAN_UNDERSCORE)
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

#if defined(__cplusplus)
extern "C" {
#endif

void isview_(IS is,Viewer viewer, int *__ierr )
{
  PetscPatchDefaultViewers_Fortran(viewer);
  *__ierr = ISView((IS)PetscToPointer( *(int*)(is) ),viewer);
}

void isequal_(IS is1,IS is2,PetscTruth *flg, int *__ierr )
{
  *__ierr = ISEqual((IS)PetscToPointer( *(int*)(is1) ),
	          (IS)PetscToPointer( *(int*)(is2) ), flg);
}

void isidentity_(IS is,PetscTruth *ident, int *__ierr )
{
  *__ierr = ISIdentity((IS)PetscToPointer( *(int*)(is) ),ident);
}
void issorted_(IS is,PetscTruth *flg, int *__ierr )
{
  *__ierr = ISSorted((IS)PetscToPointer( *(int*)(is) ),flg);
}

void ispermutation_(IS is,PetscTruth *perm, int *__ierr ){
  *__ierr = ISPermutation((IS)PetscToPointer( *(int*)(is) ),perm);
}

void isstride_(IS is,PetscTruth *flag, int *__ierr )
{
  *__ierr = ISStride((IS)PetscToPointer( *(int*)(is) ),flag);
}

void isblockgetindices_(IS x,int *fa,int *ia,int *__ierr)
{
  IS    xin = (IS)PetscToPointer( *(int*)(x) );
  int   *lx;

#if defined(PARCH_IRIX64)
  PetscErrorPrintf("PETSC ERROR: Cannot use ISBlockGetIndices() from Fortran under IRIX\n");
  PetscErrorPrintf("PETSC ERROR: Refer to troubleshooting.html for more details\n");
  MPI_Abort(PETSC_COMM_WORLD,1);
#else
  *__ierr = ISGetIndices(xin,&lx); if (*__ierr) return;
  *ia      = PetscIntAddressToFortran(fa,lx);
#endif
}

void isblockrestoreindices_(IS x,int *fa,int *ia,int *__ierr)
{
  IS    xin = (IS)PetscToPointer( *(int*)(x) );
  int *lx = PetscIntAddressFromFortran(fa,*ia);

  *__ierr = ISRestoreIndices(xin,&lx);
}

void isblock_(IS is,PetscTruth *flag, int *__ierr )
{
  *__ierr = ISBlock((IS)PetscToPointer( *(int*)(is) ),flag);
}

void isgetindices_(IS x,int *fa,int *ia,int *__ierr)
{
  IS    xin = (IS)PetscToPointer( *(int*)(x) );
  int   *lx;

#if defined(PARCH_IRIX64)
  PetscErrorPrintf("PETSC ERROR: Cannot use ISGetIndices() from Fortran under IRIX\n");
  PetscErrorPrintf("PETSC ERROR: Refer to troubleshooting.html for more details\n");
  MPI_Abort(PETSC_COMM_WORLD,1);
#else
  *__ierr = ISGetIndices(xin,&lx); if (*__ierr) return;
  *ia      = PetscIntAddressToFortran(fa,lx);
#endif
}

void isrestoreindices_(IS x,int *fa,int *ia,int *__ierr)
{
  IS    xin = (IS)PetscToPointer( *(int*)(x) );
  int *lx = PetscIntAddressFromFortran(fa,*ia);

  *__ierr = ISRestoreIndices(xin,&lx);
}

void iscreategeneral_(MPI_Comm *comm,int *n,int *idx,IS *is, int *__ierr ){
  IS ii;
  *__ierr = ISCreateGeneral(
	(MPI_Comm)PetscToPointerComm( *comm ),*n,idx,&ii);
  *(int*) is = PetscFromPointer(ii);
}

void isinvertpermutation_(IS is,IS *isout, int *__ierr )
{
  IS ii;
  *__ierr = ISInvertPermutation((IS)PetscToPointer( *(int*)(is) ),&ii);
  *(int*) isout = PetscFromPointer(ii);
}

void iscreateblock_(MPI_Comm *comm,int *bs,int *n,int *idx,IS *is, int *__ierr ){
  IS ii;
  *__ierr = ISCreateBlock(
	(MPI_Comm)PetscToPointerComm( *comm ),*bs,*n,idx,&ii);
  *(int*) is = PetscFromPointer(ii);
}

void iscreatestride_(MPI_Comm *comm,int *n,int *first,int *step,
                               IS *is, int *__ierr ){
  IS ii;
  *__ierr = ISCreateStride(
	(MPI_Comm)PetscToPointerComm( *comm ),*n,*first,*step,&ii);
  *(int*) is = PetscFromPointer(ii);
}

void isdestroy_(IS is, int *__ierr ){
  *__ierr = ISDestroy((IS)PetscToPointer( *(int*)(is) ));
  PetscRmPointer(*(int*)(is) );
}

void iscoloringcreate_(MPI_Comm *comm,int *n,int *colors,ISColoring *iscoloring, int *__ierr )
{
  ISColoring ii;

  *__ierr = ISColoringCreate((MPI_Comm)PetscToPointerComm( *comm ),*n,colors,&ii);
  *(int *) iscoloring = PetscFromPointer(ii);
}

void islocaltoglobalmappingcreate_(MPI_Comm *comm,int *n,int *indices,ISLocalToGlobalMapping 
                                   *mapping, int *__ierr )
{
  ISLocalToGlobalMapping ii;
  *__ierr = ISLocalToGlobalMappingCreate((MPI_Comm)PetscToPointerComm(*comm),*n,indices,&ii);
  *(int *) mapping = PetscFromPointer(ii);
}

void isallgather_(IS is,IS *isout, int *__ierr ){
IS islocal;
*__ierr = ISAllGather(
	(IS)PetscToPointer( *(int*)(is)) ,&islocal);
        *(int*) isout = PetscFromPointer(islocal);

}

#if defined(__cplusplus)
}
#endif
