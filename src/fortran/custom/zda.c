#ifndef lint
static char vcid[] = "$Id: zda.c,v 1.6 1996/03/05 00:06:55 curfman Exp curfman $";
#endif

#include "zpetsc.h"
#include "da.h"
#ifdef HAVE_FORTRAN_CAPS
#define dacreate1d_             DACREATE1D
#define dacreate3d_             DACREATE3D
#define dacreate2d_             DACREATE2D
#define dadestroy_              DADESTROY
#define dagetdistributedvector_ DAGETDISTRIBUTEDVECTOR
#define dagetlocalvector_       DAGETLOCALVECTOR
#define dagetscatter_           DAGETSCATTER
#define dagetglobalindices_     DAGETGLOBALINDICES
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define dacreate1d_             dacreate1d
#define dacreate3d_             dacreate3d
#define dacreate2d_             dacreate2d
#define dadestroy_              dadestroy
#define dagetdistributedvector_ dagetdistributedvector
#define dagetlocalvector_       dagetlocalvector
#define dagetscatter_           dagetscatter
#define dagetglobalindices_     dagetglobalindices
#endif

#if defined(__cplusplus)
extern "C" {
#endif

void dagetglobalindices_(DA da,int *n, int *indices, int *ia,int *__ierr )
{
  int *idx;
  *__ierr = DAGetGlobalIndices((DA)MPIR_ToPointer(*(int*)(da)),n,&idx);
  *ia     = PetscIntAddressToFortran(indices,idx);
}

void dagetdistributedvector_(DA da,Vec* g, int *__ierr )
{
  Vec v;
  *__ierr = DAGetDistributedVector((DA)MPIR_ToPointer(*(int*)(da)),&v);
  *(int*) g = MPIR_FromPointer(v);
}
void dagetlocalvector_(DA da,Vec* l, int *__ierr )
{
  Vec v;
  *__ierr = DAGetLocalVector((DA)MPIR_ToPointer(*(int*)(da)),&v);
  *(int*) l = MPIR_FromPointer(v);
}
void dagetscatter_(DA da,VecScatter *ltog,VecScatter *gtol,VecScatter *ltol,
                   int *__ierr )
{
  VecScatter l,g,ll;
  *__ierr = DAGetScatter((DA)MPIR_ToPointer(*(int*)(da)),&l,&g,&ll);
  if (!FORTRANNULL(ltog)) *(int*) ltog = MPIR_FromPointer(l);
  if (!FORTRANNULL(gtol)) *(int*) gtol = MPIR_FromPointer(g);
  if (!FORTRANNULL(ltol)) *(int*) ltol = MPIR_FromPointer(ll);
}

void dadestroy_(DA da, int *__ierr ){
  *__ierr = DADestroy(
	(DA)MPIR_ToPointer( *(int*)(da) ));
  MPIR_RmPointer(*(int*)(da));
}

void dacreate2d_(MPI_Comm comm,DAPeriodicType *wrap,DAStencilType
                  *stencil_type,int *M,int *N,int *m,int *n,int *w,
                  int *s,DA *inra, int *__ierr )
{
  DA da;
  *__ierr = DACreate2d(
	    (MPI_Comm)MPIR_ToPointer_Comm( *(int*)(comm) ),*wrap,
            *stencil_type,*M,*N,*m,*n,*w,*s,&da);
  *(int*) inra = MPIR_FromPointer(da);
}

void dacreate1d_(MPI_Comm comm,DAPeriodicType *wrap,int *M,int *w,int *s,
                 DA *inra, int *__ierr )
{
  DA da;
  *__ierr = DACreate1d(
	   (MPI_Comm)MPIR_ToPointer_Comm( *(int*)(comm) ),*wrap,*M,*w,*s,&da);
  *(int*) inra = MPIR_FromPointer(da);
}

void dacreate3d_(MPI_Comm comm,DAPeriodicType *wrap,DAStencilType 
                 *stencil_type,int *M,int *N,int *P,int *m,int *n,int *p,
                 int *w,int *s,DA *inra, int *__ierr )
{
  DA da;
  *__ierr = DACreate3d(
	   (MPI_Comm)MPIR_ToPointer_Comm(*(int*)(comm)),*wrap,*stencil_type,
           *M,*N,*P,*m,*n,*p,*w,*s,&da);
  *(int*) inra = MPIR_FromPointer(da);
}

#if defined(__cplusplus)
}
#endif
