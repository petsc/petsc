#ifndef lint
static char vcid[] = "$Id: zda.c,v 1.1 1995/08/27 00:35:57 bsmith Exp bsmith $";
#endif

#include "zpetsc.h"
#include "da.h"
#ifdef FORTRANCAPS
#define dacreate1d_             DACREATE1D
#define dacreate3d_             DACREATE3D
#define dacreate2d_             DACREATE2D
#define dadestroy_              DADESTROY
#define dagetdistributedvector_ DAGETDISTRIBUTEDVECTOR
#define dagetlocalvector_       DAGETLOCALVECTOR
#define dagetscatterctx_        DAGETSCATTERCTX
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dacreate1d_             dacreate1d
#define dacreate3d_             dacreate3d
#define dacreate2d_             dacreate2d
#define dadestroy_              dadestroy
#define dagetdistributedvector_ dagetdistributedvector
#define dagetlocalvector_       dagetlocalvector
#define dagetscatterctx_        dagetscatterctx
#endif


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
void dagetscatterctx_(DA da,VecScatterCtx *ltog,VecScatterCtx *gtol,
                      int *__ierr )
{
  VecScatterCtx l,g;
  *__ierr = DAGetScatterCtx((DA)MPIR_ToPointer(*(int*)(da)),&l,&g);
  *(int*) ltog = MPIR_FromPointer(l);
  *(int*) gtol = MPIR_FromPointer(g);
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
	    (MPI_Comm)MPIR_ToPointer( *(int*)(comm) ),*wrap,
            *stencil_type,*M,*N,*m,*n,*w,*s,&da);
  *(int*) inra = MPIR_FromPointer(da);
}

void dacreate1d_(MPI_Comm comm,DAPeriodicType *wrap,int *M,int *w,int *s,
                 DA *inra, int *__ierr )
{
  DA da;
  *__ierr = DACreate1d(
	   (MPI_Comm)MPIR_ToPointer( *(int*)(comm) ),*wrap,*M,*w,*s,&da);
  *(int*) inra = MPIR_FromPointer(da);
}

void dacreate3d_(MPI_Comm comm,DAPeriodicType *wrap,DAStencilType 
                 *stencil_type,int *M,int *N,int *P,int *m,int *n,int *p,
                 int *w,int *s,DA *inra, int *__ierr )
{
  DA da;
  *__ierr = DACreate3d(
	   (MPI_Comm)MPIR_ToPointer( *(int*)(comm) ),*wrap,*stencil_type,
           *M,*N,*P,*m,*n,*p,*w,*s,&da);
  *(int*) inra = MPIR_FromPointer(da);
}
