#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zda.c,v 1.27 1999/03/18 00:36:01 curfman Exp bsmith $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "mat.h"
#include "da.h"

#ifdef HAVE_FORTRAN_CAPS
#define dacreate1d_             DACREATE1D
#define dacreate3d_             DACREATE3D
#define dacreate2d_             DACREATE2D
#define dadestroy_              DADESTROY
#define dacreateglobalvector_   DACREATEGLOBALVECTOR
#define dacreatelocalvector_    DACREATELOCALVECTOR
#define dagetscatter_           DAGETSCATTER
#define dagetglobalindices_     DAGETGLOBALINDICES
#define daview_                 DAVIEW
#define dagetinfo_              DAGETINFO
#define dagetcoloring_          DAGETCOLORING
#define dagetislocaltoglobalmapping_ DAGETISLOCALTOGLOBALMAPPING
#define daload_                      DALOAD
#define dasetfieldname_              DASETFIELDNAME
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define daload_                 daload
#define dacreateglobalvector_   dacreateglobalvector
#define dacreatelocalvector_    dacreatelocalvector
#define daview_                 daview
#define dacreate1d_             dacreate1d
#define dacreate3d_             dacreate3d
#define dacreate2d_             dacreate2d
#define dadestroy_              dadestroy
#define dagetscatter_           dagetscatter
#define dagetglobalindices_     dagetglobalindices
#define dagetinfo_              dagetinfo
#define dagetcoloring_          dagetcoloring
#define dagetislocaltoglobalmapping_ dagetislocaltoglobalmapping
#define dasetfieldname_              dasetfieldname
#endif

EXTERN_C_BEGIN

void dasetfieldname_(DA *da,int *nf, CHAR name, int *__ierr,int len )
{
  char *t;
  FIXCHAR(name,len,t);
  *__ierr = DASetFieldName(*da,*nf,t);
  FREECHAR(name,t);
}

void daload_(Viewer *viewer,int *M,int *N,int *P,DA *da, int *__ierr )
{
  *__ierr = DALoad(*viewer,*M,*N,*P,da);
}

void dagetislocaltoglobalmapping_(DA *da,ISLocalToGlobalMapping *map, int *__ierr)
{
  *__ierr = DAGetISLocalToGlobalMapping(*da,map);
}

void dagetcoloring_(DA *da, ISColoring *coloring, Mat *J,int *__ierr)
{
  *__ierr = DAGetColoring(*da,coloring,J);
}

void daview_(DA *da,Viewer *vin, int *__ierr )
{
  Viewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *__ierr = DAView(*da,v);
}

void dagetglobalindices_(DA *da,int *n, int *indices, long *ia,int *__ierr )
{
  int *idx;
  *__ierr = DAGetGlobalIndices(*da,n,&idx);
  *ia     = PetscIntAddressToFortran(indices,idx);
}

void dacreateglobalvector_(DA *da,Vec* g, int *__ierr )
{
  *__ierr = DACreateGlobalVector(*da,g);
}

void dacreatelocalvector_(DA *da,Vec* l, int *__ierr )
{
  *__ierr = DACreateLocalVector(*da,l);
}

void dagetscatter_(DA *da,VecScatter *ltog,VecScatter *gtol,VecScatter *ltol,
                   int *__ierr )
{
  if (!FORTRANNULLINTEGER(ltog)) ltog = PETSC_NULL;
  if (!FORTRANNULLINTEGER(gtol)) gtol = PETSC_NULL;
  if (!FORTRANNULLINTEGER(ltol)) ltol = PETSC_NULL;
  *__ierr = DAGetScatter(*da,ltog,gtol,ltol);
}

void dadestroy_(DA *da, int *__ierr )
{
  *__ierr = DADestroy(*da);
}

void dacreate2d_(MPI_Comm *comm,DAPeriodicType *wrap,DAStencilType
                  *stencil_type,int *M,int *N,int *m,int *n,int *w,
                  int *s,int *lx,int *ly,DA *inra, int *__ierr )
{
  if (FORTRANNULLINTEGER(lx)) lx = PETSC_NULL;
  if (FORTRANNULLINTEGER(ly)) ly = PETSC_NULL;
  *__ierr = DACreate2d((MPI_Comm)PetscToPointerComm( *comm ),*wrap,
                       *stencil_type,*M,*N,*m,*n,*w,*s,lx,ly,inra);
}

void dacreate1d_(MPI_Comm *comm,DAPeriodicType *wrap,int *M,int *w,int *s,
                 int *lc,DA *inra, int *__ierr )
{
  if (FORTRANNULLINTEGER(lc)) lc = PETSC_NULL;
  *__ierr = DACreate1d((MPI_Comm)PetscToPointerComm(*comm),*wrap,*M,*w,*s,lc,inra);
}

void dacreate3d_(MPI_Comm *comm,DAPeriodicType *wrap,DAStencilType 
                 *stencil_type,int *M,int *N,int *P,int *m,int *n,int *p,
                 int *w,int *s,int *lx,int *ly,int *lz,DA *inra, int *__ierr )
{
  if (FORTRANNULLINTEGER(lx)) lx = PETSC_NULL;
  if (FORTRANNULLINTEGER(ly)) ly = PETSC_NULL;
  if (FORTRANNULLINTEGER(lz)) lz = PETSC_NULL;
  *__ierr = DACreate3d((MPI_Comm)PetscToPointerComm(*comm),*wrap,*stencil_type,
                        *M,*N,*P,*m,*n,*p,*w,*s,lx,ly,lz,inra);
}

void dagetinfo_(DA *da,int *dim,int *M,int *N,int *P,int *m,int *n,int *p,int *w,int *s,
                DAPeriodicType *wrap, DAStencilType *st,int *__ierr )
{
  *__ierr = DAGetInfo(*da,dim,M,N,P,m,n,p,w,s,wrap,st);
}

EXTERN_C_END

