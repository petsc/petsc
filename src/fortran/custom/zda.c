/*$Id: zda.c,v 1.34 1999/11/05 14:48:14 bsmith Exp bsmith $*/

#include "src/fortran/custom/zpetsc.h"
#include "mat.h"
#include "da.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define dagetinterpolation_          DAGETINTERPOLATION
#define dacreate1d_                  DACREATE1D
#define dacreate3d_                  DACREATE3D
#define dacreate2d_                  DACREATE2D
#define dadestroy_                   DADESTROY
#define dacreateglobalvector_        DACREATEGLOBALVECTOR
#define dacreatelocalvector_         DACREATELOCALVECTOR
#define dagetscatter_                DAGETSCATTER
#define dagetglobalindices_          DAGETGLOBALINDICES
#define daview_                      DAVIEW
#define dagetinfo_                   DAGETINFO
#define dagetcoloring_               DAGETCOLORING
#define dagetislocaltoglobalmapping_ DAGETISLOCALTOGLOBALMAPPING
#define daload_                      DALOAD
#define dasetfieldname_              DASETFIELDNAME
#define dagetfieldname_              DAGETFIELDNAME
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dagetinterpolation_          dagetinterpolation
#define daload_                      daload
#define dacreateglobalvector_        dacreateglobalvector
#define dacreatelocalvector_         dacreatelocalvector
#define daview_                      daview
#define dacreate1d_                  dacreate1d
#define dacreate3d_                  dacreate3d
#define dacreate2d_                  dacreate2d
#define dadestroy_                   dadestroy
#define dagetscatter_                dagetscatter
#define dagetglobalindices_          dagetglobalindices
#define dagetinfo_                   dagetinfo
#define dagetcoloring_               dagetcoloring
#define dagetislocaltoglobalmapping_ dagetislocaltoglobalmapping
#define dasetfieldname_              dasetfieldname
#define dagetfieldname_              dagetfieldname
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL dagetinterpolation_(DA *dac,DA *daf,Mat *A,Vec *scale,int *__ierr)
{
  if (FORTRANNULLOBJECT(scale)) scale = PETSC_NULL;
  *__ierr = DAGetInterpolation(*dac,*daf,A,scale);
}

void PETSC_STDCALL dasetfieldname_(DA *da,int *nf, CHAR name PETSC_MIXED_LEN(len),
                                   int *__ierr PETSC_END_LEN(len) )
{
  char *t;
  FIXCHAR(name,len,t);
  *__ierr = DASetFieldName(*da,*nf,t);
  FREECHAR(name,t);
}
void PETSC_STDCALL dagetfieldname(DA *da,int *nf,CHAR name PETSC_MIXED_LEN(len),
                                  int *__ierr PETSC_END_LEN(len))
{
  char *tname;

  *__ierr = DAGetFieldName(*da,*nf,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(name); int len1 = _fcdlen(name);
    *__ierr = PetscStrncpy(t,tname,len1);
  }
#else
  *__ierr = PetscStrncpy(name,tname,len);
#endif
}

void PETSC_STDCALL daload_(Viewer *viewer,int *M,int *N,int *P,DA *da, int *__ierr )
{
  *__ierr = DALoad(*viewer,*M,*N,*P,da);
}

void PETSC_STDCALL dagetislocaltoglobalmapping_(DA *da,ISLocalToGlobalMapping *map, int *__ierr)
{
  *__ierr = DAGetISLocalToGlobalMapping(*da,map);
}

void PETSC_STDCALL dagetcoloring_(DA *da, ISColoring *coloring, Mat *J,int *__ierr)
{
  *__ierr = DAGetColoring(*da,coloring,J);
}

void PETSC_STDCALL daview_(DA *da,Viewer *vin, int *__ierr )
{
  Viewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *__ierr = DAView(*da,v);
}

void PETSC_STDCALL dagetglobalindices_(DA *da,int *n, int *indices, long *ia,int *__ierr )
{
  int *idx;
  *__ierr = DAGetGlobalIndices(*da,n,&idx);
  *ia     = PetscIntAddressToFortran(indices,idx);
}

void PETSC_STDCALL dacreateglobalvector_(DA *da,Vec* g, int *__ierr )
{
  *__ierr = DACreateGlobalVector(*da,g);
}

void PETSC_STDCALL dacreatelocalvector_(DA *da,Vec* l, int *__ierr )
{
  *__ierr = DACreateLocalVector(*da,l);
}

void PETSC_STDCALL dagetscatter_(DA *da,VecScatter *ltog,VecScatter *gtol,VecScatter *ltol,
                   int *__ierr )
{
  if (!FORTRANNULLINTEGER(ltog)) ltog = PETSC_NULL;
  if (!FORTRANNULLINTEGER(gtol)) gtol = PETSC_NULL;
  if (!FORTRANNULLINTEGER(ltol)) ltol = PETSC_NULL;
  *__ierr = DAGetScatter(*da,ltog,gtol,ltol);
}

void PETSC_STDCALL dadestroy_(DA *da, int *__ierr )
{
  *__ierr = DADestroy(*da);
}

void PETSC_STDCALL dacreate2d_(MPI_Comm *comm,DAPeriodicType *wrap,DAStencilType
                  *stencil_type,int *M,int *N,int *m,int *n,int *w,
                  int *s,int *lx,int *ly,DA *inra, int *__ierr )
{
  if (FORTRANNULLINTEGER(lx)) lx = PETSC_NULL;
  if (FORTRANNULLINTEGER(ly)) ly = PETSC_NULL;
  *__ierr = DACreate2d((MPI_Comm)PetscToPointerComm( *comm ),*wrap,
                       *stencil_type,*M,*N,*m,*n,*w,*s,lx,ly,inra);
}

void PETSC_STDCALL dacreate1d_(MPI_Comm *comm,DAPeriodicType *wrap,int *M,int *w,int *s,
                 int *lc,DA *inra, int *__ierr )
{
  if (FORTRANNULLINTEGER(lc)) lc = PETSC_NULL;
  *__ierr = DACreate1d((MPI_Comm)PetscToPointerComm(*comm),*wrap,*M,*w,*s,lc,inra);
}

void PETSC_STDCALL dacreate3d_(MPI_Comm *comm,DAPeriodicType *wrap,DAStencilType 
                 *stencil_type,int *M,int *N,int *P,int *m,int *n,int *p,
                 int *w,int *s,int *lx,int *ly,int *lz,DA *inra, int *__ierr )
{
  if (FORTRANNULLINTEGER(lx)) lx = PETSC_NULL;
  if (FORTRANNULLINTEGER(ly)) ly = PETSC_NULL;
  if (FORTRANNULLINTEGER(lz)) lz = PETSC_NULL;
  *__ierr = DACreate3d((MPI_Comm)PetscToPointerComm(*comm),*wrap,*stencil_type,
                        *M,*N,*P,*m,*n,*p,*w,*s,lx,ly,lz,inra);
}

void PETSC_STDCALL dagetinfo_(DA *da,int *dim,int *M,int *N,int *P,int *m,int *n,int *p,int *w,int *s,
                DAPeriodicType *wrap, DAStencilType *st,int *__ierr )
{
  *__ierr = DAGetInfo(*da,dim,M,N,P,m,n,p,w,s,wrap,st);
}

EXTERN_C_END

