/*$Id: zda.c,v 1.49 2001/08/06 21:19:11 bsmith Exp $*/

#include "src/dm/da/daimpl.h"
#include "src/fortran/custom/zpetsc.h"
#include "petscmat.h"
#include "petscda.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define dasetlocalfunction_          DASETLOCALFUNCTION
#define dasetLocaladiforfunction_    DASETLOCALADIFORFUNCTION
#define dasetlocaladiformffunction_  DASETLOCALADIFORMFFUNCTION
#define dasetlocaljacobian_          DASETLOCALJACOBIAN
#define dagetlocalinfo_              DAGETLOCALINFO
#define dagetinterpolation_          DAGETINTERPOLATION
#define dacreate1d_                  DACREATE1D
#define dacreate3d_                  DACREATE3D
#define dacreate2d_                  DACREATE2D
#define dadestroy_                   DADESTROY
#define dacreateglobalvector_        DACREATEGLOBALVECTOR
#define dacreatenaturalvector_       DACREATENATURALVECTOR
#define dacreatelocalvector_         DACREATELOCALVECTOR
#define dagetlocalvector_            DAGETLOCALVECTOR
#define dagetglobalvector_           DAGETGLOBALVECTOR
#define darestorelocalvector_        DARESTORELOCALVECTOR
#define dagetscatter_                DAGETSCATTER
#define dagetglobalindices_          DAGETGLOBALINDICES
#define daview_                      DAVIEW
#define dagetinfo_                   DAGETINFO
#define dagetcoloring_               DAGETCOLORING
#define dagetmatrix_                 DAGETMATRIX
#define dagetislocaltoglobalmapping_ DAGETISLOCALTOGLOBALMAPPING
#define dagetislocaltoglobalmappingblck_ DAGETISLOCALTOGLOBALMAPPINGBLCK
#define daload_                      DALOAD
#define dasetfieldname_              DASETFIELDNAME
#define dagetfieldname_              DAGETFIELDNAME
#define darefine_                    DAREFINE
#define dagetao_                     DAGETAO
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dagetlocalinfo_              dagetlocalinfo
#define dagetlocalvector_            dagetlocalvector
#define dagetglobalvector_           dagetglobalvector
#define darestorelocalvector_        darestorelocalvector
#define dagetinterpolation_          dagetinterpolation
#define daload_                      daload
#define dacreateglobalvector_        dacreateglobalvector
#define dacreatenaturalvector_       dacreatenaturalvector
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
#define dagetmatrix_                 dagetmatrix
#define dagetislocaltoglobalmapping_ dagetislocaltoglobalmapping
#define dagetislocaltoglobalmappingblck_ dagetislocaltoglobalmappingblck
#define dasetfieldname_              dasetfieldname
#define dagetfieldname_              dagetfieldname
#define darefine_                    darefine
#define dagetao_                     dagetao
#define dasetlocalfunction_          dasetlocalfunction
#define dasetlocaladiforfunction_       dasetlocaladiforfunction
#define dasetlocaladiformffunction_       dasetlocaladiformffunction
#define dasetlocaljacobian_          dasetlocaljacobian
#endif



EXTERN_C_BEGIN


static void (PETSC_STDCALL *j1d)(DALocalInfo*,void*,void*,void*,int*);
static int ourlj1d(DALocalInfo *info,PetscScalar *in,Mat m,void *ptr)
{
  int ierr = 0;
  (*j1d)(info,&in[info->gxs],&m,ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static void (PETSC_STDCALL *j2d)(DALocalInfo*,void*,void*,void*,int*);
static int ourlj2d(DALocalInfo *info,PetscScalar **in,Mat m,void *ptr)
{
  int ierr = 0;
  (*j2d)(info,&in[info->gys][info->gxs],&m,ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static void (PETSC_STDCALL *j3d)(DALocalInfo*,void*,void*,void*,int*);
static int ourlj3d(DALocalInfo *info,PetscScalar ***in,Mat m,void *ptr)
{
  int ierr = 0;
  (*j3d)(info,&in[info->gzs][info->gys][info->gxs],&m,ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static void (PETSC_STDCALL *f1d)(DALocalInfo*,void*,void*,void*,int*);
static int ourlf1d(DALocalInfo *info,PetscScalar *in,PetscScalar *out,void *ptr)
{
  int ierr = 0;
  (*f1d)(info,&in[info->gxs],&out[info->xs],ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static void (PETSC_STDCALL *f2d)(DALocalInfo*,void*,void*,void*,int*);
static int ourlf2d(DALocalInfo *info,PetscScalar **in,PetscScalar **out,void *ptr)
{
  int ierr = 0;
  (*f2d)(info,&in[info->gys][info->gxs],&out[info->ys][info->xs],ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static void (PETSC_STDCALL *f3d)(DALocalInfo*,void*,void*,void*,int*);
static int ourlf3d(DALocalInfo *info,PetscScalar ***in,PetscScalar ***out,void *ptr)
{
  int ierr = 0;
  (*f3d)(info,&in[info->gzs][info->gys][info->gxs],&out[info->zs][info->ys][info->xs],ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL dasetlocalfunction_(DA *da,void (PETSC_STDCALL *func)(DALocalInfo*,void*,void*,void*,int*),int *ierr)
{
  int dim;

  *ierr = DAGetInfo(*da,&dim,0,0,0,0,0,0,0,0,0,0); if (*ierr) return;
  if (dim == 2) {
     f2d    = (void (PETSC_STDCALL *)(DALocalInfo*,void*,void*,void*,int*))func; 
    *ierr = DASetLocalFunction(*da,(DALocalFunction1)ourlf2d);
  } else if (dim == 3) {
     f3d    = (void (PETSC_STDCALL *)(DALocalInfo*,void*,void*,void*,int*))func; 
    *ierr = DASetLocalFunction(*da,(DALocalFunction1)ourlf3d);
  } else if (dim == 1) {
     f1d    = (void (PETSC_STDCALL *)(DALocalInfo*,void*,void*,void*,int*))func; 
    *ierr = DASetLocalFunction(*da,(DALocalFunction1)ourlf1d);
  } else *ierr = 1;
}


void PETSC_STDCALL dasetlocaladiforfunction_(DA *da,
void (PETSC_STDCALL *jfunc)(int*,DALocalInfo*,void*,void*,int*,void*,void*,int*,void*,int*),int *ierr)
{
  (*da)->adifor_lf = (DALocalFunction1)jfunc;
}

void PETSC_STDCALL dasetlocaladiformffunction_(DA *da,
void (PETSC_STDCALL *jfunc)(DALocalInfo*,void*,void*,void*,void*,void*,int*),int *ierr)
{
  (*da)->adiformf_lf = (DALocalFunction1)jfunc;
}

void PETSC_STDCALL dasetlocaljacobian_(DA *da,void (PETSC_STDCALL *jac)(DALocalInfo*,void*,void*,void*,int*),int *ierr)
{
  int dim;

  *ierr = DAGetInfo(*da,&dim,0,0,0,0,0,0,0,0,0,0); if (*ierr) return;
  if (dim == 2) {
     j2d    = (void (PETSC_STDCALL *)(DALocalInfo*,void*,void*,void*,int*))jac; 
    *ierr = DASetLocalJacobian(*da,(DALocalFunction1)ourlj2d);
  } else if (dim == 3) {
     j3d    = (void (PETSC_STDCALL *)(DALocalInfo*,void*,void*,void*,int*))jac;
    *ierr = DASetLocalJacobian(*da,(DALocalFunction1)ourlj3d);
  } else if (dim == 1) {
     j1d    = (void (PETSC_STDCALL *)(DALocalInfo*,void*,void*,void*,int*))jac; 
    *ierr = DASetLocalJacobian(*da,(DALocalFunction1)ourlj1d);
  } else *ierr = 1;
}

void PETSC_STDCALL dagetlocalinfo_(DA *da,DALocalInfo *ao,int *ierr)
{
  *ierr = DAGetLocalInfo(*da,ao);
}

void PETSC_STDCALL dagetao_(DA *da,AO *ao,int *ierr)
{
  *ierr = DAGetAO(*da,ao);
}

void PETSC_STDCALL darefine_(DA *da,MPI_Comm *comm,DA *daref, int *ierr)
{
  *ierr = DARefine(*da,(MPI_Comm)PetscToPointerComm(*comm),daref);
}

void PETSC_STDCALL dagetinterpolation_(DA *dac,DA *daf,Mat *A,Vec *scale,int *ierr)
{
  CHKFORTRANNULLOBJECT(scale);
  *ierr = DAGetInterpolation(*dac,*daf,A,scale);
}

void PETSC_STDCALL dasetfieldname_(DA *da,int *nf,CHAR name PETSC_MIXED_LEN(len),int *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(name,len,t);
  *ierr = DASetFieldName(*da,*nf,t);
  FREECHAR(name,t);
}
void PETSC_STDCALL dagetfieldname(DA *da,int *nf,CHAR name PETSC_MIXED_LEN(len),int *ierr PETSC_END_LEN(len))
{
  char *tname;

  *ierr = DAGetFieldName(*da,*nf,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(name); int len1 = _fcdlen(name);
    *ierr = PetscStrncpy(t,tname,len1);
  }
#else
  *ierr = PetscStrncpy(name,tname,len);
#endif
}

void PETSC_STDCALL daload_(PetscViewer *viewer,int *M,int *N,int *P,DA *da,int *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = DALoad(v,*M,*N,*P,da);
}

void PETSC_STDCALL dagetislocaltoglobalmapping_(DA *da,ISLocalToGlobalMapping *map,int *ierr)
{
  *ierr = DAGetISLocalToGlobalMapping(*da,map);
}

void PETSC_STDCALL dagetislocaltoglobalmappingblck_(DA *da,ISLocalToGlobalMapping *map,int *ierr)
{
  *ierr = DAGetISLocalToGlobalMappingBlck(*da,map);
}

void PETSC_STDCALL dagetcoloring_(DA *da,ISColoringType *ctype,ISColoring *coloring,int *ierr PETSC_END_LEN(len))
{
  *ierr = DAGetColoring(*da,*ctype,coloring);
}

void PETSC_STDCALL dagetmatrix_(DA *da,CHAR mat_type PETSC_MIXED_LEN(len),Mat *J,int *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(mat_type,len,t);
  *ierr = DAGetMatrix(*da,t,J);
  FREECHAR(mat_type,t);
}

void PETSC_STDCALL daview_(DA *da,PetscViewer *vin,int *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = DAView(*da,v);
}

void PETSC_STDCALL dagetglobalindices_(DA *da,int *n,int *indices,long *ia,int *ierr)
{
  int *idx;
  *ierr = DAGetGlobalIndices(*da,n,&idx);
  *ia   = PetscIntAddressToFortran(indices,idx);
}

void PETSC_STDCALL dacreateglobalvector_(DA *da,Vec* g,int *ierr)
{
  *ierr = DACreateGlobalVector(*da,g);
}

void PETSC_STDCALL dacreatenaturalvector_(DA *da,Vec* g,int *ierr)
{
  *ierr = DACreateNaturalVector(*da,g);
}

void PETSC_STDCALL dacreatelocalvector_(DA *da,Vec* l,int *ierr)
{
  *ierr = DACreateLocalVector(*da,l);
}

void PETSC_STDCALL dagetlocalvector_(DA *da,Vec* l,int *ierr)
{
  *ierr = DAGetLocalVector(*da,l);
}

void PETSC_STDCALL dagetglobalvector_(DA *da,Vec* l,int *ierr)
{
  *ierr = DAGetGlobalVector(*da,l);
}

void PETSC_STDCALL darestorelocalvector_(DA *da,Vec* l,int *ierr)
{
  *ierr = DARestoreLocalVector(*da,l);
}

void PETSC_STDCALL dagetscatter_(DA *da,VecScatter *ltog,VecScatter *gtol,VecScatter *ltol,int *ierr)
{
  CHKFORTRANNULLINTEGER(ltog);
  CHKFORTRANNULLINTEGER(gtol);
  CHKFORTRANNULLINTEGER(ltol);
  *ierr = DAGetScatter(*da,ltog,gtol,ltol);
}

void PETSC_STDCALL dadestroy_(DA *da,int *ierr)
{
  *ierr = DADestroy(*da);
}

void PETSC_STDCALL dacreate2d_(MPI_Comm *comm,DAPeriodicType *wrap,DAStencilType
                  *stencil_type,int *M,int *N,int *m,int *n,int *w,
                  int *s,int *lx,int *ly,DA *inra,int *ierr)
{
  CHKFORTRANNULLINTEGER(lx);
  CHKFORTRANNULLINTEGER(ly);
  *ierr = DACreate2d((MPI_Comm)PetscToPointerComm(*comm),*wrap,
                       *stencil_type,*M,*N,*m,*n,*w,*s,lx,ly,inra);
}

void PETSC_STDCALL dacreate1d_(MPI_Comm *comm,DAPeriodicType *wrap,int *M,int *w,int *s,
                 int *lc,DA *inra,int *ierr)
{
  CHKFORTRANNULLINTEGER(lc);
  *ierr = DACreate1d((MPI_Comm)PetscToPointerComm(*comm),*wrap,*M,*w,*s,lc,inra);
}

void PETSC_STDCALL dacreate3d_(MPI_Comm *comm,DAPeriodicType *wrap,DAStencilType 
                 *stencil_type,int *M,int *N,int *P,int *m,int *n,int *p,
                 int *w,int *s,int *lx,int *ly,int *lz,DA *inra,int *ierr)
{
  CHKFORTRANNULLINTEGER(lx);
  CHKFORTRANNULLINTEGER(ly);
  CHKFORTRANNULLINTEGER(lz);
  *ierr = DACreate3d((MPI_Comm)PetscToPointerComm(*comm),*wrap,*stencil_type,
                        *M,*N,*P,*m,*n,*p,*w,*s,lx,ly,lz,inra);
}

void PETSC_STDCALL dagetinfo_(DA *da,int *dim,int *M,int *N,int *P,int *m,int *n,int *p,int *w,int *s,
                DAPeriodicType *wrap,DAStencilType *st,int *ierr)
{
  CHKFORTRANNULLINTEGER(dim);
  CHKFORTRANNULLINTEGER(M);
  CHKFORTRANNULLINTEGER(N);
  CHKFORTRANNULLINTEGER(P);
  CHKFORTRANNULLINTEGER(m);
  CHKFORTRANNULLINTEGER(n);
  CHKFORTRANNULLINTEGER(p);
  CHKFORTRANNULLINTEGER(w);
  CHKFORTRANNULLINTEGER(s);
  CHKFORTRANNULLINTEGER(wrap);
  CHKFORTRANNULLINTEGER(st);
  *ierr = DAGetInfo(*da,dim,M,N,P,m,n,p,w,s,wrap,st);
}

EXTERN_C_END


