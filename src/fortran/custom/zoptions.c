/*$Id: zoptions.c,v 1.62 1999/10/24 14:04:19 bsmith Exp bsmith $*/

/*
  This file contains Fortran stubs for Options routines. 
  These are not generated automatically since they require passing strings
  between Fortran and C.
*/

#include "src/fortran/custom/zpetsc.h" 
#include "sys.h"
extern int          PetscBeganMPI;

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscgetarchtype_             PETSCGETARCHTYPE
#define optionsgetintarray_           OPTIONSGETINTARRAY
#define optionssetvalue_              OPTIONSSETVALUE
#define optionsclearvalue_            OPTIONSCLEARVALUE
#define optionshasname_               OPTIONSHASNAME
#define optionsgetint_                OPTIONSGETINT
#define optionsgetdouble_             OPTIONSGETDOUBLE
#define optionsgetdoublearray_        OPTIONSGETDOUBLEARRAY
#define optionsgetstring_             OPTIONSGETSTRING
#define petscgetprogramname           PETSCGETPROGRAMNAME
#define optionsinsertfile_            OPTIONSINSERTFILE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscgetarchtype_             petscgetarchtype
#define optionssetvalue_              optionssetvalue
#define optionsclearvalue_            optionsclearvalue
#define optionshasname_               optionshasname
#define optionsgetint_                optionsgetint
#define optionsgetdouble_             optionsgetdouble
#define optionsgetdoublearray_        optionsgetdoublearray
#define optionsgetstring_             optionsgetstring
#define optionsgetintarray_           optionsgetintarray
#define petscgetprogramname_          petscgetprogramname
#define optionsinsertfile_            optionsinsertfile
#endif

EXTERN_C_BEGIN

/* ---------------------------------------------------------------------*/

void PETSC_STDCALL optionsinsertfile_( CHAR file PETSC_MIXED_LEN(len), int *__ierr PETSC_END_LEN(len) )
{
  char *c1;

  FIXCHAR(file,len,c1);
  *__ierr = OptionsInsertFile(c1);
  FREECHAR(file,c1);
}

void PETSC_STDCALL optionssetvalue_(CHAR name PETSC_MIXED_LEN(len1),CHAR value PETSC_MIXED_LEN(len2),
                   int *__ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2) )
{
  char *c1,*c2;

  FIXCHAR(name,len1,c1);
  FIXCHAR(value,len2,c2);
  *__ierr = OptionsSetValue(c1,c2);
  FREECHAR(name,c1);
  FREECHAR(value,c2);
}

void PETSC_STDCALL optionsclearvalue_(CHAR name PETSC_MIXED_LEN(len),int *__ierr PETSC_END_LEN(len) )
{
  char *c1;

  FIXCHAR(name,len,c1);
  *__ierr = OptionsClearValue(c1);
  FREECHAR(name,c1);
}

void PETSC_STDCALL optionshasname_(CHAR pre PETSC_MIXED_LEN(len1),CHAR name PETSC_MIXED_LEN(len2),
                    PetscTruth *flg,int *__ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2) )
{
  char *c1,*c2;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  *__ierr = OptionsHasName(c1,c2,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

void PETSC_STDCALL optionsgetint_(CHAR pre PETSC_MIXED_LEN(len1),CHAR name PETSC_MIXED_LEN(len2),
                    int *ivalue,PetscTruth *flg,int *__ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2) )
{
  char *c1,*c2;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  *__ierr = OptionsGetInt(c1,c2,ivalue,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

void PETSC_STDCALL optionsgetdouble_(CHAR pre PETSC_MIXED_LEN(len1),CHAR name PETSC_MIXED_LEN(len2),
                    double *dvalue,PetscTruth *flg,int *__ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2) )
{
  char *c1,*c2;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  *__ierr = OptionsGetDouble(c1,c2,dvalue,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

void PETSC_STDCALL optionsgetdoublearray_(CHAR pre PETSC_MIXED_LEN(len1),CHAR name PETSC_MIXED_LEN(len2),
                double *dvalue,int *nmax,PetscTruth *flg,int *__ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2) )
{
  char *c1,*c2;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  *__ierr = OptionsGetDoubleArray(c1,c2,dvalue,nmax,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

void PETSC_STDCALL optionsgetintarray_(CHAR pre PETSC_MIXED_LEN(len1),CHAR name PETSC_MIXED_LEN(len2),
                   int *dvalue,int *nmax,PetscTruth *flg,int *__ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2) )
{
  char *c1,*c2;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  *__ierr = OptionsGetIntArray(c1,c2,dvalue,nmax,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

void PETSC_STDCALL optionsgetstring_(CHAR pre PETSC_MIXED_LEN(len1),CHAR name PETSC_MIXED_LEN(len2),
                    CHAR string PETSC_MIXED_LEN(len),PetscTruth *flg,
                    int *__ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2) PETSC_END_LEN(len) )
{
  char *c1,*c2,*c3;
  int  len3;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
#if defined(PETSC_USES_CPTOFCD)
    c3   = _fcdtocp(string);
    len3 = _fcdlen(string) - 1;
#else
    c3   = string;
    len3 = len - 1;
#endif

  *__ierr = OptionsGetString(c1,c2,c3,len3,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

void PETSC_STDCALL petscgetarchtype_(CHAR str PETSC_MIXED_LEN(len),int *__ierr PETSC_END_LEN(len) )
{
#if defined(PETSC_USES_CPTOFCD)
  char *tstr = _fcdtocp(str); 
  int  len1 = _fcdlen(str);
  *__ierr = PetscGetArchType(tstr,len1);
#else
  *__ierr = PetscGetArchType(str,len);
#endif
}

void PETSC_STDCALL petscgetprogramname_(CHAR name PETSC_MIXED_LEN(len_in),
                                        int *__ierr PETSC_END_LEN(len_in) )
{
  char *tmp;
  int  len;
#if defined(PETSC_USES_CPTOFCD)
  tmp = _fcdtocp(name);
  len = _fcdlen(name) - 1;
#else
  tmp = name;
  len = len_in - 1;
#endif
  *__ierr = PetscGetProgramName(tmp,len);
}

EXTERN_C_END

/*
    This is code for translating PETSc memory addresses to integer offsets 
    for Fortran.
*/
char   *PETSC_NULL_CHARACTER_Fortran;
void   *PETSC_NULL_INTEGER_Fortran;
void   *PETSC_NULL_SCALAR_Fortran;
void   *PETSC_NULL_DOUBLE_Fortran;
void   *PETSC_NULL_FUNCTION_Fortran;

long PetscIntAddressToFortran(int *base,int *addr)
{
  unsigned long tmp1 = (unsigned long) base,tmp2 = 0;
  unsigned long tmp3 = (unsigned long) addr;
  long          itmp2;

#if !defined(PETSC_HAVE_CRAY90_POINTER)
  if (tmp3 > tmp1) {
    tmp2  = (tmp3 - tmp1)/sizeof(int);
    itmp2 = (long) tmp2;
  } else {
    tmp2  = (tmp1 - tmp3)/sizeof(int);
    itmp2 = -((long) tmp2);
  }
#else
  if (tmp3 > tmp1) {
    tmp2  = (tmp3 - tmp1);
    itmp2 = (long) tmp2;
  } else {
    tmp2  = (tmp1 - tmp3);
    itmp2 = -((long) tmp2);
  }
#endif

  if (base + itmp2 != addr) {
    (*PetscErrorPrintf)("PetscIntAddressToFortran:C and Fortran arrays are\n");
    (*PetscErrorPrintf)("not commonly aligned or are too far apart to be indexed \n");
    (*PetscErrorPrintf)("by an integer. Locations: C %ld Fortran %ld\n",tmp1,tmp3);
    MPI_Abort(PETSC_COMM_WORLD,1);
  }
  return itmp2;
}

int *PetscIntAddressFromFortran(int *base,long addr)
{
  return base + addr;
}

/*
       obj - PETSc object on which request is made
       base - Fortran array address
       addr - C array address
       res  - will contain offset from C to Fortran
       shift - number of bytes that prevent base and addr from being commonly aligned
*/
int PetscScalarAddressToFortran(PetscObject obj,Scalar *base,Scalar *addr,int N,long *res)
{
  unsigned long tmp1 = (unsigned long) base,tmp2 = tmp1/sizeof(Scalar);
  unsigned long tmp3 = (unsigned long) addr;
  long          itmp2;
  int           shift;

#if !defined(PETSC_HAVE_CRAY90_POINTER)
  if (tmp3 > tmp1) {  /* C is bigger than Fortran */
    tmp2  = (tmp3 - tmp1)/sizeof(Scalar);
    itmp2 = (long) tmp2;
    shift = (sizeof(Scalar) - (int) ((tmp3 - tmp1) % sizeof(Scalar))) % sizeof(Scalar);
  } else {  
    tmp2  = (tmp1 - tmp3)/sizeof(Scalar);
    itmp2 = -((long) tmp2);
    shift = (int) ((tmp1 - tmp3) % sizeof(Scalar));
  }
#else
  if (tmp3 > tmp1) {  /* C is bigger than Fortran */
    tmp2  = (tmp3 - tmp1);
    itmp2 = (long) tmp2;
  } else {  
    tmp2  = (tmp1 - tmp3);
    itmp2 = -((long) tmp2);
  }
  shift = 0;
#endif
  
  if (shift) { 
    /* 
        Fortran and C not Scalar aligned, recover by copying values into
        memory that is aligned with the Fortran
    */
    int                  ierr;
    Scalar               *work;
    PetscObjectContainer container;

    work = (Scalar *) PetscMalloc((N+1)*sizeof(Scalar));CHKPTRQ(work); 

    /* shift work by that number of bytes */
    work = (Scalar *) (((char *) work) + shift);
    ierr = PetscMemcpy(work,addr,N*sizeof(Scalar));CHKERRQ(ierr);

    /* store in the first location in addr how much you shift it */
    ((int *)addr)[0] = shift;
 
    ierr = PetscObjectContainerCreate(PETSC_COMM_SELF,&container);CHKERRQ(ierr);
    ierr = PetscObjectContainerSetPointer(container,addr);CHKERRQ(ierr);
    ierr = PetscObjectCompose(obj,"GetArrayPtr",(PetscObject)container);CHKERRQ(ierr);

    tmp3 = (unsigned long) work;
    if (tmp3 > tmp1) {  /* C is bigger than Fortran */
      tmp2  = (tmp3 - tmp1)/sizeof(Scalar);
      itmp2 = (long) tmp2;
      shift = (sizeof(Scalar) - (int) ((tmp3 - tmp1) % sizeof(Scalar))) % sizeof(Scalar);
    } else {  
      tmp2  = (tmp1 - tmp3)/sizeof(Scalar);
      itmp2 = -((long) tmp2);
      shift = (int) ((tmp1 - tmp3) % sizeof(Scalar));
    }
    if (shift) {
      (*PetscErrorPrintf)("PetscScalarAddressToFortran:C and Fortran arrays are\n");
      (*PetscErrorPrintf)("not commonly aligned.\n");
      (*PetscErrorPrintf)("Locations/sizeof(Scalar): C %f Fortran %f\n",
                         ((double) tmp3)/sizeof(Scalar),((double) tmp1)/sizeof(Scalar));
      MPI_Abort(PETSC_COMM_WORLD,1);
    }
    PLogInfo((void *)obj,"PetscScalarAddressToFortran:Efficiency warning, copying array in XXXGetArray() due\n\
    to alignment differences between C and Fortran\n");
  }
  *res = itmp2;
  return 0;
}

/*
    obj - the PETSc object where the scalar pointer came from
    base - the Fortran array address
    addr - the Fortran offset from base
    N    - the amount of data

    lx   - the array space that is to be passed to XXXXRestoreArray()
*/     
int PetscScalarAddressFromFortran(PetscObject obj,Scalar *base,long addr,int N,Scalar **lx)
{
  int                  ierr,shift;
  PetscObjectContainer container;
  Scalar               *tlx;

  ierr = PetscObjectQuery(obj,"GetArrayPtr",(PetscObject *)&container);CHKERRQ(ierr);
  if (container) {
    ierr  = PetscObjectContainerGetPointer(container,(void **) lx);CHKERRQ(ierr);
    tlx   = base + addr;

    shift = *(int *)*lx;
    ierr  = PetscMemcpy(*lx,tlx,N*sizeof(Scalar));CHKERRQ(ierr);
    tlx   = (Scalar *) (((char *)tlx) - shift);
    ierr = PetscFree(tlx);CHKERRQ(ierr);
    ierr = PetscObjectContainerDestroy(container);CHKERRQ(ierr);
    ierr = PetscObjectCompose(obj,"GetArrayPtr",0);CHKERRQ(ierr);
  } else {
    *lx = base + addr;
  }
  return 0;
}

/*@C
    MPICCommToFortranComm - Converts a MPI_Comm represented
    in C to one appropriate to pass to a Fortran routine.

    Not collective

    Input Parameter:
.   cobj - the C MPI_Comm

    Output Parameter:
.   fobj - the Fortran MPI_Comm

    Level: advanced

    Notes:
    MPICCommToFortranComm() must be called in a C/C++ routine.
    MPI 1 does not provide a standard for mapping between
    Fortran and C MPI communicators; this routine handles the
    mapping correctly on all machines.

.keywords: Fortran, C, MPI_Comm, convert, interlanguage

.seealso: MPIFortranCommToCComm()
@*/
int MPICCommToFortranComm(MPI_Comm comm,int *fcomm)
{
  *fcomm = PetscFromPointerComm(comm);
  PetscFunctionReturn(0);
}

/*@C
    MPIFortranCommToCComm - Converts a MPI_Comm represented
    int Fortran (as an integer) to a MPI_Comm in C.

    Not collective

    Input Parameter:
.   fcomm - the Fortran MPI_Comm (an integer)

    Output Parameter:
.   comm - the C MPI_Comm

    Level: advanced

    Notes:
    MPIFortranCommToCComm() must be called in a C/C++ routine.
    MPI 1 does not provide a standard for mapping between
    Fortran and C MPI communicators; this routine handles the
    mapping correctly on all machines.

.keywords: Fortran, C, MPI_Comm, convert, interlanguage

.seealso: MPICCommToFortranComm()
@*/
int MPIFortranCommToCComm(int fcomm,MPI_Comm *comm)
{
  *comm = (MPI_Comm)PetscToPointerComm(fcomm);
  PetscFunctionReturn(0);
}



