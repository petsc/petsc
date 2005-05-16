/*
  This file contains Fortran stubs for Options routines. 
  These are not generated automatically since they require passing strings
  between Fortran and C.
*/

/*MC
   PetscFortranAddr - a variable type in Fortran that can hold a
     regular C pointer.

   Notes: Used, for example, as the file argument in PetscFOpen()

   Level: beginner

.seealso:  PetscOffset, PetscInt
M*/
/*MC
   PetscOffset - a variable type in Fortran used with VecGetArray()
     and ISGetIndices()

   Level: beginner

.seealso:  PetscFortranAddr, PetscInt
M*/

#include "zpetsc.h" 
#include "petscsys.h"
extern PetscTruth PetscBeganMPI;

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscoptionsgettruth_            PETSCOPTIONSGETTRUTH
#define petscgetarchtype_                  PETSCGETARCHTYPE
#define petscoptionsgetintarray_           PETSCOPTIONSGETINTARRAY
#define petscoptionssetvalue_              PETSCOPTIONSSETVALUE
#define petscoptionsclearvalue_            PETSCOPTIONSCLEARVALUE
#define petscoptionshasname_               PETSCOPTIONSHASNAME
#define petscoptionsgetint_                PETSCOPTIONSGETINT
#define petscoptionsgetreal_               PETSCOPTIONSGETREAL
#define petscoptionsgetrealarray_          PETSCOPTIONSGETREALARRAY
#define petscoptionsgetstring_             PETSCOPTIONSGETSTRING
#define petscgetprogramname                PETSCGETPROGRAMNAME
#define petscoptionsinsertfile_            PETSCOPTIONSINSERTFILE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscoptionsgettruth_            petscoptionsgettruth
#define petscgetarchtype_                  petscgetarchtype
#define petscoptionssetvalue_              petscoptionssetvalue
#define petscoptionsclearvalue_            petscoptionsclearvalue
#define petscoptionshasname_               petscoptionshasname
#define petscoptionsgetint_                petscoptionsgetint
#define petscoptionsgetreal_               petscoptionsgetreal
#define petscoptionsgetrealarray_          petscoptionsgetrealarray
#define petscoptionsgetstring_             petscoptionsgetstring
#define petscoptionsgetintarray_           petscoptionsgetintarray
#define petscgetprogramname_               petscgetprogramname
#define petscoptionsinsertfile_            petscoptionsinsertfile
#endif

EXTERN_C_BEGIN

/* ---------------------------------------------------------------------*/

void PETSC_STDCALL petscoptionsinsertfile_(CHAR file PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;

  FIXCHAR(file,len,c1);
  *ierr = PetscOptionsInsertFile(c1);
  FREECHAR(file,c1);
}

void PETSC_STDCALL petscoptionssetvalue_(CHAR name PETSC_MIXED_LEN(len1),CHAR value PETSC_MIXED_LEN(len2),
                   PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *c1,*c2;

  FIXCHAR(name,len1,c1);
  FIXCHAR(value,len2,c2);
  *ierr = PetscOptionsSetValue(c1,c2);
  FREECHAR(name,c1);
  FREECHAR(value,c2);
}

void PETSC_STDCALL petscoptionsclearvalue_(CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;

  FIXCHAR(name,len,c1);
  *ierr = PetscOptionsClearValue(c1);
  FREECHAR(name,c1);
}

void PETSC_STDCALL petscoptionshasname_(CHAR pre PETSC_MIXED_LEN(len1),CHAR name PETSC_MIXED_LEN(len2),
                    PetscTruth *flg,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *c1,*c2;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  *ierr = PetscOptionsHasName(c1,c2,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

void PETSC_STDCALL petscoptionsgetint_(CHAR pre PETSC_MIXED_LEN(len1),CHAR name PETSC_MIXED_LEN(len2),
                    PetscInt *ivalue,PetscTruth *flg,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *c1,*c2;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  *ierr = PetscOptionsGetInt(c1,c2,ivalue,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

void PETSC_STDCALL petscoptionsgettruth_(CHAR pre PETSC_MIXED_LEN(len1),CHAR name PETSC_MIXED_LEN(len2),
                    PetscTruth *ivalue,PetscTruth *flg,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *c1,*c2;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  *ierr = PetscOptionsGetTruth(c1,c2,ivalue,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

void PETSC_STDCALL petscoptionsgetreal_(CHAR pre PETSC_MIXED_LEN(len1),CHAR name PETSC_MIXED_LEN(len2),
                    PetscReal *dvalue,PetscTruth *flg,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *c1,*c2;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  *ierr = PetscOptionsGetReal(c1,c2,dvalue,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

void PETSC_STDCALL petscoptionsgetrealarray_(CHAR pre PETSC_MIXED_LEN(len1),CHAR name PETSC_MIXED_LEN(len2),
                PetscReal *dvalue,PetscInt *nmax,PetscTruth *flg,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *c1,*c2;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  *ierr = PetscOptionsGetRealArray(c1,c2,dvalue,nmax,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

void PETSC_STDCALL petscoptionsgetintarray_(CHAR pre PETSC_MIXED_LEN(len1),CHAR name PETSC_MIXED_LEN(len2),
                   PetscInt *dvalue,PetscInt *nmax,PetscTruth *flg,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *c1,*c2;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  *ierr = PetscOptionsGetIntArray(c1,c2,dvalue,nmax,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

void PETSC_STDCALL petscoptionsgetstring_(CHAR pre PETSC_MIXED_LEN(len1),CHAR name PETSC_MIXED_LEN(len2),
                    CHAR string PETSC_MIXED_LEN(len),PetscTruth *flg,
                    PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2) PETSC_END_LEN(len))
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

  *ierr = PetscOptionsGetString(c1,c2,c3,len3,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
  FIXRETURNCHAR(string,len);
}

void PETSC_STDCALL petscgetarchtype_(CHAR str PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
#if defined(PETSC_USES_CPTOFCD)
  char *tstr = _fcdtocp(str); 
  int  len1 = _fcdlen(str);
  *ierr = PetscGetArchType(tstr,len1);
#else
  *ierr = PetscGetArchType(str,len);
#endif
  FIXRETURNCHAR(str,len);

}

void PETSC_STDCALL petscgetprogramname_(CHAR name PETSC_MIXED_LEN(len_in),PetscErrorCode *ierr PETSC_END_LEN(len_in))
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
  *ierr = PetscGetProgramName(tmp,len);
  FIXRETURNCHAR(name,len_in);
}

EXTERN_C_END

/*
    This is code for translating PETSc memory addresses to integer offsets 
    for Fortran.
*/
char   *PETSC_NULL_CHARACTER_Fortran = 0;
void   *PETSC_NULL_INTEGER_Fortran   = 0;
void   *PETSC_NULL_OBJECT_Fortran    = 0;
void   *PETSC_NULL_Fortran           = 0;
void   *PETSC_NULL_SCALAR_Fortran    = 0;
void   *PETSC_NULL_DOUBLE_Fortran    = 0;
void   *PETSC_NULL_REAL_Fortran      = 0;
EXTERN_C_BEGIN
void   (*PETSC_NULL_FUNCTION_Fortran)(void) = 0;
EXTERN_C_END
size_t PetscIntAddressToFortran(PetscInt *base,PetscInt *addr)
{
  size_t tmp1 = (size_t) base,tmp2 = 0;
  size_t tmp3 = (size_t) addr;
  size_t itmp2;

#if !defined(PETSC_HAVE_CRAY90_POINTER)
  if (tmp3 > tmp1) {
    tmp2  = (tmp3 - tmp1)/sizeof(PetscInt);
    itmp2 = (size_t) tmp2;
  } else {
    tmp2  = (tmp1 - tmp3)/sizeof(PetscInt);
    itmp2 = -((size_t) tmp2);
  }
#else
  if (tmp3 > tmp1) {
    tmp2  = (tmp3 - tmp1);
    itmp2 = (size_t) tmp2;
  } else {
    tmp2  = (tmp1 - tmp3);
    itmp2 = -((size_t) tmp2);
  }
#endif

  if (base + itmp2 != addr) {
    (*PetscErrorPrintf)("PetscIntAddressToFortran:C and Fortran arrays are\n");
    (*PetscErrorPrintf)("not commonly aligned or are too far apart to be indexed \n");
    (*PetscErrorPrintf)("by an integer. Locations: C %uld Fortran %uld\n",tmp1,tmp3);
    MPI_Abort(PETSC_COMM_WORLD,1);
  }
  return itmp2;
}

PetscInt *PetscIntAddressFromFortran(PetscInt *base,size_t addr)
{
  return base + addr;
}

/*
       obj - PETSc object on which request is made
       base - Fortran array address
       addr - C array address
       res  - will contain offset from C to Fortran
       shift - number of bytes that prevent base and addr from being commonly aligned
       N - size of the array

*/
PetscErrorCode PetscScalarAddressToFortran(PetscObject obj,PetscScalar *base,PetscScalar *addr,PetscInt N,size_t *res)
{
  size_t   tmp1 = (size_t) base,tmp2 = tmp1/sizeof(PetscScalar);
  size_t   tmp3 = (size_t) addr;
  size_t   itmp2;
  PetscInt shift;

#if !defined(PETSC_HAVE_CRAY90_POINTER)
  if (tmp3 > tmp1) {  /* C is bigger than Fortran */
    tmp2  = (tmp3 - tmp1)/sizeof(PetscScalar);
    itmp2 = (size_t) tmp2;
    shift = (sizeof(PetscScalar) - (int)((tmp3 - tmp1) % sizeof(PetscScalar))) % sizeof(PetscScalar);
  } else {  
    tmp2  = (tmp1 - tmp3)/sizeof(PetscScalar);
    itmp2 = -((size_t) tmp2);
    shift = (int)((tmp1 - tmp3) % sizeof(PetscScalar));
  }
#else
  if (tmp3 > tmp1) {  /* C is bigger than Fortran */
    tmp2  = (tmp3 - tmp1);
    itmp2 = (size_t) tmp2;
  } else {  
    tmp2  = (tmp1 - tmp3);
    itmp2 = -((size_t) tmp2);
  }
  shift = 0;
#endif
  
  if (shift) { 
    /* 
        Fortran and C not PetscScalar aligned,recover by copying values into
        memory that is aligned with the Fortran
    */
    PetscErrorCode       ierr;
    PetscScalar          *work;
    PetscObjectContainer container;

    ierr = PetscMalloc((N+1)*sizeof(PetscScalar),&work);CHKERRQ(ierr); 

    /* shift work by that number of bytes */
    work = (PetscScalar*)(((char*)work) + shift);
    ierr = PetscMemcpy(work,addr,N*sizeof(PetscScalar));CHKERRQ(ierr);

    /* store in the first location in addr how much you shift it */
    ((PetscInt*)addr)[0] = shift;
 
    ierr = PetscObjectContainerCreate(PETSC_COMM_SELF,&container);CHKERRQ(ierr);
    ierr = PetscObjectContainerSetPointer(container,addr);CHKERRQ(ierr);
    ierr = PetscObjectCompose(obj,"GetArrayPtr",(PetscObject)container);CHKERRQ(ierr);

    tmp3 = (size_t) work;
    if (tmp3 > tmp1) {  /* C is bigger than Fortran */
      tmp2  = (tmp3 - tmp1)/sizeof(PetscScalar);
      itmp2 = (size_t) tmp2;
      shift = (sizeof(PetscScalar) - (int)((tmp3 - tmp1) % sizeof(PetscScalar))) % sizeof(PetscScalar);
    } else {  
      tmp2  = (tmp1 - tmp3)/sizeof(PetscScalar);
      itmp2 = -((size_t) tmp2);
      shift = (int)((tmp1 - tmp3) % sizeof(PetscScalar));
    }
    if (shift) {
      (*PetscErrorPrintf)("PetscScalarAddressToFortran:C and Fortran arrays are\n");
      (*PetscErrorPrintf)("not commonly aligned.\n");
      /* double/int doesn't work with ADIC */
      (*PetscErrorPrintf)("Locations/sizeof(PetscScalar): C %f Fortran %f\n",
                         ((PetscReal)tmp3)/(PetscReal)sizeof(PetscScalar),((PetscReal)tmp1)/(PetscReal)sizeof(PetscScalar));
      MPI_Abort(PETSC_COMM_WORLD,1);
    }
    ierr = PetscLogInfo(((void*)obj,"PetscScalarAddressToFortran:Efficiency warning, copying array in XXXGetArray() due\n\
    to alignment differences between C and Fortran\n"));CHKERRQ(ierr);
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
PetscErrorCode PetscScalarAddressFromFortran(PetscObject obj,PetscScalar *base,size_t addr,PetscInt N,PetscScalar **lx)
{
  PetscErrorCode       ierr;
  PetscInt             shift;
  PetscObjectContainer container;
  PetscScalar          *tlx;

  ierr = PetscObjectQuery(obj,"GetArrayPtr",(PetscObject *)&container);CHKERRQ(ierr);
  if (container) {
    ierr  = PetscObjectContainerGetPointer(container,(void**)lx);CHKERRQ(ierr);
    tlx   = base + addr;

    shift = *(PetscInt*)*lx;
    ierr  = PetscMemcpy(*lx,tlx,N*sizeof(PetscScalar));CHKERRQ(ierr);
    tlx   = (PetscScalar*)(((char *)tlx) - shift);
    ierr = PetscFree(tlx);CHKERRQ(ierr);
    ierr = PetscObjectContainerDestroy(container);CHKERRQ(ierr);
    ierr = PetscObjectCompose(obj,"GetArrayPtr",0);CHKERRQ(ierr);
  } else {
    *lx = base + addr;
  }
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "MPICCommToFortranComm"
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
PetscErrorCode MPICCommToFortranComm(MPI_Comm comm,int *fcomm)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  /* call to MPI_Comm_size() is for error checking on comm */
  ierr = MPI_Comm_size(comm,&size);
  if (ierr) SETERRQ(PETSC_ERR_ARG_CORRUPT ,"Invalid MPI communicator");

  *fcomm = PetscFromPointerComm(comm);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MPIFortranCommToCComm"
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
PetscErrorCode MPIFortranCommToCComm(int fcomm,MPI_Comm *comm)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  *comm = (MPI_Comm)PetscToPointerComm(fcomm);
  /* call to MPI_Comm_size() is for error checking on comm */
  ierr = MPI_Comm_size(*comm,&size);
  if (ierr) SETERRQ(PETSC_ERR_ARG_CORRUPT,"Invalid MPI communicator");
  PetscFunctionReturn(0);
}



