#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zoptions.c,v 1.31 1997/07/09 20:55:52 balay Exp balay $";
#endif

/*
  This file contains Fortran stubs for Options routines. 
  These are not generated automatically since they require passing strings
  between Fortran and C.
*/

#include "src/fortran/custom/zpetsc.h" 
#include "sys.h"
#include <stdio.h>
#include "pinclude/pviewer.h"
#include "pinclude/petscfix.h"
extern int          PetscBeganMPI;

#ifdef HAVE_FORTRAN_CAPS
#define petscgetarchtype_             PETSCGETARCHTYPE
#define optionsgetintarray_           OPTIONSGETINTARRAY
#define optionssetvalue_              OPTIONSSETVALUE
#define optionshasname_               OPTIONSHASNAME
#define optionsgetint_                OPTIONSGETINT
#define optionsgetdouble_             OPTIONSGETDOUBLE
#define optionsgetdoublearray_        OPTIONSGETDOUBLEARRAY
#define optionsgetstring_             OPTIONSGETSTRING
#define petscgetprogramname_          PETSCGETPROGRAMNAME
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define petscgetarchtype_             petscgetarchtype
#define optionssetvalue_              optionssetvalue
#define optionshasname_               optionshasname
#define optionsgetint_                optionsgetint
#define optionsgetdouble_             optionsgetdouble
#define optionsgetdoublearray_        optionsgetdoublearray
#define optionsgetstring_             optionsgetstring
#define optionsgetintarray_           optionsgetintarray
#define petscgetprogramname_       petscgetprogramname
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/* ---------------------------------------------------------------------*/

void optionssetvalue_(CHAR name,CHAR value,int *__ierr, int len1,int len2)
{
  char *c1,*c2;
  int  ierr;

  FIXCHAR(name,len1,c1);
  FIXCHAR(value,len2,c2);
  ierr = OptionsSetValue(c1,c2);
  FREECHAR(name,c1);
  FREECHAR(value,c2);
  *__ierr = ierr;
}

void optionshasname_(CHAR pre,CHAR name,int *flg,int *__ierr,int len1,int len2){
  char *c1,*c2;
  int  ierr;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  ierr = OptionsHasName(c1,c2,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
  *__ierr = ierr;
}

void optionsgetint_(CHAR pre,CHAR name,int *ivalue,int *flg,int *__ierr,int len1,int len2)
{
  char *c1,*c2;
  int  ierr;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  ierr = OptionsGetInt(c1,c2,ivalue,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
  *__ierr = ierr;
}

void optionsgetdouble_(CHAR pre,CHAR name,double *dvalue,int *flg,int *__ierr,
                       int len1,int len2){
  char *c1,*c2;
  int  ierr;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  ierr = OptionsGetDouble(c1,c2,dvalue,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
  *__ierr = ierr;
}

void optionsgetdoublearray_(CHAR pre,CHAR name,
              double *dvalue,int *nmax,int *flg,int *__ierr,int len1,int len2)
{
  char *c1,*c2;
  int  ierr;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  ierr = OptionsGetDoubleArray(c1,c2,dvalue,nmax,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);

  *__ierr = ierr;
}

void optionsgetintarray_(CHAR pre,CHAR name,int *dvalue,int *nmax,int *flg,
                         int *__ierr,int len1,int len2)
{
  char *c1,*c2;
  int  ierr;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  ierr = OptionsGetIntArray(c1,c2,dvalue,nmax,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);

  *__ierr = ierr;
}

void optionsgetstring_(CHAR pre,CHAR name,CHAR string,int *flg,
                       int *__ierr, int len1, int len2,int len){
  char *c1,*c2,*c3;
  int  ierr,len3;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
#if defined(USES_CPTOFCD)
    c3   = _fcdtocp(string);
    len3 = _fcdlen(string) - 1;
#else
    c3   = string;
    len3 = len - 1;
#endif

  ierr = OptionsGetString(c1,c2,c3,len3,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);

  *__ierr = ierr;
}

void petscgetarchtype_(CHAR str,int *__ierr,int len)
{
#if defined(USES_CPTOFCD)
  char *tstr = _fcdtocp(str); 
  int  len1 = _fcdlen(str);
  *__ierr = PetscGetArchType(tstr,len1);
#else
  *__ierr = PetscGetArchType(str,len);
#endif
}

void petscgetprogramname_(CHAR name, int *__ierr,int len_in )
{
  char *tmp;
  int len;
#if defined(USES_CPTOFCD)
  tmp = _fcdtocp(name);
  len = _fcdlen(name) - 1;
#else
  tmp = name;
  len = len_in - 1;
#endif
  *__ierr = PetscGetProgramName(tmp,len);
}

#if defined(__cplusplus)
}
#endif


/*
    This is code for translating PETSc memory addresses to integer offsets 
    for Fortran.
*/
void   *PETSC_NULL_Fortran;
char   *PETSC_NULL_CHARACTER_Fortran;

int PetscIntAddressToFortran(int *base,int *addr)
{
  unsigned long tmp1 = (unsigned long) base,tmp2 = tmp1/sizeof(int);
  unsigned long tmp3 = (unsigned long) addr;
  int           itmp2;

  if (tmp3 > tmp1) {
    tmp2  = (tmp3 - tmp1)/sizeof(int);
    itmp2 = (int) tmp2;
  }
  else {
    tmp2  = (tmp1 - tmp3)/sizeof(int);
    itmp2 = -((int) tmp2);
  }
  if (base + itmp2 != addr) {
    fprintf(stderr,"PetscIntAddressToFortran:C and Fortran arrays are\n");
    fprintf(stderr,"not commonly aligned or are too far apart to be indexed \n");
    fprintf(stderr,"by an integer. Locations: C %ld Fortran %ld\n",tmp1,tmp3);
    MPI_Abort(PETSC_COMM_WORLD,1);
  }
  return itmp2;
}

int *PetscIntAddressFromFortran(int *base,int addr)
{
  return base + addr;
}

int PetscScalarAddressToFortran(Scalar *base,Scalar *addr)
{
  unsigned long tmp1 = (unsigned long) base,tmp2 = tmp1/sizeof(Scalar);
  unsigned long tmp3 = (unsigned long) addr;
  int           itmp2;

  if (tmp3 > tmp1) {
    tmp2  = (tmp3 - tmp1)/sizeof(Scalar);
    itmp2 = (int) tmp2;
  }
  else {
    tmp2  = (tmp1 - tmp3)/sizeof(Scalar);
    itmp2 = -((int) tmp2);
  }
  if (base + itmp2 != addr) {
    fprintf(stderr,"PetscScalarAddressToFortran:C and Fortran arrays are\n");
    fprintf(stderr,"not commonly aligned or are too far apart to be indexed \n");
    fprintf(stderr,"by an integer. Locations: C %ld Fortran %ld\n",tmp3,tmp1);
    fprintf(stderr,"Locations/sizeof(Scalar): C %f Fortran %f\n",((double) tmp3)/sizeof(Scalar),
                                                 ((double) tmp1)/sizeof(Scalar));
    MPI_Abort(PETSC_COMM_WORLD,1);
  }
  return itmp2;
}

Scalar *PetscScalarAddressFromFortran(Scalar *base,int addr)
{
  return base + addr;
}

/*@
    PetscCObjectToFortranObject - Converts a PETSc object represented
    in C to one appropriate to pass to a Fortran routine.

    Input Parameter:
.   cobj - the PETSc C object

    Output Parameter:
.   fobj - the PETSc Fortran object

    Notes:
    PetscCObjectToFortranObject() must be called in a C/C++ routine.
    See examples petsc/src/vec/examples/ex24.c and ex24f.F

.keywords: Fortran, C, object, convert, interlanguage

.seealso: PetscFortranObjectToCObject()
@*/
int PetscCObjectToFortranObject(void *cobj,int *fobj)
{
  PetscValidHeader(cobj);
  *fobj = PetscFromPointer(cobj);
  return 0;
}

/*@
    PetscFortranObjectToCObject - Converts a PETSc object represented
    in Fortran to one appropriate for C.

    Input Parameter:
.   fobj - the PETSc Fortran object

    Output Parameter:
.   cobj - the PETSc C object

    Notes:
    PetscCObjectToFortranObject() must be called in a C/C++ routine.
    See examples petsc/src/vec/examples/ex24.c and ex24f.F

.keywords: Fortran, C, object, convert, interlanguage

.seealso: PetscCObjectToFortranObject()
@*/
int PetscFortranObjectToCObject(int fobj,void *cobj)
{
  (*(void **) cobj) = (void *) PetscToPointer(fobj);
  return 0;
}

/*@
    MPICCommToFortranComm - Converts a MPI_Comm represented
    in C to one appropriate to pass to a Fortran routine.

    Input Parameter:
.   cobj - the C MPI_Comm

    Output Parameter:
.   fobj - the Fortran MPI_Comm

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
  return 0;
}

/*@
    MPIFortranCommToCComm - Converts a MPI_Comm represented
    int Fortran (as an integer) to a MPI_Comm in C.

    Input Parameter:
.   fcomm - the Fortran MPI_Comm (an integer)

    Output Parameter:
.   comm - the C MPI_Comm

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
  return 0;
}



