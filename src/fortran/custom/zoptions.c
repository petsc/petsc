#ifndef lint
static char vcid[] = "$Id: zoptions.c,v 1.14 1996/01/30 00:40:19 bsmith Exp bsmith $";
#endif

/*
  This file contains Fortran stubs for PetscInitialize and Options routines. 
  These are not generated automatically since they require passing strings
  between Fortran and C.
*/

/*
    This is to prevent the Cray T3D version of MPI (University of Edinburgh)
  from stupidly redefining MPI_INIT(). They put this in to detect errors
  in C code, but here I do want to be calling the Fortran version from a
  C subroutine. I think their act goes against the philosophy of MPI 
  and their mpi.h file should be declared not up to the standard.
*/
#define T3DMPI_FORTRAN
#include "zpetsc.h" 
#include <stdio.h>
#include "pinclude/pviewer.h"
#include "pinclude/petscfix.h"
extern int          PetscBeganMPI;

#ifdef HAVE_FORTRAN_CAPS
#define optionsgetintarray_           OPTIONSGETINTARRAY
#define optionssetvalue_              OPTIONSSETVALUE
#define optionshasname_               OPTIONSHASNAME
#define optionsgetint_                OPTIONSGETINT
#define optionsgetdouble_             OPTIONSGETDOUBLE
#define optionsgetdoublearray_        OPTIONSGETDOUBLEARRAY
#define petscfinalize_                PETSCFINALIZE
#define petscsetcommonblock_          PETSCSETCOMMONBLOCK
#define petscsetfortranbasepointers_  PETSCSETFORTRANBASEPOINTERS
#define optionsgetstring_             OPTIONSGETSTRING
#define petscinitialize_              PETSCINITIALIZE
#define iargc_                        IARGC
#define getarg_                       GETARG
#define mpi_init_                     MPI_INIT
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define optionssetvalue_              optionssetvalue
#define optionshasname_               optionshasname
#define optionsgetint_                optionsgetint
#define optionsgetdouble_             optionsgetdouble
#define optionsgetdoublearray_        optionsgetdoublearray
#define petscfinalize_                petscfinalize
#define petscsetcommonblock_          petscsetcommonblock
#define petscsetfortranbasepointers_  petscsetfortranbasepointers
#define optionsgetstring_             optionsgetstring
#define optionsgetintarray_           optionsgetintarray
#define petscinitialize_              petscinitialize
#define mpi_init_                     mpi_init

/*
    HP-UX does not have Fortran underscore but iargc and getarg 
  do have underscores????
*/
#if !defined(PARCH_hpux)
#define iargc_                        iargc
#define getarg_                       getarg
#endif

#endif

int OptionsCheckInitial_Private(),
    OptionsCreate_Private(int*,char***,char*,char*),
    OptionsSetAlias_Private(char *,char *);

/*
    The extra _ is because the f2c compiler puts an
  extra _ at the end if the original routine name 
  contained any _.
*/
#if defined(PARCH_freebsd) | defined(PARCH_linux)
#define mpi_init_             mpi_init__
#endif

#if defined(__cplusplus)
extern "C" {
#endif
extern void mpi_init_(int*);
extern void petscsetcommonblock_(int*,int*,int*);
extern int  iargc_();
extern void getarg_(int*,char*,int);
#if defined(PARCH_t3d)
extern void PXFGETARG(int *,_fcd,int*,int*);
#endif
#if defined(__cplusplus)
}
#endif

/*
    Reads in Fortran command line argments and sends them to 
  all processors and adds them to Options database.
*/

int PETScParseFortranArgs_Private(int *argc,char ***argv)
{
  int  i, warg = 256,rank;
  char *p;

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (!rank) {
    *argc = 1 + iargc_();
  }
  MPI_Bcast(argc,1,MPI_INT,0,MPI_COMM_WORLD);

  *argv = (char **) PetscMalloc((*argc+1)*(warg*sizeof(char)+sizeof(char*))); 
  CHKPTRQ(*argv);
  (*argv)[0] = (char*) (*argv + *argc + 1);

  if (!rank) {
    PetscMemzero((*argv)[0],(*argc)*warg*sizeof(char));
    for ( i=0; i<*argc; i++ ) {
      (*argv)[i+1] = (*argv)[i] + warg;
#if defined(PARCH_t3d)
      {char *tmp = (*argv)[i]; 
       int  ierr,ilen;
       PXFGETARG(&i, _cptofcd(tmp,warg),&ilen,&ierr); CHKERRQ(ierr);
       tmp[ilen] = 0;
      } 
#else
      getarg_( &i, (*argv)[i], warg );
#endif
      /* zero out garbage at end of each argument */
      p = (*argv)[i] + warg-1;
      while (p > (*argv)[i]) {
        if (*p == ' ') *p = 0; 
        p--;
      }
    }
  }
  MPI_Bcast((*argv)[0],*argc*warg,MPI_CHAR,0,MPI_COMM_WORLD);  
  if (rank) {
    for ( i=0; i<*argc; i++ ) {
      (*argv)[i+1] = (*argv)[i] + warg;
    }
  } 
  return 0;   
}

#if defined(__cplusplus)
extern "C" {
#endif

extern int PetscInitializedCalled;

void petscinitialize_(int *err)
{
  int  flag,argc = 0,s1,s2,s3;
  char **args = 0;
  *err = 1;

  if (PetscInitializedCalled) {*err = 0; return;}
  PetscInitializedCalled = 1;

  MPI_Initialized(&flag);
  if (!flag) {
    mpi_init_(err);
    if (*err) {fprintf(stderr,"PetscInitialize:");return;}
    PetscBeganMPI = 1;
  }
#if defined(PETSC_COMPLEX)
  MPI_Type_contiguous(2,MPI_DOUBLE,&MPIU_COMPLEX);
  MPI_Type_commit(&MPIU_COMPLEX);
#endif
  PETScParseFortranArgs_Private(&argc,&args);
  *err = OptionsCreate_Private(&argc,&args,0,0); 
  if (*err) { fprintf(stderr,"PETSC ERROR: PetscInitialize:");return;}
  PetscFree(args);
  *err = OptionsCheckInitial_Private(); 
  if (*err) { fprintf(stderr,"PETSC ERROR: PetscInitialize:");return;}
  *err = ViewerInitialize_Private(); 
  if (*err) { fprintf(stderr,"PETSC ERROR: PetscInitialize:");return;}

  s1 = MPIR_FromPointer(STDOUT_VIEWER_SELF);
  s2 = MPIR_FromPointer(STDERR_VIEWER_SELF);
  s3 = MPIR_FromPointer(STDOUT_VIEWER_WORLD);
  petscsetcommonblock_(&s1,&s2,&s3);
  if (PetscBeganMPI) {
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    PLogInfo(0,"[%d] PETSc successfully started: procs %d\n",rank,size);
  }
  *err = 0;
}

void petscfinalize_(int *ierr){
  *ierr = PetscFinalize();
}

/* ---------------------------------------------------------------------*/

void optionssetvalue_(CHAR name,CHAR value,int *err, int len1,int len2)
{
  char *c1,*c2;
  int  ierr;
  FIXCHAR(name,len1,c1);
  FIXCHAR(value,len2,c2);
  ierr = OptionsSetValue(c1,c2);
  FREECHAR(name,c1);
  FREECHAR(value,c2);
  *err = ierr;
}

void optionshasname_(CHAR pre,CHAR name,int *flg,int *err,int len1,int len2){
  char *c1,*c2;
  int  ierr;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  ierr = OptionsHasName(c1,c2,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
  *err = ierr;
}

void optionsgetint_(CHAR pre,CHAR name,int *ivalue,int *flg,int *err,int len1,int len2){
  char *c1,*c2;
  int  ierr;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  ierr = OptionsGetInt(c1,c2,ivalue,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
  *err = ierr;
}

void optionsgetdouble_(CHAR pre,CHAR name,double *dvalue,int *flg,int *err,
                       int len1,int len2){
  char *c1,*c2;
  int  ierr;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  ierr = OptionsGetDouble(c1,c2,dvalue,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
  *err = ierr;
}

void optionsgetdoublearray_(CHAR pre,CHAR name,
              double *dvalue,int *nmax,int *flg,int *err,int len1,int len2)
{
  char *c1,*c2;
  int  ierr;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  ierr = OptionsGetDoubleArray(c1,c2,dvalue,nmax,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);

  *err = ierr;
}

void optionsgetintarray_(CHAR pre,CHAR name,int *dvalue,int *nmax,int *flg,
                         int *err,int len1,int len2)
{
  char *c1,*c2;
  int  ierr;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  ierr = OptionsGetIntArray(c1,c2,dvalue,nmax,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);

  *err = ierr;
}

void optionsgetstring_(CHAR pre,CHAR name,CHAR string,int *flg,
                       int *err, int len1, int len2,int len){
  char *c1,*c2,*c3;
  int  ierr,len3;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
#if defined(PARCH_t3d)
    c3   = _fcdtocp(string);
    len3 = _fcdlen(string) - 1;
#else
    c3   = string;
    len3 = len - 1;
#endif

  ierr = OptionsGetString(c1,c2,c3,len3,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);

  *err = ierr;
}

#if defined(PARCH_t3d)

void petscsetfortranbasepointers_(void *fnull,_fcd fcnull)
{
  PETSC_NULL_Fortran       = fnull;
  PETSC_NULL_CHAR_Fortran  = _fcdtocp(fcnull);
}

#else

void petscsetfortranbasepointers_(void *fnull,char *fcnull)
{
  PETSC_NULL_Fortran       = fnull;
  PETSC_NULL_CHAR_Fortran  = fcnull;
}

#endif  /* end of !defined(PARCH_t3d) */

#if defined(__cplusplus)
}
#endif


/*
    This is code for translating PETSc memory addresses to integer offsets 
    for Fortran.
*/
void   *PETSC_NULL_Fortran, *PETSC_NULL_CHAR_Fortran;

int PetscIntAddressToFortran(int *base,int *addr)
{
  return (int) (((long)addr) - ((long)base))/sizeof(int);
}

int *PetscIntAddressFromFortran(int *base,int addr)
{
  return base + addr;
}

int PetscScalarAddressToFortran(Scalar *base,Scalar *addr)
{
  long tmp1 = (long) base,tmp2 = tmp1/sizeof(Scalar);
  if (tmp2*sizeof(Scalar) != tmp1) {
    fprintf(stderr,"PetscScalarAddressToFortran:unaligned Fortran double\n");
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  return (int) (((long)addr) - ((long)base))/sizeof(Scalar);
}

Scalar *PetscScalarAddressFromFortran(Scalar *base,int addr)
{
  return base + addr;
}







