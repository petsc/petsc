#ifndef lint
static char vcid[] = "$Id: zoptions.c,v 1.10 1995/11/28 19:42:25 curfman Exp curfman $";
#endif

/*
  This file contatins Fortran stubs for PetscInitialize and Options routines. 
  These are not generated automatically since they require passing strings
  between Fortran and C.
*/

#include "zpetsc.h"
#include "petsc.h"
#include <stdio.h>
#include "pinclude/pviewer.h"
#include "pinclude/petscfix.h"
extern int          PetscBeganMPI;
#if defined(PETSC_COMPLEX)
extern MPI_Datatype  MPIU_COMPLEX;
#endif

#ifdef FORTRANCAPS
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
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
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
#endif

int OptionsCheckInitial_Private(),
    OptionsCreate_Private(int*,char***,char*,char*),
    OptionsSetAlias_Private(char *,char *);

/*
     This first block is probably only due to using MPI-CH
   I DO NOT understand why they put a __ in. It seems nuts
   and TOTALLY unneeded!
*/
#if defined(PARCH_freebsd) | defined(PARCH_linux)
#define mpi_init        mpi_init__
#define petscinitialize petscinitialize_
#define iargc           iargc_
#define getarg          getarg_
#elif defined(FORTRANCAPS)
#define petscinitialize PETSCINITIALIZE
#define mpi_init        MPI_INIT
#define iargc           IARGC
#define getarg          GETARG
#elif defined(FORTRANUNDERSCORE)
#define petscinitialize petscinitialize_
#define mpi_init        mpi_init_
#define iargc           iargc_
#define getarg          getarg_
#endif
#if defined(__cplusplus)
extern "C" {
#endif
extern void mpi_init(int*);
extern void petscsetcommonblock_(int*,int*,int*);
#if defined(__cplusplus)
}
#endif

/*
    Reads in Fortran command line argments and sends them to 
  all processors and adds them to Options database.
*/
#if defined(__cplusplus)
extern "C" {
#endif
extern int iargc_();
extern void getarg_(int*,char*,int);
#if defined(__cplusplus)
}
#endif
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
      getarg_( &i, (*argv)[i], warg );
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

int petscinitialize(int *err)
{
  int  ierr,flag,argc = 0,s1,s2,s3;
  char **args = 0;
  *err = 1;

  MPI_Initialized(&flag);
  if (!flag) {
    mpi_init(err);if (*err) SETERRQ(*err,"PetscInitialize:Could not mpi_init");
    PetscBeganMPI = 1;
  }
#if defined(PETSC_COMPLEX)
  MPI_Type_contiguous(2,MPI_DOUBLE,&MPIU_COMPLEX);
  MPI_Type_commit(&MPIU_COMPLEX);
#endif
  PETScParseFortranArgs_Private(&argc,&args);
  ierr = OptionsCreate_Private(&argc,&args,0,0); CHKERRQ(ierr);
  PetscFree(args);
  ierr = OptionsCheckInitial_Private(); CHKERRQ(ierr);
  ierr = ViewerInitialize_Private(); CHKERRQ(ierr);

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
  return 0;
}
int  petscfinalize_(int *ierr){
  return *ierr = PetscFinalize();
}


int  optionssetvalue_(char *name,char *value,int *err, int len1,int len2)
{
  char *c1,*c2;
  int  ierr;
  if (!name[len1] == 0) {
    c1 = (char *) PetscMalloc( (len1+1)*sizeof(char)); 
    PetscStrncpy(c1,name,len1);
    c1[len1] = 0;
  } else c1 = name;
  if (!value[len2] == 0) {
    c2 = (char *) PetscMalloc( (len2+1)*sizeof(char)); 
    PetscStrncpy(c2,value,len2);
    c2[len2] = 0;
  } else c2 = value;
  ierr = OptionsSetValue(c1,c2);
  if (c1 != name) PetscFree(c1);
  if (c2 != value) PetscFree(c2);
  return *err = ierr;
}

int  optionshasname_(char* pre,char *name,int *err,int len1,int len2){
  char *c1,*c2;
  int  ierr;
  if (!pre[len1] == 0) {
    c1 = (char *) PetscMalloc( (len1+1)*sizeof(char)); 
    PetscStrncpy(c1,pre,len1);
    c1[len1] = 0;
  } else c1 = pre;
  if (!name[len2] == 0) {
    c2 = (char *) PetscMalloc( (len2+1)*sizeof(char)); 
    PetscStrncpy(c2,name,len2);
    c2[len2] = 0;
  } else c2 = name;
  ierr = OptionsHasName(c1,c2);
  if (c1 != pre) PetscFree(c1);
  if (c2 != name) PetscFree(c2);
  return *err = ierr;
}

int  optionsgetint_(char*pre,char *name,int *ivalue,int *err,
                    int len1,int len2){
  char *c1,*c2;
  int  ierr;
  if (!pre[len1] == 0) {
    c1 = (char *) PetscMalloc( (len1+1)*sizeof(char)); 
    PetscStrncpy(c1,pre,len1);
    c1[len1] = 0;
  } else c1 = pre;
  if (!name[len2] == 0) {
    c2 = (char *) PetscMalloc( (len2+1)*sizeof(char)); 
    PetscStrncpy(c2,name,len2);
    c2[len2] = 0;
  } else c2 = name;
  ierr = OptionsGetInt(c1,c2,ivalue);
  if (c1 != pre) PetscFree(c1);
  if (c2 != name) PetscFree(c2);
  return *err = ierr;
}

int  optionsgetdouble_(char* pre,char *name,double *dvalue,int *err,
                       int len1,int len2){
  char *c1,*c2;
  int  ierr;
  if (!pre[len1] == 0) {
    c1 = (char *) PetscMalloc( (len1+1)*sizeof(char)); 
    PetscStrncpy(c1,pre,len1);
    c1[len1] = 0;
  } else c1 = pre;
  if (!name[len2] == 0) {
    c2 = (char *) PetscMalloc( (len2+1)*sizeof(char)); 
    PetscStrncpy(c2,name,len2);
    c2[len2] = 0;
  } else c2 = name;
  ierr = OptionsGetDouble(c1,c2,dvalue);
  if (c1 != pre) PetscFree(c1);
  if (c2 != name) PetscFree(c2);
  return *err = ierr;
}

int  optionsgetdoublearray_(char* pre,char *name,
                      double *dvalue,int *nmax,int *err,int len1,int len2){
  char *c1,*c2;
  int  ierr;
  if (!pre[len1] == 0) {
    c1 = (char *) PetscMalloc( (len1+1)*sizeof(char)); 
    PetscStrncpy(c1,pre,len1);
    c1[len1] = 0;
  } else c1 = pre;
  if (!name[len2] == 0) {
    c2 = (char *) PetscMalloc( (len2+1)*sizeof(char)); 
    PetscStrncpy(c2,name,len2);
    c2[len2] = 0;
  } else c2 = name;
  ierr = OptionsGetDoubleArray(c1,c2,dvalue,nmax);
  if (c1 != pre) PetscFree(c1);
  if (c2 != name) PetscFree(c2);
  return *err = ierr;
}

int  optionsgetintarray_(char* pre,char *name,int *dvalue,int *nmax,int *err,
                         int len1,int len2){
  char *c1,*c2;
  int  ierr;
  if (!pre[len1] == 0) {
    c1 = (char *) PetscMalloc( (len1+1)*sizeof(char)); 
    PetscStrncpy(c1,pre,len1);
    c1[len1] = 0;
  } else c1 = pre;
  if (!name[len2] == 0) {
    c2 = (char *) PetscMalloc( (len2+1)*sizeof(char)); 
    PetscStrncpy(c2,name,len2);
    c2[len2] = 0;
  } else c2 = name;
  ierr = OptionsGetIntArray(c1,c2,dvalue,nmax);
  if (c1 != pre) PetscFree(c1);
  if (c2 != name) PetscFree(c2);
  return *err = ierr;
}

int  optionsgetstring_(char *pre,char *name,char *string,
                       int *err, int len1, int len2,int len){
  char *c1,*c2;
  int  ierr;
  if (!pre[len1] == 0) {
    c1 = (char *) PetscMalloc( (len1+1)*sizeof(char)); 
    PetscStrncpy(c1,pre,len1);
    c1[len1] = 0;
  } else c1 = pre;
  if (!name[len2] == 0) {
    c2 = (char *) PetscMalloc( (len2+1)*sizeof(char)); 
    PetscStrncpy(c2,name,len2);
    c2[len2] = 0;
  } else c2 = name;
  ierr = OptionsGetString(c1,c2,string,len);
  if (c1 != pre) PetscFree(c1);
  if (c2 != name) PetscFree(c2);
  return *err = ierr;
}

/*
    This is code for translating PETSc memory addresses to integer offsets 
    for Fortran.
*/
       void   *PetscNull_Fortran;

int PetscIntAddressToFortran(int *base,int *addr)
{
  return (((int)addr) - (int)base)/sizeof(int);
}

int *PetscIntAddressFromFortran(int *base,int addr)
{
  return base + addr;
}

int PetscDoubleAddressToFortran(double *base,double *addr)
{
  return (((int)addr) - (int)base)/sizeof(double);
}

double *PetscDoubleAddressFromFortran(double *base,int addr)
{
  return base + addr;
}


#if defined(__cplusplus)
extern "C" {
#endif
void petscsetfortranbasepointers_(void *fnull)
{
  PetscNull_Fortran  = fnull;
}
#if defined(__cplusplus)
}
#endif
