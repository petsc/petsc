
#ifndef lint
static char vcid[] = "$Id: zoptions.c,v 1.1 1995/08/21 19:56:20 bsmith Exp bsmith $";
#endif
/*
    Fortran stub for PetscInitialize and Options routines. 
  These are not done automatically since they require passing 
  strings between Fortran and C.


*/
#include "zpetsc.h"
#include "petsc.h"
#if defined(HAVE_STRING_H)
#include <string.h>
#endif
#include <stdio.h>
#include "pinclude/pviewer.h"
#include "pinclude/petscfix.h"
extern int          PetscBeganMPI;
#if defined(PETSC_COMPLEX)
extern MPI_Datatype  MPIU_COMPLEX;
#endif

#ifdef FORTRANCAPS
#define optionssetvalue_       OPTIONSSETVALUE
#define optionshasname_        OPTIONSHASNAME
#define optionsgetint_         OPTIONSGETINT
#define optionsgetdouble_      OPTIONSGETDOUBLE
#define optionsgetdoublearray_ OPTIONSGETDOUBLEARRAY
#define petscfinalize_         PETSCFINALIZE
#define petscsetcommonblock_   PETSCSETCOMMONBLOCK
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define optionssetvalue_       optionssetvalue
#define optionshasname_        optionshasname
#define optionsgetint_         optionsgetint
#define optionsgetdouble_      optionsgetdouble
#define optionsgetdoublearray_ optionsgetdoublearray
#define petscfinalize_         petscfinalize
#define petscsetcommonblock_   petscsetcommonblock
#endif

int OptionsCheckInitial_Private(),
    OptionsCreate_Private(int*,char***,char*,char*),
    OptionsSetAlias_Private(char *,char *);

/*
     This first block is probably only due to using MPI-CH
   I DO NOT understand why they put a __ in. It seems nuts
   and TOTALLY unneeded!
*/
#if defined(PARCH_freebsd)
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
};
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
};
#endif
int PETScParseFortranArgs_Private(int *argc,char ***argv)
{
  int  i, warg = 256,mytid;
  char *p;
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  if (!mytid) {
    *argc = 1 + iargc_();
  }
  MPI_Bcast(argc,1,MPI_INT,0,MPI_COMM_WORLD);

  *argv = (char **) PETSCMALLOC((*argc+1)*(warg*sizeof(char)+sizeof(char*))); 
  CHKPTRQ(*argv);
  (*argv)[0] = (char*) (*argv + *argc + 1);

  if (!mytid) {
    PETSCMEMSET((*argv)[0],0,(*argc)*warg*sizeof(char));
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
  if (mytid) {
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
  PETSCFREE(args);
  ierr = OptionsCheckInitial_Private(); CHKERRQ(ierr);
  ierr = ViewerInitialize_Private(); CHKERRQ(ierr);

  s1 = MPIR_FromPointer(STDOUT_VIEWER_SELF);
  s2 = MPIR_FromPointer(STDERR_VIEWER_SELF);
  s3 = MPIR_FromPointer(STDOUT_VIEWER_WORLD);
  petscsetcommonblock_(&s1,&s2,&s3);
  if (PetscBeganMPI) {
    int mytid,numtid;
    MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
    MPI_Comm_size(MPI_COMM_WORLD,&numtid);
    PLogInfo(0,"[%d] PETSc successfully started: procs %d\n",mytid,numtid);
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
    c1 = (char *) PETSCMALLOC( (len1+1)*sizeof(char)); 
    strncpy(c1,name,len1);
    c1[len1] = 0;
  } else c1 = name;
  if (!value[len2] == 0) {
    c2 = (char *) PETSCMALLOC( (len2+1)*sizeof(char)); 
    strncpy(c2,value,len2);
    c2[len2] = 0;
  } else c2 = value;
  ierr = OptionsSetValue(c1,c2);
  if (c1 != name) PETSCFREE(c1);
  if (c2 != value) PETSCFREE(c2);
  return *err = ierr;
}

int  optionshasname_(char* pre,char *name,int *err,int len1,int len2){
  char *c1,*c2;
  int  ierr;
  if (!pre[len1] == 0) {
    c1 = (char *) PETSCMALLOC( (len1+1)*sizeof(char)); 
    strncpy(c1,pre,len1);
    c1[len1] = 0;
  } else c1 = pre;
  if (!name[len2] == 0) {
    c2 = (char *) PETSCMALLOC( (len2+1)*sizeof(char)); 
    strncpy(c2,name,len2);
    c2[len2] = 0;
  } else c2 = name;
  ierr = OptionsHasName(c1,c2);
  if (c1 != pre) PETSCFREE(c1);
  if (c2 != name) PETSCFREE(c2);
  return *err = ierr;
}

int  optionsgetint_(char*pre,char *name,int *ivalue,int *err,
                    int len1,int len2){
  char *c1,*c2;
  int  ierr;
  if (!pre[len1] == 0) {
    c1 = (char *) PETSCMALLOC( (len1+1)*sizeof(char)); 
    strncpy(c1,pre,len1);
    c1[len1] = 0;
  } else c1 = pre;
  if (!name[len2] == 0) {
    c2 = (char *) PETSCMALLOC( (len2+1)*sizeof(char)); 
    strncpy(c2,name,len2);
    c2[len2] = 0;
  } else c2 = name;
  ierr = OptionsGetInt(c1,c2,ivalue);
  if (c1 != pre) PETSCFREE(c1);
  if (c2 != name) PETSCFREE(c2);
  return *err = ierr;
}

int  optionsgetdouble_(char* pre,char *name,double *dvalue,int *err,
                       int len1,int len2){
  char *c1,*c2;
  int  ierr;
  if (!pre[len1] == 0) {
    c1 = (char *) PETSCMALLOC( (len1+1)*sizeof(char)); 
    strncpy(c1,pre,len1);
    c1[len1] = 0;
  } else c1 = pre;
  if (!name[len2] == 0) {
    c2 = (char *) PETSCMALLOC( (len2+1)*sizeof(char)); 
    strncpy(c2,name,len2);
    c2[len2] = 0;
  } else c2 = name;
  ierr = OptionsGetDouble(c1,c2,dvalue);
  if (c1 != pre) PETSCFREE(c1);
  if (c2 != name) PETSCFREE(c2);
  return *err = ierr;
}

int  optionsgetdoublearray_(char* pre,char *name,
                      double *dvalue,int *nmax,int *err,int len1,int len2){
  char *c1,*c2;
  int  ierr;
  if (!pre[len1] == 0) {
    c1 = (char *) PETSCMALLOC( (len1+1)*sizeof(char)); 
    strncpy(c1,pre,len1);
    c1[len1] = 0;
  } else c1 = pre;
  if (!name[len2] == 0) {
    c2 = (char *) PETSCMALLOC( (len2+1)*sizeof(char)); 
    strncpy(c2,name,len2);
    c2[len2] = 0;
  } else c2 = name;
  ierr = OptionsGetDoubleArray(c1,c2,dvalue,nmax);
  if (c1 != pre) PETSCFREE(c1);
  if (c2 != name) PETSCFREE(c2);
  return *err = ierr;
}
#ifdef FORTRANCAPS
#define optionsgetintarray_ OPTIONSGETINTARRAY
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define optionsgetintarray_ optionsgetintarray
#endif
int  optionsgetintarray_(char* pre,char *name,int *dvalue,int *nmax,int *err,
                         int len1,int len2){
  char *c1,*c2;
  int  ierr;
  if (!pre[len1] == 0) {
    c1 = (char *) PETSCMALLOC( (len1+1)*sizeof(char)); 
    strncpy(c1,pre,len1);
    c1[len1] = 0;
  } else c1 = pre;
  if (!name[len2] == 0) {
    c2 = (char *) PETSCMALLOC( (len2+1)*sizeof(char)); 
    strncpy(c2,name,len2);
    c2[len2] = 0;
  } else c2 = name;
  ierr = OptionsGetIntArray(c1,c2,dvalue,nmax);
  if (c1 != pre) PETSCFREE(c1);
  if (c2 != name) PETSCFREE(c2);
  return *err = ierr;
}
#ifdef FORTRANCAPS
#define optionsgetstring_ OPTIONSGETSTRING
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define optionsgetstring_ optionsgetstring
#endif
int  optionsgetstring_(char *pre,char *name,char *string,
                       int *err, int len1, int len2,int len){
  char *c1,*c2;
  int  ierr;
  if (!pre[len1] == 0) {
    c1 = (char *) PETSCMALLOC( (len1+1)*sizeof(char)); 
    strncpy(c1,pre,len1);
    c1[len1] = 0;
  } else c1 = pre;
  if (!name[len2] == 0) {
    c2 = (char *) PETSCMALLOC( (len2+1)*sizeof(char)); 
    strncpy(c2,name,len2);
    c2[len2] = 0;
  } else c2 = name;
  ierr = OptionsGetString(c1,c2,string,len);
  if (c1 != pre) PETSCFREE(c1);
  if (c2 != name) PETSCFREE(c2);
  return *err = ierr;
}
