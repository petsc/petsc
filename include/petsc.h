
#if !defined(__PETSC_PACKAGE)
#define __PETSC_PACKAGE

#include <stdio.h>

/* MPI interface */
#include "mpi.h"
#include "mpe.h"
#if defined(PETSC_COMPLEX)
#define MPI_SCALAR MPIR_dcomplex_dte
#else
#define MPI_SCALAR MPI_DOUBLE
#endif
extern FILE *MPE_fopen(MPI_Comm,char *,char *);
extern int MPE_fclose(MPI_Comm,FILE*);
extern int MPE_fprintf(MPI_Comm,FILE*,char *,...);
extern int MPE_printf(MPI_Comm,char *,...);
extern int MPE_Set_display(MPI_Comm,char **);


#if defined(PETSC_COMPLEX)
/* work around for bug in alpha g++ compiler */
#if defined(PARCH_alpha) 
#define hypot(a,b) (double) sqrt((a)*(a)+(b)*(b)) 
/* extern double hypot(double,double); */
#endif
#include <complex.h>
#define PETSCREAL(a) real(a)
#define Scalar       complex
#else
#define PETSCREAL(a) a
#define Scalar       double
#endif


/*  Macros for getting and freeing memory */
#if defined(PETSC_MALLOC)
#define MALLOC(a)       trmalloc(a,__LINE__,__FILE__)
#define FREE(a)         trfree(a,__LINE__,__FILE__)
#else
#define MALLOC(a)       malloc(a)
#define FREE(a)         free(a)
#endif
#define NEW(a)          (a *) MALLOC(sizeof(a))
#define MEMCPY(a,b,n)   memcpy((char*)(a),(char*)(b),n)
#define MEMSET(a,b,n)   memset((char*)(a),(int)(b),n)
#include <memory.h>

/*  Macros for error checking */
#if !defined(__DIR__)
#define __DIR__ 0
#endif
#define SETERR(n,s)     {return PetscError(__LINE__,__DIR__,__FILE__,s,n);}
#define SETERRA(n,s)    \
                {int _ierr = PetscError(__LINE__,__DIR__,__FILE__,s,n);\
                 MPI_Abort(MPI_COMM_WORLD,_ierr);}
#define CHKERR(n)       {if (n) SETERR(n,(char *)0);}
#define CHKERRA(n)      {if (n) SETERRA(n,(char *)0);}
#define CHKPTR(p)       if (!p) SETERR(1,"No memory");
#define CHKPTRA(p)      if (!p) SETERRA(1,"No memory");


typedef struct _PetscObject* PetscObject;

typedef struct _Viewer*      Viewer;
#define ViewerPrintf         (void *) 0
#define VIEWER_COOKIE        0x123123
#define MATLAB_VIEWER        1

/* useful Petsc routines (used often) */
extern int  PetscInitialize(int*,char***,char*,char*);
extern int  PetscFinalize();

extern int  PetscDestroy(PetscObject);
extern int  PetscView(PetscObject,Viewer);

extern int  PetscDefaultErrorHandler(int,char*,char*,char*,int,void*);
extern int  PetscAbortErrorHandler(int,char*,char*,char*,int,void* );
extern int  PetscAttachDebuggerErrorHandler(int,char*,char*,char*,int,void*); 
extern int  PetscError(int,char*,char*,char*,int);
extern int  PetscPushErrorHandler(int 
                         (*handler)(int,char*,char*,char*,int,void*),void* );
extern int  PetscPopErrorHandler();

extern int  PetscSetDebugger(char *,int,char *);
extern int  PetscAttachDebugger();

extern int PetscDefaultSignalHandler(int,void*);
extern int PetscPushSignalHandler(int (*)(int,void *),void*);
extern int PetscPopSignalHandler();
extern int PetscSetFPTrap(int);

#if defined(PETSC_MALLOC)
extern void *trmalloc(unsigned int,int,char*);
extern int  trfree(void *,int,char*);
extern int  trdump(FILE *);
#else
#include <malloc.h>
#endif

#if defined(PARCH_cray) || defined(PARCH_NCUBE)
#define FORTRANCAPS
#elif !defined(PARCH_rs6000) && !defined(PACRH_NeXT) && !defined(PACRH_HPUX)
#define FORTRANUNDERSCORE
#endif

#include <stdio.h> /* I don't like this, but? */

#endif
