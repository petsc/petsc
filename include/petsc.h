
#if !defined(__PETSC_PACKAGE)
#define __PETSC_PACKAGE

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
#define SETERR(n,s)     {return PetscError(__LINE__,__FILE__,s,n);}
#define CHKERR(n)       {if (n) SETERR(n,(char *)0);}
#define CHKPTR(p)       if (!p) SETERR(1,"No memory");


typedef struct _PetscObject* PetscObject;
typedef struct _Viewer*      Viewer;

/* useful Petsc routines (used often) */
extern int  PetscInitialize(int*,char***,char*,char*);
extern int  PetscFinalize();

extern int  PetscDestroy(PetscObject);
extern int  PetscView(PetscObject,Viewer);

extern int  PetscError(int,char*,char*,int);
extern int  PetscPushErrorHandler(int (*handler)(int,char*,char*,int,void*),void* );
extern int  PetscPopErrorHandler();

extern int  PetscDefaultErrorHandler(int,char*,char*,int,void*);
extern int  PetscAbortErrorHandler(int,char*,char*,int,void* );
extern int  PetscAttachDebuggerErrorHandler(int, char *,char *,int,void*); 

extern int  PetscSetDebugger(char *,int,char *);
extern int  PetscAttachDebugger();

#include <signal.h> /* I don't like this, but? */
#if !defined(PARCH_rs6000) && !defined(PARCH_freebsd) && !defined(PARCH_alpha)
extern int PetscSetSignalHandler(void (*)(int,int,struct sigcontext *,char*));
#else
extern int PetscSetSignalHandler(void (*)(int));
#endif

extern void *trmalloc(unsigned int,int,char*);
extern int  trfree(void *,int,char*);
#include <stdio.h> /* I don't like this, but? */
extern int  trdump(FILE *);

#if defined(PARCH_cray) || defined(PARCH_NCUBE)
#define FORTRANCAPS
#elif !defined(PARCH_rs6000) && !defined(PACRH_NeXT) && !defined(PACRH_HPUX)
#define FORTRANUNDERSCORE
#endif

#endif
