
#if !defined(__PETSC_PACKAGE)
#define __PETSC_PACKAGE

#if PETSC_COMPLEX
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
extern int PetscSetSignalHandler(void (*)(int,int,struct sigcontext *,char*));

extern void *trmalloc(unsigned int,int,char*);
extern int  trfree(void *,int,char*);
#include <stdio.h> /* I don't like this, but? */
extern int  trdump(FILE *);

#if defined(titan) || defined(cray) || defined(ncube)
#define FORTRANCAPS
#elif !defined(rs6000) && !defined(NeXT) && !defined(HPUX)
#define FORTRANUNDERSCORE
#endif

#endif
