
#if !defined(__PETSC_PACKAGE)
#define __PETSC_PACKAGE

/*  Macros for getting and freeing memory */
#define NEW(a)          (a *) MALLOC(sizeof(a))
#define MALLOC(a)       malloc(a)
#define FREE(a)         free(a)
#define MEMCPY(a,b,n)   memcpy((char*)(a),(char*)(b),n)

/*  Macros for error checking */
#define SETERR(n,s)     {return PetscErrorHandler(__LINE__,__FILE__,s,n);}
#define CHKERR(n)       {if (n) SETERR(n,(char *)0);}
#define CHKPTR(p)       if (!p) SETERR(1,"No memory");


typedef struct _PetscObject PetscObject;

/*  Useful Petsc functions */

#ifdef ANSI_ARG
#undef ANSI_ARG
#endif
#ifdef __STDC__
#define ANSI_ARGS(a) a
#else
#define ANSI_ARGS(a) ()
#endif

extern int PetscDestroy       ANSI_ARGS((PetscObject));
extern int PetscErrorHandler  ANSI_ARGS((int,char*,char*,int));

#endif
