/* adebug.c */
/* Fortran interface file */

#ifdef POINTER_64_BITS
extern void *__ToPointer();
extern int __FromPointer();
extern void __RmPointer();
#else
#define __ToPointer(a) (a)
#define __FromPointer(a) (int)(a)
#define __RmPointer(a)
#endif

#include "petsc.h"
#ifdef FORTRANCAPS
#define petscsetdebugger_ PETSCSETDEBUGGER
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define petscsetdebugger_ petscsetdebugger
#endif
void petscsetdebugger_(char *debugger,int *xterm,char *display, int *__ierr ){
*__ierr = PetscSetDebugger(debugger,*xterm,display);
}
#ifdef FORTRANCAPS
#define petscattachdebugger_ PETSCATTACHDEBUGGER
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define petscattachdebugger_ petscattachdebugger
#endif
void petscattachdebugger_(){
*__ierr = PetscAttachDebugger();
}
#ifdef FORTRANCAPS
#define petscattachdebuggererrorhandler_ PETSCATTACHDEBUGGERERRORHANDLER
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define petscattachdebuggererrorhandler_ petscattachdebuggererrorhandler
#endif
void petscattachdebuggererrorhandler_(int *line,char* dir,char* file,char* mess,
                                    int *num,void *ctx, int *__ierr ){
*__ierr = PetscAttachDebuggerErrorHandler(*line,dir,file,mess,*num,ctx);
}
