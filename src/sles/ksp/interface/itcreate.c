#ifndef lint
static char vcid[] = "$Id: itcreate.c,v 1.29 1995/05/02 16:55:21 curfman Exp curfman $";
#endif

#include "petsc.h"
#include "kspimpl.h"      /*I "ksp.h" I*/
#include <stdio.h>
#include "sys/nreg.h"
#include "sys.h"
#include "options.h"
#include "viewer.h"
#include "pviewer.h"

/*@ 
   KSPView - Prints the KSP data structure.

   Input Parameters:
.  ksp - the Krylov space context
.  viewer - the location to display context (usually 0)

.keywords: KSP, view
@*/
int KSPView(KSP ksp,Viewer viewer)
{
  PetscObject vobj = (PetscObject) viewer;
  FILE *fd;
  if (vobj->cookie == VIEWER_COOKIE && (vobj->type == FILE_VIEWER ||
                                        vobj->type == FILES_VIEWER)){
    fd = ViewerFileGetPointer_Private(viewer);
    fprintf(fd,"KSP Object\n");
    fprintf(fd,"Max. Its. %d rtol %g atol %g\n",
            ksp->max_it,ksp->rtol,ksp->atol);
  }
  return 0;
}
int _KSPView(PetscObject obj,Viewer viewer)
{
  return  KSPView((KSP) obj,viewer);
}
static NRList *__ITList = 0;
/*@
   KSPCreate - Creates the default KSP context.

   Output Parameter:
.  ksp - location to put the KSP context
.  comm - MPI communicator

   Notes:
   The default KSP method is GMRES with a restart of 10.

.keywords: KSP, create, context

.seealso: KSPSetUp(), KSPSolve(), KSPDestroy()
@*/
int KSPCreate(MPI_Comm comm,KSP *ksp)
{
  KSP ctx;
  *ksp = 0;
  PETSCHEADERCREATE(ctx,_KSP,KSP_COOKIE,KSPGMRES,comm);
  PLogObjectCreate(ctx);
  *ksp               = ctx;
  ctx->view          = _KSPView;
  ctx->prefix        = 0;

  ctx->type          = (KSPMethod) -1;
  ctx->max_it        = 10000;
  ctx->right_pre     = 0;
  ctx->use_pres      = 0;
  ctx->rtol          = 1.e-5;
  ctx->atol          = 1.e-50;
  ctx->divtol        = 1.e4;

  ctx->guess_zero    = 1;
  ctx->calc_eigs     = 0;
  ctx->calc_res      = 0;
  ctx->residual_history = 0;
  ctx->res_hist_size    = 0;
  ctx->res_act_size     = 0;
  ctx->usr_monitor= 0;
  ctx->adjust_work_vectors = 0;
  ctx->converged     = KSPDefaultConverged;
  ctx->BuildSolution = KSPDefaultBuildSolution;
  ctx->BuildResidual = KSPDefaultBuildResidual;

  ctx->vec_sol   = 0;
  ctx->vec_rhs   = 0;
  ctx->B         = 0;

  ctx->solver    = 0;
  ctx->setup     = 0;
  ctx->destroy   = 0;
  ctx->adjustwork= 0;

  ctx->MethodPrivate = 0;
  ctx->nwork         = 0;
  ctx->work          = 0;

  ctx->nmatop        = 0;
  ctx->namult        = 0;
  ctx->nbinv         = 0;
  ctx->nvectors      = 0;
  ctx->nscalar       = 0;

  ctx->monP          = 0;
  ctx->cnvP          = 0;

  ctx->setupcalled   = 0;
  /* this violates our rule about seperating abstract from implementations*/
  return KSPSetMethod(*ksp,KSPGMRES);
}

/*@
   KSPSetMethod - Builds KSP for a particular solver. 

   Input Parameter:
.  ctx      - the Krylov space context
.  itmethod - a known method

   Notes:  
   See "petsc/include/ksp.h" for available methods (for instance KSPCG 
   or KSPGMRES).

.keywords: KSP, set, method
@*/
int KSPSetMethod(KSP ctx,KSPMethod itmethod)
{
  int (*r)(KSP);
  VALIDHEADER(ctx,KSP_COOKIE);
  if (ctx->setupcalled) {
    SETERR(1,"Method cannot be called after KSPSetUp");
  }
  /* Get the function pointers for the iterative method requested */
  if (!__ITList) {KSPRegisterAll();}
  if (!__ITList) {
    SETERR(1,"Could not acquire list of KSP methods"); 
  }
  r =  (int (*)(KSP))NRFindRoutine( __ITList, (int)itmethod, (char *)0 );
  if (!r) {SETERR(1,"Unknown KSP method");}
  if (ctx->MethodPrivate) FREE(ctx->MethodPrivate);
  ctx->MethodPrivate = 0;
  return (*r)(ctx);
}

/*@C
   KSPRegister - Adds the iterative method to the KSP package,  given
   an iterative name (KSPMethod) and a function pointer.

   Input Parameters:
.  name   - for instance KSPCG, KSPGMRES, ...
.  sname  - corresponding string for name
.  create - routine to create method context

.keywords: KSP, register

.seealso: KSPRegisterAll(), KSPRegisterDestroy()
@*/
int  KSPRegister(KSPMethod name, char *sname, int  (*create)(KSP))
{
  int ierr;
  int (*dummy)(void *) = (int (*)(void *)) create;
  if (!__ITList) {ierr = NRCreate(&__ITList); CHKERR(ierr);}
  return NRRegister( __ITList, (int) name, sname, dummy );
}

/*@
   KSPRegisterDestroy - Frees the list of iterative solvers that were
   registered by KSPRegister().

.keywords: KSP, register, destroy

.seealso: KSPRegister(), KSPRegisterAll()
@*/
int KSPRegisterDestroy()
{
  if (__ITList) {
    NRDestroy( __ITList );
    __ITList = 0;
  }
  return 0;
}


/*
   KSPGetMethodFromOptions - Sets the selected KSP method from the options
   database.

   Input Parameter:
.  ctx - the KSP context

   Output Parameter:
.  itmethod - iterative method

   Returns:
   Returns 1 if the method is found; 0 otherwise.

   Options Database Key:
$  -ksp_method  itmethod
*/
int KSPGetMethodFromOptions(KSP ctx,KSPMethod *itmethod)
{
  char sbuf[50];
  if (OptionsGetString(0,ctx->prefix,"-ksp_method", sbuf, 50 )) {
    if (!__ITList) KSPRegisterAll();
    *itmethod = (KSPMethod)NRFindID( __ITList, sbuf );
    return 1;
  }
  return 0;
}

/*@C
   KSPGetMethodName - Gets the KSP method name (as a string) from 
   the method type.

   Input Parameter:
.  itmeth - KSP method

   Output Parameter:
.  name - name of KSP method

.keywords: KSP, get, method, name
@*/
int KSPGetMethodName(KSPMethod  itmeth,char **name )
{
  if (!__ITList) KSPRegisterAll();
  *name = NRFindName( __ITList, (int) itmeth );
  return 0;
}

#include <stdio.h>
/*@C
   KSPPrintMethods - Prints the KSP methods available from the options 
   database.

   Input Parameters:
.  prefix - prefix (usually "-")
.  name - the options database name (by default "kspmethod") 

.keywords: KSP, print, methods, options, database

.seealso: KSPPrintHelp()
@*/
int KSPPrintMethods(char* prefix,char *name)
{
  FuncList *entry;
  if (!__ITList) {KSPRegisterAll();}
  entry = __ITList->head;
  fprintf(stderr," %s%s (one of)",prefix,name);
  while (entry) {
    fprintf(stderr," %s",entry->name);
    entry = entry->next;
  }
  fprintf(stderr,"\n");
  return 1;
}
