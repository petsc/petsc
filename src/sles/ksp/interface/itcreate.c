#ifndef lint
static char vcid[] = "$Id: $";
#endif

#include "petsc.h"
#include "kspimpl.h"      /*I "ksp.h" I*/
#include <stdio.h>
#include "sys/nreg.h"
#include "sys.h"
#include "options.h"
/*@ 
    KSPView - Prints KSP datastructure.

  Input Parameters:
.  ksp - the Krylov space context
.  viewer - the location where to display context (usually 0)
@*/
int KSPView(KSP ksp,Viewer viewer)
{
  fprintf(stderr,"KSP Object\n");
  fprintf(stderr,"Max. Its. %d rtol %g atol %g\n",
          ksp->max_it,ksp->rtol,ksp->atol);
  return 0;
}
int _KSPView(PetscObject obj,Viewer viewer)
{
  return  KSPView((KSP) obj,viewer);
}
static NRList *__ITList = 0;
/*@
    KSPCreate - Creates default KSP context.

  Output Parameter:
.  ksp - location to put the Krylov Space context.

@*/
int KSPCreate(KSP *ksp)
{
  KSP ctx;
  *ksp = 0;
  CREATEHEADER(ctx,_KSP);
  *ksp               = ctx;
  ctx->cookie        = KSP_COOKIE;
  ctx->view          = _KSPView;
  ctx->prefix        = 0;

  ctx->method        = (KSPMETHOD) -1;
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
  KSPSetMethod - Builds KSP for a particular solver. Itmethod is,
  for instance, KSPCG or KSPGMRES.  

  Input Parameter:
.  ctx - the Krylov space context.
.  itmethod   - One of the known methods.  See "ksp.h" for
    available methods (for instance KSPCG or KSPGMRES).
 @*/
int KSPSetMethod(KSP ctx,KSPMETHOD itmethod)
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
   an iterative name (KSPMETHOD) and a function pointer.

   Input Parameters:
.      name - for instance KSPGMRES, ...
.      sname -  corresponding string for name
.      create - routine to create method context
@*/
int  KSPRegister(KSPMETHOD name, char *sname, int  (*create)(KSP))
{
  int ierr;
  int (*dummy)(void *) = (int (*)(void *)) create;
  if (!__ITList) {ierr = NRCreate(&__ITList); CHKERR(ierr);}
  return NRRegister( __ITList, (int) name, sname, dummy );
}

/*@
   KSPRegisterDestroy - Frees the list of iterative solvers
   registered by KSPRegister().
@*/
int KSPRegisterDestroy()
{
  if (__ITList) {
    NRDestroy( __ITList );
    __ITList = 0;
  }
  return 0;
}

/*@C
  KSPGetMethodFromOptions - Sets the selected method from the options
                            database.

  Input parameters:
. ctx - the KSP context

  Output parameter:
. kspmethod -  Iterative method type
. returns 1 if method found else 0.
@*/
int KSPGetMethodFromOptions(KSP ctx,KSPMETHOD *itmethod )
{
  char sbuf[50];
  if (OptionsGetString(0,ctx->prefix,"-kspmethod", sbuf, 50 )) {
    if (!__ITList) KSPRegisterAll();
    *itmethod = (KSPMETHOD)NRFindID( __ITList, sbuf );
    return 1;
  }
  return 0;
}

/*@C
   KSPGetMethodName - Get the name (as a string) from the method type

   Input Parameter:
.  itctx - Iterative context
@*/
int KSPGetMethodName(KSPMETHOD  itmeth,char **name )
{
  if (!__ITList) KSPRegisterAll();
  *name = NRFindName( __ITList, (int) itmeth );
  return 0;
}

#include <stdio.h>
/*@C
    KSPPrintMethods - Prints the Krylov space methods available 
              from the options database.

  Input Parameters:
.   name - the  options name (usually -kspmethod) 
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
