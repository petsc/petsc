#ifndef lint
static char vcid[] = "$Id: itcreate.c,v 1.41 1995/07/08 15:06:48 bsmith Exp curfman $";
#endif

#include "petsc.h"
#include "kspimpl.h"      /*I "ksp.h" I*/
#include <stdio.h>
#include "sys/nreg.h"     /*I "sys/nreg.h I*/
#include "sys.h"
#include "viewer.h"       /*I "viewer.h" I*/
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
  char *method;
  if (vobj->cookie == VIEWER_COOKIE && (vobj->type == FILE_VIEWER ||
                                        vobj->type == FILES_VIEWER)){
    fd = ViewerFileGetPointer_Private(viewer);
    MPIU_fprintf(ksp->comm,fd,"KSP Object:\n");
    KSPGetMethodName((KSPMethod)ksp->type,&method);
    MPIU_fprintf(ksp->comm,fd,"  method: %s\n",method);
    if (ksp->guess_zero) MPIU_fprintf(ksp->comm,fd,
      "  maximum iterations=%d\n, initial guess is zero",ksp->max_it);
    else MPIU_fprintf(ksp->comm,fd,"  maximum iterations=%d\n", ksp->max_it);
    fprintf(fd,"  tolerances:  relative=%g, absolute=%g, divergence=%g\n",
            ksp->rtol, ksp->atol, ksp->divtol);
    if (ksp->right_pre) MPIU_fprintf(ksp->comm,fd,"  right preconditioning\n");
    else MPIU_fprintf(ksp->comm,fd,"  left preconditioning\n");
  }
  return 0;
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
  ctx->view          = 0;
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
  ctx->buildsolution = KSPDefaultBuildSolution;
  ctx->buildresidual = KSPDefaultBuildResidual;

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

   Options Database Command:
$  -ksp_method  <method>
$      Use -help for a list of available methods
$      (for instance, cg or gmres)

   Notes:  
   See "petsc/include/ksp.h" for available methods (for instance KSPCG 
   or KSPGMRES).

.keywords: KSP, set, method
@*/
int KSPSetMethod(KSP ctx,KSPMethod itmethod)
{
  int ierr,(*r)(KSP);
  VALIDHEADER(ctx,KSP_COOKIE);
  if (ctx->setupcalled) {
    /* destroy the old private KSP context */
    ierr = (*(ctx)->destroy)((PetscObject)ctx); CHKERRQ(ierr);
    ctx->MethodPrivate = 0;
  }
  /* Get the function pointers for the iterative method requested */
  if (!__ITList) {KSPRegisterAll();}
  if (!__ITList) {
    SETERRQ(1,"KSPSetMethod: Could not acquire list of KSP methods"); 
  }
  r =  (int (*)(KSP))NRFindRoutine( __ITList, (int)itmethod, (char *)0 );
  if (!r) {SETERRQ(1,"KSPSetMethod: Unknown KSP method");}
  if (ctx->MethodPrivate) PETSCFREE(ctx->MethodPrivate);
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
  if (!__ITList) {ierr = NRCreate(&__ITList); CHKERRQ(ierr);}
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
   KSPGetMethodFromOptions_Private - Sets the selected KSP method from 
   the options database.

   Input Parameter:
.  ctx - the KSP context

   Output Parameter:
.  itmethod - iterative method

   Returns:
   Returns 1 if the method is found; 0 otherwise.
*/
int KSPGetMethodFromOptions_Private(KSP ctx,KSPMethod *itmethod)
{
  char sbuf[50];
  if (OptionsGetString(ctx->prefix,"-ksp_method", sbuf, 50 )) {
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
/*
   KSPPrintMethods_Private - Prints the KSP methods available from the options 
   database.

   Input Parameters:
.  prefix - prefix (usually "-")
.  name - the options database name (by default "ksp_method") 
*/
int KSPPrintMethods_Private(char* prefix,char *name)
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
