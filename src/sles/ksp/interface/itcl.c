/*
    Command line interface for KSP
*/

#include "petsc.h"
#include "kspimpl.h"
#include "sys.h"
#include "options.h"
/*@
    KSPSetFromOptions - Sets KSP options from the options database.
                            This must be called before KSPSetUp()
                            if the user is to be allowed to set the 
                            Krylov method. 

  Input Parameters:
.  ctx - the Krylov space context
   
   See also: KSPPrintHelp()
@*/
int KSPSetFromOptions(KSP ctx)
{
  char      string[50];
  KSPMETHOD method;
  VALIDHEADER(ctx,KSP_COOKIE);

  if (OptionsHasName(0,"-help")) {
    KSPPrintHelp(ctx);
  }
  if (KSPGetMethodFromOptions(0,ctx->namemethod,&method)) {
    KSPSetMethod(ctx,method);
  }
  OptionsGetInt(0,ctx->namemax_it,&ctx->max_it);
  OptionsGetDouble(0,ctx->namertol,&ctx->rtol);  
  OptionsGetDouble(0,ctx->nameatol,&ctx->atol);
  OptionsGetDouble(0,ctx->namedivtol,&ctx->divtol);
  if (OptionsHasName(0,"-kspmonitor")){
    KSPSetMonitor(ctx,KSPDefaultMonitor,(void *)0);
  }
  return 0;
}
  
/*@ 
    KSPPrintHelp - Prints all the  options for the KSP component.

  Input Parameters:
.  ctx - the KSP context

@*/
int KSPPrintHelp(KSP ctx)
{
  VALIDHEADER(ctx,KSP_COOKIE);
  KSPPrintMethods(ctx->namemethod);
  fprintf(stderr," %s (relative tolerance: defaults to %g)\n",
                 ctx->namertol,ctx->rtol);
  fprintf(stderr," %s (absolute tolerance: defaults to %g)\n",
                 ctx->nameatol,ctx->atol);
  fprintf(stderr," %s (divergence tolerance: defaults to %g)\n",
                 ctx->namedivtol,ctx->divtol);
  fprintf(stderr," %s (maximum iterations: defaults to %d)\n",
                 ctx->namemax_it,ctx->max_it);
  fprintf(stderr," -kspmonitor: use residual convergence monitor\n");
  return 1;
}

