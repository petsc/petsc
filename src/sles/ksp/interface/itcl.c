/*
    Command line interface for KSP
*/

#include "petsc.h"
#include "kspimpl.h"
#include "sys.h"
/*@
    KSPSetFromCommandLine - sets KSP options from the command line.
                            This must be called before KSPSetUp()
                            if the user is to be allowed to set the 
                            Krylov method. 
  Input Parameters:
.  argv,argc - the command line arguments
.  options - the options context
   
   See also: KSPPrintHelpFromCommandLine()
@*/
int KSPSetFromCommandLine(KSP ctx,int* argc,char **argv)
{
  char      string[50];
  KSPMETHOD method;
  VALIDHEADER(ctx,KSP_COOKIE);

  if (KSPGetMethodFromCommandLine(argc,argv,0,ctx->namemethod,&method)) {
    KSPSetMethod(ctx,method);
  }
  SYArgGetInt(argc,argv,0,ctx->namemax_it,&ctx->max_it);
  SYArgGetDouble(argc,argv,0,ctx->namertol,&ctx->rtol);  
  SYArgGetDouble(argc,argv,0,ctx->nameatol,&ctx->atol);
  SYArgGetDouble(argc,argv,0,ctx->namedivtol,&ctx->divtol);
  return 0;
}
  
/*@ 
    KSPPrintHelpFromCommandLine - prints all the command line options
                              for the KSP component.

  Input Parameters:
.  argc, argv - the command line arguments

@*/
int KSPPrintHelpFromCommandLine(KSP ctx,int *argc,char **argv)
{
  VALIDHEADER(ctx,KSP_COOKIE);
  if (!SYArgHasName(argc,argv,0,"-help")) return 0;
  KSPPrintMethods(ctx->namemethod);
  fprintf(stderr," %s (relative tolerance: defaults to %g)\n",
                 ctx->namertol,ctx->rtol);
  fprintf(stderr," %s (absolute tolerance: defaults to %g)\n",
                 ctx->nameatol,ctx->atol);
  fprintf(stderr," %s (divergence tolerance: defaults to %g)\n",
                 ctx->namedivtol,ctx->divtol);
  fprintf(stderr," %s (maximum iterations: defaults to %d)\n",
                 ctx->namemax_it,ctx->max_it);
  return 1;
}

