#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ec.c,v 1.8 1997/08/22 17:49:15 curfman Exp curfman $";
#endif

/*
   This is where the eigenvalue computation operations are defined
*/

#include "src/ec/ecimpl.h"        /*I "ec.h" I*/
#include "pinclude/pviewer.h"


#undef __FUNC__  
#define __FUNC__ "ECDestroy" 
/*@C
   ECDestroy - Destroys EC context that was created with ECCreate().

   Input Parameter:
.  ec - the eigenvalue computation context

.keywords: EC, destroy

.seealso: ECCreate(), ECSetUp()
@*/
int ECDestroy(EC ec)
{
  int ierr = 0;
  PetscValidHeaderSpecific(ec,EC_COOKIE);
  if (ec->destroy) ierr =  (*ec->destroy)((PetscObject)ec);
  else {if (ec->data) PetscFree(ec->data);}
  PLogObjectDestroy(ec);
  PetscHeaderDestroy(ec);
  return ierr;
}

#undef __FUNC__  
#define __FUNC__ "ECView" 
/*@ 
   ECView - Prints the EC data structure.

   Input Parameters:
.  EC - the eigenvalue computation context
.  viewer - visualization context

   Note:
   The available visualization contexts include
$     VIEWER_STDOUT_SELF - standard output (default)
$     VIEWER_STDOUT_WORLD - synchronized standard
$       output where only the first processor opens
$       the file.  All other processors send their 
$       data to the first processor to print. 

   The user can open alternative vistualization contexts with
$    ViewerFileOpenASCII() - output to a specified file

.keywords: EC, view

.seealso: ECView(), ViewerFileOpenASCII(), ECView()
@*/
int ECView(EC ec,Viewer viewer)
{
  FILE        *fd;
  char        *method;
  int         ierr;
  ViewerType  vtype;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    PetscFPrintf(ec->comm,fd,"EC Object:\n");
    ECGetType(ec,PETSC_NULL,&method);
    PetscFPrintf(ec->comm,fd,"  method: %s\n",method);
    if (ec->view) (*ec->view)((PetscObject)ec,viewer);
  } else if (vtype == STRING_VIEWER) {
    ECType type;
    ECGetType(ec,&type,&method);
    ierr = ViewerStringSPrintf(viewer," %-7.7s",method); CHKERRQ(ierr);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ECGetEigenvalues"
/*@
   ECGetEigenvalues - Returns pointers to the real and 
   imaginary components of the eigenvalues computed 
   with an EC context.

   Input Parameters:
.  ec - the eigenvalue computation context

   Output Parameters:
.  n - number of eigenvalues computed
.  rpart - array containing the real parts of the eigenvalues
.  ipart - array containing the imaginary parts of the eigenvalues

   Notes:
   ECGetEigenvalues() may be called only after ECSolve().

.keywords: EC, setup

.seealso: ECCreate(), ECSolve(), ECDestroy()
@*/
int ECGetEigenvalues(EC ec,int *n,double **rpart,double **ipart)
{
  PetscValidHeaderSpecific(ec,EC_COOKIE);

  *n     = ec->n;
  *rpart = ec->realpart;
  *ipart = ec->imagpart;
  return 0;
}


#undef __FUNC__  
#define __FUNC__ "ECGetEigenvectors"
/*@
   ECGetEigenvectors - Returns pointers to the eigenvectors
   computed  with an EC context.

   Input Parameters:
.  ec - the eigenvalue computation context

   Output Parameters:
.  n - number of eigenvectors computed
.  evecs - the eigenvectors

   Notes:
   ECGetEigenvectors() may be called only after ECSolveEigenvectors().

.keywords: EC, get, eigenvectors

.seealso: ECCreate(), ECSolve(), ECDestroy(), ECSolveEigenvectors()
@*/
int ECGetEigenvectors(EC ec,int *n,Vec *evecs)
{
  PetscValidHeaderSpecific(ec,EC_COOKIE);

  *n     = ec->n;
  *evecs = ec->evecs;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ECSetUp"
/*@
   ECSetUp - Prepares for the use of an eigenvalue solver.

   Input parameters:
.  ec - the eigenvalue computation context

.keywords: EC, setup

.seealso: ECCreate(), ECSolve(), ECDestroy()
@*/
int ECSetUp(EC ec)
{
  int ierr;
  PetscValidHeaderSpecific(ec,EC_COOKIE);

  if (ec->setupcalled > 0) return 0;
  PLogEventBegin(EC_SetUp,ec,0,0,0); 
  if (!ec->A) {SETERRQ(1,0,"Matrix must be set first");}
  if (ec->setup) { ierr = (*ec->setup)(ec); CHKERRQ(ierr);}
  ec->setupcalled = 1;
  PLogEventEnd(EC_SetUp,ec,0,0,0); 
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ECSetEigenvectorsRequired"
/*@C
   ECSetEigenvectorsRequired - Indicates that both eigenvalues and
   eigenvectors should be computed.

   Output Parameter:
.  ec - eigenvalue computation context

.keywords: EC, create, context, eigenvectors

.seealso: ECSetUp(), ECSolve(), ECDestroy(), ECSolveEigenvectors(),
          ECGetEigenvectors()
@*/
int ECSetEigenvectorsRequired(EC ec)
{
  PetscValidHeaderSpecific(ec,EC_COOKIE);
  ec->computeeigenvectors = 1;
  return 0;
}

#include "src/sys/nreg.h"
static NRList *__ECList = 0;
#undef __FUNC__  
#define __FUNC__ "ECCreate"
/*@C
   ECCreate - Creates the default EC context.

   Output Parameter:
.  ec - location to put the EC context
.  pt - either EC_EIGENVALUE or EC_GENERALIZED_EIGENVALUE
.  comm - MPI communicator

   Notes:
   The default EC type is ?????

.keywords: EC, create, context

.seealso: ECSetUp(), ECSolve(), ECDestroy(), ECRegister()
@*/
int ECCreate(MPI_Comm comm,ECProblemType pt,EC *ec)
{
  EC ctx;

  *ec = 0;
  PetscHeaderCreate(ctx,_p_EC,EC_COOKIE,EC_LAPACK,comm,ECDestroy,ECView);
  PLogObjectCreate(ctx);
  *ec                = ctx;
  ctx->view          = 0;
  ctx->type          = (ECType) -1;
  ctx->problemtype   = pt;
  ctx->solve         = 0;
  ctx->setup         = 0;
  ctx->destroy       = 0;

  ctx->computeeigenvectors = 0;

  ctx->data          = 0;
  ctx->nwork         = 0;
  ctx->work          = 0;

  ctx->n             = 0;
  ctx->realpart      = 0;
  ctx->cnvP          = 0;

  ctx->setupcalled   = 0;
  /* this violates our rule about separating abstract from implementations */
  return ECSetType(*ec,EC_LAPACK);
}

#undef __FUNC__  
#define __FUNC__ "ECSetFromOptions"
/*@
   ECSetFromOptions - Set EC options from the options database.

   Input Parameter:
.   ec - the eigenvalue computation context

   Options Database Commands:
$  -ec_type  <method>
$      Use -help for a list of available methods
$      (for instance, LAPACK, ???)
$  -ec_spectrum_portion <largest_real_part,largest_magnitude, 
$                        smallest_real_part,smallest_magnitude,
$                        interior>
$  -ec_spectrum_number number of eigenvalues requested

.keywords: EC, set, method

.seealso: ECCreate(), ECSetType()
@*/
int ECSetFromOptions(EC ec)
{
  int  ierr,flag;
  char spectrum[128];

  PetscValidHeaderSpecific(ec,EC_COOKIE);

  ierr = OptionsGetString(ec->prefix,"-ec_spectrum_portion",spectrum,128,&flag);CHKERRQ(ierr);
  if (flag) {
    if (!PetscStrcmp(spectrum,"largest_real_part")) {
      PLogInfo(ec,"Computing largest real part of spectrum");
      ec->spectrumportion = EC_LARGEST_REAL_PART;
    } else if (!PetscStrcmp(spectrum,"largest_magnitude")) {
      PLogInfo(ec,"Computing largest magnitude of spectrum");
      ec->spectrumportion = EC_LARGEST_MAGNITUDE;
    } else if (!PetscStrcmp(spectrum,"smallest_real_part")) {
      PLogInfo(ec,"Computing smallest real part of spectrum");
      ec->spectrumportion = EC_SMALLEST_REAL_PART;
    } else if (!PetscStrcmp(spectrum,"smallest_magnitude")) {
      PLogInfo(ec,"Computing smallest magnitude of spectrum");
      ec->spectrumportion = EC_SMALLEST_MAGNITUDE;
    } else if (!PetscStrcmp(spectrum,"interior")) {
      PLogInfo(ec,"Computing interior spectrum");
      ec->spectrumportion = EC_INTERIOR;
      ierr = OptionsGetScalar(ec->prefix,"-ec_spectrum_location",&ec->location,&flag);
             CHKERRQ(ierr);
      if (!flag) SETERRQ(1,1,"Must set interior spectrum location");   
    } else {
      SETERRQ(1,1,"Unknown spectrum request");
    }
  }
  ierr = OptionsGetInt(ec->prefix,"-ec_spectrum_number",&ec->n,&flag);CHKERRQ(ierr);

  if (ec->setfromoptions) {
    ierr = (*ec->setfromoptions)(ec); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(0,"-help",&flag); CHKERRQ(ierr);
  if (flag) {
    ierr = ECPrintHelp(ec); CHKERRQ(ierr);
  }
  return 0;
}

extern int ECPrintTypes_Private(MPI_Comm,char *,char *);

#undef __FUNC__  
#define __FUNC__ "ECPrintHelp" 
/*@
   ECPrintHelp - Prints a help message about the eigenvalue computations.

   Input Parameter:
.   ec - the eigenvalue computation context

   Options Database Command:
$  -help

.keywords: EC, set, help
@*/
int ECPrintHelp(EC ec)
{
  int  ierr;
  char p[64]; 

  PetscValidHeaderSpecific(ec,EC_COOKIE);

  PetscStrcpy(p,"-");
  if (ec->prefix) PetscStrcat(p,ec->prefix);
  PetscPrintf(ec->comm,"EC options --------------------------------------------------\n");
  ECPrintTypes_Private(ec->comm,p,"ec_type");
  PetscPrintf(ec->comm,"  %sec_view: print information on solvers used for eigenvalues\n",p);
  PetscPrintf(ec->comm,"  %sec_view_eigenvalues: print eigenvalues to screen\n",p);
  PetscPrintf(ec->comm,"  %sec_view_eigenvalues_draw: plot eigenvalues to screen\n",p);
  PetscPrintf(ec->comm,"  %sec_spectrum_number <n>: number of eigenvalues to compute\n",p);
  PetscPrintf(ec->comm,"  %sec_spectrum_portion <largest_real,largest_magnitude,smallest_real\n",p);
  PetscPrintf(ec->comm,"                smallest_magnitude,interior>: specify spectrum portion\n");
  PetscPrintf(ec->comm,"  %sec_spectrum_location <location>: find eigenvalues nearby.\n",p);
  PetscPrintf(ec->comm,"                Use with interior portion (listed above).\n");
  if (ec->printhelp) {
    ierr = (*ec->printhelp)(ec,p); CHKERRQ(ierr);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ECSetOperators" 
/*@
   ECSetOperators - Sets the operators for which eigenvalues are to be computed.

   Input Parameter:
.  ec - the eigenvalue computation context
.  A  - the matrix for which eigenvalues are requested
.  B  - optional matrix for generalized eigenvalue problem

.keywords: EC, set, help

.seealso: ECCreate()
@*/
int ECSetOperators(EC ec,Mat A,Mat B)
{
  PetscValidHeaderSpecific(ec,EC_COOKIE);
  PetscValidHeaderSpecific(A,MAT_COOKIE);
  if (B) {
    PetscValidHeaderSpecific(B,MAT_COOKIE);
    if (ec->problemtype == EC_GENERALIZED_EIGENVALUE) {
      SETERRQ(PETSC_ERR_ARG_WRONG,1,"Cannot set B operator for standard eigenvalue problem");
    }
  } else if (ec->problemtype == EC_GENERALIZED_EIGENVALUE) {
    SETERRQ(PETSC_ERR_ARG_WRONG,1,"Must set B operator for generalized eigenvalue problem");
  }
  ec->A = A;
  ec->B = B;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ECSetSpectrumPortion"
/*@
   ECSetSpectrumPortion - Sets the portion of the spectrum from which 
   eigenvalues will be computed.

   Input Parameter:
.   ec - the eigenvalue computation context
.   n - number of eigenvalues requested
.   portion - one of EC_LARGEST_REAL_PART, EC_LARGEST_MAGNITUDE,
               EC_SMALLEST_REAL_PART, EC_SMALLEST_MAGNITUDE,
               EC_INTERIOR
.   location - value near which you wish the spectrum computed

   Options Database Command:
$  -ec_spectrum_portion <largest_real,largest_magnitude, 
$                        smallest_real,smallest_magnitude,
$                        interior>
$  -ec_number number of eigenvalues requested

.keywords: EC, set, help, eigenvalues

.seealso: ECCreate(), ECSetOperators()
@*/
int ECSetEigenvaluePortion(EC ec,int n,ECSpectrumPortion portion,Scalar location)
{
  PetscValidHeaderSpecific(ec,EC_COOKIE);

  ec->n               = n;
  ec->spectrumportion = portion;
  ec->location        = location;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ECSolve"
/*@
   ECSolve - Computes the appropriate eigenvalues.

   Input Parameter:
.   ec - the eigenvalue computation context

.keywords: EC, set, help, eigenvalues

.seealso: ECCreate(), ECSetOperators(), ECGetEigenvalues(),
          ECSolveEigenvectors()
@*/
int ECSolve(EC ec)
{
  int ierr,flag;
  PetscValidHeaderSpecific(ec,EC_COOKIE);

  PLogEventBegin(EC_Solve,ec,0,0,0); 
  ierr = (*ec->solve)(ec); CHKERRQ(ierr);
  PLogEventEnd(EC_Solve,ec,0,0,0); 

  ierr = OptionsHasName(ec->prefix,"-ec_view_eigenvalues",&flag);CHKERRQ(ierr);
  if (flag) {
    double *rpart,*ipart;
    int    i,n,rank;
    MPI_Comm_rank(ec->comm,&rank);
    if (!rank) {
      ierr = ECGetEigenvalues(ec,&n,&rpart,&ipart); CHKERRQ(ierr);
      printf("%d eigenvalues computed\n",n);
      for ( i=0; i<n; i++ ) {
        printf("%d %g + %gi\n",i,rpart[i],ipart[i]);
      }
    }
  }  
  ierr = OptionsHasName(ec->prefix,"-ec_view_eigenvalues_draw",&flag);CHKERRQ(ierr);
  if (flag) {
    double *rpart,*ipart;
    int    i,n,rank;
    MPI_Comm_rank(ec->comm,&rank);
    if (!rank) {
      Viewer    viewer;
      Draw      draw;
      DrawSP    drawsp;

      ierr = ECGetEigenvalues(ec,&n,&rpart,&ipart); CHKERRQ(ierr);
      ierr = ViewerDrawOpenX(PETSC_COMM_SELF,0,"Eigenvalues",PETSC_DECIDE,PETSC_DECIDE,
                             300,300,&viewer); CHKERRQ(ierr);
      ierr = ViewerDrawGetDraw(viewer,&draw); CHKERRQ(ierr);
      ierr = DrawSPCreate(draw,1,&drawsp); CHKERRQ(ierr);
      for ( i=0; i<n; i++ ) {
        ierr = DrawSPAddPoint(drawsp,rpart+i,ipart+i); CHKERRQ(ierr);
      }
      ierr = DrawSPDraw(drawsp); CHKERRQ(ierr);
      ierr = DrawSPDestroy(drawsp); CHKERRQ(ierr);
      ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
    }
  }  
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ECSolveEigenvectors"
/*@
   ECSolveEigenvectors - Computes the appropriate eigenvectors

   Input Parameter:
.   ec - the eigenvalue computation context

   Notes: Must be called after ECSolve().

.keywords: EC, set, help, eigenvalues

.seealso: ECCreate(), ECSetOperators(), ECSolve(), ECGetEigenVectors(),
          ECSetEigenvectorsRequired()
@*/
int ECSolveEigenvectors(EC ec)
{
  int ierr;
  PetscValidHeaderSpecific(ec,EC_COOKIE);

  if (!ec->computeeigenvectors) {
    SETERRQ(1,1,"Must call ECSetEigenvectorsRequired() before this call");
  }

  /* PLogEventBegin(EC_SolveEigenvectors,ec,0,0,0);  */
  ierr = (*ec->solveeigenvectors)(ec); CHKERRQ(ierr);
  /* PLogEventEnd(EC_SolveEigenvectors,ec,0,0,0);  */
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ECSetType" 
/*@
   ECSetType - Builds EC for a particular solver. 

   Input Parameter:
.  ctx      - the eigenvalue computation context
.  itmethod - a known method

   Options Database Command:
$  -ec_type  <method>
$      Use -help for a list of available methods
$      (for instance, lapack, arpack)

   Notes:  
   See "petsc/include/ec.h" for available methods (for instance,
   EC_LAPACK, EC_???

  Normally, it is best to use the ECSetFromOptions() command and
  then set the EC type from the options database rather than by using
  this routine.  Using the options database provides the user with
  maximum flexibility in evaluating the many different Krylov methods.
  The ECSetType() routine is provided for those situations where it
  is necessary to set the iterative solver independently of the command
  line or options database.  This might be the case, for example, when
  the choice of iterative solver changes during the execution of the
  program, and the user's application is taking responsibility for
  choosing the appropriate method.  In other words, this routine is
  for the advanced user.

.keywords: EC, set, method

.seealso: ECCreate(), ECSetFromOptions()
@*/
int ECSetType(EC ec,ECType itmethod)
{
  int ierr,(*r)(EC);

  PetscValidHeaderSpecific(ec,EC_COOKIE);
  if (ec->type == (int) itmethod) return 0;

  if (ec->setupcalled) {
    /* destroy the old private EC context */
    ierr = (*(ec)->destroy)((PetscObject)ec); CHKERRQ(ierr);
    ec->data = 0;
  }
  /* Get the function pointers for the approach requested */
  if (!__ECList) {ierr = ECRegisterAll(); CHKERRQ(ierr);}
  if (!__ECList) SETERRQ(1,0,"Could not get list of EC types"); 
  r =  (int (*)(EC))NRFindRoutine( __ECList, (int)itmethod, (char *)0 );
  if (!r) {SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown method");}
  if (ec->data) PetscFree(ec->data);
  ec->data = 0;
  return (*r)(ec);
}

#undef __FUNC__  
#define __FUNC__ "ECRegister" 
/*@C
   ECRegister - Adds the iterative method to the EC package,  given
   an iterative name (ECType) and a function pointer.

   Input Parameters:
.  name   - for instance ECCG, ECGMRES, ...
.  sname  - corresponding string for name
.  create - routine to create method context

.keywords: EC, register

.seealso: ECRegisterAll(), ECRegisterDestroy(), ECDestroy(), ECCreate()
@*/
int  ECRegister(ECType name, char *sname, int  (*create)(EC))
{
  int ierr;
  int (*dummy)(void *) = (int (*)(void *)) create;
  if (!__ECList) {ierr = NRCreate(&__ECList); CHKERRQ(ierr);}
  return NRRegister( __ECList, (int) name, sname, dummy );
}

#undef __FUNC__  
#define __FUNC__ "ECRegisterDestroy" 
/*@C
   ECRegisterDestroy - Frees the list of EC methods that were
   registered by ECRegister().

.keywords: EC, register, destroy

.seealso: ECRegister(), ECRegisterAll()
@*/
int ECRegisterDestroy()
{
  if (__ECList) {
    NRDestroy( __ECList );
    __ECList = 0;
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ECGetTypeFromOptions_Private" 
/*
   ECGetTypeFromOptions_Private - Sets the selected EC type from 
   the options database.

   Input Parameter:
.  ec - the EC context

   Output Parameter:
.  itmethod - iterative method

   Returns:
   Returns 1 if the method is found; 0 otherwise.
*/
int ECGetTypeFromOptions_Private(EC ec,ECType *itmethod)
{
  char sbuf[50];
  int  flg,ierr;

  ierr = OptionsGetString(ec->prefix,"-ec_type", sbuf, 50,&flg); CHKERRQ(ierr);
  if (flg) {
    if (!__ECList) ECRegisterAll();
    *itmethod = (ECType)NRFindID( __ECList, sbuf );
    return 1;
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ECGetType" 
/*@C
   ECGetType - Gets the EC type and method name (as a string) from 
   the method type.

   Input Parameter:
.  ec - Krylov context 

   Output Parameters:
.  itmeth - EC method (or use PETSC_NULL)
.  name - name of EC method (or use PETSC_NULL)

.keywords: EC, get, method, name
@*/
int ECGetType(EC ec,ECType *type,char **name)
{
  int ierr;
  if (!__ECList) {ierr = ECRegisterAll(); CHKERRQ(ierr);}
  if (type) *type = (ECType) ec->type;
  if (name)  *name = NRFindName( __ECList, (int) ec->type);
  return 0;
}

#include <stdio.h>
#undef __FUNC__  
#define __FUNC__ "ECPrintTypes_Private" 
/*
   ECPrintTypes_Private - Prints the EC methods available from the options 
   database.

   Input Parameters:
.  comm   - The communicator (usually MPI_COMM_WORLD)
.  prefix - prefix (usually "-")
.  name   - the options database name (by default "ec_type") 
*/
int ECPrintTypes_Private(MPI_Comm comm,char* prefix,char *name)
{
  FuncList *entry;
  int      count = 0;

  if (!__ECList) {ECRegisterAll();}
  entry = __ECList->head;
  PetscPrintf(comm,"  %s%s (one of)",prefix,name);
  while (entry) {
    PetscPrintf(comm," %s",entry->name);
    entry = entry->next;
    count++;
    if (count == 8) PetscPrintf(comm,"\n    ");
  }
  PetscPrintf(comm,"\n");
  return 1;
}



