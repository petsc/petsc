/*$Id: pcset.c,v 1.100 2000/01/26 13:35:51 bsmith Exp bsmith $*/
/*
    Routines to set PC methods and options.
*/

#include "src/sles/pc/pcimpl.h"      /*I "pc.h" I*/
#include "sys.h"

PetscTruth PCRegisterAllCalled = PETSC_FALSE;
/*
   Contains the list of registered KSP routines
*/
FList PCList = 0;

#undef __FUNC__  
#define __FUNC__ "PCSetType"
/*@C
   PCSetType - Builds PC for a particular preconditioner.

   Collective on PC

   Input Parameter:
+  pc - the preconditioner context.
-  type - a known method

   Options Database Key:
.  -pc_type <type> - Sets PC type

   Use -help for a list of available methods (for instance,
   jacobi or bjacobi)

  Notes:
  See "petsc/include/pc.h" for available methods (for instance,
  PCJACOBI, PCILU, or PCBJACOBI).

  Normally, it is best to use the SLESSetFromOptions() command and
  then set the PC type from the options database rather than by using
  this routine.  Using the options database provides the user with
  maximum flexibility in evaluating the many different preconditioners. 
  The PCSetType() routine is provided for those situations where it
  is necessary to set the preconditioner independently of the command
  line or options database.  This might be the case, for example, when
  the choice of preconditioner changes during the execution of the
  program, and the user's application is taking responsibility for
  choosing the appropriate preconditioner.  In other words, this
  routine is not for beginners.

  Level: intermediate

.keywords: PC, set, method, type

.seealso: KSPSetType()

@*/
int PCSetType(PC pc,PCType type)
{
  int        ierr,(*r)(PC);
  PetscTruth match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  PetscValidCharPointer(type);

  ierr = PetscTypeCompare((PetscObject)pc,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (pc->ops->destroy) {ierr =  (*pc->ops->destroy)(pc);CHKERRQ(ierr);}
  pc->data        = 0;
  pc->setupcalled = 0;

  /* Get the function pointers for the method requested */
  if (!PCRegisterAllCalled) {ierr = PCRegisterAll(0);CHKERRQ(ierr);}

  /* Determine the PCCreateXXX routine for a particular preconditioner */
  ierr =  FListFind(pc->comm,PCList,type,(int (**)(void *)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(1,1,"Unable to find requested PC type %s",type);
  if (pc->data) {ierr = PetscFree(pc->data);CHKERRQ(ierr);}

  pc->ops->destroy             = (int (*)(PC)) 0;
  pc->ops->view                = (int (*)(PC,Viewer)) 0;
  pc->ops->apply               = (int (*)(PC,Vec,Vec)) 0;
  pc->ops->setup               = (int (*)(PC)) 0;
  pc->ops->applyrichardson     = (int (*)(PC,Vec,Vec,Vec,int)) 0;
  pc->ops->applyBA             = (int (*)(PC,int,Vec,Vec,Vec)) 0;
  pc->ops->setfromoptions      = (int (*)(PC)) 0;
  pc->ops->printhelp           = (int (*)(PC,char*)) 0;
  pc->ops->applytranspose      = (int (*)(PC,Vec,Vec)) 0;
  pc->ops->applyBAtranspose    = (int (*)(PC,int,Vec,Vec,Vec)) 0;
  pc->ops->presolve            = (int (*)(PC,KSP,Vec,Vec)) 0;
  pc->ops->postsolve           = (int (*)(PC,KSP,Vec,Vec)) 0;
  pc->ops->getfactoredmatrix   = (int (*)(PC,Mat*)) 0;
  pc->ops->applysymmetricleft  = (int (*)(PC,Vec,Vec)) 0;
  pc->ops->applysymmetricright = (int (*)(PC,Vec,Vec)) 0;
  pc->ops->setuponblocks       = (int (*)(PC)) 0;
  pc->modifysubmatrices   = (int (*)(PC,int,IS*,IS*,Mat*,void*)) 0;

  /* Call the PCCreateXXX routine for this particular preconditioner */
  ierr = (*r)(pc);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)pc,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCRegisterDestroy"
/*@C
   PCRegisterDestroy - Frees the list of preconditioners that were
   registered by PCRegisterDynamic().

   Not Collective

   Level: advanced

.keywords: PC, register, destroy

.seealso: PCRegisterAll(), PCRegisterAll()

@*/
int PCRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (PCList) {
    ierr = FListDestroy(PCList);CHKERRQ(ierr);
    PCList = 0;
  }
  PCRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCPrintHelp"
/*@
   PCPrintHelp - Prints all the options for the PC component.

   Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Options Database Keys:
+  -help - Prints PC options
-  -h - Prints PC options

   Level: developer

.keywords: PC, help

.seealso: PCSetFromOptions()

@*/
int PCPrintHelp(PC pc)
{
  char p[64]; 
  int  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscStrcpy(p,"-");CHKERRQ(ierr);
  if (pc->prefix) {ierr = PetscStrcat(p,pc->prefix);CHKERRQ(ierr);}
  if (!PCRegisterAllCalled) {ierr = PCRegisterAll(0);CHKERRQ(ierr);}
  ierr = (*PetscHelpPrintf)(pc->comm,"PC options --------------------------------------------------\n");CHKERRQ(ierr);
  ierr = FListPrintTypes(pc->comm,stdout,pc->prefix,"pc_type",PCList);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm,"Run program with -help %spc_type <type> for help on ",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm,"a particular method\n");CHKERRQ(ierr);
  if (pc->ops->printhelp) {
    ierr = (*pc->ops->printhelp)(pc,p);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "PCGetType"
/*@C
   PCGetType - Gets the PC method type and name (as a string) from the PC
   context.

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  name - name of preconditioner 

   Level: intermediate

.keywords: PC, get, method, name, type

.seealso: PCSetType()

@*/
int PCGetType(PC pc,PCType *meth)
{
  PetscFunctionBegin;
  *meth = (PCType) pc->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetTypeFromOptions"
/*@
   PCSetTypeFromOptions - Sets PC type from the options database; if not given
         sets default.

   Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Level: developer

.keywords: PC, set, from, options, database

.seealso: PCPrintHelp(), PCSetFromOptions(), SLESSetFromOptions(),
          SLESSetTypeFromOptions()
@*/
int PCSetTypeFromOptions(PC pc)
{
  char       type[256];
  int        ierr;
  PetscTruth flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = OptionsGetString(pc->prefix,"-pc_type",type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCSetType(pc,type);CHKERRQ(ierr);
  }
  if (!pc->type_name) {
    int size;

    ierr = MPI_Comm_size(pc->comm,&size);CHKERRQ(ierr);
    if (size == 1) {
      ierr = PCSetType(pc,PCILU);CHKERRQ(ierr);
    } else {
      ierr = PCSetType(pc,PCBJACOBI);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetFromOptions"
/*@
   PCSetFromOptions - Sets PC options from the options database.
   This routine must be called before PCSetUp() if the user is to be
   allowed to set the preconditioner method. 

   Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Level: developer

.keywords: PC, set, from, options, database

.seealso: PCPrintHelp()

@*/
int PCSetFromOptions(PC pc)
{
  int        ierr;
  PetscTruth flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PCSetTypeFromOptions(pc);CHKERRQ(ierr);
  if (pc->ops->setfromoptions) {
    ierr = (*pc->ops->setfromoptions)(pc);CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg);CHKERRQ(ierr);
  if (flg){
    ierr = PCPrintHelp(pc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
