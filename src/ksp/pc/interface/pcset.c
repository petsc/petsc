#define PETSCKSP_DLL
/*
    Routines to set PC methods and options.
*/

#include "src/ksp/pc/pcimpl.h"      /*I "petscpc.h" I*/
#include "petscsys.h"

PetscTruth PCRegisterAllCalled = PETSC_FALSE;
/*
   Contains the list of registered KSP routines
*/
PetscFList PCList = 0;

#undef __FUNCT__  
#define __FUNCT__ "PCSetType"
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
  See "petsc/include/petscpc.h" for available methods (for instance,
  PCJACOBI, PCILU, or PCBJACOBI).

  Normally, it is best to use the KSPSetFromOptions() command and
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

.seealso: KSPSetType(), PCType

@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCSetType(PC pc, PCType type)
{
  PetscErrorCode ierr,(*r)(PC);
  PetscTruth     match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidCharPointer(type,2);

  ierr = PetscTypeCompare((PetscObject)pc,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (pc->ops->destroy) {ierr =  (*pc->ops->destroy)(pc);CHKERRQ(ierr);}
  ierr = PetscFListDestroy(&pc->qlist);CHKERRQ(ierr);
  pc->data        = 0;
  pc->setupcalled = 0;

  /* Get the function pointers for the method requested */
  if (!PCRegisterAllCalled) {ierr = PCRegisterAll(0);CHKERRQ(ierr);}

  /* Determine the PCCreateXXX routine for a particular preconditioner */
  ierr =  PetscFListFind(pc->comm,PCList,type,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested PC type %s",type);
  if (pc->data) {ierr = PetscFree(pc->data);CHKERRQ(ierr);}

  pc->ops->setup               = (PetscErrorCode (*)(PC)) 0;
  pc->ops->apply               = (PetscErrorCode (*)(PC,Vec,Vec)) 0;
  pc->ops->applyrichardson     = (PetscErrorCode (*)(PC,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,PetscInt)) 0;
  pc->ops->applyBA             = (PetscErrorCode (*)(PC,PetscInt,Vec,Vec,Vec)) 0;
  pc->ops->applytranspose      = (PetscErrorCode (*)(PC,Vec,Vec)) 0;
  pc->ops->applyBAtranspose    = (PetscErrorCode (*)(PC,PetscInt,Vec,Vec,Vec)) 0;
  pc->ops->setfromoptions      = (PetscErrorCode (*)(PC)) 0;
  pc->ops->presolve            = (PetscErrorCode (*)(PC,KSP,Vec,Vec)) 0;
  pc->ops->postsolve           = (PetscErrorCode (*)(PC,KSP,Vec,Vec)) 0;
  pc->ops->getfactoredmatrix   = (PetscErrorCode (*)(PC,Mat*)) 0;
  pc->ops->applysymmetricleft  = (PetscErrorCode (*)(PC,Vec,Vec)) 0;
  pc->ops->applysymmetricright = (PetscErrorCode (*)(PC,Vec,Vec)) 0;
  pc->ops->setuponblocks       = (PetscErrorCode (*)(PC)) 0;
  pc->ops->destroy             = (PetscErrorCode (*)(PC)) 0;
  pc->ops->view                = (PetscErrorCode (*)(PC,PetscViewer)) 0;
  pc->modifysubmatrices        = (PetscErrorCode (*)(PC,PetscInt,const IS[],const IS[],Mat[],void*)) 0;

  /* Call the PCCreateXXX routine for this particular preconditioner */
  ierr = (*r)(pc);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)pc,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCRegisterDestroy"
/*@
   PCRegisterDestroy - Frees the list of preconditioners that were
   registered by PCRegisterDynamic().

   Not Collective

   Level: advanced

.keywords: PC, register, destroy

.seealso: PCRegisterAll(), PCRegisterAll()

@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PCList) {
    ierr = PetscFListDestroy(&PCList);CHKERRQ(ierr);
    PCList = 0;
  }
  PCRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGetType"
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
PetscErrorCode PETSCKSP_DLLEXPORT PCGetType(PC pc,PCType *meth)
{
  PetscFunctionBegin;
  *meth = (PCType) pc->type_name;
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode PCGetDefaultType_Private(PC,const char*[]);

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions"
/*@
   PCSetFromOptions - Sets PC options from the options database.
   This routine must be called before PCSetUp() if the user is to be
   allowed to set the preconditioner method. 

   Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Level: developer

.keywords: PC, set, from, options, database

.seealso: 

@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCSetFromOptions(PC pc)
{
  PetscErrorCode ierr;
  char           type[256];
  const char     *def;
  PetscTruth     flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);

  if (!PCRegisterAllCalled) {ierr = PCRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = PetscOptionsBegin(pc->comm,pc->prefix,"Preconditioner (PC) Options","PC");CHKERRQ(ierr);
    if (!pc->type_name) {
      ierr = PCGetDefaultType_Private(pc,&def);CHKERRQ(ierr);
    } else {
      def = pc->type_name;
    }

    ierr = PetscOptionsList("-pc_type","Preconditioner","PCSetType",PCList,def,type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCSetType(pc,type);CHKERRQ(ierr);
    } else if (!pc->type_name){
      ierr = PCSetType(pc,def);CHKERRQ(ierr);
    } 

    if (pc->ops->setfromoptions) {
      ierr = (*pc->ops->setfromoptions)(pc);CHKERRQ(ierr);
    }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
