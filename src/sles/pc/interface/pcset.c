/*$Id: pcset.c,v 1.118 2001/08/21 21:03:13 bsmith Exp $*/
/*
    Routines to set PC methods and options.
*/

#include "src/sles/pc/pcimpl.h"      /*I "petscpc.h" I*/
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
  ierr = PetscFListDestroy(&pc->qlist);CHKERRQ(ierr);
  pc->data        = 0;
  pc->setupcalled = 0;

  /* Get the function pointers for the method requested */
  if (!PCRegisterAllCalled) {ierr = PCRegisterAll(0);CHKERRQ(ierr);}

  /* Determine the PCCreateXXX routine for a particular preconditioner */
  ierr =  PetscFListFind(pc->comm,PCList,type,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(1,"Unable to find requested PC type %s",type);
  if (pc->data) {ierr = PetscFree(pc->data);CHKERRQ(ierr);}

  pc->ops->destroy             = (int (*)(PC)) 0;
  pc->ops->view                = (int (*)(PC,PetscViewer)) 0;
  pc->ops->apply               = (int (*)(PC,Vec,Vec)) 0;
  pc->ops->setup               = (int (*)(PC)) 0;
  pc->ops->applyrichardson     = (int (*)(PC,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,int)) 0;
  pc->ops->applyBA             = (int (*)(PC,int,Vec,Vec,Vec)) 0;
  pc->ops->setfromoptions      = (int (*)(PC)) 0;
  pc->ops->applytranspose      = (int (*)(PC,Vec,Vec)) 0;
  pc->ops->applyBAtranspose    = (int (*)(PC,int,Vec,Vec,Vec)) 0;
  pc->ops->presolve            = (int (*)(PC,KSP,Vec,Vec)) 0;
  pc->ops->postsolve           = (int (*)(PC,KSP,Vec,Vec)) 0;
  pc->ops->getfactoredmatrix   = (int (*)(PC,Mat*)) 0;
  pc->ops->applysymmetricleft  = (int (*)(PC,Vec,Vec)) 0;
  pc->ops->applysymmetricright = (int (*)(PC,Vec,Vec)) 0;
  pc->ops->setuponblocks       = (int (*)(PC)) 0;
  pc->modifysubmatrices        = (int (*)(PC,int,const IS[],const IS[],Mat[],void*)) 0;

  /* Call the PCCreateXXX routine for this particular preconditioner */
  ierr = (*r)(pc);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)pc,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCRegisterDestroy"
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
int PCGetType(PC pc,PCType *meth)
{
  PetscFunctionBegin;
  *meth = (PCType) pc->type_name;
  PetscFunctionReturn(0);
}

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
int PCSetFromOptions(PC pc)
{
  int        ierr;
  char       type[256],*def;
  PetscTruth flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);

  if (!PCRegisterAllCalled) {ierr = PCRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = PetscOptionsBegin(pc->comm,pc->prefix,"Preconditioner (PC) Options","PC");CHKERRQ(ierr);
    if (!pc->type_name) {
      PetscTruth ismatshell;
      int        size;

      /*
        Shell matrix (probably) cannot support Bjacobi and ILU
      */
      ierr = MPI_Comm_size(pc->comm,&size);CHKERRQ(ierr);
      if (pc->pmat) {
        ierr = PetscTypeCompare((PetscObject)pc->pmat,MATSHELL,&ismatshell);CHKERRQ(ierr);
      } else {
        ismatshell = PETSC_FALSE; /* matrix is not yet set, so guess that it will not be MATSHELL */
      }
      /* 
         MATMFFD cannot support BJacobia and ILU
      */
      if (!ismatshell) {
        ierr = PetscTypeCompare((PetscObject)pc->pmat,MATMFFD,&ismatshell);CHKERRQ(ierr);
      }

      if (ismatshell) {
        def = PCNONE;
        PetscLogInfo(pc,"PCSetOperators:Setting default PC to PCNONE since MATSHELL doesn't support\n\
    preconditioners (unless defined by the user)\n");
      } else if (size == 1) {
        ierr = PetscTypeCompare((PetscObject)pc->pmat,MATSEQSBAIJ,&flg);CHKERRQ(ierr);
        if (flg) {
          def = PCICC;
        } else {
	  def = PCILU;
        }
      } else {
        def = PCBJACOBI;
      }
    } else {
      def = pc->type_name;
    }

    ierr = PetscOptionsList("-pc_type","Preconditioner","PCSetType",PCList,def,type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCSetType(pc,type);CHKERRQ(ierr);
    }

    ierr = PetscOptionsName("-pc_constant_null_space","Add constant null space to preconditioner","PCNullSpaceAttach",&flg);CHKERRQ(ierr);
    if (flg) {
      MatNullSpace nsp;

      ierr = MatNullSpaceCreate(pc->comm,1,0,0,&nsp);CHKERRQ(ierr);
      ierr = PCNullSpaceAttach(pc,nsp);CHKERRQ(ierr);
      ierr = MatNullSpaceDestroy(nsp);CHKERRQ(ierr);
    }


    /* option is actually checked in PCSetUp() */
    if (pc->nullsp) {
      ierr = PetscOptionsName("-pc_test_null_space","Is provided null space correct","None",&flg);CHKERRQ(ierr);
    }

    /*
      Set the type if it was never set.
    */
    if (!pc->type_name) {
      ierr = PCSetType(pc,def);CHKERRQ(ierr);
    }



    if (pc->ops->setfromoptions) {
      ierr = (*pc->ops->setfromoptions)(pc);CHKERRQ(ierr);
    }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
#if defined(__cplusplus) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE) && defined(PETSC_HAVE_CXX_NAMESPACE)
  ierr = PCESISetFromOptions(pc);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
