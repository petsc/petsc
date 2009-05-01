#define PETSCKSP_DLL


#include "petscpc.h"   /*I "petscpc.h" I*/
#include "petscmg.h"   /*I "petscmg.h" I*/
#include "petscda.h"   /*I "petscda.h" I*/
#include "../src/ksp/pc/impls/mg/mgimpl.h"

const char *PCExoticTypes[] = {"face","wirebasket","PCExoticType","PC_Exotic",0};

extern PetscErrorCode DAGetWireBasketInterpolation(DA,Mat,MatReuse,Mat*);
extern PetscErrorCode DAGetFaceInterpolation(DA,Mat,MatReuse,Mat*);

typedef struct {
  DA           da;
  PCExoticType type;
  Mat          P;      /* the interpolation matrix */
} PC_Exotic;

#undef __FUNCT__  
#define __FUNCT__ "PCExoticSetType"
/*@
   PCExoticSetType - Sets the type of coarse grid interpolation to use

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  type - either PC_EXOTIC_FACE or PC_EXOTIC_WIREBASKET (defaults to face)

   Notes: The face based interpolation has 1 degree of freedom per face and ignores the 
     edge and vertex values completely in the coarse problem. For any seven point
     stencil the interpolation of a constant on all faces into the interior is that constant.

     The wirebasket interpolation has 1 degree of freedom per vertex, per edge and 
     per face. A constant on the subdomain boundary is interpolated as that constant
     in the interior of the domain. 

     The coarse grid matrix is obtained via the Galerkin computation A_c = R A R^T, hence 
     if A is nonsingular A_c is also nonsingular.

     Both interpolations are suitable for only scalar problems.

   Level: intermediate


.seealso: PCEXOTIC, PCExoticType()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCExoticSetType(PC pc,PCExoticType type)
{
  PetscErrorCode ierr,(*f)(PC,PCExoticType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCExoticSetType_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,type);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCExoticSetType_Exotic"
PetscErrorCode PETSCKSP_DLLEXPORT PCExoticSetType_Exotic(PC pc,PCExoticType type)
{
  PC_MG     **mg = (PC_MG**)pc->data;
  PC_Exotic *ctx = (PC_Exotic*) mg[0]->innerctx;

  PetscFunctionBegin;
  ctx->type = type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_Exotic"
PetscErrorCode PCSetUp_Exotic(PC pc)
{
  PetscErrorCode ierr;
  Mat            A;
  PC_MG          **mg = (PC_MG**)pc->data;
  PC_Exotic      *ex = (PC_Exotic*) mg[0]->innerctx;
  DA             da = ex->da;
  MatReuse       reuse = (ex->P) ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX;

  PetscFunctionBegin;
  ierr = PCGetOperators(pc,PETSC_NULL,&A,PETSC_NULL);CHKERRQ(ierr);
  if (ex->type == PC_EXOTIC_FACE) {
    ierr = DAGetFaceInterpolation(da,A,reuse,&ex->P);CHKERRQ(ierr);
  } else if (ex->type == PC_EXOTIC_WIREBASKET) {
    ierr = DAGetWireBasketInterpolation(da,A,reuse,&ex->P);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_ERR_PLIB,"Unknown exotic coarse space %d",ex->type);
  ierr = PCMGSetInterpolation(pc,1,ex->P);CHKERRQ(ierr);
  ierr = PCSetUp_MG(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_Exotic"
PetscErrorCode PCDestroy_Exotic(PC pc)
{
  PetscErrorCode ierr;
  PC_MG          **mg = (PC_MG**)pc->data;
  PC_Exotic      *ctx = (PC_Exotic*) mg[0]->innerctx;

  PetscFunctionBegin;
  if (ctx->da) {ierr = DADestroy(ctx->da);CHKERRQ(ierr);}
  if (ctx->P) {ierr = MatDestroy(ctx->P);CHKERRQ(ierr);}
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  ierr = PCDestroy_MG(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_Exotic_Error"
PetscErrorCode PCSetUp_Exotic_Error(PC pc)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"You are using the Exotic preconditioner but never called PCSetDA()");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetDA"
/*@
   PCSetDA - Sets the DA that is to be used by the PCEXOTIC or certain other preconditioners

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  da - the da

   Level: intermediate


.seealso: PCEXOTIC, PCExoticType()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCSetDA(PC pc,DA da)
{
  PetscErrorCode ierr,(*f)(PC,DA);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCSetDA_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,da);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PCSetDA_Exotic"
PetscErrorCode PETSCKSP_DLLEXPORT PCSetDA_Exotic(PC pc,DA da)
{
  PetscErrorCode ierr;
  PC_MG          **mg = (PC_MG**)pc->data;
  PC_Exotic      *ctx = (PC_Exotic*) mg[0]->innerctx;

  PetscFunctionBegin;
  ctx->da = da;
  pc->ops->setup = PCSetUp_Exotic;
  ierr   = PetscObjectReference((PetscObject)da);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView_Exotic"
PetscErrorCode PCView_Exotic(PC pc,PetscViewer viewer)
{
  PC_MG          **mg = (PC_MG**)pc->data;
  PetscErrorCode ierr;
  PetscTruth     iascii;
  PC_Exotic      *ctx = (PC_Exotic*) mg[0]->innerctx;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"    Exotic type = %s\n",PCExoticTypes[ctx->type]);CHKERRQ(ierr);
  }
  ierr = PCView_MG(pc,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_Exotic"
PetscErrorCode PCSetFromOptions_Exotic(PC pc)
{
  PetscErrorCode ierr;
  PetscTruth     flg;
  PC_MG          **mg = (PC_MG**)pc->data;
  PCExoticType   mgctype;
  PC_Exotic      *ctx = (PC_Exotic*) mg[0]->innerctx;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Exotic coarse space options");CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-pc_exotic_type","face or wirebasket","PCExoticSetType",PCExoticTypes,(PetscEnum)ctx->type,(PetscEnum*)&mgctype,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCExoticSetType(pc,mgctype);CHKERRQ(ierr);
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*MC
     PCEXOTIC - Two level overlapping Schwarz preconditioner with exotic (non-standard) coarse grid spaces

     This uses the PCMG infrastructure restricted to two levels and the face and wirebasket based coarse
   grid spaces. These coarse grid spaces originate in the work of Bramble, Pasciak  and Schatz, "The Construction
   of Preconditioners for Elliptic Problems by Substructing IV", Mathematics of Computation, volume 53 pages 1--24, 1989.
   They were generalized slightly in "Domain Decomposition Method for Linear Elasticity", Ph. D. thesis, Barry Smith,
   New York University, 1990. They were then explored in great detail in Dryja, Smith, Widlund, "Schwarz Analysis
   of Iterative Substructuring Methods for Elliptic Problems in Three Dimensions, SIAM Journal on Numerical 
   Analysis, volume 31. pages 1662-1694, 1994. These were developed in the context of iterative substructuring preconditioners.
   They were then ingeniously applied as coarse grid spaces for overlapping Schwarz methods by Dohrmann and Widlund.
   They refer to them as GDSW (generalized Dryja, Smith, Widlund preconditioners). See, for example, 
   Clark R. Dohrmann, Axel Klawonn, and Olof B. Widlund. Extending theory for domain decomposition algorithms to irregular subdomains. In Ulrich Langer, Marco
   Discacciati, David Keyes, Olof Widlund, and Walter Zulehner, editors, Proceedings
   of the 17th International Conference on Domain Decomposition Methods in
   Science and Engineering, held in Strobl, Austria, July 3-7, 2006, number 60 in
   Springer-Verlag, Lecture Notes in Computational Science and Engineering, pages 255-261, 2007.
   Clark R. Dohrmann, Axel Klawonn, and Olof B. Widlund. A family of energy min-
   imizing coarse spaces for overlapping Schwarz preconditioners. In Ulrich Langer,
   Marco Discacciati, David Keyes, OlofWidlund, andWalter Zulehner, editors, Proceedings
   of the 17th International Conference on Domain Decomposition Methods
   in Science and Engineering, held in Strobl, Austria, July 3-7, 2006, number 60 in
   Springer-Verlag, Lecture Notes in Computational Science and Engineering, pages 247-254, 2007
   Clark R. Dohrmann, Axel Klawonn, and Olof B. Widlund. Domain decomposition
   for less regular subdomains: Overlapping Schwarz in two dimensions. SIAM J.
   Numer. Anal., 46(4):2153-2168, 2008.
   Clark R. Dohrmann and Olof B. Widlund. An overlapping Schwarz
   algorithm for almost incompressible elasticity. Technical Report
   TR2008-912, Department of Computer Science, Courant Institute
   of Mathematical Sciences, New York University, May 2008. URL:
   http://cs.nyu.edu/csweb/Research/TechReports/TR2008-912/TR2008-912.pdf

   Options Database: The usual PCMG options are supported, such as -mg_levels_pc_type <type> -mg_coarse_pc_type <type>
      -pc_mg_type <type>

   Level: advanced

.seealso:  PCMG, PCSetDA(), PCExoticType, PCExoticSetType()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_Exotic"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_Exotic(PC pc)
{
  PetscErrorCode ierr;
  PC_Exotic      *ex;
  PC_MG          **mg;

  PetscFunctionBegin;
  /* if type was previously mg; must manually destroy it because call to PCSetType(pc,PCMG) will not destroy it */
  if (pc->ops->destroy) { ierr =  (*pc->ops->destroy)(pc);CHKERRQ(ierr); pc->data = 0;}
  ierr = PetscStrfree(((PetscObject)pc)->type_name);CHKERRQ(ierr);
  ((PetscObject)pc)->type_name = 0;

  ierr = PCSetType(pc,PCMG);CHKERRQ(ierr);
  ierr = PCMGSetLevels(pc,2,PETSC_NULL);CHKERRQ(ierr);
  ierr = PCMGSetGalerkin(pc);CHKERRQ(ierr);
  ierr = PetscNew(PC_Exotic,&ex);CHKERRQ(ierr);\
  ex->type = PC_EXOTIC_FACE;
  mg = (PC_MG**) pc->data;
  mg[0]->innerctx = ex;


  pc->ops->setfromoptions = PCSetFromOptions_Exotic;
  pc->ops->view           = PCView_Exotic;
  pc->ops->destroy        = PCDestroy_Exotic;
  pc->ops->setup          = PCSetUp_Exotic_Error;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCExoticSetType_C","PCExoticSetType_Exotic",PCExoticSetType_Exotic);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCSetDA_C","PCSetDA_Exotic",PCSetDA_Exotic);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
