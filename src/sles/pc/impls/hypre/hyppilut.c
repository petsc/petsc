/*$Id: bvec2.c,v 1.202 2001/09/12 03:26:24 bsmith Exp $*/
/*

*/

#include "src/sles/pc/pcimpl.h"          /*I "petscpc.h" I*/
EXTERN_C_BEGIN
#include "HYPRE.h"
#include "IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
EXTERN_C_END

extern int MatHYPRE_IJMatrixCreate(Mat,HYPRE_IJMatrix*);
extern int MatHYPRE_IJMatrixCopy(Mat,HYPRE_IJMatrix);
extern int VecHYPRE_IJVectorCreate(Vec,HYPRE_IJVector*);
extern int VecHYPRE_IJVectorCopy(Vec,HYPRE_IJVector);
extern int VecHYPRE_IJVectorCopyFrom(HYPRE_IJVector ij,Vec v);

/* 
   Private context (data structure) for the  preconditioner.  
*/
typedef struct {
  HYPRE_Solver       hsolver;
  HYPRE_IJMatrix     ij;
  HYPRE_IJVector     b,x;

  /* options for pilut */
  int    maxiter;
  double tol;
  int    factorrowsize;
} PC_HYPRE;


#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_HYPRE_Pilut"
static int PCSetUp_HYPRE_Pilut(PC pc)
{
  PC_HYPRE           *jac = (PC_HYPRE*)pc->data;
  int                ierr;
  HYPRE_ParCSRMatrix hmat;
  HYPRE_ParVector    bv,xv;

  PetscFunctionBegin;
  if (!jac->ij) { /* create the matrix the first time through */ 
    ierr = MatHYPRE_IJMatrixCreate(pc->pmat,&jac->ij);CHKERRQ(ierr);
  }
  if (!jac->b) {
    ierr = VecHYPRE_IJVectorCreate(pc->vec,&jac->b);CHKERRQ(ierr);
    ierr = VecHYPRE_IJVectorCreate(pc->vec,&jac->x);CHKERRQ(ierr);
  }
  ierr = MatHYPRE_IJMatrixCopy(pc->pmat,jac->ij);CHKERRQ(ierr);
  ierr = HYPRE_IJMatrixGetObject(jac->ij,(void**)&hmat);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorGetObject(jac->b,(void**)&bv);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorGetObject(jac->x,(void**)&xv);CHKERRQ(ierr);
  ierr = HYPRE_ParCSRPilutSetup(jac->hsolver,hmat,bv,xv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_HYPRE_Pilut"
static int PCApply_HYPRE_Pilut(PC pc,Vec x,Vec y)
{
  PC_HYPRE           *jac = (PC_HYPRE*)pc->data;
  int                ierr;
  HYPRE_ParCSRMatrix hmat;
  HYPRE_ParVector    bv,xv;

  PetscFunctionBegin;
  ierr = VecHYPRE_IJVectorCopy(x,jac->b);CHKERRQ(ierr);
  ierr = HYPRE_IJMatrixGetObject(jac->ij,(void**)&hmat);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorGetObject(jac->b,(void**)&bv);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorGetObject(jac->x,(void**)&xv);CHKERRQ(ierr);
  ierr = HYPRE_ParCSRPilutSolve(jac->hsolver,hmat,bv,xv);CHKERRQ(ierr);
  ierr = VecHYPRE_IJVectorCopyFrom(jac->x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_HYPRE_Pilut"
static int PCDestroy_HYPRE_Pilut(PC pc)
{
  PC_HYPRE *jac = (PC_HYPRE*)pc->data;
  int      ierr;

  PetscFunctionBegin;
  ierr = HYPRE_IJMatrixDestroy(jac->ij);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorDestroy(jac->b);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorDestroy(jac->x);CHKERRQ(ierr);
  ierr = HYPRE_ParCSRPilutDestroy(jac->hsolver);CHKERRQ(ierr);
  ierr = PetscFree(jac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_HYPRE_Pilut"
static int PCSetFromOptions_HYPRE_Pilut(PC pc)
{
  PC_HYPRE  *jac = (PC_HYPRE*)pc->data;
  int        ierr;
  PetscTruth flag;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("HYPRE Pilut Options");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-pc_hypre_pilut_maxiter","Number of iterations","None",jac->maxiter,&jac->maxiter,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = HYPRE_ParCSRPilutSetMaxIter(jac->hsolver,jac->maxiter);CHKERRQ(ierr);
    } 
    ierr = PetscOptionsReal("-pc_hypre_pilut_tol","Drop tolerance","None",jac->tol,&jac->tol,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = HYPRE_ParCSRPilutSetDropTolerance(jac->hsolver,jac->tol);CHKERRQ(ierr);
    } 
    ierr = PetscOptionsInt("-pc_hypre_pilut_factorrowsize","FactorRowSize","None",jac->factorrowsize,&jac->factorrowsize,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = HYPRE_ParCSRPilutSetFactorRowSize(jac->hsolver,jac->factorrowsize);CHKERRQ(ierr);
    } 
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView_HYPRE_Pilut"
static int PCView_HYPRE_Pilut(PC pc,PetscViewer viewer)
{
  PC_HYPRE    *jac = (PC_HYPRE*)pc->data;
  int         ierr;
  PetscTruth  isascii,isstring;
  PetscViewer sviewer;


  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Pilut preconditioning\n");CHKERRQ(ierr);
    if (jac->maxiter != PETSC_DEFAULT) {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Pilut: number of iterations %d\n",jac->maxiter);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Pilut: default number of iterations \n");CHKERRQ(ierr);
    }
    if (jac->tol != PETSC_DEFAULT) {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Pilut: drop tolerance %g\n",jac->tol);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Pilut: default drop tolerance \n");CHKERRQ(ierr);
    }
    if (jac->factorrowsize != PETSC_DEFAULT) {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Pilut: factor row size %d\n",jac->factorrowsize);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Pilut: default factor row size \n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "PCHYPRESetType_HYPRE"
static int PCHYPRESetType_HYPRE(PC pc,char *name)
{
  PC_HYPRE   *jac = (PC_HYPRE*)pc->data;
  int        ierr;
  PetscTruth flag;

  PetscFunctionBegin;
  if (pc->ops->setup) {
    SETERRQ(1,"Cannot set the HYPRE preconditioner type once it has been set");
  }

  ierr = PetscStrcmp("pilut",name,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr                    = HYPRE_ParCSRPilutCreate(pc->comm,&jac->hsolver);
    pc->ops->setup          = PCSetUp_HYPRE_Pilut;
    pc->ops->apply          = PCApply_HYPRE_Pilut;
    pc->ops->setfromoptions = PCSetFromOptions_HYPRE_Pilut;
    pc->ops->destroy        = PCDestroy_HYPRE_Pilut;
    pc->ops->view           = PCView_HYPRE_Pilut;
    jac->maxiter            = PETSC_DEFAULT;
    jac->tol                = PETSC_DEFAULT;
    jac->factorrowsize      = PETSC_DEFAULT;
  }
  PetscFunctionReturn(0);
}

/*
    It only gets here if the HYPRE type has not been set before the call to 
   ...SetFromOptions() which actually is most of the time
*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_HYPRE"
static int PCSetFromOptions_HYPRE(PC pc)
{
  PC_HYPRE  *jac = (PC_HYPRE*)pc->data;
  int        ierr;
  char       buff[32],*type[] = {"pilut","none","none","none"};
  PetscTruth flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("HYPRE preconditioner options");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-pc_hypre_type","HYPRE preconditioner type","PCHYPRESetType",type,4,"pilut",buff,32,&flg);CHKERRQ(ierr);

    
    if (PetscOptionsPublishCount) {   /* force the default if it was not yet set and user did not set with option */
      if (!flg && !pc->ops->apply) {
        flg  = PETSC_TRUE;
        ierr = PetscStrcpy(buff,"pilut");CHKERRQ(ierr);
      }
    }

    if (flg) {
      ierr = PCHYPRESetType_HYPRE(pc,buff);CHKERRQ(ierr);
    } 
    if (pc->ops->setfromoptions) {
      ierr = pc->ops->setfromoptions(pc);CHKERRQ(ierr);
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_HYPRE"
int PCCreate_HYPRE(PC pc)
{
  PC_HYPRE *jac;
  int       ierr;

  PetscFunctionBegin;
  ierr                    = PetscNew(PC_HYPRE,&jac);CHKERRQ(ierr);
  pc->data                = jac;
  pc->ops->setfromoptions = PCSetFromOptions_HYPRE;
  PetscFunctionReturn(0);
}
EXTERN_C_END
