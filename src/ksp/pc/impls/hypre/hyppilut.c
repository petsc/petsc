/*

*/

#include "src/ksp/pc/pcimpl.h"          /*I "petscpc.h" I*/
EXTERN_C_BEGIN
#include "HYPRE.h"
#include "IJ_mv.h"
#include "parcsr_ls.h"
EXTERN_C_END

EXTERN PetscErrorCode MatHYPRE_IJMatrixCreate(Mat,HYPRE_IJMatrix*);
EXTERN PetscErrorCode MatHYPRE_IJMatrixCopy(Mat,HYPRE_IJMatrix);
EXTERN PetscErrorCode VecHYPRE_IJVectorCreate(Vec,HYPRE_IJVector*);

/* 
   Private context (data structure) for the  preconditioner.  
*/
typedef struct {
  HYPRE_Solver       hsolver;
  HYPRE_IJMatrix     ij;
  HYPRE_IJVector     b,x;

  PetscErrorCode (*destroy)(HYPRE_Solver);
  PetscErrorCode (*solve)(HYPRE_Solver,HYPRE_ParCSRMatrix,HYPRE_ParVector,HYPRE_ParVector);
  PetscErrorCode (*setup)(HYPRE_Solver,HYPRE_ParCSRMatrix,HYPRE_ParVector,HYPRE_ParVector);
  
  MPI_Comm          comm_hypre;

  /* options for pilut and BoomerAMG*/
  int                maxiter;
  double             tol;
  PetscTruth         applyrichardson;

  /* options for pilut */
  int                factorrowsize;

  /* options for parasails */
  int                nlevels;
  double             threshhold;
  double             filter;
  int                sym;
  double             loadbal;
  int                logging;
  int                ruse;
  int                symt;

  /* options for euclid */
  PetscTruth         bjilu;
  int                levels;

  /* options for euclid and BoomerAMG */
  PetscTruth         printstatistics;

  /* options for BoomerAMG */
  int                maxlevels;
  double             strongthreshold;
  double             maxrowsum;
  int                gridsweeps[4];
  int                coarsentype;
  int                measuretype;
  int                relaxtype[4];
  double             relaxweight;
  double             outerrelaxweight;
  int                relaxorder;
  int                **gridrelaxpoints;
} PC_HYPRE;


#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_HYPRE"
static PetscErrorCode PCSetUp_HYPRE(PC pc)
{
  PC_HYPRE           *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode     ierr;
  HYPRE_ParCSRMatrix hmat;
  HYPRE_ParVector    bv,xv;
  int                hierr;

  PetscFunctionBegin;
  if (!jac->ij) { /* create the matrix the first time through */ 
    ierr = MatHYPRE_IJMatrixCreate(pc->pmat,&jac->ij);CHKERRQ(ierr);
  }
  if (!jac->b) {
    Vec vec;
    ierr = MatGetVecs(pc->pmat,&vec,0);CHKERRQ(ierr);
    ierr = VecHYPRE_IJVectorCreate(vec,&jac->b);CHKERRQ(ierr);
    ierr = VecHYPRE_IJVectorCreate(vec,&jac->x);CHKERRQ(ierr);
    ierr = VecDestroy(vec);CHKERRQ(ierr);
  }
  ierr = MatHYPRE_IJMatrixCopy(pc->pmat,jac->ij);CHKERRQ(ierr);
  ierr = HYPRE_IJMatrixGetObject(jac->ij,(void**)&hmat);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorGetObject(jac->b,(void**)&bv);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorGetObject(jac->x,(void**)&xv);CHKERRQ(ierr);
  hierr = (*jac->setup)(jac->hsolver,hmat,bv,xv);
  if (hierr) SETERRQ1(PETSC_ERR_LIB,"Error in HYPRE setup, error code %d",hierr);
  PetscFunctionReturn(0);
}

/*
    Replaces the address where the HYPRE vector points to its data with the address of
  PETSc's data. Saves the old address so it can be reset when we are finished with it.
  Allows use to get the data into a HYPRE vector without the cost of memcopies 
*/
#define HYPREReplacePointer(b,newvalue,savedvalue) {\
   hypre_ParVector *par_vector   = (hypre_ParVector *)hypre_IJVectorObject(((hypre_IJVector*)b));\
   hypre_Vector    *local_vector = hypre_ParVectorLocalVector(par_vector);\
   savedvalue         = local_vector->data;\
   local_vector->data = newvalue;}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_HYPRE"
static PetscErrorCode PCApply_HYPRE(PC pc,Vec b,Vec x)
{
  PC_HYPRE           *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode     ierr;
  HYPRE_ParCSRMatrix hmat;
  PetscScalar        *bv,*xv;
  HYPRE_ParVector    jbv,jxv;
  PetscScalar        *sbv,*sxv; 
  PetscScalar        zero=0.0;
  int                hierr;

  PetscFunctionBegin;
  if (!jac->applyrichardson) {ierr = VecSet(&zero,x);CHKERRQ(ierr);}
  ierr = VecGetArray(b,&bv);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xv);CHKERRQ(ierr);
  HYPREReplacePointer(jac->b,bv,sbv);
  HYPREReplacePointer(jac->x,xv,sxv);

  ierr = HYPRE_IJMatrixGetObject(jac->ij,(void**)&hmat);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorGetObject(jac->b,(void**)&jbv);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorGetObject(jac->x,(void**)&jxv);CHKERRQ(ierr);
  hierr = (*jac->solve)(jac->hsolver,hmat,jbv,jxv);
  /* error code of 1 in boomerAMG merely means convergence not achieved */
  if (hierr && (hierr != 1 || jac->solve != HYPRE_BoomerAMGSolve)) SETERRQ1(PETSC_ERR_LIB,"Error in HYPRE solver, error code %d",hierr);
  
  HYPREReplacePointer(jac->b,sbv,bv);
  HYPREReplacePointer(jac->x,sxv,xv);
  ierr = VecRestoreArray(x,&xv);CHKERRQ(ierr);
  ierr = VecRestoreArray(b,&bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_HYPRE"
static PetscErrorCode PCDestroy_HYPRE(PC pc)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = HYPRE_IJMatrixDestroy(jac->ij);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorDestroy(jac->b);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorDestroy(jac->x);CHKERRQ(ierr);
  ierr = MPI_Comm_free(&(jac->comm_hypre));CHKERRQ(ierr);
  ierr = (*jac->destroy)(jac->hsolver);CHKERRQ(ierr);
  ierr = PetscFree(jac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_HYPRE_Pilut"
static PetscErrorCode PCSetFromOptions_HYPRE_Pilut(PC pc)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscTruth     flag;

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
static PetscErrorCode PCView_HYPRE_Pilut(PC pc,PetscViewer viewer)
{
  PC_HYPRE    *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscTruth  iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Pilut preconditioning\n");CHKERRQ(ierr);
    if (jac->maxiter != PETSC_DEFAULT) {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Pilut: maximum number of iterations %d\n",jac->maxiter);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Pilut: default maximum number of iterations \n");CHKERRQ(ierr);
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

/* --------------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_HYPRE_Euclid"
static PetscErrorCode PCSetFromOptions_HYPRE_Euclid(PC pc)
{
  PC_HYPRE  *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscTruth flag;
  char       *args[2];

  PetscFunctionBegin;
  jac->bjilu              = PETSC_FALSE;
  jac->levels             = 1;

  ierr = PetscOptionsHead("HYPRE Euclid Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_hypre_euclid_levels","Number of levels of fill ILU(k)","None",jac->levels,&jac->levels,&flag);CHKERRQ(ierr);
  if (flag) {
    char levels[16];
    if (jac->levels < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Number of levels %d must be nonegative",jac->levels);
    sprintf(levels,"%d",jac->levels);
    args[0] = (char*)"-level"; args[1] = levels;
    ierr = HYPRE_EuclidSetParams(jac->hsolver,2,args);CHKERRQ(ierr);
  } 
  ierr = PetscOptionsLogical("-pc_hypre_euclid_bj","Use block Jacobi ILU(k)","None",jac->bjilu,&jac->bjilu,PETSC_NULL);CHKERRQ(ierr);
  if (jac->bjilu) {
    args[0] =(char*) "-bj"; args[1] = (char*)"1";
    ierr = HYPRE_EuclidSetParams(jac->hsolver,2,args);CHKERRQ(ierr);
  } 
    
  ierr = PetscOptionsLogical("-pc_hypre_euclid_print_statistics","Print statistics","None",jac->printstatistics,&jac->printstatistics,PETSC_NULL);CHKERRQ(ierr);
  if (jac->printstatistics) {
    args[0] = (char*)"-eu_stats"; args[1] = (char*)"1";
    ierr = HYPRE_EuclidSetParams(jac->hsolver,2,args);CHKERRQ(ierr);
    args[0] = (char*)"-eu_mem"; args[1] = (char*)"1";
    ierr = HYPRE_EuclidSetParams(jac->hsolver,2,args);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView_HYPRE_Euclid"
static PetscErrorCode PCView_HYPRE_Euclid(PC pc,PetscViewer viewer)
{
  PC_HYPRE    *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscTruth  iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Euclid preconditioning\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Euclid: number of levels %d\n",jac->levels);CHKERRQ(ierr);
    if (jac->bjilu) {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Euclid: Using block Jacobi ILU instead of parallel ILU\n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------------*/

static const char *HYPREBoomerAMGCoarsenType[] = {"CLJP","Ruge-Stueben","","modifiedRuge-Stueben","","","Falgout"};
static const char *HYPREBoomerAMGMeasureType[] = {"local","global"};
static const char *HYPREBoomerAMGRelaxType[]   = {"Jacobi","sequential-Gauss-Seidel","","SOR/Jacobi","backward-SOR/Jacobi","","symmetric-SOR/Jacobi",
                                                  "","","Gaussian-elimination"};
#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_HYPRE_BoomerAMG"
static PetscErrorCode PCSetFromOptions_HYPRE_BoomerAMG(PC pc)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  int            n,indx;
  PetscTruth     flg, tmp_truth;
  double         tmpdbl, twodbl[2];

  PetscFunctionBegin;
  /* these defaults match the hypre defaults */
  jac->maxlevels        = 25;
  jac->maxiter          = 20;  
  jac->tol              = 1.e-7;
  jac->strongthreshold  = .25;
  jac->maxrowsum        = .9;
  jac->coarsentype      = 6;
  jac->measuretype      = 0;
  jac->applyrichardson  = PETSC_FALSE;  
  jac->gridsweeps[0]    = jac->gridsweeps[1] = jac->gridsweeps[2] = jac->gridsweeps[3]  = 1;
  jac->relaxtype[0]     = jac->relaxtype[1]  = jac->relaxtype[2]  = 3;
  jac->relaxtype[3]     = 9;
  jac->relaxweight      = 1.0;
  jac->outerrelaxweight = 1.0;
  jac->relaxorder       = 1;
  
  ierr = PetscOptionsHead("HYPRE BoomerAMG Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_hypre_boomeramg_max_levels","Number of levels (of grids) allowed","None",jac->maxlevels,&jac->maxlevels,&flg);CHKERRQ(ierr);
  if (flg) {
    if (jac->maxlevels < 2) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Number of levels %d must be at least two",jac->maxlevels);
  }
  ierr = HYPRE_BoomerAMGSetMaxLevels(jac->hsolver,jac->maxlevels);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_hypre_boomeramg_max_iter","Maximum iterations used","None",jac->maxiter,&jac->maxiter,&flg);CHKERRQ(ierr);
  if (flg) {
    if (jac->maxiter < 1) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Number of iterations %d must be at least one",jac->maxiter);
  }
  ierr = HYPRE_BoomerAMGSetMaxIter(jac->hsolver,jac->maxiter);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-pc_hypre_boomeramg_tol","Convergence tolerance","None",jac->tol,&jac->tol,&flg);CHKERRQ(ierr);
  if (flg) {
    if (jac->tol < 0.0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Tolerance %g must be great than or equal zero",jac->tol);
  }
  ierr = HYPRE_BoomerAMGSetTol(jac->hsolver,jac->tol);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-pc_hypre_boomeramg_strong_threshold","Threshold for being strongly connected","None",jac->strongthreshold,&jac->strongthreshold,&flg);CHKERRQ(ierr);
  if (flg) {
    if (jac->strongthreshold < 0.0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Strong threshold %g must be great than or equal zero",jac->strongthreshold);
  }
  ierr = HYPRE_BoomerAMGSetStrongThreshold(jac->hsolver,jac->strongthreshold);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-pc_hypre_boomeramg_max_row_sum","Maximum row sum","None",jac->maxrowsum,&jac->maxrowsum,&flg);CHKERRQ(ierr);
  if (flg) {
    if (jac->maxrowsum < 0.0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Maximum row sum %g must be greater than zero",jac->maxrowsum);
    if (jac->maxrowsum > 1.0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Maximum row sum %g must be less than or equal one",jac->maxrowsum);
  } 
  ierr = HYPRE_BoomerAMGSetMaxRowSum(jac->hsolver,jac->maxrowsum);CHKERRQ(ierr);

  /* Grid sweeps */
  ierr = PetscOptionsInt("-pc_hypre_boomeramg_grid_sweeps_all","Number of sweeps for all grid levels (fine, up, and down)","None", jac->gridsweeps[0], &indx ,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = HYPRE_BoomerAMGSetNumSweeps(jac->hsolver,indx);CHKERRQ(ierr);
    /* modify the jac structure so we can view the updated options with PC_View */ 
    jac->gridsweeps[0] = indx;
    jac->gridsweeps[1] = jac->gridsweeps[2] = jac->gridsweeps[0];
    jac->gridsweeps[3] = 1;  /*The coarse level is not affected by this function - hypre code sets to 1*/
  } 
  ierr = PetscOptionsInt("-pc_hypre_boomeramg_grid_sweeps_fine","Number of sweeps for the fine level","None", jac->gridsweeps[0], &indx ,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = HYPRE_BoomerAMGSetCycleNumSweeps(jac->hsolver,indx, 0);CHKERRQ(ierr);
    jac->gridsweeps[0] = indx;
  }
  ierr = PetscOptionsInt("-pc_hypre_boomeramg_grid_sweeps_down","Number of sweeps for the down cycles","None",jac->gridsweeps[2], &indx ,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = HYPRE_BoomerAMGSetCycleNumSweeps(jac->hsolver,indx, 1);CHKERRQ(ierr);
    jac->gridsweeps[1] = indx;
  }
  ierr = PetscOptionsInt("-pc_hypre_boomeramg_grid_sweeps_up","Number of sweeps for the up cycles","None", jac->gridsweeps[1], &indx ,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = HYPRE_BoomerAMGSetCycleNumSweeps(jac->hsolver,indx, 2);CHKERRQ(ierr);
    jac->gridsweeps[2] = indx;
  }
  ierr = PetscOptionsInt("-pc_hypre_boomeramg_grid_sweeps_coarse","Number of sweeps for the coarse level","None", jac->gridsweeps[3], &indx ,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = HYPRE_BoomerAMGSetCycleNumSweeps(jac->hsolver,indx, 3);CHKERRQ(ierr);
    jac->gridsweeps[3] = indx;
  }

  /* Relax type */
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_relax_type_all","Relax type for fine, up, and down cycles (coarse level set to gaussian elimination)","None",HYPREBoomerAMGRelaxType,10,HYPREBoomerAMGRelaxType[3],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->relaxtype[0] = jac->relaxtype[1] = jac->relaxtype[2] = indx;
    jac->relaxtype[3] = 9; /* hypre code sets coarse grid to 9 (G.E.)*/
    ierr = hypre_BoomerAMGSetRelaxType(jac->hsolver, indx);CHKERRQ(ierr); 
  }
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_relax_type_fine","Relax type on fine grid","None",HYPREBoomerAMGRelaxType,10,HYPREBoomerAMGRelaxType[3],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->relaxtype[0] = indx;
    ierr = HYPRE_BoomerAMGSetCycleRelaxType(jac->hsolver, indx, 0);CHKERRQ(ierr); 
  }
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_relax_type_down","Relax type for the down cycles","None",HYPREBoomerAMGRelaxType,10,HYPREBoomerAMGRelaxType[3],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->relaxtype[1] = indx;
    ierr = HYPRE_BoomerAMGSetCycleRelaxType(jac->hsolver, indx, 1);CHKERRQ(ierr); 
  }
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_relax_type_up","Relax type for the up cycles","None",HYPREBoomerAMGRelaxType,10,HYPREBoomerAMGRelaxType[3],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->relaxtype[2] = indx;
    ierr = HYPRE_BoomerAMGSetCycleRelaxType(jac->hsolver, indx, 2);CHKERRQ(ierr); 
  }
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_relax_type_coarse","Relax type on coarse grid","None",HYPREBoomerAMGRelaxType,10,HYPREBoomerAMGRelaxType[9],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->relaxtype[3] = indx;
    ierr = HYPRE_BoomerAMGSetCycleRelaxType(jac->hsolver, indx, 3);CHKERRQ(ierr); 
  }

  /* Relaxation Weight */
  ierr = PetscOptionsReal("-pc_hypre_boomeramg_relax_weight_all","Relaxation weight for all levels (0 = hypre estimates, -k = determined with k CG steps)","None",jac->relaxweight, &tmpdbl ,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = hypre_BoomerAMGSetRelaxWt( jac->hsolver, tmpdbl);CHKERRQ(ierr); 
    jac->relaxweight = tmpdbl;
  }

  n=2;
  twodbl[0] = twodbl[1] = 1.0;
  ierr = PetscOptionsRealArray("-pc_hypre_boomeramg_relax_weight_level","Set the relaxation weight for a particular level (weight,level)","None",twodbl, &n, &flg);CHKERRQ(ierr);
  if (flg) {
    if (n == 2) {
      indx =  PetscAbsReal(twodbl[1]); 
      ierr = hypre_BoomerAMGSetLevelRelaxWt( jac->hsolver, twodbl[0], indx);CHKERRQ(ierr); 
    } else {
      SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Relax weight level: you must provide 2 values seperated by a comma (and no space), you provided %d",n);
    }
  }

  /* Outer relaxation Weight */
  ierr = PetscOptionsReal("-pc_hypre_boomeramg_outer_relax_weight_all","Outer relaxation weight for all levels ( -k = determined with k CG steps)","None",jac->outerrelaxweight, &tmpdbl ,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = hypre_BoomerAMGSetOuterWt( jac->hsolver, tmpdbl);CHKERRQ(ierr); 
    jac->outerrelaxweight = tmpdbl;
  }

  n=2;
  twodbl[0] = twodbl[1] = 1.0;
  ierr = PetscOptionsRealArray("-pc_hypre_boomeramg_outer_relax_weight_level","Set the outer relaxation weight for a particular level (weight,level)","None",twodbl, &n, &flg);CHKERRQ(ierr);
  if (flg) {
    if (n == 2) {
      indx =  PetscAbsReal(twodbl[1]); 
      ierr = hypre_BoomerAMGSetLevelOuterWt( jac->hsolver, twodbl[0], indx);CHKERRQ(ierr); 
    } else {
      SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Relax weight outer level: You must provide 2 values seperated by a comma (and no space), you provided %d",n);
    }
  }

  /* the Relax Order */ 
  /* ierr = PetscOptionsName("-pc_hypre_boomeramg_no_CF", "Do not use CF-relaxation", "None", &flg);CHKERRQ(ierr); */
  ierr = PetscOptionsLogical( "-pc_hypre_boomeramg_no_CF", "Do not use CF-relaxation", "None", PETSC_FALSE, &tmp_truth, &flg);CHKERRQ(ierr);

  if (flg) {
    jac->relaxorder = 0;
    ierr = hypre_BoomerAMGSetRelaxOrder( jac->hsolver, jac->relaxorder);CHKERRQ(ierr); 
  }
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_measure_type","Measure type","None",HYPREBoomerAMGMeasureType,2,HYPREBoomerAMGMeasureType[0],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->measuretype = indx;
  }
  ierr = HYPRE_BoomerAMGSetMeasureType(jac->hsolver,jac->measuretype);CHKERRQ(ierr); 
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_coarsen_type","Coarsen type","None",HYPREBoomerAMGCoarsenType,7,HYPREBoomerAMGCoarsenType[6],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->coarsentype = indx;
  }
  ierr = HYPRE_BoomerAMGSetCoarsenType(jac->hsolver,jac->coarsentype);CHKERRQ(ierr); 
  ierr = PetscOptionsLogical("-pc_hypre_boomeramg_print_statistics","Print statistics","None",jac->printstatistics,&jac->printstatistics,PETSC_NULL);CHKERRQ(ierr);
  if (jac->printstatistics) {
    ierr = HYPRE_BoomerAMGSetPrintLevel(jac->hsolver,3);CHKERRQ(ierr);
    ierr = HYPRE_BoomerAMGSetDebugFlag(jac->hsolver,3);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyRichardson_BoomerAMG"
static PetscErrorCode PCApplyRichardson_BoomerAMG(PC pc,Vec b,Vec y,Vec w,PetscReal rtol,PetscReal abstol, PetscReal dtol,PetscInt its)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscLogInfo(pc,"PCApplyRichardson_hypre_BoomerAMG: Warning, convergence critera ignored, using %D iterations\n",its);
  ierr = HYPRE_BoomerAMGSetMaxIter(jac->hsolver,its);CHKERRQ(ierr);
  ierr = HYPRE_BoomerAMGSetTol(jac->hsolver,rtol);CHKERRQ(ierr);
  jac->applyrichardson = PETSC_TRUE;
  ierr = PCApply_HYPRE(pc,b,y);CHKERRQ(ierr);
  jac->applyrichardson = PETSC_FALSE;
  ierr = HYPRE_BoomerAMGSetTol(jac->hsolver,jac->tol);CHKERRQ(ierr);
  ierr = HYPRE_BoomerAMGSetMaxIter(jac->hsolver,jac->maxiter);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PCView_HYPRE_BoomerAMG"
static PetscErrorCode PCView_HYPRE_BoomerAMG(PC pc,PetscViewer viewer)
{
  PC_HYPRE    *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscTruth  iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG preconditioning\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Maximum number of levels %d\n",jac->maxlevels);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Maximum number of iterations %d\n",jac->maxiter);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Convergence tolerance %g\n",jac->tol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Threshold for strong coupling %g\n",jac->strongthreshold);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Maximum row sums %g\n",jac->maxrowsum);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Sweeps on fine grid %d\n",jac->gridsweeps[0]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Sweeps down         %d\n",jac->gridsweeps[1]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Sweeps up           %d\n",jac->gridsweeps[2]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Sweeps on coarse    %d\n",jac->gridsweeps[3]);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Relax on fine grid  %s\n",HYPREBoomerAMGRelaxType[jac->relaxtype[0]]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Relax down          %s\n",HYPREBoomerAMGRelaxType[jac->relaxtype[1]]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Relax up            %s\n",HYPREBoomerAMGRelaxType[jac->relaxtype[2]]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Relax on coarse     %s\n",HYPREBoomerAMGRelaxType[jac->relaxtype[3]]);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Relax weight  (all)      %g\n",jac->relaxweight);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Outer relax weight (all) %g\n",jac->outerrelaxweight);CHKERRQ(ierr); 

    if (jac->relaxorder) {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Using CF-relaxation\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Not using CF-relaxation\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Measure type        %s\n",HYPREBoomerAMGMeasureType[jac->measuretype]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Coarsen type        %s\n",HYPREBoomerAMGCoarsenType[jac->coarsentype]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_HYPRE_ParaSails"
static PetscErrorCode PCSetFromOptions_HYPRE_ParaSails(PC pc)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  int            indx;
  PetscTruth     flag;
  const char     *symtlist[] = {"nonsymmetric","SPD","nonsymmetric,SPD"};

  PetscFunctionBegin;
  jac->nlevels     = 1;
  jac->threshhold  = .1;
  jac->filter      = .1;
  jac->loadbal     = 0;
  if (PetscLogPrintInfo) {
    jac->logging     = (int) PETSC_TRUE;
  } else {
    jac->logging     = (int) PETSC_FALSE;
  }
  jac->ruse = (int) PETSC_FALSE;
  jac->symt   = 0;

  ierr = PetscOptionsHead("HYPRE ParaSails Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_hypre_parasails_nlevels","Number of number of levels","None",jac->nlevels,&jac->nlevels,0);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_hypre_parasails_thresh","Threshold","None",jac->threshhold,&jac->threshhold,0);CHKERRQ(ierr);
  ierr = HYPRE_ParaSailsSetParams(jac->hsolver,jac->threshhold,jac->nlevels);CHKERRQ(ierr);

  ierr = PetscOptionsReal("-pc_hypre_parasails_filter","filter","None",jac->filter,&jac->filter,0);CHKERRQ(ierr);
  ierr = HYPRE_ParaSailsSetFilter(jac->hsolver,jac->filter);CHKERRQ(ierr);

  ierr = PetscOptionsReal("-pc_hypre_parasails_loadbal","Load balance","None",jac->loadbal,&jac->loadbal,0);CHKERRQ(ierr);
  ierr = HYPRE_ParaSailsSetLoadbal(jac->hsolver,jac->loadbal);CHKERRQ(ierr);

  ierr = PetscOptionsLogical("-pc_hypre_parasails_logging","Print info to screen","None",(PetscTruth)jac->logging,(PetscTruth*)&jac->logging,0);CHKERRQ(ierr);
  ierr = HYPRE_ParaSailsSetLogging(jac->hsolver,jac->logging);CHKERRQ(ierr);

  ierr = PetscOptionsLogical("-pc_hypre_parasails_reuse","Reuse nonzero pattern in preconditioner","None",(PetscTruth)jac->ruse,(PetscTruth*)&jac->ruse,0);CHKERRQ(ierr);
  ierr = HYPRE_ParaSailsSetReuse(jac->hsolver,jac->ruse);CHKERRQ(ierr);

  ierr = PetscOptionsEList("-pc_hypre_parasails_sym","Symmetry of matrix and preconditioner","None",symtlist,3,symtlist[0],&indx,&flag);CHKERRQ(ierr);
  if (flag) {
    jac->symt = indx;
  }
  ierr = HYPRE_ParaSailsSetSym(jac->hsolver,jac->symt);CHKERRQ(ierr);

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView_HYPRE_ParaSails"
static PetscErrorCode PCView_HYPRE_ParaSails(PC pc,PetscViewer viewer)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscTruth     iascii;
  const char     *symt;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails preconditioning\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails: nlevels %d\n",jac->nlevels);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails: threshold %g\n",jac->threshhold);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails: filter %g\n",jac->filter);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails: load balance %g\n",jac->loadbal);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails: reuse nonzero structure %s\n",jac->ruse ? "true" : "false");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails: print info to screen %s\n",jac->logging ? "true" : "false");CHKERRQ(ierr);
    if (!jac->symt) {
      symt = "nonsymmetric matrix and preconditioner";
    } else if (jac->symt == 1) {
      symt = "SPD matrix and preconditioner";
    } else if (jac->symt == 2) {
      symt = "nonsymmetric matrix but SPD preconditioner";
    } else {
      SETERRQ1(PETSC_ERR_ARG_WRONG,"Unknown HYPRE ParaSails symmetric option %d",jac->symt);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails: %s\n",symt);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "PCHYPRESetType_HYPRE"
static PetscErrorCode PCHYPRESetType_HYPRE(PC pc,const char name[])
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscTruth     flag;

  PetscFunctionBegin;
  if (pc->ops->setup) {
    SETERRQ(PETSC_ERR_ORDER,"Cannot set the HYPRE preconditioner type once it has been set");
  }

  pc->ops->setup          = PCSetUp_HYPRE;
  pc->ops->apply          = PCApply_HYPRE;
  pc->ops->destroy        = PCDestroy_HYPRE;

  jac->maxiter            = PETSC_DEFAULT;
  jac->tol                = PETSC_DEFAULT;
  jac->printstatistics    = PetscLogPrintInfo;

  ierr = PetscStrcmp("pilut",name,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr                    = HYPRE_ParCSRPilutCreate(jac->comm_hypre,&jac->hsolver);
    pc->ops->setfromoptions = PCSetFromOptions_HYPRE_Pilut;
    pc->ops->view           = PCView_HYPRE_Pilut;
    jac->destroy            = HYPRE_ParCSRPilutDestroy;
    jac->setup              = HYPRE_ParCSRPilutSetup;
    jac->solve              = HYPRE_ParCSRPilutSolve;
    jac->factorrowsize      = PETSC_DEFAULT;
    PetscFunctionReturn(0);
  }
  ierr = PetscStrcmp("parasails",name,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr                    = HYPRE_ParaSailsCreate(jac->comm_hypre,&jac->hsolver);
    pc->ops->setfromoptions = PCSetFromOptions_HYPRE_ParaSails;
    pc->ops->view           = PCView_HYPRE_ParaSails;
    jac->destroy            = HYPRE_ParaSailsDestroy;
    jac->setup              = HYPRE_ParaSailsSetup;
    jac->solve              = HYPRE_ParaSailsSolve;
    PetscFunctionReturn(0);
  }
  ierr = PetscStrcmp("euclid",name,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr                    = HYPRE_EuclidCreate(jac->comm_hypre,&jac->hsolver);
    pc->ops->setfromoptions = PCSetFromOptions_HYPRE_Euclid;
    pc->ops->view           = PCView_HYPRE_Euclid;
    jac->destroy            = HYPRE_EuclidDestroy;
    jac->setup              = HYPRE_EuclidSetup;
    jac->solve              = HYPRE_EuclidSolve;
    PetscFunctionReturn(0);
  }
  ierr = PetscStrcmp("boomeramg",name,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr                     = HYPRE_BoomerAMGCreate(&jac->hsolver);
    pc->ops->setfromoptions  = PCSetFromOptions_HYPRE_BoomerAMG;
    pc->ops->view            = PCView_HYPRE_BoomerAMG;
    jac->destroy             = HYPRE_BoomerAMGDestroy;
    jac->setup               = HYPRE_BoomerAMGSetup;
    jac->solve               = HYPRE_BoomerAMGSolve;
    pc->ops->applyrichardson = PCApplyRichardson_BoomerAMG;
    PetscFunctionReturn(0);
  }
  SETERRQ1(PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown HYPRE preconditioner %s; Choices are pilut, parasails, euclid, boomeramg",name);
  PetscFunctionReturn(0);
}

/*
    It only gets here if the HYPRE type has not been set before the call to 
   ...SetFromOptions() which actually is most of the time
*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_HYPRE"
static PetscErrorCode PCSetFromOptions_HYPRE(PC pc)
{
  PetscErrorCode ierr;
  int            indx;
  const char     *type[] = {"pilut","parasails","boomeramg","euclid"};
  PetscTruth     flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("HYPRE preconditioner options");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-pc_hypre_type","HYPRE preconditioner type","PCHYPRESetType",type,4,"pilut",&indx,&flg);CHKERRQ(ierr);
  if (PetscOptionsPublishCount) {   /* force the default if it was not yet set and user did not set with option */
    if (!flg && !pc->ops->apply) {
      flg   = PETSC_TRUE;
      indx = 0;
    }
  }
  if (flg) {
    ierr = PCHYPRESetType_HYPRE(pc,type[indx]);CHKERRQ(ierr);
  } 
  if (pc->ops->setfromoptions) {
    ierr = pc->ops->setfromoptions(pc);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCHYPRESetType"
/*@C
     PCHYPRESetType - Sets which hypre preconditioner you wish to use

   Input Parameters:
+     pc - the preconditioner context
-     name - either  pilut, parasails, boomeramg, euclid

   Options Database Keys:
   -pc_hypre_type - One of pilut, parasails, boomeramg, euclid
 
   Level: intermediate

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCHYPRE

@*/
PetscErrorCode PCHYPRESetType(PC pc,const char name[])
{
  PetscErrorCode ierr,(*f)(PC,const char[]);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidCharPointer(name,2);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCHYPRESetType_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,name);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

/*MC
     PCHYPRE - Allows you to use the matrix element based preconditioners in the LLNL package hypre

   Options Database Keys:
+   -pc_hypre_type - One of pilut, parasails, boomerAMG, euclid
-   Too many others to list, run with -pc_type hypre -pc_hypre_type XXX -help to see options for the XXX
          preconditioner
 
   Level: intermediate

   Notes: Apart from pc_hypre_type (for which there is PCHYPRESetType()),
          the many hypre options can ONLY be set via the options database (e.g. the command line
          or with PetscOptionsSetValue(), there are no functions to set them)

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCHYPRESetType()

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_HYPRE"
PetscErrorCode PCCreate_HYPRE(PC pc)
{
  PC_HYPRE       *jac;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr                     = PetscNew(PC_HYPRE,&jac);CHKERRQ(ierr);
  pc->data                 = jac;
  pc->ops->setfromoptions  = PCSetFromOptions_HYPRE;
  /* Com_dup for hypre */
  ierr = MPI_Comm_dup(pc->comm,&(jac->comm_hypre));CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCHYPRESetType_C","PCHYPRESetType_HYPRE",PCHYPRESetType_HYPRE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
