#include <petsc/private/pcimpl.h>     /*I "petscpc.h" I*/
#include <petsc/private/kspimpl.h>    /*  This is needed to provide the appropriate PETSC_EXTERN for KSP_Solve_FS ....*/
#include <petscdm.h>

const char *const PCFieldSplitSchurPreTypes[] = {"SELF","SELFP","A11","USER","FULL","PCFieldSplitSchurPreType","PC_FIELDSPLIT_SCHUR_PRE_",0};
const char *const PCFieldSplitSchurFactTypes[] = {"DIAG","LOWER","UPPER","FULL","PCFieldSplitSchurFactType","PC_FIELDSPLIT_SCHUR_FACT_",0};

PetscLogEvent KSP_Solve_FS_0,KSP_Solve_FS_1,KSP_Solve_FS_S,KSP_Solve_FS_U,KSP_Solve_FS_L,KSP_Solve_FS_2,KSP_Solve_FS_3,KSP_Solve_FS_4;

typedef struct _PC_FieldSplitLink *PC_FieldSplitLink;
struct _PC_FieldSplitLink {
  KSP               ksp;
  Vec               x,y,z;
  char              *splitname;
  PetscInt          nfields;
  PetscInt          *fields,*fields_col;
  VecScatter        sctx;
  IS                is,is_col;
  PC_FieldSplitLink next,previous;
  PetscLogEvent     event;
};

typedef struct {
  PCCompositeType type;
  PetscBool       defaultsplit;                    /* Flag for a system with a set of 'k' scalar fields with the same layout (and bs = k) */
  PetscBool       splitdefined;                    /* Flag is set after the splits have been defined, to prevent more splits from being added */
  PetscInt        bs;                              /* Block size for IS and Mat structures */
  PetscInt        nsplits;                         /* Number of field divisions defined */
  Vec             *x,*y,w1,w2;
  Mat             *mat;                            /* The diagonal block for each split */
  Mat             *pmat;                           /* The preconditioning diagonal block for each split */
  Mat             *Afield;                         /* The rows of the matrix associated with each split */
  PetscBool       issetup;

  /* Only used when Schur complement preconditioning is used */
  Mat                       B;                     /* The (0,1) block */
  Mat                       C;                     /* The (1,0) block */
  Mat                       schur;                 /* The Schur complement S = A11 - A10 A00^{-1} A01, the KSP here, kspinner, is H_1 in [El08] */
  Mat                       schurp;                /* Assembled approximation to S built by MatSchurComplement to be used as a preconditioning matrix when solving with S */
  Mat                       schur_user;            /* User-provided preconditioning matrix for the Schur complement */
  PCFieldSplitSchurPreType  schurpre;              /* Determines which preconditioning matrix is used for the Schur complement */
  PCFieldSplitSchurFactType schurfactorization;
  KSP                       kspschur;              /* The solver for S */
  KSP                       kspupper;              /* The solver for A in the upper diagonal part of the factorization (H_2 in [El08]) */
  PetscScalar               schurscale;            /* Scaling factor for the Schur complement solution with DIAG factorization */

  /* Only used when Golub-Kahan bidiagonalization preconditioning is used */
  Mat                       H;                     /* The modified matrix H = A00 + nu*A01*A01'              */
  PetscReal                 gkbtol;                /* Stopping tolerance for lower bound estimate            */
  PetscInt                  gkbdelay;              /* The delay window for the stopping criterion            */
  PetscReal                 gkbnu;                 /* Parameter for augmented Lagrangian H = A + nu*A01*A01' */
  PetscInt                  gkbmaxit;              /* Maximum number of iterations for outer loop            */
  PetscBool                 gkbmonitor;            /* Monitor for gkb iterations and the lower bound error   */
  PetscViewer               gkbviewer;             /* Viewer context for gkbmonitor                          */
  Vec                       u,v,d,Hu;              /* Work vectors for the GKB algorithm                     */
  PetscScalar               *vecz;                 /* Contains intermediate values, eg for lower bound       */

  PC_FieldSplitLink         head;
  PetscBool                 isrestrict;             /* indicates PCFieldSplitRestrictIS() has been last called on this object, hack */
  PetscBool                 suboptionsset;          /* Indicates that the KSPSetFromOptions() has been called on the sub-KSPs */
  PetscBool                 dm_splits;              /* Whether to use DMCreateFieldDecomposition() whenever possible */
  PetscBool                 diag_use_amat;          /* Whether to extract diagonal matrix blocks from Amat, rather than Pmat (weaker than -pc_use_amat) */
  PetscBool                 offdiag_use_amat;       /* Whether to extract off-diagonal matrix blocks from Amat, rather than Pmat (weaker than -pc_use_amat) */
  PetscBool                 detect;                 /* Whether to form 2-way split by finding zero diagonal entries */
} PC_FieldSplit;

/*
    Notes:
    there is no particular reason that pmat, x, and y are stored as arrays in PC_FieldSplit instead of
   inside PC_FieldSplitLink, just historical. If you want to be able to add new fields after already using the
   PC you could change this.
*/

/* This helper is so that setting a user-provided preconditioning matrix is orthogonal to choosing to use it.  This way the
* application-provided FormJacobian can provide this matrix without interfering with the user's (command-line) choices. */
static Mat FieldSplitSchurPre(PC_FieldSplit *jac)
{
  switch (jac->schurpre) {
  case PC_FIELDSPLIT_SCHUR_PRE_SELF: return jac->schur;
  case PC_FIELDSPLIT_SCHUR_PRE_SELFP: return jac->schurp;
  case PC_FIELDSPLIT_SCHUR_PRE_A11: return jac->pmat[1];
  case PC_FIELDSPLIT_SCHUR_PRE_FULL: /* We calculate this and store it in schur_user */
  case PC_FIELDSPLIT_SCHUR_PRE_USER: /* Use a user-provided matrix if it is given, otherwise diagonal block */
  default:
    return jac->schur_user ? jac->schur_user : jac->pmat[1];
  }
}


#include <petscdraw.h>
static PetscErrorCode PCView_FieldSplit(PC pc,PetscViewer viewer)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PetscBool         iascii,isdraw;
  PetscInt          i,j;
  PC_FieldSplitLink ilink = jac->head;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  if (iascii) {
    if (jac->bs > 0) {
      ierr = PetscViewerASCIIPrintf(viewer,"  FieldSplit with %s composition: total splits = %D, blocksize = %D\n",PCCompositeTypes[jac->type],jac->nsplits,jac->bs);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  FieldSplit with %s composition: total splits = %D\n",PCCompositeTypes[jac->type],jac->nsplits);CHKERRQ(ierr);
    }
    if (pc->useAmat) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using Amat (not Pmat) as operator for blocks\n");CHKERRQ(ierr);
    }
    if (jac->diag_use_amat) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using Amat (not Pmat) as operator for diagonal blocks\n");CHKERRQ(ierr);
    }
    if (jac->offdiag_use_amat) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using Amat (not Pmat) as operator for off-diagonal blocks\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  Solver info for each split is in the following KSP objects:\n");CHKERRQ(ierr);
    for (i=0; i<jac->nsplits; i++) {
      if (ilink->fields) {
        ierr = PetscViewerASCIIPrintf(viewer,"Split number %D Fields ",i);CHKERRQ(ierr);
        ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
        for (j=0; j<ilink->nfields; j++) {
          if (j > 0) {
            ierr = PetscViewerASCIIPrintf(viewer,",");CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer," %D",ilink->fields[j]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"Split number %D Defined by IS\n",i);CHKERRQ(ierr);
      }
      ierr  = KSPView(ilink->ksp,viewer);CHKERRQ(ierr);
      ilink = ilink->next;
    }
  }

 if (isdraw) {
    PetscDraw draw;
    PetscReal x,y,w,wd;

    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawGetCurrentPoint(draw,&x,&y);CHKERRQ(ierr);
    w    = 2*PetscMin(1.0 - x,x);
    wd   = w/(jac->nsplits + 1);
    x    = x - wd*(jac->nsplits-1)/2.0;
    for (i=0; i<jac->nsplits; i++) {
      ierr  = PetscDrawPushCurrentPoint(draw,x,y);CHKERRQ(ierr);
      ierr  = KSPView(ilink->ksp,viewer);CHKERRQ(ierr);
      ierr  = PetscDrawPopCurrentPoint(draw);CHKERRQ(ierr);
      x    += wd;
      ilink = ilink->next;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_FieldSplit_Schur(PC pc,PetscViewer viewer)
{
  PC_FieldSplit              *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode             ierr;
  PetscBool                  iascii,isdraw;
  PetscInt                   i,j;
  PC_FieldSplitLink          ilink = jac->head;
  MatSchurComplementAinvType atype;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  if (iascii) {
    if (jac->bs > 0) {
      ierr = PetscViewerASCIIPrintf(viewer,"  FieldSplit with Schur preconditioner, blocksize = %D, factorization %s\n",jac->bs,PCFieldSplitSchurFactTypes[jac->schurfactorization]);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  FieldSplit with Schur preconditioner, factorization %s\n",PCFieldSplitSchurFactTypes[jac->schurfactorization]);CHKERRQ(ierr);
    }
    if (pc->useAmat) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using Amat (not Pmat) as operator for blocks\n");CHKERRQ(ierr);
    }
    switch (jac->schurpre) {
    case PC_FIELDSPLIT_SCHUR_PRE_SELF:
      ierr = PetscViewerASCIIPrintf(viewer,"  Preconditioner for the Schur complement formed from S itself\n");CHKERRQ(ierr);
      break;
    case PC_FIELDSPLIT_SCHUR_PRE_SELFP:
      ierr = MatSchurComplementGetAinvType(jac->schur,&atype);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Preconditioner for the Schur complement formed from Sp, an assembled approximation to S, which uses A00's %sdiagonal's inverse\n",atype == MAT_SCHUR_COMPLEMENT_AINV_DIAG ? "" : (atype == MAT_SCHUR_COMPLEMENT_AINV_BLOCK_DIAG ? "block " : "lumped "));CHKERRQ(ierr);break;
    case PC_FIELDSPLIT_SCHUR_PRE_A11:
      ierr = PetscViewerASCIIPrintf(viewer,"  Preconditioner for the Schur complement formed from A11\n");CHKERRQ(ierr);
      break;
    case PC_FIELDSPLIT_SCHUR_PRE_FULL:
      ierr = PetscViewerASCIIPrintf(viewer,"  Preconditioner for the Schur complement formed from the exact Schur complement\n");CHKERRQ(ierr);
      break;
    case PC_FIELDSPLIT_SCHUR_PRE_USER:
      if (jac->schur_user) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Preconditioner for the Schur complement formed from user provided matrix\n");CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  Preconditioner for the Schur complement formed from A11\n");CHKERRQ(ierr);
      }
      break;
    default:
      SETERRQ1(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_OUTOFRANGE, "Invalid Schur preconditioning type: %d", jac->schurpre);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  Split info:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    for (i=0; i<jac->nsplits; i++) {
      if (ilink->fields) {
        ierr = PetscViewerASCIIPrintf(viewer,"Split number %D Fields ",i);CHKERRQ(ierr);
        ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
        for (j=0; j<ilink->nfields; j++) {
          if (j > 0) {
            ierr = PetscViewerASCIIPrintf(viewer,",");CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer," %D",ilink->fields[j]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"Split number %D Defined by IS\n",i);CHKERRQ(ierr);
      }
      ilink = ilink->next;
    }
    ierr = PetscViewerASCIIPrintf(viewer,"KSP solver for A00 block\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    if (jac->head) {
      ierr = KSPView(jac->head->ksp,viewer);CHKERRQ(ierr);
    } else  {ierr = PetscViewerASCIIPrintf(viewer,"  not yet available\n");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    if (jac->head && jac->kspupper != jac->head->ksp) {
      ierr = PetscViewerASCIIPrintf(viewer,"KSP solver for upper A00 in upper triangular factor \n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      if (jac->kspupper) {ierr = KSPView(jac->kspupper,viewer);CHKERRQ(ierr);}
      else {ierr = PetscViewerASCIIPrintf(viewer,"  not yet available\n");CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"KSP solver for S = A11 - A10 inv(A00) A01 \n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    if (jac->kspschur) {
      ierr = KSPView(jac->kspschur,viewer);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  not yet available\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else if (isdraw && jac->head) {
    PetscDraw draw;
    PetscReal x,y,w,wd,h;
    PetscInt  cnt = 2;
    char      str[32];

    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawGetCurrentPoint(draw,&x,&y);CHKERRQ(ierr);
    if (jac->kspupper != jac->head->ksp) cnt++;
    w  = 2*PetscMin(1.0 - x,x);
    wd = w/(cnt + 1);

    ierr = PetscSNPrintf(str,32,"Schur fact. %s",PCFieldSplitSchurFactTypes[jac->schurfactorization]);CHKERRQ(ierr);
    ierr = PetscDrawStringBoxed(draw,x,y,PETSC_DRAW_RED,PETSC_DRAW_BLACK,str,NULL,&h);CHKERRQ(ierr);
    y   -= h;
    if (jac->schurpre == PC_FIELDSPLIT_SCHUR_PRE_USER &&  !jac->schur_user) {
      ierr = PetscSNPrintf(str,32,"Prec. for Schur from %s",PCFieldSplitSchurPreTypes[PC_FIELDSPLIT_SCHUR_PRE_A11]);CHKERRQ(ierr);
    } else {
      ierr = PetscSNPrintf(str,32,"Prec. for Schur from %s",PCFieldSplitSchurPreTypes[jac->schurpre]);CHKERRQ(ierr);
    }
    ierr = PetscDrawStringBoxed(draw,x+wd*(cnt-1)/2.0,y,PETSC_DRAW_RED,PETSC_DRAW_BLACK,str,NULL,&h);CHKERRQ(ierr);
    y   -= h;
    x    = x - wd*(cnt-1)/2.0;

    ierr = PetscDrawPushCurrentPoint(draw,x,y);CHKERRQ(ierr);
    ierr = KSPView(jac->head->ksp,viewer);CHKERRQ(ierr);
    ierr = PetscDrawPopCurrentPoint(draw);CHKERRQ(ierr);
    if (jac->kspupper != jac->head->ksp) {
      x   += wd;
      ierr = PetscDrawPushCurrentPoint(draw,x,y);CHKERRQ(ierr);
      ierr = KSPView(jac->kspupper,viewer);CHKERRQ(ierr);
      ierr = PetscDrawPopCurrentPoint(draw);CHKERRQ(ierr);
    }
    x   += wd;
    ierr = PetscDrawPushCurrentPoint(draw,x,y);CHKERRQ(ierr);
    ierr = KSPView(jac->kspschur,viewer);CHKERRQ(ierr);
    ierr = PetscDrawPopCurrentPoint(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_FieldSplit_GKB(PC pc,PetscViewer viewer)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PetscBool         iascii,isdraw;
  PetscInt          i,j;
  PC_FieldSplitLink ilink = jac->head;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  if (iascii) {
    if (jac->bs > 0) {
      ierr = PetscViewerASCIIPrintf(viewer,"  FieldSplit with %s composition: total splits = %D, blocksize = %D\n",PCCompositeTypes[jac->type],jac->nsplits,jac->bs);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  FieldSplit with %s composition: total splits = %D\n",PCCompositeTypes[jac->type],jac->nsplits);CHKERRQ(ierr);
    }
    if (pc->useAmat) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using Amat (not Pmat) as operator for blocks\n");CHKERRQ(ierr);
    }
    if (jac->diag_use_amat) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using Amat (not Pmat) as operator for diagonal blocks\n");CHKERRQ(ierr);
    }
    if (jac->offdiag_use_amat) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using Amat (not Pmat) as operator for off-diagonal blocks\n");CHKERRQ(ierr);
    }

    ierr = PetscViewerASCIIPrintf(viewer,"  Stopping tolerance=%.1e, delay in error estimate=%D, maximum iterations=%D\n",jac->gkbtol,jac->gkbdelay,jac->gkbmaxit);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Solver info for H = A00 + nu*A01*A01' matrix:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);

    if (ilink->fields) {
      ierr = PetscViewerASCIIPrintf(viewer,"Split number %D Fields ",0);CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
      for (j=0; j<ilink->nfields; j++) {
        if (j > 0) {
          ierr = PetscViewerASCIIPrintf(viewer,",");CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer," %D",ilink->fields[j]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
    } else {
        ierr = PetscViewerASCIIPrintf(viewer,"Split number %D Defined by IS\n",0);CHKERRQ(ierr);
    }
    ierr  = KSPView(ilink->ksp,viewer);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }

 if (isdraw) {
    PetscDraw draw;
    PetscReal x,y,w,wd;

    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawGetCurrentPoint(draw,&x,&y);CHKERRQ(ierr);
    w    = 2*PetscMin(1.0 - x,x);
    wd   = w/(jac->nsplits + 1);
    x    = x - wd*(jac->nsplits-1)/2.0;
    for (i=0; i<jac->nsplits; i++) {
      ierr  = PetscDrawPushCurrentPoint(draw,x,y);CHKERRQ(ierr);
      ierr  = KSPView(ilink->ksp,viewer);CHKERRQ(ierr);
      ierr  = PetscDrawPopCurrentPoint(draw);CHKERRQ(ierr);
      x    += wd;
      ilink = ilink->next;
    }
  }
  PetscFunctionReturn(0);
}


/* Precondition: jac->bs is set to a meaningful value */
static PetscErrorCode PCFieldSplitSetRuntimeSplits_Private(PC pc)
{
  PetscErrorCode ierr;
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;
  PetscInt       i,nfields,*ifields,nfields_col,*ifields_col;
  PetscBool      flg,flg_col;
  char           optionname[128],splitname[8],optionname_col[128];

  PetscFunctionBegin;
  ierr = PetscMalloc1(jac->bs,&ifields);CHKERRQ(ierr);
  ierr = PetscMalloc1(jac->bs,&ifields_col);CHKERRQ(ierr);
  for (i=0,flg=PETSC_TRUE;; i++) {
    ierr        = PetscSNPrintf(splitname,sizeof(splitname),"%D",i);CHKERRQ(ierr);
    ierr        = PetscSNPrintf(optionname,sizeof(optionname),"-pc_fieldsplit_%D_fields",i);CHKERRQ(ierr);
    ierr        = PetscSNPrintf(optionname_col,sizeof(optionname_col),"-pc_fieldsplit_%D_fields_col",i);CHKERRQ(ierr);
    nfields     = jac->bs;
    nfields_col = jac->bs;
    ierr        = PetscOptionsGetIntArray(((PetscObject)pc)->options,((PetscObject)pc)->prefix,optionname,ifields,&nfields,&flg);CHKERRQ(ierr);
    ierr        = PetscOptionsGetIntArray(((PetscObject)pc)->options,((PetscObject)pc)->prefix,optionname_col,ifields_col,&nfields_col,&flg_col);CHKERRQ(ierr);
    if (!flg) break;
    else if (flg && !flg_col) {
      if (!nfields) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot list zero fields");
      ierr = PCFieldSplitSetFields(pc,splitname,nfields,ifields,ifields);CHKERRQ(ierr);
    } else {
      if (!nfields || !nfields_col) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot list zero fields");
      if (nfields != nfields_col) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Number of row and column fields must match");
      ierr = PCFieldSplitSetFields(pc,splitname,nfields,ifields,ifields_col);CHKERRQ(ierr);
    }
  }
  if (i > 0) {
    /* Makes command-line setting of splits take precedence over setting them in code.
       Otherwise subsequent calls to PCFieldSplitSetIS() or PCFieldSplitSetFields() would
       create new splits, which would probably not be what the user wanted. */
    jac->splitdefined = PETSC_TRUE;
  }
  ierr = PetscFree(ifields);CHKERRQ(ierr);
  ierr = PetscFree(ifields_col);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCFieldSplitSetDefaults(PC pc)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilink = jac->head;
  PetscBool         fieldsplit_default = PETSC_FALSE,coupling = PETSC_FALSE;
  PetscInt          i;

  PetscFunctionBegin;
  /*
   Kinda messy, but at least this now uses DMCreateFieldDecomposition().
   Should probably be rewritten.
   */
  if (!ilink) {
    ierr = PetscOptionsGetBool(((PetscObject)pc)->options,((PetscObject)pc)->prefix,"-pc_fieldsplit_detect_coupling",&coupling,NULL);CHKERRQ(ierr);
    if (pc->dm && jac->dm_splits && !jac->detect && !coupling) {
      PetscInt  numFields, f, i, j;
      char      **fieldNames;
      IS        *fields;
      DM        *dms;
      DM        subdm[128];
      PetscBool flg;

      ierr = DMCreateFieldDecomposition(pc->dm, &numFields, &fieldNames, &fields, &dms);CHKERRQ(ierr);
      /* Allow the user to prescribe the splits */
      for (i = 0, flg = PETSC_TRUE;; i++) {
        PetscInt ifields[128];
        IS       compField;
        char     optionname[128], splitname[8];
        PetscInt nfields = numFields;

        ierr = PetscSNPrintf(optionname, sizeof(optionname), "-pc_fieldsplit_%D_fields", i);CHKERRQ(ierr);
        ierr = PetscOptionsGetIntArray(((PetscObject)pc)->options,((PetscObject)pc)->prefix, optionname, ifields, &nfields, &flg);CHKERRQ(ierr);
        if (!flg) break;
        if (numFields > 128) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Cannot currently support %d > 128 fields", numFields);
        ierr = DMCreateSubDM(pc->dm, nfields, ifields, &compField, &subdm[i]);CHKERRQ(ierr);
        if (nfields == 1) {
          ierr = PCFieldSplitSetIS(pc, fieldNames[ifields[0]], compField);CHKERRQ(ierr);
        } else {
          ierr = PetscSNPrintf(splitname, sizeof(splitname), "%D", i);CHKERRQ(ierr);
          ierr = PCFieldSplitSetIS(pc, splitname, compField);CHKERRQ(ierr);
        }
        ierr = ISDestroy(&compField);CHKERRQ(ierr);
        for (j = 0; j < nfields; ++j) {
          f    = ifields[j];
          ierr = PetscFree(fieldNames[f]);CHKERRQ(ierr);
          ierr = ISDestroy(&fields[f]);CHKERRQ(ierr);
        }
      }
      if (i == 0) {
        for (f = 0; f < numFields; ++f) {
          ierr = PCFieldSplitSetIS(pc, fieldNames[f], fields[f]);CHKERRQ(ierr);
          ierr = PetscFree(fieldNames[f]);CHKERRQ(ierr);
          ierr = ISDestroy(&fields[f]);CHKERRQ(ierr);
        }
      } else {
        for (j=0; j<numFields; j++) {
          ierr = DMDestroy(dms+j);CHKERRQ(ierr);
        }
        ierr = PetscFree(dms);CHKERRQ(ierr);
        ierr = PetscMalloc1(i, &dms);CHKERRQ(ierr);
        for (j = 0; j < i; ++j) dms[j] = subdm[j];
      }
      ierr = PetscFree(fieldNames);CHKERRQ(ierr);
      ierr = PetscFree(fields);CHKERRQ(ierr);
      if (dms) {
        ierr = PetscInfo(pc, "Setting up physics based fieldsplit preconditioner using the embedded DM\n");CHKERRQ(ierr);
        for (ilink = jac->head, i = 0; ilink; ilink = ilink->next, ++i) {
          const char *prefix;
          ierr = PetscObjectGetOptionsPrefix((PetscObject)(ilink->ksp),&prefix);CHKERRQ(ierr);
          ierr = PetscObjectSetOptionsPrefix((PetscObject)(dms[i]), prefix);CHKERRQ(ierr);
          ierr = KSPSetDM(ilink->ksp, dms[i]);CHKERRQ(ierr);
          ierr = KSPSetDMActive(ilink->ksp, PETSC_FALSE);CHKERRQ(ierr);
          {
            PetscErrorCode (*func)(KSP,Mat,Mat,void*);
            void            *ctx;

            ierr = DMKSPGetComputeOperators(pc->dm, &func, &ctx);CHKERRQ(ierr);
            ierr = DMKSPSetComputeOperators(dms[i],  func,  ctx);CHKERRQ(ierr);
          }
          ierr = PetscObjectIncrementTabLevel((PetscObject)dms[i],(PetscObject)ilink->ksp,0);CHKERRQ(ierr);
          ierr = DMDestroy(&dms[i]);CHKERRQ(ierr);
        }
        ierr = PetscFree(dms);CHKERRQ(ierr);
      }
    } else {
      if (jac->bs <= 0) {
        if (pc->pmat) {
          ierr = MatGetBlockSize(pc->pmat,&jac->bs);CHKERRQ(ierr);
        } else jac->bs = 1;
      }

      if (jac->detect) {
        IS       zerodiags,rest;
        PetscInt nmin,nmax;

        ierr = MatGetOwnershipRange(pc->mat,&nmin,&nmax);CHKERRQ(ierr);
        ierr = MatFindZeroDiagonals(pc->mat,&zerodiags);CHKERRQ(ierr);
        ierr = ISComplement(zerodiags,nmin,nmax,&rest);CHKERRQ(ierr);
        ierr = PCFieldSplitSetIS(pc,"0",rest);CHKERRQ(ierr);
        ierr = PCFieldSplitSetIS(pc,"1",zerodiags);CHKERRQ(ierr);
        ierr = ISDestroy(&zerodiags);CHKERRQ(ierr);
        ierr = ISDestroy(&rest);CHKERRQ(ierr);
      } else if (coupling) {
        IS       coupling,rest;
        PetscInt nmin,nmax;

        ierr = MatGetOwnershipRange(pc->mat,&nmin,&nmax);CHKERRQ(ierr);
        ierr = MatFindOffBlockDiagonalEntries(pc->mat,&coupling);CHKERRQ(ierr);
        ierr = ISCreateStride(PetscObjectComm((PetscObject)pc->mat),nmax-nmin,nmin,1,&rest);CHKERRQ(ierr);
        ierr = ISSetIdentity(rest);CHKERRQ(ierr);
        ierr = PCFieldSplitSetIS(pc,"0",rest);CHKERRQ(ierr);
        ierr = PCFieldSplitSetIS(pc,"1",coupling);CHKERRQ(ierr);
        ierr = ISDestroy(&coupling);CHKERRQ(ierr);
        ierr = ISDestroy(&rest);CHKERRQ(ierr);
      } else {
        ierr = PetscOptionsGetBool(((PetscObject)pc)->options,((PetscObject)pc)->prefix,"-pc_fieldsplit_default",&fieldsplit_default,NULL);CHKERRQ(ierr);
        if (!fieldsplit_default) {
          /* Allow user to set fields from command line,  if bs was known at the time of PCSetFromOptions_FieldSplit()
           then it is set there. This is not ideal because we should only have options set in XXSetFromOptions(). */
          ierr = PCFieldSplitSetRuntimeSplits_Private(pc);CHKERRQ(ierr);
          if (jac->splitdefined) {ierr = PetscInfo(pc,"Splits defined using the options database\n");CHKERRQ(ierr);}
        }
        if ((fieldsplit_default || !jac->splitdefined) && !jac->isrestrict) {
          Mat       M = pc->pmat;
          PetscBool isnest;

          ierr = PetscInfo(pc,"Using default splitting of fields\n");CHKERRQ(ierr);
          ierr = PetscObjectTypeCompare((PetscObject)pc->pmat,MATNEST,&isnest);CHKERRQ(ierr);
          if (!isnest) {
            M    = pc->mat;
            ierr = PetscObjectTypeCompare((PetscObject)pc->mat,MATNEST,&isnest);CHKERRQ(ierr);
          }
          if (isnest) {
            IS       *fields;
            PetscInt nf;

            ierr = MatNestGetSize(M,&nf,NULL);CHKERRQ(ierr);
            ierr = PetscMalloc1(nf,&fields);CHKERRQ(ierr);
            ierr = MatNestGetISs(M,fields,NULL);CHKERRQ(ierr);
            for (i=0;i<nf;i++) {
              ierr = PCFieldSplitSetIS(pc,NULL,fields[i]);CHKERRQ(ierr);
            }
            ierr = PetscFree(fields);CHKERRQ(ierr);
          } else {
            for (i=0; i<jac->bs; i++) {
              char splitname[8];
              ierr = PetscSNPrintf(splitname,sizeof(splitname),"%D",i);CHKERRQ(ierr);
              ierr = PCFieldSplitSetFields(pc,splitname,1,&i,&i);CHKERRQ(ierr);
            }
            jac->defaultsplit = PETSC_TRUE;
          }
        }
      }
    }
  } else if (jac->nsplits == 1) {
    if (ilink->is) {
      IS       is2;
      PetscInt nmin,nmax;

      ierr = MatGetOwnershipRange(pc->mat,&nmin,&nmax);CHKERRQ(ierr);
      ierr = ISComplement(ilink->is,nmin,nmax,&is2);CHKERRQ(ierr);
      ierr = PCFieldSplitSetIS(pc,"1",is2);CHKERRQ(ierr);
      ierr = ISDestroy(&is2);CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Must provide at least two sets of fields to PCFieldSplit()");
  }

  if (jac->nsplits < 2) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"Unhandled case, must have at least two fields, not %d", jac->nsplits);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGolubKahanComputeExplicitOperator(Mat A,Mat B,Mat C,Mat *H,PetscReal gkbnu)
{
  PetscErrorCode    ierr;
  Mat               BT,T;
  PetscReal         nrmT,nrmB;

  PetscFunctionBegin;
  ierr = MatHermitianTranspose(C,MAT_INITIAL_MATRIX,&T);CHKERRQ(ierr);            /* Test if augmented matrix is symmetric */
  ierr = MatAXPY(T,-1.0,B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(T,NORM_1,&nrmT);CHKERRQ(ierr);
  ierr = MatNorm(B,NORM_1,&nrmB);CHKERRQ(ierr);
  if (nrmB > 0) {
    if (nrmT/nrmB >= PETSC_SMALL) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Matrix is not symmetric/hermitian, GKB is not applicable.");
    }
  }
  /* Compute augmented Lagrangian matrix H = A00 + nu*A01*A01'. This corresponds to */
  /* setting N := 1/nu*I in [Ar13].                                                 */
  ierr = MatHermitianTranspose(B,MAT_INITIAL_MATRIX,&BT);CHKERRQ(ierr);
  ierr = MatMatMult(B,BT,MAT_INITIAL_MATRIX,PETSC_DEFAULT,H);CHKERRQ(ierr);       /* H = A01*A01'          */
  ierr = MatAYPX(*H,gkbnu,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);             /* H = A00 + nu*A01*A01' */

  ierr = MatDestroy(&BT);CHKERRQ(ierr);
  ierr = MatDestroy(&T);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscOptionsFindPairPrefix_Private(PetscOptions,const char pre[], const char name[],const char *value[],PetscBool *flg);

static PetscErrorCode PCSetUp_FieldSplit(PC pc)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilink;
  PetscInt          i,nsplit;
  PetscBool         sorted, sorted_col;

  PetscFunctionBegin;
  pc->failedreason = PC_NOERROR;
  ierr   = PCFieldSplitSetDefaults(pc);CHKERRQ(ierr);
  nsplit = jac->nsplits;
  ilink  = jac->head;

  /* get the matrices for each split */
  if (!jac->issetup) {
    PetscInt rstart,rend,nslots,bs;

    jac->issetup = PETSC_TRUE;

    /* This is done here instead of in PCFieldSplitSetFields() because may not have matrix at that point */
    if (jac->defaultsplit || !ilink->is) {
      if (jac->bs <= 0) jac->bs = nsplit;
    }
    bs     = jac->bs;
    ierr   = MatGetOwnershipRange(pc->pmat,&rstart,&rend);CHKERRQ(ierr);
    nslots = (rend - rstart)/bs;
    for (i=0; i<nsplit; i++) {
      if (jac->defaultsplit) {
        ierr = ISCreateStride(PetscObjectComm((PetscObject)pc),nslots,rstart+i,nsplit,&ilink->is);CHKERRQ(ierr);
        ierr = ISDuplicate(ilink->is,&ilink->is_col);CHKERRQ(ierr);
      } else if (!ilink->is) {
        if (ilink->nfields > 1) {
          PetscInt *ii,*jj,j,k,nfields = ilink->nfields,*fields = ilink->fields,*fields_col = ilink->fields_col;
          ierr = PetscMalloc1(ilink->nfields*nslots,&ii);CHKERRQ(ierr);
          ierr = PetscMalloc1(ilink->nfields*nslots,&jj);CHKERRQ(ierr);
          for (j=0; j<nslots; j++) {
            for (k=0; k<nfields; k++) {
              ii[nfields*j + k] = rstart + bs*j + fields[k];
              jj[nfields*j + k] = rstart + bs*j + fields_col[k];
            }
          }
          ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pc),nslots*nfields,ii,PETSC_OWN_POINTER,&ilink->is);CHKERRQ(ierr);
          ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pc),nslots*nfields,jj,PETSC_OWN_POINTER,&ilink->is_col);CHKERRQ(ierr);
          ierr = ISSetBlockSize(ilink->is, nfields);CHKERRQ(ierr);
          ierr = ISSetBlockSize(ilink->is_col, nfields);CHKERRQ(ierr);
        } else {
          ierr = ISCreateStride(PetscObjectComm((PetscObject)pc),nslots,rstart+ilink->fields[0],bs,&ilink->is);CHKERRQ(ierr);
          ierr = ISCreateStride(PetscObjectComm((PetscObject)pc),nslots,rstart+ilink->fields_col[0],bs,&ilink->is_col);CHKERRQ(ierr);
        }
      }
      ierr = ISSorted(ilink->is,&sorted);CHKERRQ(ierr);
      if (ilink->is_col) { ierr = ISSorted(ilink->is_col,&sorted_col);CHKERRQ(ierr); }
      if (!sorted || !sorted_col) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Fields must be sorted when creating split");
      ilink = ilink->next;
    }
  }

  ilink = jac->head;
  if (!jac->pmat) {
    Vec xtmp;

    ierr = MatCreateVecs(pc->pmat,&xtmp,NULL);CHKERRQ(ierr);
    ierr = PetscMalloc1(nsplit,&jac->pmat);CHKERRQ(ierr);
    ierr = PetscMalloc2(nsplit,&jac->x,nsplit,&jac->y);CHKERRQ(ierr);
    for (i=0; i<nsplit; i++) {
      MatNullSpace sp;

      /* Check for preconditioning matrix attached to IS */
      ierr = PetscObjectQuery((PetscObject) ilink->is, "pmat", (PetscObject*) &jac->pmat[i]);CHKERRQ(ierr);
      if (jac->pmat[i]) {
        ierr = PetscObjectReference((PetscObject) jac->pmat[i]);CHKERRQ(ierr);
        if (jac->type == PC_COMPOSITE_SCHUR) {
          jac->schur_user = jac->pmat[i];

          ierr = PetscObjectReference((PetscObject) jac->schur_user);CHKERRQ(ierr);
        }
      } else {
        const char *prefix;
        ierr = MatCreateSubMatrix(pc->pmat,ilink->is,ilink->is_col,MAT_INITIAL_MATRIX,&jac->pmat[i]);CHKERRQ(ierr);
        ierr = KSPGetOptionsPrefix(ilink->ksp,&prefix);CHKERRQ(ierr);
        ierr = MatSetOptionsPrefix(jac->pmat[i],prefix);CHKERRQ(ierr);
        ierr = MatViewFromOptions(jac->pmat[i],NULL,"-mat_view");CHKERRQ(ierr);
      }
      /* create work vectors for each split */
      ierr     = MatCreateVecs(jac->pmat[i],&jac->x[i],&jac->y[i]);CHKERRQ(ierr);
      ilink->x = jac->x[i]; ilink->y = jac->y[i]; ilink->z = NULL;
      /* compute scatter contexts needed by multiplicative versions and non-default splits */
      ierr = VecScatterCreate(xtmp,ilink->is,jac->x[i],NULL,&ilink->sctx);CHKERRQ(ierr);
      ierr = PetscObjectQuery((PetscObject) ilink->is, "nearnullspace", (PetscObject*) &sp);CHKERRQ(ierr);
      if (sp) {
        ierr = MatSetNearNullSpace(jac->pmat[i], sp);CHKERRQ(ierr);
      }
      ilink = ilink->next;
    }
    ierr = VecDestroy(&xtmp);CHKERRQ(ierr);
  } else {
    MatReuse scall;
    if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
      for (i=0; i<nsplit; i++) {
        ierr = MatDestroy(&jac->pmat[i]);CHKERRQ(ierr);
      }
      scall = MAT_INITIAL_MATRIX;
    } else scall = MAT_REUSE_MATRIX;

    for (i=0; i<nsplit; i++) {
      Mat pmat;

      /* Check for preconditioning matrix attached to IS */
      ierr = PetscObjectQuery((PetscObject) ilink->is, "pmat", (PetscObject*) &pmat);CHKERRQ(ierr);
      if (!pmat) {
        ierr = MatCreateSubMatrix(pc->pmat,ilink->is,ilink->is_col,scall,&jac->pmat[i]);CHKERRQ(ierr);
      }
      ilink = ilink->next;
    }
  }
  if (jac->diag_use_amat) {
    ilink = jac->head;
    if (!jac->mat) {
      ierr = PetscMalloc1(nsplit,&jac->mat);CHKERRQ(ierr);
      for (i=0; i<nsplit; i++) {
        ierr  = MatCreateSubMatrix(pc->mat,ilink->is,ilink->is_col,MAT_INITIAL_MATRIX,&jac->mat[i]);CHKERRQ(ierr);
        ilink = ilink->next;
      }
    } else {
      MatReuse scall;
      if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
        for (i=0; i<nsplit; i++) {
          ierr = MatDestroy(&jac->mat[i]);CHKERRQ(ierr);
        }
        scall = MAT_INITIAL_MATRIX;
      } else scall = MAT_REUSE_MATRIX;

      for (i=0; i<nsplit; i++) {
        ierr  = MatCreateSubMatrix(pc->mat,ilink->is,ilink->is_col,scall,&jac->mat[i]);CHKERRQ(ierr);
        ilink = ilink->next;
      }
    }
  } else {
    jac->mat = jac->pmat;
  }

  /* Check for null space attached to IS */
  ilink = jac->head;
  for (i=0; i<nsplit; i++) {
    MatNullSpace sp;

    ierr = PetscObjectQuery((PetscObject) ilink->is, "nullspace", (PetscObject*) &sp);CHKERRQ(ierr);
    if (sp) {
      ierr = MatSetNullSpace(jac->mat[i], sp);CHKERRQ(ierr);
    }
    ilink = ilink->next;
  }

  if (jac->type != PC_COMPOSITE_ADDITIVE  && jac->type != PC_COMPOSITE_SCHUR && jac->type != PC_COMPOSITE_GKB) {
    /* extract the rows of the matrix associated with each field: used for efficient computation of residual inside algorithm */
    /* FIXME: Can/should we reuse jac->mat whenever (jac->diag_use_amat) is true? */
    ilink = jac->head;
    if (nsplit == 2 && jac->type == PC_COMPOSITE_MULTIPLICATIVE) {
      /* special case need where Afield[0] is not needed and only certain columns of Afield[1] are needed since update is only on those rows of the solution */
      if (!jac->Afield) {
        ierr = PetscCalloc1(nsplit,&jac->Afield);CHKERRQ(ierr);
        if (jac->offdiag_use_amat) {
          ierr = MatCreateSubMatrix(pc->mat,ilink->next->is,ilink->is,MAT_INITIAL_MATRIX,&jac->Afield[1]);CHKERRQ(ierr);
        } else {
          ierr = MatCreateSubMatrix(pc->pmat,ilink->next->is,ilink->is,MAT_INITIAL_MATRIX,&jac->Afield[1]);CHKERRQ(ierr);
        }
      } else {
        MatReuse scall;

        if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
          ierr  = MatDestroy(&jac->Afield[1]);CHKERRQ(ierr);
          scall = MAT_INITIAL_MATRIX;
        } else scall = MAT_REUSE_MATRIX;

        if (jac->offdiag_use_amat) {
          ierr = MatCreateSubMatrix(pc->mat,ilink->next->is,ilink->is,scall,&jac->Afield[1]);CHKERRQ(ierr);
        } else {
          ierr = MatCreateSubMatrix(pc->pmat,ilink->next->is,ilink->is,scall,&jac->Afield[1]);CHKERRQ(ierr);
        }
      }
    } else {
      if (!jac->Afield) {
        ierr = PetscMalloc1(nsplit,&jac->Afield);CHKERRQ(ierr);
        for (i=0; i<nsplit; i++) {
          if (jac->offdiag_use_amat) {
            ierr = MatCreateSubMatrix(pc->mat,ilink->is,NULL,MAT_INITIAL_MATRIX,&jac->Afield[i]);CHKERRQ(ierr);
          } else {
            ierr = MatCreateSubMatrix(pc->pmat,ilink->is,NULL,MAT_INITIAL_MATRIX,&jac->Afield[i]);CHKERRQ(ierr);
          }
          ilink = ilink->next;
        }
      } else {
        MatReuse scall;
        if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
          for (i=0; i<nsplit; i++) {
            ierr = MatDestroy(&jac->Afield[i]);CHKERRQ(ierr);
          }
          scall = MAT_INITIAL_MATRIX;
        } else scall = MAT_REUSE_MATRIX;

        for (i=0; i<nsplit; i++) {
          if (jac->offdiag_use_amat) {
            ierr = MatCreateSubMatrix(pc->mat,ilink->is,NULL,scall,&jac->Afield[i]);CHKERRQ(ierr);
          } else {
            ierr = MatCreateSubMatrix(pc->pmat,ilink->is,NULL,scall,&jac->Afield[i]);CHKERRQ(ierr);
          }
          ilink = ilink->next;
        }
      }
    }
  }

  if (jac->type == PC_COMPOSITE_SCHUR) {
    IS          ccis;
    PetscBool   isspd;
    PetscInt    rstart,rend;
    char        lscname[256];
    PetscObject LSC_L;

    if (nsplit != 2) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_INCOMP,"To use Schur complement preconditioner you must have exactly 2 fields");

    /* If pc->mat is SPD, don't scale by -1 the Schur complement */
    if (jac->schurscale == (PetscScalar)-1.0) {
      ierr = MatGetOption(pc->pmat,MAT_SPD,&isspd);CHKERRQ(ierr);
      jac->schurscale = (isspd == PETSC_TRUE) ? 1.0 : -1.0;
    }

    /* When extracting off-diagonal submatrices, we take complements from this range */
    ierr = MatGetOwnershipRangeColumn(pc->mat,&rstart,&rend);CHKERRQ(ierr);

    if (jac->schur) {
      KSP      kspA = jac->head->ksp, kspInner = NULL, kspUpper = jac->kspupper;
      MatReuse scall;

      if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
        scall = MAT_INITIAL_MATRIX;
        ierr  = MatDestroy(&jac->B);CHKERRQ(ierr);
        ierr  = MatDestroy(&jac->C);CHKERRQ(ierr);
      } else scall = MAT_REUSE_MATRIX;

      ierr  = MatSchurComplementGetKSP(jac->schur, &kspInner);CHKERRQ(ierr);
      ilink = jac->head;
      ierr  = ISComplement(ilink->is_col,rstart,rend,&ccis);CHKERRQ(ierr);
      if (jac->offdiag_use_amat) {
	ierr = MatCreateSubMatrix(pc->mat,ilink->is,ccis,scall,&jac->B);CHKERRQ(ierr);
      } else {
	ierr = MatCreateSubMatrix(pc->pmat,ilink->is,ccis,scall,&jac->B);CHKERRQ(ierr);
      }
      ierr  = ISDestroy(&ccis);CHKERRQ(ierr);
      ilink = ilink->next;
      ierr  = ISComplement(ilink->is_col,rstart,rend,&ccis);CHKERRQ(ierr);
      if (jac->offdiag_use_amat) {
	ierr = MatCreateSubMatrix(pc->mat,ilink->is,ccis,scall,&jac->C);CHKERRQ(ierr);
      } else {
	ierr = MatCreateSubMatrix(pc->pmat,ilink->is,ccis,scall,&jac->C);CHKERRQ(ierr);
      }
      ierr  = ISDestroy(&ccis);CHKERRQ(ierr);
      ierr  = MatSchurComplementUpdateSubMatrices(jac->schur,jac->mat[0],jac->pmat[0],jac->B,jac->C,jac->mat[1]);CHKERRQ(ierr);
      if (jac->schurpre == PC_FIELDSPLIT_SCHUR_PRE_SELFP) {
	ierr = MatDestroy(&jac->schurp);CHKERRQ(ierr);
	ierr = MatSchurComplementGetPmat(jac->schur,MAT_INITIAL_MATRIX,&jac->schurp);CHKERRQ(ierr);
      }
      if (kspA != kspInner) {
        ierr = KSPSetOperators(kspA,jac->mat[0],jac->pmat[0]);CHKERRQ(ierr);
      }
      if (kspUpper != kspA) {
        ierr = KSPSetOperators(kspUpper,jac->mat[0],jac->pmat[0]);CHKERRQ(ierr);
      }
      ierr = KSPSetOperators(jac->kspschur,jac->schur,FieldSplitSchurPre(jac));CHKERRQ(ierr);
    } else {
      const char   *Dprefix;
      char         schurprefix[256], schurmatprefix[256];
      char         schurtestoption[256];
      MatNullSpace sp;
      PetscBool    flg;
      KSP          kspt;

      /* extract the A01 and A10 matrices */
      ilink = jac->head;
      ierr  = ISComplement(ilink->is_col,rstart,rend,&ccis);CHKERRQ(ierr);
      if (jac->offdiag_use_amat) {
	ierr = MatCreateSubMatrix(pc->mat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->B);CHKERRQ(ierr);
      } else {
	ierr = MatCreateSubMatrix(pc->pmat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->B);CHKERRQ(ierr);
      }
      ierr  = ISDestroy(&ccis);CHKERRQ(ierr);
      ilink = ilink->next;
      ierr  = ISComplement(ilink->is_col,rstart,rend,&ccis);CHKERRQ(ierr);
      if (jac->offdiag_use_amat) {
	ierr = MatCreateSubMatrix(pc->mat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->C);CHKERRQ(ierr);
      } else {
	ierr = MatCreateSubMatrix(pc->pmat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->C);CHKERRQ(ierr);
      }
      ierr = ISDestroy(&ccis);CHKERRQ(ierr);

      /* Use mat[0] (diagonal block of Amat) preconditioned by pmat[0] to define Schur complement */
      ierr = MatCreate(((PetscObject)jac->mat[0])->comm,&jac->schur);CHKERRQ(ierr);
      ierr = MatSetType(jac->schur,MATSCHURCOMPLEMENT);CHKERRQ(ierr);
      ierr = MatSchurComplementSetSubMatrices(jac->schur,jac->mat[0],jac->pmat[0],jac->B,jac->C,jac->mat[1]);CHKERRQ(ierr);
      ierr = PetscSNPrintf(schurmatprefix, sizeof(schurmatprefix), "%sfieldsplit_%s_", ((PetscObject)pc)->prefix ? ((PetscObject)pc)->prefix : "", ilink->splitname);CHKERRQ(ierr);
      ierr = MatSetOptionsPrefix(jac->schur,schurmatprefix);CHKERRQ(ierr);
      ierr = MatSchurComplementGetKSP(jac->schur,&kspt);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(kspt,schurmatprefix);CHKERRQ(ierr);

      /* Note: this is not true in general */
      ierr = MatGetNullSpace(jac->mat[1], &sp);CHKERRQ(ierr);
      if (sp) {
        ierr = MatSetNullSpace(jac->schur, sp);CHKERRQ(ierr);
      }

      ierr = PetscSNPrintf(schurtestoption, sizeof(schurtestoption), "-fieldsplit_%s_inner_", ilink->splitname);CHKERRQ(ierr);
      ierr = PetscOptionsFindPairPrefix_Private(((PetscObject)pc)->options,((PetscObject)pc)->prefix, schurtestoption, NULL, &flg);CHKERRQ(ierr);
      if (flg) {
        DM  dmInner;
        KSP kspInner;
        PC  pcInner;

        ierr = MatSchurComplementGetKSP(jac->schur, &kspInner);CHKERRQ(ierr);
        ierr = KSPReset(kspInner);CHKERRQ(ierr);
        ierr = KSPSetOperators(kspInner,jac->mat[0],jac->pmat[0]);CHKERRQ(ierr);
        ierr = PetscSNPrintf(schurprefix, sizeof(schurprefix), "%sfieldsplit_%s_inner_", ((PetscObject)pc)->prefix ? ((PetscObject)pc)->prefix : "", ilink->splitname);CHKERRQ(ierr);
        /* Indent this deeper to emphasize the "inner" nature of this solver. */
        ierr = PetscObjectIncrementTabLevel((PetscObject)kspInner, (PetscObject) pc, 2);CHKERRQ(ierr);
        ierr = PetscObjectIncrementTabLevel((PetscObject)kspInner->pc, (PetscObject) pc, 2);CHKERRQ(ierr);
        ierr = KSPSetOptionsPrefix(kspInner, schurprefix);CHKERRQ(ierr);

        /* Set DM for new solver */
        ierr = KSPGetDM(jac->head->ksp, &dmInner);CHKERRQ(ierr);
        ierr = KSPSetDM(kspInner, dmInner);CHKERRQ(ierr);
        ierr = KSPSetDMActive(kspInner, PETSC_FALSE);CHKERRQ(ierr);

        /* Defaults to PCKSP as preconditioner */
        ierr = KSPGetPC(kspInner, &pcInner);CHKERRQ(ierr);
        ierr = PCSetType(pcInner, PCKSP);CHKERRQ(ierr);
        ierr = PCKSPSetKSP(pcInner, jac->head->ksp);CHKERRQ(ierr);
      } else {
         /* Use the outer solver for the inner solve, but revert the KSPPREONLY from PCFieldSplitSetFields_FieldSplit or
          * PCFieldSplitSetIS_FieldSplit. We don't want KSPPREONLY because it makes the Schur complement inexact,
          * preventing Schur complement reduction to be an accurate solve. Usually when an iterative solver is used for
          * S = D - C A_inner^{-1} B, we expect S to be defined using an accurate definition of A_inner^{-1}, so we make
          * GMRES the default. Note that it is also common to use PREONLY for S, in which case S may not be used
          * directly, and the user is responsible for setting an inexact method for fieldsplit's A^{-1}. */
        ierr = KSPSetType(jac->head->ksp,KSPGMRES);CHKERRQ(ierr);
        ierr = MatSchurComplementSetKSP(jac->schur,jac->head->ksp);CHKERRQ(ierr);
      }
      ierr = KSPSetOperators(jac->head->ksp,jac->mat[0],jac->pmat[0]);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(jac->head->ksp);CHKERRQ(ierr);
      ierr = MatSetFromOptions(jac->schur);CHKERRQ(ierr);

      ierr = PetscObjectTypeCompare((PetscObject)jac->schur, MATSCHURCOMPLEMENT, &flg);CHKERRQ(ierr);
      if (flg) { /* Need to do this otherwise PCSetUp_KSP will overwrite the amat of jac->head->ksp */
        KSP kspInner;
        PC  pcInner;

        ierr = MatSchurComplementGetKSP(jac->schur, &kspInner);CHKERRQ(ierr);
        ierr = KSPGetPC(kspInner, &pcInner);CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)pcInner, PCKSP, &flg);CHKERRQ(ierr);
        if (flg) {
          KSP ksp;

          ierr = PCKSPGetKSP(pcInner, &ksp);CHKERRQ(ierr);
          if (ksp == jac->head->ksp) {
            ierr = PCSetUseAmat(pcInner, PETSC_TRUE);CHKERRQ(ierr);
          }
        }
      }
      ierr = PetscSNPrintf(schurtestoption, sizeof(schurtestoption), "-fieldsplit_%s_upper_", ilink->splitname);CHKERRQ(ierr);
      ierr = PetscOptionsFindPairPrefix_Private(((PetscObject)pc)->options,((PetscObject)pc)->prefix, schurtestoption, NULL, &flg);CHKERRQ(ierr);
      if (flg) {
        DM dmInner;

        ierr = PetscSNPrintf(schurprefix, sizeof(schurprefix), "%sfieldsplit_%s_upper_", ((PetscObject)pc)->prefix ? ((PetscObject)pc)->prefix : "", ilink->splitname);CHKERRQ(ierr);
        ierr = KSPCreate(PetscObjectComm((PetscObject)pc), &jac->kspupper);CHKERRQ(ierr);
        ierr = KSPSetErrorIfNotConverged(jac->kspupper,pc->erroriffailure);CHKERRQ(ierr);
        ierr = KSPSetOptionsPrefix(jac->kspupper, schurprefix);CHKERRQ(ierr);
        ierr = PetscObjectIncrementTabLevel((PetscObject)jac->kspupper, (PetscObject) pc, 1);CHKERRQ(ierr);
        ierr = PetscObjectIncrementTabLevel((PetscObject)jac->kspupper->pc, (PetscObject) pc, 1);CHKERRQ(ierr);
        ierr = KSPGetDM(jac->head->ksp, &dmInner);CHKERRQ(ierr);
        ierr = KSPSetDM(jac->kspupper, dmInner);CHKERRQ(ierr);
        ierr = KSPSetDMActive(jac->kspupper, PETSC_FALSE);CHKERRQ(ierr);
        ierr = KSPSetFromOptions(jac->kspupper);CHKERRQ(ierr);
        ierr = KSPSetOperators(jac->kspupper,jac->mat[0],jac->pmat[0]);CHKERRQ(ierr);
        ierr = VecDuplicate(jac->head->x, &jac->head->z);CHKERRQ(ierr);
      } else {
        jac->kspupper = jac->head->ksp;
        ierr          = PetscObjectReference((PetscObject) jac->head->ksp);CHKERRQ(ierr);
      }

      if (jac->schurpre == PC_FIELDSPLIT_SCHUR_PRE_SELFP) {
	ierr = MatSchurComplementGetPmat(jac->schur,MAT_INITIAL_MATRIX,&jac->schurp);CHKERRQ(ierr);
      }
      ierr = KSPCreate(PetscObjectComm((PetscObject)pc),&jac->kspschur);CHKERRQ(ierr);
      ierr = KSPSetErrorIfNotConverged(jac->kspschur,pc->erroriffailure);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)jac->kspschur);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)jac->kspschur,(PetscObject)pc,1);CHKERRQ(ierr);
      if (jac->schurpre == PC_FIELDSPLIT_SCHUR_PRE_SELF) {
        PC pcschur;
        ierr = KSPGetPC(jac->kspschur,&pcschur);CHKERRQ(ierr);
        ierr = PCSetType(pcschur,PCNONE);CHKERRQ(ierr);
        /* Note: This is bad if there exist preconditioners for MATSCHURCOMPLEMENT */
      } else if (jac->schurpre == PC_FIELDSPLIT_SCHUR_PRE_FULL) {
        ierr = MatSchurComplementComputeExplicitOperator(jac->schur, &jac->schur_user);CHKERRQ(ierr);
      }
      ierr = KSPSetOperators(jac->kspschur,jac->schur,FieldSplitSchurPre(jac));CHKERRQ(ierr);
      ierr = KSPGetOptionsPrefix(jac->head->next->ksp, &Dprefix);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(jac->kspschur,         Dprefix);CHKERRQ(ierr);
      /* propagate DM */
      {
        DM sdm;
        ierr = KSPGetDM(jac->head->next->ksp, &sdm);CHKERRQ(ierr);
        if (sdm) {
          ierr = KSPSetDM(jac->kspschur, sdm);CHKERRQ(ierr);
          ierr = KSPSetDMActive(jac->kspschur, PETSC_FALSE);CHKERRQ(ierr);
        }
      }
      /* really want setfromoptions called in PCSetFromOptions_FieldSplit(), but it is not ready yet */
      /* need to call this every time, since the jac->kspschur is freshly created, otherwise its options never get set */
      ierr = KSPSetFromOptions(jac->kspschur);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(jac->schur,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(jac->schur,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    /* HACK: special support to forward L and Lp matrices that might be used by PCLSC */
    ierr = PetscSNPrintf(lscname,sizeof(lscname),"%s_LSC_L",ilink->splitname);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)pc->mat,lscname,(PetscObject*)&LSC_L);CHKERRQ(ierr);
    if (!LSC_L) {ierr = PetscObjectQuery((PetscObject)pc->pmat,lscname,(PetscObject*)&LSC_L);CHKERRQ(ierr);}
    if (LSC_L) {ierr = PetscObjectCompose((PetscObject)jac->schur,"LSC_L",(PetscObject)LSC_L);CHKERRQ(ierr);}
    ierr = PetscSNPrintf(lscname,sizeof(lscname),"%s_LSC_Lp",ilink->splitname);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)pc->pmat,lscname,(PetscObject*)&LSC_L);CHKERRQ(ierr);
    if (!LSC_L) {ierr = PetscObjectQuery((PetscObject)pc->mat,lscname,(PetscObject*)&LSC_L);CHKERRQ(ierr);}
    if (LSC_L) {ierr = PetscObjectCompose((PetscObject)jac->schur,"LSC_Lp",(PetscObject)LSC_L);CHKERRQ(ierr);}
  } else if (jac->type == PC_COMPOSITE_GKB) {
    IS          ccis;
    PetscInt    rstart,rend;

    if (nsplit != 2) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_INCOMP,"To use GKB preconditioner you must have exactly 2 fields");

    ilink = jac->head;

    /* When extracting off-diagonal submatrices, we take complements from this range */
    ierr = MatGetOwnershipRangeColumn(pc->mat,&rstart,&rend);CHKERRQ(ierr);

    ierr = ISComplement(ilink->is_col,rstart,rend,&ccis);CHKERRQ(ierr);
    if (jac->offdiag_use_amat) {
      ierr = MatCreateSubMatrix(pc->mat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->B);CHKERRQ(ierr);
    } else {
      ierr = MatCreateSubMatrix(pc->pmat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->B);CHKERRQ(ierr);
    }
    ierr  = ISDestroy(&ccis);CHKERRQ(ierr);
    /* Create work vectors for GKB algorithm */
    ierr  = VecDuplicate(ilink->x,&jac->u);CHKERRQ(ierr);
    ierr  = VecDuplicate(ilink->x,&jac->Hu);CHKERRQ(ierr);
    ierr  = VecDuplicate(ilink->x,&jac->w2);CHKERRQ(ierr);
    ilink = ilink->next;
    ierr  = ISComplement(ilink->is_col,rstart,rend,&ccis);CHKERRQ(ierr);
    if (jac->offdiag_use_amat) {
      ierr = MatCreateSubMatrix(pc->mat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->C);CHKERRQ(ierr);
    } else {
      ierr = MatCreateSubMatrix(pc->pmat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->C);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&ccis);CHKERRQ(ierr);
    /* Create work vectors for GKB algorithm */
    ierr = VecDuplicate(ilink->x,&jac->v);CHKERRQ(ierr);
    ierr = VecDuplicate(ilink->x,&jac->d);CHKERRQ(ierr);
    ierr = VecDuplicate(ilink->x,&jac->w1);CHKERRQ(ierr);
    ierr = MatGolubKahanComputeExplicitOperator(jac->mat[0],jac->B,jac->C,&jac->H,jac->gkbnu);CHKERRQ(ierr);
    ierr = PetscCalloc1(jac->gkbdelay,&jac->vecz);CHKERRQ(ierr);

    ilink = jac->head;
    ierr  = KSPSetOperators(ilink->ksp,jac->H,jac->H);CHKERRQ(ierr);
    if (!jac->suboptionsset) {ierr = KSPSetFromOptions(ilink->ksp);CHKERRQ(ierr);}
    /* Create gkb_monitor context */
    if (jac->gkbmonitor) {
      PetscInt  tablevel;
      ierr = PetscViewerCreate(PETSC_COMM_WORLD,&jac->gkbviewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(jac->gkbviewer,PETSCVIEWERASCII);CHKERRQ(ierr);
      ierr = PetscObjectGetTabLevel((PetscObject)ilink->ksp,&tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIISetTab(jac->gkbviewer,tablevel);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)ilink->ksp,(PetscObject)ilink->ksp,1);CHKERRQ(ierr);
    }
  } else {
    /* set up the individual splits' PCs */
    i     = 0;
    ilink = jac->head;
    while (ilink) {
      ierr = KSPSetOperators(ilink->ksp,jac->mat[i],jac->pmat[i]);CHKERRQ(ierr);
      /* really want setfromoptions called in PCSetFromOptions_FieldSplit(), but it is not ready yet */
      if (!jac->suboptionsset) {ierr = KSPSetFromOptions(ilink->ksp);CHKERRQ(ierr);}
      i++;
      ilink = ilink->next;
    }
  }

  jac->suboptionsset = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#define FieldSplitSplitSolveAdd(ilink,xx,yy) \
  (VecScatterBegin(ilink->sctx,xx,ilink->x,INSERT_VALUES,SCATTER_FORWARD) || \
   VecScatterEnd(ilink->sctx,xx,ilink->x,INSERT_VALUES,SCATTER_FORWARD) || \
   PetscLogEventBegin(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL) ||\
   KSPSolve(ilink->ksp,ilink->x,ilink->y) ||                               \
   KSPCheckSolve(ilink->ksp,pc,ilink->y)  || \
   PetscLogEventEnd(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL) ||\
   VecScatterBegin(ilink->sctx,ilink->y,yy,ADD_VALUES,SCATTER_REVERSE) ||  \
   VecScatterEnd(ilink->sctx,ilink->y,yy,ADD_VALUES,SCATTER_REVERSE))

static PetscErrorCode PCApply_FieldSplit_Schur(PC pc,Vec x,Vec y)
{
  PC_FieldSplit      *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode     ierr;
  PC_FieldSplitLink  ilinkA = jac->head, ilinkD = ilinkA->next;
  KSP                kspA   = ilinkA->ksp, kspLower = kspA, kspUpper = jac->kspupper;

  PetscFunctionBegin;
  switch (jac->schurfactorization) {
  case PC_FIELDSPLIT_SCHUR_FACT_DIAG:
    /* [A00 0; 0 -S], positive definite, suitable for MINRES */
    ierr = VecScatterBegin(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterBegin(ilinkD->sctx,x,ilinkD->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(ilinkA->event,kspA,ilinkA->x,ilinkA->y,NULL);CHKERRQ(ierr);
    ierr = KSPSolve(kspA,ilinkA->x,ilinkA->y);CHKERRQ(ierr);
    ierr = KSPCheckSolve(kspA,pc,ilinkA->y);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ilinkA->event,kspA,ilinkA->x,ilinkA->y,NULL);CHKERRQ(ierr);
    ierr = VecScatterBegin(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(ilinkD->sctx,x,ilinkD->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(KSP_Solve_FS_S,jac->kspschur,ilinkD->x,ilinkD->y,NULL);CHKERRQ(ierr);
    ierr = KSPSolve(jac->kspschur,ilinkD->x,ilinkD->y);CHKERRQ(ierr);
    ierr = KSPCheckSolve(jac->kspschur,pc,ilinkD->y);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(KSP_Solve_FS_S,jac->kspschur,ilinkD->x,ilinkD->y,NULL);CHKERRQ(ierr);
    ierr = VecScale(ilinkD->y,jac->schurscale);CHKERRQ(ierr);
    ierr = VecScatterEnd(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    break;
  case PC_FIELDSPLIT_SCHUR_FACT_LOWER:
    /* [A00 0; A10 S], suitable for left preconditioning */
    ierr = VecScatterBegin(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(ilinkA->event,kspA,ilinkA->x,ilinkA->y,NULL);CHKERRQ(ierr);
    ierr = KSPSolve(kspA,ilinkA->x,ilinkA->y);CHKERRQ(ierr);
    ierr = KSPCheckSolve(kspA,pc,ilinkA->y);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ilinkA->event,kspA,ilinkA->x,ilinkA->y,NULL);CHKERRQ(ierr);
    ierr = MatMult(jac->C,ilinkA->y,ilinkD->x);CHKERRQ(ierr);
    ierr = VecScale(ilinkD->x,-1.);CHKERRQ(ierr);
    ierr = VecScatterBegin(ilinkD->sctx,x,ilinkD->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterBegin(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(ilinkD->sctx,x,ilinkD->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(KSP_Solve_FS_S,jac->kspschur,ilinkD->x,ilinkD->y,NULL);CHKERRQ(ierr);
    ierr = KSPSolve(jac->kspschur,ilinkD->x,ilinkD->y);CHKERRQ(ierr);
    ierr = KSPCheckSolve(jac->kspschur,pc,ilinkD->y);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(KSP_Solve_FS_S,jac->kspschur,ilinkD->x,ilinkD->y,NULL);CHKERRQ(ierr);
    ierr = VecScatterEnd(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    break;
  case PC_FIELDSPLIT_SCHUR_FACT_UPPER:
    /* [A00 A01; 0 S], suitable for right preconditioning */
    ierr = VecScatterBegin(ilinkD->sctx,x,ilinkD->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ilinkD->sctx,x,ilinkD->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(KSP_Solve_FS_S,jac->kspschur,ilinkD->x,ilinkD->y,NULL);CHKERRQ(ierr);
    ierr = KSPSolve(jac->kspschur,ilinkD->x,ilinkD->y);CHKERRQ(ierr);
    ierr = KSPCheckSolve(jac->kspschur,pc,ilinkD->y);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(KSP_Solve_FS_S,jac->kspschur,ilinkD->x,ilinkD->y,NULL);CHKERRQ(ierr);    ierr = MatMult(jac->B,ilinkD->y,ilinkA->x);CHKERRQ(ierr);
    ierr = VecScale(ilinkA->x,-1.);CHKERRQ(ierr);
    ierr = VecScatterBegin(ilinkA->sctx,x,ilinkA->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterBegin(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(ilinkA->sctx,x,ilinkA->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(ilinkA->event,kspA,ilinkA->x,ilinkA->y,NULL);CHKERRQ(ierr);
    ierr = KSPSolve(kspA,ilinkA->x,ilinkA->y);CHKERRQ(ierr);
    ierr = KSPCheckSolve(kspA,pc,ilinkA->y);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ilinkA->event,kspA,ilinkA->x,ilinkA->y,NULL);CHKERRQ(ierr);
    ierr = VecScatterEnd(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    break;
  case PC_FIELDSPLIT_SCHUR_FACT_FULL:
    /* [1 0; A10 A00^{-1} 1] [A00 0; 0 S] [1 A00^{-1}A01; 0 1] */
    ierr = VecScatterBegin(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(KSP_Solve_FS_L,kspLower,ilinkA->x,ilinkA->y,NULL);CHKERRQ(ierr);
    ierr = KSPSolve(kspLower,ilinkA->x,ilinkA->y);CHKERRQ(ierr);
    ierr = KSPCheckSolve(kspLower,pc,ilinkA->y);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(KSP_Solve_FS_L,kspLower,ilinkA->x,ilinkA->y,NULL);CHKERRQ(ierr);
    ierr = MatMult(jac->C,ilinkA->y,ilinkD->x);CHKERRQ(ierr);
    ierr = VecScale(ilinkD->x,-1.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(ilinkD->sctx,x,ilinkD->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ilinkD->sctx,x,ilinkD->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    ierr = PetscLogEventBegin(KSP_Solve_FS_S,jac->kspschur,ilinkD->x,ilinkD->y,NULL);CHKERRQ(ierr);
    ierr = KSPSolve(jac->kspschur,ilinkD->x,ilinkD->y);CHKERRQ(ierr);
    ierr = KSPCheckSolve(jac->kspschur,pc,ilinkD->y);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(KSP_Solve_FS_S,jac->kspschur,ilinkD->x,ilinkD->y,NULL);CHKERRQ(ierr);
    ierr = VecScatterBegin(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

    if (kspUpper == kspA) {
      ierr = MatMult(jac->B,ilinkD->y,ilinkA->y);CHKERRQ(ierr);
      ierr = VecAXPY(ilinkA->x,-1.0,ilinkA->y);CHKERRQ(ierr);
      ierr = PetscLogEventBegin(ilinkA->event,kspA,ilinkA->x,ilinkA->y,NULL);CHKERRQ(ierr);
      ierr = KSPSolve(kspA,ilinkA->x,ilinkA->y);CHKERRQ(ierr);
      ierr = KSPCheckSolve(kspA,pc,ilinkA->y);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(ilinkA->event,kspA,ilinkA->x,ilinkA->y,NULL);CHKERRQ(ierr);
    } else {
      ierr = PetscLogEventBegin(ilinkA->event,kspA,ilinkA->x,ilinkA->y,NULL);CHKERRQ(ierr);
      ierr = KSPSolve(kspA,ilinkA->x,ilinkA->y);CHKERRQ(ierr);
      ierr = KSPCheckSolve(kspA,pc,ilinkA->y);CHKERRQ(ierr);
      ierr = MatMult(jac->B,ilinkD->y,ilinkA->x);CHKERRQ(ierr);
      ierr = PetscLogEventBegin(KSP_Solve_FS_U,kspUpper,ilinkA->x,ilinkA->z,NULL);CHKERRQ(ierr);
      ierr = KSPSolve(kspUpper,ilinkA->x,ilinkA->z);CHKERRQ(ierr);
      ierr = KSPCheckSolve(kspUpper,pc,ilinkA->z);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(KSP_Solve_FS_U,kspUpper,ilinkA->x,ilinkA->z,NULL);CHKERRQ(ierr);
      ierr = VecAXPY(ilinkA->y,-1.0,ilinkA->z);CHKERRQ(ierr);
    }
    ierr = VecScatterEnd(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_FieldSplit(PC pc,Vec x,Vec y)
{
  PC_FieldSplit      *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode     ierr;
  PC_FieldSplitLink  ilink = jac->head;
  PetscInt           cnt,bs;

  PetscFunctionBegin;
  if (jac->type == PC_COMPOSITE_ADDITIVE) {
    if (jac->defaultsplit) {
      ierr = VecGetBlockSize(x,&bs);CHKERRQ(ierr);
      if (jac->bs > 0 && bs != jac->bs) SETERRQ2(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Blocksize of x vector %D does not match fieldsplit blocksize %D",bs,jac->bs);
      ierr = VecGetBlockSize(y,&bs);CHKERRQ(ierr);
      if (jac->bs > 0 && bs != jac->bs) SETERRQ2(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Blocksize of y vector %D does not match fieldsplit blocksize %D",bs,jac->bs);
      ierr = VecStrideGatherAll(x,jac->x,INSERT_VALUES);CHKERRQ(ierr);
      while (ilink) {
        ierr = PetscLogEventBegin(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL);CHKERRQ(ierr);
        ierr = KSPSolve(ilink->ksp,ilink->x,ilink->y);CHKERRQ(ierr);
        ierr = KSPCheckSolve(ilink->ksp,pc,ilink->y);CHKERRQ(ierr);
        ierr = PetscLogEventEnd(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL);CHKERRQ(ierr);
        ilink = ilink->next;
      }
      ierr = VecStrideScatterAll(jac->y,y,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      ierr = VecSet(y,0.0);CHKERRQ(ierr);
      while (ilink) {
        ierr = FieldSplitSplitSolveAdd(ilink,x,y);CHKERRQ(ierr);
        ilink = ilink->next;
      }
    }
  } else if (jac->type == PC_COMPOSITE_MULTIPLICATIVE && jac->nsplits == 2) {
    ierr = VecSet(y,0.0);CHKERRQ(ierr);
    /* solve on first block for first block variables */
    ierr = VecScatterBegin(ilink->sctx,x,ilink->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ilink->sctx,x,ilink->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL);CHKERRQ(ierr);
    ierr = KSPSolve(ilink->ksp,ilink->x,ilink->y);CHKERRQ(ierr);
    ierr = KSPCheckSolve(ilink->ksp,pc,ilink->y);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL);CHKERRQ(ierr);
    ierr = VecScatterBegin(ilink->sctx,ilink->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(ilink->sctx,ilink->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

    /* compute the residual only onto second block variables using first block variables */
    ierr = MatMult(jac->Afield[1],ilink->y,ilink->next->x);CHKERRQ(ierr);
    ilink = ilink->next;
    ierr = VecScale(ilink->x,-1.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(ilink->sctx,x,ilink->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ilink->sctx,x,ilink->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    /* solve on second block variables */
    ierr = PetscLogEventBegin(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL);CHKERRQ(ierr);
    ierr = KSPSolve(ilink->ksp,ilink->x,ilink->y);CHKERRQ(ierr);
    ierr = KSPCheckSolve(ilink->ksp,pc,ilink->y);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL);CHKERRQ(ierr);
    ierr = VecScatterBegin(ilink->sctx,ilink->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(ilink->sctx,ilink->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  } else if (jac->type == PC_COMPOSITE_MULTIPLICATIVE || jac->type == PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE) {
    if (!jac->w1) {
      ierr = VecDuplicate(x,&jac->w1);CHKERRQ(ierr);
      ierr = VecDuplicate(x,&jac->w2);CHKERRQ(ierr);
    }
    ierr = VecSet(y,0.0);CHKERRQ(ierr);
    ierr = FieldSplitSplitSolveAdd(ilink,x,y);CHKERRQ(ierr);
    cnt  = 1;
    while (ilink->next) {
      ilink = ilink->next;
      /* compute the residual only over the part of the vector needed */
      ierr = MatMult(jac->Afield[cnt++],y,ilink->x);CHKERRQ(ierr);
      ierr = VecScale(ilink->x,-1.0);CHKERRQ(ierr);
      ierr = VecScatterBegin(ilink->sctx,x,ilink->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(ilink->sctx,x,ilink->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = PetscLogEventBegin(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL);CHKERRQ(ierr);
      ierr = KSPSolve(ilink->ksp,ilink->x,ilink->y);CHKERRQ(ierr);
      ierr = KSPCheckSolve(ilink->ksp,pc,ilink->y);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL);CHKERRQ(ierr);
      ierr = VecScatterBegin(ilink->sctx,ilink->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(ilink->sctx,ilink->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    }
    if (jac->type == PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE) {
      cnt -= 2;
      while (ilink->previous) {
        ilink = ilink->previous;
        /* compute the residual only over the part of the vector needed */
        ierr = MatMult(jac->Afield[cnt--],y,ilink->x);CHKERRQ(ierr);
        ierr = VecScale(ilink->x,-1.0);CHKERRQ(ierr);
        ierr = VecScatterBegin(ilink->sctx,x,ilink->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(ilink->sctx,x,ilink->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = PetscLogEventBegin(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL);CHKERRQ(ierr);
        ierr = KSPSolve(ilink->ksp,ilink->x,ilink->y);CHKERRQ(ierr);
        ierr = KSPCheckSolve(ilink->ksp,pc,ilink->y);CHKERRQ(ierr);
        ierr = PetscLogEventEnd(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL);CHKERRQ(ierr);
        ierr = VecScatterBegin(ilink->sctx,ilink->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        ierr = VecScatterEnd(ilink->sctx,ilink->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      }
    }
  } else SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Unsupported or unknown composition",(int) jac->type);
  PetscFunctionReturn(0);
}


static PetscErrorCode PCApply_FieldSplit_GKB(PC pc,Vec x,Vec y)
{
  PC_FieldSplit      *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode     ierr;
  PC_FieldSplitLink  ilinkA = jac->head,ilinkD = ilinkA->next;
  KSP                ksp = ilinkA->ksp;
  Vec                u,v,Hu,d,work1,work2;
  PetscScalar        alpha,z,nrmz2,*vecz;
  PetscReal          lowbnd,nu,beta;
  PetscInt           j,iterGKB;

  PetscFunctionBegin;
  ierr = VecScatterBegin(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ilinkD->sctx,x,ilinkD->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ilinkD->sctx,x,ilinkD->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  u     = jac->u;
  v     = jac->v;
  Hu    = jac->Hu;
  d     = jac->d;
  work1 = jac->w1;
  work2 = jac->w2;
  vecz  = jac->vecz;

  /* Change RHS to comply with matrix regularization H = A + nu*B*B' */
  /* Add q = q + nu*B*b */
  if (jac->gkbnu) {
    nu = jac->gkbnu;
    ierr = VecScale(ilinkD->x,jac->gkbnu);CHKERRQ(ierr);
    ierr = MatMultAdd(jac->B,ilinkD->x,ilinkA->x,ilinkA->x);CHKERRQ(ierr);            /* q = q + nu*B*b */
  } else {
    /* Situation when no augmented Lagrangian is used. Then we set inner  */
    /* matrix N = I in [Ar13], and thus nu = 1.                           */
    nu = 1;
  }

  /* Transform rhs from [q,tilde{b}] to [0,b] */
  ierr = PetscLogEventBegin(ilinkA->event,ksp,ilinkA->x,ilinkA->y,NULL);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,ilinkA->x,ilinkA->y);CHKERRQ(ierr);
  ierr = KSPCheckSolve(ksp,pc,ilinkA->y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ilinkA->event,ksp,ilinkA->x,ilinkA->y,NULL);CHKERRQ(ierr);
  ierr = MatMultHermitianTranspose(jac->B,ilinkA->y,work1);CHKERRQ(ierr);
  ierr = VecAXPBY(work1,1.0/nu,-1.0,ilinkD->x);CHKERRQ(ierr);            /* c = b - B'*x        */

  /* First step of algorithm */
  ierr  = VecNorm(work1,NORM_2,&beta);CHKERRQ(ierr);                   /* beta = sqrt(nu*c'*c)*/
  KSPCheckDot(ksp,beta);
  beta  = PetscSqrtScalar(nu)*beta;
  ierr  = VecAXPBY(v,nu/beta,0.0,work1);CHKERRQ(ierr);                   /* v = nu/beta *c      */
  ierr  = MatMult(jac->B,v,work2);CHKERRQ(ierr);                       /* u = H^{-1}*B*v      */
  ierr  = PetscLogEventBegin(ilinkA->event,ksp,work2,u,NULL);CHKERRQ(ierr);
  ierr  = KSPSolve(ksp,work2,u);CHKERRQ(ierr);
  ierr  = KSPCheckSolve(ksp,pc,u);CHKERRQ(ierr);
  ierr  = PetscLogEventEnd(ilinkA->event,ksp,work2,u,NULL);CHKERRQ(ierr);
  ierr  = MatMult(jac->H,u,Hu);CHKERRQ(ierr);                          /* alpha = u'*H*u      */
  ierr  = VecDot(Hu,u,&alpha);CHKERRQ(ierr);
  KSPCheckDot(ksp,alpha);
  if (PetscRealPart(alpha) <= 0.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_NOT_CONVERGED,"GKB preconditioner diverged, H is not positive definite");
  alpha = PetscSqrtScalar(PetscAbsScalar(alpha));
  ierr  = VecScale(u,1.0/alpha);CHKERRQ(ierr);
  ierr  = VecAXPBY(d,1.0/alpha,0.0,v);CHKERRQ(ierr);                       /* v = nu/beta *c      */

  z = beta/alpha;
  vecz[1] = z;

  /* Computation of first iterate x(1) and p(1) */
  ierr = VecAXPY(ilinkA->y,z,u);CHKERRQ(ierr);
  ierr = VecCopy(d,ilinkD->y);CHKERRQ(ierr);
  ierr = VecScale(ilinkD->y,-z);CHKERRQ(ierr);

  iterGKB = 1; lowbnd = 2*jac->gkbtol;
  if (jac->gkbmonitor) {
      ierr = PetscViewerASCIIPrintf(jac->gkbviewer,"%3D GKB Lower bound estimate %14.12e\n",iterGKB,lowbnd);CHKERRQ(ierr);
  }

  while (iterGKB < jac->gkbmaxit && lowbnd > jac->gkbtol) {
    iterGKB += 1;
    ierr  = MatMultHermitianTranspose(jac->B,u,work1);CHKERRQ(ierr); /* v <- nu*(B'*u-alpha/nu*v) */
    ierr  = VecAXPBY(v,nu,-alpha,work1);CHKERRQ(ierr);
    ierr  = VecNorm(v,NORM_2,&beta);CHKERRQ(ierr);                   /* beta = sqrt(nu)*v'*v      */
    beta  = beta/PetscSqrtScalar(nu);
    ierr  = VecScale(v,1.0/beta);CHKERRQ(ierr);
    ierr  = MatMult(jac->B,v,work2);CHKERRQ(ierr);                  /* u <- H^{-1}*(B*v-beta*H*u) */
    ierr  = MatMult(jac->H,u,Hu);CHKERRQ(ierr);
    ierr  = VecAXPY(work2,-beta,Hu);CHKERRQ(ierr);
    ierr  = PetscLogEventBegin(ilinkA->event,ksp,work2,u,NULL);CHKERRQ(ierr);
    ierr  = KSPSolve(ksp,work2,u);CHKERRQ(ierr);
    ierr  = KSPCheckSolve(ksp,pc,u);CHKERRQ(ierr);
    ierr  = PetscLogEventEnd(ilinkA->event,ksp,work2,u,NULL);CHKERRQ(ierr);
    ierr  = MatMult(jac->H,u,Hu);CHKERRQ(ierr);                      /* alpha = u'*H*u            */
    ierr  = VecDot(Hu,u,&alpha);CHKERRQ(ierr);
    KSPCheckDot(ksp,alpha);
    if (PetscRealPart(alpha) <= 0.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_NOT_CONVERGED,"GKB preconditioner diverged, H is not positive definite");
    alpha = PetscSqrtScalar(PetscAbsScalar(alpha));
    ierr  = VecScale(u,1.0/alpha);CHKERRQ(ierr);

    z = -beta/alpha*z;                                            /* z <- beta/alpha*z     */
    vecz[0] = z;

    /* Computation of new iterate x(i+1) and p(i+1) */
    ierr = VecAXPBY(d,1.0/alpha,-beta/alpha,v);CHKERRQ(ierr);       /* d = (v-beta*d)/alpha */
    ierr = VecAXPY(ilinkA->y,z,u);CHKERRQ(ierr);                  /* r = r + z*u          */
    ierr = VecAXPY(ilinkD->y,-z,d);CHKERRQ(ierr);                 /* p = p - z*d          */
    ierr = MatMult(jac->H,ilinkA->y,Hu);CHKERRQ(ierr);            /* ||u||_H = u'*H*u     */
    ierr = VecDot(Hu,ilinkA->y,&nrmz2);CHKERRQ(ierr);

    /* Compute Lower Bound estimate */
    if (iterGKB > jac->gkbdelay) {
      lowbnd = 0.0;
      for (j=0; j<jac->gkbdelay; j++) {
        lowbnd += PetscAbsScalar(vecz[j]*vecz[j]);
      }
      lowbnd = PetscSqrtScalar(lowbnd/PetscAbsScalar(nrmz2));
    }

    for (j=0; j<jac->gkbdelay-1; j++) {
      vecz[jac->gkbdelay-j-1] = vecz[jac->gkbdelay-j-2];
    }
    if (jac->gkbmonitor) {
      ierr = PetscViewerASCIIPrintf(jac->gkbviewer,"%3D GKB Lower bound estimate %14.12e\n",iterGKB,lowbnd);CHKERRQ(ierr);
    }
  }

  ierr = VecScatterBegin(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#define FieldSplitSplitSolveAddTranspose(ilink,xx,yy) \
  (VecScatterBegin(ilink->sctx,xx,ilink->y,INSERT_VALUES,SCATTER_FORWARD) || \
   VecScatterEnd(ilink->sctx,xx,ilink->y,INSERT_VALUES,SCATTER_FORWARD) || \
   PetscLogEventBegin(ilink->event,ilink->ksp,ilink->y,ilink->x,NULL) || \
   KSPSolveTranspose(ilink->ksp,ilink->y,ilink->x) ||                  \
   KSPCheckSolve(ilink->ksp,pc,ilink->x) || \
   PetscLogEventEnd(ilink->event,ilink->ksp,ilink->y,ilink->x,NULL) ||   \
   VecScatterBegin(ilink->sctx,ilink->x,yy,ADD_VALUES,SCATTER_REVERSE) || \
   VecScatterEnd(ilink->sctx,ilink->x,yy,ADD_VALUES,SCATTER_REVERSE))

static PetscErrorCode PCApplyTranspose_FieldSplit(PC pc,Vec x,Vec y)
{
  PC_FieldSplit      *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode     ierr;
  PC_FieldSplitLink  ilink = jac->head;
  PetscInt           bs;

  PetscFunctionBegin;
  if (jac->type == PC_COMPOSITE_ADDITIVE) {
    if (jac->defaultsplit) {
      ierr = VecGetBlockSize(x,&bs);CHKERRQ(ierr);
      if (jac->bs > 0 && bs != jac->bs) SETERRQ2(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Blocksize of x vector %D does not match fieldsplit blocksize %D",bs,jac->bs);
      ierr = VecGetBlockSize(y,&bs);CHKERRQ(ierr);
      if (jac->bs > 0 && bs != jac->bs) SETERRQ2(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Blocksize of y vector %D does not match fieldsplit blocksize %D",bs,jac->bs);
      ierr = VecStrideGatherAll(x,jac->x,INSERT_VALUES);CHKERRQ(ierr);
      while (ilink) {
        ierr = PetscLogEventBegin(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL);CHKERRQ(ierr);
        ierr = KSPSolveTranspose(ilink->ksp,ilink->x,ilink->y);CHKERRQ(ierr);
        ierr = KSPCheckSolve(ilink->ksp,pc,ilink->y);CHKERRQ(ierr);
        ierr = PetscLogEventEnd(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL);CHKERRQ(ierr);
        ilink = ilink->next;
      }
      ierr = VecStrideScatterAll(jac->y,y,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      ierr = VecSet(y,0.0);CHKERRQ(ierr);
      while (ilink) {
        ierr = FieldSplitSplitSolveAddTranspose(ilink,x,y);CHKERRQ(ierr);
        ilink = ilink->next;
      }
    }
  } else {
    if (!jac->w1) {
      ierr = VecDuplicate(x,&jac->w1);CHKERRQ(ierr);
      ierr = VecDuplicate(x,&jac->w2);CHKERRQ(ierr);
    }
    ierr = VecSet(y,0.0);CHKERRQ(ierr);
    if (jac->type == PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE) {
      ierr = FieldSplitSplitSolveAddTranspose(ilink,x,y);CHKERRQ(ierr);
      while (ilink->next) {
        ilink = ilink->next;
        ierr  = MatMultTranspose(pc->mat,y,jac->w1);CHKERRQ(ierr);
        ierr  = VecWAXPY(jac->w2,-1.0,jac->w1,x);CHKERRQ(ierr);
        ierr  = FieldSplitSplitSolveAddTranspose(ilink,jac->w2,y);CHKERRQ(ierr);
      }
      while (ilink->previous) {
        ilink = ilink->previous;
        ierr  = MatMultTranspose(pc->mat,y,jac->w1);CHKERRQ(ierr);
        ierr  = VecWAXPY(jac->w2,-1.0,jac->w1,x);CHKERRQ(ierr);
        ierr  = FieldSplitSplitSolveAddTranspose(ilink,jac->w2,y);CHKERRQ(ierr);
      }
    } else {
      while (ilink->next) {   /* get to last entry in linked list */
        ilink = ilink->next;
      }
      ierr = FieldSplitSplitSolveAddTranspose(ilink,x,y);CHKERRQ(ierr);
      while (ilink->previous) {
        ilink = ilink->previous;
        ierr  = MatMultTranspose(pc->mat,y,jac->w1);CHKERRQ(ierr);
        ierr  = VecWAXPY(jac->w2,-1.0,jac->w1,x);CHKERRQ(ierr);
        ierr  = FieldSplitSplitSolveAddTranspose(ilink,jac->w2,y);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_FieldSplit(PC pc)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilink = jac->head,next;

  PetscFunctionBegin;
  while (ilink) {
    ierr  = KSPDestroy(&ilink->ksp);CHKERRQ(ierr);
    ierr  = VecDestroy(&ilink->x);CHKERRQ(ierr);
    ierr  = VecDestroy(&ilink->y);CHKERRQ(ierr);
    ierr  = VecDestroy(&ilink->z);CHKERRQ(ierr);
    ierr  = VecScatterDestroy(&ilink->sctx);CHKERRQ(ierr);
    ierr  = ISDestroy(&ilink->is);CHKERRQ(ierr);
    ierr  = ISDestroy(&ilink->is_col);CHKERRQ(ierr);
    ierr  = PetscFree(ilink->splitname);CHKERRQ(ierr);
    ierr  = PetscFree(ilink->fields);CHKERRQ(ierr);
    ierr  = PetscFree(ilink->fields_col);CHKERRQ(ierr);
    next  = ilink->next;
    ierr  = PetscFree(ilink);CHKERRQ(ierr);
    ilink = next;
  }
  jac->head = NULL;
  ierr = PetscFree2(jac->x,jac->y);CHKERRQ(ierr);
  if (jac->mat && jac->mat != jac->pmat) {
    ierr = MatDestroyMatrices(jac->nsplits,&jac->mat);CHKERRQ(ierr);
  } else if (jac->mat) {
    jac->mat = NULL;
  }
  if (jac->pmat) {ierr = MatDestroyMatrices(jac->nsplits,&jac->pmat);CHKERRQ(ierr);}
  if (jac->Afield) {ierr = MatDestroyMatrices(jac->nsplits,&jac->Afield);CHKERRQ(ierr);}
  jac->nsplits = 0;
  ierr = VecDestroy(&jac->w1);CHKERRQ(ierr);
  ierr = VecDestroy(&jac->w2);CHKERRQ(ierr);
  ierr = MatDestroy(&jac->schur);CHKERRQ(ierr);
  ierr = MatDestroy(&jac->schurp);CHKERRQ(ierr);
  ierr = MatDestroy(&jac->schur_user);CHKERRQ(ierr);
  ierr = KSPDestroy(&jac->kspschur);CHKERRQ(ierr);
  ierr = KSPDestroy(&jac->kspupper);CHKERRQ(ierr);
  ierr = MatDestroy(&jac->B);CHKERRQ(ierr);
  ierr = MatDestroy(&jac->C);CHKERRQ(ierr);
  ierr = MatDestroy(&jac->H);CHKERRQ(ierr);
  ierr = VecDestroy(&jac->u);CHKERRQ(ierr);
  ierr = VecDestroy(&jac->v);CHKERRQ(ierr);
  ierr = VecDestroy(&jac->Hu);CHKERRQ(ierr);
  ierr = VecDestroy(&jac->d);CHKERRQ(ierr);
  ierr = PetscFree(jac->vecz);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&jac->gkbviewer);CHKERRQ(ierr);
  jac->isrestrict = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_FieldSplit(PC pc)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PCReset_FieldSplit(pc);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSchurGetSubKSP_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitGetSubKSP_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetFields_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetIS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetBlockSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetSchurPre_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitGetSchurPre_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetSchurFactType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitRestrictIS_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_FieldSplit(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PetscErrorCode  ierr;
  PetscInt        bs;
  PetscBool       flg;
  PC_FieldSplit   *jac = (PC_FieldSplit*)pc->data;
  PCCompositeType ctype;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"FieldSplit options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_fieldsplit_dm_splits","Whether to use DMCreateFieldDecomposition() for splits","PCFieldSplitSetDMSplits",jac->dm_splits,&jac->dm_splits,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_fieldsplit_block_size","Blocksize that defines number of fields","PCFieldSplitSetBlockSize",jac->bs,&bs,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCFieldSplitSetBlockSize(pc,bs);CHKERRQ(ierr);
  }
  jac->diag_use_amat = pc->useAmat;
  ierr = PetscOptionsBool("-pc_fieldsplit_diag_use_amat","Use Amat (not Pmat) to extract diagonal fieldsplit blocks", "PCFieldSplitSetDiagUseAmat",jac->diag_use_amat,&jac->diag_use_amat,NULL);CHKERRQ(ierr);
  jac->offdiag_use_amat = pc->useAmat;
  ierr = PetscOptionsBool("-pc_fieldsplit_off_diag_use_amat","Use Amat (not Pmat) to extract off-diagonal fieldsplit blocks", "PCFieldSplitSetOffDiagUseAmat",jac->offdiag_use_amat,&jac->offdiag_use_amat,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_fieldsplit_detect_saddle_point","Form 2-way split by detecting zero diagonal entries", "PCFieldSplitSetDetectSaddlePoint",jac->detect,&jac->detect,NULL);CHKERRQ(ierr);
  ierr = PCFieldSplitSetDetectSaddlePoint(pc,jac->detect);CHKERRQ(ierr); /* Sets split type and Schur PC type */
  ierr = PetscOptionsEnum("-pc_fieldsplit_type","Type of composition","PCFieldSplitSetType",PCCompositeTypes,(PetscEnum)jac->type,(PetscEnum*)&ctype,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCFieldSplitSetType(pc,ctype);CHKERRQ(ierr);
  }
  /* Only setup fields once */
  if ((jac->bs > 0) && (jac->nsplits == 0)) {
    /* only allow user to set fields from command line if bs is already known.
       otherwise user can set them in PCFieldSplitSetDefaults() */
    ierr = PCFieldSplitSetRuntimeSplits_Private(pc);CHKERRQ(ierr);
    if (jac->splitdefined) {ierr = PetscInfo(pc,"Splits defined using the options database\n");CHKERRQ(ierr);}
  }
  if (jac->type == PC_COMPOSITE_SCHUR) {
    ierr = PetscOptionsGetEnum(((PetscObject)pc)->options,((PetscObject)pc)->prefix,"-pc_fieldsplit_schur_factorization_type",PCFieldSplitSchurFactTypes,(PetscEnum*)&jac->schurfactorization,&flg);CHKERRQ(ierr);
    if (flg) {ierr = PetscInfo(pc,"Deprecated use of -pc_fieldsplit_schur_factorization_type\n");CHKERRQ(ierr);}
    ierr = PetscOptionsEnum("-pc_fieldsplit_schur_fact_type","Which off-diagonal parts of the block factorization to use","PCFieldSplitSetSchurFactType",PCFieldSplitSchurFactTypes,(PetscEnum)jac->schurfactorization,(PetscEnum*)&jac->schurfactorization,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-pc_fieldsplit_schur_precondition","How to build preconditioner for Schur complement","PCFieldSplitSetSchurPre",PCFieldSplitSchurPreTypes,(PetscEnum)jac->schurpre,(PetscEnum*)&jac->schurpre,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsScalar("-pc_fieldsplit_schur_scale","Scale Schur complement","PCFieldSplitSetSchurScale",jac->schurscale,&jac->schurscale,NULL);CHKERRQ(ierr);
  } else if (jac->type == PC_COMPOSITE_GKB) {
    ierr = PetscOptionsReal("-pc_fieldsplit_gkb_tol","The tolerance for the lower bound stopping criterion","PCFieldSplitGKBTol",jac->gkbtol,&jac->gkbtol,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-pc_fieldsplit_gkb_delay","The delay value for lower bound criterion","PCFieldSplitGKBDelay",jac->gkbdelay,&jac->gkbdelay,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-pc_fieldsplit_gkb_nu","Parameter in augmented Lagrangian approach","PCFieldSplitGKBNu",jac->gkbnu,&jac->gkbnu,NULL);CHKERRQ(ierr);
    if (jac->gkbnu < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nu cannot be less than 0: value %f",jac->gkbnu);
    ierr = PetscOptionsInt("-pc_fieldsplit_gkb_maxit","Maximum allowed number of iterations","PCFieldSplitGKBMaxit",jac->gkbmaxit,&jac->gkbmaxit,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-pc_fieldsplit_gkb_monitor","Prints number of GKB iterations and error","PCFieldSplitGKB",jac->gkbmonitor,&jac->gkbmonitor,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------------------------------*/

static PetscErrorCode  PCFieldSplitSetFields_FieldSplit(PC pc,const char splitname[],PetscInt n,const PetscInt *fields,const PetscInt *fields_col)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilink,next = jac->head;
  char              prefix[128];
  PetscInt          i;

  PetscFunctionBegin;
  if (jac->splitdefined) {
    ierr = PetscInfo1(pc,"Ignoring new split \"%s\" because the splits have already been defined\n",splitname);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  for (i=0; i<n; i++) {
    if (fields[i] >= jac->bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Field %D requested but only %D exist",fields[i],jac->bs);
    if (fields[i] < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative field %D requested",fields[i]);
  }
  ierr = PetscNew(&ilink);CHKERRQ(ierr);
  if (splitname) {
    ierr = PetscStrallocpy(splitname,&ilink->splitname);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc1(3,&ilink->splitname);CHKERRQ(ierr);
    ierr = PetscSNPrintf(ilink->splitname,2,"%s",jac->nsplits);CHKERRQ(ierr);
  }
  ilink->event = jac->nsplits < 5 ? KSP_Solve_FS_0 + jac->nsplits : KSP_Solve_FS_0 + 4; /* Any split great than 4 gets logged in the 4th split */
  ierr = PetscMalloc1(n,&ilink->fields);CHKERRQ(ierr);
  ierr = PetscArraycpy(ilink->fields,fields,n);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&ilink->fields_col);CHKERRQ(ierr);
  ierr = PetscArraycpy(ilink->fields_col,fields_col,n);CHKERRQ(ierr);

  ilink->nfields = n;
  ilink->next    = NULL;
  ierr           = KSPCreate(PetscObjectComm((PetscObject)pc),&ilink->ksp);CHKERRQ(ierr);
  ierr           = KSPSetErrorIfNotConverged(ilink->ksp,pc->erroriffailure);CHKERRQ(ierr);
  ierr           = PetscObjectIncrementTabLevel((PetscObject)ilink->ksp,(PetscObject)pc,1);CHKERRQ(ierr);
  ierr           = KSPSetType(ilink->ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr           = PetscLogObjectParent((PetscObject)pc,(PetscObject)ilink->ksp);CHKERRQ(ierr);

  ierr = PetscSNPrintf(prefix,sizeof(prefix),"%sfieldsplit_%s_",((PetscObject)pc)->prefix ? ((PetscObject)pc)->prefix : "",ilink->splitname);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ilink->ksp,prefix);CHKERRQ(ierr);

  if (!next) {
    jac->head       = ilink;
    ilink->previous = NULL;
  } else {
    while (next->next) {
      next = next->next;
    }
    next->next      = ilink;
    ilink->previous = next;
  }
  jac->nsplits++;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCFieldSplitSchurGetSubKSP_FieldSplit(PC pc,PetscInt *n,KSP **subksp)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *subksp = NULL;
  if (n) *n = 0;
  if (jac->type == PC_COMPOSITE_SCHUR) {
    PetscInt nn;

    if (!jac->schur) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must call KSPSetUp() or PCSetUp() before calling PCFieldSplitSchurGetSubKSP()");
    if (jac->nsplits != 2) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"Unexpected number of splits %D != 2",jac->nsplits);
    nn   = jac->nsplits + (jac->kspupper != jac->head->ksp ? 1 : 0);
    ierr = PetscMalloc1(nn,subksp);CHKERRQ(ierr);
    (*subksp)[0] = jac->head->ksp;
    (*subksp)[1] = jac->kspschur;
    if (jac->kspupper != jac->head->ksp) (*subksp)[2] = jac->kspupper;
    if (n) *n = nn;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCFieldSplitGetSubKSP_FieldSplit_Schur(PC pc,PetscInt *n,KSP **subksp)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!jac->schur) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must call KSPSetUp() or PCSetUp() before calling PCFieldSplitGetSubKSP()");
  ierr = PetscMalloc1(jac->nsplits,subksp);CHKERRQ(ierr);
  ierr = MatSchurComplementGetKSP(jac->schur,*subksp);CHKERRQ(ierr);

  (*subksp)[1] = jac->kspschur;
  if (n) *n = jac->nsplits;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCFieldSplitGetSubKSP_FieldSplit(PC pc,PetscInt *n,KSP **subksp)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PetscInt          cnt   = 0;
  PC_FieldSplitLink ilink = jac->head;

  PetscFunctionBegin;
  ierr = PetscMalloc1(jac->nsplits,subksp);CHKERRQ(ierr);
  while (ilink) {
    (*subksp)[cnt++] = ilink->ksp;
    ilink            = ilink->next;
  }
  if (cnt != jac->nsplits) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Corrupt PCFIELDSPLIT object: number of splits in linked list %D does not match number in object %D",cnt,jac->nsplits);
  if (n) *n = jac->nsplits;
  PetscFunctionReturn(0);
}

/*@C
    PCFieldSplitRestrictIS - Restricts the fieldsplit ISs to be within a given IS.

    Input Parameters:
+   pc  - the preconditioner context
-   is - the index set that defines the indices to which the fieldsplit is to be restricted

    Level: advanced

@*/
PetscErrorCode  PCFieldSplitRestrictIS(PC pc,IS isy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(isy,IS_CLASSID,2);
  ierr = PetscTryMethod(pc,"PCFieldSplitRestrictIS_C",(PC,IS),(pc,isy));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode  PCFieldSplitRestrictIS_FieldSplit(PC pc, IS isy)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilink = jac->head, next;
  PetscInt          localsize,size,sizez,i;
  const PetscInt    *ind, *indz;
  PetscInt          *indc, *indcz;
  PetscBool         flg;

  PetscFunctionBegin;
  ierr = ISGetLocalSize(isy,&localsize);CHKERRQ(ierr);
  ierr = MPI_Scan(&localsize,&size,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)isy));CHKERRQ(ierr);
  size -= localsize;
  while(ilink) {
    IS isrl,isr;
    PC subpc;
    ierr          = ISEmbed(ilink->is, isy, PETSC_TRUE, &isrl);CHKERRQ(ierr);
    ierr          = ISGetLocalSize(isrl,&localsize);CHKERRQ(ierr);
    ierr          = PetscMalloc1(localsize,&indc);CHKERRQ(ierr);
    ierr          = ISGetIndices(isrl,&ind);CHKERRQ(ierr);
    ierr          = PetscArraycpy(indc,ind,localsize);CHKERRQ(ierr);
    ierr          = ISRestoreIndices(isrl,&ind);CHKERRQ(ierr);
    ierr          = ISDestroy(&isrl);CHKERRQ(ierr);
    for (i=0; i<localsize; i++) *(indc+i) += size;
    ierr          = ISCreateGeneral(PetscObjectComm((PetscObject)isy),localsize,indc,PETSC_OWN_POINTER,&isr);CHKERRQ(ierr);
    ierr          = PetscObjectReference((PetscObject)isr);CHKERRQ(ierr);
    ierr          = ISDestroy(&ilink->is);CHKERRQ(ierr);
    ilink->is     = isr;
    ierr          = PetscObjectReference((PetscObject)isr);CHKERRQ(ierr);
    ierr          = ISDestroy(&ilink->is_col);CHKERRQ(ierr);
    ilink->is_col = isr;
    ierr          = ISDestroy(&isr);CHKERRQ(ierr);
    ierr          = KSPGetPC(ilink->ksp, &subpc);CHKERRQ(ierr);
    ierr          = PetscObjectTypeCompare((PetscObject)subpc,PCFIELDSPLIT,&flg);CHKERRQ(ierr);
    if(flg) {
      IS iszl,isz;
      MPI_Comm comm;
      ierr   = ISGetLocalSize(ilink->is,&localsize);CHKERRQ(ierr);
      comm   = PetscObjectComm((PetscObject)ilink->is);
      ierr   = ISEmbed(isy, ilink->is, PETSC_TRUE, &iszl);CHKERRQ(ierr);
      ierr   = MPI_Scan(&localsize,&sizez,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
      sizez -= localsize;
      ierr   = ISGetLocalSize(iszl,&localsize);CHKERRQ(ierr);
      ierr   = PetscMalloc1(localsize,&indcz);CHKERRQ(ierr);
      ierr   = ISGetIndices(iszl,&indz);CHKERRQ(ierr);
      ierr   = PetscArraycpy(indcz,indz,localsize);CHKERRQ(ierr);
      ierr   = ISRestoreIndices(iszl,&indz);CHKERRQ(ierr);
      ierr   = ISDestroy(&iszl);CHKERRQ(ierr);
      for (i=0; i<localsize; i++) *(indcz+i) += sizez;
      ierr   = ISCreateGeneral(comm,localsize,indcz,PETSC_OWN_POINTER,&isz);CHKERRQ(ierr);
      ierr   = PCFieldSplitRestrictIS(subpc,isz);CHKERRQ(ierr);
      ierr   = ISDestroy(&isz);CHKERRQ(ierr);
    }
    next = ilink->next;
    ilink = next;
  }
  jac->isrestrict = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCFieldSplitSetIS_FieldSplit(PC pc,const char splitname[],IS is)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilink, next = jac->head;
  char              prefix[128];

  PetscFunctionBegin;
  if (jac->splitdefined) {
    ierr = PetscInfo1(pc,"Ignoring new split \"%s\" because the splits have already been defined\n",splitname);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscNew(&ilink);CHKERRQ(ierr);
  if (splitname) {
    ierr = PetscStrallocpy(splitname,&ilink->splitname);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc1(8,&ilink->splitname);CHKERRQ(ierr);
    ierr = PetscSNPrintf(ilink->splitname,7,"%D",jac->nsplits);CHKERRQ(ierr);
  }
  ilink->event  = jac->nsplits < 5 ? KSP_Solve_FS_0 + jac->nsplits : KSP_Solve_FS_0 + 4; /* Any split great than 4 gets logged in the 4th split */
  ierr          = PetscObjectReference((PetscObject)is);CHKERRQ(ierr);
  ierr          = ISDestroy(&ilink->is);CHKERRQ(ierr);
  ilink->is     = is;
  ierr          = PetscObjectReference((PetscObject)is);CHKERRQ(ierr);
  ierr          = ISDestroy(&ilink->is_col);CHKERRQ(ierr);
  ilink->is_col = is;
  ilink->next   = NULL;
  ierr          = KSPCreate(PetscObjectComm((PetscObject)pc),&ilink->ksp);CHKERRQ(ierr);
  ierr          = KSPSetErrorIfNotConverged(ilink->ksp,pc->erroriffailure);CHKERRQ(ierr);
  ierr          = PetscObjectIncrementTabLevel((PetscObject)ilink->ksp,(PetscObject)pc,1);CHKERRQ(ierr);
  ierr          = KSPSetType(ilink->ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr          = PetscLogObjectParent((PetscObject)pc,(PetscObject)ilink->ksp);CHKERRQ(ierr);

  ierr = PetscSNPrintf(prefix,sizeof(prefix),"%sfieldsplit_%s_",((PetscObject)pc)->prefix ? ((PetscObject)pc)->prefix : "",ilink->splitname);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ilink->ksp,prefix);CHKERRQ(ierr);

  if (!next) {
    jac->head       = ilink;
    ilink->previous = NULL;
  } else {
    while (next->next) {
      next = next->next;
    }
    next->next      = ilink;
    ilink->previous = next;
  }
  jac->nsplits++;
  PetscFunctionReturn(0);
}

/*@C
    PCFieldSplitSetFields - Sets the fields for one particular split in the field split preconditioner

    Logically Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
.   splitname - name of this split, if NULL the number of the split is used
.   n - the number of fields in this split
-   fields - the fields in this split

    Level: intermediate

    Notes:
    Use PCFieldSplitSetIS() to set a completely general set of indices as a field.

     The PCFieldSplitSetFields() is for defining fields as strided blocks. For example, if the block
     size is three then one can define a field as 0, or 1 or 2 or 0,1 or 0,2 or 1,2 which mean
     0xx3xx6xx9xx12 ... x1xx4xx7xx ... xx2xx5xx8xx.. 01x34x67x... 0x1x3x5x7.. x12x45x78x....
     where the numbered entries indicate what is in the field.

     This function is called once per split (it creates a new split each time).  Solve options
     for this split will be available under the prefix -fieldsplit_SPLITNAME_.

     Developer Note: This routine does not actually create the IS representing the split, that is delayed
     until PCSetUp_FieldSplit(), because information about the vector/matrix layouts may not be
     available when this routine is called.

.seealso: PCFieldSplitGetSubKSP(), PCFIELDSPLIT, PCFieldSplitSetBlockSize(), PCFieldSplitSetIS()

@*/
PetscErrorCode  PCFieldSplitSetFields(PC pc,const char splitname[],PetscInt n,const PetscInt *fields,const PetscInt *fields_col)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidCharPointer(splitname,2);
  if (n < 1) SETERRQ2(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Provided number of fields %D in split \"%s\" not positive",n,splitname);
  PetscValidIntPointer(fields,3);
  ierr = PetscTryMethod(pc,"PCFieldSplitSetFields_C",(PC,const char[],PetscInt,const PetscInt*,const PetscInt*),(pc,splitname,n,fields,fields_col));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    PCFieldSplitSetDiagUseAmat - set flag indicating whether to extract diagonal blocks from Amat (rather than Pmat)

    Logically Collective on PC

    Input Parameters:
+   pc  - the preconditioner object
-   flg - boolean flag indicating whether or not to use Amat to extract the diagonal blocks from

    Options Database:
.     -pc_fieldsplit_diag_use_amat

    Level: intermediate

.seealso: PCFieldSplitGetDiagUseAmat(), PCFieldSplitSetOffDiagUseAmat(), PCFIELDSPLIT

@*/
PetscErrorCode  PCFieldSplitSetDiagUseAmat(PC pc,PetscBool flg)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;
  PetscBool      isfs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCFIELDSPLIT,&isfs);CHKERRQ(ierr);
  if (!isfs) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"PC not of type %s",PCFIELDSPLIT);
  jac->diag_use_amat = flg;
  PetscFunctionReturn(0);
}

/*@
    PCFieldSplitGetDiagUseAmat - get the flag indicating whether to extract diagonal blocks from Amat (rather than Pmat)

    Logically Collective on PC

    Input Parameters:
.   pc  - the preconditioner object

    Output Parameters:
.   flg - boolean flag indicating whether or not to use Amat to extract the diagonal blocks from


    Level: intermediate

.seealso: PCFieldSplitSetDiagUseAmat(), PCFieldSplitGetOffDiagUseAmat(), PCFIELDSPLIT

@*/
PetscErrorCode  PCFieldSplitGetDiagUseAmat(PC pc,PetscBool *flg)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;
  PetscBool      isfs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCFIELDSPLIT,&isfs);CHKERRQ(ierr);
  if (!isfs) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"PC not of type %s",PCFIELDSPLIT);
  *flg = jac->diag_use_amat;
  PetscFunctionReturn(0);
}

/*@
    PCFieldSplitSetOffDiagUseAmat - set flag indicating whether to extract off-diagonal blocks from Amat (rather than Pmat)

    Logically Collective on PC

    Input Parameters:
+   pc  - the preconditioner object
-   flg - boolean flag indicating whether or not to use Amat to extract the off-diagonal blocks from

    Options Database:
.     -pc_fieldsplit_off_diag_use_amat

    Level: intermediate

.seealso: PCFieldSplitGetOffDiagUseAmat(), PCFieldSplitSetDiagUseAmat(), PCFIELDSPLIT

@*/
PetscErrorCode  PCFieldSplitSetOffDiagUseAmat(PC pc,PetscBool flg)
{
  PC_FieldSplit *jac = (PC_FieldSplit*)pc->data;
  PetscBool      isfs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCFIELDSPLIT,&isfs);CHKERRQ(ierr);
  if (!isfs) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"PC not of type %s",PCFIELDSPLIT);
  jac->offdiag_use_amat = flg;
  PetscFunctionReturn(0);
}

/*@
    PCFieldSplitGetOffDiagUseAmat - get the flag indicating whether to extract off-diagonal blocks from Amat (rather than Pmat)

    Logically Collective on PC

    Input Parameters:
.   pc  - the preconditioner object

    Output Parameters:
.   flg - boolean flag indicating whether or not to use Amat to extract the off-diagonal blocks from


    Level: intermediate

.seealso: PCFieldSplitSetOffDiagUseAmat(), PCFieldSplitGetDiagUseAmat(), PCFIELDSPLIT

@*/
PetscErrorCode  PCFieldSplitGetOffDiagUseAmat(PC pc,PetscBool *flg)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;
  PetscBool      isfs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCFIELDSPLIT,&isfs);CHKERRQ(ierr);
  if (!isfs) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"PC not of type %s",PCFIELDSPLIT);
  *flg = jac->offdiag_use_amat;
  PetscFunctionReturn(0);
}



/*@C
    PCFieldSplitSetIS - Sets the exact elements for field

    Logically Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
.   splitname - name of this split, if NULL the number of the split is used
-   is - the index set that defines the vector elements in this field


    Notes:
    Use PCFieldSplitSetFields(), for fields defined by strided types.

    This function is called once per split (it creates a new split each time).  Solve options
    for this split will be available under the prefix -fieldsplit_SPLITNAME_.

    Level: intermediate

.seealso: PCFieldSplitGetSubKSP(), PCFIELDSPLIT, PCFieldSplitSetBlockSize()

@*/
PetscErrorCode  PCFieldSplitSetIS(PC pc,const char splitname[],IS is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (splitname) PetscValidCharPointer(splitname,2);
  PetscValidHeaderSpecific(is,IS_CLASSID,3);
  ierr = PetscTryMethod(pc,"PCFieldSplitSetIS_C",(PC,const char[],IS),(pc,splitname,is));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    PCFieldSplitGetIS - Retrieves the elements for a field as an IS

    Logically Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
-   splitname - name of this split

    Output Parameter:
-   is - the index set that defines the vector elements in this field, or NULL if the field is not found

    Level: intermediate

.seealso: PCFieldSplitGetSubKSP(), PCFIELDSPLIT, PCFieldSplitSetIS()

@*/
PetscErrorCode PCFieldSplitGetIS(PC pc,const char splitname[],IS *is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidCharPointer(splitname,2);
  PetscValidPointer(is,3);
  {
    PC_FieldSplit     *jac  = (PC_FieldSplit*) pc->data;
    PC_FieldSplitLink ilink = jac->head;
    PetscBool         found;

    *is = NULL;
    while (ilink) {
      ierr = PetscStrcmp(ilink->splitname, splitname, &found);CHKERRQ(ierr);
      if (found) {
        *is = ilink->is;
        break;
      }
      ilink = ilink->next;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
    PCFieldSplitGetISByIndex - Retrieves the elements for a given index field as an IS

    Logically Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
-   index - index of this split

    Output Parameter:
-   is - the index set that defines the vector elements in this field

    Level: intermediate

.seealso: PCFieldSplitGetSubKSP(), PCFIELDSPLIT, PCFieldSplitGetIS(), PCFieldSplitSetIS()

@*/
PetscErrorCode PCFieldSplitGetISByIndex(PC pc,PetscInt index,IS *is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (index < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative field %D requested",index);
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(is,3);
  {
    PC_FieldSplit     *jac  = (PC_FieldSplit*) pc->data;
    PC_FieldSplitLink ilink = jac->head;
    PetscInt          i     = 0;
    if (index >= jac->nsplits) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Field %D requested but only %D exist",index,jac->nsplits);

    while (i < index) {
      ilink = ilink->next;
      ++i;
    }
    ierr = PCFieldSplitGetIS(pc,ilink->splitname,is);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
    PCFieldSplitSetBlockSize - Sets the block size for defining where fields start in the
      fieldsplit preconditioner. If not set the matrix block size is used.

    Logically Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
-   bs - the block size

    Level: intermediate

.seealso: PCFieldSplitGetSubKSP(), PCFIELDSPLIT, PCFieldSplitSetFields()

@*/
PetscErrorCode  PCFieldSplitSetBlockSize(PC pc,PetscInt bs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,bs,2);
  ierr = PetscTryMethod(pc,"PCFieldSplitSetBlockSize_C",(PC,PetscInt),(pc,bs));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PCFieldSplitGetSubKSP - Gets the KSP contexts for all splits

   Collective on KSP

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  n - the number of splits
-  subksp - the array of KSP contexts

   Note:
   After PCFieldSplitGetSubKSP() the array of KSPs is to be freed by the user with PetscFree()
   (not the KSP just the array that contains them).

   You must call PCSetUp() before calling PCFieldSplitGetSubKSP().

   If the fieldsplit is of type PC_COMPOSITE_SCHUR, it returns the KSP object used inside the
   Schur complement and the KSP object used to iterate over the Schur complement.
   To access all the KSP objects used in PC_COMPOSITE_SCHUR, use PCFieldSplitSchurGetSubKSP().

   If the fieldsplit is of type PC_COMPOSITE_GKB, it returns the KSP object used to solve the
   inner linear system defined by the matrix H in each loop.

   Fortran Usage: You must pass in a KSP array that is large enough to contain all the local KSPs.
      You can call PCFieldSplitGetSubKSP(pc,n,PETSC_NULL_KSP,ierr) to determine how large the
      KSP array must be.


   Level: advanced

.seealso: PCFIELDSPLIT
@*/
PetscErrorCode  PCFieldSplitGetSubKSP(PC pc,PetscInt *n,KSP *subksp[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (n) PetscValidIntPointer(n,2);
  ierr = PetscUseMethod(pc,"PCFieldSplitGetSubKSP_C",(PC,PetscInt*,KSP **),(pc,n,subksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PCFieldSplitSchurGetSubKSP - Gets the KSP contexts used inside the Schur complement based PCFIELDSPLIT

   Collective on KSP

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  n - the number of splits
-  subksp - the array of KSP contexts

   Note:
   After PCFieldSplitSchurGetSubKSP() the array of KSPs is to be freed by the user with PetscFree()
   (not the KSP just the array that contains them).

   You must call PCSetUp() before calling PCFieldSplitSchurGetSubKSP().

   If the fieldsplit type is of type PC_COMPOSITE_SCHUR, it returns (in order)
   - the KSP used for the (1,1) block
   - the KSP used for the Schur complement (not the one used for the interior Schur solver)
   - the KSP used for the (1,1) block in the upper triangular factor (if different from that of the (1,1) block).

   It returns a null array if the fieldsplit is not of type PC_COMPOSITE_SCHUR; in this case, you should use PCFieldSplitGetSubKSP().

   Fortran Usage: You must pass in a KSP array that is large enough to contain all the local KSPs.
      You can call PCFieldSplitSchurGetSubKSP(pc,n,PETSC_NULL_KSP,ierr) to determine how large the
      KSP array must be.

   Level: advanced

.seealso: PCFIELDSPLIT
@*/
PetscErrorCode  PCFieldSplitSchurGetSubKSP(PC pc,PetscInt *n,KSP *subksp[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (n) PetscValidIntPointer(n,2);
  ierr = PetscUseMethod(pc,"PCFieldSplitSchurGetSubKSP_C",(PC,PetscInt*,KSP **),(pc,n,subksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    PCFieldSplitSetSchurPre -  Indicates from what operator the preconditioner is constructucted for the Schur complement.
      The default is the A11 matrix. 

    Collective on PC

    Input Parameters:
+   pc      - the preconditioner context
.   ptype   - which matrix to use for preconditioning the Schur complement: PC_FIELDSPLIT_SCHUR_PRE_A11 (default), PC_FIELDSPLIT_SCHUR_PRE_SELF, PC_FIELDSPLIT_SCHUR_PRE_USER
              PC_FIELDSPLIT_SCHUR_PRE_SELFP, and PC_FIELDSPLIT_SCHUR_PRE_FULL
-   userpre - matrix to use for preconditioning, or NULL

    Options Database:
+    -pc_fieldsplit_schur_precondition <self,selfp,user,a11,full> - default is a11. See notes for meaning of various arguments
-    -fieldsplit_1_pc_type <pctype> - the preconditioner algorithm that is used to construct the preconditioner from the operator

    Notes:
$    If ptype is
$        a11 - the preconditioner for the Schur complement is generated from the block diagonal part of the preconditioner
$        matrix associated with the Schur complement (i.e. A11), not the Schur complement matrix
$        self - the preconditioner for the Schur complement is generated from the symbolic representation of the Schur complement matrix:
$             The only preconditioner that currently works with this symbolic respresentation matrix object is the PCLSC
$             preconditioner
$        user - the preconditioner for the Schur complement is generated from the user provided matrix (pre argument
$             to this function).
$        selfp - the preconditioning for the Schur complement is generated from an explicitly-assembled approximation Sp = A11 - A10 inv(diag(A00)) A01
$             This is only a good preconditioner when diag(A00) is a good preconditioner for A00. Optionally, A00 can be
$             lumped before extracting the diagonal using the additional option -fieldsplit_1_mat_schur_complement_ainv_type lump
$        full - the preconditioner for the Schur complement is generated from the exact Schur complement matrix representation computed internally by PCFIELDSPLIT (this is expensive)
$             useful mostly as a test that the Schur complement approach can work for your problem

     When solving a saddle point problem, where the A11 block is identically zero, using a11 as the ptype only makes sense
    with the additional option -fieldsplit_1_pc_type none. Usually for saddle point problems one would use a ptype of self and
    -fieldsplit_1_pc_type lsc which uses the least squares commutator to compute a preconditioner for the Schur complement.

    Level: intermediate

.seealso: PCFieldSplitGetSchurPre(), PCFieldSplitGetSubKSP(), PCFIELDSPLIT, PCFieldSplitSetFields(), PCFieldSplitSchurPreType,
          MatSchurComplementSetAinvType(), PCLSC

@*/
PetscErrorCode PCFieldSplitSetSchurPre(PC pc,PCFieldSplitSchurPreType ptype,Mat pre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCFieldSplitSetSchurPre_C",(PC,PCFieldSplitSchurPreType,Mat),(pc,ptype,pre));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCFieldSplitSchurPrecondition(PC pc,PCFieldSplitSchurPreType ptype,Mat pre) {return PCFieldSplitSetSchurPre(pc,ptype,pre);} /* Deprecated name */

/*@
    PCFieldSplitGetSchurPre - For Schur complement fieldsplit, determine how the Schur complement will be
    preconditioned.  See PCFieldSplitSetSchurPre() for details.

    Logically Collective on PC

    Input Parameters:
.   pc      - the preconditioner context

    Output Parameters:
+   ptype   - which matrix to use for preconditioning the Schur complement: PC_FIELDSPLIT_SCHUR_PRE_A11, PC_FIELDSPLIT_SCHUR_PRE_SELF, PC_FIELDSPLIT_PRE_USER
-   userpre - matrix to use for preconditioning (with PC_FIELDSPLIT_PRE_USER), or NULL

    Level: intermediate

.seealso: PCFieldSplitSetSchurPre(), PCFieldSplitGetSubKSP(), PCFIELDSPLIT, PCFieldSplitSetFields(), PCFieldSplitSchurPreType, PCLSC

@*/
PetscErrorCode PCFieldSplitGetSchurPre(PC pc,PCFieldSplitSchurPreType *ptype,Mat *pre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCFieldSplitGetSchurPre_C",(PC,PCFieldSplitSchurPreType*,Mat*),(pc,ptype,pre));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    PCFieldSplitSchurGetS -  extract the MatSchurComplement object used by this PC in case it needs to be configured separately

    Not collective

    Input Parameter:
.   pc      - the preconditioner context

    Output Parameter:
.   S       - the Schur complement matrix

    Notes:
    This matrix should not be destroyed using MatDestroy(); rather, use PCFieldSplitSchurRestoreS().

    Level: advanced

.seealso: PCFieldSplitGetSubKSP(), PCFIELDSPLIT, PCFieldSplitSchurPreType, PCFieldSplitSetSchurPre(), MatSchurComplement, PCFieldSplitSchurRestoreS()

@*/
PetscErrorCode  PCFieldSplitSchurGetS(PC pc,Mat *S)
{
  PetscErrorCode ierr;
  const char*    t;
  PetscBool      isfs;
  PC_FieldSplit  *jac;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscObjectGetType((PetscObject)pc,&t);CHKERRQ(ierr);
  ierr = PetscStrcmp(t,PCFIELDSPLIT,&isfs);CHKERRQ(ierr);
  if (!isfs) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Expected PC of type PCFIELDSPLIT, got %s instead",t);
  jac = (PC_FieldSplit*)pc->data;
  if (jac->type != PC_COMPOSITE_SCHUR) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Expected PCFIELDSPLIT of type SCHUR, got %D instead",jac->type);
  if (S) *S = jac->schur;
  PetscFunctionReturn(0);
}

/*@
    PCFieldSplitSchurRestoreS -  restores the MatSchurComplement object used by this PC

    Not collective

    Input Parameters:
+   pc      - the preconditioner context
-   S       - the Schur complement matrix

    Level: advanced

.seealso: PCFieldSplitGetSubKSP(), PCFIELDSPLIT, PCFieldSplitSchurPreType, PCFieldSplitSetSchurPre(), MatSchurComplement, PCFieldSplitSchurGetS()

@*/
PetscErrorCode  PCFieldSplitSchurRestoreS(PC pc,Mat *S)
{
  PetscErrorCode ierr;
  const char*    t;
  PetscBool      isfs;
  PC_FieldSplit  *jac;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscObjectGetType((PetscObject)pc,&t);CHKERRQ(ierr);
  ierr = PetscStrcmp(t,PCFIELDSPLIT,&isfs);CHKERRQ(ierr);
  if (!isfs) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Expected PC of type PCFIELDSPLIT, got %s instead",t);
  jac = (PC_FieldSplit*)pc->data;
  if (jac->type != PC_COMPOSITE_SCHUR) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Expected PCFIELDSPLIT of type SCHUR, got %D instead",jac->type);
  if (!S || *S != jac->schur) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"MatSchurComplement restored is not the same as gotten");
  PetscFunctionReturn(0);
}


static PetscErrorCode  PCFieldSplitSetSchurPre_FieldSplit(PC pc,PCFieldSplitSchurPreType ptype,Mat pre)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  jac->schurpre = ptype;
  if (ptype == PC_FIELDSPLIT_SCHUR_PRE_USER && pre) {
    ierr            = MatDestroy(&jac->schur_user);CHKERRQ(ierr);
    jac->schur_user = pre;
    ierr            = PetscObjectReference((PetscObject)jac->schur_user);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCFieldSplitGetSchurPre_FieldSplit(PC pc,PCFieldSplitSchurPreType *ptype,Mat *pre)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;

  PetscFunctionBegin;
  *ptype = jac->schurpre;
  *pre   = jac->schur_user;
  PetscFunctionReturn(0);
}

/*@
    PCFieldSplitSetSchurFactType -  sets which blocks of the approximate block factorization to retain in the preconditioner

    Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
-   ftype - which blocks of factorization to retain, PC_FIELDSPLIT_SCHUR_FACT_FULL is default

    Options Database:
.     -pc_fieldsplit_schur_fact_type <diag,lower,upper,full> default is full


    Level: intermediate

    Notes:
    The FULL factorization is

$   (A   B)  = (1       0) (A   0) (1  Ainv*B)  = L D U
$   (C   E)    (C*Ainv  1) (0   S) (0     1  )

    where S = E - C*Ainv*B. In practice, the full factorization is applied via block triangular solves with the grouping L*(D*U). UPPER uses D*U, LOWER uses L*D,
    and DIAG is the diagonal part with the sign of S flipped (because this makes the preconditioner positive definite for many formulations, thus allowing the use of KSPMINRES). Sign flipping of S can be turned off with PCFieldSplitSetSchurScale().

$    If A and S are solved exactly
$      *) FULL factorization is a direct solver.
$      *) The preconditioned operator with LOWER or UPPER has all eigenvalues equal to 1 and minimal polynomial of degree 2, so KSPGMRES converges in 2 iterations.
$      *) With DIAG, the preconditioned operator has three distinct nonzero eigenvalues and minimal polynomial of degree at most 4, so KSPGMRES converges in at most 4 iterations.

    If the iteration count is very low, consider using KSPFGMRES or KSPGCR which can use one less preconditioner
    application in this case. Note that the preconditioned operator may be highly non-normal, so such fast convergence may not be observed in practice.

    For symmetric problems in which A is positive definite and S is negative definite, DIAG can be used with KSPMINRES.

    Note that a flexible method like KSPFGMRES or KSPGCR must be used if the fieldsplit preconditioner is nonlinear (e.g. a few iterations of a Krylov method is used to solve with A or S).

    References:
+   1. - Murphy, Golub, and Wathen, A note on preconditioning indefinite linear systems, SIAM J. Sci. Comput., 21 (2000).
-   2. - Ipsen, A note on preconditioning nonsymmetric matrices, SIAM J. Sci. Comput., 23 (2001).

.seealso: PCFieldSplitGetSubKSP(), PCFIELDSPLIT, PCFieldSplitSetFields(), PCFieldSplitSchurPreType, PCFieldSplitSetSchurScale()
@*/
PetscErrorCode  PCFieldSplitSetSchurFactType(PC pc,PCFieldSplitSchurFactType ftype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCFieldSplitSetSchurFactType_C",(PC,PCFieldSplitSchurFactType),(pc,ftype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCFieldSplitSetSchurFactType_FieldSplit(PC pc,PCFieldSplitSchurFactType ftype)
{
  PC_FieldSplit *jac = (PC_FieldSplit*)pc->data;

  PetscFunctionBegin;
  jac->schurfactorization = ftype;
  PetscFunctionReturn(0);
}

/*@
    PCFieldSplitSetSchurScale -  Controls the sign flip of S for PC_FIELDSPLIT_SCHUR_FACT_DIAG.

    Collective on PC

    Input Parameters:
+   pc    - the preconditioner context
-   scale - scaling factor for the Schur complement

    Options Database:
.     -pc_fieldsplit_schur_scale - default is -1.0

    Level: intermediate

.seealso: PCFIELDSPLIT, PCFieldSplitSetFields(), PCFieldSplitSchurFactType, PCFieldSplitSetSchurScale()
@*/
PetscErrorCode PCFieldSplitSetSchurScale(PC pc,PetscScalar scale)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveScalar(pc,scale,2);
  ierr = PetscTryMethod(pc,"PCFieldSplitSetSchurScale_C",(PC,PetscScalar),(pc,scale));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCFieldSplitSetSchurScale_FieldSplit(PC pc,PetscScalar scale)
{
  PC_FieldSplit *jac = (PC_FieldSplit*)pc->data;

  PetscFunctionBegin;
  jac->schurscale = scale;
  PetscFunctionReturn(0);
}

/*@C
   PCFieldSplitGetSchurBlocks - Gets all matrix blocks for the Schur complement

   Collective on KSP

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  A00 - the (0,0) block
.  A01 - the (0,1) block
.  A10 - the (1,0) block
-  A11 - the (1,1) block

   Level: advanced

.seealso: PCFIELDSPLIT
@*/
PetscErrorCode  PCFieldSplitGetSchurBlocks(PC pc,Mat *A00,Mat *A01,Mat *A10, Mat *A11)
{
  PC_FieldSplit *jac = (PC_FieldSplit*) pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (jac->type != PC_COMPOSITE_SCHUR) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG, "FieldSplit is not using a Schur complement approach.");
  if (A00) *A00 = jac->pmat[0];
  if (A01) *A01 = jac->B;
  if (A10) *A10 = jac->C;
  if (A11) *A11 = jac->pmat[1];
  PetscFunctionReturn(0);
}

/*@
    PCFieldSplitSetGKBTol -  Sets the solver tolerance for the generalized Golub-Kahan bidiagonalization preconditioner.

    Collective on PC

    Notes:
    The generalized GKB algorithm uses a lower bound estimate of the error in energy norm as stopping criterion.
    It stops once the lower bound estimate undershoots the required solver tolerance. Although the actual error might be bigger than
    this estimate, the stopping criterion is satisfactory in practical cases [A13].

[Ar13] Generalized Golub-Kahan bidiagonalization and stopping criteria, SIAM J. Matrix Anal. Appl., Vol. 34, No. 2, pp. 571-592, 2013.

    Input Parameters:
+   pc        - the preconditioner context
-   tolerance - the solver tolerance

    Options Database:
.     -pc_fieldsplit_gkb_tol - default is 1e-5

    Level: intermediate

.seealso: PCFIELDSPLIT, PCFieldSplitSetGKBDelay(), PCFieldSplitSetGKBNu(), PCFieldSplitSetGKBMaxit()
@*/
PetscErrorCode PCFieldSplitSetGKBTol(PC pc,PetscReal tolerance)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveReal(pc,tolerance,2);
  ierr = PetscTryMethod(pc,"PCFieldSplitSetGKBTol_C",(PC,PetscReal),(pc,tolerance));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCFieldSplitSetGKBTol_FieldSplit(PC pc,PetscReal tolerance)
{
  PC_FieldSplit *jac = (PC_FieldSplit*)pc->data;

  PetscFunctionBegin;
  jac->gkbtol = tolerance;
  PetscFunctionReturn(0);
}


/*@
    PCFieldSplitSetGKBMaxit -  Sets the maximum number of iterations for the generalized Golub-Kahan bidiagonalization
    preconditioner.

    Collective on PC

    Input Parameters:
+   pc     - the preconditioner context
-   maxit  - the maximum number of iterations

    Options Database:
.     -pc_fieldsplit_gkb_maxit - default is 100

    Level: intermediate

.seealso: PCFIELDSPLIT, PCFieldSplitSetGKBDelay(), PCFieldSplitSetGKBTol(), PCFieldSplitSetGKBNu()
@*/
PetscErrorCode PCFieldSplitSetGKBMaxit(PC pc,PetscInt maxit)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,maxit,2);
  ierr = PetscTryMethod(pc,"PCFieldSplitSetGKBMaxit_C",(PC,PetscInt),(pc,maxit));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCFieldSplitSetGKBMaxit_FieldSplit(PC pc,PetscInt maxit)
{
  PC_FieldSplit *jac = (PC_FieldSplit*)pc->data;

  PetscFunctionBegin;
  jac->gkbmaxit = maxit;
  PetscFunctionReturn(0);
}

/*@
    PCFieldSplitSetGKBDelay -  Sets the delay in the lower bound error estimate in the generalized Golub-Kahan bidiagonalization
    preconditioner.

    Collective on PC

    Notes:
    The algorithm uses a lower bound estimate of the error in energy norm as stopping criterion. The lower bound of the error ||u-u^k||_H
    is expressed as a truncated sum. The error at iteration k can only be measured at iteration (k + delay), and thus the algorithm needs
    at least (delay + 1) iterations to stop. For more details on the generalized Golub-Kahan bidiagonalization method and its lower bound stopping criterion, please refer to

[Ar13] Generalized Golub-Kahan bidiagonalization and stopping criteria, SIAM J. Matrix Anal. Appl., Vol. 34, No. 2, pp. 571-592, 2013.

    Input Parameters:
+   pc     - the preconditioner context
-   delay  - the delay window in the lower bound estimate

    Options Database:
.     -pc_fieldsplit_gkb_delay - default is 5

    Level: intermediate

.seealso: PCFIELDSPLIT, PCFieldSplitSetGKBNu(), PCFieldSplitSetGKBTol(), PCFieldSplitSetGKBMaxit()
@*/
PetscErrorCode PCFieldSplitSetGKBDelay(PC pc,PetscInt delay)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,delay,2);
  ierr = PetscTryMethod(pc,"PCFieldSplitSetGKBDelay_C",(PC,PetscInt),(pc,delay));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCFieldSplitSetGKBDelay_FieldSplit(PC pc,PetscInt delay)
{
  PC_FieldSplit *jac = (PC_FieldSplit*)pc->data;

  PetscFunctionBegin;
  jac->gkbdelay = delay;
  PetscFunctionReturn(0);
}

/*@
    PCFieldSplitSetGKBNu -  Sets the scalar value nu >= 0 in the transformation H = A00 + nu*A01*A01' of the (1,1) block in the Golub-Kahan bidiagonalization preconditioner.

    Collective on PC

    Notes:
    This shift is in general done to obtain better convergence properties for the outer loop of the algorithm. This is often achieved by chosing nu sufficiently big. However,
    if nu is chosen too big, the matrix H might be badly conditioned and the solution of the linear system Hx = b in the inner loop gets difficult. It is therefore
    necessary to find a good balance in between the convergence of the inner and outer loop.

    For nu = 0, no shift is done. In this case A00 has to be positive definite. The matrix N in [Ar13] is then chosen as identity.

[Ar13] Generalized Golub-Kahan bidiagonalization and stopping criteria, SIAM J. Matrix Anal. Appl., Vol. 34, No. 2, pp. 571-592, 2013.

    Input Parameters:
+   pc     - the preconditioner context
-   nu     - the shift parameter

    Options Database:
.     -pc_fieldsplit_gkb_nu - default is 1

    Level: intermediate

.seealso: PCFIELDSPLIT, PCFieldSplitSetGKBDelay(), PCFieldSplitSetGKBTol(), PCFieldSplitSetGKBMaxit()
@*/
PetscErrorCode PCFieldSplitSetGKBNu(PC pc,PetscReal nu)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveReal(pc,nu,2);
  ierr = PetscTryMethod(pc,"PCFieldSplitSetGKBNu_C",(PC,PetscReal),(pc,nu));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCFieldSplitSetGKBNu_FieldSplit(PC pc,PetscReal nu)
{
  PC_FieldSplit *jac = (PC_FieldSplit*)pc->data;

  PetscFunctionBegin;
  jac->gkbnu = nu;
  PetscFunctionReturn(0);
}


static PetscErrorCode  PCFieldSplitSetType_FieldSplit(PC pc,PCCompositeType type)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  jac->type = type;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitGetSubKSP_C",0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetSchurPre_C",0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitGetSchurPre_C",0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetSchurFactType_C",0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetSchurScale_C",0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetGKBTol_C",0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetGKBMaxit_C",0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetGKBNu_C",0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetGKBDelay_C",0);CHKERRQ(ierr);

  if (type == PC_COMPOSITE_SCHUR) {
    pc->ops->apply = PCApply_FieldSplit_Schur;
    pc->ops->view  = PCView_FieldSplit_Schur;

    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitGetSubKSP_C",PCFieldSplitGetSubKSP_FieldSplit_Schur);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetSchurPre_C",PCFieldSplitSetSchurPre_FieldSplit);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitGetSchurPre_C",PCFieldSplitGetSchurPre_FieldSplit);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetSchurFactType_C",PCFieldSplitSetSchurFactType_FieldSplit);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetSchurScale_C",PCFieldSplitSetSchurScale_FieldSplit);CHKERRQ(ierr);
  } else if (type == PC_COMPOSITE_GKB){
    pc->ops->apply = PCApply_FieldSplit_GKB;
    pc->ops->view  = PCView_FieldSplit_GKB;

    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitGetSubKSP_C",PCFieldSplitGetSubKSP_FieldSplit);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetGKBTol_C",PCFieldSplitSetGKBTol_FieldSplit);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetGKBMaxit_C",PCFieldSplitSetGKBMaxit_FieldSplit);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetGKBNu_C",PCFieldSplitSetGKBNu_FieldSplit);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetGKBDelay_C",PCFieldSplitSetGKBDelay_FieldSplit);CHKERRQ(ierr);
  } else {
    pc->ops->apply = PCApply_FieldSplit;
    pc->ops->view  = PCView_FieldSplit;

    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitGetSubKSP_C",PCFieldSplitGetSubKSP_FieldSplit);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCFieldSplitSetBlockSize_FieldSplit(PC pc,PetscInt bs)
{
  PC_FieldSplit *jac = (PC_FieldSplit*)pc->data;

  PetscFunctionBegin;
  if (bs < 1) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Blocksize must be positive, you gave %D",bs);
  if (jac->bs > 0 && jac->bs != bs) SETERRQ2(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Cannot change fieldsplit blocksize from %D to %D after it has been set",jac->bs,bs);
  jac->bs = bs;
  PetscFunctionReturn(0);
}

/*@
   PCFieldSplitSetType - Sets the type of fieldsplit preconditioner.

   Collective on PC

   Input Parameter:
+  pc - the preconditioner context
-  type - PC_COMPOSITE_ADDITIVE, PC_COMPOSITE_MULTIPLICATIVE (default), PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE, PC_COMPOSITE_SPECIAL, PC_COMPOSITE_SCHUR

   Options Database Key:
.  -pc_fieldsplit_type <type: one of multiplicative, additive, symmetric_multiplicative, special, schur> - Sets fieldsplit preconditioner type

   Level: Intermediate

.seealso: PCCompositeSetType()

@*/
PetscErrorCode  PCFieldSplitSetType(PC pc,PCCompositeType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCFieldSplitSetType_C",(PC,PCCompositeType),(pc,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PCFieldSplitGetType - Gets the type of fieldsplit preconditioner.

  Not collective

  Input Parameter:
. pc - the preconditioner context

  Output Parameter:
. type - PC_COMPOSITE_ADDITIVE, PC_COMPOSITE_MULTIPLICATIVE (default), PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE, PC_COMPOSITE_SPECIAL, PC_COMPOSITE_SCHUR

  Level: Intermediate

.seealso: PCCompositeSetType()
@*/
PetscErrorCode PCFieldSplitGetType(PC pc, PCCompositeType *type)
{
  PC_FieldSplit *jac = (PC_FieldSplit*) pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidIntPointer(type,2);
  *type = jac->type;
  PetscFunctionReturn(0);
}

/*@
   PCFieldSplitSetDMSplits - Flags whether DMCreateFieldDecomposition() should be used to define the splits, whenever possible.

   Logically Collective

   Input Parameters:
+  pc   - the preconditioner context
-  flg  - boolean indicating whether to use field splits defined by the DM

   Options Database Key:
.  -pc_fieldsplit_dm_splits

   Level: Intermediate

.seealso: PCFieldSplitGetDMSplits()

@*/
PetscErrorCode  PCFieldSplitSetDMSplits(PC pc,PetscBool flg)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;
  PetscBool      isfs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveBool(pc,flg,2);
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCFIELDSPLIT,&isfs);CHKERRQ(ierr);
  if (isfs) {
    jac->dm_splits = flg;
  }
  PetscFunctionReturn(0);
}


/*@
   PCFieldSplitGetDMSplits - Returns flag indicating whether DMCreateFieldDecomposition() should be used to define the splits, whenever possible.

   Logically Collective

   Input Parameter:
.  pc   - the preconditioner context

   Output Parameter:
.  flg  - boolean indicating whether to use field splits defined by the DM

   Level: Intermediate

.seealso: PCFieldSplitSetDMSplits()

@*/
PetscErrorCode  PCFieldSplitGetDMSplits(PC pc,PetscBool* flg)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;
  PetscBool      isfs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCFIELDSPLIT,&isfs);CHKERRQ(ierr);
  if (isfs) {
    if(flg) *flg = jac->dm_splits;
  }
  PetscFunctionReturn(0);
}

/*@
   PCFieldSplitGetDetectSaddlePoint - Returns flag indicating whether PCFieldSplit will attempt to automatically determine fields based on zero diagonal entries.

   Logically Collective

   Input Parameter:
.  pc   - the preconditioner context

   Output Parameter:
.  flg  - boolean indicating whether to detect fields or not

   Level: Intermediate

.seealso: PCFIELDSPLIT, PCFieldSplitSetDetectSaddlePoint()

@*/
PetscErrorCode PCFieldSplitGetDetectSaddlePoint(PC pc,PetscBool *flg)
{
  PC_FieldSplit *jac = (PC_FieldSplit*)pc->data;

  PetscFunctionBegin;
  *flg = jac->detect;
  PetscFunctionReturn(0);
}

/*@
   PCFieldSplitSetDetectSaddlePoint - Sets flag indicating whether PCFieldSplit will attempt to automatically determine fields based on zero diagonal entries.

   Logically Collective

   Notes:
   Also sets the split type to PC_COMPOSITE_SCHUR (see PCFieldSplitSetType()) and the Schur preconditioner type to PC_FIELDSPLIT_SCHUR_PRE_SELF (see PCFieldSplitSetSchurPre()).

   Input Parameter:
.  pc   - the preconditioner context

   Output Parameter:
.  flg  - boolean indicating whether to detect fields or not

   Options Database Key:
.  -pc_fieldsplit_detect_saddle_point

   Level: Intermediate

.seealso: PCFIELDSPLIT, PCFieldSplitSetDetectSaddlePoint(), PCFieldSplitSetType(), PCFieldSplitSetSchurPre()

@*/
PetscErrorCode PCFieldSplitSetDetectSaddlePoint(PC pc,PetscBool flg)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  jac->detect = flg;
  if (jac->detect) {
    ierr = PCFieldSplitSetType(pc,PC_COMPOSITE_SCHUR);CHKERRQ(ierr);
    ierr = PCFieldSplitSetSchurPre(pc,PC_FIELDSPLIT_SCHUR_PRE_SELF,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
/*MC
   PCFIELDSPLIT - Preconditioner created by combining separate preconditioners for individual
                  fields or groups of fields. See the users manual section "Solving Block Matrices" for more details.

     To set options on the solvers for each block append -fieldsplit_ to all the PC
        options database keys. For example, -fieldsplit_pc_type ilu -fieldsplit_pc_factor_levels 1

     To set the options on the solvers separate for each block call PCFieldSplitGetSubKSP()
         and set the options directly on the resulting KSP object

   Level: intermediate

   Options Database Keys:
+   -pc_fieldsplit_%d_fields <a,b,..> - indicates the fields to be used in the %d'th split
.   -pc_fieldsplit_default - automatically add any fields to additional splits that have not
                              been supplied explicitly by -pc_fieldsplit_%d_fields
.   -pc_fieldsplit_block_size <bs> - size of block that defines fields (i.e. there are bs fields)
.   -pc_fieldsplit_type <additive,multiplicative,symmetric_multiplicative,schur,gkb> - type of relaxation or factorization splitting
.   -pc_fieldsplit_schur_precondition <self,selfp,user,a11,full> - default is a11; see PCFieldSplitSetSchurPre()
.   -pc_fieldsplit_schur_fact_type <diag,lower,upper,full> - set factorization type when using -pc_fieldsplit_type schur; see PCFieldSplitSetSchurFactType()
-   -pc_fieldsplit_detect_saddle_point - automatically finds rows with zero diagonal and uses Schur complement with no preconditioner as the solver

     Options prefixes for inner solvers when using the Schur complement preconditioner are -fieldsplit_0_ and -fieldsplit_1_ .
     For all other solvers they are -fieldsplit_%d_ for the dth field; use -fieldsplit_ for all fields.
     The options prefix for the inner solver when using the Golub-Kahan biadiagonalization preconditioner is -fieldsplit_0_

   Notes:
    Use PCFieldSplitSetFields() to set fields defined by "strided" entries and PCFieldSplitSetIS()
     to define a field by an arbitrary collection of entries.

      If no fields are set the default is used. The fields are defined by entries strided by bs,
      beginning at 0 then 1, etc to bs-1. The block size can be set with PCFieldSplitSetBlockSize(),
      if this is not called the block size defaults to the blocksize of the second matrix passed
      to KSPSetOperators()/PCSetOperators().

$     For the Schur complement preconditioner if J = ( A00 A01 )
$                                                    ( A10 A11 )
$     the preconditioner using full factorization is
$              ( I   -ksp(A00) A01 ) ( inv(A00)     0  ) (     I          0  )
$              ( 0         I       ) (   0      ksp(S) ) ( -A10 ksp(A00)  I  )
     where the action of inv(A00) is applied using the KSP solver with prefix -fieldsplit_0_.  S is the Schur complement
$              S = A11 - A10 ksp(A00) A01
     which is usually dense and not stored explicitly.  The action of ksp(S) is computed using the KSP solver with prefix -fieldsplit_splitname_ (where splitname was given
     in providing the SECOND split or 1 if not give). For PCFieldSplitGetSubKSP() when field number is 0,
     it returns the KSP associated with -fieldsplit_0_ while field number 1 gives -fieldsplit_1_ KSP. By default
     A11 is used to construct a preconditioner for S, use PCFieldSplitSetSchurPre() for all the possible ways to construct the preconditioner for S.

     The factorization type is set using -pc_fieldsplit_schur_fact_type <diag, lower, upper, full>. The full is shown above,
     diag gives
$              ( inv(A00)     0   )
$              (   0      -ksp(S) )
     note that slightly counter intuitively there is a negative in front of the ksp(S) so that the preconditioner is positive definite. For SPD matrices J, the sign flip
     can be turned off with PCFieldSplitSetSchurScale() or by command line -pc_fieldsplit_schur_scale 1.0. The lower factorization is the inverse of
$              (  A00   0 )
$              (  A10   S )
     where the inverses of A00 and S are applied using KSPs. The upper factorization is the inverse of
$              ( A00 A01 )
$              (  0   S  )
     where again the inverses of A00 and S are applied using KSPs.

     If only one set of indices (one IS) is provided with PCFieldSplitSetIS() then the complement of that IS
     is used automatically for a second block.

     The fieldsplit preconditioner cannot currently be used with the BAIJ or SBAIJ data formats if the blocksize is larger than 1.
     Generally it should be used with the AIJ format.

     The forms of these preconditioners are closely related if not identical to forms derived as "Distributive Iterations", see,
     for example, page 294 in "Principles of Computational Fluid Dynamics" by Pieter Wesseling. Note that one can also use PCFIELDSPLIT
     inside a smoother resulting in "Distributive Smoothers".

   There is a nice discussion of block preconditioners in

[El08] A taxonomy and comparison of parallel block multi-level preconditioners for the incompressible Navier-Stokes equations
       Howard Elman, V.E. Howle, John Shadid, Robert Shuttleworth, Ray Tuminaro, Journal of Computational Physics 227 (2008) 1790--1808
       http://chess.cs.umd.edu/~elman/papers/tax.pdf

   The Constrained Pressure Preconditioner (CPR) can be implemented using PCCOMPOSITE with PCGALERKIN. CPR first solves an R A P subsystem, updates the
   residual on all variables (PCCompositeSetType(pc,PC_COMPOSITE_MULTIPLICATIVE)), and then applies a simple ILU like preconditioner on all the variables.

   The generalized Golub-Kahan bidiagonalization preconditioner (gkb) can be applied to symmetric 2x2 block matrices of the shape
$        ( A00  A01 )
$        ( A01' 0   )
   with A00 positive semi-definite. The implementation follows [Ar13]. Therein, we choose N := 1/nu * I and the (1,1)-block of the matrix is modified to H = A00 + nu*A01*A01'.
   A linear system Hx = b has to be solved in each iteration of the GKB algorithm. This solver is chosen with the option prefix -fieldsplit_0_.

[Ar13] Generalized Golub-Kahan bidiagonalization and stopping criteria, SIAM J. Matrix Anal. Appl., Vol. 34, No. 2, pp. 571-592, 2013.

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCLSC,
           PCFieldSplitGetSubKSP(), PCFieldSplitSchurGetSubKSP(), PCFieldSplitSetFields(),
           PCFieldSplitSetType(), PCFieldSplitSetIS(), PCFieldSplitSetSchurPre(), PCFieldSplitSetSchurFactType(),
           MatSchurComplementSetAinvType(), PCFieldSplitSetSchurScale(), PCFieldSplitSetDetectSaddlePoint()
M*/

PETSC_EXTERN PetscErrorCode PCCreate_FieldSplit(PC pc)
{
  PetscErrorCode ierr;
  PC_FieldSplit  *jac;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,&jac);CHKERRQ(ierr);

  jac->bs                 = -1;
  jac->nsplits            = 0;
  jac->type               = PC_COMPOSITE_MULTIPLICATIVE;
  jac->schurpre           = PC_FIELDSPLIT_SCHUR_PRE_USER; /* Try user preconditioner first, fall back on diagonal */
  jac->schurfactorization = PC_FIELDSPLIT_SCHUR_FACT_FULL;
  jac->schurscale         = -1.0;
  jac->dm_splits          = PETSC_TRUE;
  jac->detect             = PETSC_FALSE;
  jac->gkbtol             = 1e-5;
  jac->gkbdelay           = 5;
  jac->gkbnu              = 1;
  jac->gkbmaxit           = 100;
  jac->gkbmonitor         = PETSC_FALSE;

  pc->data = (void*)jac;

  pc->ops->apply           = PCApply_FieldSplit;
  pc->ops->applytranspose  = PCApplyTranspose_FieldSplit;
  pc->ops->setup           = PCSetUp_FieldSplit;
  pc->ops->reset           = PCReset_FieldSplit;
  pc->ops->destroy         = PCDestroy_FieldSplit;
  pc->ops->setfromoptions  = PCSetFromOptions_FieldSplit;
  pc->ops->view            = PCView_FieldSplit;
  pc->ops->applyrichardson = 0;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSchurGetSubKSP_C",PCFieldSplitSchurGetSubKSP_FieldSplit);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitGetSubKSP_C",PCFieldSplitGetSubKSP_FieldSplit);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetFields_C",PCFieldSplitSetFields_FieldSplit);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetIS_C",PCFieldSplitSetIS_FieldSplit);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetType_C",PCFieldSplitSetType_FieldSplit);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetBlockSize_C",PCFieldSplitSetBlockSize_FieldSplit);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitRestrictIS_C",PCFieldSplitRestrictIS_FieldSplit);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
