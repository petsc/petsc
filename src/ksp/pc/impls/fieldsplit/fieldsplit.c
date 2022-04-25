#include <petsc/private/pcimpl.h>     /*I "petscpc.h" I*/
#include <petsc/private/kspimpl.h>    /*  This is needed to provide the appropriate PETSC_EXTERN for KSP_Solve_FS ....*/
#include <petscdm.h>

const char *const PCFieldSplitSchurPreTypes[] = {"SELF","SELFP","A11","USER","FULL","PCFieldSplitSchurPreType","PC_FIELDSPLIT_SCHUR_PRE_",NULL};
const char *const PCFieldSplitSchurFactTypes[] = {"DIAG","LOWER","UPPER","FULL","PCFieldSplitSchurFactType","PC_FIELDSPLIT_SCHUR_FACT_",NULL};

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

  /* Used only when setting coordinates with PCSetCoordinates */
  PetscInt dim;
  PetscInt ndofs;
  PetscReal *coords;
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
  PetscBool                 coordinates_set;        /* Whether PCSetCoordinates has been called */
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
  PetscBool         iascii,isdraw;
  PetscInt          i,j;
  PC_FieldSplitLink ilink = jac->head;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw));
  if (iascii) {
    if (jac->bs > 0) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  FieldSplit with %s composition: total splits = %" PetscInt_FMT ", blocksize = %" PetscInt_FMT "\n",PCCompositeTypes[jac->type],jac->nsplits,jac->bs));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  FieldSplit with %s composition: total splits = %" PetscInt_FMT "\n",PCCompositeTypes[jac->type],jac->nsplits));
    }
    if (pc->useAmat) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  using Amat (not Pmat) as operator for blocks\n"));
    }
    if (jac->diag_use_amat) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  using Amat (not Pmat) as operator for diagonal blocks\n"));
    }
    if (jac->offdiag_use_amat) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  using Amat (not Pmat) as operator for off-diagonal blocks\n"));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Solver info for each split is in the following KSP objects:\n"));
    for (i=0; i<jac->nsplits; i++) {
      if (ilink->fields) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"Split number %" PetscInt_FMT " Fields ",i));
        PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
        for (j=0; j<ilink->nfields; j++) {
          if (j > 0) {
            PetscCall(PetscViewerASCIIPrintf(viewer,","));
          }
          PetscCall(PetscViewerASCIIPrintf(viewer," %" PetscInt_FMT,ilink->fields[j]));
        }
        PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
        PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer,"Split number %" PetscInt_FMT " Defined by IS\n",i));
      }
      PetscCall(KSPView(ilink->ksp,viewer));
      ilink = ilink->next;
    }
  }

 if (isdraw) {
    PetscDraw draw;
    PetscReal x,y,w,wd;

    PetscCall(PetscViewerDrawGetDraw(viewer,0,&draw));
    PetscCall(PetscDrawGetCurrentPoint(draw,&x,&y));
    w    = 2*PetscMin(1.0 - x,x);
    wd   = w/(jac->nsplits + 1);
    x    = x - wd*(jac->nsplits-1)/2.0;
    for (i=0; i<jac->nsplits; i++) {
      PetscCall(PetscDrawPushCurrentPoint(draw,x,y));
      PetscCall(KSPView(ilink->ksp,viewer));
      PetscCall(PetscDrawPopCurrentPoint(draw));
      x    += wd;
      ilink = ilink->next;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_FieldSplit_Schur(PC pc,PetscViewer viewer)
{
  PC_FieldSplit              *jac = (PC_FieldSplit*)pc->data;
  PetscBool                  iascii,isdraw;
  PetscInt                   i,j;
  PC_FieldSplitLink          ilink = jac->head;
  MatSchurComplementAinvType atype;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw));
  if (iascii) {
    if (jac->bs > 0) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  FieldSplit with Schur preconditioner, blocksize = %" PetscInt_FMT ", factorization %s\n",jac->bs,PCFieldSplitSchurFactTypes[jac->schurfactorization]));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  FieldSplit with Schur preconditioner, factorization %s\n",PCFieldSplitSchurFactTypes[jac->schurfactorization]));
    }
    if (pc->useAmat) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  using Amat (not Pmat) as operator for blocks\n"));
    }
    switch (jac->schurpre) {
    case PC_FIELDSPLIT_SCHUR_PRE_SELF:
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Preconditioner for the Schur complement formed from S itself\n"));
      break;
    case PC_FIELDSPLIT_SCHUR_PRE_SELFP:
      PetscCall(MatSchurComplementGetAinvType(jac->schur,&atype));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Preconditioner for the Schur complement formed from Sp, an assembled approximation to S, which uses A00's %sdiagonal's inverse\n",atype == MAT_SCHUR_COMPLEMENT_AINV_DIAG ? "" : (atype == MAT_SCHUR_COMPLEMENT_AINV_BLOCK_DIAG ? "block " : "lumped ")));break;
    case PC_FIELDSPLIT_SCHUR_PRE_A11:
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Preconditioner for the Schur complement formed from A11\n"));
      break;
    case PC_FIELDSPLIT_SCHUR_PRE_FULL:
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Preconditioner for the Schur complement formed from the exact Schur complement\n"));
      break;
    case PC_FIELDSPLIT_SCHUR_PRE_USER:
      if (jac->schur_user) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"  Preconditioner for the Schur complement formed from user provided matrix\n"));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer,"  Preconditioner for the Schur complement formed from A11\n"));
      }
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_OUTOFRANGE, "Invalid Schur preconditioning type: %d", jac->schurpre);
    }
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Split info:\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    for (i=0; i<jac->nsplits; i++) {
      if (ilink->fields) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"Split number %" PetscInt_FMT " Fields ",i));
        PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
        for (j=0; j<ilink->nfields; j++) {
          if (j > 0) {
            PetscCall(PetscViewerASCIIPrintf(viewer,","));
          }
          PetscCall(PetscViewerASCIIPrintf(viewer," %" PetscInt_FMT,ilink->fields[j]));
        }
        PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
        PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer,"Split number %" PetscInt_FMT " Defined by IS\n",i));
      }
      ilink = ilink->next;
    }
    PetscCall(PetscViewerASCIIPrintf(viewer,"KSP solver for A00 block\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    if (jac->head) {
      PetscCall(KSPView(jac->head->ksp,viewer));
    } else  PetscCall(PetscViewerASCIIPrintf(viewer,"  not yet available\n"));
    PetscCall(PetscViewerASCIIPopTab(viewer));
    if (jac->head && jac->kspupper != jac->head->ksp) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"KSP solver for upper A00 in upper triangular factor \n"));
      PetscCall(PetscViewerASCIIPushTab(viewer));
      if (jac->kspupper) PetscCall(KSPView(jac->kspupper,viewer));
      else PetscCall(PetscViewerASCIIPrintf(viewer,"  not yet available\n"));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer,"KSP solver for S = A11 - A10 inv(A00) A01 \n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    if (jac->kspschur) {
      PetscCall(KSPView(jac->kspschur,viewer));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  not yet available\n"));
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  } else if (isdraw && jac->head) {
    PetscDraw draw;
    PetscReal x,y,w,wd,h;
    PetscInt  cnt = 2;
    char      str[32];

    PetscCall(PetscViewerDrawGetDraw(viewer,0,&draw));
    PetscCall(PetscDrawGetCurrentPoint(draw,&x,&y));
    if (jac->kspupper != jac->head->ksp) cnt++;
    w  = 2*PetscMin(1.0 - x,x);
    wd = w/(cnt + 1);

    PetscCall(PetscSNPrintf(str,32,"Schur fact. %s",PCFieldSplitSchurFactTypes[jac->schurfactorization]));
    PetscCall(PetscDrawStringBoxed(draw,x,y,PETSC_DRAW_RED,PETSC_DRAW_BLACK,str,NULL,&h));
    y   -= h;
    if (jac->schurpre == PC_FIELDSPLIT_SCHUR_PRE_USER &&  !jac->schur_user) {
      PetscCall(PetscSNPrintf(str,32,"Prec. for Schur from %s",PCFieldSplitSchurPreTypes[PC_FIELDSPLIT_SCHUR_PRE_A11]));
    } else {
      PetscCall(PetscSNPrintf(str,32,"Prec. for Schur from %s",PCFieldSplitSchurPreTypes[jac->schurpre]));
    }
    PetscCall(PetscDrawStringBoxed(draw,x+wd*(cnt-1)/2.0,y,PETSC_DRAW_RED,PETSC_DRAW_BLACK,str,NULL,&h));
    y   -= h;
    x    = x - wd*(cnt-1)/2.0;

    PetscCall(PetscDrawPushCurrentPoint(draw,x,y));
    PetscCall(KSPView(jac->head->ksp,viewer));
    PetscCall(PetscDrawPopCurrentPoint(draw));
    if (jac->kspupper != jac->head->ksp) {
      x   += wd;
      PetscCall(PetscDrawPushCurrentPoint(draw,x,y));
      PetscCall(KSPView(jac->kspupper,viewer));
      PetscCall(PetscDrawPopCurrentPoint(draw));
    }
    x   += wd;
    PetscCall(PetscDrawPushCurrentPoint(draw,x,y));
    PetscCall(KSPView(jac->kspschur,viewer));
    PetscCall(PetscDrawPopCurrentPoint(draw));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_FieldSplit_GKB(PC pc,PetscViewer viewer)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscBool         iascii,isdraw;
  PetscInt          i,j;
  PC_FieldSplitLink ilink = jac->head;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw));
  if (iascii) {
    if (jac->bs > 0) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  FieldSplit with %s composition: total splits = %" PetscInt_FMT ", blocksize = %" PetscInt_FMT "\n",PCCompositeTypes[jac->type],jac->nsplits,jac->bs));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  FieldSplit with %s composition: total splits = %" PetscInt_FMT "\n",PCCompositeTypes[jac->type],jac->nsplits));
    }
    if (pc->useAmat) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  using Amat (not Pmat) as operator for blocks\n"));
    }
    if (jac->diag_use_amat) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  using Amat (not Pmat) as operator for diagonal blocks\n"));
    }
    if (jac->offdiag_use_amat) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  using Amat (not Pmat) as operator for off-diagonal blocks\n"));
    }

    PetscCall(PetscViewerASCIIPrintf(viewer,"  Stopping tolerance=%.1e, delay in error estimate=%" PetscInt_FMT ", maximum iterations=%" PetscInt_FMT "\n",(double)jac->gkbtol,jac->gkbdelay,jac->gkbmaxit));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Solver info for H = A00 + nu*A01*A01' matrix:\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));

    if (ilink->fields) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"Split number 0 Fields "));
      PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
      for (j=0; j<ilink->nfields; j++) {
        if (j > 0) {
          PetscCall(PetscViewerASCIIPrintf(viewer,","));
        }
        PetscCall(PetscViewerASCIIPrintf(viewer," %" PetscInt_FMT,ilink->fields[j]));
      }
      PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
      PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
    } else {
        PetscCall(PetscViewerASCIIPrintf(viewer,"Split number 0 Defined by IS\n"));
    }
    PetscCall(KSPView(ilink->ksp,viewer));

    PetscCall(PetscViewerASCIIPopTab(viewer));
  }

 if (isdraw) {
    PetscDraw draw;
    PetscReal x,y,w,wd;

    PetscCall(PetscViewerDrawGetDraw(viewer,0,&draw));
    PetscCall(PetscDrawGetCurrentPoint(draw,&x,&y));
    w    = 2*PetscMin(1.0 - x,x);
    wd   = w/(jac->nsplits + 1);
    x    = x - wd*(jac->nsplits-1)/2.0;
    for (i=0; i<jac->nsplits; i++) {
      PetscCall(PetscDrawPushCurrentPoint(draw,x,y));
      PetscCall(KSPView(ilink->ksp,viewer));
      PetscCall(PetscDrawPopCurrentPoint(draw));
      x    += wd;
      ilink = ilink->next;
    }
  }
  PetscFunctionReturn(0);
}

/* Precondition: jac->bs is set to a meaningful value */
static PetscErrorCode PCFieldSplitSetRuntimeSplits_Private(PC pc)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;
  PetscInt       i,nfields,*ifields,nfields_col,*ifields_col;
  PetscBool      flg,flg_col;
  char           optionname[128],splitname[8],optionname_col[128];

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(jac->bs,&ifields));
  PetscCall(PetscMalloc1(jac->bs,&ifields_col));
  for (i=0,flg=PETSC_TRUE;; i++) {
    PetscCall(PetscSNPrintf(splitname,sizeof(splitname),"%" PetscInt_FMT,i));
    PetscCall(PetscSNPrintf(optionname,sizeof(optionname),"-pc_fieldsplit_%" PetscInt_FMT "_fields",i));
    PetscCall(PetscSNPrintf(optionname_col,sizeof(optionname_col),"-pc_fieldsplit_%" PetscInt_FMT "_fields_col",i));
    nfields     = jac->bs;
    nfields_col = jac->bs;
    PetscCall(PetscOptionsGetIntArray(((PetscObject)pc)->options,((PetscObject)pc)->prefix,optionname,ifields,&nfields,&flg));
    PetscCall(PetscOptionsGetIntArray(((PetscObject)pc)->options,((PetscObject)pc)->prefix,optionname_col,ifields_col,&nfields_col,&flg_col));
    if (!flg) break;
    else if (flg && !flg_col) {
      PetscCheck(nfields,PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot list zero fields");
      PetscCall(PCFieldSplitSetFields(pc,splitname,nfields,ifields,ifields));
    } else {
      PetscCheck(nfields && nfields_col,PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot list zero fields");
      PetscCheck(nfields == nfields_col,PETSC_COMM_SELF,PETSC_ERR_USER,"Number of row and column fields must match");
      PetscCall(PCFieldSplitSetFields(pc,splitname,nfields,ifields,ifields_col));
    }
  }
  if (i > 0) {
    /* Makes command-line setting of splits take precedence over setting them in code.
       Otherwise subsequent calls to PCFieldSplitSetIS() or PCFieldSplitSetFields() would
       create new splits, which would probably not be what the user wanted. */
    jac->splitdefined = PETSC_TRUE;
  }
  PetscCall(PetscFree(ifields));
  PetscCall(PetscFree(ifields_col));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCFieldSplitSetDefaults(PC pc)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PC_FieldSplitLink ilink = jac->head;
  PetscBool         fieldsplit_default = PETSC_FALSE,coupling = PETSC_FALSE;
  PetscInt          i;

  PetscFunctionBegin;
  /*
   Kinda messy, but at least this now uses DMCreateFieldDecomposition().
   Should probably be rewritten.
   */
  if (!ilink) {
    PetscCall(PetscOptionsGetBool(((PetscObject)pc)->options,((PetscObject)pc)->prefix,"-pc_fieldsplit_detect_coupling",&coupling,NULL));
    if (pc->dm && jac->dm_splits && !jac->detect && !coupling) {
      PetscInt  numFields, f, i, j;
      char      **fieldNames;
      IS        *fields;
      DM        *dms;
      DM        subdm[128];
      PetscBool flg;

      PetscCall(DMCreateFieldDecomposition(pc->dm, &numFields, &fieldNames, &fields, &dms));
      /* Allow the user to prescribe the splits */
      for (i = 0, flg = PETSC_TRUE;; i++) {
        PetscInt ifields[128];
        IS       compField;
        char     optionname[128], splitname[8];
        PetscInt nfields = numFields;

        PetscCall(PetscSNPrintf(optionname, sizeof(optionname), "-pc_fieldsplit_%" PetscInt_FMT "_fields", i));
        PetscCall(PetscOptionsGetIntArray(((PetscObject)pc)->options,((PetscObject)pc)->prefix, optionname, ifields, &nfields, &flg));
        if (!flg) break;
        PetscCheck(numFields <= 128,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Cannot currently support %" PetscInt_FMT " > 128 fields", numFields);
        PetscCall(DMCreateSubDM(pc->dm, nfields, ifields, &compField, &subdm[i]));
        if (nfields == 1) {
          PetscCall(PCFieldSplitSetIS(pc, fieldNames[ifields[0]], compField));
        } else {
          PetscCall(PetscSNPrintf(splitname, sizeof(splitname), "%" PetscInt_FMT, i));
          PetscCall(PCFieldSplitSetIS(pc, splitname, compField));
        }
        PetscCall(ISDestroy(&compField));
        for (j = 0; j < nfields; ++j) {
          f    = ifields[j];
          PetscCall(PetscFree(fieldNames[f]));
          PetscCall(ISDestroy(&fields[f]));
        }
      }
      if (i == 0) {
        for (f = 0; f < numFields; ++f) {
          PetscCall(PCFieldSplitSetIS(pc, fieldNames[f], fields[f]));
          PetscCall(PetscFree(fieldNames[f]));
          PetscCall(ISDestroy(&fields[f]));
        }
      } else {
        for (j=0; j<numFields; j++) {
          PetscCall(DMDestroy(dms+j));
        }
        PetscCall(PetscFree(dms));
        PetscCall(PetscMalloc1(i, &dms));
        for (j = 0; j < i; ++j) dms[j] = subdm[j];
      }
      PetscCall(PetscFree(fieldNames));
      PetscCall(PetscFree(fields));
      if (dms) {
        PetscCall(PetscInfo(pc, "Setting up physics based fieldsplit preconditioner using the embedded DM\n"));
        for (ilink = jac->head, i = 0; ilink; ilink = ilink->next, ++i) {
          const char *prefix;
          PetscCall(PetscObjectGetOptionsPrefix((PetscObject)(ilink->ksp),&prefix));
          PetscCall(PetscObjectSetOptionsPrefix((PetscObject)(dms[i]), prefix));
          PetscCall(KSPSetDM(ilink->ksp, dms[i]));
          PetscCall(KSPSetDMActive(ilink->ksp, PETSC_FALSE));
          {
            PetscErrorCode (*func)(KSP,Mat,Mat,void*);
            void            *ctx;

            PetscCall(DMKSPGetComputeOperators(pc->dm, &func, &ctx));
            PetscCall(DMKSPSetComputeOperators(dms[i],  func,  ctx));
          }
          PetscCall(PetscObjectIncrementTabLevel((PetscObject)dms[i],(PetscObject)ilink->ksp,0));
          PetscCall(DMDestroy(&dms[i]));
        }
        PetscCall(PetscFree(dms));
      }
    } else {
      if (jac->bs <= 0) {
        if (pc->pmat) {
          PetscCall(MatGetBlockSize(pc->pmat,&jac->bs));
        } else jac->bs = 1;
      }

      if (jac->detect) {
        IS       zerodiags,rest;
        PetscInt nmin,nmax;

        PetscCall(MatGetOwnershipRange(pc->mat,&nmin,&nmax));
        if (jac->diag_use_amat) {
          PetscCall(MatFindZeroDiagonals(pc->mat,&zerodiags));
        } else {
          PetscCall(MatFindZeroDiagonals(pc->pmat,&zerodiags));
        }
        PetscCall(ISComplement(zerodiags,nmin,nmax,&rest));
        PetscCall(PCFieldSplitSetIS(pc,"0",rest));
        PetscCall(PCFieldSplitSetIS(pc,"1",zerodiags));
        PetscCall(ISDestroy(&zerodiags));
        PetscCall(ISDestroy(&rest));
      } else if (coupling) {
        IS       coupling,rest;
        PetscInt nmin,nmax;

        PetscCall(MatGetOwnershipRange(pc->mat,&nmin,&nmax));
        if (jac->offdiag_use_amat) {
          PetscCall(MatFindOffBlockDiagonalEntries(pc->mat,&coupling));
        } else {
          PetscCall(MatFindOffBlockDiagonalEntries(pc->pmat,&coupling));
        }
        PetscCall(ISCreateStride(PetscObjectComm((PetscObject)pc->mat),nmax-nmin,nmin,1,&rest));
        PetscCall(ISSetIdentity(rest));
        PetscCall(PCFieldSplitSetIS(pc,"0",rest));
        PetscCall(PCFieldSplitSetIS(pc,"1",coupling));
        PetscCall(ISDestroy(&coupling));
        PetscCall(ISDestroy(&rest));
      } else {
        PetscCall(PetscOptionsGetBool(((PetscObject)pc)->options,((PetscObject)pc)->prefix,"-pc_fieldsplit_default",&fieldsplit_default,NULL));
        if (!fieldsplit_default) {
          /* Allow user to set fields from command line,  if bs was known at the time of PCSetFromOptions_FieldSplit()
           then it is set there. This is not ideal because we should only have options set in XXSetFromOptions(). */
          PetscCall(PCFieldSplitSetRuntimeSplits_Private(pc));
          if (jac->splitdefined) PetscCall(PetscInfo(pc,"Splits defined using the options database\n"));
        }
        if ((fieldsplit_default || !jac->splitdefined) && !jac->isrestrict) {
          Mat       M = pc->pmat;
          PetscBool isnest;

          PetscCall(PetscInfo(pc,"Using default splitting of fields\n"));
          PetscCall(PetscObjectTypeCompare((PetscObject)pc->pmat,MATNEST,&isnest));
          if (!isnest) {
            M    = pc->mat;
            PetscCall(PetscObjectTypeCompare((PetscObject)pc->mat,MATNEST,&isnest));
          }
          if (isnest) {
            IS       *fields;
            PetscInt nf;

            PetscCall(MatNestGetSize(M,&nf,NULL));
            PetscCall(PetscMalloc1(nf,&fields));
            PetscCall(MatNestGetISs(M,fields,NULL));
            for (i=0;i<nf;i++) {
              PetscCall(PCFieldSplitSetIS(pc,NULL,fields[i]));
            }
            PetscCall(PetscFree(fields));
          } else {
            for (i=0; i<jac->bs; i++) {
              char splitname[8];
              PetscCall(PetscSNPrintf(splitname,sizeof(splitname),"%" PetscInt_FMT,i));
              PetscCall(PCFieldSplitSetFields(pc,splitname,1,&i,&i));
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

      PetscCall(MatGetOwnershipRange(pc->mat,&nmin,&nmax));
      PetscCall(ISComplement(ilink->is,nmin,nmax,&is2));
      PetscCall(PCFieldSplitSetIS(pc,"1",is2));
      PetscCall(ISDestroy(&is2));
    } else SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Must provide at least two sets of fields to PCFieldSplit()");
  }

  PetscCheck(jac->nsplits >= 2,PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"Unhandled case, must have at least two fields, not %" PetscInt_FMT, jac->nsplits);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGolubKahanComputeExplicitOperator(Mat A,Mat B,Mat C,Mat *H,PetscReal gkbnu)
{
  Mat               BT,T;
  PetscReal         nrmT,nrmB;

  PetscFunctionBegin;
  PetscCall(MatHermitianTranspose(C,MAT_INITIAL_MATRIX,&T));            /* Test if augmented matrix is symmetric */
  PetscCall(MatAXPY(T,-1.0,B,DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatNorm(T,NORM_1,&nrmT));
  PetscCall(MatNorm(B,NORM_1,&nrmB));
  if (nrmB > 0) {
    if (nrmT/nrmB >= PETSC_SMALL) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Matrix is not symmetric/hermitian, GKB is not applicable.");
    }
  }
  /* Compute augmented Lagrangian matrix H = A00 + nu*A01*A01'. This corresponds to */
  /* setting N := 1/nu*I in [Ar13].                                                 */
  PetscCall(MatHermitianTranspose(B,MAT_INITIAL_MATRIX,&BT));
  PetscCall(MatMatMult(B,BT,MAT_INITIAL_MATRIX,PETSC_DEFAULT,H));       /* H = A01*A01'          */
  PetscCall(MatAYPX(*H,gkbnu,A,DIFFERENT_NONZERO_PATTERN));             /* H = A00 + nu*A01*A01' */

  PetscCall(MatDestroy(&BT));
  PetscCall(MatDestroy(&T));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscOptionsFindPairPrefix_Private(PetscOptions,const char pre[], const char name[],const char *value[],PetscBool *flg);

static PetscErrorCode PCSetUp_FieldSplit(PC pc)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PC_FieldSplitLink ilink;
  PetscInt          i,nsplit;
  PetscBool         sorted, sorted_col;

  PetscFunctionBegin;
  pc->failedreason = PC_NOERROR;
  PetscCall(PCFieldSplitSetDefaults(pc));
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
    PetscCall(MatGetOwnershipRange(pc->pmat,&rstart,&rend));
    nslots = (rend - rstart)/bs;
    for (i=0; i<nsplit; i++) {
      if (jac->defaultsplit) {
        PetscCall(ISCreateStride(PetscObjectComm((PetscObject)pc),nslots,rstart+i,nsplit,&ilink->is));
        PetscCall(ISDuplicate(ilink->is,&ilink->is_col));
      } else if (!ilink->is) {
        if (ilink->nfields > 1) {
          PetscInt *ii,*jj,j,k,nfields = ilink->nfields,*fields = ilink->fields,*fields_col = ilink->fields_col;
          PetscCall(PetscMalloc1(ilink->nfields*nslots,&ii));
          PetscCall(PetscMalloc1(ilink->nfields*nslots,&jj));
          for (j=0; j<nslots; j++) {
            for (k=0; k<nfields; k++) {
              ii[nfields*j + k] = rstart + bs*j + fields[k];
              jj[nfields*j + k] = rstart + bs*j + fields_col[k];
            }
          }
          PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)pc),nslots*nfields,ii,PETSC_OWN_POINTER,&ilink->is));
          PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)pc),nslots*nfields,jj,PETSC_OWN_POINTER,&ilink->is_col));
          PetscCall(ISSetBlockSize(ilink->is, nfields));
          PetscCall(ISSetBlockSize(ilink->is_col, nfields));
        } else {
          PetscCall(ISCreateStride(PetscObjectComm((PetscObject)pc),nslots,rstart+ilink->fields[0],bs,&ilink->is));
          PetscCall(ISCreateStride(PetscObjectComm((PetscObject)pc),nslots,rstart+ilink->fields_col[0],bs,&ilink->is_col));
        }
      }
      PetscCall(ISSorted(ilink->is,&sorted));
      if (ilink->is_col) PetscCall(ISSorted(ilink->is_col,&sorted_col));
      PetscCheck(sorted && sorted_col,PETSC_COMM_SELF,PETSC_ERR_USER,"Fields must be sorted when creating split");
      ilink = ilink->next;
    }
  }

  ilink = jac->head;
  if (!jac->pmat) {
    Vec xtmp;

    PetscCall(MatCreateVecs(pc->pmat,&xtmp,NULL));
    PetscCall(PetscMalloc1(nsplit,&jac->pmat));
    PetscCall(PetscMalloc2(nsplit,&jac->x,nsplit,&jac->y));
    for (i=0; i<nsplit; i++) {
      MatNullSpace sp;

      /* Check for preconditioning matrix attached to IS */
      PetscCall(PetscObjectQuery((PetscObject) ilink->is, "pmat", (PetscObject*) &jac->pmat[i]));
      if (jac->pmat[i]) {
        PetscCall(PetscObjectReference((PetscObject) jac->pmat[i]));
        if (jac->type == PC_COMPOSITE_SCHUR) {
          jac->schur_user = jac->pmat[i];

          PetscCall(PetscObjectReference((PetscObject) jac->schur_user));
        }
      } else {
        const char *prefix;
        PetscCall(MatCreateSubMatrix(pc->pmat,ilink->is,ilink->is_col,MAT_INITIAL_MATRIX,&jac->pmat[i]));
        PetscCall(KSPGetOptionsPrefix(ilink->ksp,&prefix));
        PetscCall(MatSetOptionsPrefix(jac->pmat[i],prefix));
        PetscCall(MatViewFromOptions(jac->pmat[i],NULL,"-mat_view"));
      }
      /* create work vectors for each split */
      PetscCall(MatCreateVecs(jac->pmat[i],&jac->x[i],&jac->y[i]));
      ilink->x = jac->x[i]; ilink->y = jac->y[i]; ilink->z = NULL;
      /* compute scatter contexts needed by multiplicative versions and non-default splits */
      PetscCall(VecScatterCreate(xtmp,ilink->is,jac->x[i],NULL,&ilink->sctx));
      PetscCall(PetscObjectQuery((PetscObject) ilink->is, "nearnullspace", (PetscObject*) &sp));
      if (sp) {
        PetscCall(MatSetNearNullSpace(jac->pmat[i], sp));
      }
      ilink = ilink->next;
    }
    PetscCall(VecDestroy(&xtmp));
  } else {
    MatReuse scall;
    if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
      for (i=0; i<nsplit; i++) {
        PetscCall(MatDestroy(&jac->pmat[i]));
      }
      scall = MAT_INITIAL_MATRIX;
    } else scall = MAT_REUSE_MATRIX;

    for (i=0; i<nsplit; i++) {
      Mat pmat;

      /* Check for preconditioning matrix attached to IS */
      PetscCall(PetscObjectQuery((PetscObject) ilink->is, "pmat", (PetscObject*) &pmat));
      if (!pmat) {
        PetscCall(MatCreateSubMatrix(pc->pmat,ilink->is,ilink->is_col,scall,&jac->pmat[i]));
      }
      ilink = ilink->next;
    }
  }
  if (jac->diag_use_amat) {
    ilink = jac->head;
    if (!jac->mat) {
      PetscCall(PetscMalloc1(nsplit,&jac->mat));
      for (i=0; i<nsplit; i++) {
        PetscCall(MatCreateSubMatrix(pc->mat,ilink->is,ilink->is_col,MAT_INITIAL_MATRIX,&jac->mat[i]));
        ilink = ilink->next;
      }
    } else {
      MatReuse scall;
      if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
        for (i=0; i<nsplit; i++) {
          PetscCall(MatDestroy(&jac->mat[i]));
        }
        scall = MAT_INITIAL_MATRIX;
      } else scall = MAT_REUSE_MATRIX;

      for (i=0; i<nsplit; i++) {
        PetscCall(MatCreateSubMatrix(pc->mat,ilink->is,ilink->is_col,scall,&jac->mat[i]));
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

    PetscCall(PetscObjectQuery((PetscObject) ilink->is, "nullspace", (PetscObject*) &sp));
    if (sp) {
      PetscCall(MatSetNullSpace(jac->mat[i], sp));
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
        PetscCall(PetscCalloc1(nsplit,&jac->Afield));
        if (jac->offdiag_use_amat) {
          PetscCall(MatCreateSubMatrix(pc->mat,ilink->next->is,ilink->is,MAT_INITIAL_MATRIX,&jac->Afield[1]));
        } else {
          PetscCall(MatCreateSubMatrix(pc->pmat,ilink->next->is,ilink->is,MAT_INITIAL_MATRIX,&jac->Afield[1]));
        }
      } else {
        MatReuse scall;

        if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
          PetscCall(MatDestroy(&jac->Afield[1]));
          scall = MAT_INITIAL_MATRIX;
        } else scall = MAT_REUSE_MATRIX;

        if (jac->offdiag_use_amat) {
          PetscCall(MatCreateSubMatrix(pc->mat,ilink->next->is,ilink->is,scall,&jac->Afield[1]));
        } else {
          PetscCall(MatCreateSubMatrix(pc->pmat,ilink->next->is,ilink->is,scall,&jac->Afield[1]));
        }
      }
    } else {
      if (!jac->Afield) {
        PetscCall(PetscMalloc1(nsplit,&jac->Afield));
        for (i=0; i<nsplit; i++) {
          if (jac->offdiag_use_amat) {
            PetscCall(MatCreateSubMatrix(pc->mat,ilink->is,NULL,MAT_INITIAL_MATRIX,&jac->Afield[i]));
          } else {
            PetscCall(MatCreateSubMatrix(pc->pmat,ilink->is,NULL,MAT_INITIAL_MATRIX,&jac->Afield[i]));
          }
          ilink = ilink->next;
        }
      } else {
        MatReuse scall;
        if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
          for (i=0; i<nsplit; i++) {
            PetscCall(MatDestroy(&jac->Afield[i]));
          }
          scall = MAT_INITIAL_MATRIX;
        } else scall = MAT_REUSE_MATRIX;

        for (i=0; i<nsplit; i++) {
          if (jac->offdiag_use_amat) {
            PetscCall(MatCreateSubMatrix(pc->mat,ilink->is,NULL,scall,&jac->Afield[i]));
          } else {
            PetscCall(MatCreateSubMatrix(pc->pmat,ilink->is,NULL,scall,&jac->Afield[i]));
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

    PetscCheck(nsplit == 2,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_INCOMP,"To use Schur complement preconditioner you must have exactly 2 fields");

    /* If pc->mat is SPD, don't scale by -1 the Schur complement */
    if (jac->schurscale == (PetscScalar)-1.0) {
      PetscCall(MatGetOption(pc->pmat,MAT_SPD,&isspd));
      jac->schurscale = (isspd == PETSC_TRUE) ? 1.0 : -1.0;
    }

    /* When extracting off-diagonal submatrices, we take complements from this range */
    PetscCall(MatGetOwnershipRangeColumn(pc->mat,&rstart,&rend));

    if (jac->schur) {
      KSP      kspA = jac->head->ksp, kspInner = NULL, kspUpper = jac->kspupper;
      MatReuse scall;

      if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
        scall = MAT_INITIAL_MATRIX;
        PetscCall(MatDestroy(&jac->B));
        PetscCall(MatDestroy(&jac->C));
      } else scall = MAT_REUSE_MATRIX;

      PetscCall(MatSchurComplementGetKSP(jac->schur, &kspInner));
      ilink = jac->head;
      PetscCall(ISComplement(ilink->is_col,rstart,rend,&ccis));
      if (jac->offdiag_use_amat) {
        PetscCall(MatCreateSubMatrix(pc->mat,ilink->is,ccis,scall,&jac->B));
      } else {
        PetscCall(MatCreateSubMatrix(pc->pmat,ilink->is,ccis,scall,&jac->B));
      }
      PetscCall(ISDestroy(&ccis));
      ilink = ilink->next;
      PetscCall(ISComplement(ilink->is_col,rstart,rend,&ccis));
      if (jac->offdiag_use_amat) {
        PetscCall(MatCreateSubMatrix(pc->mat,ilink->is,ccis,scall,&jac->C));
      } else {
        PetscCall(MatCreateSubMatrix(pc->pmat,ilink->is,ccis,scall,&jac->C));
      }
      PetscCall(ISDestroy(&ccis));
      PetscCall(MatSchurComplementUpdateSubMatrices(jac->schur,jac->mat[0],jac->pmat[0],jac->B,jac->C,jac->mat[1]));
      if (jac->schurpre == PC_FIELDSPLIT_SCHUR_PRE_SELFP) {
        PetscCall(MatDestroy(&jac->schurp));
        PetscCall(MatSchurComplementGetPmat(jac->schur,MAT_INITIAL_MATRIX,&jac->schurp));
      }
      if (kspA != kspInner) {
        PetscCall(KSPSetOperators(kspA,jac->mat[0],jac->pmat[0]));
      }
      if (kspUpper != kspA) {
        PetscCall(KSPSetOperators(kspUpper,jac->mat[0],jac->pmat[0]));
      }
      PetscCall(KSPSetOperators(jac->kspschur,jac->schur,FieldSplitSchurPre(jac)));
    } else {
      const char   *Dprefix;
      char         schurprefix[256], schurmatprefix[256];
      char         schurtestoption[256];
      MatNullSpace sp;
      PetscBool    flg;
      KSP          kspt;

      /* extract the A01 and A10 matrices */
      ilink = jac->head;
      PetscCall(ISComplement(ilink->is_col,rstart,rend,&ccis));
      if (jac->offdiag_use_amat) {
        PetscCall(MatCreateSubMatrix(pc->mat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->B));
      } else {
        PetscCall(MatCreateSubMatrix(pc->pmat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->B));
      }
      PetscCall(ISDestroy(&ccis));
      ilink = ilink->next;
      PetscCall(ISComplement(ilink->is_col,rstart,rend,&ccis));
      if (jac->offdiag_use_amat) {
        PetscCall(MatCreateSubMatrix(pc->mat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->C));
      } else {
        PetscCall(MatCreateSubMatrix(pc->pmat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->C));
      }
      PetscCall(ISDestroy(&ccis));

      /* Use mat[0] (diagonal block of Amat) preconditioned by pmat[0] to define Schur complement */
      PetscCall(MatCreate(((PetscObject)jac->mat[0])->comm,&jac->schur));
      PetscCall(MatSetType(jac->schur,MATSCHURCOMPLEMENT));
      PetscCall(MatSchurComplementSetSubMatrices(jac->schur,jac->mat[0],jac->pmat[0],jac->B,jac->C,jac->mat[1]));
      PetscCall(PetscSNPrintf(schurmatprefix, sizeof(schurmatprefix), "%sfieldsplit_%s_", ((PetscObject)pc)->prefix ? ((PetscObject)pc)->prefix : "", ilink->splitname));
      PetscCall(MatSetOptionsPrefix(jac->schur,schurmatprefix));
      PetscCall(MatSchurComplementGetKSP(jac->schur,&kspt));
      PetscCall(KSPSetOptionsPrefix(kspt,schurmatprefix));

      /* Note: this is not true in general */
      PetscCall(MatGetNullSpace(jac->mat[1], &sp));
      if (sp) {
        PetscCall(MatSetNullSpace(jac->schur, sp));
      }

      PetscCall(PetscSNPrintf(schurtestoption, sizeof(schurtestoption), "-fieldsplit_%s_inner_", ilink->splitname));
      PetscCall(PetscOptionsFindPairPrefix_Private(((PetscObject)pc)->options,((PetscObject)pc)->prefix, schurtestoption, NULL, &flg));
      if (flg) {
        DM  dmInner;
        KSP kspInner;
        PC  pcInner;

        PetscCall(MatSchurComplementGetKSP(jac->schur, &kspInner));
        PetscCall(KSPReset(kspInner));
        PetscCall(KSPSetOperators(kspInner,jac->mat[0],jac->pmat[0]));
        PetscCall(PetscSNPrintf(schurprefix, sizeof(schurprefix), "%sfieldsplit_%s_inner_", ((PetscObject)pc)->prefix ? ((PetscObject)pc)->prefix : "", ilink->splitname));
        /* Indent this deeper to emphasize the "inner" nature of this solver. */
        PetscCall(PetscObjectIncrementTabLevel((PetscObject)kspInner, (PetscObject) pc, 2));
        PetscCall(PetscObjectIncrementTabLevel((PetscObject)kspInner->pc, (PetscObject) pc, 2));
        PetscCall(KSPSetOptionsPrefix(kspInner, schurprefix));

        /* Set DM for new solver */
        PetscCall(KSPGetDM(jac->head->ksp, &dmInner));
        PetscCall(KSPSetDM(kspInner, dmInner));
        PetscCall(KSPSetDMActive(kspInner, PETSC_FALSE));

        /* Defaults to PCKSP as preconditioner */
        PetscCall(KSPGetPC(kspInner, &pcInner));
        PetscCall(PCSetType(pcInner, PCKSP));
        PetscCall(PCKSPSetKSP(pcInner, jac->head->ksp));
      } else {
         /* Use the outer solver for the inner solve, but revert the KSPPREONLY from PCFieldSplitSetFields_FieldSplit or
          * PCFieldSplitSetIS_FieldSplit. We don't want KSPPREONLY because it makes the Schur complement inexact,
          * preventing Schur complement reduction to be an accurate solve. Usually when an iterative solver is used for
          * S = D - C A_inner^{-1} B, we expect S to be defined using an accurate definition of A_inner^{-1}, so we make
          * GMRES the default. Note that it is also common to use PREONLY for S, in which case S may not be used
          * directly, and the user is responsible for setting an inexact method for fieldsplit's A^{-1}. */
        PetscCall(KSPSetType(jac->head->ksp,KSPGMRES));
        PetscCall(MatSchurComplementSetKSP(jac->schur,jac->head->ksp));
      }
      PetscCall(KSPSetOperators(jac->head->ksp,jac->mat[0],jac->pmat[0]));
      PetscCall(KSPSetFromOptions(jac->head->ksp));
      PetscCall(MatSetFromOptions(jac->schur));

      PetscCall(PetscObjectTypeCompare((PetscObject)jac->schur, MATSCHURCOMPLEMENT, &flg));
      if (flg) { /* Need to do this otherwise PCSetUp_KSP will overwrite the amat of jac->head->ksp */
        KSP kspInner;
        PC  pcInner;

        PetscCall(MatSchurComplementGetKSP(jac->schur, &kspInner));
        PetscCall(KSPGetPC(kspInner, &pcInner));
        PetscCall(PetscObjectTypeCompare((PetscObject)pcInner, PCKSP, &flg));
        if (flg) {
          KSP ksp;

          PetscCall(PCKSPGetKSP(pcInner, &ksp));
          if (ksp == jac->head->ksp) {
            PetscCall(PCSetUseAmat(pcInner, PETSC_TRUE));
          }
        }
      }
      PetscCall(PetscSNPrintf(schurtestoption, sizeof(schurtestoption), "-fieldsplit_%s_upper_", ilink->splitname));
      PetscCall(PetscOptionsFindPairPrefix_Private(((PetscObject)pc)->options,((PetscObject)pc)->prefix, schurtestoption, NULL, &flg));
      if (flg) {
        DM dmInner;

        PetscCall(PetscSNPrintf(schurprefix, sizeof(schurprefix), "%sfieldsplit_%s_upper_", ((PetscObject)pc)->prefix ? ((PetscObject)pc)->prefix : "", ilink->splitname));
        PetscCall(KSPCreate(PetscObjectComm((PetscObject)pc), &jac->kspupper));
        PetscCall(KSPSetErrorIfNotConverged(jac->kspupper,pc->erroriffailure));
        PetscCall(KSPSetOptionsPrefix(jac->kspupper, schurprefix));
        PetscCall(PetscObjectIncrementTabLevel((PetscObject)jac->kspupper, (PetscObject) pc, 1));
        PetscCall(PetscObjectIncrementTabLevel((PetscObject)jac->kspupper->pc, (PetscObject) pc, 1));
        PetscCall(KSPGetDM(jac->head->ksp, &dmInner));
        PetscCall(KSPSetDM(jac->kspupper, dmInner));
        PetscCall(KSPSetDMActive(jac->kspupper, PETSC_FALSE));
        PetscCall(KSPSetFromOptions(jac->kspupper));
        PetscCall(KSPSetOperators(jac->kspupper,jac->mat[0],jac->pmat[0]));
        PetscCall(VecDuplicate(jac->head->x, &jac->head->z));
      } else {
        jac->kspupper = jac->head->ksp;
        PetscCall(PetscObjectReference((PetscObject) jac->head->ksp));
      }

      if (jac->schurpre == PC_FIELDSPLIT_SCHUR_PRE_SELFP) {
        PetscCall(MatSchurComplementGetPmat(jac->schur,MAT_INITIAL_MATRIX,&jac->schurp));
      }
      PetscCall(KSPCreate(PetscObjectComm((PetscObject)pc),&jac->kspschur));
      PetscCall(KSPSetErrorIfNotConverged(jac->kspschur,pc->erroriffailure));
      PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)jac->kspschur));
      PetscCall(PetscObjectIncrementTabLevel((PetscObject)jac->kspschur,(PetscObject)pc,1));
      if (jac->schurpre == PC_FIELDSPLIT_SCHUR_PRE_SELF) {
        PC pcschur;
        PetscCall(KSPGetPC(jac->kspschur,&pcschur));
        PetscCall(PCSetType(pcschur,PCNONE));
        /* Note: This is bad if there exist preconditioners for MATSCHURCOMPLEMENT */
      } else if (jac->schurpre == PC_FIELDSPLIT_SCHUR_PRE_FULL) {
        PetscCall(MatSchurComplementComputeExplicitOperator(jac->schur, &jac->schur_user));
      }
      PetscCall(KSPSetOperators(jac->kspschur,jac->schur,FieldSplitSchurPre(jac)));
      PetscCall(KSPGetOptionsPrefix(jac->head->next->ksp, &Dprefix));
      PetscCall(KSPSetOptionsPrefix(jac->kspschur,         Dprefix));
      /* propagate DM */
      {
        DM sdm;
        PetscCall(KSPGetDM(jac->head->next->ksp, &sdm));
        if (sdm) {
          PetscCall(KSPSetDM(jac->kspschur, sdm));
          PetscCall(KSPSetDMActive(jac->kspschur, PETSC_FALSE));
        }
      }
      /* really want setfromoptions called in PCSetFromOptions_FieldSplit(), but it is not ready yet */
      /* need to call this every time, since the jac->kspschur is freshly created, otherwise its options never get set */
      PetscCall(KSPSetFromOptions(jac->kspschur));
    }
    PetscCall(MatAssemblyBegin(jac->schur,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(jac->schur,MAT_FINAL_ASSEMBLY));

    /* HACK: special support to forward L and Lp matrices that might be used by PCLSC */
    PetscCall(PetscSNPrintf(lscname,sizeof(lscname),"%s_LSC_L",ilink->splitname));
    PetscCall(PetscObjectQuery((PetscObject)pc->mat,lscname,(PetscObject*)&LSC_L));
    if (!LSC_L) PetscCall(PetscObjectQuery((PetscObject)pc->pmat,lscname,(PetscObject*)&LSC_L));
    if (LSC_L) PetscCall(PetscObjectCompose((PetscObject)jac->schur,"LSC_L",(PetscObject)LSC_L));
    PetscCall(PetscSNPrintf(lscname,sizeof(lscname),"%s_LSC_Lp",ilink->splitname));
    PetscCall(PetscObjectQuery((PetscObject)pc->pmat,lscname,(PetscObject*)&LSC_L));
    if (!LSC_L) PetscCall(PetscObjectQuery((PetscObject)pc->mat,lscname,(PetscObject*)&LSC_L));
    if (LSC_L) PetscCall(PetscObjectCompose((PetscObject)jac->schur,"LSC_Lp",(PetscObject)LSC_L));
  } else if (jac->type == PC_COMPOSITE_GKB) {
    IS          ccis;
    PetscInt    rstart,rend;

    PetscCheck(nsplit == 2,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_INCOMP,"To use GKB preconditioner you must have exactly 2 fields");

    ilink = jac->head;

    /* When extracting off-diagonal submatrices, we take complements from this range */
    PetscCall(MatGetOwnershipRangeColumn(pc->mat,&rstart,&rend));

    PetscCall(ISComplement(ilink->is_col,rstart,rend,&ccis));
    if (jac->offdiag_use_amat) {
      PetscCall(MatCreateSubMatrix(pc->mat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->B));
    } else {
      PetscCall(MatCreateSubMatrix(pc->pmat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->B));
    }
    PetscCall(ISDestroy(&ccis));
    /* Create work vectors for GKB algorithm */
    PetscCall(VecDuplicate(ilink->x,&jac->u));
    PetscCall(VecDuplicate(ilink->x,&jac->Hu));
    PetscCall(VecDuplicate(ilink->x,&jac->w2));
    ilink = ilink->next;
    PetscCall(ISComplement(ilink->is_col,rstart,rend,&ccis));
    if (jac->offdiag_use_amat) {
      PetscCall(MatCreateSubMatrix(pc->mat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->C));
    } else {
      PetscCall(MatCreateSubMatrix(pc->pmat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->C));
    }
    PetscCall(ISDestroy(&ccis));
    /* Create work vectors for GKB algorithm */
    PetscCall(VecDuplicate(ilink->x,&jac->v));
    PetscCall(VecDuplicate(ilink->x,&jac->d));
    PetscCall(VecDuplicate(ilink->x,&jac->w1));
    PetscCall(MatGolubKahanComputeExplicitOperator(jac->mat[0],jac->B,jac->C,&jac->H,jac->gkbnu));
    PetscCall(PetscCalloc1(jac->gkbdelay,&jac->vecz));

    ilink = jac->head;
    PetscCall(KSPSetOperators(ilink->ksp,jac->H,jac->H));
    if (!jac->suboptionsset) PetscCall(KSPSetFromOptions(ilink->ksp));
    /* Create gkb_monitor context */
    if (jac->gkbmonitor) {
      PetscInt  tablevel;
      PetscCall(PetscViewerCreate(PETSC_COMM_WORLD,&jac->gkbviewer));
      PetscCall(PetscViewerSetType(jac->gkbviewer,PETSCVIEWERASCII));
      PetscCall(PetscObjectGetTabLevel((PetscObject)ilink->ksp,&tablevel));
      PetscCall(PetscViewerASCIISetTab(jac->gkbviewer,tablevel));
      PetscCall(PetscObjectIncrementTabLevel((PetscObject)ilink->ksp,(PetscObject)ilink->ksp,1));
    }
  } else {
    /* set up the individual splits' PCs */
    i     = 0;
    ilink = jac->head;
    while (ilink) {
      PetscCall(KSPSetOperators(ilink->ksp,jac->mat[i],jac->pmat[i]));
      /* really want setfromoptions called in PCSetFromOptions_FieldSplit(), but it is not ready yet */
      if (!jac->suboptionsset) PetscCall(KSPSetFromOptions(ilink->ksp));
      i++;
      ilink = ilink->next;
    }
  }

  /* Set coordinates to the sub PC objects whenever these are set */
  if (jac->coordinates_set) {
    PC pc_coords;
    if (jac->type == PC_COMPOSITE_SCHUR) {
      // Head is first block.
      PetscCall(KSPGetPC(jac->head->ksp, &pc_coords));
      PetscCall(PCSetCoordinates(pc_coords, jac->head->dim, jac->head->ndofs, jac->head->coords));
      // Second one is Schur block, but its KSP object is in kspschur.
      PetscCall(KSPGetPC(jac->kspschur, &pc_coords));
      PetscCall(PCSetCoordinates(pc_coords, jac->head->next->dim, jac->head->next->ndofs, jac->head->next->coords));
    } else if (jac->type == PC_COMPOSITE_GKB) {
      PetscCall(PetscInfo(pc, "Warning: Setting coordinates does nothing for the GKB Fieldpslit preconditioner"));
    } else {
      ilink = jac->head;
      while (ilink) {
        PetscCall(KSPGetPC(ilink->ksp, &pc_coords));
        PetscCall(PCSetCoordinates(pc_coords, ilink->dim, ilink->ndofs, ilink->coords));
        ilink = ilink->next;
      }
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
  PC_FieldSplitLink  ilinkA = jac->head, ilinkD = ilinkA->next;
  KSP                kspA   = ilinkA->ksp, kspLower = kspA, kspUpper = jac->kspupper;

  PetscFunctionBegin;
  switch (jac->schurfactorization) {
  case PC_FIELDSPLIT_SCHUR_FACT_DIAG:
    /* [A00 0; 0 -S], positive definite, suitable for MINRES */
    PetscCall(VecScatterBegin(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterBegin(ilinkD->sctx,x,ilinkD->x,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(PetscLogEventBegin(ilinkA->event,kspA,ilinkA->x,ilinkA->y,NULL));
    PetscCall(KSPSolve(kspA,ilinkA->x,ilinkA->y));
    PetscCall(KSPCheckSolve(kspA,pc,ilinkA->y));
    PetscCall(PetscLogEventEnd(ilinkA->event,kspA,ilinkA->x,ilinkA->y,NULL));
    PetscCall(VecScatterBegin(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(ilinkD->sctx,x,ilinkD->x,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(PetscLogEventBegin(KSP_Solve_FS_S,jac->kspschur,ilinkD->x,ilinkD->y,NULL));
    PetscCall(KSPSolve(jac->kspschur,ilinkD->x,ilinkD->y));
    PetscCall(KSPCheckSolve(jac->kspschur,pc,ilinkD->y));
    PetscCall(PetscLogEventEnd(KSP_Solve_FS_S,jac->kspschur,ilinkD->x,ilinkD->y,NULL));
    PetscCall(VecScale(ilinkD->y,jac->schurscale));
    PetscCall(VecScatterEnd(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterBegin(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE));
    break;
  case PC_FIELDSPLIT_SCHUR_FACT_LOWER:
    /* [A00 0; A10 S], suitable for left preconditioning */
    PetscCall(VecScatterBegin(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(PetscLogEventBegin(ilinkA->event,kspA,ilinkA->x,ilinkA->y,NULL));
    PetscCall(KSPSolve(kspA,ilinkA->x,ilinkA->y));
    PetscCall(KSPCheckSolve(kspA,pc,ilinkA->y));
    PetscCall(PetscLogEventEnd(ilinkA->event,kspA,ilinkA->x,ilinkA->y,NULL));
    PetscCall(MatMult(jac->C,ilinkA->y,ilinkD->x));
    PetscCall(VecScale(ilinkD->x,-1.));
    PetscCall(VecScatterBegin(ilinkD->sctx,x,ilinkD->x,ADD_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterBegin(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(ilinkD->sctx,x,ilinkD->x,ADD_VALUES,SCATTER_FORWARD));
    PetscCall(PetscLogEventBegin(KSP_Solve_FS_S,jac->kspschur,ilinkD->x,ilinkD->y,NULL));
    PetscCall(KSPSolve(jac->kspschur,ilinkD->x,ilinkD->y));
    PetscCall(KSPCheckSolve(jac->kspschur,pc,ilinkD->y));
    PetscCall(PetscLogEventEnd(KSP_Solve_FS_S,jac->kspschur,ilinkD->x,ilinkD->y,NULL));
    PetscCall(VecScatterEnd(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterBegin(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE));
    break;
  case PC_FIELDSPLIT_SCHUR_FACT_UPPER:
    /* [A00 A01; 0 S], suitable for right preconditioning */
    PetscCall(VecScatterBegin(ilinkD->sctx,x,ilinkD->x,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ilinkD->sctx,x,ilinkD->x,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(PetscLogEventBegin(KSP_Solve_FS_S,jac->kspschur,ilinkD->x,ilinkD->y,NULL));
    PetscCall(KSPSolve(jac->kspschur,ilinkD->x,ilinkD->y));
    PetscCall(KSPCheckSolve(jac->kspschur,pc,ilinkD->y));
    PetscCall(PetscLogEventEnd(KSP_Solve_FS_S,jac->kspschur,ilinkD->x,ilinkD->y,NULL));    PetscCall(MatMult(jac->B,ilinkD->y,ilinkA->x));
    PetscCall(VecScale(ilinkA->x,-1.));
    PetscCall(VecScatterBegin(ilinkA->sctx,x,ilinkA->x,ADD_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterBegin(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(ilinkA->sctx,x,ilinkA->x,ADD_VALUES,SCATTER_FORWARD));
    PetscCall(PetscLogEventBegin(ilinkA->event,kspA,ilinkA->x,ilinkA->y,NULL));
    PetscCall(KSPSolve(kspA,ilinkA->x,ilinkA->y));
    PetscCall(KSPCheckSolve(kspA,pc,ilinkA->y));
    PetscCall(PetscLogEventEnd(ilinkA->event,kspA,ilinkA->x,ilinkA->y,NULL));
    PetscCall(VecScatterEnd(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterBegin(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE));
    break;
  case PC_FIELDSPLIT_SCHUR_FACT_FULL:
    /* [1 0; A10 A00^{-1} 1] [A00 0; 0 S] [1 A00^{-1}A01; 0 1] */
    PetscCall(VecScatterBegin(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(PetscLogEventBegin(KSP_Solve_FS_L,kspLower,ilinkA->x,ilinkA->y,NULL));
    PetscCall(KSPSolve(kspLower,ilinkA->x,ilinkA->y));
    PetscCall(KSPCheckSolve(kspLower,pc,ilinkA->y));
    PetscCall(PetscLogEventEnd(KSP_Solve_FS_L,kspLower,ilinkA->x,ilinkA->y,NULL));
    PetscCall(MatMult(jac->C,ilinkA->y,ilinkD->x));
    PetscCall(VecScale(ilinkD->x,-1.0));
    PetscCall(VecScatterBegin(ilinkD->sctx,x,ilinkD->x,ADD_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ilinkD->sctx,x,ilinkD->x,ADD_VALUES,SCATTER_FORWARD));

    PetscCall(PetscLogEventBegin(KSP_Solve_FS_S,jac->kspschur,ilinkD->x,ilinkD->y,NULL));
    PetscCall(KSPSolve(jac->kspschur,ilinkD->x,ilinkD->y));
    PetscCall(KSPCheckSolve(jac->kspschur,pc,ilinkD->y));
    PetscCall(PetscLogEventEnd(KSP_Solve_FS_S,jac->kspschur,ilinkD->x,ilinkD->y,NULL));
    PetscCall(VecScatterBegin(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE));

    if (kspUpper == kspA) {
      PetscCall(MatMult(jac->B,ilinkD->y,ilinkA->y));
      PetscCall(VecAXPY(ilinkA->x,-1.0,ilinkA->y));
      PetscCall(PetscLogEventBegin(ilinkA->event,kspA,ilinkA->x,ilinkA->y,NULL));
      PetscCall(KSPSolve(kspA,ilinkA->x,ilinkA->y));
      PetscCall(KSPCheckSolve(kspA,pc,ilinkA->y));
      PetscCall(PetscLogEventEnd(ilinkA->event,kspA,ilinkA->x,ilinkA->y,NULL));
    } else {
      PetscCall(PetscLogEventBegin(ilinkA->event,kspA,ilinkA->x,ilinkA->y,NULL));
      PetscCall(KSPSolve(kspA,ilinkA->x,ilinkA->y));
      PetscCall(KSPCheckSolve(kspA,pc,ilinkA->y));
      PetscCall(MatMult(jac->B,ilinkD->y,ilinkA->x));
      PetscCall(PetscLogEventBegin(KSP_Solve_FS_U,kspUpper,ilinkA->x,ilinkA->z,NULL));
      PetscCall(KSPSolve(kspUpper,ilinkA->x,ilinkA->z));
      PetscCall(KSPCheckSolve(kspUpper,pc,ilinkA->z));
      PetscCall(PetscLogEventEnd(KSP_Solve_FS_U,kspUpper,ilinkA->x,ilinkA->z,NULL));
      PetscCall(VecAXPY(ilinkA->y,-1.0,ilinkA->z));
    }
    PetscCall(VecScatterEnd(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterBegin(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_FieldSplit(PC pc,Vec x,Vec y)
{
  PC_FieldSplit      *jac = (PC_FieldSplit*)pc->data;
  PC_FieldSplitLink  ilink = jac->head;
  PetscInt           cnt,bs;

  PetscFunctionBegin;
  if (jac->type == PC_COMPOSITE_ADDITIVE) {
    if (jac->defaultsplit) {
      PetscCall(VecGetBlockSize(x,&bs));
      PetscCheckFalse(jac->bs > 0 && bs != jac->bs,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Blocksize of x vector %" PetscInt_FMT " does not match fieldsplit blocksize %" PetscInt_FMT,bs,jac->bs);
      PetscCall(VecGetBlockSize(y,&bs));
      PetscCheckFalse(jac->bs > 0 && bs != jac->bs,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Blocksize of y vector %" PetscInt_FMT " does not match fieldsplit blocksize %" PetscInt_FMT,bs,jac->bs);
      PetscCall(VecStrideGatherAll(x,jac->x,INSERT_VALUES));
      while (ilink) {
        PetscCall(PetscLogEventBegin(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL));
        PetscCall(KSPSolve(ilink->ksp,ilink->x,ilink->y));
        PetscCall(KSPCheckSolve(ilink->ksp,pc,ilink->y));
        PetscCall(PetscLogEventEnd(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL));
        ilink = ilink->next;
      }
      PetscCall(VecStrideScatterAll(jac->y,y,INSERT_VALUES));
    } else {
      PetscCall(VecSet(y,0.0));
      while (ilink) {
        PetscCall(FieldSplitSplitSolveAdd(ilink,x,y));
        ilink = ilink->next;
      }
    }
  } else if (jac->type == PC_COMPOSITE_MULTIPLICATIVE && jac->nsplits == 2) {
    PetscCall(VecSet(y,0.0));
    /* solve on first block for first block variables */
    PetscCall(VecScatterBegin(ilink->sctx,x,ilink->x,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ilink->sctx,x,ilink->x,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(PetscLogEventBegin(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL));
    PetscCall(KSPSolve(ilink->ksp,ilink->x,ilink->y));
    PetscCall(KSPCheckSolve(ilink->ksp,pc,ilink->y));
    PetscCall(PetscLogEventEnd(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL));
    PetscCall(VecScatterBegin(ilink->sctx,ilink->y,y,ADD_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(ilink->sctx,ilink->y,y,ADD_VALUES,SCATTER_REVERSE));

    /* compute the residual only onto second block variables using first block variables */
    PetscCall(MatMult(jac->Afield[1],ilink->y,ilink->next->x));
    ilink = ilink->next;
    PetscCall(VecScale(ilink->x,-1.0));
    PetscCall(VecScatterBegin(ilink->sctx,x,ilink->x,ADD_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ilink->sctx,x,ilink->x,ADD_VALUES,SCATTER_FORWARD));

    /* solve on second block variables */
    PetscCall(PetscLogEventBegin(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL));
    PetscCall(KSPSolve(ilink->ksp,ilink->x,ilink->y));
    PetscCall(KSPCheckSolve(ilink->ksp,pc,ilink->y));
    PetscCall(PetscLogEventEnd(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL));
    PetscCall(VecScatterBegin(ilink->sctx,ilink->y,y,ADD_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(ilink->sctx,ilink->y,y,ADD_VALUES,SCATTER_REVERSE));
  } else if (jac->type == PC_COMPOSITE_MULTIPLICATIVE || jac->type == PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE) {
    if (!jac->w1) {
      PetscCall(VecDuplicate(x,&jac->w1));
      PetscCall(VecDuplicate(x,&jac->w2));
    }
    PetscCall(VecSet(y,0.0));
    PetscCall(FieldSplitSplitSolveAdd(ilink,x,y));
    cnt  = 1;
    while (ilink->next) {
      ilink = ilink->next;
      /* compute the residual only over the part of the vector needed */
      PetscCall(MatMult(jac->Afield[cnt++],y,ilink->x));
      PetscCall(VecScale(ilink->x,-1.0));
      PetscCall(VecScatterBegin(ilink->sctx,x,ilink->x,ADD_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(ilink->sctx,x,ilink->x,ADD_VALUES,SCATTER_FORWARD));
      PetscCall(PetscLogEventBegin(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL));
      PetscCall(KSPSolve(ilink->ksp,ilink->x,ilink->y));
      PetscCall(KSPCheckSolve(ilink->ksp,pc,ilink->y));
      PetscCall(PetscLogEventEnd(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL));
      PetscCall(VecScatterBegin(ilink->sctx,ilink->y,y,ADD_VALUES,SCATTER_REVERSE));
      PetscCall(VecScatterEnd(ilink->sctx,ilink->y,y,ADD_VALUES,SCATTER_REVERSE));
    }
    if (jac->type == PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE) {
      cnt -= 2;
      while (ilink->previous) {
        ilink = ilink->previous;
        /* compute the residual only over the part of the vector needed */
        PetscCall(MatMult(jac->Afield[cnt--],y,ilink->x));
        PetscCall(VecScale(ilink->x,-1.0));
        PetscCall(VecScatterBegin(ilink->sctx,x,ilink->x,ADD_VALUES,SCATTER_FORWARD));
        PetscCall(VecScatterEnd(ilink->sctx,x,ilink->x,ADD_VALUES,SCATTER_FORWARD));
        PetscCall(PetscLogEventBegin(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL));
        PetscCall(KSPSolve(ilink->ksp,ilink->x,ilink->y));
        PetscCall(KSPCheckSolve(ilink->ksp,pc,ilink->y));
        PetscCall(PetscLogEventEnd(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL));
        PetscCall(VecScatterBegin(ilink->sctx,ilink->y,y,ADD_VALUES,SCATTER_REVERSE));
        PetscCall(VecScatterEnd(ilink->sctx,ilink->y,y,ADD_VALUES,SCATTER_REVERSE));
      }
    }
  } else SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Unsupported or unknown composition %d",(int) jac->type);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_FieldSplit_GKB(PC pc,Vec x,Vec y)
{
  PC_FieldSplit      *jac = (PC_FieldSplit*)pc->data;
  PC_FieldSplitLink  ilinkA = jac->head,ilinkD = ilinkA->next;
  KSP                ksp = ilinkA->ksp;
  Vec                u,v,Hu,d,work1,work2;
  PetscScalar        alpha,z,nrmz2,*vecz;
  PetscReal          lowbnd,nu,beta;
  PetscInt           j,iterGKB;

  PetscFunctionBegin;
  PetscCall(VecScatterBegin(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterBegin(ilinkD->sctx,x,ilinkD->x,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ilinkD->sctx,x,ilinkD->x,INSERT_VALUES,SCATTER_FORWARD));

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
    PetscCall(VecScale(ilinkD->x,jac->gkbnu));
    PetscCall(MatMultAdd(jac->B,ilinkD->x,ilinkA->x,ilinkA->x));            /* q = q + nu*B*b */
  } else {
    /* Situation when no augmented Lagrangian is used. Then we set inner  */
    /* matrix N = I in [Ar13], and thus nu = 1.                           */
    nu = 1;
  }

  /* Transform rhs from [q,tilde{b}] to [0,b] */
  PetscCall(PetscLogEventBegin(ilinkA->event,ksp,ilinkA->x,ilinkA->y,NULL));
  PetscCall(KSPSolve(ksp,ilinkA->x,ilinkA->y));
  PetscCall(KSPCheckSolve(ksp,pc,ilinkA->y));
  PetscCall(PetscLogEventEnd(ilinkA->event,ksp,ilinkA->x,ilinkA->y,NULL));
  PetscCall(MatMultHermitianTranspose(jac->B,ilinkA->y,work1));
  PetscCall(VecAXPBY(work1,1.0/nu,-1.0,ilinkD->x));            /* c = b - B'*x        */

  /* First step of algorithm */
  PetscCall(VecNorm(work1,NORM_2,&beta));                   /* beta = sqrt(nu*c'*c)*/
  KSPCheckDot(ksp,beta);
  beta  = PetscSqrtReal(nu)*beta;
  PetscCall(VecAXPBY(v,nu/beta,0.0,work1));                   /* v = nu/beta *c      */
  PetscCall(MatMult(jac->B,v,work2));                       /* u = H^{-1}*B*v      */
  PetscCall(PetscLogEventBegin(ilinkA->event,ksp,work2,u,NULL));
  PetscCall(KSPSolve(ksp,work2,u));
  PetscCall(KSPCheckSolve(ksp,pc,u));
  PetscCall(PetscLogEventEnd(ilinkA->event,ksp,work2,u,NULL));
  PetscCall(MatMult(jac->H,u,Hu));                          /* alpha = u'*H*u      */
  PetscCall(VecDot(Hu,u,&alpha));
  KSPCheckDot(ksp,alpha);
  PetscCheck(PetscRealPart(alpha) > 0.0,PETSC_COMM_SELF,PETSC_ERR_NOT_CONVERGED,"GKB preconditioner diverged, H is not positive definite");
  alpha = PetscSqrtReal(PetscAbsScalar(alpha));
  PetscCall(VecScale(u,1.0/alpha));
  PetscCall(VecAXPBY(d,1.0/alpha,0.0,v));                       /* v = nu/beta *c      */

  z = beta/alpha;
  vecz[1] = z;

  /* Computation of first iterate x(1) and p(1) */
  PetscCall(VecAXPY(ilinkA->y,z,u));
  PetscCall(VecCopy(d,ilinkD->y));
  PetscCall(VecScale(ilinkD->y,-z));

  iterGKB = 1; lowbnd = 2*jac->gkbtol;
  if (jac->gkbmonitor) {
    PetscCall(PetscViewerASCIIPrintf(jac->gkbviewer,"%3" PetscInt_FMT " GKB Lower bound estimate %14.12e\n",iterGKB,(double)lowbnd));
  }

  while (iterGKB < jac->gkbmaxit && lowbnd > jac->gkbtol) {
    iterGKB += 1;
    PetscCall(MatMultHermitianTranspose(jac->B,u,work1)); /* v <- nu*(B'*u-alpha/nu*v) */
    PetscCall(VecAXPBY(v,nu,-alpha,work1));
    PetscCall(VecNorm(v,NORM_2,&beta));                   /* beta = sqrt(nu)*v'*v      */
    beta  = beta/PetscSqrtReal(nu);
    PetscCall(VecScale(v,1.0/beta));
    PetscCall(MatMult(jac->B,v,work2));                  /* u <- H^{-1}*(B*v-beta*H*u) */
    PetscCall(MatMult(jac->H,u,Hu));
    PetscCall(VecAXPY(work2,-beta,Hu));
    PetscCall(PetscLogEventBegin(ilinkA->event,ksp,work2,u,NULL));
    PetscCall(KSPSolve(ksp,work2,u));
    PetscCall(KSPCheckSolve(ksp,pc,u));
    PetscCall(PetscLogEventEnd(ilinkA->event,ksp,work2,u,NULL));
    PetscCall(MatMult(jac->H,u,Hu));                      /* alpha = u'*H*u            */
    PetscCall(VecDot(Hu,u,&alpha));
    KSPCheckDot(ksp,alpha);
    PetscCheck(PetscRealPart(alpha) > 0.0,PETSC_COMM_SELF,PETSC_ERR_NOT_CONVERGED,"GKB preconditioner diverged, H is not positive definite");
    alpha = PetscSqrtReal(PetscAbsScalar(alpha));
    PetscCall(VecScale(u,1.0/alpha));

    z = -beta/alpha*z;                                            /* z <- beta/alpha*z     */
    vecz[0] = z;

    /* Computation of new iterate x(i+1) and p(i+1) */
    PetscCall(VecAXPBY(d,1.0/alpha,-beta/alpha,v));       /* d = (v-beta*d)/alpha */
    PetscCall(VecAXPY(ilinkA->y,z,u));                  /* r = r + z*u          */
    PetscCall(VecAXPY(ilinkD->y,-z,d));                 /* p = p - z*d          */
    PetscCall(MatMult(jac->H,ilinkA->y,Hu));            /* ||u||_H = u'*H*u     */
    PetscCall(VecDot(Hu,ilinkA->y,&nrmz2));

    /* Compute Lower Bound estimate */
    if (iterGKB > jac->gkbdelay) {
      lowbnd = 0.0;
      for (j=0; j<jac->gkbdelay; j++) {
        lowbnd += PetscAbsScalar(vecz[j]*vecz[j]);
      }
      lowbnd = PetscSqrtReal(lowbnd/PetscAbsScalar(nrmz2));
    }

    for (j=0; j<jac->gkbdelay-1; j++) {
      vecz[jac->gkbdelay-j-1] = vecz[jac->gkbdelay-j-2];
    }
    if (jac->gkbmonitor) {
      PetscCall(PetscViewerASCIIPrintf(jac->gkbviewer,"%3" PetscInt_FMT " GKB Lower bound estimate %14.12e\n",iterGKB,(double)lowbnd));
    }
  }

  PetscCall(VecScatterBegin(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterBegin(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE));

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
  PC_FieldSplitLink  ilink = jac->head;
  PetscInt           bs;

  PetscFunctionBegin;
  if (jac->type == PC_COMPOSITE_ADDITIVE) {
    if (jac->defaultsplit) {
      PetscCall(VecGetBlockSize(x,&bs));
      PetscCheckFalse(jac->bs > 0 && bs != jac->bs,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Blocksize of x vector %" PetscInt_FMT " does not match fieldsplit blocksize %" PetscInt_FMT,bs,jac->bs);
      PetscCall(VecGetBlockSize(y,&bs));
      PetscCheckFalse(jac->bs > 0 && bs != jac->bs,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Blocksize of y vector %" PetscInt_FMT " does not match fieldsplit blocksize %" PetscInt_FMT,bs,jac->bs);
      PetscCall(VecStrideGatherAll(x,jac->x,INSERT_VALUES));
      while (ilink) {
        PetscCall(PetscLogEventBegin(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL));
        PetscCall(KSPSolveTranspose(ilink->ksp,ilink->x,ilink->y));
        PetscCall(KSPCheckSolve(ilink->ksp,pc,ilink->y));
        PetscCall(PetscLogEventEnd(ilink->event,ilink->ksp,ilink->x,ilink->y,NULL));
        ilink = ilink->next;
      }
      PetscCall(VecStrideScatterAll(jac->y,y,INSERT_VALUES));
    } else {
      PetscCall(VecSet(y,0.0));
      while (ilink) {
        PetscCall(FieldSplitSplitSolveAddTranspose(ilink,x,y));
        ilink = ilink->next;
      }
    }
  } else {
    if (!jac->w1) {
      PetscCall(VecDuplicate(x,&jac->w1));
      PetscCall(VecDuplicate(x,&jac->w2));
    }
    PetscCall(VecSet(y,0.0));
    if (jac->type == PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE) {
      PetscCall(FieldSplitSplitSolveAddTranspose(ilink,x,y));
      while (ilink->next) {
        ilink = ilink->next;
        PetscCall(MatMultTranspose(pc->mat,y,jac->w1));
        PetscCall(VecWAXPY(jac->w2,-1.0,jac->w1,x));
        PetscCall(FieldSplitSplitSolveAddTranspose(ilink,jac->w2,y));
      }
      while (ilink->previous) {
        ilink = ilink->previous;
        PetscCall(MatMultTranspose(pc->mat,y,jac->w1));
        PetscCall(VecWAXPY(jac->w2,-1.0,jac->w1,x));
        PetscCall(FieldSplitSplitSolveAddTranspose(ilink,jac->w2,y));
      }
    } else {
      while (ilink->next) {   /* get to last entry in linked list */
        ilink = ilink->next;
      }
      PetscCall(FieldSplitSplitSolveAddTranspose(ilink,x,y));
      while (ilink->previous) {
        ilink = ilink->previous;
        PetscCall(MatMultTranspose(pc->mat,y,jac->w1));
        PetscCall(VecWAXPY(jac->w2,-1.0,jac->w1,x));
        PetscCall(FieldSplitSplitSolveAddTranspose(ilink,jac->w2,y));
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_FieldSplit(PC pc)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PC_FieldSplitLink ilink = jac->head,next;

  PetscFunctionBegin;
  while (ilink) {
    PetscCall(KSPDestroy(&ilink->ksp));
    PetscCall(VecDestroy(&ilink->x));
    PetscCall(VecDestroy(&ilink->y));
    PetscCall(VecDestroy(&ilink->z));
    PetscCall(VecScatterDestroy(&ilink->sctx));
    PetscCall(ISDestroy(&ilink->is));
    PetscCall(ISDestroy(&ilink->is_col));
    PetscCall(PetscFree(ilink->splitname));
    PetscCall(PetscFree(ilink->fields));
    PetscCall(PetscFree(ilink->fields_col));
    next  = ilink->next;
    PetscCall(PetscFree(ilink));
    ilink = next;
  }
  jac->head = NULL;
  PetscCall(PetscFree2(jac->x,jac->y));
  if (jac->mat && jac->mat != jac->pmat) {
    PetscCall(MatDestroyMatrices(jac->nsplits,&jac->mat));
  } else if (jac->mat) {
    jac->mat = NULL;
  }
  if (jac->pmat) PetscCall(MatDestroyMatrices(jac->nsplits,&jac->pmat));
  if (jac->Afield) PetscCall(MatDestroyMatrices(jac->nsplits,&jac->Afield));
  jac->nsplits = 0;
  PetscCall(VecDestroy(&jac->w1));
  PetscCall(VecDestroy(&jac->w2));
  PetscCall(MatDestroy(&jac->schur));
  PetscCall(MatDestroy(&jac->schurp));
  PetscCall(MatDestroy(&jac->schur_user));
  PetscCall(KSPDestroy(&jac->kspschur));
  PetscCall(KSPDestroy(&jac->kspupper));
  PetscCall(MatDestroy(&jac->B));
  PetscCall(MatDestroy(&jac->C));
  PetscCall(MatDestroy(&jac->H));
  PetscCall(VecDestroy(&jac->u));
  PetscCall(VecDestroy(&jac->v));
  PetscCall(VecDestroy(&jac->Hu));
  PetscCall(VecDestroy(&jac->d));
  PetscCall(PetscFree(jac->vecz));
  PetscCall(PetscViewerDestroy(&jac->gkbviewer));
  jac->isrestrict = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_FieldSplit(PC pc)
{
  PetscFunctionBegin;
  PetscCall(PCReset_FieldSplit(pc));
  PetscCall(PetscFree(pc->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSchurGetSubKSP_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitGetSubKSP_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetFields_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetIS_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetBlockSize_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetSchurPre_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitGetSchurPre_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetSchurFactType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitRestrictIS_C",NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_FieldSplit(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PetscInt        bs;
  PetscBool       flg;
  PC_FieldSplit   *jac = (PC_FieldSplit*)pc->data;
  PCCompositeType ctype;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"FieldSplit options");
  PetscCall(PetscOptionsBool("-pc_fieldsplit_dm_splits","Whether to use DMCreateFieldDecomposition() for splits","PCFieldSplitSetDMSplits",jac->dm_splits,&jac->dm_splits,NULL));
  PetscCall(PetscOptionsInt("-pc_fieldsplit_block_size","Blocksize that defines number of fields","PCFieldSplitSetBlockSize",jac->bs,&bs,&flg));
  if (flg) {
    PetscCall(PCFieldSplitSetBlockSize(pc,bs));
  }
  jac->diag_use_amat = pc->useAmat;
  PetscCall(PetscOptionsBool("-pc_fieldsplit_diag_use_amat","Use Amat (not Pmat) to extract diagonal fieldsplit blocks", "PCFieldSplitSetDiagUseAmat",jac->diag_use_amat,&jac->diag_use_amat,NULL));
  jac->offdiag_use_amat = pc->useAmat;
  PetscCall(PetscOptionsBool("-pc_fieldsplit_off_diag_use_amat","Use Amat (not Pmat) to extract off-diagonal fieldsplit blocks", "PCFieldSplitSetOffDiagUseAmat",jac->offdiag_use_amat,&jac->offdiag_use_amat,NULL));
  PetscCall(PetscOptionsBool("-pc_fieldsplit_detect_saddle_point","Form 2-way split by detecting zero diagonal entries", "PCFieldSplitSetDetectSaddlePoint",jac->detect,&jac->detect,NULL));
  PetscCall(PCFieldSplitSetDetectSaddlePoint(pc,jac->detect)); /* Sets split type and Schur PC type */
  PetscCall(PetscOptionsEnum("-pc_fieldsplit_type","Type of composition","PCFieldSplitSetType",PCCompositeTypes,(PetscEnum)jac->type,(PetscEnum*)&ctype,&flg));
  if (flg) {
    PetscCall(PCFieldSplitSetType(pc,ctype));
  }
  /* Only setup fields once */
  if ((jac->bs > 0) && (jac->nsplits == 0)) {
    /* only allow user to set fields from command line if bs is already known.
       otherwise user can set them in PCFieldSplitSetDefaults() */
    PetscCall(PCFieldSplitSetRuntimeSplits_Private(pc));
    if (jac->splitdefined) PetscCall(PetscInfo(pc,"Splits defined using the options database\n"));
  }
  if (jac->type == PC_COMPOSITE_SCHUR) {
    PetscCall(PetscOptionsGetEnum(((PetscObject)pc)->options,((PetscObject)pc)->prefix,"-pc_fieldsplit_schur_factorization_type",PCFieldSplitSchurFactTypes,(PetscEnum*)&jac->schurfactorization,&flg));
    if (flg) PetscCall(PetscInfo(pc,"Deprecated use of -pc_fieldsplit_schur_factorization_type\n"));
    PetscCall(PetscOptionsEnum("-pc_fieldsplit_schur_fact_type","Which off-diagonal parts of the block factorization to use","PCFieldSplitSetSchurFactType",PCFieldSplitSchurFactTypes,(PetscEnum)jac->schurfactorization,(PetscEnum*)&jac->schurfactorization,NULL));
    PetscCall(PetscOptionsEnum("-pc_fieldsplit_schur_precondition","How to build preconditioner for Schur complement","PCFieldSplitSetSchurPre",PCFieldSplitSchurPreTypes,(PetscEnum)jac->schurpre,(PetscEnum*)&jac->schurpre,NULL));
    PetscCall(PetscOptionsScalar("-pc_fieldsplit_schur_scale","Scale Schur complement","PCFieldSplitSetSchurScale",jac->schurscale,&jac->schurscale,NULL));
  } else if (jac->type == PC_COMPOSITE_GKB) {
    PetscCall(PetscOptionsReal("-pc_fieldsplit_gkb_tol","The tolerance for the lower bound stopping criterion","PCFieldSplitGKBTol",jac->gkbtol,&jac->gkbtol,NULL));
    PetscCall(PetscOptionsInt("-pc_fieldsplit_gkb_delay","The delay value for lower bound criterion","PCFieldSplitGKBDelay",jac->gkbdelay,&jac->gkbdelay,NULL));
    PetscCall(PetscOptionsReal("-pc_fieldsplit_gkb_nu","Parameter in augmented Lagrangian approach","PCFieldSplitGKBNu",jac->gkbnu,&jac->gkbnu,NULL));
    PetscCheck(jac->gkbnu >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nu cannot be less than 0: value %g",(double)jac->gkbnu);
    PetscCall(PetscOptionsInt("-pc_fieldsplit_gkb_maxit","Maximum allowed number of iterations","PCFieldSplitGKBMaxit",jac->gkbmaxit,&jac->gkbmaxit,NULL));
    PetscCall(PetscOptionsBool("-pc_fieldsplit_gkb_monitor","Prints number of GKB iterations and error","PCFieldSplitGKB",jac->gkbmonitor,&jac->gkbmonitor,NULL));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------------------------------*/

static PetscErrorCode  PCFieldSplitSetFields_FieldSplit(PC pc,const char splitname[],PetscInt n,const PetscInt *fields,const PetscInt *fields_col)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PC_FieldSplitLink ilink,next = jac->head;
  char              prefix[128];
  PetscInt          i;

  PetscFunctionBegin;
  if (jac->splitdefined) {
    PetscCall(PetscInfo(pc,"Ignoring new split \"%s\" because the splits have already been defined\n",splitname));
    PetscFunctionReturn(0);
  }
  for (i=0; i<n; i++) {
    PetscCheck(fields[i] < jac->bs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Field %" PetscInt_FMT " requested but only %" PetscInt_FMT " exist",fields[i],jac->bs);
    PetscCheck(fields[i] >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative field %" PetscInt_FMT " requested",fields[i]);
  }
  PetscCall(PetscNew(&ilink));
  if (splitname) {
    PetscCall(PetscStrallocpy(splitname,&ilink->splitname));
  } else {
    PetscCall(PetscMalloc1(3,&ilink->splitname));
    PetscCall(PetscSNPrintf(ilink->splitname,2,"%" PetscInt_FMT,jac->nsplits));
  }
  ilink->event = jac->nsplits < 5 ? KSP_Solve_FS_0 + jac->nsplits : KSP_Solve_FS_0 + 4; /* Any split great than 4 gets logged in the 4th split */
  PetscCall(PetscMalloc1(n,&ilink->fields));
  PetscCall(PetscArraycpy(ilink->fields,fields,n));
  PetscCall(PetscMalloc1(n,&ilink->fields_col));
  PetscCall(PetscArraycpy(ilink->fields_col,fields_col,n));

  ilink->nfields = n;
  ilink->next    = NULL;
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)pc),&ilink->ksp));
  PetscCall(KSPSetErrorIfNotConverged(ilink->ksp,pc->erroriffailure));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)ilink->ksp,(PetscObject)pc,1));
  PetscCall(KSPSetType(ilink->ksp,KSPPREONLY));
  PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)ilink->ksp));

  PetscCall(PetscSNPrintf(prefix,sizeof(prefix),"%sfieldsplit_%s_",((PetscObject)pc)->prefix ? ((PetscObject)pc)->prefix : "",ilink->splitname));
  PetscCall(KSPSetOptionsPrefix(ilink->ksp,prefix));

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

  PetscFunctionBegin;
  *subksp = NULL;
  if (n) *n = 0;
  if (jac->type == PC_COMPOSITE_SCHUR) {
    PetscInt nn;

    PetscCheck(jac->schur,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must call KSPSetUp() or PCSetUp() before calling PCFieldSplitSchurGetSubKSP()");
    PetscCheck(jac->nsplits == 2,PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"Unexpected number of splits %" PetscInt_FMT " != 2",jac->nsplits);
    nn   = jac->nsplits + (jac->kspupper != jac->head->ksp ? 1 : 0);
    PetscCall(PetscMalloc1(nn,subksp));
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

  PetscFunctionBegin;
  PetscCheck(jac->schur,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must call KSPSetUp() or PCSetUp() before calling PCFieldSplitGetSubKSP()");
  PetscCall(PetscMalloc1(jac->nsplits,subksp));
  PetscCall(MatSchurComplementGetKSP(jac->schur,*subksp));

  (*subksp)[1] = jac->kspschur;
  if (n) *n = jac->nsplits;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCFieldSplitGetSubKSP_FieldSplit(PC pc,PetscInt *n,KSP **subksp)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscInt          cnt   = 0;
  PC_FieldSplitLink ilink = jac->head;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(jac->nsplits,subksp));
  while (ilink) {
    (*subksp)[cnt++] = ilink->ksp;
    ilink            = ilink->next;
  }
  PetscCheck(cnt == jac->nsplits,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Corrupt PCFIELDSPLIT object: number of splits in linked list %" PetscInt_FMT " does not match number in object %" PetscInt_FMT,cnt,jac->nsplits);
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(isy,IS_CLASSID,2);
  PetscTryMethod(pc,"PCFieldSplitRestrictIS_C",(PC,IS),(pc,isy));
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCFieldSplitRestrictIS_FieldSplit(PC pc, IS isy)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PC_FieldSplitLink ilink = jac->head, next;
  PetscInt          localsize,size,sizez,i;
  const PetscInt    *ind, *indz;
  PetscInt          *indc, *indcz;
  PetscBool         flg;

  PetscFunctionBegin;
  PetscCall(ISGetLocalSize(isy,&localsize));
  PetscCallMPI(MPI_Scan(&localsize,&size,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)isy)));
  size -= localsize;
  while (ilink) {
    IS isrl,isr;
    PC subpc;
    PetscCall(ISEmbed(ilink->is, isy, PETSC_TRUE, &isrl));
    PetscCall(ISGetLocalSize(isrl,&localsize));
    PetscCall(PetscMalloc1(localsize,&indc));
    PetscCall(ISGetIndices(isrl,&ind));
    PetscCall(PetscArraycpy(indc,ind,localsize));
    PetscCall(ISRestoreIndices(isrl,&ind));
    PetscCall(ISDestroy(&isrl));
    for (i=0; i<localsize; i++) *(indc+i) += size;
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)isy),localsize,indc,PETSC_OWN_POINTER,&isr));
    PetscCall(PetscObjectReference((PetscObject)isr));
    PetscCall(ISDestroy(&ilink->is));
    ilink->is     = isr;
    PetscCall(PetscObjectReference((PetscObject)isr));
    PetscCall(ISDestroy(&ilink->is_col));
    ilink->is_col = isr;
    PetscCall(ISDestroy(&isr));
    PetscCall(KSPGetPC(ilink->ksp, &subpc));
    PetscCall(PetscObjectTypeCompare((PetscObject)subpc,PCFIELDSPLIT,&flg));
    if (flg) {
      IS iszl,isz;
      MPI_Comm comm;
      PetscCall(ISGetLocalSize(ilink->is,&localsize));
      comm   = PetscObjectComm((PetscObject)ilink->is);
      PetscCall(ISEmbed(isy, ilink->is, PETSC_TRUE, &iszl));
      PetscCallMPI(MPI_Scan(&localsize,&sizez,1,MPIU_INT,MPI_SUM,comm));
      sizez -= localsize;
      PetscCall(ISGetLocalSize(iszl,&localsize));
      PetscCall(PetscMalloc1(localsize,&indcz));
      PetscCall(ISGetIndices(iszl,&indz));
      PetscCall(PetscArraycpy(indcz,indz,localsize));
      PetscCall(ISRestoreIndices(iszl,&indz));
      PetscCall(ISDestroy(&iszl));
      for (i=0; i<localsize; i++) *(indcz+i) += sizez;
      PetscCall(ISCreateGeneral(comm,localsize,indcz,PETSC_OWN_POINTER,&isz));
      PetscCall(PCFieldSplitRestrictIS(subpc,isz));
      PetscCall(ISDestroy(&isz));
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
  PC_FieldSplitLink ilink, next = jac->head;
  char              prefix[128];

  PetscFunctionBegin;
  if (jac->splitdefined) {
    PetscCall(PetscInfo(pc,"Ignoring new split \"%s\" because the splits have already been defined\n",splitname));
    PetscFunctionReturn(0);
  }
  PetscCall(PetscNew(&ilink));
  if (splitname) {
    PetscCall(PetscStrallocpy(splitname,&ilink->splitname));
  } else {
    PetscCall(PetscMalloc1(8,&ilink->splitname));
    PetscCall(PetscSNPrintf(ilink->splitname,7,"%" PetscInt_FMT,jac->nsplits));
  }
  ilink->event  = jac->nsplits < 5 ? KSP_Solve_FS_0 + jac->nsplits : KSP_Solve_FS_0 + 4; /* Any split great than 4 gets logged in the 4th split */
  PetscCall(PetscObjectReference((PetscObject)is));
  PetscCall(ISDestroy(&ilink->is));
  ilink->is     = is;
  PetscCall(PetscObjectReference((PetscObject)is));
  PetscCall(ISDestroy(&ilink->is_col));
  ilink->is_col = is;
  ilink->next   = NULL;
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)pc),&ilink->ksp));
  PetscCall(KSPSetErrorIfNotConverged(ilink->ksp,pc->erroriffailure));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)ilink->ksp,(PetscObject)pc,1));
  PetscCall(KSPSetType(ilink->ksp,KSPPREONLY));
  PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)ilink->ksp));

  PetscCall(PetscSNPrintf(prefix,sizeof(prefix),"%sfieldsplit_%s_",((PetscObject)pc)->prefix ? ((PetscObject)pc)->prefix : "",ilink->splitname));
  PetscCall(KSPSetOptionsPrefix(ilink->ksp,prefix));

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

.seealso: `PCFieldSplitGetSubKSP()`, `PCFIELDSPLIT`, `PCFieldSplitSetBlockSize()`, `PCFieldSplitSetIS()`

@*/
PetscErrorCode  PCFieldSplitSetFields(PC pc,const char splitname[],PetscInt n,const PetscInt *fields,const PetscInt *fields_col)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidCharPointer(splitname,2);
  PetscCheck(n >= 1,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Provided number of fields %" PetscInt_FMT " in split \"%s\" not positive",n,splitname);
  PetscValidIntPointer(fields,4);
  PetscTryMethod(pc,"PCFieldSplitSetFields_C",(PC,const char[],PetscInt,const PetscInt*,const PetscInt*),(pc,splitname,n,fields,fields_col));
  PetscFunctionReturn(0);
}

/*@
    PCFieldSplitSetDiagUseAmat - set flag indicating whether to extract diagonal blocks from Amat (rather than Pmat)

    Logically Collective on PC

    Input Parameters:
+   pc  - the preconditioner object
-   flg - boolean flag indicating whether or not to use Amat to extract the diagonal blocks from

    Options Database:
.   -pc_fieldsplit_diag_use_amat - use the Amat to provide the diagonal blocks

    Level: intermediate

.seealso: `PCFieldSplitGetDiagUseAmat()`, `PCFieldSplitSetOffDiagUseAmat()`, `PCFIELDSPLIT`

@*/
PetscErrorCode  PCFieldSplitSetDiagUseAmat(PC pc,PetscBool flg)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;
  PetscBool      isfs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCFIELDSPLIT,&isfs));
  PetscCheck(isfs,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"PC not of type %s",PCFIELDSPLIT);
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

.seealso: `PCFieldSplitSetDiagUseAmat()`, `PCFieldSplitGetOffDiagUseAmat()`, `PCFIELDSPLIT`

@*/
PetscErrorCode  PCFieldSplitGetDiagUseAmat(PC pc,PetscBool *flg)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;
  PetscBool      isfs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCFIELDSPLIT,&isfs));
  PetscCheck(isfs,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"PC not of type %s",PCFIELDSPLIT);
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
.     -pc_fieldsplit_off_diag_use_amat <bool> - use the Amat to extract the off-diagonal blocks

    Level: intermediate

.seealso: `PCFieldSplitGetOffDiagUseAmat()`, `PCFieldSplitSetDiagUseAmat()`, `PCFIELDSPLIT`

@*/
PetscErrorCode  PCFieldSplitSetOffDiagUseAmat(PC pc,PetscBool flg)
{
  PC_FieldSplit *jac = (PC_FieldSplit*)pc->data;
  PetscBool      isfs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCFIELDSPLIT,&isfs));
  PetscCheck(isfs,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"PC not of type %s",PCFIELDSPLIT);
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

.seealso: `PCFieldSplitSetOffDiagUseAmat()`, `PCFieldSplitGetDiagUseAmat()`, `PCFIELDSPLIT`

@*/
PetscErrorCode  PCFieldSplitGetOffDiagUseAmat(PC pc,PetscBool *flg)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;
  PetscBool      isfs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCFIELDSPLIT,&isfs));
  PetscCheck(isfs,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"PC not of type %s",PCFIELDSPLIT);
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

.seealso: `PCFieldSplitGetSubKSP()`, `PCFIELDSPLIT`, `PCFieldSplitSetBlockSize()`

@*/
PetscErrorCode  PCFieldSplitSetIS(PC pc,const char splitname[],IS is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (splitname) PetscValidCharPointer(splitname,2);
  PetscValidHeaderSpecific(is,IS_CLASSID,3);
  PetscTryMethod(pc,"PCFieldSplitSetIS_C",(PC,const char[],IS),(pc,splitname,is));
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

.seealso: `PCFieldSplitGetSubKSP()`, `PCFIELDSPLIT`, `PCFieldSplitSetIS()`

@*/
PetscErrorCode PCFieldSplitGetIS(PC pc,const char splitname[],IS *is)
{
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
      PetscCall(PetscStrcmp(ilink->splitname, splitname, &found));
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

.seealso: `PCFieldSplitGetSubKSP()`, `PCFIELDSPLIT`, `PCFieldSplitGetIS()`, `PCFieldSplitSetIS()`

@*/
PetscErrorCode PCFieldSplitGetISByIndex(PC pc,PetscInt index,IS *is)
{
  PetscFunctionBegin;
  PetscCheck(index >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative field %" PetscInt_FMT " requested",index);
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(is,3);
  {
    PC_FieldSplit     *jac  = (PC_FieldSplit*) pc->data;
    PC_FieldSplitLink ilink = jac->head;
    PetscInt          i     = 0;
    PetscCheck(index < jac->nsplits,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Field %" PetscInt_FMT " requested but only %" PetscInt_FMT " exist",index,jac->nsplits);

    while (i < index) {
      ilink = ilink->next;
      ++i;
    }
    PetscCall(PCFieldSplitGetIS(pc,ilink->splitname,is));
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

.seealso: `PCFieldSplitGetSubKSP()`, `PCFIELDSPLIT`, `PCFieldSplitSetFields()`

@*/
PetscErrorCode  PCFieldSplitSetBlockSize(PC pc,PetscInt bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,bs,2);
  PetscTryMethod(pc,"PCFieldSplitSetBlockSize_C",(PC,PetscInt),(pc,bs));
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

.seealso: `PCFIELDSPLIT`
@*/
PetscErrorCode  PCFieldSplitGetSubKSP(PC pc,PetscInt *n,KSP *subksp[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (n) PetscValidIntPointer(n,2);
  PetscUseMethod(pc,"PCFieldSplitGetSubKSP_C",(PC,PetscInt*,KSP **),(pc,n,subksp));
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

.seealso: `PCFIELDSPLIT`
@*/
PetscErrorCode  PCFieldSplitSchurGetSubKSP(PC pc,PetscInt *n,KSP *subksp[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (n) PetscValidIntPointer(n,2);
  PetscUseMethod(pc,"PCFieldSplitSchurGetSubKSP_C",(PC,PetscInt*,KSP **),(pc,n,subksp));
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

.seealso: `PCFieldSplitGetSchurPre()`, `PCFieldSplitGetSubKSP()`, `PCFIELDSPLIT`, `PCFieldSplitSetFields()`, `PCFieldSplitSchurPreType`,
          `MatSchurComplementSetAinvType()`, `PCLSC`

@*/
PetscErrorCode PCFieldSplitSetSchurPre(PC pc,PCFieldSplitSchurPreType ptype,Mat pre)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCFieldSplitSetSchurPre_C",(PC,PCFieldSplitSchurPreType,Mat),(pc,ptype,pre));
  PetscFunctionReturn(0);
}

PetscErrorCode PCFieldSplitSchurPrecondition(PC pc,PCFieldSplitSchurPreType ptype,Mat pre) {return PCFieldSplitSetSchurPre(pc,ptype,pre);} /* Deprecated name */

/*@
    PCFieldSplitGetSchurPre - For Schur complement fieldsplit, determine how the Schur complement will be
    preconditioned.  See PCFieldSplitSetSchurPre() for details.

    Logically Collective on PC

    Input Parameter:
.   pc      - the preconditioner context

    Output Parameters:
+   ptype   - which matrix to use for preconditioning the Schur complement: PC_FIELDSPLIT_SCHUR_PRE_A11, PC_FIELDSPLIT_SCHUR_PRE_SELF, PC_FIELDSPLIT_PRE_USER
-   userpre - matrix to use for preconditioning (with PC_FIELDSPLIT_PRE_USER), or NULL

    Level: intermediate

.seealso: `PCFieldSplitSetSchurPre()`, `PCFieldSplitGetSubKSP()`, `PCFIELDSPLIT`, `PCFieldSplitSetFields()`, `PCFieldSplitSchurPreType`, `PCLSC`

@*/
PetscErrorCode PCFieldSplitGetSchurPre(PC pc,PCFieldSplitSchurPreType *ptype,Mat *pre)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscUseMethod(pc,"PCFieldSplitGetSchurPre_C",(PC,PCFieldSplitSchurPreType*,Mat*),(pc,ptype,pre));
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

.seealso: `PCFieldSplitGetSubKSP()`, `PCFIELDSPLIT`, `PCFieldSplitSchurPreType`, `PCFieldSplitSetSchurPre()`, `MatSchurComplement`, `PCFieldSplitSchurRestoreS()`

@*/
PetscErrorCode  PCFieldSplitSchurGetS(PC pc,Mat *S)
{
  const char*    t;
  PetscBool      isfs;
  PC_FieldSplit  *jac;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCall(PetscObjectGetType((PetscObject)pc,&t));
  PetscCall(PetscStrcmp(t,PCFIELDSPLIT,&isfs));
  PetscCheck(isfs,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Expected PC of type PCFIELDSPLIT, got %s instead",t);
  jac = (PC_FieldSplit*)pc->data;
  PetscCheck(jac->type == PC_COMPOSITE_SCHUR,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Expected PCFIELDSPLIT of type SCHUR, got %d instead",jac->type);
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

.seealso: `PCFieldSplitGetSubKSP()`, `PCFIELDSPLIT`, `PCFieldSplitSchurPreType`, `PCFieldSplitSetSchurPre()`, `MatSchurComplement`, `PCFieldSplitSchurGetS()`

@*/
PetscErrorCode  PCFieldSplitSchurRestoreS(PC pc,Mat *S)
{
  const char*    t;
  PetscBool      isfs;
  PC_FieldSplit  *jac;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCall(PetscObjectGetType((PetscObject)pc,&t));
  PetscCall(PetscStrcmp(t,PCFIELDSPLIT,&isfs));
  PetscCheck(isfs,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Expected PC of type PCFIELDSPLIT, got %s instead",t);
  jac = (PC_FieldSplit*)pc->data;
  PetscCheck(jac->type == PC_COMPOSITE_SCHUR,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Expected PCFIELDSPLIT of type SCHUR, got %d instead",jac->type);
  PetscCheck(S && (*S == jac->schur),PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"MatSchurComplement restored is not the same as gotten");
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCFieldSplitSetSchurPre_FieldSplit(PC pc,PCFieldSplitSchurPreType ptype,Mat pre)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;

  PetscFunctionBegin;
  jac->schurpre = ptype;
  if (ptype == PC_FIELDSPLIT_SCHUR_PRE_USER && pre) {
    PetscCall(MatDestroy(&jac->schur_user));
    jac->schur_user = pre;
    PetscCall(PetscObjectReference((PetscObject)jac->schur_user));
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
.     -pc_fieldsplit_schur_fact_type <diag,lower,upper,full> - default is full

    Level: intermediate

    Notes:
    The FULL factorization is

.vb
   (A   B)  = (1       0) (A   0) (1  Ainv*B)  = L D U
   (C   E)    (C*Ainv  1) (0   S) (0     1)
.vb
    where S = E - C*Ainv*B. In practice, the full factorization is applied via block triangular solves with the grouping L*(D*U). UPPER uses D*U, LOWER uses L*D,
    and DIAG is the diagonal part with the sign of S flipped (because this makes the preconditioner positive definite for many formulations, thus allowing the use of KSPMINRES). Sign flipping of S can be turned off with PCFieldSplitSetSchurScale().

    If A and S are solved exactly
.vb
      *) FULL factorization is a direct solver.
      *) The preconditioned operator with LOWER or UPPER has all eigenvalues equal to 1 and minimal polynomial of degree 2, so KSPGMRES converges in 2 iterations.
      *) With DIAG, the preconditioned operator has three distinct nonzero eigenvalues and minimal polynomial of degree at most 4, so KSPGMRES converges in at most 4 iterations.
.ve

    If the iteration count is very low, consider using KSPFGMRES or KSPGCR which can use one less preconditioner
    application in this case. Note that the preconditioned operator may be highly non-normal, so such fast convergence may not be observed in practice.

    For symmetric problems in which A is positive definite and S is negative definite, DIAG can be used with KSPMINRES.

    Note that a flexible method like KSPFGMRES or KSPGCR must be used if the fieldsplit preconditioner is nonlinear (e.g. a few iterations of a Krylov method is used to solve with A or S).

    References:
+   * - Murphy, Golub, and Wathen, A note on preconditioning indefinite linear systems, SIAM J. Sci. Comput., 21 (2000).
-   * - Ipsen, A note on preconditioning nonsymmetric matrices, SIAM J. Sci. Comput., 23 (2001).

.seealso: `PCFieldSplitGetSubKSP()`, `PCFIELDSPLIT`, `PCFieldSplitSetFields()`, `PCFieldSplitSchurPreType`, `PCFieldSplitSetSchurScale()`
@*/
PetscErrorCode  PCFieldSplitSetSchurFactType(PC pc,PCFieldSplitSchurFactType ftype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCFieldSplitSetSchurFactType_C",(PC,PCFieldSplitSchurFactType),(pc,ftype));
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

.seealso: `PCFIELDSPLIT`, `PCFieldSplitSetFields()`, `PCFieldSplitSchurFactType`, `PCFieldSplitSetSchurScale()`
@*/
PetscErrorCode PCFieldSplitSetSchurScale(PC pc,PetscScalar scale)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveScalar(pc,scale,2);
  PetscTryMethod(pc,"PCFieldSplitSetSchurScale_C",(PC,PetscScalar),(pc,scale));
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

.seealso: `PCFIELDSPLIT`
@*/
PetscErrorCode  PCFieldSplitGetSchurBlocks(PC pc,Mat *A00,Mat *A01,Mat *A10, Mat *A11)
{
  PC_FieldSplit *jac = (PC_FieldSplit*) pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCheck(jac->type == PC_COMPOSITE_SCHUR,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG, "FieldSplit is not using a Schur complement approach.");
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

.seealso: `PCFIELDSPLIT`, `PCFieldSplitSetGKBDelay()`, `PCFieldSplitSetGKBNu()`, `PCFieldSplitSetGKBMaxit()`
@*/
PetscErrorCode PCFieldSplitSetGKBTol(PC pc,PetscReal tolerance)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveReal(pc,tolerance,2);
  PetscTryMethod(pc,"PCFieldSplitSetGKBTol_C",(PC,PetscReal),(pc,tolerance));
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

.seealso: `PCFIELDSPLIT`, `PCFieldSplitSetGKBDelay()`, `PCFieldSplitSetGKBTol()`, `PCFieldSplitSetGKBNu()`
@*/
PetscErrorCode PCFieldSplitSetGKBMaxit(PC pc,PetscInt maxit)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,maxit,2);
  PetscTryMethod(pc,"PCFieldSplitSetGKBMaxit_C",(PC,PetscInt),(pc,maxit));
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

.seealso: `PCFIELDSPLIT`, `PCFieldSplitSetGKBNu()`, `PCFieldSplitSetGKBTol()`, `PCFieldSplitSetGKBMaxit()`
@*/
PetscErrorCode PCFieldSplitSetGKBDelay(PC pc,PetscInt delay)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,delay,2);
  PetscTryMethod(pc,"PCFieldSplitSetGKBDelay_C",(PC,PetscInt),(pc,delay));
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

.seealso: `PCFIELDSPLIT`, `PCFieldSplitSetGKBDelay()`, `PCFieldSplitSetGKBTol()`, `PCFieldSplitSetGKBMaxit()`
@*/
PetscErrorCode PCFieldSplitSetGKBNu(PC pc,PetscReal nu)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveReal(pc,nu,2);
  PetscTryMethod(pc,"PCFieldSplitSetGKBNu_C",(PC,PetscReal),(pc,nu));
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

  PetscFunctionBegin;
  jac->type = type;

  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitGetSubKSP_C",0));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetSchurPre_C",0));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitGetSchurPre_C",0));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetSchurFactType_C",0));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetSchurScale_C",0));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetGKBTol_C",0));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetGKBMaxit_C",0));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetGKBNu_C",0));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetGKBDelay_C",0));

  if (type == PC_COMPOSITE_SCHUR) {
    pc->ops->apply = PCApply_FieldSplit_Schur;
    pc->ops->view  = PCView_FieldSplit_Schur;

    PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitGetSubKSP_C",PCFieldSplitGetSubKSP_FieldSplit_Schur));
    PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetSchurPre_C",PCFieldSplitSetSchurPre_FieldSplit));
    PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitGetSchurPre_C",PCFieldSplitGetSchurPre_FieldSplit));
    PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetSchurFactType_C",PCFieldSplitSetSchurFactType_FieldSplit));
    PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetSchurScale_C",PCFieldSplitSetSchurScale_FieldSplit));
  } else if (type == PC_COMPOSITE_GKB) {
    pc->ops->apply = PCApply_FieldSplit_GKB;
    pc->ops->view  = PCView_FieldSplit_GKB;

    PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitGetSubKSP_C",PCFieldSplitGetSubKSP_FieldSplit));
    PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetGKBTol_C",PCFieldSplitSetGKBTol_FieldSplit));
    PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetGKBMaxit_C",PCFieldSplitSetGKBMaxit_FieldSplit));
    PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetGKBNu_C",PCFieldSplitSetGKBNu_FieldSplit));
    PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetGKBDelay_C",PCFieldSplitSetGKBDelay_FieldSplit));
  } else {
    pc->ops->apply = PCApply_FieldSplit;
    pc->ops->view  = PCView_FieldSplit;

    PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitGetSubKSP_C",PCFieldSplitGetSubKSP_FieldSplit));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCFieldSplitSetBlockSize_FieldSplit(PC pc,PetscInt bs)
{
  PC_FieldSplit *jac = (PC_FieldSplit*)pc->data;

  PetscFunctionBegin;
  PetscCheck(bs >= 1,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Blocksize must be positive, you gave %" PetscInt_FMT,bs);
  PetscCheckFalse(jac->bs > 0 && jac->bs != bs,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Cannot change fieldsplit blocksize from %" PetscInt_FMT " to %" PetscInt_FMT " after it has been set",jac->bs,bs);
  jac->bs = bs;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetCoordinates_FieldSplit(PC pc, PetscInt dim, PetscInt nloc, PetscReal coords[])
{
  PC_FieldSplit *   jac = (PC_FieldSplit*)pc->data;
  PC_FieldSplitLink ilink_current = jac->head;
  PetscInt          ii;
  IS                is_owned;

  PetscFunctionBegin;
  jac->coordinates_set = PETSC_TRUE; // Internal flag
  PetscCall(MatGetOwnershipIS(pc->mat,&is_owned,PETSC_NULL));

  ii=0;
  while (ilink_current) {
    // For each IS, embed it to get local coords indces
    IS        is_coords;
    PetscInt  ndofs_block;
    const PetscInt *block_dofs_enumeration; // Numbering of the dofs relevant to the current block

    // Setting drop to true for safety. It should make no difference.
    PetscCall(ISEmbed(ilink_current->is, is_owned, PETSC_TRUE, &is_coords));
    PetscCall(ISGetLocalSize(is_coords, &ndofs_block));
    PetscCall(ISGetIndices(is_coords, &block_dofs_enumeration));

    // Allocate coordinates vector and set it directly
    PetscCall(PetscMalloc1(ndofs_block * dim, &(ilink_current->coords)));
    for (PetscInt dof=0;dof<ndofs_block;++dof) {
      for (PetscInt d=0;d<dim;++d) {
        (ilink_current->coords)[dim*dof + d] = coords[dim * block_dofs_enumeration[dof] + d];
      }
    }
    ilink_current->dim = dim;
    ilink_current->ndofs = ndofs_block;
    PetscCall(ISRestoreIndices(is_coords, &block_dofs_enumeration));
    PetscCall(ISDestroy(&is_coords));
    ilink_current = ilink_current->next;
    ++ii;
  }
  PetscCall(ISDestroy(&is_owned));
  PetscFunctionReturn(0);
}

/*@
   PCFieldSplitSetType - Sets the type of fieldsplit preconditioner.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  type - PC_COMPOSITE_ADDITIVE, PC_COMPOSITE_MULTIPLICATIVE (default), PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE, PC_COMPOSITE_SPECIAL, PC_COMPOSITE_SCHUR

   Options Database Key:
.  -pc_fieldsplit_type <type: one of multiplicative, additive, symmetric_multiplicative, special, schur> - Sets fieldsplit preconditioner type

   Level: Intermediate

.seealso: `PCCompositeSetType()`

@*/
PetscErrorCode  PCFieldSplitSetType(PC pc,PCCompositeType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCFieldSplitSetType_C",(PC,PCCompositeType),(pc,type));
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

.seealso: `PCCompositeSetType()`
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
.  -pc_fieldsplit_dm_splits <bool> - use the field splits defined by the DM

   Level: Intermediate

.seealso: `PCFieldSplitGetDMSplits()`

@*/
PetscErrorCode  PCFieldSplitSetDMSplits(PC pc,PetscBool flg)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;
  PetscBool      isfs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveBool(pc,flg,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCFIELDSPLIT,&isfs));
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

.seealso: `PCFieldSplitSetDMSplits()`

@*/
PetscErrorCode  PCFieldSplitGetDMSplits(PC pc,PetscBool* flg)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;
  PetscBool      isfs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCFIELDSPLIT,&isfs));
  if (isfs) {
    if (flg) *flg = jac->dm_splits;
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

.seealso: `PCFIELDSPLIT`, `PCFieldSplitSetDetectSaddlePoint()`

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

   Input Parameter:
.  pc   - the preconditioner context

   Output Parameter:
.  flg  - boolean indicating whether to detect fields or not

   Options Database Key:
.  -pc_fieldsplit_detect_saddle_point <bool> - detect and use the saddle point

   Notes:
   Also sets the split type to PC_COMPOSITE_SCHUR (see PCFieldSplitSetType()) and the Schur preconditioner type to PC_FIELDSPLIT_SCHUR_PRE_SELF (see PCFieldSplitSetSchurPre()).

   Level: Intermediate

.seealso: `PCFIELDSPLIT`, `PCFieldSplitSetDetectSaddlePoint()`, `PCFieldSplitSetType()`, `PCFieldSplitSetSchurPre()`

@*/
PetscErrorCode PCFieldSplitSetDetectSaddlePoint(PC pc,PetscBool flg)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;

  PetscFunctionBegin;
  jac->detect = flg;
  if (jac->detect) {
    PetscCall(PCFieldSplitSetType(pc,PC_COMPOSITE_SCHUR));
    PetscCall(PCFieldSplitSetSchurPre(pc,PC_FIELDSPLIT_SCHUR_PRE_SELF,NULL));
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
/*MC
   PCFIELDSPLIT - Preconditioner created by combining separate preconditioners for individual
                  fields or groups of fields. See [the users manual section on "Solving Block Matrices"](sec_block_matrices) for more details.

     To set options on the solvers for each block append `-fieldsplit_` to all the PC
        options database keys. For example, `-fieldsplit_pc_type ilu -fieldsplit_pc_factor_levels 1`

     To set the options on the solvers separate for each block call `PCFieldSplitGetSubKSP()`
         and set the options directly on the resulting `KSP` object

   Level: intermediate

   Options Database Keys:
+   -pc_fieldsplit_%d_fields <a,b,..> - indicates the fields to be used in the `%d`'th split
.   -pc_fieldsplit_default - automatically add any fields to additional splits that have not
                              been supplied explicitly by `-pc_fieldsplit_%d_fields`
.   -pc_fieldsplit_block_size <bs> - size of block that defines fields (i.e. there are bs fields)
.   -pc_fieldsplit_type <additive,multiplicative,symmetric_multiplicative,schur,gkb> - type of relaxation or factorization splitting
.   -pc_fieldsplit_schur_precondition <self,selfp,user,a11,full> - default is a11; see `PCFieldSplitSetSchurPre()`
.   -pc_fieldsplit_schur_fact_type <diag,lower,upper,full> - set factorization type when using `-pc_fieldsplit_type schur`; see `PCFieldSplitSetSchurFactType()`
-   -pc_fieldsplit_detect_saddle_point - automatically finds rows with zero diagonal and uses Schur complement with no preconditioner as the solver

     Options prefixes for inner solvers when using the Schur complement preconditioner are `-fieldsplit_0_` and `-fieldsplit_1_` .
     For all other solvers they are `-fieldsplit_%d_` for the `d`th field; use `-fieldsplit_` for all fields.
     The options prefix for the inner solver when using the Golub-Kahan biadiagonalization preconditioner is `-fieldsplit_0_`

   Notes:
    Use `PCFieldSplitSetFields()` to set fields defined by "strided" entries and `PCFieldSplitSetIS()`
     to define a field by an arbitrary collection of entries.

      If no fields are set the default is used. The fields are defined by entries strided by bs,
      beginning at 0 then 1, etc to bs-1. The block size can be set with `PCFieldSplitSetBlockSize()`,
      if this is not called the block size defaults to the blocksize of the second matrix passed
      to `KSPSetOperators()`/`PCSetOperators()`.

      For the Schur complement preconditioner if

      ```{math}
      J = \left[\begin{array}{cc} A_{00} & A_{01} \\ A_{10} & A_{11} \end{array}\right]
      ```

      the preconditioner using `full` factorization is logically
      ```{math}
      \left[\begin{array}{cc} I & -\text{ksp}(A_{00}) \\ 0 & I \end{array}\right] \left[\begin{array}{cc} \text{inv}(A_{00}) & 0 \\ 0 & \text{ksp}(S) \end{array}\right] \left[\begin{array}{cc} I & 0 \\ -A_{10} \text{ksp}(A_{00}) & I \end{array}\right]
      ```
     where the action of $\text{inv}(A_{00})$ is applied using the KSP solver with prefix `-fieldsplit_0_`.  $S$ is the Schur complement
     ```{math}
     S = A_{11} - A10 \text{ksp}(A_{00}) A_{01}
     ```
     which is usually dense and not stored explicitly.  The action of $\text{ksp}(S)$ is computed using the KSP solver with prefix `-fieldsplit_splitname_` (where `splitname` was given
     in providing the SECOND split or 1 if not given). For `PCFieldSplitGetSub\text{ksp}()` when field number is 0,
     it returns the KSP associated with `-fieldsplit_0_` while field number 1 gives `-fieldsplit_1_` KSP. By default
     $A_{11}$ is used to construct a preconditioner for $S$, use `PCFieldSplitSetSchurPre()` for all the possible ways to construct the preconditioner for $S$.

     The factorization type is set using `-pc_fieldsplit_schur_fact_type <diag, lower, upper, full>`. `full` is shown above,
     `diag` gives
      ```{math}
      \left[\begin{array}{cc} \text{inv}(A_{00}) & 0 \\  0 & -\text{ksp}(S) \end{array}\right]
      ```
     Note that, slightly counter intuitively, there is a negative in front of the $\text{ksp}(S)$  so that the preconditioner is positive definite. For SPD matrices $J$, the sign flip
     can be turned off with `PCFieldSplitSetSchurScale()` or by command line `-pc_fieldsplit_schur_scale 1.0`. The `lower` factorization is the inverse of
      ```{math}
      \left[\begin{array}{cc} A_{00} & 0 \\  A_{10} & S \end{array}\right]
      ```
     where the inverses of A_{00} and S are applied using KSPs. The upper factorization is the inverse of
      ```{math}
      \left[\begin{array}{cc} A_{00} & A_{01} \\  0 & S \end{array}\right]
      ```
     where again the inverses of $A_{00}$ and $S$ are applied using `KSP`s.

     If only one set of indices (one `IS`) is provided with `PCFieldSplitSetIS()` then the complement of that `IS`
     is used automatically for a second block.

     The fieldsplit preconditioner cannot currently be used with the BAIJ or SBAIJ data formats if the blocksize is larger than 1.
     Generally it should be used with the AIJ format.

     The forms of these preconditioners are closely related if not identical to forms derived as "Distributive Iterations", see,
     for example, page 294 in "Principles of Computational Fluid Dynamics" by Pieter Wesseling {cite}`Wesseling2009`. Note that one can also use `PCFIELDSPLIT`
     inside a smoother resulting in "Distributive Smoothers".

     References:

     See "A taxonomy and comparison of parallel block multi-level preconditioners for the incompressible Navier-Stokes equations" {cite}`elman2008tcp`.

     The Constrained Pressure Preconditioner (CPR) can be implemented using `PCCOMPOSITE` with `PCGALERKIN`. CPR first solves an $R A P$ subsystem, updates the
     residual on all variables (`PCCompositeSetType(pc,PC_COMPOSITE_MULTIPLICATIVE)`), and then applies a simple ILU like preconditioner on all the variables.

     The generalized Golub-Kahan bidiagonalization preconditioner (GKB) can be applied to symmetric $2 \times 2$ block matrices of the shape
     ```{math}
     \left[\begin{array}{cc} A_{00} & A_{01} \\ A_{01}' & 0 \end{array}\right]
     ```
     with $A_{00}$ positive semi-definite. The implementation follows {cite}`Arioli2013`. Therein, we choose $N := 1/\nu * I$ and the $(1,1)$-block of the matrix is modified to $H = _{A00} + \nu*A_{01}*A_{01}'$.
     A linear system $Hx = b$ has to be solved in each iteration of the GKB algorithm. This solver is chosen with the option prefix `-fieldsplit_0_`.

     ```{bibliography}
     :filter: docname in docnames
     ```

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PC`, `PCLSC`,
          `PCFieldSplitGetSubKSP()`, `PCFieldSplitSchurGetSubKSP()`, `PCFieldSplitSetFields()`,
          `PCFieldSplitSetType()`, `PCFieldSplitSetIS()`, `PCFieldSplitSetSchurPre()`, `PCFieldSplitSetSchurFactType()`,
          `MatSchurComplementSetAinvType()`, `PCFieldSplitSetSchurScale()`, `PCFieldSplitSetDetectSaddlePoint()`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_FieldSplit(PC pc)
{
  PC_FieldSplit  *jac;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(pc,&jac));

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
  jac->coordinates_set    = PETSC_FALSE;

  pc->data = (void*)jac;

  pc->ops->apply           = PCApply_FieldSplit;
  pc->ops->applytranspose  = PCApplyTranspose_FieldSplit;
  pc->ops->setup           = PCSetUp_FieldSplit;
  pc->ops->reset           = PCReset_FieldSplit;
  pc->ops->destroy         = PCDestroy_FieldSplit;
  pc->ops->setfromoptions  = PCSetFromOptions_FieldSplit;
  pc->ops->view            = PCView_FieldSplit;
  pc->ops->applyrichardson = NULL;

  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSchurGetSubKSP_C",PCFieldSplitSchurGetSubKSP_FieldSplit));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitGetSubKSP_C",PCFieldSplitGetSubKSP_FieldSplit));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetFields_C",PCFieldSplitSetFields_FieldSplit));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetIS_C",PCFieldSplitSetIS_FieldSplit));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetType_C",PCFieldSplitSetType_FieldSplit));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitSetBlockSize_C",PCFieldSplitSetBlockSize_FieldSplit));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFieldSplitRestrictIS_C",PCFieldSplitRestrictIS_FieldSplit));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",PCSetCoordinates_FieldSplit));
  PetscFunctionReturn(0);
}
