
#include <petsc-private/pcimpl.h>     /*I "petscpc.h" I*/
#include <petscdmcomposite.h>   /*I "petscdmcomposite.h" I*/

/*
  There is a nice discussion of block preconditioners in

[El08] A taxonomy and comparison of parallel block multi-level preconditioners for the incompressible Navier–Stokes equations
       Howard Elman, V.E. Howle, John Shadid, Robert Shuttleworth, Ray Tuminaro, Journal of Computational Physics 227 (2008) 1790--1808
       http://chess.cs.umd.edu/~elman/papers/tax.pdf
*/

const char *const PCFieldSplitSchurPreTypes[] = {"SELF","DIAG","USER","PCFieldSplitSchurPreType","PC_FIELDSPLIT_SCHUR_PRE_",0};
const char *const PCFieldSplitSchurFactTypes[] = {"DIAG","LOWER","UPPER","FULL","PCFieldSplitSchurFactType","PC_FIELDSPLIT_SCHUR_FACT_",0};

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
};

typedef struct {
  PCCompositeType                    type;
  PetscBool                          defaultsplit; /* Flag for a system with a set of 'k' scalar fields with the same layout (and bs = k) */
  PetscBool                          splitdefined; /* Flag is set after the splits have been defined, to prevent more splits from being added */
  PetscBool                          realdiagonal; /* Flag to use the diagonal blocks of mat preconditioned by pmat, instead of just pmat */
  PetscInt                           bs;           /* Block size for IS and Mat structures */
  PetscInt                           nsplits;      /* Number of field divisions defined */
  Vec                                *x,*y,w1,w2;
  Mat                                *mat;         /* The diagonal block for each split */
  Mat                                *pmat;        /* The preconditioning diagonal block for each split */
  Mat                                *Afield;      /* The rows of the matrix associated with each split */
  PetscBool                          issetup;
  /* Only used when Schur complement preconditioning is used */
  Mat                                B;            /* The (0,1) block */
  Mat                                C;            /* The (1,0) block */
  Mat                                schur;        /* The Schur complement S = A11 - A10 A00^{-1} A01, the KSP here, kspinner, is H_1 in [El08] */
  Mat                                schur_user;   /* User-provided preconditioning matrix for the Schur complement */
  PCFieldSplitSchurPreType           schurpre; /* Determines which preconditioning matrix is used for the Schur complement */
  PCFieldSplitSchurFactType          schurfactorization;
  KSP                                kspschur;     /* The solver for S */
  KSP                                kspupper;     /* The solver for A in the upper diagonal part of the factorization (H_2 in [El08]) */
  PC_FieldSplitLink                  head;
  PetscBool                          reset;         /* indicates PCReset() has been last called on this object, hack */
  PetscBool                          suboptionsset; /* Indicates that the KSPSetFromOptions() has been called on the sub-KSPs */
} PC_FieldSplit;

/*
    Notes: there is no particular reason that pmat, x, and y are stored as arrays in PC_FieldSplit instead of
   inside PC_FieldSplitLink, just historical. If you want to be able to add new fields after already using the
   PC you could change this.
*/

/* This helper is so that setting a user-provided preconditioning matrix is orthogonal to choosing to use it.  This way the
* application-provided FormJacobian can provide this matrix without interfering with the user's (command-line) choices. */
static Mat FieldSplitSchurPre(PC_FieldSplit *jac)
{
  switch (jac->schurpre) {
    case PC_FIELDSPLIT_SCHUR_PRE_SELF: return jac->schur;
    case PC_FIELDSPLIT_SCHUR_PRE_DIAG: return jac->pmat[1];
    case PC_FIELDSPLIT_SCHUR_PRE_USER: /* Use a user-provided matrix if it is given, otherwise diagonal block */
    default:
      return jac->schur_user ? jac->schur_user : jac->pmat[1];
  }
}


#undef __FUNCT__
#define __FUNCT__ "PCView_FieldSplit"
static PetscErrorCode PCView_FieldSplit(PC pc,PetscViewer viewer)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscInt          i,j;
  PC_FieldSplitLink ilink = jac->head;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (jac->bs > 0) {
      ierr = PetscViewerASCIIPrintf(viewer,"  FieldSplit with %s composition: total splits = %D, blocksize = %D\n",PCCompositeTypes[jac->type],jac->nsplits,jac->bs);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  FieldSplit with %s composition: total splits = %D\n",PCCompositeTypes[jac->type],jac->nsplits);CHKERRQ(ierr);
    }
    if (jac->realdiagonal) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using actual matrix for blocks rather than preconditioner matrix\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  Solver info for each split is in the following KSP objects:\n");CHKERRQ(ierr);
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
      ierr = KSPView(ilink->ksp,viewer);CHKERRQ(ierr);
      ilink = ilink->next;
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for PCFieldSplit",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCView_FieldSplit_Schur"
static PetscErrorCode PCView_FieldSplit_Schur(PC pc,PetscViewer viewer)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscInt          i,j;
  PC_FieldSplitLink ilink = jac->head;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (jac->bs > 0) {
      ierr = PetscViewerASCIIPrintf(viewer,"  FieldSplit with Schur preconditioner, blocksize = %D, factorization %s\n",jac->bs,PCFieldSplitSchurFactTypes[jac->schurfactorization]);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  FieldSplit with Schur preconditioner, factorization %s\n",PCFieldSplitSchurFactTypes[jac->schurfactorization]);CHKERRQ(ierr);
    }
    if (jac->realdiagonal) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using actual matrix for blocks rather than preconditioner matrix\n");CHKERRQ(ierr);
    }
    switch(jac->schurpre) {
    case PC_FIELDSPLIT_SCHUR_PRE_SELF:
      ierr = PetscViewerASCIIPrintf(viewer,"  Preconditioner for the Schur complement formed from S itself\n");CHKERRQ(ierr);break;
    case PC_FIELDSPLIT_SCHUR_PRE_DIAG:
      ierr = PetscViewerASCIIPrintf(viewer,"  Preconditioner for the Schur complement formed from the block diagonal part of A11\n");CHKERRQ(ierr);break;
    case PC_FIELDSPLIT_SCHUR_PRE_USER:
      if (jac->schur_user) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Preconditioner for the Schur complement formed from user provided matrix\n");CHKERRQ(ierr);
      } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  Preconditioner for the Schur complement formed from the block diagonal part of A11\n");CHKERRQ(ierr);
      }
      break;
    default:
      SETERRQ1(((PetscObject) pc)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid Schur preconditioning type: %d", jac->schurpre);
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
    ierr = PetscViewerASCIIPrintf(viewer,"KSP solver for A00 block \n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = KSPView(jac->head->ksp,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    if (jac->kspupper != jac->head->ksp) {
      ierr = PetscViewerASCIIPrintf(viewer,"KSP solver for upper A00 in upper triangular factor \n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = KSPView(jac->kspupper,viewer);CHKERRQ(ierr);
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
  } else {
    SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for PCFieldSplit",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitSetRuntimeSplits_Private"
/* Precondition: jac->bs is set to a meaningful value */
static PetscErrorCode PCFieldSplitSetRuntimeSplits_Private(PC pc)
{
  PetscErrorCode ierr;
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;
  PetscInt       i,nfields,*ifields,nfields_col,*ifields_col;
  PetscBool      flg,flg_col;
  char           optionname[128],splitname[8],optionname_col[128];

  PetscFunctionBegin;
  ierr = PetscMalloc(jac->bs*sizeof(PetscInt),&ifields);CHKERRQ(ierr);
  ierr = PetscMalloc(jac->bs*sizeof(PetscInt),&ifields_col);CHKERRQ(ierr);
  for (i=0,flg=PETSC_TRUE; ; i++) {
    ierr = PetscSNPrintf(splitname,sizeof(splitname),"%D",i);CHKERRQ(ierr);
    ierr = PetscSNPrintf(optionname,sizeof(optionname),"-pc_fieldsplit_%D_fields",i);CHKERRQ(ierr);
    ierr = PetscSNPrintf(optionname_col,sizeof(optionname_col),"-pc_fieldsplit_%D_fields_col",i);CHKERRQ(ierr);
    nfields = jac->bs;
    nfields_col = jac->bs;
    ierr    = PetscOptionsGetIntArray(((PetscObject)pc)->prefix,optionname,ifields,&nfields,&flg);CHKERRQ(ierr);
    ierr    = PetscOptionsGetIntArray(((PetscObject)pc)->prefix,optionname_col,ifields_col,&nfields_col,&flg_col);CHKERRQ(ierr);
    if (!flg) break;
    else if (flg && !flg_col) {
      if (!nfields) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot list zero fields");
      ierr = PCFieldSplitSetFields(pc,splitname,nfields,ifields,ifields);CHKERRQ(ierr);
    }
    else {
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

#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitSetDefaults"
static PetscErrorCode PCFieldSplitSetDefaults(PC pc)
{
  PC_FieldSplit     *jac  = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilink = jac->head;
  PetscBool         fieldsplit_default = PETSC_FALSE,stokes = PETSC_FALSE;
  PetscInt          i;

  PetscFunctionBegin;
  /*
   Kinda messy, but at least this now uses DMCreateFieldDecomposition() even with jac->reset.
   Should probably be rewritten.
   */
  if (!ilink || jac->reset) {
    ierr = PetscOptionsGetBool(((PetscObject)pc)->prefix,"-pc_fieldsplit_detect_saddle_point",&stokes,PETSC_NULL);CHKERRQ(ierr);
    if (pc->dm && !stokes) {
      PetscInt     numFields, f, i, j;
      char       **fieldNames;
      IS          *fields;
      DM          *dms;
      DM           subdm[128];
      PetscBool    flg;

      ierr = DMCreateFieldDecomposition(pc->dm, &numFields, &fieldNames, &fields, &dms);CHKERRQ(ierr);
      /* Allow the user to prescribe the splits */
      for (i = 0, flg = PETSC_TRUE; ; i++) {
        PetscInt ifields[128];
        IS       compField;
        char     optionname[128], splitname[8];
        PetscInt nfields = numFields;

        ierr = PetscSNPrintf(optionname, sizeof(optionname), "-pc_fieldsplit_%D_fields", i);CHKERRQ(ierr);
        ierr = PetscOptionsGetIntArray(((PetscObject) pc)->prefix, optionname, ifields, &nfields, &flg);CHKERRQ(ierr);
        if (!flg) break;
        if (numFields > 128) SETERRQ1(((PetscObject) pc)->comm,PETSC_ERR_SUP,"Cannot currently support %d > 128 fields", numFields);
        ierr = DMCreateSubDM(pc->dm, nfields, ifields, &compField, &subdm[i]);CHKERRQ(ierr);
        if (nfields == 1) {
          ierr = PCFieldSplitSetIS(pc, fieldNames[ifields[0]], compField);CHKERRQ(ierr);
          /* ierr = PetscPrintf(((PetscObject) pc)->comm, "%s Field Indices:", fieldNames[ifields[0]]);CHKERRQ(ierr);
             ierr = ISView(compField, PETSC_NULL);CHKERRQ(ierr); */
        } else {
          ierr = PetscSNPrintf(splitname, sizeof(splitname), "%D", i);CHKERRQ(ierr);
          ierr = PCFieldSplitSetIS(pc, splitname, compField);CHKERRQ(ierr);
          /* ierr = PetscPrintf(((PetscObject) pc)->comm, "%s Field Indices:", splitname);CHKERRQ(ierr);
             ierr = ISView(compField, PETSC_NULL);CHKERRQ(ierr); */
        }
        ierr = ISDestroy(&compField);CHKERRQ(ierr);
        for (j = 0; j < nfields; ++j) {
          f = ifields[j];
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
        ierr = PetscMalloc(i * sizeof(DM), &dms);CHKERRQ(ierr);
        for (j = 0; j < i; ++j) {
          dms[j] = subdm[j];
        }
      }
      ierr = PetscFree(fieldNames);CHKERRQ(ierr);
      ierr = PetscFree(fields);CHKERRQ(ierr);
      if (dms) {
        ierr = PetscInfo(pc, "Setting up physics based fieldsplit preconditioner using the embedded DM\n");CHKERRQ(ierr);
        for (ilink = jac->head, i = 0; ilink; ilink = ilink->next, ++i) {
          const char *prefix;
          ierr = PetscObjectGetOptionsPrefix((PetscObject)(ilink->ksp),&prefix); CHKERRQ(ierr);
          ierr = PetscObjectSetOptionsPrefix((PetscObject)(dms[i]), prefix);     CHKERRQ(ierr);
          ierr = KSPSetDM(ilink->ksp, dms[i]);CHKERRQ(ierr);
          ierr = KSPSetDMActive(ilink->ksp, PETSC_FALSE);CHKERRQ(ierr);
          ierr = PetscObjectIncrementTabLevel((PetscObject)dms[i],(PetscObject)ilink->ksp,0); CHKERRQ(ierr);
          ierr = DMDestroy(&dms[i]); CHKERRQ(ierr);
        }
        ierr = PetscFree(dms);CHKERRQ(ierr);
      }
    } else {
      if (jac->bs <= 0) {
        if (pc->pmat) {
          ierr   = MatGetBlockSize(pc->pmat,&jac->bs);CHKERRQ(ierr);
        } else {
          jac->bs = 1;
        }
      }

      ierr = PetscOptionsGetBool(((PetscObject)pc)->prefix,"-pc_fieldsplit_default",&fieldsplit_default,PETSC_NULL);CHKERRQ(ierr);
      if (stokes) {
        IS       zerodiags,rest;
        PetscInt nmin,nmax;

        ierr = MatGetOwnershipRange(pc->mat,&nmin,&nmax);CHKERRQ(ierr);
        ierr = MatFindZeroDiagonals(pc->mat,&zerodiags);CHKERRQ(ierr);
        ierr = ISComplement(zerodiags,nmin,nmax,&rest);CHKERRQ(ierr);
        if (jac->reset) {
          jac->head->is       = rest;
          jac->head->next->is = zerodiags;
        }
        else {
          ierr = PCFieldSplitSetIS(pc,"0",rest);CHKERRQ(ierr);
          ierr = PCFieldSplitSetIS(pc,"1",zerodiags);CHKERRQ(ierr);
        }
        ierr = ISDestroy(&zerodiags);CHKERRQ(ierr);
        ierr = ISDestroy(&rest);CHKERRQ(ierr);
      } else {
        if (jac->reset)
          SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Cases not yet handled when PCReset() was used");
        if (!fieldsplit_default) {
          /* Allow user to set fields from command line,  if bs was known at the time of PCSetFromOptions_FieldSplit()
           then it is set there. This is not ideal because we should only have options set in XXSetFromOptions(). */
          ierr = PCFieldSplitSetRuntimeSplits_Private(pc);CHKERRQ(ierr);
          if (jac->splitdefined) {ierr = PetscInfo(pc,"Splits defined using the options database\n");CHKERRQ(ierr);}
        }
        if (fieldsplit_default || !jac->splitdefined) {
          ierr = PetscInfo(pc,"Using default splitting of fields\n");CHKERRQ(ierr);
          for (i=0; i<jac->bs; i++) {
            char splitname[8];
            ierr = PetscSNPrintf(splitname,sizeof(splitname),"%D",i);CHKERRQ(ierr);
            ierr = PCFieldSplitSetFields(pc,splitname,1,&i,&i);CHKERRQ(ierr);
          }
          jac->defaultsplit = PETSC_TRUE;
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
    } else SETERRQ(((PetscObject) pc)->comm,PETSC_ERR_SUP,"Must provide at least two sets of fields to PCFieldSplit()");
  }


  if (jac->nsplits < 2) SETERRQ1(((PetscObject) pc)->comm,PETSC_ERR_PLIB,"Unhandled case, must have at least two fields, not %d", jac->nsplits);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscOptionsFindPairPrefix_Private(const char pre[], const char name[], char *value[], PetscBool *flg);

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_FieldSplit"
static PetscErrorCode PCSetUp_FieldSplit(PC pc)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilink;
  PetscInt          i,nsplit;
  MatStructure      flag = pc->flag;
  PetscBool         sorted, sorted_col;

  PetscFunctionBegin;
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
        ierr = ISCreateStride(((PetscObject)pc)->comm,nslots,rstart+i,nsplit,&ilink->is);CHKERRQ(ierr);
        ierr = ISDuplicate(ilink->is,&ilink->is_col);CHKERRQ(ierr);
      } else if (!ilink->is) {
        if (ilink->nfields > 1) {
          PetscInt   *ii,*jj,j,k,nfields = ilink->nfields,*fields = ilink->fields,*fields_col = ilink->fields_col;
          ierr = PetscMalloc(ilink->nfields*nslots*sizeof(PetscInt),&ii);CHKERRQ(ierr);
          ierr = PetscMalloc(ilink->nfields*nslots*sizeof(PetscInt),&jj);CHKERRQ(ierr);
          for (j=0; j<nslots; j++) {
            for (k=0; k<nfields; k++) {
              ii[nfields*j + k] = rstart + bs*j + fields[k];
              jj[nfields*j + k] = rstart + bs*j + fields_col[k];
            }
          }
          ierr = ISCreateGeneral(((PetscObject)pc)->comm,nslots*nfields,ii,PETSC_OWN_POINTER,&ilink->is);CHKERRQ(ierr);
          ierr = ISCreateGeneral(((PetscObject)pc)->comm,nslots*nfields,jj,PETSC_OWN_POINTER,&ilink->is_col);CHKERRQ(ierr);
        } else {
          ierr = ISCreateStride(((PetscObject)pc)->comm,nslots,rstart+ilink->fields[0],bs,&ilink->is);CHKERRQ(ierr);
          ierr = ISCreateStride(((PetscObject)pc)->comm,nslots,rstart+ilink->fields_col[0],bs,&ilink->is_col);CHKERRQ(ierr);
       }
      }
      ierr = ISSorted(ilink->is,&sorted);CHKERRQ(ierr);
      if (ilink->is_col) { ierr = ISSorted(ilink->is_col,&sorted_col);CHKERRQ(ierr); }
      if (!sorted || !sorted_col) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Fields must be sorted when creating split");
      ilink = ilink->next;
    }
  }

  ilink  = jac->head;
  if (!jac->pmat) {
    Vec xtmp;

    ierr = MatGetVecs(pc->pmat,&xtmp,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscMalloc(nsplit*sizeof(Mat),&jac->pmat);CHKERRQ(ierr);
    ierr = PetscMalloc2(nsplit,Vec,&jac->x,nsplit,Vec,&jac->y);CHKERRQ(ierr);
    for (i=0; i<nsplit; i++) {
      MatNullSpace sp;

      /* Check for preconditioning matrix attached to IS */
      ierr = PetscObjectQuery((PetscObject) ilink->is, "pmat", (PetscObject *) &jac->pmat[i]);CHKERRQ(ierr);
      if (jac->pmat[i]) {
        ierr = PetscObjectReference((PetscObject) jac->pmat[i]);CHKERRQ(ierr);
        if (jac->type == PC_COMPOSITE_SCHUR) {
          jac->schur_user = jac->pmat[i];
          ierr = PetscObjectReference((PetscObject) jac->schur_user);CHKERRQ(ierr);
        }
      } else {
        ierr = MatGetSubMatrix(pc->pmat,ilink->is,ilink->is_col,MAT_INITIAL_MATRIX,&jac->pmat[i]);CHKERRQ(ierr);
      }
      /* create work vectors for each split */
      ierr = MatGetVecs(jac->pmat[i],&jac->x[i],&jac->y[i]);CHKERRQ(ierr);
      ilink->x = jac->x[i]; ilink->y = jac->y[i]; ilink->z = PETSC_NULL;
      /* compute scatter contexts needed by multiplicative versions and non-default splits */
      ierr = VecScatterCreate(xtmp,ilink->is,jac->x[i],PETSC_NULL,&ilink->sctx);CHKERRQ(ierr);
      /* Check for null space attached to IS */
      ierr = PetscObjectQuery((PetscObject) ilink->is, "nullspace", (PetscObject *) &sp);CHKERRQ(ierr);
      if (sp) {
        ierr  = MatSetNullSpace(jac->pmat[i], sp);CHKERRQ(ierr);
      }
      ierr = PetscObjectQuery((PetscObject) ilink->is, "nearnullspace", (PetscObject *) &sp);CHKERRQ(ierr);
      if (sp) {
        ierr  = MatSetNearNullSpace(jac->pmat[i], sp);CHKERRQ(ierr);
      }
      ilink = ilink->next;
    }
    ierr = VecDestroy(&xtmp);CHKERRQ(ierr);
  } else {
    for (i=0; i<nsplit; i++) {
      Mat pmat;

      /* Check for preconditioning matrix attached to IS */
      ierr = PetscObjectQuery((PetscObject) ilink->is, "pmat", (PetscObject *) &pmat);CHKERRQ(ierr);
      if (!pmat) {
        ierr = MatGetSubMatrix(pc->pmat,ilink->is,ilink->is_col,MAT_REUSE_MATRIX,&jac->pmat[i]);CHKERRQ(ierr);
      }
      ilink = ilink->next;
    }
  }
  if (jac->realdiagonal) {
    ilink = jac->head;
    if (!jac->mat) {
      ierr = PetscMalloc(nsplit*sizeof(Mat),&jac->mat);CHKERRQ(ierr);
      for (i=0; i<nsplit; i++) {
        ierr = MatGetSubMatrix(pc->mat,ilink->is,ilink->is_col,MAT_INITIAL_MATRIX,&jac->mat[i]);CHKERRQ(ierr);
        ilink = ilink->next;
      }
    } else {
      for (i=0; i<nsplit; i++) {
        if (jac->mat[i]) {ierr = MatGetSubMatrix(pc->mat,ilink->is,ilink->is_col,MAT_REUSE_MATRIX,&jac->mat[i]);CHKERRQ(ierr);}
        ilink = ilink->next;
      }
    }
  } else {
    jac->mat = jac->pmat;
  }

  if (jac->type != PC_COMPOSITE_ADDITIVE  && jac->type != PC_COMPOSITE_SCHUR) {
    /* extract the rows of the matrix associated with each field: used for efficient computation of residual inside algorithm */
    ilink  = jac->head;
    if (!jac->Afield) {
      ierr = PetscMalloc(nsplit*sizeof(Mat),&jac->Afield);CHKERRQ(ierr);
      for (i=0; i<nsplit; i++) {
        ierr = MatGetSubMatrix(pc->mat,ilink->is,PETSC_NULL,MAT_INITIAL_MATRIX,&jac->Afield[i]);CHKERRQ(ierr);
        ilink = ilink->next;
      }
    } else {
      for (i=0; i<nsplit; i++) {
        ierr = MatGetSubMatrix(pc->mat,ilink->is,PETSC_NULL,MAT_REUSE_MATRIX,&jac->Afield[i]);CHKERRQ(ierr);
        ilink = ilink->next;
      }
    }
  }

  if (jac->type == PC_COMPOSITE_SCHUR) {
    IS       ccis;
    PetscInt rstart,rend;
    char     lscname[256];
    PetscObject LSC_L;
    if (nsplit != 2) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_INCOMP,"To use Schur complement preconditioner you must have exactly 2 fields");

    /* When extracting off-diagonal submatrices, we take complements from this range */
    ierr  = MatGetOwnershipRangeColumn(pc->mat,&rstart,&rend);CHKERRQ(ierr);

    /* need to handle case when one is resetting up the preconditioner */
    if (jac->schur) {
      KSP kspA = jac->head->ksp, kspInner = PETSC_NULL, kspUpper = jac->kspupper;

      ierr = MatSchurComplementGetKSP(jac->schur, &kspInner);CHKERRQ(ierr);
      ilink = jac->head;
      ierr  = ISComplement(ilink->is_col,rstart,rend,&ccis);CHKERRQ(ierr);
      ierr  = MatGetSubMatrix(pc->mat,ilink->is,ccis,MAT_REUSE_MATRIX,&jac->B);CHKERRQ(ierr);
      ierr  = ISDestroy(&ccis);CHKERRQ(ierr);
      ilink = ilink->next;
      ierr  = ISComplement(ilink->is_col,rstart,rend,&ccis);CHKERRQ(ierr);
      ierr  = MatGetSubMatrix(pc->mat,ilink->is,ccis,MAT_REUSE_MATRIX,&jac->C);CHKERRQ(ierr);
      ierr  = ISDestroy(&ccis);CHKERRQ(ierr);
      ierr  = MatSchurComplementUpdate(jac->schur,jac->mat[0],jac->pmat[0],jac->B,jac->C,jac->mat[1],pc->flag);CHKERRQ(ierr);
      if (kspA != kspInner) {
        ierr = KSPSetOperators(kspA,jac->mat[0],jac->pmat[0],pc->flag);CHKERRQ(ierr);
      }
      if (kspUpper != kspA) {
        ierr = KSPSetOperators(kspUpper,jac->mat[0],jac->pmat[0],pc->flag);CHKERRQ(ierr);
      }
      ierr  = KSPSetOperators(jac->kspschur,jac->schur,FieldSplitSchurPre(jac),pc->flag);CHKERRQ(ierr);
     } else {
      KSP ksp;
      const char  *Dprefix;
      char schurprefix[256];
      char schurtestoption[256];
      MatNullSpace sp;
      PetscBool    flg;

      /* extract the A01 and A10 matrices */
      ilink = jac->head;
      ierr  = ISComplement(ilink->is_col,rstart,rend,&ccis);CHKERRQ(ierr);
      ierr  = MatGetSubMatrix(pc->mat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->B);CHKERRQ(ierr);
      ierr  = ISDestroy(&ccis);CHKERRQ(ierr);
      ilink = ilink->next;
      ierr = ISComplement(ilink->is_col,rstart,rend,&ccis);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(pc->mat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->C);CHKERRQ(ierr);
      ierr = ISDestroy(&ccis);CHKERRQ(ierr);

      /* Use mat[0] (diagonal block of the real matrix) preconditioned by pmat[0] to define Schur complement */
      ierr = MatCreate(((PetscObject)jac->mat[0])->comm,&jac->schur);CHKERRQ(ierr);
      ierr = MatSetType(jac->schur,MATSCHURCOMPLEMENT);CHKERRQ(ierr);
      ierr = MatSchurComplementGetKSP(jac->schur, &ksp);CHKERRQ(ierr);
      ierr = PetscSNPrintf(schurprefix, sizeof(schurprefix), "%sfieldsplit_%s_inner_", ((PetscObject)pc)->prefix ? ((PetscObject)pc)->prefix : "", ilink->splitname);CHKERRQ(ierr);
      /* Indent this deeper to emphasize the "inner" nature of this solver. */
      ierr = PetscObjectIncrementTabLevel((PetscObject)ksp, (PetscObject) pc, 2);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(ksp, schurprefix);CHKERRQ(ierr);
      ierr = MatSchurComplementSet(jac->schur,jac->mat[0],jac->pmat[0],jac->B,jac->C,jac->mat[1]);CHKERRQ(ierr);

      ierr = MatGetNullSpace(jac->pmat[1], &sp);CHKERRQ(ierr);
      if (sp) {
        ierr  = MatSetNullSpace(jac->schur, sp);CHKERRQ(ierr);
      }

      ierr = PetscSNPrintf(schurtestoption, sizeof(schurtestoption), "-fieldsplit_%s_inner_", ilink->splitname);CHKERRQ(ierr);
      ierr = PetscOptionsFindPairPrefix_Private(((PetscObject)pc)->prefix, schurtestoption, PETSC_NULL, &flg);CHKERRQ(ierr);
      if (flg) {
        DM dmInner;

        /* Set DM for new solver */
        ierr = KSPGetDM(jac->head->ksp, &dmInner);CHKERRQ(ierr);
        ierr = KSPSetDM(ksp, dmInner);CHKERRQ(ierr);
        ierr = KSPSetDMActive(ksp, PETSC_FALSE);CHKERRQ(ierr);
        ierr = KSPSetOperators(jac->head->ksp,jac->mat[0],jac->pmat[0],flag);CHKERRQ(ierr);
        ierr = KSPSetFromOptions(jac->head->ksp);CHKERRQ(ierr);
      } else {
        ierr = MatSchurComplementSetKSP(jac->schur, jac->head->ksp);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject) jac->head->ksp);CHKERRQ(ierr);
      }
      ierr  = MatSetFromOptions(jac->schur);CHKERRQ(ierr);

      ierr = PetscSNPrintf(schurtestoption, sizeof(schurtestoption), "-fieldsplit_%s_upper_", ilink->splitname);CHKERRQ(ierr);
      ierr = PetscOptionsFindPairPrefix_Private(((PetscObject)pc)->prefix, schurtestoption, PETSC_NULL, &flg);CHKERRQ(ierr);
      if (flg) {
        DM dmInner;

        ierr = PetscSNPrintf(schurprefix, sizeof(schurprefix), "%sfieldsplit_%s_upper_", ((PetscObject)pc)->prefix ? ((PetscObject)pc)->prefix : "", ilink->splitname);CHKERRQ(ierr);
        ierr = KSPCreate(((PetscObject) pc)->comm, &jac->kspupper);CHKERRQ(ierr);
        ierr = KSPSetOptionsPrefix(jac->kspupper, schurprefix);CHKERRQ(ierr);
        ierr = KSPGetDM(jac->head->ksp, &dmInner);CHKERRQ(ierr);
        ierr = KSPSetDM(jac->kspupper, dmInner);CHKERRQ(ierr);
        ierr = KSPSetDMActive(jac->kspupper, PETSC_FALSE);CHKERRQ(ierr);
        ierr = KSPSetFromOptions(jac->kspupper);CHKERRQ(ierr);
        ierr = KSPSetOperators(jac->kspupper,jac->mat[0],jac->pmat[0],flag);CHKERRQ(ierr);
        ierr = VecDuplicate(jac->head->x, &jac->head->z);CHKERRQ(ierr);
      } else {
        jac->kspupper = jac->head->ksp;
        ierr = PetscObjectReference((PetscObject) jac->head->ksp);CHKERRQ(ierr);
      }

      ierr  = KSPCreate(((PetscObject)pc)->comm,&jac->kspschur);               CHKERRQ(ierr);
      ierr  = PetscLogObjectParent((PetscObject)pc,(PetscObject)jac->kspschur);CHKERRQ(ierr);
      ierr  = PetscObjectIncrementTabLevel((PetscObject)jac->kspschur,(PetscObject)pc,1);           CHKERRQ(ierr);
      ierr  = KSPSetOperators(jac->kspschur,jac->schur,FieldSplitSchurPre(jac),DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      if (jac->schurpre == PC_FIELDSPLIT_SCHUR_PRE_SELF) {
        PC pcschur;
        ierr = KSPGetPC(jac->kspschur,&pcschur);CHKERRQ(ierr);
        ierr = PCSetType(pcschur,PCNONE);CHKERRQ(ierr);
        /* Note: This is bad if there exist preconditioners for MATSCHURCOMPLEMENT */
      }
      ierr  = KSPGetOptionsPrefix(jac->head->next->ksp, &Dprefix); CHKERRQ(ierr);
      ierr  = KSPSetOptionsPrefix(jac->kspschur,         Dprefix); CHKERRQ(ierr);
      /* really want setfromoptions called in PCSetFromOptions_FieldSplit(), but it is not ready yet */
      /* need to call this every time, since the jac->kspschur is freshly created, otherwise its options never get set */
      ierr = KSPSetFromOptions(jac->kspschur);CHKERRQ(ierr);
    }

    /* HACK: special support to forward L and Lp matrices that might be used by PCLSC */
    ierr = PetscSNPrintf(lscname,sizeof(lscname),"%s_LSC_L",ilink->splitname);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)pc->mat,lscname,(PetscObject*)&LSC_L);CHKERRQ(ierr);
    if (!LSC_L) {ierr = PetscObjectQuery((PetscObject)pc->pmat,lscname,(PetscObject*)&LSC_L);CHKERRQ(ierr);}
    if (LSC_L) {ierr = PetscObjectCompose((PetscObject)jac->schur,"LSC_L",(PetscObject)LSC_L);CHKERRQ(ierr);}
    ierr = PetscSNPrintf(lscname,sizeof(lscname),"%s_LSC_Lp",ilink->splitname);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)pc->pmat,lscname,(PetscObject*)&LSC_L);CHKERRQ(ierr);
    if (!LSC_L) {ierr = PetscObjectQuery((PetscObject)pc->mat,lscname,(PetscObject*)&LSC_L);CHKERRQ(ierr);}
    if (LSC_L) {ierr = PetscObjectCompose((PetscObject)jac->schur,"LSC_Lp",(PetscObject)LSC_L);CHKERRQ(ierr);}
  } else {
    /* set up the individual splits' PCs */
    i    = 0;
    ilink = jac->head;
    while (ilink) {
      ierr = KSPSetOperators(ilink->ksp,jac->mat[i],jac->pmat[i],flag);CHKERRQ(ierr);
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
     KSPSolve(ilink->ksp,ilink->x,ilink->y) || \
     VecScatterBegin(ilink->sctx,ilink->y,yy,ADD_VALUES,SCATTER_REVERSE) || \
     VecScatterEnd(ilink->sctx,ilink->y,yy,ADD_VALUES,SCATTER_REVERSE))

#undef __FUNCT__
#define __FUNCT__ "PCApply_FieldSplit_Schur"
static PetscErrorCode PCApply_FieldSplit_Schur(PC pc,Vec x,Vec y)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilinkA = jac->head, ilinkD = ilinkA->next;
  KSP               kspA = ilinkA->ksp, kspLower = kspA, kspUpper = jac->kspupper;

  PetscFunctionBegin;
  switch (jac->schurfactorization) {
    case PC_FIELDSPLIT_SCHUR_FACT_DIAG:
      /* [A00 0; 0 -S], positive definite, suitable for MINRES */
      ierr = VecScatterBegin(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterBegin(ilinkD->sctx,x,ilinkD->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = KSPSolve(kspA,ilinkA->x,ilinkA->y);CHKERRQ(ierr);
      ierr = VecScatterBegin(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(ilinkD->sctx,x,ilinkD->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = KSPSolve(jac->kspschur,ilinkD->x,ilinkD->y);CHKERRQ(ierr);
      ierr = VecScale(ilinkD->y,-1.);CHKERRQ(ierr);
      ierr = VecScatterBegin(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      break;
    case PC_FIELDSPLIT_SCHUR_FACT_LOWER:
      /* [A00 0; A10 S], suitable for left preconditioning */
      ierr = VecScatterBegin(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = KSPSolve(kspA,ilinkA->x,ilinkA->y);CHKERRQ(ierr);
      ierr = MatMult(jac->C,ilinkA->y,ilinkD->x);CHKERRQ(ierr);
      ierr = VecScale(ilinkD->x,-1.);CHKERRQ(ierr);
      ierr = VecScatterBegin(ilinkD->sctx,x,ilinkD->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterBegin(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(ilinkD->sctx,x,ilinkD->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = KSPSolve(jac->kspschur,ilinkD->x,ilinkD->y);CHKERRQ(ierr);
      ierr = VecScatterBegin(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      break;
    case PC_FIELDSPLIT_SCHUR_FACT_UPPER:
      /* [A00 A01; 0 S], suitable for right preconditioning */
      ierr = VecScatterBegin(ilinkD->sctx,x,ilinkD->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(ilinkD->sctx,x,ilinkD->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = KSPSolve(jac->kspschur,ilinkD->x,ilinkD->y);CHKERRQ(ierr);
      ierr = MatMult(jac->B,ilinkD->y,ilinkA->x);CHKERRQ(ierr);
      ierr = VecScale(ilinkA->x,-1.);CHKERRQ(ierr);
      ierr = VecScatterBegin(ilinkA->sctx,x,ilinkA->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterBegin(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(ilinkA->sctx,x,ilinkA->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = KSPSolve(kspA,ilinkA->x,ilinkA->y);CHKERRQ(ierr);
      ierr = VecScatterBegin(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      break;
    case PC_FIELDSPLIT_SCHUR_FACT_FULL:
      /* [1 0; A10 A00^{-1} 1] [A00 0; 0 S] [1 A00^{-1}A01; 0 1], an exact solve if applied exactly, needs one extra solve with A */
      ierr = VecScatterBegin(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = KSPSolve(kspLower,ilinkA->x,ilinkA->y);CHKERRQ(ierr);
      ierr = MatMult(jac->C,ilinkA->y,ilinkD->x);CHKERRQ(ierr);
      ierr = VecScale(ilinkD->x,-1.0);CHKERRQ(ierr);
      ierr = VecScatterBegin(ilinkD->sctx,x,ilinkD->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(ilinkD->sctx,x,ilinkD->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

      ierr = KSPSolve(jac->kspschur,ilinkD->x,ilinkD->y);CHKERRQ(ierr);
      ierr = VecScatterBegin(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

      if (kspUpper == kspA) {
        ierr = MatMult(jac->B,ilinkD->y,ilinkA->y);CHKERRQ(ierr);
        ierr = VecAXPY(ilinkA->x,-1.0,ilinkA->y);CHKERRQ(ierr);
        ierr = KSPSolve(kspA,ilinkA->x,ilinkA->y);CHKERRQ(ierr);
      } else {
        ierr = KSPSolve(kspA,ilinkA->x,ilinkA->y);CHKERRQ(ierr);
        ierr = MatMult(jac->B,ilinkD->y,ilinkA->x);CHKERRQ(ierr);
        ierr = KSPSolve(kspUpper,ilinkA->x,ilinkA->z);CHKERRQ(ierr);
        ierr = VecAXPY(ilinkA->y,-1.0,ilinkA->z);CHKERRQ(ierr);
      }
      ierr = VecScatterBegin(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_FieldSplit"
static PetscErrorCode PCApply_FieldSplit(PC pc,Vec x,Vec y)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilink = jac->head;
  PetscInt          cnt,bs;

  PetscFunctionBegin;
  x->map->bs = jac->bs;
  y->map->bs = jac->bs;
  CHKMEMQ;
  if (jac->type == PC_COMPOSITE_ADDITIVE) {
    if (jac->defaultsplit) {
      ierr = VecGetBlockSize(x,&bs);CHKERRQ(ierr);
      if (jac->bs > 0 && bs != jac->bs) SETERRQ2(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Blocksize of x vector %D does not match fieldsplit blocksize %D",bs,jac->bs);
      ierr = VecGetBlockSize(y,&bs);CHKERRQ(ierr);
      if (jac->bs > 0 && bs != jac->bs) SETERRQ2(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Blocksize of y vector %D does not match fieldsplit blocksize %D",bs,jac->bs);
      ierr = VecStrideGatherAll(x,jac->x,INSERT_VALUES);CHKERRQ(ierr);
      while (ilink) {
        ierr = KSPSolve(ilink->ksp,ilink->x,ilink->y);CHKERRQ(ierr);
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
  } else if (jac->type == PC_COMPOSITE_MULTIPLICATIVE || jac->type == PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE) {
    if (!jac->w1) {
      ierr = VecDuplicate(x,&jac->w1);CHKERRQ(ierr);
      ierr = VecDuplicate(x,&jac->w2);CHKERRQ(ierr);
    }
    ierr = VecSet(y,0.0);CHKERRQ(ierr);
    ierr = FieldSplitSplitSolveAdd(ilink,x,y);CHKERRQ(ierr);
    cnt = 1;
    while (ilink->next) {
      ilink = ilink->next;
      /* compute the residual only over the part of the vector needed */
      ierr = MatMult(jac->Afield[cnt++],y,ilink->x);CHKERRQ(ierr);
      ierr = VecScale(ilink->x,-1.0);CHKERRQ(ierr);
      ierr = VecScatterBegin(ilink->sctx,x,ilink->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(ilink->sctx,x,ilink->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = KSPSolve(ilink->ksp,ilink->x,ilink->y);CHKERRQ(ierr);
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
        ierr = KSPSolve(ilink->ksp,ilink->x,ilink->y);CHKERRQ(ierr);
        ierr = VecScatterBegin(ilink->sctx,ilink->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        ierr = VecScatterEnd(ilink->sctx,ilink->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      }
    }
  } else SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Unsupported or unknown composition",(int) jac->type);
  CHKMEMQ;
  PetscFunctionReturn(0);
}

#define FieldSplitSplitSolveAddTranspose(ilink,xx,yy) \
    (VecScatterBegin(ilink->sctx,xx,ilink->y,INSERT_VALUES,SCATTER_FORWARD) || \
     VecScatterEnd(ilink->sctx,xx,ilink->y,INSERT_VALUES,SCATTER_FORWARD) || \
     KSPSolveTranspose(ilink->ksp,ilink->y,ilink->x) || \
     VecScatterBegin(ilink->sctx,ilink->x,yy,ADD_VALUES,SCATTER_REVERSE) || \
     VecScatterEnd(ilink->sctx,ilink->x,yy,ADD_VALUES,SCATTER_REVERSE))

#undef __FUNCT__
#define __FUNCT__ "PCApplyTranspose_FieldSplit"
static PetscErrorCode PCApplyTranspose_FieldSplit(PC pc,Vec x,Vec y)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilink = jac->head;
  PetscInt          bs;

  PetscFunctionBegin;
  CHKMEMQ;
  if (jac->type == PC_COMPOSITE_ADDITIVE) {
    if (jac->defaultsplit) {
      ierr = VecGetBlockSize(x,&bs);CHKERRQ(ierr);
      if (jac->bs > 0 && bs != jac->bs) SETERRQ2(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Blocksize of x vector %D does not match fieldsplit blocksize %D",bs,jac->bs);
      ierr = VecGetBlockSize(y,&bs);CHKERRQ(ierr);
      if (jac->bs > 0 && bs != jac->bs) SETERRQ2(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Blocksize of y vector %D does not match fieldsplit blocksize %D",bs,jac->bs);
      ierr = VecStrideGatherAll(x,jac->x,INSERT_VALUES);CHKERRQ(ierr);
      while (ilink) {
	ierr = KSPSolveTranspose(ilink->ksp,ilink->x,ilink->y);CHKERRQ(ierr);
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
  CHKMEMQ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCReset_FieldSplit"
static PetscErrorCode PCReset_FieldSplit(PC pc)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilink = jac->head,next;

  PetscFunctionBegin;
  while (ilink) {
    ierr = KSPReset(ilink->ksp);CHKERRQ(ierr);
    ierr = VecDestroy(&ilink->x);CHKERRQ(ierr);
    ierr = VecDestroy(&ilink->y);CHKERRQ(ierr);
    ierr = VecDestroy(&ilink->z);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&ilink->sctx);CHKERRQ(ierr);
    ierr = ISDestroy(&ilink->is);CHKERRQ(ierr);
    ierr = ISDestroy(&ilink->is_col);CHKERRQ(ierr);
    next = ilink->next;
    ilink = next;
  }
  ierr = PetscFree2(jac->x,jac->y);CHKERRQ(ierr);
  if (jac->mat && jac->mat != jac->pmat) {
    ierr = MatDestroyMatrices(jac->nsplits,&jac->mat);CHKERRQ(ierr);
  } else if (jac->mat) {
    jac->mat = PETSC_NULL;
  }
  if (jac->pmat) {ierr = MatDestroyMatrices(jac->nsplits,&jac->pmat);CHKERRQ(ierr);}
  if (jac->Afield) {ierr = MatDestroyMatrices(jac->nsplits,&jac->Afield);CHKERRQ(ierr);}
  ierr = VecDestroy(&jac->w1);CHKERRQ(ierr);
  ierr = VecDestroy(&jac->w2);CHKERRQ(ierr);
  ierr = MatDestroy(&jac->schur);CHKERRQ(ierr);
  ierr = MatDestroy(&jac->schur_user);CHKERRQ(ierr);
  ierr = KSPDestroy(&jac->kspschur);CHKERRQ(ierr);
  ierr = MatDestroy(&jac->B);CHKERRQ(ierr);
  ierr = MatDestroy(&jac->C);CHKERRQ(ierr);
  jac->reset = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_FieldSplit"
static PetscErrorCode PCDestroy_FieldSplit(PC pc)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilink = jac->head,next;

  PetscFunctionBegin;
  ierr = PCReset_FieldSplit(pc);CHKERRQ(ierr);
  while (ilink) {
    ierr = KSPDestroy(&ilink->ksp);CHKERRQ(ierr);
    next = ilink->next;
    ierr = PetscFree(ilink->splitname);CHKERRQ(ierr);
    ierr = PetscFree(ilink->fields);CHKERRQ(ierr);
    ierr = PetscFree(ilink->fields_col);CHKERRQ(ierr);
    ierr = PetscFree(ilink);CHKERRQ(ierr);
    ilink = next;
  }
  ierr = PetscFree2(jac->x,jac->y);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitGetSubKSP_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSetFields_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSetIS_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSetType_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSetBlockSize_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSchurPrecondition_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSetSchurFactType_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_FieldSplit"
static PetscErrorCode PCSetFromOptions_FieldSplit(PC pc)
{
  PetscErrorCode  ierr;
  PetscInt        bs;
  PetscBool       flg,stokes = PETSC_FALSE;
  PC_FieldSplit   *jac = (PC_FieldSplit*)pc->data;
  PCCompositeType ctype;
  DM              ddm;
  char            ddm_name[1025];

  PetscFunctionBegin;
  ierr = PetscOptionsHead("FieldSplit options");CHKERRQ(ierr);
  if (pc->dm) {
    /* Allow the user to request a decomposition DM by name */
    ierr = PetscStrncpy(ddm_name, "", 1024); CHKERRQ(ierr);
    ierr = PetscOptionsString("-pc_fieldsplit_decomposition", "Name of the DM defining the composition", "PCSetDM", ddm_name, ddm_name,1024,&flg); CHKERRQ(ierr);
    if (flg) {
      ierr = DMCreateFieldDecompositionDM(pc->dm, ddm_name, &ddm); CHKERRQ(ierr);
      if (!ddm) {
        SETERRQ1(((PetscObject)pc)->comm, PETSC_ERR_ARG_WRONGSTATE, "Uknown field decomposition name %s", ddm_name);
      }
      ierr = PetscInfo(pc,"Using field decomposition DM defined using options database\n");CHKERRQ(ierr);
      ierr = PCSetDM(pc,ddm); CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsBool("-pc_fieldsplit_real_diagonal","Use diagonal blocks of the operator","PCFieldSplitSetRealDiagonal",jac->realdiagonal,&jac->realdiagonal,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_fieldsplit_block_size","Blocksize that defines number of fields","PCFieldSplitSetBlockSize",jac->bs,&bs,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCFieldSplitSetBlockSize(pc,bs);CHKERRQ(ierr);
  }

  ierr = PetscOptionsGetBool(((PetscObject)pc)->prefix,"-pc_fieldsplit_detect_saddle_point",&stokes,PETSC_NULL);CHKERRQ(ierr);
  if (stokes) {
    ierr = PCFieldSplitSetType(pc,PC_COMPOSITE_SCHUR);CHKERRQ(ierr);
    jac->schurpre = PC_FIELDSPLIT_SCHUR_PRE_SELF;
  }

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
    ierr = PetscOptionsGetEnum(((PetscObject)pc)->prefix,"-pc_fieldsplit_schur_factorization_type",PCFieldSplitSchurFactTypes,(PetscEnum*)&jac->schurfactorization,&flg);CHKERRQ(ierr);
    if (flg) {ierr = PetscInfo(pc,"Deprecated use of -pc_fieldsplit_schur_factorization_type\n");CHKERRQ(ierr);}
    ierr = PetscOptionsEnum("-pc_fieldsplit_schur_fact_type","Which off-diagonal parts of the block factorization to use","PCFieldSplitSetSchurFactType",PCFieldSplitSchurFactTypes,(PetscEnum)jac->schurfactorization,(PetscEnum*)&jac->schurfactorization,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-pc_fieldsplit_schur_precondition","How to build preconditioner for Schur complement","PCFieldSplitSchurPrecondition",PCFieldSplitSchurPreTypes,(PetscEnum)jac->schurpre,(PetscEnum*)&jac->schurpre,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitSetFields_FieldSplit"
PetscErrorCode  PCFieldSplitSetFields_FieldSplit(PC pc,const char splitname[],PetscInt n,const PetscInt *fields,const PetscInt *fields_col)
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
  ierr = PetscNew(struct _PC_FieldSplitLink,&ilink);CHKERRQ(ierr);
  if (splitname) {
    ierr = PetscStrallocpy(splitname,&ilink->splitname);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc(3*sizeof(char),&ilink->splitname);CHKERRQ(ierr);
    ierr = PetscSNPrintf(ilink->splitname,2,"%s",jac->nsplits);CHKERRQ(ierr);
  }
  ierr = PetscMalloc(n*sizeof(PetscInt),&ilink->fields);CHKERRQ(ierr);
  ierr = PetscMemcpy(ilink->fields,fields,n*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscInt),&ilink->fields_col);CHKERRQ(ierr);
  ierr = PetscMemcpy(ilink->fields_col,fields_col,n*sizeof(PetscInt));CHKERRQ(ierr);
  ilink->nfields = n;
  ilink->next    = PETSC_NULL;
  ierr           = KSPCreate(((PetscObject)pc)->comm,&ilink->ksp);CHKERRQ(ierr);
  ierr           = PetscObjectIncrementTabLevel((PetscObject)ilink->ksp,(PetscObject)pc,1);CHKERRQ(ierr);
  ierr           = KSPSetType(ilink->ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr           = PetscLogObjectParent((PetscObject)pc,(PetscObject)ilink->ksp);CHKERRQ(ierr);

  ierr = PetscSNPrintf(prefix,sizeof(prefix),"%sfieldsplit_%s_",((PetscObject)pc)->prefix?((PetscObject)pc)->prefix:"",ilink->splitname);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ilink->ksp,prefix);CHKERRQ(ierr);

  if (!next) {
    jac->head       = ilink;
    ilink->previous = PETSC_NULL;
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
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitGetSubKSP_FieldSplit_Schur"
PetscErrorCode  PCFieldSplitGetSubKSP_FieldSplit_Schur(PC pc,PetscInt *n,KSP **subksp)
{
  PC_FieldSplit *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(jac->nsplits*sizeof(KSP),subksp);CHKERRQ(ierr);
  ierr = MatSchurComplementGetKSP(jac->schur,*subksp);CHKERRQ(ierr);
  (*subksp)[1] = jac->kspschur;
  if (n) *n = jac->nsplits;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitGetSubKSP_FieldSplit"
PetscErrorCode  PCFieldSplitGetSubKSP_FieldSplit(PC pc,PetscInt *n,KSP **subksp)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PetscInt          cnt = 0;
  PC_FieldSplitLink ilink = jac->head;

  PetscFunctionBegin;
  ierr = PetscMalloc(jac->nsplits*sizeof(KSP),subksp);CHKERRQ(ierr);
  while (ilink) {
    (*subksp)[cnt++] = ilink->ksp;
    ilink = ilink->next;
  }
  if (cnt != jac->nsplits) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Corrupt PCFIELDSPLIT object: number of splits in linked list %D does not match number in object %D",cnt,jac->nsplits);
  if (n) *n = jac->nsplits;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitSetIS_FieldSplit"
PetscErrorCode  PCFieldSplitSetIS_FieldSplit(PC pc,const char splitname[],IS is)
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
  ierr = PetscNew(struct _PC_FieldSplitLink,&ilink);CHKERRQ(ierr);
  if (splitname) {
    ierr = PetscStrallocpy(splitname,&ilink->splitname);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc(8*sizeof(char),&ilink->splitname);CHKERRQ(ierr);
    ierr = PetscSNPrintf(ilink->splitname,7,"%D",jac->nsplits);CHKERRQ(ierr);
  }
  ierr = PetscObjectReference((PetscObject)is);CHKERRQ(ierr);
  ierr = ISDestroy(&ilink->is);CHKERRQ(ierr);
  ilink->is      = is;
  ierr = PetscObjectReference((PetscObject)is);CHKERRQ(ierr);
  ierr = ISDestroy(&ilink->is_col);CHKERRQ(ierr);
  ilink->is_col  = is;
  ilink->next    = PETSC_NULL;
  ierr           = KSPCreate(((PetscObject)pc)->comm,&ilink->ksp);CHKERRQ(ierr);
  ierr           = PetscObjectIncrementTabLevel((PetscObject)ilink->ksp,(PetscObject)pc,1);CHKERRQ(ierr);
  ierr           = KSPSetType(ilink->ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr           = PetscLogObjectParent((PetscObject)pc,(PetscObject)ilink->ksp);CHKERRQ(ierr);

  ierr = PetscSNPrintf(prefix,sizeof(prefix),"%sfieldsplit_%s_",((PetscObject)pc)->prefix?((PetscObject)pc)->prefix:"",ilink->splitname);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ilink->ksp,prefix);CHKERRQ(ierr);

  if (!next) {
    jac->head       = ilink;
    ilink->previous = PETSC_NULL;
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
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitSetFields"
/*@
    PCFieldSplitSetFields - Sets the fields for one particular split in the field split preconditioner

    Logically Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
.   splitname - name of this split, if PETSC_NULL the number of the split is used
.   n - the number of fields in this split
-   fields - the fields in this split

    Level: intermediate

    Notes: Use PCFieldSplitSetIS() to set a completely general set of indices as a field.

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
  if (n < 1) SETERRQ2(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Provided number of fields %D in split \"%s\" not positive",n,splitname);
  PetscValidIntPointer(fields,3);
  ierr = PetscTryMethod(pc,"PCFieldSplitSetFields_C",(PC,const char[],PetscInt,const PetscInt *,const PetscInt *),(pc,splitname,n,fields,fields_col));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitSetIS"
/*@
    PCFieldSplitSetIS - Sets the exact elements for field

    Logically Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
.   splitname - name of this split, if PETSC_NULL the number of the split is used
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

#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitGetIS"
/*@
    PCFieldSplitGetIS - Retrieves the elements for a field as an IS

    Logically Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
-   splitname - name of this split

    Output Parameter:
-   is - the index set that defines the vector elements in this field, or PETSC_NULL if the field is not found

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
    PC_FieldSplit    *jac   = (PC_FieldSplit *) pc->data;
    PC_FieldSplitLink ilink = jac->head;
    PetscBool         found;

    *is = PETSC_NULL;
    while(ilink) {
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

#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitSetBlockSize"
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

#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitGetSubKSP"
/*@C
   PCFieldSplitGetSubKSP - Gets the KSP contexts for all splits

   Collective on KSP

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  n - the number of splits
-  pc - the array of KSP contexts

   Note:
   After PCFieldSplitGetSubKSP() the array of KSPs IS to be freed by the user
   (not the KSP just the array that contains them).

   You must call KSPSetUp() before calling PCFieldSplitGetSubKSP().

   Fortran Usage: You must pass in a KSP array that is large enough to contain all the local KSPs.
      You can call PCFieldSplitGetSubKSP(pc,n,PETSC_NULL_OBJECT,ierr) to determine how large the
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

#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitSchurPrecondition"
/*@
    PCFieldSplitSchurPrecondition -  Indicates if the Schur complement is preconditioned by a preconditioner constructed by the
      A11 matrix. Otherwise no preconditioner is used.

    Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
.   ptype - which matrix to use for preconditioning the Schur complement, PC_FIELDSPLIT_SCHUR_PRE_DIAG (diag) is default
-   userpre - matrix to use for preconditioning, or PETSC_NULL

    Options Database:
.     -pc_fieldsplit_schur_precondition <self,user,diag> default is diag

    Notes:
$    If ptype is
$        user then the preconditioner for the Schur complement is generated by the provided matrix (pre argument
$             to this function).
$        diag then the preconditioner for the Schur complement is generated by the block diagonal part of the original
$             matrix associated with the Schur complement (i.e. A11)
$        self the preconditioner for the Schur complement is generated from the Schur complement matrix itself:
$             The only preconditioner that currently works directly with the Schur complement matrix object is the PCLSC
$             preconditioner

     When solving a saddle point problem, where the A11 block is identically zero, using diag as the ptype only makes sense
    with the additional option -fieldsplit_1_pc_type none. Usually for saddle point problems one would use a ptype of self and
    -fieldsplit_1_pc_type lsc which uses the least squares commutator compute a preconditioner for the Schur complement.

    Developer Notes: This is a terrible name, gives no good indication of what the function does and should also have Set in
     the name since it sets a proceedure to use.

    Level: intermediate

.seealso: PCFieldSplitGetSubKSP(), PCFIELDSPLIT, PCFieldSplitSetFields(), PCFieldSplitSchurPreType, PCLSC

@*/
PetscErrorCode  PCFieldSplitSchurPrecondition(PC pc,PCFieldSplitSchurPreType ptype,Mat pre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCFieldSplitSchurPrecondition_C",(PC,PCFieldSplitSchurPreType,Mat),(pc,ptype,pre));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitSchurPrecondition_FieldSplit"
PetscErrorCode  PCFieldSplitSchurPrecondition_FieldSplit(PC pc,PCFieldSplitSchurPreType ptype,Mat pre)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  jac->schurpre = ptype;
  if (pre) {
    ierr = MatDestroy(&jac->schur_user);CHKERRQ(ierr);
    jac->schur_user = pre;
    ierr = PetscObjectReference((PetscObject)jac->schur_user);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitSetSchurFactType"
/*@
    PCFieldSplitSetSchurFactType -  sets which blocks of the approximate block factorization to retain

    Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
-   ftype - which blocks of factorization to retain, PC_FIELDSPLIT_SCHUR_FACT_FULL is default

    Options Database:
.     -pc_fieldsplit_schur_fact_type <diag,lower,upper,full> default is full


    Level: intermediate

    Notes:
    The FULL factorization is

$   (A   B)  = (1       0) (A   0) (1  Ainv*B)
$   (C   D)    (C*Ainv  1) (0   S) (0     1  )

    where S = D - C*Ainv*B. In practice, the full factorization is applied via block triangular solves with the grouping L*(D*U). UPPER uses D*U, LOWER uses L*D,
    and DIAG is the diagonal part with the sign of S flipped (because this makes the preconditioner positive definite for many formulations, thus allowing the use of KSPMINRES).

    If applied exactly, FULL factorization is a direct solver. The preconditioned operator with LOWER or UPPER has all eigenvalues equal to 1 and minimal polynomial
    of degree 2, so KSPGMRES converges in 2 iterations. If the iteration count is very low, consider using KSPFGMRES or KSPGCR which can use one less preconditioner
    application in this case. Note that the preconditioned operator may be highly non-normal, so such fast convergence may not be observed in practice. With DIAG,
    the preconditioned operator has three distinct nonzero eigenvalues and minimal polynomial of degree at most 4, so KSPGMRES converges in at most 4 iterations.

    For symmetric problems in which A is positive definite and S is negative definite, DIAG can be used with KSPMINRES. Note that a flexible method like KSPFGMRES
    or KSPGCR must be used if the fieldsplit preconditioner is nonlinear (e.g. a few iterations of a Krylov method is used inside a split).

    References:
    Murphy, Golub, and Wathen, A note on preconditioning indefinite linear systems, SIAM J. Sci. Comput., 21 (2000) pp. 1969-1972.

    Ipsen, A note on preconditioning nonsymmetric matrices, SIAM J. Sci. Comput., 23 (2001), pp. 1050-1051.

.seealso: PCFieldSplitGetSubKSP(), PCFIELDSPLIT, PCFieldSplitSetFields(), PCFieldSplitSchurPreType
@*/
PetscErrorCode  PCFieldSplitSetSchurFactType(PC pc,PCFieldSplitSchurFactType ftype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCFieldSplitSetSchurFactType_C",(PC,PCFieldSplitSchurFactType),(pc,ftype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitSetSchurFactType_FieldSplit"
PETSC_EXTERN_C PetscErrorCode PCFieldSplitSetSchurFactType_FieldSplit(PC pc,PCFieldSplitSchurFactType ftype)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;

  PetscFunctionBegin;
  jac->schurfactorization = ftype;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitGetSchurBlocks"
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
  PC_FieldSplit *jac = (PC_FieldSplit *) pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (jac->type != PC_COMPOSITE_SCHUR) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONG, "FieldSplit is not using a Schur complement approach.");
  if (A00) *A00 = jac->pmat[0];
  if (A01) *A01 = jac->B;
  if (A10) *A10 = jac->C;
  if (A11) *A11 = jac->pmat[1];
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitSetType_FieldSplit"
PetscErrorCode  PCFieldSplitSetType_FieldSplit(PC pc,PCCompositeType type)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  jac->type = type;
  if (type == PC_COMPOSITE_SCHUR) {
    pc->ops->apply = PCApply_FieldSplit_Schur;
    pc->ops->view  = PCView_FieldSplit_Schur;
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitGetSubKSP_C","PCFieldSplitGetSubKSP_FieldSplit_Schur",PCFieldSplitGetSubKSP_FieldSplit_Schur);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSchurPrecondition_C","PCFieldSplitSchurPrecondition_FieldSplit",PCFieldSplitSchurPrecondition_FieldSplit);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSetSchurFactType_C","PCFieldSplitSetSchurFactType_FieldSplit",PCFieldSplitSetSchurFactType_FieldSplit);CHKERRQ(ierr);

  } else {
    pc->ops->apply = PCApply_FieldSplit;
    pc->ops->view  = PCView_FieldSplit;
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitGetSubKSP_C","PCFieldSplitGetSubKSP_FieldSplit",PCFieldSplitGetSubKSP_FieldSplit);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSchurPrecondition_C","",0);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSetSchurFactType_C","",0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitSetBlockSize_FieldSplit"
PetscErrorCode  PCFieldSplitSetBlockSize_FieldSplit(PC pc,PetscInt bs)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;

  PetscFunctionBegin;
  if (bs < 1) SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Blocksize must be positive, you gave %D",bs);
  if (jac->bs > 0 && jac->bs != bs) SETERRQ2(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot change fieldsplit blocksize from %D to %D after it has been set",jac->bs,bs);
  jac->bs = bs;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitSetType"
/*@
   PCFieldSplitSetType - Sets the type of fieldsplit preconditioner.

   Collective on PC

   Input Parameter:
.  pc - the preconditioner context
.  type - PC_COMPOSITE_ADDITIVE, PC_COMPOSITE_MULTIPLICATIVE (default), PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE, PC_COMPOSITE_SPECIAL, PC_COMPOSITE_SCHUR

   Options Database Key:
.  -pc_fieldsplit_type <type: one of multiplicative, additive, symmetric_multiplicative, special, schur> - Sets fieldsplit preconditioner type

   Level: Intermediate

.keywords: PC, set, type, composite preconditioner, additive, multiplicative

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

#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitGetType"
/*@
  PCFieldSplitGetType - Gets the type of fieldsplit preconditioner.

  Not collective

  Input Parameter:
. pc - the preconditioner context

  Output Parameter:
. type - PC_COMPOSITE_ADDITIVE, PC_COMPOSITE_MULTIPLICATIVE (default), PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE, PC_COMPOSITE_SPECIAL, PC_COMPOSITE_SCHUR

  Level: Intermediate

.keywords: PC, set, type, composite preconditioner, additive, multiplicative
.seealso: PCCompositeSetType()
@*/
PetscErrorCode PCFieldSplitGetType(PC pc, PCCompositeType *type)
{
  PC_FieldSplit *jac = (PC_FieldSplit *) pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidIntPointer(type,2);
  *type = jac->type;
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
.   -pc_fieldsplit_type <additive,multiplicative,symmetric_multiplicative,schur> - type of relaxation or factorization splitting
.   -pc_fieldsplit_schur_precondition <self,user,diag> - default is diag
.   -pc_fieldsplit_detect_saddle_point - automatically finds rows with zero or negative diagonal and uses Schur complement with no preconditioner as the solver

-    Options prefix for inner solvers when using Schur complement preconditioner are -fieldsplit_0_ and -fieldsplit_1_
     for all other solvers they are -fieldsplit_%d_ for the dth field, use -fieldsplit_ for all fields

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
$              ( I   -A10 ksp(A00) ) ( inv(A00)     0  ) (     I          0  )
$              ( 0         I       ) (   0      ksp(S) ) ( -A10 ksp(A00)  I  )
     where the action of inv(A00) is applied using the KSP solver with prefix -fieldsplit_0_. The action of
     ksp(S) is computed using the KSP solver with prefix -fieldsplit_splitname_ (where splitname was given
     in providing the SECOND split or 1 if not give). For PCFieldSplitGetKSP() when field number is 0,
     it returns the KSP associated with -fieldsplit_0_ while field number 1 gives -fieldsplit_1_ KSP. By default
     A11 is used to construct a preconditioner for S, use PCFieldSplitSchurPrecondition() to turn on or off this
     option. You can use the preconditioner PCLSC to precondition the Schur complement with -fieldsplit_1_pc_type lsc. The
     factorization type is set using -pc_fieldsplit_schur_fact_type <diag, lower, upper, full>. The full is shown above,
     diag gives
$              ( inv(A00)     0   )
$              (   0      -ksp(S) )
     note that slightly counter intuitively there is a negative in front of the ksp(S) so that the preconditioner is positive definite. The lower factorization is the inverse of
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

   Concepts: physics based preconditioners, block preconditioners

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, Block_Preconditioners, PCLSC,
           PCFieldSplitGetSubKSP(), PCFieldSplitSetFields(), PCFieldSplitSetType(), PCFieldSplitSetIS(), PCFieldSplitSchurPrecondition()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCCreate_FieldSplit"
PetscErrorCode  PCCreate_FieldSplit(PC pc)
{
  PetscErrorCode ierr;
  PC_FieldSplit  *jac;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,PC_FieldSplit,&jac);CHKERRQ(ierr);
  jac->bs        = -1;
  jac->nsplits   = 0;
  jac->type      = PC_COMPOSITE_MULTIPLICATIVE;
  jac->schurpre  = PC_FIELDSPLIT_SCHUR_PRE_USER; /* Try user preconditioner first, fall back on diagonal */
  jac->schurfactorization = PC_FIELDSPLIT_SCHUR_FACT_FULL;

  pc->data     = (void*)jac;

  pc->ops->apply             = PCApply_FieldSplit;
  pc->ops->applytranspose    = PCApplyTranspose_FieldSplit;
  pc->ops->setup             = PCSetUp_FieldSplit;
  pc->ops->reset             = PCReset_FieldSplit;
  pc->ops->destroy           = PCDestroy_FieldSplit;
  pc->ops->setfromoptions    = PCSetFromOptions_FieldSplit;
  pc->ops->view              = PCView_FieldSplit;
  pc->ops->applyrichardson   = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitGetSubKSP_C","PCFieldSplitGetSubKSP_FieldSplit",
                    PCFieldSplitGetSubKSP_FieldSplit);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSetFields_C","PCFieldSplitSetFields_FieldSplit",
                    PCFieldSplitSetFields_FieldSplit);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSetIS_C","PCFieldSplitSetIS_FieldSplit",
                    PCFieldSplitSetIS_FieldSplit);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSetType_C","PCFieldSplitSetType_FieldSplit",
                    PCFieldSplitSetType_FieldSplit);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSetBlockSize_C","PCFieldSplitSetBlockSize_FieldSplit",
                    PCFieldSplitSetBlockSize_FieldSplit);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END



