#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/ksp/pc/impls/bddc/bddc.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>
#include <../src/mat/impls/dense/seq/dense.h>
#include <petscdmplex.h>
#include <petscblaslapack.h>
#include <petsc/private/sfimpl.h>
#include <petsc/private/dmpleximpl.h>
#include <petscdmda.h>

static PetscErrorCode MatMPIAIJRestrict(Mat,MPI_Comm,Mat*);

/* if range is true,  it returns B s.t. span{B} = range(A)
   if range is false, it returns B s.t. range(B) _|_ range(A) */
PetscErrorCode MatDenseOrthogonalRangeOrComplement(Mat A, PetscBool range, PetscInt lw, PetscScalar *work, PetscReal *rwork, Mat *B)
{
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar    *uwork,*data,*U, ds = 0.;
  PetscReal      *sing;
  PetscBLASInt   bM,bN,lwork,lierr,di = 1;
  PetscInt       ulw,i,nr,nc,n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_MISSING_LAPACK_GESVD)
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"LAPACK _GESVD not available");
#else
  ierr = MatGetSize(A,&nr,&nc);CHKERRQ(ierr);
  if (!nr || !nc) PetscFunctionReturn(0);

  /* workspace */
  if (!work) {
    ulw  = PetscMax(PetscMax(1,5*PetscMin(nr,nc)),3*PetscMin(nr,nc)+PetscMax(nr,nc));
    ierr = PetscMalloc1(ulw,&uwork);CHKERRQ(ierr);
  } else {
    ulw   = lw;
    uwork = work;
  }
  n = PetscMin(nr,nc);
  if (!rwork) {
    ierr = PetscMalloc1(n,&sing);CHKERRQ(ierr);
  } else {
    sing = rwork;
  }

  /* SVD */
  ierr = PetscMalloc1(nr*nr,&U);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(nr,&bM);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(nc,&bN);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ulw,&lwork);CHKERRQ(ierr);
  ierr = MatDenseGetArray(A,&data);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("A","N",&bM,&bN,data,&bM,sing,U,&bM,&ds,&di,uwork,&lwork,&lierr));
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GESVD Lapack routine %d",(int)lierr);
  ierr = MatDenseRestoreArray(A,&data);CHKERRQ(ierr);
  for (i=0;i<n;i++) if (sing[i] < PETSC_SMALL) break;
  if (!rwork) {
    ierr = PetscFree(sing);CHKERRQ(ierr);
  }
  if (!work) {
    ierr = PetscFree(uwork);CHKERRQ(ierr);
  }
  /* create B */
  if (!range) {
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,nr,nr-i,NULL,B);CHKERRQ(ierr);
    ierr = MatDenseGetArray(*B,&data);CHKERRQ(ierr);
    ierr = PetscArraycpy(data,U+nr*i,(nr-i)*nr);CHKERRQ(ierr);
  } else {
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,nr,i,NULL,B);CHKERRQ(ierr);
    ierr = MatDenseGetArray(*B,&data);CHKERRQ(ierr);
    ierr = PetscArraycpy(data,U,i*nr);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(*B,&data);CHKERRQ(ierr);
  ierr = PetscFree(U);CHKERRQ(ierr);
#endif
#else /* PETSC_USE_COMPLEX */
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented for complexes");
#endif
  PetscFunctionReturn(0);
}

/* TODO REMOVE */
#if defined(PRINT_GDET)
static int inc = 0;
static int lev = 0;
#endif

PetscErrorCode PCBDDCComputeNedelecChangeEdge(Mat lG, IS edge, IS extrow, IS extcol, IS corners, Mat* Gins, Mat* GKins, PetscScalar cvals[2], PetscScalar *work, PetscReal *rwork)
{
  PetscErrorCode ierr;
  Mat            GE,GEd;
  PetscInt       rsize,csize,esize;
  PetscScalar    *ptr;

  PetscFunctionBegin;
  ierr = ISGetSize(edge,&esize);CHKERRQ(ierr);
  if (!esize) PetscFunctionReturn(0);
  ierr = ISGetSize(extrow,&rsize);CHKERRQ(ierr);
  ierr = ISGetSize(extcol,&csize);CHKERRQ(ierr);

  /* gradients */
  ptr  = work + 5*esize;
  ierr = MatCreateSubMatrix(lG,extrow,extcol,MAT_INITIAL_MATRIX,&GE);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,rsize,csize,ptr,Gins);CHKERRQ(ierr);
  ierr = MatConvert(GE,MATSEQDENSE,MAT_REUSE_MATRIX,Gins);CHKERRQ(ierr);
  ierr = MatDestroy(&GE);CHKERRQ(ierr);

  /* constants */
  ptr += rsize*csize;
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,esize,csize,ptr,&GEd);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix(lG,edge,extcol,MAT_INITIAL_MATRIX,&GE);CHKERRQ(ierr);
  ierr = MatConvert(GE,MATSEQDENSE,MAT_REUSE_MATRIX,&GEd);CHKERRQ(ierr);
  ierr = MatDestroy(&GE);CHKERRQ(ierr);
  ierr = MatDenseOrthogonalRangeOrComplement(GEd,PETSC_FALSE,5*esize,work,rwork,GKins);CHKERRQ(ierr);
  ierr = MatDestroy(&GEd);CHKERRQ(ierr);

  if (corners) {
    Mat               GEc;
    const PetscScalar *vals;
    PetscScalar       v;

    ierr = MatCreateSubMatrix(lG,edge,corners,MAT_INITIAL_MATRIX,&GEc);CHKERRQ(ierr);
    ierr = MatTransposeMatMult(GEc,*GKins,MAT_INITIAL_MATRIX,1.0,&GEd);CHKERRQ(ierr);
    ierr = MatDenseGetArrayRead(GEd,&vals);CHKERRQ(ierr);
    /* v    = PetscAbsScalar(vals[0]) */;
    v    = 1.;
    cvals[0] = vals[0]/v;
    cvals[1] = vals[1]/v;
    ierr = MatDenseRestoreArrayRead(GEd,&vals);CHKERRQ(ierr);
    ierr = MatScale(*GKins,1./v);CHKERRQ(ierr);
#if defined(PRINT_GDET)
    {
      PetscViewer viewer;
      char filename[256];
      sprintf(filename,"Gdet_l%d_r%d_cc%d.m",lev,PetscGlobalRank,inc++);
      ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF,filename,&viewer);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)GEc,"GEc");CHKERRQ(ierr);
      ierr = MatView(GEc,viewer);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)(*GKins),"GK");CHKERRQ(ierr);
      ierr = MatView(*GKins,viewer);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)GEd,"Gproj");CHKERRQ(ierr);
      ierr = MatView(GEd,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
#endif
    ierr = MatDestroy(&GEd);CHKERRQ(ierr);
    ierr = MatDestroy(&GEc);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCNedelecSupport(PC pc)
{
  PC_BDDC                *pcbddc = (PC_BDDC*)pc->data;
  Mat_IS                 *matis = (Mat_IS*)pc->pmat->data;
  Mat                    G,T,conn,lG,lGt,lGis,lGall,lGe,lGinit;
  Vec                    tvec;
  PetscSF                sfv;
  ISLocalToGlobalMapping el2g,vl2g,fl2g,al2g;
  MPI_Comm               comm;
  IS                     lned,primals,allprimals,nedfieldlocal;
  IS                     *eedges,*extrows,*extcols,*alleedges;
  PetscBT                btv,bte,btvc,btb,btbd,btvcand,btvi,btee,bter;
  PetscScalar            *vals,*work;
  PetscReal              *rwork;
  const PetscInt         *idxs,*ii,*jj,*iit,*jjt;
  PetscInt               ne,nv,Lv,order,n,field;
  PetscInt               n_neigh,*neigh,*n_shared,**shared;
  PetscInt               i,j,extmem,cum,maxsize,nee;
  PetscInt               *extrow,*extrowcum,*marks,*vmarks,*gidxs;
  PetscInt               *sfvleaves,*sfvroots;
  PetscInt               *corners,*cedges;
  PetscInt               *ecount,**eneighs,*vcount,**vneighs;
#if defined(PETSC_USE_DEBUG)
  PetscInt               *emarks;
#endif
  PetscBool              print,eerr,done,lrc[2],conforming,global,singular,setprimal;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  /* If the discrete gradient is defined for a subset of dofs and global is true,
     it assumes G is given in global ordering for all the dofs.
     Otherwise, the ordering is global for the Nedelec field */
  order      = pcbddc->nedorder;
  conforming = pcbddc->conforming;
  field      = pcbddc->nedfield;
  global     = pcbddc->nedglobal;
  setprimal  = PETSC_FALSE;
  print      = PETSC_FALSE;
  singular   = PETSC_FALSE;

  /* Command line customization */
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)pc),((PetscObject)pc)->prefix,"BDDC Nedelec options","PC");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_nedelec_field_primal","All edge dofs set as primals: Toselli's algorithm C",NULL,setprimal,&setprimal,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_nedelec_singular","Infer nullspace from discrete gradient",NULL,singular,&singular,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_bddc_nedelec_order","Test variable order code (to be removed)",NULL,order,&order,NULL);CHKERRQ(ierr);
  /* print debug info TODO: to be removed */
  ierr = PetscOptionsBool("-pc_bddc_nedelec_print","Print debug info",NULL,print,&print,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Return if there are no edges in the decomposition and the problem is not singular */
  ierr = MatGetLocalToGlobalMapping(pc->pmat,&al2g,NULL);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetSize(al2g,&n);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  if (!singular) {
    ierr   = VecGetArrayRead(matis->counter,(const PetscScalar**)&vals);CHKERRQ(ierr);
    lrc[0] = PETSC_FALSE;
    for (i=0;i<n;i++) {
      if (PetscRealPart(vals[i]) > 2.) {
        lrc[0] = PETSC_TRUE;
        break;
      }
    }
    ierr = VecRestoreArrayRead(matis->counter,(const PetscScalar**)&vals);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(&lrc[0],&lrc[1],1,MPIU_BOOL,MPI_LOR,comm);CHKERRQ(ierr);
    if (!lrc[1]) PetscFunctionReturn(0);
  }

  /* Get Nedelec field */
  if (pcbddc->n_ISForDofsLocal && field >= pcbddc->n_ISForDofsLocal) SETERRQ2(comm,PETSC_ERR_USER,"Invalid field for Nedelec %D: number of fields is %D",field,pcbddc->n_ISForDofsLocal);
  if (pcbddc->n_ISForDofsLocal && field >= 0) {
    ierr          = PetscObjectReference((PetscObject)pcbddc->ISForDofsLocal[field]);CHKERRQ(ierr);
    nedfieldlocal = pcbddc->ISForDofsLocal[field];
    ierr          = ISGetLocalSize(nedfieldlocal,&ne);CHKERRQ(ierr);
  } else if (!pcbddc->n_ISForDofsLocal && field != PETSC_DECIDE) {
    ne            = n;
    nedfieldlocal = NULL;
    global        = PETSC_TRUE;
  } else if (field == PETSC_DECIDE) {
    PetscInt rst,ren,*idx;

    ierr = PetscArrayzero(matis->sf_leafdata,n);CHKERRQ(ierr);
    ierr = PetscArrayzero(matis->sf_rootdata,pc->pmat->rmap->n);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(pcbddc->discretegradient,&rst,&ren);CHKERRQ(ierr);
    for (i=rst;i<ren;i++) {
      PetscInt nc;

      ierr = MatGetRow(pcbddc->discretegradient,i,&nc,NULL,NULL);CHKERRQ(ierr);
      if (nc > 1) matis->sf_rootdata[i-rst] = 1;
      ierr = MatRestoreRow(pcbddc->discretegradient,i,&nc,NULL,NULL);CHKERRQ(ierr);
    }
    ierr = PetscSFBcastBegin(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata);CHKERRQ(ierr);
    ierr = PetscMalloc1(n,&idx);CHKERRQ(ierr);
    for (i=0,ne=0;i<n;i++) if (matis->sf_leafdata[i]) idx[ne++] = i;
    ierr = ISCreateGeneral(comm,ne,idx,PETSC_OWN_POINTER,&nedfieldlocal);CHKERRQ(ierr);
  } else {
    SETERRQ(comm,PETSC_ERR_USER,"When multiple fields are present, the Nedelec field has to be specified");
  }

  /* Sanity checks */
  if (!order && !conforming) SETERRQ(comm,PETSC_ERR_SUP,"Variable order and non-conforming spaces are not supported at the same time");
  if (pcbddc->user_ChangeOfBasisMatrix) SETERRQ(comm,PETSC_ERR_SUP,"Cannot generate Nedelec support with user defined change of basis");
  if (order && ne%order) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"The number of local edge dofs %D it's not a multiple of the order %D",ne,order);

  /* Just set primal dofs and return */
  if (setprimal) {
    IS       enedfieldlocal;
    PetscInt *eidxs;

    ierr = PetscMalloc1(ne,&eidxs);CHKERRQ(ierr);
    ierr = VecGetArrayRead(matis->counter,(const PetscScalar**)&vals);CHKERRQ(ierr);
    if (nedfieldlocal) {
      ierr = ISGetIndices(nedfieldlocal,&idxs);CHKERRQ(ierr);
      for (i=0,cum=0;i<ne;i++) {
        if (PetscRealPart(vals[idxs[i]]) > 2.) {
          eidxs[cum++] = idxs[i];
        }
      }
      ierr = ISRestoreIndices(nedfieldlocal,&idxs);CHKERRQ(ierr);
    } else {
      for (i=0,cum=0;i<ne;i++) {
        if (PetscRealPart(vals[i]) > 2.) {
          eidxs[cum++] = i;
        }
      }
    }
    ierr = VecRestoreArrayRead(matis->counter,(const PetscScalar**)&vals);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm,cum,eidxs,PETSC_COPY_VALUES,&enedfieldlocal);CHKERRQ(ierr);
    ierr = PCBDDCSetPrimalVerticesLocalIS(pc,enedfieldlocal);CHKERRQ(ierr);
    ierr = PetscFree(eidxs);CHKERRQ(ierr);
    ierr = ISDestroy(&nedfieldlocal);CHKERRQ(ierr);
    ierr = ISDestroy(&enedfieldlocal);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* Compute some l2g maps */
  if (nedfieldlocal) {
    IS is;

    /* need to map from the local Nedelec field to local numbering */
    ierr = ISLocalToGlobalMappingCreateIS(nedfieldlocal,&fl2g);CHKERRQ(ierr);
    /* need to map from the local Nedelec field to global numbering for the whole dofs*/
    ierr = ISLocalToGlobalMappingApplyIS(al2g,nedfieldlocal,&is);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreateIS(is,&al2g);CHKERRQ(ierr);
    /* need to map from the local Nedelec field to global numbering (for Nedelec only) */
    if (global) {
      ierr = PetscObjectReference((PetscObject)al2g);CHKERRQ(ierr);
      el2g = al2g;
    } else {
      IS gis;

      ierr = ISRenumber(is,NULL,NULL,&gis);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingCreateIS(gis,&el2g);CHKERRQ(ierr);
      ierr = ISDestroy(&gis);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&is);CHKERRQ(ierr);
  } else {
    /* restore default */
    pcbddc->nedfield = -1;
    /* one ref for the destruction of al2g, one for el2g */
    ierr = PetscObjectReference((PetscObject)al2g);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)al2g);CHKERRQ(ierr);
    el2g = al2g;
    fl2g = NULL;
  }

  /* Start communication to drop connections for interior edges (for cc analysis only) */
  ierr = PetscArrayzero(matis->sf_leafdata,n);CHKERRQ(ierr);
  ierr = PetscArrayzero(matis->sf_rootdata,pc->pmat->rmap->n);CHKERRQ(ierr);
  if (nedfieldlocal) {
    ierr = ISGetIndices(nedfieldlocal,&idxs);CHKERRQ(ierr);
    for (i=0;i<ne;i++) matis->sf_leafdata[idxs[i]] = 1;
    ierr = ISRestoreIndices(nedfieldlocal,&idxs);CHKERRQ(ierr);
  } else {
    for (i=0;i<ne;i++) matis->sf_leafdata[i] = 1;
  }
  ierr = PetscSFReduceBegin(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,MPI_SUM);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,MPI_SUM);CHKERRQ(ierr);

  if (!singular) { /* drop connections with interior edges to avoid unneeded communications and memory movements */
    ierr = MatDuplicate(pcbddc->discretegradient,MAT_COPY_VALUES,&G);CHKERRQ(ierr);
    ierr = MatSetOption(G,MAT_KEEP_NONZERO_PATTERN,PETSC_FALSE);CHKERRQ(ierr);
    if (global) {
      PetscInt rst;

      ierr = MatGetOwnershipRange(G,&rst,NULL);CHKERRQ(ierr);
      for (i=0,cum=0;i<pc->pmat->rmap->n;i++) {
        if (matis->sf_rootdata[i] < 2) {
          matis->sf_rootdata[cum++] = i + rst;
        }
      }
      ierr = MatSetOption(G,MAT_NO_OFF_PROC_ZERO_ROWS,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatZeroRows(G,cum,matis->sf_rootdata,0.,NULL,NULL);CHKERRQ(ierr);
    } else {
      PetscInt *tbz;

      ierr = PetscMalloc1(ne,&tbz);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata);CHKERRQ(ierr);
      ierr = ISGetIndices(nedfieldlocal,&idxs);CHKERRQ(ierr);
      for (i=0,cum=0;i<ne;i++)
        if (matis->sf_leafdata[idxs[i]] == 1)
          tbz[cum++] = i;
      ierr = ISRestoreIndices(nedfieldlocal,&idxs);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingApply(el2g,cum,tbz,tbz);CHKERRQ(ierr);
      ierr = MatZeroRows(G,cum,tbz,0.,NULL,NULL);CHKERRQ(ierr);
      ierr = PetscFree(tbz);CHKERRQ(ierr);
    }
  } else { /* we need the entire G to infer the nullspace */
    ierr = PetscObjectReference((PetscObject)pcbddc->discretegradient);CHKERRQ(ierr);
    G    = pcbddc->discretegradient;
  }

  /* Extract subdomain relevant rows of G */
  ierr = ISLocalToGlobalMappingGetIndices(el2g,&idxs);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,ne,idxs,PETSC_USE_POINTER,&lned);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix(G,lned,NULL,MAT_INITIAL_MATRIX,&lGall);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreIndices(el2g,&idxs);CHKERRQ(ierr);
  ierr = ISDestroy(&lned);CHKERRQ(ierr);
  ierr = MatConvert(lGall,MATIS,MAT_INITIAL_MATRIX,&lGis);CHKERRQ(ierr);
  ierr = MatDestroy(&lGall);CHKERRQ(ierr);
  ierr = MatISGetLocalMat(lGis,&lG);CHKERRQ(ierr);

  /* SF for nodal dofs communications */
  ierr = MatGetLocalSize(G,NULL,&Lv);CHKERRQ(ierr);
  ierr = MatGetLocalToGlobalMapping(lGis,NULL,&vl2g);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)vl2g);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetSize(vl2g,&nv);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm,&sfv);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(vl2g,&idxs);CHKERRQ(ierr);
  ierr = PetscSFSetGraphLayout(sfv,lGis->cmap,nv,NULL,PETSC_OWN_POINTER,idxs);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreIndices(vl2g,&idxs);CHKERRQ(ierr);
  i    = singular ? 2 : 1;
  ierr = PetscMalloc2(i*nv,&sfvleaves,i*Lv,&sfvroots);CHKERRQ(ierr);

  /* Destroy temporary G created in MATIS format and modified G */
  ierr = PetscObjectReference((PetscObject)lG);CHKERRQ(ierr);
  ierr = MatDestroy(&lGis);CHKERRQ(ierr);
  ierr = MatDestroy(&G);CHKERRQ(ierr);

  if (print) {
    ierr = PetscObjectSetName((PetscObject)lG,"initial_lG");CHKERRQ(ierr);
    ierr = MatView(lG,NULL);CHKERRQ(ierr);
  }

  /* Save lG for values insertion in change of basis */
  ierr = MatDuplicate(lG,MAT_COPY_VALUES,&lGinit);CHKERRQ(ierr);

  /* Analyze the edge-nodes connections (duplicate lG) */
  ierr = MatDuplicate(lG,MAT_COPY_VALUES,&lGe);CHKERRQ(ierr);
  ierr = MatSetOption(lGe,MAT_KEEP_NONZERO_PATTERN,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscBTCreate(nv,&btv);CHKERRQ(ierr);
  ierr = PetscBTCreate(ne,&bte);CHKERRQ(ierr);
  ierr = PetscBTCreate(ne,&btb);CHKERRQ(ierr);
  ierr = PetscBTCreate(ne,&btbd);CHKERRQ(ierr);
  ierr = PetscBTCreate(nv,&btvcand);CHKERRQ(ierr);
  /* need to import the boundary specification to ensure the
     proper detection of coarse edges' endpoints */
  if (pcbddc->DirichletBoundariesLocal) {
    IS is;

    if (fl2g) {
      ierr = ISGlobalToLocalMappingApplyIS(fl2g,IS_GTOLM_MASK,pcbddc->DirichletBoundariesLocal,&is);CHKERRQ(ierr);
    } else {
      is = pcbddc->DirichletBoundariesLocal;
    }
    ierr = ISGetLocalSize(is,&cum);CHKERRQ(ierr);
    ierr = ISGetIndices(is,&idxs);CHKERRQ(ierr);
    for (i=0;i<cum;i++) {
      if (idxs[i] >= 0) {
        ierr = PetscBTSet(btb,idxs[i]);CHKERRQ(ierr);
        ierr = PetscBTSet(btbd,idxs[i]);CHKERRQ(ierr);
      }
    }
    ierr = ISRestoreIndices(is,&idxs);CHKERRQ(ierr);
    if (fl2g) {
      ierr = ISDestroy(&is);CHKERRQ(ierr);
    }
  }
  if (pcbddc->NeumannBoundariesLocal) {
    IS is;

    if (fl2g) {
      ierr = ISGlobalToLocalMappingApplyIS(fl2g,IS_GTOLM_MASK,pcbddc->NeumannBoundariesLocal,&is);CHKERRQ(ierr);
    } else {
      is = pcbddc->NeumannBoundariesLocal;
    }
    ierr = ISGetLocalSize(is,&cum);CHKERRQ(ierr);
    ierr = ISGetIndices(is,&idxs);CHKERRQ(ierr);
    for (i=0;i<cum;i++) {
      if (idxs[i] >= 0) {
        ierr = PetscBTSet(btb,idxs[i]);CHKERRQ(ierr);
      }
    }
    ierr = ISRestoreIndices(is,&idxs);CHKERRQ(ierr);
    if (fl2g) {
      ierr = ISDestroy(&is);CHKERRQ(ierr);
    }
  }

  /* Count neighs per dof */
  ierr = ISLocalToGlobalMappingGetNodeInfo(el2g,NULL,&ecount,&eneighs);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetNodeInfo(vl2g,NULL,&vcount,&vneighs);CHKERRQ(ierr);

  /* need to remove coarse faces' dofs and coarse edges' dirichlet dofs
     for proper detection of coarse edges' endpoints */
  ierr = PetscBTCreate(ne,&btee);CHKERRQ(ierr);
  for (i=0;i<ne;i++) {
    if ((ecount[i] > 2 && !PetscBTLookup(btbd,i)) || (ecount[i] == 2 && PetscBTLookup(btb,i))) {
      ierr = PetscBTSet(btee,i);CHKERRQ(ierr);
    }
  }
  ierr = PetscMalloc1(ne,&marks);CHKERRQ(ierr);
  if (!conforming) {
    ierr = MatTranspose(lGe,MAT_INITIAL_MATRIX,&lGt);CHKERRQ(ierr);
    ierr = MatGetRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done);CHKERRQ(ierr);
  }
  ierr = MatGetRowIJ(lGe,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done);CHKERRQ(ierr);
  ierr = MatSeqAIJGetArray(lGe,&vals);CHKERRQ(ierr);
  cum  = 0;
  for (i=0;i<ne;i++) {
    /* eliminate rows corresponding to edge dofs belonging to coarse faces */
    if (!PetscBTLookup(btee,i)) {
      marks[cum++] = i;
      continue;
    }
    /* set badly connected edge dofs as primal */
    if (!conforming) {
      if (ii[i+1]-ii[i] != order + 1) { /* every row of G on the coarse edge should list order+1 nodal dofs */
        marks[cum++] = i;
        ierr = PetscBTSet(bte,i);CHKERRQ(ierr);
        for (j=ii[i];j<ii[i+1];j++) {
          ierr = PetscBTSet(btv,jj[j]);CHKERRQ(ierr);
        }
      } else {
        /* every edge dofs should be connected trough a certain number of nodal dofs
           to other edge dofs belonging to coarse edges
           - at most 2 endpoints
           - order-1 interior nodal dofs
           - no undefined nodal dofs (nconn < order)
        */
        PetscInt ends = 0,ints = 0, undef = 0;
        for (j=ii[i];j<ii[i+1];j++) {
          PetscInt v = jj[j],k;
          PetscInt nconn = iit[v+1]-iit[v];
          for (k=iit[v];k<iit[v+1];k++) if (!PetscBTLookup(btee,jjt[k])) nconn--;
          if (nconn > order) ends++;
          else if (nconn == order) ints++;
          else undef++;
        }
        if (undef || ends > 2 || ints != order -1) {
          marks[cum++] = i;
          ierr = PetscBTSet(bte,i);CHKERRQ(ierr);
          for (j=ii[i];j<ii[i+1];j++) {
            ierr = PetscBTSet(btv,jj[j]);CHKERRQ(ierr);
          }
        }
      }
    }
    /* We assume the order on the element edge is ii[i+1]-ii[i]-1 */
    if (!order && ii[i+1] != ii[i]) {
      PetscScalar val = 1./(ii[i+1]-ii[i]-1);
      for (j=ii[i];j<ii[i+1];j++) vals[j] = val;
    }
  }
  ierr = PetscBTDestroy(&btee);CHKERRQ(ierr);
  ierr = MatSeqAIJRestoreArray(lGe,&vals);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(lGe,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done);CHKERRQ(ierr);
  if (!conforming) {
    ierr = MatRestoreRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done);CHKERRQ(ierr);
    ierr = MatDestroy(&lGt);CHKERRQ(ierr);
  }
  ierr = MatZeroRows(lGe,cum,marks,0.,NULL,NULL);CHKERRQ(ierr);

  /* identify splitpoints and corner candidates */
  ierr = MatTranspose(lGe,MAT_INITIAL_MATRIX,&lGt);CHKERRQ(ierr);
  if (print) {
    ierr = PetscObjectSetName((PetscObject)lGe,"edgerestr_lG");CHKERRQ(ierr);
    ierr = MatView(lGe,NULL);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)lGt,"edgerestr_lGt");CHKERRQ(ierr);
    ierr = MatView(lGt,NULL);CHKERRQ(ierr);
  }
  ierr = MatGetRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done);CHKERRQ(ierr);
  ierr = MatSeqAIJGetArray(lGt,&vals);CHKERRQ(ierr);
  for (i=0;i<nv;i++) {
    PetscInt  ord = order, test = ii[i+1]-ii[i], vc = vcount[i];
    PetscBool sneighs = PETSC_TRUE, bdir = PETSC_FALSE;
    if (!order) { /* variable order */
      PetscReal vorder = 0.;

      for (j=ii[i];j<ii[i+1];j++) vorder += PetscRealPart(vals[j]);
      test = PetscFloorReal(vorder+10.*PETSC_SQRT_MACHINE_EPSILON);
      if (vorder-test > PETSC_SQRT_MACHINE_EPSILON) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected value for vorder: %g (%D)",vorder,test);
      ord  = 1;
    }
#if defined(PETSC_USE_DEBUG)
    if (test%ord) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected number of edge dofs %D connected with nodal dof %D with order %D",test,i,ord);
#endif
    for (j=ii[i];j<ii[i+1] && sneighs;j++) {
      if (PetscBTLookup(btbd,jj[j])) {
        bdir = PETSC_TRUE;
        break;
      }
      if (vc != ecount[jj[j]]) {
        sneighs = PETSC_FALSE;
      } else {
        PetscInt k,*vn = vneighs[i], *en = eneighs[jj[j]];
        for (k=0;k<vc;k++) {
          if (vn[k] != en[k]) {
            sneighs = PETSC_FALSE;
            break;
          }
        }
      }
    }
    if (!sneighs || test >= 3*ord || bdir) { /* splitpoints */
      if (print) PetscPrintf(PETSC_COMM_SELF,"SPLITPOINT %D (%D %D %D)\n",i,!sneighs,test >= 3*ord,bdir);
      ierr = PetscBTSet(btv,i);CHKERRQ(ierr);
    } else if (test == ord) {
      if (order == 1 || (!order && ii[i+1]-ii[i] == 1)) {
        if (print) PetscPrintf(PETSC_COMM_SELF,"ENDPOINT %D\n",i);
        ierr = PetscBTSet(btv,i);CHKERRQ(ierr);
      } else {
        if (print) PetscPrintf(PETSC_COMM_SELF,"CORNER CANDIDATE %D\n",i);
        ierr = PetscBTSet(btvcand,i);CHKERRQ(ierr);
      }
    }
  }
  ierr = ISLocalToGlobalMappingRestoreNodeInfo(el2g,NULL,&ecount,&eneighs);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreNodeInfo(vl2g,NULL,&vcount,&vneighs);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&btbd);CHKERRQ(ierr);

  /* a candidate is valid if it is connected to another candidate via a non-primal edge dof */
  if (order != 1) {
    if (print) PetscPrintf(PETSC_COMM_SELF,"INSPECTING CANDIDATES\n");
    ierr = MatGetRowIJ(lGe,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done);CHKERRQ(ierr);
    for (i=0;i<nv;i++) {
      if (PetscBTLookup(btvcand,i)) {
        PetscBool found = PETSC_FALSE;
        for (j=ii[i];j<ii[i+1] && !found;j++) {
          PetscInt k,e = jj[j];
          if (PetscBTLookup(bte,e)) continue;
          for (k=iit[e];k<iit[e+1];k++) {
            PetscInt v = jjt[k];
            if (v != i && PetscBTLookup(btvcand,v)) {
              found = PETSC_TRUE;
              break;
            }
          }
        }
        if (!found) {
          if (print) PetscPrintf(PETSC_COMM_SELF,"  CANDIDATE %D CLEARED\n",i);
          ierr = PetscBTClear(btvcand,i);CHKERRQ(ierr);
        } else {
          if (print) PetscPrintf(PETSC_COMM_SELF,"  CANDIDATE %D ACCEPTED\n",i);
        }
      }
    }
    ierr = MatRestoreRowIJ(lGe,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done);CHKERRQ(ierr);
  }
  ierr = MatSeqAIJRestoreArray(lGt,&vals);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done);CHKERRQ(ierr);
  ierr = MatDestroy(&lGe);CHKERRQ(ierr);

  /* Get the local G^T explicitly */
  ierr = MatDestroy(&lGt);CHKERRQ(ierr);
  ierr = MatTranspose(lG,MAT_INITIAL_MATRIX,&lGt);CHKERRQ(ierr);
  ierr = MatSetOption(lGt,MAT_KEEP_NONZERO_PATTERN,PETSC_FALSE);CHKERRQ(ierr);

  /* Mark interior nodal dofs */
  ierr = ISLocalToGlobalMappingGetInfo(vl2g,&n_neigh,&neigh,&n_shared,&shared);CHKERRQ(ierr);
  ierr = PetscBTCreate(nv,&btvi);CHKERRQ(ierr);
  for (i=1;i<n_neigh;i++) {
    for (j=0;j<n_shared[i];j++) {
      ierr = PetscBTSet(btvi,shared[i][j]);CHKERRQ(ierr);
    }
  }
  ierr = ISLocalToGlobalMappingRestoreInfo(vl2g,&n_neigh,&neigh,&n_shared,&shared);CHKERRQ(ierr);

  /* communicate corners and splitpoints */
  ierr = PetscMalloc1(nv,&vmarks);CHKERRQ(ierr);
  ierr = PetscArrayzero(sfvleaves,nv);CHKERRQ(ierr);
  ierr = PetscArrayzero(sfvroots,Lv);CHKERRQ(ierr);
  for (i=0;i<nv;i++) if (PetscUnlikely(PetscBTLookup(btv,i))) sfvleaves[i] = 1;

  if (print) {
    IS tbz;

    cum = 0;
    for (i=0;i<nv;i++)
      if (sfvleaves[i])
        vmarks[cum++] = i;

    ierr = ISCreateGeneral(PETSC_COMM_SELF,cum,vmarks,PETSC_COPY_VALUES,&tbz);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)tbz,"corners_to_be_zeroed_local");CHKERRQ(ierr);
    ierr = ISView(tbz,NULL);CHKERRQ(ierr);
    ierr = ISDestroy(&tbz);CHKERRQ(ierr);
  }

  ierr = PetscSFReduceBegin(sfv,MPIU_INT,sfvleaves,sfvroots,MPI_SUM);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(sfv,MPIU_INT,sfvleaves,sfvroots,MPI_SUM);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sfv,MPIU_INT,sfvroots,sfvleaves);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sfv,MPIU_INT,sfvroots,sfvleaves);CHKERRQ(ierr);

  /* Zero rows of lGt corresponding to identified corners
     and interior nodal dofs */
  cum = 0;
  for (i=0;i<nv;i++) {
    if (sfvleaves[i]) {
      vmarks[cum++] = i;
      ierr = PetscBTSet(btv,i);CHKERRQ(ierr);
    }
    if (!PetscBTLookup(btvi,i)) vmarks[cum++] = i;
  }
  ierr = PetscBTDestroy(&btvi);CHKERRQ(ierr);
  if (print) {
    IS tbz;

    ierr = ISCreateGeneral(PETSC_COMM_SELF,cum,vmarks,PETSC_COPY_VALUES,&tbz);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)tbz,"corners_to_be_zeroed_with_interior");CHKERRQ(ierr);
    ierr = ISView(tbz,NULL);CHKERRQ(ierr);
    ierr = ISDestroy(&tbz);CHKERRQ(ierr);
  }
  ierr = MatZeroRows(lGt,cum,vmarks,0.,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscFree(vmarks);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfv);CHKERRQ(ierr);
  ierr = PetscFree2(sfvleaves,sfvroots);CHKERRQ(ierr);

  /* Recompute G */
  ierr = MatDestroy(&lG);CHKERRQ(ierr);
  ierr = MatTranspose(lGt,MAT_INITIAL_MATRIX,&lG);CHKERRQ(ierr);
  if (print) {
    ierr = PetscObjectSetName((PetscObject)lG,"used_lG");CHKERRQ(ierr);
    ierr = MatView(lG,NULL);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)lGt,"used_lGt");CHKERRQ(ierr);
    ierr = MatView(lGt,NULL);CHKERRQ(ierr);
  }

  /* Get primal dofs (if any) */
  cum = 0;
  for (i=0;i<ne;i++) {
    if (PetscUnlikely(PetscBTLookup(bte,i))) marks[cum++] = i;
  }
  if (fl2g) {
    ierr = ISLocalToGlobalMappingApply(fl2g,cum,marks,marks);CHKERRQ(ierr);
  }
  ierr = ISCreateGeneral(comm,cum,marks,PETSC_COPY_VALUES,&primals);CHKERRQ(ierr);
  if (print) {
    ierr = PetscObjectSetName((PetscObject)primals,"prescribed_primal_dofs");CHKERRQ(ierr);
    ierr = ISView(primals,NULL);CHKERRQ(ierr);
  }
  ierr = PetscBTDestroy(&bte);CHKERRQ(ierr);
  /* TODO: what if the user passed in some of them ?  */
  ierr = PCBDDCSetPrimalVerticesLocalIS(pc,primals);CHKERRQ(ierr);
  ierr = ISDestroy(&primals);CHKERRQ(ierr);

  /* Compute edge connectivity */
  ierr = PetscObjectSetOptionsPrefix((PetscObject)lG,"econn_");CHKERRQ(ierr);
  ierr = MatMatMultSymbolic(lG,lGt,PETSC_DEFAULT,&conn);CHKERRQ(ierr);
  ierr = MatGetRowIJ(conn,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done);CHKERRQ(ierr);
  if (fl2g) {
    PetscBT   btf;
    PetscInt  *iia,*jja,*iiu,*jju;
    PetscBool rest = PETSC_FALSE,free = PETSC_FALSE;

    /* create CSR for all local dofs */
    ierr = PetscMalloc1(n+1,&iia);CHKERRQ(ierr);
    if (pcbddc->mat_graph->nvtxs_csr) { /* the user has passed in a CSR graph */
      if (pcbddc->mat_graph->nvtxs_csr != n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Invalid size of CSR graph %D. Should be %D",pcbddc->mat_graph->nvtxs_csr,n);
      iiu = pcbddc->mat_graph->xadj;
      jju = pcbddc->mat_graph->adjncy;
    } else if (pcbddc->use_local_adj) {
      rest = PETSC_TRUE;
      ierr = MatGetRowIJ(matis->A,0,PETSC_TRUE,PETSC_FALSE,&i,(const PetscInt**)&iiu,(const PetscInt**)&jju,&done);CHKERRQ(ierr);
    } else {
      free   = PETSC_TRUE;
      ierr   = PetscMalloc2(n+1,&iiu,n,&jju);CHKERRQ(ierr);
      iiu[0] = 0;
      for (i=0;i<n;i++) {
        iiu[i+1] = i+1;
        jju[i]   = -1;
      }
    }

    /* import sizes of CSR */
    iia[0] = 0;
    for (i=0;i<n;i++) iia[i+1] = iiu[i+1]-iiu[i];

    /* overwrite entries corresponding to the Nedelec field */
    ierr = PetscBTCreate(n,&btf);CHKERRQ(ierr);
    ierr = ISGetIndices(nedfieldlocal,&idxs);CHKERRQ(ierr);
    for (i=0;i<ne;i++) {
      ierr = PetscBTSet(btf,idxs[i]);CHKERRQ(ierr);
      iia[idxs[i]+1] = ii[i+1]-ii[i];
    }

    /* iia in CSR */
    for (i=0;i<n;i++) iia[i+1] += iia[i];

    /* jja in CSR */
    ierr = PetscMalloc1(iia[n],&jja);CHKERRQ(ierr);
    for (i=0;i<n;i++)
      if (!PetscBTLookup(btf,i))
        for (j=0;j<iiu[i+1]-iiu[i];j++)
          jja[iia[i]+j] = jju[iiu[i]+j];

    /* map edge dofs connectivity */
    if (jj) {
      ierr = ISLocalToGlobalMappingApply(fl2g,ii[ne],jj,(PetscInt *)jj);CHKERRQ(ierr);
      for (i=0;i<ne;i++) {
        PetscInt e = idxs[i];
        for (j=0;j<ii[i+1]-ii[i];j++) jja[iia[e]+j] = jj[ii[i]+j];
      }
    }
    ierr = ISRestoreIndices(nedfieldlocal,&idxs);CHKERRQ(ierr);
    ierr = PCBDDCSetLocalAdjacencyGraph(pc,n,iia,jja,PETSC_OWN_POINTER);CHKERRQ(ierr);
    if (rest) {
      ierr = MatRestoreRowIJ(matis->A,0,PETSC_TRUE,PETSC_FALSE,&i,(const PetscInt**)&iiu,(const PetscInt**)&jju,&done);CHKERRQ(ierr);
    }
    if (free) {
      ierr = PetscFree2(iiu,jju);CHKERRQ(ierr);
    }
    ierr = PetscBTDestroy(&btf);CHKERRQ(ierr);
  } else {
    ierr = PCBDDCSetLocalAdjacencyGraph(pc,n,ii,jj,PETSC_USE_POINTER);CHKERRQ(ierr);
  }

  /* Analyze interface for edge dofs */
  ierr = PCBDDCAnalyzeInterface(pc);CHKERRQ(ierr);
  pcbddc->mat_graph->twodim = PETSC_FALSE;

  /* Get coarse edges in the edge space */
  ierr = PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,NULL,NULL,&nee,&alleedges,&allprimals);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(conn,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done);CHKERRQ(ierr);

  if (fl2g) {
    ierr = ISGlobalToLocalMappingApplyIS(fl2g,IS_GTOLM_DROP,allprimals,&primals);CHKERRQ(ierr);
    ierr = PetscMalloc1(nee,&eedges);CHKERRQ(ierr);
    for (i=0;i<nee;i++) {
      ierr = ISGlobalToLocalMappingApplyIS(fl2g,IS_GTOLM_DROP,alleedges[i],&eedges[i]);CHKERRQ(ierr);
    }
  } else {
    eedges  = alleedges;
    primals = allprimals;
  }

  /* Mark fine edge dofs with their coarse edge id */
  ierr = PetscArrayzero(marks,ne);CHKERRQ(ierr);
  ierr = ISGetLocalSize(primals,&cum);CHKERRQ(ierr);
  ierr = ISGetIndices(primals,&idxs);CHKERRQ(ierr);
  for (i=0;i<cum;i++) marks[idxs[i]] = nee+1;
  ierr = ISRestoreIndices(primals,&idxs);CHKERRQ(ierr);
  if (print) {
    ierr = PetscObjectSetName((PetscObject)primals,"obtained_primal_dofs");CHKERRQ(ierr);
    ierr = ISView(primals,NULL);CHKERRQ(ierr);
  }

  maxsize = 0;
  for (i=0;i<nee;i++) {
    PetscInt size,mark = i+1;

    ierr = ISGetLocalSize(eedges[i],&size);CHKERRQ(ierr);
    ierr = ISGetIndices(eedges[i],&idxs);CHKERRQ(ierr);
    for (j=0;j<size;j++) marks[idxs[j]] = mark;
    ierr = ISRestoreIndices(eedges[i],&idxs);CHKERRQ(ierr);
    maxsize = PetscMax(maxsize,size);
  }

  /* Find coarse edge endpoints */
  ierr = MatGetRowIJ(lG,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done);CHKERRQ(ierr);
  ierr = MatGetRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done);CHKERRQ(ierr);
  for (i=0;i<nee;i++) {
    PetscInt mark = i+1,size;

    ierr = ISGetLocalSize(eedges[i],&size);CHKERRQ(ierr);
    if (!size && nedfieldlocal) continue;
    if (!size) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected zero sized edge %D",i);
    ierr = ISGetIndices(eedges[i],&idxs);CHKERRQ(ierr);
    if (print) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"ENDPOINTS ANALYSIS EDGE %D\n",i);CHKERRQ(ierr);
      ierr = ISView(eedges[i],NULL);CHKERRQ(ierr);
    }
    for (j=0;j<size;j++) {
      PetscInt k, ee = idxs[j];
      if (print) PetscPrintf(PETSC_COMM_SELF,"  idx %D\n",ee);
      for (k=ii[ee];k<ii[ee+1];k++) {
        if (print) PetscPrintf(PETSC_COMM_SELF,"    inspect %D\n",jj[k]);
        if (PetscBTLookup(btv,jj[k])) {
          if (print) PetscPrintf(PETSC_COMM_SELF,"      corner found (already set) %D\n",jj[k]);
        } else if (PetscBTLookup(btvcand,jj[k])) { /* is it ok? */
          PetscInt  k2;
          PetscBool corner = PETSC_FALSE;
          for (k2 = iit[jj[k]];k2 < iit[jj[k]+1];k2++) {
            if (print) PetscPrintf(PETSC_COMM_SELF,"        INSPECTING %D: mark %D (ref mark %D), boundary %D\n",jjt[k2],marks[jjt[k2]],mark,!!PetscBTLookup(btb,jjt[k2]));
            /* it's a corner if either is connected with an edge dof belonging to a different cc or
               if the edge dof lie on the natural part of the boundary */
            if ((marks[jjt[k2]] && marks[jjt[k2]] != mark) || (!marks[jjt[k2]] && PetscBTLookup(btb,jjt[k2]))) {
              corner = PETSC_TRUE;
              break;
            }
          }
          if (corner) { /* found the nodal dof corresponding to the endpoint of the edge */
            if (print) PetscPrintf(PETSC_COMM_SELF,"        corner found %D\n",jj[k]);
            ierr = PetscBTSet(btv,jj[k]);CHKERRQ(ierr);
          } else {
            if (print) PetscPrintf(PETSC_COMM_SELF,"        no corners found\n");
          }
        }
      }
    }
    ierr = ISRestoreIndices(eedges[i],&idxs);CHKERRQ(ierr);
  }
  ierr = MatRestoreRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(lG,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&btb);CHKERRQ(ierr);

  /* Reset marked primal dofs */
  ierr = ISGetLocalSize(primals,&cum);CHKERRQ(ierr);
  ierr = ISGetIndices(primals,&idxs);CHKERRQ(ierr);
  for (i=0;i<cum;i++) marks[idxs[i]] = 0;
  ierr = ISRestoreIndices(primals,&idxs);CHKERRQ(ierr);

  /* Now use the initial lG */
  ierr = MatDestroy(&lG);CHKERRQ(ierr);
  ierr = MatDestroy(&lGt);CHKERRQ(ierr);
  lG   = lGinit;
  ierr = MatTranspose(lG,MAT_INITIAL_MATRIX,&lGt);CHKERRQ(ierr);

  /* Compute extended cols indices */
  ierr = PetscBTCreate(nv,&btvc);CHKERRQ(ierr);
  ierr = PetscBTCreate(nee,&bter);CHKERRQ(ierr);
  ierr = MatGetRowIJ(lG,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done);CHKERRQ(ierr);
  ierr = MatSeqAIJGetMaxRowNonzeros(lG,&i);CHKERRQ(ierr);
  i   *= maxsize;
  ierr = PetscCalloc1(nee,&extcols);CHKERRQ(ierr);
  ierr = PetscMalloc2(i,&extrow,i,&gidxs);CHKERRQ(ierr);
  eerr = PETSC_FALSE;
  for (i=0;i<nee;i++) {
    PetscInt size,found = 0;

    cum  = 0;
    ierr = ISGetLocalSize(eedges[i],&size);CHKERRQ(ierr);
    if (!size && nedfieldlocal) continue;
    if (!size) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected zero sized edge %D",i);
    ierr = ISGetIndices(eedges[i],&idxs);CHKERRQ(ierr);
    ierr = PetscBTMemzero(nv,btvc);CHKERRQ(ierr);
    for (j=0;j<size;j++) {
      PetscInt k,ee = idxs[j];
      for (k=ii[ee];k<ii[ee+1];k++) {
        PetscInt vv = jj[k];
        if (!PetscBTLookup(btv,vv)) extrow[cum++] = vv;
        else if (!PetscBTLookupSet(btvc,vv)) found++;
      }
    }
    ierr = ISRestoreIndices(eedges[i],&idxs);CHKERRQ(ierr);
    ierr = PetscSortRemoveDupsInt(&cum,extrow);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingApply(vl2g,cum,extrow,gidxs);CHKERRQ(ierr);
    ierr = PetscSortIntWithArray(cum,gidxs,extrow);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,cum,extrow,PETSC_COPY_VALUES,&extcols[i]);CHKERRQ(ierr);
    /* it may happen that endpoints are not defined at this point
       if it is the case, mark this edge for a second pass */
    if (cum != size -1 || found != 2) {
      ierr = PetscBTSet(bter,i);CHKERRQ(ierr);
      if (print) {
        ierr = PetscObjectSetName((PetscObject)eedges[i],"error_edge");CHKERRQ(ierr);
        ierr = ISView(eedges[i],NULL);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)extcols[i],"error_extcol");CHKERRQ(ierr);
        ierr = ISView(extcols[i],NULL);CHKERRQ(ierr);
      }
      eerr = PETSC_TRUE;
    }
  }
  /* if (eerr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected SIZE OF EDGE > EXTCOL FIRST PASS"); */
  ierr = MPIU_Allreduce(&eerr,&done,1,MPIU_BOOL,MPI_LOR,comm);CHKERRQ(ierr);
  if (done) {
    PetscInt *newprimals;

    ierr = PetscMalloc1(ne,&newprimals);CHKERRQ(ierr);
    ierr = ISGetLocalSize(primals,&cum);CHKERRQ(ierr);
    ierr = ISGetIndices(primals,&idxs);CHKERRQ(ierr);
    ierr = PetscArraycpy(newprimals,idxs,cum);CHKERRQ(ierr);
    ierr = ISRestoreIndices(primals,&idxs);CHKERRQ(ierr);
    ierr = MatGetRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done);CHKERRQ(ierr);
    if (print) PetscPrintf(PETSC_COMM_SELF,"DOING SECOND PASS (eerr %D)\n",eerr);
    for (i=0;i<nee;i++) {
      PetscBool has_candidates = PETSC_FALSE;
      if (PetscBTLookup(bter,i)) {
        PetscInt size,mark = i+1;

        ierr = ISGetLocalSize(eedges[i],&size);CHKERRQ(ierr);
        ierr = ISGetIndices(eedges[i],&idxs);CHKERRQ(ierr);
        /* for (j=0;j<size;j++) newprimals[cum++] = idxs[j]; */
        for (j=0;j<size;j++) {
          PetscInt k,ee = idxs[j];
          if (print) PetscPrintf(PETSC_COMM_SELF,"Inspecting edge dof %D [%D %D)\n",ee,ii[ee],ii[ee+1]);
          for (k=ii[ee];k<ii[ee+1];k++) {
            /* set all candidates located on the edge as corners */
            if (PetscBTLookup(btvcand,jj[k])) {
              PetscInt k2,vv = jj[k];
              has_candidates = PETSC_TRUE;
              if (print) PetscPrintf(PETSC_COMM_SELF,"  Candidate set to vertex %D\n",vv);
              ierr = PetscBTSet(btv,vv);CHKERRQ(ierr);
              /* set all edge dofs connected to candidate as primals */
              for (k2=iit[vv];k2<iit[vv+1];k2++) {
                if (marks[jjt[k2]] == mark) {
                  PetscInt k3,ee2 = jjt[k2];
                  if (print) PetscPrintf(PETSC_COMM_SELF,"    Connected edge dof set to primal %D\n",ee2);
                  newprimals[cum++] = ee2;
                  /* finally set the new corners */
                  for (k3=ii[ee2];k3<ii[ee2+1];k3++) {
                    if (print) PetscPrintf(PETSC_COMM_SELF,"      Connected nodal dof set to vertex %D\n",jj[k3]);
                    ierr = PetscBTSet(btv,jj[k3]);CHKERRQ(ierr);
                  }
                }
              }
            } else {
              if (print) PetscPrintf(PETSC_COMM_SELF,"  Not a candidate vertex %D\n",jj[k]);
            }
          }
        }
        if (!has_candidates) { /* circular edge */
          PetscInt k, ee = idxs[0],*tmarks;

          ierr = PetscCalloc1(ne,&tmarks);CHKERRQ(ierr);
          if (print) PetscPrintf(PETSC_COMM_SELF,"  Circular edge %D\n",i);
          for (k=ii[ee];k<ii[ee+1];k++) {
            PetscInt k2;
            if (print) PetscPrintf(PETSC_COMM_SELF,"    Set to corner %D\n",jj[k]);
            ierr = PetscBTSet(btv,jj[k]);CHKERRQ(ierr);
            for (k2=iit[jj[k]];k2<iit[jj[k]+1];k2++) tmarks[jjt[k2]]++;
          }
          for (j=0;j<size;j++) {
            if (tmarks[idxs[j]] > 1) {
              if (print) PetscPrintf(PETSC_COMM_SELF,"  Edge dof set to primal %D\n",idxs[j]);
              newprimals[cum++] = idxs[j];
            }
          }
          ierr = PetscFree(tmarks);CHKERRQ(ierr);
        }
        ierr = ISRestoreIndices(eedges[i],&idxs);CHKERRQ(ierr);
      }
      ierr = ISDestroy(&extcols[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(extcols);CHKERRQ(ierr);
    ierr = MatRestoreRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done);CHKERRQ(ierr);
    ierr = PetscSortRemoveDupsInt(&cum,newprimals);CHKERRQ(ierr);
    if (fl2g) {
      ierr = ISLocalToGlobalMappingApply(fl2g,cum,newprimals,newprimals);CHKERRQ(ierr);
      ierr = ISDestroy(&primals);CHKERRQ(ierr);
      for (i=0;i<nee;i++) {
        ierr = ISDestroy(&eedges[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(eedges);CHKERRQ(ierr);
    }
    ierr = PCBDDCGraphRestoreCandidatesIS(pcbddc->mat_graph,NULL,NULL,&nee,&alleedges,&allprimals);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm,cum,newprimals,PETSC_COPY_VALUES,&primals);CHKERRQ(ierr);
    ierr = PetscFree(newprimals);CHKERRQ(ierr);
    ierr = PCBDDCSetPrimalVerticesLocalIS(pc,primals);CHKERRQ(ierr);
    ierr = ISDestroy(&primals);CHKERRQ(ierr);
    ierr = PCBDDCAnalyzeInterface(pc);CHKERRQ(ierr);
    pcbddc->mat_graph->twodim = PETSC_FALSE;
    ierr = PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,NULL,NULL,&nee,&alleedges,&allprimals);CHKERRQ(ierr);
    if (fl2g) {
      ierr = ISGlobalToLocalMappingApplyIS(fl2g,IS_GTOLM_DROP,allprimals,&primals);CHKERRQ(ierr);
      ierr = PetscMalloc1(nee,&eedges);CHKERRQ(ierr);
      for (i=0;i<nee;i++) {
        ierr = ISGlobalToLocalMappingApplyIS(fl2g,IS_GTOLM_DROP,alleedges[i],&eedges[i]);CHKERRQ(ierr);
      }
    } else {
      eedges  = alleedges;
      primals = allprimals;
    }
    ierr = PetscCalloc1(nee,&extcols);CHKERRQ(ierr);

    /* Mark again */
    ierr = PetscArrayzero(marks,ne);CHKERRQ(ierr);
    for (i=0;i<nee;i++) {
      PetscInt size,mark = i+1;

      ierr = ISGetLocalSize(eedges[i],&size);CHKERRQ(ierr);
      ierr = ISGetIndices(eedges[i],&idxs);CHKERRQ(ierr);
      for (j=0;j<size;j++) marks[idxs[j]] = mark;
      ierr = ISRestoreIndices(eedges[i],&idxs);CHKERRQ(ierr);
    }
    if (print) {
      ierr = PetscObjectSetName((PetscObject)primals,"obtained_primal_dofs_secondpass");CHKERRQ(ierr);
      ierr = ISView(primals,NULL);CHKERRQ(ierr);
    }

    /* Recompute extended cols */
    eerr = PETSC_FALSE;
    for (i=0;i<nee;i++) {
      PetscInt size;

      cum  = 0;
      ierr = ISGetLocalSize(eedges[i],&size);CHKERRQ(ierr);
      if (!size && nedfieldlocal) continue;
      if (!size) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected zero sized edge %D",i);
      ierr = ISGetIndices(eedges[i],&idxs);CHKERRQ(ierr);
      for (j=0;j<size;j++) {
        PetscInt k,ee = idxs[j];
        for (k=ii[ee];k<ii[ee+1];k++) if (!PetscBTLookup(btv,jj[k])) extrow[cum++] = jj[k];
      }
      ierr = ISRestoreIndices(eedges[i],&idxs);CHKERRQ(ierr);
      ierr = PetscSortRemoveDupsInt(&cum,extrow);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingApply(vl2g,cum,extrow,gidxs);CHKERRQ(ierr);
      ierr = PetscSortIntWithArray(cum,gidxs,extrow);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF,cum,extrow,PETSC_COPY_VALUES,&extcols[i]);CHKERRQ(ierr);
      if (cum != size -1) {
        if (print) {
          ierr = PetscObjectSetName((PetscObject)eedges[i],"error_edge_secondpass");CHKERRQ(ierr);
          ierr = ISView(eedges[i],NULL);CHKERRQ(ierr);
          ierr = PetscObjectSetName((PetscObject)extcols[i],"error_extcol_secondpass");CHKERRQ(ierr);
          ierr = ISView(extcols[i],NULL);CHKERRQ(ierr);
        }
        eerr = PETSC_TRUE;
      }
    }
  }
  ierr = MatRestoreRowIJ(lG,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done);CHKERRQ(ierr);
  ierr = PetscFree2(extrow,gidxs);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&bter);CHKERRQ(ierr);
  if (print) { ierr = PCBDDCGraphASCIIView(pcbddc->mat_graph,5,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr); }
  /* an error should not occur at this point */
  if (eerr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected SIZE OF EDGE > EXTCOL SECOND PASS");

  /* Check the number of endpoints */
  ierr = MatGetRowIJ(lG,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done);CHKERRQ(ierr);
  ierr = PetscMalloc1(2*nee,&corners);CHKERRQ(ierr);
  ierr = PetscMalloc1(nee,&cedges);CHKERRQ(ierr);
  for (i=0;i<nee;i++) {
    PetscInt size, found = 0, gc[2];

    /* init with defaults */
    cedges[i] = corners[i*2] = corners[i*2+1] = -1;
    ierr = ISGetLocalSize(eedges[i],&size);CHKERRQ(ierr);
    if (!size && nedfieldlocal) continue;
    if (!size) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected zero sized edge %D",i);
    ierr = ISGetIndices(eedges[i],&idxs);CHKERRQ(ierr);
    ierr = PetscBTMemzero(nv,btvc);CHKERRQ(ierr);
    for (j=0;j<size;j++) {
      PetscInt k,ee = idxs[j];
      for (k=ii[ee];k<ii[ee+1];k++) {
        PetscInt vv = jj[k];
        if (PetscBTLookup(btv,vv) && !PetscBTLookupSet(btvc,vv)) {
          if (found == 2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Found more then two corners for edge %D",i);
          corners[i*2+found++] = vv;
        }
      }
    }
    if (found != 2) {
      PetscInt e;
      if (fl2g) {
        ierr = ISLocalToGlobalMappingApply(fl2g,1,idxs,&e);CHKERRQ(ierr);
      } else {
        e = idxs[0];
      }
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Found %D corners for edge %D (astart %D, estart %D)",found,i,e,idxs[0]);
    }

    /* get primal dof index on this coarse edge */
    ierr = ISLocalToGlobalMappingApply(vl2g,2,corners+2*i,gc);CHKERRQ(ierr);
    if (gc[0] > gc[1]) {
      PetscInt swap  = corners[2*i];
      corners[2*i]   = corners[2*i+1];
      corners[2*i+1] = swap;
    }
    cedges[i] = idxs[size-1];
    ierr = ISRestoreIndices(eedges[i],&idxs);CHKERRQ(ierr);
    if (print) PetscPrintf(PETSC_COMM_SELF,"EDGE %D: ce %D, corners (%D,%D)\n",i,cedges[i],corners[2*i],corners[2*i+1]);
  }
  ierr = MatRestoreRowIJ(lG,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&btvc);CHKERRQ(ierr);

#if defined(PETSC_USE_DEBUG)
  /* Inspects columns of lG (rows of lGt) and make sure the change of basis will
     not interfere with neighbouring coarse edges */
  ierr = PetscMalloc1(nee+1,&emarks);CHKERRQ(ierr);
  ierr = MatGetRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done);CHKERRQ(ierr);
  for (i=0;i<nv;i++) {
    PetscInt emax = 0,eemax = 0;

    if (ii[i+1]==ii[i] || PetscBTLookup(btv,i)) continue;
    ierr = PetscArrayzero(emarks,nee+1);CHKERRQ(ierr);
    for (j=ii[i];j<ii[i+1];j++) emarks[marks[jj[j]]]++;
    for (j=1;j<nee+1;j++) {
      if (emax < emarks[j]) {
        emax = emarks[j];
        eemax = j;
      }
    }
    /* not relevant for edges */
    if (!eemax) continue;

    for (j=ii[i];j<ii[i+1];j++) {
      if (marks[jj[j]] && marks[jj[j]] != eemax) {
        SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_SUP,"Found 2 coarse edges (id %D and %D) connected through the %D nodal dof at edge dof %D",marks[jj[j]]-1,eemax,i,jj[j]);
      }
    }
  }
  ierr = PetscFree(emarks);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done);CHKERRQ(ierr);
#endif

  /* Compute extended rows indices for edge blocks of the change of basis */
  ierr = MatGetRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done);CHKERRQ(ierr);
  ierr = MatSeqAIJGetMaxRowNonzeros(lGt,&extmem);CHKERRQ(ierr);
  extmem *= maxsize;
  ierr = PetscMalloc1(extmem*nee,&extrow);CHKERRQ(ierr);
  ierr = PetscMalloc1(nee,&extrows);CHKERRQ(ierr);
  ierr = PetscCalloc1(nee,&extrowcum);CHKERRQ(ierr);
  for (i=0;i<nv;i++) {
    PetscInt mark = 0,size,start;

    if (ii[i+1]==ii[i] || PetscBTLookup(btv,i)) continue;
    for (j=ii[i];j<ii[i+1];j++)
      if (marks[jj[j]] && !mark)
        mark = marks[jj[j]];

    /* not relevant */
    if (!mark) continue;

    /* import extended row */
    mark--;
    start = mark*extmem+extrowcum[mark];
    size = ii[i+1]-ii[i];
    if (extrowcum[mark] + size > extmem) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not enough memory allocated %D > %D",extrowcum[mark] + size,extmem);
    ierr = PetscArraycpy(extrow+start,jj+ii[i],size);CHKERRQ(ierr);
    extrowcum[mark] += size;
  }
  ierr = MatRestoreRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done);CHKERRQ(ierr);
  ierr = MatDestroy(&lGt);CHKERRQ(ierr);
  ierr = PetscFree(marks);CHKERRQ(ierr);

  /* Compress extrows */
  cum  = 0;
  for (i=0;i<nee;i++) {
    PetscInt size = extrowcum[i],*start = extrow + i*extmem;
    ierr = PetscSortRemoveDupsInt(&size,start);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,size,start,PETSC_USE_POINTER,&extrows[i]);CHKERRQ(ierr);
    cum  = PetscMax(cum,size);
  }
  ierr = PetscFree(extrowcum);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&btv);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&btvcand);CHKERRQ(ierr);

  /* Workspace for lapack inner calls and VecSetValues */
  ierr = PetscMalloc2((5+cum+maxsize)*maxsize,&work,maxsize,&rwork);CHKERRQ(ierr);

  /* Create change of basis matrix (preallocation can be improved) */
  ierr = MatCreate(comm,&T);CHKERRQ(ierr);
  ierr = MatSetSizes(T,pc->pmat->rmap->n,pc->pmat->rmap->n,
                       pc->pmat->rmap->N,pc->pmat->rmap->N);CHKERRQ(ierr);
  ierr = MatSetType(T,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(T,10,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(T,10,NULL,10,NULL);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(T,al2g,al2g);CHKERRQ(ierr);
  ierr = MatSetOption(T,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatSetOption(T,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&al2g);CHKERRQ(ierr);

  /* Defaults to identity */
  ierr = MatCreateVecs(pc->pmat,&tvec,NULL);CHKERRQ(ierr);
  ierr = VecSet(tvec,1.0);CHKERRQ(ierr);
  ierr = MatDiagonalSet(T,tvec,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecDestroy(&tvec);CHKERRQ(ierr);

  /* Create discrete gradient for the coarser level if needed */
  ierr = MatDestroy(&pcbddc->nedcG);CHKERRQ(ierr);
  ierr = ISDestroy(&pcbddc->nedclocal);CHKERRQ(ierr);
  if (pcbddc->current_level < pcbddc->max_levels) {
    ISLocalToGlobalMapping cel2g,cvl2g;
    IS                     wis,gwis;
    PetscInt               cnv,cne;

    ierr = ISCreateGeneral(comm,nee,cedges,PETSC_COPY_VALUES,&wis);CHKERRQ(ierr);
    if (fl2g) {
      ierr = ISLocalToGlobalMappingApplyIS(fl2g,wis,&pcbddc->nedclocal);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectReference((PetscObject)wis);CHKERRQ(ierr);
      pcbddc->nedclocal = wis;
    }
    ierr = ISLocalToGlobalMappingApplyIS(el2g,wis,&gwis);CHKERRQ(ierr);
    ierr = ISDestroy(&wis);CHKERRQ(ierr);
    ierr = ISRenumber(gwis,NULL,&cne,&wis);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreateIS(wis,&cel2g);CHKERRQ(ierr);
    ierr = ISDestroy(&wis);CHKERRQ(ierr);
    ierr = ISDestroy(&gwis);CHKERRQ(ierr);

    ierr = ISCreateGeneral(comm,2*nee,corners,PETSC_USE_POINTER,&wis);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingApplyIS(vl2g,wis,&gwis);CHKERRQ(ierr);
    ierr = ISDestroy(&wis);CHKERRQ(ierr);
    ierr = ISRenumber(gwis,NULL,&cnv,&wis);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreateIS(wis,&cvl2g);CHKERRQ(ierr);
    ierr = ISDestroy(&wis);CHKERRQ(ierr);
    ierr = ISDestroy(&gwis);CHKERRQ(ierr);

    ierr = MatCreate(comm,&pcbddc->nedcG);CHKERRQ(ierr);
    ierr = MatSetSizes(pcbddc->nedcG,PETSC_DECIDE,PETSC_DECIDE,cne,cnv);CHKERRQ(ierr);
    ierr = MatSetType(pcbddc->nedcG,MATAIJ);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(pcbddc->nedcG,2,NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(pcbddc->nedcG,2,NULL,2,NULL);CHKERRQ(ierr);
    ierr = MatSetLocalToGlobalMapping(pcbddc->nedcG,cel2g,cvl2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&cel2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&cvl2g);CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingDestroy(&vl2g);CHKERRQ(ierr);

#if defined(PRINT_GDET)
  inc = 0;
  lev = pcbddc->current_level;
#endif

  /* Insert values in the change of basis matrix */
  for (i=0;i<nee;i++) {
    Mat         Gins = NULL, GKins = NULL;
    IS          cornersis = NULL;
    PetscScalar cvals[2];

    if (pcbddc->nedcG) {
      ierr = ISCreateGeneral(PETSC_COMM_SELF,2,corners+2*i,PETSC_USE_POINTER,&cornersis);CHKERRQ(ierr);
    }
    ierr = PCBDDCComputeNedelecChangeEdge(lG,eedges[i],extrows[i],extcols[i],cornersis,&Gins,&GKins,cvals,work,rwork);CHKERRQ(ierr);
    if (Gins && GKins) {
      const PetscScalar *data;
      const PetscInt    *rows,*cols;
      PetscInt          nrh,nch,nrc,ncc;

      ierr = ISGetIndices(eedges[i],&cols);CHKERRQ(ierr);
      /* H1 */
      ierr = ISGetIndices(extrows[i],&rows);CHKERRQ(ierr);
      ierr = MatGetSize(Gins,&nrh,&nch);CHKERRQ(ierr);
      ierr = MatDenseGetArrayRead(Gins,&data);CHKERRQ(ierr);
      ierr = MatSetValuesLocal(T,nrh,rows,nch,cols,data,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatDenseRestoreArrayRead(Gins,&data);CHKERRQ(ierr);
      ierr = ISRestoreIndices(extrows[i],&rows);CHKERRQ(ierr);
      /* complement */
      ierr = MatGetSize(GKins,&nrc,&ncc);CHKERRQ(ierr);
      if (!ncc) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Constant function has not been generated for coarse edge %D",i);
      if (ncc + nch != nrc) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB,"The sum of the number of columns of GKins %D and Gins %D does not match %D for coarse edge %D",ncc,nch,nrc,i);
      if (ncc != 1 && pcbddc->nedcG) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot generate the coarse discrete gradient for coarse edge %D with ncc %D",i,ncc);
      ierr = MatDenseGetArrayRead(GKins,&data);CHKERRQ(ierr);
      ierr = MatSetValuesLocal(T,nrc,cols,ncc,cols+nch,data,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatDenseRestoreArrayRead(GKins,&data);CHKERRQ(ierr);

      /* coarse discrete gradient */
      if (pcbddc->nedcG) {
        PetscInt cols[2];

        cols[0] = 2*i;
        cols[1] = 2*i+1;
        ierr = MatSetValuesLocal(pcbddc->nedcG,1,&i,2,cols,cvals,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(eedges[i],&cols);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&extrows[i]);CHKERRQ(ierr);
    ierr = ISDestroy(&extcols[i]);CHKERRQ(ierr);
    ierr = ISDestroy(&cornersis);CHKERRQ(ierr);
    ierr = MatDestroy(&Gins);CHKERRQ(ierr);
    ierr = MatDestroy(&GKins);CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingDestroy(&el2g);CHKERRQ(ierr);

  /* Start assembling */
  ierr = MatAssemblyBegin(T,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (pcbddc->nedcG) {
    ierr = MatAssemblyBegin(pcbddc->nedcG,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  /* Free */
  if (fl2g) {
    ierr = ISDestroy(&primals);CHKERRQ(ierr);
    for (i=0;i<nee;i++) {
      ierr = ISDestroy(&eedges[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(eedges);CHKERRQ(ierr);
  }

  /* hack mat_graph with primal dofs on the coarse edges */
  {
    PCBDDCGraph graph   = pcbddc->mat_graph;
    PetscInt    *oqueue = graph->queue;
    PetscInt    *ocptr  = graph->cptr;
    PetscInt    ncc,*idxs;

    /* find first primal edge */
    if (pcbddc->nedclocal) {
      ierr = ISGetIndices(pcbddc->nedclocal,(const PetscInt**)&idxs);CHKERRQ(ierr);
    } else {
      if (fl2g) {
        ierr = ISLocalToGlobalMappingApply(fl2g,nee,cedges,cedges);CHKERRQ(ierr);
      }
      idxs = cedges;
    }
    cum = 0;
    while (cum < nee && cedges[cum] < 0) cum++;

    /* adapt connected components */
    ierr = PetscMalloc2(graph->nvtxs+1,&graph->cptr,ocptr[graph->ncc],&graph->queue);CHKERRQ(ierr);
    graph->cptr[0] = 0;
    for (i=0,ncc=0;i<graph->ncc;i++) {
      PetscInt lc = ocptr[i+1]-ocptr[i];
      if (cum != nee && oqueue[ocptr[i+1]-1] == cedges[cum]) { /* this cc has a primal dof */
        graph->cptr[ncc+1] = graph->cptr[ncc]+1;
        graph->queue[graph->cptr[ncc]] = cedges[cum];
        ncc++;
        lc--;
        cum++;
        while (cum < nee && cedges[cum] < 0) cum++;
      }
      graph->cptr[ncc+1] = graph->cptr[ncc] + lc;
      for (j=0;j<lc;j++) graph->queue[graph->cptr[ncc]+j] = oqueue[ocptr[i]+j];
      ncc++;
    }
    graph->ncc = ncc;
    if (pcbddc->nedclocal) {
      ierr = ISRestoreIndices(pcbddc->nedclocal,(const PetscInt**)&idxs);CHKERRQ(ierr);
    }
    ierr = PetscFree2(ocptr,oqueue);CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingDestroy(&fl2g);CHKERRQ(ierr);
  ierr = PCBDDCGraphRestoreCandidatesIS(pcbddc->mat_graph,NULL,NULL,&nee,&alleedges,&allprimals);CHKERRQ(ierr);
  ierr = PCBDDCGraphResetCSR(pcbddc->mat_graph);CHKERRQ(ierr);
  ierr = MatDestroy(&conn);CHKERRQ(ierr);

  ierr = ISDestroy(&nedfieldlocal);CHKERRQ(ierr);
  ierr = PetscFree(extrow);CHKERRQ(ierr);
  ierr = PetscFree2(work,rwork);CHKERRQ(ierr);
  ierr = PetscFree(corners);CHKERRQ(ierr);
  ierr = PetscFree(cedges);CHKERRQ(ierr);
  ierr = PetscFree(extrows);CHKERRQ(ierr);
  ierr = PetscFree(extcols);CHKERRQ(ierr);
  ierr = MatDestroy(&lG);CHKERRQ(ierr);

  /* Complete assembling */
  ierr = MatAssemblyEnd(T,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (pcbddc->nedcG) {
    ierr = MatAssemblyEnd(pcbddc->nedcG,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if 0
    ierr = PetscObjectSetName((PetscObject)pcbddc->nedcG,"coarse_G");CHKERRQ(ierr);
    ierr = MatView(pcbddc->nedcG,NULL);CHKERRQ(ierr);
#endif
  }

  /* set change of basis */
  ierr = PCBDDCSetChangeOfBasisMat(pc,T,singular);CHKERRQ(ierr);
  ierr = MatDestroy(&T);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* the near-null space of BDDC carries information on quadrature weights,
   and these can be collinear -> so cheat with MatNullSpaceCreate
   and create a suitable set of basis vectors first */
PetscErrorCode PCBDDCNullSpaceCreate(MPI_Comm comm, PetscBool has_const, PetscInt nvecs, Vec quad_vecs[], MatNullSpace *nnsp)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0;i<nvecs;i++) {
    PetscInt first,last;

    ierr = VecGetOwnershipRange(quad_vecs[i],&first,&last);CHKERRQ(ierr);
    if (last-first < 2*nvecs && has_const) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
    if (i>=first && i < last) {
      PetscScalar *data;
      ierr = VecGetArray(quad_vecs[i],&data);CHKERRQ(ierr);
      if (!has_const) {
        data[i-first] = 1.;
      } else {
        data[2*i-first] = 1./PetscSqrtReal(2.);
        data[2*i-first+1] = -1./PetscSqrtReal(2.);
      }
      ierr = VecRestoreArray(quad_vecs[i],&data);CHKERRQ(ierr);
    }
    ierr = PetscObjectStateIncrease((PetscObject)quad_vecs[i]);CHKERRQ(ierr);
  }
  ierr = MatNullSpaceCreate(comm,has_const,nvecs,quad_vecs,nnsp);CHKERRQ(ierr);
  for (i=0;i<nvecs;i++) { /* reset vectors */
    PetscInt first,last;
    ierr = VecLockReadPop(quad_vecs[i]);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(quad_vecs[i],&first,&last);CHKERRQ(ierr);
    if (i>=first && i < last) {
      PetscScalar *data;
      ierr = VecGetArray(quad_vecs[i],&data);CHKERRQ(ierr);
      if (!has_const) {
        data[i-first] = 0.;
      } else {
        data[2*i-first] = 0.;
        data[2*i-first+1] = 0.;
      }
      ierr = VecRestoreArray(quad_vecs[i],&data);CHKERRQ(ierr);
    }
    ierr = PetscObjectStateIncrease((PetscObject)quad_vecs[i]);CHKERRQ(ierr);
    ierr = VecLockReadPush(quad_vecs[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCComputeNoNetFlux(Mat A, Mat divudotp, PetscBool transpose, IS vl2l, PCBDDCGraph graph, MatNullSpace *nnsp)
{
  Mat                    loc_divudotp;
  Vec                    p,v,vins,quad_vec,*quad_vecs;
  ISLocalToGlobalMapping map;
  PetscScalar            *vals;
  const PetscScalar      *array;
  PetscInt               i,maxneighs,maxsize,*gidxs;
  PetscInt               n_neigh,*neigh,*n_shared,**shared;
  PetscMPIInt            rank;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = ISLocalToGlobalMappingGetInfo(graph->l2gmap,&n_neigh,&neigh,&n_shared,&shared);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&n_neigh,&maxneighs,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
  if (!maxneighs) {
    ierr  = ISLocalToGlobalMappingRestoreInfo(graph->l2gmap,&n_neigh,&neigh,&n_shared,&shared);CHKERRQ(ierr);
    *nnsp = NULL;
    PetscFunctionReturn(0);
  }
  maxsize = 0;
  for (i=0;i<n_neigh;i++) maxsize = PetscMax(n_shared[i],maxsize);
  ierr = PetscMalloc2(maxsize,&gidxs,maxsize,&vals);CHKERRQ(ierr);
  /* create vectors to hold quadrature weights */
  ierr = MatCreateVecs(A,&quad_vec,NULL);CHKERRQ(ierr);
  if (!transpose) {
    ierr = MatGetLocalToGlobalMapping(A,&map,NULL);CHKERRQ(ierr);
  } else {
    ierr = MatGetLocalToGlobalMapping(A,NULL,&map);CHKERRQ(ierr);
  }
  ierr = VecDuplicateVecs(quad_vec,maxneighs,&quad_vecs);CHKERRQ(ierr);
  ierr = VecDestroy(&quad_vec);CHKERRQ(ierr);
  ierr = PCBDDCNullSpaceCreate(PetscObjectComm((PetscObject)A),PETSC_FALSE,maxneighs,quad_vecs,nnsp);CHKERRQ(ierr);
  for (i=0;i<maxneighs;i++) {
    ierr = VecLockReadPop(quad_vecs[i]);CHKERRQ(ierr);
  }

  /* compute local quad vec */
  ierr = MatISGetLocalMat(divudotp,&loc_divudotp);CHKERRQ(ierr);
  if (!transpose) {
    ierr = MatCreateVecs(loc_divudotp,&v,&p);CHKERRQ(ierr);
  } else {
    ierr = MatCreateVecs(loc_divudotp,&p,&v);CHKERRQ(ierr);
  }
  ierr = VecSet(p,1.);CHKERRQ(ierr);
  if (!transpose) {
    ierr = MatMultTranspose(loc_divudotp,p,v);CHKERRQ(ierr);
  } else {
    ierr = MatMult(loc_divudotp,p,v);CHKERRQ(ierr);
  }
  if (vl2l) {
    Mat        lA;
    VecScatter sc;

    ierr = MatISGetLocalMat(A,&lA);CHKERRQ(ierr);
    ierr = MatCreateVecs(lA,&vins,NULL);CHKERRQ(ierr);
    ierr = VecScatterCreate(v,NULL,vins,vl2l,&sc);CHKERRQ(ierr);
    ierr = VecScatterBegin(sc,v,vins,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(sc,v,vins,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&sc);CHKERRQ(ierr);
  } else {
    vins = v;
  }
  ierr = VecGetArrayRead(vins,&array);CHKERRQ(ierr);
  ierr = VecDestroy(&p);CHKERRQ(ierr);

  /* insert in global quadrature vecs */
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)A),&rank);CHKERRQ(ierr);
  for (i=0;i<n_neigh;i++) {
    const PetscInt    *idxs;
    PetscInt          idx,nn,j;

    idxs = shared[i];
    nn   = n_shared[i];
    for (j=0;j<nn;j++) vals[j] = array[idxs[j]];
    ierr = PetscFindInt(rank,graph->count[idxs[0]],graph->neighbours_set[idxs[0]],&idx);CHKERRQ(ierr);
    idx  = -(idx+1);
    ierr = ISLocalToGlobalMappingApply(map,nn,idxs,gidxs);CHKERRQ(ierr);
    ierr = VecSetValues(quad_vecs[idx],nn,gidxs,vals,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingRestoreInfo(graph->l2gmap,&n_neigh,&neigh,&n_shared,&shared);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vins,&array);CHKERRQ(ierr);
  if (vl2l) {
    ierr = VecDestroy(&vins);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = PetscFree2(gidxs,vals);CHKERRQ(ierr);

  /* assemble near null space */
  for (i=0;i<maxneighs;i++) {
    ierr = VecAssemblyBegin(quad_vecs[i]);CHKERRQ(ierr);
  }
  for (i=0;i<maxneighs;i++) {
    ierr = VecAssemblyEnd(quad_vecs[i]);CHKERRQ(ierr);
    ierr = VecViewFromOptions(quad_vecs[i],NULL,"-pc_bddc_quad_vecs_view");CHKERRQ(ierr);
    ierr = VecLockReadPush(quad_vecs[i]);CHKERRQ(ierr);
  }
  ierr = VecDestroyVecs(maxneighs,&quad_vecs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCAddPrimalVerticesLocalIS(PC pc, IS primalv)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (primalv) {
    if (pcbddc->user_primal_vertices_local) {
      IS list[2], newp;

      list[0] = primalv;
      list[1] = pcbddc->user_primal_vertices_local;
      ierr = ISConcatenate(PetscObjectComm((PetscObject)pc),2,list,&newp);CHKERRQ(ierr);
      ierr = ISSortRemoveDups(newp);CHKERRQ(ierr);
      ierr = ISDestroy(&list[1]);CHKERRQ(ierr);
      pcbddc->user_primal_vertices_local = newp;
    } else {
      ierr = PCBDDCSetPrimalVerticesLocalIS(pc,primalv);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode func_coords_private(PetscInt dim, PetscReal t, const PetscReal X[], PetscInt Nf, PetscScalar *out, void *ctx)
{
  PetscInt f, *comp  = (PetscInt *)ctx;

  PetscFunctionBegin;
  for (f=0;f<Nf;f++) out[f] = X[*comp];
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCComputeLocalTopologyInfo(PC pc)
{
  PetscErrorCode ierr;
  Vec            local,global;
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  Mat_IS         *matis = (Mat_IS*)pc->pmat->data;
  PetscBool      monolithic = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)pc),((PetscObject)pc)->prefix,"BDDC topology options","PC");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_monolithic","Discard any information on dofs splitting",NULL,monolithic,&monolithic,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  /* need to convert from global to local topology information and remove references to information in global ordering */
  ierr = MatCreateVecs(pc->pmat,&global,NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(matis->A,&local,NULL);CHKERRQ(ierr);
  ierr = VecPinToCPU(global,PETSC_TRUE);CHKERRQ(ierr);
  ierr = VecPinToCPU(local,PETSC_TRUE);CHKERRQ(ierr);
  if (monolithic) { /* just get block size to properly compute vertices */
    if (pcbddc->vertex_size == 1) {
      ierr = MatGetBlockSize(pc->pmat,&pcbddc->vertex_size);CHKERRQ(ierr);
    }
    goto boundary;
  }

  if (pcbddc->user_provided_isfordofs) {
    if (pcbddc->n_ISForDofs) {
      PetscInt i;

      ierr = PetscMalloc1(pcbddc->n_ISForDofs,&pcbddc->ISForDofsLocal);CHKERRQ(ierr);
      for (i=0;i<pcbddc->n_ISForDofs;i++) {
        PetscInt bs;

        ierr = PCBDDCGlobalToLocal(matis->rctx,global,local,pcbddc->ISForDofs[i],&pcbddc->ISForDofsLocal[i]);CHKERRQ(ierr);
        ierr = ISGetBlockSize(pcbddc->ISForDofs[i],&bs);CHKERRQ(ierr);
        ierr = ISSetBlockSize(pcbddc->ISForDofsLocal[i],bs);CHKERRQ(ierr);
        ierr = ISDestroy(&pcbddc->ISForDofs[i]);CHKERRQ(ierr);
      }
      pcbddc->n_ISForDofsLocal = pcbddc->n_ISForDofs;
      pcbddc->n_ISForDofs = 0;
      ierr = PetscFree(pcbddc->ISForDofs);CHKERRQ(ierr);
    }
  } else {
    if (!pcbddc->n_ISForDofsLocal) { /* field split not present */
      DM dm;

      ierr = MatGetDM(pc->pmat, &dm);CHKERRQ(ierr);
      if (!dm) {
        ierr = PCGetDM(pc, &dm);CHKERRQ(ierr);
      }
      if (dm) {
        IS      *fields;
        PetscInt nf,i;

        ierr = DMCreateFieldDecomposition(dm,&nf,NULL,&fields,NULL);CHKERRQ(ierr);
        ierr = PetscMalloc1(nf,&pcbddc->ISForDofsLocal);CHKERRQ(ierr);
        for (i=0;i<nf;i++) {
          PetscInt bs;

          ierr = PCBDDCGlobalToLocal(matis->rctx,global,local,fields[i],&pcbddc->ISForDofsLocal[i]);CHKERRQ(ierr);
          ierr = ISGetBlockSize(fields[i],&bs);CHKERRQ(ierr);
          ierr = ISSetBlockSize(pcbddc->ISForDofsLocal[i],bs);CHKERRQ(ierr);
          ierr = ISDestroy(&fields[i]);CHKERRQ(ierr);
        }
        ierr = PetscFree(fields);CHKERRQ(ierr);
        pcbddc->n_ISForDofsLocal = nf;
      } else { /* See if MATIS has fields attached by the conversion from MatNest */
        PetscContainer   c;

        ierr = PetscObjectQuery((PetscObject)pc->pmat,"_convert_nest_lfields",(PetscObject*)&c);CHKERRQ(ierr);
        if (c) {
          MatISLocalFields lf;
          ierr = PetscContainerGetPointer(c,(void**)&lf);CHKERRQ(ierr);
          ierr = PCBDDCSetDofsSplittingLocal(pc,lf->nr,lf->rf);CHKERRQ(ierr);
        } else { /* fallback, create the default fields if bs > 1 */
          PetscInt i, n = matis->A->rmap->n;
          ierr = MatGetBlockSize(pc->pmat,&i);CHKERRQ(ierr);
          if (i > 1) {
            pcbddc->n_ISForDofsLocal = i;
            ierr = PetscMalloc1(pcbddc->n_ISForDofsLocal,&pcbddc->ISForDofsLocal);CHKERRQ(ierr);
            for (i=0;i<pcbddc->n_ISForDofsLocal;i++) {
              ierr = ISCreateStride(PetscObjectComm((PetscObject)pc),n/pcbddc->n_ISForDofsLocal,i,pcbddc->n_ISForDofsLocal,&pcbddc->ISForDofsLocal[i]);CHKERRQ(ierr);
            }
          }
        }
      }
    } else {
      PetscInt i;
      for (i=0;i<pcbddc->n_ISForDofsLocal;i++) {
        ierr = PCBDDCConsistencyCheckIS(pc,MPI_LAND,&pcbddc->ISForDofsLocal[i]);CHKERRQ(ierr);
      }
    }
  }

boundary:
  if (!pcbddc->DirichletBoundariesLocal && pcbddc->DirichletBoundaries) {
    ierr = PCBDDCGlobalToLocal(matis->rctx,global,local,pcbddc->DirichletBoundaries,&pcbddc->DirichletBoundariesLocal);CHKERRQ(ierr);
  } else if (pcbddc->DirichletBoundariesLocal) {
    ierr = PCBDDCConsistencyCheckIS(pc,MPI_LAND,&pcbddc->DirichletBoundariesLocal);CHKERRQ(ierr);
  }
  if (!pcbddc->NeumannBoundariesLocal && pcbddc->NeumannBoundaries) {
    ierr = PCBDDCGlobalToLocal(matis->rctx,global,local,pcbddc->NeumannBoundaries,&pcbddc->NeumannBoundariesLocal);CHKERRQ(ierr);
  } else if (pcbddc->NeumannBoundariesLocal) {
    ierr = PCBDDCConsistencyCheckIS(pc,MPI_LOR,&pcbddc->NeumannBoundariesLocal);CHKERRQ(ierr);
  }
  if (!pcbddc->user_primal_vertices_local && pcbddc->user_primal_vertices) {
    ierr = PCBDDCGlobalToLocal(matis->rctx,global,local,pcbddc->user_primal_vertices,&pcbddc->user_primal_vertices_local);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  /* detect local disconnected subdomains if requested (use matis->A) */
  if (pcbddc->detect_disconnected) {
    IS        primalv = NULL;
    PetscInt  i;
    PetscBool filter = pcbddc->detect_disconnected_filter;

    for (i=0;i<pcbddc->n_local_subs;i++) {
      ierr = ISDestroy(&pcbddc->local_subs[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(pcbddc->local_subs);CHKERRQ(ierr);
    ierr = PCBDDCDetectDisconnectedComponents(pc,filter,&pcbddc->n_local_subs,&pcbddc->local_subs,&primalv);CHKERRQ(ierr);
    ierr = PCBDDCAddPrimalVerticesLocalIS(pc,primalv);CHKERRQ(ierr);
    ierr = ISDestroy(&primalv);CHKERRQ(ierr);
  }
  /* early stage corner detection */
  {
    DM dm;

    ierr = MatGetDM(pc->pmat,&dm);CHKERRQ(ierr);
    if (!dm) {
      ierr = PCGetDM(pc,&dm);CHKERRQ(ierr);
    }
    if (dm) {
      PetscBool isda;

      ierr = PetscObjectTypeCompare((PetscObject)dm,DMDA,&isda);CHKERRQ(ierr);
      if (isda) {
        ISLocalToGlobalMapping l2l;
        IS                     corners;
        Mat                    lA;
        PetscBool              gl,lo;

        {
          Vec               cvec;
          const PetscScalar *coords;
          PetscInt          dof,n,cdim;
          PetscBool         memc = PETSC_TRUE;

          ierr = DMDAGetInfo(dm,NULL,NULL,NULL,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
          ierr = DMGetCoordinates(dm,&cvec);CHKERRQ(ierr);
          ierr = VecGetLocalSize(cvec,&n);CHKERRQ(ierr);
          ierr = VecGetBlockSize(cvec,&cdim);CHKERRQ(ierr);
          n   /= cdim;
          ierr = PetscFree(pcbddc->mat_graph->coords);CHKERRQ(ierr);
          ierr = PetscMalloc1(dof*n*cdim,&pcbddc->mat_graph->coords);CHKERRQ(ierr);
          ierr = VecGetArrayRead(cvec,&coords);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
          memc = PETSC_FALSE;
#endif
          if (dof != 1) memc = PETSC_FALSE;
          if (memc) {
            ierr = PetscArraycpy(pcbddc->mat_graph->coords,coords,cdim*n*dof);CHKERRQ(ierr);
          } else { /* BDDC graph does not use any blocked information, we need to replicate the data */
            PetscReal *bcoords = pcbddc->mat_graph->coords;
            PetscInt  i, b, d;

            for (i=0;i<n;i++) {
              for (b=0;b<dof;b++) {
                for (d=0;d<cdim;d++) {
                  bcoords[i*dof*cdim + b*cdim + d] = PetscRealPart(coords[i*cdim+d]);
                }
              }
            }
          }
          ierr = VecRestoreArrayRead(cvec,&coords);CHKERRQ(ierr);
          pcbddc->mat_graph->cdim  = cdim;
          pcbddc->mat_graph->cnloc = dof*n;
          pcbddc->mat_graph->cloc  = PETSC_FALSE;
        }
        ierr = DMDAGetSubdomainCornersIS(dm,&corners);CHKERRQ(ierr);
        ierr = MatISGetLocalMat(pc->pmat,&lA);CHKERRQ(ierr);
        ierr = MatGetLocalToGlobalMapping(lA,&l2l,NULL);CHKERRQ(ierr);
        ierr = MatISRestoreLocalMat(pc->pmat,&lA);CHKERRQ(ierr);
        lo   = (PetscBool)(l2l && corners);
        ierr = MPIU_Allreduce(&lo,&gl,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
        if (gl) { /* From PETSc's DMDA */
          const PetscInt    *idx;
          PetscInt          dof,bs,*idxout,n;

          ierr = DMDAGetInfo(dm,NULL,NULL,NULL,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
          ierr = ISLocalToGlobalMappingGetBlockSize(l2l,&bs);CHKERRQ(ierr);
          ierr = ISGetLocalSize(corners,&n);CHKERRQ(ierr);
          ierr = ISGetIndices(corners,&idx);CHKERRQ(ierr);
          if (bs == dof) {
            ierr = PetscMalloc1(n,&idxout);CHKERRQ(ierr);
            ierr = ISLocalToGlobalMappingApplyBlock(l2l,n,idx,idxout);CHKERRQ(ierr);
          } else { /* the original DMDA local-to-local map have been modified */
            PetscInt i,d;

            ierr = PetscMalloc1(dof*n,&idxout);CHKERRQ(ierr);
            for (i=0;i<n;i++) for (d=0;d<dof;d++) idxout[dof*i+d] = dof*idx[i]+d;
            ierr = ISLocalToGlobalMappingApply(l2l,dof*n,idxout,idxout);CHKERRQ(ierr);

            bs = 1;
            n *= dof;
          }
          ierr = ISRestoreIndices(corners,&idx);CHKERRQ(ierr);
          ierr = DMDARestoreSubdomainCornersIS(dm,&corners);CHKERRQ(ierr);
          ierr = ISCreateBlock(PetscObjectComm((PetscObject)pc),bs,n,idxout,PETSC_OWN_POINTER,&corners);CHKERRQ(ierr);
          ierr = PCBDDCAddPrimalVerticesLocalIS(pc,corners);CHKERRQ(ierr);
          ierr = ISDestroy(&corners);CHKERRQ(ierr);
          pcbddc->corner_selected  = PETSC_TRUE;
          pcbddc->corner_selection = PETSC_TRUE;
        }
        if (corners) {
          ierr = DMDARestoreSubdomainCornersIS(dm,&corners);CHKERRQ(ierr);
        }
      }
    }
  }
  if (pcbddc->corner_selection && !pcbddc->mat_graph->cdim) {
    DM dm;

    ierr = MatGetDM(pc->pmat,&dm);CHKERRQ(ierr);
    if (!dm) {
      ierr = PCGetDM(pc,&dm);CHKERRQ(ierr);
    }
    if (dm) { /* this can get very expensive, I need to find a faster alternative */
      Vec            vcoords;
      PetscSection   section;
      PetscReal      *coords;
      PetscInt       d,cdim,nl,nf,**ctxs;
      PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal *, PetscInt, PetscScalar *, void *);

      ierr = DMGetCoordinateDim(dm,&cdim);CHKERRQ(ierr);
      ierr = DMGetLocalSection(dm,&section);CHKERRQ(ierr);
      ierr = PetscSectionGetNumFields(section,&nf);CHKERRQ(ierr);
      ierr = DMCreateGlobalVector(dm,&vcoords);CHKERRQ(ierr);
      ierr = VecGetLocalSize(vcoords,&nl);CHKERRQ(ierr);
      ierr = PetscMalloc1(nl*cdim,&coords);CHKERRQ(ierr);
      ierr = PetscMalloc2(nf,&funcs,nf,&ctxs);CHKERRQ(ierr);
      ierr = PetscMalloc1(nf,&ctxs[0]);CHKERRQ(ierr);
      for (d=0;d<nf;d++) funcs[d] = func_coords_private;
      for (d=1;d<nf;d++) ctxs[d] = ctxs[d-1] + 1;
      for (d=0;d<cdim;d++) {
        PetscInt          i;
        const PetscScalar *v;

        for (i=0;i<nf;i++) ctxs[i][0] = d;
        ierr = DMProjectFunction(dm,0.0,funcs,(void**)ctxs,INSERT_VALUES,vcoords);CHKERRQ(ierr);
        ierr = VecGetArrayRead(vcoords,&v);CHKERRQ(ierr);
        for (i=0;i<nl;i++) coords[i*cdim+d] = PetscRealPart(v[i]);
        ierr = VecRestoreArrayRead(vcoords,&v);CHKERRQ(ierr);
      }
      ierr = VecDestroy(&vcoords);CHKERRQ(ierr);
      ierr = PCSetCoordinates(pc,cdim,nl,coords);CHKERRQ(ierr);
      ierr = PetscFree(coords);CHKERRQ(ierr);
      ierr = PetscFree(ctxs[0]);CHKERRQ(ierr);
      ierr = PetscFree2(funcs,ctxs);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCConsistencyCheckIS(PC pc, MPI_Op mop, IS *is)
{
  Mat_IS          *matis = (Mat_IS*)(pc->pmat->data);
  PetscErrorCode  ierr;
  IS              nis;
  const PetscInt  *idxs;
  PetscInt        i,nd,n = matis->A->rmap->n,*nidxs,nnd;
  PetscBool       *ld;

  PetscFunctionBegin;
  if (mop != MPI_LAND && mop != MPI_LOR) SETERRQ(PetscObjectComm((PetscObject)(pc)),PETSC_ERR_SUP,"Supported are MPI_LAND and MPI_LOR");
  if (mop == MPI_LAND) {
    /* init rootdata with true */
    ld   = (PetscBool*) matis->sf_rootdata;
    for (i=0;i<pc->pmat->rmap->n;i++) ld[i] = PETSC_TRUE;
  } else {
    ierr = PetscArrayzero(matis->sf_rootdata,pc->pmat->rmap->n);CHKERRQ(ierr);
  }
  ierr = PetscArrayzero(matis->sf_leafdata,n);CHKERRQ(ierr);
  ierr = ISGetLocalSize(*is,&nd);CHKERRQ(ierr);
  ierr = ISGetIndices(*is,&idxs);CHKERRQ(ierr);
  ld   = (PetscBool*) matis->sf_leafdata;
  for (i=0;i<nd;i++)
    if (-1 < idxs[i] && idxs[i] < n)
      ld[idxs[i]] = PETSC_TRUE;
  ierr = ISRestoreIndices(*is,&idxs);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(matis->sf,MPIU_BOOL,matis->sf_leafdata,matis->sf_rootdata,mop);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(matis->sf,MPIU_BOOL,matis->sf_leafdata,matis->sf_rootdata,mop);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(matis->sf,MPIU_BOOL,matis->sf_rootdata,matis->sf_leafdata);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(matis->sf,MPIU_BOOL,matis->sf_rootdata,matis->sf_leafdata);CHKERRQ(ierr);
  if (mop == MPI_LAND) {
    ierr = PetscMalloc1(nd,&nidxs);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc1(n,&nidxs);CHKERRQ(ierr);
  }
  for (i=0,nnd=0;i<n;i++)
    if (ld[i])
      nidxs[nnd++] = i;
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)(*is)),nnd,nidxs,PETSC_OWN_POINTER,&nis);CHKERRQ(ierr);
  ierr = ISDestroy(is);CHKERRQ(ierr);
  *is  = nis;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCBenignRemoveInterior(PC pc,Vec r,Vec z)
{
  PC_IS             *pcis = (PC_IS*)(pc->data);
  PC_BDDC           *pcbddc = (PC_BDDC*)(pc->data);
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!pcbddc->benign_have_null) {
    PetscFunctionReturn(0);
  }
  if (pcbddc->ChangeOfBasisMatrix) {
    Vec swap;

    ierr = MatMultTranspose(pcbddc->ChangeOfBasisMatrix,r,pcbddc->work_change);CHKERRQ(ierr);
    swap = pcbddc->work_change;
    pcbddc->work_change = r;
    r = swap;
  }
  ierr = VecScatterBegin(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = KSPSolve(pcbddc->ksp_D,pcis->vec1_D,pcis->vec2_D);CHKERRQ(ierr);
  ierr = KSPCheckSolve(pcbddc->ksp_D,pc,pcis->vec2_D);CHKERRQ(ierr);
  ierr = VecSet(z,0.);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcis->global_to_D,pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->global_to_D,pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  if (pcbddc->ChangeOfBasisMatrix) {
    pcbddc->work_change = r;
    ierr = VecCopy(z,pcbddc->work_change);CHKERRQ(ierr);
    ierr = MatMult(pcbddc->ChangeOfBasisMatrix,pcbddc->work_change,z);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCBenignMatMult_Private_Private(Mat A, Vec x, Vec y, PetscBool transpose)
{
  PCBDDCBenignMatMult_ctx ctx;
  PetscErrorCode          ierr;
  PetscBool               apply_right,apply_left,reset_x;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&ctx);CHKERRQ(ierr);
  if (transpose) {
    apply_right = ctx->apply_left;
    apply_left = ctx->apply_right;
  } else {
    apply_right = ctx->apply_right;
    apply_left = ctx->apply_left;
  }
  reset_x = PETSC_FALSE;
  if (apply_right) {
    const PetscScalar *ax;
    PetscInt          nl,i;

    ierr = VecGetLocalSize(x,&nl);CHKERRQ(ierr);
    ierr = VecGetArrayRead(x,&ax);CHKERRQ(ierr);
    ierr = PetscArraycpy(ctx->work,ax,nl);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(x,&ax);CHKERRQ(ierr);
    for (i=0;i<ctx->benign_n;i++) {
      PetscScalar    sum,val;
      const PetscInt *idxs;
      PetscInt       nz,j;
      ierr = ISGetLocalSize(ctx->benign_zerodiag_subs[i],&nz);CHKERRQ(ierr);
      ierr = ISGetIndices(ctx->benign_zerodiag_subs[i],&idxs);CHKERRQ(ierr);
      sum = 0.;
      if (ctx->apply_p0) {
        val = ctx->work[idxs[nz-1]];
        for (j=0;j<nz-1;j++) {
          sum += ctx->work[idxs[j]];
          ctx->work[idxs[j]] += val;
        }
      } else {
        for (j=0;j<nz-1;j++) {
          sum += ctx->work[idxs[j]];
        }
      }
      ctx->work[idxs[nz-1]] -= sum;
      ierr = ISRestoreIndices(ctx->benign_zerodiag_subs[i],&idxs);CHKERRQ(ierr);
    }
    ierr = VecPlaceArray(x,ctx->work);CHKERRQ(ierr);
    reset_x = PETSC_TRUE;
  }
  if (transpose) {
    ierr = MatMultTranspose(ctx->A,x,y);CHKERRQ(ierr);
  } else {
    ierr = MatMult(ctx->A,x,y);CHKERRQ(ierr);
  }
  if (reset_x) {
    ierr = VecResetArray(x);CHKERRQ(ierr);
  }
  if (apply_left) {
    PetscScalar *ay;
    PetscInt    i;

    ierr = VecGetArray(y,&ay);CHKERRQ(ierr);
    for (i=0;i<ctx->benign_n;i++) {
      PetscScalar    sum,val;
      const PetscInt *idxs;
      PetscInt       nz,j;
      ierr = ISGetLocalSize(ctx->benign_zerodiag_subs[i],&nz);CHKERRQ(ierr);
      ierr = ISGetIndices(ctx->benign_zerodiag_subs[i],&idxs);CHKERRQ(ierr);
      val = -ay[idxs[nz-1]];
      if (ctx->apply_p0) {
        sum = 0.;
        for (j=0;j<nz-1;j++) {
          sum += ay[idxs[j]];
          ay[idxs[j]] += val;
        }
        ay[idxs[nz-1]] += sum;
      } else {
        for (j=0;j<nz-1;j++) {
          ay[idxs[j]] += val;
        }
        ay[idxs[nz-1]] = 0.;
      }
      ierr = ISRestoreIndices(ctx->benign_zerodiag_subs[i],&idxs);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(y,&ay);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCBenignMatMultTranspose_Private(Mat A, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCBDDCBenignMatMult_Private_Private(A,x,y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCBenignMatMult_Private(Mat A, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCBDDCBenignMatMult_Private_Private(A,x,y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCBenignShellMat(PC pc, PetscBool restore)
{
  PC_IS                   *pcis = (PC_IS*)pc->data;
  PC_BDDC                 *pcbddc = (PC_BDDC*)pc->data;
  PCBDDCBenignMatMult_ctx ctx;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  if (!restore) {
    Mat                A_IB,A_BI;
    PetscScalar        *work;
    PCBDDCReuseSolvers reuse = pcbddc->sub_schurs ? pcbddc->sub_schurs->reuse_solver : NULL;

    if (pcbddc->benign_original_mat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Benign original mat has not been restored");
    if (!pcbddc->benign_change || !pcbddc->benign_n || pcbddc->benign_change_explicit) PetscFunctionReturn(0);
    ierr = PetscMalloc1(pcis->n,&work);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,&A_IB);CHKERRQ(ierr);
    ierr = MatSetSizes(A_IB,pcis->n-pcis->n_B,pcis->n_B,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetType(A_IB,MATSHELL);CHKERRQ(ierr);
    ierr = MatShellSetOperation(A_IB,MATOP_MULT,(void (*)(void))PCBDDCBenignMatMult_Private);CHKERRQ(ierr);
    ierr = MatShellSetOperation(A_IB,MATOP_MULT_TRANSPOSE,(void (*)(void))PCBDDCBenignMatMultTranspose_Private);CHKERRQ(ierr);
    ierr = PetscNew(&ctx);CHKERRQ(ierr);
    ierr = MatShellSetContext(A_IB,ctx);CHKERRQ(ierr);
    ctx->apply_left = PETSC_TRUE;
    ctx->apply_right = PETSC_FALSE;
    ctx->apply_p0 = PETSC_FALSE;
    ctx->benign_n = pcbddc->benign_n;
    if (reuse) {
      ctx->benign_zerodiag_subs = reuse->benign_zerodiag_subs;
      ctx->free = PETSC_FALSE;
    } else { /* TODO: could be optimized for successive solves */
      ISLocalToGlobalMapping N_to_D;
      PetscInt               i;

      ierr = ISLocalToGlobalMappingCreateIS(pcis->is_I_local,&N_to_D);CHKERRQ(ierr);
      ierr = PetscMalloc1(pcbddc->benign_n,&ctx->benign_zerodiag_subs);CHKERRQ(ierr);
      for (i=0;i<pcbddc->benign_n;i++) {
        ierr = ISGlobalToLocalMappingApplyIS(N_to_D,IS_GTOLM_DROP,pcbddc->benign_zerodiag_subs[i],&ctx->benign_zerodiag_subs[i]);CHKERRQ(ierr);
      }
      ierr = ISLocalToGlobalMappingDestroy(&N_to_D);CHKERRQ(ierr);
      ctx->free = PETSC_TRUE;
    }
    ctx->A = pcis->A_IB;
    ctx->work = work;
    ierr = MatSetUp(A_IB);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A_IB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A_IB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    pcis->A_IB = A_IB;

    /* A_BI as A_IB^T */
    ierr = MatCreateTranspose(A_IB,&A_BI);CHKERRQ(ierr);
    pcbddc->benign_original_mat = pcis->A_BI;
    pcis->A_BI = A_BI;
  } else {
    if (!pcbddc->benign_original_mat) {
      PetscFunctionReturn(0);
    }
    ierr = MatShellGetContext(pcis->A_IB,&ctx);CHKERRQ(ierr);
    ierr = MatDestroy(&pcis->A_IB);CHKERRQ(ierr);
    pcis->A_IB = ctx->A;
    ctx->A = NULL;
    ierr = MatDestroy(&pcis->A_BI);CHKERRQ(ierr);
    pcis->A_BI = pcbddc->benign_original_mat;
    pcbddc->benign_original_mat = NULL;
    if (ctx->free) {
      PetscInt i;
      for (i=0;i<ctx->benign_n;i++) {
        ierr = ISDestroy(&ctx->benign_zerodiag_subs[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(ctx->benign_zerodiag_subs);CHKERRQ(ierr);
    }
    ierr = PetscFree(ctx->work);CHKERRQ(ierr);
    ierr = PetscFree(ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* used just in bddc debug mode */
PetscErrorCode PCBDDCBenignProject(PC pc, IS is1, IS is2, Mat *B)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  Mat_IS         *matis = (Mat_IS*)pc->pmat->data;
  Mat            An;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatPtAP(matis->A,pcbddc->benign_change,MAT_INITIAL_MATRIX,2.0,&An);CHKERRQ(ierr);
  ierr = MatZeroRowsColumns(An,pcbddc->benign_n,pcbddc->benign_p0_lidx,1.0,NULL,NULL);CHKERRQ(ierr);
  if (is1) {
    ierr = MatCreateSubMatrix(An,is1,is2,MAT_INITIAL_MATRIX,B);CHKERRQ(ierr);
    ierr = MatDestroy(&An);CHKERRQ(ierr);
  } else {
    *B = An;
  }
  PetscFunctionReturn(0);
}

/* TODO: add reuse flag */
PetscErrorCode MatSeqAIJCompress(Mat A, Mat *B)
{
  Mat            Bt;
  PetscScalar    *a,*bdata;
  const PetscInt *ii,*ij;
  PetscInt       m,n,i,nnz,*bii,*bij;
  PetscBool      flg_row;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetSize(A,&n,&m);CHKERRQ(ierr);
  ierr = MatGetRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&n,&ii,&ij,&flg_row);CHKERRQ(ierr);
  ierr = MatSeqAIJGetArray(A,&a);CHKERRQ(ierr);
  nnz = n;
  for (i=0;i<ii[n];i++) {
    if (PetscLikely(PetscAbsScalar(a[i]) > PETSC_SMALL)) nnz++;
  }
  ierr = PetscMalloc1(n+1,&bii);CHKERRQ(ierr);
  ierr = PetscMalloc1(nnz,&bij);CHKERRQ(ierr);
  ierr = PetscMalloc1(nnz,&bdata);CHKERRQ(ierr);
  nnz = 0;
  bii[0] = 0;
  for (i=0;i<n;i++) {
    PetscInt j;
    for (j=ii[i];j<ii[i+1];j++) {
      PetscScalar entry = a[j];
      if (PetscLikely(PetscAbsScalar(entry) > PETSC_SMALL) || (n == m && ij[j] == i)) {
        bij[nnz] = ij[j];
        bdata[nnz] = entry;
        nnz++;
      }
    }
    bii[i+1] = nnz;
  }
  ierr = MatSeqAIJRestoreArray(A,&a);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJWithArrays(PetscObjectComm((PetscObject)A),n,m,bii,bij,bdata,&Bt);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&n,&ii,&ij,&flg_row);CHKERRQ(ierr);
  {
    Mat_SeqAIJ *b = (Mat_SeqAIJ*)(Bt->data);
    b->free_a = PETSC_TRUE;
    b->free_ij = PETSC_TRUE;
  }
  if (*B == A) {
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }
  *B = Bt;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCDetectDisconnectedComponents(PC pc, PetscBool filter, PetscInt *ncc, IS* cc[], IS* primalv)
{
  Mat                    B = NULL;
  DM                     dm;
  IS                     is_dummy,*cc_n;
  ISLocalToGlobalMapping l2gmap_dummy;
  PCBDDCGraph            graph;
  PetscInt               *xadj_filtered = NULL,*adjncy_filtered = NULL;
  PetscInt               i,n;
  PetscInt               *xadj,*adjncy;
  PetscBool              isplex = PETSC_FALSE;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  if (ncc) *ncc = 0;
  if (cc) *cc = NULL;
  if (primalv) *primalv = NULL;
  ierr = PCBDDCGraphCreate(&graph);CHKERRQ(ierr);
  ierr = MatGetDM(pc->pmat,&dm);CHKERRQ(ierr);
  if (!dm) {
    ierr = PCGetDM(pc,&dm);CHKERRQ(ierr);
  }
  if (dm) {
    ierr = PetscObjectTypeCompare((PetscObject)dm,DMPLEX,&isplex);CHKERRQ(ierr);
  }
  if (filter) isplex = PETSC_FALSE;

  if (isplex) { /* this code has been modified from plexpartition.c */
    PetscInt       p, pStart, pEnd, a, adjSize, idx, size, nroots;
    PetscInt      *adj = NULL;
    IS             cellNumbering;
    const PetscInt *cellNum;
    PetscBool      useCone, useClosure;
    PetscSection   section;
    PetscSegBuffer adjBuffer;
    PetscSF        sfPoint;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = DMPlexGetHeightStratum(dm, 0, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(sfPoint, &nroots, NULL, NULL, NULL);CHKERRQ(ierr);
    /* Build adjacency graph via a section/segbuffer */
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &section);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(section, pStart, pEnd);CHKERRQ(ierr);
    ierr = PetscSegBufferCreate(sizeof(PetscInt),1000,&adjBuffer);CHKERRQ(ierr);
    /* Always use FVM adjacency to create partitioner graph */
    ierr = DMGetBasicAdjacency(dm, &useCone, &useClosure);CHKERRQ(ierr);
    ierr = DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMPlexGetCellNumbering(dm, &cellNumbering);CHKERRQ(ierr);
    ierr = ISGetIndices(cellNumbering, &cellNum);CHKERRQ(ierr);
    for (n = 0, p = pStart; p < pEnd; p++) {
      /* Skip non-owned cells in parallel (ParMetis expects no overlap) */
      if (nroots > 0) {if (cellNum[p] < 0) continue;}
      adjSize = PETSC_DETERMINE;
      ierr = DMPlexGetAdjacency(dm, p, &adjSize, &adj);CHKERRQ(ierr);
      for (a = 0; a < adjSize; ++a) {
        const PetscInt point = adj[a];
        if (pStart <= point && point < pEnd) {
          PetscInt *PETSC_RESTRICT pBuf;
          ierr = PetscSectionAddDof(section, p, 1);CHKERRQ(ierr);
          ierr = PetscSegBufferGetInts(adjBuffer, 1, &pBuf);CHKERRQ(ierr);
          *pBuf = point;
        }
      }
      n++;
    }
    ierr = DMSetBasicAdjacency(dm, useCone, useClosure);CHKERRQ(ierr);
    /* Derive CSR graph from section/segbuffer */
    ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(section, &size);CHKERRQ(ierr);
    ierr = PetscMalloc1(n+1, &xadj);CHKERRQ(ierr);
    for (idx = 0, p = pStart; p < pEnd; p++) {
      if (nroots > 0) {if (cellNum[p] < 0) continue;}
      ierr = PetscSectionGetOffset(section, p, &(xadj[idx++]));CHKERRQ(ierr);
    }
    xadj[n] = size;
    ierr = PetscSegBufferExtractAlloc(adjBuffer, &adjncy);CHKERRQ(ierr);
    /* Clean up */
    ierr = PetscSegBufferDestroy(&adjBuffer);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
    ierr = PetscFree(adj);CHKERRQ(ierr);
    graph->xadj = xadj;
    graph->adjncy = adjncy;
  } else {
    Mat       A;
    PetscBool isseqaij, flg_row;

    ierr = MatISGetLocalMat(pc->pmat,&A);CHKERRQ(ierr);
    if (!A->rmap->N || !A->cmap->N) {
      ierr = PCBDDCGraphDestroy(&graph);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATSEQAIJ,&isseqaij);CHKERRQ(ierr);
    if (!isseqaij && filter) {
      PetscBool isseqdense;

      ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQDENSE,&isseqdense);CHKERRQ(ierr);
      if (!isseqdense) {
        ierr = MatConvert(A,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
      } else { /* TODO: rectangular case and LDA */
        PetscScalar *array;
        PetscReal   chop=1.e-6;

        ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
        ierr = MatDenseGetArray(B,&array);CHKERRQ(ierr);
        ierr = MatGetSize(B,&n,NULL);CHKERRQ(ierr);
        for (i=0;i<n;i++) {
          PetscInt j;
          for (j=i+1;j<n;j++) {
            PetscReal thresh = chop*(PetscAbsScalar(array[i*(n+1)])+PetscAbsScalar(array[j*(n+1)]));
            if (PetscAbsScalar(array[i*n+j]) < thresh) array[i*n+j] = 0.;
            if (PetscAbsScalar(array[j*n+i]) < thresh) array[j*n+i] = 0.;
          }
        }
        ierr = MatDenseRestoreArray(B,&array);CHKERRQ(ierr);
        ierr = MatConvert(B,MATSEQAIJ,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
      B = A;
    }
    ierr = MatGetRowIJ(B,0,PETSC_TRUE,PETSC_FALSE,&n,(const PetscInt**)&xadj,(const PetscInt**)&adjncy,&flg_row);CHKERRQ(ierr);

    /* if filter is true, then removes entries lower than PETSC_SMALL in magnitude */
    if (filter) {
      PetscScalar *data;
      PetscInt    j,cum;

      ierr = PetscCalloc2(n+1,&xadj_filtered,xadj[n],&adjncy_filtered);CHKERRQ(ierr);
      ierr = MatSeqAIJGetArray(B,&data);CHKERRQ(ierr);
      cum = 0;
      for (i=0;i<n;i++) {
        PetscInt t;

        for (j=xadj[i];j<xadj[i+1];j++) {
          if (PetscUnlikely(PetscAbsScalar(data[j]) < PETSC_SMALL)) {
            continue;
          }
          adjncy_filtered[cum+xadj_filtered[i]++] = adjncy[j];
        }
        t = xadj_filtered[i];
        xadj_filtered[i] = cum;
        cum += t;
      }
      ierr = MatSeqAIJRestoreArray(B,&data);CHKERRQ(ierr);
      graph->xadj = xadj_filtered;
      graph->adjncy = adjncy_filtered;
    } else {
      graph->xadj = xadj;
      graph->adjncy = adjncy;
    }
  }
  /* compute local connected components using PCBDDCGraph */
  ierr = ISCreateStride(PETSC_COMM_SELF,n,0,1,&is_dummy);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(is_dummy,&l2gmap_dummy);CHKERRQ(ierr);
  ierr = ISDestroy(&is_dummy);CHKERRQ(ierr);
  ierr = PCBDDCGraphInit(graph,l2gmap_dummy,n,PETSC_MAX_INT);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&l2gmap_dummy);CHKERRQ(ierr);
  ierr = PCBDDCGraphSetUp(graph,1,NULL,NULL,0,NULL,NULL);CHKERRQ(ierr);
  ierr = PCBDDCGraphComputeConnectedComponents(graph);CHKERRQ(ierr);

  /* partial clean up */
  ierr = PetscFree2(xadj_filtered,adjncy_filtered);CHKERRQ(ierr);
  if (B) {
    PetscBool flg_row;
    ierr = MatRestoreRowIJ(B,0,PETSC_TRUE,PETSC_FALSE,&n,(const PetscInt**)&xadj,(const PetscInt**)&adjncy,&flg_row);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }
  if (isplex) {
    ierr = PetscFree(xadj);CHKERRQ(ierr);
    ierr = PetscFree(adjncy);CHKERRQ(ierr);
  }

  /* get back data */
  if (isplex) {
    if (ncc) *ncc = graph->ncc;
    if (cc || primalv) {
      Mat          A;
      PetscBT      btv,btvt;
      PetscSection subSection;
      PetscInt     *ids,cum,cump,*cids,*pids;

      ierr = DMPlexGetSubdomainSection(dm,&subSection);CHKERRQ(ierr);
      ierr = MatISGetLocalMat(pc->pmat,&A);CHKERRQ(ierr);
      ierr = PetscMalloc3(A->rmap->n,&ids,graph->ncc+1,&cids,A->rmap->n,&pids);CHKERRQ(ierr);
      ierr = PetscBTCreate(A->rmap->n,&btv);CHKERRQ(ierr);
      ierr = PetscBTCreate(A->rmap->n,&btvt);CHKERRQ(ierr);

      cids[0] = 0;
      for (i = 0, cump = 0, cum = 0; i < graph->ncc; i++) {
        PetscInt j;

        ierr = PetscBTMemzero(A->rmap->n,btvt);CHKERRQ(ierr);
        for (j = graph->cptr[i]; j < graph->cptr[i+1]; j++) {
          PetscInt k, size, *closure = NULL, cell = graph->queue[j];

          ierr = DMPlexGetTransitiveClosure(dm,cell,PETSC_TRUE,&size,&closure);CHKERRQ(ierr);
          for (k = 0; k < 2*size; k += 2) {
            PetscInt s, pp, p = closure[k], off, dof, cdof;

            ierr = PetscSectionGetConstraintDof(subSection,p,&cdof);CHKERRQ(ierr);
            ierr = PetscSectionGetOffset(subSection,p,&off);CHKERRQ(ierr);
            ierr = PetscSectionGetDof(subSection,p,&dof);CHKERRQ(ierr);
            for (s = 0; s < dof-cdof; s++) {
              if (PetscBTLookupSet(btvt,off+s)) continue;
              if (!PetscBTLookup(btv,off+s)) {
                ids[cum++] = off+s;
              } else { /* cross-vertex */
                pids[cump++] = off+s;
              }
            }
            ierr = DMPlexGetTreeParent(dm,p,&pp,NULL);CHKERRQ(ierr);
            if (pp != p) {
              ierr = PetscSectionGetConstraintDof(subSection,pp,&cdof);CHKERRQ(ierr);
              ierr = PetscSectionGetOffset(subSection,pp,&off);CHKERRQ(ierr);
              ierr = PetscSectionGetDof(subSection,pp,&dof);CHKERRQ(ierr);
              for (s = 0; s < dof-cdof; s++) {
                if (PetscBTLookupSet(btvt,off+s)) continue;
                if (!PetscBTLookup(btv,off+s)) {
                  ids[cum++] = off+s;
                } else { /* cross-vertex */
                  pids[cump++] = off+s;
                }
              }
            }
          }
          ierr = DMPlexRestoreTransitiveClosure(dm,cell,PETSC_TRUE,&size,&closure);CHKERRQ(ierr);
        }
        cids[i+1] = cum;
        /* mark dofs as already assigned */
        for (j = cids[i]; j < cids[i+1]; j++) {
          ierr = PetscBTSet(btv,ids[j]);CHKERRQ(ierr);
        }
      }
      if (cc) {
        ierr = PetscMalloc1(graph->ncc,&cc_n);CHKERRQ(ierr);
        for (i = 0; i < graph->ncc; i++) {
          ierr = ISCreateGeneral(PETSC_COMM_SELF,cids[i+1]-cids[i],ids+cids[i],PETSC_COPY_VALUES,&cc_n[i]);CHKERRQ(ierr);
        }
        *cc = cc_n;
      }
      if (primalv) {
        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pc),cump,pids,PETSC_COPY_VALUES,primalv);CHKERRQ(ierr);
      }
      ierr = PetscFree3(ids,cids,pids);CHKERRQ(ierr);
      ierr = PetscBTDestroy(&btv);CHKERRQ(ierr);
      ierr = PetscBTDestroy(&btvt);CHKERRQ(ierr);
    }
  } else {
    if (ncc) *ncc = graph->ncc;
    if (cc) {
      ierr = PetscMalloc1(graph->ncc,&cc_n);CHKERRQ(ierr);
      for (i=0;i<graph->ncc;i++) {
        ierr = ISCreateGeneral(PETSC_COMM_SELF,graph->cptr[i+1]-graph->cptr[i],graph->queue+graph->cptr[i],PETSC_COPY_VALUES,&cc_n[i]);CHKERRQ(ierr);
      }
      *cc = cc_n;
    }
  }
  /* clean up graph */
  graph->xadj = 0;
  graph->adjncy = 0;
  ierr = PCBDDCGraphDestroy(&graph);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCBenignCheck(PC pc, IS zerodiag)
{
  PC_BDDC*       pcbddc = (PC_BDDC*)pc->data;
  PC_IS*         pcis = (PC_IS*)(pc->data);
  IS             dirIS = NULL;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCBDDCGraphGetDirichletDofs(pcbddc->mat_graph,&dirIS);CHKERRQ(ierr);
  if (zerodiag) {
    Mat            A;
    Vec            vec3_N;
    PetscScalar    *vals;
    const PetscInt *idxs;
    PetscInt       nz,*count;

    /* p0 */
    ierr = VecSet(pcis->vec1_N,0.);CHKERRQ(ierr);
    ierr = PetscMalloc1(pcis->n,&vals);CHKERRQ(ierr);
    ierr = ISGetLocalSize(zerodiag,&nz);CHKERRQ(ierr);
    ierr = ISGetIndices(zerodiag,&idxs);CHKERRQ(ierr);
    for (i=0;i<nz;i++) vals[i] = 1.;
    ierr = VecSetValues(pcis->vec1_N,nz,idxs,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(pcis->vec1_N);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(pcis->vec1_N);CHKERRQ(ierr);
    /* v_I */
    ierr = VecSetRandom(pcis->vec2_N,NULL);CHKERRQ(ierr);
    for (i=0;i<nz;i++) vals[i] = 0.;
    ierr = VecSetValues(pcis->vec2_N,nz,idxs,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = ISRestoreIndices(zerodiag,&idxs);CHKERRQ(ierr);
    ierr = ISGetIndices(pcis->is_B_local,&idxs);CHKERRQ(ierr);
    for (i=0;i<pcis->n_B;i++) vals[i] = 0.;
    ierr = VecSetValues(pcis->vec2_N,pcis->n_B,idxs,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = ISRestoreIndices(pcis->is_B_local,&idxs);CHKERRQ(ierr);
    if (dirIS) {
      PetscInt n;

      ierr = ISGetLocalSize(dirIS,&n);CHKERRQ(ierr);
      ierr = ISGetIndices(dirIS,&idxs);CHKERRQ(ierr);
      for (i=0;i<n;i++) vals[i] = 0.;
      ierr = VecSetValues(pcis->vec2_N,n,idxs,vals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = ISRestoreIndices(dirIS,&idxs);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(pcis->vec2_N);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(pcis->vec2_N);CHKERRQ(ierr);
    ierr = VecDuplicate(pcis->vec1_N,&vec3_N);CHKERRQ(ierr);
    ierr = VecSet(vec3_N,0.);CHKERRQ(ierr);
    ierr = MatISGetLocalMat(pc->pmat,&A);CHKERRQ(ierr);
    ierr = MatMult(A,pcis->vec1_N,vec3_N);CHKERRQ(ierr);
    ierr = VecDot(vec3_N,pcis->vec2_N,&vals[0]);CHKERRQ(ierr);
    if (PetscAbsScalar(vals[0]) > 1.e-1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Benign trick can not be applied! b(v_I,p_0) = %1.6e (should be numerically 0.)",PetscAbsScalar(vals[0]));
    ierr = PetscFree(vals);CHKERRQ(ierr);
    ierr = VecDestroy(&vec3_N);CHKERRQ(ierr);

    /* there should not be any pressure dofs lying on the interface */
    ierr = PetscCalloc1(pcis->n,&count);CHKERRQ(ierr);
    ierr = ISGetIndices(pcis->is_B_local,&idxs);CHKERRQ(ierr);
    for (i=0;i<pcis->n_B;i++) count[idxs[i]]++;
    ierr = ISRestoreIndices(pcis->is_B_local,&idxs);CHKERRQ(ierr);
    ierr = ISGetIndices(zerodiag,&idxs);CHKERRQ(ierr);
    for (i=0;i<nz;i++) if (count[idxs[i]]) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Benign trick can not be applied! pressure dof %D is an interface dof",idxs[i]);
    ierr = ISRestoreIndices(zerodiag,&idxs);CHKERRQ(ierr);
    ierr = PetscFree(count);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&dirIS);CHKERRQ(ierr);

  /* check PCBDDCBenignGetOrSetP0 */
  ierr = VecSetRandom(pcis->vec1_global,NULL);CHKERRQ(ierr);
  for (i=0;i<pcbddc->benign_n;i++) pcbddc->benign_p0[i] = -PetscGlobalRank-i;
  ierr = PCBDDCBenignGetOrSetP0(pc,pcis->vec1_global,PETSC_FALSE);CHKERRQ(ierr);
  for (i=0;i<pcbddc->benign_n;i++) pcbddc->benign_p0[i] = 1;
  ierr = PCBDDCBenignGetOrSetP0(pc,pcis->vec1_global,PETSC_TRUE);CHKERRQ(ierr);
  for (i=0;i<pcbddc->benign_n;i++) {
    PetscInt val = PetscRealPart(pcbddc->benign_p0[i]);
    if (val != -PetscGlobalRank-i) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error testing PCBDDCBenignGetOrSetP0! Found %g at %D instead of %g",PetscRealPart(pcbddc->benign_p0[i]),i,-PetscGlobalRank-i);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCBenignDetectSaddlePoint(PC pc, PetscBool reuse, IS *zerodiaglocal)
{
  PC_BDDC*       pcbddc = (PC_BDDC*)pc->data;
  IS             pressures = NULL,zerodiag = NULL,*bzerodiag = NULL,zerodiag_save,*zerodiag_subs;
  PetscInt       nz,n,benign_n,bsp = 1;
  PetscInt       *interior_dofs,n_interior_dofs,nneu;
  PetscBool      sorted,have_null,has_null_pressures,recompute_zerodiag,checkb;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (reuse) goto project_b0;
  ierr = PetscSFDestroy(&pcbddc->benign_sf);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->benign_B0);CHKERRQ(ierr);
  for (n=0;n<pcbddc->benign_n;n++) {
    ierr = ISDestroy(&pcbddc->benign_zerodiag_subs[n]);CHKERRQ(ierr);
  }
  ierr = PetscFree(pcbddc->benign_zerodiag_subs);CHKERRQ(ierr);
  has_null_pressures = PETSC_TRUE;
  have_null = PETSC_TRUE;
  /* if a local information on dofs is present, gets pressure dofs from command line (uses the last field is not provided)
     Without local information, it uses only the zerodiagonal dofs (ok if the pressure block is all zero and it is a scalar field)
     Checks if all the pressure dofs in each subdomain have a zero diagonal
     If not, a change of basis on pressures is not needed
     since the local Schur complements are already SPD
  */
  if (pcbddc->n_ISForDofsLocal) {
    IS        iP = NULL;
    PetscInt  p,*pp;
    PetscBool flg;

    ierr = PetscMalloc1(pcbddc->n_ISForDofsLocal,&pp);CHKERRQ(ierr);
    n    = pcbddc->n_ISForDofsLocal;
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)pc),((PetscObject)pc)->prefix,"BDDC benign options","PC");CHKERRQ(ierr);
    ierr = PetscOptionsIntArray("-pc_bddc_pressure_field","Field id for pressures",NULL,pp,&n,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (!flg) {
      n = 1;
      pp[0] = pcbddc->n_ISForDofsLocal-1;
    }

    bsp = 0;
    for (p=0;p<n;p++) {
      PetscInt bs;

      if (pp[p] < 0 || pp[p] > pcbddc->n_ISForDofsLocal-1) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"Invalid field id for pressures %D",pp[p]);
      ierr = ISGetBlockSize(pcbddc->ISForDofsLocal[pp[p]],&bs);CHKERRQ(ierr);
      bsp += bs;
    }
    ierr = PetscMalloc1(bsp,&bzerodiag);CHKERRQ(ierr);
    bsp  = 0;
    for (p=0;p<n;p++) {
      const PetscInt *idxs;
      PetscInt       b,bs,npl,*bidxs;

      ierr = ISGetBlockSize(pcbddc->ISForDofsLocal[pp[p]],&bs);CHKERRQ(ierr);
      ierr = ISGetLocalSize(pcbddc->ISForDofsLocal[pp[p]],&npl);CHKERRQ(ierr);
      ierr = ISGetIndices(pcbddc->ISForDofsLocal[pp[p]],&idxs);CHKERRQ(ierr);
      ierr = PetscMalloc1(npl/bs,&bidxs);CHKERRQ(ierr);
      for (b=0;b<bs;b++) {
        PetscInt i;

        for (i=0;i<npl/bs;i++) bidxs[i] = idxs[bs*i+b];
        ierr = ISCreateGeneral(PETSC_COMM_SELF,npl/bs,bidxs,PETSC_COPY_VALUES,&bzerodiag[bsp]);CHKERRQ(ierr);
        bsp++;
      }
      ierr = PetscFree(bidxs);CHKERRQ(ierr);
      ierr = ISRestoreIndices(pcbddc->ISForDofsLocal[pp[p]],&idxs);CHKERRQ(ierr);
    }
    ierr = ISConcatenate(PETSC_COMM_SELF,bsp,bzerodiag,&pressures);CHKERRQ(ierr);

    /* remove zeroed out pressures if we are setting up a BDDC solver for a saddle-point FETI-DP */
    ierr = PetscObjectQuery((PetscObject)pc,"__KSPFETIDP_lP",(PetscObject*)&iP);CHKERRQ(ierr);
    if (iP) {
      IS newpressures;

      ierr = ISDifference(pressures,iP,&newpressures);CHKERRQ(ierr);
      ierr = ISDestroy(&pressures);CHKERRQ(ierr);
      pressures = newpressures;
    }
    ierr = ISSorted(pressures,&sorted);CHKERRQ(ierr);
    if (!sorted) {
      ierr = ISSort(pressures);CHKERRQ(ierr);
    }
    ierr = PetscFree(pp);CHKERRQ(ierr);
  }

  /* pcis has not been setup yet, so get the local size from the subdomain matrix */
  ierr = MatGetLocalSize(pcbddc->local_mat,&n,NULL);CHKERRQ(ierr);
  if (!n) pcbddc->benign_change_explicit = PETSC_TRUE;
  ierr = MatFindZeroDiagonals(pcbddc->local_mat,&zerodiag);CHKERRQ(ierr);
  ierr = ISSorted(zerodiag,&sorted);CHKERRQ(ierr);
  if (!sorted) {
    ierr = ISSort(zerodiag);CHKERRQ(ierr);
  }
  ierr = PetscObjectReference((PetscObject)zerodiag);CHKERRQ(ierr);
  zerodiag_save = zerodiag;
  ierr = ISGetLocalSize(zerodiag,&nz);CHKERRQ(ierr);
  if (!nz) {
    if (n) have_null = PETSC_FALSE;
    has_null_pressures = PETSC_FALSE;
    ierr = ISDestroy(&zerodiag);CHKERRQ(ierr);
  }
  recompute_zerodiag = PETSC_FALSE;

  /* in case disconnected subdomains info is present, split the pressures accordingly (otherwise the benign trick could fail) */
  zerodiag_subs    = NULL;
  benign_n         = 0;
  n_interior_dofs  = 0;
  interior_dofs    = NULL;
  nneu             = 0;
  if (pcbddc->NeumannBoundariesLocal) {
    ierr = ISGetLocalSize(pcbddc->NeumannBoundariesLocal,&nneu);CHKERRQ(ierr);
  }
  checkb = (PetscBool)(!pcbddc->NeumannBoundariesLocal || pcbddc->current_level);
  if (checkb) { /* need to compute interior nodes */
    PetscInt n,i,j;
    PetscInt n_neigh,*neigh,*n_shared,**shared;
    PetscInt *iwork;

    ierr = ISLocalToGlobalMappingGetSize(pc->pmat->rmap->mapping,&n);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetInfo(pc->pmat->rmap->mapping,&n_neigh,&neigh,&n_shared,&shared);CHKERRQ(ierr);
    ierr = PetscCalloc1(n,&iwork);CHKERRQ(ierr);
    ierr = PetscMalloc1(n,&interior_dofs);CHKERRQ(ierr);
    for (i=1;i<n_neigh;i++)
      for (j=0;j<n_shared[i];j++)
          iwork[shared[i][j]] += 1;
    for (i=0;i<n;i++)
      if (!iwork[i])
        interior_dofs[n_interior_dofs++] = i;
    ierr = PetscFree(iwork);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingRestoreInfo(pc->pmat->rmap->mapping,&n_neigh,&neigh,&n_shared,&shared);CHKERRQ(ierr);
  }
  if (has_null_pressures) {
    IS             *subs;
    PetscInt       nsubs,i,j,nl;
    const PetscInt *idxs;
    PetscScalar    *array;
    Vec            *work;
    Mat_IS*        matis = (Mat_IS*)(pc->pmat->data);

    subs  = pcbddc->local_subs;
    nsubs = pcbddc->n_local_subs;
    /* these vectors are needed to check if the constant on pressures is in the kernel of the local operator B (i.e. B(v_I,p0) should be zero) */
    if (checkb) {
      ierr = VecDuplicateVecs(matis->y,2,&work);CHKERRQ(ierr);
      ierr = ISGetLocalSize(zerodiag,&nl);CHKERRQ(ierr);
      ierr = ISGetIndices(zerodiag,&idxs);CHKERRQ(ierr);
      /* work[0] = 1_p */
      ierr = VecSet(work[0],0.);CHKERRQ(ierr);
      ierr = VecGetArray(work[0],&array);CHKERRQ(ierr);
      for (j=0;j<nl;j++) array[idxs[j]] = 1.;
      ierr = VecRestoreArray(work[0],&array);CHKERRQ(ierr);
      /* work[0] = 1_v */
      ierr = VecSet(work[1],1.);CHKERRQ(ierr);
      ierr = VecGetArray(work[1],&array);CHKERRQ(ierr);
      for (j=0;j<nl;j++) array[idxs[j]] = 0.;
      ierr = VecRestoreArray(work[1],&array);CHKERRQ(ierr);
      ierr = ISRestoreIndices(zerodiag,&idxs);CHKERRQ(ierr);
    }

    if (nsubs > 1 || bsp > 1) {
      IS       *is;
      PetscInt b,totb;

      totb  = bsp;
      is    = bsp > 1 ? bzerodiag : &zerodiag;
      nsubs = PetscMax(nsubs,1);
      ierr  = PetscCalloc1(nsubs*totb,&zerodiag_subs);CHKERRQ(ierr);
      for (b=0;b<totb;b++) {
        for (i=0;i<nsubs;i++) {
          ISLocalToGlobalMapping l2g;
          IS                     t_zerodiag_subs;
          PetscInt               nl;

          if (subs) {
            ierr = ISLocalToGlobalMappingCreateIS(subs[i],&l2g);CHKERRQ(ierr);
          } else {
            IS tis;

            ierr = MatGetLocalSize(pcbddc->local_mat,&nl,NULL);CHKERRQ(ierr);
            ierr = ISCreateStride(PETSC_COMM_SELF,nl,0,1,&tis);CHKERRQ(ierr);
            ierr = ISLocalToGlobalMappingCreateIS(tis,&l2g);CHKERRQ(ierr);
            ierr = ISDestroy(&tis);CHKERRQ(ierr);
          }
          ierr = ISGlobalToLocalMappingApplyIS(l2g,IS_GTOLM_DROP,is[b],&t_zerodiag_subs);CHKERRQ(ierr);
          ierr = ISGetLocalSize(t_zerodiag_subs,&nl);CHKERRQ(ierr);
          if (nl) {
            PetscBool valid = PETSC_TRUE;

            if (checkb) {
              ierr = VecSet(matis->x,0);CHKERRQ(ierr);
              ierr = ISGetLocalSize(subs[i],&nl);CHKERRQ(ierr);
              ierr = ISGetIndices(subs[i],&idxs);CHKERRQ(ierr);
              ierr = VecGetArray(matis->x,&array);CHKERRQ(ierr);
              for (j=0;j<nl;j++) array[idxs[j]] = 1.;
              ierr = VecRestoreArray(matis->x,&array);CHKERRQ(ierr);
              ierr = ISRestoreIndices(subs[i],&idxs);CHKERRQ(ierr);
              ierr = VecPointwiseMult(matis->x,work[0],matis->x);CHKERRQ(ierr);
              ierr = MatMult(matis->A,matis->x,matis->y);CHKERRQ(ierr);
              ierr = VecPointwiseMult(matis->y,work[1],matis->y);CHKERRQ(ierr);
              ierr = VecGetArray(matis->y,&array);CHKERRQ(ierr);
              for (j=0;j<n_interior_dofs;j++) {
                if (PetscAbsScalar(array[interior_dofs[j]]) > PETSC_SMALL) {
                  valid = PETSC_FALSE;
                  break;
                }
              }
              ierr = VecRestoreArray(matis->y,&array);CHKERRQ(ierr);
            }
            if (valid && nneu) {
              const PetscInt *idxs;
              PetscInt       nzb;

              ierr = ISGetIndices(pcbddc->NeumannBoundariesLocal,&idxs);CHKERRQ(ierr);
              ierr = ISGlobalToLocalMappingApply(l2g,IS_GTOLM_DROP,nneu,idxs,&nzb,NULL);CHKERRQ(ierr);
              ierr = ISRestoreIndices(pcbddc->NeumannBoundariesLocal,&idxs);CHKERRQ(ierr);
              if (nzb) valid = PETSC_FALSE;
            }
            if (valid && pressures) {
              IS       t_pressure_subs,tmp;
              PetscInt i1,i2;

              ierr = ISGlobalToLocalMappingApplyIS(l2g,IS_GTOLM_DROP,pressures,&t_pressure_subs);CHKERRQ(ierr);
              ierr = ISEmbed(t_zerodiag_subs,t_pressure_subs,PETSC_TRUE,&tmp);CHKERRQ(ierr);
              ierr = ISGetLocalSize(tmp,&i1);CHKERRQ(ierr);
              ierr = ISGetLocalSize(t_zerodiag_subs,&i2);CHKERRQ(ierr);
              if (i2 != i1) valid = PETSC_FALSE;
              ierr = ISDestroy(&t_pressure_subs);CHKERRQ(ierr);
              ierr = ISDestroy(&tmp);CHKERRQ(ierr);
            }
            if (valid) {
              ierr = ISLocalToGlobalMappingApplyIS(l2g,t_zerodiag_subs,&zerodiag_subs[benign_n]);CHKERRQ(ierr);
              benign_n++;
            } else recompute_zerodiag = PETSC_TRUE;
          }
          ierr = ISDestroy(&t_zerodiag_subs);CHKERRQ(ierr);
          ierr = ISLocalToGlobalMappingDestroy(&l2g);CHKERRQ(ierr);
        }
      }
    } else { /* there's just one subdomain (or zero if they have not been detected */
      PetscBool valid = PETSC_TRUE;

      if (nneu) valid = PETSC_FALSE;
      if (valid && pressures) {
        ierr = ISEqual(pressures,zerodiag,&valid);CHKERRQ(ierr);
      }
      if (valid && checkb) {
        ierr = MatMult(matis->A,work[0],matis->x);CHKERRQ(ierr);
        ierr = VecPointwiseMult(matis->x,work[1],matis->x);CHKERRQ(ierr);
        ierr = VecGetArray(matis->x,&array);CHKERRQ(ierr);
        for (j=0;j<n_interior_dofs;j++) {
          if (PetscAbsScalar(array[interior_dofs[j]]) > PETSC_SMALL) {
            valid = PETSC_FALSE;
            break;
          }
        }
        ierr = VecRestoreArray(matis->x,&array);CHKERRQ(ierr);
      }
      if (valid) {
        benign_n = 1;
        ierr = PetscMalloc1(benign_n,&zerodiag_subs);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)zerodiag);CHKERRQ(ierr);
        zerodiag_subs[0] = zerodiag;
      }
    }
    if (checkb) {
      ierr = VecDestroyVecs(2,&work);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(interior_dofs);CHKERRQ(ierr);

  if (!benign_n) {
    PetscInt n;

    ierr = ISDestroy(&zerodiag);CHKERRQ(ierr);
    recompute_zerodiag = PETSC_FALSE;
    ierr = MatGetLocalSize(pcbddc->local_mat,&n,NULL);CHKERRQ(ierr);
    if (n) have_null = PETSC_FALSE;
  }

  /* final check for null pressures */
  if (zerodiag && pressures) {
    ierr = ISEqual(pressures,zerodiag,&have_null);CHKERRQ(ierr);
  }

  if (recompute_zerodiag) {
    ierr = ISDestroy(&zerodiag);CHKERRQ(ierr);
    if (benign_n == 1) {
      ierr = PetscObjectReference((PetscObject)zerodiag_subs[0]);CHKERRQ(ierr);
      zerodiag = zerodiag_subs[0];
    } else {
      PetscInt i,nzn,*new_idxs;

      nzn = 0;
      for (i=0;i<benign_n;i++) {
        PetscInt ns;
        ierr = ISGetLocalSize(zerodiag_subs[i],&ns);CHKERRQ(ierr);
        nzn += ns;
      }
      ierr = PetscMalloc1(nzn,&new_idxs);CHKERRQ(ierr);
      nzn = 0;
      for (i=0;i<benign_n;i++) {
        PetscInt ns,*idxs;
        ierr = ISGetLocalSize(zerodiag_subs[i],&ns);CHKERRQ(ierr);
        ierr = ISGetIndices(zerodiag_subs[i],(const PetscInt**)&idxs);CHKERRQ(ierr);
        ierr = PetscArraycpy(new_idxs+nzn,idxs,ns);CHKERRQ(ierr);
        ierr = ISRestoreIndices(zerodiag_subs[i],(const PetscInt**)&idxs);CHKERRQ(ierr);
        nzn += ns;
      }
      ierr = PetscSortInt(nzn,new_idxs);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF,nzn,new_idxs,PETSC_OWN_POINTER,&zerodiag);CHKERRQ(ierr);
    }
    have_null = PETSC_FALSE;
  }

  /* determines if the coarse solver will be singular or not */
  ierr = MPIU_Allreduce(&have_null,&pcbddc->benign_null,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);

  /* Prepare matrix to compute no-net-flux */
  if (pcbddc->compute_nonetflux && !pcbddc->divudotp) {
    Mat                    A,loc_divudotp;
    ISLocalToGlobalMapping rl2g,cl2g,l2gmap;
    IS                     row,col,isused = NULL;
    PetscInt               M,N,n,st,n_isused;

    if (pressures) {
      isused = pressures;
    } else {
      isused = zerodiag_save;
    }
    ierr = MatGetLocalToGlobalMapping(pc->pmat,&l2gmap,NULL);CHKERRQ(ierr);
    ierr = MatISGetLocalMat(pc->pmat,&A);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,&n,NULL);CHKERRQ(ierr);
    if (!isused && n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Don't know how to extract div u dot p! Please provide the pressure field");
    n_isused = 0;
    if (isused) {
      ierr = ISGetLocalSize(isused,&n_isused);CHKERRQ(ierr);
    }
    ierr = MPI_Scan(&n_isused,&st,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
    st = st-n_isused;
    if (n) {
      const PetscInt *gidxs;

      ierr = MatCreateSubMatrix(A,isused,NULL,MAT_INITIAL_MATRIX,&loc_divudotp);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingGetIndices(l2gmap,&gidxs);CHKERRQ(ierr);
      /* TODO: extend ISCreateStride with st = PETSC_DECIDE */
      ierr = ISCreateStride(PetscObjectComm((PetscObject)pc),n_isused,st,1,&row);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pc),n,gidxs,PETSC_COPY_VALUES,&col);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingRestoreIndices(l2gmap,&gidxs);CHKERRQ(ierr);
    } else {
      ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,0,0,1,NULL,&loc_divudotp);CHKERRQ(ierr);
      ierr = ISCreateStride(PetscObjectComm((PetscObject)pc),n_isused,st,1,&row);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pc),0,NULL,PETSC_COPY_VALUES,&col);CHKERRQ(ierr);
    }
    ierr = MatGetSize(pc->pmat,NULL,&N);CHKERRQ(ierr);
    ierr = ISGetSize(row,&M);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreateIS(row,&rl2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreateIS(col,&cl2g);CHKERRQ(ierr);
    ierr = ISDestroy(&row);CHKERRQ(ierr);
    ierr = ISDestroy(&col);CHKERRQ(ierr);
    ierr = MatCreate(PetscObjectComm((PetscObject)pc),&pcbddc->divudotp);CHKERRQ(ierr);
    ierr = MatSetType(pcbddc->divudotp,MATIS);CHKERRQ(ierr);
    ierr = MatSetSizes(pcbddc->divudotp,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
    ierr = MatSetLocalToGlobalMapping(pcbddc->divudotp,rl2g,cl2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&rl2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&cl2g);CHKERRQ(ierr);
    ierr = MatISSetLocalMat(pcbddc->divudotp,loc_divudotp);CHKERRQ(ierr);
    ierr = MatDestroy(&loc_divudotp);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(pcbddc->divudotp,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(pcbddc->divudotp,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&zerodiag_save);CHKERRQ(ierr);
  ierr = ISDestroy(&pressures);CHKERRQ(ierr);
  if (bzerodiag) {
    PetscInt i;

    for (i=0;i<bsp;i++) {
      ierr = ISDestroy(&bzerodiag[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(bzerodiag);CHKERRQ(ierr);
  }
  pcbddc->benign_n = benign_n;
  pcbddc->benign_zerodiag_subs = zerodiag_subs;

  /* determines if the problem has subdomains with 0 pressure block */
  have_null = (PetscBool)(!!pcbddc->benign_n);
  ierr = MPIU_Allreduce(&have_null,&pcbddc->benign_have_null,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);

project_b0:
  ierr = MatGetLocalSize(pcbddc->local_mat,&n,NULL);CHKERRQ(ierr);
  /* change of basis and p0 dofs */
  if (pcbddc->benign_n) {
    PetscInt i,s,*nnz;

    /* local change of basis for pressures */
    ierr = MatDestroy(&pcbddc->benign_change);CHKERRQ(ierr);
    ierr = MatCreate(PetscObjectComm((PetscObject)pcbddc->local_mat),&pcbddc->benign_change);CHKERRQ(ierr);
    ierr = MatSetType(pcbddc->benign_change,MATAIJ);CHKERRQ(ierr);
    ierr = MatSetSizes(pcbddc->benign_change,n,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = PetscMalloc1(n,&nnz);CHKERRQ(ierr);
    for (i=0;i<n;i++) nnz[i] = 1; /* defaults to identity */
    for (i=0;i<pcbddc->benign_n;i++) {
      const PetscInt *idxs;
      PetscInt       nzs,j;

      ierr = ISGetLocalSize(pcbddc->benign_zerodiag_subs[i],&nzs);CHKERRQ(ierr);
      ierr = ISGetIndices(pcbddc->benign_zerodiag_subs[i],&idxs);CHKERRQ(ierr);
      for (j=0;j<nzs-1;j++) nnz[idxs[j]] = 2; /* change on pressures */
      nnz[idxs[nzs-1]] = nzs; /* last local pressure dof in subdomain */
      ierr = ISRestoreIndices(pcbddc->benign_zerodiag_subs[i],&idxs);CHKERRQ(ierr);
    }
    ierr = MatSeqAIJSetPreallocation(pcbddc->benign_change,0,nnz);CHKERRQ(ierr);
    ierr = MatSetOption(pcbddc->benign_change,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscFree(nnz);CHKERRQ(ierr);
    /* set identity by default */
    for (i=0;i<n;i++) {
      ierr = MatSetValue(pcbddc->benign_change,i,i,1.,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = PetscFree3(pcbddc->benign_p0_lidx,pcbddc->benign_p0_gidx,pcbddc->benign_p0);CHKERRQ(ierr);
    ierr = PetscMalloc3(pcbddc->benign_n,&pcbddc->benign_p0_lidx,pcbddc->benign_n,&pcbddc->benign_p0_gidx,pcbddc->benign_n,&pcbddc->benign_p0);CHKERRQ(ierr);
    /* set change on pressures */
    for (s=0;s<pcbddc->benign_n;s++) {
      PetscScalar    *array;
      const PetscInt *idxs;
      PetscInt       nzs;

      ierr = ISGetLocalSize(pcbddc->benign_zerodiag_subs[s],&nzs);CHKERRQ(ierr);
      ierr = ISGetIndices(pcbddc->benign_zerodiag_subs[s],&idxs);CHKERRQ(ierr);
      for (i=0;i<nzs-1;i++) {
        PetscScalar vals[2];
        PetscInt    cols[2];

        cols[0] = idxs[i];
        cols[1] = idxs[nzs-1];
        vals[0] = 1.;
        vals[1] = 1.;
        ierr = MatSetValues(pcbddc->benign_change,1,cols,2,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = PetscMalloc1(nzs,&array);CHKERRQ(ierr);
      for (i=0;i<nzs-1;i++) array[i] = -1.;
      array[nzs-1] = 1.;
      ierr = MatSetValues(pcbddc->benign_change,1,idxs+nzs-1,nzs,idxs,array,INSERT_VALUES);CHKERRQ(ierr);
      /* store local idxs for p0 */
      pcbddc->benign_p0_lidx[s] = idxs[nzs-1];
      ierr = ISRestoreIndices(pcbddc->benign_zerodiag_subs[s],&idxs);CHKERRQ(ierr);
      ierr = PetscFree(array);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(pcbddc->benign_change,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(pcbddc->benign_change,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    /* project if needed */
    if (pcbddc->benign_change_explicit) {
      Mat M;

      ierr = MatPtAP(pcbddc->local_mat,pcbddc->benign_change,MAT_INITIAL_MATRIX,2.0,&M);CHKERRQ(ierr);
      ierr = MatDestroy(&pcbddc->local_mat);CHKERRQ(ierr);
      ierr = MatSeqAIJCompress(M,&pcbddc->local_mat);CHKERRQ(ierr);
      ierr = MatDestroy(&M);CHKERRQ(ierr);
    }
    /* store global idxs for p0 */
    ierr = ISLocalToGlobalMappingApply(pc->pmat->rmap->mapping,pcbddc->benign_n,pcbddc->benign_p0_lidx,pcbddc->benign_p0_gidx);CHKERRQ(ierr);
  }
  *zerodiaglocal = zerodiag;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCBenignGetOrSetP0(PC pc, Vec v, PetscBool get)
{
  PC_BDDC*       pcbddc = (PC_BDDC*)pc->data;
  PetscScalar    *array;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!pcbddc->benign_sf) {
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)pc),&pcbddc->benign_sf);CHKERRQ(ierr);
    ierr = PetscSFSetGraphLayout(pcbddc->benign_sf,pc->pmat->rmap,pcbddc->benign_n,NULL,PETSC_OWN_POINTER,pcbddc->benign_p0_gidx);CHKERRQ(ierr);
  }
  if (get) {
    ierr = VecGetArrayRead(v,(const PetscScalar**)&array);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(pcbddc->benign_sf,MPIU_SCALAR,array,pcbddc->benign_p0);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(pcbddc->benign_sf,MPIU_SCALAR,array,pcbddc->benign_p0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(v,(const PetscScalar**)&array);CHKERRQ(ierr);
  } else {
    ierr = VecGetArray(v,&array);CHKERRQ(ierr);
    ierr = PetscSFReduceBegin(pcbddc->benign_sf,MPIU_SCALAR,pcbddc->benign_p0,array,MPIU_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(pcbddc->benign_sf,MPIU_SCALAR,pcbddc->benign_p0,array,MPIU_REPLACE);CHKERRQ(ierr);
    ierr = VecRestoreArray(v,&array);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCBenignPopOrPushB0(PC pc, PetscBool pop)
{
  PC_BDDC*       pcbddc = (PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* TODO: add error checking
    - avoid nested pop (or push) calls.
    - cannot push before pop.
    - cannot call this if pcbddc->local_mat is NULL
  */
  if (!pcbddc->benign_n) {
    PetscFunctionReturn(0);
  }
  if (pop) {
    if (pcbddc->benign_change_explicit) {
      IS       is_p0;
      MatReuse reuse;

      /* extract B_0 */
      reuse = MAT_INITIAL_MATRIX;
      if (pcbddc->benign_B0) {
        reuse = MAT_REUSE_MATRIX;
      }
      ierr = ISCreateGeneral(PETSC_COMM_SELF,pcbddc->benign_n,pcbddc->benign_p0_lidx,PETSC_COPY_VALUES,&is_p0);CHKERRQ(ierr);
      ierr = MatCreateSubMatrix(pcbddc->local_mat,is_p0,NULL,reuse,&pcbddc->benign_B0);CHKERRQ(ierr);
      /* remove rows and cols from local problem */
      ierr = MatSetOption(pcbddc->local_mat,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatSetOption(pcbddc->local_mat,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatZeroRowsColumnsIS(pcbddc->local_mat,is_p0,1.0,NULL,NULL);CHKERRQ(ierr);
      ierr = ISDestroy(&is_p0);CHKERRQ(ierr);
    } else {
      Mat_IS      *matis = (Mat_IS*)pc->pmat->data;
      PetscScalar *vals;
      PetscInt    i,n,*idxs_ins;

      ierr = VecGetLocalSize(matis->y,&n);CHKERRQ(ierr);
      ierr = PetscMalloc2(n,&idxs_ins,n,&vals);CHKERRQ(ierr);
      if (!pcbddc->benign_B0) {
        PetscInt *nnz;
        ierr = MatCreate(PetscObjectComm((PetscObject)pcbddc->local_mat),&pcbddc->benign_B0);CHKERRQ(ierr);
        ierr = MatSetType(pcbddc->benign_B0,MATAIJ);CHKERRQ(ierr);
        ierr = MatSetSizes(pcbddc->benign_B0,pcbddc->benign_n,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
        ierr = PetscMalloc1(pcbddc->benign_n,&nnz);CHKERRQ(ierr);
        for (i=0;i<pcbddc->benign_n;i++) {
          ierr = ISGetLocalSize(pcbddc->benign_zerodiag_subs[i],&nnz[i]);CHKERRQ(ierr);
          nnz[i] = n - nnz[i];
        }
        ierr = MatSeqAIJSetPreallocation(pcbddc->benign_B0,0,nnz);CHKERRQ(ierr);
        ierr = MatSetOption(pcbddc->benign_B0,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
        ierr = PetscFree(nnz);CHKERRQ(ierr);
      }

      for (i=0;i<pcbddc->benign_n;i++) {
        PetscScalar *array;
        PetscInt    *idxs,j,nz,cum;

        ierr = VecSet(matis->x,0.);CHKERRQ(ierr);
        ierr = ISGetLocalSize(pcbddc->benign_zerodiag_subs[i],&nz);CHKERRQ(ierr);
        ierr = ISGetIndices(pcbddc->benign_zerodiag_subs[i],(const PetscInt**)&idxs);CHKERRQ(ierr);
        for (j=0;j<nz;j++) vals[j] = 1.;
        ierr = VecSetValues(matis->x,nz,idxs,vals,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecAssemblyBegin(matis->x);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(matis->x);CHKERRQ(ierr);
        ierr = VecSet(matis->y,0.);CHKERRQ(ierr);
        ierr = MatMult(matis->A,matis->x,matis->y);CHKERRQ(ierr);
        ierr = VecGetArray(matis->y,&array);CHKERRQ(ierr);
        cum = 0;
        for (j=0;j<n;j++) {
          if (PetscUnlikely(PetscAbsScalar(array[j]) > PETSC_SMALL)) {
            vals[cum] = array[j];
            idxs_ins[cum] = j;
            cum++;
          }
        }
        ierr = MatSetValues(pcbddc->benign_B0,1,&i,cum,idxs_ins,vals,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecRestoreArray(matis->y,&array);CHKERRQ(ierr);
        ierr = ISRestoreIndices(pcbddc->benign_zerodiag_subs[i],(const PetscInt**)&idxs);CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(pcbddc->benign_B0,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(pcbddc->benign_B0,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = PetscFree2(idxs_ins,vals);CHKERRQ(ierr);
    }
  } else { /* push */
    if (pcbddc->benign_change_explicit) {
      PetscInt i;

      for (i=0;i<pcbddc->benign_n;i++) {
        PetscScalar *B0_vals;
        PetscInt    *B0_cols,B0_ncol;

        ierr = MatGetRow(pcbddc->benign_B0,i,&B0_ncol,(const PetscInt**)&B0_cols,(const PetscScalar**)&B0_vals);CHKERRQ(ierr);
        ierr = MatSetValues(pcbddc->local_mat,1,pcbddc->benign_p0_lidx+i,B0_ncol,B0_cols,B0_vals,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatSetValues(pcbddc->local_mat,B0_ncol,B0_cols,1,pcbddc->benign_p0_lidx+i,B0_vals,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatSetValue(pcbddc->local_mat,pcbddc->benign_p0_lidx[i],pcbddc->benign_p0_lidx[i],0.0,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatRestoreRow(pcbddc->benign_B0,i,&B0_ncol,(const PetscInt**)&B0_cols,(const PetscScalar**)&B0_vals);CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(pcbddc->local_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(pcbddc->local_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cannot push B0!");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCAdaptiveSelection(PC pc)
{
  PC_BDDC*        pcbddc = (PC_BDDC*)pc->data;
  PCBDDCSubSchurs sub_schurs = pcbddc->sub_schurs;
  PetscBLASInt    B_dummyint,B_neigs,B_ierr,B_lwork;
  PetscBLASInt    *B_iwork,*B_ifail;
  PetscScalar     *work,lwork;
  PetscScalar     *St,*S,*eigv;
  PetscScalar     *Sarray,*Starray;
  PetscReal       *eigs,thresh,lthresh,uthresh;
  PetscInt        i,nmax,nmin,nv,cum,mss,cum2,cumarray,maxneigs;
  PetscBool       allocated_S_St;
#if defined(PETSC_USE_COMPLEX)
  PetscReal       *rwork;
#endif
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!sub_schurs) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Adaptive selection of constraints requires SubSchurs data");
  if (!sub_schurs->schur_explicit) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Adaptive selection of constraints requires MUMPS and/or MKL_CPARDISO");
  if (sub_schurs->n_subs && (!sub_schurs->is_symmetric)) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"Adaptive selection not yet implemented for this matrix pencil (herm %d, symm %d, posdef %d)",sub_schurs->is_hermitian,sub_schurs->is_symmetric,sub_schurs->is_posdef);
  ierr = PetscLogEventBegin(PC_BDDC_AdaptiveSetUp[pcbddc->current_level],pc,0,0,0);CHKERRQ(ierr);

  if (pcbddc->dbg_flag) {
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Check adaptive selection of constraints\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer);CHKERRQ(ierr);
  }

  if (pcbddc->dbg_flag) {
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d cc %D (%d,%d).\n",PetscGlobalRank,sub_schurs->n_subs,sub_schurs->is_hermitian,sub_schurs->is_posdef);CHKERRQ(ierr);
  }

  /* max size of subsets */
  mss = 0;
  for (i=0;i<sub_schurs->n_subs;i++) {
    PetscInt subset_size;

    ierr = ISGetLocalSize(sub_schurs->is_subs[i],&subset_size);CHKERRQ(ierr);
    mss = PetscMax(mss,subset_size);
  }

  /* min/max and threshold */
  nmax = pcbddc->adaptive_nmax > 0 ? pcbddc->adaptive_nmax : mss;
  nmin = pcbddc->adaptive_nmin > 0 ? pcbddc->adaptive_nmin : 0;
  nmax = PetscMax(nmin,nmax);
  allocated_S_St = PETSC_FALSE;
  if (nmin || !sub_schurs->is_posdef) { /* XXX */
    allocated_S_St = PETSC_TRUE;
  }

  /* allocate lapack workspace */
  cum = cum2 = 0;
  maxneigs = 0;
  for (i=0;i<sub_schurs->n_subs;i++) {
    PetscInt n,subset_size;

    ierr = ISGetLocalSize(sub_schurs->is_subs[i],&subset_size);CHKERRQ(ierr);
    n = PetscMin(subset_size,nmax);
    cum += subset_size;
    cum2 += subset_size*n;
    maxneigs = PetscMax(maxneigs,n);
  }
  lwork = 0;
  if (mss) {
    if (sub_schurs->is_symmetric) {
      PetscScalar  sdummy = 0.;
      PetscBLASInt B_itype = 1;
      PetscBLASInt B_N = mss, idummy = 0;
      PetscReal    rdummy = 0.,zero = 0.0;
      PetscReal    eps = 0.0; /* dlamch? */

      B_lwork = -1;
      /* some implementations may complain about NULL pointers, even if we are querying */
      S = &sdummy;
      St = &sdummy;
      eigs = &rdummy;
      eigv = &sdummy;
      B_iwork = &idummy;
      B_ifail = &idummy;
#if defined(PETSC_USE_COMPLEX)
      rwork = &rdummy;
#endif
      thresh = 1.0;
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
      PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&zero,&thresh,&B_dummyint,&B_dummyint,&eps,&B_neigs,eigs,eigv,&B_N,&lwork,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
      PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&zero,&thresh,&B_dummyint,&B_dummyint,&eps,&B_neigs,eigs,eigv,&B_N,&lwork,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
      if (B_ierr != 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to SYGVX Lapack routine %d",(int)B_ierr);
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not yet implemented");
  }

  nv = 0;
  if (sub_schurs->is_vertices && pcbddc->use_vertices) { /* complement set of active subsets, each entry is a vertex (boundary made by active subsets, vertices and dirichlet dofs) */
    ierr = ISGetLocalSize(sub_schurs->is_vertices,&nv);CHKERRQ(ierr);
  }
  ierr = PetscBLASIntCast((PetscInt)PetscRealPart(lwork),&B_lwork);CHKERRQ(ierr);
  if (allocated_S_St) {
    ierr = PetscMalloc2(mss*mss,&S,mss*mss,&St);CHKERRQ(ierr);
  }
  ierr = PetscMalloc5(mss*mss,&eigv,mss,&eigs,B_lwork,&work,5*mss,&B_iwork,mss,&B_ifail);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc1(7*mss,&rwork);CHKERRQ(ierr);
#endif
  ierr = PetscMalloc5(nv+sub_schurs->n_subs,&pcbddc->adaptive_constraints_n,
                      nv+sub_schurs->n_subs+1,&pcbddc->adaptive_constraints_idxs_ptr,
                      nv+sub_schurs->n_subs+1,&pcbddc->adaptive_constraints_data_ptr,
                      nv+cum,&pcbddc->adaptive_constraints_idxs,
                      nv+cum2,&pcbddc->adaptive_constraints_data);CHKERRQ(ierr);
  ierr = PetscArrayzero(pcbddc->adaptive_constraints_n,nv+sub_schurs->n_subs);CHKERRQ(ierr);

  maxneigs = 0;
  cum = cumarray = 0;
  pcbddc->adaptive_constraints_idxs_ptr[0] = 0;
  pcbddc->adaptive_constraints_data_ptr[0] = 0;
  if (sub_schurs->is_vertices && pcbddc->use_vertices) {
    const PetscInt *idxs;

    ierr = ISGetIndices(sub_schurs->is_vertices,&idxs);CHKERRQ(ierr);
    for (cum=0;cum<nv;cum++) {
      pcbddc->adaptive_constraints_n[cum] = 1;
      pcbddc->adaptive_constraints_idxs[cum] = idxs[cum];
      pcbddc->adaptive_constraints_data[cum] = 1.0;
      pcbddc->adaptive_constraints_idxs_ptr[cum+1] = pcbddc->adaptive_constraints_idxs_ptr[cum]+1;
      pcbddc->adaptive_constraints_data_ptr[cum+1] = pcbddc->adaptive_constraints_data_ptr[cum]+1;
    }
    ierr = ISRestoreIndices(sub_schurs->is_vertices,&idxs);CHKERRQ(ierr);
  }

  if (mss) { /* multilevel */
    ierr = MatSeqAIJGetArray(sub_schurs->sum_S_Ej_inv_all,&Sarray);CHKERRQ(ierr);
    ierr = MatSeqAIJGetArray(sub_schurs->sum_S_Ej_tilda_all,&Starray);CHKERRQ(ierr);
  }

  lthresh = pcbddc->adaptive_threshold[0];
  uthresh = pcbddc->adaptive_threshold[1];
  for (i=0;i<sub_schurs->n_subs;i++) {
    const PetscInt *idxs;
    PetscReal      upper,lower;
    PetscInt       j,subset_size,eigs_start = 0;
    PetscBLASInt   B_N;
    PetscBool      same_data = PETSC_FALSE;
    PetscBool      scal = PETSC_FALSE;

    if (pcbddc->use_deluxe_scaling) {
      upper = PETSC_MAX_REAL;
      lower = uthresh;
    } else {
      if (!sub_schurs->is_posdef) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not yet implemented without deluxe scaling");
      upper = 1./uthresh;
      lower = 0.;
    }
    ierr = ISGetLocalSize(sub_schurs->is_subs[i],&subset_size);CHKERRQ(ierr);
    ierr = ISGetIndices(sub_schurs->is_subs[i],&idxs);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(subset_size,&B_N);CHKERRQ(ierr);
    /* this is experimental: we assume the dofs have been properly grouped to have
       the diagonal blocks Schur complements either positive or negative definite (true for Stokes) */
    if (!sub_schurs->is_posdef) {
      Mat T;

      for (j=0;j<subset_size;j++) {
        if (PetscRealPart(*(Sarray+cumarray+j*(subset_size+1))) < 0.0) {
          ierr = MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,Sarray+cumarray,&T);CHKERRQ(ierr);
          ierr = MatScale(T,-1.0);CHKERRQ(ierr);
          ierr = MatDestroy(&T);CHKERRQ(ierr);
          ierr = MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,Starray+cumarray,&T);CHKERRQ(ierr);
          ierr = MatScale(T,-1.0);CHKERRQ(ierr);
          ierr = MatDestroy(&T);CHKERRQ(ierr);
          if (sub_schurs->change_primal_sub) {
            PetscInt       nz,k;
            const PetscInt *idxs;

            ierr = ISGetLocalSize(sub_schurs->change_primal_sub[i],&nz);CHKERRQ(ierr);
            ierr = ISGetIndices(sub_schurs->change_primal_sub[i],&idxs);CHKERRQ(ierr);
            for (k=0;k<nz;k++) {
              *( Sarray + cumarray + idxs[k]*(subset_size+1)) *= -1.0;
              *(Starray + cumarray + idxs[k]*(subset_size+1))  = 0.0;
            }
            ierr = ISRestoreIndices(sub_schurs->change_primal_sub[i],&idxs);CHKERRQ(ierr);
          }
          scal = PETSC_TRUE;
          break;
        }
      }
    }

    if (allocated_S_St) { /* S and S_t should be copied since we could need them later */
      if (sub_schurs->is_symmetric) {
        PetscInt j,k;
        if (sub_schurs->n_subs == 1) { /* zeroing memory to use PetscArraycmp() later */
          ierr = PetscArrayzero(S,subset_size*subset_size);CHKERRQ(ierr);
          ierr = PetscArrayzero(St,subset_size*subset_size);CHKERRQ(ierr);
        }
        for (j=0;j<subset_size;j++) {
          for (k=j;k<subset_size;k++) {
            S [j*subset_size+k] = Sarray [cumarray+j*subset_size+k];
            St[j*subset_size+k] = Starray[cumarray+j*subset_size+k];
          }
        }
      } else {
        ierr = PetscArraycpy(S,Sarray+cumarray,subset_size*subset_size);CHKERRQ(ierr);
        ierr = PetscArraycpy(St,Starray+cumarray,subset_size*subset_size);CHKERRQ(ierr);
      }
    } else {
      S = Sarray + cumarray;
      St = Starray + cumarray;
    }
    /* see if we can save some work */
    if (sub_schurs->n_subs == 1 && pcbddc->use_deluxe_scaling) {
      ierr = PetscArraycmp(S,St,subset_size*subset_size,&same_data);CHKERRQ(ierr);
    }

    if (same_data && !sub_schurs->change) { /* there's no need of constraints here */
      B_neigs = 0;
    } else {
      if (sub_schurs->is_symmetric) {
        PetscBLASInt B_itype = 1;
        PetscBLASInt B_IL, B_IU;
        PetscReal    eps = -1.0; /* dlamch? */
        PetscInt     nmin_s;
        PetscBool    compute_range;

        B_neigs = 0;
        compute_range = (PetscBool)!same_data;
        if (nmin >= subset_size) compute_range = PETSC_FALSE;

        if (pcbddc->dbg_flag) {
          PetscInt nc = 0;

          if (sub_schurs->change_primal_sub) {
            ierr = ISGetLocalSize(sub_schurs->change_primal_sub[i],&nc);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Computing for sub %D/%D size %D count %D fid %D (range %d) (change %D).\n",i,sub_schurs->n_subs,subset_size,pcbddc->mat_graph->count[idxs[0]]+1,pcbddc->mat_graph->which_dof[idxs[0]],compute_range,nc);CHKERRQ(ierr);
        }

        ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
        if (compute_range) {

          /* ask for eigenvalues larger than thresh */
          if (sub_schurs->is_posdef) {
#if defined(PETSC_USE_COMPLEX)
            PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&lower,&upper,&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
            PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&lower,&upper,&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
            ierr = PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0);CHKERRQ(ierr);
          } else { /* no theory so far, but it works nicely */
            PetscInt  recipe = 0,recipe_m = 1;
            PetscReal bb[2];

            ierr = PetscOptionsGetInt(NULL,((PetscObject)pc)->prefix,"-pc_bddc_adaptive_recipe",&recipe,NULL);CHKERRQ(ierr);
            switch (recipe) {
            case 0:
              if (scal) { bb[0] = PETSC_MIN_REAL; bb[1] = lthresh; }
              else { bb[0] = uthresh; bb[1] = PETSC_MAX_REAL; }
#if defined(PETSC_USE_COMPLEX)
              PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
              PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
              ierr = PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0);CHKERRQ(ierr);
              break;
            case 1:
              bb[0] = PETSC_MIN_REAL; bb[1] = lthresh*lthresh;
#if defined(PETSC_USE_COMPLEX)
              PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
              PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
              ierr = PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0);CHKERRQ(ierr);
              if (!scal) {
                PetscBLASInt B_neigs2 = 0;

                bb[0] = PetscMax(lthresh*lthresh,uthresh); bb[1] = PETSC_MAX_REAL;
                ierr = PetscArraycpy(S,Sarray+cumarray,subset_size*subset_size);CHKERRQ(ierr);
                ierr = PetscArraycpy(St,Starray+cumarray,subset_size*subset_size);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
                ierr = PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0);CHKERRQ(ierr);
                B_neigs += B_neigs2;
              }
              break;
            case 2:
              if (scal) {
                bb[0] = PETSC_MIN_REAL;
                bb[1] = 0;
#if defined(PETSC_USE_COMPLEX)
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
                ierr = PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0);CHKERRQ(ierr);
              } else {
                PetscBLASInt B_neigs2 = 0;
                PetscBool    import = PETSC_FALSE;

                lthresh = PetscMax(lthresh,0.0);
                if (lthresh > 0.0) {
                  bb[0] = PETSC_MIN_REAL;
                  bb[1] = lthresh*lthresh;

                  import = PETSC_TRUE;
#if defined(PETSC_USE_COMPLEX)
                  PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
                  PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
                  ierr = PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0);CHKERRQ(ierr);
                }
                bb[0] = PetscMax(lthresh*lthresh,uthresh);
                bb[1] = PETSC_MAX_REAL;
                if (import) {
                  ierr = PetscArraycpy(S,Sarray+cumarray,subset_size*subset_size);CHKERRQ(ierr);
                  ierr = PetscArraycpy(St,Starray+cumarray,subset_size*subset_size);CHKERRQ(ierr);
                }
#if defined(PETSC_USE_COMPLEX)
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
                ierr = PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0);CHKERRQ(ierr);
                B_neigs += B_neigs2;
              }
              break;
            case 3:
              if (scal) {
                ierr = PetscOptionsGetInt(NULL,((PetscObject)pc)->prefix,"-pc_bddc_adaptive_recipe3_min_scal",&recipe_m,NULL);CHKERRQ(ierr);
              } else {
                ierr = PetscOptionsGetInt(NULL,((PetscObject)pc)->prefix,"-pc_bddc_adaptive_recipe3_min",&recipe_m,NULL);CHKERRQ(ierr);
              }
              if (!scal) {
                bb[0] = uthresh;
                bb[1] = PETSC_MAX_REAL;
#if defined(PETSC_USE_COMPLEX)
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
                ierr = PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0);CHKERRQ(ierr);
              }
              if (recipe_m > 0 && B_N - B_neigs > 0) {
                PetscBLASInt B_neigs2 = 0;

                B_IL = 1;
                ierr = PetscBLASIntCast(PetscMin(recipe_m,B_N - B_neigs),&B_IU);CHKERRQ(ierr);
                ierr = PetscArraycpy(S,Sarray+cumarray,subset_size*subset_size);CHKERRQ(ierr);
                ierr = PetscArraycpy(St,Starray+cumarray,subset_size*subset_size);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","I","L",&B_N,St,&B_N,S,&B_N,&lower,&upper,&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","I","L",&B_N,St,&B_N,S,&B_N,&lower,&upper,&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
                ierr = PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0);CHKERRQ(ierr);
                B_neigs += B_neigs2;
              }
              break;
            case 4:
              bb[0] = PETSC_MIN_REAL; bb[1] = lthresh;
#if defined(PETSC_USE_COMPLEX)
              PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
              PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
              ierr = PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0);CHKERRQ(ierr);
              {
                PetscBLASInt B_neigs2 = 0;

                bb[0] = PetscMax(lthresh+PETSC_SMALL,uthresh); bb[1] = PETSC_MAX_REAL;
                ierr = PetscArraycpy(S,Sarray+cumarray,subset_size*subset_size);CHKERRQ(ierr);
                ierr = PetscArraycpy(St,Starray+cumarray,subset_size*subset_size);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
                ierr = PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0);CHKERRQ(ierr);
                B_neigs += B_neigs2;
              }
              break;
            case 5: /* same as before: first compute all eigenvalues, then filter */
#if defined(PETSC_USE_COMPLEX)
              PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","A","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
              PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","A","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
              ierr = PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0);CHKERRQ(ierr);
              {
                PetscInt e,k,ne;
                for (e=0,ne=0;e<B_neigs;e++) {
                  if (eigs[e] < lthresh || eigs[e] > uthresh) {
                    for (k=0;k<B_N;k++) S[ne*B_N+k] = eigv[e*B_N+k];
                    eigs[ne] = eigs[e];
                    ne++;
                  }
                }
                ierr = PetscArraycpy(eigv,S,B_N*ne);CHKERRQ(ierr);
                B_neigs = ne;
              }
              break;
            default:
              SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Unknown recipe %D",recipe);
              break;
            }
          }
        } else if (!same_data) { /* this is just to see all the eigenvalues */
          B_IU = PetscMax(1,PetscMin(B_N,nmax));
          B_IL = 1;
#if defined(PETSC_USE_COMPLEX)
          PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","I","L",&B_N,St,&B_N,S,&B_N,&lower,&upper,&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
          PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","I","L",&B_N,St,&B_N,S,&B_N,&lower,&upper,&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
          ierr = PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0);CHKERRQ(ierr);
        } else { /* same_data is true, so just get the adaptive functional requested by the user */
          PetscInt k;
          if (!sub_schurs->change_primal_sub) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"This should not happen");
          ierr = ISGetLocalSize(sub_schurs->change_primal_sub[i],&nmax);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(nmax,&B_neigs);CHKERRQ(ierr);
          nmin = nmax;
          ierr = PetscArrayzero(eigv,subset_size*nmax);CHKERRQ(ierr);
          for (k=0;k<nmax;k++) {
            eigs[k] = 1./PETSC_SMALL;
            eigv[k*(subset_size+1)] = 1.0;
          }
        }
        ierr = PetscFPTrapPop();CHKERRQ(ierr);
        if (B_ierr) {
          if (B_ierr < 0 ) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYGVX Lapack routine: illegal value for argument %d",-(int)B_ierr);
          else if (B_ierr <= B_N) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYGVX Lapack routine: %d eigenvalues failed to converge",(int)B_ierr);
          else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYGVX Lapack routine: leading minor of order %d is not positive definite",(int)B_ierr-B_N-1);
        }

        if (B_neigs > nmax) {
          if (pcbddc->dbg_flag) {
            ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"   found %d eigs, more than maximum required %D.\n",B_neigs,nmax);CHKERRQ(ierr);
          }
          if (pcbddc->use_deluxe_scaling) eigs_start = scal ? 0 : B_neigs-nmax;
          B_neigs = nmax;
        }

        nmin_s = PetscMin(nmin,B_N);
        if (B_neigs < nmin_s) {
          PetscBLASInt B_neigs2 = 0;

          if (pcbddc->use_deluxe_scaling) {
            if (scal) {
              B_IU = nmin_s;
              B_IL = B_neigs + 1;
            } else {
              B_IL = B_N - nmin_s + 1;
              B_IU = B_N - B_neigs;
            }
          } else {
            B_IL = B_neigs + 1;
            B_IU = nmin_s;
          }
          if (pcbddc->dbg_flag) {
            ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"   found %d eigs, less than minimum required %D. Asking for %d to %d incl (fortran like)\n",B_neigs,nmin,B_IL,B_IU);CHKERRQ(ierr);
          }
          if (sub_schurs->is_symmetric) {
            PetscInt j,k;
            for (j=0;j<subset_size;j++) {
              for (k=j;k<subset_size;k++) {
                S [j*subset_size+k] = Sarray [cumarray+j*subset_size+k];
                St[j*subset_size+k] = Starray[cumarray+j*subset_size+k];
              }
            }
          } else {
            ierr = PetscArraycpy(S,Sarray+cumarray,subset_size*subset_size);CHKERRQ(ierr);
            ierr = PetscArraycpy(St,Starray+cumarray,subset_size*subset_size);CHKERRQ(ierr);
          }
          ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
          PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","I","L",&B_N,St,&B_N,S,&B_N,&lower,&upper,&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*subset_size,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
          PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","I","L",&B_N,St,&B_N,S,&B_N,&lower,&upper,&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*subset_size,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
          ierr = PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0);CHKERRQ(ierr);
          ierr = PetscFPTrapPop();CHKERRQ(ierr);
          B_neigs += B_neigs2;
        }
        if (B_ierr) {
          if (B_ierr < 0 ) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYGVX Lapack routine: illegal value for argument %d",-(int)B_ierr);
          else if (B_ierr <= B_N) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYGVX Lapack routine: %d eigenvalues failed to converge",(int)B_ierr);
          else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYGVX Lapack routine: leading minor of order %d is not positive definite",(int)B_ierr-B_N-1);
        }
        if (pcbddc->dbg_flag) {
          ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"   -> Got %d eigs\n",B_neigs);CHKERRQ(ierr);
          for (j=0;j<B_neigs;j++) {
            if (eigs[j] == 0.0) {
              ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"     Inf\n");CHKERRQ(ierr);
            } else {
              if (pcbddc->use_deluxe_scaling) {
                ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"     %1.6e\n",eigs[j+eigs_start]);CHKERRQ(ierr);
              } else {
                ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"     %1.6e\n",1./eigs[j+eigs_start]);CHKERRQ(ierr);
              }
            }
          }
        }
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not yet implemented");
    }
    /* change the basis back to the original one */
    if (sub_schurs->change) {
      Mat change,phi,phit;

      if (pcbddc->dbg_flag > 2) {
        PetscInt ii;
        for (ii=0;ii<B_neigs;ii++) {
          ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"   -> Eigenvector (old basis) %d/%d (%d)\n",ii,B_neigs,B_N);CHKERRQ(ierr);
          for (j=0;j<B_N;j++) {
#if defined(PETSC_USE_COMPLEX)
            PetscReal r = PetscRealPart(eigv[(ii+eigs_start)*subset_size+j]);
            PetscReal c = PetscImaginaryPart(eigv[(ii+eigs_start)*subset_size+j]);
            ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"       %1.4e + %1.4e i\n",r,c);CHKERRQ(ierr);
#else
            ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"       %1.4e\n",eigv[(ii+eigs_start)*subset_size+j]);CHKERRQ(ierr);
#endif
          }
        }
      }
      ierr = KSPGetOperators(sub_schurs->change[i],&change,NULL);CHKERRQ(ierr);
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,subset_size,B_neigs,eigv+eigs_start*subset_size,&phit);CHKERRQ(ierr);
      ierr = MatMatMult(change,phit,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&phi);CHKERRQ(ierr);
      ierr = MatCopy(phi,phit,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatDestroy(&phit);CHKERRQ(ierr);
      ierr = MatDestroy(&phi);CHKERRQ(ierr);
    }
    maxneigs = PetscMax(B_neigs,maxneigs);
    pcbddc->adaptive_constraints_n[i+nv] = B_neigs;
    if (B_neigs) {
      ierr = PetscArraycpy(pcbddc->adaptive_constraints_data+pcbddc->adaptive_constraints_data_ptr[cum],eigv+eigs_start*subset_size,B_neigs*subset_size);CHKERRQ(ierr);

      if (pcbddc->dbg_flag > 1) {
        PetscInt ii;
        for (ii=0;ii<B_neigs;ii++) {
          ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"   -> Eigenvector %d/%d (%d)\n",ii,B_neigs,B_N);CHKERRQ(ierr);
          for (j=0;j<B_N;j++) {
#if defined(PETSC_USE_COMPLEX)
            PetscReal r = PetscRealPart(pcbddc->adaptive_constraints_data[ii*subset_size+j+pcbddc->adaptive_constraints_data_ptr[cum]]);
            PetscReal c = PetscImaginaryPart(pcbddc->adaptive_constraints_data[ii*subset_size+j+pcbddc->adaptive_constraints_data_ptr[cum]]);
            ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"       %1.4e + %1.4e i\n",r,c);CHKERRQ(ierr);
#else
            ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"       %1.4e\n",pcbddc->adaptive_constraints_data[ii*subset_size+j+pcbddc->adaptive_constraints_data_ptr[cum]]);CHKERRQ(ierr);
#endif
          }
        }
      }
      ierr = PetscArraycpy(pcbddc->adaptive_constraints_idxs+pcbddc->adaptive_constraints_idxs_ptr[cum],idxs,subset_size);CHKERRQ(ierr);
      pcbddc->adaptive_constraints_idxs_ptr[cum+1] = pcbddc->adaptive_constraints_idxs_ptr[cum] + subset_size;
      pcbddc->adaptive_constraints_data_ptr[cum+1] = pcbddc->adaptive_constraints_data_ptr[cum] + subset_size*B_neigs;
      cum++;
    }
    ierr = ISRestoreIndices(sub_schurs->is_subs[i],&idxs);CHKERRQ(ierr);
    /* shift for next computation */
    cumarray += subset_size*subset_size;
  }
  if (pcbddc->dbg_flag) {
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
  }

  if (mss) {
    ierr = MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_inv_all,&Sarray);CHKERRQ(ierr);
    ierr = MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_tilda_all,&Starray);CHKERRQ(ierr);
    /* destroy matrices (junk) */
    ierr = MatDestroy(&sub_schurs->sum_S_Ej_inv_all);CHKERRQ(ierr);
    ierr = MatDestroy(&sub_schurs->sum_S_Ej_tilda_all);CHKERRQ(ierr);
  }
  if (allocated_S_St) {
    ierr = PetscFree2(S,St);CHKERRQ(ierr);
  }
  ierr = PetscFree5(eigv,eigs,work,B_iwork,B_ifail);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#endif
  if (pcbddc->dbg_flag) {
    PetscInt maxneigs_r;
    ierr = MPIU_Allreduce(&maxneigs,&maxneigs_r,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Maximum number of constraints per cc %D\n",maxneigs_r);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(PC_BDDC_AdaptiveSetUp[pcbddc->current_level],pc,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSetUpSolvers(PC pc)
{
  PetscScalar    *coarse_submat_vals;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Setup local scatters R_to_B and (optionally) R_to_D */
  /* PCBDDCSetUpLocalWorkVectors should be called first! */
  ierr = PCBDDCSetUpLocalScatters(pc);CHKERRQ(ierr);

  /* Setup local neumann solver ksp_R */
  /* PCBDDCSetUpLocalScatters should be called first! */
  ierr = PCBDDCSetUpLocalSolvers(pc,PETSC_FALSE,PETSC_TRUE);CHKERRQ(ierr);

  /*
     Setup local correction and local part of coarse basis.
     Gives back the dense local part of the coarse matrix in column major ordering
  */
  ierr = PCBDDCSetUpCorrection(pc,&coarse_submat_vals);CHKERRQ(ierr);

  /* Compute total number of coarse nodes and setup coarse solver */
  ierr = PCBDDCSetUpCoarseSolver(pc,coarse_submat_vals);CHKERRQ(ierr);

  /* free */
  ierr = PetscFree(coarse_submat_vals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCResetCustomization(PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISDestroy(&pcbddc->user_primal_vertices);CHKERRQ(ierr);
  ierr = ISDestroy(&pcbddc->user_primal_vertices_local);CHKERRQ(ierr);
  ierr = ISDestroy(&pcbddc->NeumannBoundaries);CHKERRQ(ierr);
  ierr = ISDestroy(&pcbddc->NeumannBoundariesLocal);CHKERRQ(ierr);
  ierr = ISDestroy(&pcbddc->DirichletBoundaries);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&pcbddc->onearnullspace);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->onearnullvecs_state);CHKERRQ(ierr);
  ierr = ISDestroy(&pcbddc->DirichletBoundariesLocal);CHKERRQ(ierr);
  ierr = PCBDDCSetDofsSplitting(pc,0,NULL);CHKERRQ(ierr);
  ierr = PCBDDCSetDofsSplittingLocal(pc,0,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCResetTopography(PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&pcbddc->nedcG);CHKERRQ(ierr);
  ierr = ISDestroy(&pcbddc->nedclocal);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->discretegradient);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->user_ChangeOfBasisMatrix);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->ChangeOfBasisMatrix);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->switch_static_change);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->work_change);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->ConstraintMatrix);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->divudotp);CHKERRQ(ierr);
  ierr = ISDestroy(&pcbddc->divudotp_vl2l);CHKERRQ(ierr);
  ierr = PCBDDCGraphDestroy(&pcbddc->mat_graph);CHKERRQ(ierr);
  for (i=0;i<pcbddc->n_local_subs;i++) {
    ierr = ISDestroy(&pcbddc->local_subs[i]);CHKERRQ(ierr);
  }
  pcbddc->n_local_subs = 0;
  ierr = PetscFree(pcbddc->local_subs);CHKERRQ(ierr);
  ierr = PCBDDCSubSchursDestroy(&pcbddc->sub_schurs);CHKERRQ(ierr);
  pcbddc->graphanalyzed        = PETSC_FALSE;
  pcbddc->recompute_topography = PETSC_TRUE;
  pcbddc->corner_selected      = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCResetSolvers(PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&pcbddc->coarse_vec);CHKERRQ(ierr);
  if (pcbddc->coarse_phi_B) {
    PetscScalar *array;
    ierr = MatDenseGetArray(pcbddc->coarse_phi_B,&array);CHKERRQ(ierr);
    ierr = PetscFree(array);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&pcbddc->coarse_phi_B);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->coarse_phi_D);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->coarse_psi_B);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->coarse_psi_D);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->vec1_P);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->vec1_C);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->local_auxmat2);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->local_auxmat1);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->vec1_R);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->vec2_R);CHKERRQ(ierr);
  ierr = ISDestroy(&pcbddc->is_R_local);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&pcbddc->R_to_B);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&pcbddc->R_to_D);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&pcbddc->coarse_loc_to_glob);CHKERRQ(ierr);
  ierr = KSPReset(pcbddc->ksp_D);CHKERRQ(ierr);
  ierr = KSPReset(pcbddc->ksp_R);CHKERRQ(ierr);
  ierr = KSPReset(pcbddc->coarse_ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->local_mat);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->primal_indices_local_idxs);CHKERRQ(ierr);
  ierr = PetscFree2(pcbddc->local_primal_ref_node,pcbddc->local_primal_ref_mult);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->global_primal_indices);CHKERRQ(ierr);
  ierr = ISDestroy(&pcbddc->coarse_subassembling);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->benign_change);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->benign_vec);CHKERRQ(ierr);
  ierr = PCBDDCBenignShellMat(pc,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->benign_B0);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&pcbddc->benign_sf);CHKERRQ(ierr);
  if (pcbddc->benign_zerodiag_subs) {
    PetscInt i;
    for (i=0;i<pcbddc->benign_n;i++) {
      ierr = ISDestroy(&pcbddc->benign_zerodiag_subs[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(pcbddc->benign_zerodiag_subs);CHKERRQ(ierr);
  }
  ierr = PetscFree3(pcbddc->benign_p0_lidx,pcbddc->benign_p0_gidx,pcbddc->benign_p0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSetUpLocalWorkVectors(PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PC_IS          *pcis = (PC_IS*)pc->data;
  VecType        impVecType;
  PetscInt       n_constraints,n_R,old_size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  n_constraints = pcbddc->local_primal_size - pcbddc->benign_n - pcbddc->n_vertices;
  n_R = pcis->n - pcbddc->n_vertices;
  ierr = VecGetType(pcis->vec1_N,&impVecType);CHKERRQ(ierr);
  /* local work vectors (try to avoid unneeded work)*/
  /* R nodes */
  old_size = -1;
  if (pcbddc->vec1_R) {
    ierr = VecGetSize(pcbddc->vec1_R,&old_size);CHKERRQ(ierr);
  }
  if (n_R != old_size) {
    ierr = VecDestroy(&pcbddc->vec1_R);CHKERRQ(ierr);
    ierr = VecDestroy(&pcbddc->vec2_R);CHKERRQ(ierr);
    ierr = VecCreate(PetscObjectComm((PetscObject)pcis->vec1_N),&pcbddc->vec1_R);CHKERRQ(ierr);
    ierr = VecSetSizes(pcbddc->vec1_R,PETSC_DECIDE,n_R);CHKERRQ(ierr);
    ierr = VecSetType(pcbddc->vec1_R,impVecType);CHKERRQ(ierr);
    ierr = VecDuplicate(pcbddc->vec1_R,&pcbddc->vec2_R);CHKERRQ(ierr);
  }
  /* local primal dofs */
  old_size = -1;
  if (pcbddc->vec1_P) {
    ierr = VecGetSize(pcbddc->vec1_P,&old_size);CHKERRQ(ierr);
  }
  if (pcbddc->local_primal_size != old_size) {
    ierr = VecDestroy(&pcbddc->vec1_P);CHKERRQ(ierr);
    ierr = VecCreate(PetscObjectComm((PetscObject)pcis->vec1_N),&pcbddc->vec1_P);CHKERRQ(ierr);
    ierr = VecSetSizes(pcbddc->vec1_P,PETSC_DECIDE,pcbddc->local_primal_size);CHKERRQ(ierr);
    ierr = VecSetType(pcbddc->vec1_P,impVecType);CHKERRQ(ierr);
  }
  /* local explicit constraints */
  old_size = -1;
  if (pcbddc->vec1_C) {
    ierr = VecGetSize(pcbddc->vec1_C,&old_size);CHKERRQ(ierr);
  }
  if (n_constraints && n_constraints != old_size) {
    ierr = VecDestroy(&pcbddc->vec1_C);CHKERRQ(ierr);
    ierr = VecCreate(PetscObjectComm((PetscObject)pcis->vec1_N),&pcbddc->vec1_C);CHKERRQ(ierr);
    ierr = VecSetSizes(pcbddc->vec1_C,PETSC_DECIDE,n_constraints);CHKERRQ(ierr);
    ierr = VecSetType(pcbddc->vec1_C,impVecType);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSetUpCorrection(PC pc, PetscScalar **coarse_submat_vals_n)
{
  PetscErrorCode  ierr;
  /* pointers to pcis and pcbddc */
  PC_IS*          pcis = (PC_IS*)pc->data;
  PC_BDDC*        pcbddc = (PC_BDDC*)pc->data;
  PCBDDCSubSchurs sub_schurs = pcbddc->sub_schurs;
  /* submatrices of local problem */
  Mat             A_RV,A_VR,A_VV,local_auxmat2_R;
  /* submatrices of local coarse problem */
  Mat             S_VV,S_CV,S_VC,S_CC;
  /* working matrices */
  Mat             C_CR;
  /* additional working stuff */
  PC              pc_R;
  Mat             F,Brhs = NULL;
  Vec             dummy_vec;
  PetscBool       isLU,isCHOL,need_benign_correction,sparserhs;
  PetscScalar     *coarse_submat_vals; /* TODO: use a PETSc matrix */
  PetscScalar     *work;
  PetscInt        *idx_V_B;
  PetscInt        lda_rhs,n,n_vertices,n_constraints,*p0_lidx_I;
  PetscInt        i,n_R,n_D,n_B;
  PetscScalar     one=1.0,m_one=-1.0;

  PetscFunctionBegin;
  if (!pcbddc->symmetric_primal && pcbddc->benign_n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Non-symmetric primal basis computation with benign trick not yet implemented");
  ierr = PetscLogEventBegin(PC_BDDC_CorrectionSetUp[pcbddc->current_level],pc,0,0,0);CHKERRQ(ierr);

  /* Set Non-overlapping dimensions */
  n_vertices = pcbddc->n_vertices;
  n_constraints = pcbddc->local_primal_size - pcbddc->benign_n - n_vertices;
  n_B = pcis->n_B;
  n_D = pcis->n - n_B;
  n_R = pcis->n - n_vertices;

  /* vertices in boundary numbering */
  ierr = PetscMalloc1(n_vertices,&idx_V_B);CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApply(pcis->BtoNmap,IS_GTOLM_DROP,n_vertices,pcbddc->local_primal_ref_node,&i,idx_V_B);CHKERRQ(ierr);
  if (i != n_vertices) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in boundary numbering for BDDC vertices! %D != %D",n_vertices,i);

  /* Subdomain contribution (Non-overlapping) to coarse matrix  */
  ierr = PetscCalloc1(pcbddc->local_primal_size*pcbddc->local_primal_size,&coarse_submat_vals);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,n_vertices,n_vertices,coarse_submat_vals,&S_VV);CHKERRQ(ierr);
  ierr = MatSeqDenseSetLDA(S_VV,pcbddc->local_primal_size);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,n_constraints,n_vertices,coarse_submat_vals+n_vertices,&S_CV);CHKERRQ(ierr);
  ierr = MatSeqDenseSetLDA(S_CV,pcbddc->local_primal_size);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,n_vertices,n_constraints,coarse_submat_vals+pcbddc->local_primal_size*n_vertices,&S_VC);CHKERRQ(ierr);
  ierr = MatSeqDenseSetLDA(S_VC,pcbddc->local_primal_size);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,n_constraints,n_constraints,coarse_submat_vals+(pcbddc->local_primal_size+1)*n_vertices,&S_CC);CHKERRQ(ierr);
  ierr = MatSeqDenseSetLDA(S_CC,pcbddc->local_primal_size);CHKERRQ(ierr);

  /* determine if can use MatSolve routines instead of calling KSPSolve on ksp_R */
  ierr = KSPGetPC(pcbddc->ksp_R,&pc_R);CHKERRQ(ierr);
  ierr = PCSetUp(pc_R);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)pc_R,PCLU,&isLU);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)pc_R,PCCHOLESKY,&isCHOL);CHKERRQ(ierr);
  lda_rhs = n_R;
  need_benign_correction = PETSC_FALSE;
  if (isLU || isCHOL) {
    ierr = PCFactorGetMatrix(pc_R,&F);CHKERRQ(ierr);
  } else if (sub_schurs && sub_schurs->reuse_solver) {
    PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;
    MatFactorType      type;

    F = reuse_solver->F;
    ierr = MatGetFactorType(F,&type);CHKERRQ(ierr);
    if (type == MAT_FACTOR_CHOLESKY) isCHOL = PETSC_TRUE;
    if (type == MAT_FACTOR_LU) isLU = PETSC_TRUE;
    ierr = MatGetSize(F,&lda_rhs,NULL);CHKERRQ(ierr);
    need_benign_correction = (PetscBool)(!!reuse_solver->benign_n);
  } else F = NULL;

  /* determine if we can use a sparse right-hand side */
  sparserhs = PETSC_FALSE;
  if (F) {
    MatSolverType solver;

    ierr = MatFactorGetSolverType(F,&solver);CHKERRQ(ierr);
    ierr = PetscStrcmp(solver,MATSOLVERMUMPS,&sparserhs);CHKERRQ(ierr);
  }

  /* allocate workspace */
  n = 0;
  if (n_constraints) {
    n += lda_rhs*n_constraints;
  }
  if (n_vertices) {
    n = PetscMax(2*lda_rhs*n_vertices,n);
    n = PetscMax((lda_rhs+n_B)*n_vertices,n);
  }
  if (!pcbddc->symmetric_primal) {
    n = PetscMax(2*lda_rhs*pcbddc->local_primal_size,n);
  }
  ierr = PetscMalloc1(n,&work);CHKERRQ(ierr);

  /* create dummy vector to modify rhs and sol of MatMatSolve (work array will never be used) */
  dummy_vec = NULL;
  if (need_benign_correction && lda_rhs != n_R && F) {
    ierr = VecCreate(PetscObjectComm((PetscObject)pcis->vec1_N),&dummy_vec);CHKERRQ(ierr);
    ierr = VecSetSizes(dummy_vec,lda_rhs,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetType(dummy_vec,((PetscObject)pcis->vec1_N)->type_name);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&pcbddc->local_auxmat1);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->local_auxmat2);CHKERRQ(ierr);

  /* Precompute stuffs needed for preprocessing and application of BDDC*/
  if (n_constraints) {
    Mat         M3,C_B;
    IS          is_aux;
    PetscScalar *array,*array2;

    /* Extract constraints on R nodes: C_{CR}  */
    ierr = ISCreateStride(PETSC_COMM_SELF,n_constraints,n_vertices,1,&is_aux);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(pcbddc->ConstraintMatrix,is_aux,pcbddc->is_R_local,MAT_INITIAL_MATRIX,&C_CR);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(pcbddc->ConstraintMatrix,is_aux,pcis->is_B_local,MAT_INITIAL_MATRIX,&C_B);CHKERRQ(ierr);

    /* Assemble         local_auxmat2_R =        (- A_{RR}^{-1} C^T_{CR}) needed by BDDC setup */
    /* Assemble pcbddc->local_auxmat2   = R_to_B (- A_{RR}^{-1} C^T_{CR}) needed by BDDC application */
    if (!sparserhs) {
      ierr = PetscArrayzero(work,lda_rhs*n_constraints);CHKERRQ(ierr);
      for (i=0;i<n_constraints;i++) {
        const PetscScalar *row_cmat_values;
        const PetscInt    *row_cmat_indices;
        PetscInt          size_of_constraint,j;

        ierr = MatGetRow(C_CR,i,&size_of_constraint,&row_cmat_indices,&row_cmat_values);CHKERRQ(ierr);
        for (j=0;j<size_of_constraint;j++) {
          work[row_cmat_indices[j]+i*lda_rhs] = -row_cmat_values[j];
        }
        ierr = MatRestoreRow(C_CR,i,&size_of_constraint,&row_cmat_indices,&row_cmat_values);CHKERRQ(ierr);
      }
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,lda_rhs,n_constraints,work,&Brhs);CHKERRQ(ierr);
    } else {
      Mat tC_CR;

      ierr = MatScale(C_CR,-1.0);CHKERRQ(ierr);
      if (lda_rhs != n_R) {
        PetscScalar *aa;
        PetscInt    r,*ii,*jj;
        PetscBool   done;

        ierr = MatGetRowIJ(C_CR,0,PETSC_FALSE,PETSC_FALSE,&r,(const PetscInt**)&ii,(const PetscInt**)&jj,&done);CHKERRQ(ierr);
        if (!done) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"GetRowIJ failed");
        ierr = MatSeqAIJGetArray(C_CR,&aa);CHKERRQ(ierr);
        ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,n_constraints,lda_rhs,ii,jj,aa,&tC_CR);CHKERRQ(ierr);
        ierr = MatRestoreRowIJ(C_CR,0,PETSC_FALSE,PETSC_FALSE,&r,(const PetscInt**)&ii,(const PetscInt**)&jj,&done);CHKERRQ(ierr);
        if (!done) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"RestoreRowIJ failed");
      } else {
        ierr  = PetscObjectReference((PetscObject)C_CR);CHKERRQ(ierr);
        tC_CR = C_CR;
      }
      ierr = MatCreateTranspose(tC_CR,&Brhs);CHKERRQ(ierr);
      ierr = MatDestroy(&tC_CR);CHKERRQ(ierr);
    }
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,lda_rhs,n_constraints,NULL,&local_auxmat2_R);CHKERRQ(ierr);
    if (F) {
      if (need_benign_correction) {
        PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

        /* rhs is already zero on interior dofs, no need to change the rhs */
        ierr = PetscArrayzero(reuse_solver->benign_save_vals,pcbddc->benign_n);CHKERRQ(ierr);
      }
      ierr = MatMatSolve(F,Brhs,local_auxmat2_R);CHKERRQ(ierr);
      if (need_benign_correction) {
        PetscScalar        *marr;
        PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

        ierr = MatDenseGetArray(local_auxmat2_R,&marr);CHKERRQ(ierr);
        if (lda_rhs != n_R) {
          for (i=0;i<n_constraints;i++) {
            ierr = VecPlaceArray(dummy_vec,marr+i*lda_rhs);CHKERRQ(ierr);
            ierr = PCBDDCReuseSolversBenignAdapt(reuse_solver,dummy_vec,NULL,PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
            ierr = VecResetArray(dummy_vec);CHKERRQ(ierr);
          }
        } else {
          for (i=0;i<n_constraints;i++) {
            ierr = VecPlaceArray(pcbddc->vec1_R,marr+i*lda_rhs);CHKERRQ(ierr);
            ierr = PCBDDCReuseSolversBenignAdapt(reuse_solver,pcbddc->vec1_R,NULL,PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
            ierr = VecResetArray(pcbddc->vec1_R);CHKERRQ(ierr);
          }
        }
        ierr = MatDenseRestoreArray(local_auxmat2_R,&marr);CHKERRQ(ierr);
      }
    } else {
      PetscScalar *marr;

      ierr = MatDenseGetArray(local_auxmat2_R,&marr);CHKERRQ(ierr);
      for (i=0;i<n_constraints;i++) {
        ierr = VecPlaceArray(pcbddc->vec1_R,work+i*lda_rhs);CHKERRQ(ierr);
        ierr = VecPlaceArray(pcbddc->vec2_R,marr+i*lda_rhs);CHKERRQ(ierr);
        ierr = KSPSolve(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec2_R);CHKERRQ(ierr);
        ierr = KSPCheckSolve(pcbddc->ksp_R,pc,pcbddc->vec2_R);CHKERRQ(ierr);
        ierr = VecResetArray(pcbddc->vec1_R);CHKERRQ(ierr);
        ierr = VecResetArray(pcbddc->vec2_R);CHKERRQ(ierr);
      }
      ierr = MatDenseRestoreArray(local_auxmat2_R,&marr);CHKERRQ(ierr);
    }
    if (sparserhs) {
      ierr = MatScale(C_CR,-1.0);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&Brhs);CHKERRQ(ierr);
    if (!pcbddc->switch_static) {
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,n_B,n_constraints,NULL,&pcbddc->local_auxmat2);CHKERRQ(ierr);
      ierr = MatDenseGetArray(pcbddc->local_auxmat2,&array);CHKERRQ(ierr);
      ierr = MatDenseGetArray(local_auxmat2_R,&array2);CHKERRQ(ierr);
      for (i=0;i<n_constraints;i++) {
        ierr = VecPlaceArray(pcbddc->vec1_R,array2+i*lda_rhs);CHKERRQ(ierr);
        ierr = VecPlaceArray(pcis->vec1_B,array+i*n_B);CHKERRQ(ierr);
        ierr = VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecResetArray(pcis->vec1_B);CHKERRQ(ierr);
        ierr = VecResetArray(pcbddc->vec1_R);CHKERRQ(ierr);
      }
      ierr = MatDenseRestoreArray(local_auxmat2_R,&array2);CHKERRQ(ierr);
      ierr = MatDenseRestoreArray(pcbddc->local_auxmat2,&array);CHKERRQ(ierr);
      ierr = MatMatMult(C_B,pcbddc->local_auxmat2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&M3);CHKERRQ(ierr);
    } else {
      if (lda_rhs != n_R) {
        IS dummy;

        ierr = ISCreateStride(PETSC_COMM_SELF,n_R,0,1,&dummy);CHKERRQ(ierr);
        ierr = MatCreateSubMatrix(local_auxmat2_R,dummy,NULL,MAT_INITIAL_MATRIX,&pcbddc->local_auxmat2);CHKERRQ(ierr);
        ierr = ISDestroy(&dummy);CHKERRQ(ierr);
      } else {
        ierr = PetscObjectReference((PetscObject)local_auxmat2_R);CHKERRQ(ierr);
        pcbddc->local_auxmat2 = local_auxmat2_R;
      }
      ierr = MatMatMult(C_CR,pcbddc->local_auxmat2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&M3);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&is_aux);CHKERRQ(ierr);
    /* Assemble explicitly S_CC = ( C_{CR} A_{RR}^{-1} C^T_{CR} )^{-1}  */
    ierr = MatScale(M3,m_one);CHKERRQ(ierr);
    if (isCHOL) {
      ierr = MatCholeskyFactor(M3,NULL,NULL);CHKERRQ(ierr);
    } else {
      ierr = MatLUFactor(M3,NULL,NULL,NULL);CHKERRQ(ierr);
    }
    ierr = MatSeqDenseInvertFactors_Private(M3);CHKERRQ(ierr);
    /* Assemble local_auxmat1 = S_CC*C_{CB} needed by BDDC application in KSP and in preproc */
    ierr = MatMatMult(M3,C_B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&pcbddc->local_auxmat1);CHKERRQ(ierr);
    ierr = MatDestroy(&C_B);CHKERRQ(ierr);
    ierr = MatCopy(M3,S_CC,SAME_NONZERO_PATTERN);CHKERRQ(ierr); /* S_CC can have a different LDA, MatMatSolve doesn't support it */
    ierr = MatDestroy(&M3);CHKERRQ(ierr);
  }

  /* Get submatrices from subdomain matrix */
  if (n_vertices) {
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    PetscBool oldpin;
#endif
    PetscBool isaij;
    IS        is_aux;

    if (sub_schurs && sub_schurs->reuse_solver) { /* is_R_local is not sorted, ISComplement doesn't like it */
      IS tis;

      ierr = ISDuplicate(pcbddc->is_R_local,&tis);CHKERRQ(ierr);
      ierr = ISSort(tis);CHKERRQ(ierr);
      ierr = ISComplement(tis,0,pcis->n,&is_aux);CHKERRQ(ierr);
      ierr = ISDestroy(&tis);CHKERRQ(ierr);
    } else {
      ierr = ISComplement(pcbddc->is_R_local,0,pcis->n,&is_aux);CHKERRQ(ierr);
    }
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    oldpin = pcbddc->local_mat->pinnedtocpu;
#endif
    ierr = MatPinToCPU(pcbddc->local_mat,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(pcbddc->local_mat,pcbddc->is_R_local,is_aux,MAT_INITIAL_MATRIX,&A_RV);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(pcbddc->local_mat,is_aux,pcbddc->is_R_local,MAT_INITIAL_MATRIX,&A_VR);CHKERRQ(ierr);
    ierr = PetscObjectBaseTypeCompare((PetscObject)A_VR,MATSEQAIJ,&isaij);CHKERRQ(ierr);
    if (!isaij) { /* TODO REMOVE: MatMatMult(A_VR,A_RRmA_RV) below may raise an error */
      ierr = MatConvert(A_VR,MATSEQAIJ,MAT_INPLACE_MATRIX,&A_VR);CHKERRQ(ierr);
    }
    ierr = MatCreateSubMatrix(pcbddc->local_mat,is_aux,is_aux,MAT_INITIAL_MATRIX,&A_VV);CHKERRQ(ierr);
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    ierr = MatPinToCPU(pcbddc->local_mat,oldpin);CHKERRQ(ierr);
#endif
    ierr = ISDestroy(&is_aux);CHKERRQ(ierr);
  }

  /* Matrix of coarse basis functions (local) */
  if (pcbddc->coarse_phi_B) {
    PetscInt on_B,on_primal,on_D=n_D;
    if (pcbddc->coarse_phi_D) {
      ierr = MatGetSize(pcbddc->coarse_phi_D,&on_D,NULL);CHKERRQ(ierr);
    }
    ierr = MatGetSize(pcbddc->coarse_phi_B,&on_B,&on_primal);CHKERRQ(ierr);
    if (on_B != n_B || on_primal != pcbddc->local_primal_size || on_D != n_D) {
      PetscScalar *marray;

      ierr = MatDenseGetArray(pcbddc->coarse_phi_B,&marray);CHKERRQ(ierr);
      ierr = PetscFree(marray);CHKERRQ(ierr);
      ierr = MatDestroy(&pcbddc->coarse_phi_B);CHKERRQ(ierr);
      ierr = MatDestroy(&pcbddc->coarse_psi_B);CHKERRQ(ierr);
      ierr = MatDestroy(&pcbddc->coarse_phi_D);CHKERRQ(ierr);
      ierr = MatDestroy(&pcbddc->coarse_psi_D);CHKERRQ(ierr);
    }
  }

  if (!pcbddc->coarse_phi_B) {
    PetscScalar *marr;

    /* memory size */
    n = n_B*pcbddc->local_primal_size;
    if (pcbddc->switch_static || pcbddc->dbg_flag) n += n_D*pcbddc->local_primal_size;
    if (!pcbddc->symmetric_primal) n *= 2;
    ierr  = PetscCalloc1(n,&marr);CHKERRQ(ierr);
    ierr  = MatCreateSeqDense(PETSC_COMM_SELF,n_B,pcbddc->local_primal_size,marr,&pcbddc->coarse_phi_B);CHKERRQ(ierr);
    marr += n_B*pcbddc->local_primal_size;
    if (pcbddc->switch_static || pcbddc->dbg_flag) {
      ierr  = MatCreateSeqDense(PETSC_COMM_SELF,n_D,pcbddc->local_primal_size,marr,&pcbddc->coarse_phi_D);CHKERRQ(ierr);
      marr += n_D*pcbddc->local_primal_size;
    }
    if (!pcbddc->symmetric_primal) {
      ierr  = MatCreateSeqDense(PETSC_COMM_SELF,n_B,pcbddc->local_primal_size,marr,&pcbddc->coarse_psi_B);CHKERRQ(ierr);
      marr += n_B*pcbddc->local_primal_size;
      if (pcbddc->switch_static || pcbddc->dbg_flag) {
        ierr = MatCreateSeqDense(PETSC_COMM_SELF,n_D,pcbddc->local_primal_size,marr,&pcbddc->coarse_psi_D);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscObjectReference((PetscObject)pcbddc->coarse_phi_B);CHKERRQ(ierr);
      pcbddc->coarse_psi_B = pcbddc->coarse_phi_B;
      if (pcbddc->switch_static || pcbddc->dbg_flag) {
        ierr = PetscObjectReference((PetscObject)pcbddc->coarse_phi_D);CHKERRQ(ierr);
        pcbddc->coarse_psi_D = pcbddc->coarse_phi_D;
      }
    }
  }

  /* We are now ready to evaluate coarse basis functions and subdomain contribution to coarse problem */
  p0_lidx_I = NULL;
  if (pcbddc->benign_n && (pcbddc->switch_static || pcbddc->dbg_flag)) {
    const PetscInt *idxs;

    ierr = ISGetIndices(pcis->is_I_local,&idxs);CHKERRQ(ierr);
    ierr = PetscMalloc1(pcbddc->benign_n,&p0_lidx_I);CHKERRQ(ierr);
    for (i=0;i<pcbddc->benign_n;i++) {
      ierr = PetscFindInt(pcbddc->benign_p0_lidx[i],pcis->n-pcis->n_B,idxs,&p0_lidx_I[i]);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(pcis->is_I_local,&idxs);CHKERRQ(ierr);
  }

  /* vertices */
  if (n_vertices) {
    PetscBool restoreavr = PETSC_FALSE;

    ierr = MatConvert(A_VV,MATDENSE,MAT_INPLACE_MATRIX,&A_VV);CHKERRQ(ierr);

    if (n_R) {
      Mat               A_RRmA_RV,A_RV_bcorr=NULL,S_VVt; /* S_VVt with LDA=N */
      PetscBLASInt      B_N,B_one = 1;
      const PetscScalar *x;
      PetscScalar       *y;

      ierr = MatScale(A_RV,m_one);CHKERRQ(ierr);
      if (need_benign_correction) {
        ISLocalToGlobalMapping RtoN;
        IS                     is_p0;
        PetscInt               *idxs_p0,n;

        ierr = PetscMalloc1(pcbddc->benign_n,&idxs_p0);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingCreateIS(pcbddc->is_R_local,&RtoN);CHKERRQ(ierr);
        ierr = ISGlobalToLocalMappingApply(RtoN,IS_GTOLM_DROP,pcbddc->benign_n,pcbddc->benign_p0_lidx,&n,idxs_p0);CHKERRQ(ierr);
        if (n != pcbddc->benign_n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in R numbering for benign p0! %D != %D",n,pcbddc->benign_n);
        ierr = ISLocalToGlobalMappingDestroy(&RtoN);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PETSC_COMM_SELF,n,idxs_p0,PETSC_OWN_POINTER,&is_p0);CHKERRQ(ierr);
        ierr = MatCreateSubMatrix(A_RV,is_p0,NULL,MAT_INITIAL_MATRIX,&A_RV_bcorr);CHKERRQ(ierr);
        ierr = ISDestroy(&is_p0);CHKERRQ(ierr);
      }

      ierr = MatCreateSeqDense(PETSC_COMM_SELF,lda_rhs,n_vertices,work,&A_RRmA_RV);CHKERRQ(ierr);
      if (!sparserhs || need_benign_correction) {
        if (lda_rhs == n_R) {
          ierr = MatConvert(A_RV,MATDENSE,MAT_INPLACE_MATRIX,&A_RV);CHKERRQ(ierr);
        } else {
          PetscScalar    *av,*array;
          const PetscInt *xadj,*adjncy;
          PetscInt       n;
          PetscBool      flg_row;

          array = work+lda_rhs*n_vertices;
          ierr = PetscArrayzero(array,lda_rhs*n_vertices);CHKERRQ(ierr);
          ierr = MatConvert(A_RV,MATSEQAIJ,MAT_INPLACE_MATRIX,&A_RV);CHKERRQ(ierr);
          ierr = MatGetRowIJ(A_RV,0,PETSC_FALSE,PETSC_FALSE,&n,&xadj,&adjncy,&flg_row);CHKERRQ(ierr);
          ierr = MatSeqAIJGetArray(A_RV,&av);CHKERRQ(ierr);
          for (i=0;i<n;i++) {
            PetscInt j;
            for (j=xadj[i];j<xadj[i+1];j++) array[lda_rhs*adjncy[j]+i] = av[j];
          }
          ierr = MatRestoreRowIJ(A_RV,0,PETSC_FALSE,PETSC_FALSE,&n,&xadj,&adjncy,&flg_row);CHKERRQ(ierr);
          ierr = MatDestroy(&A_RV);CHKERRQ(ierr);
          ierr = MatCreateSeqDense(PETSC_COMM_SELF,lda_rhs,n_vertices,array,&A_RV);CHKERRQ(ierr);
        }
        if (need_benign_correction) {
          PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;
          PetscScalar        *marr;

          ierr = MatDenseGetArray(A_RV,&marr);CHKERRQ(ierr);
          /* need \Phi^T A_RV = (I+L)A_RV, L given by

                 | 0 0  0 | (V)
             L = | 0 0 -1 | (P-p0)
                 | 0 0 -1 | (p0)

          */
          for (i=0;i<reuse_solver->benign_n;i++) {
            const PetscScalar *vals;
            const PetscInt    *idxs,*idxs_zero;
            PetscInt          n,j,nz;

            ierr = ISGetLocalSize(reuse_solver->benign_zerodiag_subs[i],&nz);CHKERRQ(ierr);
            ierr = ISGetIndices(reuse_solver->benign_zerodiag_subs[i],&idxs_zero);CHKERRQ(ierr);
            ierr = MatGetRow(A_RV_bcorr,i,&n,&idxs,&vals);CHKERRQ(ierr);
            for (j=0;j<n;j++) {
              PetscScalar val = vals[j];
              PetscInt    k,col = idxs[j];
              for (k=0;k<nz;k++) marr[idxs_zero[k]+lda_rhs*col] -= val;
            }
            ierr = MatRestoreRow(A_RV_bcorr,i,&n,&idxs,&vals);CHKERRQ(ierr);
            ierr = ISRestoreIndices(reuse_solver->benign_zerodiag_subs[i],&idxs_zero);CHKERRQ(ierr);
          }
          ierr = MatDenseRestoreArray(A_RV,&marr);CHKERRQ(ierr);
        }
        ierr = PetscObjectReference((PetscObject)A_RV);CHKERRQ(ierr);
        Brhs = A_RV;
      } else {
        Mat tA_RVT,A_RVT;

        if (!pcbddc->symmetric_primal) {
          /* A_RV already scaled by -1 */
          ierr = MatTranspose(A_RV,MAT_INITIAL_MATRIX,&A_RVT);CHKERRQ(ierr);
        } else {
          restoreavr = PETSC_TRUE;
          ierr  = MatScale(A_VR,-1.0);CHKERRQ(ierr);
          ierr  = PetscObjectReference((PetscObject)A_VR);CHKERRQ(ierr);
          A_RVT = A_VR;
        }
        if (lda_rhs != n_R) {
          PetscScalar *aa;
          PetscInt    r,*ii,*jj;
          PetscBool   done;

          ierr = MatGetRowIJ(A_RVT,0,PETSC_FALSE,PETSC_FALSE,&r,(const PetscInt**)&ii,(const PetscInt**)&jj,&done);CHKERRQ(ierr);
          if (!done) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"GetRowIJ failed");
          ierr = MatSeqAIJGetArray(A_RVT,&aa);CHKERRQ(ierr);
          ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,n_vertices,lda_rhs,ii,jj,aa,&tA_RVT);CHKERRQ(ierr);
          ierr = MatRestoreRowIJ(A_RVT,0,PETSC_FALSE,PETSC_FALSE,&r,(const PetscInt**)&ii,(const PetscInt**)&jj,&done);CHKERRQ(ierr);
          if (!done) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"RestoreRowIJ failed");
        } else {
          ierr   = PetscObjectReference((PetscObject)A_RVT);CHKERRQ(ierr);
          tA_RVT = A_RVT;
        }
        ierr = MatCreateTranspose(tA_RVT,&Brhs);CHKERRQ(ierr);
        ierr = MatDestroy(&tA_RVT);CHKERRQ(ierr);
        ierr = MatDestroy(&A_RVT);CHKERRQ(ierr);
      }
      if (F) {
        /* need to correct the rhs */
        if (need_benign_correction) {
          PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;
          PetscScalar        *marr;

          ierr = MatDenseGetArray(Brhs,&marr);CHKERRQ(ierr);
          if (lda_rhs != n_R) {
            for (i=0;i<n_vertices;i++) {
              ierr = VecPlaceArray(dummy_vec,marr+i*lda_rhs);CHKERRQ(ierr);
              ierr = PCBDDCReuseSolversBenignAdapt(reuse_solver,dummy_vec,NULL,PETSC_FALSE,PETSC_TRUE);CHKERRQ(ierr);
              ierr = VecResetArray(dummy_vec);CHKERRQ(ierr);
            }
          } else {
            for (i=0;i<n_vertices;i++) {
              ierr = VecPlaceArray(pcbddc->vec1_R,marr+i*lda_rhs);CHKERRQ(ierr);
              ierr = PCBDDCReuseSolversBenignAdapt(reuse_solver,pcbddc->vec1_R,NULL,PETSC_FALSE,PETSC_TRUE);CHKERRQ(ierr);
              ierr = VecResetArray(pcbddc->vec1_R);CHKERRQ(ierr);
            }
          }
          ierr = MatDenseRestoreArray(Brhs,&marr);CHKERRQ(ierr);
        }
        ierr = MatMatSolve(F,Brhs,A_RRmA_RV);CHKERRQ(ierr);
        if (restoreavr) {
          ierr = MatScale(A_VR,-1.0);CHKERRQ(ierr);
        }
        /* need to correct the solution */
        if (need_benign_correction) {
          PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;
          PetscScalar        *marr;

          ierr = MatDenseGetArray(A_RRmA_RV,&marr);CHKERRQ(ierr);
          if (lda_rhs != n_R) {
            for (i=0;i<n_vertices;i++) {
              ierr = VecPlaceArray(dummy_vec,marr+i*lda_rhs);CHKERRQ(ierr);
              ierr = PCBDDCReuseSolversBenignAdapt(reuse_solver,dummy_vec,NULL,PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
              ierr = VecResetArray(dummy_vec);CHKERRQ(ierr);
            }
          } else {
            for (i=0;i<n_vertices;i++) {
              ierr = VecPlaceArray(pcbddc->vec1_R,marr+i*lda_rhs);CHKERRQ(ierr);
              ierr = PCBDDCReuseSolversBenignAdapt(reuse_solver,pcbddc->vec1_R,NULL,PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
              ierr = VecResetArray(pcbddc->vec1_R);CHKERRQ(ierr);
            }
          }
          ierr = MatDenseRestoreArray(A_RRmA_RV,&marr);CHKERRQ(ierr);
        }
      } else {
        ierr = MatDenseGetArray(Brhs,&y);CHKERRQ(ierr);
        for (i=0;i<n_vertices;i++) {
          ierr = VecPlaceArray(pcbddc->vec1_R,y+i*lda_rhs);CHKERRQ(ierr);
          ierr = VecPlaceArray(pcbddc->vec2_R,work+i*lda_rhs);CHKERRQ(ierr);
          ierr = KSPSolve(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec2_R);CHKERRQ(ierr);
          ierr = KSPCheckSolve(pcbddc->ksp_R,pc,pcbddc->vec2_R);CHKERRQ(ierr);
          ierr = VecResetArray(pcbddc->vec1_R);CHKERRQ(ierr);
          ierr = VecResetArray(pcbddc->vec2_R);CHKERRQ(ierr);
        }
        ierr = MatDenseRestoreArray(Brhs,&y);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&A_RV);CHKERRQ(ierr);
      ierr = MatDestroy(&Brhs);CHKERRQ(ierr);
      /* S_VV and S_CV */
      if (n_constraints) {
        Mat B;

        ierr = PetscArrayzero(work+lda_rhs*n_vertices,n_B*n_vertices);CHKERRQ(ierr);
        for (i=0;i<n_vertices;i++) {
          ierr = VecPlaceArray(pcbddc->vec1_R,work+i*lda_rhs);CHKERRQ(ierr);
          ierr = VecPlaceArray(pcis->vec1_B,work+lda_rhs*n_vertices+i*n_B);CHKERRQ(ierr);
          ierr = VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
          ierr = VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
          ierr = VecResetArray(pcis->vec1_B);CHKERRQ(ierr);
          ierr = VecResetArray(pcbddc->vec1_R);CHKERRQ(ierr);
        }
        ierr = MatCreateSeqDense(PETSC_COMM_SELF,n_B,n_vertices,work+lda_rhs*n_vertices,&B);CHKERRQ(ierr);
        ierr = MatMatMult(pcbddc->local_auxmat1,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&S_CV);CHKERRQ(ierr);
        ierr = MatDestroy(&B);CHKERRQ(ierr);
        ierr = MatCreateSeqDense(PETSC_COMM_SELF,lda_rhs,n_vertices,work+lda_rhs*n_vertices,&B);CHKERRQ(ierr);
        ierr = MatMatMult(local_auxmat2_R,S_CV,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B);CHKERRQ(ierr);
        ierr = MatScale(S_CV,m_one);CHKERRQ(ierr);
        ierr = PetscBLASIntCast(lda_rhs*n_vertices,&B_N);CHKERRQ(ierr);
        PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&B_N,&one,work+lda_rhs*n_vertices,&B_one,work,&B_one));
        ierr = MatDestroy(&B);CHKERRQ(ierr);
      }
      if (lda_rhs != n_R) {
        ierr = MatDestroy(&A_RRmA_RV);CHKERRQ(ierr);
        ierr = MatCreateSeqDense(PETSC_COMM_SELF,n_R,n_vertices,work,&A_RRmA_RV);CHKERRQ(ierr);
        ierr = MatSeqDenseSetLDA(A_RRmA_RV,lda_rhs);CHKERRQ(ierr);
      }
      ierr = MatMatMult(A_VR,A_RRmA_RV,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&S_VVt);CHKERRQ(ierr);
      /* need A_VR * \Phi * A_RRmA_RV = A_VR * (I+L)^T * A_RRmA_RV, L given as before */
      if (need_benign_correction) {
        PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;
        PetscScalar      *marr,*sums;

        ierr = PetscMalloc1(n_vertices,&sums);CHKERRQ(ierr);
        ierr = MatDenseGetArray(S_VVt,&marr);CHKERRQ(ierr);
        for (i=0;i<reuse_solver->benign_n;i++) {
          const PetscScalar *vals;
          const PetscInt    *idxs,*idxs_zero;
          PetscInt          n,j,nz;

          ierr = ISGetLocalSize(reuse_solver->benign_zerodiag_subs[i],&nz);CHKERRQ(ierr);
          ierr = ISGetIndices(reuse_solver->benign_zerodiag_subs[i],&idxs_zero);CHKERRQ(ierr);
          for (j=0;j<n_vertices;j++) {
            PetscInt k;
            sums[j] = 0.;
            for (k=0;k<nz;k++) sums[j] += work[idxs_zero[k]+j*lda_rhs];
          }
          ierr = MatGetRow(A_RV_bcorr,i,&n,&idxs,&vals);CHKERRQ(ierr);
          for (j=0;j<n;j++) {
            PetscScalar val = vals[j];
            PetscInt k;
            for (k=0;k<n_vertices;k++) {
              marr[idxs[j]+k*n_vertices] += val*sums[k];
            }
          }
          ierr = MatRestoreRow(A_RV_bcorr,i,&n,&idxs,&vals);CHKERRQ(ierr);
          ierr = ISRestoreIndices(reuse_solver->benign_zerodiag_subs[i],&idxs_zero);CHKERRQ(ierr);
        }
        ierr = PetscFree(sums);CHKERRQ(ierr);
        ierr = MatDenseRestoreArray(S_VVt,&marr);CHKERRQ(ierr);
        ierr = MatDestroy(&A_RV_bcorr);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&A_RRmA_RV);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(n_vertices*n_vertices,&B_N);CHKERRQ(ierr);
      ierr = MatDenseGetArrayRead(A_VV,&x);CHKERRQ(ierr);
      ierr = MatDenseGetArray(S_VVt,&y);CHKERRQ(ierr);
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&B_N,&one,x,&B_one,y,&B_one));
      ierr = MatDenseRestoreArrayRead(A_VV,&x);CHKERRQ(ierr);
      ierr = MatDenseRestoreArray(S_VVt,&y);CHKERRQ(ierr);
      ierr = MatCopy(S_VVt,S_VV,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatDestroy(&S_VVt);CHKERRQ(ierr);
    } else {
      ierr = MatCopy(A_VV,S_VV,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&A_VV);CHKERRQ(ierr);

    /* coarse basis functions */
    for (i=0;i<n_vertices;i++) {
      PetscScalar *y;

      ierr = VecPlaceArray(pcbddc->vec1_R,work+lda_rhs*i);CHKERRQ(ierr);
      ierr = MatDenseGetArray(pcbddc->coarse_phi_B,&y);CHKERRQ(ierr);
      ierr = VecPlaceArray(pcis->vec1_B,y+n_B*i);CHKERRQ(ierr);
      ierr = VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      y[n_B*i+idx_V_B[i]] = 1.0;
      ierr = MatDenseRestoreArray(pcbddc->coarse_phi_B,&y);CHKERRQ(ierr);
      ierr = VecResetArray(pcis->vec1_B);CHKERRQ(ierr);

      if (pcbddc->switch_static || pcbddc->dbg_flag) {
        PetscInt j;

        ierr = MatDenseGetArray(pcbddc->coarse_phi_D,&y);CHKERRQ(ierr);
        ierr = VecPlaceArray(pcis->vec1_D,y+n_D*i);CHKERRQ(ierr);
        ierr = VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecResetArray(pcis->vec1_D);CHKERRQ(ierr);
        for (j=0;j<pcbddc->benign_n;j++) y[n_D*i+p0_lidx_I[j]] = 0.0;
        ierr = MatDenseRestoreArray(pcbddc->coarse_phi_D,&y);CHKERRQ(ierr);
      }
      ierr = VecResetArray(pcbddc->vec1_R);CHKERRQ(ierr);
    }
    /* if n_R == 0 the object is not destroyed */
    ierr = MatDestroy(&A_RV);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&dummy_vec);CHKERRQ(ierr);

  if (n_constraints) {
    Mat B;

    ierr = MatCreateSeqDense(PETSC_COMM_SELF,lda_rhs,n_constraints,work,&B);CHKERRQ(ierr);
    ierr = MatScale(S_CC,m_one);CHKERRQ(ierr);
    ierr = MatMatMult(local_auxmat2_R,S_CC,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B);CHKERRQ(ierr);
    ierr = MatScale(S_CC,m_one);CHKERRQ(ierr);
    if (n_vertices) {
      if (isCHOL || need_benign_correction) { /* if we can solve the interior problem with cholesky, we should also be fine with transposing here */
        ierr = MatTranspose(S_CV,MAT_REUSE_MATRIX,&S_VC);CHKERRQ(ierr);
      } else {
        Mat S_VCt;

        if (lda_rhs != n_R) {
          ierr = MatDestroy(&B);CHKERRQ(ierr);
          ierr = MatCreateSeqDense(PETSC_COMM_SELF,n_R,n_constraints,work,&B);CHKERRQ(ierr);
          ierr = MatSeqDenseSetLDA(B,lda_rhs);CHKERRQ(ierr);
        }
        ierr = MatMatMult(A_VR,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&S_VCt);CHKERRQ(ierr);
        ierr = MatCopy(S_VCt,S_VC,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
        ierr = MatDestroy(&S_VCt);CHKERRQ(ierr);
      }
    }
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    /* coarse basis functions */
    for (i=0;i<n_constraints;i++) {
      PetscScalar *y;

      ierr = VecPlaceArray(pcbddc->vec1_R,work+lda_rhs*i);CHKERRQ(ierr);
      ierr = MatDenseGetArray(pcbddc->coarse_phi_B,&y);CHKERRQ(ierr);
      ierr = VecPlaceArray(pcis->vec1_B,y+n_B*(i+n_vertices));CHKERRQ(ierr);
      ierr = VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = MatDenseRestoreArray(pcbddc->coarse_phi_B,&y);CHKERRQ(ierr);
      ierr = VecResetArray(pcis->vec1_B);CHKERRQ(ierr);
      if (pcbddc->switch_static || pcbddc->dbg_flag) {
        PetscInt j;

        ierr = MatDenseGetArray(pcbddc->coarse_phi_D,&y);CHKERRQ(ierr);
        ierr = VecPlaceArray(pcis->vec1_D,y+n_D*(i+n_vertices));CHKERRQ(ierr);
        ierr = VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecResetArray(pcis->vec1_D);CHKERRQ(ierr);
        for (j=0;j<pcbddc->benign_n;j++) y[n_D*i+p0_lidx_I[j]] = 0.0;
        ierr = MatDenseRestoreArray(pcbddc->coarse_phi_D,&y);CHKERRQ(ierr);
      }
      ierr = VecResetArray(pcbddc->vec1_R);CHKERRQ(ierr);
    }
  }
  if (n_constraints) {
    ierr = MatDestroy(&local_auxmat2_R);CHKERRQ(ierr);
  }
  ierr = PetscFree(p0_lidx_I);CHKERRQ(ierr);

  /* coarse matrix entries relative to B_0 */
  if (pcbddc->benign_n) {
    Mat               B0_B,B0_BPHI;
    IS                is_dummy;
    const PetscScalar *data;
    PetscInt          j;

    ierr = ISCreateStride(PETSC_COMM_SELF,pcbddc->benign_n,0,1,&is_dummy);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(pcbddc->benign_B0,is_dummy,pcis->is_B_local,MAT_INITIAL_MATRIX,&B0_B);CHKERRQ(ierr);
    ierr = ISDestroy(&is_dummy);CHKERRQ(ierr);
    ierr = MatMatMult(B0_B,pcbddc->coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&B0_BPHI);CHKERRQ(ierr);
    ierr = MatConvert(B0_BPHI,MATSEQDENSE,MAT_INPLACE_MATRIX,&B0_BPHI);CHKERRQ(ierr);
    ierr = MatDenseGetArrayRead(B0_BPHI,&data);CHKERRQ(ierr);
    for (j=0;j<pcbddc->benign_n;j++) {
      PetscInt primal_idx = pcbddc->local_primal_size - pcbddc->benign_n + j;
      for (i=0;i<pcbddc->local_primal_size;i++) {
        coarse_submat_vals[primal_idx*pcbddc->local_primal_size+i] = data[i*pcbddc->benign_n+j];
        coarse_submat_vals[i*pcbddc->local_primal_size+primal_idx] = data[i*pcbddc->benign_n+j];
      }
    }
    ierr = MatDenseRestoreArrayRead(B0_BPHI,&data);CHKERRQ(ierr);
    ierr = MatDestroy(&B0_B);CHKERRQ(ierr);
    ierr = MatDestroy(&B0_BPHI);CHKERRQ(ierr);
  }

  /* compute other basis functions for non-symmetric problems */
  if (!pcbddc->symmetric_primal) {
    Mat         B_V=NULL,B_C=NULL;
    PetscScalar *marray;

    if (n_constraints) {
      Mat S_CCT,C_CRT;

      ierr = MatTranspose(C_CR,MAT_INITIAL_MATRIX,&C_CRT);CHKERRQ(ierr);
      ierr = MatTranspose(S_CC,MAT_INITIAL_MATRIX,&S_CCT);CHKERRQ(ierr);
      ierr = MatMatMult(C_CRT,S_CCT,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B_C);CHKERRQ(ierr);
      ierr = MatDestroy(&S_CCT);CHKERRQ(ierr);
      if (n_vertices) {
        Mat S_VCT;

        ierr = MatTranspose(S_VC,MAT_INITIAL_MATRIX,&S_VCT);CHKERRQ(ierr);
        ierr = MatMatMult(C_CRT,S_VCT,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B_V);CHKERRQ(ierr);
        ierr = MatDestroy(&S_VCT);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&C_CRT);CHKERRQ(ierr);
    } else {
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,n_R,n_vertices,NULL,&B_V);CHKERRQ(ierr);
    }
    if (n_vertices && n_R) {
      PetscScalar    *av,*marray;
      const PetscInt *xadj,*adjncy;
      PetscInt       n;
      PetscBool      flg_row;

      /* B_V = B_V - A_VR^T */
      ierr = MatConvert(A_VR,MATSEQAIJ,MAT_INPLACE_MATRIX,&A_VR);CHKERRQ(ierr);
      ierr = MatGetRowIJ(A_VR,0,PETSC_FALSE,PETSC_FALSE,&n,&xadj,&adjncy,&flg_row);CHKERRQ(ierr);
      ierr = MatSeqAIJGetArray(A_VR,&av);CHKERRQ(ierr);
      ierr = MatDenseGetArray(B_V,&marray);CHKERRQ(ierr);
      for (i=0;i<n;i++) {
        PetscInt j;
        for (j=xadj[i];j<xadj[i+1];j++) marray[i*n_R + adjncy[j]] -= av[j];
      }
      ierr = MatDenseRestoreArray(B_V,&marray);CHKERRQ(ierr);
      ierr = MatRestoreRowIJ(A_VR,0,PETSC_FALSE,PETSC_FALSE,&n,&xadj,&adjncy,&flg_row);CHKERRQ(ierr);
      ierr = MatDestroy(&A_VR);CHKERRQ(ierr);
    }

    /* currently there's no support for MatTransposeMatSolve(F,B,X) */
    if (n_vertices) {
      ierr = MatDenseGetArray(B_V,&marray);CHKERRQ(ierr);
      for (i=0;i<n_vertices;i++) {
        ierr = VecPlaceArray(pcbddc->vec1_R,marray+i*n_R);CHKERRQ(ierr);
        ierr = VecPlaceArray(pcbddc->vec2_R,work+i*n_R);CHKERRQ(ierr);
        ierr = KSPSolveTranspose(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec2_R);CHKERRQ(ierr);
        ierr = KSPCheckSolve(pcbddc->ksp_R,pc,pcbddc->vec2_R);CHKERRQ(ierr);
        ierr = VecResetArray(pcbddc->vec1_R);CHKERRQ(ierr);
        ierr = VecResetArray(pcbddc->vec2_R);CHKERRQ(ierr);
      }
      ierr = MatDenseRestoreArray(B_V,&marray);CHKERRQ(ierr);
    }
    if (B_C) {
      ierr = MatDenseGetArray(B_C,&marray);CHKERRQ(ierr);
      for (i=n_vertices;i<n_constraints+n_vertices;i++) {
        ierr = VecPlaceArray(pcbddc->vec1_R,marray+(i-n_vertices)*n_R);CHKERRQ(ierr);
        ierr = VecPlaceArray(pcbddc->vec2_R,work+i*n_R);CHKERRQ(ierr);
        ierr = KSPSolveTranspose(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec2_R);CHKERRQ(ierr);
        ierr = KSPCheckSolve(pcbddc->ksp_R,pc,pcbddc->vec2_R);CHKERRQ(ierr);
        ierr = VecResetArray(pcbddc->vec1_R);CHKERRQ(ierr);
        ierr = VecResetArray(pcbddc->vec2_R);CHKERRQ(ierr);
      }
      ierr = MatDenseRestoreArray(B_C,&marray);CHKERRQ(ierr);
    }
    /* coarse basis functions */
    for (i=0;i<pcbddc->local_primal_size;i++) {
      PetscScalar *y;

      ierr = VecPlaceArray(pcbddc->vec1_R,work+i*n_R);CHKERRQ(ierr);
      ierr = MatDenseGetArray(pcbddc->coarse_psi_B,&y);CHKERRQ(ierr);
      ierr = VecPlaceArray(pcis->vec1_B,y+n_B*i);CHKERRQ(ierr);
      ierr = VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      if (i<n_vertices) {
        y[n_B*i+idx_V_B[i]] = 1.0;
      }
      ierr = MatDenseRestoreArray(pcbddc->coarse_psi_B,&y);CHKERRQ(ierr);
      ierr = VecResetArray(pcis->vec1_B);CHKERRQ(ierr);

      if (pcbddc->switch_static || pcbddc->dbg_flag) {
        ierr = MatDenseGetArray(pcbddc->coarse_psi_D,&y);CHKERRQ(ierr);
        ierr = VecPlaceArray(pcis->vec1_D,y+n_D*i);CHKERRQ(ierr);
        ierr = VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecResetArray(pcis->vec1_D);CHKERRQ(ierr);
        ierr = MatDenseRestoreArray(pcbddc->coarse_psi_D,&y);CHKERRQ(ierr);
      }
      ierr = VecResetArray(pcbddc->vec1_R);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&B_V);CHKERRQ(ierr);
    ierr = MatDestroy(&B_C);CHKERRQ(ierr);
  }

  /* free memory */
  ierr = PetscFree(idx_V_B);CHKERRQ(ierr);
  ierr = MatDestroy(&S_VV);CHKERRQ(ierr);
  ierr = MatDestroy(&S_CV);CHKERRQ(ierr);
  ierr = MatDestroy(&S_VC);CHKERRQ(ierr);
  ierr = MatDestroy(&S_CC);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  if (n_vertices) {
    ierr = MatDestroy(&A_VR);CHKERRQ(ierr);
  }
  if (n_constraints) {
    ierr = MatDestroy(&C_CR);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(PC_BDDC_CorrectionSetUp[pcbddc->current_level],pc,0,0,0);CHKERRQ(ierr);

  /* Checking coarse_sub_mat and coarse basis functios */
  /* Symmetric case     : It should be \Phi^{(j)^T} A^{(j)} \Phi^{(j)}=coarse_sub_mat */
  /* Non-symmetric case : It should be \Psi^{(j)^T} A^{(j)} \Phi^{(j)}=coarse_sub_mat */
  if (pcbddc->dbg_flag) {
    Mat         coarse_sub_mat;
    Mat         AUXMAT,TM1,TM2,TM3,TM4;
    Mat         coarse_phi_D,coarse_phi_B;
    Mat         coarse_psi_D,coarse_psi_B;
    Mat         A_II,A_BB,A_IB,A_BI;
    Mat         C_B,CPHI;
    IS          is_dummy;
    Vec         mones;
    MatType     checkmattype=MATSEQAIJ;
    PetscReal   real_value;

    if (pcbddc->benign_n && !pcbddc->benign_change_explicit) {
      Mat A;
      ierr = PCBDDCBenignProject(pc,NULL,NULL,&A);CHKERRQ(ierr);
      ierr = MatCreateSubMatrix(A,pcis->is_I_local,pcis->is_I_local,MAT_INITIAL_MATRIX,&A_II);CHKERRQ(ierr);
      ierr = MatCreateSubMatrix(A,pcis->is_I_local,pcis->is_B_local,MAT_INITIAL_MATRIX,&A_IB);CHKERRQ(ierr);
      ierr = MatCreateSubMatrix(A,pcis->is_B_local,pcis->is_I_local,MAT_INITIAL_MATRIX,&A_BI);CHKERRQ(ierr);
      ierr = MatCreateSubMatrix(A,pcis->is_B_local,pcis->is_B_local,MAT_INITIAL_MATRIX,&A_BB);CHKERRQ(ierr);
      ierr = MatDestroy(&A);CHKERRQ(ierr);
    } else {
      ierr = MatConvert(pcis->A_II,checkmattype,MAT_INITIAL_MATRIX,&A_II);CHKERRQ(ierr);
      ierr = MatConvert(pcis->A_IB,checkmattype,MAT_INITIAL_MATRIX,&A_IB);CHKERRQ(ierr);
      ierr = MatConvert(pcis->A_BI,checkmattype,MAT_INITIAL_MATRIX,&A_BI);CHKERRQ(ierr);
      ierr = MatConvert(pcis->A_BB,checkmattype,MAT_INITIAL_MATRIX,&A_BB);CHKERRQ(ierr);
    }
    ierr = MatConvert(pcbddc->coarse_phi_D,checkmattype,MAT_INITIAL_MATRIX,&coarse_phi_D);CHKERRQ(ierr);
    ierr = MatConvert(pcbddc->coarse_phi_B,checkmattype,MAT_INITIAL_MATRIX,&coarse_phi_B);CHKERRQ(ierr);
    if (!pcbddc->symmetric_primal) {
      ierr = MatConvert(pcbddc->coarse_psi_D,checkmattype,MAT_INITIAL_MATRIX,&coarse_psi_D);CHKERRQ(ierr);
      ierr = MatConvert(pcbddc->coarse_psi_B,checkmattype,MAT_INITIAL_MATRIX,&coarse_psi_B);CHKERRQ(ierr);
    }
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,pcbddc->local_primal_size,pcbddc->local_primal_size,coarse_submat_vals,&coarse_sub_mat);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Check coarse sub mat computation (symmetric %d)\n",pcbddc->symmetric_primal);CHKERRQ(ierr);
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    if (!pcbddc->symmetric_primal) {
      ierr = MatMatMult(A_II,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&AUXMAT);CHKERRQ(ierr);
      ierr = MatTransposeMatMult(coarse_psi_D,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM1);CHKERRQ(ierr);
      ierr = MatDestroy(&AUXMAT);CHKERRQ(ierr);
      ierr = MatMatMult(A_BB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&AUXMAT);CHKERRQ(ierr);
      ierr = MatTransposeMatMult(coarse_psi_B,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM2);CHKERRQ(ierr);
      ierr = MatDestroy(&AUXMAT);CHKERRQ(ierr);
      ierr = MatMatMult(A_IB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&AUXMAT);CHKERRQ(ierr);
      ierr = MatTransposeMatMult(coarse_psi_D,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM3);CHKERRQ(ierr);
      ierr = MatDestroy(&AUXMAT);CHKERRQ(ierr);
      ierr = MatMatMult(A_BI,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&AUXMAT);CHKERRQ(ierr);
      ierr = MatTransposeMatMult(coarse_psi_B,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM4);CHKERRQ(ierr);
      ierr = MatDestroy(&AUXMAT);CHKERRQ(ierr);
    } else {
      ierr = MatPtAP(A_II,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&TM1);CHKERRQ(ierr);
      ierr = MatPtAP(A_BB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&TM2);CHKERRQ(ierr);
      ierr = MatMatMult(A_IB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&AUXMAT);CHKERRQ(ierr);
      ierr = MatTransposeMatMult(coarse_phi_D,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM3);CHKERRQ(ierr);
      ierr = MatDestroy(&AUXMAT);CHKERRQ(ierr);
      ierr = MatMatMult(A_BI,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&AUXMAT);CHKERRQ(ierr);
      ierr = MatTransposeMatMult(coarse_phi_B,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM4);CHKERRQ(ierr);
      ierr = MatDestroy(&AUXMAT);CHKERRQ(ierr);
    }
    ierr = MatAXPY(TM1,one,TM2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(TM1,one,TM3,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(TM1,one,TM4,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatConvert(TM1,MATSEQDENSE,MAT_INPLACE_MATRIX,&TM1);CHKERRQ(ierr);
    if (pcbddc->benign_n) {
      Mat               B0_B,B0_BPHI;
      const PetscScalar *data2;
      PetscScalar       *data;
      PetscInt          j;

      ierr = ISCreateStride(PETSC_COMM_SELF,pcbddc->benign_n,0,1,&is_dummy);CHKERRQ(ierr);
      ierr = MatCreateSubMatrix(pcbddc->benign_B0,is_dummy,pcis->is_B_local,MAT_INITIAL_MATRIX,&B0_B);CHKERRQ(ierr);
      ierr = MatMatMult(B0_B,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&B0_BPHI);CHKERRQ(ierr);
      ierr = MatConvert(B0_BPHI,MATSEQDENSE,MAT_INPLACE_MATRIX,&B0_BPHI);CHKERRQ(ierr);
      ierr = MatDenseGetArray(TM1,&data);CHKERRQ(ierr);
      ierr = MatDenseGetArrayRead(B0_BPHI,&data2);CHKERRQ(ierr);
      for (j=0;j<pcbddc->benign_n;j++) {
        PetscInt primal_idx = pcbddc->local_primal_size - pcbddc->benign_n + j;
        for (i=0;i<pcbddc->local_primal_size;i++) {
          data[primal_idx*pcbddc->local_primal_size+i] += data2[i*pcbddc->benign_n+j];
          data[i*pcbddc->local_primal_size+primal_idx] += data2[i*pcbddc->benign_n+j];
        }
      }
      ierr = MatDenseRestoreArray(TM1,&data);CHKERRQ(ierr);
      ierr = MatDenseRestoreArrayRead(B0_BPHI,&data2);CHKERRQ(ierr);
      ierr = MatDestroy(&B0_B);CHKERRQ(ierr);
      ierr = ISDestroy(&is_dummy);CHKERRQ(ierr);
      ierr = MatDestroy(&B0_BPHI);CHKERRQ(ierr);
    }
#if 0
  {
    PetscViewer viewer;
    char filename[256];
    sprintf(filename,"details_local_coarse_mat%d_level%d.m",PetscGlobalRank,pcbddc->current_level);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF,filename,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)coarse_sub_mat,"computed");CHKERRQ(ierr);
    ierr = MatView(coarse_sub_mat,viewer);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)TM1,"projected");CHKERRQ(ierr);
    ierr = MatView(TM1,viewer);CHKERRQ(ierr);
    if (pcbddc->coarse_phi_B) {
      ierr = PetscObjectSetName((PetscObject)pcbddc->coarse_phi_B,"phi_B");CHKERRQ(ierr);
      ierr = MatView(pcbddc->coarse_phi_B,viewer);CHKERRQ(ierr);
    }
    if (pcbddc->coarse_phi_D) {
      ierr = PetscObjectSetName((PetscObject)pcbddc->coarse_phi_D,"phi_D");CHKERRQ(ierr);
      ierr = MatView(pcbddc->coarse_phi_D,viewer);CHKERRQ(ierr);
    }
    if (pcbddc->coarse_psi_B) {
      ierr = PetscObjectSetName((PetscObject)pcbddc->coarse_psi_B,"psi_B");CHKERRQ(ierr);
      ierr = MatView(pcbddc->coarse_psi_B,viewer);CHKERRQ(ierr);
    }
    if (pcbddc->coarse_psi_D) {
      ierr = PetscObjectSetName((PetscObject)pcbddc->coarse_psi_D,"psi_D");CHKERRQ(ierr);
      ierr = MatView(pcbddc->coarse_psi_D,viewer);CHKERRQ(ierr);
    }
    ierr = PetscObjectSetName((PetscObject)pcbddc->local_mat,"A");CHKERRQ(ierr);
    ierr = MatView(pcbddc->local_mat,viewer);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)pcbddc->ConstraintMatrix,"C");CHKERRQ(ierr);
    ierr = MatView(pcbddc->ConstraintMatrix,viewer);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)pcis->is_I_local,"I");CHKERRQ(ierr);
    ierr = ISView(pcis->is_I_local,viewer);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)pcis->is_B_local,"B");CHKERRQ(ierr);
    ierr = ISView(pcis->is_B_local,viewer);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)pcbddc->is_R_local,"R");CHKERRQ(ierr);
    ierr = ISView(pcbddc->is_R_local,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
#endif
    ierr = MatAXPY(TM1,m_one,coarse_sub_mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(TM1,NORM_FROBENIUS,&real_value);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d          matrix error % 1.14e\n",PetscGlobalRank,real_value);CHKERRQ(ierr);

    /* check constraints */
    ierr = ISCreateStride(PETSC_COMM_SELF,pcbddc->local_primal_size-pcbddc->benign_n,0,1,&is_dummy);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(pcbddc->ConstraintMatrix,is_dummy,pcis->is_B_local,MAT_INITIAL_MATRIX,&C_B);CHKERRQ(ierr);
    if (!pcbddc->benign_n) { /* TODO: add benign case */
      ierr = MatMatMult(C_B,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&CPHI);CHKERRQ(ierr);
    } else {
      PetscScalar *data;
      Mat         tmat;
      ierr = MatDenseGetArray(pcbddc->coarse_phi_B,&data);CHKERRQ(ierr);
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,pcis->n_B,pcbddc->local_primal_size-pcbddc->benign_n,data,&tmat);CHKERRQ(ierr);
      ierr = MatDenseRestoreArray(pcbddc->coarse_phi_B,&data);CHKERRQ(ierr);
      ierr = MatMatMult(C_B,tmat,MAT_INITIAL_MATRIX,1.0,&CPHI);CHKERRQ(ierr);
      ierr = MatDestroy(&tmat);CHKERRQ(ierr);
    }
    ierr = MatCreateVecs(CPHI,&mones,NULL);CHKERRQ(ierr);
    ierr = VecSet(mones,-1.0);CHKERRQ(ierr);
    ierr = MatDiagonalSet(CPHI,mones,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatNorm(CPHI,NORM_FROBENIUS,&real_value);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d phi constraints error % 1.14e\n",PetscGlobalRank,real_value);CHKERRQ(ierr);
    if (!pcbddc->symmetric_primal) {
      ierr = MatMatMult(C_B,coarse_psi_B,MAT_REUSE_MATRIX,1.0,&CPHI);CHKERRQ(ierr);
      ierr = VecSet(mones,-1.0);CHKERRQ(ierr);
      ierr = MatDiagonalSet(CPHI,mones,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatNorm(CPHI,NORM_FROBENIUS,&real_value);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d psi constraints error % 1.14e\n",PetscGlobalRank,real_value);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&C_B);CHKERRQ(ierr);
    ierr = MatDestroy(&CPHI);CHKERRQ(ierr);
    ierr = ISDestroy(&is_dummy);CHKERRQ(ierr);
    ierr = VecDestroy(&mones);CHKERRQ(ierr);
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    ierr = MatDestroy(&A_II);CHKERRQ(ierr);
    ierr = MatDestroy(&A_BB);CHKERRQ(ierr);
    ierr = MatDestroy(&A_IB);CHKERRQ(ierr);
    ierr = MatDestroy(&A_BI);CHKERRQ(ierr);
    ierr = MatDestroy(&TM1);CHKERRQ(ierr);
    ierr = MatDestroy(&TM2);CHKERRQ(ierr);
    ierr = MatDestroy(&TM3);CHKERRQ(ierr);
    ierr = MatDestroy(&TM4);CHKERRQ(ierr);
    ierr = MatDestroy(&coarse_phi_D);CHKERRQ(ierr);
    ierr = MatDestroy(&coarse_phi_B);CHKERRQ(ierr);
    if (!pcbddc->symmetric_primal) {
      ierr = MatDestroy(&coarse_psi_D);CHKERRQ(ierr);
      ierr = MatDestroy(&coarse_psi_B);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&coarse_sub_mat);CHKERRQ(ierr);
  }
  /* FINAL CUDA support (we cannot currently mix viennacl and cuda vectors */
  {
    PetscBool gpu;

    ierr = PetscObjectTypeCompare((PetscObject)pcis->vec1_N,VECSEQCUDA,&gpu);CHKERRQ(ierr);
    if (gpu) {
      if (pcbddc->local_auxmat1) {
        ierr = MatConvert(pcbddc->local_auxmat1,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&pcbddc->local_auxmat1);CHKERRQ(ierr);
      }
      if (pcbddc->local_auxmat2) {
        ierr = MatConvert(pcbddc->local_auxmat2,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&pcbddc->local_auxmat2);CHKERRQ(ierr);
      }
      if (pcbddc->coarse_phi_B) {
        ierr = MatConvert(pcbddc->coarse_phi_B,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&pcbddc->coarse_phi_B);CHKERRQ(ierr);
      }
      if (pcbddc->coarse_phi_D) {
        ierr = MatConvert(pcbddc->coarse_phi_D,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&pcbddc->coarse_phi_D);CHKERRQ(ierr);
      }
      if (pcbddc->coarse_psi_B) {
        ierr = MatConvert(pcbddc->coarse_psi_B,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&pcbddc->coarse_psi_B);CHKERRQ(ierr);
      }
      if (pcbddc->coarse_psi_D) {
        ierr = MatConvert(pcbddc->coarse_psi_D,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&pcbddc->coarse_psi_D);CHKERRQ(ierr);
      }
    }
  }
  /* get back data */
  *coarse_submat_vals_n = coarse_submat_vals;
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateSubMatrixUnsorted(Mat A, IS isrow, IS iscol, Mat* B)
{
  Mat            *work_mat;
  IS             isrow_s,iscol_s;
  PetscBool      rsorted,csorted;
  PetscInt       rsize,*idxs_perm_r=NULL,csize,*idxs_perm_c=NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISSorted(isrow,&rsorted);CHKERRQ(ierr);
  ierr = ISSorted(iscol,&csorted);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrow,&rsize);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscol,&csize);CHKERRQ(ierr);

  if (!rsorted) {
    const PetscInt *idxs;
    PetscInt *idxs_sorted,i;

    ierr = PetscMalloc1(rsize,&idxs_perm_r);CHKERRQ(ierr);
    ierr = PetscMalloc1(rsize,&idxs_sorted);CHKERRQ(ierr);
    for (i=0;i<rsize;i++) {
      idxs_perm_r[i] = i;
    }
    ierr = ISGetIndices(isrow,&idxs);CHKERRQ(ierr);
    ierr = PetscSortIntWithPermutation(rsize,idxs,idxs_perm_r);CHKERRQ(ierr);
    for (i=0;i<rsize;i++) {
      idxs_sorted[i] = idxs[idxs_perm_r[i]];
    }
    ierr = ISRestoreIndices(isrow,&idxs);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,rsize,idxs_sorted,PETSC_OWN_POINTER,&isrow_s);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectReference((PetscObject)isrow);CHKERRQ(ierr);
    isrow_s = isrow;
  }

  if (!csorted) {
    if (isrow == iscol) {
      ierr = PetscObjectReference((PetscObject)isrow_s);CHKERRQ(ierr);
      iscol_s = isrow_s;
    } else {
      const PetscInt *idxs;
      PetscInt       *idxs_sorted,i;

      ierr = PetscMalloc1(csize,&idxs_perm_c);CHKERRQ(ierr);
      ierr = PetscMalloc1(csize,&idxs_sorted);CHKERRQ(ierr);
      for (i=0;i<csize;i++) {
        idxs_perm_c[i] = i;
      }
      ierr = ISGetIndices(iscol,&idxs);CHKERRQ(ierr);
      ierr = PetscSortIntWithPermutation(csize,idxs,idxs_perm_c);CHKERRQ(ierr);
      for (i=0;i<csize;i++) {
        idxs_sorted[i] = idxs[idxs_perm_c[i]];
      }
      ierr = ISRestoreIndices(iscol,&idxs);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF,csize,idxs_sorted,PETSC_OWN_POINTER,&iscol_s);CHKERRQ(ierr);
    }
  } else {
    ierr = PetscObjectReference((PetscObject)iscol);CHKERRQ(ierr);
    iscol_s = iscol;
  }

  ierr = MatCreateSubMatrices(A,1,&isrow_s,&iscol_s,MAT_INITIAL_MATRIX,&work_mat);CHKERRQ(ierr);

  if (!rsorted || !csorted) {
    Mat      new_mat;
    IS       is_perm_r,is_perm_c;

    if (!rsorted) {
      PetscInt *idxs_r,i;
      ierr = PetscMalloc1(rsize,&idxs_r);CHKERRQ(ierr);
      for (i=0;i<rsize;i++) {
        idxs_r[idxs_perm_r[i]] = i;
      }
      ierr = PetscFree(idxs_perm_r);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF,rsize,idxs_r,PETSC_OWN_POINTER,&is_perm_r);CHKERRQ(ierr);
    } else {
      ierr = ISCreateStride(PETSC_COMM_SELF,rsize,0,1,&is_perm_r);CHKERRQ(ierr);
    }
    ierr = ISSetPermutation(is_perm_r);CHKERRQ(ierr);

    if (!csorted) {
      if (isrow_s == iscol_s) {
        ierr = PetscObjectReference((PetscObject)is_perm_r);CHKERRQ(ierr);
        is_perm_c = is_perm_r;
      } else {
        PetscInt *idxs_c,i;
        if (!idxs_perm_c) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Permutation array not present");
        ierr = PetscMalloc1(csize,&idxs_c);CHKERRQ(ierr);
        for (i=0;i<csize;i++) {
          idxs_c[idxs_perm_c[i]] = i;
        }
        ierr = PetscFree(idxs_perm_c);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PETSC_COMM_SELF,csize,idxs_c,PETSC_OWN_POINTER,&is_perm_c);CHKERRQ(ierr);
      }
    } else {
      ierr = ISCreateStride(PETSC_COMM_SELF,csize,0,1,&is_perm_c);CHKERRQ(ierr);
    }
    ierr = ISSetPermutation(is_perm_c);CHKERRQ(ierr);

    ierr = MatPermute(work_mat[0],is_perm_r,is_perm_c,&new_mat);CHKERRQ(ierr);
    ierr = MatDestroy(&work_mat[0]);CHKERRQ(ierr);
    work_mat[0] = new_mat;
    ierr = ISDestroy(&is_perm_r);CHKERRQ(ierr);
    ierr = ISDestroy(&is_perm_c);CHKERRQ(ierr);
  }

  ierr = PetscObjectReference((PetscObject)work_mat[0]);CHKERRQ(ierr);
  *B = work_mat[0];
  ierr = MatDestroyMatrices(1,&work_mat);CHKERRQ(ierr);
  ierr = ISDestroy(&isrow_s);CHKERRQ(ierr);
  ierr = ISDestroy(&iscol_s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCComputeLocalMatrix(PC pc, Mat ChangeOfBasisMatrix)
{
  Mat_IS*        matis = (Mat_IS*)pc->pmat->data;
  PC_BDDC*       pcbddc = (PC_BDDC*)pc->data;
  Mat            new_mat,lA;
  IS             is_local,is_global;
  PetscInt       local_size;
  PetscBool      isseqaij;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&pcbddc->local_mat);CHKERRQ(ierr);
  ierr = MatGetSize(matis->A,&local_size,NULL);CHKERRQ(ierr);
  ierr = ISCreateStride(PetscObjectComm((PetscObject)matis->A),local_size,0,1,&is_local);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyIS(pc->pmat->rmap->mapping,is_local,&is_global);CHKERRQ(ierr);
  ierr = ISDestroy(&is_local);CHKERRQ(ierr);
  ierr = MatCreateSubMatrixUnsorted(ChangeOfBasisMatrix,is_global,is_global,&new_mat);CHKERRQ(ierr);
  ierr = ISDestroy(&is_global);CHKERRQ(ierr);

  if (pcbddc->dbg_flag) {
    Vec       x,x_change;
    PetscReal error;

    ierr = MatCreateVecs(ChangeOfBasisMatrix,&x,&x_change);CHKERRQ(ierr);
    ierr = VecSetRandom(x,NULL);CHKERRQ(ierr);
    ierr = MatMult(ChangeOfBasisMatrix,x,x_change);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->cctx,x,matis->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(matis->cctx,x,matis->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = MatMult(new_mat,matis->x,matis->y);CHKERRQ(ierr);
    if (!pcbddc->change_interior) {
      const PetscScalar *x,*y,*v;
      PetscReal         lerror = 0.;
      PetscInt          i;

      ierr = VecGetArrayRead(matis->x,&x);CHKERRQ(ierr);
      ierr = VecGetArrayRead(matis->y,&y);CHKERRQ(ierr);
      ierr = VecGetArrayRead(matis->counter,&v);CHKERRQ(ierr);
      for (i=0;i<local_size;i++)
        if (PetscRealPart(v[i]) < 1.5 && PetscAbsScalar(x[i]-y[i]) > lerror)
          lerror = PetscAbsScalar(x[i]-y[i]);
      ierr = VecRestoreArrayRead(matis->x,&x);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(matis->y,&y);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(matis->counter,&v);CHKERRQ(ierr);
      ierr = MPIU_Allreduce(&lerror,&error,1,MPIU_REAL,MPI_MAX,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
      if (error > PETSC_SMALL) {
        if (!pcbddc->user_ChangeOfBasisMatrix || pcbddc->current_level) {
          SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"Error global vs local change on I: %1.6e",error);
        } else {
          SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"Error global vs local change on I: %1.6e",error);
        }
      }
    }
    ierr = VecScatterBegin(matis->rctx,matis->y,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(matis->rctx,matis->y,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecAXPY(x,-1.0,x_change);CHKERRQ(ierr);
    ierr = VecNorm(x,NORM_INFINITY,&error);CHKERRQ(ierr);
    if (error > PETSC_SMALL) {
      if (!pcbddc->user_ChangeOfBasisMatrix || pcbddc->current_level) {
        SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"Error global vs local change on N: %1.6e",error);
      } else {
        SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"Error global vs local change on N: %1.6e",error);
      }
    }
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&x_change);CHKERRQ(ierr);
  }

  /* lA is present if we are setting up an inner BDDC for a saddle point FETI-DP */
  ierr = PetscObjectQuery((PetscObject)pc,"__KSPFETIDP_lA" ,(PetscObject*)&lA);CHKERRQ(ierr);

  /* TODO: HOW TO WORK WITH BAIJ and SBAIJ and SEQDENSE? */
  ierr = PetscObjectBaseTypeCompare((PetscObject)matis->A,MATSEQAIJ,&isseqaij);CHKERRQ(ierr);
  if (isseqaij) {
    ierr = MatDestroy(&pcbddc->local_mat);CHKERRQ(ierr);
    ierr = MatPtAP(matis->A,new_mat,MAT_INITIAL_MATRIX,2.0,&pcbddc->local_mat);CHKERRQ(ierr);
    if (lA) {
      Mat work;
      ierr = MatPtAP(lA,new_mat,MAT_INITIAL_MATRIX,2.0,&work);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)pc,"__KSPFETIDP_lA" ,(PetscObject)work);CHKERRQ(ierr);
      ierr = MatDestroy(&work);CHKERRQ(ierr);
    }
  } else {
    Mat work_mat;

    ierr = MatDestroy(&pcbddc->local_mat);CHKERRQ(ierr);
    ierr = MatConvert(matis->A,MATSEQAIJ,MAT_INITIAL_MATRIX,&work_mat);CHKERRQ(ierr);
    ierr = MatPtAP(work_mat,new_mat,MAT_INITIAL_MATRIX,2.0,&pcbddc->local_mat);CHKERRQ(ierr);
    ierr = MatDestroy(&work_mat);CHKERRQ(ierr);
    if (lA) {
      Mat work;
      ierr = MatConvert(lA,MATSEQAIJ,MAT_INITIAL_MATRIX,&work_mat);CHKERRQ(ierr);
      ierr = MatPtAP(work_mat,new_mat,MAT_INITIAL_MATRIX,2.0,&work);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)pc,"__KSPFETIDP_lA" ,(PetscObject)work);CHKERRQ(ierr);
      ierr = MatDestroy(&work);CHKERRQ(ierr);
    }
  }
  if (matis->A->symmetric_set) {
    ierr = MatSetOption(pcbddc->local_mat,MAT_SYMMETRIC,matis->A->symmetric);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    ierr = MatSetOption(pcbddc->local_mat,MAT_HERMITIAN,matis->A->symmetric);CHKERRQ(ierr);
#endif
  }
  ierr = MatDestroy(&new_mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSetUpLocalScatters(PC pc)
{
  PC_IS*          pcis = (PC_IS*)(pc->data);
  PC_BDDC*        pcbddc = (PC_BDDC*)pc->data;
  PCBDDCSubSchurs sub_schurs = pcbddc->sub_schurs;
  PetscInt        *idx_R_local=NULL;
  PetscInt        n_vertices,i,j,n_R,n_D,n_B;
  PetscInt        vbs,bs;
  PetscBT         bitmask=NULL;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /*
    No need to setup local scatters if
      - primal space is unchanged
        AND
      - we actually have locally some primal dofs (could not be true in multilevel or for isolated subdomains)
        AND
      - we are not in debugging mode (this is needed since there are Synchronized prints at the end of the subroutine
  */
  if (!pcbddc->new_primal_space_local && pcbddc->local_primal_size && !pcbddc->dbg_flag) {
    PetscFunctionReturn(0);
  }
  /* destroy old objects */
  ierr = ISDestroy(&pcbddc->is_R_local);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&pcbddc->R_to_B);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&pcbddc->R_to_D);CHKERRQ(ierr);
  /* Set Non-overlapping dimensions */
  n_B = pcis->n_B;
  n_D = pcis->n - n_B;
  n_vertices = pcbddc->n_vertices;

  /* Dohrmann's notation: dofs splitted in R (Remaining: all dofs but the vertices) and V (Vertices) */

  /* create auxiliary bitmask and allocate workspace */
  if (!sub_schurs || !sub_schurs->reuse_solver) {
    ierr = PetscMalloc1(pcis->n-n_vertices,&idx_R_local);CHKERRQ(ierr);
    ierr = PetscBTCreate(pcis->n,&bitmask);CHKERRQ(ierr);
    for (i=0;i<n_vertices;i++) {
      ierr = PetscBTSet(bitmask,pcbddc->local_primal_ref_node[i]);CHKERRQ(ierr);
    }

    for (i=0, n_R=0; i<pcis->n; i++) {
      if (!PetscBTLookup(bitmask,i)) {
        idx_R_local[n_R++] = i;
      }
    }
  } else { /* A different ordering (already computed) is present if we are reusing the Schur solver */
    PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

    ierr = ISGetIndices(reuse_solver->is_R,(const PetscInt**)&idx_R_local);CHKERRQ(ierr);
    ierr = ISGetLocalSize(reuse_solver->is_R,&n_R);CHKERRQ(ierr);
  }

  /* Block code */
  vbs = 1;
  ierr = MatGetBlockSize(pcbddc->local_mat,&bs);CHKERRQ(ierr);
  if (bs>1 && !(n_vertices%bs)) {
    PetscBool is_blocked = PETSC_TRUE;
    PetscInt  *vary;
    if (!sub_schurs || !sub_schurs->reuse_solver) {
      ierr = PetscMalloc1(pcis->n/bs,&vary);CHKERRQ(ierr);
      ierr = PetscArrayzero(vary,pcis->n/bs);CHKERRQ(ierr);
      /* Verify that the vertex indices correspond to each element in a block (code taken from sbaij2.c) */
      /* it is ok to check this way since local_primal_ref_node are always sorted by local numbering and idx_R_local is obtained as a complement */
      for (i=0; i<n_vertices; i++) vary[pcbddc->local_primal_ref_node[i]/bs]++;
      for (i=0; i<pcis->n/bs; i++) {
        if (vary[i]!=0 && vary[i]!=bs) {
          is_blocked = PETSC_FALSE;
          break;
        }
      }
      ierr = PetscFree(vary);CHKERRQ(ierr);
    } else {
      /* Verify directly the R set */
      for (i=0; i<n_R/bs; i++) {
        PetscInt j,node=idx_R_local[bs*i];
        for (j=1; j<bs; j++) {
          if (node != idx_R_local[bs*i+j]-j) {
            is_blocked = PETSC_FALSE;
            break;
          }
        }
      }
    }
    if (is_blocked) { /* build compressed IS for R nodes (complement of vertices) */
      vbs = bs;
      for (i=0;i<n_R/vbs;i++) {
        idx_R_local[i] = idx_R_local[vbs*i]/vbs;
      }
    }
  }
  ierr = ISCreateBlock(PETSC_COMM_SELF,vbs,n_R/vbs,idx_R_local,PETSC_COPY_VALUES,&pcbddc->is_R_local);CHKERRQ(ierr);
  if (sub_schurs && sub_schurs->reuse_solver) {
    PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

    ierr = ISRestoreIndices(reuse_solver->is_R,(const PetscInt**)&idx_R_local);CHKERRQ(ierr);
    ierr = ISDestroy(&reuse_solver->is_R);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)pcbddc->is_R_local);CHKERRQ(ierr);
    reuse_solver->is_R = pcbddc->is_R_local;
  } else {
    ierr = PetscFree(idx_R_local);CHKERRQ(ierr);
  }

  /* print some info if requested */
  if (pcbddc->dbg_flag) {
    ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d local dimensions\n",PetscGlobalRank);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"local_size = %D, dirichlet_size = %D, boundary_size = %D\n",pcis->n,n_D,n_B);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"r_size = %D, v_size = %D, constraints = %D, local_primal_size = %D\n",n_R,n_vertices,pcbddc->local_primal_size-n_vertices-pcbddc->benign_n,pcbddc->local_primal_size);CHKERRQ(ierr);
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
  }

  /* VecScatters pcbddc->R_to_B and (optionally) pcbddc->R_to_D */
  if (!sub_schurs || !sub_schurs->reuse_solver) {
    IS       is_aux1,is_aux2;
    PetscInt *aux_array1,*aux_array2,*is_indices,*idx_R_local;

    ierr = ISGetIndices(pcbddc->is_R_local,(const PetscInt**)&idx_R_local);CHKERRQ(ierr);
    ierr = PetscMalloc1(pcis->n_B-n_vertices,&aux_array1);CHKERRQ(ierr);
    ierr = PetscMalloc1(pcis->n_B-n_vertices,&aux_array2);CHKERRQ(ierr);
    ierr = ISGetIndices(pcis->is_I_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (i=0; i<n_D; i++) {
      ierr = PetscBTSet(bitmask,is_indices[i]);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(pcis->is_I_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (i=0, j=0; i<n_R; i++) {
      if (!PetscBTLookup(bitmask,idx_R_local[i])) {
        aux_array1[j++] = i;
      }
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,j,aux_array1,PETSC_OWN_POINTER,&is_aux1);CHKERRQ(ierr);
    ierr = ISGetIndices(pcis->is_B_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (i=0, j=0; i<n_B; i++) {
      if (!PetscBTLookup(bitmask,is_indices[i])) {
        aux_array2[j++] = i;
      }
    }
    ierr = ISRestoreIndices(pcis->is_B_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,j,aux_array2,PETSC_OWN_POINTER,&is_aux2);CHKERRQ(ierr);
    ierr = VecScatterCreate(pcbddc->vec1_R,is_aux1,pcis->vec1_B,is_aux2,&pcbddc->R_to_B);CHKERRQ(ierr);
    ierr = ISDestroy(&is_aux1);CHKERRQ(ierr);
    ierr = ISDestroy(&is_aux2);CHKERRQ(ierr);

    if (pcbddc->switch_static || pcbddc->dbg_flag) {
      ierr = PetscMalloc1(n_D,&aux_array1);CHKERRQ(ierr);
      for (i=0, j=0; i<n_R; i++) {
        if (PetscBTLookup(bitmask,idx_R_local[i])) {
          aux_array1[j++] = i;
        }
      }
      ierr = ISCreateGeneral(PETSC_COMM_SELF,j,aux_array1,PETSC_OWN_POINTER,&is_aux1);CHKERRQ(ierr);
      ierr = VecScatterCreate(pcbddc->vec1_R,is_aux1,pcis->vec1_D,(IS)0,&pcbddc->R_to_D);CHKERRQ(ierr);
      ierr = ISDestroy(&is_aux1);CHKERRQ(ierr);
    }
    ierr = PetscBTDestroy(&bitmask);CHKERRQ(ierr);
    ierr = ISRestoreIndices(pcbddc->is_R_local,(const PetscInt**)&idx_R_local);CHKERRQ(ierr);
  } else {
    PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;
    IS                 tis;
    PetscInt           schur_size;

    ierr = ISGetLocalSize(reuse_solver->is_B,&schur_size);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,schur_size,n_D,1,&tis);CHKERRQ(ierr);
    ierr = VecScatterCreate(pcbddc->vec1_R,tis,pcis->vec1_B,reuse_solver->is_B,&pcbddc->R_to_B);CHKERRQ(ierr);
    ierr = ISDestroy(&tis);CHKERRQ(ierr);
    if (pcbddc->switch_static || pcbddc->dbg_flag) {
      ierr = ISCreateStride(PETSC_COMM_SELF,n_D,0,1,&tis);CHKERRQ(ierr);
      ierr = VecScatterCreate(pcbddc->vec1_R,tis,pcis->vec1_D,(IS)0,&pcbddc->R_to_D);CHKERRQ(ierr);
      ierr = ISDestroy(&tis);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatNullSpacePropagateAny_Private(Mat A, IS is, Mat B)
{
  MatNullSpace   NullSpace;
  Mat            dmat;
  const Vec      *nullvecs;
  Vec            v,v2,*nullvecs2;
  VecScatter     sct = NULL;
  PetscContainer c;
  PetscScalar    *ddata;
  PetscInt       k,nnsp_size,bsiz,bsiz2,n,N,bs;
  PetscBool      nnsp_has_cnst;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!is && !B) { /* MATIS */
    Mat_IS* matis = (Mat_IS*)A->data;

    if (!B) {
      ierr = MatISGetLocalMat(A,&B);CHKERRQ(ierr);
    }
    sct  = matis->cctx;
    ierr = PetscObjectReference((PetscObject)sct);CHKERRQ(ierr);
  } else {
    ierr = MatGetNullSpace(B,&NullSpace);CHKERRQ(ierr);
    if (!NullSpace) {
      ierr = MatGetNearNullSpace(B,&NullSpace);CHKERRQ(ierr);
    }
    if (NullSpace) PetscFunctionReturn(0);
  }
  ierr = MatGetNullSpace(A,&NullSpace);CHKERRQ(ierr);
  if (!NullSpace) {
    ierr = MatGetNearNullSpace(A,&NullSpace);CHKERRQ(ierr);
  }
  if (!NullSpace) PetscFunctionReturn(0);

  ierr = MatCreateVecs(A,&v,NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(B,&v2,NULL);CHKERRQ(ierr);
  if (!sct) {
    ierr = VecScatterCreate(v,is,v2,NULL,&sct);CHKERRQ(ierr);
  }
  ierr = MatNullSpaceGetVecs(NullSpace,&nnsp_has_cnst,&nnsp_size,(const Vec**)&nullvecs);CHKERRQ(ierr);
  bsiz = bsiz2 = nnsp_size+!!nnsp_has_cnst;
  ierr = PetscMalloc1(bsiz,&nullvecs2);CHKERRQ(ierr);
  ierr = VecGetBlockSize(v2,&bs);CHKERRQ(ierr);
  ierr = VecGetSize(v2,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v2,&n);CHKERRQ(ierr);
  ierr = PetscMalloc1(n*bsiz,&ddata);CHKERRQ(ierr);
  for (k=0;k<nnsp_size;k++) {
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)B),bs,n,N,ddata + n*k,&nullvecs2[k]);CHKERRQ(ierr);
    ierr = VecScatterBegin(sct,nullvecs[k],nullvecs2[k],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(sct,nullvecs[k],nullvecs2[k],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  if (nnsp_has_cnst) {
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)B),bs,n,N,ddata + n*nnsp_size,&nullvecs2[nnsp_size]);CHKERRQ(ierr);
    ierr = VecSet(nullvecs2[nnsp_size],1.0);CHKERRQ(ierr);
  }
  ierr = PCBDDCOrthonormalizeVecs(&bsiz2,nullvecs2);CHKERRQ(ierr);
  ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject)B),PETSC_FALSE,bsiz2,nullvecs2,&NullSpace);CHKERRQ(ierr);

  ierr = MatCreateDense(PetscObjectComm((PetscObject)B),n,PETSC_DECIDE,N,bsiz2,ddata,&dmat);CHKERRQ(ierr);
  ierr = PetscContainerCreate(PetscObjectComm((PetscObject)B),&c);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(c,ddata);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(c,PetscContainerUserDestroyDefault);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)dmat,"_PBDDC_Null_dmat_arr",(PetscObject)c);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&c);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)NullSpace,"_PBDDC_Null_dmat",(PetscObject)dmat);CHKERRQ(ierr);
  ierr = MatDestroy(&dmat);CHKERRQ(ierr);

  for (k=0;k<bsiz;k++) {
    ierr = VecDestroy(&nullvecs2[k]);CHKERRQ(ierr);
  }
  ierr = PetscFree(nullvecs2);CHKERRQ(ierr);
  ierr = MatSetNearNullSpace(B,NullSpace);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&NullSpace);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&v2);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&sct);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSetUpLocalSolvers(PC pc, PetscBool dirichlet, PetscBool neumann)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PC_IS          *pcis = (PC_IS*)pc->data;
  PC             pc_temp;
  Mat            A_RR;
  MatNullSpace   nnsp;
  MatReuse       reuse;
  PetscScalar    m_one = -1.0;
  PetscReal      value;
  PetscInt       n_D,n_R;
  PetscBool      issbaij,opts;
  PetscErrorCode ierr;
  void           (*f)(void) = 0;
  char           dir_prefix[256],neu_prefix[256],str_level[16];
  size_t         len;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PC_BDDC_LocalSolvers[pcbddc->current_level],pc,0,0,0);CHKERRQ(ierr);
  /* approximate solver, propagate NearNullSpace if needed */
  if (!pc->setupcalled && (pcbddc->NullSpace_corr[0] || pcbddc->NullSpace_corr[2])) {
    MatNullSpace gnnsp1,gnnsp2;
    PetscBool    lhas,ghas;

    ierr = MatGetNearNullSpace(pcbddc->local_mat,&nnsp);CHKERRQ(ierr);
    ierr = MatGetNearNullSpace(pc->pmat,&gnnsp1);CHKERRQ(ierr);
    ierr = MatGetNullSpace(pc->pmat,&gnnsp2);CHKERRQ(ierr);
    lhas = nnsp ? PETSC_TRUE : PETSC_FALSE;
    ierr = MPIU_Allreduce(&lhas,&ghas,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
    if (!ghas && (gnnsp1 || gnnsp2)) {
      ierr = MatNullSpacePropagateAny_Private(pc->pmat,NULL,NULL);CHKERRQ(ierr);
    }
  }

  /* compute prefixes */
  ierr = PetscStrcpy(dir_prefix,"");CHKERRQ(ierr);
  ierr = PetscStrcpy(neu_prefix,"");CHKERRQ(ierr);
  if (!pcbddc->current_level) {
    ierr = PetscStrncpy(dir_prefix,((PetscObject)pc)->prefix,sizeof(dir_prefix));CHKERRQ(ierr);
    ierr = PetscStrncpy(neu_prefix,((PetscObject)pc)->prefix,sizeof(neu_prefix));CHKERRQ(ierr);
    ierr = PetscStrlcat(dir_prefix,"pc_bddc_dirichlet_",sizeof(dir_prefix));CHKERRQ(ierr);
    ierr = PetscStrlcat(neu_prefix,"pc_bddc_neumann_",sizeof(neu_prefix));CHKERRQ(ierr);
  } else {
    ierr = PetscSNPrintf(str_level,sizeof(str_level),"l%d_",(int)(pcbddc->current_level));CHKERRQ(ierr);
    ierr = PetscStrlen(((PetscObject)pc)->prefix,&len);CHKERRQ(ierr);
    len -= 15; /* remove "pc_bddc_coarse_" */
    if (pcbddc->current_level>1) len -= 3; /* remove "lX_" with X level number */
    if (pcbddc->current_level>10) len -= 1; /* remove another char from level number */
    /* Nonstandard use of PetscStrncpy() to only copy a portion of the input string */
    ierr = PetscStrncpy(dir_prefix,((PetscObject)pc)->prefix,len+1);CHKERRQ(ierr);
    ierr = PetscStrncpy(neu_prefix,((PetscObject)pc)->prefix,len+1);CHKERRQ(ierr);
    ierr = PetscStrlcat(dir_prefix,"pc_bddc_dirichlet_",sizeof(dir_prefix));CHKERRQ(ierr);
    ierr = PetscStrlcat(neu_prefix,"pc_bddc_neumann_",sizeof(neu_prefix));CHKERRQ(ierr);
    ierr = PetscStrlcat(dir_prefix,str_level,sizeof(dir_prefix));CHKERRQ(ierr);
    ierr = PetscStrlcat(neu_prefix,str_level,sizeof(neu_prefix));CHKERRQ(ierr);
  }

  /* DIRICHLET PROBLEM */
  if (dirichlet) {
    PCBDDCSubSchurs sub_schurs = pcbddc->sub_schurs;
    if (pcbddc->benign_n && !pcbddc->benign_change_explicit) {
      if (!sub_schurs || !sub_schurs->reuse_solver) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not yet implemented");
      if (pcbddc->dbg_flag) {
        Mat    A_IIn;

        ierr = PCBDDCBenignProject(pc,pcis->is_I_local,pcis->is_I_local,&A_IIn);CHKERRQ(ierr);
        ierr = MatDestroy(&pcis->A_II);CHKERRQ(ierr);
        pcis->A_II = A_IIn;
      }
    }
    if (pcbddc->local_mat->symmetric_set) {
      ierr = MatSetOption(pcis->A_II,MAT_SYMMETRIC,pcbddc->local_mat->symmetric);CHKERRQ(ierr);
    }
    /* Matrix for Dirichlet problem is pcis->A_II */
    n_D  = pcis->n - pcis->n_B;
    opts = PETSC_FALSE;
    if (!pcbddc->ksp_D) { /* create object if not yet build */
      opts = PETSC_TRUE;
      ierr = KSPCreate(PETSC_COMM_SELF,&pcbddc->ksp_D);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)pcbddc->ksp_D,(PetscObject)pc,1);CHKERRQ(ierr);
      /* default */
      ierr = KSPSetType(pcbddc->ksp_D,KSPPREONLY);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(pcbddc->ksp_D,dir_prefix);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)pcis->pA_II,MATSEQSBAIJ,&issbaij);CHKERRQ(ierr);
      ierr = KSPGetPC(pcbddc->ksp_D,&pc_temp);CHKERRQ(ierr);
      if (issbaij) {
        ierr = PCSetType(pc_temp,PCCHOLESKY);CHKERRQ(ierr);
      } else {
        ierr = PCSetType(pc_temp,PCLU);CHKERRQ(ierr);
      }
      ierr = KSPSetErrorIfNotConverged(pcbddc->ksp_D,pc->erroriffailure);CHKERRQ(ierr);
    }
    ierr = MatSetOptionsPrefix(pcis->pA_II,((PetscObject)pcbddc->ksp_D)->prefix);CHKERRQ(ierr);
    ierr = KSPSetOperators(pcbddc->ksp_D,pcis->A_II,pcis->pA_II);CHKERRQ(ierr);
    /* Allow user's customization */
    if (opts) {
      ierr = KSPSetFromOptions(pcbddc->ksp_D);CHKERRQ(ierr);
    }
    ierr = MatGetNearNullSpace(pcis->pA_II,&nnsp);CHKERRQ(ierr);
    if (pcbddc->NullSpace_corr[0] && !nnsp) { /* approximate solver, propagate NearNullSpace */
      ierr = MatNullSpacePropagateAny_Private(pcbddc->local_mat,pcis->is_I_local,pcis->pA_II);CHKERRQ(ierr);
    }
    ierr = MatGetNearNullSpace(pcis->pA_II,&nnsp);CHKERRQ(ierr);
    ierr = KSPGetPC(pcbddc->ksp_D,&pc_temp);CHKERRQ(ierr);
    ierr = PetscObjectQueryFunction((PetscObject)pc_temp,"PCSetCoordinates_C",&f);CHKERRQ(ierr);
    if (f && pcbddc->mat_graph->cloc && !nnsp) {
      PetscReal      *coords = pcbddc->mat_graph->coords,*scoords;
      const PetscInt *idxs;
      PetscInt       cdim = pcbddc->mat_graph->cdim,nl,i,d;

      ierr = ISGetLocalSize(pcis->is_I_local,&nl);CHKERRQ(ierr);
      ierr = ISGetIndices(pcis->is_I_local,&idxs);CHKERRQ(ierr);
      ierr = PetscMalloc1(nl*cdim,&scoords);CHKERRQ(ierr);
      for (i=0;i<nl;i++) {
        for (d=0;d<cdim;d++) {
          scoords[i*cdim+d] = coords[idxs[i]*cdim+d];
        }
      }
      ierr = ISRestoreIndices(pcis->is_I_local,&idxs);CHKERRQ(ierr);
      ierr = PCSetCoordinates(pc_temp,cdim,nl,scoords);CHKERRQ(ierr);
      ierr = PetscFree(scoords);CHKERRQ(ierr);
    }
    if (sub_schurs && sub_schurs->reuse_solver) {
      PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

      ierr = KSPSetPC(pcbddc->ksp_D,reuse_solver->interior_solver);CHKERRQ(ierr);
    }

    /* umfpack interface has a bug when matrix dimension is zero. TODO solve from umfpack interface */
    if (!n_D) {
      ierr = KSPGetPC(pcbddc->ksp_D,&pc_temp);CHKERRQ(ierr);
      ierr = PCSetType(pc_temp,PCNONE);CHKERRQ(ierr);
    }
    ierr = KSPSetUp(pcbddc->ksp_D);CHKERRQ(ierr);
    /* set ksp_D into pcis data */
    ierr = PetscObjectReference((PetscObject)pcbddc->ksp_D);CHKERRQ(ierr);
    ierr = KSPDestroy(&pcis->ksp_D);CHKERRQ(ierr);
    pcis->ksp_D = pcbddc->ksp_D;
  }

  /* NEUMANN PROBLEM */
  A_RR = 0;
  if (neumann) {
    PCBDDCSubSchurs sub_schurs = pcbddc->sub_schurs;
    PetscInt        ibs,mbs;
    PetscBool       issbaij, reuse_neumann_solver;
    Mat_IS*         matis = (Mat_IS*)pc->pmat->data;

    reuse_neumann_solver = PETSC_FALSE;
    if (sub_schurs && sub_schurs->reuse_solver) {
      IS iP;

      reuse_neumann_solver = PETSC_TRUE;
      ierr = PetscObjectQuery((PetscObject)sub_schurs->A,"__KSPFETIDP_iP",(PetscObject*)&iP);CHKERRQ(ierr);
      if (iP) reuse_neumann_solver = PETSC_FALSE;
    }
    /* Matrix for Neumann problem is A_RR -> we need to create/reuse it at this point */
    ierr = ISGetSize(pcbddc->is_R_local,&n_R);CHKERRQ(ierr);
    if (pcbddc->ksp_R) { /* already created ksp */
      PetscInt nn_R;
      ierr = KSPGetOperators(pcbddc->ksp_R,NULL,&A_RR);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)A_RR);CHKERRQ(ierr);
      ierr = MatGetSize(A_RR,&nn_R,NULL);CHKERRQ(ierr);
      if (nn_R != n_R) { /* old ksp is not reusable, so reset it */
        ierr = KSPReset(pcbddc->ksp_R);CHKERRQ(ierr);
        ierr = MatDestroy(&A_RR);CHKERRQ(ierr);
        reuse = MAT_INITIAL_MATRIX;
      } else { /* same sizes, but nonzero pattern depend on primal vertices so it can be changed */
        if (pcbddc->new_primal_space_local) { /* we are not sure the matrix will have the same nonzero pattern */
          ierr = MatDestroy(&A_RR);CHKERRQ(ierr);
          reuse = MAT_INITIAL_MATRIX;
        } else { /* safe to reuse the matrix */
          reuse = MAT_REUSE_MATRIX;
        }
      }
      /* last check */
      if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
        ierr = MatDestroy(&A_RR);CHKERRQ(ierr);
        reuse = MAT_INITIAL_MATRIX;
      }
    } else { /* first time, so we need to create the matrix */
      reuse = MAT_INITIAL_MATRIX;
    }
    /* convert pcbddc->local_mat if needed later in PCBDDCSetUpCorrection
       TODO: Get Rid of these conversions */
    ierr = MatGetBlockSize(pcbddc->local_mat,&mbs);CHKERRQ(ierr);
    ierr = ISGetBlockSize(pcbddc->is_R_local,&ibs);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)pcbddc->local_mat,MATSEQSBAIJ,&issbaij);CHKERRQ(ierr);
    if (ibs != mbs) { /* need to convert to SEQAIJ to extract any submatrix with is_R_local */
      if (matis->A == pcbddc->local_mat) {
        ierr = MatDestroy(&pcbddc->local_mat);CHKERRQ(ierr);
        ierr = MatConvert(matis->A,MATSEQAIJ,MAT_INITIAL_MATRIX,&pcbddc->local_mat);CHKERRQ(ierr);
      } else {
        ierr = MatConvert(pcbddc->local_mat,MATSEQAIJ,MAT_INPLACE_MATRIX,&pcbddc->local_mat);CHKERRQ(ierr);
      }
    } else if (issbaij) { /* need to convert to BAIJ to get offdiagonal blocks */
      if (matis->A == pcbddc->local_mat) {
        ierr = MatDestroy(&pcbddc->local_mat);CHKERRQ(ierr);
        ierr = MatConvert(matis->A,mbs > 1 ? MATSEQBAIJ : MATSEQAIJ,MAT_INITIAL_MATRIX,&pcbddc->local_mat);CHKERRQ(ierr);
      } else {
        ierr = MatConvert(pcbddc->local_mat,mbs > 1 ? MATSEQBAIJ : MATSEQAIJ,MAT_INPLACE_MATRIX,&pcbddc->local_mat);CHKERRQ(ierr);
      }
    }
    /* extract A_RR */
    if (reuse_neumann_solver) {
      PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

      if (pcbddc->dbg_flag) { /* we need A_RR to test the solver later */
        ierr = MatDestroy(&A_RR);CHKERRQ(ierr);
        if (reuse_solver->benign_n) { /* we are not using the explicit change of basis on the pressures */
          ierr = PCBDDCBenignProject(pc,pcbddc->is_R_local,pcbddc->is_R_local,&A_RR);CHKERRQ(ierr);
        } else {
          ierr = MatCreateSubMatrix(pcbddc->local_mat,pcbddc->is_R_local,pcbddc->is_R_local,MAT_INITIAL_MATRIX,&A_RR);CHKERRQ(ierr);
        }
      } else {
        ierr = MatDestroy(&A_RR);CHKERRQ(ierr);
        ierr = PCGetOperators(reuse_solver->correction_solver,&A_RR,NULL);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)A_RR);CHKERRQ(ierr);
      }
    } else { /* we have to build the neumann solver, so we need to extract the relevant matrix */
      ierr = MatCreateSubMatrix(pcbddc->local_mat,pcbddc->is_R_local,pcbddc->is_R_local,reuse,&A_RR);CHKERRQ(ierr);
    }
    if (pcbddc->local_mat->symmetric_set) {
      ierr = MatSetOption(A_RR,MAT_SYMMETRIC,pcbddc->local_mat->symmetric);CHKERRQ(ierr);
    }
    opts = PETSC_FALSE;
    if (!pcbddc->ksp_R) { /* create object if not present */
      opts = PETSC_TRUE;
      ierr = KSPCreate(PETSC_COMM_SELF,&pcbddc->ksp_R);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)pcbddc->ksp_R,(PetscObject)pc,1);CHKERRQ(ierr);
      /* default */
      ierr = KSPSetType(pcbddc->ksp_R,KSPPREONLY);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(pcbddc->ksp_R,neu_prefix);CHKERRQ(ierr);
      ierr = KSPGetPC(pcbddc->ksp_R,&pc_temp);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)A_RR,MATSEQSBAIJ,&issbaij);CHKERRQ(ierr);
      if (issbaij) {
        ierr = PCSetType(pc_temp,PCCHOLESKY);CHKERRQ(ierr);
      } else {
        ierr = PCSetType(pc_temp,PCLU);CHKERRQ(ierr);
      }
      ierr = KSPSetErrorIfNotConverged(pcbddc->ksp_R,pc->erroriffailure);CHKERRQ(ierr);
    }
    ierr = KSPSetOperators(pcbddc->ksp_R,A_RR,A_RR);CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(A_RR,((PetscObject)pcbddc->ksp_R)->prefix);CHKERRQ(ierr);
    if (opts) { /* Allow user's customization once */
      ierr = KSPSetFromOptions(pcbddc->ksp_R);CHKERRQ(ierr);
    }
    ierr = MatGetNearNullSpace(A_RR,&nnsp);CHKERRQ(ierr);
    if (pcbddc->NullSpace_corr[2] && !nnsp) { /* approximate solver, propagate NearNullSpace */
      ierr = MatNullSpacePropagateAny_Private(pcbddc->local_mat,pcbddc->is_R_local,A_RR);CHKERRQ(ierr);
    }
    ierr = MatGetNearNullSpace(A_RR,&nnsp);CHKERRQ(ierr);
    ierr = KSPGetPC(pcbddc->ksp_R,&pc_temp);CHKERRQ(ierr);
    ierr = PetscObjectQueryFunction((PetscObject)pc_temp,"PCSetCoordinates_C",&f);CHKERRQ(ierr);
    if (f && pcbddc->mat_graph->cloc && !nnsp) {
      PetscReal      *coords = pcbddc->mat_graph->coords,*scoords;
      const PetscInt *idxs;
      PetscInt       cdim = pcbddc->mat_graph->cdim,nl,i,d;

      ierr = ISGetLocalSize(pcbddc->is_R_local,&nl);CHKERRQ(ierr);
      ierr = ISGetIndices(pcbddc->is_R_local,&idxs);CHKERRQ(ierr);
      ierr = PetscMalloc1(nl*cdim,&scoords);CHKERRQ(ierr);
      for (i=0;i<nl;i++) {
        for (d=0;d<cdim;d++) {
          scoords[i*cdim+d] = coords[idxs[i]*cdim+d];
        }
      }
      ierr = ISRestoreIndices(pcbddc->is_R_local,&idxs);CHKERRQ(ierr);
      ierr = PCSetCoordinates(pc_temp,cdim,nl,scoords);CHKERRQ(ierr);
      ierr = PetscFree(scoords);CHKERRQ(ierr);
    }

    /* umfpack interface has a bug when matrix dimension is zero. TODO solve from umfpack interface */
    if (!n_R) {
      ierr = KSPGetPC(pcbddc->ksp_R,&pc_temp);CHKERRQ(ierr);
      ierr = PCSetType(pc_temp,PCNONE);CHKERRQ(ierr);
    }
    /* Reuse solver if it is present */
    if (reuse_neumann_solver) {
      PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

      ierr = KSPSetPC(pcbddc->ksp_R,reuse_solver->correction_solver);CHKERRQ(ierr);
    }
    ierr = KSPSetUp(pcbddc->ksp_R);CHKERRQ(ierr);
  }

  if (pcbddc->dbg_flag) {
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(PC_BDDC_LocalSolvers[pcbddc->current_level],pc,0,0,0);CHKERRQ(ierr);

  /* adapt Dirichlet and Neumann solvers if a nullspace correction has been requested */
  if (pcbddc->NullSpace_corr[0]) {
    ierr = PCBDDCSetUseExactDirichlet(pc,PETSC_FALSE);CHKERRQ(ierr);
  }
  if (dirichlet && pcbddc->NullSpace_corr[0] && !pcbddc->switch_static) {
    ierr = PCBDDCNullSpaceAssembleCorrection(pc,PETSC_TRUE,pcbddc->NullSpace_corr[1]);CHKERRQ(ierr);
  }
  if (neumann && pcbddc->NullSpace_corr[2]) {
    ierr = PCBDDCNullSpaceAssembleCorrection(pc,PETSC_FALSE,pcbddc->NullSpace_corr[3]);CHKERRQ(ierr);
  }
  /* check Dirichlet and Neumann solvers */
  if (pcbddc->dbg_flag) {
    if (dirichlet) { /* Dirichlet */
      ierr = VecSetRandom(pcis->vec1_D,NULL);CHKERRQ(ierr);
      ierr = MatMult(pcis->A_II,pcis->vec1_D,pcis->vec2_D);CHKERRQ(ierr);
      ierr = KSPSolve(pcbddc->ksp_D,pcis->vec2_D,pcis->vec2_D);CHKERRQ(ierr);
      ierr = KSPCheckSolve(pcbddc->ksp_D,pc,pcis->vec2_D);CHKERRQ(ierr);
      ierr = VecAXPY(pcis->vec1_D,m_one,pcis->vec2_D);CHKERRQ(ierr);
      ierr = VecNorm(pcis->vec1_D,NORM_INFINITY,&value);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Dirichlet solve (%s) = % 1.14e \n",PetscGlobalRank,((PetscObject)(pcbddc->ksp_D))->prefix,value);CHKERRQ(ierr);
      ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    }
    if (neumann) { /* Neumann */
      ierr = VecSetRandom(pcbddc->vec1_R,NULL);CHKERRQ(ierr);
      ierr = MatMult(A_RR,pcbddc->vec1_R,pcbddc->vec2_R);CHKERRQ(ierr);
      ierr = KSPSolve(pcbddc->ksp_R,pcbddc->vec2_R,pcbddc->vec2_R);CHKERRQ(ierr);
      ierr = KSPCheckSolve(pcbddc->ksp_R,pc,pcbddc->vec2_R);CHKERRQ(ierr);
      ierr = VecAXPY(pcbddc->vec1_R,m_one,pcbddc->vec2_R);CHKERRQ(ierr);
      ierr = VecNorm(pcbddc->vec1_R,NORM_INFINITY,&value);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Neumann solve (%s) = % 1.14e\n",PetscGlobalRank,((PetscObject)(pcbddc->ksp_R))->prefix,value);CHKERRQ(ierr);
      ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    }
  }
  /* free Neumann problem's matrix */
  ierr = MatDestroy(&A_RR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCBDDCSolveSubstructureCorrection(PC pc, Vec inout_B, Vec inout_D, PetscBool applytranspose)
{
  PetscErrorCode  ierr;
  PC_BDDC*        pcbddc = (PC_BDDC*)(pc->data);
  PCBDDCSubSchurs sub_schurs = pcbddc->sub_schurs;
  PetscBool       reuse_solver = sub_schurs ? ( sub_schurs->reuse_solver ? PETSC_TRUE : PETSC_FALSE ) : PETSC_FALSE;

  PetscFunctionBegin;
  if (!reuse_solver) {
    ierr = VecSet(pcbddc->vec1_R,0.);CHKERRQ(ierr);
  }
  if (!pcbddc->switch_static) {
    if (applytranspose && pcbddc->local_auxmat1) {
      ierr = MatMultTranspose(pcbddc->local_auxmat2,inout_B,pcbddc->vec1_C);CHKERRQ(ierr);
      ierr = MatMultTransposeAdd(pcbddc->local_auxmat1,pcbddc->vec1_C,inout_B,inout_B);CHKERRQ(ierr);
    }
    if (!reuse_solver) {
      ierr = VecScatterBegin(pcbddc->R_to_B,inout_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(pcbddc->R_to_B,inout_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    } else {
      PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

      ierr = VecScatterBegin(reuse_solver->correction_scatter_B,inout_B,reuse_solver->rhs_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(reuse_solver->correction_scatter_B,inout_B,reuse_solver->rhs_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    }
  } else {
    ierr = VecScatterBegin(pcbddc->R_to_B,inout_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcbddc->R_to_B,inout_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcbddc->R_to_D,inout_D,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcbddc->R_to_D,inout_D,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    if (applytranspose && pcbddc->local_auxmat1) {
      ierr = MatMultTranspose(pcbddc->local_auxmat2,pcbddc->vec1_R,pcbddc->vec1_C);CHKERRQ(ierr);
      ierr = MatMultTransposeAdd(pcbddc->local_auxmat1,pcbddc->vec1_C,inout_B,inout_B);CHKERRQ(ierr);
      ierr = VecScatterBegin(pcbddc->R_to_B,inout_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(pcbddc->R_to_B,inout_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    }
  }
  if (!reuse_solver || pcbddc->switch_static) {
    if (applytranspose) {
      ierr = KSPSolveTranspose(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec1_R);CHKERRQ(ierr);
    } else {
      ierr = KSPSolve(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec1_R);CHKERRQ(ierr);
    }
    ierr = KSPCheckSolve(pcbddc->ksp_R,pc,pcbddc->vec1_R);CHKERRQ(ierr);
  } else {
    PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

    if (applytranspose) {
      ierr = MatFactorSolveSchurComplementTranspose(reuse_solver->F,reuse_solver->rhs_B,reuse_solver->sol_B);CHKERRQ(ierr);
    } else {
      ierr = MatFactorSolveSchurComplement(reuse_solver->F,reuse_solver->rhs_B,reuse_solver->sol_B);CHKERRQ(ierr);
    }
  }
  ierr = VecSet(inout_B,0.);CHKERRQ(ierr);
  if (!pcbddc->switch_static) {
    if (!reuse_solver) {
      ierr = VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,inout_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,inout_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    } else {
      PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

      ierr = VecScatterBegin(reuse_solver->correction_scatter_B,reuse_solver->sol_B,inout_B,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(reuse_solver->correction_scatter_B,reuse_solver->sol_B,inout_B,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    }
    if (!applytranspose && pcbddc->local_auxmat1) {
      ierr = MatMult(pcbddc->local_auxmat1,inout_B,pcbddc->vec1_C);CHKERRQ(ierr);
      ierr = MatMultAdd(pcbddc->local_auxmat2,pcbddc->vec1_C,inout_B,inout_B);CHKERRQ(ierr);
    }
  } else {
    ierr = VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,inout_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,inout_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,inout_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,inout_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    if (!applytranspose && pcbddc->local_auxmat1) {
      ierr = MatMult(pcbddc->local_auxmat1,inout_B,pcbddc->vec1_C);CHKERRQ(ierr);
      ierr = MatMultAdd(pcbddc->local_auxmat2,pcbddc->vec1_C,pcbddc->vec1_R,pcbddc->vec1_R);CHKERRQ(ierr);
    }
    ierr = VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,inout_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,inout_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,inout_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,inout_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* parameter apply transpose determines if the interface preconditioner should be applied transposed or not */
PetscErrorCode  PCBDDCApplyInterfacePreconditioner(PC pc, PetscBool applytranspose)
{
  PetscErrorCode ierr;
  PC_BDDC*        pcbddc = (PC_BDDC*)(pc->data);
  PC_IS*            pcis = (PC_IS*)  (pc->data);
  const PetscScalar zero = 0.0;

  PetscFunctionBegin;
  /* Application of PSI^T or PHI^T (depending on applytranspose, see comment above) */
  if (!pcbddc->benign_apply_coarse_only) {
    if (applytranspose) {
      ierr = MatMultTranspose(pcbddc->coarse_phi_B,pcis->vec1_B,pcbddc->vec1_P);CHKERRQ(ierr);
      if (pcbddc->switch_static) { ierr = MatMultTransposeAdd(pcbddc->coarse_phi_D,pcis->vec1_D,pcbddc->vec1_P,pcbddc->vec1_P);CHKERRQ(ierr); }
    } else {
      ierr = MatMultTranspose(pcbddc->coarse_psi_B,pcis->vec1_B,pcbddc->vec1_P);CHKERRQ(ierr);
      if (pcbddc->switch_static) { ierr = MatMultTransposeAdd(pcbddc->coarse_psi_D,pcis->vec1_D,pcbddc->vec1_P,pcbddc->vec1_P);CHKERRQ(ierr); }
    }
  } else {
    ierr = VecSet(pcbddc->vec1_P,zero);CHKERRQ(ierr);
  }

  /* add p0 to the last value of vec1_P holding the coarse dof relative to p0 */
  if (pcbddc->benign_n) {
    PetscScalar *array;
    PetscInt    j;

    ierr = VecGetArray(pcbddc->vec1_P,&array);CHKERRQ(ierr);
    for (j=0;j<pcbddc->benign_n;j++) array[pcbddc->local_primal_size-pcbddc->benign_n+j] += pcbddc->benign_p0[j];
    ierr = VecRestoreArray(pcbddc->vec1_P,&array);CHKERRQ(ierr);
  }

  /* start communications from local primal nodes to rhs of coarse solver */
  ierr = VecSet(pcbddc->coarse_vec,zero);CHKERRQ(ierr);
  ierr = PCBDDCScatterCoarseDataBegin(pc,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = PCBDDCScatterCoarseDataEnd(pc,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* Coarse solution -> rhs and sol updated inside PCBDDCScattarCoarseDataBegin/End */
  if (pcbddc->coarse_ksp) {
    Mat          coarse_mat;
    Vec          rhs,sol;
    MatNullSpace nullsp;
    PetscBool    isbddc = PETSC_FALSE;

    if (pcbddc->benign_have_null) {
      PC        coarse_pc;

      ierr = KSPGetPC(pcbddc->coarse_ksp,&coarse_pc);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)coarse_pc,PCBDDC,&isbddc);CHKERRQ(ierr);
      /* we need to propagate to coarser levels the need for a possible benign correction */
      if (isbddc && pcbddc->benign_apply_coarse_only && !pcbddc->benign_skip_correction) {
        PC_BDDC* coarsepcbddc = (PC_BDDC*)(coarse_pc->data);
        coarsepcbddc->benign_skip_correction = PETSC_FALSE;
        coarsepcbddc->benign_apply_coarse_only = PETSC_TRUE;
      }
    }
    ierr = KSPGetRhs(pcbddc->coarse_ksp,&rhs);CHKERRQ(ierr);
    ierr = KSPGetSolution(pcbddc->coarse_ksp,&sol);CHKERRQ(ierr);
    ierr = KSPGetOperators(pcbddc->coarse_ksp,&coarse_mat,NULL);CHKERRQ(ierr);
    if (applytranspose) {
      if (pcbddc->benign_apply_coarse_only) SETERRQ(PetscObjectComm((PetscObject)pcbddc->coarse_ksp),PETSC_ERR_SUP,"Not yet implemented");
      ierr = KSPSolveTranspose(pcbddc->coarse_ksp,rhs,sol);CHKERRQ(ierr);
      ierr = KSPCheckSolve(pcbddc->coarse_ksp,pc,sol);CHKERRQ(ierr);
      ierr = MatGetTransposeNullSpace(coarse_mat,&nullsp);CHKERRQ(ierr);
      if (nullsp) {
        ierr = MatNullSpaceRemove(nullsp,sol);CHKERRQ(ierr);
      }
    } else {
      ierr = MatGetNullSpace(coarse_mat,&nullsp);CHKERRQ(ierr);
      if (pcbddc->benign_apply_coarse_only && isbddc) { /* need just to apply the coarse preconditioner during presolve */
        PC        coarse_pc;

        if (nullsp) {
          ierr = MatNullSpaceRemove(nullsp,rhs);CHKERRQ(ierr);
        }
        ierr = KSPGetPC(pcbddc->coarse_ksp,&coarse_pc);CHKERRQ(ierr);
        ierr = PCPreSolve(coarse_pc,pcbddc->coarse_ksp);CHKERRQ(ierr);
        ierr = PCBDDCBenignRemoveInterior(coarse_pc,rhs,sol);CHKERRQ(ierr);
        ierr = PCPostSolve(coarse_pc,pcbddc->coarse_ksp);CHKERRQ(ierr);
      } else {
        ierr = KSPSolve(pcbddc->coarse_ksp,rhs,sol);CHKERRQ(ierr);
        ierr = KSPCheckSolve(pcbddc->coarse_ksp,pc,sol);CHKERRQ(ierr);
        if (nullsp) {
          ierr = MatNullSpaceRemove(nullsp,sol);CHKERRQ(ierr);
        }
      }
    }
    /* we don't need the benign correction at coarser levels anymore */
    if (pcbddc->benign_have_null && isbddc) {
      PC        coarse_pc;
      PC_BDDC*  coarsepcbddc;

      ierr = KSPGetPC(pcbddc->coarse_ksp,&coarse_pc);CHKERRQ(ierr);
      coarsepcbddc = (PC_BDDC*)(coarse_pc->data);
      coarsepcbddc->benign_skip_correction = PETSC_TRUE;
      coarsepcbddc->benign_apply_coarse_only = PETSC_FALSE;
    }
  }

  /* Local solution on R nodes */
  if (pcis->n && !pcbddc->benign_apply_coarse_only) {
    ierr = PCBDDCSolveSubstructureCorrection(pc,pcis->vec1_B,pcis->vec1_D,applytranspose);CHKERRQ(ierr);
  }
  /* communications from coarse sol to local primal nodes */
  ierr = PCBDDCScatterCoarseDataBegin(pc,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = PCBDDCScatterCoarseDataEnd(pc,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  /* Sum contributions from the two levels */
  if (!pcbddc->benign_apply_coarse_only) {
    if (applytranspose) {
      ierr = MatMultAdd(pcbddc->coarse_psi_B,pcbddc->vec1_P,pcis->vec1_B,pcis->vec1_B);CHKERRQ(ierr);
      if (pcbddc->switch_static) { ierr = MatMultAdd(pcbddc->coarse_psi_D,pcbddc->vec1_P,pcis->vec1_D,pcis->vec1_D);CHKERRQ(ierr); }
    } else {
      ierr = MatMultAdd(pcbddc->coarse_phi_B,pcbddc->vec1_P,pcis->vec1_B,pcis->vec1_B);CHKERRQ(ierr);
      if (pcbddc->switch_static) { ierr = MatMultAdd(pcbddc->coarse_phi_D,pcbddc->vec1_P,pcis->vec1_D,pcis->vec1_D);CHKERRQ(ierr); }
    }
    /* store p0 */
    if (pcbddc->benign_n) {
      PetscScalar *array;
      PetscInt    j;

      ierr = VecGetArray(pcbddc->vec1_P,&array);CHKERRQ(ierr);
      for (j=0;j<pcbddc->benign_n;j++) pcbddc->benign_p0[j] = array[pcbddc->local_primal_size-pcbddc->benign_n+j];
      ierr = VecRestoreArray(pcbddc->vec1_P,&array);CHKERRQ(ierr);
    }
  } else { /* expand the coarse solution */
    if (applytranspose) {
      ierr = MatMult(pcbddc->coarse_psi_B,pcbddc->vec1_P,pcis->vec1_B);CHKERRQ(ierr);
    } else {
      ierr = MatMult(pcbddc->coarse_phi_B,pcbddc->vec1_P,pcis->vec1_B);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCScatterCoarseDataBegin(PC pc,InsertMode imode, ScatterMode smode)
{
  PC_BDDC*          pcbddc = (PC_BDDC*)(pc->data);
  Vec               from,to;
  const PetscScalar *array;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (smode == SCATTER_REVERSE) { /* from global to local -> get data from coarse solution */
    from = pcbddc->coarse_vec;
    to = pcbddc->vec1_P;
    if (pcbddc->coarse_ksp) { /* get array from coarse processes */
      Vec tvec;

      ierr = KSPGetRhs(pcbddc->coarse_ksp,&tvec);CHKERRQ(ierr);
      ierr = VecResetArray(tvec);CHKERRQ(ierr);
      ierr = KSPGetSolution(pcbddc->coarse_ksp,&tvec);CHKERRQ(ierr);
      ierr = VecGetArrayRead(tvec,&array);CHKERRQ(ierr);
      ierr = VecPlaceArray(from,array);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(tvec,&array);CHKERRQ(ierr);
    }
  } else { /* from local to global -> put data in coarse right hand side */
    from = pcbddc->vec1_P;
    to = pcbddc->coarse_vec;
  }
  ierr = VecScatterBegin(pcbddc->coarse_loc_to_glob,from,to,imode,smode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCScatterCoarseDataEnd(PC pc, InsertMode imode, ScatterMode smode)
{
  PC_BDDC*          pcbddc = (PC_BDDC*)(pc->data);
  Vec               from,to;
  const PetscScalar *array;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (smode == SCATTER_REVERSE) { /* from global to local -> get data from coarse solution */
    from = pcbddc->coarse_vec;
    to = pcbddc->vec1_P;
  } else { /* from local to global -> put data in coarse right hand side */
    from = pcbddc->vec1_P;
    to = pcbddc->coarse_vec;
  }
  ierr = VecScatterEnd(pcbddc->coarse_loc_to_glob,from,to,imode,smode);CHKERRQ(ierr);
  if (smode == SCATTER_FORWARD) {
    if (pcbddc->coarse_ksp) { /* get array from coarse processes */
      Vec tvec;

      ierr = KSPGetRhs(pcbddc->coarse_ksp,&tvec);CHKERRQ(ierr);
      ierr = VecGetArrayRead(to,&array);CHKERRQ(ierr);
      ierr = VecPlaceArray(tvec,array);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(to,&array);CHKERRQ(ierr);
    }
  } else {
    if (pcbddc->coarse_ksp) { /* restore array of pcbddc->coarse_vec */
     ierr = VecResetArray(from);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCConstraintsSetUp(PC pc)
{
  PetscErrorCode    ierr;
  PC_IS*            pcis = (PC_IS*)(pc->data);
  PC_BDDC*          pcbddc = (PC_BDDC*)pc->data;
  Mat_IS*           matis = (Mat_IS*)pc->pmat->data;
  /* one and zero */
  PetscScalar       one=1.0,zero=0.0;
  /* space to store constraints and their local indices */
  PetscScalar       *constraints_data;
  PetscInt          *constraints_idxs,*constraints_idxs_B;
  PetscInt          *constraints_idxs_ptr,*constraints_data_ptr;
  PetscInt          *constraints_n;
  /* iterators */
  PetscInt          i,j,k,total_counts,total_counts_cc,cum;
  /* BLAS integers */
  PetscBLASInt      lwork,lierr;
  PetscBLASInt      Blas_N,Blas_M,Blas_K,Blas_one=1;
  PetscBLASInt      Blas_LDA,Blas_LDB,Blas_LDC;
  /* reuse */
  PetscInt          olocal_primal_size,olocal_primal_size_cc;
  PetscInt          *olocal_primal_ref_node,*olocal_primal_ref_mult;
  /* change of basis */
  PetscBool         qr_needed;
  PetscBT           change_basis,qr_needed_idx;
  /* auxiliary stuff */
  PetscInt          *nnz,*is_indices;
  PetscInt          ncc;
  /* some quantities */
  PetscInt          n_vertices,total_primal_vertices,valid_constraints;
  PetscInt          size_of_constraint,max_size_of_constraint=0,max_constraints,temp_constraints;
  PetscReal         tol; /* tolerance for retaining eigenmodes */

  PetscFunctionBegin;
  tol  = PetscSqrtReal(PETSC_SMALL);
  /* Destroy Mat objects computed previously */
  ierr = MatDestroy(&pcbddc->ChangeOfBasisMatrix);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->ConstraintMatrix);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->switch_static_change);CHKERRQ(ierr);
  /* save info on constraints from previous setup (if any) */
  olocal_primal_size = pcbddc->local_primal_size;
  olocal_primal_size_cc = pcbddc->local_primal_size_cc;
  ierr = PetscMalloc2(olocal_primal_size_cc,&olocal_primal_ref_node,olocal_primal_size_cc,&olocal_primal_ref_mult);CHKERRQ(ierr);
  ierr = PetscArraycpy(olocal_primal_ref_node,pcbddc->local_primal_ref_node,olocal_primal_size_cc);CHKERRQ(ierr);
  ierr = PetscArraycpy(olocal_primal_ref_mult,pcbddc->local_primal_ref_mult,olocal_primal_size_cc);CHKERRQ(ierr);
  ierr = PetscFree2(pcbddc->local_primal_ref_node,pcbddc->local_primal_ref_mult);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->primal_indices_local_idxs);CHKERRQ(ierr);

  if (!pcbddc->adaptive_selection) {
    IS           ISForVertices,*ISForFaces,*ISForEdges;
    MatNullSpace nearnullsp;
    const Vec    *nearnullvecs;
    Vec          *localnearnullsp;
    PetscScalar  *array;
    PetscInt     n_ISForFaces,n_ISForEdges,nnsp_size;
    PetscBool    nnsp_has_cnst;
    /* LAPACK working arrays for SVD or POD */
    PetscBool    skip_lapack,boolforchange;
    PetscScalar  *work;
    PetscReal    *singular_vals;
#if defined(PETSC_USE_COMPLEX)
    PetscReal    *rwork;
#endif
    PetscScalar  *temp_basis = NULL,*correlation_mat = NULL;
    PetscBLASInt dummy_int=1;
    PetscScalar  dummy_scalar=1.;
    PetscBool    use_pod = PETSC_FALSE;

    /* MKL SVD with same input gives different results on different processes! */
#if defined(PETSC_MISSING_LAPACK_GESVD) || defined(PETSC_HAVE_MKL)
    use_pod = PETSC_TRUE;
#endif
    /* Get index sets for faces, edges and vertices from graph */
    ierr = PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,&n_ISForFaces,&ISForFaces,&n_ISForEdges,&ISForEdges,&ISForVertices);CHKERRQ(ierr);
    /* print some info */
    if (pcbddc->dbg_flag && (!pcbddc->sub_schurs || pcbddc->sub_schurs_rebuild)) {
      PetscInt nv;

      ierr = PCBDDCGraphASCIIView(pcbddc->mat_graph,pcbddc->dbg_flag,pcbddc->dbg_viewer);CHKERRQ(ierr);
      ierr = ISGetSize(ISForVertices,&nv);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"--------------------------------------------------------------\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d got %02d local candidate vertices (%D)\n",PetscGlobalRank,nv,pcbddc->use_vertices);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d got %02d local candidate edges    (%D)\n",PetscGlobalRank,n_ISForEdges,pcbddc->use_edges);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d got %02d local candidate faces    (%D)\n",PetscGlobalRank,n_ISForFaces,pcbddc->use_faces);CHKERRQ(ierr);
      ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(pcbddc->dbg_viewer);CHKERRQ(ierr);
    }

    /* free unneeded index sets */
    if (!pcbddc->use_vertices) {
      ierr = ISDestroy(&ISForVertices);CHKERRQ(ierr);
    }
    if (!pcbddc->use_edges) {
      for (i=0;i<n_ISForEdges;i++) {
        ierr = ISDestroy(&ISForEdges[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(ISForEdges);CHKERRQ(ierr);
      n_ISForEdges = 0;
    }
    if (!pcbddc->use_faces) {
      for (i=0;i<n_ISForFaces;i++) {
        ierr = ISDestroy(&ISForFaces[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(ISForFaces);CHKERRQ(ierr);
      n_ISForFaces = 0;
    }

    /* check if near null space is attached to global mat */
    if (pcbddc->use_nnsp) {
      ierr = MatGetNearNullSpace(pc->pmat,&nearnullsp);CHKERRQ(ierr);
    } else nearnullsp = NULL;

    if (nearnullsp) {
      ierr = MatNullSpaceGetVecs(nearnullsp,&nnsp_has_cnst,&nnsp_size,&nearnullvecs);CHKERRQ(ierr);
      /* remove any stored info */
      ierr = MatNullSpaceDestroy(&pcbddc->onearnullspace);CHKERRQ(ierr);
      ierr = PetscFree(pcbddc->onearnullvecs_state);CHKERRQ(ierr);
      /* store information for BDDC solver reuse */
      ierr = PetscObjectReference((PetscObject)nearnullsp);CHKERRQ(ierr);
      pcbddc->onearnullspace = nearnullsp;
      ierr = PetscMalloc1(nnsp_size,&pcbddc->onearnullvecs_state);CHKERRQ(ierr);
      for (i=0;i<nnsp_size;i++) {
        ierr = PetscObjectStateGet((PetscObject)nearnullvecs[i],&pcbddc->onearnullvecs_state[i]);CHKERRQ(ierr);
      }
    } else { /* if near null space is not provided BDDC uses constants by default */
      nnsp_size = 0;
      nnsp_has_cnst = PETSC_TRUE;
    }
    /* get max number of constraints on a single cc */
    max_constraints = nnsp_size;
    if (nnsp_has_cnst) max_constraints++;

    /*
         Evaluate maximum storage size needed by the procedure
         - Indices for connected component i stored at "constraints_idxs + constraints_idxs_ptr[i]"
         - Values for constraints on connected component i stored at "constraints_data + constraints_data_ptr[i]"
         There can be multiple constraints per connected component
                                                                                                                                                           */
    n_vertices = 0;
    if (ISForVertices) {
      ierr = ISGetSize(ISForVertices,&n_vertices);CHKERRQ(ierr);
    }
    ncc = n_vertices+n_ISForFaces+n_ISForEdges;
    ierr = PetscMalloc3(ncc+1,&constraints_idxs_ptr,ncc+1,&constraints_data_ptr,ncc,&constraints_n);CHKERRQ(ierr);

    total_counts = n_ISForFaces+n_ISForEdges;
    total_counts *= max_constraints;
    total_counts += n_vertices;
    ierr = PetscBTCreate(total_counts,&change_basis);CHKERRQ(ierr);

    total_counts = 0;
    max_size_of_constraint = 0;
    for (i=0;i<n_ISForEdges+n_ISForFaces;i++) {
      IS used_is;
      if (i<n_ISForEdges) {
        used_is = ISForEdges[i];
      } else {
        used_is = ISForFaces[i-n_ISForEdges];
      }
      ierr = ISGetSize(used_is,&j);CHKERRQ(ierr);
      total_counts += j;
      max_size_of_constraint = PetscMax(j,max_size_of_constraint);
    }
    ierr = PetscMalloc3(total_counts*max_constraints+n_vertices,&constraints_data,total_counts+n_vertices,&constraints_idxs,total_counts+n_vertices,&constraints_idxs_B);CHKERRQ(ierr);

    /* get local part of global near null space vectors */
    ierr = PetscMalloc1(nnsp_size,&localnearnullsp);CHKERRQ(ierr);
    for (k=0;k<nnsp_size;k++) {
      ierr = VecDuplicate(pcis->vec1_N,&localnearnullsp[k]);CHKERRQ(ierr);
      ierr = VecScatterBegin(matis->rctx,nearnullvecs[k],localnearnullsp[k],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(matis->rctx,nearnullvecs[k],localnearnullsp[k],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    }

    /* whether or not to skip lapack calls */
    skip_lapack = PETSC_TRUE;
    if (n_ISForFaces+n_ISForEdges && max_constraints > 1 && !pcbddc->use_nnsp_true) skip_lapack = PETSC_FALSE;

    /* First we issue queries to allocate optimal workspace for LAPACKgesvd (or LAPACKsyev if SVD is missing) */
    if (!skip_lapack) {
      PetscScalar temp_work;

      if (use_pod) {
        /* Proper Orthogonal Decomposition (POD) using the snapshot method */
        ierr = PetscMalloc1(max_constraints*max_constraints,&correlation_mat);CHKERRQ(ierr);
        ierr = PetscMalloc1(max_constraints,&singular_vals);CHKERRQ(ierr);
        ierr = PetscMalloc1(max_size_of_constraint*max_constraints,&temp_basis);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        ierr = PetscMalloc1(3*max_constraints,&rwork);CHKERRQ(ierr);
#endif
        /* now we evaluate the optimal workspace using query with lwork=-1 */
        ierr = PetscBLASIntCast(max_constraints,&Blas_N);CHKERRQ(ierr);
        ierr = PetscBLASIntCast(max_constraints,&Blas_LDA);CHKERRQ(ierr);
        lwork = -1;
        ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
        PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","U",&Blas_N,correlation_mat,&Blas_LDA,singular_vals,&temp_work,&lwork,&lierr));
#else
        PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","U",&Blas_N,correlation_mat,&Blas_LDA,singular_vals,&temp_work,&lwork,rwork,&lierr));
#endif
        ierr = PetscFPTrapPop();CHKERRQ(ierr);
        if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to SYEV Lapack routine %d",(int)lierr);
      } else {
#if !defined(PETSC_MISSING_LAPACK_GESVD)
        /* SVD */
        PetscInt max_n,min_n;
        max_n = max_size_of_constraint;
        min_n = max_constraints;
        if (max_size_of_constraint < max_constraints) {
          min_n = max_size_of_constraint;
          max_n = max_constraints;
        }
        ierr = PetscMalloc1(min_n,&singular_vals);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        ierr = PetscMalloc1(5*min_n,&rwork);CHKERRQ(ierr);
#endif
        /* now we evaluate the optimal workspace using query with lwork=-1 */
        lwork = -1;
        ierr = PetscBLASIntCast(max_n,&Blas_M);CHKERRQ(ierr);
        ierr = PetscBLASIntCast(min_n,&Blas_N);CHKERRQ(ierr);
        ierr = PetscBLASIntCast(max_n,&Blas_LDA);CHKERRQ(ierr);
        ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
        PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&Blas_M,&Blas_N,&constraints_data[0],&Blas_LDA,singular_vals,&dummy_scalar,&dummy_int,&dummy_scalar,&dummy_int,&temp_work,&lwork,&lierr));
#else
        PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&Blas_M,&Blas_N,&constraints_data[0],&Blas_LDA,singular_vals,&dummy_scalar,&dummy_int,&dummy_scalar,&dummy_int,&temp_work,&lwork,rwork,&lierr));
#endif
        ierr = PetscFPTrapPop();CHKERRQ(ierr);
        if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to GESVD Lapack routine %d",(int)lierr);
#else
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"This should not happen");
#endif /* on missing GESVD */
      }
      /* Allocate optimal workspace */
      ierr = PetscBLASIntCast((PetscInt)PetscRealPart(temp_work),&lwork);CHKERRQ(ierr);
      ierr = PetscMalloc1(lwork,&work);CHKERRQ(ierr);
    }
    /* Now we can loop on constraining sets */
    total_counts = 0;
    constraints_idxs_ptr[0] = 0;
    constraints_data_ptr[0] = 0;
    /* vertices */
    if (n_vertices) {
      ierr = ISGetIndices(ISForVertices,(const PetscInt**)&is_indices);CHKERRQ(ierr);
      ierr = PetscArraycpy(constraints_idxs,is_indices,n_vertices);CHKERRQ(ierr);
      for (i=0;i<n_vertices;i++) {
        constraints_n[total_counts] = 1;
        constraints_data[total_counts] = 1.0;
        constraints_idxs_ptr[total_counts+1] = constraints_idxs_ptr[total_counts]+1;
        constraints_data_ptr[total_counts+1] = constraints_data_ptr[total_counts]+1;
        total_counts++;
      }
      ierr = ISRestoreIndices(ISForVertices,(const PetscInt**)&is_indices);CHKERRQ(ierr);
      n_vertices = total_counts;
    }

    /* edges and faces */
    total_counts_cc = total_counts;
    for (ncc=0;ncc<n_ISForEdges+n_ISForFaces;ncc++) {
      IS        used_is;
      PetscBool idxs_copied = PETSC_FALSE;

      if (ncc<n_ISForEdges) {
        used_is = ISForEdges[ncc];
        boolforchange = pcbddc->use_change_of_basis; /* change or not the basis on the edge */
      } else {
        used_is = ISForFaces[ncc-n_ISForEdges];
        boolforchange = (PetscBool)(pcbddc->use_change_of_basis && pcbddc->use_change_on_faces); /* change or not the basis on the face */
      }
      temp_constraints = 0;          /* zero the number of constraints I have on this conn comp */

      ierr = ISGetSize(used_is,&size_of_constraint);CHKERRQ(ierr);
      ierr = ISGetIndices(used_is,(const PetscInt**)&is_indices);CHKERRQ(ierr);
      /* change of basis should not be performed on local periodic nodes */
      if (pcbddc->mat_graph->mirrors && pcbddc->mat_graph->mirrors[is_indices[0]]) boolforchange = PETSC_FALSE;
      if (nnsp_has_cnst) {
        PetscScalar quad_value;

        ierr = PetscArraycpy(constraints_idxs + constraints_idxs_ptr[total_counts_cc],is_indices,size_of_constraint);CHKERRQ(ierr);
        idxs_copied = PETSC_TRUE;

        if (!pcbddc->use_nnsp_true) {
          quad_value = (PetscScalar)(1.0/PetscSqrtReal((PetscReal)size_of_constraint));
        } else {
          quad_value = 1.0;
        }
        for (j=0;j<size_of_constraint;j++) {
          constraints_data[constraints_data_ptr[total_counts_cc]+j] = quad_value;
        }
        temp_constraints++;
        total_counts++;
      }
      for (k=0;k<nnsp_size;k++) {
        PetscReal real_value;
        PetscScalar *ptr_to_data;

        ierr = VecGetArrayRead(localnearnullsp[k],(const PetscScalar**)&array);CHKERRQ(ierr);
        ptr_to_data = &constraints_data[constraints_data_ptr[total_counts_cc]+temp_constraints*size_of_constraint];
        for (j=0;j<size_of_constraint;j++) {
          ptr_to_data[j] = array[is_indices[j]];
        }
        ierr = VecRestoreArrayRead(localnearnullsp[k],(const PetscScalar**)&array);CHKERRQ(ierr);
        /* check if array is null on the connected component */
        ierr = PetscBLASIntCast(size_of_constraint,&Blas_N);CHKERRQ(ierr);
        PetscStackCallBLAS("BLASasum",real_value = BLASasum_(&Blas_N,ptr_to_data,&Blas_one));
        if (real_value > tol*size_of_constraint) { /* keep indices and values */
          temp_constraints++;
          total_counts++;
          if (!idxs_copied) {
            ierr = PetscArraycpy(constraints_idxs + constraints_idxs_ptr[total_counts_cc],is_indices,size_of_constraint);CHKERRQ(ierr);
            idxs_copied = PETSC_TRUE;
          }
        }
      }
      ierr = ISRestoreIndices(used_is,(const PetscInt**)&is_indices);CHKERRQ(ierr);
      valid_constraints = temp_constraints;
      if (!pcbddc->use_nnsp_true && temp_constraints) {
        if (temp_constraints == 1) { /* just normalize the constraint */
          PetscScalar norm,*ptr_to_data;

          ptr_to_data = &constraints_data[constraints_data_ptr[total_counts_cc]];
          ierr = PetscBLASIntCast(size_of_constraint,&Blas_N);CHKERRQ(ierr);
          PetscStackCallBLAS("BLASdot",norm = BLASdot_(&Blas_N,ptr_to_data,&Blas_one,ptr_to_data,&Blas_one));
          norm = 1.0/PetscSqrtReal(PetscRealPart(norm));
          PetscStackCallBLAS("BLASscal",BLASscal_(&Blas_N,&norm,ptr_to_data,&Blas_one));
        } else { /* perform SVD */
          PetscScalar *ptr_to_data = &constraints_data[constraints_data_ptr[total_counts_cc]];

          if (use_pod) {
            /* SVD: Y = U*S*V^H                -> U (eigenvectors of Y*Y^H) = Y*V*(S)^\dag
               POD: Y^H*Y = V*D*V^H, D = S^H*S -> U = Y*V*D^(-1/2)
               -> When PETSC_USE_COMPLEX and PETSC_MISSING_LAPACK_GESVD are defined
                  the constraints basis will differ (by a complex factor with absolute value equal to 1)
                  from that computed using LAPACKgesvd
               -> This is due to a different computation of eigenvectors in LAPACKheev
               -> The quality of the POD-computed basis will be the same */
            ierr = PetscArrayzero(correlation_mat,temp_constraints*temp_constraints);CHKERRQ(ierr);
            /* Store upper triangular part of correlation matrix */
            ierr = PetscBLASIntCast(size_of_constraint,&Blas_N);CHKERRQ(ierr);
            ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
            for (j=0;j<temp_constraints;j++) {
              for (k=0;k<j+1;k++) {
                PetscStackCallBLAS("BLASdot",correlation_mat[j*temp_constraints+k] = BLASdot_(&Blas_N,ptr_to_data+k*size_of_constraint,&Blas_one,ptr_to_data+j*size_of_constraint,&Blas_one));
              }
            }
            /* compute eigenvalues and eigenvectors of correlation matrix */
            ierr = PetscBLASIntCast(temp_constraints,&Blas_N);CHKERRQ(ierr);
            ierr = PetscBLASIntCast(temp_constraints,&Blas_LDA);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
            PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","U",&Blas_N,correlation_mat,&Blas_LDA,singular_vals,work,&lwork,&lierr));
#else
            PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","U",&Blas_N,correlation_mat,&Blas_LDA,singular_vals,work,&lwork,rwork,&lierr));
#endif
            ierr = PetscFPTrapPop();CHKERRQ(ierr);
            if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYEV Lapack routine %d",(int)lierr);
            /* retain eigenvalues greater than tol: note that LAPACKsyev gives eigs in ascending order */
            j = 0;
            while (j < temp_constraints && singular_vals[j]/singular_vals[temp_constraints-1] < tol) j++;
            total_counts = total_counts-j;
            valid_constraints = temp_constraints-j;
            /* scale and copy POD basis into used quadrature memory */
            ierr = PetscBLASIntCast(size_of_constraint,&Blas_M);CHKERRQ(ierr);
            ierr = PetscBLASIntCast(temp_constraints,&Blas_N);CHKERRQ(ierr);
            ierr = PetscBLASIntCast(temp_constraints,&Blas_K);CHKERRQ(ierr);
            ierr = PetscBLASIntCast(size_of_constraint,&Blas_LDA);CHKERRQ(ierr);
            ierr = PetscBLASIntCast(temp_constraints,&Blas_LDB);CHKERRQ(ierr);
            ierr = PetscBLASIntCast(size_of_constraint,&Blas_LDC);CHKERRQ(ierr);
            if (j<temp_constraints) {
              PetscInt ii;
              for (k=j;k<temp_constraints;k++) singular_vals[k] = 1.0/PetscSqrtReal(singular_vals[k]);
              ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
              PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&Blas_M,&Blas_N,&Blas_K,&one,ptr_to_data,&Blas_LDA,correlation_mat,&Blas_LDB,&zero,temp_basis,&Blas_LDC));
              ierr = PetscFPTrapPop();CHKERRQ(ierr);
              for (k=0;k<temp_constraints-j;k++) {
                for (ii=0;ii<size_of_constraint;ii++) {
                  ptr_to_data[k*size_of_constraint+ii] = singular_vals[temp_constraints-1-k]*temp_basis[(temp_constraints-1-k)*size_of_constraint+ii];
                }
              }
            }
          } else {
#if !defined(PETSC_MISSING_LAPACK_GESVD)
            ierr = PetscBLASIntCast(size_of_constraint,&Blas_M);CHKERRQ(ierr);
            ierr = PetscBLASIntCast(temp_constraints,&Blas_N);CHKERRQ(ierr);
            ierr = PetscBLASIntCast(size_of_constraint,&Blas_LDA);CHKERRQ(ierr);
            ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
            PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&Blas_M,&Blas_N,ptr_to_data,&Blas_LDA,singular_vals,&dummy_scalar,&dummy_int,&dummy_scalar,&dummy_int,work,&lwork,&lierr));
#else
            PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&Blas_M,&Blas_N,ptr_to_data,&Blas_LDA,singular_vals,&dummy_scalar,&dummy_int,&dummy_scalar,&dummy_int,work,&lwork,rwork,&lierr));
#endif
            if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GESVD Lapack routine %d",(int)lierr);
            ierr = PetscFPTrapPop();CHKERRQ(ierr);
            /* retain eigenvalues greater than tol: note that LAPACKgesvd gives eigs in descending order */
            k = temp_constraints;
            if (k > size_of_constraint) k = size_of_constraint;
            j = 0;
            while (j < k && singular_vals[k-j-1]/singular_vals[0] < tol) j++;
            valid_constraints = k-j;
            total_counts = total_counts-temp_constraints+valid_constraints;
#else
            SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"This should not happen");
#endif /* on missing GESVD */
          }
        }
      }
      /* update pointers information */
      if (valid_constraints) {
        constraints_n[total_counts_cc] = valid_constraints;
        constraints_idxs_ptr[total_counts_cc+1] = constraints_idxs_ptr[total_counts_cc]+size_of_constraint;
        constraints_data_ptr[total_counts_cc+1] = constraints_data_ptr[total_counts_cc]+size_of_constraint*valid_constraints;
        /* set change_of_basis flag */
        if (boolforchange) {
          PetscBTSet(change_basis,total_counts_cc);
        }
        total_counts_cc++;
      }
    }
    /* free workspace */
    if (!skip_lapack) {
      ierr = PetscFree(work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscFree(rwork);CHKERRQ(ierr);
#endif
      ierr = PetscFree(singular_vals);CHKERRQ(ierr);
      ierr = PetscFree(correlation_mat);CHKERRQ(ierr);
      ierr = PetscFree(temp_basis);CHKERRQ(ierr);
    }
    for (k=0;k<nnsp_size;k++) {
      ierr = VecDestroy(&localnearnullsp[k]);CHKERRQ(ierr);
    }
    ierr = PetscFree(localnearnullsp);CHKERRQ(ierr);
    /* free index sets of faces, edges and vertices */
    for (i=0;i<n_ISForFaces;i++) {
      ierr = ISDestroy(&ISForFaces[i]);CHKERRQ(ierr);
    }
    if (n_ISForFaces) {
      ierr = PetscFree(ISForFaces);CHKERRQ(ierr);
    }
    for (i=0;i<n_ISForEdges;i++) {
      ierr = ISDestroy(&ISForEdges[i]);CHKERRQ(ierr);
    }
    if (n_ISForEdges) {
      ierr = PetscFree(ISForEdges);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&ISForVertices);CHKERRQ(ierr);
  } else {
    PCBDDCSubSchurs sub_schurs = pcbddc->sub_schurs;

    total_counts = 0;
    n_vertices = 0;
    if (sub_schurs->is_vertices && pcbddc->use_vertices) {
      ierr = ISGetLocalSize(sub_schurs->is_vertices,&n_vertices);CHKERRQ(ierr);
    }
    max_constraints = 0;
    total_counts_cc = 0;
    for (i=0;i<sub_schurs->n_subs+n_vertices;i++) {
      total_counts += pcbddc->adaptive_constraints_n[i];
      if (pcbddc->adaptive_constraints_n[i]) total_counts_cc++;
      max_constraints = PetscMax(max_constraints,pcbddc->adaptive_constraints_n[i]);
    }
    constraints_idxs_ptr = pcbddc->adaptive_constraints_idxs_ptr;
    constraints_data_ptr = pcbddc->adaptive_constraints_data_ptr;
    constraints_idxs = pcbddc->adaptive_constraints_idxs;
    constraints_data = pcbddc->adaptive_constraints_data;
    /* constraints_n differs from pcbddc->adaptive_constraints_n */
    ierr = PetscMalloc1(total_counts_cc,&constraints_n);CHKERRQ(ierr);
    total_counts_cc = 0;
    for (i=0;i<sub_schurs->n_subs+n_vertices;i++) {
      if (pcbddc->adaptive_constraints_n[i]) {
        constraints_n[total_counts_cc++] = pcbddc->adaptive_constraints_n[i];
      }
    }

    max_size_of_constraint = 0;
    for (i=0;i<total_counts_cc;i++) max_size_of_constraint = PetscMax(max_size_of_constraint,constraints_idxs_ptr[i+1]-constraints_idxs_ptr[i]);
    ierr = PetscMalloc1(constraints_idxs_ptr[total_counts_cc],&constraints_idxs_B);CHKERRQ(ierr);
    /* Change of basis */
    ierr = PetscBTCreate(total_counts_cc,&change_basis);CHKERRQ(ierr);
    if (pcbddc->use_change_of_basis) {
      for (i=0;i<sub_schurs->n_subs;i++) {
        if (PetscBTLookup(sub_schurs->is_edge,i) || pcbddc->use_change_on_faces) {
          ierr = PetscBTSet(change_basis,i+n_vertices);CHKERRQ(ierr);
        }
      }
    }
  }
  pcbddc->local_primal_size = total_counts;
  ierr = PetscMalloc1(pcbddc->local_primal_size+pcbddc->benign_n,&pcbddc->primal_indices_local_idxs);CHKERRQ(ierr);

  /* map constraints_idxs in boundary numbering */
  ierr = ISGlobalToLocalMappingApply(pcis->BtoNmap,IS_GTOLM_DROP,constraints_idxs_ptr[total_counts_cc],constraints_idxs,&i,constraints_idxs_B);CHKERRQ(ierr);
  if (i != constraints_idxs_ptr[total_counts_cc]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in boundary numbering for constraints indices %D != %D",constraints_idxs_ptr[total_counts_cc],i);

  /* Create constraint matrix */
  ierr = MatCreate(PETSC_COMM_SELF,&pcbddc->ConstraintMatrix);CHKERRQ(ierr);
  ierr = MatSetType(pcbddc->ConstraintMatrix,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(pcbddc->ConstraintMatrix,pcbddc->local_primal_size,pcis->n,pcbddc->local_primal_size,pcis->n);CHKERRQ(ierr);

  /* find primal_dofs: subdomain corners plus dofs selected as primal after change of basis */
  /* determine if a QR strategy is needed for change of basis */
  qr_needed = pcbddc->use_qr_single;
  ierr = PetscBTCreate(total_counts_cc,&qr_needed_idx);CHKERRQ(ierr);
  total_primal_vertices=0;
  pcbddc->local_primal_size_cc = 0;
  for (i=0;i<total_counts_cc;i++) {
    size_of_constraint = constraints_idxs_ptr[i+1]-constraints_idxs_ptr[i];
    if (size_of_constraint == 1 && pcbddc->mat_graph->custom_minimal_size) {
      pcbddc->primal_indices_local_idxs[total_primal_vertices++] = constraints_idxs[constraints_idxs_ptr[i]];
      pcbddc->local_primal_size_cc += 1;
    } else if (PetscBTLookup(change_basis,i)) {
      for (k=0;k<constraints_n[i];k++) {
        pcbddc->primal_indices_local_idxs[total_primal_vertices++] = constraints_idxs[constraints_idxs_ptr[i]+k];
      }
      pcbddc->local_primal_size_cc += constraints_n[i];
      if (constraints_n[i] > 1 || pcbddc->use_qr_single) {
        PetscBTSet(qr_needed_idx,i);
        qr_needed = PETSC_TRUE;
      }
    } else {
      pcbddc->local_primal_size_cc += 1;
    }
  }
  /* note that the local variable n_vertices used below stores the number of pointwise constraints */
  pcbddc->n_vertices = total_primal_vertices;
  /* permute indices in order to have a sorted set of vertices */
  ierr = PetscSortInt(total_primal_vertices,pcbddc->primal_indices_local_idxs);CHKERRQ(ierr);
  ierr = PetscMalloc2(pcbddc->local_primal_size_cc+pcbddc->benign_n,&pcbddc->local_primal_ref_node,pcbddc->local_primal_size_cc+pcbddc->benign_n,&pcbddc->local_primal_ref_mult);CHKERRQ(ierr);
  ierr = PetscArraycpy(pcbddc->local_primal_ref_node,pcbddc->primal_indices_local_idxs,total_primal_vertices);CHKERRQ(ierr);
  for (i=0;i<total_primal_vertices;i++) pcbddc->local_primal_ref_mult[i] = 1;

  /* nonzero structure of constraint matrix */
  /* and get reference dof for local constraints */
  ierr = PetscMalloc1(pcbddc->local_primal_size,&nnz);CHKERRQ(ierr);
  for (i=0;i<total_primal_vertices;i++) nnz[i] = 1;

  j = total_primal_vertices;
  total_counts = total_primal_vertices;
  cum = total_primal_vertices;
  for (i=n_vertices;i<total_counts_cc;i++) {
    if (!PetscBTLookup(change_basis,i)) {
      pcbddc->local_primal_ref_node[cum] = constraints_idxs[constraints_idxs_ptr[i]];
      pcbddc->local_primal_ref_mult[cum] = constraints_n[i];
      cum++;
      size_of_constraint = constraints_idxs_ptr[i+1]-constraints_idxs_ptr[i];
      for (k=0;k<constraints_n[i];k++) {
        pcbddc->primal_indices_local_idxs[total_counts++] = constraints_idxs[constraints_idxs_ptr[i]+k];
        nnz[j+k] = size_of_constraint;
      }
      j += constraints_n[i];
    }
  }
  ierr = MatSeqAIJSetPreallocation(pcbddc->ConstraintMatrix,0,nnz);CHKERRQ(ierr);
  ierr = MatSetOption(pcbddc->ConstraintMatrix,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);

  /* set values in constraint matrix */
  for (i=0;i<total_primal_vertices;i++) {
    ierr = MatSetValue(pcbddc->ConstraintMatrix,i,pcbddc->local_primal_ref_node[i],1.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  total_counts = total_primal_vertices;
  for (i=n_vertices;i<total_counts_cc;i++) {
    if (!PetscBTLookup(change_basis,i)) {
      PetscInt *cols;

      size_of_constraint = constraints_idxs_ptr[i+1]-constraints_idxs_ptr[i];
      cols = constraints_idxs+constraints_idxs_ptr[i];
      for (k=0;k<constraints_n[i];k++) {
        PetscInt    row = total_counts+k;
        PetscScalar *vals;

        vals = constraints_data+constraints_data_ptr[i]+k*size_of_constraint;
        ierr = MatSetValues(pcbddc->ConstraintMatrix,1,&row,size_of_constraint,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
      }
      total_counts += constraints_n[i];
    }
  }
  /* assembling */
  ierr = MatAssemblyBegin(pcbddc->ConstraintMatrix,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(pcbddc->ConstraintMatrix,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatViewFromOptions(pcbddc->ConstraintMatrix,(PetscObject)pc,"-pc_bddc_constraint_mat_view");CHKERRQ(ierr);

  /* Create matrix for change of basis. We don't need it in case pcbddc->use_change_of_basis is FALSE */
  if (pcbddc->use_change_of_basis) {
    /* dual and primal dofs on a single cc */
    PetscInt     dual_dofs,primal_dofs;
    /* working stuff for GEQRF */
    PetscScalar  *qr_basis = NULL,*qr_tau = NULL,*qr_work = NULL,lqr_work_t;
    PetscBLASInt lqr_work;
    /* working stuff for UNGQR */
    PetscScalar  *gqr_work = NULL,lgqr_work_t;
    PetscBLASInt lgqr_work;
    /* working stuff for TRTRS */
    PetscScalar  *trs_rhs = NULL;
    PetscBLASInt Blas_NRHS;
    /* pointers for values insertion into change of basis matrix */
    PetscInt     *start_rows,*start_cols;
    PetscScalar  *start_vals;
    /* working stuff for values insertion */
    PetscBT      is_primal;
    PetscInt     *aux_primal_numbering_B;
    /* matrix sizes */
    PetscInt     global_size,local_size;
    /* temporary change of basis */
    Mat          localChangeOfBasisMatrix;
    /* extra space for debugging */
    PetscScalar  *dbg_work = NULL;

    /* local temporary change of basis acts on local interfaces -> dimension is n_B x n_B */
    ierr = MatCreate(PETSC_COMM_SELF,&localChangeOfBasisMatrix);CHKERRQ(ierr);
    ierr = MatSetType(localChangeOfBasisMatrix,MATAIJ);CHKERRQ(ierr);
    ierr = MatSetSizes(localChangeOfBasisMatrix,pcis->n,pcis->n,pcis->n,pcis->n);CHKERRQ(ierr);
    /* nonzeros for local mat */
    ierr = PetscMalloc1(pcis->n,&nnz);CHKERRQ(ierr);
    if (!pcbddc->benign_change || pcbddc->fake_change) {
      for (i=0;i<pcis->n;i++) nnz[i]=1;
    } else {
      const PetscInt *ii;
      PetscInt       n;
      PetscBool      flg_row;
      ierr = MatGetRowIJ(pcbddc->benign_change,0,PETSC_FALSE,PETSC_FALSE,&n,&ii,NULL,&flg_row);CHKERRQ(ierr);
      for (i=0;i<n;i++) nnz[i] = ii[i+1]-ii[i];
      ierr = MatRestoreRowIJ(pcbddc->benign_change,0,PETSC_FALSE,PETSC_FALSE,&n,&ii,NULL,&flg_row);CHKERRQ(ierr);
    }
    for (i=n_vertices;i<total_counts_cc;i++) {
      if (PetscBTLookup(change_basis,i)) {
        size_of_constraint = constraints_idxs_ptr[i+1]-constraints_idxs_ptr[i];
        if (PetscBTLookup(qr_needed_idx,i)) {
          for (j=0;j<size_of_constraint;j++) nnz[constraints_idxs[constraints_idxs_ptr[i]+j]] = size_of_constraint;
        } else {
          nnz[constraints_idxs[constraints_idxs_ptr[i]]] = size_of_constraint;
          for (j=1;j<size_of_constraint;j++) nnz[constraints_idxs[constraints_idxs_ptr[i]+j]] = 2;
        }
      }
    }
    ierr = MatSeqAIJSetPreallocation(localChangeOfBasisMatrix,0,nnz);CHKERRQ(ierr);
    ierr = MatSetOption(localChangeOfBasisMatrix,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscFree(nnz);CHKERRQ(ierr);
    /* Set interior change in the matrix */
    if (!pcbddc->benign_change || pcbddc->fake_change) {
      for (i=0;i<pcis->n;i++) {
        ierr = MatSetValue(localChangeOfBasisMatrix,i,i,1.0,INSERT_VALUES);CHKERRQ(ierr);
      }
    } else {
      const PetscInt *ii,*jj;
      PetscScalar    *aa;
      PetscInt       n;
      PetscBool      flg_row;
      ierr = MatGetRowIJ(pcbddc->benign_change,0,PETSC_FALSE,PETSC_FALSE,&n,&ii,&jj,&flg_row);CHKERRQ(ierr);
      ierr = MatSeqAIJGetArray(pcbddc->benign_change,&aa);CHKERRQ(ierr);
      for (i=0;i<n;i++) {
        ierr = MatSetValues(localChangeOfBasisMatrix,1,&i,ii[i+1]-ii[i],jj+ii[i],aa+ii[i],INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatSeqAIJRestoreArray(pcbddc->benign_change,&aa);CHKERRQ(ierr);
      ierr = MatRestoreRowIJ(pcbddc->benign_change,0,PETSC_FALSE,PETSC_FALSE,&n,&ii,&jj,&flg_row);CHKERRQ(ierr);
    }

    if (pcbddc->dbg_flag) {
      ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"--------------------------------------------------------------\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Checking change of basis computation for subdomain %04d\n",PetscGlobalRank);CHKERRQ(ierr);
    }


    /* Now we loop on the constraints which need a change of basis */
    /*
       Change of basis matrix is evaluated similarly to the FIRST APPROACH in
       Klawonn and Widlund, Dual-primal FETI-DP methods for linear elasticity, (see Sect 6.2.1)

       Basic blocks of change of basis matrix T computed by

          - Using the following block transformation if there is only a primal dof on the cc (and -pc_bddc_use_qr_single is not specified)

            | 1        0   ...        0         s_1/S |
            | 0        1   ...        0         s_2/S |
            |              ...                        |
            | 0        ...            1     s_{n-1}/S |
            | -s_1/s_n ...    -s_{n-1}/s_n      s_n/S |

            with S = \sum_{i=1}^n s_i^2
            NOTE: in the above example, the primal dof is the last one of the edge in LOCAL ordering
                  in the current implementation, the primal dof is the first one of the edge in GLOBAL ordering

          - QR decomposition of constraints otherwise
    */
    if (qr_needed && max_size_of_constraint) {
      /* space to store Q */
      ierr = PetscMalloc1(max_size_of_constraint*max_size_of_constraint,&qr_basis);CHKERRQ(ierr);
      /* array to store scaling factors for reflectors */
      ierr = PetscMalloc1(max_constraints,&qr_tau);CHKERRQ(ierr);
      /* first we issue queries for optimal work */
      ierr = PetscBLASIntCast(max_size_of_constraint,&Blas_M);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(max_constraints,&Blas_N);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(max_size_of_constraint,&Blas_LDA);CHKERRQ(ierr);
      lqr_work = -1;
      PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&Blas_M,&Blas_N,qr_basis,&Blas_LDA,qr_tau,&lqr_work_t,&lqr_work,&lierr));
      if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to GEQRF Lapack routine %d",(int)lierr);
      ierr = PetscBLASIntCast((PetscInt)PetscRealPart(lqr_work_t),&lqr_work);CHKERRQ(ierr);
      ierr = PetscMalloc1((PetscInt)PetscRealPart(lqr_work_t),&qr_work);CHKERRQ(ierr);
      lgqr_work = -1;
      ierr = PetscBLASIntCast(max_size_of_constraint,&Blas_M);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(max_size_of_constraint,&Blas_N);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(max_constraints,&Blas_K);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(max_size_of_constraint,&Blas_LDA);CHKERRQ(ierr);
      if (Blas_K>Blas_M) Blas_K=Blas_M; /* adjust just for computing optimal work */
      PetscStackCallBLAS("LAPACKorgqr",LAPACKorgqr_(&Blas_M,&Blas_N,&Blas_K,qr_basis,&Blas_LDA,qr_tau,&lgqr_work_t,&lgqr_work,&lierr));
      if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to ORGQR/UNGQR Lapack routine %d",(int)lierr);
      ierr = PetscBLASIntCast((PetscInt)PetscRealPart(lgqr_work_t),&lgqr_work);CHKERRQ(ierr);
      ierr = PetscMalloc1((PetscInt)PetscRealPart(lgqr_work_t),&gqr_work);CHKERRQ(ierr);
      /* array to store rhs and solution of triangular solver */
      ierr = PetscMalloc1(max_constraints*max_constraints,&trs_rhs);CHKERRQ(ierr);
      /* allocating workspace for check */
      if (pcbddc->dbg_flag) {
        ierr = PetscMalloc1(max_size_of_constraint*(max_constraints+max_size_of_constraint),&dbg_work);CHKERRQ(ierr);
      }
    }
    /* array to store whether a node is primal or not */
    ierr = PetscBTCreate(pcis->n_B,&is_primal);CHKERRQ(ierr);
    ierr = PetscMalloc1(total_primal_vertices,&aux_primal_numbering_B);CHKERRQ(ierr);
    ierr = ISGlobalToLocalMappingApply(pcis->BtoNmap,IS_GTOLM_DROP,total_primal_vertices,pcbddc->local_primal_ref_node,&i,aux_primal_numbering_B);CHKERRQ(ierr);
    if (i != total_primal_vertices) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in boundary numbering for BDDC vertices! %D != %D",total_primal_vertices,i);
    for (i=0;i<total_primal_vertices;i++) {
      ierr = PetscBTSet(is_primal,aux_primal_numbering_B[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(aux_primal_numbering_B);CHKERRQ(ierr);

    /* loop on constraints and see whether or not they need a change of basis and compute it */
    for (total_counts=n_vertices;total_counts<total_counts_cc;total_counts++) {
      size_of_constraint = constraints_idxs_ptr[total_counts+1]-constraints_idxs_ptr[total_counts];
      if (PetscBTLookup(change_basis,total_counts)) {
        /* get constraint info */
        primal_dofs = constraints_n[total_counts];
        dual_dofs = size_of_constraint-primal_dofs;

        if (pcbddc->dbg_flag) {
          ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Constraints %D: %D need a change of basis (size %D)\n",total_counts,primal_dofs,size_of_constraint);CHKERRQ(ierr);
        }

        if (PetscBTLookup(qr_needed_idx,total_counts)) { /* QR */

          /* copy quadrature constraints for change of basis check */
          if (pcbddc->dbg_flag) {
            ierr = PetscArraycpy(dbg_work,&constraints_data[constraints_data_ptr[total_counts]],size_of_constraint*primal_dofs);CHKERRQ(ierr);
          }
          /* copy temporary constraints into larger work vector (in order to store all columns of Q) */
          ierr = PetscArraycpy(qr_basis,&constraints_data[constraints_data_ptr[total_counts]],size_of_constraint*primal_dofs);CHKERRQ(ierr);

          /* compute QR decomposition of constraints */
          ierr = PetscBLASIntCast(size_of_constraint,&Blas_M);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(primal_dofs,&Blas_N);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(size_of_constraint,&Blas_LDA);CHKERRQ(ierr);
          ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
          PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&Blas_M,&Blas_N,qr_basis,&Blas_LDA,qr_tau,qr_work,&lqr_work,&lierr));
          if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GEQRF Lapack routine %d",(int)lierr);
          ierr = PetscFPTrapPop();CHKERRQ(ierr);

          /* explictly compute R^-T */
          ierr = PetscArrayzero(trs_rhs,primal_dofs*primal_dofs);CHKERRQ(ierr);
          for (j=0;j<primal_dofs;j++) trs_rhs[j*(primal_dofs+1)] = 1.0;
          ierr = PetscBLASIntCast(primal_dofs,&Blas_N);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(primal_dofs,&Blas_NRHS);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(size_of_constraint,&Blas_LDA);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(primal_dofs,&Blas_LDB);CHKERRQ(ierr);
          ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
          PetscStackCallBLAS("LAPACKtrtrs",LAPACKtrtrs_("U","T","N",&Blas_N,&Blas_NRHS,qr_basis,&Blas_LDA,trs_rhs,&Blas_LDB,&lierr));
          if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in TRTRS Lapack routine %d",(int)lierr);
          ierr = PetscFPTrapPop();CHKERRQ(ierr);

          /* explicitly compute all columns of Q (Q = [Q1 | Q2] ) overwriting QR factorization in qr_basis */
          ierr = PetscBLASIntCast(size_of_constraint,&Blas_M);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(size_of_constraint,&Blas_N);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(primal_dofs,&Blas_K);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(size_of_constraint,&Blas_LDA);CHKERRQ(ierr);
          ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
          PetscStackCallBLAS("LAPACKorgqr",LAPACKorgqr_(&Blas_M,&Blas_N,&Blas_K,qr_basis,&Blas_LDA,qr_tau,gqr_work,&lgqr_work,&lierr));
          if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in ORGQR/UNGQR Lapack routine %d",(int)lierr);
          ierr = PetscFPTrapPop();CHKERRQ(ierr);

          /* first primal_dofs columns of Q need to be re-scaled in order to be unitary w.r.t constraints
             i.e. C_{pxn}*Q_{nxn} should be equal to [I_pxp | 0_pxd] (see check below)
             where n=size_of_constraint, p=primal_dofs, d=dual_dofs (n=p+d), I and 0 identity and null matrix resp. */
          ierr = PetscBLASIntCast(size_of_constraint,&Blas_M);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(primal_dofs,&Blas_N);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(primal_dofs,&Blas_K);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(size_of_constraint,&Blas_LDA);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(primal_dofs,&Blas_LDB);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(size_of_constraint,&Blas_LDC);CHKERRQ(ierr);
          ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
          PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&Blas_M,&Blas_N,&Blas_K,&one,qr_basis,&Blas_LDA,trs_rhs,&Blas_LDB,&zero,constraints_data+constraints_data_ptr[total_counts],&Blas_LDC));
          ierr = PetscFPTrapPop();CHKERRQ(ierr);
          ierr = PetscArraycpy(qr_basis,&constraints_data[constraints_data_ptr[total_counts]],size_of_constraint*primal_dofs);CHKERRQ(ierr);

          /* insert values in change of basis matrix respecting global ordering of new primal dofs */
          start_rows = &constraints_idxs[constraints_idxs_ptr[total_counts]];
          /* insert cols for primal dofs */
          for (j=0;j<primal_dofs;j++) {
            start_vals = &qr_basis[j*size_of_constraint];
            start_cols = &constraints_idxs[constraints_idxs_ptr[total_counts]+j];
            ierr = MatSetValues(localChangeOfBasisMatrix,size_of_constraint,start_rows,1,start_cols,start_vals,INSERT_VALUES);CHKERRQ(ierr);
          }
          /* insert cols for dual dofs */
          for (j=0,k=0;j<dual_dofs;k++) {
            if (!PetscBTLookup(is_primal,constraints_idxs_B[constraints_idxs_ptr[total_counts]+k])) {
              start_vals = &qr_basis[(primal_dofs+j)*size_of_constraint];
              start_cols = &constraints_idxs[constraints_idxs_ptr[total_counts]+k];
              ierr = MatSetValues(localChangeOfBasisMatrix,size_of_constraint,start_rows,1,start_cols,start_vals,INSERT_VALUES);CHKERRQ(ierr);
              j++;
            }
          }

          /* check change of basis */
          if (pcbddc->dbg_flag) {
            PetscInt   ii,jj;
            PetscBool valid_qr=PETSC_TRUE;
            ierr = PetscBLASIntCast(primal_dofs,&Blas_M);CHKERRQ(ierr);
            ierr = PetscBLASIntCast(size_of_constraint,&Blas_N);CHKERRQ(ierr);
            ierr = PetscBLASIntCast(size_of_constraint,&Blas_K);CHKERRQ(ierr);
            ierr = PetscBLASIntCast(size_of_constraint,&Blas_LDA);CHKERRQ(ierr);
            ierr = PetscBLASIntCast(size_of_constraint,&Blas_LDB);CHKERRQ(ierr);
            ierr = PetscBLASIntCast(primal_dofs,&Blas_LDC);CHKERRQ(ierr);
            ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
            PetscStackCallBLAS("BLASgemm",BLASgemm_("T","N",&Blas_M,&Blas_N,&Blas_K,&one,dbg_work,&Blas_LDA,qr_basis,&Blas_LDB,&zero,&dbg_work[size_of_constraint*primal_dofs],&Blas_LDC));
            ierr = PetscFPTrapPop();CHKERRQ(ierr);
            for (jj=0;jj<size_of_constraint;jj++) {
              for (ii=0;ii<primal_dofs;ii++) {
                if (ii != jj && PetscAbsScalar(dbg_work[size_of_constraint*primal_dofs+jj*primal_dofs+ii]) > 1.e-12) valid_qr = PETSC_FALSE;
                if (ii == jj && PetscAbsScalar(dbg_work[size_of_constraint*primal_dofs+jj*primal_dofs+ii]-(PetscReal)1) > 1.e-12) valid_qr = PETSC_FALSE;
              }
            }
            if (!valid_qr) {
              ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"\t-> wrong change of basis!\n");CHKERRQ(ierr);
              for (jj=0;jj<size_of_constraint;jj++) {
                for (ii=0;ii<primal_dofs;ii++) {
                  if (ii != jj && PetscAbsScalar(dbg_work[size_of_constraint*primal_dofs+jj*primal_dofs+ii]) > 1.e-12) {
                    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"\tQr basis function %D is not orthogonal to constraint %D (%1.14e)!\n",jj,ii,PetscAbsScalar(dbg_work[size_of_constraint*primal_dofs+jj*primal_dofs+ii]));CHKERRQ(ierr);
                  }
                  if (ii == jj && PetscAbsScalar(dbg_work[size_of_constraint*primal_dofs+jj*primal_dofs+ii]-(PetscReal)1) > 1.e-12) {
                    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"\tQr basis function %D is not unitary w.r.t constraint %D (%1.14e)!\n",jj,ii,PetscAbsScalar(dbg_work[size_of_constraint*primal_dofs+jj*primal_dofs+ii]));CHKERRQ(ierr);
                  }
                }
              }
            } else {
              ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"\t-> right change of basis!\n");CHKERRQ(ierr);
            }
          }
        } else { /* simple transformation block */
          PetscInt    row,col;
          PetscScalar val,norm;

          ierr = PetscBLASIntCast(size_of_constraint,&Blas_N);CHKERRQ(ierr);
          PetscStackCallBLAS("BLASdot",norm = BLASdot_(&Blas_N,constraints_data+constraints_data_ptr[total_counts],&Blas_one,constraints_data+constraints_data_ptr[total_counts],&Blas_one));
          for (j=0;j<size_of_constraint;j++) {
            PetscInt row_B = constraints_idxs_B[constraints_idxs_ptr[total_counts]+j];
            row = constraints_idxs[constraints_idxs_ptr[total_counts]+j];
            if (!PetscBTLookup(is_primal,row_B)) {
              col = constraints_idxs[constraints_idxs_ptr[total_counts]];
              ierr = MatSetValue(localChangeOfBasisMatrix,row,row,1.0,INSERT_VALUES);CHKERRQ(ierr);
              ierr = MatSetValue(localChangeOfBasisMatrix,row,col,constraints_data[constraints_data_ptr[total_counts]+j]/norm,INSERT_VALUES);CHKERRQ(ierr);
            } else {
              for (k=0;k<size_of_constraint;k++) {
                col = constraints_idxs[constraints_idxs_ptr[total_counts]+k];
                if (row != col) {
                  val = -constraints_data[constraints_data_ptr[total_counts]+k]/constraints_data[constraints_data_ptr[total_counts]];
                } else {
                  val = constraints_data[constraints_data_ptr[total_counts]]/norm;
                }
                ierr = MatSetValue(localChangeOfBasisMatrix,row,col,val,INSERT_VALUES);CHKERRQ(ierr);
              }
            }
          }
          if (pcbddc->dbg_flag) {
            ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"\t-> using standard change of basis\n");CHKERRQ(ierr);
          }
        }
      } else {
        if (pcbddc->dbg_flag) {
          ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Constraint %D does not need a change of basis (size %D)\n",total_counts,size_of_constraint);CHKERRQ(ierr);
        }
      }
    }

    /* free workspace */
    if (qr_needed) {
      if (pcbddc->dbg_flag) {
        ierr = PetscFree(dbg_work);CHKERRQ(ierr);
      }
      ierr = PetscFree(trs_rhs);CHKERRQ(ierr);
      ierr = PetscFree(qr_tau);CHKERRQ(ierr);
      ierr = PetscFree(qr_work);CHKERRQ(ierr);
      ierr = PetscFree(gqr_work);CHKERRQ(ierr);
      ierr = PetscFree(qr_basis);CHKERRQ(ierr);
    }
    ierr = PetscBTDestroy(&is_primal);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(localChangeOfBasisMatrix,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(localChangeOfBasisMatrix,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    /* assembling of global change of variable */
    if (!pcbddc->fake_change) {
      Mat      tmat;
      PetscInt bs;

      ierr = VecGetSize(pcis->vec1_global,&global_size);CHKERRQ(ierr);
      ierr = VecGetLocalSize(pcis->vec1_global,&local_size);CHKERRQ(ierr);
      ierr = MatDuplicate(pc->pmat,MAT_DO_NOT_COPY_VALUES,&tmat);CHKERRQ(ierr);
      ierr = MatISSetLocalMat(tmat,localChangeOfBasisMatrix);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(tmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(tmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatCreate(PetscObjectComm((PetscObject)pc),&pcbddc->ChangeOfBasisMatrix);CHKERRQ(ierr);
      ierr = MatSetType(pcbddc->ChangeOfBasisMatrix,MATAIJ);CHKERRQ(ierr);
      ierr = MatGetBlockSize(pc->pmat,&bs);CHKERRQ(ierr);
      ierr = MatSetBlockSize(pcbddc->ChangeOfBasisMatrix,bs);CHKERRQ(ierr);
      ierr = MatSetSizes(pcbddc->ChangeOfBasisMatrix,local_size,local_size,global_size,global_size);CHKERRQ(ierr);
      ierr = MatISSetMPIXAIJPreallocation_Private(tmat,pcbddc->ChangeOfBasisMatrix,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatConvert(tmat,MATAIJ,MAT_REUSE_MATRIX,&pcbddc->ChangeOfBasisMatrix);CHKERRQ(ierr);
      ierr = MatDestroy(&tmat);CHKERRQ(ierr);
      ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
      ierr = VecSet(pcis->vec1_N,1.0);CHKERRQ(ierr);
      ierr = VecScatterBegin(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecReciprocal(pcis->vec1_global);CHKERRQ(ierr);
      ierr = MatDiagonalScale(pcbddc->ChangeOfBasisMatrix,pcis->vec1_global,NULL);CHKERRQ(ierr);

      /* check */
      if (pcbddc->dbg_flag) {
        PetscReal error;
        Vec       x,x_change;

        ierr = VecDuplicate(pcis->vec1_global,&x);CHKERRQ(ierr);
        ierr = VecDuplicate(pcis->vec1_global,&x_change);CHKERRQ(ierr);
        ierr = VecSetRandom(x,NULL);CHKERRQ(ierr);
        ierr = VecCopy(x,pcis->vec1_global);CHKERRQ(ierr);
        ierr = VecScatterBegin(matis->rctx,x,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(matis->rctx,x,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = MatMult(localChangeOfBasisMatrix,pcis->vec1_N,pcis->vec2_N);CHKERRQ(ierr);
        ierr = VecScatterBegin(matis->rctx,pcis->vec2_N,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        ierr = VecScatterEnd(matis->rctx,pcis->vec2_N,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        ierr = MatMult(pcbddc->ChangeOfBasisMatrix,pcis->vec1_global,x_change);CHKERRQ(ierr);
        ierr = VecAXPY(x,-1.0,x_change);CHKERRQ(ierr);
        ierr = VecNorm(x,NORM_INFINITY,&error);CHKERRQ(ierr);
        if (error > PETSC_SMALL) {
          SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"Error global vs local change on N: %1.6e",error);
        }
        ierr = VecDestroy(&x);CHKERRQ(ierr);
        ierr = VecDestroy(&x_change);CHKERRQ(ierr);
      }
      /* adapt sub_schurs computed (if any) */
      if (pcbddc->use_deluxe_scaling) {
        PCBDDCSubSchurs sub_schurs=pcbddc->sub_schurs;

        if (pcbddc->use_change_of_basis && pcbddc->adaptive_userdefined) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Cannot mix automatic change of basis, adaptive selection and user-defined constraints");
        if (sub_schurs && sub_schurs->S_Ej_all) {
          Mat                    S_new,tmat;
          IS                     is_all_N,is_V_Sall = NULL;

          ierr = ISLocalToGlobalMappingApplyIS(pcis->BtoNmap,sub_schurs->is_Ej_all,&is_all_N);CHKERRQ(ierr);
          ierr = MatCreateSubMatrix(localChangeOfBasisMatrix,is_all_N,is_all_N,MAT_INITIAL_MATRIX,&tmat);CHKERRQ(ierr);
          if (pcbddc->deluxe_zerorows) {
            ISLocalToGlobalMapping NtoSall;
            IS                     is_V;
            ierr = ISCreateGeneral(PETSC_COMM_SELF,pcbddc->n_vertices,pcbddc->local_primal_ref_node,PETSC_COPY_VALUES,&is_V);CHKERRQ(ierr);
            ierr = ISLocalToGlobalMappingCreateIS(is_all_N,&NtoSall);CHKERRQ(ierr);
            ierr = ISGlobalToLocalMappingApplyIS(NtoSall,IS_GTOLM_DROP,is_V,&is_V_Sall);CHKERRQ(ierr);
            ierr = ISLocalToGlobalMappingDestroy(&NtoSall);CHKERRQ(ierr);
            ierr = ISDestroy(&is_V);CHKERRQ(ierr);
          }
          ierr = ISDestroy(&is_all_N);CHKERRQ(ierr);
          ierr = MatPtAP(sub_schurs->S_Ej_all,tmat,MAT_INITIAL_MATRIX,1.0,&S_new);CHKERRQ(ierr);
          ierr = MatDestroy(&sub_schurs->S_Ej_all);CHKERRQ(ierr);
          ierr = PetscObjectReference((PetscObject)S_new);CHKERRQ(ierr);
          if (pcbddc->deluxe_zerorows) {
            const PetscScalar *array;
            const PetscInt    *idxs_V,*idxs_all;
            PetscInt          i,n_V;

            ierr = MatZeroRowsColumnsIS(S_new,is_V_Sall,1.,NULL,NULL);CHKERRQ(ierr);
            ierr = ISGetLocalSize(is_V_Sall,&n_V);CHKERRQ(ierr);
            ierr = ISGetIndices(is_V_Sall,&idxs_V);CHKERRQ(ierr);
            ierr = ISGetIndices(sub_schurs->is_Ej_all,&idxs_all);CHKERRQ(ierr);
            ierr = VecGetArrayRead(pcis->D,&array);CHKERRQ(ierr);
            for (i=0;i<n_V;i++) {
              PetscScalar val;
              PetscInt    idx;

              idx = idxs_V[i];
              val = array[idxs_all[idxs_V[i]]];
              ierr = MatSetValue(S_new,idx,idx,val,INSERT_VALUES);CHKERRQ(ierr);
            }
            ierr = MatAssemblyBegin(S_new,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
            ierr = MatAssemblyEnd(S_new,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
            ierr = VecRestoreArrayRead(pcis->D,&array);CHKERRQ(ierr);
            ierr = ISRestoreIndices(sub_schurs->is_Ej_all,&idxs_all);CHKERRQ(ierr);
            ierr = ISRestoreIndices(is_V_Sall,&idxs_V);CHKERRQ(ierr);
          }
          sub_schurs->S_Ej_all = S_new;
          ierr = MatDestroy(&S_new);CHKERRQ(ierr);
          if (sub_schurs->sum_S_Ej_all) {
            ierr = MatPtAP(sub_schurs->sum_S_Ej_all,tmat,MAT_INITIAL_MATRIX,1.0,&S_new);CHKERRQ(ierr);
            ierr = MatDestroy(&sub_schurs->sum_S_Ej_all);CHKERRQ(ierr);
            ierr = PetscObjectReference((PetscObject)S_new);CHKERRQ(ierr);
            if (pcbddc->deluxe_zerorows) {
              ierr = MatZeroRowsColumnsIS(S_new,is_V_Sall,1.,NULL,NULL);CHKERRQ(ierr);
            }
            sub_schurs->sum_S_Ej_all = S_new;
            ierr = MatDestroy(&S_new);CHKERRQ(ierr);
          }
          ierr = ISDestroy(&is_V_Sall);CHKERRQ(ierr);
          ierr = MatDestroy(&tmat);CHKERRQ(ierr);
        }
        /* destroy any change of basis context in sub_schurs */
        if (sub_schurs && sub_schurs->change) {
          PetscInt i;

          for (i=0;i<sub_schurs->n_subs;i++) {
            ierr = KSPDestroy(&sub_schurs->change[i]);CHKERRQ(ierr);
          }
          ierr = PetscFree(sub_schurs->change);CHKERRQ(ierr);
        }
      }
      if (pcbddc->switch_static) { /* need to save the local change */
        pcbddc->switch_static_change = localChangeOfBasisMatrix;
      } else {
        ierr = MatDestroy(&localChangeOfBasisMatrix);CHKERRQ(ierr);
      }
      /* determine if any process has changed the pressures locally */
      pcbddc->change_interior = pcbddc->benign_have_null;
    } else { /* fake change (get back change of basis into ConstraintMatrix and info on qr) */
      ierr = MatDestroy(&pcbddc->ConstraintMatrix);CHKERRQ(ierr);
      pcbddc->ConstraintMatrix = localChangeOfBasisMatrix;
      pcbddc->use_qr_single = qr_needed;
    }
  } else if (pcbddc->user_ChangeOfBasisMatrix || pcbddc->benign_saddle_point) {
    if (!pcbddc->benign_have_null && pcbddc->user_ChangeOfBasisMatrix) {
      ierr = PetscObjectReference((PetscObject)pcbddc->user_ChangeOfBasisMatrix);CHKERRQ(ierr);
      pcbddc->ChangeOfBasisMatrix = pcbddc->user_ChangeOfBasisMatrix;
    } else {
      Mat benign_global = NULL;
      if (pcbddc->benign_have_null) {
        Mat M;

        pcbddc->change_interior = PETSC_TRUE;
        ierr = VecCopy(matis->counter,pcis->vec1_N);CHKERRQ(ierr);
        ierr = VecReciprocal(pcis->vec1_N);CHKERRQ(ierr);
        ierr = MatDuplicate(pc->pmat,MAT_DO_NOT_COPY_VALUES,&benign_global);CHKERRQ(ierr);
        if (pcbddc->benign_change) {
          ierr = MatDuplicate(pcbddc->benign_change,MAT_COPY_VALUES,&M);CHKERRQ(ierr);
          ierr = MatDiagonalScale(M,pcis->vec1_N,NULL);CHKERRQ(ierr);
        } else {
          ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,pcis->n,pcis->n,1,NULL,&M);CHKERRQ(ierr);
          ierr = MatDiagonalSet(M,pcis->vec1_N,INSERT_VALUES);CHKERRQ(ierr);
        }
        ierr = MatISSetLocalMat(benign_global,M);CHKERRQ(ierr);
        ierr = MatDestroy(&M);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(benign_global,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(benign_global,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      }
      if (pcbddc->user_ChangeOfBasisMatrix) {
        ierr = MatMatMult(pcbddc->user_ChangeOfBasisMatrix,benign_global,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&pcbddc->ChangeOfBasisMatrix);CHKERRQ(ierr);
        ierr = MatDestroy(&benign_global);CHKERRQ(ierr);
      } else if (pcbddc->benign_have_null) {
        pcbddc->ChangeOfBasisMatrix = benign_global;
      }
    }
    if (pcbddc->switch_static && pcbddc->ChangeOfBasisMatrix) { /* need to save the local change */
      IS             is_global;
      const PetscInt *gidxs;

      ierr = ISLocalToGlobalMappingGetIndices(pc->pmat->rmap->mapping,&gidxs);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pc),pcis->n,gidxs,PETSC_COPY_VALUES,&is_global);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingRestoreIndices(pc->pmat->rmap->mapping,&gidxs);CHKERRQ(ierr);
      ierr = MatCreateSubMatrixUnsorted(pcbddc->ChangeOfBasisMatrix,is_global,is_global,&pcbddc->switch_static_change);CHKERRQ(ierr);
      ierr = ISDestroy(&is_global);CHKERRQ(ierr);
    }
  }
  if (!pcbddc->fake_change && pcbddc->ChangeOfBasisMatrix && !pcbddc->work_change) {
    ierr = VecDuplicate(pcis->vec1_global,&pcbddc->work_change);CHKERRQ(ierr);
  }

  if (!pcbddc->fake_change) {
    /* add pressure dofs to set of primal nodes for numbering purposes */
    for (i=0;i<pcbddc->benign_n;i++) {
      pcbddc->local_primal_ref_node[pcbddc->local_primal_size_cc] = pcbddc->benign_p0_lidx[i];
      pcbddc->primal_indices_local_idxs[pcbddc->local_primal_size] = pcbddc->benign_p0_lidx[i];
      pcbddc->local_primal_ref_mult[pcbddc->local_primal_size_cc] = 1;
      pcbddc->local_primal_size_cc++;
      pcbddc->local_primal_size++;
    }

    /* check if a new primal space has been introduced (also take into account benign trick) */
    pcbddc->new_primal_space_local = PETSC_TRUE;
    if (olocal_primal_size == pcbddc->local_primal_size) {
      ierr = PetscArraycmp(pcbddc->local_primal_ref_node,olocal_primal_ref_node,olocal_primal_size_cc,&pcbddc->new_primal_space_local);CHKERRQ(ierr);
      pcbddc->new_primal_space_local = (PetscBool)(!pcbddc->new_primal_space_local);
      if (!pcbddc->new_primal_space_local) {
        ierr = PetscArraycmp(pcbddc->local_primal_ref_mult,olocal_primal_ref_mult,olocal_primal_size_cc,&pcbddc->new_primal_space_local);CHKERRQ(ierr);
        pcbddc->new_primal_space_local = (PetscBool)(!pcbddc->new_primal_space_local);
      }
    }
    /* new_primal_space will be used for numbering of coarse dofs, so it should be the same across all subdomains */
    ierr = MPIU_Allreduce(&pcbddc->new_primal_space_local,&pcbddc->new_primal_space,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
  }
  ierr = PetscFree2(olocal_primal_ref_node,olocal_primal_ref_mult);CHKERRQ(ierr);

  /* flush dbg viewer */
  if (pcbddc->dbg_flag) {
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
  }

  /* free workspace */
  ierr = PetscBTDestroy(&qr_needed_idx);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&change_basis);CHKERRQ(ierr);
  if (!pcbddc->adaptive_selection) {
    ierr = PetscFree3(constraints_idxs_ptr,constraints_data_ptr,constraints_n);CHKERRQ(ierr);
    ierr = PetscFree3(constraints_data,constraints_idxs,constraints_idxs_B);CHKERRQ(ierr);
  } else {
    ierr = PetscFree5(pcbddc->adaptive_constraints_n,
                      pcbddc->adaptive_constraints_idxs_ptr,
                      pcbddc->adaptive_constraints_data_ptr,
                      pcbddc->adaptive_constraints_idxs,
                      pcbddc->adaptive_constraints_data);CHKERRQ(ierr);
    ierr = PetscFree(constraints_n);CHKERRQ(ierr);
    ierr = PetscFree(constraints_idxs_B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCAnalyzeInterface(PC pc)
{
  ISLocalToGlobalMapping map;
  PC_BDDC                *pcbddc = (PC_BDDC*)pc->data;
  Mat_IS                 *matis  = (Mat_IS*)pc->pmat->data;
  PetscInt               i,N;
  PetscBool              rcsr = PETSC_FALSE;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  if (pcbddc->recompute_topography) {
    pcbddc->graphanalyzed = PETSC_FALSE;
    /* Reset previously computed graph */
    ierr = PCBDDCGraphReset(pcbddc->mat_graph);CHKERRQ(ierr);
    /* Init local Graph struct */
    ierr = MatGetSize(pc->pmat,&N,NULL);CHKERRQ(ierr);
    ierr = MatGetLocalToGlobalMapping(pc->pmat,&map,NULL);CHKERRQ(ierr);
    ierr = PCBDDCGraphInit(pcbddc->mat_graph,map,N,pcbddc->graphmaxcount);CHKERRQ(ierr);

    if (pcbddc->user_primal_vertices_local && !pcbddc->user_primal_vertices) {
      ierr = PCBDDCConsistencyCheckIS(pc,MPI_LOR,&pcbddc->user_primal_vertices_local);CHKERRQ(ierr);
    }
    /* Check validity of the csr graph passed in by the user */
    if (pcbddc->mat_graph->nvtxs_csr && pcbddc->mat_graph->nvtxs_csr != pcbddc->mat_graph->nvtxs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid size of local CSR graph! Found %D, expected %D",pcbddc->mat_graph->nvtxs_csr,pcbddc->mat_graph->nvtxs);

    /* Set default CSR adjacency of local dofs if not provided by the user with PCBDDCSetLocalAdjacencyGraph */
    if (!pcbddc->mat_graph->xadj && pcbddc->use_local_adj) {
      PetscInt  *xadj,*adjncy;
      PetscInt  nvtxs;
      PetscBool flg_row=PETSC_FALSE;

      ierr = MatGetRowIJ(matis->A,0,PETSC_TRUE,PETSC_FALSE,&nvtxs,(const PetscInt**)&xadj,(const PetscInt**)&adjncy,&flg_row);CHKERRQ(ierr);
      if (flg_row) {
        ierr = PCBDDCSetLocalAdjacencyGraph(pc,nvtxs,xadj,adjncy,PETSC_COPY_VALUES);CHKERRQ(ierr);
        pcbddc->computed_rowadj = PETSC_TRUE;
      }
      ierr = MatRestoreRowIJ(matis->A,0,PETSC_TRUE,PETSC_FALSE,&nvtxs,(const PetscInt**)&xadj,(const PetscInt**)&adjncy,&flg_row);CHKERRQ(ierr);
      rcsr = PETSC_TRUE;
    }
    if (pcbddc->dbg_flag) {
      ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    }

    if (pcbddc->mat_graph->cdim && !pcbddc->mat_graph->cloc) {
      PetscReal    *lcoords;
      PetscInt     n;
      MPI_Datatype dimrealtype;

      /* TODO: support for blocked */
      if (pcbddc->mat_graph->cnloc != pc->pmat->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Invalid number of local coordinates! Got %D, expected %D",pcbddc->mat_graph->cnloc,pc->pmat->rmap->n);
      ierr = MatGetLocalSize(matis->A,&n,NULL);CHKERRQ(ierr);
      ierr = PetscMalloc1(pcbddc->mat_graph->cdim*n,&lcoords);CHKERRQ(ierr);
      ierr = MPI_Type_contiguous(pcbddc->mat_graph->cdim,MPIU_REAL,&dimrealtype);CHKERRQ(ierr);
      ierr = MPI_Type_commit(&dimrealtype);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(matis->sf,dimrealtype,pcbddc->mat_graph->coords,lcoords);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(matis->sf,dimrealtype,pcbddc->mat_graph->coords,lcoords);CHKERRQ(ierr);
      ierr = MPI_Type_free(&dimrealtype);CHKERRQ(ierr);
      ierr = PetscFree(pcbddc->mat_graph->coords);CHKERRQ(ierr);

      pcbddc->mat_graph->coords = lcoords;
      pcbddc->mat_graph->cloc   = PETSC_TRUE;
      pcbddc->mat_graph->cnloc  = n;
    }
    if (pcbddc->mat_graph->cnloc && pcbddc->mat_graph->cnloc != pcbddc->mat_graph->nvtxs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Invalid number of local subdomain coordinates! Got %D, expected %D",pcbddc->mat_graph->cnloc,pcbddc->mat_graph->nvtxs);
    pcbddc->mat_graph->active_coords = (PetscBool)(pcbddc->corner_selection && !pcbddc->corner_selected);

    /* Setup of Graph */
    pcbddc->mat_graph->commsizelimit = 0; /* don't use the COMM_SELF variant of the graph */
    ierr = PCBDDCGraphSetUp(pcbddc->mat_graph,pcbddc->vertex_size,pcbddc->NeumannBoundariesLocal,pcbddc->DirichletBoundariesLocal,pcbddc->n_ISForDofsLocal,pcbddc->ISForDofsLocal,pcbddc->user_primal_vertices_local);CHKERRQ(ierr);

    /* attach info on disconnected subdomains if present */
    if (pcbddc->n_local_subs) {
      PetscInt *local_subs,n,totn;

      ierr = MatGetLocalSize(matis->A,&n,NULL);CHKERRQ(ierr);
      ierr = PetscMalloc1(n,&local_subs);CHKERRQ(ierr);
      for (i=0;i<n;i++) local_subs[i] = pcbddc->n_local_subs;
      for (i=0;i<pcbddc->n_local_subs;i++) {
        const PetscInt *idxs;
        PetscInt       nl,j;

        ierr = ISGetLocalSize(pcbddc->local_subs[i],&nl);CHKERRQ(ierr);
        ierr = ISGetIndices(pcbddc->local_subs[i],&idxs);CHKERRQ(ierr);
        for (j=0;j<nl;j++) local_subs[idxs[j]] = i;
        ierr = ISRestoreIndices(pcbddc->local_subs[i],&idxs);CHKERRQ(ierr);
      }
      for (i=0,totn=0;i<n;i++) totn = PetscMax(totn,local_subs[i]);
      pcbddc->mat_graph->n_local_subs = totn + 1;
      pcbddc->mat_graph->local_subs = local_subs;
    }
  }

  if (!pcbddc->graphanalyzed) {
    /* Graph's connected components analysis */
    ierr = PCBDDCGraphComputeConnectedComponents(pcbddc->mat_graph);CHKERRQ(ierr);
    pcbddc->graphanalyzed = PETSC_TRUE;
    pcbddc->corner_selected = pcbddc->corner_selection;
  }
  if (rcsr) pcbddc->mat_graph->nvtxs_csr = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCOrthonormalizeVecs(PetscInt *nio, Vec vecs[])
{
  PetscInt       i,j,n;
  PetscScalar    *alphas;
  PetscReal      norm,*onorms;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  n = *nio;
  if (!n) PetscFunctionReturn(0);
  ierr = PetscMalloc2(n,&alphas,n,&onorms);CHKERRQ(ierr);
  ierr = VecNormalize(vecs[0],&norm);CHKERRQ(ierr);
  if (norm < PETSC_SMALL) {
    onorms[0] = 0.0;
    ierr = VecSet(vecs[0],0.0);CHKERRQ(ierr);
  } else {
    onorms[0] = norm;
  }

  for (i=1;i<n;i++) {
    ierr = VecMDot(vecs[i],i,vecs,alphas);CHKERRQ(ierr);
    for (j=0;j<i;j++) alphas[j] = PetscConj(-alphas[j]);
    ierr = VecMAXPY(vecs[i],i,alphas,vecs);CHKERRQ(ierr);
    ierr = VecNormalize(vecs[i],&norm);CHKERRQ(ierr);
    if (norm < PETSC_SMALL) {
      onorms[i] = 0.0;
      ierr = VecSet(vecs[i],0.0);CHKERRQ(ierr);
    } else {
      onorms[i] = norm;
    }
  }
  /* push nonzero vectors at the beginning */
  for (i=0;i<n;i++) {
    if (onorms[i] == 0.0) {
      for (j=i+1;j<n;j++) {
        if (onorms[j] != 0.0) {
          ierr = VecCopy(vecs[j],vecs[i]);CHKERRQ(ierr);
          onorms[j] = 0.0;
        }
      }
    }
  }
  for (i=0,*nio=0;i<n;i++) *nio += onorms[i] != 0.0 ? 1 : 0;
  ierr = PetscFree2(alphas,onorms);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCMatISGetSubassemblingPattern(Mat mat, PetscInt *n_subdomains, PetscInt redprocs, IS* is_sends, PetscBool *have_void)
{
  Mat            A;
  PetscInt       n_neighs,*neighs,*n_shared,**shared;
  PetscMPIInt    size,rank,color;
  PetscInt       *xadj,*adjncy;
  PetscInt       *adjncy_wgt,*v_wgt,*ranks_send_to_idx;
  PetscInt       im_active,active_procs,N,n,i,j,threshold = 2;
  PetscInt       void_procs,*procs_candidates = NULL;
  PetscInt       xadj_count,*count;
  PetscBool      ismatis,use_vwgt=PETSC_FALSE;
  PetscSubcomm   psubcomm;
  MPI_Comm       subcomm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATIS,&ismatis);CHKERRQ(ierr);
  if (!ismatis) SETERRQ1(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot use %s on a matrix object which is not of type MATIS",PETSC_FUNCTION_NAME);
  PetscValidLogicalCollectiveInt(mat,*n_subdomains,2);
  PetscValidLogicalCollectiveInt(mat,redprocs,3);
  if (*n_subdomains <=0) SETERRQ1(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONG,"Invalid number of subdomains requested %D",*n_subdomains);

  if (have_void) *have_void = PETSC_FALSE;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)mat),&rank);CHKERRQ(ierr);
  ierr = MatISGetLocalMat(mat,&A);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&n,NULL);CHKERRQ(ierr);
  im_active = !!n;
  ierr = MPIU_Allreduce(&im_active,&active_procs,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)mat));CHKERRQ(ierr);
  void_procs = size - active_procs;
  /* get ranks of of non-active processes in mat communicator */
  if (void_procs) {
    PetscInt ncand;

    if (have_void) *have_void = PETSC_TRUE;
    ierr = PetscMalloc1(size,&procs_candidates);CHKERRQ(ierr);
    ierr = MPI_Allgather(&im_active,1,MPIU_INT,procs_candidates,1,MPIU_INT,PetscObjectComm((PetscObject)mat));CHKERRQ(ierr);
    for (i=0,ncand=0;i<size;i++) {
      if (!procs_candidates[i]) {
        procs_candidates[ncand++] = i;
      }
    }
    /* force n_subdomains to be not greater that the number of non-active processes */
    *n_subdomains = PetscMin(void_procs,*n_subdomains);
  }

  /* number of subdomains requested greater than active processes or matrix size -> just shift the matrix
     number of subdomains requested 1 -> send to master or first candidate in voids  */
  ierr = MatGetSize(mat,&N,NULL);CHKERRQ(ierr);
  if (active_procs < *n_subdomains || *n_subdomains == 1 || N <= *n_subdomains) {
    PetscInt issize,isidx,dest;
    if (*n_subdomains == 1) dest = 0;
    else dest = rank;
    if (im_active) {
      issize = 1;
      if (procs_candidates) { /* shift the pattern on non-active candidates (if any) */
        isidx = procs_candidates[dest];
      } else {
        isidx = dest;
      }
    } else {
      issize = 0;
      isidx = -1;
    }
    if (*n_subdomains != 1) *n_subdomains = active_procs;
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)mat),issize,&isidx,PETSC_COPY_VALUES,is_sends);CHKERRQ(ierr);
    ierr = PetscFree(procs_candidates);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscOptionsGetBool(NULL,NULL,"-matis_partitioning_use_vwgt",&use_vwgt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-matis_partitioning_threshold",&threshold,NULL);CHKERRQ(ierr);
  threshold = PetscMax(threshold,2);

  /* Get info on mapping */
  ierr = ISLocalToGlobalMappingGetInfo(mat->rmap->mapping,&n_neighs,&neighs,&n_shared,&shared);CHKERRQ(ierr);

  /* build local CSR graph of subdomains' connectivity */
  ierr = PetscMalloc1(2,&xadj);CHKERRQ(ierr);
  xadj[0] = 0;
  xadj[1] = PetscMax(n_neighs-1,0);
  ierr = PetscMalloc1(xadj[1],&adjncy);CHKERRQ(ierr);
  ierr = PetscMalloc1(xadj[1],&adjncy_wgt);CHKERRQ(ierr);
  ierr = PetscCalloc1(n,&count);CHKERRQ(ierr);
  for (i=1;i<n_neighs;i++)
    for (j=0;j<n_shared[i];j++)
      count[shared[i][j]] += 1;

  xadj_count = 0;
  for (i=1;i<n_neighs;i++) {
    for (j=0;j<n_shared[i];j++) {
      if (count[shared[i][j]] < threshold) {
        adjncy[xadj_count] = neighs[i];
        adjncy_wgt[xadj_count] = n_shared[i];
        xadj_count++;
        break;
      }
    }
  }
  xadj[1] = xadj_count;
  ierr = PetscFree(count);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreInfo(mat->rmap->mapping,&n_neighs,&neighs,&n_shared,&shared);CHKERRQ(ierr);
  ierr = PetscSortIntWithArray(xadj[1],adjncy,adjncy_wgt);CHKERRQ(ierr);

  ierr = PetscMalloc1(1,&ranks_send_to_idx);CHKERRQ(ierr);

  /* Restrict work on active processes only */
  ierr = PetscMPIIntCast(im_active,&color);CHKERRQ(ierr);
  if (void_procs) {
    ierr = PetscSubcommCreate(PetscObjectComm((PetscObject)mat),&psubcomm);CHKERRQ(ierr);
    ierr = PetscSubcommSetNumber(psubcomm,2);CHKERRQ(ierr); /* 2 groups, active process and not active processes */
    ierr = PetscSubcommSetTypeGeneral(psubcomm,color,rank);CHKERRQ(ierr);
    subcomm = PetscSubcommChild(psubcomm);
  } else {
    psubcomm = NULL;
    subcomm = PetscObjectComm((PetscObject)mat);
  }

  v_wgt = NULL;
  if (!color) {
    ierr = PetscFree(xadj);CHKERRQ(ierr);
    ierr = PetscFree(adjncy);CHKERRQ(ierr);
    ierr = PetscFree(adjncy_wgt);CHKERRQ(ierr);
  } else {
    Mat             subdomain_adj;
    IS              new_ranks,new_ranks_contig;
    MatPartitioning partitioner;
    PetscInt        rstart=0,rend=0;
    PetscInt        *is_indices,*oldranks;
    PetscMPIInt     size;
    PetscBool       aggregate;

    ierr = MPI_Comm_size(subcomm,&size);CHKERRQ(ierr);
    if (void_procs) {
      PetscInt prank = rank;
      ierr = PetscMalloc1(size,&oldranks);CHKERRQ(ierr);
      ierr = MPI_Allgather(&prank,1,MPIU_INT,oldranks,1,MPIU_INT,subcomm);CHKERRQ(ierr);
      for (i=0;i<xadj[1];i++) {
        ierr = PetscFindInt(adjncy[i],size,oldranks,&adjncy[i]);CHKERRQ(ierr);
      }
      ierr = PetscSortIntWithArray(xadj[1],adjncy,adjncy_wgt);CHKERRQ(ierr);
    } else {
      oldranks = NULL;
    }
    aggregate = ((redprocs > 0 && redprocs < size) ? PETSC_TRUE : PETSC_FALSE);
    if (aggregate) { /* TODO: all this part could be made more efficient */
      PetscInt    lrows,row,ncols,*cols;
      PetscMPIInt nrank;
      PetscScalar *vals;

      ierr = MPI_Comm_rank(subcomm,&nrank);CHKERRQ(ierr);
      lrows = 0;
      if (nrank<redprocs) {
        lrows = size/redprocs;
        if (nrank<size%redprocs) lrows++;
      }
      ierr = MatCreateAIJ(subcomm,lrows,lrows,size,size,50,NULL,50,NULL,&subdomain_adj);CHKERRQ(ierr);
      ierr = MatGetOwnershipRange(subdomain_adj,&rstart,&rend);CHKERRQ(ierr);
      ierr = MatSetOption(subdomain_adj,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatSetOption(subdomain_adj,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
      row = nrank;
      ncols = xadj[1]-xadj[0];
      cols = adjncy;
      ierr = PetscMalloc1(ncols,&vals);CHKERRQ(ierr);
      for (i=0;i<ncols;i++) vals[i] = adjncy_wgt[i];
      ierr = MatSetValues(subdomain_adj,1,&row,ncols,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(subdomain_adj,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(subdomain_adj,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = PetscFree(xadj);CHKERRQ(ierr);
      ierr = PetscFree(adjncy);CHKERRQ(ierr);
      ierr = PetscFree(adjncy_wgt);CHKERRQ(ierr);
      ierr = PetscFree(vals);CHKERRQ(ierr);
      if (use_vwgt) {
        Vec               v;
        const PetscScalar *array;
        PetscInt          nl;

        ierr = MatCreateVecs(subdomain_adj,&v,NULL);CHKERRQ(ierr);
        ierr = VecSetValue(v,row,(PetscScalar)n,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecAssemblyBegin(v);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(v);CHKERRQ(ierr);
        ierr = VecGetLocalSize(v,&nl);CHKERRQ(ierr);
        ierr = VecGetArrayRead(v,&array);CHKERRQ(ierr);
        ierr = PetscMalloc1(nl,&v_wgt);CHKERRQ(ierr);
        for (i=0;i<nl;i++) v_wgt[i] = (PetscInt)PetscRealPart(array[i]);
        ierr = VecRestoreArrayRead(v,&array);CHKERRQ(ierr);
        ierr = VecDestroy(&v);CHKERRQ(ierr);
      }
    } else {
      ierr = MatCreateMPIAdj(subcomm,1,(PetscInt)size,xadj,adjncy,adjncy_wgt,&subdomain_adj);CHKERRQ(ierr);
      if (use_vwgt) {
        ierr = PetscMalloc1(1,&v_wgt);CHKERRQ(ierr);
        v_wgt[0] = n;
      }
    }
    /* ierr = MatView(subdomain_adj,0);CHKERRQ(ierr); */

    /* Partition */
    ierr = MatPartitioningCreate(subcomm,&partitioner);CHKERRQ(ierr);
#if defined(PETSC_HAVE_PTSCOTCH)
    ierr = MatPartitioningSetType(partitioner,MATPARTITIONINGPTSCOTCH);CHKERRQ(ierr);
#elif defined(PETSC_HAVE_PARMETIS)
    ierr = MatPartitioningSetType(partitioner,MATPARTITIONINGPARMETIS);CHKERRQ(ierr);
#else
    ierr = MatPartitioningSetType(partitioner,MATPARTITIONINGAVERAGE);CHKERRQ(ierr);
#endif
    ierr = MatPartitioningSetAdjacency(partitioner,subdomain_adj);CHKERRQ(ierr);
    if (v_wgt) {
      ierr = MatPartitioningSetVertexWeights(partitioner,v_wgt);CHKERRQ(ierr);
    }
    *n_subdomains = PetscMin((PetscInt)size,*n_subdomains);
    ierr = MatPartitioningSetNParts(partitioner,*n_subdomains);CHKERRQ(ierr);
    ierr = MatPartitioningSetFromOptions(partitioner);CHKERRQ(ierr);
    ierr = MatPartitioningApply(partitioner,&new_ranks);CHKERRQ(ierr);
    /* ierr = MatPartitioningView(partitioner,0);CHKERRQ(ierr); */

    /* renumber new_ranks to avoid "holes" in new set of processors */
    ierr = ISRenumber(new_ranks,NULL,NULL,&new_ranks_contig);CHKERRQ(ierr);
    ierr = ISDestroy(&new_ranks);CHKERRQ(ierr);
    ierr = ISGetIndices(new_ranks_contig,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    if (!aggregate) {
      if (procs_candidates) { /* shift the pattern on non-active candidates (if any) */
#if defined(PETSC_USE_DEBUG)
        if (!oldranks) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"This should not happen");
#endif
        ranks_send_to_idx[0] = procs_candidates[oldranks[is_indices[0]]];
      } else if (oldranks) {
        ranks_send_to_idx[0] = oldranks[is_indices[0]];
      } else {
        ranks_send_to_idx[0] = is_indices[0];
      }
    } else {
      PetscInt    idx = 0;
      PetscMPIInt tag;
      MPI_Request *reqs;

      ierr = PetscObjectGetNewTag((PetscObject)subdomain_adj,&tag);CHKERRQ(ierr);
      ierr = PetscMalloc1(rend-rstart,&reqs);CHKERRQ(ierr);
      for (i=rstart;i<rend;i++) {
        ierr = MPI_Isend(is_indices+i-rstart,1,MPIU_INT,i,tag,subcomm,&reqs[i-rstart]);CHKERRQ(ierr);
      }
      ierr = MPI_Recv(&idx,1,MPIU_INT,MPI_ANY_SOURCE,tag,subcomm,MPI_STATUS_IGNORE);CHKERRQ(ierr);
      ierr = MPI_Waitall(rend-rstart,reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
      ierr = PetscFree(reqs);CHKERRQ(ierr);
      if (procs_candidates) { /* shift the pattern on non-active candidates (if any) */
#if defined(PETSC_USE_DEBUG)
        if (!oldranks) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"This should not happen");
#endif
        ranks_send_to_idx[0] = procs_candidates[oldranks[idx]];
      } else if (oldranks) {
        ranks_send_to_idx[0] = oldranks[idx];
      } else {
        ranks_send_to_idx[0] = idx;
      }
    }
    ierr = ISRestoreIndices(new_ranks_contig,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    /* clean up */
    ierr = PetscFree(oldranks);CHKERRQ(ierr);
    ierr = ISDestroy(&new_ranks_contig);CHKERRQ(ierr);
    ierr = MatDestroy(&subdomain_adj);CHKERRQ(ierr);
    ierr = MatPartitioningDestroy(&partitioner);CHKERRQ(ierr);
  }
  ierr = PetscSubcommDestroy(&psubcomm);CHKERRQ(ierr);
  ierr = PetscFree(procs_candidates);CHKERRQ(ierr);

  /* assemble parallel IS for sends */
  i = 1;
  if (!color) i=0;
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)mat),i,ranks_send_to_idx,PETSC_OWN_POINTER,is_sends);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef enum {MATDENSE_PRIVATE=0,MATAIJ_PRIVATE,MATBAIJ_PRIVATE,MATSBAIJ_PRIVATE}MatTypePrivate;

PetscErrorCode PCBDDCMatISSubassemble(Mat mat, IS is_sends, PetscInt n_subdomains, PetscBool restrict_comm, PetscBool restrict_full, PetscBool reuse, Mat *mat_n, PetscInt nis, IS isarray[], PetscInt nvecs, Vec nnsp_vec[])
{
  Mat                    local_mat;
  IS                     is_sends_internal;
  PetscInt               rows,cols,new_local_rows;
  PetscInt               i,bs,buf_size_idxs,buf_size_idxs_is,buf_size_vals,buf_size_vecs;
  PetscBool              ismatis,isdense,newisdense,destroy_mat;
  ISLocalToGlobalMapping l2gmap;
  PetscInt*              l2gmap_indices;
  const PetscInt*        is_indices;
  MatType                new_local_type;
  /* buffers */
  PetscInt               *ptr_idxs,*send_buffer_idxs,*recv_buffer_idxs;
  PetscInt               *ptr_idxs_is,*send_buffer_idxs_is,*recv_buffer_idxs_is;
  PetscInt               *recv_buffer_idxs_local;
  PetscScalar            *ptr_vals,*recv_buffer_vals;
  const PetscScalar      *send_buffer_vals;
  PetscScalar            *ptr_vecs,*send_buffer_vecs,*recv_buffer_vecs;
  /* MPI */
  MPI_Comm               comm,comm_n;
  PetscSubcomm           subcomm;
  PetscMPIInt            n_sends,n_recvs,size;
  PetscMPIInt            *iflags,*ilengths_idxs,*ilengths_vals,*ilengths_idxs_is;
  PetscMPIInt            *onodes,*onodes_is,*olengths_idxs,*olengths_idxs_is,*olengths_vals;
  PetscMPIInt            len,tag_idxs,tag_idxs_is,tag_vals,tag_vecs,source_dest;
  MPI_Request            *send_req_idxs,*send_req_idxs_is,*send_req_vals,*send_req_vecs;
  MPI_Request            *recv_req_idxs,*recv_req_idxs_is,*recv_req_vals,*recv_req_vecs;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATIS,&ismatis);CHKERRQ(ierr);
  if (!ismatis) SETERRQ1(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot use %s on a matrix object which is not of type MATIS",PETSC_FUNCTION_NAME);
  PetscValidLogicalCollectiveInt(mat,n_subdomains,3);
  PetscValidLogicalCollectiveBool(mat,restrict_comm,4);
  PetscValidLogicalCollectiveBool(mat,restrict_full,5);
  PetscValidLogicalCollectiveBool(mat,reuse,6);
  PetscValidLogicalCollectiveInt(mat,nis,8);
  PetscValidLogicalCollectiveInt(mat,nvecs,10);
  if (nvecs) {
    if (nvecs > 1) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Just 1 vector supported");
    PetscValidHeaderSpecific(nnsp_vec[0],VEC_CLASSID,11);
  }
  /* further checks */
  ierr = MatISGetLocalMat(mat,&local_mat);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)local_mat,MATSEQDENSE,&isdense);CHKERRQ(ierr);
  if (!isdense) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Currently cannot subassemble MATIS when local matrix type is not of type SEQDENSE");
  ierr = MatGetSize(local_mat,&rows,&cols);CHKERRQ(ierr);
  if (rows != cols) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Local MATIS matrices should be square");
  if (reuse && *mat_n) {
    PetscInt mrows,mcols,mnrows,mncols;
    PetscValidHeaderSpecific(*mat_n,MAT_CLASSID,7);
    ierr = PetscObjectTypeCompare((PetscObject)*mat_n,MATIS,&ismatis);CHKERRQ(ierr);
    if (!ismatis) SETERRQ(PetscObjectComm((PetscObject)*mat_n),PETSC_ERR_SUP,"Cannot reuse a matrix which is not of type MATIS");
    ierr = MatGetSize(mat,&mrows,&mcols);CHKERRQ(ierr);
    ierr = MatGetSize(*mat_n,&mnrows,&mncols);CHKERRQ(ierr);
    if (mrows != mnrows) SETERRQ2(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix! Wrong number of rows %D != %D",mrows,mnrows);
    if (mcols != mncols) SETERRQ2(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix! Wrong number of cols %D != %D",mcols,mncols);
  }
  ierr = MatGetBlockSize(local_mat,&bs);CHKERRQ(ierr);
  PetscValidLogicalCollectiveInt(mat,bs,0);

  /* prepare IS for sending if not provided */
  if (!is_sends) {
    if (!n_subdomains) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"You should specify either an IS or a target number of subdomains");
    ierr = PCBDDCMatISGetSubassemblingPattern(mat,&n_subdomains,0,&is_sends_internal,NULL);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectReference((PetscObject)is_sends);CHKERRQ(ierr);
    is_sends_internal = is_sends;
  }

  /* get comm */
  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);

  /* compute number of sends */
  ierr = ISGetLocalSize(is_sends_internal,&i);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(i,&n_sends);CHKERRQ(ierr);

  /* compute number of receives */
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscMalloc1(size,&iflags);CHKERRQ(ierr);
  ierr = PetscArrayzero(iflags,size);CHKERRQ(ierr);
  ierr = ISGetIndices(is_sends_internal,&is_indices);CHKERRQ(ierr);
  for (i=0;i<n_sends;i++) iflags[is_indices[i]] = 1;
  ierr = PetscGatherNumberOfMessages(comm,iflags,NULL,&n_recvs);CHKERRQ(ierr);
  ierr = PetscFree(iflags);CHKERRQ(ierr);

  /* restrict comm if requested */
  subcomm = 0;
  destroy_mat = PETSC_FALSE;
  if (restrict_comm) {
    PetscMPIInt color,subcommsize;

    color = 0;
    if (restrict_full) {
      if (!n_recvs) color = 1; /* processes not receiving anything will not partecipate in new comm (full restriction) */
    } else {
      if (!n_recvs && n_sends) color = 1; /* just those processes that are sending but not receiving anything will not partecipate in new comm */
    }
    ierr = MPIU_Allreduce(&color,&subcommsize,1,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
    subcommsize = size - subcommsize;
    /* check if reuse has been requested */
    if (reuse) {
      if (*mat_n) {
        PetscMPIInt subcommsize2;
        ierr = MPI_Comm_size(PetscObjectComm((PetscObject)*mat_n),&subcommsize2);CHKERRQ(ierr);
        if (subcommsize != subcommsize2) SETERRQ2(PetscObjectComm((PetscObject)*mat_n),PETSC_ERR_PLIB,"Cannot reuse matrix! wrong subcomm size %d != %d",subcommsize,subcommsize2);
        comm_n = PetscObjectComm((PetscObject)*mat_n);
      } else {
        comm_n = PETSC_COMM_SELF;
      }
    } else { /* MAT_INITIAL_MATRIX */
      PetscMPIInt rank;

      ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
      ierr = PetscSubcommCreate(comm,&subcomm);CHKERRQ(ierr);
      ierr = PetscSubcommSetNumber(subcomm,2);CHKERRQ(ierr);
      ierr = PetscSubcommSetTypeGeneral(subcomm,color,rank);CHKERRQ(ierr);
      comm_n = PetscSubcommChild(subcomm);
    }
    /* flag to destroy *mat_n if not significative */
    if (color) destroy_mat = PETSC_TRUE;
  } else {
    comm_n = comm;
  }

  /* prepare send/receive buffers */
  ierr = PetscMalloc1(size,&ilengths_idxs);CHKERRQ(ierr);
  ierr = PetscArrayzero(ilengths_idxs,size);CHKERRQ(ierr);
  ierr = PetscMalloc1(size,&ilengths_vals);CHKERRQ(ierr);
  ierr = PetscArrayzero(ilengths_vals,size);CHKERRQ(ierr);
  if (nis) {
    ierr = PetscCalloc1(size,&ilengths_idxs_is);CHKERRQ(ierr);
  }

  /* Get data from local matrices */
  if (!isdense) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Subassembling of AIJ local matrices not yet implemented");
    /* TODO: See below some guidelines on how to prepare the local buffers */
    /*
       send_buffer_vals should contain the raw values of the local matrix
       send_buffer_idxs should contain:
       - MatType_PRIVATE type
       - PetscInt        size_of_l2gmap
       - PetscInt        global_row_indices[size_of_l2gmap]
       - PetscInt        all_other_info_which_is_needed_to_compute_preallocation_and_set_values
    */
  else {
    ierr = MatDenseGetArrayRead(local_mat,&send_buffer_vals);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetSize(mat->rmap->mapping,&i);CHKERRQ(ierr);
    ierr = PetscMalloc1(i+2,&send_buffer_idxs);CHKERRQ(ierr);
    send_buffer_idxs[0] = (PetscInt)MATDENSE_PRIVATE;
    send_buffer_idxs[1] = i;
    ierr = ISLocalToGlobalMappingGetIndices(mat->rmap->mapping,(const PetscInt**)&ptr_idxs);CHKERRQ(ierr);
    ierr = PetscArraycpy(&send_buffer_idxs[2],ptr_idxs,i);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingRestoreIndices(mat->rmap->mapping,(const PetscInt**)&ptr_idxs);CHKERRQ(ierr);
    ierr = PetscMPIIntCast(i,&len);CHKERRQ(ierr);
    for (i=0;i<n_sends;i++) {
      ilengths_vals[is_indices[i]] = len*len;
      ilengths_idxs[is_indices[i]] = len+2;
    }
  }
  ierr = PetscGatherMessageLengths2(comm,n_sends,n_recvs,ilengths_idxs,ilengths_vals,&onodes,&olengths_idxs,&olengths_vals);CHKERRQ(ierr);
  /* additional is (if any) */
  if (nis) {
    PetscMPIInt psum;
    PetscInt j;
    for (j=0,psum=0;j<nis;j++) {
      PetscInt plen;
      ierr = ISGetLocalSize(isarray[j],&plen);CHKERRQ(ierr);
      ierr = PetscMPIIntCast(plen,&len);CHKERRQ(ierr);
      psum += len+1; /* indices + lenght */
    }
    ierr = PetscMalloc1(psum,&send_buffer_idxs_is);CHKERRQ(ierr);
    for (j=0,psum=0;j<nis;j++) {
      PetscInt plen;
      const PetscInt *is_array_idxs;
      ierr = ISGetLocalSize(isarray[j],&plen);CHKERRQ(ierr);
      send_buffer_idxs_is[psum] = plen;
      ierr = ISGetIndices(isarray[j],&is_array_idxs);CHKERRQ(ierr);
      ierr = PetscArraycpy(&send_buffer_idxs_is[psum+1],is_array_idxs,plen);CHKERRQ(ierr);
      ierr = ISRestoreIndices(isarray[j],&is_array_idxs);CHKERRQ(ierr);
      psum += plen+1; /* indices + lenght */
    }
    for (i=0;i<n_sends;i++) {
      ilengths_idxs_is[is_indices[i]] = psum;
    }
    ierr = PetscGatherMessageLengths(comm,n_sends,n_recvs,ilengths_idxs_is,&onodes_is,&olengths_idxs_is);CHKERRQ(ierr);
  }
  ierr = MatISRestoreLocalMat(mat,&local_mat);CHKERRQ(ierr);

  buf_size_idxs = 0;
  buf_size_vals = 0;
  buf_size_idxs_is = 0;
  buf_size_vecs = 0;
  for (i=0;i<n_recvs;i++) {
    buf_size_idxs += (PetscInt)olengths_idxs[i];
    buf_size_vals += (PetscInt)olengths_vals[i];
    if (nis) buf_size_idxs_is += (PetscInt)olengths_idxs_is[i];
    if (nvecs) buf_size_vecs += (PetscInt)olengths_idxs[i];
  }
  ierr = PetscMalloc1(buf_size_idxs,&recv_buffer_idxs);CHKERRQ(ierr);
  ierr = PetscMalloc1(buf_size_vals,&recv_buffer_vals);CHKERRQ(ierr);
  ierr = PetscMalloc1(buf_size_idxs_is,&recv_buffer_idxs_is);CHKERRQ(ierr);
  ierr = PetscMalloc1(buf_size_vecs,&recv_buffer_vecs);CHKERRQ(ierr);

  /* get new tags for clean communications */
  ierr = PetscObjectGetNewTag((PetscObject)mat,&tag_idxs);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)mat,&tag_vals);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)mat,&tag_idxs_is);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)mat,&tag_vecs);CHKERRQ(ierr);

  /* allocate for requests */
  ierr = PetscMalloc1(n_sends,&send_req_idxs);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_sends,&send_req_vals);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_sends,&send_req_idxs_is);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_sends,&send_req_vecs);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_recvs,&recv_req_idxs);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_recvs,&recv_req_vals);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_recvs,&recv_req_idxs_is);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_recvs,&recv_req_vecs);CHKERRQ(ierr);

  /* communications */
  ptr_idxs = recv_buffer_idxs;
  ptr_vals = recv_buffer_vals;
  ptr_idxs_is = recv_buffer_idxs_is;
  ptr_vecs = recv_buffer_vecs;
  for (i=0;i<n_recvs;i++) {
    source_dest = onodes[i];
    ierr = MPI_Irecv(ptr_idxs,olengths_idxs[i],MPIU_INT,source_dest,tag_idxs,comm,&recv_req_idxs[i]);CHKERRQ(ierr);
    ierr = MPI_Irecv(ptr_vals,olengths_vals[i],MPIU_SCALAR,source_dest,tag_vals,comm,&recv_req_vals[i]);CHKERRQ(ierr);
    ptr_idxs += olengths_idxs[i];
    ptr_vals += olengths_vals[i];
    if (nis) {
      source_dest = onodes_is[i];
      ierr = MPI_Irecv(ptr_idxs_is,olengths_idxs_is[i],MPIU_INT,source_dest,tag_idxs_is,comm,&recv_req_idxs_is[i]);CHKERRQ(ierr);
      ptr_idxs_is += olengths_idxs_is[i];
    }
    if (nvecs) {
      source_dest = onodes[i];
      ierr = MPI_Irecv(ptr_vecs,olengths_idxs[i]-2,MPIU_SCALAR,source_dest,tag_vecs,comm,&recv_req_vecs[i]);CHKERRQ(ierr);
      ptr_vecs += olengths_idxs[i]-2;
    }
  }
  for (i=0;i<n_sends;i++) {
    ierr = PetscMPIIntCast(is_indices[i],&source_dest);CHKERRQ(ierr);
    ierr = MPI_Isend(send_buffer_idxs,ilengths_idxs[source_dest],MPIU_INT,source_dest,tag_idxs,comm,&send_req_idxs[i]);CHKERRQ(ierr);
    ierr = MPI_Isend(send_buffer_vals,ilengths_vals[source_dest],MPIU_SCALAR,source_dest,tag_vals,comm,&send_req_vals[i]);CHKERRQ(ierr);
    if (nis) {
      ierr = MPI_Isend(send_buffer_idxs_is,ilengths_idxs_is[source_dest],MPIU_INT,source_dest,tag_idxs_is,comm,&send_req_idxs_is[i]);CHKERRQ(ierr);
    }
    if (nvecs) {
      ierr = VecGetArray(nnsp_vec[0],&send_buffer_vecs);CHKERRQ(ierr);
      ierr = MPI_Isend(send_buffer_vecs,ilengths_idxs[source_dest]-2,MPIU_SCALAR,source_dest,tag_vecs,comm,&send_req_vecs[i]);CHKERRQ(ierr);
    }
  }
  ierr = ISRestoreIndices(is_sends_internal,&is_indices);CHKERRQ(ierr);
  ierr = ISDestroy(&is_sends_internal);CHKERRQ(ierr);

  /* assemble new l2g map */
  ierr = MPI_Waitall(n_recvs,recv_req_idxs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ptr_idxs = recv_buffer_idxs;
  new_local_rows = 0;
  for (i=0;i<n_recvs;i++) {
    new_local_rows += *(ptr_idxs+1); /* second element is the local size of the l2gmap */
    ptr_idxs += olengths_idxs[i];
  }
  ierr = PetscMalloc1(new_local_rows,&l2gmap_indices);CHKERRQ(ierr);
  ptr_idxs = recv_buffer_idxs;
  new_local_rows = 0;
  for (i=0;i<n_recvs;i++) {
    ierr = PetscArraycpy(&l2gmap_indices[new_local_rows],ptr_idxs+2,*(ptr_idxs+1));CHKERRQ(ierr);
    new_local_rows += *(ptr_idxs+1); /* second element is the local size of the l2gmap */
    ptr_idxs += olengths_idxs[i];
  }
  ierr = PetscSortRemoveDupsInt(&new_local_rows,l2gmap_indices);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(comm_n,1,new_local_rows,l2gmap_indices,PETSC_COPY_VALUES,&l2gmap);CHKERRQ(ierr);
  ierr = PetscFree(l2gmap_indices);CHKERRQ(ierr);

  /* infer new local matrix type from received local matrices type */
  /* currently if all local matrices are of type X, then the resulting matrix will be of type X, except for the dense case */
  /* it also assumes that if the block size is set, than it is the same among all local matrices (see checks at the beginning of the function) */
  if (n_recvs) {
    MatTypePrivate new_local_type_private = (MatTypePrivate)send_buffer_idxs[0];
    ptr_idxs = recv_buffer_idxs;
    for (i=0;i<n_recvs;i++) {
      if ((PetscInt)new_local_type_private != *ptr_idxs) {
        new_local_type_private = MATAIJ_PRIVATE;
        break;
      }
      ptr_idxs += olengths_idxs[i];
    }
    switch (new_local_type_private) {
      case MATDENSE_PRIVATE:
        new_local_type = MATSEQAIJ;
        bs = 1;
        break;
      case MATAIJ_PRIVATE:
        new_local_type = MATSEQAIJ;
        bs = 1;
        break;
      case MATBAIJ_PRIVATE:
        new_local_type = MATSEQBAIJ;
        break;
      case MATSBAIJ_PRIVATE:
        new_local_type = MATSEQSBAIJ;
        break;
      default:
        SETERRQ2(comm,PETSC_ERR_SUP,"Unsupported private type %d in %s",new_local_type_private,PETSC_FUNCTION_NAME);
        break;
    }
  } else { /* by default, new_local_type is seqaij */
    new_local_type = MATSEQAIJ;
    bs = 1;
  }

  /* create MATIS object if needed */
  if (!reuse) {
    ierr = MatGetSize(mat,&rows,&cols);CHKERRQ(ierr);
    ierr = MatCreateIS(comm_n,bs,PETSC_DECIDE,PETSC_DECIDE,rows,cols,l2gmap,NULL,mat_n);CHKERRQ(ierr);
  } else {
    /* it also destroys the local matrices */
    if (*mat_n) {
      ierr = MatSetLocalToGlobalMapping(*mat_n,l2gmap,l2gmap);CHKERRQ(ierr);
    } else { /* this is a fake object */
      ierr = MatCreateIS(comm_n,bs,PETSC_DECIDE,PETSC_DECIDE,rows,cols,l2gmap,NULL,mat_n);CHKERRQ(ierr);
    }
  }
  ierr = MatISGetLocalMat(*mat_n,&local_mat);CHKERRQ(ierr);
  ierr = MatSetType(local_mat,new_local_type);CHKERRQ(ierr);

  ierr = MPI_Waitall(n_recvs,recv_req_vals,MPI_STATUSES_IGNORE);CHKERRQ(ierr);

  /* Global to local map of received indices */
  ierr = PetscMalloc1(buf_size_idxs,&recv_buffer_idxs_local);CHKERRQ(ierr); /* needed for values insertion */
  ierr = ISGlobalToLocalMappingApply(l2gmap,IS_GTOLM_MASK,buf_size_idxs,recv_buffer_idxs,&i,recv_buffer_idxs_local);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&l2gmap);CHKERRQ(ierr);

  /* restore attributes -> type of incoming data and its size */
  buf_size_idxs = 0;
  for (i=0;i<n_recvs;i++) {
    recv_buffer_idxs_local[buf_size_idxs] = recv_buffer_idxs[buf_size_idxs];
    recv_buffer_idxs_local[buf_size_idxs+1] = recv_buffer_idxs[buf_size_idxs+1];
    buf_size_idxs += (PetscInt)olengths_idxs[i];
  }
  ierr = PetscFree(recv_buffer_idxs);CHKERRQ(ierr);

  /* set preallocation */
  ierr = PetscObjectTypeCompare((PetscObject)local_mat,MATSEQDENSE,&newisdense);CHKERRQ(ierr);
  if (!newisdense) {
    PetscInt *new_local_nnz=0;

    ptr_idxs = recv_buffer_idxs_local;
    if (n_recvs) {
      ierr = PetscCalloc1(new_local_rows,&new_local_nnz);CHKERRQ(ierr);
    }
    for (i=0;i<n_recvs;i++) {
      PetscInt j;
      if (*ptr_idxs == (PetscInt)MATDENSE_PRIVATE) { /* preallocation provided for dense case only */
        for (j=0;j<*(ptr_idxs+1);j++) {
          new_local_nnz[*(ptr_idxs+2+j)] += *(ptr_idxs+1);
        }
      } else {
        /* TODO */
      }
      ptr_idxs += olengths_idxs[i];
    }
    if (new_local_nnz) {
      for (i=0;i<new_local_rows;i++) new_local_nnz[i] = PetscMin(new_local_nnz[i],new_local_rows);
      ierr = MatSeqAIJSetPreallocation(local_mat,0,new_local_nnz);CHKERRQ(ierr);
      for (i=0;i<new_local_rows;i++) new_local_nnz[i] /= bs;
      ierr = MatSeqBAIJSetPreallocation(local_mat,bs,0,new_local_nnz);CHKERRQ(ierr);
      for (i=0;i<new_local_rows;i++) new_local_nnz[i] = PetscMax(new_local_nnz[i]-i,0);
      ierr = MatSeqSBAIJSetPreallocation(local_mat,bs,0,new_local_nnz);CHKERRQ(ierr);
    } else {
      ierr = MatSetUp(local_mat);CHKERRQ(ierr);
    }
    ierr = PetscFree(new_local_nnz);CHKERRQ(ierr);
  } else {
    ierr = MatSetUp(local_mat);CHKERRQ(ierr);
  }

  /* set values */
  ptr_vals = recv_buffer_vals;
  ptr_idxs = recv_buffer_idxs_local;
  for (i=0;i<n_recvs;i++) {
    if (*ptr_idxs == (PetscInt)MATDENSE_PRIVATE) { /* values insertion provided for dense case only */
      ierr = MatSetOption(local_mat,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatSetValues(local_mat,*(ptr_idxs+1),ptr_idxs+2,*(ptr_idxs+1),ptr_idxs+2,ptr_vals,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(local_mat,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(local_mat,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatSetOption(local_mat,MAT_ROW_ORIENTED,PETSC_TRUE);CHKERRQ(ierr);
    } else {
      /* TODO */
    }
    ptr_idxs += olengths_idxs[i];
    ptr_vals += olengths_vals[i];
  }
  ierr = MatAssemblyBegin(local_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(local_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatISRestoreLocalMat(*mat_n,&local_mat);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*mat_n,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat_n,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree(recv_buffer_vals);CHKERRQ(ierr);

#if 0
  if (!restrict_comm) { /* check */
    Vec       lvec,rvec;
    PetscReal infty_error;

    ierr = MatCreateVecs(mat,&rvec,&lvec);CHKERRQ(ierr);
    ierr = VecSetRandom(rvec,NULL);CHKERRQ(ierr);
    ierr = MatMult(mat,rvec,lvec);CHKERRQ(ierr);
    ierr = VecScale(lvec,-1.0);CHKERRQ(ierr);
    ierr = MatMultAdd(*mat_n,rvec,lvec,lvec);CHKERRQ(ierr);
    ierr = VecNorm(lvec,NORM_INFINITY,&infty_error);CHKERRQ(ierr);
    ierr = PetscPrintf(PetscObjectComm((PetscObject)mat),"Infinity error subassembling %1.6e\n",infty_error);
    ierr = VecDestroy(&rvec);CHKERRQ(ierr);
    ierr = VecDestroy(&lvec);CHKERRQ(ierr);
  }
#endif

  /* assemble new additional is (if any) */
  if (nis) {
    PetscInt **temp_idxs,*count_is,j,psum;

    ierr = MPI_Waitall(n_recvs,recv_req_idxs_is,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    ierr = PetscCalloc1(nis,&count_is);CHKERRQ(ierr);
    ptr_idxs = recv_buffer_idxs_is;
    psum = 0;
    for (i=0;i<n_recvs;i++) {
      for (j=0;j<nis;j++) {
        PetscInt plen = *(ptr_idxs); /* first element is the local size of IS's indices */
        count_is[j] += plen; /* increment counting of buffer for j-th IS */
        psum += plen;
        ptr_idxs += plen+1; /* shift pointer to received data */
      }
    }
    ierr = PetscMalloc1(nis,&temp_idxs);CHKERRQ(ierr);
    ierr = PetscMalloc1(psum,&temp_idxs[0]);CHKERRQ(ierr);
    for (i=1;i<nis;i++) {
      temp_idxs[i] = temp_idxs[i-1]+count_is[i-1];
    }
    ierr = PetscArrayzero(count_is,nis);CHKERRQ(ierr);
    ptr_idxs = recv_buffer_idxs_is;
    for (i=0;i<n_recvs;i++) {
      for (j=0;j<nis;j++) {
        PetscInt plen = *(ptr_idxs); /* first element is the local size of IS's indices */
        ierr = PetscArraycpy(&temp_idxs[j][count_is[j]],ptr_idxs+1,plen);CHKERRQ(ierr);
        count_is[j] += plen; /* increment starting point of buffer for j-th IS */
        ptr_idxs += plen+1; /* shift pointer to received data */
      }
    }
    for (i=0;i<nis;i++) {
      ierr = ISDestroy(&isarray[i]);CHKERRQ(ierr);
      ierr = PetscSortRemoveDupsInt(&count_is[i],temp_idxs[i]);CHKERRQ(ierr);CHKERRQ(ierr);
      ierr = ISCreateGeneral(comm_n,count_is[i],temp_idxs[i],PETSC_COPY_VALUES,&isarray[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(count_is);CHKERRQ(ierr);
    ierr = PetscFree(temp_idxs[0]);CHKERRQ(ierr);
    ierr = PetscFree(temp_idxs);CHKERRQ(ierr);
  }
  /* free workspace */
  ierr = PetscFree(recv_buffer_idxs_is);CHKERRQ(ierr);
  ierr = MPI_Waitall(n_sends,send_req_idxs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = PetscFree(send_buffer_idxs);CHKERRQ(ierr);
  ierr = MPI_Waitall(n_sends,send_req_vals,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  if (isdense) {
    ierr = MatISGetLocalMat(mat,&local_mat);CHKERRQ(ierr);
    ierr = MatDenseRestoreArrayRead(local_mat,&send_buffer_vals);CHKERRQ(ierr);
    ierr = MatISRestoreLocalMat(mat,&local_mat);CHKERRQ(ierr);
  } else {
    /* ierr = PetscFree(send_buffer_vals);CHKERRQ(ierr); */
  }
  if (nis) {
    ierr = MPI_Waitall(n_sends,send_req_idxs_is,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    ierr = PetscFree(send_buffer_idxs_is);CHKERRQ(ierr);
  }

  if (nvecs) {
    ierr = MPI_Waitall(n_recvs,recv_req_vecs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    ierr = MPI_Waitall(n_sends,send_req_vecs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    ierr = VecRestoreArray(nnsp_vec[0],&send_buffer_vecs);CHKERRQ(ierr);
    ierr = VecDestroy(&nnsp_vec[0]);CHKERRQ(ierr);
    ierr = VecCreate(comm_n,&nnsp_vec[0]);CHKERRQ(ierr);
    ierr = VecSetSizes(nnsp_vec[0],new_local_rows,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetType(nnsp_vec[0],VECSTANDARD);CHKERRQ(ierr);
    /* set values */
    ptr_vals = recv_buffer_vecs;
    ptr_idxs = recv_buffer_idxs_local;
    ierr = VecGetArray(nnsp_vec[0],&send_buffer_vecs);CHKERRQ(ierr);
    for (i=0;i<n_recvs;i++) {
      PetscInt j;
      for (j=0;j<*(ptr_idxs+1);j++) {
        send_buffer_vecs[*(ptr_idxs+2+j)] += *(ptr_vals + j);
      }
      ptr_idxs += olengths_idxs[i];
      ptr_vals += olengths_idxs[i]-2;
    }
    ierr = VecRestoreArray(nnsp_vec[0],&send_buffer_vecs);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(nnsp_vec[0]);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(nnsp_vec[0]);CHKERRQ(ierr);
  }

  ierr = PetscFree(recv_buffer_vecs);CHKERRQ(ierr);
  ierr = PetscFree(recv_buffer_idxs_local);CHKERRQ(ierr);
  ierr = PetscFree(recv_req_idxs);CHKERRQ(ierr);
  ierr = PetscFree(recv_req_vals);CHKERRQ(ierr);
  ierr = PetscFree(recv_req_vecs);CHKERRQ(ierr);
  ierr = PetscFree(recv_req_idxs_is);CHKERRQ(ierr);
  ierr = PetscFree(send_req_idxs);CHKERRQ(ierr);
  ierr = PetscFree(send_req_vals);CHKERRQ(ierr);
  ierr = PetscFree(send_req_vecs);CHKERRQ(ierr);
  ierr = PetscFree(send_req_idxs_is);CHKERRQ(ierr);
  ierr = PetscFree(ilengths_vals);CHKERRQ(ierr);
  ierr = PetscFree(ilengths_idxs);CHKERRQ(ierr);
  ierr = PetscFree(olengths_vals);CHKERRQ(ierr);
  ierr = PetscFree(olengths_idxs);CHKERRQ(ierr);
  ierr = PetscFree(onodes);CHKERRQ(ierr);
  if (nis) {
    ierr = PetscFree(ilengths_idxs_is);CHKERRQ(ierr);
    ierr = PetscFree(olengths_idxs_is);CHKERRQ(ierr);
    ierr = PetscFree(onodes_is);CHKERRQ(ierr);
  }
  ierr = PetscSubcommDestroy(&subcomm);CHKERRQ(ierr);
  if (destroy_mat) { /* destroy mat is true only if restrict comm is true and process will not partecipate */
    ierr = MatDestroy(mat_n);CHKERRQ(ierr);
    for (i=0;i<nis;i++) {
      ierr = ISDestroy(&isarray[i]);CHKERRQ(ierr);
    }
    if (nvecs) { /* need to match VecDestroy nnsp_vec called in the other code path */
      ierr = VecDestroy(&nnsp_vec[0]);CHKERRQ(ierr);
    }
    *mat_n = NULL;
  }
  PetscFunctionReturn(0);
}

/* temporary hack into ksp private data structure */
#include <petsc/private/kspimpl.h>

PetscErrorCode PCBDDCSetUpCoarseSolver(PC pc,PetscScalar* coarse_submat_vals)
{
  PC_BDDC                *pcbddc = (PC_BDDC*)pc->data;
  PC_IS                  *pcis = (PC_IS*)pc->data;
  Mat                    coarse_mat,coarse_mat_is,coarse_submat_dense;
  Mat                    coarsedivudotp = NULL;
  Mat                    coarseG,t_coarse_mat_is;
  MatNullSpace           CoarseNullSpace = NULL;
  ISLocalToGlobalMapping coarse_islg;
  IS                     coarse_is,*isarray,corners;
  PetscInt               i,im_active=-1,active_procs=-1;
  PetscInt               nis,nisdofs,nisneu,nisvert;
  PetscInt               coarse_eqs_per_proc;
  PC                     pc_temp;
  PCType                 coarse_pc_type;
  KSPType                coarse_ksp_type;
  PetscBool              multilevel_requested,multilevel_allowed;
  PetscBool              coarse_reuse;
  PetscInt               ncoarse,nedcfield;
  PetscBool              compute_vecs = PETSC_FALSE;
  PetscScalar            *array;
  MatReuse               coarse_mat_reuse;
  PetscBool              restr, full_restr, have_void;
  PetscMPIInt            size;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PC_BDDC_CoarseSetUp[pcbddc->current_level],pc,0,0,0);CHKERRQ(ierr);
  /* Assign global numbering to coarse dofs */
  if (pcbddc->new_primal_space || pcbddc->coarse_size == -1) { /* a new primal space is present or it is the first initialization, so recompute global numbering */
    PetscInt ocoarse_size;
    compute_vecs = PETSC_TRUE;

    pcbddc->new_primal_space = PETSC_TRUE;
    ocoarse_size = pcbddc->coarse_size;
    ierr = PetscFree(pcbddc->global_primal_indices);CHKERRQ(ierr);
    ierr = PCBDDCComputePrimalNumbering(pc,&pcbddc->coarse_size,&pcbddc->global_primal_indices);CHKERRQ(ierr);
    /* see if we can avoid some work */
    if (pcbddc->coarse_ksp) { /* coarse ksp has already been created */
      /* if the coarse size is different or we are using adaptive selection, better to not reuse the coarse matrix */
      if (ocoarse_size != pcbddc->coarse_size || pcbddc->adaptive_selection) {
        ierr = KSPReset(pcbddc->coarse_ksp);CHKERRQ(ierr);
        coarse_reuse = PETSC_FALSE;
      } else { /* we can safely reuse already computed coarse matrix */
        coarse_reuse = PETSC_TRUE;
      }
    } else { /* there's no coarse ksp, so we need to create the coarse matrix too */
      coarse_reuse = PETSC_FALSE;
    }
    /* reset any subassembling information */
    if (!coarse_reuse || pcbddc->recompute_topography) {
      ierr = ISDestroy(&pcbddc->coarse_subassembling);CHKERRQ(ierr);
    }
  } else { /* primal space is unchanged, so we can reuse coarse matrix */
    coarse_reuse = PETSC_TRUE;
  }
  if (coarse_reuse && pcbddc->coarse_ksp) {
    ierr = KSPGetOperators(pcbddc->coarse_ksp,&coarse_mat,NULL);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)coarse_mat);CHKERRQ(ierr);
    coarse_mat_reuse = MAT_REUSE_MATRIX;
  } else {
    coarse_mat = NULL;
    coarse_mat_reuse = MAT_INITIAL_MATRIX;
  }

  /* creates temporary l2gmap and IS for coarse indexes */
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pc),pcbddc->local_primal_size,pcbddc->global_primal_indices,PETSC_COPY_VALUES,&coarse_is);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(coarse_is,&coarse_islg);CHKERRQ(ierr);

  /* creates temporary MATIS object for coarse matrix */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,pcbddc->local_primal_size,pcbddc->local_primal_size,coarse_submat_vals,&coarse_submat_dense);CHKERRQ(ierr);
  ierr = MatCreateIS(PetscObjectComm((PetscObject)pc),1,PETSC_DECIDE,PETSC_DECIDE,pcbddc->coarse_size,pcbddc->coarse_size,coarse_islg,NULL,&t_coarse_mat_is);CHKERRQ(ierr);
  ierr = MatISSetLocalMat(t_coarse_mat_is,coarse_submat_dense);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(t_coarse_mat_is,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(t_coarse_mat_is,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatDestroy(&coarse_submat_dense);CHKERRQ(ierr);

  /* count "active" (i.e. with positive local size) and "void" processes */
  im_active = !!(pcis->n);
  ierr = MPIU_Allreduce(&im_active,&active_procs,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);

  /* determine number of processes partecipating to coarse solver and compute subassembling pattern */
  /* restr : whether we want to exclude senders (which are not receivers) from the subassembling pattern */
  /* full_restr : just use the receivers from the subassembling pattern */
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size);CHKERRQ(ierr);
  coarse_mat_is        = NULL;
  multilevel_allowed   = PETSC_FALSE;
  multilevel_requested = PETSC_FALSE;
  coarse_eqs_per_proc  = PetscMin(PetscMax(pcbddc->coarse_size,1),pcbddc->coarse_eqs_per_proc);
  if (coarse_eqs_per_proc < 0) coarse_eqs_per_proc = pcbddc->coarse_size;
  if (pcbddc->current_level < pcbddc->max_levels) multilevel_requested = PETSC_TRUE;
  if (pcbddc->coarse_size <= pcbddc->coarse_eqs_limit) multilevel_requested = PETSC_FALSE;
  if (multilevel_requested) {
    ncoarse    = active_procs/pcbddc->coarsening_ratio;
    restr      = PETSC_FALSE;
    full_restr = PETSC_FALSE;
  } else {
    ncoarse    = pcbddc->coarse_size/coarse_eqs_per_proc + !!(pcbddc->coarse_size%coarse_eqs_per_proc);
    restr      = PETSC_TRUE;
    full_restr = PETSC_TRUE;
  }
  if (!pcbddc->coarse_size || size == 1) multilevel_allowed = multilevel_requested = restr = full_restr = PETSC_FALSE;
  ncoarse = PetscMax(1,ncoarse);
  if (!pcbddc->coarse_subassembling) {
    if (pcbddc->coarsening_ratio > 1) {
      if (multilevel_requested) {
        ierr = PCBDDCMatISGetSubassemblingPattern(pc->pmat,&ncoarse,pcbddc->coarse_adj_red,&pcbddc->coarse_subassembling,&have_void);CHKERRQ(ierr);
      } else {
        ierr = PCBDDCMatISGetSubassemblingPattern(t_coarse_mat_is,&ncoarse,pcbddc->coarse_adj_red,&pcbddc->coarse_subassembling,&have_void);CHKERRQ(ierr);
      }
    } else {
      PetscMPIInt rank;

      ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank);CHKERRQ(ierr);
      have_void = (active_procs == (PetscInt)size) ? PETSC_FALSE : PETSC_TRUE;
      ierr = ISCreateStride(PetscObjectComm((PetscObject)pc),1,rank,1,&pcbddc->coarse_subassembling);CHKERRQ(ierr);
    }
  } else { /* if a subassembling pattern exists, then we can reuse the coarse ksp and compute the number of process involved */
    PetscInt    psum;
    if (pcbddc->coarse_ksp) psum = 1;
    else psum = 0;
    ierr = MPIU_Allreduce(&psum,&ncoarse,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
    have_void = ncoarse < size ? PETSC_TRUE : PETSC_FALSE;
  }
  /* determine if we can go multilevel */
  if (multilevel_requested) {
    if (ncoarse > 1) multilevel_allowed = PETSC_TRUE; /* found enough processes */
    else restr = full_restr = PETSC_TRUE; /* 1 subdomain, use a direct solver */
  }
  if (multilevel_allowed && have_void) restr = PETSC_TRUE;

  /* dump subassembling pattern */
  if (pcbddc->dbg_flag && multilevel_allowed) {
    ierr = ISView(pcbddc->coarse_subassembling,pcbddc->dbg_viewer);CHKERRQ(ierr);
  }
  /* compute dofs splitting and neumann boundaries for coarse dofs */
  nedcfield = -1;
  corners = NULL;
  if (multilevel_allowed && !coarse_reuse && (pcbddc->n_ISForDofsLocal || pcbddc->NeumannBoundariesLocal || pcbddc->nedclocal || pcbddc->corner_selected)) { /* protects from unneeded computations */
    PetscInt               *tidxs,*tidxs2,nout,tsize,i;
    const PetscInt         *idxs;
    ISLocalToGlobalMapping tmap;

    /* create map between primal indices (in local representative ordering) and local primal numbering */
    ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,1,pcbddc->local_primal_size,pcbddc->primal_indices_local_idxs,PETSC_COPY_VALUES,&tmap);CHKERRQ(ierr);
    /* allocate space for temporary storage */
    ierr = PetscMalloc1(pcbddc->local_primal_size,&tidxs);CHKERRQ(ierr);
    ierr = PetscMalloc1(pcbddc->local_primal_size,&tidxs2);CHKERRQ(ierr);
    /* allocate for IS array */
    nisdofs = pcbddc->n_ISForDofsLocal;
    if (pcbddc->nedclocal) {
      if (pcbddc->nedfield > -1) {
        nedcfield = pcbddc->nedfield;
      } else {
        nedcfield = 0;
        if (nisdofs) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"This should not happen (%D)",nisdofs);
        nisdofs = 1;
      }
    }
    nisneu = !!pcbddc->NeumannBoundariesLocal;
    nisvert = 0; /* nisvert is not used */
    nis = nisdofs + nisneu + nisvert;
    ierr = PetscMalloc1(nis,&isarray);CHKERRQ(ierr);
    /* dofs splitting */
    for (i=0;i<nisdofs;i++) {
      /* ierr = ISView(pcbddc->ISForDofsLocal[i],0);CHKERRQ(ierr); */
      if (nedcfield != i) {
        ierr = ISGetLocalSize(pcbddc->ISForDofsLocal[i],&tsize);CHKERRQ(ierr);
        ierr = ISGetIndices(pcbddc->ISForDofsLocal[i],&idxs);CHKERRQ(ierr);
        ierr = ISGlobalToLocalMappingApply(tmap,IS_GTOLM_DROP,tsize,idxs,&nout,tidxs);CHKERRQ(ierr);
        ierr = ISRestoreIndices(pcbddc->ISForDofsLocal[i],&idxs);CHKERRQ(ierr);
      } else {
        ierr = ISGetLocalSize(pcbddc->nedclocal,&tsize);CHKERRQ(ierr);
        ierr = ISGetIndices(pcbddc->nedclocal,&idxs);CHKERRQ(ierr);
        ierr = ISGlobalToLocalMappingApply(tmap,IS_GTOLM_DROP,tsize,idxs,&nout,tidxs);CHKERRQ(ierr);
        if (tsize != nout) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Failed when mapping coarse nedelec field! %D != %D",tsize,nout);
        ierr = ISRestoreIndices(pcbddc->nedclocal,&idxs);CHKERRQ(ierr);
      }
      ierr = ISLocalToGlobalMappingApply(coarse_islg,nout,tidxs,tidxs2);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pc),nout,tidxs2,PETSC_COPY_VALUES,&isarray[i]);CHKERRQ(ierr);
      /* ierr = ISView(isarray[i],0);CHKERRQ(ierr); */
    }
    /* neumann boundaries */
    if (pcbddc->NeumannBoundariesLocal) {
      /* ierr = ISView(pcbddc->NeumannBoundariesLocal,0);CHKERRQ(ierr); */
      ierr = ISGetLocalSize(pcbddc->NeumannBoundariesLocal,&tsize);CHKERRQ(ierr);
      ierr = ISGetIndices(pcbddc->NeumannBoundariesLocal,&idxs);CHKERRQ(ierr);
      ierr = ISGlobalToLocalMappingApply(tmap,IS_GTOLM_DROP,tsize,idxs,&nout,tidxs);CHKERRQ(ierr);
      ierr = ISRestoreIndices(pcbddc->NeumannBoundariesLocal,&idxs);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingApply(coarse_islg,nout,tidxs,tidxs2);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pc),nout,tidxs2,PETSC_COPY_VALUES,&isarray[nisdofs]);CHKERRQ(ierr);
      /* ierr = ISView(isarray[nisdofs],0);CHKERRQ(ierr); */
    }
    /* coordinates */
    if (pcbddc->corner_selected) {
      ierr = PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&corners);CHKERRQ(ierr);
      ierr = ISGetLocalSize(corners,&tsize);CHKERRQ(ierr);
      ierr = ISGetIndices(corners,&idxs);CHKERRQ(ierr);
      ierr = ISGlobalToLocalMappingApply(tmap,IS_GTOLM_DROP,tsize,idxs,&nout,tidxs);CHKERRQ(ierr);
      if (tsize != nout) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Failed when mapping corners! %D != %D",tsize,nout);
      ierr = ISRestoreIndices(corners,&idxs);CHKERRQ(ierr);
      ierr = PCBDDCGraphRestoreCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&corners);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingApply(coarse_islg,nout,tidxs,tidxs2);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pc),nout,tidxs2,PETSC_COPY_VALUES,&corners);CHKERRQ(ierr);
    }
    ierr = PetscFree(tidxs);CHKERRQ(ierr);
    ierr = PetscFree(tidxs2);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&tmap);CHKERRQ(ierr);
  } else {
    nis = 0;
    nisdofs = 0;
    nisneu = 0;
    nisvert = 0;
    isarray = NULL;
  }
  /* destroy no longer needed map */
  ierr = ISLocalToGlobalMappingDestroy(&coarse_islg);CHKERRQ(ierr);

  /* subassemble */
  if (multilevel_allowed) {
    Vec       vp[1];
    PetscInt  nvecs = 0;
    PetscBool reuse,reuser;

    if (coarse_mat) reuse = PETSC_TRUE;
    else reuse = PETSC_FALSE;
    ierr = MPIU_Allreduce(&reuse,&reuser,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
    vp[0] = NULL;
    if (pcbddc->benign_have_null) { /* propagate no-net-flux quadrature to coarser level */
      ierr = VecCreate(PetscObjectComm((PetscObject)pc),&vp[0]);CHKERRQ(ierr);
      ierr = VecSetSizes(vp[0],pcbddc->local_primal_size,PETSC_DECIDE);CHKERRQ(ierr);
      ierr = VecSetType(vp[0],VECSTANDARD);CHKERRQ(ierr);
      nvecs = 1;

      if (pcbddc->divudotp) {
        Mat      B,loc_divudotp;
        Vec      v,p;
        IS       dummy;
        PetscInt np;

        ierr = MatISGetLocalMat(pcbddc->divudotp,&loc_divudotp);CHKERRQ(ierr);
        ierr = MatGetSize(loc_divudotp,&np,NULL);CHKERRQ(ierr);
        ierr = ISCreateStride(PETSC_COMM_SELF,np,0,1,&dummy);CHKERRQ(ierr);
        ierr = MatCreateSubMatrix(loc_divudotp,dummy,pcis->is_B_local,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
        ierr = MatCreateVecs(B,&v,&p);CHKERRQ(ierr);
        ierr = VecSet(p,1.);CHKERRQ(ierr);
        ierr = MatMultTranspose(B,p,v);CHKERRQ(ierr);
        ierr = VecDestroy(&p);CHKERRQ(ierr);
        ierr = MatDestroy(&B);CHKERRQ(ierr);
        ierr = VecGetArray(vp[0],&array);CHKERRQ(ierr);
        ierr = VecPlaceArray(pcbddc->vec1_P,array);CHKERRQ(ierr);
        ierr = VecRestoreArray(vp[0],&array);CHKERRQ(ierr);
        ierr = MatMultTranspose(pcbddc->coarse_phi_B,v,pcbddc->vec1_P);CHKERRQ(ierr);
        ierr = VecResetArray(pcbddc->vec1_P);CHKERRQ(ierr);
        ierr = ISDestroy(&dummy);CHKERRQ(ierr);
        ierr = VecDestroy(&v);CHKERRQ(ierr);
      }
    }
    if (reuser) {
      ierr = PCBDDCMatISSubassemble(t_coarse_mat_is,pcbddc->coarse_subassembling,0,restr,full_restr,PETSC_TRUE,&coarse_mat,nis,isarray,nvecs,vp);CHKERRQ(ierr);
    } else {
      ierr = PCBDDCMatISSubassemble(t_coarse_mat_is,pcbddc->coarse_subassembling,0,restr,full_restr,PETSC_FALSE,&coarse_mat_is,nis,isarray,nvecs,vp);CHKERRQ(ierr);
    }
    if (vp[0]) { /* vp[0] could have been placed on a different set of processes */
      PetscScalar       *arraym;
      const PetscScalar *arrayv;
      PetscInt          nl;
      ierr = VecGetLocalSize(vp[0],&nl);CHKERRQ(ierr);
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,1,nl,NULL,&coarsedivudotp);CHKERRQ(ierr);
      ierr = MatDenseGetArray(coarsedivudotp,&arraym);CHKERRQ(ierr);
      ierr = VecGetArrayRead(vp[0],&arrayv);CHKERRQ(ierr);
      ierr = PetscArraycpy(arraym,arrayv,nl);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(vp[0],&arrayv);CHKERRQ(ierr);
      ierr = MatDenseRestoreArray(coarsedivudotp,&arraym);CHKERRQ(ierr);
      ierr = VecDestroy(&vp[0]);CHKERRQ(ierr);
    } else {
      ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,0,0,1,NULL,&coarsedivudotp);CHKERRQ(ierr);
    }
  } else {
    ierr = PCBDDCMatISSubassemble(t_coarse_mat_is,pcbddc->coarse_subassembling,0,restr,full_restr,PETSC_FALSE,&coarse_mat_is,0,NULL,0,NULL);CHKERRQ(ierr);
  }
  if (coarse_mat_is || coarse_mat) {
    if (!multilevel_allowed) {
      ierr = MatConvert(coarse_mat_is,MATAIJ,coarse_mat_reuse,&coarse_mat);CHKERRQ(ierr);
    } else {
      /* if this matrix is present, it means we are not reusing the coarse matrix */
      if (coarse_mat_is) {
        if (coarse_mat) SETERRQ(PetscObjectComm((PetscObject)coarse_mat_is),PETSC_ERR_PLIB,"This should not happen");
        ierr = PetscObjectReference((PetscObject)coarse_mat_is);CHKERRQ(ierr);
        coarse_mat = coarse_mat_is;
      }
    }
  }
  ierr = MatDestroy(&t_coarse_mat_is);CHKERRQ(ierr);
  ierr = MatDestroy(&coarse_mat_is);CHKERRQ(ierr);

  /* create local to global scatters for coarse problem */
  if (compute_vecs) {
    PetscInt lrows;
    ierr = VecDestroy(&pcbddc->coarse_vec);CHKERRQ(ierr);
    if (coarse_mat) {
      ierr = MatGetLocalSize(coarse_mat,&lrows,NULL);CHKERRQ(ierr);
    } else {
      lrows = 0;
    }
    ierr = VecCreate(PetscObjectComm((PetscObject)pc),&pcbddc->coarse_vec);CHKERRQ(ierr);
    ierr = VecSetSizes(pcbddc->coarse_vec,lrows,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetType(pcbddc->coarse_vec,coarse_mat ? coarse_mat->defaultvectype : VECSTANDARD);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&pcbddc->coarse_loc_to_glob);CHKERRQ(ierr);
    ierr = VecScatterCreate(pcbddc->vec1_P,NULL,pcbddc->coarse_vec,coarse_is,&pcbddc->coarse_loc_to_glob);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&coarse_is);CHKERRQ(ierr);

  /* set defaults for coarse KSP and PC */
  if (multilevel_allowed) {
    coarse_ksp_type = KSPRICHARDSON;
    coarse_pc_type  = PCBDDC;
  } else {
    coarse_ksp_type = KSPPREONLY;
    coarse_pc_type  = PCREDUNDANT;
  }

  /* print some info if requested */
  if (pcbddc->dbg_flag) {
    if (!multilevel_allowed) {
      ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
      if (multilevel_requested) {
        ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Not enough active processes on level %D (active processes %D, coarsening ratio %D)\n",pcbddc->current_level,active_procs,pcbddc->coarsening_ratio);CHKERRQ(ierr);
      } else if (pcbddc->max_levels) {
        ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Maximum number of requested levels reached (%D)\n",pcbddc->max_levels);CHKERRQ(ierr);
      }
      ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    }
  }

  /* communicate coarse discrete gradient */
  coarseG = NULL;
  if (pcbddc->nedcG && multilevel_allowed) {
    MPI_Comm ccomm;
    if (coarse_mat) {
      ccomm = PetscObjectComm((PetscObject)coarse_mat);
    } else {
      ccomm = MPI_COMM_NULL;
    }
    ierr = MatMPIAIJRestrict(pcbddc->nedcG,ccomm,&coarseG);CHKERRQ(ierr);
  }

  /* create the coarse KSP object only once with defaults */
  if (coarse_mat) {
    PetscBool   isredundant,isbddc,force,valid;
    PetscViewer dbg_viewer = NULL;

    if (pcbddc->dbg_flag) {
      dbg_viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)coarse_mat));
      ierr = PetscViewerASCIIAddTab(dbg_viewer,2*pcbddc->current_level);CHKERRQ(ierr);
    }
    if (!pcbddc->coarse_ksp) {
      char   prefix[256],str_level[16];
      size_t len;

      ierr = KSPCreate(PetscObjectComm((PetscObject)coarse_mat),&pcbddc->coarse_ksp);CHKERRQ(ierr);
      ierr = KSPSetErrorIfNotConverged(pcbddc->coarse_ksp,pc->erroriffailure);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)pcbddc->coarse_ksp,(PetscObject)pc,1);CHKERRQ(ierr);
      ierr = KSPSetTolerances(pcbddc->coarse_ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,1);CHKERRQ(ierr);
      ierr = KSPSetOperators(pcbddc->coarse_ksp,coarse_mat,coarse_mat);CHKERRQ(ierr);
      ierr = KSPSetType(pcbddc->coarse_ksp,coarse_ksp_type);CHKERRQ(ierr);
      ierr = KSPSetNormType(pcbddc->coarse_ksp,KSP_NORM_NONE);CHKERRQ(ierr);
      ierr = KSPGetPC(pcbddc->coarse_ksp,&pc_temp);CHKERRQ(ierr);
      /* TODO is this logic correct? should check for coarse_mat type */
      ierr = PCSetType(pc_temp,coarse_pc_type);CHKERRQ(ierr);
      /* prefix */
      ierr = PetscStrcpy(prefix,"");CHKERRQ(ierr);
      ierr = PetscStrcpy(str_level,"");CHKERRQ(ierr);
      if (!pcbddc->current_level) {
        ierr = PetscStrncpy(prefix,((PetscObject)pc)->prefix,sizeof(prefix));CHKERRQ(ierr);
        ierr = PetscStrlcat(prefix,"pc_bddc_coarse_",sizeof(prefix));CHKERRQ(ierr);
      } else {
        ierr = PetscStrlen(((PetscObject)pc)->prefix,&len);CHKERRQ(ierr);
        if (pcbddc->current_level>1) len -= 3; /* remove "lX_" with X level number */
        if (pcbddc->current_level>10) len -= 1; /* remove another char from level number */
        /* Nonstandard use of PetscStrncpy() to copy only a portion of the string */
        ierr = PetscStrncpy(prefix,((PetscObject)pc)->prefix,len+1);CHKERRQ(ierr);
        ierr = PetscSNPrintf(str_level,sizeof(str_level),"l%d_",(int)(pcbddc->current_level));CHKERRQ(ierr);
        ierr = PetscStrlcat(prefix,str_level,sizeof(prefix));CHKERRQ(ierr);
      }
      ierr = KSPSetOptionsPrefix(pcbddc->coarse_ksp,prefix);CHKERRQ(ierr);
      /* propagate BDDC info to the next level (these are dummy calls if pc_temp is not of type PCBDDC) */
      ierr = PCBDDCSetLevel(pc_temp,pcbddc->current_level+1);CHKERRQ(ierr);
      ierr = PCBDDCSetCoarseningRatio(pc_temp,pcbddc->coarsening_ratio);CHKERRQ(ierr);
      ierr = PCBDDCSetLevels(pc_temp,pcbddc->max_levels);CHKERRQ(ierr);
      /* allow user customization */
      ierr = KSPSetFromOptions(pcbddc->coarse_ksp);CHKERRQ(ierr);
      /* get some info after set from options */
      ierr = KSPGetPC(pcbddc->coarse_ksp,&pc_temp);CHKERRQ(ierr);
      /* multilevel cannot be done with coarse PC different from BDDC, NN, HPDDM, unless forced to */
      force = PETSC_FALSE;
      ierr = PetscOptionsGetBool(NULL,((PetscObject)pc_temp)->prefix,"-pc_type_forced",&force,NULL);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompareAny((PetscObject)pc_temp,&valid,PCBDDC,PCNN,PCHPDDM,"");CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)pc_temp,PCBDDC,&isbddc);CHKERRQ(ierr);
      if (multilevel_allowed && !force && !valid) {
        isbddc = PETSC_TRUE;
        ierr   = PCSetType(pc_temp,PCBDDC);CHKERRQ(ierr);
        ierr   = PCBDDCSetLevel(pc_temp,pcbddc->current_level+1);CHKERRQ(ierr);
        ierr   = PCBDDCSetCoarseningRatio(pc_temp,pcbddc->coarsening_ratio);CHKERRQ(ierr);
        ierr   = PCBDDCSetLevels(pc_temp,pcbddc->max_levels);CHKERRQ(ierr);
        if (pc_temp->ops->setfromoptions) { /* need to setfromoptions again, skipping the pc_type */
          ierr = PetscObjectOptionsBegin((PetscObject)pc_temp);CHKERRQ(ierr);
          ierr = (*pc_temp->ops->setfromoptions)(PetscOptionsObject,pc_temp);CHKERRQ(ierr);
          ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)pc_temp);CHKERRQ(ierr);
          ierr = PetscOptionsEnd();CHKERRQ(ierr);
          pc_temp->setfromoptionscalled++;
        }
      }
    }
    /* propagate BDDC info to the next level (these are dummy calls if pc_temp is not of type PCBDDC) */
    ierr = KSPGetPC(pcbddc->coarse_ksp,&pc_temp);CHKERRQ(ierr);
    if (nisdofs) {
      ierr = PCBDDCSetDofsSplitting(pc_temp,nisdofs,isarray);CHKERRQ(ierr);
      for (i=0;i<nisdofs;i++) {
        ierr = ISDestroy(&isarray[i]);CHKERRQ(ierr);
      }
    }
    if (nisneu) {
      ierr = PCBDDCSetNeumannBoundaries(pc_temp,isarray[nisdofs]);CHKERRQ(ierr);
      ierr = ISDestroy(&isarray[nisdofs]);CHKERRQ(ierr);
    }
    if (nisvert) {
      ierr = PCBDDCSetPrimalVerticesIS(pc_temp,isarray[nis-1]);CHKERRQ(ierr);
      ierr = ISDestroy(&isarray[nis-1]);CHKERRQ(ierr);
    }
    if (coarseG) {
      ierr = PCBDDCSetDiscreteGradient(pc_temp,coarseG,1,nedcfield,PETSC_FALSE,PETSC_TRUE);CHKERRQ(ierr);
    }

    /* get some info after set from options */
    ierr = PetscObjectTypeCompare((PetscObject)pc_temp,PCBDDC,&isbddc);CHKERRQ(ierr);

    /* multilevel can only be requested via -pc_bddc_levels or PCBDDCSetLevels */
    if (isbddc && !multilevel_allowed) {
      ierr = PCSetType(pc_temp,coarse_pc_type);CHKERRQ(ierr);
    }
    /* multilevel cannot be done with coarse PC different from BDDC, NN, HPDDM, unless forced to */
    force = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL,((PetscObject)pc_temp)->prefix,"-pc_type_forced",&force,NULL);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompareAny((PetscObject)pc_temp,&valid,PCBDDC,PCNN,PCHPDDM,"");CHKERRQ(ierr);
    if (multilevel_requested && multilevel_allowed && !valid && !force) {
      ierr = PCSetType(pc_temp,PCBDDC);CHKERRQ(ierr);
    }
    ierr = PetscObjectTypeCompare((PetscObject)pc_temp,PCREDUNDANT,&isredundant);CHKERRQ(ierr);
    if (isredundant) {
      KSP inner_ksp;
      PC  inner_pc;

      ierr = PCRedundantGetKSP(pc_temp,&inner_ksp);CHKERRQ(ierr);
      ierr = KSPGetPC(inner_ksp,&inner_pc);CHKERRQ(ierr);
    }

    /* parameters which miss an API */
    ierr = PetscObjectTypeCompare((PetscObject)pc_temp,PCBDDC,&isbddc);CHKERRQ(ierr);
    if (isbddc) {
      PC_BDDC* pcbddc_coarse = (PC_BDDC*)pc_temp->data;

      pcbddc_coarse->detect_disconnected = PETSC_TRUE;
      pcbddc_coarse->coarse_eqs_per_proc = pcbddc->coarse_eqs_per_proc;
      pcbddc_coarse->coarse_eqs_limit    = pcbddc->coarse_eqs_limit;
      pcbddc_coarse->benign_saddle_point = pcbddc->benign_have_null;
      if (pcbddc_coarse->benign_saddle_point) {
        Mat                    coarsedivudotp_is;
        ISLocalToGlobalMapping l2gmap,rl2g,cl2g;
        IS                     row,col;
        const PetscInt         *gidxs;
        PetscInt               n,st,M,N;

        ierr = MatGetSize(coarsedivudotp,&n,NULL);CHKERRQ(ierr);
        ierr = MPI_Scan(&n,&st,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)coarse_mat));CHKERRQ(ierr);
        st   = st-n;
        ierr = ISCreateStride(PetscObjectComm((PetscObject)coarse_mat),1,st,1,&row);CHKERRQ(ierr);
        ierr = MatGetLocalToGlobalMapping(coarse_mat,&l2gmap,NULL);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingGetSize(l2gmap,&n);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingGetIndices(l2gmap,&gidxs);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)coarse_mat),n,gidxs,PETSC_COPY_VALUES,&col);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingRestoreIndices(l2gmap,&gidxs);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingCreateIS(row,&rl2g);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingCreateIS(col,&cl2g);CHKERRQ(ierr);
        ierr = ISGetSize(row,&M);CHKERRQ(ierr);
        ierr = MatGetSize(coarse_mat,&N,NULL);CHKERRQ(ierr);
        ierr = ISDestroy(&row);CHKERRQ(ierr);
        ierr = ISDestroy(&col);CHKERRQ(ierr);
        ierr = MatCreate(PetscObjectComm((PetscObject)coarse_mat),&coarsedivudotp_is);CHKERRQ(ierr);
        ierr = MatSetType(coarsedivudotp_is,MATIS);CHKERRQ(ierr);
        ierr = MatSetSizes(coarsedivudotp_is,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
        ierr = MatSetLocalToGlobalMapping(coarsedivudotp_is,rl2g,cl2g);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingDestroy(&rl2g);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingDestroy(&cl2g);CHKERRQ(ierr);
        ierr = MatISSetLocalMat(coarsedivudotp_is,coarsedivudotp);CHKERRQ(ierr);
        ierr = MatDestroy(&coarsedivudotp);CHKERRQ(ierr);
        ierr = PCBDDCSetDivergenceMat(pc_temp,coarsedivudotp_is,PETSC_FALSE,NULL);CHKERRQ(ierr);
        ierr = MatDestroy(&coarsedivudotp_is);CHKERRQ(ierr);
        pcbddc_coarse->adaptive_userdefined = PETSC_TRUE;
        if (pcbddc->adaptive_threshold[0] == 0.0) pcbddc_coarse->deluxe_zerorows = PETSC_TRUE;
      }
    }

    /* propagate symmetry info of coarse matrix */
    ierr = MatSetOption(coarse_mat,MAT_STRUCTURALLY_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    if (pc->pmat->symmetric_set) {
      ierr = MatSetOption(coarse_mat,MAT_SYMMETRIC,pc->pmat->symmetric);CHKERRQ(ierr);
    }
    if (pc->pmat->hermitian_set) {
      ierr = MatSetOption(coarse_mat,MAT_HERMITIAN,pc->pmat->hermitian);CHKERRQ(ierr);
    }
    if (pc->pmat->spd_set) {
      ierr = MatSetOption(coarse_mat,MAT_SPD,pc->pmat->spd);CHKERRQ(ierr);
    }
    if (pcbddc->benign_saddle_point && !pcbddc->benign_have_null) {
      ierr = MatSetOption(coarse_mat,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);
    }
    /* set operators */
    ierr = MatViewFromOptions(coarse_mat,(PetscObject)pc,"-pc_bddc_coarse_mat_view");CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(coarse_mat,((PetscObject)pcbddc->coarse_ksp)->prefix);CHKERRQ(ierr);
    ierr = KSPSetOperators(pcbddc->coarse_ksp,coarse_mat,coarse_mat);CHKERRQ(ierr);
    if (pcbddc->dbg_flag) {
      ierr = PetscViewerASCIISubtractTab(dbg_viewer,2*pcbddc->current_level);CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy(&coarseG);CHKERRQ(ierr);
  ierr = PetscFree(isarray);CHKERRQ(ierr);
#if 0
  {
    PetscViewer viewer;
    char filename[256];
    sprintf(filename,"coarse_mat_level%d.m",pcbddc->current_level);
    ierr = PetscViewerASCIIOpen(PetscObjectComm((PetscObject)coarse_mat),filename,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = MatView(coarse_mat,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
#endif

  if (corners) {
    Vec            gv;
    IS             is;
    const PetscInt *idxs;
    PetscInt       i,d,N,n,cdim = pcbddc->mat_graph->cdim;
    PetscScalar    *coords;

    if (!pcbddc->mat_graph->cloc) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing local coordinates");
    ierr = VecGetSize(pcbddc->coarse_vec,&N);CHKERRQ(ierr);
    ierr = VecGetLocalSize(pcbddc->coarse_vec,&n);CHKERRQ(ierr);
    ierr = VecCreate(PetscObjectComm((PetscObject)pcbddc->coarse_vec),&gv);CHKERRQ(ierr);
    ierr = VecSetBlockSize(gv,cdim);CHKERRQ(ierr);
    ierr = VecSetSizes(gv,n*cdim,N*cdim);CHKERRQ(ierr);
    ierr = VecSetType(gv,VECSTANDARD);CHKERRQ(ierr);
    ierr = VecSetFromOptions(gv);CHKERRQ(ierr);
    ierr = VecSet(gv,PETSC_MAX_REAL);CHKERRQ(ierr); /* we only propagate coordinates from vertices constraints */

    ierr = PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&is);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
    ierr = ISGetIndices(is,&idxs);CHKERRQ(ierr);
    ierr = PetscMalloc1(n*cdim,&coords);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      for (d=0;d<cdim;d++) {
        coords[cdim*i+d] = pcbddc->mat_graph->coords[cdim*idxs[i]+d];
      }
    }
    ierr = ISRestoreIndices(is,&idxs);CHKERRQ(ierr);
    ierr = PCBDDCGraphRestoreCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&is);CHKERRQ(ierr);

    ierr = ISGetLocalSize(corners,&n);CHKERRQ(ierr);
    ierr = ISGetIndices(corners,&idxs);CHKERRQ(ierr);
    ierr = VecSetValuesBlocked(gv,n,idxs,coords,INSERT_VALUES);CHKERRQ(ierr);
    ierr = ISRestoreIndices(corners,&idxs);CHKERRQ(ierr);
    ierr = PetscFree(coords);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(gv);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(gv);CHKERRQ(ierr);
    ierr = VecGetArray(gv,&coords);CHKERRQ(ierr);
    if (pcbddc->coarse_ksp) {
      PC        coarse_pc;
      PetscBool isbddc;

      ierr = KSPGetPC(pcbddc->coarse_ksp,&coarse_pc);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)coarse_pc,PCBDDC,&isbddc);CHKERRQ(ierr);
      if (isbddc) { /* coarse coordinates have PETSC_MAX_REAL, specific for BDDC */
        PetscReal *realcoords;

        ierr = VecGetLocalSize(gv,&n);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        ierr = PetscMalloc1(n,&realcoords);CHKERRQ(ierr);
        for (i=0;i<n;i++) realcoords[i] = PetscRealPart(coords[i]);
#else
        realcoords = coords;
#endif
        ierr = PCSetCoordinates(coarse_pc,cdim,n/cdim,realcoords);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        ierr = PetscFree(realcoords);CHKERRQ(ierr);
#endif
      }
    }
    ierr = VecRestoreArray(gv,&coords);CHKERRQ(ierr);
    ierr = VecDestroy(&gv);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&corners);CHKERRQ(ierr);

  if (pcbddc->coarse_ksp) {
    Vec crhs,csol;

    ierr = KSPGetSolution(pcbddc->coarse_ksp,&csol);CHKERRQ(ierr);
    ierr = KSPGetRhs(pcbddc->coarse_ksp,&crhs);CHKERRQ(ierr);
    if (!csol) {
      ierr = MatCreateVecs(coarse_mat,&((pcbddc->coarse_ksp)->vec_sol),NULL);CHKERRQ(ierr);
    }
    if (!crhs) {
      ierr = MatCreateVecs(coarse_mat,NULL,&((pcbddc->coarse_ksp)->vec_rhs));CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy(&coarsedivudotp);CHKERRQ(ierr);

  /* compute null space for coarse solver if the benign trick has been requested */
  if (pcbddc->benign_null) {

    ierr = VecSet(pcbddc->vec1_P,0.);CHKERRQ(ierr);
    for (i=0;i<pcbddc->benign_n;i++) {
      ierr = VecSetValue(pcbddc->vec1_P,pcbddc->local_primal_size-pcbddc->benign_n+i,1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(pcbddc->vec1_P);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(pcbddc->vec1_P);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcbddc->coarse_loc_to_glob,pcbddc->vec1_P,pcbddc->coarse_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcbddc->coarse_loc_to_glob,pcbddc->vec1_P,pcbddc->coarse_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    if (coarse_mat) {
      Vec         nullv;
      PetscScalar *array,*array2;
      PetscInt    nl;

      ierr = MatCreateVecs(coarse_mat,&nullv,NULL);CHKERRQ(ierr);
      ierr = VecGetLocalSize(nullv,&nl);CHKERRQ(ierr);
      ierr = VecGetArrayRead(pcbddc->coarse_vec,(const PetscScalar**)&array);CHKERRQ(ierr);
      ierr = VecGetArray(nullv,&array2);CHKERRQ(ierr);
      ierr = PetscArraycpy(array2,array,nl);CHKERRQ(ierr);
      ierr = VecRestoreArray(nullv,&array2);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(pcbddc->coarse_vec,(const PetscScalar**)&array);CHKERRQ(ierr);
      ierr = VecNormalize(nullv,NULL);CHKERRQ(ierr);
      ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject)coarse_mat),PETSC_FALSE,1,&nullv,&CoarseNullSpace);CHKERRQ(ierr);
      ierr = VecDestroy(&nullv);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(PC_BDDC_CoarseSetUp[pcbddc->current_level],pc,0,0,0);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(PC_BDDC_CoarseSolver[pcbddc->current_level],pc,0,0,0);CHKERRQ(ierr);
  if (pcbddc->coarse_ksp) {
    PetscBool ispreonly;

    if (CoarseNullSpace) {
      PetscBool isnull;
      ierr = MatNullSpaceTest(CoarseNullSpace,coarse_mat,&isnull);CHKERRQ(ierr);
      if (isnull) {
        ierr = MatSetNullSpace(coarse_mat,CoarseNullSpace);CHKERRQ(ierr);
      }
      /* TODO: add local nullspaces (if any) */
    }
    /* setup coarse ksp */
    ierr = KSPSetUp(pcbddc->coarse_ksp);CHKERRQ(ierr);
    /* Check coarse problem if in debug mode or if solving with an iterative method */
    ierr = PetscObjectTypeCompare((PetscObject)pcbddc->coarse_ksp,KSPPREONLY,&ispreonly);CHKERRQ(ierr);
    if (pcbddc->dbg_flag || (!ispreonly && pcbddc->use_coarse_estimates) ) {
      KSP       check_ksp;
      KSPType   check_ksp_type;
      PC        check_pc;
      Vec       check_vec,coarse_vec;
      PetscReal abs_infty_error,infty_error,lambda_min=1.0,lambda_max=1.0;
      PetscInt  its;
      PetscBool compute_eigs;
      PetscReal *eigs_r,*eigs_c;
      PetscInt  neigs;
      const char *prefix;

      /* Create ksp object suitable for estimation of extreme eigenvalues */
      ierr = KSPCreate(PetscObjectComm((PetscObject)pcbddc->coarse_ksp),&check_ksp);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)check_ksp,(PetscObject)pcbddc->coarse_ksp,0);CHKERRQ(ierr);
      ierr = KSPSetErrorIfNotConverged(pcbddc->coarse_ksp,PETSC_FALSE);CHKERRQ(ierr);
      ierr = KSPSetOperators(check_ksp,coarse_mat,coarse_mat);CHKERRQ(ierr);
      ierr = KSPSetTolerances(check_ksp,1.e-12,1.e-12,PETSC_DEFAULT,pcbddc->coarse_size);CHKERRQ(ierr);
      /* prevent from setup unneeded object */
      ierr = KSPGetPC(check_ksp,&check_pc);CHKERRQ(ierr);
      ierr = PCSetType(check_pc,PCNONE);CHKERRQ(ierr);
      if (ispreonly) {
        check_ksp_type = KSPPREONLY;
        compute_eigs = PETSC_FALSE;
      } else {
        check_ksp_type = KSPGMRES;
        compute_eigs = PETSC_TRUE;
      }
      ierr = KSPSetType(check_ksp,check_ksp_type);CHKERRQ(ierr);
      ierr = KSPSetComputeSingularValues(check_ksp,compute_eigs);CHKERRQ(ierr);
      ierr = KSPSetComputeEigenvalues(check_ksp,compute_eigs);CHKERRQ(ierr);
      ierr = KSPGMRESSetRestart(check_ksp,pcbddc->coarse_size+1);CHKERRQ(ierr);
      ierr = KSPGetOptionsPrefix(pcbddc->coarse_ksp,&prefix);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(check_ksp,prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(check_ksp,"check_");CHKERRQ(ierr);
      ierr = KSPSetFromOptions(check_ksp);CHKERRQ(ierr);
      ierr = KSPSetUp(check_ksp);CHKERRQ(ierr);
      ierr = KSPGetPC(pcbddc->coarse_ksp,&check_pc);CHKERRQ(ierr);
      ierr = KSPSetPC(check_ksp,check_pc);CHKERRQ(ierr);
      /* create random vec */
      ierr = MatCreateVecs(coarse_mat,&coarse_vec,&check_vec);CHKERRQ(ierr);
      ierr = VecSetRandom(check_vec,NULL);CHKERRQ(ierr);
      ierr = MatMult(coarse_mat,check_vec,coarse_vec);CHKERRQ(ierr);
      /* solve coarse problem */
      ierr = KSPSolve(check_ksp,coarse_vec,coarse_vec);CHKERRQ(ierr);
      ierr = KSPCheckSolve(check_ksp,pc,coarse_vec);CHKERRQ(ierr);
      /* set eigenvalue estimation if preonly has not been requested */
      if (compute_eigs) {
        ierr = PetscMalloc1(pcbddc->coarse_size+1,&eigs_r);CHKERRQ(ierr);
        ierr = PetscMalloc1(pcbddc->coarse_size+1,&eigs_c);CHKERRQ(ierr);
        ierr = KSPComputeEigenvalues(check_ksp,pcbddc->coarse_size+1,eigs_r,eigs_c,&neigs);CHKERRQ(ierr);
        if (neigs) {
          lambda_max = eigs_r[neigs-1];
          lambda_min = eigs_r[0];
          if (pcbddc->use_coarse_estimates) {
            if (lambda_max>=lambda_min) { /* using PETSC_SMALL since lambda_max == lambda_min is not allowed by KSPChebyshevSetEigenvalues */
              ierr = KSPChebyshevSetEigenvalues(pcbddc->coarse_ksp,lambda_max+PETSC_SMALL,lambda_min);CHKERRQ(ierr);
              ierr = KSPRichardsonSetScale(pcbddc->coarse_ksp,2.0/(lambda_max+lambda_min));CHKERRQ(ierr);
            }
          }
        }
      }

      /* check coarse problem residual error */
      if (pcbddc->dbg_flag) {
        PetscViewer dbg_viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)pcbddc->coarse_ksp));
        ierr = PetscViewerASCIIAddTab(dbg_viewer,2*(pcbddc->current_level+1));CHKERRQ(ierr);
        ierr = VecAXPY(check_vec,-1.0,coarse_vec);CHKERRQ(ierr);
        ierr = VecNorm(check_vec,NORM_INFINITY,&infty_error);CHKERRQ(ierr);
        ierr = MatMult(coarse_mat,check_vec,coarse_vec);CHKERRQ(ierr);
        ierr = VecNorm(coarse_vec,NORM_INFINITY,&abs_infty_error);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(dbg_viewer,"Coarse problem details (use estimates %d)\n",pcbddc->use_coarse_estimates);CHKERRQ(ierr);
        ierr = PetscObjectPrintClassNamePrefixType((PetscObject)(pcbddc->coarse_ksp),dbg_viewer);CHKERRQ(ierr);
        ierr = PetscObjectPrintClassNamePrefixType((PetscObject)(check_pc),dbg_viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(dbg_viewer,"Coarse problem exact infty_error   : %1.6e\n",infty_error);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(dbg_viewer,"Coarse problem residual infty_error: %1.6e\n",abs_infty_error);CHKERRQ(ierr);
        if (CoarseNullSpace) {
          ierr = PetscViewerASCIIPrintf(dbg_viewer,"Coarse problem is singular\n");CHKERRQ(ierr);
        }
        if (compute_eigs) {
          PetscReal          lambda_max_s,lambda_min_s;
          KSPConvergedReason reason;
          ierr = KSPGetType(check_ksp,&check_ksp_type);CHKERRQ(ierr);
          ierr = KSPGetIterationNumber(check_ksp,&its);CHKERRQ(ierr);
          ierr = KSPGetConvergedReason(check_ksp,&reason);CHKERRQ(ierr);
          ierr = KSPComputeExtremeSingularValues(check_ksp,&lambda_max_s,&lambda_min_s);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPrintf(dbg_viewer,"Coarse problem eigenvalues (estimated with %d iterations of %s, conv reason %d): %1.6e %1.6e (%1.6e %1.6e)\n",its,check_ksp_type,reason,lambda_min,lambda_max,lambda_min_s,lambda_max_s);CHKERRQ(ierr);
          for (i=0;i<neigs;i++) {
            ierr = PetscViewerASCIIPrintf(dbg_viewer,"%1.6e %1.6ei\n",eigs_r[i],eigs_c[i]);CHKERRQ(ierr);
          }
        }
        ierr = PetscViewerFlush(dbg_viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIISubtractTab(dbg_viewer,2*(pcbddc->current_level+1));CHKERRQ(ierr);
      }
      ierr = VecDestroy(&check_vec);CHKERRQ(ierr);
      ierr = VecDestroy(&coarse_vec);CHKERRQ(ierr);
      ierr = KSPDestroy(&check_ksp);CHKERRQ(ierr);
      if (compute_eigs) {
        ierr = PetscFree(eigs_r);CHKERRQ(ierr);
        ierr = PetscFree(eigs_c);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatNullSpaceDestroy(&CoarseNullSpace);CHKERRQ(ierr);
  /* print additional info */
  if (pcbddc->dbg_flag) {
    /* waits until all processes reaches this point */
    ierr = PetscBarrier((PetscObject)pc);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Coarse solver setup completed at level %D\n",pcbddc->current_level);CHKERRQ(ierr);
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
  }

  /* free memory */
  ierr = MatDestroy(&coarse_mat);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PC_BDDC_CoarseSolver[pcbddc->current_level],pc,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCComputePrimalNumbering(PC pc,PetscInt* coarse_size_n,PetscInt** local_primal_indices_n)
{
  PC_BDDC*       pcbddc = (PC_BDDC*)pc->data;
  PC_IS*         pcis = (PC_IS*)pc->data;
  Mat_IS*        matis = (Mat_IS*)pc->pmat->data;
  IS             subset,subset_mult,subset_n;
  PetscInt       local_size,coarse_size=0;
  PetscInt       *local_primal_indices=NULL;
  const PetscInt *t_local_primal_indices;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Compute global number of coarse dofs */
  if (pcbddc->local_primal_size && !pcbddc->local_primal_ref_node) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"BDDC ConstraintsSetUp should be called first");
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)(pc->pmat)),pcbddc->local_primal_size_cc,pcbddc->local_primal_ref_node,PETSC_COPY_VALUES,&subset_n);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyIS(pcis->mapping,subset_n,&subset);CHKERRQ(ierr);
  ierr = ISDestroy(&subset_n);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)(pc->pmat)),pcbddc->local_primal_size_cc,pcbddc->local_primal_ref_mult,PETSC_COPY_VALUES,&subset_mult);CHKERRQ(ierr);
  ierr = ISRenumber(subset,subset_mult,&coarse_size,&subset_n);CHKERRQ(ierr);
  ierr = ISDestroy(&subset);CHKERRQ(ierr);
  ierr = ISDestroy(&subset_mult);CHKERRQ(ierr);
  ierr = ISGetLocalSize(subset_n,&local_size);CHKERRQ(ierr);
  if (local_size != pcbddc->local_primal_size) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid number of local primal indices computed %D != %D",local_size,pcbddc->local_primal_size);
  ierr = PetscMalloc1(local_size,&local_primal_indices);CHKERRQ(ierr);
  ierr = ISGetIndices(subset_n,&t_local_primal_indices);CHKERRQ(ierr);
  ierr = PetscArraycpy(local_primal_indices,t_local_primal_indices,local_size);CHKERRQ(ierr);
  ierr = ISRestoreIndices(subset_n,&t_local_primal_indices);CHKERRQ(ierr);
  ierr = ISDestroy(&subset_n);CHKERRQ(ierr);

  /* check numbering */
  if (pcbddc->dbg_flag) {
    PetscScalar coarsesum,*array,*array2;
    PetscInt    i;
    PetscBool   set_error = PETSC_FALSE,set_error_reduced = PETSC_FALSE;

    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Check coarse indices\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer);CHKERRQ(ierr);
    /* counter */
    ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
    ierr = VecSet(pcis->vec1_N,1.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->rctx,pcis->vec1_global,pcis->vec2_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(matis->rctx,pcis->vec1_global,pcis->vec2_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecSet(pcis->vec1_N,0.0);CHKERRQ(ierr);
    for (i=0;i<pcbddc->local_primal_size;i++) {
      ierr = VecSetValue(pcis->vec1_N,pcbddc->primal_indices_local_idxs[i],1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(pcis->vec1_N);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(pcis->vec1_N);CHKERRQ(ierr);
    ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->rctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(matis->rctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->vec2_N,&array2);CHKERRQ(ierr);
    for (i=0;i<pcis->n;i++) {
      if (array[i] != 0.0 && array[i] != array2[i]) {
        PetscInt owned = (PetscInt)PetscRealPart(array[i]),gi;
        PetscInt neigh = (PetscInt)PetscRealPart(array2[i]);
        set_error = PETSC_TRUE;
        ierr = ISLocalToGlobalMappingApply(pcis->mapping,1,&i,&gi);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d: local index %D (gid %D) owned by %D processes instead of %D!\n",PetscGlobalRank,i,gi,owned,neigh);CHKERRQ(ierr);
      }
    }
    ierr = VecRestoreArray(pcis->vec2_N,&array2);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(&set_error,&set_error_reduced,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    for (i=0;i<pcis->n;i++) {
      if (PetscRealPart(array[i]) > 0.0) array[i] = 1.0/PetscRealPart(array[i]);
    }
    ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecSum(pcis->vec1_global,&coarsesum);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Size of coarse problem is %D (%lf)\n",coarse_size,PetscRealPart(coarsesum));CHKERRQ(ierr);
    if (pcbddc->dbg_flag > 1 || set_error_reduced) {
      PetscInt *gidxs;

      ierr = PetscMalloc1(pcbddc->local_primal_size,&gidxs);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingApply(pcis->mapping,pcbddc->local_primal_size,pcbddc->primal_indices_local_idxs,gidxs);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Distribution of local primal indices\n");CHKERRQ(ierr);
      ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d\n",PetscGlobalRank);CHKERRQ(ierr);
      for (i=0;i<pcbddc->local_primal_size;i++) {
        ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"local_primal_indices[%D]=%D (%D,%D)\n",i,local_primal_indices[i],pcbddc->primal_indices_local_idxs[i],gidxs[i]);CHKERRQ(ierr);
      }
      ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
      ierr = PetscFree(gidxs);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer);CHKERRQ(ierr);
    if (set_error_reduced) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"BDDC Numbering of coarse dofs failed");
  }

  /* get back data */
  *coarse_size_n = coarse_size;
  *local_primal_indices_n = local_primal_indices;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCGlobalToLocal(VecScatter g2l_ctx,Vec gwork, Vec lwork, IS globalis, IS* localis)
{
  IS             localis_t;
  PetscInt       i,lsize,*idxs,n;
  PetscScalar    *vals;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* get indices in local ordering exploiting local to global map */
  ierr = ISGetLocalSize(globalis,&lsize);CHKERRQ(ierr);
  ierr = PetscMalloc1(lsize,&vals);CHKERRQ(ierr);
  for (i=0;i<lsize;i++) vals[i] = 1.0;
  ierr = ISGetIndices(globalis,(const PetscInt**)&idxs);CHKERRQ(ierr);
  ierr = VecSet(gwork,0.0);CHKERRQ(ierr);
  ierr = VecSet(lwork,0.0);CHKERRQ(ierr);
  if (idxs) { /* multilevel guard */
    ierr = VecSetOption(gwork,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);CHKERRQ(ierr);
    ierr = VecSetValues(gwork,lsize,idxs,vals,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(gwork);CHKERRQ(ierr);
  ierr = ISRestoreIndices(globalis,(const PetscInt**)&idxs);CHKERRQ(ierr);
  ierr = PetscFree(vals);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(gwork);CHKERRQ(ierr);
  /* now compute set in local ordering */
  ierr = VecScatterBegin(g2l_ctx,gwork,lwork,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(g2l_ctx,gwork,lwork,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArrayRead(lwork,(const PetscScalar**)&vals);CHKERRQ(ierr);
  ierr = VecGetSize(lwork,&n);CHKERRQ(ierr);
  for (i=0,lsize=0;i<n;i++) {
    if (PetscRealPart(vals[i]) > 0.5) {
      lsize++;
    }
  }
  ierr = PetscMalloc1(lsize,&idxs);CHKERRQ(ierr);
  for (i=0,lsize=0;i<n;i++) {
    if (PetscRealPart(vals[i]) > 0.5) {
      idxs[lsize++] = i;
    }
  }
  ierr = VecRestoreArrayRead(lwork,(const PetscScalar**)&vals);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)gwork),lsize,idxs,PETSC_OWN_POINTER,&localis_t);CHKERRQ(ierr);
  *localis = localis_t;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSetUpSubSchurs(PC pc)
{
  PC_IS               *pcis=(PC_IS*)pc->data;
  PC_BDDC             *pcbddc=(PC_BDDC*)pc->data;
  PCBDDCSubSchurs     sub_schurs=pcbddc->sub_schurs;
  Mat                 S_j;
  PetscInt            *used_xadj,*used_adjncy;
  PetscBool           free_used_adj;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PC_BDDC_Schurs[pcbddc->current_level],pc,0,0,0);CHKERRQ(ierr);
  /* decide the adjacency to be used for determining internal problems for local schur on subsets */
  free_used_adj = PETSC_FALSE;
  if (pcbddc->sub_schurs_layers == -1) {
    used_xadj = NULL;
    used_adjncy = NULL;
  } else {
    if (pcbddc->sub_schurs_use_useradj && pcbddc->mat_graph->xadj) {
      used_xadj = pcbddc->mat_graph->xadj;
      used_adjncy = pcbddc->mat_graph->adjncy;
    } else if (pcbddc->computed_rowadj) {
      used_xadj = pcbddc->mat_graph->xadj;
      used_adjncy = pcbddc->mat_graph->adjncy;
    } else {
      PetscBool      flg_row=PETSC_FALSE;
      const PetscInt *xadj,*adjncy;
      PetscInt       nvtxs;

      ierr = MatGetRowIJ(pcbddc->local_mat,0,PETSC_TRUE,PETSC_FALSE,&nvtxs,&xadj,&adjncy,&flg_row);CHKERRQ(ierr);
      if (flg_row) {
        ierr = PetscMalloc2(nvtxs+1,&used_xadj,xadj[nvtxs],&used_adjncy);CHKERRQ(ierr);
        ierr = PetscArraycpy(used_xadj,xadj,nvtxs+1);CHKERRQ(ierr);
        ierr = PetscArraycpy(used_adjncy,adjncy,xadj[nvtxs]);CHKERRQ(ierr);
        free_used_adj = PETSC_TRUE;
      } else {
        pcbddc->sub_schurs_layers = -1;
        used_xadj = NULL;
        used_adjncy = NULL;
      }
      ierr = MatRestoreRowIJ(pcbddc->local_mat,0,PETSC_TRUE,PETSC_FALSE,&nvtxs,&xadj,&adjncy,&flg_row);CHKERRQ(ierr);
    }
  }

  /* setup sub_schurs data */
  ierr = MatCreateSchurComplement(pcis->A_II,pcis->pA_II,pcis->A_IB,pcis->A_BI,pcis->A_BB,&S_j);CHKERRQ(ierr);
  if (!sub_schurs->schur_explicit) {
    /* pcbddc->ksp_D up to date only if not using MatFactor with Schur complement support */
    ierr = MatSchurComplementSetKSP(S_j,pcbddc->ksp_D);CHKERRQ(ierr);
    ierr = PCBDDCSubSchursSetUp(sub_schurs,NULL,S_j,PETSC_FALSE,used_xadj,used_adjncy,pcbddc->sub_schurs_layers,NULL,pcbddc->adaptive_selection,PETSC_FALSE,PETSC_FALSE,0,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  } else {
    Mat       change = NULL;
    Vec       scaling = NULL;
    IS        change_primal = NULL, iP;
    PetscInt  benign_n;
    PetscBool reuse_solvers = (PetscBool)!pcbddc->use_change_of_basis;
    PetscBool need_change = PETSC_FALSE;
    PetscBool discrete_harmonic = PETSC_FALSE;

    if (!pcbddc->use_vertices && reuse_solvers) {
      PetscInt n_vertices;

      ierr = ISGetLocalSize(sub_schurs->is_vertices,&n_vertices);CHKERRQ(ierr);
      reuse_solvers = (PetscBool)!n_vertices;
    }
    if (!pcbddc->benign_change_explicit) {
      benign_n = pcbddc->benign_n;
    } else {
      benign_n = 0;
    }
    /* sub_schurs->change is a local object; instead, PCBDDCConstraintsSetUp and the quantities used in the test below are logically collective on pc.
       We need a global reduction to avoid possible deadlocks.
       We assume that sub_schurs->change is created once, and then reused for different solves, unless the topography has been recomputed */
    if (pcbddc->adaptive_userdefined || (pcbddc->deluxe_zerorows && !pcbddc->use_change_of_basis)) {
      PetscBool have_loc_change = (PetscBool)(!!sub_schurs->change);
      ierr = MPIU_Allreduce(&have_loc_change,&need_change,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
      need_change = (PetscBool)(!need_change);
    }
    /* If the user defines additional constraints, we import them here.
       We need to compute the change of basis according to the quadrature weights attached to pmat via MatSetNearNullSpace, and this could not be done (at the moment) without some hacking */
    if (need_change) {
      PC_IS   *pcisf;
      PC_BDDC *pcbddcf;
      PC      pcf;

      if (pcbddc->sub_schurs_rebuild) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot compute change of basis with a different graph");
      ierr = PCCreate(PetscObjectComm((PetscObject)pc),&pcf);CHKERRQ(ierr);
      ierr = PCSetOperators(pcf,pc->mat,pc->pmat);CHKERRQ(ierr);
      ierr = PCSetType(pcf,PCBDDC);CHKERRQ(ierr);

      /* hacks */
      pcisf                        = (PC_IS*)pcf->data;
      pcisf->is_B_local            = pcis->is_B_local;
      pcisf->vec1_N                = pcis->vec1_N;
      pcisf->BtoNmap               = pcis->BtoNmap;
      pcisf->n                     = pcis->n;
      pcisf->n_B                   = pcis->n_B;
      pcbddcf                      = (PC_BDDC*)pcf->data;
      ierr                         = PetscFree(pcbddcf->mat_graph);CHKERRQ(ierr);
      pcbddcf->mat_graph           = pcbddc->mat_graph;
      pcbddcf->use_faces           = PETSC_TRUE;
      pcbddcf->use_change_of_basis = PETSC_TRUE;
      pcbddcf->use_change_on_faces = PETSC_TRUE;
      pcbddcf->use_qr_single       = PETSC_TRUE;
      pcbddcf->fake_change         = PETSC_TRUE;

      /* setup constraints so that we can get information on primal vertices and change of basis (in local numbering) */
      ierr = PCBDDCConstraintsSetUp(pcf);CHKERRQ(ierr);
      sub_schurs->change_with_qr = pcbddcf->use_qr_single;
      ierr = ISCreateGeneral(PETSC_COMM_SELF,pcbddcf->n_vertices,pcbddcf->local_primal_ref_node,PETSC_COPY_VALUES,&change_primal);CHKERRQ(ierr);
      change = pcbddcf->ConstraintMatrix;
      pcbddcf->ConstraintMatrix = NULL;

      /* free unneeded memory allocated in PCBDDCConstraintsSetUp */
      ierr = PetscFree(pcbddcf->sub_schurs);CHKERRQ(ierr);
      ierr = MatNullSpaceDestroy(&pcbddcf->onearnullspace);CHKERRQ(ierr);
      ierr = PetscFree2(pcbddcf->local_primal_ref_node,pcbddcf->local_primal_ref_mult);CHKERRQ(ierr);
      ierr = PetscFree(pcbddcf->primal_indices_local_idxs);CHKERRQ(ierr);
      ierr = PetscFree(pcbddcf->onearnullvecs_state);CHKERRQ(ierr);
      ierr = PetscFree(pcf->data);CHKERRQ(ierr);
      pcf->ops->destroy = NULL;
      pcf->ops->reset   = NULL;
      ierr = PCDestroy(&pcf);CHKERRQ(ierr);
    }
    if (!pcbddc->use_deluxe_scaling) scaling = pcis->D;

    ierr = PetscObjectQuery((PetscObject)pc,"__KSPFETIDP_iP",(PetscObject*)&iP);CHKERRQ(ierr);
    if (iP) {
      ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)iP),sub_schurs->prefix,"BDDC sub_schurs options","PC");CHKERRQ(ierr);
      ierr = PetscOptionsBool("-sub_schurs_discrete_harmonic",NULL,NULL,discrete_harmonic,&discrete_harmonic,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsEnd();CHKERRQ(ierr);
    }
    if (discrete_harmonic) {
      Mat A;
      ierr = MatDuplicate(pcbddc->local_mat,MAT_COPY_VALUES,&A);CHKERRQ(ierr);
      ierr = MatZeroRowsColumnsIS(A,iP,1.0,NULL,NULL);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)A,"__KSPFETIDP_iP",(PetscObject)iP);CHKERRQ(ierr);
      ierr = PCBDDCSubSchursSetUp(sub_schurs,A,S_j,pcbddc->sub_schurs_exact_schur,used_xadj,used_adjncy,pcbddc->sub_schurs_layers,scaling,pcbddc->adaptive_selection,reuse_solvers,pcbddc->benign_saddle_point,benign_n,pcbddc->benign_p0_lidx,pcbddc->benign_zerodiag_subs,change,change_primal);CHKERRQ(ierr);
      ierr = MatDestroy(&A);CHKERRQ(ierr);
    } else {
      ierr = PCBDDCSubSchursSetUp(sub_schurs,pcbddc->local_mat,S_j,pcbddc->sub_schurs_exact_schur,used_xadj,used_adjncy,pcbddc->sub_schurs_layers,scaling,pcbddc->adaptive_selection,reuse_solvers,pcbddc->benign_saddle_point,benign_n,pcbddc->benign_p0_lidx,pcbddc->benign_zerodiag_subs,change,change_primal);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&change);CHKERRQ(ierr);
    ierr = ISDestroy(&change_primal);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&S_j);CHKERRQ(ierr);

  /* free adjacency */
  if (free_used_adj) {
    ierr = PetscFree2(used_xadj,used_adjncy);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(PC_BDDC_Schurs[pcbddc->current_level],pc,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCInitSubSchurs(PC pc)
{
  PC_IS               *pcis=(PC_IS*)pc->data;
  PC_BDDC             *pcbddc=(PC_BDDC*)pc->data;
  PCBDDCGraph         graph;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* attach interface graph for determining subsets */
  if (pcbddc->sub_schurs_rebuild) { /* in case rebuild has been requested, it uses a graph generated only by the neighbouring information */
    IS       verticesIS,verticescomm;
    PetscInt vsize,*idxs;

    ierr = PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&verticesIS);CHKERRQ(ierr);
    ierr = ISGetSize(verticesIS,&vsize);CHKERRQ(ierr);
    ierr = ISGetIndices(verticesIS,(const PetscInt**)&idxs);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pc),vsize,idxs,PETSC_COPY_VALUES,&verticescomm);CHKERRQ(ierr);
    ierr = ISRestoreIndices(verticesIS,(const PetscInt**)&idxs);CHKERRQ(ierr);
    ierr = PCBDDCGraphRestoreCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&verticesIS);CHKERRQ(ierr);
    ierr = PCBDDCGraphCreate(&graph);CHKERRQ(ierr);
    ierr = PCBDDCGraphInit(graph,pcbddc->mat_graph->l2gmap,pcbddc->mat_graph->nvtxs_global,pcbddc->graphmaxcount);CHKERRQ(ierr);
    ierr = PCBDDCGraphSetUp(graph,pcbddc->mat_graph->custom_minimal_size,NULL,pcbddc->DirichletBoundariesLocal,0,NULL,verticescomm);CHKERRQ(ierr);
    ierr = ISDestroy(&verticescomm);CHKERRQ(ierr);
    ierr = PCBDDCGraphComputeConnectedComponents(graph);CHKERRQ(ierr);
  } else {
    graph = pcbddc->mat_graph;
  }
  /* print some info */
  if (pcbddc->dbg_flag && !pcbddc->sub_schurs_rebuild) {
    IS       vertices;
    PetscInt nv,nedges,nfaces;
    ierr = PCBDDCGraphASCIIView(graph,pcbddc->dbg_flag,pcbddc->dbg_viewer);CHKERRQ(ierr);
    ierr = PCBDDCGraphGetCandidatesIS(graph,&nfaces,NULL,&nedges,NULL,&vertices);CHKERRQ(ierr);
    ierr = ISGetSize(vertices,&nv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"--------------------------------------------------------------\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d got %02d local candidate vertices (%D)\n",PetscGlobalRank,(int)nv,pcbddc->use_vertices);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d got %02d local candidate edges    (%D)\n",PetscGlobalRank,(int)nedges,pcbddc->use_edges);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d got %02d local candidate faces    (%D)\n",PetscGlobalRank,(int)nfaces,pcbddc->use_faces);CHKERRQ(ierr);
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(pcbddc->dbg_viewer);CHKERRQ(ierr);
    ierr = PCBDDCGraphRestoreCandidatesIS(graph,&nfaces,NULL,&nedges,NULL,&vertices);CHKERRQ(ierr);
  }

  /* sub_schurs init */
  if (!pcbddc->sub_schurs) {
    ierr = PCBDDCSubSchursCreate(&pcbddc->sub_schurs);CHKERRQ(ierr);
  }
  ierr = PCBDDCSubSchursInit(pcbddc->sub_schurs,((PetscObject)pc)->prefix,pcis->is_I_local,pcis->is_B_local,graph,pcis->BtoNmap,pcbddc->sub_schurs_rebuild);CHKERRQ(ierr);

  /* free graph struct */
  if (pcbddc->sub_schurs_rebuild) {
    ierr = PCBDDCGraphDestroy(&graph);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCCheckOperator(PC pc)
{
  PC_IS               *pcis=(PC_IS*)pc->data;
  PC_BDDC             *pcbddc=(PC_BDDC*)pc->data;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (pcbddc->n_vertices == pcbddc->local_primal_size) {
    IS             zerodiag = NULL;
    Mat            S_j,B0_B=NULL;
    Vec            dummy_vec=NULL,vec_check_B,vec_scale_P;
    PetscScalar    *p0_check,*array,*array2;
    PetscReal      norm;
    PetscInt       i;

    /* B0 and B0_B */
    if (zerodiag) {
      IS       dummy;

      ierr = ISCreateStride(PETSC_COMM_SELF,pcbddc->benign_n,0,1,&dummy);CHKERRQ(ierr);
      ierr = MatCreateSubMatrix(pcbddc->benign_B0,dummy,pcis->is_B_local,MAT_INITIAL_MATRIX,&B0_B);CHKERRQ(ierr);
      ierr = MatCreateVecs(B0_B,NULL,&dummy_vec);CHKERRQ(ierr);
      ierr = ISDestroy(&dummy);CHKERRQ(ierr);
    }
    /* I need a primal vector to scale primal nodes since BDDC sums contibutions */
    ierr = VecDuplicate(pcbddc->vec1_P,&vec_scale_P);CHKERRQ(ierr);
    ierr = VecSet(pcbddc->vec1_P,1.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcbddc->coarse_loc_to_glob,pcbddc->vec1_P,pcbddc->coarse_vec,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcbddc->coarse_loc_to_glob,pcbddc->vec1_P,pcbddc->coarse_vec,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcbddc->coarse_loc_to_glob,pcbddc->coarse_vec,vec_scale_P,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcbddc->coarse_loc_to_glob,pcbddc->coarse_vec,vec_scale_P,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecReciprocal(vec_scale_P);CHKERRQ(ierr);
    /* S_j */
    ierr = MatCreateSchurComplement(pcis->A_II,pcis->pA_II,pcis->A_IB,pcis->A_BI,pcis->A_BB,&S_j);CHKERRQ(ierr);
    ierr = MatSchurComplementSetKSP(S_j,pcbddc->ksp_D);CHKERRQ(ierr);

    /* mimic vector in \widetilde{W}_\Gamma */
    ierr = VecSetRandom(pcis->vec1_N,NULL);CHKERRQ(ierr);
    /* continuous in primal space */
    ierr = VecSetRandom(pcbddc->coarse_vec,NULL);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcbddc->coarse_loc_to_glob,pcbddc->coarse_vec,pcbddc->vec1_P,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcbddc->coarse_loc_to_glob,pcbddc->coarse_vec,pcbddc->vec1_P,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecGetArray(pcbddc->vec1_P,&array);CHKERRQ(ierr);
    ierr = PetscCalloc1(pcbddc->benign_n,&p0_check);CHKERRQ(ierr);
    for (i=0;i<pcbddc->benign_n;i++) p0_check[i] = array[pcbddc->local_primal_size-pcbddc->benign_n+i];
    ierr = VecSetValues(pcis->vec1_N,pcbddc->local_primal_size,pcbddc->local_primal_ref_node,array,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecRestoreArray(pcbddc->vec1_P,&array);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(pcis->vec1_N);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(pcis->vec1_N);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec2_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->N_to_B,pcis->vec1_N,pcis->vec2_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecDuplicate(pcis->vec2_B,&vec_check_B);CHKERRQ(ierr);
    ierr = VecCopy(pcis->vec2_B,vec_check_B);CHKERRQ(ierr);

    /* assemble rhs for coarse problem */
    /* widetilde{S}_\Gamma w_\Gamma + \widetilde{B0}^T_B p0 */
    /* local with Schur */
    ierr = MatMult(S_j,pcis->vec2_B,pcis->vec1_B);CHKERRQ(ierr);
    if (zerodiag) {
      ierr = VecGetArray(dummy_vec,&array);CHKERRQ(ierr);
      for (i=0;i<pcbddc->benign_n;i++) array[i] = p0_check[i];
      ierr = VecRestoreArray(dummy_vec,&array);CHKERRQ(ierr);
      ierr = MatMultTransposeAdd(B0_B,dummy_vec,pcis->vec1_B,pcis->vec1_B);CHKERRQ(ierr);
    }
    /* sum on primal nodes the local contributions */
    ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_B,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->N_to_B,pcis->vec1_B,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    ierr = VecGetArray(pcbddc->vec1_P,&array2);CHKERRQ(ierr);
    for (i=0;i<pcbddc->local_primal_size;i++) array2[i] = array[pcbddc->local_primal_ref_node[i]];
    ierr = VecRestoreArray(pcbddc->vec1_P,&array2);CHKERRQ(ierr);
    ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    ierr = VecSet(pcbddc->coarse_vec,0.);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcbddc->coarse_loc_to_glob,pcbddc->vec1_P,pcbddc->coarse_vec,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcbddc->coarse_loc_to_glob,pcbddc->vec1_P,pcbddc->coarse_vec,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcbddc->coarse_loc_to_glob,pcbddc->coarse_vec,pcbddc->vec1_P,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcbddc->coarse_loc_to_glob,pcbddc->coarse_vec,pcbddc->vec1_P,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecGetArray(pcbddc->vec1_P,&array);CHKERRQ(ierr);
    /* scale primal nodes (BDDC sums contibutions) */
    ierr = VecPointwiseMult(pcbddc->vec1_P,vec_scale_P,pcbddc->vec1_P);CHKERRQ(ierr);
    ierr = VecSetValues(pcis->vec1_N,pcbddc->local_primal_size,pcbddc->local_primal_ref_node,array,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecRestoreArray(pcbddc->vec1_P,&array);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(pcis->vec1_N);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(pcis->vec1_N);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    /* global: \widetilde{B0}_B w_\Gamma */
    if (zerodiag) {
      ierr = MatMult(B0_B,pcis->vec2_B,dummy_vec);CHKERRQ(ierr);
      ierr = VecGetArray(dummy_vec,&array);CHKERRQ(ierr);
      for (i=0;i<pcbddc->benign_n;i++) pcbddc->benign_p0[i] = array[i];
      ierr = VecRestoreArray(dummy_vec,&array);CHKERRQ(ierr);
    }
    /* BDDC */
    ierr = VecSet(pcis->vec1_D,0.);CHKERRQ(ierr);
    ierr = PCBDDCApplyInterfacePreconditioner(pc,PETSC_FALSE);CHKERRQ(ierr);

    ierr = VecCopy(pcis->vec1_B,pcis->vec2_B);CHKERRQ(ierr);
    ierr = VecAXPY(pcis->vec1_B,-1.0,vec_check_B);CHKERRQ(ierr);
    ierr = VecNorm(pcis->vec1_B,NORM_INFINITY,&norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] BDDC local error is %1.4e\n",PetscGlobalRank,norm);CHKERRQ(ierr);
    for (i=0;i<pcbddc->benign_n;i++) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] BDDC p0[%D] error is %1.4e\n",PetscGlobalRank,i,PetscAbsScalar(pcbddc->benign_p0[i]-p0_check[i]));CHKERRQ(ierr);
    }
    ierr = PetscFree(p0_check);CHKERRQ(ierr);
    ierr = VecDestroy(&vec_scale_P);CHKERRQ(ierr);
    ierr = VecDestroy(&vec_check_B);CHKERRQ(ierr);
    ierr = VecDestroy(&dummy_vec);CHKERRQ(ierr);
    ierr = MatDestroy(&S_j);CHKERRQ(ierr);
    ierr = MatDestroy(&B0_B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#include <../src/mat/impls/aij/mpi/mpiaij.h>
PetscErrorCode MatMPIAIJRestrict(Mat A, MPI_Comm ccomm, Mat *B)
{
  Mat            At;
  IS             rows;
  PetscInt       rst,ren;
  PetscErrorCode ierr;
  PetscLayout    rmap;

  PetscFunctionBegin;
  rst = ren = 0;
  if (ccomm != MPI_COMM_NULL) {
    ierr = PetscLayoutCreate(ccomm,&rmap);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(rmap,A->rmap->N);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize(rmap,1);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(rmap);CHKERRQ(ierr);
    ierr = PetscLayoutGetRange(rmap,&rst,&ren);CHKERRQ(ierr);
  }
  ierr = ISCreateStride(PetscObjectComm((PetscObject)A),ren-rst,rst,1,&rows);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix(A,rows,NULL,MAT_INITIAL_MATRIX,&At);CHKERRQ(ierr);
  ierr = ISDestroy(&rows);CHKERRQ(ierr);

  if (ccomm != MPI_COMM_NULL) {
    Mat_MPIAIJ *a,*b;
    IS         from,to;
    Vec        gvec;
    PetscInt   lsize;

    ierr = MatCreate(ccomm,B);CHKERRQ(ierr);
    ierr = MatSetSizes(*B,ren-rst,PETSC_DECIDE,PETSC_DECIDE,At->cmap->N);CHKERRQ(ierr);
    ierr = MatSetType(*B,MATAIJ);CHKERRQ(ierr);
    ierr = PetscLayoutDestroy(&((*B)->rmap));CHKERRQ(ierr);
    ierr = PetscLayoutSetUp((*B)->cmap);CHKERRQ(ierr);
    a    = (Mat_MPIAIJ*)At->data;
    b    = (Mat_MPIAIJ*)(*B)->data;
    ierr = MPI_Comm_size(ccomm,&b->size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(ccomm,&b->rank);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)a->A);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)a->B);CHKERRQ(ierr);
    b->A = a->A;
    b->B = a->B;

    b->donotstash      = a->donotstash;
    b->roworiented     = a->roworiented;
    b->rowindices      = 0;
    b->rowvalues       = 0;
    b->getrowactive    = PETSC_FALSE;

    (*B)->rmap         = rmap;
    (*B)->factortype   = A->factortype;
    (*B)->assembled    = PETSC_TRUE;
    (*B)->insertmode   = NOT_SET_VALUES;
    (*B)->preallocated = PETSC_TRUE;

    if (a->colmap) {
#if defined(PETSC_USE_CTABLE)
      ierr = PetscTableCreateCopy(a->colmap,&b->colmap);CHKERRQ(ierr);
#else
      ierr = PetscMalloc1(At->cmap->N,&b->colmap);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)*B,At->cmap->N*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscArraycpy(b->colmap,a->colmap,At->cmap->N);CHKERRQ(ierr);
#endif
    } else b->colmap = 0;
    if (a->garray) {
      PetscInt len;
      len  = a->B->cmap->n;
      ierr = PetscMalloc1(len+1,&b->garray);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)(*B),len*sizeof(PetscInt));CHKERRQ(ierr);
      if (len) { ierr = PetscArraycpy(b->garray,a->garray,len);CHKERRQ(ierr); }
    } else b->garray = 0;

    ierr    = PetscObjectReference((PetscObject)a->lvec);CHKERRQ(ierr);
    b->lvec = a->lvec;
    ierr    = PetscLogObjectParent((PetscObject)*B,(PetscObject)b->lvec);CHKERRQ(ierr);

    /* cannot use VecScatterCopy */
    ierr = VecGetLocalSize(b->lvec,&lsize);CHKERRQ(ierr);
    ierr = ISCreateGeneral(ccomm,lsize,b->garray,PETSC_USE_POINTER,&from);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,lsize,0,1,&to);CHKERRQ(ierr);
    ierr = MatCreateVecs(*B,&gvec,NULL);CHKERRQ(ierr);
    ierr = VecScatterCreate(gvec,from,b->lvec,to,&b->Mvctx);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)*B,(PetscObject)b->Mvctx);CHKERRQ(ierr);
    ierr = ISDestroy(&from);CHKERRQ(ierr);
    ierr = ISDestroy(&to);CHKERRQ(ierr);
    ierr = VecDestroy(&gvec);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&At);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
