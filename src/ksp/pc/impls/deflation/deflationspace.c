#include <../src/ksp/pc/impls/deflation/deflation.h> /*I "petscksp.h" I*/

PetscScalar db2[] = {0.7071067811865476,0.7071067811865476};

PetscScalar db4[] = {-0.12940952255092145,0.22414386804185735,0.836516303737469,0.48296291314469025};

PetscScalar db8[] = {-0.010597401784997278,
0.032883011666982945,
0.030841381835986965,
-0.18703481171888114,
-0.02798376941698385,
0.6308807679295904,
0.7148465705525415,
0.23037781330885523};

PetscScalar db16[] = {-0.00011747678400228192,
0.0006754494059985568,
-0.0003917403729959771,
-0.00487035299301066,
0.008746094047015655,
0.013981027917015516,
-0.04408825393106472,
-0.01736930100202211,
0.128747426620186,
0.00047248457399797254,
-0.2840155429624281,
-0.015829105256023893,
0.5853546836548691,
0.6756307362980128,
0.3128715909144659,
0.05441584224308161};

PetscScalar biorth22[] = {0.0,
-0.1767766952966369,
0.3535533905932738,
1.0606601717798214,
0.3535533905932738,
-0.1767766952966369};

PetscScalar meyer[] = {0.0,-1.009999956941423e-12,8.519459636796214e-09,-1.111944952595278e-08,-1.0798819539621958e-08,6.066975741351135e-08,-1.0866516536735883e-07,8.200680650386481e-08,1.1783004497663934e-07,-5.506340565252278e-07,1.1307947017916706e-06,-1.489549216497156e-06,7.367572885903746e-07,3.20544191334478e-06,-1.6312699734552807e-05,6.554305930575149e-05,-0.0006011502343516092,-0.002704672124643725,0.002202534100911002,0.006045814097323304,-0.006387718318497156,-0.011061496392513451,0.015270015130934803,0.017423434103729693,-0.03213079399021176,-0.024348745906078023,0.0637390243228016,0.030655091960824263,-0.13284520043622938,-0.035087555656258346,0.44459300275757724,0.7445855923188063,0.44459300275757724,-0.035087555656258346,-0.13284520043622938,0.030655091960824263,0.0637390243228016,-0.024348745906078023,-0.03213079399021176,0.017423434103729693,0.015270015130934803,-0.011061496392513451,-0.006387718318497156,0.006045814097323304,0.002202534100911002,-0.002704672124643725,-0.0006011502343516092,6.554305930575149e-05,-1.6312699734552807e-05,3.20544191334478e-06,7.367572885903746e-07,-1.489549216497156e-06,1.1307947017916706e-06,-5.506340565252278e-07,1.1783004497663934e-07,8.200680650386481e-08,-1.0866516536735883e-07,6.066975741351135e-08,-1.0798819539621958e-08,-1.111944952595278e-08,8.519459636796214e-09,-1.009999956941423e-12};

static PetscErrorCode PCDeflationCreateSpaceWave(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt ncoeffs,PetscScalar *coeffs,PetscBool trunc,Mat *H)
{
  Mat            defl;
  PetscInt       i,j,k,ilo,ihi,*Iidx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(ncoeffs,&Iidx);CHKERRQ(ierr);

  ierr = MatCreate(comm,&defl);CHKERRQ(ierr);
  ierr = MatSetSizes(defl,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetUp(defl);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(defl,ncoeffs,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(defl,ncoeffs,NULL,ncoeffs,NULL);CHKERRQ(ierr);
  ierr = MatSetOption(defl,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(defl,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);

  /* Alg 735 Taswell: fvecmat */
  k = ncoeffs -2;
  if (trunc) k = k/2;

  ierr = MatGetOwnershipRange(defl,&ilo,&ihi);CHKERRQ(ierr);
  for (i=0; i<ncoeffs; i++) {
    Iidx[i] = i+ilo*2 -k;
    if (Iidx[i] >= N) Iidx[i] = PETSC_MIN_INT;
  }
  for (i=ilo; i<ihi; i++) {
    ierr = MatSetValues(defl,1,&i,ncoeffs,Iidx,coeffs,INSERT_VALUES);CHKERRQ(ierr);
    for (j=0; j<ncoeffs; j++) {
      Iidx[j] += 2;
      if (Iidx[j] >= N) Iidx[j] = PETSC_MIN_INT;
    }
  }

  ierr = MatAssemblyBegin(defl,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(defl,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscFree(Iidx);CHKERRQ(ierr);
  *H = defl;
  PetscFunctionReturn(0);
}

PetscErrorCode PCDeflationGetSpaceHaar(PC pc,Mat *W,PetscInt size)
{
  Mat            A,defl;
  PetscInt       i,j,len,ilo,ihi,*Iidx,m,M;
  PetscScalar    *col,val;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Haar basis wavelet, level=size */
  len = pow(2,size);
  ierr = PetscMalloc2(len,&col,len,&Iidx);CHKERRQ(ierr);
  val = 1./pow(2,size/2.);
  for (i=0; i<len; i++) col[i] = val;

  ierr = PCGetOperators(pc,NULL,&A);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,NULL);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&defl);CHKERRQ(ierr);
  ierr = MatSetSizes(defl,m,PETSC_DECIDE,M,(PetscInt)ceil(M/(float)len));CHKERRQ(ierr);
  ierr = MatSetUp(defl);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(defl,size,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(defl,size,NULL,size,NULL);CHKERRQ(ierr);
  ierr = MatSetOption(defl,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  ierr = MatGetOwnershipRangeColumn(defl,&ilo,&ihi);CHKERRQ(ierr);
  for (i=0; i<len; i++) Iidx[i] = i+ilo*len;
  if (M%len && ihi == (int)ceil(M/(float)len)) ihi -= 1;
  for (i=ilo; i<ihi; i++) {
    ierr = MatSetValues(defl,len,Iidx,1,&i,col,INSERT_VALUES);CHKERRQ(ierr);
    for (j=0; j<len; j++) Iidx[j] += len;
  }
  if (M%len && ihi+1 == ceil(M/(float)len)) {
    len = M%len;
    val = 1./pow(pow(2,len),0.5);
    for (i=0; i<len; i++) col[i] = val;
    ierr = MatSetValues(defl,len,Iidx,1,&ihi,col,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(defl,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(defl,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscFree2(col,Iidx);CHKERRQ(ierr);
  *W = defl;
  PetscFunctionReturn(0);
}

PetscErrorCode PCDeflationGetSpaceWave(PC pc,Mat *W,PetscInt size,PetscInt ncoeffs,PetscScalar *coeffs,PetscBool trunc)
{
  Mat            A,*H,defl;
  PetscInt       i,m,M,Mdefl,Ndefl;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  ierr = PetscMalloc1(size,&H);CHKERRQ(ierr);
  ierr = PCGetOperators(pc,&A,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,NULL);CHKERRQ(ierr);
  Mdefl = M;
  Ndefl = M;
  for (i=0; i<size; i++) {
    if (Mdefl%2)  {
      if (trunc) Mdefl = (PetscInt)PetscCeilReal(Mdefl/2.);
      else       Mdefl = (PetscInt)PetscFloorReal((ncoeffs+Mdefl-1)/2.);
    } else       Mdefl = Mdefl/2;
    ierr = PCDeflationCreateSpaceWave(comm,PETSC_DECIDE,m,Mdefl,Ndefl,ncoeffs,coeffs,trunc,&H[i]);CHKERRQ(ierr);
    ierr = MatGetLocalSize(H[i],&m,NULL);CHKERRQ(ierr);
    Ndefl = Mdefl;
  }
  ierr = MatCreateComposite(comm,size,H,&defl);CHKERRQ(ierr);
  ierr = MatCompositeSetType(defl,MAT_COMPOSITE_MULTIPLICATIVE);CHKERRQ(ierr);
  *W = defl;

  for (i=0; i<size; i++) {
    ierr = MatDestroy(&H[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(H);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCDeflationGetSpaceAggregation(PC pc,Mat *W)
{
  Mat            A,defl;
  PetscInt       i,ilo,ihi,*Iidx,M;
  PetscMPIInt    m;
  PetscScalar    *col;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCGetOperators(pc,&A,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(A,&ilo,&ihi);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&m);CHKERRMPI(ierr);
  ierr = MatCreate(comm,&defl);CHKERRQ(ierr);
  ierr = MatSetSizes(defl,ihi-ilo,1,M,m);CHKERRQ(ierr);
  ierr = MatSetUp(defl);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(defl,1,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(defl,1,NULL,0,NULL);CHKERRQ(ierr);
  ierr = MatSetOption(defl,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(defl,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);

  ierr = PetscMalloc2(ihi-ilo,&col,ihi-ilo,&Iidx);CHKERRQ(ierr);
  for (i=ilo; i<ihi; i++) {
    Iidx[i-ilo] = i;
    col[i-ilo] = 1;
  }
  ierr = MPI_Comm_rank(comm,&m);CHKERRMPI(ierr);
  i = m;
  ierr = MatSetValues(defl,ihi-ilo,Iidx,1,&i,col,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(defl,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(defl,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscFree2(col,Iidx);CHKERRQ(ierr);
  *W = defl;
  PetscFunctionReturn(0);
}

PetscErrorCode PCDeflationComputeSpace(PC pc)
{
  Mat            defl;
  PetscBool      transp=PETSC_TRUE;
  PC_Deflation   *def = (PC_Deflation*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (def->spacesize < 1) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG,"Wrong PCDeflation space size specified: %D",def->spacesize);
  switch (def->spacetype) {
    case PC_DEFLATION_SPACE_HAAR:
      transp = PETSC_FALSE;
      ierr = PCDeflationGetSpaceHaar(pc,&defl,def->spacesize);CHKERRQ(ierr);break;
    case PC_DEFLATION_SPACE_DB2:
      ierr = PCDeflationGetSpaceWave(pc,&defl,def->spacesize,2,db2,PetscNot(def->extendsp));CHKERRQ(ierr);break;
    case PC_DEFLATION_SPACE_DB4:
      ierr = PCDeflationGetSpaceWave(pc,&defl,def->spacesize,4,db4,PetscNot(def->extendsp));CHKERRQ(ierr);break;
    case PC_DEFLATION_SPACE_DB8:
      ierr = PCDeflationGetSpaceWave(pc,&defl,def->spacesize,8,db8,PetscNot(def->extendsp));CHKERRQ(ierr);break;
    case PC_DEFLATION_SPACE_DB16:
      ierr = PCDeflationGetSpaceWave(pc,&defl,def->spacesize,16,db16,PetscNot(def->extendsp));CHKERRQ(ierr);break;
    case PC_DEFLATION_SPACE_BIORTH22:
      ierr = PCDeflationGetSpaceWave(pc,&defl,def->spacesize,6,biorth22,PetscNot(def->extendsp));CHKERRQ(ierr);break;
    case PC_DEFLATION_SPACE_MEYER:
      ierr = PCDeflationGetSpaceWave(pc,&defl,def->spacesize,62,meyer,PetscNot(def->extendsp));CHKERRQ(ierr);break;
    case PC_DEFLATION_SPACE_AGGREGATION:
      transp = PETSC_FALSE;
      ierr = PCDeflationGetSpaceAggregation(pc,&defl);CHKERRQ(ierr);break;
    default: SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG,"Wrong PCDeflationSpaceType specified");
  }

  ierr = PCDeflationSetSpace(pc,defl,transp);CHKERRQ(ierr);
  ierr = MatDestroy(&defl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

