static char help[] = "Demonstrates BuildTwoSided functions.\n";

#include <petscsys.h>

typedef struct {
  PetscInt    rank;
  PetscScalar value;
  char        ok[3];
} Unit;

static PetscErrorCode MakeDatatype(MPI_Datatype *dtype)
{
  PetscErrorCode ierr;
  MPI_Datatype dtypes[3],tmptype;
  PetscMPIInt  lengths[3];
  MPI_Aint     displs[3];
  Unit         dummy;

  PetscFunctionBegin;
  dtypes[0] = MPIU_INT;
  dtypes[1] = MPIU_SCALAR;
  dtypes[2] = MPI_CHAR;
  lengths[0] = 1;
  lengths[1] = 1;
  lengths[2] = 3;
  /* Curse the evil beings that made std::complex a non-POD type. */
  displs[0] = (char*)&dummy.rank - (char*)&dummy;  /* offsetof(Unit,rank); */
  displs[1] = (char*)&dummy.value - (char*)&dummy; /* offsetof(Unit,value); */
  displs[2] = (char*)&dummy.ok - (char*)&dummy;    /* offsetof(Unit,ok); */
  ierr = MPI_Type_create_struct(3,lengths,displs,dtypes,&tmptype);CHKERRMPI(ierr);
  ierr = MPI_Type_commit(&tmptype);CHKERRMPI(ierr);
  ierr = MPI_Type_create_resized(tmptype,0,sizeof(Unit),dtype);CHKERRMPI(ierr);
  ierr = MPI_Type_commit(dtype);CHKERRMPI(ierr);
  ierr = MPI_Type_free(&tmptype);CHKERRMPI(ierr);
  {
    MPI_Aint lb,extent;
    ierr = MPI_Type_get_extent(*dtype,&lb,&extent);CHKERRMPI(ierr);
    PetscAssertFalse(extent != sizeof(Unit),PETSC_COMM_WORLD,PETSC_ERR_LIB,"New type has extent %d != sizeof(Unit) %d",(int)extent,(int)sizeof(Unit));
  }
  PetscFunctionReturn(0);
}

struct FCtx {
  PetscMPIInt rank;
  PetscMPIInt nto;
  PetscMPIInt *toranks;
  Unit *todata;
  PetscSegBuffer seg;
};

static PetscErrorCode FSend(MPI_Comm comm,const PetscMPIInt tag[],PetscMPIInt tonum,PetscMPIInt rank,void *todata,MPI_Request req[],void *ctx)
{
  struct FCtx *fctx = (struct FCtx*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscAssertFalse(rank != fctx->toranks[tonum],PETSC_COMM_SELF,PETSC_ERR_PLIB,"Rank %d does not match toranks[%d] %d",rank,tonum,fctx->toranks[tonum]);
  PetscAssertFalse(fctx->rank != *(PetscMPIInt*)todata,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Todata %d does not match rank %d",*(PetscMPIInt*)todata,fctx->rank);
  ierr = MPI_Isend(&fctx->todata[tonum].rank,1,MPIU_INT,rank,tag[0],comm,&req[0]);CHKERRMPI(ierr);
  ierr = MPI_Isend(&fctx->todata[tonum].value,1,MPIU_SCALAR,rank,tag[1],comm,&req[1]);CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FRecv(MPI_Comm comm,const PetscMPIInt tag[],PetscMPIInt rank,void *fromdata,MPI_Request req[],void *ctx)
{
  struct FCtx *fctx = (struct FCtx*)ctx;
  PetscErrorCode ierr;
  Unit           *buf;

  PetscFunctionBegin;
  PetscAssertFalse(*(PetscMPIInt*)fromdata != rank,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Dummy data %d from rank %d corrupt",*(PetscMPIInt*)fromdata,rank);
  ierr = PetscSegBufferGet(fctx->seg,1,&buf);CHKERRQ(ierr);
  ierr = MPI_Irecv(&buf->rank,1,MPIU_INT,rank,tag[0],comm,&req[0]);CHKERRMPI(ierr);
  ierr = MPI_Irecv(&buf->value,1,MPIU_SCALAR,rank,tag[1],comm,&req[1]);CHKERRMPI(ierr);
  buf->ok[0] = 'o';
  buf->ok[1] = 'k';
  buf->ok[2] = 0;
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size,*toranks,*fromranks,nto,nfrom;
  PetscInt       i,n;
  PetscBool      verbose,build_twosided_f;
  Unit           *todata,*fromdata;
  MPI_Datatype   dtype;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  verbose = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-verbose",&verbose,NULL);CHKERRQ(ierr);
  build_twosided_f = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-build_twosided_f",&build_twosided_f,NULL);CHKERRQ(ierr);

  for (i=1,nto=0; i<size; i*=2) nto++;
  ierr = PetscMalloc2(nto,&todata,nto,&toranks);CHKERRQ(ierr);
  for (n=0,i=1; i<size; n++,i*=2) {
    toranks[n] = (rank+i) % size;
    todata[n].rank  = (rank+i) % size;
    todata[n].value = (PetscScalar)rank;
    todata[n].ok[0] = 'o';
    todata[n].ok[1] = 'k';
    todata[n].ok[2] = 0;
  }
  if (verbose) {
    for (i=0; i<nto; i++) {
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] TO %d: {%" PetscInt_FMT ", %g, \"%s\"}\n",rank,toranks[i],todata[i].rank,(double)PetscRealPart(todata[i].value),todata[i].ok);CHKERRQ(ierr);
    }
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
  }

  ierr = MakeDatatype(&dtype);CHKERRQ(ierr);

  if (build_twosided_f) {
    struct FCtx fctx;
    PetscMPIInt *todummy,*fromdummy;
    fctx.rank    = rank;
    fctx.nto     = nto;
    fctx.toranks = toranks;
    fctx.todata  = todata;
    ierr = PetscSegBufferCreate(sizeof(Unit),1,&fctx.seg);CHKERRQ(ierr);
    ierr = PetscMalloc1(nto,&todummy);CHKERRQ(ierr);
    for (i=0; i<nto; i++) todummy[i] = rank;
    ierr = PetscCommBuildTwoSidedF(PETSC_COMM_WORLD,1,MPI_INT,nto,toranks,todummy,&nfrom,&fromranks,&fromdummy,2,FSend,FRecv,&fctx);CHKERRQ(ierr);
    ierr = PetscFree(todummy);CHKERRQ(ierr);
    ierr = PetscFree(fromdummy);CHKERRQ(ierr);
    ierr = PetscSegBufferExtractAlloc(fctx.seg,&fromdata);CHKERRQ(ierr);
    ierr = PetscSegBufferDestroy(&fctx.seg);CHKERRQ(ierr);
  } else {
    ierr = PetscCommBuildTwoSided(PETSC_COMM_WORLD,1,dtype,nto,toranks,todata,&nfrom,&fromranks,&fromdata);CHKERRQ(ierr);
  }
  ierr = MPI_Type_free(&dtype);CHKERRMPI(ierr);

  if (verbose) {
    PetscInt *iranks,*iperm;
    ierr = PetscMalloc2(nfrom,&iranks,nfrom,&iperm);CHKERRQ(ierr);
    for (i=0; i<nfrom; i++) {
      iranks[i] = fromranks[i];
      iperm[i] = i;
    }
    /* Receive ordering is non-deterministic in general, so sort to make verbose output deterministic. */
    ierr = PetscSortIntWithPermutation(nfrom,iranks,iperm);CHKERRQ(ierr);
    for (i=0; i<nfrom; i++) {
      PetscInt ip = iperm[i];
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] FROM %d: {%" PetscInt_FMT ", %g, \"%s\"}\n",rank,fromranks[ip],fromdata[ip].rank,(double)PetscRealPart(fromdata[ip].value),fromdata[ip].ok);CHKERRQ(ierr);
    }
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
    ierr = PetscFree2(iranks,iperm);CHKERRQ(ierr);
  }

  PetscAssertFalse(nto != nfrom,PETSC_COMM_SELF,PETSC_ERR_PLIB,"[%d] From ranks %d does not match To ranks %d",rank,nto,nfrom);
  for (i=1; i<size; i*=2) {
    PetscMPIInt expected_rank = (rank-i+size)%size;
    PetscBool flg;
    for (n=0; n<nfrom; n++) {
      if (expected_rank == fromranks[n]) goto found;
    }
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"[%d] Could not find expected from rank %d",rank,expected_rank);
    found:
    PetscAssertFalse(PetscRealPart(fromdata[n].value) != expected_rank,PETSC_COMM_SELF,PETSC_ERR_PLIB,"[%d] Got data %g from rank %d",rank,(double)PetscRealPart(fromdata[n].value),expected_rank);
    ierr = PetscStrcmp(fromdata[n].ok,"ok",&flg);CHKERRQ(ierr);
    PetscAssertFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"[%d] Got string %s from rank %d",rank,fromdata[n].ok,expected_rank);
  }
  ierr = PetscFree2(todata,toranks);CHKERRQ(ierr);
  ierr = PetscFree(fromdata);CHKERRQ(ierr);
  ierr = PetscFree(fromranks);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 4
      args: -verbose -build_twosided allreduce

   test:
      suffix: f
      nsize: 4
      args: -verbose -build_twosided_f -build_twosided allreduce
      output_file: output/ex8_1.out

   test:
      suffix: f_ibarrier
      nsize: 4
      args: -verbose -build_twosided_f -build_twosided ibarrier
      output_file: output/ex8_1.out
      requires: defined(PETSC_HAVE_MPI_IBARRIER)

   test:
      suffix: ibarrier
      nsize: 4
      args: -verbose -build_twosided ibarrier
      output_file: output/ex8_1.out
      requires: defined(PETSC_HAVE_MPI_IBARRIER)

   test:
      suffix: redscatter
      requires: mpi_reduce_scatter_block
      nsize: 4
      args: -verbose -build_twosided redscatter
      output_file: output/ex8_1.out

TEST*/
