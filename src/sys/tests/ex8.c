static char help[] = "Demonstrates BuildTwoSided functions.\n";

#include <petscsys.h>

typedef struct {
  PetscInt    rank;
  PetscScalar value;
  char        ok[3];
} Unit;

static PetscErrorCode MakeDatatype(MPI_Datatype *dtype)
{
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
  CHKERRMPI(MPI_Type_create_struct(3,lengths,displs,dtypes,&tmptype));
  CHKERRMPI(MPI_Type_commit(&tmptype));
  CHKERRMPI(MPI_Type_create_resized(tmptype,0,sizeof(Unit),dtype));
  CHKERRMPI(MPI_Type_commit(dtype));
  CHKERRMPI(MPI_Type_free(&tmptype));
  {
    MPI_Aint lb,extent;
    CHKERRMPI(MPI_Type_get_extent(*dtype,&lb,&extent));
    PetscCheckFalse(extent != sizeof(Unit),PETSC_COMM_WORLD,PETSC_ERR_LIB,"New type has extent %d != sizeof(Unit) %d",(int)extent,(int)sizeof(Unit));
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

  PetscFunctionBegin;
  PetscCheckFalse(rank != fctx->toranks[tonum],PETSC_COMM_SELF,PETSC_ERR_PLIB,"Rank %d does not match toranks[%d] %d",rank,tonum,fctx->toranks[tonum]);
  PetscCheckFalse(fctx->rank != *(PetscMPIInt*)todata,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Todata %d does not match rank %d",*(PetscMPIInt*)todata,fctx->rank);
  CHKERRMPI(MPI_Isend(&fctx->todata[tonum].rank,1,MPIU_INT,rank,tag[0],comm,&req[0]));
  CHKERRMPI(MPI_Isend(&fctx->todata[tonum].value,1,MPIU_SCALAR,rank,tag[1],comm,&req[1]));
  PetscFunctionReturn(0);
}

static PetscErrorCode FRecv(MPI_Comm comm,const PetscMPIInt tag[],PetscMPIInt rank,void *fromdata,MPI_Request req[],void *ctx)
{
  struct FCtx *fctx = (struct FCtx*)ctx;
  Unit           *buf;

  PetscFunctionBegin;
  PetscCheckFalse(*(PetscMPIInt*)fromdata != rank,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Dummy data %d from rank %d corrupt",*(PetscMPIInt*)fromdata,rank);
  CHKERRQ(PetscSegBufferGet(fctx->seg,1,&buf));
  CHKERRMPI(MPI_Irecv(&buf->rank,1,MPIU_INT,rank,tag[0],comm,&req[0]));
  CHKERRMPI(MPI_Irecv(&buf->value,1,MPIU_SCALAR,rank,tag[1],comm,&req[1]));
  buf->ok[0] = 'o';
  buf->ok[1] = 'k';
  buf->ok[2] = 0;
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscMPIInt    rank,size,*toranks,*fromranks,nto,nfrom;
  PetscInt       i,n;
  PetscBool      verbose,build_twosided_f;
  Unit           *todata,*fromdata;
  MPI_Datatype   dtype;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  verbose = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-verbose",&verbose,NULL));
  build_twosided_f = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-build_twosided_f",&build_twosided_f,NULL));

  for (i=1,nto=0; i<size; i*=2) nto++;
  CHKERRQ(PetscMalloc2(nto,&todata,nto,&toranks));
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
      CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] TO %d: {%" PetscInt_FMT ", %g, \"%s\"}\n",rank,toranks[i],todata[i].rank,(double)PetscRealPart(todata[i].value),todata[i].ok));
    }
    CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  }

  CHKERRQ(MakeDatatype(&dtype));

  if (build_twosided_f) {
    struct FCtx fctx;
    PetscMPIInt *todummy,*fromdummy;
    fctx.rank    = rank;
    fctx.nto     = nto;
    fctx.toranks = toranks;
    fctx.todata  = todata;
    CHKERRQ(PetscSegBufferCreate(sizeof(Unit),1,&fctx.seg));
    CHKERRQ(PetscMalloc1(nto,&todummy));
    for (i=0; i<nto; i++) todummy[i] = rank;
    CHKERRQ(PetscCommBuildTwoSidedF(PETSC_COMM_WORLD,1,MPI_INT,nto,toranks,todummy,&nfrom,&fromranks,&fromdummy,2,FSend,FRecv,&fctx));
    CHKERRQ(PetscFree(todummy));
    CHKERRQ(PetscFree(fromdummy));
    CHKERRQ(PetscSegBufferExtractAlloc(fctx.seg,&fromdata));
    CHKERRQ(PetscSegBufferDestroy(&fctx.seg));
  } else {
    CHKERRQ(PetscCommBuildTwoSided(PETSC_COMM_WORLD,1,dtype,nto,toranks,todata,&nfrom,&fromranks,&fromdata));
  }
  CHKERRMPI(MPI_Type_free(&dtype));

  if (verbose) {
    PetscInt *iranks,*iperm;
    CHKERRQ(PetscMalloc2(nfrom,&iranks,nfrom,&iperm));
    for (i=0; i<nfrom; i++) {
      iranks[i] = fromranks[i];
      iperm[i] = i;
    }
    /* Receive ordering is non-deterministic in general, so sort to make verbose output deterministic. */
    CHKERRQ(PetscSortIntWithPermutation(nfrom,iranks,iperm));
    for (i=0; i<nfrom; i++) {
      PetscInt ip = iperm[i];
      CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] FROM %d: {%" PetscInt_FMT ", %g, \"%s\"}\n",rank,fromranks[ip],fromdata[ip].rank,(double)PetscRealPart(fromdata[ip].value),fromdata[ip].ok));
    }
    CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
    CHKERRQ(PetscFree2(iranks,iperm));
  }

  PetscCheckFalse(nto != nfrom,PETSC_COMM_SELF,PETSC_ERR_PLIB,"[%d] From ranks %d does not match To ranks %d",rank,nto,nfrom);
  for (i=1; i<size; i*=2) {
    PetscMPIInt expected_rank = (rank-i+size)%size;
    PetscBool flg;
    for (n=0; n<nfrom; n++) {
      if (expected_rank == fromranks[n]) goto found;
    }
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"[%d] Could not find expected from rank %d",rank,expected_rank);
    found:
    PetscCheckFalse(PetscRealPart(fromdata[n].value) != expected_rank,PETSC_COMM_SELF,PETSC_ERR_PLIB,"[%d] Got data %g from rank %d",rank,(double)PetscRealPart(fromdata[n].value),expected_rank);
    CHKERRQ(PetscStrcmp(fromdata[n].ok,"ok",&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"[%d] Got string %s from rank %d",rank,fromdata[n].ok,expected_rank);
  }
  CHKERRQ(PetscFree2(todata,toranks));
  CHKERRQ(PetscFree(fromdata));
  CHKERRQ(PetscFree(fromranks));
  CHKERRQ(PetscFinalize());
  return 0;
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
      requires: defined(PETSC_HAVE_MPI_NONBLOCKING_COLLECTIVES)

   test:
      suffix: ibarrier
      nsize: 4
      args: -verbose -build_twosided ibarrier
      output_file: output/ex8_1.out
      requires: defined(PETSC_HAVE_MPI_NONBLOCKING_COLLECTIVES)

   test:
      suffix: redscatter
      requires: mpi_reduce_scatter_block
      nsize: 4
      args: -verbose -build_twosided redscatter
      output_file: output/ex8_1.out

TEST*/
