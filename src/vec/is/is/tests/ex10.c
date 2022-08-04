
static char help[] = "Tests ISFilter().\n\n";

#include <petscis.h>
#include <petscviewer.h>

static PetscErrorCode CreateIS(MPI_Comm comm, PetscInt n, PetscInt first, PetscInt step, IS *is)
{
  PetscInt       *idx, i, j;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  *is = NULL;
  first += rank;
  PetscCall(PetscMalloc1(n,&idx));
  for (i=0,j=first; i<n; i++,j+=step) idx[i] = j;
  PetscCall(ISCreateGeneral(comm,n,idx,PETSC_OWN_POINTER,is));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  IS             is;
  PetscInt       n=10, N, first=0, step=0, start, end;
  PetscMPIInt    rank;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-first",&first,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-step",&step,NULL));
  start = 0; end = n;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-start",&start,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-end",&end,NULL));

  PetscCall(CreateIS(comm, n, first, step, &is));
  PetscCall(ISGeneralFilter(is, start, end));
  PetscCall(ISView(is,PETSC_VIEWER_STDOUT_(comm)));
  PetscCall(ISGetSize(is, &N));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_(comm), "global size: %" PetscInt_FMT "\n", N));

  PetscCall(ISDestroy(&is));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      suffix: 1
      nsize: 4
      args: -n 6
      args: -first -2
      args: -step 1
      args: -start {{-2 4}separate output} -end {{2 6}separate output}

TEST*/
