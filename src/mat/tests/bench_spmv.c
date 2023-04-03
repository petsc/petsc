static char help[] = "Driver for benchmarking SpMV.";

#include <petscmat.h>
#include "cJSON.h"
#include "mmloader.h"

char *read_file(const char *filename)
{
  FILE  *file       = NULL;
  long   length     = 0;
  char  *content    = NULL;
  size_t read_chars = 0;

  /* open in read binary mode */
  file = fopen(filename, "rb");
  if (file) {
    /* get the length */
    fseek(file, 0, SEEK_END);
    length = ftell(file);
    fseek(file, 0, SEEK_SET);
    /* allocate content buffer */
    content = (char *)malloc((size_t)length + sizeof(""));
    /* read the file into memory */
    read_chars          = fread(content, sizeof(char), (size_t)length, file);
    content[read_chars] = '\0';
    fclose(file);
  }
  return content;
}

void write_file(const char *filename, const char *content)
{
  FILE *file = NULL;
  file       = fopen(filename, "w");
  if (file) { fputs(content, file); }
  fclose(file);
}

int ParseJSON(const char *const inputjsonfile, char ***outputfilenames, char ***outputgroupnames, char ***outputmatnames, int *nmat)
{
  char        *content     = read_file(inputjsonfile);
  cJSON       *matrix_json = NULL;
  const cJSON *problem = NULL, *elem = NULL;
  const cJSON *item = NULL;
  char       **filenames, **groupnames, **matnames;
  int          i, n;
  if (!content) return 0;
  matrix_json = cJSON_Parse(content);
  if (!matrix_json) return 0;
  n          = cJSON_GetArraySize(matrix_json);
  *nmat      = n;
  filenames  = (char **)malloc(sizeof(char *) * n);
  groupnames = (char **)malloc(sizeof(char *) * n);
  matnames   = (char **)malloc(sizeof(char *) * n);
  for (i = 0; i < n; i++) {
    elem         = cJSON_GetArrayItem(matrix_json, i);
    item         = cJSON_GetObjectItemCaseSensitive(elem, "filename");
    filenames[i] = (char *)malloc(sizeof(char) * (strlen(item->valuestring) + 1));
    strcpy(filenames[i], item->valuestring);
    problem       = cJSON_GetObjectItemCaseSensitive(elem, "problem");
    item          = cJSON_GetObjectItemCaseSensitive(problem, "group");
    groupnames[i] = (char *)malloc(sizeof(char) * strlen(item->valuestring) + 1);
    strcpy(groupnames[i], item->valuestring);
    item        = cJSON_GetObjectItemCaseSensitive(problem, "name");
    matnames[i] = (char *)malloc(sizeof(char) * strlen(item->valuestring) + 1);
    strcpy(matnames[i], item->valuestring);
  }
  cJSON_Delete(matrix_json);
  free(content);
  *outputfilenames  = filenames;
  *outputgroupnames = groupnames;
  *outputmatnames   = matnames;
  return 0;
}

int UpdateJSON(const char *const inputjsonfile, PetscReal *spmv_times, PetscReal starting_spmv_time, const char *const matformat, PetscBool use_gpu, PetscInt repetitions)
{
  char  *content     = read_file(inputjsonfile);
  cJSON *matrix_json = NULL;
  cJSON *elem        = NULL;
  int    i, n;
  if (!content) return 0;
  matrix_json = cJSON_Parse(content);
  if (!matrix_json) return 0;
  n = cJSON_GetArraySize(matrix_json);
  for (i = 0; i < n; i++) {
    cJSON *spmv   = NULL;
    cJSON *format = NULL;
    elem          = cJSON_GetArrayItem(matrix_json, i);
    spmv          = cJSON_GetObjectItem(elem, "spmv");
    if (spmv) {
      format = cJSON_GetObjectItem(spmv, matformat);
      if (format) {
        cJSON_SetNumberValue(cJSON_GetObjectItem(format, "time"), (spmv_times[i] - ((i == 0) ? starting_spmv_time : spmv_times[i - 1])) / repetitions);
        cJSON_SetIntValue(cJSON_GetObjectItem(format, "repetitions"), repetitions);
      } else {
        format = cJSON_CreateObject();
        cJSON_AddItemToObject(spmv, matformat, format);
        cJSON_AddNumberToObject(format, "time", (spmv_times[i] - ((i == 0) ? starting_spmv_time : spmv_times[i - 1])) / repetitions);
        cJSON_AddNumberToObject(format, "repetitions", repetitions);
      }
    } else {
      spmv = cJSON_CreateObject();
      cJSON_AddItemToObject(elem, "spmv", spmv);
      format = cJSON_CreateObject();
      cJSON_AddItemToObject(spmv, matformat, format);
      cJSON_AddNumberToObject(format, "time", (spmv_times[i] - ((i == 0) ? starting_spmv_time : spmv_times[i - 1])) / repetitions);
      cJSON_AddNumberToObject(format, "repetitions", repetitions);
    }
  }
  free(content);
  content = cJSON_Print(matrix_json);
  write_file(inputjsonfile, content);
  cJSON_Delete(matrix_json);
  free(content);
  return 0;
}

/*
  For GPU formats, we keep two copies of the matrix on CPU and one copy on GPU.
  The extra CPU copy allows us to destroy the GPU matrix and recreate it efficiently
  in each repetition. As a result,  each MatMult call is fresh, and we can capture
  the first-time overhead (e.g. of CuSparse SpMV), and avoids the cache effect
  during consecutive calls.
*/
PetscErrorCode TimedSpMV(Mat A, Vec b, PetscReal *time, const char *petscmatformat, PetscBool use_gpu, PetscInt repetitions)
{
  Mat            A2 = NULL;
  PetscInt       i;
  Vec            u;
  PetscLogDouble vstart = 0, vend = 0;
  PetscBool      isaijcusparse, isaijkokkos;

  PetscFunctionBeginUser;
  PetscCall(PetscStrcmp(petscmatformat, MATAIJCUSPARSE, &isaijcusparse));
  PetscCall(PetscStrcmp(petscmatformat, MATAIJKOKKOS, &isaijkokkos));
  if (isaijcusparse) PetscCall(VecSetType(b, VECCUDA));
  if (isaijkokkos) PetscCall(VecSetType(b, VECKOKKOS));
  PetscCall(VecDuplicate(b, &u));
  if (time) *time = 0.0;
  for (i = 0; i < repetitions; i++) {
    if (use_gpu) {
      PetscCall(MatDestroy(&A2));
      PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &A2));
      PetscCall(MatConvert(A2, petscmatformat, MAT_INPLACE_MATRIX, &A2));
    } else A2 = A;
    /* Timing MatMult */
    if (time) PetscCall(PetscTime(&vstart));

    PetscCall(MatMult(A2, b, u));

    if (time) {
      PetscCall(PetscTime(&vend));
      *time += (PetscReal)(vend - vstart);
    }
  }
  PetscCall(VecDestroy(&u));
  if (repetitions > 0 && use_gpu) PetscCall(MatDestroy(&A2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogSpMVTime(PetscReal *gputime, PetscReal *cputime, PetscReal *gpuflops, const char *petscmatformat)
{
  PetscLogEvent      event;
  PetscEventPerfInfo eventInfo;
  //PetscReal          gpuflopRate;

  // if (matformat) {
  //   PetscCall(PetscLogEventGetId("MatCUDACopyTo", &event));
  // } else {
  //  PetscCall(PetscLogEventGetId("MatCUSPARSCopyTo", &event));
  // }
  // PetscCall(PetscLogEventGetPerfInfo(PETSC_DETERMINE, event, &eventInfo));
  // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%.4e ", eventInfo.time));

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventGetId("MatMult", &event));
  PetscCall(PetscLogEventGetPerfInfo(PETSC_DETERMINE, event, &eventInfo));
  //gpuflopRate = eventInfo.GpuFlops/eventInfo.GpuTime;
  // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%.2f %.4e %.4e\n", gpuflopRate/1.e6, eventInfo.GpuTime, eventInfo.time));
  if (cputime) *cputime = eventInfo.time;
#if defined(PETSC_HAVE_DEVICE)
  if (gputime) *gputime = eventInfo.GpuTime;
  if (gpuflops) *gpuflops = eventInfo.GpuFlops / 1.e6;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MapToPetscMatType(const char *matformat, PetscBool use_gpu, char **petscmatformat)
{
  PetscBool iscsr, issell, iscsrkokkos;

  PetscFunctionBeginUser;
  PetscCall(PetscStrcmp(matformat, "csr", &iscsr));
  if (iscsr) {
    if (use_gpu) PetscCall(PetscStrallocpy(MATAIJCUSPARSE, petscmatformat));
    else PetscCall(PetscStrallocpy(MATAIJ, petscmatformat));
  } else {
    PetscCall(PetscStrcmp(matformat, "sell", &issell));
    if (issell) {
      if (use_gpu) PetscCall(PetscStrallocpy(MATSELL, petscmatformat)); // placeholder for SELLCUDA
      else PetscCall(PetscStrallocpy(MATSELL, petscmatformat));
    } else {
      PetscCall(PetscStrcmp(matformat, "csrkokkos", &iscsrkokkos));
      if (iscsrkokkos) PetscCall(PetscStrallocpy(MATAIJKOKKOS, petscmatformat));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  PetscInt    nmat = 1, nformats = 5, i, j, repetitions = 1;
  Mat         A;
  Vec         b;
  char        jfilename[PETSC_MAX_PATH_LEN];
  char        filename[PETSC_MAX_PATH_LEN], bfilename[PETSC_MAX_PATH_LEN];
  char        groupname[PETSC_MAX_PATH_LEN], matname[PETSC_MAX_PATH_LEN];
  char       *matformats[5];
  char      **filenames = NULL, **groupnames = NULL, **matnames = NULL;
  char        ordering[256] = MATORDERINGRCM;
  PetscBool   bflg, flg1, flg2, flg3, use_gpu = PETSC_FALSE, permute = PETSC_FALSE;
  IS          rowperm = NULL, colperm = NULL;
  PetscViewer fd;
  PetscReal   starting_spmv_time = 0, *spmv_times;

  PetscCall(PetscOptionsInsertString(NULL, "-log_view_gpu_time -log_view :/dev/null"));
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetStringArray(NULL, NULL, "-formats", matformats, &nformats, &flg1));
  if (!flg1) {
    nformats = 1;
    PetscCall(PetscStrallocpy("csr", &matformats[0]));
  }
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_gpu", &use_gpu, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-repetitions", &repetitions, NULL));
  /* Read matrix and RHS */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-groupname", groupname, PETSC_MAX_PATH_LEN, NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-matname", matname, PETSC_MAX_PATH_LEN, NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-ABIN", filename, PETSC_MAX_PATH_LEN, &flg1));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-AMTX", filename, PETSC_MAX_PATH_LEN, &flg2));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-AJSON", jfilename, PETSC_MAX_PATH_LEN, &flg3));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Extra options", "");
  PetscCall(PetscOptionsFList("-permute", "Permute matrix and vector to solving in new ordering", "", MatOrderingList, ordering, ordering, sizeof(ordering), &permute));
  PetscOptionsEnd();
#if !defined(PETSC_HAVE_DEVICE)
  PetscCheck(!use_gpu, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "To use the option -use_gpu 1, PETSc must be configured with GPU support");
#endif
  PetscCheck(flg1 || flg2 || flg3, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Must indicate an input file with the -ABIN or -AMTX or -AJSON depending on the file format");
  if (flg3) {
    ParseJSON(jfilename, &filenames, &groupnames, &matnames, &nmat);
    PetscCall(PetscCalloc1(nmat, &spmv_times));
  } else if (flg2) {
    PetscCall(MatCreateFromMTX(&A, filename, PETSC_TRUE));
  } else if (flg1) {
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &fd));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetType(A, MATAIJ));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatLoad(A, fd));
    PetscCall(PetscViewerDestroy(&fd));
  }
  if (permute) {
    Mat Aperm;
    PetscCall(MatGetOrdering(A, ordering, &rowperm, &colperm));
    PetscCall(MatPermute(A, rowperm, colperm, &Aperm));
    PetscCall(MatDestroy(&A));
    A = Aperm; /* Replace original operator with permuted version */
  }
  /* Let the vec object trigger the first CUDA call, which takes a relatively long time to init CUDA */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-b", bfilename, PETSC_MAX_PATH_LEN, &bflg));
  if (bflg) {
    PetscViewer fb;
    PetscCall(VecCreate(PETSC_COMM_WORLD, &b));
    PetscCall(VecSetFromOptions(b));
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, bfilename, FILE_MODE_READ, &fb));
    PetscCall(VecLoad(b, fb));
    PetscCall(PetscViewerDestroy(&fb));
  }

  for (j = 0; j < nformats; j++) {
    char *petscmatformat = NULL;
    PetscCall(MapToPetscMatType(matformats[j], use_gpu, &petscmatformat));
    PetscCheck(petscmatformat, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Invalid mat format %s, supported options include csr and sell.", matformats[j]);
    if (flg3) { // mat names specified in a JSON file
      for (i = 0; i < nmat; i++) {
        PetscCall(MatCreateFromMTX(&A, filenames[i], PETSC_TRUE));
        if (!bflg) {
          PetscCall(MatCreateVecs(A, &b, NULL));
          PetscCall(VecSet(b, 1.0));
        }
        PetscCall(TimedSpMV(A, b, NULL, petscmatformat, use_gpu, repetitions));
        if (use_gpu) PetscCall(PetscLogSpMVTime(&spmv_times[i], NULL, NULL, petscmatformat));
        else PetscCall(PetscLogSpMVTime(NULL, &spmv_times[i], NULL, petscmatformat));
        PetscCall(MatDestroy(&A));
        if (!bflg) PetscCall(VecDestroy(&b));
      }
      UpdateJSON(jfilename, spmv_times, starting_spmv_time, matformats[j], use_gpu, repetitions);
      starting_spmv_time = spmv_times[nmat - 1];
    } else {
      PetscReal spmv_time;
      if (!bflg) {
        PetscCall(MatCreateVecs(A, &b, NULL));
        PetscCall(VecSet(b, 1.0));
      }
      PetscCall(TimedSpMV(A, b, &spmv_time, petscmatformat, use_gpu, repetitions));
      if (!bflg) PetscCall(VecDestroy(&b));
    }
    PetscCall(PetscFree(petscmatformat));
  }
  if (flg3) {
    for (i = 0; i < nmat; i++) {
      free(filenames[i]);
      free(groupnames[i]);
      free(matnames[i]);
    }
    free(filenames);
    free(groupnames);
    free(matnames);
    PetscCall(PetscFree(spmv_times));
  }
  for (j = 0; j < nformats; j++) PetscCall(PetscFree(matformats[j]));
  if (flg1 || flg2) PetscCall(MatDestroy(&A));
  if (bflg) PetscCall(VecDestroy(&b));
  PetscCall(ISDestroy(&rowperm));
  PetscCall(ISDestroy(&colperm));
  PetscCall(PetscFinalize());
  return 0;
}
/*TEST

   build:
      requires:  !complex double !windows_compilers !defined(PETSC_USE_64BIT_INDICES)
      depends: mmloader.c mmio.c cJSON.c

   test:
      suffix: 1
      args: -AMTX ${wPETSC_DIR}/share/petsc/datafiles/matrices/amesos2_test_mat0.mtx

   test:
      suffix: 2
      args:-AMTX ${wPETSC_DIR}/share/petsc/datafiles/matrices/amesos2_test_mat0.mtx -use_gpu
      output_file: output/bench_spmv_1.out
      requires: cuda

TEST*/
