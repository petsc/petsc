/*************************************************************************************
 *    M A R I T I M E  R E S E A R C H  I N S T I T U T E  N E T H E R L A N D S     *
 *************************************************************************************
 *    authors: Koos Huijssen, Christiaan M. Klaij                                    *
 *************************************************************************************
 *    content: Viewer for writing XML output                                         *
 *************************************************************************************/
#include <petscviewer.h>
#include <petsc/private/logimpl.h>
#include "xmlviewer.h"
#include "lognested.h"

static PetscErrorCode PetscViewerXMLStartSection(PetscViewer viewer, const char *name, const char *desc)
{
  PetscInt XMLSectionDepthPetsc;
  int      XMLSectionDepth;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIGetTab(viewer, &XMLSectionDepthPetsc));
  PetscCall(PetscCIntCast(XMLSectionDepthPetsc, &XMLSectionDepth));
  if (!desc) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "%*s<%s>\n", 2 * XMLSectionDepth, "", name));
  } else {
    PetscCall(PetscViewerASCIIPrintf(viewer, "%*s<%s desc=\"%s\">\n", 2 * XMLSectionDepth, "", name, desc));
  }
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Initialize a viewer to XML, and initialize the XMLDepth static parameter */
static PetscErrorCode PetscViewerInitASCII_XML(PetscViewer viewer)
{
  MPI_Comm comm;
  char     PerfScript[PETSC_MAX_PATH_LEN + 40];

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCall(PetscViewerASCIIPrintf(viewer, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"));
  PetscCall(PetscStrreplace(comm, "<?xml-stylesheet type=\"text/xsl\" href=\"performance_xml2html.xsl\"?>", PerfScript, sizeof(PerfScript)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%s\n", PerfScript));
  PetscCall(PetscViewerXMLStartSection(viewer, "root", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerXMLEndSection(PetscViewer viewer, const char *name)
{
  PetscInt XMLSectionDepthPetsc;
  int      XMLSectionDepth;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIGetTab(viewer, &XMLSectionDepthPetsc));
  PetscCall(PetscCIntCast(XMLSectionDepthPetsc, &XMLSectionDepth));
  if (XMLSectionDepth > 0) PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscViewerASCIIGetTab(viewer, &XMLSectionDepthPetsc));
  PetscCall(PetscCIntCast(XMLSectionDepthPetsc, &XMLSectionDepth));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%*s</%s>\n", 2 * XMLSectionDepth, "", name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Initialize a viewer to XML, and initialize the XMLDepth static parameter */
static PetscErrorCode PetscViewerFinalASCII_XML(PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerXMLEndSection(viewer, "root"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerXMLPutString(PetscViewer viewer, const char *name, const char *desc, const char *value)
{
  PetscInt XMLSectionDepthPetsc;
  int      XMLSectionDepth;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIGetTab(viewer, &XMLSectionDepthPetsc));
  PetscCall(PetscCIntCast(XMLSectionDepthPetsc, &XMLSectionDepth));
  if (!desc) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "%*s<%s>%s</%s>\n", 2 * XMLSectionDepth, "", name, value, name));
  } else {
    PetscCall(PetscViewerASCIIPrintf(viewer, "%*s<%s desc=\"%s\">%s</%s>\n", 2 * XMLSectionDepth, "", name, desc, value, name));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerXMLPutInt(PetscViewer viewer, const char *name, const char *desc, int value)
{
  PetscInt XMLSectionDepthPetsc;
  int      XMLSectionDepth;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIGetTab(viewer, &XMLSectionDepthPetsc));
  PetscCall(PetscCIntCast(XMLSectionDepthPetsc, &XMLSectionDepth));
  if (!desc) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "%*s<%s>%d</%s>\n", 2 * XMLSectionDepth, "", name, value, name));
  } else {
    PetscCall(PetscViewerASCIIPrintf(viewer, "%*s<%s desc=\"%s\">%d</%s>\n", 2 * XMLSectionDepth, "", name, desc, value, name));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerXMLPutDouble(PetscViewer viewer, const char *name, PetscLogDouble value, const char *format)
{
  PetscInt XMLSectionDepthPetsc;
  int      XMLSectionDepth;
  char     buffer[1024];

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIGetTab(viewer, &XMLSectionDepthPetsc));
  PetscCall(PetscCIntCast(XMLSectionDepthPetsc, &XMLSectionDepth));
  PetscCall(PetscSNPrintf(buffer, sizeof(buffer), "%*s<%s>%s</%s>\n", 2 * XMLSectionDepth, "", name, format, name));
  PetscCall(PetscViewerASCIIPrintf(viewer, buffer, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPrintExeSpecs(PetscViewer viewer)
{
  char        arch[128], hostname[128], username[128], pname[PETSC_MAX_PATH_LEN], date[128];
  char        version[256], buildoptions[128] = "";
  PetscMPIInt size;
  size_t      len;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)viewer), &size));
  PetscCall(PetscGetArchType(arch, sizeof(arch)));
  PetscCall(PetscGetHostName(hostname, sizeof(hostname)));
  PetscCall(PetscGetUserName(username, sizeof(username)));
  PetscCall(PetscGetProgramName(pname, sizeof(pname)));
  PetscCall(PetscGetDate(date, sizeof(date)));
  PetscCall(PetscGetVersion(version, sizeof(version)));

  PetscCall(PetscViewerXMLStartSection(viewer, "runspecification", "Run Specification"));
  PetscCall(PetscViewerXMLPutString(viewer, "executable", "Executable", pname));
  PetscCall(PetscViewerXMLPutString(viewer, "architecture", "Architecture", arch));
  PetscCall(PetscViewerXMLPutString(viewer, "hostname", "Host", hostname));
  PetscCall(PetscViewerXMLPutInt(viewer, "nprocesses", "Number of processes", size));
  PetscCall(PetscViewerXMLPutString(viewer, "user", "Run by user", username));
  PetscCall(PetscViewerXMLPutString(viewer, "date", "Started at", date));
  PetscCall(PetscViewerXMLPutString(viewer, "petscrelease", "PETSc Release", version));

  if (PetscDefined(USE_DEBUG)) PetscCall(PetscStrlcat(buildoptions, "Debug ", sizeof(buildoptions)));
  if (PetscDefined(USE_COMPLEX)) PetscCall(PetscStrlcat(buildoptions, "Complex ", sizeof(buildoptions)));
  if (PetscDefined(USE_REAL_SINGLE)) {
    PetscCall(PetscStrlcat(buildoptions, "Single ", sizeof(buildoptions)));
  } else if (PetscDefined(USE_REAL___FLOAT128)) {
    PetscCall(PetscStrlcat(buildoptions, "Quadruple ", sizeof(buildoptions)));
  } else if (PetscDefined(USE_REAL___FP16)) {
    PetscCall(PetscStrlcat(buildoptions, "Half ", sizeof(buildoptions)));
  }
  if (PetscDefined(USE_64BIT_INDICES)) PetscCall(PetscStrlcat(buildoptions, "Int64 ", sizeof(buildoptions)));
#if defined(__cplusplus)
  PetscCall(PetscStrlcat(buildoptions, "C++ ", sizeof(buildoptions)));
#endif
  PetscCall(PetscStrlen(buildoptions, &len));
  if (len) PetscCall(PetscViewerXMLPutString(viewer, "petscbuildoptions", "PETSc build options", buildoptions));
  PetscCall(PetscViewerXMLEndSection(viewer, "runspecification"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPrintXMLGlobalPerformanceElement(PetscViewer viewer, const char *name, const char *desc, PetscLogDouble local_val, const PetscBool print_average, const PetscBool print_total)
{
  PetscLogDouble min, tot, ratio, avg;
  MPI_Comm       comm;
  PetscMPIInt    rank, size;
  PetscLogDouble valrank[2], max[2];

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)viewer), &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  valrank[0] = local_val;
  valrank[1] = (PetscLogDouble)rank;
  PetscCallMPI(MPIU_Allreduce(&local_val, &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
  PetscCallMPI(MPIU_Allreduce(valrank, &max, 1, MPIU_2PETSCLOGDOUBLE, MPI_MAXLOC, comm));
  PetscCallMPI(MPIU_Allreduce(&local_val, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
  avg = tot / ((PetscLogDouble)size);
  if (min != 0.0) ratio = max[0] / min;
  else ratio = 0.0;

  PetscCall(PetscViewerXMLStartSection(viewer, name, desc));
  PetscCall(PetscViewerXMLPutDouble(viewer, "max", max[0], "%e"));
  PetscCall(PetscViewerXMLPutInt(viewer, "maxrank", "rank at which max was found", (PetscMPIInt)max[1]));
  PetscCall(PetscViewerXMLPutDouble(viewer, "ratio", ratio, "%f"));
  if (print_average) PetscCall(PetscViewerXMLPutDouble(viewer, "average", avg, "%e"));
  if (print_total) PetscCall(PetscViewerXMLPutDouble(viewer, "total", tot, "%e"));
  PetscCall(PetscViewerXMLEndSection(viewer, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Print the global performance: max, max/min, average and total of
 *      time, objects, flops, flops/sec, memory, MPI messages, MPI message lengths, MPI reductions.
 */
static PetscErrorCode PetscPrintGlobalPerformance(PetscViewer viewer, PetscLogDouble locTotalTime, PetscLogHandler default_handler)
{
  PetscLogDouble  flops, mem, red, mess;
  PetscInt        num_objects;
  const PetscBool print_total_yes = PETSC_TRUE, print_total_no = PETSC_FALSE, print_average_no = PETSC_FALSE, print_average_yes = PETSC_TRUE;

  PetscFunctionBegin;
  /* Must preserve reduction count before we go on */
  red = petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;

  /* Calculate summary information */
  PetscCall(PetscViewerXMLStartSection(viewer, "globalperformance", "Global performance"));

  /*   Time */
  PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "time", "Time (sec)", locTotalTime, print_average_yes, print_total_no));

  /*   Objects */
  PetscCall(PetscLogHandlerGetNumObjects(default_handler, &num_objects));
  PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "objects", "Objects", (PetscLogDouble)num_objects, print_average_yes, print_total_no));

  /*   Flop */
  PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "mflop", "MFlop", petsc_TotalFlops / 1.0E6, print_average_yes, print_total_yes));

  /*   Flop/sec -- Must talk to Barry here */
  if (locTotalTime != 0.0) flops = petsc_TotalFlops / locTotalTime;
  else flops = 0.0;
  PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "mflops", "MFlop/sec", flops / 1.0E6, print_average_yes, print_total_yes));

  /*   Memory */
  PetscCall(PetscMallocGetMaximumUsage(&mem));
  if (mem > 0.0) PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "memory", "Memory (MiB)", mem / 1024.0 / 1024.0, print_average_yes, print_total_yes));
  /*   Messages */
  mess = 0.5 * (petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct);
  PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "messagetransfers", "MPI Message Transfers", mess, print_average_yes, print_total_yes));

  /*   Message Volume */
  mess = 0.5 * (petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len);
  PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "messagevolume", "MPI Message Volume (MiB)", mess / 1024.0 / 1024.0, print_average_yes, print_total_yes));

  /*   Reductions */
  PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "reductions", "MPI Reductions", red, print_average_no, print_total_no));
  PetscCall(PetscViewerXMLEndSection(viewer, "globalperformance"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Print the global performance: max, max/min, average and total of
 *      time, objects, flops, flops/sec, memory, MPI messages, MPI message lengths, MPI reductions.
 */
static PetscErrorCode PetscPrintXMLNestedLinePerfResults(PetscViewer viewer, const char *name, PetscLogDouble value, PetscLogDouble minthreshold, PetscLogDouble maxthreshold, PetscLogDouble minmaxtreshold)
{
  MPI_Comm       comm; /* MPI communicator in reduction */
  PetscMPIInt    rank; /* rank of this process */
  PetscLogDouble val_in[2], max[2], min[2];
  PetscLogDouble minvalue, maxvalue, tot;
  PetscMPIInt    size;
  PetscMPIInt    minLoc, maxLoc;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  val_in[0] = value;
  val_in[1] = (PetscLogDouble)rank;
  PetscCallMPI(MPIU_Allreduce(val_in, max, 1, MPIU_2PETSCLOGDOUBLE, MPI_MAXLOC, comm));
  PetscCallMPI(MPIU_Allreduce(val_in, min, 1, MPIU_2PETSCLOGDOUBLE, MPI_MINLOC, comm));
  maxvalue = max[0];
  maxLoc   = (PetscMPIInt)max[1];
  minvalue = min[0];
  minLoc   = (PetscMPIInt)min[1];
  PetscCallMPI(MPIU_Allreduce(&value, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));

  if (maxvalue < maxthreshold && minvalue >= minthreshold) {
    /* One call per parent or NO value: don't print */
  } else {
    PetscCall(PetscViewerXMLStartSection(viewer, name, NULL));
    if (maxvalue > minvalue * minmaxtreshold) {
      PetscCall(PetscViewerXMLPutDouble(viewer, "avgvalue", tot / size, "%g"));
      PetscCall(PetscViewerXMLPutDouble(viewer, "minvalue", minvalue, "%g"));
      PetscCall(PetscViewerXMLPutDouble(viewer, "maxvalue", maxvalue, "%g"));
      PetscCall(PetscViewerXMLPutInt(viewer, "minloc", NULL, minLoc));
      PetscCall(PetscViewerXMLPutInt(viewer, "maxloc", NULL, maxLoc));
    } else {
      PetscCall(PetscViewerXMLPutDouble(viewer, "value", tot / size, "%g"));
    }
    PetscCall(PetscViewerXMLEndSection(viewer, name));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedTreePrintLine(PetscViewer viewer, const PetscEventPerfInfo *perfInfo, int childCount, int parentCount, const char *name, PetscLogDouble totalTime)
{
  PetscLogDouble time = perfInfo->time;
  PetscLogDouble timeMx;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCallMPI(MPIU_Allreduce(&time, &timeMx, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscCall(PetscViewerXMLPutString(viewer, "name", NULL, name));
  PetscCall(PetscPrintXMLNestedLinePerfResults(viewer, "time", time / totalTime * 100.0, 0, 0, 1.02));
  PetscCall(PetscPrintXMLNestedLinePerfResults(viewer, "ncalls", parentCount > 0 ? ((PetscLogDouble)childCount) / ((PetscLogDouble)parentCount) : 1.0, 0.99, 1.01, 1.02));
  PetscCall(PetscPrintXMLNestedLinePerfResults(viewer, "mflops", time >= timeMx * 0.001 ? 1e-6 * perfInfo->flops / time : 0, 0, 0.01, 1.05));
  PetscCall(PetscPrintXMLNestedLinePerfResults(viewer, "mbps", time >= timeMx * 0.001 ? perfInfo->messageLength / (1024 * 1024 * time) : 0, 0, 0.01, 1.05));
  PetscCall(PetscPrintXMLNestedLinePerfResults(viewer, "nreductsps", time >= timeMx * 0.001 ? perfInfo->numReductions / time : 0, 0, 0.01, 1.05));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscNestedNameGetBase(const char name[], const char *base[])
{
  size_t n;

  PetscFunctionBegin;
  PetscCall(PetscStrlen(name, &n));
  while (n > 0 && name[n - 1] != ';') n--;
  *base = &name[n];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedTreePrint(PetscViewer viewer, double total_time, double threshold_time, const PetscNestedEventNode *parent_node, PetscEventPerfInfo *parent_info, const PetscNestedEventNode tree[], PetscEventPerfInfo perf[], PetscLogNestedType type, PetscBool print_events)
{
  PetscInt           num_children = 0, num_printed;
  PetscInt           num_nodes    = parent_node->num_descendants;
  PetscInt          *perm;
  PetscReal         *times; // Not PetscLogDouble, to reuse PetscSortRealWithArrayInt() below
  PetscEventPerfInfo other;

  PetscFunctionBegin;
  for (PetscInt node = 0; node < num_nodes; node += 1 + tree[node].num_descendants) {
    PetscAssert(tree[node].parent == parent_node->id, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Failed tree invariant");
    num_children++;
  }
  num_printed = num_children;
  PetscCall(PetscMemzero(&other, sizeof(other)));
  PetscCall(PetscMalloc2(num_children + 2, &times, num_children + 2, &perm));
  for (PetscInt i = 0, node = 0; node < num_nodes; i++, node += 1 + tree[node].num_descendants) {
    PetscLogDouble child_time = perf[node].time;

    perm[i] = node;
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &child_time, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, PetscObjectComm((PetscObject)viewer)));
    times[i] = -child_time;

    parent_info->time -= perf[node].time;
    parent_info->flops -= perf[node].flops;
    parent_info->numMessages -= perf[node].numMessages;
    parent_info->messageLength -= perf[node].messageLength;
    parent_info->numReductions -= perf[node].numReductions;
    if (child_time < threshold_time) {
      PetscEventPerfInfo *add_to = (type == PETSC_LOG_NESTED_XML) ? &other : parent_info;

      add_to->time += perf[node].time;
      add_to->flops += perf[node].flops;
      add_to->numMessages += perf[node].numMessages;
      add_to->messageLength += perf[node].messageLength;
      add_to->numReductions += perf[node].numReductions;
      add_to->count += perf[node].count;
      num_printed--;
    }
  }
  perm[num_children]      = -1;
  times[num_children]     = -parent_info->time;
  perm[num_children + 1]  = -2;
  times[num_children + 1] = -other.time;
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &times[num_children], 2, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject)viewer)));
  // Sync the time with allreduce results, otherwise it could result in code path divergence through num_printed and early return.
  other.time = -times[num_children + 1];
  if (type == PETSC_LOG_NESTED_FLAMEGRAPH) {
    /* The output is given as an integer in microseconds because otherwise the file cannot be read
     * by apps such as speedscope (https://speedscope.app/). */
    PetscCall(PetscViewerASCIIPrintf(viewer, "%s %" PetscInt64_FMT "\n", parent_node->name, (PetscInt64)(-times[num_children] * 1e6)));
  }
  if (other.time > 0.0) num_printed++;
  // number of items other than "self" that will be printed
  if (num_printed == 0) {
    PetscCall(PetscFree2(times, perm));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  // sort descending by time
  PetscCall(PetscSortRealWithArrayInt(num_children + 2, times, perm));
  if (type == PETSC_LOG_NESTED_XML && print_events) PetscCall(PetscViewerXMLStartSection(viewer, "events", NULL));
  for (PetscInt i = 0; i < num_children + 2; i++) {
    PetscInt       node       = perm[i];
    PetscLogDouble child_time = -times[i];

    if (child_time >= threshold_time || (node < 0 && child_time > 0.0)) {
      if (type == PETSC_LOG_NESTED_XML) {
        PetscCall(PetscViewerXMLStartSection(viewer, "event", NULL));
        if (node == -1) {
          PetscCall(PetscLogNestedTreePrintLine(viewer, parent_info, 0, 0, "self", total_time));
        } else if (node == -2) {
          PetscCall(PetscLogNestedTreePrintLine(viewer, &other, other.count, parent_info->count, "other", total_time));
        } else {
          const char *base_name = NULL;
          PetscCall(PetscNestedNameGetBase(tree[node].name, &base_name));
          PetscCall(PetscLogNestedTreePrintLine(viewer, &perf[node], perf[node].count, parent_info->count, base_name, total_time));
          PetscCall(PetscLogNestedTreePrint(viewer, total_time, threshold_time, &tree[node], &perf[node], &tree[node + 1], &perf[node + 1], type, PETSC_TRUE));
        }
        PetscCall(PetscViewerXMLEndSection(viewer, "event"));
      } else if (node >= 0) {
        PetscCall(PetscLogNestedTreePrint(viewer, total_time, threshold_time, &tree[node], &perf[node], &tree[node + 1], &perf[node + 1], type, PETSC_FALSE));
      }
    }
  }
  if (type == PETSC_LOG_NESTED_XML && print_events) PetscCall(PetscViewerXMLEndSection(viewer, "events"));
  PetscCall(PetscFree2(times, perm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedTreePrintTop(PetscViewer viewer, PetscNestedEventTree *tree, PetscLogDouble threshold, PetscLogNestedType type)
{
  PetscNestedEventNode *main_stage;
  PetscNestedEventNode *tree_rem;
  PetscEventPerfInfo   *main_stage_perf;
  PetscEventPerfInfo   *perf_rem;
  PetscLogDouble        time;
  PetscLogDouble        threshold_time;

  PetscFunctionBegin;
  main_stage      = &tree->nodes[0];
  tree_rem        = &tree->nodes[1];
  main_stage_perf = &tree->perf[0];
  perf_rem        = &tree->perf[1];
  time            = main_stage_perf->time;
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &time, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, tree->comm));
  /* Print (or ignore) the children in ascending order of total time */
  if (type == PETSC_LOG_NESTED_XML) {
    PetscCall(PetscViewerXMLStartSection(viewer, "timertree", "Timings tree"));
    PetscCall(PetscViewerXMLPutDouble(viewer, "totaltime", time, "%f"));
    PetscCall(PetscViewerXMLPutDouble(viewer, "timethreshold", threshold, "%f"));
  }
  threshold_time = time * (threshold / 100.0 + 1.e-12);
  PetscCall(PetscLogNestedTreePrint(viewer, time, threshold_time, main_stage, main_stage_perf, tree_rem, perf_rem, type, PETSC_FALSE));
  if (type == PETSC_LOG_NESTED_XML) PetscCall(PetscViewerXMLEndSection(viewer, "timertree"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogHandlerView_Nested_XML(PetscLogHandler_Nested nested, PetscNestedEventTree *tree, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerInitASCII_XML(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "<!-- PETSc Performance Summary: -->\n"));
  PetscCall(PetscViewerXMLStartSection(viewer, "petscroot", NULL));

  // Print global information about this run
  PetscCall(PetscPrintExeSpecs(viewer));

  if (PetscDefined(USE_LOG)) {
    PetscEventPerfInfo *main_stage_info;
    PetscLogDouble      locTotalTime;

    PetscCall(PetscLogHandlerGetEventPerfInfo(nested->handler, 0, 0, &main_stage_info));
    locTotalTime = main_stage_info->time;
    PetscCall(PetscPrintGlobalPerformance(viewer, locTotalTime, nested->handler));
  }
  PetscCall(PetscLogNestedTreePrintTop(viewer, tree, nested->threshold, PETSC_LOG_NESTED_XML));
  PetscCall(PetscViewerXMLEndSection(viewer, "petscroot"));
  PetscCall(PetscViewerFinalASCII_XML(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogHandlerView_Nested_Flamegraph(PetscLogHandler_Nested nested, PetscNestedEventTree *tree, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscLogNestedTreePrintTop(viewer, tree, nested->threshold, PETSC_LOG_NESTED_FLAMEGRAPH));
  PetscFunctionReturn(PETSC_SUCCESS);
}
