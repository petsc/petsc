#include <petscviewer.h>
#include "lognested.h"
#include "xmlviewer.h"

PETSC_INTERN PetscErrorCode PetscLogHandlerNestedSetThreshold(PetscLogHandler h, PetscLogDouble newThresh, PetscLogDouble *oldThresh)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->data;

  PetscFunctionBegin;
  if (oldThresh) *oldThresh = nested->threshold;
  if (newThresh == (PetscLogDouble)PETSC_DECIDE) newThresh = 0.01;
  if (newThresh == (PetscLogDouble)PETSC_DEFAULT) newThresh = 0.01;
  nested->threshold = PetscMax(newThresh, 0.0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogEventGetNestedEvent(PetscLogHandler h, PetscLogEvent e, PetscLogEvent *nested_event)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->data;
  NestedIdPair           key;
  PetscHashIter          iter;
  PetscBool              missing;
  PetscLogState          state;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscIntStackTop(nested->nested_stack, &key.root));
  key.leaf = NestedIdFromEvent(e);
  PetscCall(PetscNestedHashPut(nested->pair_map, key, &iter, &missing));
  if (missing) {
    // register a new nested event
    char              name[BUFSIZ];
    PetscLogEventInfo event_info;
    PetscLogEventInfo nested_event_info;

    PetscCall(PetscLogStateEventGetInfo(state, e, &event_info));
    PetscCall(PetscLogStateEventGetInfo(nested->state, key.root, &nested_event_info));
    PetscCall(PetscSNPrintf(name, sizeof(name) - 1, "%s;%s", nested_event_info.name, event_info.name));
    PetscCall(PetscLogStateEventRegister(nested->state, name, event_info.classid, nested_event));
    PetscCall(PetscLogStateEventSetCollective(nested->state, *nested_event, event_info.collective));
    PetscCall(PetscNestedHashIterSet(nested->pair_map, iter, *nested_event));
  } else {
    PetscCall(PetscNestedHashIterGet(nested->pair_map, iter, nested_event));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogStageGetNestedEvent(PetscLogHandler h, PetscLogStage stage, PetscLogEvent *nested_event)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->data;
  NestedIdPair           key;
  PetscHashIter          iter;
  PetscBool              missing;
  PetscLogState          state;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscIntStackTop(nested->nested_stack, &key.root));
  key.leaf = NestedIdFromStage(stage);
  PetscCall(PetscNestedHashPut(nested->pair_map, key, &iter, &missing));
  if (missing) {
    PetscLogStageInfo stage_info;
    char              name[BUFSIZ];
    PetscBool         collective = PETSC_TRUE;

    PetscCall(PetscLogStateStageGetInfo(state, stage, &stage_info));
    if (key.root >= 0) {
      PetscLogEventInfo nested_event_info;

      PetscCall(PetscLogStateEventGetInfo(nested->state, key.root, &nested_event_info));
      PetscCall(PetscSNPrintf(name, sizeof(name) - 1, "%s;%s", nested_event_info.name, stage_info.name));
      collective = nested_event_info.collective;
    } else {
      PetscCall(PetscSNPrintf(name, sizeof(name) - 1, "%s", stage_info.name));
    }
    PetscCall(PetscLogStateEventRegister(nested->state, name, nested->nested_stage_id, nested_event));
    PetscCall(PetscLogStateEventSetCollective(nested->state, *nested_event, collective));
    PetscCall(PetscNestedHashIterSet(nested->pair_map, iter, *nested_event));
  } else {
    PetscCall(PetscNestedHashIterGet(nested->pair_map, iter, nested_event));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedFindNestedId(PetscLogHandler h, NestedId orig_id, PetscInt *pop_count)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->data;
  PetscInt               count, i;

  PetscFunctionBegin;
  // stop before zero cause there is a null event at the bottom of the stack
  for (i = nested->orig_stack->top, count = 0; i > 0; i--) {
    count++;
    if (nested->orig_stack->stack[i] == orig_id) break;
  }
  *pop_count = count;
  if (count == 1) PetscFunctionReturn(PETSC_SUCCESS); // Normal function, just the top of the stack is being popped.
  if (orig_id > 0) {
    PetscLogEvent     event_id = NestedIdToEvent(orig_id);
    PetscLogState     state;
    PetscLogEventInfo event_info;

    PetscCall(PetscLogHandlerGetState(h, &state));
    PetscCall(PetscLogStateEventGetInfo(state, event_id, &event_info));
    PetscCheck(i > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Tried to end event %s, but it is not in the event stack", event_info.name);
  } else {
    PetscLogStage     stage_id = NestedIdToStage(orig_id);
    PetscLogState     state;
    PetscLogStageInfo stage_info;

    PetscCall(PetscLogHandlerGetState(h, &state));
    PetscCall(PetscLogStateStageGetInfo(state, stage_id, &stage_info));
    PetscCheck(i > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Tried to pop stage %s, but it is not in the stage stack", stage_info.name);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedCheckNested(PetscLogHandler h, NestedId leaf, PetscLogEvent nested_event)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->data;
  NestedIdPair           key;
  NestedId               val;

  PetscFunctionBegin;
  PetscCall(PetscIntStackTop(nested->nested_stack, &key.root));
  key.leaf = leaf;
  PetscCall(PetscNestedHashGet(nested->pair_map, key, &val));
  PetscCheck(val == nested_event, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Logging events and stages are not nested, nested logging cannot be used");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventBegin_Nested(PetscLogHandler h, PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->data;
  PetscLogEvent          nested_event;

  PetscFunctionBegin;
  PetscCall(PetscLogEventGetNestedEvent(h, e, &nested_event));
  PetscCall(PetscLogHandlerEventBegin(nested->handler, nested_event, o1, o2, o3, o4));
  PetscCall(PetscIntStackPush(nested->nested_stack, nested_event));
  PetscCall(PetscIntStackPush(nested->orig_stack, NestedIdFromEvent(e)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerNestedEventEnd(PetscLogHandler h, NestedId id, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->data;
  PetscInt               pop_count;

  PetscFunctionBegin;
  PetscCall(PetscLogNestedFindNestedId(h, id, &pop_count));
  for (PetscInt c = 0; c < pop_count; c++) {
    PetscLogEvent nested_event;
    PetscLogEvent nested_id;

    PetscCall(PetscIntStackPop(nested->nested_stack, &nested_event));
    PetscCall(PetscIntStackPop(nested->orig_stack, &nested_id));
    if (PetscDefined(USE_DEBUG)) PetscCall(PetscLogNestedCheckNested(h, nested_id, nested_event));
    if ((pop_count > 1) && (c + 1 < pop_count)) {
      if (nested_id > 0) {
        PetscLogEvent     event_id = NestedIdToEvent(nested_id);
        PetscLogState     state;
        PetscLogEventInfo event_info;

        PetscCall(PetscLogHandlerGetState(h, &state));
        PetscCall(PetscLogStateEventGetInfo(state, event_id, &event_info));
        PetscCall(PetscInfo(h, "Log event %s wasn't ended, ending it to maintain stack property for nested log handler\n", event_info.name));
      }
    }
    PetscCall(PetscLogHandlerEventEnd(nested->handler, nested_event, o1, o2, o3, o4));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventEnd_Nested(PetscLogHandler h, PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscFunctionBegin;
  PetscCall(PetscLogHandlerNestedEventEnd(h, NestedIdFromEvent(e), o1, o2, o3, o4));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventSync_Nested(PetscLogHandler h, PetscLogEvent e, MPI_Comm comm)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->data;
  PetscLogEvent          nested_event;

  PetscFunctionBegin;
  PetscCall(PetscLogEventGetNestedEvent(h, e, &nested_event));
  PetscCall(PetscLogHandlerEventSync(nested->handler, nested_event, comm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerStagePush_Nested(PetscLogHandler h, PetscLogStage stage)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->data;
  PetscLogEvent          nested_event;

  PetscFunctionBegin;
  if (nested->nested_stage_id == -1) PetscCall(PetscClassIdRegister("LogNestedStage", &nested->nested_stage_id));
  PetscCall(PetscLogStageGetNestedEvent(h, stage, &nested_event));
  PetscCall(PetscLogHandlerEventBegin(nested->handler, nested_event, NULL, NULL, NULL, NULL));
  PetscCall(PetscIntStackPush(nested->nested_stack, nested_event));
  PetscCall(PetscIntStackPush(nested->orig_stack, NestedIdFromStage(stage)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerStagePop_Nested(PetscLogHandler h, PetscLogStage stage)
{
  PetscFunctionBegin;
  PetscCall(PetscLogHandlerNestedEventEnd(h, NestedIdFromStage(stage), NULL, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerContextCreate_Nested(MPI_Comm comm, PetscLogHandler_Nested *nested_p)
{
  PetscLogStage          root_stage;
  PetscLogHandler_Nested nested;

  PetscFunctionBegin;
  PetscCall(PetscNew(nested_p));
  nested = *nested_p;
  PetscCall(PetscLogStateCreate(&nested->state));
  PetscCall(PetscIntStackCreate(&nested->nested_stack));
  PetscCall(PetscIntStackCreate(&nested->orig_stack));
  nested->nested_stage_id = -1;
  nested->threshold       = 0.01;
  PetscCall(PetscNestedHashCreate(&nested->pair_map));
  PetscCall(PetscLogHandlerCreate(comm, &nested->handler));
  PetscCall(PetscLogHandlerSetType(nested->handler, PETSCLOGHANDLERDEFAULT));
  PetscCall(PetscLogHandlerSetState(nested->handler, nested->state));
  PetscCall(PetscLogStateStageRegister(nested->state, "", &root_stage));
  PetscAssert(root_stage == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "root stage not zero");
  PetscCall(PetscLogHandlerStagePush(nested->handler, root_stage));
  PetscCall(PetscLogStateStagePush(nested->state, root_stage));
  PetscCall(PetscIntStackPush(nested->nested_stack, -1));
  PetscCall(PetscIntStackPush(nested->orig_stack, -1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerObjectCreate_Nested(PetscLogHandler h, PetscObject obj)
{
  PetscClassId           classid;
  PetscInt               num_registered, num_nested_registered;
  PetscLogState          state;
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->data;

  PetscFunctionBegin;
  // register missing objects
  PetscCall(PetscObjectGetClassId(obj, &classid));
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscLogStateGetNumClasses(nested->state, &num_nested_registered));
  PetscCall(PetscLogStateGetNumClasses(state, &num_registered));
  for (PetscLogClass c = num_nested_registered; c < num_registered; c++) {
    PetscLogClassInfo class_info;
    PetscLogClass     nested_c;

    PetscCall(PetscLogStateClassGetInfo(state, c, &class_info));
    PetscCall(PetscLogStateClassRegister(nested->state, class_info.name, class_info.classid, &nested_c));
  }
  PetscCall(PetscLogHandlerObjectCreate(nested->handler, obj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerObjectDestroy_Nested(PetscLogHandler h, PetscObject obj)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->data;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerObjectDestroy(nested->handler, obj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerDestroy_Nested(PetscLogHandler h)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->data;

  PetscFunctionBegin;
  PetscCall(PetscLogStateStagePop(nested->state));
  PetscCall(PetscLogHandlerStagePop(nested->handler, 0));
  PetscCall(PetscLogStateDestroy(&nested->state));
  PetscCall(PetscIntStackDestroy(nested->nested_stack));
  PetscCall(PetscIntStackDestroy(nested->orig_stack));
  PetscCall(PetscNestedHashDestroy(&nested->pair_map));
  PetscCall(PetscLogHandlerDestroy(&nested->handler));
  PetscCall(PetscFree(nested));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedEventNodesOrderDepthFirst(PetscInt num_nodes, PetscInt parent, PetscNestedEventNode tree[], PetscInt *num_descendants)
{
  PetscInt node, start_loc;

  PetscFunctionBegin;
  node      = 0;
  start_loc = 0;
  while (node < num_nodes) {
    if (tree[node].parent == parent) {
      PetscInt             num_this_descendants = 0;
      PetscNestedEventNode tmp                  = tree[start_loc];
      tree[start_loc]                           = tree[node];
      tree[node]                                = tmp;
      PetscCall(PetscLogNestedEventNodesOrderDepthFirst(num_nodes - start_loc - 1, tree[start_loc].id, &tree[start_loc + 1], &num_this_descendants));
      tree[start_loc].num_descendants = num_this_descendants;
      *num_descendants += 1 + num_this_descendants;
      start_loc += 1 + num_this_descendants;
      node = start_loc;
    } else {
      node++;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedCreatePerfNodes(MPI_Comm comm, PetscLogHandler_Nested nested, PetscLogGlobalNames global_events, PetscNestedEventNode **tree_p, PetscEventPerfInfo **perf_p)
{
  PetscMPIInt           size;
  PetscInt              num_nodes;
  PetscInt              num_map_entries;
  PetscEventPerfInfo   *perf;
  NestedIdPair         *keys;
  NestedId             *vals;
  PetscInt              offset;
  PetscInt              num_descendants;
  PetscNestedEventNode *tree;

  PetscFunctionBegin;
  PetscCall(PetscLogGlobalNamesGetSize(global_events, NULL, &num_nodes));
  PetscCall(PetscCalloc1(num_nodes, &tree));
  for (PetscInt node = 0; node < num_nodes; node++) {
    tree[node].id = node;
    PetscCall(PetscLogGlobalNamesGlobalGetName(global_events, node, &tree[node].name));
    tree[node].parent = -1;
  }
  PetscCall(PetscNestedHashGetSize(nested->pair_map, &num_map_entries));
  PetscCall(PetscMalloc2(num_map_entries, &keys, num_map_entries, &vals));
  offset = 0;
  PetscCall(PetscNestedHashGetPairs(nested->pair_map, &offset, keys, vals));
  for (PetscInt k = 0; k < num_map_entries; k++) {
    NestedId root_local = keys[k].root;
    NestedId leaf_local = vals[k];
    PetscInt root_global;
    PetscInt leaf_global;

    PetscCall(PetscLogGlobalNamesLocalGetGlobal(global_events, leaf_local, &leaf_global));
    if (root_local >= 0) {
      PetscCall(PetscLogGlobalNamesLocalGetGlobal(global_events, root_local, &root_global));
      tree[leaf_global].parent = root_global;
    }
  }
  PetscCall(PetscFree2(keys, vals));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size > 1) { // get missing parents from other processes
    PetscInt *parents;

    PetscCall(PetscMalloc1(num_nodes, &parents));
    for (PetscInt node = 0; node < num_nodes; node++) parents[node] = tree[node].parent;
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, parents, num_nodes, MPIU_INT, MPI_MAX, comm));
    for (PetscInt node = 0; node < num_nodes; node++) tree[node].parent = parents[node];
    PetscCall(PetscFree(parents));
  }

  num_descendants = 0;
  PetscCall(PetscLogNestedEventNodesOrderDepthFirst(num_nodes, -1, tree, &num_descendants));
  PetscAssert(num_descendants == num_nodes, comm, PETSC_ERR_PLIB, "Failed tree ordering invariant");

  PetscCall(PetscCalloc1(num_nodes, &perf));
  for (PetscInt tree_node = 0; tree_node < num_nodes; tree_node++) {
    PetscInt global_id = tree[tree_node].id;
    PetscInt event_id;

    PetscCall(PetscLogGlobalNamesGlobalGetLocal(global_events, global_id, &event_id));
    if (event_id >= 0) {
      PetscEventPerfInfo *event_info;

      PetscCall(PetscLogHandlerGetEventPerfInfo(nested->handler, 0, event_id, &event_info));
      perf[tree_node] = *event_info;
    } else {
      PetscCall(PetscArrayzero(&perf[tree_node], 1));
    }
  }
  *tree_p = tree;
  *perf_p = perf;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerView_Nested(PetscLogHandler handler, PetscViewer viewer)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)handler->data;
  PetscNestedEventNode  *nodes;
  PetscEventPerfInfo    *perf;
  PetscLogGlobalNames    global_events;
  PetscNestedEventTree   tree;
  PetscViewerFormat      format;
  MPI_Comm               comm = PetscObjectComm((PetscObject)viewer);

  PetscFunctionBegin;
  PetscCall(PetscLogRegistryCreateGlobalEventNames(comm, nested->state->registry, &global_events));
  PetscCall(PetscLogNestedCreatePerfNodes(comm, nested, global_events, &nodes, &perf));
  tree.comm          = comm;
  tree.global_events = global_events;
  tree.perf          = perf;
  tree.nodes         = nodes;
  PetscCall(PetscViewerGetFormat(viewer, &format));
  if (format == PETSC_VIEWER_ASCII_XML) {
    PetscCall(PetscLogHandlerView_Nested_XML(nested, &tree, viewer));
  } else if (format == PETSC_VIEWER_ASCII_FLAMEGRAPH) {
    PetscCall(PetscLogHandlerView_Nested_Flamegraph(nested, &tree, viewer));
  } else SETERRQ(comm, PETSC_ERR_ARG_INCOMP, "No nested viewer for this format");
  PetscCall(PetscLogGlobalNamesDestroy(&global_events));
  PetscCall(PetscFree(tree.nodes));
  PetscCall(PetscFree(tree.perf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCLOGHANDLERNESTED - PETSCLOGHANDLERNESTED = "nested" -  A `PetscLogHandler` that collects data for PETSc's
  XML and flamegraph profiling log viewers.  A log handler of this type is created and started by
  by `PetscLogNestedBegin()`.

  Level: developer

.seealso: [](ch_profiling), `PetscLogHandler`
M*/

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Nested(PetscLogHandler handler)
{
  PetscFunctionBegin;
  PetscCall(PetscLogHandlerContextCreate_Nested(PetscObjectComm((PetscObject)handler), (PetscLogHandler_Nested *)&handler->data));
  handler->ops->destroy       = PetscLogHandlerDestroy_Nested;
  handler->ops->stagepush     = PetscLogHandlerStagePush_Nested;
  handler->ops->stagepop      = PetscLogHandlerStagePop_Nested;
  handler->ops->eventbegin    = PetscLogHandlerEventBegin_Nested;
  handler->ops->eventend      = PetscLogHandlerEventEnd_Nested;
  handler->ops->eventsync     = PetscLogHandlerEventSync_Nested;
  handler->ops->objectcreate  = PetscLogHandlerObjectCreate_Nested;
  handler->ops->objectdestroy = PetscLogHandlerObjectDestroy_Nested;
  handler->ops->view          = PetscLogHandlerView_Nested;
  PetscFunctionReturn(PETSC_SUCCESS);
}
