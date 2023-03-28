
#include <petsc/private/matimpl.h> /*I "petscmatcoarsen.h" I*/

/* Logging support */
PetscClassId MAT_COARSEN_CLASSID;

PetscFunctionList MatCoarsenList              = NULL;
PetscBool         MatCoarsenRegisterAllCalled = PETSC_FALSE;

/*@C
   MatCoarsenRegister - Adds a new sparse matrix coarsening algorithm to the matrix package.

   Logically Collective

   Input Parameters:
+  sname - name of coarsen (for example `MATCOARSENMIS`)
-  function - function pointer that creates the coarsen type

   Level: developer

   Sample usage:
.vb
   MatCoarsenRegister("my_agg",MyAggCreate);
.ve

   Then, your aggregator can be chosen with the procedural interface via
$     MatCoarsenSetType(agg,"my_agg")
   or at runtime via the option
$     -mat_coarsen_type my_agg

.seealso: `MatCoarsen`, `MatCoarsenType`, `MatCoarsenSetType()`, `MatCoarsenCreate()`, `MatCoarsenRegisterDestroy()`, `MatCoarsenRegisterAll()`
@*/
PetscErrorCode MatCoarsenRegister(const char sname[], PetscErrorCode (*function)(MatCoarsen))
{
  PetscFunctionBegin;
  PetscCall(MatInitializePackage());
  PetscCall(PetscFunctionListAdd(&MatCoarsenList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   MatCoarsenGetType - Gets the Coarsen method type and name (as a string)
        from the coarsen context.

   Not Collective

   Input Parameter:
.  coarsen - the coarsen context

   Output Parameter:
.  type - coarsener type

   Level: advanced

.seealso: `MatCoarsen`, `MatCoarsenCreate()`, `MatCoarsenType`, `MatCoarsenSetType()`, `MatCoarsenRegister()`
@*/
PetscErrorCode MatCoarsenGetType(MatCoarsen coarsen, MatCoarsenType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarsen, MAT_COARSEN_CLASSID, 1);
  PetscValidPointer(type, 2);
  *type = ((PetscObject)coarsen)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatCoarsenApply - Gets a coarsen for a matrix.

   Collective

   Input Parameter:
.   coarsen - the coarsen

   Options Database Keys:
   To specify the coarsen through the options database, use one of
   the following
$    -mat_coarsen_type mis|hem|misk
   To see the coarsen result
$    -mat_coarsen_view

   Level: advanced

   Notes:
    Use `MatCoarsenGetData()` to access the results of the coarsening

   The user can define additional coarsens; see `MatCoarsenRegister()`.

.seealso: `MatCoarsen`, `MatCoarseSetFromOptions()`, `MatCoarsenSetType()`, `MatCoarsenRegister()`, `MatCoarsenCreate()`,
          `MatCoarsenDestroy()`, `MatCoarsenSetAdjacency()`
          `MatCoarsenGetData()`
@*/
PetscErrorCode MatCoarsenApply(MatCoarsen coarser)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarser, MAT_COARSEN_CLASSID, 1);
  PetscValidPointer(coarser, 1);
  PetscCheck(coarser->graph->assembled, PetscObjectComm((PetscObject)coarser), PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled matrix");
  PetscCheck(!coarser->graph->factortype, PetscObjectComm((PetscObject)coarser), PETSC_ERR_ARG_WRONGSTATE, "Not for factored matrix");
  PetscCall(PetscLogEventBegin(MAT_Coarsen, coarser, 0, 0, 0));
  PetscUseTypeMethod(coarser, apply);
  PetscCall(PetscLogEventEnd(MAT_Coarsen, coarser, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatCoarsenSetAdjacency - Sets the adjacency graph (matrix) of the thing to be coarsened.

   Collective

   Input Parameters:
+  agg - the coarsen context
-  adj - the adjacency matrix

   Level: advanced

.seealso: `MatCoarsen`, `MatCoarsenSetFromOptions()`, `Mat`, `MatCoarsenCreate()`, `MatCoarsenApply()`
@*/
PetscErrorCode MatCoarsenSetAdjacency(MatCoarsen agg, Mat adj)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(agg, MAT_COARSEN_CLASSID, 1);
  PetscValidHeaderSpecific(adj, MAT_CLASSID, 2);
  agg->graph = adj;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatCoarsenSetStrictAggs - Set whether to keep strict (non overlapping) aggregates in the linked list of aggregates for a coarsen context

   Logically Collective

   Input Parameters:
+  agg - the coarsen context
-  str - `PETSC_TRUE` keep strict aggregates, `PETSC_FALSE` allow overlap

   Level: advanced

.seealso: `MatCoarsen`, `MatCoarsenCreate()`, `MatCoarsenSetFromOptions()`
@*/
PetscErrorCode MatCoarsenSetStrictAggs(MatCoarsen agg, PetscBool str)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(agg, MAT_COARSEN_CLASSID, 1);
  agg->strict_aggs = str;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatCoarsenDestroy - Destroys the coarsen context.

   Collective

   Input Parameter:
.  agg - the coarsen context

   Level: advanced

.seealso: `MatCoarsen`, `MatCoarsenCreate()`
@*/
PetscErrorCode MatCoarsenDestroy(MatCoarsen *agg)
{
  PetscFunctionBegin;
  if (!*agg) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*agg), MAT_COARSEN_CLASSID, 1);
  if (--((PetscObject)(*agg))->refct > 0) {
    *agg = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if ((*agg)->ops->destroy) PetscCall((*(*agg)->ops->destroy)((*agg)));

  if ((*agg)->agg_lists) PetscCall(PetscCDDestroy((*agg)->agg_lists));

  PetscCall(PetscHeaderDestroy(agg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatCoarsenCreate - Creates a coarsen context.

   Collective

   Input Parameter:
.   comm - MPI communicator

   Output Parameter:
.  newcrs - location to put the context

   Level: advanced

.seealso: `MatCoarsen`, `MatCoarsenSetType()`, `MatCoarsenApply()`, `MatCoarsenDestroy()`,
          `MatCoarsenSetAdjacency()`, `MatCoarsenGetData()`

@*/
PetscErrorCode MatCoarsenCreate(MPI_Comm comm, MatCoarsen *newcrs)
{
  MatCoarsen agg;

  PetscFunctionBegin;
  *newcrs = NULL;

  PetscCall(MatInitializePackage());
  PetscCall(PetscHeaderCreate(agg, MAT_COARSEN_CLASSID, "MatCoarsen", "Matrix/graph coarsen", "MatCoarsen", comm, MatCoarsenDestroy, MatCoarsenView));

  *newcrs = agg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   MatCoarsenViewFromOptions - View the coarsener from the options database

   Collective

   Input Parameters:
+  A - the coarsen context
.  obj - Optional object that provides the prefix for the option name
-  name - command line option (usually `-mat_coarsen_view`)

  Options Database Key:
.  -mat_coarsen_view [viewertype]:... - the viewer and its options

  Note:
.vb
    If no value is provided ascii:stdout is used
       ascii[:[filename][:[format][:append]]]    defaults to stdout - format can be one of ascii_info, ascii_info_detail, or ascii_matlab,
                                                  for example ascii::ascii_info prints just the information about the object not all details
                                                  unless :append is given filename opens in write mode, overwriting what was already there
       binary[:[filename][:[format][:append]]]   defaults to the file binaryoutput
       draw[:drawtype[:filename]]                for example, draw:tikz, draw:tikz:figure.tex  or draw:x
       socket[:port]                             defaults to the standard output port
       saws[:communicatorname]                    publishes object to the Scientific Application Webserver (SAWs)
.ve

   Level: intermediate

.seealso: `MatCoarsen`, `MatCoarsenView`, `PetscObjectViewFromOptions()`, `MatCoarsenCreate()`
@*/
PetscErrorCode MatCoarsenViewFromOptions(MatCoarsen A, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_COARSEN_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   MatCoarsenView - Prints the coarsen data structure.

   Collective

   Input Parameters:
+  agg - the coarsen context
-  viewer - optional visualization context

   For viewing the options database see `MatCoarsenViewFromOptions()`

   Level: advanced

.seealso: `MatCoarsen`, `PetscViewer`, `PetscViewerASCIIOpen()`, `MatCoarsenViewFromOptions`
@*/
PetscErrorCode MatCoarsenView(MatCoarsen agg, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(agg, MAT_COARSEN_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)agg), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(agg, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)agg, viewer));
  if (agg->ops->view) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscUseTypeMethod(agg, view, viewer);
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   MatCoarsenSetType - Sets the type of aggregator to use

   Collective

   Input Parameters:
+  coarser - the coarsen context.
-  type - a known coarsening method

   Options Database Key:
.  -mat_coarsen_type  <type> - (for instance, misk), use -help for a list of available methods

   Level: advanced

.seealso: `MatCoarsen`, `MatCoarsenCreate()`, `MatCoarsenApply()`, `MatCoarsenType`, `MatCoarsenGetType()`
@*/
PetscErrorCode MatCoarsenSetType(MatCoarsen coarser, MatCoarsenType type)
{
  PetscBool match;
  PetscErrorCode (*r)(MatCoarsen);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarser, MAT_COARSEN_CLASSID, 1);
  PetscValidCharPointer(type, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)coarser, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscTryTypeMethod(coarser, destroy);
  coarser->ops->destroy = NULL;
  PetscCall(PetscMemzero(coarser->ops, sizeof(struct _MatCoarsenOps)));

  PetscCall(PetscFunctionListFind(MatCoarsenList, type, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)coarser), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown coarsen type %s", type);
  PetscCall((*r)(coarser));

  PetscCall(PetscFree(((PetscObject)coarser)->type_name));
  PetscCall(PetscStrallocpy(type, &((PetscObject)coarser)->type_name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   MatCoarsenSetGreedyOrdering - Sets the ordering of the vertices to use with a greedy coarsening method

   Logically Collective

   Input Parameters:
+  coarser - the coarsen context
-  perm - vertex ordering of (greedy) algorithm

   Level: advanced

   Note:
   The `IS` weights is freed by PETSc, the user should not destroy it or change it after this call

.seealso: `MatCoarsen`, `MatCoarsenType`, `MatCoarsenCreate()`, `MatCoarsenSetType()`
@*/
PetscErrorCode MatCoarsenSetGreedyOrdering(MatCoarsen coarser, const IS perm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarser, MAT_COARSEN_CLASSID, 1);
  coarser->perm = perm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   MatCoarsenGetData - Gets the weights for vertices for a coarsener.

   Logically Collective

   Input Parameter:
.  coarser - the coarsen context

   Output Parameter:
.  llist - linked list of aggregates

   Level: advanced

.seealso: `MatCoarsen`, `MatCoarsenApply()`, `MatCoarsenCreate()`, `MatCoarsenSetType()`
@*/
PetscErrorCode MatCoarsenGetData(MatCoarsen coarser, PetscCoarsenData **llist)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarser, MAT_COARSEN_CLASSID, 1);
  PetscCheck(coarser->agg_lists, PetscObjectComm((PetscObject)coarser), PETSC_ERR_ARG_WRONGSTATE, "No linked list - generate it or call ApplyCoarsen");
  *llist             = coarser->agg_lists;
  coarser->agg_lists = NULL; /* giving up ownership */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatCoarsenSetFromOptions - Sets various coarsen options from the options database.

   Collective

   Input Parameter:
.  coarser - the coarsen context.

   Options Database Key:
.  -mat_coarsen_type  <type> - (for instance, mis), use -help for a list of available methods

   Level: advanced

   Note:
   Sets the `MatCoarsenType` to `MATCOARSENMISK` if has not been set previously

.seealso: `MatCoarsen`, `MatCoarsenType`, `MatCoarsenApply()`, `MatCoarsenCreate()`, `MatCoarsenSetType()`
@*/
PetscErrorCode MatCoarsenSetFromOptions(MatCoarsen coarser)
{
  PetscBool   flag;
  char        type[256];
  const char *def;

  PetscFunctionBegin;
  PetscObjectOptionsBegin((PetscObject)coarser);
  if (!((PetscObject)coarser)->type_name) {
    def = MATCOARSENMISK;
  } else {
    def = ((PetscObject)coarser)->type_name;
  }

  PetscCall(PetscOptionsFList("-mat_coarsen_type", "Type of aggregator", "MatCoarsenSetType", MatCoarsenList, def, type, 256, &flag));
  if (flag) PetscCall(MatCoarsenSetType(coarser, type));
  /*
   Set the type if it was never set.
   */
  if (!((PetscObject)coarser)->type_name) PetscCall(MatCoarsenSetType(coarser, def));

  PetscTryTypeMethod(coarser, setfromoptions, PetscOptionsObject);
  PetscOptionsEnd();

  PetscFunctionReturn(PETSC_SUCCESS);
}
