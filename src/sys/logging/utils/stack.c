
/*
     This defines part of the private API for logging performance information. It is intended to be used only by the
   PETSc PetscLog...() interface and not elsewhere, nor by users. Hence the prototypes for these functions are NOT
   in the public PETSc include files.

*/
#include <petsc/private/logimpl.h> /*I    "petscsys.h"   I*/

/*@C
  PetscIntStackDestroy - This function destroys a stack.

  Not Collective

  Input Parameter:
. stack - The stack

  Level: developer

.seealso: `PetscIntStackCreate()`, `PetscIntStackEmpty()`, `PetscIntStackPush()`, `PetscIntStackPop()`, `PetscIntStackTop()`
@*/
PetscErrorCode PetscIntStackDestroy(PetscIntStack stack)
{
  PetscFunctionBegin;
  PetscValidPointer(stack, 1);
  PetscCall(PetscFree(stack->stack));
  PetscCall(PetscFree(stack));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscIntStackEmpty - This function determines whether any items have been pushed.

  Not Collective

  Input Parameter:
. stack - The stack

  Output Parameter:
. empty - `PETSC_TRUE` if the stack is empty

  Level: developer

.seealso: `PetscIntStackCreate()`, `PetscIntStackDestroy()`, `PetscIntStackPush()`, `PetscIntStackPop()`, `PetscIntStackTop()`
@*/
PetscErrorCode PetscIntStackEmpty(PetscIntStack stack, PetscBool *empty)
{
  PetscFunctionBegin;
  PetscValidPointer(stack, 1);
  PetscValidBoolPointer(empty, 2);
  *empty = stack->top == -1 ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscIntStackTop - This function returns the top of the stack.

  Not Collective

  Input Parameter:
. stack - The stack

  Output Parameter:
. top - The integer on top of the stack

  Level: developer

.seealso: `PetscIntStackCreate()`, `PetscIntStackDestroy()`, `PetscIntStackEmpty()`, `PetscIntStackPush()`, `PetscIntStackPop()`
@*/
PetscErrorCode PetscIntStackTop(PetscIntStack stack, int *top)
{
  PetscFunctionBegin;
  PetscValidPointer(stack, 1);
  PetscValidIntPointer(top, 2);
  *top = stack->stack[stack->top];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscIntStackPush - This function pushes an integer on the stack.

  Not Collective

  Input Parameters:
+ stack - The stack
- item  - The integer to push

  Level: developer

.seealso: `PetscIntStackCreate()`, `PetscIntStackDestroy()`, `PetscIntStackEmpty()`, `PetscIntStackPop()`, `PetscIntStackTop()`
@*/
PetscErrorCode PetscIntStackPush(PetscIntStack stack, int item)
{
  PetscFunctionBegin;
  PetscValidPointer(stack, 1);
  if (++stack->top >= stack->max) {
    stack->max *= 2;
    PetscCall(PetscRealloc(stack->max * sizeof(*stack->stack), &stack->stack));
  }
  stack->stack[stack->top] = item;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscIntStackPop - This function pops an integer from the stack.

  Not Collective

  Input Parameter:
. stack - The stack

  Output Parameter:
. item  - The integer popped

  Level: developer

.seealso: `PetscIntStackCreate()`, `PetscIntStackDestroy()`, `PetscIntStackEmpty()`, `PetscIntStackPush()`, `PetscIntStackTop()`
@*/
PetscErrorCode PetscIntStackPop(PetscIntStack stack, int *item)
{
  PetscFunctionBegin;
  PetscValidPointer(stack, 1);
  PetscCheck(stack->top != -1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Stack is empty");
  if (item) {
    PetscValidIntPointer(item, 2);
    PetscCall(PetscIntStackTop(stack, item));
  }
  --stack->top;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscIntStackCreate - This function creates a stack.

  Not Collective

  Output Parameter:
. stack - The stack

  Level: developer

.seealso: `PetscIntStackDestroy()`, `PetscIntStackEmpty()`, `PetscIntStackPush()`, `PetscIntStackPop()`, `PetscIntStackTop()`
@*/
PetscErrorCode PetscIntStackCreate(PetscIntStack *stack)
{
  PetscFunctionBegin;
  PetscValidPointer(stack, 1);
  PetscCall(PetscNew(stack));

  (*stack)->top = -1;
  (*stack)->max = 128;

  PetscCall(PetscCalloc1((*stack)->max, &(*stack)->stack));
  PetscFunctionReturn(PETSC_SUCCESS);
}
