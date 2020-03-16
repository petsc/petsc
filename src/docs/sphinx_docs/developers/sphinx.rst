Sphinx Documentation Guidelines
===============================

Avoid duplicating information whenever possible:

* Prefer to link to other documentation instead of reiterating information found there.
* Use the ``includeliteral`` directive to directly include pieces of source code, e.g.

.. code-block:: rst

    .. literalinclude:: /../../../src/sys/error/err.c
       :language: c
       :start-at: PetscErrorCode PetscError(
       :end-at: PetscFunctionReturn(0)
       :append: }
