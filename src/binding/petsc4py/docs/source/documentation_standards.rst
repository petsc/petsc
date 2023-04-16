Documentation standards for PETSc4py
====================================

Subject to exceptions given below, new contributions to PETSc4py **must**
include `type annotations <python:typing>` for function parameters and results,
and docstrings on every class, function and method.

The documentation should be consistent with the corresponding C API
documentation, including copying text where this is appropriate. More in-depth
documentation from the C API (such as extended discussions of algorithmic or
performance factors) should not be copied.

Docstring standards
-------------------
Docstrings are to be written in `numpydoc:format` format.

The first line of a function or method docstring must be a short description of
the method in imperative mood ("Return the norm of the matrix.") "Return" is
to be preferred over "Get" in this sentence. A blank line must follow this
description.

If the corresponding C API documentation lists a function as being collective,
then this information must be repeated on the next line of the docstring.  E.g.
"Not collective.", "Logically collective on X.", "Collective."

The initial description section can contain more information if this is useful.
In particular, if there is a PETSc manual chapter about a class, then this
should be referred to from here.

Use double backticks around literals (like strings and numbers). E.g.
\`\`2\`\`, \`\`"foo"\`\`.

Reference PETSc functions simply using backticks. eg: `petsc.KSP`. refers to
the PETSc C documentation for KSP. Do **not** use URLs in docstrings. Always
use Intersphinx references.

The following sections describe the use of numpydoc sections. Other sections
allowed by numpydoc may be included if they are useful.

Parameters
..........

This is required unless there are no parameters, or it will be completely
obvious to even a novice user what the parameters do.

Types should only be specified in this section if for some reason the types
provided by typing prove to be inadequate. If no type is being specified, do
not include a colon (``:``) to the right of the parameter name.

Use `Sys.getDefaultComm` when specifying the default communicator.

Returns
.......

This should only be specified if the return value is not obvious from the
initial description and typing.

If a "Returns" section is required, the type of the returned items *must* be
specified, even if this duplicates typing information.

See Also
........

If any of the following apply, then this section is required. The order of
entries is as follows. Other links are permitted in this section if they add
information useful to users.

Every ``setFromOptions`` must include the link \`petsc_options\`.

Any closely related part of the PETSc4py API not already linked in the
docstring should appear (e.g. setters and getters should cross-refer).

If there is a corresponding C API documentation page, this must be linked from
the "See also" section. E.g. \`petsc.MatSetValues\`.

End docstring with an empty line - "closing three quotation marks must be on a
line by itself, preferably preceded by a blank line"

.. warning::

    The docstrings must not cause Sphinx warnings.


Type hint standards
-------------------

If returning ``self``, use ``-> Self`` in function signature.

Type hints are not required when the static type signature includes a PETSc
type (e.g. ``Vec x``). These will be automatically generated. This will also
work for ``= None``. When using type hints, use spacing around the equals in
any ``= None``.

Communicators in type signatures must use Python typing instead of c-typing
(i.e. ``comm: Comm`` not ``Comm comm``). This is because communicators
can come from ``mpi4py`` and not just the ``petsc4py.PETSc.Comm`` class.

For petsc4py native types that are can be strings, the type is ``argument:
KSP.Type | str`` (not eg: ``KSPType argument``). If the type is strictly an
enum the ``| str`` can be omitted. Full signature example::

    def setType(self, ksp_type: KSP.Type | str) -> None:

If a NumPy is returned, use ``ArrayInt``/``ArrayReal``/``ArrayScalar`` as the
return type.
