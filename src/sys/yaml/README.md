This directory contains a partial and slightly modified copy of
(LibYAML)[https://pyyaml.org/wiki/LibYAML] sources corresponding to
release 0.2.5. A copy of the LibYAML [license](License) is also included.

A list of the modifications follow:

* The emitter API and other output-related parts have been removed,
  as we are only interested in the input-related parts.
* `yaml_get_version()` and `yaml_get_version_string()` have been
  removed, as we do not need them.
* The constant `0` as been replaced by `NULL` in a few places to
  silence `-Wzero-as-null-pointer-constant` when using C++ compilers.
* The macro `YAML_DECLARE()` in `yaml.h` has been modified to specify
  `static` visibility for all LibYAML symbols.

Thanks to the exceptionally good source code organization in LibYAML,
the removals and minor modifications occur in large contiguous blocks
of code. This will make it quite easy to merge back upstream changes
to keep this copy properly synchronized and maintained, or even
incorporate some of the removed features if such need ever arises. We
recommend using a merge tool like [`meld`](https://meldmerge.org/) to
perform these future maintenance updates.
