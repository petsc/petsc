#ifndef RO_CONFIG
#define RO_CONFIG 1

/* HASH_SIZE is used to hash line numbers that have exceptions
   into a table of the same size. We do "linenumber % HASH_SIZE".
   If you know what you are doing, you can adjust this depending
   on how many exceptions you get. Note that an array of this size
   is allocated, so do not make it too large. */

#define HASH_SIZE 11

/* INITIAL_MAX_FILES controls the amount of space initially allocated for
   various per-{(file,routine) pair} data structures. If you have more than
   this, you will avoid a few reallocations by adjusting it.

   This can also be set with the user function "reportonce_files(int)". */

#define INITIAL_MAX_FILES 30

/* FILE_GROWTH_INCREMENT indicates how many (file,routine) pairs we will
   add to our capacity each time we exceed the existing capacity. */

#define FILE_GROWTH_INCREMENT 20


#endif /* RO_CONFIG */

