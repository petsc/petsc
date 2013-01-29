#!/bin/bash

#
# Usage:   Simply call this script from $PETSC_DIR.
#          You can pass an optional parameter for selecting a subfolder within src/
#
# Example: src/contrib/style/rununcrustify.sh vec
#          


UNCRUSTIFY=../../uncrustify/bin/uncrustify #change this as required
UNCRUSTIFYCONF=bin/uncrustify.cfg
DIFFTOOL=meld
TMPDIR=src_uncrustified


echo "Uncrustifier for PETSc. Opens one file after another in your favorite diff-viewer. Use Ctrl+C to exit."

# Create directory hierarchy of src/ in TMPDIR
rm -rf $TMPDIR
cp -R src $TMPDIR
find $TMPDIR -type f | xargs rm

# Now uncrustify one file after another and launch a diffviewer on the output
for f in `find src/$1 -name *.[ch] -or -name *.cu`
do
  # Skip automatically generated Fortran stubs:
  if [[ "$f" == *ftn-auto* ]]
  then
    continue
  fi

  TMPFILE=$TMPDIR/${f#src/}

  # Run uncrustify, place output in TMPDIR
  $UNCRUSTIFY -c $UNCRUSTIFYCONF -f $f -o $TMPFILE

  # Run a difftool (e.g. meld) and incorporate updates into file
  
  filediff=`diff --brief $f $TMPFILE | grep "differ"`
  if [ -n "$filediff" ]
  then
    dummy=`$DIFFTOOL $f $TMPFILE`
  fi
done

