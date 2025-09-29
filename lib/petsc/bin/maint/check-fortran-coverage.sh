#!/bin/bash -e

echo "Running \"make fortranbindings\" on previous and current commit to determine differences"
repo_root="$(git rev-parse --show-toplevel)"
TMPDIR=$(mktemp -d)
before=$(git rev-parse HEAD~1)
after=$(git rev-parse HEAD)

extract_subroutines() {
  find "$1" -type f \( -name '*.F' -o -name '*.h90' -o -name '*.f' -o -name '*.f90' \) \
    -exec grep -hiE '^[[:space:]]*subroutine[[:space:]]+[A-Za-z0-9_]+' {} + | \
    awk '{for(i=1;i<=NF;i++) if($i=="subroutine") print $(i+1)}' | sort | uniq
}

for rev in $before $after; do
  git worktree add -f "$TMPDIR/wt_$rev" $rev
  (
    cd "$TMPDIR/wt_$rev"
    export PETSC_DIR="$TMPDIR/wt_$rev"
    ./configure --with-fortran-bindings=1
    petsc_arch=$(find . -maxdepth 1 -type d -name 'arch-*' | head -n1 | sed 's|^\./||')
    # make PETSC_DIR="$TMPDIR/wt_$rev" PETSC_ARCH="$petsc_arch" all check
    cp -r "$petsc_arch/ftn" "$TMPDIR/$rev"
  )
  git worktree remove --force "$TMPDIR/wt_$rev"
done

DIFF_FILE="fortran_bindings_diff.txt"
diff -ru "$TMPDIR/$before" "$TMPDIR/$after" > "$DIFF_FILE" || true
count_before=$(extract_subroutines "$TMPDIR/$before" | wc -l)
count_after=$(extract_subroutines "$TMPDIR/$after" | wc -l)
echo "Before ($before): $count_before Fortran stubs"
echo "After  ($after): $count_after Fortran stubs"
echo "Full diff saved in $DIFF_FILE"
