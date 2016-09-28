#!/bin/sh
total=0
success=0
failed=0
todo=0
skip=0
failures=""
for file in counts/*.counts; do
  total_str=`grep total $file | cut -f2 -d" "`
  let total=$total+$total_str
  #
  failed_str=`grep failed $file | cut -f2 -d" "`
  let failed=$failed+$failed_str
  #
  success_str=`grep success $file | cut -f2 -d" "`
  let success=$success+$success_str
  #
  todo_str=`grep todo $file | cut -f2 -d" "`
  if test -n "${todo_str}"; then 
    let todo=$todo+$todo_str
  fi
  #
  skip_str=`grep skip $file | cut -f2 -d" "`
  if test -n "${skip_str}"; then 
    let skip=$skip+$skip_str
  fi

  failures_fromc=`grep failures $file | cut -f2- -d" "`
  if test -n "${failures_fromc//[[:space:]]}"; then
    failures="$failures $failures_fromc"
  fi
done
echo "# FAILED $failures"
#echo "# failed ${failed}/${total} tests; ${percent}% ok"
#
percent=`echo "scale=3; ${success}/${total} * 100" | bc -l`
echo "# success ${success}/${total} tests (${percent}%) "
#
percent=`echo "scale=3; ${failed}/${total} * 100" | bc -l`
echo "# failed ${failed}/${total} tests (${percent}%) "
#
percent=`echo "scale=3; ${todo}/${total} * 100" | bc -l`
echo "# todo ${todo}/${total} tests (${percent}%) "
#
percent=`echo "scale=3; ${skip}/${total} * 100" | bc -l`
echo "# skip ${skip}/${total} tests (${percent}%) "
