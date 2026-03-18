input="valid_matrix_set.csv"
{
  read
  i=1
  while IFS=',' read -r id group data a b c
  do
    for data1 in `find "/data/matrix" -name "$data.mtx"`
    do
      echo -n $data1
      timeout -s 9 4m ./test_cg $data1 -ksp_max_it 10000 -ksp_monitor -ksp_type cg -mat_type aijcusparse -vec_type cuda -use_gpu_aware_mpi 0 -pc_type none -ksp_norm_type unpreconditioned
    done
    i=`expr $i + 1`
  done 
} < "$input"
