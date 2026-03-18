#make clean
rm -rf arch-linux-c-opt/
# 避免 Python 导入 rlcg/utils 而非 PETSc 的 config/utils（导致 generatefortranbindings 找不到）
unset PYTHONPATH
./configure --download-f2cblaslapack --with-mpi=0 --with-cuda=1 --download-cusp \
  COPTFLAGS=-O3 \
  CXXOPTFLAGS="-O3 -std=c++14" \
  FOPTFLAGS=-O3 \
  --with-debugging=0 \
  --with-cxx-dialect=C++14
make all
cd src/ksp/ksp/tutorials/
rm test_cg
make test_cg
