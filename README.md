# README #

### What is this repository for? ###

Host the PETSc numerical library package: https://petsc.org

### How do I get set up? ###

* Download: https://petsc.org/release/download
* Install: https://petsc.org/release/install

### Contribution guidelines ###

* See the file [CONTRIBUTING](./CONTRIBUTING)

### Who do I talk to? ###

* https://gitlab.com/petsc/petsc/-/issues
* petsc-maint@mcs.anl.gov
* https://petsc.org/release/community/mailing/

### Code of Conduct ###

* See the file [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)

---

## 编译 PETSc CG 并测试

### 前置条件

* OpenMPI（可选，当前脚本使用 `--with-mpi=0`）
* CUDA 工具链（推荐 CUDA 12.x，CUDA 11.5 与 GCC 11 存在兼容性问题）
* 若在 conda 环境中，编译前建议 `unset PYTHONPATH`（脚本已包含）

### 编译

```bash
cd baselines/petsc
bash compile_PETSc.sh
```

编译完成后，`test_cg` 可执行文件位于 `src/ksp/ksp/tutorials/test_cg`。

### 测试

1. 设置环境变量（在项目根目录或 `baselines/petsc` 下执行）：

```bash
export PETSC_DIR=/path/to/rlcg/baselines/petsc
export PETSC_ARCH=arch-linux-c-opt
```

2. 单矩阵测试（需将 `matrix.mtx` 替换为实际矩阵文件路径）：

```bash
cd $PETSC_DIR/src/ksp/ksp/tutorials
./test_cg /path/to/matrix.mtx -ksp_max_it 10000 -ksp_monitor -ksp_type cg \
  -mat_type aijcusparse -vec_type cuda -use_gpu_aware_mpi 0 -pc_type none \
  -ksp_norm_type unpreconditioned -ksp_rtol 1e-10
```

3. 批量测试（需准备 `valid_matrix_set.csv`，矩阵文件放在 `$HOME/data/matrix` 或 `/data/matrix`）：

```bash
cd $PETSC_DIR/src/ksp/ksp/tutorials
# 确保 valid_matrix_set.csv 存在，且矩阵 .mtx 文件在指定目录
bash test_PETSc.sh
```

`valid_matrix_set.csv` 格式需包含矩阵名称列（如 `Name`），脚本会据此在矩阵目录中查找对应的 `{name}.mtx` 文件。可将项目根目录的 `valid_matrix_set.csv` 复制到该目录，或修改 `test_PETSc.sh` 中的 `input` 和矩阵路径。
