#!/bin/bash

# Before running this script, make sure set up the directory for storing the downloaded matrix files (ARCHIVE_LOCATION)
ARCHIVE_LOCATION=""
SSGET=./ssget

if [ ! "${BENCHMARK}" ]; then
    BENCHMARK="spmv"
    echo "BENCHMARK   environment variable not set - assuming \"${BENCHMARK}\"" 1>&2
fi

if [ ! "${DRY_RUN}" ]; then
    DRY_RUN="false"
    echo "DRY_RUN     environment variable not set - assuming \"${DRY_RUN}\"" 1>&2
fi

if [ ! "${EXECUTOR}" ]; then
    EXECUTOR="cuda"
    echo "EXECUTOR    environment variable not set - assuming \"${EXECUTOR}\"" 1>&2
fi

if [ ! "${SEGMENTS}" ]; then
    echo "SEGMENTS    environment variable not set - running entire suite" 1>&2
    SEGMENTS=1
    SEGMENT_ID=1
elif [ ! "${SEGMENT_ID}" ]; then
    echo "SEGMENT_ID  environment variable not set - exiting" 1>&2
    exit 1
fi

if [ ! "${FORMATS}" ]; then
    echo "FORMATS    environment variable not set - assuming \"csr\"" 1>&2
    FORMATS="csr"
fi

if [ ! "${DEVICE_ID}" ]; then
    DEVICE_ID="0"
    echo "DEVICE_ID   environment variable not set - assuming \"${DEVICE_ID}\"" 1>&2
fi

if [ ! "${SINGLE_JSON}" ]; then
    SINGLE_JSON="false"
    echo "SINGLE_JSON environment variable not set - assuming \"${SINGLE_JSONR}\"" 1>&2
fi

if [ ! "${LAUNCHER}" ]; then
    LAUNCHER=""
    echo "LAUNCHER    environment variable not set - assuming \"${LAUNCHER}\"" 1>&2
fi

# This allows using a matrix list file for benchmarking.
# The file should contains a suitesparse matrix on each line.
# The allowed formats to target suitesparse matrix is:
#   id or group/name or name.
# Example:
# 1903
# Freescale/circuit5M
# thermal2
if [ ! "${MATRIX_LIST_FILE}" ]; then
    use_matrix_list_file=0
elif [ -f "${MATRIX_LIST_FILE}" ]; then
    use_matrix_list_file=1
else
    echo -e "A matrix list file was set to ${MATRIX_LIST_FILE} but it cannot be found."
    exit 1
fi

# Runs the SpMV benchmarks for all SpMV formats by using file $1 as the input,
# and updating it with the results. Backups are created after each
# benchmark run, to prevent data loss in case of a crash. Once the benchmarking
# is completed, the backups and the results are combined, and the newest file is
# taken as the final result.
run_spmv_benchmarks() {
    [ "${DRY_RUN}" == "true" ] && return
    if [ "${EXECUTOR}" == "cuda" ]; then
        ${LAUNCHER} ../mat/tests/bench_spmv -formats "${FORMATS}" -repetitions 5 -use_gpu -AJSON "$1"
    else
        ${LAUNCHER} ../mat/tests/bench_spmv -formats "${FORMATS}" -repetitions 5 -AJSON "$1"
    fi
}

NUM_PROBLEMS="$(${SSGET} -n)"

# Creates an input file for $1-th problem in the SuiteSparse collection
generate_suite_sparse_input() {
    INPUT=$(${SSGET} -i "$1" -e)
    cat << EOT
[{
    "filename": "${INPUT}",
    "problem": $(${SSGET} -i "$1" -j)
}]
EOT
}

# Append an input file for $1-th problem in the SuiteSparse collection
append_suite_sparse_input() {
    INPUT=$(${SSGET} -i "$1" -e)
    cat << EOT
 {
    "filename": "${INPUT}",
    "problem": $(${SSGET} -i "$1" -j)
 },
EOT
}

parse_matrix_list() {
    local source_list_file=$1
    local benchmark_list=""
    local id=0
    for mtx in $(cat ${source_list_file}); do
	echo $mtx >&2
        if [[ ! "$mtx" =~ ^[0-9]+$ ]]; then
            if [[ "$mtx" =~ ^[a-zA-Z0-9_-]+$ ]]; then
                id=$(${SSGET} -s "[ @name == $mtx ]")
            elif [[ "$mtx" =~ ^([a-zA-Z0-9_-]+)\/([a-zA-Z0-9_-]+)$ ]]; then
                local group="${BASH_REMATCH[1]}"
                local name="${BASH_REMATCH[2]}"
                id=$(${SSGET} -s "[ @name == $name ] && [ @group == $group ]")
            else
                >&2 echo -e "Could not recognize entry $mtx."
            fi
        else
            id=$mtx
        fi
        benchmark_list="$benchmark_list $id"
    done
    echo "$benchmark_list"
}

if [ $use_matrix_list_file -eq 1 ]; then
    MATRIX_LIST=($(parse_matrix_list $MATRIX_LIST_FILE))
    NUM_PROBLEMS=${#MATRIX_LIST[@]}
fi

RESULT_DIR="results/${SYSTEM_NAME}/${EXECUTOR}/SuiteSparse"
if [ "${SINGLE_JSON}" == "true" ]; then
    RESULT_FILE="${RESULT_DIR}/SEGMENT${SEGMENT_ID}.json"
    cat << EOT >"${RESULT_FILE}"
[
EOT
    mkdir -p "$(dirname "${RESULT_FILE}")"
fi
LOOP_START=$((1 + (${NUM_PROBLEMS}) * (${SEGMENT_ID} - 1) / ${SEGMENTS}))
LOOP_END=$((1 + (${NUM_PROBLEMS}) * (${SEGMENT_ID}) / ${SEGMENTS}))

for (( p=${LOOP_START}; p < ${LOOP_END}; ++p )); do
    if [ $use_matrix_list_file -eq 1 ]; then
        i=${MATRIX_LIST[$((p-1))]}
    else
        i=$p
    fi
    if [ "${BENCHMARK}" == "preconditioner" ]; then
        break
    fi
    if [ "$(${SSGET} -i "$i" -preal)" = "0" ] || [ "$(${SSGET} -i "$i" -pbinary)" = "1" ]; then
        [ "${DRY_RUN}" != "true" ] && ${SSGET} -i "$i" -c >/dev/null
        continue
    fi
    # filter matrices for spmv tests
    if [ "${BENCHMARK}" == "spmv" ]; then
	# deselect non-square matrices and matrices with more than 2B non zeros
        if [ "$(${SSGET} -i "$i" -pcols)" != "$(${SSGET} -i "$i" -prows)" ] || [ "$(${SSGET} -i "$i" -pnonzeros)" -gt 2000000000 ]; then
            [ "${DRY_RUN}" != "true" ] && ${SSGET} -i "$i" -c >/dev/null
            continue
	    fi
    fi
    PREFIX="(PROB$p/${NUM_PROBLEMS} ID$i SEG${SEGMENT_ID}):\t"
    GROUP=$(${SSGET} -i "$i" -pgroup)
    NAME=$(${SSGET} -i "$i" -pname)
    if [ "${SINGLE_JSON}" == "false" ]; then
        RESULT_FILE="${RESULT_DIR}/${GROUP}/${NAME}.json"
        mkdir -p "$(dirname "${RESULT_FILE}")"
        echo -e "${PREFIX}Extracting the matrix for ${GROUP}/${NAME}" 1>&2
        generate_suite_sparse_input "$i" >"${RESULT_FILE}"
        echo -e "${PREFIX}Running SpMV for ${GROUP}/${NAME}" 1>&2
        run_spmv_benchmarks "${RESULT_FILE}"
        echo -e "${PREFIX}Cleaning up problem ${GROUP}/${NAME}" 1>&2
        [ "${DRY_RUN}" != "true" ] && ${SSGET} -i "$i" -c >/dev/null
    else
        append_suite_sparse_input "$i" >>"${RESULT_FILE}"
    fi
done
if [ "${SINGLE_JSON}" == "true" ]; then
    cat << EOT >"${RESULT_FILE}"
]
EOT
    echo -e "${PREFIX}Running SpMV for SEG${SEGMENT_ID}" 1>&2
    run_spmv_benchmarks "${RESULT_FILE}"
    for (( p=${LOOP_START}; p < ${LOOP_END}; ++p )); do
        if [ $use_matrix_list_file -eq 1 ]; then
            i=${MATRIX_LIST[$((p-1))]}
        else
            i=$p
	fi
	echo -e "${PREFIX}Cleaning up problem ${i}" 1>&2
        [ "${DRY_RUN}" != "true" ] && ${SSGET} -i "$i" -c >/dev/null
    done
fi
