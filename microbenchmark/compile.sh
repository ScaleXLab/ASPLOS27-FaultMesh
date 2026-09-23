#!/bin/bash

WORKLOADS=("2DCONV" "ATAX" "bfs" "BICG" "GEMM" "GESUMMV" "hellinger" "MVT" "nw" "XSBench")
ARCH="${ARCH:-sm_80}"

BASE_DIR=$(pwd)

echo "Starting compilation for all workloads..."

for workload in "${WORKLOADS[@]}"; do
    if [ -d "$workload" ]; then
        echo "=========================================="
        echo "Processing directory: $workload"
        echo "=========================================="
        
        cd "$workload" || exit

        case "$workload" in
            "bfs")
                echo "[BFS] Compiling main.cu..."
                nvcc -arch="${ARCH}" main.cu -o bfs
		chmod +x run

                if [ -d "inputgen" ]; then
                    echo "[BFS] Building inputgen..."
                    cd inputgen
                    rm -f graphgen
                    make
                    cd .. 
                else
                    echo "[Error] 'inputgen' directory not found inside bfs."
                fi

                echo "[BFS] Skipping legacy dataset generation; current benchmark generates graphs in-code."
                ;;

            "hellinger")
                echo "[Hellinger] Compiling main.cu..."
                nvcc -arch="${ARCH}" main.cu -o hellinger
		chmod +x run
                ;;

            *)
		chmod +x run
                if [ -f "Makefile" ] || [ -f "makefile" ]; then
                    echo "[$workload] Running make..."
                    if [ "$workload" = "XSBench" ]; then
                        make clean 2>/dev/null || true
                        make SM_VERSION=80 ARCH="${ARCH}"
                    else
                        make ARCH="${ARCH}"
                    fi
                else
                    echo "[Warning] No Makefile found in $workload."
                fi
                ;;
        esac

        cd "$BASE_DIR"
        echo ""
    else
        echo "[Warning] Directory '$workload' does not exist. Skipping."
    fi
done

cd "$BASE_DIR"
nvcc -arch="${ARCH}" hostpin.cu -o hostpin

echo "All tasks completed."
