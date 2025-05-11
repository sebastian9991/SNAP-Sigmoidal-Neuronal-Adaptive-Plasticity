#!/bin/bash



main() {
    
    echo "Running end-to-end Experiments for all learning rules."
    uv run -m tests.sequential_tests.seq_forget_test

}


main "${@}"
