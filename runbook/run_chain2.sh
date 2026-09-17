#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/.."; L=logs
modal run modal_app.py::pack_pools_fast > $L/2026-09-17-pack_pools_fast.log 2>&1; rc=$?; echo "exit=$rc" >> $L/2026-09-17-pack_pools_fast.log
[ $rc -ne 0 ] && { echo "pack_pools_fast failed" > $L/CHAIN2_FAILED; exit 1; }
(modal run modal_app.py::spiral_fit --sense CW  --steps 1500 --z-begin 10000 --z-end 11000 > $L/2026-09-17-fit_CW_1500.log 2>&1;  echo "exit=$?" >> $L/2026-09-17-fit_CW_1500.log) &
(modal run modal_app.py::spiral_fit --sense ACW --steps 1500 --z-begin 10000 --z-end 11000 > $L/2026-09-17-fit_ACW_1500.log 2>&1; echo "exit=$?" >> $L/2026-09-17-fit_ACW_1500.log) &
wait; echo done > $L/CHAIN2_DONE
