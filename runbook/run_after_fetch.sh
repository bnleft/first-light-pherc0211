#!/usr/bin/env bash
# Chain: wait for fetch_dataset -> pack_pools -> CW and ACW 1,500-step fits in parallel.
set -u
cd "$(dirname "$0")/.."
L=logs
until grep -q "^exit=" $L/2026-09-17-fetch_dataset.log; do sleep 30; done
if ! grep -q "^exit=0" $L/2026-09-17-fetch_dataset.log; then echo "fetch failed" > $L/CHAIN_FAILED; exit 1; fi
modal run modal_app.py::pack_pools > $L/2026-09-17-pack_pools.log 2>&1; rc=$?; echo "exit=$rc" >> $L/2026-09-17-pack_pools.log
[ $rc -ne 0 ] && { echo "pack_pools failed" > $L/CHAIN_FAILED; exit 1; }
(modal run modal_app.py::spiral_fit --sense CW  --steps 1500 --z-begin 10000 --z-end 11000 > $L/2026-09-17-fit_CW_1500.log 2>&1;  echo "exit=$?" >> $L/2026-09-17-fit_CW_1500.log) &
(modal run modal_app.py::spiral_fit --sense ACW --steps 1500 --z-begin 10000 --z-end 11000 > $L/2026-09-17-fit_ACW_1500.log 2>&1; echo "exit=$?" >> $L/2026-09-17-fit_ACW_1500.log) &
wait
echo "chain done" > $L/CHAIN_DONE
