# Reproduce the saavg-cos results. Override BLAS pinning var for your backend:
#   Apple Accelerate -> VECLIB_MAXIMUM_THREADS, OpenBLAS/MKL -> OMP_NUM_THREADS
PY ?= python
PIN ?= VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1

.PHONY: help test lookup validate figures map risk benchmark accuracy convergence clean

help:
	@echo "make test         - fast correctness tests (pytest)"
	@echo "make validate     - validate COS vs Monte-Carlo (a few minutes)"
	@echo "make lookup       - build the full all-branches COS IM lookup zarr (~2-3 min)"
	@echo "make figures      - regenerate the fast figures (map, risk, mp scaling)"
	@echo "make accuracy     - regenerate density+dispersion figures (MC-heavy, minutes)"
	@echo "make convergence  - regenerate convergence figure (very MC-heavy, ~10-20 min)"
	@echo "make clean        - remove the generated lookup zarr"

test:
	$(PY) -m pytest -q tests

lookup:
	$(PIN) $(PY) scripts/prepare_im_lookup.py

validate:
	$(PY) scripts/validate_against_mc.py

figures: map risk benchmark

map:
	$(PY) scripts/make_hazard_map.py

risk:
	$(PY) scripts/compare_risk_metrics.py

benchmark:
	$(PY) scripts/benchmark_multiprocessing.py

accuracy:
	$(PY) scripts/figure_density_dispersion.py

convergence:
	$(PY) scripts/figure_convergence.py

clean:
	rm -rf data/im_lookup_cos.zarr
