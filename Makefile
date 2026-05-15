PYTHON ?= python3

test: test-moleculedataset test-core test-utils test-units

# pytest returns error code 5 if no test is collected.
test-core:
	OMP_NUM_THREADS=4 ${PYTHON} -m pytest -vv -n 4 --dist load --disable-warnings ./bytemol/core/tests

test-units:
	OMP_NUM_THREADS=4 ${PYTHON} -m pytest -vv -n 4 --dist load --disable-warnings ./bytemol/units/tests

test-utils:
	OMP_NUM_THREADS=4 ${PYTHON} -m pytest -vv -n 4 --dist loadfile --disable-warnings ./bytemol/utils/tests

test-moleculedataset:
	OMP_NUM_THREADS=4 ${PYTHON} -m pytest -vv -n 4 --dist loadfile --disable-warnings ./moleculedataset/tests