TMPDIR := $(shell mktemp -d)
TESTDIR := $(shell cd ../tests; pwd)

python:
	+$(MAKE) -C model
	+$(MAKE) -C descriptor
	f2py -c -m fmodules model.f90 descriptor/cutoffs.f90 descriptor/gaussian.f90 descriptor/zernike.f90 model/neuralnetwork.f90

py2tests:
	mkdir -p $(TMPDIR)/py2_amptests
	cd $(TMPDIR)/py2_amptests && nosetests -v $(TESTDIR)

py3tests:
	mkdir -p $(TMPDIR)/py3_amptests
	cd $(TMPDIR)/py3_amptests && nosetests3 -v $(TESTDIR)
