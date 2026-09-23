CUDA_ARCH ?= $(if $(SM),sm_$(SM),$(if $(SM_VERSION),sm_$(SM_VERSION),$(if $(ARCH),$(ARCH),sm_80)))

all:
	nvcc -arch=${CUDA_ARCH} ${CUFILES} ${DEF} -o ${EXECUTABLE}
clean:
	rm -f *~ *.exe
