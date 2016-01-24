# Copyright 2015 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

#Flags for directing the runtime makefile what to include
DEBUG           ?= 0		# Include debugging symbols
OUTPUT_LEVEL    ?= LEVEL_DEBUG	# Compile time print level
SHARED_LOWLEVEL ?= 0		# Use the shared low level
ALT_MAPPERS     ?= 0		# Compile the alternative mappers

# Put the binary file name here
OUTFILE		?= ckt_sim
# List all the application source files here
GEN_SRC		?= circuit.cc circuit_cpu.cc circuit_mapper.cc	# .cc files
GEN_GPU_SRC	?= circuit_gpu.cu				# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?=
CC_FLAGS	?= -DLEGION_PROF
NVCC_FLAGS	?=
GASNET_FLAGS	?=
LD_FLAGS	?=

###########################################################################
#
#   Don't change anything below here
#   
###########################################################################

include $(LG_RT_DIR)/runtime.mk

cleanall:
	@$(RM) -rf $(ALL_OBJS) $(OUTFILE)
