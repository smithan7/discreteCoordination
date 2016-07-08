################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/agent.cpp \
../src/denseGraphCoord.cpp \
../src/frontier.cpp \
../src/graph.cpp \
../src/miniGraph.cpp \
../src/treeNode.cpp \
../src/world.cpp 

OBJS += \
./src/agent.o \
./src/denseGraphCoord.o \
./src/frontier.o \
./src/graph.o \
./src/miniGraph.o \
./src/treeNode.o \
./src/world.o 

CPP_DEPS += \
./src/agent.d \
./src/denseGraphCoord.d \
./src/frontier.d \
./src/graph.d \
./src/miniGraph.d \
./src/treeNode.d \
./src/world.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/include/opencv -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


