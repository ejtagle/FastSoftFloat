cmake_minimum_required(VERSION 3.15)

project(EmbeddedUnitTests LANGUAGES C CXX ASM)

find_bsp(
	ID com.sysprogs.project_importers.stm32.cubemx
	SOURCE_PROJECT EmbeddedUnitTests.ioc
	FRAMEWORKS com.sysprogs.embedded.semihosting_and_profiler
	CONFIGURATION
		"com.sysprogs.toolchainoptions.arm.libctype=--specs=nano.specs -u _printf_float"
		com.sysprogs.toolchainoptions.arm.syscallspecs=--specs=rdimon.specs)

bsp_include_directories(${BSP_ROOT}/Tests/Inc)

bsp_compile_flags(-Wno-psabi -ffast-math)
bsp_compile_flags($<$<COMPILE_LANGUAGE:CXX>:-std=c++23 -Wno-volatile>)

add_bsp_based_executable(
	NAME EmbeddedUnitTests
	SOURCES ${BSP_ROOT}/Tests/Src/tests.cpp
	GENERATE_BIN
	GENERATE_MAP
	GENERATE_HEX)

target_link_libraries(EmbeddedUnitTests PRIVATE -Wl,-gc-sections)
set_source_files_properties(${BSP_ROOT}/Core/Src/syscalls.c PROPERTIES HEADER_FILE_ONLY TRUE)

