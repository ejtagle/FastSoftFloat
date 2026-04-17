SET(CMAKE_SYSTEM_NAME Generic)
SET(VISUALGDB_TOOLCHAIN_TYPE Embedded)
SET(VISUALGDB_TOOLCHAIN_SUBTYPE GCC)
if(DEFINED CCACHE_EXE)
SET(CMAKE_C_COMPILER "${CCACHE_EXE}" "$ENV{TOOLCHAIN_ROOT}/bin/arm-none-eabi-gcc.exe")
SET(CMAKE_CXX_COMPILER "${CCACHE_EXE}" "$ENV{TOOLCHAIN_ROOT}/bin/arm-none-eabi-g++.exe")
SET(CMAKE_ASM_COMPILER "${CCACHE_EXE}" "$ENV{TOOLCHAIN_ROOT}/bin/arm-none-eabi-g++.exe")
SET(CMAKE_AR "${TOOLCHAIN_ROOT}/bin/arm-none-eabi-ar.exe")
SET(CMAKE_OBJCOPY "${TOOLCHAIN_ROOT}/bin/arm-none-eabi-objcopy.exe")
else()
SET(CMAKE_C_COMPILER "$ENV{TOOLCHAIN_ROOT}/bin/arm-none-eabi-gcc.exe")
SET(CMAKE_CXX_COMPILER "$ENV{TOOLCHAIN_ROOT}/bin/arm-none-eabi-g++.exe")
SET(CMAKE_ASM_COMPILER "$ENV{TOOLCHAIN_ROOT}/bin/arm-none-eabi-g++.exe")
endif()
SET(CMAKE_LD "${TOOLCHAIN_ROOT}/bin/arm-none-eabi-ld.exe")
SET(CMAKE_IMPORT_LIBRARY_SUFFIX "-implib.o") #Used for CMSE import libraries

if(EXISTS "$ENV{TOOLCHAIN_ROOT}/Qt/v5-CMake/Qt5Cross.cmake")
	include("$ENV{TOOLCHAIN_ROOT}/Qt/v5-CMake/Qt5Cross.cmake")
endif()

if(EXISTS "$ENV{TOOLCHAIN_ROOT}/Qt/v6-CMake/Qt6Cross.cmake")
	include("$ENV{TOOLCHAIN_ROOT}/Qt/v6-CMake/Qt6Cross.cmake")
endif()

#Barebone toolchain cannot link executables without a device-specific linker script
SET(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
SET(CMAKE_OBJECT_PATH_MAX 240)
set(KNOWN_PATH_PREFIX_VARIABLES "" CACHE STRING "Specifies the list of variables like 'BSP_ROOT' that will be substituted by the IDE when referencing source files." FORCE)

function (visualgdb_toolchain_load_default_configuration)
    if(NOT DEFINED BSP_CONFIGURATION_com.sysprogs.toolchainoptions.arm.libctype)
        set(BSP_CONFIGURATION_com.sysprogs.toolchainoptions.arm.libctype "--specs=nano.specs")
        set(BSP_CONFIGURATION_com.sysprogs.toolchainoptions.arm.libctype "${BSP_CONFIGURATION_com.sysprogs.toolchainoptions.arm.libctype}" PARENT_SCOPE)
    endif()

    if(NOT DEFINED BSP_CONFIGURATION_com.sysprogs.toolchainoptions.arm.syscallspecs)
        set(BSP_CONFIGURATION_com.sysprogs.toolchainoptions.arm.syscallspecs "--specs=nosys.specs")
        set(BSP_CONFIGURATION_com.sysprogs.toolchainoptions.arm.syscallspecs "${BSP_CONFIGURATION_com.sysprogs.toolchainoptions.arm.syscallspecs}" PARENT_SCOPE)
    endif()

endfunction() #visualgdb_toolchain_load_default_configuration

function (visualgdb_toolchain_compute_flags)
    set(_includes "${_includes}" PARENT_SCOPE)
    set(_defines "${_defines}" PARENT_SCOPE)
    set(_cflags "${_cflags}" PARENT_SCOPE)
    set(_ldflags ${_ldflags} "${BSP_CONFIGURATION_com.sysprogs.toolchainoptions.arm.libctype}" "${BSP_CONFIGURATION_com.sysprogs.toolchainoptions.arm.syscallspecs}" "-Wl,--no-warn-rwx-segments")
    set(_ldflags "${_ldflags}" PARENT_SCOPE)
endfunction() #visualgdb_toolchain_compute_flags



set (VISUALGDB_COMPONENT_LISTS_DIR "C:/Users/ejtagle/AppData/Local/VisualGDB")
if (NOT DEFINED VISUALGDB_BUILD_SYSTEM_ROOT)
	set (VISUALGDB_BUILD_SYSTEM_ROOT "C:/Program Files (x86)/Sysprogs/VisualGDB/CMake/embedded")
endif()

include ("${VISUALGDB_BUILD_SYSTEM_ROOT}/root.cmake")
