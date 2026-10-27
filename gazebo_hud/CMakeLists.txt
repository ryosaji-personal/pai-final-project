cmake_minimum_required(VERSION 3.10)

project(ObstacleHud)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC OFF)

find_package(Qt5 REQUIRED COMPONENTS
    Core
    Qml
    Quick
    QuickControls2
)

find_package(ignition-gui6 REQUIRED)
find_package(ignition-transport11 REQUIRED)
find_package(ignition-msgs8 REQUIRED)

qt5_add_resources(
    RESOURCES
    hud.qrc
)

# Qt Meta-Object Compiler
qt5_wrap_cpp(MOC_SOURCES
    ObstacleHud.hh
)

add_library(ObstacleHud SHARED
    ObstacleHud.cc
    ${MOC_SOURCES}
    ${RESOURCES}
)

target_link_libraries(ObstacleHud
    PRIVATE
    ignition-gui6::ignition-gui6
    ignition-transport11::ignition-transport11
    ignition-msgs8::ignition-msgs8
    Qt5::Core
    Qt5::Qml
    Qt5::Quick
    Qt5::QuickControls2
)