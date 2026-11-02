
# Find Doxygen
find_package(Doxygen)

# Version hints are exported by Conan to keep tool versions traceable.
set(PYCANHA_OPTION_DOXYGEN_VERSION "1.15.0" CACHE STRING "Pinned Doxygen version hint")
set(
    PYCANHA_OPTION_DOXYGEN_AWESOME_CSS_VERSION
    "v2.5.0"
    CACHE STRING
    "Pinned doxygen-awesome-css version"
)

if(DOXYGEN_FOUND)
    if(NOT "${DOXYGEN_VERSION}" STREQUAL "${PYCANHA_OPTION_DOXYGEN_VERSION}")
        message(WARNING
            "Doxygen version mismatch: found ${DOXYGEN_VERSION}, "
            "pinned hint is ${PYCANHA_OPTION_DOXYGEN_VERSION}.")
    endif()
endif()

# The call to doxygen_add_docs generates the Doxyfile configuration file
# and configuration variables listed in Doxygen documentation such "CALL_GRAPH"
# are set-up to the value of the corresponding DOXYGEN_* variables.
# See: https://cmake.org/cmake/help/latest/module/FindDoxygen.html
#      https://www.doxygen.nl/manual/config.html


# Modern theme look with Doxygen Awesome CSS
# Check https://github.com/jothepro/doxygen-awesome-css
include(FetchContent)
FetchContent_Declare(
    doxygen-awesome-css
    GIT_REPOSITORY
    https://github.com/jothepro/doxygen-awesome-css.git
    GIT_TAG
    ${PYCANHA_OPTION_DOXYGEN_AWESOME_CSS_VERSION}
)    
FetchContent_MakeAvailable(doxygen-awesome-css)

set(DOXYGEN_HTML_EXTRA_STYLESHEET
${doxygen-awesome-css_SOURCE_DIR}/doxygen-awesome.css 
${doxygen-awesome-css_SOURCE_DIR}/doxygen-awesome-sidebar-only.css)

set(DOXYGEN_GENERATE_TREEVIEW  YES)
set(DOXYGEN_HTML_COLORSTYLE    LIGHT)
set(DOXYGEN_FULL_SIDEBAR       NO)
set(DOXYGEN_DISABLE_INDEX      NO)



set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_HTML_OUTPUT   ${PROJECT_BINARY_DIR}/html_docs)
set(DOXYGEN_HAVE_DOT YES)
set(DOXYGEN_DOT_IMAGE_FORMAT svg)
set(DOXYGEN_DOT_TRANSPARENT YES)
set(DOXYGEN_EXTRACT_PRIVATE YES)
set(DOXYGEN_CALL_GRAPH YES)



set(DOXYGEN_PROJECT_BRIEF "The C++ core of pycanha")
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE docs/introduction.md)


# Add a target to generate the Doxygen documentation
# In doxygen, files are processed in the order you put them in the list
# so I put them manually to set the order.
doxygen_add_docs(
    doc
    ${PROJECT_SOURCE_DIR}/README.md
    ${PROJECT_SOURCE_DIR}/docs/introduction.md
    ${PROJECT_SOURCE_DIR}/docs/index.md
    ${PROJECT_SOURCE_DIR}/docs/tools.md
    ${PROJECT_SOURCE_DIR}/docs/workflow.md
    ${PROJECT_SOURCE_DIR}/docs/cicd.md
    ${PROJECT_SOURCE_DIR}/docs/faq.md
    ${PROJECT_SOURCE_DIR}/${LIB_NAME}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Generate HTML documentation"
)

