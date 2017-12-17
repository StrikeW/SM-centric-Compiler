set(LLVM_LINK_COMPONENTS support)

add_clang_executable(sm-centric
  sm-centric.cpp
)
    
target_link_libraries(sm-centric
				clangTooling
     		clangBasic
				clangFrontend
				clangLex
				clangAST
				clangDriver
)
