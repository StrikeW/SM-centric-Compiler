# CSC 512 Final Project #

Xiangqing Ding(xding3)

## Overview ##
In this project, a source-to-source translator that can transform CUDA code to an [SM-Centric](./smcPaper_ics2015.pdf) form is designed and implemented based on LLVM and Clang. In the **Build and Run** section, instructions of compiling and running the translator are provided. Then the details of how the translator is implemented is described in the **Implementation** section. In the end, some specifications about test cases will be described.

## Build and Run ##

### Assumptions ###

1. The translator is built and run under the same environment as the LLVM Virtual Machine provided. Otherwise, the instructions are not guaranteed to work. 
2. The original CUDA file to be translated should be with the suffix ".cu" to ensure the naming of generated file

### Build ###

1. Direct to the clang tool source folder (**/home/ubuntu/llvm/llvm/tools/clang/tools**)
2. Create a folder named **sm-centric**
3. Put the **SM\_centric\_transformation.cpp** (Source code) and **CMakeLists.txt** (Dependency file) under the **sm-centric** folder. Both files can be found in the **Source Code** folder within the submitted files
4. Modify the **CMakeLists.txt** in the clang tool folder. Append one line to the end of the file: `add_clang_subdirectory(sm-centric)`
5. Direct to the build-release folder (**/home/ubuntu/llvm/build-release**)
6. Run `ninja sm-centric`

Once the translator is built successfully, message like following will be displayed:

	ubuntu@ubuntu:~/llvm/build-release$ ninja sm-centric
	[2/2] Linking CXX executable bin/sm-centric

### Run ###

Run the translator on one file:

	cd ~/llvm/build-release
	bin/sm-centric path/filename.cu -- --cuda-host-only [options]

Run the translator on multiple files:
	
	cd ~/llvm/build-release
	bin/sm-centric path/filename.cu path2/filename2.cu ... -- --cuda-host-only [options]

After that, a SM-Centric form CUDA file will be generated under the the same folder with the original file. For instance, if the original file is named *test.cu*, the name of the generated file should be *test_smc.cu* and in the same path with *test.cu*

### Specification ###

1. The path of original files could be relative path or absolute path
2. When running the translator on test cases, dependency path should be specified in the command. However, without the dependency path, the error messages like `'helper_cuda.h' file not found` could be ignored, which may not influence the result of the translator.






## Implementation ##


### Main Tasks ###

1. **Adding smc.h Header**

When the program creates/opens the generated file, it will firstly put the include statement at the start of the file. Though the include statement is required to be put after other headers, it does not any difference for current test cases when it is put at the start of the file. 

	...
		// Add smc header
        outputFile << "#include \"smc.h\""
                   << "\n";
        // Put the other transformed code into stream
        outputFile << std::string(RewriteBuf->begin(), RewriteBuf->end());
	...

2. **Rewriting Kernel Function Definition**

This task can be divided several steps:

+ **Check whether a function is a kernel function**

In CUDA programming, the kernel function is marked with "\_\_global\_\_". To check whether current function is a kernel function, *hasAttr<CUDAGlobalAttr\>()* is called when a *FunctionDecl* node is visited. In addition, *hasBody()* method is called to check whether the function is defined. After that, more transformations can be made on the kernel functions. 

    bool VisitFunctionDecl(FunctionDecl *func)
    {
        // Check if kernal functions (function with __global__ )
        if (func->hasAttr<CUDAGlobalAttr>())
        {
            if (func->hasBody())
            {
				.....// More transformations
            }
        }
        return true;
    }

+ **Add the *\_\_SMC\_Begin* and the *\_\_SMC\_End* to the definition of the kernel function**

To add the *\_\_SMC\_Begin*, the start location of function definition should be found. This is achieved by calling *getLocStart()* on the function body, which is a *Stmt* object returned by *getBody()* method. After that, *\_\_SMC\_End* can also be added in the end of function definition in the same way.

				...
                // Add __SMC_Begin to the beginning of the definition of the kernel function
                std::stringstream kernal_begin;
                kernal_begin << "\n\t"
                             << "/*-------------|SMC Change: Add __SMC_Begin|-------------*/\n\t" // highlight
                             << "__SMC_Begin"
                             << "\n";
                st = funcBody->getLocStart(); // Get the start of the function definition
                SMC_Rewriter.InsertTextAfterToken(st, kernal_begin.str());
				...
                // Add __SMC_End to the end of the definition of the kernel function
                std::stringstream kernal_end;
                kernal_end << "\n\t"
                           << "/*-------------|SMC Change: Add __SMC_End|-------------*/\n\t" // highlight
                           << "__SMC_End"
                           << "\n";
                st = funcBody->getLocEnd(); // Get the end of the function definition
                SMC_Rewriter.InsertTextBefore(st, kernal_end.str());
				...
 

+ **Add parameters to the definition of the kernel function**

The next step is to add parameters to the kernel function. To locate the parameters of function, we firstly get the *TypeSourceInfo* of the function. And then the location after the last parameter can be retrieved, and new parameters will be appended.

				...
                // Add parameters to the end of the parameter list of the definition of the kernel function
                std::stringstream kernal_parameters;
                kernal_parameters << ",\n\t"
                                  << "dim3 __SMC_orgGridDim, "
                                  << "int __SMC_workersNeeded, "
                                  << "int *__SMC_workerCount, "
                                  << "int * __SMC_newChunkSeq, "
                                  << "int * __SMC_seqEnds";
                st = func->getTypeSourceInfo()->getTypeLoc().getEndLoc(); // Get the location after the last parameter
                SMC_Rewriter.InsertText(st, kernal_parameters.str(), false, true); 
				...

+ **Replacing the references of *blockIdx.x* and *blockIdx.y***

The last step of this task is to replace all the references of *blockIdx.x* and *blockIdx.y* within kernel function definition. It is achieved by implementing a recursive method called *rewriteMemberExpr(Stmt \*s)*. This function firstly tries to cast the input *Stmt* object into a *MemberExpr* object. If it succeeds, it means that the node is originally a *MemberExpr*. Then it checks whether this member expression is blockIdx.x or blockIdx.y. If they are, replace them with corresponding SMC expression.

    // A resurisve method that rewrites all member expressions in a statement
    void rewriteMemberExpr(Stmt *s)
    {
        if (MemberExpr *memberExpr = dyn_cast<MemberExpr>(s))
        {
            SourceRange sr;

            //  Get base of member expression
            if (OpaqueValueExpr *ovExpr = dyn_cast<OpaqueValueExpr>(memberExpr->getBase()))
            {
                // Get type of the base
                std::string baseType = ovExpr->getType().getAsString();
                // Check if it is blockIdx
                if (baseType == "const struct __cuda_builtin_blockIdx_t")
                {
                    // Get member of member expression
                    std::string memberName = memberExpr->getMemberDecl()->getNameAsString();

                    // Replace the references of blockIdx.x
                    if (memberName == "__fetch_builtin_x")
                    {
                        std::stringstream blockIdx_x;
                        blockIdx_x << "(int)fmodf((float)__SMC_chunkID, (float)__SMC_orgGridDim.x)"
                                   << "/*-------------|SMC Change: Replace blockIdx.x|-------------*/";
                        sr = memberExpr->getSourceRange(); // Get the range of the member expression
                        SMC_Rewriter.ReplaceText(sr, blockIdx_x.str());
                    }
                    // Replace the references of blockIdx.x
                    else if (memberName == "__fetch_builtin_y")
                    {
                        std::stringstream blockIdx_y;
                        blockIdx_y << "(int)(__SMC_chunkID/__SMC_orgGridDim.x)"
                                   << "/*-------------|SMC Change: Replace blockIdx.y|-------------*/";
                        sr = memberExpr->getSourceRange(); // Get the range of the member expression
                        SMC_Rewriter.ReplaceText(sr, blockIdx_y.str());
                    }
                }
            }
        }
		else
		...
	}

If the *Stmt* cannot be cast into the *MemberExpr* (not a *MemberExpr* node), the recursive method will be called on each child *Stmt* node.

		...
        else
        {
            for (Stmt::child_iterator ci = s->child_begin(), ce = s->child_end(); ci != ce; ++ci)
            {
                if (*ci)
                    rewriteMemberExpr(*ci);
            }
        }
		...

3. **Rewriting Kernel Function Call**

Due to the *CUDAKernelCallExpr* node from the Clang AST, it is much easier to locate the call for CUDA kernel functions by writing hooks function within the AST Visitor.

    // Visit all the CUDAKernelCallExpr nodes
    bool VisitCUDAKernelCallExpr(CUDAKernelCallExpr *cudaExpr)
    {
		...
        return true;
    }

After locating the call for CUDA kernel functions, the first step is inserting SMC initialization function before the call.

		...
        // Add __SMC_init() right before the call of the GPU kernel function
        std::stringstream smc_init;
        smc_init << "/*-------------|SMC Change: Add __SMC_init()|-------------*/\n" // highlight
                 << "__SMC_init();\n\n";
        st = cudaExpr->getLocStart(); // Get the location before the function call
        SMC_Rewriter.InsertText(st, smc_init.str(), false, true); 
		...

Then arguments are added to the function call. The location after the last arguments is also the end location of the call, we can simply call get *cudaExpr->getLocEnd()* to get the location.

		...
        // Append arguments to the end of the GPU kernel function call
        std::stringstream kernal_arguments;
        kernal_arguments << ", "
                         << "__SMC_orgGridDim, "
                         << "__SMC_workersNeeded, "
                         << "__SMC_workerCount, "
                         << "__SMC_newChunkSeq, "
                         << "__SMC_seqEnds";
        st = cudaExpr->getLocEnd(); // Get the location after the last argument
        SMC_Rewriter.InsertTextBefore(st, kernal_arguments.str());	
		...

4. **Rewriting grid() Declaration**

Rewriting grid declaration is the most difficult part in this project. The grid variable to be used in execution configuration is not necessarily named with "grid", for example:

	// This also works
	...
	dim3 gridx(...);
	...
	KernelCall<<<gridx, threads>>>(...);
	...

As a consequence, we need to retrieve the grid variable name from each call for CUDA kernel functions. And then we check each *VarDecl* node with the grid name list to see whether it needs rewriting. To solve this problem, a Preprocessor is implemented and run to put all grid variables name in a global string list. The structure of the Preprocessor is similar to the one of the Translator and within the same source file.
	
	...
    // Visit all the CUDAKernelCallExpr nodes
    bool VisitCUDAKernelCallExpr(CUDAKernelCallExpr *cudaExpr)
    {
        SourceLocation st;
        // Get configuration
        CallExpr *config = cudaExpr->getConfig();
        // Get grid configuration: the first argument
        if (CXXConstructExpr *gridConfig = dyn_cast<CXXConstructExpr>(config->getArg(0)))
        {
            if (ImplicitCastExpr *cast = dyn_cast<ImplicitCastExpr>(gridConfig->getArg(0)))
            {
                if (DeclRefExpr *grid = dyn_cast<DeclRefExpr>(cast->getSubExpr()))
                {
                    std::string gridName = grid->getNameInfo().getAsString();
                    // Put grid variable name into the global list
                    gridList.push_back(gridName);
                }
            }
        }

        return true;
    }
	...

After the Preprocessor finishes its work, the Translator could now check whether a *VarDecl* node is a declaration for the grid argument. If it is, it is replaced with the expression __SMC_orgGridDim while the arguments are kept unchanged. After that

	...
    // Visit all the VarDecl nodes
    bool VisitVarDecl(VarDecl *varDecl)
    {
        SourceLocation st;

        // Check whether it is grid initialization
        std::list<std::string>::iterator finder = std::find(gridList.begin(), gridList.end(), varDecl->getNameAsString());
        if (finder != gridList.end())
        {
            if (varDecl->hasInit())
            {
                // Replace the call of function grid(...) with dim3 __SMC_orgGridDim(...)
                std::stringstream smc_grid;
                smc_grid << "__SMC_orgGridDim";
                st = varDecl->getInit()->getLocStart(); // Get the location before initialization
                SMC_Rewriter.ReplaceText(st, finder->length(), smc_grid.str());

                // Add highlight to SMC change
                std::stringstream highlight;
                highlight << "/*-------------|SMC Change: Change Grid()|-------------*/\n";
                st = varDecl->getLocStart(); // Get the location after the last parameter
                SMC_Rewriter.InsertText(st, highlight.str(), false, true);
            }
        }
        return true;
    }
	...

### Extra Features ###

+ The translator could translate multiple files at the same time.
+ The translator can work on the original file containing multiple kernel functions
+ The name of grid argument of execution configuration is not limited to "grid". 
+ The grid in the original program could be one-dimension, two-dimension or three-dimension.
	




## Test Cases ##

To run the test cases: 

	cd ~/llvm/build-release
	bin/sm-centric path/test_1.cu ... -- --cuda-host-only

### Test Case 1 ###

The test case 1 is derived from the given file *matrixAdd_org.cu*. Some parts of expected results can be referred to the *matrixAdd_smc.cu*. Some modification are made to test whether the translator can handle different situations:

1. The *main()* function is deleted so that when the test case is run, there is no need to add dependency. 
2. In the definition of kernel function *matrixAddCUDA()*, statements below are added to test whether the translator can detect other forms of references of *blockIdx.x* and *blockIdx.y*. These two references are expected to be transformed. 

	...
    printf("Referrence of blockIdx.x: %d.\n", blockIdx.x); //line 43
    printf("Referrence of blockIdx.y: %d.\n", blockIdx.y); //line 44
	...

3. In the definition of function *matrixAdd()*, the name of grid argument and declaration are changed from "grid" to "newGrid" while the "dim3 grid(...)" is kept. This modification is aimed to test whether the translator can handle different name of grid variables. It is expected that "newGrid(...)" will be transformed while "grid(...)" keeps unchanged.
	
	...
	dim3 grid(w / threads.x, h / threads.y); //line 111
    dim3 newGrid(w / threads.x, h / threads.y); //line 112
	...
    matrixAddCUDA<<< newGrid, threads >>>(d_C, d_A, d_B, w, h); //line 117
	
4. In the definition of function *matrixAdd()*, another call for *matrixAddCUDA()* is added to test whether the translator can handle multiple kernel function calls. It is expected that this call will also be transformed.

	matrixAddCUDA<<< newGrid, threads >>>(d_C, d_A, d_B, w, h);// line 147



### Test Case 2 ###
 
The test case 2 is derived from the given file *matrixMul_org.cu*. Some parts of expected results can be referred to the *matrixMul_smc.cu*. Some modification are made to test whether the translator can handle different situations:

1. The main() function is deleted so that when the test case is run, there is no need to add dependency. 
2. In the definition of function *matrixMultiply()*, a new grid variable is declared and the execution configuration of one kernel call is changed to test whether the translator can handle different execution configurations. Also, it tests whether the grid variables can be one-dimension and three-dimension. It is expected all statements below will be transformed.

	...
    dim3 gridx(dimsB.x / threads.x); // line 202

    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y, 2); // line 204
	...

    ...
    if (block_size == 16)
    {
        matrixMulCUDA<16><<< gridx, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x); // line 212
    }
    else
    {
        matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x); // line 216
    }



### Test Case 3 ###

This test file is simply a combination of test case 1 and test case 2. It is designed for testing whether the translator can work on the original file containing multiple kernel functions (Extra Feature).


### Limitations ###

For current implementation, the only limitation is that the *#include "smc.h"* statement is added at the beginning of the generated file instead of after all existing #include statements. This doesn't make any difference to current test cases. However, it can be achieved in future work by extending PPCallBacks and overriding the IncludeDirective() method.


