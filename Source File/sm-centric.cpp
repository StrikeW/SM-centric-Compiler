// System library
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <list>
#include <algorithm>
// Clang library
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

// Namespaces
using namespace std;
using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;

// Global variable
std::list<std::string> gridList = {}; // The list storing all the names of variables holding grid configuraion

static llvm::cl::OptionCategory SMC("SMC");

// The AST Visitor for the prepocessor.
// The prepocessor finds all "grid" arguments and stores their name to the global list
class SMC_Preprocessor_Visitor : public RecursiveASTVisitor<SMC_Preprocessor_Visitor>
{
  public:
    SMC_Preprocessor_Visitor(Rewriter &R) : SMC_Rewriter(R) {}

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

  private:
    Rewriter &SMC_Rewriter;
};

// Recursive AST Visitor for main SMC translator
class SMC_Translator_Visitor : public RecursiveASTVisitor<SMC_Translator_Visitor>
{
  public:
    SMC_Translator_Visitor(Rewriter &R) : SMC_Rewriter(R) {}

    // Visit all the FunctionDecl nodes
    bool VisitFunctionDecl(FunctionDecl *func)
    {
        // Check if kernal functions (function with __global__ )
        if (func->hasAttr<CUDAGlobalAttr>())
        {
            if (func->hasBody())
            {
                // Get the body of function defination
                Stmt *funcBody = func->getBody();

                // Create a source location
                SourceLocation st;

                // Add __SMC_Begin to the beginning of the definition of the kernel function
                std::stringstream kernal_begin;
                kernal_begin << "\n\t"
                             << "/*-------------|SMC Change: Add __SMC_Begin|-------------*/\n\t" // highlight
                             << "__SMC_Begin"
                             << "\n";
                st = funcBody->getLocStart(); // Get the start of the function definition
                SMC_Rewriter.InsertTextAfterToken(st, kernal_begin.str());

                // Add __SMC_End to the end of the definition of the kernel function
                std::stringstream kernal_end;
                kernal_end << "\n\t"
                           << "/*-------------|SMC Change: Add __SMC_End|-------------*/\n\t" // highlight
                           << "__SMC_End"
                           << "\n";
                st = funcBody->getLocEnd(); // Get the end of the function definition
                SMC_Rewriter.InsertTextBefore(st, kernal_end.str());

                // Add parameters to the end of the parameter list of the definition of the kernel function
                std::stringstream kernal_parameters;
                kernal_parameters << ","
                                  << "dim3 __SMC_orgGridDim, "
                                  << "int __SMC_workersNeeded, "
                                  << "int *__SMC_workerCount, "
                                  << "int * __SMC_newChunkSeq, "
                                  << "int * __SMC_seqEnds";
                st = func->getTypeSourceInfo()->getTypeLoc().getEndLoc(); // Get the location after the last parameter
                SMC_Rewriter.InsertText(st, kernal_parameters.str(), false, true);

                // Add highlight to SMC change
                std::stringstream highlight;
                highlight << "/*-------------|SMC Change: Add parameters to kernel function|-------------*/\n";
                st = func->getLocStart(); // Get the location of the func
                SMC_Rewriter.InsertText(st, highlight.str(), true, true);

                // Rewrites all member expressions in the function body
                rewriteMemberExpr(funcBody);
            }
        }

        return true;
    }

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

    // Visit all the CUDAKernelCallExpr nodes
    bool VisitCUDAKernelCallExpr(CUDAKernelCallExpr *cudaExpr)
    {
        SourceLocation st;

        // Add __SMC_init() right before the call of the GPU kernel function
        std::stringstream smc_init;
        smc_init << "/*-------------|SMC Change: Add __SMC_init()|-------------*/\n" // highlight
                 << "__SMC_init();\n\n";
        st = cudaExpr->getLocStart(); // Get the location before the function call
        SMC_Rewriter.InsertText(st, smc_init.str(), false, true);

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

        // Add highlight to SMC change
        std::stringstream highlight;
        highlight << "/*-------------|SMC Change: Add arguments to kernel function call|-------------*/\n";
        st = cudaExpr->getLocStart(); // Get the location after the last parameter
        SMC_Rewriter.InsertText(st, highlight.str(), true, true);

        return true;
    }

  private:
    Rewriter &SMC_Rewriter;
    bool flag = true;

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
                    // Replace the references of blockIdx.y
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
        {
            for (Stmt::child_iterator ci = s->child_begin(), ce = s->child_end(); ci != ce; ++ci)
            {
                if (*ci)
                    rewriteMemberExpr(*ci);
            }
        }
    }
};

// Implementation of the ASTConsumer interface for reading an AST produced by the Clang parser.
class SMC_Preprocessor_ASTConsumer : public ASTConsumer
{
  public:
    SMC_Preprocessor_ASTConsumer(Rewriter &R) : Preprocessor(R) {}

    virtual void HandleTranslationUnit(ASTContext &Context)
    {
        // Set traversing start
        Preprocessor.TraverseDecl(Context.getTranslationUnitDecl());
    }

  private:
    SMC_Preprocessor_Visitor Preprocessor;
};

// Implementation of the ASTConsumer interface for reading an AST produced by the Clang parser.
class SMC_Translator_ASTConsumer : public ASTConsumer
{
  public:
    SMC_Translator_ASTConsumer(Rewriter &R) : Translator(R) {}

    virtual void HandleTranslationUnit(ASTContext &Context)
    {
        // Set traversing start
        Translator.TraverseDecl(Context.getTranslationUnitDecl());
    }

  private:
    SMC_Translator_Visitor Translator;
};

// For each source file provided to the tool, a new Preprocessr FrontendAction is created.
class SMC_Preprocessor_FrontendAction : public ASTFrontendAction
{
  public:
    SMC_Preprocessor_FrontendAction() {}

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override
    {
        rewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
        return llvm::make_unique<SMC_Preprocessor_ASTConsumer>(rewriter);
    }

  private:
    Rewriter rewriter;
};

// For each source file provided to the tool, a new Translator FrontendAction is created.
class SMC_Translator_FrontendAction : public ASTFrontendAction
{
  public:
    SMC_Translator_FrontendAction() {}

    // Callback at the end of processing a single input.
    void EndSourceFileAction() override
    {
        // Get rewrite buffer
        const RewriteBuffer *RewriteBuf = rewriter.getRewriteBufferFor(rewriter.getSourceMgr().getMainFileID());
        // Create output stream
        ofstream outputFile;
        // Get target file name (include path)
        filename = std::string(getCurrentFile());
        // Add "_smc" before ".cu"
        filename.insert(filename.length() - 3, "_smc");

        // Open/Create output file
        if (!filename.empty()) // If file name is already initialized
        {
            outputFile.open(filename);
        }
        else // If file name is not initialized
        {
            outputFile.open("output.cu");
        }
        // Add smc header
        outputFile << "/*-------------|SMC Change: Add smc.h header|-------------*/\n"
                   << "#include \"smc.h\""
                   << "\n";
        // Put the other transformed code into stream
        outputFile << std::string(RewriteBuf->begin(), RewriteBuf->end());

        // Close output stream
        outputFile.close();
    }

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override
    {
        rewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
        return llvm::make_unique<SMC_Translator_ASTConsumer>(rewriter);
    }

  private:
    Rewriter rewriter;
    std::string filename;
};

int main(int argc, const char **argv)
{
    // Check argument
    if (argc < 2)
    {
        llvm::errs() << "Usage: /sm-centric [options] <filename.cu>\n";
        return 1;
    }

    // Bind tool with arguments
    CommonOptionsParser op(argc, argv, SMC);
    ClangTool Tool(op.getCompilations(), op.getSourcePathList());

    // Run the preprocessor frontend action first to get the "grid" list
    Tool.run(newFrontendActionFactory<SMC_Preprocessor_FrontendAction>().get());

    // Run the main translator frontend action to transform CUDA
    return Tool.run(newFrontendActionFactory<SMC_Translator_FrontendAction>().get());
}
