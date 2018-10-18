module Interpreter

open Microsoft.FSharp.Compiler.SourceCodeServices
open Microsoft.FSharp.Compiler.Interactive.Shell

open System
open System.IO
open System.Text

let initSession () =
    // Intialize output and input streams
    let sbOut = new StringBuilder()
    let sbErr = new StringBuilder()
    let inStream = new StringReader("")
    let outStream = new StringWriter(sbOut)
    let errStream = new StringWriter(sbErr)

    // Build command line arguments & start FSI session
    let argv = [| "/fsi.exe" |]
    let allArgs = Array.append argv [|"--noninteractive"|]

    let fsiConfig = FsiEvaluationSession.GetDefaultConfiguration()
    FsiEvaluationSession.Create(fsiConfig, allArgs, inStream, outStream, errStream)

let private fail loc (ex: exn) (errs: FSharpErrorInfo[]) =
    printfn "ERROR in %s: %s" loc ex.Message
    for r in errs do
        printfn "(%i,%i): (%i,%i) %A: %s" r.StartLineAlternate r.StartColumn r.EndLineAlternate r.EndColumn r.Severity r.Message
    None

let evalScript (fsiSession: FsiEvaluationSession) (path: string) =
    match fsiSession.EvalScriptNonThrowing path with
    | Choice1Of2 _, _ ->
        let types = fsiSession.DynamicAssembly.DefinedTypes
        // for t in Seq.rev types do
        //     t.GetMembers() |> Array.map (fun m -> m.Name)
        //     |> printfn "TYPE %s METHODS %A" t.FullName
        types |> Seq.rev |> Seq.tryPick (fun t ->
            match t.GetMethod("run") with
            | null -> None
            | m -> Some m)
        |> Option.map (fun meth ->
            let res = meth.Invoke(null, [||])
            printfn "Script evaluation complete"
            // printfn "RESULT %A" res
            res)
    | Choice2Of2 ex, errs ->
        fail "script" ex errs
