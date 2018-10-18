module Server

open System.IO
open System.Collections.Concurrent
open Saturn
open Giraffe

let cache = ConcurrentDictionary<string, obj>()

let endpointPipe = pipeline {
    plug fetchSession
    plug head
    plug requestId
}

let topRouter = router {
    // pipe_through headerPipe
    not_found_handler (setStatusCode 404 >=> text "404")

    get "/" (fun f c ->
        match cache.TryGetValue("default") with
        | true, value -> json value f c
        | false, _ -> json (box [||]) f c)

    // getf "/name/%s" helloWorldName
    // getf "/name/%s/%i" helloWorldNameAge
}

let app = application {
    pipe_through endpointPipe

    use_router topRouter
    url "http://0.0.0.0:8085/"
    memory_cache
    use_cors "CORS_policy" (fun b -> b.AllowAnyOrigin() |> ignore)
    use_static "static"
    use_gzip
}

let evalScript fsiSession path =
    Interpreter.evalScript fsiSession path
    |> Option.iter (fun v ->
        cache.AddOrUpdate("default", v, (fun _ _ -> v)) |> ignore)

[<EntryPoint>]
let main _ =
    let fsiSession = Interpreter.initSession()
    let scriptFile = Path.GetFullPath("../Bicycles.fsx")
    evalScript fsiSession scriptFile

    let watcher =
        new FileSystemWatcher(
            Path.GetDirectoryName(scriptFile),
            Filter=Path.GetFileName(scriptFile),
            EnableRaisingEvents=true)
    watcher.Changed.Add(fun _ ->
        printfn "Script changed, evaluating..."
        evalScript fsiSession scriptFile)

    run app
    0
