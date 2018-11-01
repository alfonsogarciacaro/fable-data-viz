module Server

open System
open System.IO
open System.Collections.Concurrent
open Microsoft.Extensions.Configuration
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
    host_config (fun builder ->
        builder.ConfigureAppConfiguration(Action<_,IConfigurationBuilder>(fun ctx config ->
            config.AddJsonFile("appsettings.json") |> ignore
        ))
    )
}

let evalScript fsiSession path =
    Interpreter.evalScript fsiSession path
    |> Option.iter (fun v ->
        cache.AddOrUpdate("default", v, (fun _ _ -> v)) |> ignore)

[<EntryPoint>]
let main _ =
    let fsiSession = Interpreter.initSession()
    let watchingDir = Path.GetFullPath("..")
    printfn "Watching dir %s" watchingDir
    // printfn "Scripts found %A" <| Directory.GetFiles(watchingDir, "*.fsx")
    do
        Directory.GetFiles(watchingDir, "*.fsx")
        |> Seq.tryHead
        |> Option.iter (evalScript fsiSession)

    let watcher =
        new FileSystemWatcher(watchingDir,
                              Filter="*.fsx",
                              EnableRaisingEvents=true)
    watcher.Changed.Add(fun ev ->
        printfn "Script changed: %s" ev.FullPath
        evalScript fsiSession ev.FullPath)

    run app
    0
