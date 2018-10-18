(*
The goal in this exercise is to create a model to predict
how many bicycles will be used in a day, given the
available information.

Data source: UC Irvine Machine Learning dataset, "bike sharing":
https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset

We will first explore the data, using the CSV type provider
and Google Charts to visualize it.

Then we will develop a regression model to predict usage,
using Math.NET and a bit of linear algebra, starting simple,
and progressively increasing power and complexity.
*)


(*
Step 1. Opening the data using the CSV Type Provider
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Type Providers make it extremely easy to access data in
various formats: point to a sample, generate a type based
on the sample, and start exploring using IntelliSense.
*)

// TODO: run the following code, step-by-step

// #load "../.paket/load/net46/main.group.fsx"
#I "../packages/FSharp.Data/lib/net45"
#I "../packages/MathNet.Numerics/lib/net461"
#I "../packages/MathNet.Numerics.FSharp/lib/net45"

#r "FSharp.Data.dll"
#r "MathNet.Numerics.dll"
#r "MathNet.Numerics.FSharp.dll"

open System
open System.IO
open FSharp.Data

// we create a type based on sample data
type Dataset = CsvProvider<"../data/day.csv">
type Datapoint = Dataset.Row

// we can now read data...
let dataset = Dataset.Load("../data/day.csv")
let data = dataset.Rows

let run() =
    data
    |> Seq.map (fun day -> dict ["instant", day.Instant; "cnt", day.Cnt * -50])
    |> Seq.toArray

(*
// ... which is statically typed
let firstObservation = data |> Seq.head
printfn "First date: %s" (firstObservation.Dteday.ToShortDateString())

// we can print out the file headers
match dataset.Headers with
| None -> printfn "No headers found"
| Some(headers) ->
    headers
    |> Seq.iter (fun header -> printfn "%s" header)

// ... or print the total number of users for 5 first days
data
|> Seq.take 5
|> Seq.iter (fun day -> printfn "Total users: %i" day.Cnt)


// TODO: explore the dataset

// what was the day with the most wind?

data
|> Seq.maxBy (fun day -> day.Windspeed)
|> fun r -> r.Season

// what is the average number of riders?

data
|> Seq.averageBy (fun day -> float day.Cnt)

// what is the average number of riders
// during holidays? On Sundays?

data
|> Seq.filter (fun day -> day.Holiday)
|> Seq.averageBy (fun day -> float day.Cnt)

data
|> Seq.filter (fun day -> day.Dteday.DayOfWeek = System.DayOfWeek.Sunday)
|> Seq.averageBy (fun day -> float day.Cnt)

// or, fancier...
data
|> Seq.groupBy (fun day -> day.Dteday.DayOfWeek)
|> Seq.map (fun (day,group) ->
    day, group |> Seq.averageBy(fun day -> float day.Cnt))
|> Seq.toArray
*)

(*
Step 2. Visually exploring the data using Google Charts
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

It is usually easier to understand a dataset by looking at
it with charts. Let's use Google Charts to visualize our data.
*)


// // TODO: run the following code, step-by-step


// // Creating basic charts with Google Charts

// Chart.Column [ for i in 1 .. 10 -> i, 10 * i ] |> Chart.Show

// Chart.Bar [ for i in 1 .. 10 -> i, 10 * i ] |> Chart.Show

// Chart.Line [ for i in 1 .. 10 -> i, 10 * i ] |> Chart.Show

// Chart.Scatter [ (1,1); (2,3); (3,8)] |> Chart.Show

// Chart.Line [ "Monday", 1; "Tuesday", 2; "Wednesday", 3 ] |> Chart.Show



// // Using additional features

// Chart.Line [ "Monday", 1; "Tuesday", 2; "Wednesday", 3 ]
// |> Chart.WithTitle("Value by day")
// |> Chart.Show

// // Combining charts

// Chart.Line [
//     [ for x in 1 .. 10 -> x, x ]
//     [ for x in 3 .. 12 -> x, x + 5 ]
//     ]
// |> Chart.Show

// // TODO: plot the number of users, Cnt, over time.
// // Is the curve flat? Is there a trend?

// data
// |> Seq.map (fun day -> day.Instant, day.Cnt)
// |> Chart.Line
// |> Chart.Show

// // Scatterplots (Chart.Scatter) are often helpful to see if
// // some features / variables are related.

// // TODO: plot windspeed against count, temperature against
// // count. Which one seems more informative?

// data
// |> Seq.map (fun day -> day.Windspeed, day.Cnt)
// |> Chart.Scatter
// |> Chart.Show

// data
// |> Seq.map (fun day -> day.Temp, day.Cnt)
// |> Chart.Scatter
// |> Chart.Show

// (*
// Step 3. Defining a Regression Model to predict Cnt
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// We want to create a model that, given data for a day, will
// return a number, the predicted total number of users for
// that day. This is called a Regression Model.
// *)

// // TODO: run the following code, step-by-step


// // Model structure: take a day, predict the number of users
// type Model = Datapoint -> float

// // Our first model will be a straight line through the data
// // The parameter theta is called the 'coefficient'.
// let prediction (theta:float) (day:Datapoint) =
//     theta * (float day.Instant)

// // compare the results of 3 models, using 3 different
// // values for Theta (1.0, 5.0, 20.0), against the 'real'
// // value we are trying to predict.
// Chart.Line [
//     (data |> Seq.map (fun day -> day.Instant, float day.Cnt))
//     (data |> Seq.map (fun day -> day.Instant, prediction 1.0 day))
//     (data |> Seq.map (fun day -> day.Instant, prediction 5.0 day))
//     (data |> Seq.map (fun day -> day.Instant, prediction 20.0 day))
//     ]
// |> Chart.Show

// (*
// Step 4. What is a 'good model'?
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// What is a good model? A model where for each point, the
// error (the difference between the correct and the predicted
// values) is small.
// *)


// // TODO: run the following code, step-by-step

// // There are many possible ways to measure error; for
// // instance, the absolute difference between the 2.

// let absError (model:Model) (day:Datapoint) =
//     abs (model day - (float day.Cnt))

// let meanAbsError (model:Model) (data:Datapoint seq) =
//     data |> Seq.averageBy (absError model)


// // TODO: for theta = 0.0, 1.0, .. 20.0, compute the
// // model error. What is the best value of Theta?

// [ 0.0 .. 20.0 ]
// |> Seq.map (fun theta -> theta, prediction theta)
// |> Seq.map (fun (theta,model) -> theta, meanAbsError model data)
// |> Chart.Line
// |> Chart.Show

// (*
// Step 5. More complex models
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// So far we have used only one coefficient. We can build more
// complex models, by adding coefficients, for instance:
// Prediction = Theta0 + Theta1 * day.Instant + Theta2 * day.Temp
// *)


// // TODO: run the following code, step-by-step

// let complexPrediction (theta0:float,theta1:float,theta2:float) (day:Datapoint) =
//     theta0 + theta1 * (float day.Instant) + theta2 * (float day.Temp)

// // now we have 3 coefficients, instead of just one
// // note that complexModel still has a signature of
// // Datapoint -> float: it is still a valid model
// let complexModel = complexPrediction (500.0, 7.5, 2500.0)

// let complexModelError = meanAbsError complexModel data

// Chart.Line [
//     (data |> Seq.map (fun day -> day.Instant, float day.Cnt))
//     (data |> Seq.map (fun day -> day.Instant, complexModel day))
//     ]
// |> Chart.Show


// (*
// Step 6. Using Math.NET & Algebra to find the best Theta
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// WARNING: MATH ALERT. The following section contains a bit
// of algebra. If you don't follow everything, that's OK: we
// are just using algebra to create an algorithm that will
// automatically find the best model. In Section 7, we will
// use our algorithm, showing a practical usage.

// Manually searching for the best value of Theta is painful.
// Fortunately, using Linear Algebra, we can find in one shot
// the best solution for Theta, using what is called the
// 'normal form'.
// *)

// // TODO: run the following code, step-by-step


// open MathNet
// open MathNet.Numerics.LinearAlgebra
// open MathNet.Numerics.LinearAlgebra.Double

// type Vec = Vector<float>
// type Mat = Matrix<float>

// // instead of defining explicitly coefficients
// // (theta0,theta1,theta2, ...), we are going to
// // transform our model: we use a vector Theta to
// // store all these values, and we will transform
// // datapoints into another vector, so that our
// // prediction becomes simply the product of 2 vectors:
// let predict (theta:Vec) (v:Vec) = theta * v

// // using Normal Form to estimate a model.
// // For more information, see:
// // https://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)
// let estimate (Y:Vec) (X:Mat) =
//     (X.Transpose() * X).Inverse() * X.Transpose() * Y

// // We create a 'Featurizer', a function that will turn
// // a Datapoint into a Vector:
// type Featurizer = Datapoint -> float list

// let predictor (f:Featurizer) (theta:Vec) =
//     f >> vector >> (*) theta

// let evaluate = meanAbsError

// // Given a Featurizer (what data we want to use
// // from the Datapoints) and data, this will
// // return the 'best' coefficients, as well as a
// // function that predicts a value, given a Datapoint:
// let model (f:Featurizer) (data:Datapoint seq) =
//     let Yt, Xt =
//         data
//         |> Seq.toList
//         |> List.map (fun obs -> float obs.Cnt, f obs)
//         |> List.unzip
//     let theta = estimate (vector Yt) (matrix Xt)
//     let predict = predictor f theta
//     theta,predict



// (*
// Step 7. Using our algorithm to create better models
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// Now that we have an algorithm in place, let's use it on our
// dataset, to see if we can produce better prediction models!
// *)

// // illustration: show no constant, add a constant

// // we separated the dataset into 2 parts, one for training,
// // one for testing the quality of our model.
// let train = Dataset.Load("../../data/train.csv")
// let test = Dataset.Load("../../data/test.csv")

// // for our first model, we will use a constant (1.0), and
// // day.Instant as features.
// let featurizer0 (obs:Datapoint) =
//     [   1.0;
//         float obs.Instant; ]

// let (theta0,model0) = model featurizer0 train.Rows

// Chart.Line [
//     [ for obs in data -> obs.Instant, float obs.Cnt ]
//     [ for obs in data -> obs.Instant, model0 obs ] ]
// |> Chart.Show

// // we can now compare how good the model does,
// // both on train and test sets. A good model should
// // have similar performance on both.
// evaluate model0 train.Rows |> printfn "Training set: %.0f"
// evaluate model0 test.Rows |> printfn "Testing set: %.0f"

// Chart.Scatter [for obs in data -> float obs.Cnt, model0 obs ]
// |> Chart.Show

// theta0 |> Seq.iteri (fun i x -> printfn "Coeff. %i: %.1f" i x)


// // TODO: add temperature to the features
// // is this better? worse?

// let featurizer1 (obs:Datapoint) =
//     [   1.0
//         float obs.Instant
//         float obs.Temp
//     ]

// let (theta1,model1) = model featurizer1 train.Rows

// evaluate model1 test.Rows |> printfn "Testing set: %.0f"

// // TODO: visualize the result

// Chart.Line [
//     [ for obs in data -> obs.Instant, float obs.Cnt ]
//     [ for obs in data -> obs.Instant, model1 obs ] ]
// |> Chart.Show

// Chart.Scatter [for obs in data -> float obs.Cnt, model1 obs ]
// |> Chart.Show

// // TODO: try out some others, eg windspeed



// (*
// Step 8. Using discrete features
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// So far we have used numeric values (ex: temperature, wind)
// as model input. Let's see if we can use information from
// values that are not numbers, such as "is it a holiday"?
// *)


// let featurizer2 (obs:Datapoint) =
//     [   1.0
//         float obs.Instant
//         float obs.Temp
//         (if obs.Holiday then 1.0 else 0.0)  ]

// let (theta2,model2) = model featurizer2 train.Rows

// evaluate model2 test.Rows |> printfn "Testing set: %.0f"

// Chart.Scatter [for obs in data -> float obs.Cnt, model2 obs ]
// |> Chart.Show

// // Coeff. 3 shows you the impact of holidays:
// // on average, we lose 760 users.
// theta2 |> Seq.iteri (fun i x -> printfn "Coeff. %i: %.1f" i x)


// // TODO: are Mondays bigger days than Saturdays?





// (*
// Bonus section: even more features!
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// At that point, we pretty much threw everything we had into
// our model - is that all we can do? Well, no: we can do
// potentially more, by combining features into new ones.

// *)

// // If you look at the temperature vs. users chart, it
// // looks like very low or very high temperatures result in
// // lower usage:

// data
// |> Seq.map (fun day -> day.Temp, day.Cnt)
// |> Chart.Scatter
// |> Chart.Show

// // instead of modelling usage = theta0 + theta1 x temp,
// // you could do instead
// // usage =  theta0 + theta1 x temp + theta2 x (temp x temp).


// // TODO: include square of the temperature in the model

// let featurizer3 (obs:Datapoint) =
//     [   1.0
//         float obs.Instant
//         float obs.Temp
//         float (obs.Temp * obs.Temp)
//         (if obs.Holiday then 1.0 else 0.0)  ]

// let (theta3,model3) = model featurizer3 train.Rows

// evaluate model3 test.Rows |> printfn "Testing set: %.0f"
// evaluate model3 train.Rows |> printfn "Testing set: %.0f"

// Chart.Scatter [for obs in data -> float obs.Cnt, model3 obs ]
// |> Chart.Show

// // TODO: what happens as you add more features?
// // Is the quality improving the same way on train and test?


// (*
// Bonus section: accelerate computations with MKL
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// One of the beauties of algebra is, a lot of work has gone
// into it to make it run fast. Math.NET supports MKL, which
// runs algebra on a dedicated piece of hardware on the CPU.

// To use it, install the corresponding NuGet package,
// configure Math.NET to use MKL, and compare speed:

// System.Environment.CurrentDirectory <- __SOURCE_DIRECTORY__
// open MathNet.Numerics
// open MathNet.Numerics.Providers.LinearAlgebra.Mkl
// Control.LinearAlgebraProvider <- MklLinearAlgebraProvider()
// *)
