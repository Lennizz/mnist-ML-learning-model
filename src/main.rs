// region: Libraries

// ML
use mnist::*;
use ndarray::prelude::*;
use ndarray_rand::{RandomExt};
use ndarray_rand::rand_distr::Uniform;

// CLI
use serde::{Serialize, Deserialize};
use clap::{Parser, Subcommand};
use core::f32;
use std::cmp::max;
use std::usize;
use std::{fs::File, io::{BufReader, BufWriter}, path::PathBuf};

use std::io::{self, Error, ErrorKind};
use std::io::prelude::*;
use std::time::Instant;

// Mutli-threading
use rayon::prelude::*;

// Random
use rand::seq::index::sample;


// endregion

// region: Helper functions

fn print_data(image_num: usize, data: &Array3<f32>, labels: &Array2<f32>){
    println!("Label: {}", labels[[image_num, 0]]);

    for y in 0..28 {
        for x in 0..28 {
            let pixel_opacity = data[[image_num, y, x]];
            
            let pixel = if pixel_opacity > 0.5 { "##" } else if pixel_opacity > 0.1 { ".." } else { "  " };
            print!("{}", pixel);
        }
        println!();
    }
}

fn load_mnist () -> (ArrayBase<ndarray::OwnedRepr<f32>, Dim<[usize; 3]>>, Array2<f32>, ArrayBase<ndarray::OwnedRepr<f32>, Dim<[usize; 3]>>, Array2<f32>) {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let image_num = 0;
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data: ArrayBase<ndarray::OwnedRepr<f32>, Dim<[usize; 3]>, f32> = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.0);
    println!("{:#.1?}\n",train_data.slice(s![image_num, .., ..]));

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);
    println!("The first digit is a {:?}",train_labels.slice(s![image_num, ..]) );

    let test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);

    let test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);

    (train_data, train_labels, test_data, test_labels)
}

fn sigmoid_function(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn _lable_to_array(lable: f32) -> Array1<f32>{
    let mut array = Array1::zeros(10);
    array[lable as usize] = 1.0;
    array
}

fn get_highest_value_array(array: &Array1<f32>) -> (f32, usize){
    let mut max = (f32::MIN, 0);
    for (i, element) in array.iter().enumerate(){
        if *element > max.0 {
            max = (*element, i);
        }
    }

    max
}

fn get_sample_data(data: &Array3<f32>, lables: &Array2<f32>, iterations: usize) -> (Array3<f32>, Array2<f32>) {
    let mut rng = rand::rng();
    
    let data_length = data.len_of(Axis(0));
    let divisior = max(
        250 / max((iterations) / 20000, 1), 
        1);

    let sample_size = data_length / divisior;

    let samples = sample(&mut rng, data_length, sample_size).into_vec();
    
    (data.select(Axis(0), &samples), lables.select(Axis(0), &samples))
}

fn softmax(mut result: Array1<f32>) -> Array1<f32>{
    let max_value = get_highest_value_array(&result).0;
    result.mapv_inplace(|element| element - max_value);

    result.mapv_inplace(|element| element.exp());
    let result_sum = result.sum();

    result.mapv_inplace(|element| element / result_sum);

    result
}

fn randomize_evolution(from_amount: usize, to_amount: usize, randomness: f32) -> (Array2<f32>, Array1<f32>){
    let probability_threshold: f32 = 0.05;
    
    let mut random_weight_change = Array::random(
        (to_amount, from_amount), 
        Uniform::new(-randomness, randomness).unwrap()
    );

    let probabilities_weight = Array::random(
        (to_amount, from_amount), 
        Uniform::new(0.0, 1.0).unwrap()
    );

    for row in 0..to_amount {
        for col in 0..from_amount {
            if probabilities_weight[[row, col]] >= probability_threshold {
                random_weight_change[[row, col]] = 0.0;
            }
        }
    }

    let mut random_bias_change = Array::random(
        to_amount, 
        Uniform::new(-randomness, randomness).unwrap()
    );

    let probabilities_bias = Array::random(
        to_amount, 
        Uniform::new(0.0, 1.0).unwrap()
    );

    for i in 0..to_amount {
        if probabilities_bias[i] >= probability_threshold {
            random_bias_change[i] = 0.0;
        }
    }

    (random_weight_change, random_bias_change)
}

// endregion

// region: main

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Create { path, layers } => {
            let brain = NerualNetwork::new(path.clone(), layers.clone());
            match brain.save(){
                Ok(_) => {},
                Err(e) => eprintln!("CREATION OF ML FAILED: {}", e),  
            }
        }

        Commands::Start { path } => {
            let mut brain = match NerualNetwork::load(path.clone()) {
                Ok(brain) => brain,
                Err(e) => {
                    eprintln!("LOADING OF ML FAILED: {}", e);
                    return;
                },  
            };

            ml_function_loop(&mut brain);
        }
    }
    
}

// endregion

// region: CLI system

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Start {
        #[arg(short = 'p', long)]
        path: PathBuf,
    },
    Create {
        #[arg(short = 'p', long)]
        path: PathBuf,

        #[arg(short = 'l', long)]
        layers: Vec<usize>,
    }
}

// endregion

// region: Machine Learning functions

#[derive(Serialize, Deserialize, Debug, Clone)]
struct NerualNetwork{
    path: PathBuf,
    iteration: u32,
    layers: Vec<Layer>,
} 

impl NerualNetwork {
    fn new(path: PathBuf, layer_data: Vec<usize>) -> NerualNetwork{
        let mut nodes: Vec<usize> = vec![784];
        nodes.append(&mut layer_data.clone());
        nodes.push(10);

        let mut layers: Vec<Layer> = vec![];
        for values in nodes.windows(2){
            layers.push(Layer::new(values[0], values[1]));
        }

        NerualNetwork { 
            path, 
            iteration: 0, 
            layers
        }
    }

    fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(&self.path)?;
        let writer = BufWriter::new(file);
        
        serde_json::to_writer_pretty(writer, &self)?;
        
        println!("Saved network to {:?}", self.path);
        Ok(())
    }

    fn load(path: PathBuf) -> Result<NerualNetwork, Box<dyn std::error::Error>> {
        let file = File::open(&path)?;
        let reader = BufReader::new(file);

        let mut network: NerualNetwork = serde_json::from_reader(reader)?;
        network.path = path;
        
        Ok(network)
    }

    fn train (&mut self, data: &ArrayBase<ndarray::OwnedRepr<f32>, Dim<[usize; 3]>>, labels: &Array2<f32>, epocs: usize, offspring_count: usize) -> Result<(), Box<dyn std::error::Error>> {
        let data_length = data.len_of(Axis(0));
        let labels_length = labels.len();
        
        if data_length != labels_length{
            return Err(Box::new(Error::new(
                ErrorKind::QuotaExceeded, 
                format!("Training data and training labels are out of sync, sizes are not the same: {} - {}", data_length, labels_length)
            )));
        }

        for current_epoc in 0..epocs{
            let start_rate = 0.5;
            let decay_speed = 0.001;
            let current_randomness = start_rate / (1.0 + (decay_speed * self.iteration as f32));
            
            let mut offspring: Vec<Vec<Layer>> = vec![];

            offspring.push(self.layers.clone());
            
            for _ in 0..offspring_count {
                offspring.push(self.create_offspring(current_randomness));
            }
    
            let (sampled_data, sample_lables) = get_sample_data(data, labels, self.iteration as usize);
            let sample_data_length = sampled_data.len_of(Axis(0));
            
            let mut strongest: (f32, usize) = (f32::MAX, 0);
            
            let results: Vec<(f32, usize)> = offspring.par_iter().enumerate().map(|(child_index, child)| {
                let mut total_cost = 0.0;
                
                for i in 0..sample_data_length{
                    let training_image = sampled_data.index_axis(Axis(0), i);
                    let expected_value = sample_lables[[i, 0]];
    
                    let cost = NerualNetwork::evaluate_data(training_image.flatten().view(), &child, expected_value);               
                    
                    total_cost += cost as f32;
                }
                
                let average_cost = total_cost / sample_data_length as f32;
    
                (average_cost, child_index)
            }).collect();

            for offspring_result in results{
                if offspring_result.0 < strongest.0 {
                    strongest = offspring_result
                } 
            }
            
            self.layers = offspring[strongest.1].clone();
            
            self.iteration += 1;

            if current_epoc % 100 == 0{
                println!("Progress: {} -- {}", current_epoc, epocs);
            }
        }
        
        Ok(())
    }

    fn create_offspring(&self, randomness: f32) -> Vec<Layer> {
        let mut evolved_layers: Vec<Layer>= vec![];
        
        for layer in self.layers.clone(){

            let (
                random_weight_change, 
                random_bias_change
            ) = randomize_evolution(layer.from_amount, layer.node_amount, randomness);

            let new_weights = layer.weights + random_weight_change;
            let new_bias = layer.bias + random_bias_change;

            let evolved_layer = Layer {
                from_amount: layer.from_amount,
                node_amount: layer.node_amount,
                weights: new_weights,
                bias: new_bias,
            };

            evolved_layers.push(evolved_layer);
        }

        evolved_layers
    }

    fn evaluate_data(in_data: ArrayView1<f32>, layers: &Vec<Layer>, expected_value: f32) -> f32{
        let (output_layer, hidden_layers) = layers.split_last().unwrap();

        if hidden_layers.is_empty(){
            let current_node_value = output_layer.weights.dot(&in_data) + &output_layer.bias;
            let result = softmax(current_node_value);
            let correctness = result[expected_value as usize] + 1e-7_f32;
            
            let cost = -(correctness.ln());
            return cost;
        }

        let first_layer = &hidden_layers[0];

        let mut current_node_value: Array1<f32> = first_layer.weights.dot(&in_data) + &first_layer.bias;
        current_node_value.mapv_inplace(|value| sigmoid_function(value));

        for layer in &hidden_layers[1..]{
            current_node_value = layer.weights.dot(&current_node_value) + &layer.bias;
            current_node_value.mapv_inplace(|value| sigmoid_function(value));
        }

        current_node_value = output_layer.weights.dot(&current_node_value) + &output_layer.bias;

        let result = softmax(current_node_value);
        let correctness = result[expected_value as usize] + 1e-7_f32;
        
        let cost = -(correctness.ln());
        cost
    }

    fn predict_data_set(&self, data: &Array3<f32>, labels: &Array2<f32>) {
        let data_length = data.len_of(Axis(0));
        let mut correct_guesses = 0;

        for i in 0..data_length{
            let image = data.index_axis(Axis(0), i);
            let expected_value = labels[[i, 0]] as usize;

            let predicted_value = self.predict(image.flatten().view());

            if predicted_value == expected_value {
                correct_guesses += 1;
            }
        }

        let accuracy = (correct_guesses as f32 / data_length as f32) * 100.0; 
        println!("Test resulted in an accuracy of: '{:.2}%' - {} out of {}", accuracy, correct_guesses, data_length);
    }

    fn predict_showcase(&self, data: &Array3<f32>, labels: &Array2<f32>) {
        let data_length = data.len_of(Axis(0));
        let index: usize = rand::random_range(0..data_length);

        let image = data.index_axis(Axis(0), index);
        let actual_label = labels[[index, 0]];

        let predicted_value = self.predict(image.flatten().view());

        println!("\n==============================================================");
        println!("                   MNIST PREDICTION SHOWCASE                  ");
        println!("==============================================================\n");

        print_data(index, &data, &labels);

        println!("\n  Neural Network Guess: >> {} <<\n", predicted_value);

        if predicted_value == actual_label as usize {
            println!("  Result: SUCCESS! ✅");
        } else {
            println!("  Result: FAILED! ❌");
        }
        println!("==============================================================\n");
    }

    fn predict(&self, image: ArrayView1<f32>) -> usize{
        let layers = &self.layers;

        let (output_layer, hidden_layers) = layers.split_last().unwrap();

        if hidden_layers.is_empty(){
            let current_node_value = output_layer.weights.dot(&image) + &output_layer.bias;
            return get_highest_value_array(&current_node_value).1;
        }

        let first_layer = &hidden_layers[0];

        let mut current_node_value: Array1<f32> = first_layer.weights.dot(&image) + &first_layer.bias;
        current_node_value.mapv_inplace(|value| sigmoid_function(value));

        for layer in &hidden_layers[1..]{
            current_node_value = layer.weights.dot(&current_node_value) + &layer.bias;
            current_node_value.mapv_inplace(|value| sigmoid_function(value));
        }

        current_node_value = output_layer.weights.dot(&current_node_value) + &output_layer.bias;

        get_highest_value_array(&current_node_value).1
    }
}


#[derive(Serialize, Deserialize, Debug, Clone)]
struct Layer {
    from_amount: usize,
    node_amount: usize,
    weights: Array2<f32>,
    bias: Array1<f32>
}

impl Layer {
    fn new(from_amount: usize, node_amount: usize) -> Layer{
        Layer { 
            from_amount, 
            node_amount, 
            weights: Array::random(
                (node_amount, from_amount), 
                Uniform::new(-1.0 as f32, 1.0 as f32).unwrap()
            ),
            bias: Array1::zeros(node_amount),
        }
    }
}

fn ml_function_loop (brain: &mut NerualNetwork) {
    let (train_data, 
        train_labels, 
        test_data, 
        test_labels) = load_mnist();

    println!("--------------------------------");
    println!("      ML Interactive Mode.");
    println!("Commands: \n  - train, repeat, test,\n  - test-showcase, save, exit");
    println!("--------------------------------");

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(_) => {
                let parts: Vec<&str> = input.trim().split_ascii_whitespace().collect();

                if parts.len() == 0{
                    continue;
                } else if parts.len() > 2 {
                    eprintln!("Error: to many arguments");
                    continue;
                }

                let command = *parts.get(0).unwrap();

                let input = match parts.get(1) {
                    Some(arg) => match arg.parse::<usize>() {
                        Ok(num) => num,
                        Err(_) => {
                            eprintln!("Error: Training epochs must be a number, got '{}'", arg);
                            continue;
                        }
                    },
                    None => 500,
                };

                match command {
                    "train" => {
                        let start = Instant::now();

                        match brain.train(&train_data, &train_labels, input, 100) {
                            Ok(_) => {},
                            Err(e) => {eprintln!("Error: training failed, reason '{}'", e)},
                        }

                        let duration = start.elapsed();
                        println!("Training took: '{:?}', did '{}' epocs", duration, input)
                    },
                    "repeat" => {
                        loop {
                            match brain.train(&train_data, &train_labels, 1000, 100) {
                                Ok(_) => {},
                                Err(e) => {eprintln!("Error: training failed, reason '{}'", e)},
                            }    

                            brain.predict_data_set(&test_data, &test_labels);

                            match brain.save() {
                                Ok(_) => {},
                                Err(e) => {eprintln!("Error: saving failed, reason '{}'", e)},
                            }                  
                        }
                    },
                    "test" => {
                        brain.predict_data_set(&test_data, &test_labels);
                    },
                    "test-showcase" => {
                        brain.predict_showcase(&test_data, &test_labels);
                    },
                    "save" => {
                        match brain.save() {
                            Ok(_) => {},
                            Err(e) => {eprintln!("Error: saving failed, reason '{}'", e)},
                        }
                    },
                    "exit" => {
                        return;
                    },
                    _ => {
                        eprintln!("Error: not valid command used, got '{}'", command);
                        continue;
                    }
                }
                
            }
            Err(e) => {
                eprintln!("Error: Reading of command failed: {}", e);
            }
        }
    }
}

// endregion