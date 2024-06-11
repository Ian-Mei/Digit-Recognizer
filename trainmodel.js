// Import necessary libraries
import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';

// Function to load and preprocess data
async function loadData() {
  const trainData = fs.readFileSync('kaggle_datasets/train.csv', 'utf8');
  const testData = fs.readFileSync('kaggle_datasets/test.csv', 'utf8');

  // Parse CSV data
  const parseCSV = (data) => {
    return data.trim().split('\n').map(line => line.split(',').map(Number));
  };

  const trainArray = parseCSV(trainData);
  const testArray = parseCSV(testData);

  // Extract labels and features
  const train_X = trainArray.map(row => row.slice(1).map(value => value / 255.0));
  const train_Y = trainArray.map(row => row[0]);
  const test_X = testArray.map(row => row.map(value => value / 255.0));

  // Reshape data
  const train_X_reshaped = tf.tensor4d(train_X, [train_X.length, 28, 28, 1]);
  const test_X_reshaped = tf.tensor4d(test_X, [test_X.length, 28, 28, 1]);
  const train_Y_reshaped = tf.oneHot(tf.tensor1d(train_Y, 'int32'), 10);

  return { train_X_reshaped, train_Y_reshaped, test_X_reshaped };
}

// Function to define the model
function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.conv2d({ filters: 32, kernelSize: [5, 5], padding: 'same', activation: 'relu', inputShape: [28, 28, 1] }));
  model.add(tf.layers.conv2d({ filters: 32, kernelSize: [5, 5], padding: 'same', activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.dropout({ rate: 0.25 }));

  model.add(tf.layers.conv2d({ filters: 64, kernelSize: [3, 3], padding: 'same', activation: 'relu' }));
  model.add(tf.layers.conv2d({ filters: 64, kernelSize: [3, 3], padding: 'same', activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
  model.add(tf.layers.dropout({ rate: 0.25 }));

  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.5 }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

  model.compile({
    optimizer: tf.train.rmsprop(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

// Function to train the model
async function trainModel(model, train_X, train_Y) {
  const batchSize = 86;
  const epochs = 40;

  await model.fit(train_X, train_Y, {
    batchSize,
    epochs,
    validationSplit: 0.1,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1} / ${epochs}: loss=${logs.loss.toFixed(4)}, accuracy=${logs.acc.toFixed(4)}, val_loss=${logs.val_loss.toFixed(4)}, val_accuracy=${logs.val_acc.toFixed(4)}`);
      }
    }
  });
}

// Main function to load data, create model, and train
async function main() {
  const { train_X_reshaped, train_Y_reshaped } = await loadData();
  const model = createModel();
  await trainModel(model, train_X_reshaped, train_Y_reshaped);

  // Save the model
  await model.save('file://model');
}

// Execute main function
main();
