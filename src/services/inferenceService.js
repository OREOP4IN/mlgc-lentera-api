const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');

async function predictClassification(model, image) {
  try {
    const tensor = tf.node
      .decodeJpeg(image)
      .resizeNearestNeighbor([224, 224])
      .expandDims()
      .toFloat()
  
    const prediction = model.predict(tensor);
    const score = await prediction.data();
    const confidenceScore = Math.max(...score) * 100;
  
    // const classes = ['Melanocytic nevus', 'Squamous cell carcinoma', 'Vascular lesion'];
  
    // const classResult = tf.argMax(prediction, 1).dataSync()[0];
    // const label = classes[classResult];

    // if (label === 'Squamous cell carcinoma') {
    //   explanation = "Squamous cell carcinoma adalah jenis kanker kulit yang umum dijumpai. Penyakit ini sering tumbuh pada bagian-bagian tubuh yang sering terkena sinar UV."
    //   suggestion = "Segera konsultasi dengan dokter terdekat untuk meminimalisasi penyebaran kanker."
    // }
  
    // if (label === 'Vascular lesion') {
    //   explanation = "Vascular lesion adalah penyakit yang dikategorikan sebagai kanker atau tumor di mana penyakit ini sering muncul pada bagian kepala dan leher."
    //   suggestion = "Segera konsultasi dengan dokter terdekat untuk mengetahui detail terkait tingkat bahaya penyakit."
    // }
  
    // return { confidenceScore, label, explanation, suggestion };
  
    let result, suggestion;
  
    if (confidenceScore > 50) {
      result = "Cancer";
      suggestion = "Segera periksa ke dokter!";
    }else {
      result = "Non-cancer";
      suggestion = "Anda sehat!";
    }

    return { result, suggestion };
  } catch (error) {
    throw new InputError(`Terjadi kesalahan input: ${error.message}`);
  } 
}
 
module.exports = predictClassification;