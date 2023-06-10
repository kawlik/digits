import * as tf from "@tensorflow/tfjs";

// images properties
const IMAGE_X = 28;
const IMAGE_Y = 28;
const IMAGE_SIZE = IMAGE_X * IMAGE_Y;

// dataset properties
const NUM_CLASSESS = 10;
const NUM_ELEMENTS = 65000;

// dataset locations
const HREF_INPUTS = "/mnist.inputs.png";
const HREF_LABELS = "/mnist.labels.bin";

// export loader function
export async function loadMNIST(ratio = 0.8) {
	const [inputs, labels] = await Promise.all([loadInputs(), loadLabels()]);

	// clamp ratio
	if (ratio < 0) ratio = 0;
	if (ratio > 1) ratio = 1;

	// calc dataset sizes
	const numTrainElements = Math.floor(NUM_ELEMENTS * ratio);
	const numTestElements = NUM_ELEMENTS - numTrainElements;

	// shuffle datasets indeces
	const shuffledTrainIndices = tf.util.createShuffledIndices(numTrainElements);
	const shuffledTestIndices = tf.util.createShuffledIndices(numTestElements);

	// split datasets
	const trainInputs = inputs.slice(0, numTrainElements * IMAGE_SIZE);
	const trainLabels = labels.slice(0, numTrainElements * NUM_CLASSESS);
	const testInputs = inputs.slice(numTrainElements * IMAGE_SIZE);
	const testLabels = labels.slice(numTrainElements * NUM_CLASSESS);

	// controlers
	let shuffledTrainIndex = 0;
	let shuffledTestIndex = 0;

	return {
		nextTrainBatch(batchSize: number = 1) {
			if (batchSize < 1) batchSize = 1;

			return loadBatch(batchSize, [trainInputs, trainLabels], () => {
				shuffledTrainIndex += 1;
				shuffledTrainIndex %= shuffledTrainIndices.length;

				return shuffledTrainIndices[shuffledTrainIndex];
			});
		},
		nextTestBatch(batchSize: number = 1) {
			if (batchSize < 1) batchSize = 1;

			return loadBatch(batchSize, [testInputs, testLabels], () => {
				shuffledTestIndex += 1;
				shuffledTestIndex %= shuffledTestIndices.length;

				return shuffledTestIndices[shuffledTestIndex];
			});
		},
	};
}

// inputs loader
async function loadInputs(): Promise<Float32Array> {
	const cvs = document.createElement("canvas");
	const ctx = cvs.getContext("2d");
	const img = new Image();

	return new Promise((resolve, reject) => {
		if (!ctx) return reject();

		// process loaded image
		img.addEventListener("load", () => {
			const datasetBuff = new ArrayBuffer(NUM_ELEMENTS * IMAGE_SIZE * 4);
			const batchSize = 5000;

			// resize image
			img.width = img.naturalWidth;
			img.height = img.naturalHeight;

			// resize canvas
			cvs.width = img.naturalWidth;
			cvs.height = batchSize;

			// iterate over batches
			for (let i = 0; i < NUM_ELEMENTS; i += batchSize) {
				ctx.drawImage(img, 0, i, img.width, batchSize, 0, 0, cvs.width, cvs.height);

				// get image batched data
				const batchData = ctx.getImageData(0, 0, cvs.width, cvs.height);
				const batchBuff = new Float32Array(
					datasetBuff,
					IMAGE_SIZE * i * 4,
					IMAGE_SIZE * batchSize
				);

				// iterate over pixels
				for (let j = 0; j < batchData.data.length / 4; j++) {
					batchBuff[j] = batchData.data[j * 4] / 255;
				}
			}

			// resolve
			return resolve(new Float32Array(datasetBuff));
		});

		// load inputs image
		img.src = HREF_INPUTS;
	});
}

// labels loader
async function loadLabels(): Promise<Uint8Array> {
	const req = await fetch(HREF_LABELS);
	const res = new Uint8Array(await req.arrayBuffer());

	return res;
}

// dataset loaser
function loadBatch(
	batchSize: number,
	datasets: [Float32Array, Uint8Array],
	getIndex: () => number
) {
	const inputs = new Float32Array(batchSize * IMAGE_SIZE);
	const labels = new Uint8Array(batchSize * NUM_CLASSESS);

	for (let i = 0; i < batchSize; i++) {
		const index = getIndex();

		// slice data item
		const image = datasets[0].slice(index * IMAGE_SIZE, index * IMAGE_SIZE + IMAGE_SIZE);
		const label = datasets[1].slice(
			index * NUM_CLASSESS,
			index * NUM_CLASSESS + NUM_CLASSESS
		);

		// append data item
		inputs.set(image, i * IMAGE_SIZE);
		labels.set(label, i * NUM_CLASSESS);
	}

	return {
		inputs: tf.tensor2d(inputs, [batchSize, IMAGE_SIZE]),
		labels: tf.tensor2d(labels, [batchSize, NUM_CLASSESS]),
	};
}
