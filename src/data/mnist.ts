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
	if (ratio < 0.1) ratio = 0.1;
	if (ratio > 0.9) ratio = 0.9;

	// calc dataset sizes
	const numTrain = Math.round(NUM_ELEMENTS * ratio);
	const numTest = NUM_ELEMENTS - numTrain;
	const train = {
		inputs: new Array<Float32Array>(numTrain),
		labels: new Array<Uint8Array>(numTrain),
	};

	const test = {
		inputs: new Array<Float32Array>(numTest),
		labels: new Array<Uint8Array>(numTest),
	};

	// prepare controll variables
	let numTrainItems = 0;
	let numTestItems = 0;

	for (let i = 0; i < NUM_ELEMENTS; i++) {
		const dataset = (() => {
			// special cases
			if (NUM_ELEMENTS - i <= numTrain - numTrainItems) return train;
			if (NUM_ELEMENTS - i <= numTest - numTestItems) return test;

			// default case
			return Math.random() < ratio ? train : test;
		})();

		const numItems = dataset === train ? numTrainItems : numTestItems;

		dataset.inputs[numItems] = new Float32Array(IMAGE_SIZE);
		dataset.labels[numItems] = new Uint8Array(NUM_CLASSESS);

		for (let j = 0; j < IMAGE_SIZE; j++) {
			dataset.inputs[numItems][j] = inputs[IMAGE_SIZE * i + j];
		}

		for (let j = 0; j < NUM_CLASSESS; j++) {
			dataset.labels[numItems][j] = labels[NUM_CLASSESS * i + j];
		}

		if (dataset === train) {
			numTrainItems++;
		} else {
			numTestItems++;
		}
	}

	return [train, test];
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
			const batchSize = 1000;

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
