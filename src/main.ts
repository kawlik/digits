import { loadMNIST } from "./data/mnist";

const cvs = document.createElement("canvas");
const ctx = cvs.getContext("2d")!;

cvs.width = 28;
cvs.height = 28;

document.body.append(cvs);

loadMNIST().then(([train, test]) => {
	const inputIndex = Math.floor(Math.random() * train.inputs.length);
	const inputData = new Uint8ClampedArray(train.inputs[inputIndex].length * 4);

	// process picture
	for (let i = 0; i < train.inputs[inputIndex].length; i++) {
		inputData[4 * i + 0] = train.inputs[inputIndex][i] * 255;
		inputData[4 * i + 1] = train.inputs[inputIndex][i] * 255;
		inputData[4 * i + 2] = train.inputs[inputIndex][i] * 255;
		inputData[4 * i + 3] = 255;
	}

	// process image
	const imageData = new ImageData(inputData, 28, 28);

	// draw image
	ctx.putImageData(imageData, 0, 0);
});
