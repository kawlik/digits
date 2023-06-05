import { loadMNIST } from "./data/mnist";

loadMNIST().then((dataset) => {
	console.log(dataset.nextTrainBatch(20));
});
