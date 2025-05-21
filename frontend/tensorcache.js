import { CPUTensor } from "./kernel.js";


export class SafeTensorCache {
    /**
     * A basic safetensor cache with no evictions
     */
    constructor() {
        this.models = new Map();
    }

    /**
     * Get a tensor from the cache
     * @param {string} model_name
     * @param {string} tensor_name
     * @returns {CPUTensor | null}
     */
    peekTensor(model_name, tensor_name) {
        const model = this.models.get(model_name);
        if (!model) {
            return null;
        }
        return model[tensor_name];
    }

    /**
     * Put tensor into cache
     * @param {string} model_name
     * @param {string} tensor_name
     * @param {CPUTensor} tensor
     */
    putTensor(model_name, tensor_name, tensor) {
        let model = this.models.get(model_name);
        if (!model) {
            model = new Map();
            this.models.set(model_name, model);
        }
        model.set(tensor_name, tensor);
    }

    /**
     * Get a tensor from the cache
     * @param {string} model_name
     * @param {string} tensor_name
     * @returns {CPUTensor | null}
     */
    async getTensor(model_name, tensor_name) {
        let peeked = this.peekTensor(model_name, tensor_name);
        if (peeked) {
            return peeked;
        }
        const encodedModelName = btoa(model_name);
        const response = await fetch(`/safetensor/${encodedModelName}/${tensor_name}`);

        if (!response.ok) {
            throw new Error(`Failed to fetch tensor: ${response.status} ${response.statusText}`);
        }

        const buffer = await response.arrayBuffer();
        const view = new DataView(buffer);
        const [tensor] = CPUTensor.decode(view, 0);
        this.putTensor(model_name, tensor_name, tensor);
        return tensor;
    }
}