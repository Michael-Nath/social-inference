import { CPUKernel, CPUTensor, GPUKernel, GPUTensor } from "./kernel.js";


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
        console.log(tensor_name);
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

        const reader = response.body.getReader();

        // Read the initial chunk, which must contain at least the 4-byte header size
        let { value: firstChunk, done: streamDone } = await reader.read();
        if (streamDone || !firstChunk || firstChunk.byteLength < 4) {
            throw new Error("Stream ended or initial chunk too small for header size.");
        }

        // Extract header size (4 bytes, big-endian)
        const headerSizeView = new DataView(firstChunk.buffer, firstChunk.byteOffset, 4);
        const headerSize = headerSizeView.getUint32(0, false);

        let headerBytes = new Uint8Array(headerSize);
        let headerBytesReadCount = 0;
        let tensorDataChunks = [];
        let tensorDataLength = 0;

        // Process data from the first chunk (after the 4-byte header size part)
        let dataAfterHeaderSize = firstChunk.slice(4);

        // How much of dataAfterHeaderSize is part of the header?
        const headerPortionInFirstChunk = Math.min(dataAfterHeaderSize.byteLength, headerSize);
        if (headerPortionInFirstChunk > 0) {
            headerBytes.set(dataAfterHeaderSize.subarray(0, headerPortionInFirstChunk), headerBytesReadCount);
            headerBytesReadCount += headerPortionInFirstChunk;
        }

        // If there's data remaining in dataAfterHeaderSize, it's the start of the tensor data
        if (dataAfterHeaderSize.byteLength > headerPortionInFirstChunk) {
            const tensorPortionInFirstChunk = dataAfterHeaderSize.subarray(headerPortionInFirstChunk);
            tensorDataChunks.push(tensorPortionInFirstChunk);
            tensorDataLength += tensorPortionInFirstChunk.byteLength;
        }

        // Loop to read the rest of the header if not fully read yet
        while (headerBytesReadCount < headerSize) {
            ({ value: firstChunk, done: streamDone } = await reader.read()); // Re-using firstChunk variable for subsequent chunks
            if (streamDone) {
                if (headerBytesReadCount < headerSize) {
                    throw new Error("Stream ended prematurely while reading tensor header.");
                }
                break; // Should not happen if headerBytesReadCount < headerSize due to check above
            }
            if (!firstChunk) continue; // Should not happen with ReadableStream

            const currentChunk = firstChunk;
            const neededForHeader = headerSize - headerBytesReadCount;
            const headerPortionInCurrentChunk = Math.min(currentChunk.byteLength, neededForHeader);

            if (headerPortionInCurrentChunk > 0) {
                headerBytes.set(currentChunk.subarray(0, headerPortionInCurrentChunk), headerBytesReadCount);
                headerBytesReadCount += headerPortionInCurrentChunk;
            }

            // If there's data remaining in currentChunk, it's tensor data
            if (currentChunk.byteLength > headerPortionInCurrentChunk) {
                const tensorPortionInCurrentChunk = currentChunk.subarray(headerPortionInCurrentChunk);
                tensorDataChunks.push(tensorPortionInCurrentChunk);
                tensorDataLength += tensorPortionInCurrentChunk.byteLength;
            }
        }

        const headerJson = new TextDecoder().decode(headerBytes);
        const header = JSON.parse(headerJson);
        const { dtype, shape } = header;

        // Read any remaining tensor data
        while (!streamDone) {
            ({ value: firstChunk, done: streamDone } = await reader.read()); // Re-using firstChunk for subsequent chunks
            if (streamDone) break;
            if (firstChunk && firstChunk.byteLength > 0) {
                tensorDataChunks.push(firstChunk);
                tensorDataLength += firstChunk.byteLength;
            }
        }

        // Assemble the full tensor data
        const fullTensorData = new Uint8Array(tensorDataLength);
        let offset = 0;
        for (const chunk of tensorDataChunks) {
            fullTensorData.set(chunk, offset);
            offset += chunk.byteLength;
        }

        const tensor = new CPUTensor({
            data: fullTensorData.buffer,
            shape,
            dtype,
        });

        this.putTensor(model_name, tensor_name, tensor);
        return tensor;
    }
    

}