/**
 * @typedef {import('./worker.js').CPUTensor} CPUTensor
 * @typedef {import('./worker.js').PartitionWork} PartitionWork
 * @typedef {import('./worker.js').PartitionWorkResult} PartitionWorkResult
 * @typedef {import('./worker.js').InputAssignment} InputAssignment
 * @typedef {import('./worker.js').OutputAssignment} OutputAssignment
 */

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

/**
 * Writes a 32-bit big-endian integer to the DataView.
 * @param {DataView} view The DataView to write to.
 * @param {number} offset The offset to start writing at.
 * @param {number} value The integer value to write.
 * @returns {number} The new offset after writing.
 */
export function writeBEInt(view, offset, value) {
    view.setInt32(offset, value, false); // false for big-endian
    return offset + 4;
}

/**
 * Writes a boolean to the DataView.
 * @param {DataView} view The DataView to write to.
 * @param {number} offset The offset to start writing at.
 * @param {boolean} value The boolean value to write.
 * @returns {number} The new offset after writing.
 */
export function writeBool(view, offset, value) {
    if (value == true) {
        view.setUint8(offset, true);
    } else {
        view.setUint8(offset, false); 
    }
    return offset + 1;
}

/**
 * Reads a 32-bit big-endian integer from the DataView.
 * @param {DataView} view The DataView to read from.
 * @param {number} offset The offset to start reading at.
 * @returns {[number, number]} A tuple containing the read integer and the new offset.
 */
export function readBEInt(view, offset) {
    return [view.getInt32(offset, false), offset + 4];
}

/**
 * Calculates the size of an encoded string.
 * @param {string} str The string to encode.
 * @returns {number} The size in bytes of the encoded string.
 */
export function sizeEncodedString(str) {
    return 4 + textEncoder.encode(str).byteLength;
}

/**
 * Writes a length-prefixed UTF-8 string to the DataView.
 * @param {DataView} view The DataView to write to.
 * @param {number} offset The offset to start writing at.
 * @param {string} str The string to write.
 * @returns {number} The new offset after writing.
 */
export function writeEncodedString(view, offset, str) {
    const encodedString = textEncoder.encode(str);
    offset = writeBEInt(view, offset, encodedString.byteLength);
    new Uint8Array(view.buffer, view.byteOffset + offset, encodedString.byteLength).set(encodedString);
    return offset + encodedString.byteLength;
}

/**
 * Reads a length-prefixed UTF-8 string from the DataView.
 * @param {DataView} view The DataView to read from.
 * @param {number} offset The offset to start reading at.
 * @returns {[string, number]} A tuple containing the read string and the new offset.
 */
export function readEncodedString(view, offset) {
    let length;
    [length, offset] = readBEInt(view, offset);
    const encodedString = new Uint8Array(view.buffer, view.byteOffset + offset, length);
    return [textDecoder.decode(encodedString), offset + length];
}